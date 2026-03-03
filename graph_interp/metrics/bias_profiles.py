"""Active distance bin discovery and bias profile utilities.

Provides functions to discover which spatial-position distance bins
actually appear in a set of molecular graphs, pre-filter SMILES to
connected graphs, and compute entropy of learned bias profiles.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from ..data import build_batch_from_smiles


# ---------------------------------------------------------------------------
# Distance bin discovery
# ---------------------------------------------------------------------------

def discover_active_distance_bins(
    smiles_list: list[str],
    model: Any,
    device: torch.device,
    max_graphs: int = 200,
    sentinel: int = 100,
) -> tuple[list[int], dict[int, int], int]:
    """Scan graphs to find which spatial_pos distance bins actually appear.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES pool to scan.
    model : GraphormerEncoder
        Pre-trained model (used for batch building).
    device : torch.device
        Computation device.
    max_graphs : int
        Maximum number of graphs to scan.
    sentinel : int
        Value above which a distance is treated as disconnected / padding.
        The Graphormer ``spatial_pos`` offset adds 1, so a distance of
        *sentinel* or more typically means unreachable.

    Returns
    -------
    active_bins : list[int]
        Sorted list of distance values that appear at least once.
    bin_counts : dict[int, int]
        Mapping from distance value to number of (i, j) pairs observed.
    max_real_dist : int
        Maximum distance value seen (excluding sentinel / padding).
    """
    bin_counter: Counter[int] = Counter()
    max_real_dist = 0

    for smi in smiles_list[:max_graphs]:
        try:
            batch, n_atoms, dist = build_batch_from_smiles(smi, model=model, device=device)
        except Exception:
            continue

        # dist is the raw numpy shortest-path distance matrix [n, n]
        dist_arr = np.asarray(dist, dtype=np.int64)
        for i in range(n_atoms):
            for j in range(n_atoms):
                d = int(dist_arr[i, j])
                if d < sentinel and d >= 0:
                    bin_counter[d] += 1
                    if d > max_real_dist:
                        max_real_dist = d

    active_bins = sorted(bin_counter.keys())
    bin_counts = dict(bin_counter)
    return active_bins, bin_counts, max_real_dist


# ---------------------------------------------------------------------------
# Pre-filter connected SMILES
# ---------------------------------------------------------------------------

def filter_connected_smiles(
    smiles_list: list[str],
    model: Any,
    device: torch.device,
    sentinel: int = 100,
    max_diameter: int | None = None,
    verbose: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Pre-filter SMILES to only connected graphs.

    A graph is considered disconnected if any off-diagonal entry in its
    shortest-path distance matrix equals or exceeds *sentinel*.

    Parameters
    ----------
    smiles_list : list[str]
        Full SMILES pool to filter.
    model : GraphormerEncoder
        Pre-trained model (used for batch building).
    device : torch.device
        Computation device.
    sentinel : int
        Distance value at or above which a pair is disconnected.
    max_diameter : int or None
        If given, also exclude graphs whose diameter exceeds this value.
    verbose : bool
        Print summary statistics.

    Returns
    -------
    filtered : list[str]
        SMILES that passed the connectivity (and diameter) filter.
    stats : dict
        Summary with keys: ``total``, ``disconnected``,
        ``too_large_diameter``, ``kept``, ``diameter_mean``,
        ``diameter_max``.
    """
    filtered: list[str] = []
    diameters: list[int] = []
    n_disconnected = 0
    n_too_large = 0

    for smi in smiles_list:
        try:
            batch, n_atoms, dist = build_batch_from_smiles(smi, model=model, device=device)
        except Exception:
            n_disconnected += 1
            continue

        dist_arr = np.asarray(dist, dtype=np.int64)

        # Check connectivity: all off-diagonal distances must be < sentinel
        off_diag = dist_arr[np.triu_indices(n_atoms, k=1)]
        if np.any(off_diag >= sentinel):
            n_disconnected += 1
            continue

        diameter = int(off_diag.max()) if len(off_diag) > 0 else 0

        if max_diameter is not None and diameter > max_diameter:
            n_too_large += 1
            continue

        filtered.append(smi)
        diameters.append(diameter)

    stats: dict[str, Any] = {
        "total": len(smiles_list),
        "disconnected": n_disconnected,
        "too_large_diameter": n_too_large,
        "kept": len(filtered),
        "diameter_mean": float(np.mean(diameters)) if diameters else 0.0,
        "diameter_max": int(np.max(diameters)) if diameters else 0,
    }

    if verbose:
        print(
            f"  [filter] {stats['kept']}/{stats['total']} kept "
            f"({stats['disconnected']} disconnected, "
            f"{stats['too_large_diameter']} too-large-diameter). "
            f"Mean diameter={stats['diameter_mean']:.1f}, "
            f"max={stats['diameter_max']}"
        )

    return filtered, stats


# ---------------------------------------------------------------------------
# Bias profile entropy
# ---------------------------------------------------------------------------

def bias_profile_entropy(
    bias_profiles: np.ndarray,
    active_bins: list[int] | None = None,
    exclude_bins: list[int] | None = None,
    hop_min: int | None = 1,
    hop_max: int | None = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Compute entropy of learned spatial-bias profiles.

    The raw bias values are softmax-normalised over the selected bins to
    form a probability distribution, then Shannon entropy (in bits) is
    computed per head.

    Parameters
    ----------
    bias_profiles : np.ndarray
        Shape ``[H, D_bins]`` -- learned spatial bias weights from the model.
    active_bins : list[int] or None
        Indices into the second axis of *bias_profiles* to keep.  If
        ``None``, all bins are used.
    exclude_bins : list[int] or None
        Indices to explicitly drop (applied after *active_bins*).
    hop_min, hop_max : int or None
        Optional inclusive hop-distance window. When provided, only bins
        with ``hop_min <= d <= hop_max`` are used for normalisation and
        entropy computation.

    Returns
    -------
    entropy : np.ndarray  [H]
        Shannon entropy in bits for each head.
    entropy_norm : np.ndarray  [H]
        Entropy normalised by log2(n_active) so values are in [0, 1].
    peak_idx : np.ndarray  [H]
        Index (into *used_bins*) of the highest-probability bin per head.
    peak_d_label : np.ndarray  [H]
        The actual distance-bin value corresponding to the peak.
    profiles_prob : np.ndarray  [H, n_active]
        Softmax-normalised profiles over active bins.
    used_bins : list[int]
        The final list of bin indices used.
    """
    H, D_bins = bias_profiles.shape

    # Determine which bins to use
    if active_bins is not None:
        used_bins = sorted(b for b in active_bins if 0 <= b < D_bins)
    else:
        used_bins = list(range(D_bins))

    if exclude_bins is not None:
        exclude_set = set(exclude_bins)
        used_bins = [b for b in used_bins if b not in exclude_set]
    if hop_min is not None:
        used_bins = [b for b in used_bins if b >= int(hop_min)]
    if hop_max is not None:
        used_bins = [b for b in used_bins if b <= int(hop_max)]

    n_active = len(used_bins)
    if n_active == 0:
        raise ValueError("No active bins remaining after filtering.")

    # Extract and softmax normalise
    raw = bias_profiles[:, used_bins]  # [H, n_active]
    # Softmax along bins axis
    raw_shifted = raw - raw.max(axis=1, keepdims=True)
    exp_raw = np.exp(raw_shifted)
    profiles_prob = exp_raw / exp_raw.sum(axis=1, keepdims=True)  # [H, n_active]

    # Entropy in bits
    eps = 1e-12
    entropy = -(profiles_prob * np.log2(profiles_prob + eps)).sum(axis=1)  # [H]

    # Normalised entropy
    max_ent = np.log2(n_active) if n_active > 1 else 1.0
    entropy_norm = entropy / max_ent

    # Peak bin per head
    peak_idx = np.argmax(profiles_prob, axis=1)  # [H]
    peak_d_label = np.array([used_bins[int(p)] for p in peak_idx])

    return entropy, entropy_norm, peak_idx, peak_d_label, profiles_prob, used_bins
