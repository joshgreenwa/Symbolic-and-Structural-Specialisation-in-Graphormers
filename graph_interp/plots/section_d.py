"""Section D: What Does Structural Specialisation Look Like? (Figures 13-14)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..chemistry import plot_molecule_attention_tripanel
from . import LAYER_LABEL


def fig13_structural_head_attention(
    A_list_per_mol: list,
    smiles_list: list[str],
    layer: int = 11,
    head: int = 14,
    mol_indices: list[int] | None = None,
) -> list[Figure]:
    """Fig 13: 3-panel attention visualisation of L11,H14 for selected molecules.

    Parameters
    ----------
    A_list_per_mol : list
        ``A_list_per_mol[i][layer]`` is ``[H, T, T]`` attention for molecule *i*.
    smiles_list : list[str]
        SMILES strings.
    layer, head : int
        Which layer and head to visualise.
    mol_indices : list[int], optional
        Indices into *smiles_list* to show (default ``[0, 5, 80]``).
    """
    if mol_indices is None:
        mol_indices = [0, 5, 80]
    mol_indices = [i for i in mol_indices if i < len(smiles_list)]

    figs = []
    for idx in mol_indices:
        smi = smiles_list[idx]
        A_layer = A_list_per_mol[idx][layer]
        fig = plot_molecule_attention_tripanel(
            smi, A_layer, head_idx=head, layer_idx=layer,
        )
        fig.suptitle(
            f"L{layer} H{head} — Molecule {idx} (structural head)",
            fontsize=13, fontweight="bold", y=1.02,
        )
        figs.append(fig)
    return figs


def fig14_attention_mass_vs_hop(
    mean_M: np.ndarray,
    layer: int = 11,
    head: int = 14,
    d_cap: int | None = 10,
    active_bins: list[int] | None = None,
    N: int | None = None,
    **kw,
) -> Figure:
    """Fig 14: Attention mass vs hop distance for L11,H14.

    Parameters
    ----------
    mean_M : np.ndarray  [L, H, B]
        Averaged distance mass profiles.
    active_bins : list[int], optional
        If provided, only show these bins.
    """
    M = mean_M[layer, head]  # [B]
    if active_bins is not None:
        bins = [b for b in active_bins if b < len(M)]
        vals = M[bins]
        labels_x = [str(b) for b in bins]
    else:
        B = M.shape[0]
        bins = list(range(B))
        vals = M
        if d_cap is not None:
            labels_x = [str(i) if i <= d_cap else f">{d_cap}" for i in range(B)]
        else:
            labels_x = [str(i) for i in range(B)]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(vals))
    ax.bar(x, vals, color="steelblue", alpha=0.8)
    ax.set_xlabel(r"Hop distance $d$", fontsize=11)
    ax.set_ylabel(r"Attention mass $M(d)$", fontsize=11)
    title = rf"Attention mass vs hop distance — L{layer} H{head}"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
