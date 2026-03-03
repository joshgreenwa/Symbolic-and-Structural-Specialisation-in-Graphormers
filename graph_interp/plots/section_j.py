"""Section J: Learned Bias Profiles (Figures 42-44)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import HEAD_LABEL


def fig42_bias_profile_entropy(
    entropy: np.ndarray,
    entropy_norm: np.ndarray | None = None,
    N: int | None = None,
    **kw,
) -> Figure:
    """Fig 42: Bias profile entropy per head — bar chart with viridis colouring.

    Parameters
    ----------
    entropy : np.ndarray  [H]
        Shannon entropy (bits) per head from ``bias_profile_entropy()``.
    entropy_norm : np.ndarray  [H], optional
        Normalised entropy (0-1).  Used for colouring when provided.
    """
    H = len(entropy)
    cvals = entropy_norm if entropy_norm is not None else entropy
    colors = plt.cm.viridis(cvals / max(cvals.max(), 1e-12))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(H), entropy, color=colors)
    ax.set_xlabel(HEAD_LABEL)
    ax.set_ylabel("Entropy (bits)")
    title = r"Bias profile entropy per head (active bins only)"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(np.arange(H))
    ax.set_xticklabels(np.arange(H), fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig43_peak_hop_distance(
    peak_d_label: np.ndarray | None = None,
    profiles_prob: np.ndarray | None = None,
    used_bins: list[int] | None = None,
    N: int | None = None,
    **kw,
) -> Figure:
    """Fig 43: Mode hop distance per head.

    Parameters
    ----------
    peak_d_label : np.ndarray  [H], optional
        Precomputed peak/mode distance-bin value per head.
    profiles_prob : np.ndarray [H, n_active], optional
        Distance-mass distributions.  If provided with ``used_bins``,
        mode distance is computed as ``argmax_d p(d)`` per head.
    used_bins : list[int], optional
        Distance-bin labels corresponding to columns of ``profiles_prob``.
    """
    if profiles_prob is not None and used_bins is not None:
        probs = np.asarray(profiles_prob, dtype=float)
        used = np.asarray(used_bins, dtype=float)
        if probs.ndim != 2:
            raise ValueError(f"profiles_prob must be [H, n_bins], got {probs.shape}")
        if probs.shape[1] != used.shape[0]:
            raise ValueError("profiles_prob width must match len(used_bins).")
        peak_d_label = used[np.argmax(probs, axis=1)]
    elif peak_d_label is None:
        raise ValueError("Provide either peak_d_label or (profiles_prob, used_bins).")

    peak_d_label = np.asarray(peak_d_label, dtype=float)
    H = len(peak_d_label)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    cvals = (peak_d_label - peak_d_label.min()) / max(float(peak_d_label.max() - peak_d_label.min()), 1e-12)
    colors = plt.cm.YlOrRd(0.35 + 0.6 * cvals)
    ax.bar(np.arange(H), peak_d_label, color=colors, alpha=0.9, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(HEAD_LABEL)
    ax.set_ylabel(r"Mode distance $d_{\mathrm{mode}}$")
    title = r"Mode hop distance per head ($d_{\mathrm{mode}}=\arg\max_d\;p(d)$)"
    title += "\n(highest individual distance-mass per head)"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(np.arange(0, H, 4))
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def fig44_bias_profiles_gallery(
    profiles_prob: np.ndarray,
    used_bins: list[int],
    cols: int = 8,
    N: int | None = None,
    **kw,
) -> Figure:
    """Fig 44: Gallery of softmax-normalised bias profiles (active bins only).

    Parameters
    ----------
    profiles_prob : np.ndarray  [H, n_active]
        Softmax-normalised profiles from ``bias_profile_entropy()``.
    used_bins : list[int]
        Actual distance-bin values for the x-axis labels.
    """
    H, n_active = profiles_prob.shape
    rows = (H + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten()

    x = np.arange(n_active)
    bin_labels = [str(b) for b in used_bins]

    for h in range(H):
        ax = axes_flat[h]
        ax.bar(x, profiles_prob[h], color="steelblue", alpha=0.85)
        ax.set_title(f"H{h}", fontsize=8)
        if h >= H - cols:
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, fontsize=5, rotation=45)
            ax.set_xlabel("Hop $d$", fontsize=7)
        if h % cols == 0:
            ax.set_ylabel(r"$p(d)$", fontsize=7)
        ax.tick_params(labelsize=5)

    for h in range(H, len(axes_flat)):
        axes_flat[h].set_visible(False)

    suptitle = r"Learned spatial bias profiles (softmax-normalised, hop distances 1–8)"
    if N is not None:
        suptitle += f" ({N} graphs)"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig
