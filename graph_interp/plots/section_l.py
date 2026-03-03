"""Section L: Extra Early-Layer Specialisation — PCA grids (Figures 49-53)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from ..chemistry import pca_2d, LABELS_FOCUS_EXPANDED


def _prepare_head_label_ids(labels, n_heads: int, n_mols: int):
    """Convert labels into integer ids shaped [H, N] and return label names."""
    arr = np.asarray(labels)
    if arr.ndim == 1:
        if arr.shape[0] != n_mols:
            raise ValueError(f"1D labels must have length N={n_mols}, got {arr.shape[0]}")
        arr = np.broadcast_to(arr.reshape(1, -1), (n_heads, n_mols))
    elif arr.ndim == 2:
        if arr.shape == (n_mols, n_heads):
            arr = arr.T
        elif arr.shape != (n_heads, n_mols):
            raise ValueError(f"2D labels must be [H,N] or [N,H], got {arr.shape}")
    else:
        raise ValueError("labels must be 1D or 2D.")

    if np.issubdtype(arr.dtype, np.integer):
        y = arr.astype(int)
        label_names = LABELS_FOCUS_EXPANDED
        y = np.clip(y, 0, len(label_names) - 1)
        return y, label_names

    unique = sorted(set(arr.reshape(-1).tolist()))
    name_to_id = {name: i for i, name in enumerate(unique)}
    y = np.vectorize(lambda z: name_to_id[str(z)])(arr).astype(int)
    return y, unique


def fig_pca_grid(
    embeddings_per_head: np.ndarray,
    labels: list[str] | np.ndarray,
    layer: int,
    n_heads: int = 32,
    cols: int = 8,
    point_size: int = 10,
    alpha: float = 0.8,
    focus_thresh: float | None = None,
    diffuse_thresh: float | None = None,
    legend_ncol: int | None = None,
    legend_fontsize: float = 6.8,
    N: int | None = None,
) -> Figure:
    """Generic PCA grid of all heads at one layer.

    Parameters
    ----------
    embeddings_per_head : np.ndarray  [H, N_mols, d]
        Pooled head output vectors (mean of A@V over node queries).
    labels : list[str] or np.ndarray
        Either per-molecule labels [N] or head-specific labels [H, N].
    layer : int
        Layer index (used in title).
    N : int, optional
        Number of molecules (shown in title).
    """
    H = min(n_heads, embeddings_per_head.shape[0])
    rows = (H + cols - 1) // cols
    n_mols = embeddings_per_head.shape[1]

    y_ids, label_names = _prepare_head_label_ids(labels, H, n_mols)
    n_colors = max(len(label_names), 1)
    color_list = plt.cm.tab20(np.linspace(0, 1, n_colors))
    cmap = ListedColormap(color_list)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.25, rows * 2.25))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    for h in range(H):
        ax = axes_flat[h]
        X = np.asarray(embeddings_per_head[h], dtype=float)
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
        Z = pca_2d(X)
        c = y_ids[h]
        ax.scatter(
            Z[:, 0], Z[:, 1], c=c, cmap=cmap, s=point_size, alpha=alpha,
            vmin=0, vmax=max(n_colors - 1, 1), edgecolors="none",
        )
        ax.set_title(f"H{h}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.12)

    for h in range(H, len(axes_flat)):
        axes_flat[h].set_visible(False)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=color_list[i], markeredgecolor="none",
            markersize=6, label=lab,
        )
        for i, lab in enumerate(label_names)
    ]

    if legend_ncol is None:
        # Keep legend horizontal and compact: many columns, few rows.
        legend_ncol = min(len(label_names), max(8, cols * 2))
    legend_ncol = max(1, int(legend_ncol))
    legend_rows = int(np.ceil(len(label_names) / legend_ncol))

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.012),
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    suptitle = rf"PCA of $A \cdot V$ head output — Layer {layer} (all heads)"
    if N is not None:
        suptitle += f" — {N} molecules"
    if focus_thresh is not None or diffuse_thresh is not None:
        suptitle += " | "
        suptitle += rf"$\tau_{{focus}}={focus_thresh if focus_thresh is not None else 'default'}$"
        suptitle += rf", $\tau_{{diffuse}}={diffuse_thresh if diffuse_thresh is not None else 'default'}$"
    fig.suptitle(suptitle, fontsize=12, y=0.985)

    # Reserve dedicated space for the horizontal legend below the subplot grid.
    bottom_margin = 0.085 + 0.045 * legend_rows
    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=bottom_margin,
        wspace=0.18,
        hspace=0.24,
    )
    return fig


def fig49_pca_layer0(embeddings_per_head: np.ndarray, labels: list[str] | np.ndarray,
                      N: int | None = None, **kw) -> Figure:
    """Fig 49: PCA grid — Layer 0."""
    return fig_pca_grid(embeddings_per_head, labels, layer=0, N=N, **kw)


def fig50_pca_layer1(embeddings_per_head: np.ndarray, labels: list[str] | np.ndarray,
                      N: int | None = None, **kw) -> Figure:
    """Fig 50: PCA grid — Layer 1."""
    return fig_pca_grid(embeddings_per_head, labels, layer=1, N=N, **kw)


def fig51_pca_layer2(embeddings_per_head: np.ndarray, labels: list[str] | np.ndarray,
                      N: int | None = None, **kw) -> Figure:
    """Fig 51: PCA grid — Layer 2."""
    return fig_pca_grid(embeddings_per_head, labels, layer=2, N=N, **kw)


def fig52_pca_layer3(embeddings_per_head: np.ndarray, labels: list[str] | np.ndarray,
                      N: int | None = None, **kw) -> Figure:
    """Fig 52: PCA grid — Layer 3."""
    return fig_pca_grid(embeddings_per_head, labels, layer=3, N=N, **kw)


def fig53_pca_layer11(embeddings_per_head: np.ndarray, labels: list[str] | np.ndarray,
                       N: int | None = None, **kw) -> Figure:
    """Fig 53: PCA grid — Layer 11."""
    return fig_pca_grid(embeddings_per_head, labels, layer=11, N=N, **kw)
