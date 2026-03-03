"""Section A: Baseline Scores and Interpretation (Figures 1-5)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import plot_LH_heatmap, plot_scatter_colored_by_layer, plot_scatter_colored_by_head, LAYER_LABEL, HEAD_LABEL


def fig01_symbolic_score_heatmap(mean_sym: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 1: Symbolic score heatmap."""
    title = r"Symbolic scores $\bar{s}_{\mathrm{sym}}^{(\ell,h)}$"
    if N is not None:
        title += f" (Avg. over {N} graphs)"
    return plot_LH_heatmap(
        mean_sym, title=title,
        vmin=0.0, vmax=1.0, cbar_label=r"$\bar{s}_{\mathrm{sym}}$", **kw,
    )


def fig02_structural_score_heatmap(mean_struct: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 2: Structural score heatmap."""
    title = r"Structural scores $\bar{s}_{\mathrm{struct}}^{(\ell,h)}$"
    if N is not None:
        title += f" (Avg. over {N} graphs)"
    return plot_LH_heatmap(
        mean_struct, title=title,
        vmin=0.0, vmax=1.0, cbar_label=r"$\bar{s}_{\mathrm{struct}}$", **kw,
    )


def fig03_score_scatter_by_layer(
    mean_struct: np.ndarray, mean_sym: np.ndarray,
    annotate_heads: list | None = None, N: int | None = None, **kw,
) -> Figure:
    """Fig 3: Score distribution colored by layer."""
    if annotate_heads is None:
        annotate_heads = [(1, 24, "L1,H24"), (11, 14, "L11,H14")]
    title = r"Per-head scores $(\bar{s}_{\mathrm{struct}},\;\bar{s}_{\mathrm{sym}})$ coloured by layer"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_scatter_colored_by_layer(
        mean_struct, mean_sym,
        xlabel=r"$\bar{s}_{\mathrm{struct}}$", ylabel=r"$\bar{s}_{\mathrm{sym}}$",
        title=title,
        xlim=(0, 1), ylim=(0, 1), figsize=(8, 7),
        annotate_heads=annotate_heads, **kw,
    )


def fig04_score_scatter_by_head(
    mean_struct: np.ndarray, mean_sym: np.ndarray,
    N: int | None = None, **kw,
) -> Figure:
    """Fig 4: Score distribution colored by head index."""
    title = r"Per-head scores coloured by head index $h$"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_scatter_colored_by_head(
        mean_struct, mean_sym,
        xlabel=r"$\bar{s}_{\mathrm{struct}}$", ylabel=r"$\bar{s}_{\mathrm{sym}}$",
        title=title, figsize=(8, 7), **kw,
    )


def fig04b_centered_cossim_scatter_by_layer(
    variants: dict[str, np.ndarray],
    annotate_heads: list | None = None,
    N: int | None = None,
    **kw,
) -> Figure:
    """Per-head centered cosine scores (struct vs symbolic), coloured by layer."""
    key_struct = "ccossim_struct_w" if "ccossim_struct_w" in variants else "ccossim_struct"
    key_sym = "ccossim_symb_w" if "ccossim_symb_w" in variants else "ccossim_symb"
    if key_struct not in variants or key_sym not in variants:
        raise KeyError("Expected centered cosine keys in variants: ccossim_struct(_w), ccossim_symb(_w).")
    if annotate_heads is None:
        annotate_heads = [(1, 24, "L1,H24"), (11, 14, "L11,H14")]

    x = variants[key_struct]
    y = variants[key_sym]
    pad = 0.02
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))

    title = r"Per-head centered cosine scores $(\bar{c}_{\mathrm{struct}},\;\bar{c}_{\mathrm{sym}})$ coloured by layer"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_scatter_colored_by_layer(
        x, y,
        xlabel=r"Centered structural cosine $\bar{c}_{\mathrm{struct}}$",
        ylabel=r"Centered symbolic cosine $\bar{c}_{\mathrm{sym}}$",
        title=title,
        xlim=(xmin - pad, xmax + pad),
        ylim=(ymin - pad, ymax + pad),
        figsize=(8, 7),
        annotate_heads=annotate_heads,
        **kw,
    )


def fig05_scoring_variants_comparison(
    variants: dict[str, np.ndarray],
    N: int | None = None,
    figsize: tuple = (18, 10),
    **kw,
) -> Figure:
    """Fig 5: Six-panel comparison of scoring variants.

    Parameters
    ----------
    variants : dict
        Output of ``aggregate_all_variants()``.  Expected keys include
        ``tv_contrast_w``, ``ccossim_struct_w``, ``ccossim_symb_w``,
        ``logit_contrast_w``, ``orig_struct_w``, ``orig_symb_w``.
    N : int, optional
        Number of graphs (shown in suptitle).
    """
    panels = [
        ("tv_contrast_w",     r"TV Contrast $\rho_{\mathrm{TV}}^{(\ell,h)}$ (weighted)"),
        ("ccossim_struct_w",  r"Centered Cosine — Structural (weighted)"),
        ("ccossim_symb_w",    r"Centered Cosine — Symbolic (weighted)"),
        ("logit_contrast_w",  r"Logit Contrast $\rho_{\mathrm{logit}}^{(\ell,h)}$ (weighted)"),
        ("orig_struct_w",     r"Original $\bar{s}_{\mathrm{struct}}$ (weighted)"),
        ("orig_symb_w",       r"Original $\bar{s}_{\mathrm{sym}}$ (weighted)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, panels):
        if key not in variants:
            ax.set_visible(False)
            continue
        mat = variants[key]
        L, H = mat.shape
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(HEAD_LABEL, fontsize=9)
        ax.set_ylabel(LAYER_LABEL, fontsize=9)
        ax.set_xticks(np.arange(H))
        ax.set_xticklabels(np.arange(H), fontsize=5)
        ax.set_yticks(np.arange(L))
        ax.set_yticklabels(np.arange(L), fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    suptitle = r"Scoring Variants Comparison"
    if N is not None:
        suptitle += f" (Avg. over {N} graphs)"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
