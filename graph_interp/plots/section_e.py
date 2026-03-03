"""Section E: Bias Routes, Dot Selects Within (Figures 15-19)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import plot_LH_heatmap, LAYER_LABEL


def fig15_containment_heatmap(containment_LH: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 15: Fraction of |DeltaA| on bias top-k keys."""
    title = r"Containment: fraction of $|\Delta A|$ on bias top-$k$"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_LH_heatmap(
        containment_LH, title=title,
        vmin=0.0, vmax=1.0, cmap="YlOrRd", cbar_label="Containment", **kw,
    )


def fig16_structural_score_topk_vs_botk(
    cossim_results: dict, N: int | None = None, **kw,
) -> Figure:
    """Fig 16: Structural score for bias-favored vs unfavored keys vs layer."""
    L = cossim_results["struct_all"].shape[0]
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for key, label, color, ls in [
        ("struct_all", "All keys", "black", "-"),
        ("struct_topk", "Bias top-$k$", "darkred", "--"),
        ("struct_botk", "Bias bottom-$k$", "navy", "--"),
    ]:
        data = cossim_results[key]
        mu = data.mean(axis=1)
        mn, mx = data.min(axis=1), data.max(axis=1)
        ax.fill_between(x, mn, mx, alpha=0.08, color=color)
        ax.plot(x, mu, marker="o", linestyle=ls, label=label, markersize=4, linewidth=2, color=color)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"$\bar{s}_{\mathrm{struct}}$")
    title = r"Structural score: bias-favoured vs unfavoured keys"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig17_symbolic_score_topk_vs_botk(
    cossim_results: dict, N: int | None = None, **kw,
) -> Figure:
    """Fig 17: Symbolic score for bias-favored vs unfavored keys vs layer."""
    L = cossim_results["symb_all"].shape[0]
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for key, label, color, ls in [
        ("symb_all", "All keys", "black", "-"),
        ("symb_topk", "Bias top-$k$", "darkred", "--"),
        ("symb_botk", "Bias bottom-$k$", "navy", "--"),
    ]:
        data = cossim_results[key]
        mu = data.mean(axis=1)
        mn, mx = data.min(axis=1), data.max(axis=1)
        ax.fill_between(x, mn, mx, alpha=0.08, color=color)
        ax.plot(x, mu, marker="o", linestyle=ls, label=label, markersize=4, linewidth=2, color=color)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"$\bar{s}_{\mathrm{sym}}$")
    title = (
        r"Symbolic score: bias-favoured vs unfavoured keys"
        "\n(high in top-$k$ means dot selects within bias-favoured candidates)"
    )
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig18_symbolic_gap(cossim_results: dict, N: int | None = None, **kw) -> Figure:
    """Fig 18: Symbolic score gap (top-k minus bottom-k) vs layer."""
    gap = cossim_results["symb_topk"] - cossim_results["symb_botk"]
    mu = gap.mean(axis=1)
    mn, mx = gap.min(axis=1), gap.max(axis=1)
    L = len(gap)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.fill_between(x, mn, mx, alpha=0.12, color="purple")
    ax.plot(x, mu, marker="o", color="purple", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"$\Delta\bar{s}_{\mathrm{sym}}$ (top-$k$ $-$ bottom-$k$)")
    title = (
        r"Symbolic score gap: top-$k$ minus bottom-$k$"
        "\n(positive = stronger equivariance inside bias-favoured key set)"
    )
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig19_structural_gap(cossim_results: dict, N: int | None = None, **kw) -> Figure:
    """Fig 19: Structural score gap (top-k minus bottom-k) vs layer."""
    gap = cossim_results["struct_topk"] - cossim_results["struct_botk"]
    mu = gap.mean(axis=1)
    mn, mx = gap.min(axis=1), gap.max(axis=1)
    L = len(gap)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.fill_between(x, mn, mx, alpha=0.12, color="coral")
    ax.plot(x, mu, marker="o", color="coral", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"$\Delta\bar{s}_{\mathrm{struct}}$ (top-$k$ $-$ bottom-$k$)")
    title = r"Structural score gap: top-$k$ $-$ bottom-$k$"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig
