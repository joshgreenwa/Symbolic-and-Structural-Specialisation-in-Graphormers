"""Shared plotting helpers and constants."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

LAYER_LABEL = r"Layer Index $\ell$"
HEAD_LABEL = r"Head Index $h$"


def _flatten_LH(mat: np.ndarray):
    """Flatten [L, H] -> (values, layer_ids, head_ids)."""
    L, H = mat.shape
    x = mat.reshape(-1)
    layer = np.repeat(np.arange(L), H)
    head = np.tile(np.arange(H), L)
    return x, layer, head


def annotate_head_points(ax, xs, ys, heads_list):
    """Draw red circle annotations with arrows for specific heads.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to annotate on.
    xs : np.ndarray
        [L, H] x-coordinates of all heads.
    ys : np.ndarray
        [L, H] y-coordinates of all heads.
    heads_list : list of (layer, head, label) tuples
        Each tuple identifies a head to annotate and the text label to display.
    """
    if heads_list is None:
        return
    for item in heads_list:
        if len(item) >= 4:
            layer, head, label, style = item[0], item[1], item[2], dict(item[3] or {})
        else:
            layer, head, label = item
            style = {}
        px, py = xs[layer, head], ys[layer, head]
        # L1,H24: force a vertical upward arrow from below the point.
        if layer == 1 and head == 24:
            style = {
                "xytext": (0, -30),
                "ha": "center",
                "va": "top",
                **style,
            }
        else:
            style = {
                "xytext": (15, 15),
                "ha": "left",
                "va": "bottom",
                **style,
            }
        # Red hollow circle
        ax.scatter([px], [py], s=120, facecolors="none",
                   edgecolors="red", linewidths=2, zorder=5)
        # Arrow with label
        ax.annotate(
            label,
            xy=(px, py),
            xytext=style["xytext"],
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="red",
            ha=style["ha"],
            va=style["va"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5),
            zorder=6,
        )


def plot_LH_heatmap(
    mat: np.ndarray, title: str = "",
    vmin: float | None = None, vmax: float | None = None,
    cmap: str = "viridis", cbar_label: str = "",
    figsize: tuple | None = None,
    N: int | None = None,
) -> Figure:
    """Layer x Head heatmap. Returns Figure.

    Parameters
    ----------
    N : int, optional
        If provided, allows ``{N}`` placeholder in *title* to be formatted.
    """
    L, H = mat.shape
    if figsize is None:
        figsize = (max(6.0, H * 0.35), max(4.0, L * 0.45))
    if N is not None and "{N}" in title:
        title = title.format(N=N)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(HEAD_LABEL)
    ax.set_ylabel(LAYER_LABEL)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cb.set_label(cbar_label)
    ax.set_xticks(np.arange(H))
    ax.set_xticklabels(np.arange(H), fontsize=7)
    ax.set_yticks(np.arange(L))
    ax.set_yticklabels(np.arange(L))
    fig.tight_layout()
    return fig


def plot_layerwise_line(
    ys: dict[str, np.ndarray],
    title: str = "", xlabel: str = "", ylabel: str = "",
    yerr: dict[str, np.ndarray] | None = None,
    figsize: tuple = (7.2, 4.2),
) -> Figure:
    """Line plot with optional error bands. *ys*: {label: [L]}."""
    fig, ax = plt.subplots(figsize=figsize)
    L = None
    for label, y in ys.items():
        L = len(y)
        x = np.arange(L)
        ax.plot(x, y, marker="o", label=label, markersize=4)
        if yerr and label in yerr:
            ax.fill_between(x, y - yerr[label], y + yerr[label], alpha=0.2)
    ax.set_xlabel(xlabel or LAYER_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if L is not None:
        ax.set_xticks(np.arange(L))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_scatter_colored_by_layer(
    xs: np.ndarray, ys: np.ndarray,
    xlabel: str = "", ylabel: str = "", title: str = "",
    xlim: tuple | None = None, ylim: tuple | None = None,
    figsize: tuple = (8, 7),
    annotate_heads: list | None = None,
) -> Figure:
    """Scatter where each point is one head, colored by layer.

    Parameters
    ----------
    annotate_heads : list of (layer, head, label) tuples, optional
        When provided, draw red hollow circles and arrows for each annotated head.
    """
    L, H = xs.shape
    xf, layers, _ = _flatten_LH(xs)
    yf, _, _ = _flatten_LH(ys)

    cmap = plt.cm.get_cmap("viridis", L)
    colors = cmap(layers)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xf, yf, s=18, alpha=0.85, c=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(l),
               markeredgecolor="none", markersize=7, label=f"Layer {l}")
        for l in range(L)
    ]
    ax.legend(handles=handles, title=LAYER_LABEL, bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0.0, fontsize=7)

    # Annotate specific heads if requested
    if annotate_heads is not None:
        annotate_head_points(ax, xs, ys, annotate_heads)

    fig.tight_layout()
    return fig


def plot_scatter_colored_by_head(
    xs: np.ndarray, ys: np.ndarray,
    xlabel: str = "", ylabel: str = "", title: str = "",
    figsize: tuple = (8, 7),
) -> Figure:
    """Scatter where each point is one head, colored by head index."""
    L, H = xs.shape
    xf, _, heads = _flatten_LH(xs)
    yf, _, _ = _flatten_LH(ys)

    cmap = plt.cm.get_cmap("tab20", H)
    colors = cmap(heads)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xf, yf, s=18, alpha=0.85, c=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(h),
               markeredgecolor="none", markersize=5, label=f"H{h}")
        for h in range(0, H, max(1, H // 8))
    ]
    ax.legend(handles=handles, title=HEAD_LABEL, bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0.0, fontsize=6)
    fig.tight_layout()
    return fig


def plot_bar_chart(
    values: np.ndarray, xlabel: str = "", ylabel: str = "",
    title: str = "", colors=None, figsize: tuple = (7.2, 4.0),
    xlabels: list | None = None,
) -> Figure:
    """Simple bar chart. values: [N]."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(values))
    ax.bar(x, values, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
