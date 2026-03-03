"""Section F: CLS Token Dynamics (Figures 20-28)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import plot_LH_heatmap, LAYER_LABEL


def fig20_cls_structural_heatmap(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 20: CLS structural score heatmap."""
    title = r"CLS structural score $\bar{s}_{\mathrm{struct}}^{\mathrm{CLS}}$"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_LH_heatmap(
        cls_metrics["struct"], title=title,
        vmin=0.0, vmax=1.0, cbar_label=r"$\bar{s}_{\mathrm{struct}}$", **kw,
    )


def fig21_cls_centered_heatmaps(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 21: CLS centered cosine similarity heatmaps (struct and sym)."""
    struct = cls_metrics["struct"]
    sym = cls_metrics["sym"]
    cs = struct - struct.mean(axis=1, keepdims=True)
    cy = sym - sym.mean(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, lbl in [
        (axes[0], cs, r"Centered $\bar{s}_{\mathrm{struct}}^{\mathrm{CLS}}$"),
        (axes[1], cy, r"Centered $\bar{s}_{\mathrm{sym}}^{\mathrm{CLS}}$"),
    ]:
        L, H = data.shape
        im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel(r"Head Index $h$", fontsize=10)
        ax.set_ylabel(r"Layer Index $\ell$", fontsize=10)
        ax.set_xticks(np.arange(H))
        ax.set_xticklabels(np.arange(H), fontsize=6)
        ax.set_yticks(np.arange(L))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    suptitle = "CLS centred cosine scores"
    if N is not None:
        suptitle += f" ({N} graphs)"
    fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def fig22_cls_entropy_heatmap(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 22: CLS attention entropy per head."""
    title = r"CLS attention entropy $H^{\mathrm{CLS}}$ (bits)"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_LH_heatmap(
        cls_metrics["cls_ent"], title=title,
        cmap="YlOrRd", cbar_label="Entropy (bits)", **kw,
    )


def fig23_cls_gini_heatmap(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 23: CLS attention Gini coefficient per head."""
    title = r"CLS attention Gini coefficient"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_LH_heatmap(
        cls_metrics["cls_gini"], title=title,
        vmin=0.0, vmax=1.0, cbar_label="Gini", **kw,
    )


def _line_with_error(
    ax, x, y, yerr=None, label="", color=None, marker="o",
):
    """Plot a line with optional fill_between ±1σ error band."""
    ax.plot(x, y, marker=marker, label=label, markersize=4, color=color)
    if yerr is not None:
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)


def fig24_cls_entropy_across_layers(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 24: CLS attention entropy across layers (mean ± std over heads)."""
    ent = cls_metrics["cls_ent"].mean(axis=1)
    ent_std = cls_metrics.get("std_cls_ent")
    if ent_std is not None:
        ent_std = ent_std.mean(axis=1) if ent_std.ndim == 2 else ent_std
    L = len(ent)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _line_with_error(ax, x, ent, yerr=ent_std, label="CLS entropy", color="tab:blue")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel("Entropy (bits)")
    title = r"CLS attention entropy across layers"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig25_cls_max_attention(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 25: CLS max attention weight across layers."""
    if "cls_max_attn" in cls_metrics:
        max_attn = cls_metrics["cls_max_attn"]
    else:
        max_attn = np.power(2.0, -cls_metrics["cls_ent"].mean(axis=1))
    L = len(max_attn)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, max_attn, marker="o", markersize=5, color="tab:red")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"Max attention weight")
    title = r"CLS max attention weight across layers"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig26_cls_vs_node_entropy(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 26: CLS vs node query attention entropy (with error bars)."""
    cls_ent = cls_metrics["cls_ent"].mean(axis=1)
    node_ent = cls_metrics["node_ent"].mean(axis=1)
    cls_std = cls_metrics.get("std_cls_ent")
    node_std = cls_metrics.get("std_node_ent")
    if cls_std is not None:
        cls_std = cls_std.mean(axis=1) if cls_std.ndim == 2 else cls_std
    if node_std is not None:
        node_std = node_std.mean(axis=1) if node_std.ndim == 2 else node_std
    L = len(cls_ent)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _line_with_error(ax, x, cls_ent, yerr=cls_std, label=r"CLS $\to$ nodes", color="tab:blue")
    _line_with_error(ax, x, node_ent, yerr=node_std, label=r"Node $\to$ nodes", color="tab:orange", marker="s")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel("Entropy (bits)")
    title = r"CLS vs node query: attention entropy"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig27_cls_vs_node_dot_spread(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 27: CLS vs node query dot-logit spread (with error bars)."""
    cls_std_val = cls_metrics["cls_std"]
    node_std_val = cls_metrics["node_std"]
    cls_err = cls_metrics.get("std_cls_std")
    node_err = cls_metrics.get("std_node_std")
    L = len(cls_std_val)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _line_with_error(ax, x, cls_std_val, yerr=cls_err, label=r"CLS dot std", color="tab:blue")
    _line_with_error(ax, x, node_std_val, yerr=node_err, label=r"Node dot std", color="tab:orange", marker="s")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"Std of scaled dot-product logits")
    title = r"CLS vs node query: dot-logit spread"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig28_cls_similarity_to_mean(cls_metrics: dict, N: int | None = None, **kw) -> Figure:
    """Fig 28: cos(CLS, mean(nodes)) across layers (with error bars)."""
    sim = cls_metrics["cls_sim"]
    sim_std = cls_metrics.get("std_cls_sim")
    L = len(sim)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _line_with_error(ax, x, sim, yerr=sim_std, label=r"$\cos(\mathrm{CLS},\;\overline{\mathrm{node}})$", color="tab:green")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"$\cos(\mathrm{CLS},\;\overline{\mathrm{node}})$")
    title = r"CLS similarity to mean of node representations"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
