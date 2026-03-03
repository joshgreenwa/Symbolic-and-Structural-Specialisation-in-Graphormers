"""Section B: Dot-Product vs Bias Dominance (Figures 6-9)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import plot_scatter_colored_by_layer, _flatten_LH, LAYER_LABEL, HEAD_LABEL


def fig06_logit_spread(dot_std_per_layer: np.ndarray, bias_std_per_layer: np.ndarray,
                       dot_std_err: np.ndarray | None = None,
                       bias_std_err: np.ndarray | None = None,
                       N: int | None = None, **kw) -> Figure:
    """Fig 6: Std of logits across keys vs layer."""
    L = len(dot_std_per_layer)
    x = np.arange(L)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, dot_std_per_layer, marker="o", label=r"$\mathrm{std}(\mathbf{d})$ (dot-product)", markersize=4, color="tab:blue")
    ax.plot(x, bias_std_per_layer, marker="s", label=r"$\mathrm{std}(\mathbf{b})$ (bias)", markersize=4, color="tab:orange")
    if dot_std_err is not None:
        ax.fill_between(x, dot_std_per_layer - dot_std_err, dot_std_per_layer + dot_std_err,
                        alpha=0.2, color="tab:blue")
    if bias_std_err is not None:
        ax.fill_between(x, bias_std_per_layer - bias_std_err, bias_std_per_layer + bias_std_err,
                        alpha=0.2, color="tab:orange")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"Std of logits across keys")
    title = r"Logit spread: $\mathrm{std}(\mathbf{d})$ vs $\mathrm{std}(\mathbf{b})$"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig07_variance_ratio(var_ratio_per_layer: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 7: Var(dot)/Var(bias) bar chart per layer."""
    L = len(var_ratio_per_layer)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    dot_color = "#2a9d8f"
    bias_color = "#b23a48"
    colors = [dot_color if r >= 1.0 else bias_color for r in var_ratio_per_layer]
    ax.bar(np.arange(L), var_ratio_per_layer, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label=r"$r_\ell = 1$")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"$r_\ell = \mathrm{Var}(\mathbf{d}_\ell) \;/\; \mathrm{Var}(\mathbf{b}_\ell)$")
    title = r"Dot/bias variance ratio $r_\ell$ across layers"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(np.arange(L))
    ax.legend(handles=[
        Patch(facecolor=dot_color, edgecolor="black", linewidth=0.5, label=r"Dot-variance dominant ($r_\ell ≥ 1$)"),
        Patch(facecolor=bias_color, edgecolor="black", linewidth=0.5, label=r"Bias-variance dominant ($r_\ell < 1$)"),
        Line2D([0], [0], linestyle="--", color="gray", linewidth=1.0, label=r"$r_\ell = 1$"),
    ], fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    # Equation annotation box
    ax.text(0.97, 0.95, r"$r_\ell = \mathrm{Var}(\mathbf{d}) \;/\; \mathrm{Var}(\mathbf{b})$",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    return fig


def fig08_structural_dom_vs_bias_ratio(
    mean_struct: np.ndarray, mean_sym: np.ndarray,
    log_r: np.ndarray, N: int | None = None, **kw,
) -> Figure:
    """Fig 8: (s_struct - s_sym) vs log10(r), colored by layer."""
    diff = mean_struct - mean_sym
    title = r"$(\bar{s}_{\mathrm{struct}} - \bar{s}_{\mathrm{sym}})$ vs $\log_{10}\!\left(\mathrm{std}(\mathbf{b})/\mathrm{std}(\mathbf{d})\right)$"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_scatter_colored_by_layer(
        log_r, diff,
        xlabel=r"$\log_{10}\!\left(\mathrm{std}(\mathbf{b})/\mathrm{std}(\mathbf{d})\right)$",
        ylabel=r"$\bar{s}_{\mathrm{struct}} - \bar{s}_{\mathrm{sym}}$",
        title=title, **kw,
    )


def fig09_bias_ratio_per_head(log_r: np.ndarray, head_indices: list[int] | None = None,
                               N: int | None = None, **kw) -> Figure:
    """Fig 9: log10(r) vs layer, one line per selected head."""
    L, H = log_r.shape
    if head_indices is None:
        head_indices = list(range(0, H, max(1, H // 6)))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(L)
    palette = plt.cm.cividis(np.linspace(0.15, 0.95, len(head_indices)))
    for color, h in zip(palette, head_indices):
        ax.plot(x, log_r[:, h], marker="o", markersize=3.5, linewidth=1.6, label=f"H{h}", color=color)
    ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"$\log_{10}\!\left(\mathrm{std}(\mathbf{b}^{(\ell,h)})/\mathrm{std}(\mathbf{d}^{(\ell,h)})\right)$")
    title = r"Bias std ratio (log$_{10}$) per head"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    return fig
