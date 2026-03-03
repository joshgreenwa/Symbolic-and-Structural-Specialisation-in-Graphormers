"""Section I: KL Divergence Scores (Figures 37-41)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats as scipy_stats

from . import plot_LH_heatmap, plot_scatter_colored_by_layer, _flatten_LH, LAYER_LABEL


def fig37_kl_dot_only_heatmap(mean_kl_dot: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 37: KL divergence heatmap — dot-product only."""
    title = r"Normalised $D_{\mathrm{KL}}(\mathrm{full} \| \mathrm{dot\text{-}only})$"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_LH_heatmap(
        mean_kl_dot, title=title,
        vmin=0.0, vmax=1.0, cbar_label=r"$D_{\mathrm{KL}}$ (normalised)", **kw,
    )


def fig38_kl_bias_only_heatmap(mean_kl_bias: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 38: KL divergence heatmap — bias only."""
    title = r"Normalised $D_{\mathrm{KL}}(\mathrm{full} \| \mathrm{bias\text{-}only})$"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_LH_heatmap(
        mean_kl_bias, title=title,
        vmin=0.0, vmax=1.0, cbar_label=r"$D_{\mathrm{KL}}$ (normalised)", **kw,
    )


def fig39_structural_dom_vs_kl_diff(
    mean_struct: np.ndarray, mean_sym: np.ndarray,
    mean_kl_dot: np.ndarray, mean_kl_bias: np.ndarray,
    N: int | None = None, **kw,
) -> Figure:
    """Fig 39: (s_struct - s_sym) vs (KL(dot-only) - KL(bias-only)) with Pearson r."""
    diff_score = mean_struct - mean_sym
    diff_kl = mean_kl_dot - mean_kl_bias

    xf, layers, _ = _flatten_LH(diff_kl)
    yf, _, _ = _flatten_LH(diff_score)

    r, p = scipy_stats.pearsonr(xf, yf)

    fig, ax = plt.subplots(figsize=(7, 6))
    L = diff_score.shape[0]
    cmap = plt.cm.get_cmap("viridis", L)
    sc = ax.scatter(xf, yf, c=layers, cmap=cmap, s=18, alpha=0.8)
    fig.colorbar(sc, ax=ax, label=r"Layer $\ell$")
    ax.set_xlabel(r"$D_{\mathrm{KL}}(\mathrm{dot}) - D_{\mathrm{KL}}(\mathrm{bias})$")
    ax.set_ylabel(r"$\bar{s}_{\mathrm{struct}} - \bar{s}_{\mathrm{sym}}$")
    title = rf"Score vs KL difference (Pearson $r={r:.3f}$, $p={p:.1e}$)"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig40_kl_disagreement(
    mean_struct: np.ndarray, mean_sym: np.ndarray,
    mean_kl_dot: np.ndarray, mean_kl_bias: np.ndarray,
    N: int | None = None, **kw,
) -> Figure:
    """Fig 40: Mean |residual| of score vs KL across layers."""
    diff_score = mean_struct - mean_sym
    diff_kl = mean_kl_dot - mean_kl_bias
    resid = np.abs(diff_score - diff_kl)
    resid_struct = np.abs(mean_struct - (1 - mean_kl_dot))
    resid_sym = np.abs(mean_sym - (1 - mean_kl_bias))

    L = resid.shape[0]
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(x, resid.mean(axis=1), marker="o", label="|resid| difference", markersize=4.5, linewidth=2.0, color="#3f7cac")
    ax.plot(x, resid_struct.mean(axis=1), marker="s", label="|resid| structural", markersize=4.5, linewidth=1.8, color="#2a9d8f")
    ax.plot(x, resid_sym.mean(axis=1), marker="^", label="|resid| symbolic", markersize=4.5, linewidth=1.8, color="#e76f51")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel("Mean |residual|")
    title = "Score vs KL disagreement across layers"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def fig41_signed_bias(
    mean_struct: np.ndarray, mean_sym: np.ndarray,
    mean_kl_dot: np.ndarray, mean_kl_bias: np.ndarray,
    N: int | None = None, **kw,
) -> Figure:
    """Fig 41: Signed bias (residual direction) per layer."""
    resid_struct = (mean_struct - (1 - mean_kl_dot)).mean(axis=1)
    resid_sym = (mean_sym - (1 - mean_kl_bias)).mean(axis=1)
    L = len(resid_struct)
    x = np.arange(L)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - width / 2, resid_struct, width, label="Structural residual", color="#355070", alpha=0.9)
    ax.bar(x + width / 2, resid_sym, width, label="Symbolic residual", color="#b56576", alpha=0.9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel("Mean signed residual")
    title = "Signed bias per layer"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(framealpha=0.92, fontsize=9)
    ax.grid(True, alpha=0.22, axis="y")
    fig.tight_layout()
    return fig
