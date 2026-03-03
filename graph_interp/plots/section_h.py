"""Section H: Head Ablation — L1, H24 (Figures 33-36)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import LAYER_LABEL, HEAD_LABEL


def fig33_ablation_tv_across_layers(
    mean_tv: np.ndarray, max_tv: np.ndarray,
    ablated_layer: int = 1, N: int | None = None, **kw,
) -> Figure:
    """Fig 33: Attention TV distance from ablating L1,H24 across layers."""
    L = len(mean_tv)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, mean_tv, marker="o", label="Mean TV", markersize=5, color="tab:blue")
    ax.plot(x, max_tv, marker="s", label="Max TV", markersize=4, color="tab:red", alpha=0.7)
    ax.fill_between(x, 0, mean_tv, alpha=0.15, color="tab:blue")
    ax.axvline(ablated_layer, color="gray", linestyle="--", linewidth=1,
               label=f"Ablated layer {ablated_layer}")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel("TV distance")
    title = r"Attention effect of ablating L1,H24"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig34_ablation_effect_heatmap(
    effect_LH: np.ndarray,
    ablate_heads: dict[int, list[int]] | None = None,
    metric: str = "TV",
    N: int | None = None,
    **kw,
) -> Figure:
    """Fig 34: Per-head downstream attention effect heatmap from ablation."""
    if ablate_heads is None:
        ablate_heads = {1: [24]}

    L, H = effect_LH.shape
    max_abl_layer = max(ablate_heads.keys()) if ablate_heads else -1

    masked = effect_LH.astype(float).copy()
    if max_abl_layer >= 0:
        masked[:max_abl_layer + 1, :] = np.nan
    finite_mask = np.isfinite(masked)
    vmax = float(np.nanmax(masked)) if finite_mask.any() else float(np.nanmax(effect_LH))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0.0, vmax=vmax)
    for layer_idx, heads in ablate_heads.items():
        for h in heads:
            ax.plot(h, layer_idx, "kx", markersize=10, markeredgewidth=2.5)

    title = (
        r"Ablation: L1,H24"
        "\n"
        rf"Downstream attention {metric} divergence ($\times$ = ablated head)"
    )
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xlabel(HEAD_LABEL)
    ax.set_ylabel(LAYER_LABEL)
    ax.set_xticks(range(0, H, 4))
    ax.set_yticks(range(L))
    fig.colorbar(im, ax=ax, label=f"{metric} distance", fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig35_perturbation_concentration(gini_per_layer: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 35: Gini coefficient of node-level perturbation vs layer."""
    L = len(gini_per_layer)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, gini_per_layer, marker="o", markersize=5, color="tab:purple")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel("Gini coefficient")
    title = r"Perturbation concentration from ablating L1,H24"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig36_cls_perturbation(
    cls_perturbation: np.ndarray,
    cls_perturbation_err: np.ndarray | None = None,
    N: int | None = None,
    **kw,
) -> Figure:
    """Fig 36: CLS hidden state L2 perturbation vs layer."""
    L = len(cls_perturbation)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if cls_perturbation_err is not None:
        ax.errorbar(
            x, cls_perturbation, yerr=cls_perturbation_err,
            marker="o", markersize=5, color="#c44536", ecolor="#8f2d56",
            elinewidth=1.2, capsize=3, linewidth=2,
        )
        ax.fill_between(
            x, cls_perturbation - cls_perturbation_err,
            cls_perturbation + cls_perturbation_err,
            color="#c44536", alpha=0.15,
        )
    else:
        ax.plot(x, cls_perturbation, marker="o", markersize=5, color="#c44536", linewidth=2)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"CLS $\ell_2$ perturbation")
    title = r"CLS token perturbation from ablating L1,H24"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
