"""Section G: Representation Dynamics (Figures 29-32)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import LAYER_LABEL


def fig29_node_similarity_across_layers(
    mean_sim: np.ndarray, min_sim: np.ndarray, max_sim: np.ndarray,
    std_sim: np.ndarray | None = None,
    N: int | None = None, **kw,
) -> Figure:
    """Fig 29: Mean pairwise cosine similarity with uncertainty bands."""
    L = len(mean_sim)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    main = "#3b6ea8"
    ax.fill_between(x, min_sim, max_sim, alpha=0.14, color=main, label="Min–Max range")
    if std_sim is not None:
        ax.fill_between(x, mean_sim - std_sim, mean_sim + std_sim, alpha=0.24, color=main, label="Mean ± std")
    ax.plot(x, mean_sim, marker="o", label="Mean", markersize=5, color=main, linewidth=2.2)
    ax.plot(x, max_sim, marker="s", linestyle="--", color="#1f355e", linewidth=1.2, alpha=0.7, label="Max pair")
    ax.plot(x, min_sim, marker="^", linestyle="--", color="#7da9d9", linewidth=1.2, alpha=0.9, label="Min pair")
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel("Pairwise cosine similarity")
    title = "T1: Node representation similarity"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig30_delta_similarity(delta_sim: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 30: Change in mean similarity per layer transition."""
    L = len(delta_sim)
    x = np.arange(L)

    colors = ["#d32f2f" if d > 0 else "#1976d2" for d in delta_sim]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x, delta_sim, color=colors, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Layer transition $\ell \to \ell\!+\!1$")
    ax.set_ylabel(r"$\Delta$ mean cosine similarity")
    title = "Change in node similarity per layer transition\n(red = homogenizing, blue = differentiating)"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}→{i+1}" for i in range(L)], fontsize=7, rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig31_sublayer_similarity(
    sim_input: np.ndarray, sim_attn: np.ndarray, sim_ffn: np.ndarray,
    N: int | None = None, **kw,
) -> Figure:
    """Fig 31: Similarity at sublayer stages: input, after attention, after FFN."""
    L = len(sim_attn)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    c_input = "#2f4f4f"
    c_attn = "#6a4c93"
    c_ffn = "#1982c4"
    if not np.all(np.isnan(sim_input)):
        ax.plot(x, sim_input, marker="^", label="Layer input", markersize=4.5, linewidth=1.8, color=c_input)
    ax.plot(x, sim_attn, marker="o", label="After attention", markersize=4.5, linewidth=2.0, color=c_attn)
    ax.plot(x, sim_ffn, marker="s", label="After FFN", markersize=4.5, linewidth=2.0, color=c_ffn)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel("Mean pairwise cosine similarity")
    title = "Node similarity at sublayer stages"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(framealpha=0.92, fontsize=9)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def fig32_attn_vs_ffn_contribution(
    sim_input: np.ndarray, sim_attn: np.ndarray, sim_ffn: np.ndarray,
    N: int | None = None, **kw,
) -> Figure:
    """Fig 32: Per-layer contribution to homogenisation: attention vs FFN."""
    L = len(sim_attn)
    x = np.arange(L)
    width = 0.35

    delta_attn = sim_attn - sim_input
    delta_ffn = sim_ffn - sim_attn

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - width / 2, delta_attn, width, label=r"$\Delta$ Attention", color="#bc5090", alpha=0.86)
    ax.bar(x + width / 2, delta_ffn, width, label=r"$\Delta$ FFN", color="#0081a7", alpha=0.86)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(LAYER_LABEL)
    ax.set_ylabel(r"$\Delta$ mean pairwise cosine")
    title = r"Homogenisation contribution: Attention vs FFN"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(framealpha=0.92, fontsize=9)
    ax.grid(True, alpha=0.22, axis="y")
    fig.tight_layout()
    return fig
