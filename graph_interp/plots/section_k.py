"""Section K: Entropy Diagnostics (Figures 45-48)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import plot_LH_heatmap, plot_scatter_colored_by_layer, _flatten_LH


def fig45_entropy_heatmap(mean_ent_bits: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 45: Attention entropy heatmap (node->node)."""
    title = r"Attention entropy $H^{(\ell,h)}$ (bits)"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_LH_heatmap(
        mean_ent_bits, title=title,
        cmap="YlOrRd", cbar_label="Entropy (bits)", **kw,
    )


def fig46_normalized_entropy_heatmap(mean_ent_norm: np.ndarray, N: int | None = None, **kw) -> Figure:
    """Fig 46: Normalised attention entropy heatmap."""
    title = r"Normalised attention entropy $H / H_{\max}$"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_LH_heatmap(
        mean_ent_norm, title=title,
        vmin=0.0, vmax=1.0, cmap="YlOrRd", cbar_label=r"$H / H_{\max}$", **kw,
    )


def fig47_struct_vs_entropy(mean_struct: np.ndarray, mean_ent_bits: np.ndarray,
                             N: int | None = None, **kw) -> Figure:
    """Fig 47: Structural score vs entropy scatter, coloured by layer."""
    title = r"$\bar{s}_{\mathrm{struct}}$ vs entropy"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_scatter_colored_by_layer(
        mean_ent_bits, mean_struct,
        xlabel="Entropy (bits)", ylabel=r"$\bar{s}_{\mathrm{struct}}$",
        title=title, **kw,
    )


def fig48_symbolic_vs_entropy(mean_sym: np.ndarray, mean_ent_bits: np.ndarray,
                               N: int | None = None, **kw) -> Figure:
    """Fig 48: Symbolic score vs entropy scatter, coloured by layer."""
    title = r"$\bar{s}_{\mathrm{sym}}$ vs entropy"
    if N is not None:
        title += f" ({N} graphs)"
    return plot_scatter_colored_by_layer(
        mean_ent_bits, mean_sym,
        xlabel="Entropy (bits)", ylabel=r"$\bar{s}_{\mathrm{sym}}$",
        title=title, **kw,
    )
