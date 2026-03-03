"""Section C: Early-Layer Symbolic Specialisation (Figures 10-12)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..chemistry import (
    label_head_focus_coarse,
    plot_molecule_attention_tripanel,
    pca_2d,
    LABELS_COARSE,
    LABELS_FOCUS_EXPANDED,
)


def fig10_symbolic_head_attention(
    A_list_per_mol: list,
    smiles_list: list[str],
    layer: int = 1,
    head: int = 24,
    mol_indices: list[int] | None = None,
) -> list[Figure]:
    """Fig 10: 3-panel attention visualisation of L1,H24 for selected molecules.

    Each molecule produces one Figure with three panels:
      (1) plain RDKit molecule, (2) attention inflow overlay,
      (3) node-to-node attention matrix.

    Parameters
    ----------
    A_list_per_mol : list
        ``A_list_per_mol[i][layer]`` is ``[H, T, T]`` attention for molecule *i*.
    smiles_list : list[str]
        SMILES strings corresponding to each molecule.
    layer, head : int
        Which layer and head to visualise.
    mol_indices : list[int], optional
        Indices into *smiles_list* to show (default ``[0, 5, 80]``).
    """
    if mol_indices is None:
        mol_indices = [0, 5, 80]
    # Clamp indices to available range
    mol_indices = [i for i in mol_indices if i < len(smiles_list)]

    figs = []
    for idx in mol_indices:
        smi = smiles_list[idx]
        A_layer = A_list_per_mol[idx][layer]  # [H, T, T]
        fig = plot_molecule_attention_tripanel(
            smi, A_layer, head_idx=head, layer_idx=layer, right_panel="matrix",
        )
        fig.suptitle(
            f"L{layer} H{head} — Molecule {idx} (symbolic head)",
            fontsize=13, fontweight="bold", y=1.02,
        )
        figs.append(fig)
    return figs


def fig11_pca_head_output(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    layer: int = 1,
    head: int = 24,
    N: int | None = None,
    title: str | None = None,
) -> Figure:
    """Fig 11: PCA of query-averaged head output (A@V pooled over node queries).

    Parameters
    ----------
    embeddings : np.ndarray  [N_molecules, d]
        Pooled head output vectors (mean over node queries of A@V).
    labels : list[str]  [N_molecules]
        Per-molecule category labels from ``label_head_focus_coarse``.
    """
    Z = pca_2d(embeddings)
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be 1D for fig11/fig12.")

    if np.issubdtype(labels_arr.dtype, np.integer):
        label_names = LABELS_FOCUS_EXPANDED
        y = labels_arr.astype(int)
    else:
        unique = sorted(set(labels_arr.tolist()))
        label_names = unique
        name_to_id = {lab: i for i, lab in enumerate(unique)}
        y = np.array([name_to_id[str(v)] for v in labels_arr], dtype=int)

    n_lab = max(len(label_names), 1)
    cmap = plt.cm.get_cmap("tab20", n_lab)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for i, lab in enumerate(label_names):
        mask = (y == i)
        if not np.any(mask):
            continue
        ax.scatter(
            Z[mask, 0], Z[mask, 1], s=18, alpha=0.78, color=cmap(i),
            label=lab, edgecolors="white", linewidths=0.2,
        )
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    if title is None:
        title = rf"PCA of $A \cdot V$ head output — L{layer} H{head}"
        if N is not None:
            title += f" ({N} molecules)"
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, loc="best", framealpha=0.92, ncol=1)
    ax.grid(True, alpha=0.16)
    fig.tight_layout()
    return fig


def fig12_pca_head_output_second(
    embeddings: np.ndarray,
    labels: list[str],
    layer: int = 0,
    head: int = 0,
    N: int | None = None,
    title: str | None = None,
) -> Figure:
    """Fig 12: PCA of query-averaged head output for a second early-layer head."""
    return fig11_pca_head_output(
        embeddings, labels, layer=layer, head=head, N=N, title=title,
    )
