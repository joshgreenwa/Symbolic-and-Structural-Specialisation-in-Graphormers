"""RDKit utilities: atom feature labels, molecule coordinates, SMARTS patterns."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor


# ── SMARTS patterns ──────────────────────────────────────────
_PATT_CARBONYL_O = Chem.MolFromSmarts("[O]=[C]")
_PATT_HYDROXYL_O = Chem.MolFromSmarts("[O;H1]")
_PATT_ESTER_O = Chem.MolFromSmarts("[O]([C])[C](=O)")
_PATT_CARBOXYL_O = Chem.MolFromSmarts("[O;H1][C](=O)")


def _patt(smarts: str):
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"Bad SMARTS: {smarts}")
    return patt


def _anchor_idx(patt):
    for i, atom in enumerate(patt.GetAtoms()):
        if atom.GetAtomMapNum() == 1:
            return i
    raise ValueError("Pattern missing atom-map anchor ':1'.")


# Expanded focus-labelling SMARTS (anchor atom has map number :1)
_P_CARBONYL_O = _patt("[O:1]=[C]")
_P_CARBOXY_O = _patt("[O-:1][C](=O)")
_P_ESTER_O = _patt("[O:1][C](=O)")
_P_N_NITRO = _patt("[N+:1](=O)[O-]")
_P_N_NITRILE = _patt("[N:1]#[C]")
_P_N_AMIDE = _patt("[N:1][C](=O)")
_P_N_AROM = _patt("[n:1]")

LABELS_COARSE = ["focus=carbonyl O", "focus=hydroxyl O", "focus=nitrogen", "focus=other"]
LABELS_EXPANDED = [
    "C (sp3)", "C (sp2/arom)", "N", "O (hydroxyl)", "O (carbonyl)",
    "O (ester/carboxyl)", "F", "S", "Cl", "Br", "other",
]

LABELS_FOCUS_EXPANDED = [
    "O: carbonyl",
    "O: hydroxyl",
    "O: ester/carboxyl",
    "O: other",
    "N: amide",
    "N: aromatic",
    "N: nitro",
    "N: nitrile",
    "N: other",
    "S: sulfur",
    "X: halogen",
    "P: phosphorus",
    "Ring: junction",
    "Ring: aromatic",
    "Ring: aliphatic",
    "Branch: degree>=3",
    "Charge: +",
    "Charge: -",
    "other/diffuse",
]


def get_molecule(smiles: str):
    """Parse SMILES into an RDKit Mol object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return mol


def get_atom_symbols(smiles: str) -> list[str]:
    """Return per-atom element symbols."""
    mol = get_molecule(smiles)
    return [a.GetSymbol() for a in mol.GetAtoms()]


def get_molecule_coords(smiles: str):
    """2D coordinates from RDKit. Returns dict {atom_idx: (x, y)}."""
    mol = get_molecule(smiles)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    coords = {}
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords[i] = (pos.x, pos.y)
    return coords


def get_bonds(smiles: str) -> list[tuple[int, int]]:
    """Return list of (atom_i, atom_j) bonds."""
    mol = get_molecule(smiles)
    return [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]


def _focus_atoms(p_atom: np.ndarray, focus_mass: float = 0.80):
    """Indices of atoms carrying *focus_mass* fraction of attention."""
    order = np.argsort(-p_atom)
    cum = 0.0
    sel = []
    for idx in order:
        sel.append(int(idx))
        cum += float(p_atom[idx])
        if cum >= focus_mass:
            break
    return sel


def label_head_focus_coarse(smiles: str, p_atom: np.ndarray, focus_mass: float = 0.80) -> str:
    """Coarse label: carbonyl O / hydroxyl O / nitrogen / other."""
    mol = get_molecule(smiles)
    focus = _focus_atoms(p_atom, focus_mass)

    carbonyl_O = {m[0] for m in mol.GetSubstructMatches(_PATT_CARBONYL_O, uniquify=True)}
    hydroxyl_O = {m[0] for m in mol.GetSubstructMatches(_PATT_HYDROXYL_O, uniquify=True)}

    mass = {"carbonyl": 0.0, "hydroxyl": 0.0, "nitrogen": 0.0, "other": 0.0}
    for ai in focus:
        a = mol.GetAtomWithIdx(ai)
        sym = a.GetSymbol()
        w = float(p_atom[ai])
        if sym == "O" and ai in carbonyl_O:
            mass["carbonyl"] += w
        elif sym == "O" and ai in hydroxyl_O:
            mass["hydroxyl"] += w
        elif sym in ("N", "n"):
            mass["nitrogen"] += w
        else:
            mass["other"] += w

    best = max(mass.items(), key=lambda kv: kv[1])[0]
    return {
        "carbonyl": "focus=carbonyl O",
        "hydroxyl": "focus=hydroxyl O",
        "nitrogen": "focus=nitrogen",
    }.get(best, "focus=other")


def atom_categories_expanded(mol) -> list[str | None]:
    """Expanded atom-level chemistry tags used for head-focus PCA coloring.

    Priority:
      1) high-signal O/N functional SMARTS
      2) element-level fallbacks (O/N/S/P/halogen)
      3) structural tags (ring/branch)
      4) formal charge tags
      5) None (mapped to other/diffuse later)
    """
    n = mol.GetNumAtoms()
    cat: list[str | None] = [None] * n

    def assign_from_pattern(patt, label: str):
        ai = _anchor_idx(patt)
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            idx = int(match[ai])
            if 0 <= idx < n and cat[idx] is None:
                cat[idx] = label

    assign_from_pattern(_P_CARBONYL_O, "O: carbonyl")
    assign_from_pattern(_P_CARBOXY_O, "O: ester/carboxyl")
    assign_from_pattern(_P_ESTER_O, "O: ester/carboxyl")
    assign_from_pattern(_P_N_NITRO, "N: nitro")
    assign_from_pattern(_P_N_NITRILE, "N: nitrile")
    assign_from_pattern(_P_N_AMIDE, "N: amide")
    assign_from_pattern(_P_N_AROM, "N: aromatic")

    ringinfo = mol.GetRingInfo()
    num_atom_rings = [ringinfo.NumAtomRings(i) for i in range(n)]

    for i, atom in enumerate(mol.GetAtoms()):
        if cat[i] is not None:
            continue
        sym = atom.GetSymbol()
        if sym == "O":
            cat[i] = "O: hydroxyl" if atom.GetTotalNumHs() >= 1 else "O: other"
        elif sym in ("N", "n"):
            cat[i] = "N: other"
        elif sym in ("S", "s"):
            cat[i] = "S: sulfur"
        elif sym in ("P", "p"):
            cat[i] = "P: phosphorus"
        elif sym in ("F", "Cl", "Br", "I"):
            cat[i] = "X: halogen"

    for i, atom in enumerate(mol.GetAtoms()):
        if cat[i] is not None:
            continue
        if num_atom_rings[i] >= 2:
            cat[i] = "Ring: junction"
            continue
        if num_atom_rings[i] >= 1:
            cat[i] = "Ring: aromatic" if atom.GetIsAromatic() else "Ring: aliphatic"
            continue
        if int(atom.GetDegree()) >= 3:
            cat[i] = "Branch: degree>=3"
            continue

    for i, atom in enumerate(mol.GetAtoms()):
        if cat[i] is not None:
            continue
        formal_charge = int(atom.GetFormalCharge())
        if formal_charge > 0:
            cat[i] = "Charge: +"
        elif formal_charge < 0:
            cat[i] = "Charge: -"

    return cat


def label_head_focus_expanded(
    smiles: str,
    p_atom: np.ndarray,
    focus_mass: float = 0.75,
    diffuse_thresh: float = 0.35,
) -> str:
    """Expanded chemistry focus label from per-atom inbound attention mass."""
    mol = get_molecule(smiles)
    cat_map = atom_categories_expanded(mol)
    focus = _focus_atoms(np.asarray(p_atom, dtype=float), focus_mass)

    mass_by_cat = {lab: 0.0 for lab in LABELS_FOCUS_EXPANDED}
    for ai in focus:
        w = float(p_atom[ai])
        lab = cat_map[ai] if ai < len(cat_map) else None
        if lab is None:
            lab = "other/diffuse"
        mass_by_cat[lab] += w

    best_label, best_mass = max(mass_by_cat.items(), key=lambda kv: kv[1])
    if best_mass < float(diffuse_thresh):
        return "other/diffuse"
    return best_label


def label_heads_from_attention(
    smiles: str,
    A_layer,
    focus_mass: float = 0.75,
    diffuse_thresh: float = 0.35,
) -> list[str]:
    """Head-specific focus labels from one layer attention tensor [H,T,T]."""
    A_np = A_layer.detach().float().cpu().numpy() if hasattr(A_layer, "detach") else np.asarray(A_layer)
    if A_np.ndim != 3:
        raise ValueError(f"A_layer must be [H,T,T], got shape {A_np.shape}")
    H, T, _ = A_np.shape
    if T < 2:
        return ["other/diffuse"] * H

    A_nn = A_np[:, 1:, 1:]  # [H,n,n]
    inflow = A_nn.sum(axis=1)  # [H,n], inbound over node queries
    denom = np.clip(inflow.sum(axis=1, keepdims=True), 1e-12, None)
    p_atom = inflow / denom

    labels = []
    for h in range(H):
        labels.append(label_head_focus_expanded(
            smiles, p_atom[h], focus_mass=focus_mass, diffuse_thresh=diffuse_thresh
        ))
    return labels


def label_atom_expanded(smiles: str) -> list[str]:
    """Expanded per-atom labels for PCA coloring."""
    mol = get_molecule(smiles)
    carbonyl_O = {m[0] for m in mol.GetSubstructMatches(_PATT_CARBONYL_O, uniquify=True)}
    hydroxyl_O = {m[0] for m in mol.GetSubstructMatches(_PATT_HYDROXYL_O, uniquify=True)}
    ester_O = set()
    if _PATT_ESTER_O is not None:
        ester_O = {m[0] for m in mol.GetSubstructMatches(_PATT_ESTER_O, uniquify=True)}
    carboxyl_O = set()
    if _PATT_CARBOXYL_O is not None:
        carboxyl_O = {m[0] for m in mol.GetSubstructMatches(_PATT_CARBOXYL_O, uniquify=True)}

    labels = []
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        idx = a.GetIdx()
        if sym == "C":
            if a.GetIsAromatic() or a.GetHybridization().name == "SP2":
                labels.append("C (sp2/arom)")
            else:
                labels.append("C (sp3)")
        elif sym == "N":
            labels.append("N")
        elif sym == "O":
            if idx in carbonyl_O:
                labels.append("O (carbonyl)")
            elif idx in hydroxyl_O:
                labels.append("O (hydroxyl)")
            elif idx in ester_O or idx in carboxyl_O:
                labels.append("O (ester/carboxyl)")
            else:
                labels.append("O (hydroxyl)")  # default O
        elif sym == "F":
            labels.append("F")
        elif sym == "S":
            labels.append("S")
        elif sym == "Cl":
            labels.append("Cl")
        elif sym == "Br":
            labels.append("Br")
        else:
            labels.append("other")
    return labels


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Simple 2-component PCA. X: [N, d] -> [N, 2]."""
    Xc = X - X.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(Xc)) < 1e-10:
        return np.zeros((X.shape[0], 2), dtype=float)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt.T[:, :2]


# ── Molecule visualization for 3-panel attention displays ────


def draw_molecule_plain(smiles, label_mode="atom", figsize=(4.2, 4.2), dpi=220):
    """Render molecule with atom-index labels using RDKit's MolDraw2DCairo.

    Args:
        smiles: SMILES string
        label_mode: "atom" labels 0..n-1, "token" labels 1..n
        figsize: figure size in inches
        dpi: resolution

    Returns:
        PIL.Image of the rendered molecule
    """
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    import io

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES.")
    rdDepictor.Compute2DCoords(mol)
    n_atoms = mol.GetNumAtoms()
    lab = (lambda i: str(i)) if label_mode == "atom" else (lambda i: str(i + 1))
    W, H = int(figsize[0] * dpi), int(figsize[1] * dpi)
    drawer = rdMolDraw2D.MolDraw2DCairo(W, H)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    for i in range(n_atoms):
        opts.atomLabels[i] = lab(i)
    drawer.DrawMolecule(mol, legend="")
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png))


def draw_molecule_attention_overlay(smiles, inflow, label_mode="atom", log_scale=True,
                                     radius_range=(0.22, 0.62), cmap_name="Reds",
                                     figsize=(4.2, 4.2), dpi=220):
    """Render molecule with attention-weighted atom highlights.

    Args:
        smiles: SMILES string
        inflow: [n_atoms] array of attention weights
        label_mode: "atom" labels 0..n-1, "token" labels 1..n
        log_scale: apply log1p transform before mapping colours
        radius_range: (min, max) highlight radius
        cmap_name: matplotlib colormap name
        figsize: figure size in inches
        dpi: resolution

    Returns:
        (PIL.Image, vmin, vmax, cmap_name)
    """
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    import io
    import matplotlib.pyplot as plt

    inflow = np.asarray(inflow, dtype=np.float32)
    v = np.log1p(inflow) if log_scale else inflow
    vmin, vmax = float(v.min()), float(v.max())
    denom = (vmax - vmin) if vmax > vmin else 1.0
    vn = (v - vmin) / denom
    cmap = plt.get_cmap(cmap_name)
    cols = cmap(vn)
    atom_colors = {i: (float(cols[i, 0]), float(cols[i, 1]), float(cols[i, 2])) for i in range(len(vn))}
    r0, r1 = radius_range
    atom_radii = {i: float(r0 + (r1 - r0) * vn[i]) for i in range(len(vn))}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES.")
    rdDepictor.Compute2DCoords(mol)
    n_atoms = mol.GetNumAtoms()
    if n_atoms != len(inflow):
        raise RuntimeError(f"Atom count mismatch: RDKit {n_atoms}, inflow {len(inflow)}")
    lab = (lambda i: str(i)) if label_mode == "atom" else (lambda i: str(i + 1))
    W, H = int(figsize[0] * dpi), int(figsize[1] * dpi)
    drawer = rdMolDraw2D.MolDraw2DCairo(W, H)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    for i in range(n_atoms):
        opts.atomLabels[i] = lab(i)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        legend=""
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png)), vmin, vmax, cmap_name


def _get_head_matrix_nodekeys(A_all, head_idx, mode="node_cond", eps=1e-12):
    """Extract node-conditioned attention matrix for one head.

    Args:
        A_all: [H, T, T] attention tensor for one layer
        head_idx: which head to extract
        mode: "raw" returns A[1:,1:] directly; "node_cond" row-normalises

    Returns:
        [n, n] tensor (node queries x node keys)
    """
    A = A_all[head_idx]  # [T, T]
    T = A.shape[0]
    A_q_nodes = A[1:, 1:]  # node queries to node keys
    if mode == "raw":
        return A_q_nodes
    if mode == "node_cond":
        if hasattr(A_q_nodes, "detach"):
            row_mass = A_q_nodes.sum(dim=-1, keepdim=True).clamp_min(eps)
            return A_q_nodes / row_mass
        row_mass = np.clip(A_q_nodes.sum(axis=-1, keepdims=True), eps, None)
        return A_q_nodes / row_mass
    raise ValueError(f"mode must be 'raw' or 'node_cond', got {mode}")


def _sparsify_edges(M, method="mass", k=2, mass=0.9, threshold=0.05, drop_self=True):
    """Sparsify attention matrix to list of (query, key, weight) edges.

    Args:
        M: [Q, n] attention matrix
        method: "topk", "mass", or "threshold"
        k: top-k count (for method="topk")
        mass: cumulative mass cutoff (for method="mass")
        threshold: minimum weight (for method="threshold")
        drop_self: if True, ignore self-attention (diagonal)

    Returns:
        list of (query_idx, key_idx, weight) tuples
    """
    import torch

    Q, n = M.shape
    M_np = M.detach().float().cpu().numpy() if torch.is_tensor(M) else np.asarray(M)
    edges = []
    for q in range(Q):
        row = M_np[q].copy()
        if drop_self and q < n:
            row[q] = -np.inf
        if method == "topk":
            kk = min(k, n)
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.argsort(-row[idx])]
            for j in idx:
                w = row[j]
                if np.isfinite(w) and w > 0:
                    edges.append((q, int(j), float(w)))
        elif method == "mass":
            order = np.argsort(-row)
            cum = 0.0
            for j in order:
                w = row[j]
                if not np.isfinite(w) or w <= 0:
                    continue
                edges.append((q, int(j), float(w)))
                cum += float(w)
                if cum >= mass:
                    break
        elif method == "threshold":
            idx = np.where(row >= threshold)[0]
            idx = idx[np.argsort(-row[idx])]
            for j in idx:
                edges.append((q, int(j), float(row[j])))
    return edges


def draw_attention_coupling_graph(ax, smiles, A_all, head_idx,
                                   mode="node_cond", sparsify="mass",
                                   k=2, mass=0.9, threshold=0.05,
                                   drop_self=True, include_bonds=True,
                                   width_scale=7.5):
    """Draw attention coupling graph (query->key edges) on RDKit 2D layout.

    Args:
        ax: matplotlib Axes to draw on
        smiles: SMILES string
        A_all: [H, T, T] attention tensor for one layer
        head_idx: which head to visualise
        mode: "node_cond" or "raw"
        sparsify: "mass", "topk", or "threshold"
        k: top-k count (for sparsify="topk")
        mass: cumulative mass cutoff (for sparsify="mass")
        threshold: minimum weight (for sparsify="threshold")
        drop_self: if True, ignore self-attention edges
        include_bonds: draw bond scaffold underneath
        width_scale: scale factor for edge widths

    Returns:
        (n_edges, w_max)
    """
    import networkx as nx

    pos_atoms = get_molecule_coords(smiles)
    bonds_list = get_bonds(smiles)
    symbols = get_atom_symbols(smiles)
    M = _get_head_matrix_nodekeys(A_all, head_idx, mode=mode)
    edges = _sparsify_edges(M, method=sparsify, k=k, mass=mass,
                            threshold=threshold, drop_self=drop_self)
    n = len(pos_atoms)
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    pos = dict(pos_atoms)
    weights = []
    for q, j, w in edges:
        G.add_edge(int(q), int(j), weight=float(w))
        weights.append(float(w))
    # Draw bond scaffold
    if include_bonds and len(bonds_list) > 0:
        bond_graph = nx.Graph(bonds_list)
        nx.draw_networkx_edges(bond_graph, pos=pos, ax=ax, width=1.1, alpha=0.30)
    # Nodes and labels
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=380)
    labels = {i: str(i) for i in range(n)}
    nx.draw_networkx_labels(G, pos=pos, labels=labels, ax=ax, font_size=9)
    # Directed attention edges
    w_max = max(weights) if weights else 1.0
    widths = [width_scale * (G[u][v]["weight"] / (w_max + 1e-12)) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax,
        arrows=True, arrowstyle="-|>", arrowsize=12,
        width=widths, alpha=0.9, connectionstyle="arc3,rad=0.10",
    )
    ax.set_axis_off()
    return len(weights), w_max


def plot_molecule_attention_tripanel(smiles, A_layer, head_idx, layer_idx=None,
                                      log_scale=True, right_panel="coupling",
                                      figsize=(13.6, 4.5)):
    """Create 3-panel figure: plain molecule, attention overlay, and matrix/coupling.

    Args:
        smiles: SMILES string
        A_layer: [H, T, T] attention tensor for one layer
        head_idx: which head
        layer_idx: layer number for title (optional)
        log_scale: use log1p for inflow visualization
        right_panel: "coupling" or "matrix"
        figsize: figure size

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    # Compute inflow: sum over node queries of attention to each key
    T = A_layer.shape[1]
    n = T - 1
    A_h = A_layer[head_idx]  # [T, T]
    if hasattr(A_h, "detach"):
        inflow = A_h[1:, 1:].sum(dim=0).detach().float().cpu().numpy()  # [n]
    else:
        inflow = np.asarray(A_h[1:, 1:], dtype=float).sum(axis=0)

    img_plain = draw_molecule_plain(smiles, label_mode="atom")
    img_overlay, vmin, vmax, cmap_name = draw_molecule_attention_overlay(
        smiles, inflow, log_scale=log_scale
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # Panel 1: plain molecule
    axes[0].imshow(img_plain)
    axes[0].set_axis_off()
    axes[0].set_title("Molecule structure", fontsize=11)
    # Panel 2: attention overlay
    axes[1].imshow(img_overlay)
    axes[1].set_axis_off()
    axes[1].set_title(f"Key inflow ({'log1p' if log_scale else 'raw'})", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name),
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("log1p(inflow)" if log_scale else "inflow", fontsize=9)
    # Panel 3: coupling graph or full node->node attention matrix
    if right_panel == "matrix":
        M = _get_head_matrix_nodekeys(A_layer, head_idx, mode="raw")
        M_np = M.detach().float().cpu().numpy() if hasattr(M, "detach") else np.asarray(M, dtype=float)
        im = axes[2].imshow(M_np, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0)
        axes[2].set_title(r"Node$\to$node attention matrix $A_{qk}$", fontsize=11)
        axes[2].set_xlabel("Key index", fontsize=9)
        axes[2].set_ylabel("Query index", fontsize=9)
        axes[2].tick_params(labelsize=7)
        cb2 = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
        cb2.set_label("Attention weight", fontsize=9)
    else:
        n_edges, wmax = draw_attention_coupling_graph(
            axes[2], smiles, A_layer, head_idx,
            mode="node_cond", sparsify="mass", mass=0.9
        )
        axes[2].set_title(f"Attention coupling (mass\u22650.9, {n_edges} edges)", fontsize=11)

    Ltxt = f"L{layer_idx} " if layer_idx is not None else ""
    fig.suptitle(f"{Ltxt}H{head_idx} \u2014 Molecule attention analysis",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
