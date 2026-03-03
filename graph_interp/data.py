"""Dataset loading and Graphormer batch building from SMILES strings."""

import numpy as np
import torch
from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph

from .config import MAX_DIST, DEFAULT_POOL_SIZE


# ── Dataset loading ──────────────────────────────────────────

def load_dataset(root: str = "pcqm4mv2"):
    """Load PCQM4Mv2 dataset (SMILES-only when possible)."""
    try:
        ds = PCQM4Mv2Dataset(root=root, only_smiles=True)
    except TypeError:
        ds = PCQM4Mv2Dataset(root=root)
    return ds


def get_smiles(ds, i: int) -> str:
    """Extract a SMILES string from dataset index *i*."""
    x = ds[i]
    if isinstance(x, dict) and "smiles" in x:
        return x["smiles"]
    if isinstance(x, (tuple, list)) and len(x) >= 1 and isinstance(x[0], str):
        return x[0]
    if isinstance(x, str):
        return x
    raise ValueError(
        f"Could not extract SMILES at index {i}; "
        f"got type={type(x)} keys={getattr(x, 'keys', lambda: [])()}"
    )


def load_smiles_pool(ds=None, pool_size: int = DEFAULT_POOL_SIZE, root: str = "pcqm4mv2"):
    """Return a list of SMILES strings from the first *pool_size* dataset entries."""
    if ds is None:
        ds = load_dataset(root=root)
    return [get_smiles(ds, i) for i in range(min(pool_size, len(ds)))]


# ── Batch building internals ─────────────────────────────────

def _convert_to_single_emb(x: np.ndarray, offset: int = 512) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    out = x.copy()
    for i in range(out.shape[1]):
        out[:, i] = out[:, i] + 1 + offset * i
    return out


def _build_undirected_adj(edge_index: np.ndarray, edge_feat: np.ndarray):
    E = edge_index.shape[1]
    adj: dict[int, list] = {u: [] for u in range(int(edge_index.max()) + 1)}
    for e in range(E):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        ef = edge_feat[e]
        adj.setdefault(u, []).append((v, ef))
        adj.setdefault(v, []).append((u, ef))
    return adj


def _all_pairs_shortest_paths(adj, n: int):
    INF = 10**9
    dist = np.full((n, n), INF, dtype=np.int64)
    parent = np.full((n, n), -1, dtype=np.int64)
    parent_ef = np.empty((n, n), dtype=object)

    for s in range(n):
        dist[s, s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]; head += 1
            du = dist[s, u]
            for w, ef in adj.get(u, []):
                if 0 <= w < n and dist[s, w] > du + 1:
                    dist[s, w] = du + 1
                    parent[s, w] = u
                    parent_ef[s, w] = ef
                    q.append(w)
    return dist, parent, parent_ef


def _reconstruct_path_edgefeats(parent, parent_ef, s: int, t: int):
    if s == t or parent[s, t] == -1:
        return []
    out = []
    cur = t
    while cur != s:
        p = parent[s, cur]
        if p == -1:
            return []
        out.append(parent_ef[s, cur])
        cur = p
    out.reverse()
    return out


def _make_graphormer_batch(g: dict, max_dist: int, spatial_pos_max: int = 20, offset: int = 512):
    edge_index = np.asarray(g["edge_index"], dtype=np.int64)
    node_feat = np.asarray(g["node_feat"], dtype=np.int64)
    edge_feat = np.asarray(g["edge_feat"], dtype=np.int64)
    n = int(g["num_nodes"])
    Fe = edge_feat.shape[1]

    node_feat_se = _convert_to_single_emb(node_feat, offset=offset)
    edge_feat_se = _convert_to_single_emb(edge_feat, offset=offset)

    input_nodes = torch.from_numpy(node_feat_se).long()

    in_deg = np.zeros(n, dtype=np.int64)
    out_deg = np.zeros(n, dtype=np.int64)
    for u, v in zip(edge_index[0], edge_index[1]):
        out_deg[u] += 1
        in_deg[v] += 1
    in_degree = torch.from_numpy(in_deg + 1).long()
    out_degree = torch.from_numpy(out_deg + 1).long()

    adj = _build_undirected_adj(edge_index, edge_feat_se)
    dist, parent, parent_ef = _all_pairs_shortest_paths(adj, n)

    dist_clip = np.minimum(dist, spatial_pos_max)
    spatial_pos = torch.from_numpy(dist_clip + 1).long()

    attn_bias = torch.zeros((n + 1, n + 1), dtype=torch.float32)
    attn_edge_type = torch.zeros((n, n, Fe), dtype=torch.long)
    input_edges = torch.zeros((n, n, max_dist, Fe), dtype=torch.long)

    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        attn_edge_type[u, v, :] = torch.from_numpy(edge_feat_se[e]).long()

    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            d = dist[s, t]
            if d >= 10**9 or d <= 0:
                continue
            efs = _reconstruct_path_edgefeats(parent, parent_ef, s, t)
            if not efs:
                continue
            L = min(len(efs), max_dist)
            for k in range(L):
                input_edges[s, t, k, :] = torch.from_numpy(efs[k]).long()

    batch = {
        "input_nodes": input_nodes.unsqueeze(0),
        "input_edges": input_edges.unsqueeze(0),
        "attn_bias": attn_bias.unsqueeze(0),
        "in_degree": in_degree.unsqueeze(0),
        "out_degree": out_degree.unsqueeze(0),
        "spatial_pos": spatial_pos.unsqueeze(0),
        "attn_edge_type": attn_edge_type.unsqueeze(0),
    }
    return batch, n, dist


def _batch_to_device(batch, device, max_dist: int = 5, max_degree: int = 512):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if k == "spatial_pos":
                v = v.long()
                v = torch.where(v > max_dist, torch.full_like(v, max_dist + 1), v)
                v = v.clamp_min(0)
            elif k in ("in_degree", "out_degree"):
                v = v.long().clamp(min=0, max=max_degree)
            elif k in ("x", "attn_edge_type", "edge_input"):
                v = v.long().clamp_min(0)
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ── Public API ───────────────────────────────────────────────

def build_batch_from_smiles(smiles: str, model, device: torch.device, offset: int = 512):
    """Build a Graphormer-ready batch dict from a SMILES string.

    Returns:
        batch: dict of tensors on *device*
        n: number of atoms (nodes)
        dist: [n, n] numpy shortest-path distance matrix
    """
    g = smiles2graph(smiles)
    spatial_pos_max = int(getattr(model.config, "spatial_pos_max", 20))
    batch, n, dist = _make_graphormer_batch(
        g,
        max_dist=int(model.config.multi_hop_max_dist),
        spatial_pos_max=spatial_pos_max,
        offset=offset,
    )
    max_dist = int(getattr(model.config, "max_dist", MAX_DIST))
    batch = _batch_to_device(batch, device, max_dist=max_dist)
    return batch, n, dist
