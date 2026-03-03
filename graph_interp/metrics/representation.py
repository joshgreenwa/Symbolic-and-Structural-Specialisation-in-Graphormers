"""Node representation similarity dynamics across layers."""

import numpy as np
import torch
import torch.nn.functional as F

from ..extraction import extract_hidden_states
from ..data import build_batch_from_smiles


def _pairwise_cosine(X: torch.Tensor, eps: float = 1e-12):
    """Mean, min, max pairwise cosine similarity among rows of X [n, D]."""
    X_norm = F.normalize(X, dim=-1, eps=eps)
    sim = X_norm @ X_norm.T  # [n, n]
    n = sim.shape[0]
    if n < 2:
        return 1.0, 1.0, 1.0
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    vals = sim[mask]
    return float(vals.mean()), float(vals.min()), float(vals.max())


@torch.no_grad()
def node_similarity_profile(
    model, batch_device: dict,
    ablate_heads: dict | None = None,
    ablation_mode: str = "zero",
):
    """Pairwise cosine similarity of node reps at each layer.

    Returns:
        mean_sim: [L] mean pairwise cosine similarity
        min_sim:  [L]
        max_sim:  [L]
    """
    A_list, hidden_list, final = extract_hidden_states(
        model, batch_device, ablate_heads=ablate_heads, ablation_mode=ablation_mode
    )
    L = len(hidden_list)
    mean_sim = np.empty(L)
    min_sim = np.empty(L)
    max_sim = np.empty(L)

    for l_idx in range(L):
        h = hidden_list[l_idx]  # [T, D]
        nodes = h[1:]  # exclude CLS
        mn, mi, mx = _pairwise_cosine(nodes)
        mean_sim[l_idx] = mn
        min_sim[l_idx] = mi
        max_sim[l_idx] = mx

    return mean_sim, min_sim, max_sim


@torch.no_grad()
def sublayer_similarity_breakdown(model, batch_device: dict):
    """Similarity at sublayer stages: input, after-attention, after-FFN.

    Returns:
        sim_input:      [L]
        sim_after_attn: [L]
        sim_after_ffn:  [L]
    """
    A_list, hidden_list, sublayer_list, final = extract_hidden_states(
        model, batch_device, return_sublayers=True
    )
    L = len(hidden_list)
    sim_input = np.empty(L)
    sim_attn = np.empty(L)
    sim_ffn = np.empty(L)

    # For input of layer 0, we need the embedding — approximate with sublayer_list from layer before
    # Actually hidden_list[l] = after FFN of layer l. sublayer_list[l] = after attn of layer l.
    # input of layer l = hidden_list[l-1] for l>0, or embedding for l=0.
    # We don't have the raw embedding easily, so we skip layer 0 input and use sublayer[0] as first available.

    for l_idx in range(L):
        # after-attention
        after_attn = sublayer_list[l_idx][1:]
        sim_attn[l_idx] = _pairwise_cosine(after_attn)[0]

        # after-FFN
        after_ffn = hidden_list[l_idx][1:]
        sim_ffn[l_idx] = _pairwise_cosine(after_ffn)[0]

        # input = previous layer's output (or first sublayer for l=0)
        if l_idx == 0:
            sim_input[l_idx] = float("nan")  # not available without embedding
        else:
            inp = hidden_list[l_idx - 1][1:]
            sim_input[l_idx] = _pairwise_cosine(inp)[0]

    return sim_input, sim_attn, sim_ffn


@torch.no_grad()
def aggregate_representation_dynamics(
    model, smiles_list: list[str], device: torch.device,
    n_graphs: int | None = None,
):
    """Average node similarity profile over graphs.

    Returns:
        mean_sim: [L] mean pairwise cosine (averaged over graphs)
        min_sim:  [L]
        max_sim:  [L]
        delta_sim: [L-1] change in mean similarity per layer transition
    """
    if n_graphs is not None:
        smiles_list = smiles_list[:n_graphs]

    sum_mean = sum_min = sum_max = None
    used = 0

    for smi in smiles_list:
        batch, n, dist = build_batch_from_smiles(smi, model=model, device=device)
        mn, mi, mx = node_similarity_profile(model, batch)

        if sum_mean is None:
            sum_mean, sum_min, sum_max = mn.copy(), mi.copy(), mx.copy()
        else:
            sum_mean += mn; sum_min += mi; sum_max += mx
        used += 1

    d = max(used, 1)
    mean_sim = sum_mean / d
    min_sim = sum_min / d
    max_sim = sum_max / d
    delta_sim = np.diff(mean_sim)

    return mean_sim, min_sim, max_sim, delta_sim


@torch.no_grad()
def aggregate_sublayer_breakdown(
    model, smiles_list: list[str], device: torch.device,
    n_graphs: int | None = None,
):
    """Average sublayer similarity breakdown over graphs.

    Returns: (sim_input [L], sim_attn [L], sim_ffn [L])
    """
    if n_graphs is not None:
        smiles_list = smiles_list[:n_graphs]

    s_in = s_at = s_ff = None
    used = 0

    for smi in smiles_list:
        batch, n, dist = build_batch_from_smiles(smi, model=model, device=device)
        si, sa, sf = sublayer_similarity_breakdown(model, batch)

        if s_in is None:
            s_in, s_at, s_ff = si.copy(), sa.copy(), sf.copy()
        else:
            s_in += si; s_at += sa; s_ff += sf
        used += 1

    d = max(used, 1)
    return s_in / d, s_at / d, s_ff / d
