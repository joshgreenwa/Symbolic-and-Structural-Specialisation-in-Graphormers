"""Structural vs symbolic scoring via key-permutation interventions."""

import numpy as np
import torch

from ..extraction import extract_attention, node_only_conditional
from ..data import build_batch_from_smiles
from ..config import DEFAULT_NUM_PERMS, DEFAULT_TAU, DEFAULT_QUERY_SUBSAMPLE, DEFAULT_N_GRAPHS
from .welford import WelfordAccumulator


def cossim_lastdim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12):
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1).clamp_min(eps) * b.norm(dim=-1).clamp_min(eps)
    return (num / den).clamp(-1.0, 1.0)


def sample_transpositions(n: int, K: int, seed: int = 0):
    """Sample K transpositions (u,v) uniformly with u!=v in {0..n-1}."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, n, size=K)
    v = rng.integers(0, n, size=K)
    same = (u == v)
    while np.any(same):
        v[same] = rng.integers(0, n, size=np.sum(same))
        same = (u == v)
    return list(zip(u.tolist(), v.tolist()))


def _swap_perm(n: int, u: int, v: int, device):
    perm = torch.arange(n, device=device)
    perm[u], perm[v] = perm[v].clone(), perm[u].clone()
    return perm


@torch.no_grad()
def score_layer_struct_sym(
    dot_all: torch.Tensor,
    bias_all: torch.Tensor,
    A_all: torch.Tensor,
    num_perms: int = DEFAULT_NUM_PERMS,
    tau: float = DEFAULT_TAU,
    seed: int = 0,
    query_subsample: int | None = DEFAULT_QUERY_SUBSAMPLE,
):
    """Per-head structural and symbolic scores for one layer.

    Returns: (s_struct_head [H], s_sym_head [H])
    """
    device = dot_all.device
    H, T, _ = dot_all.shape
    n = T - 1

    dot = dot_all[:, 1:, 1:]
    bias = bias_all[:, 1:, 1:]
    p_base, _ = node_only_conditional(A_all)

    q_idx = torch.arange(n, device=device)
    if query_subsample is not None and n > query_subsample:
        g = torch.Generator(device=device)
        g.manual_seed(seed + 12345)
        q_idx = q_idx[torch.randperm(n, generator=g, device=device)[:query_subsample]].sort().values
    Q = q_idx.numel()

    dot_q = dot[:, q_idx, :]
    bias_q = bias[:, q_idx, :]
    p_q = p_base[:, q_idx, :]

    uv_list = sample_transpositions(n=n, K=num_perms, seed=seed)

    delta = torch.empty((num_perms, H, Q), device=device, dtype=p_q.dtype)
    for t, (u, v) in enumerate(uv_list):
        delta[t] = (p_q[:, :, u] - p_q[:, :, v]).abs() / max(tau, 1e-12)
    alpha = torch.softmax(delta, dim=0)

    s_struct = torch.zeros((H, Q), device=device, dtype=p_q.dtype)
    s_sym = torch.zeros((H, Q), device=device, dtype=p_q.dtype)

    for t, (u, v) in enumerate(uv_list):
        perm = _swap_perm(n, u, v, device=device)
        logits = dot_q[:, :, perm] + bias_q
        p_pi = torch.softmax(logits.float(), dim=-1).to(p_q.dtype)
        pP = p_q[:, :, perm]
        w = alpha[t]
        s_struct += w * cossim_lastdim(p_pi, p_q)
        s_sym += w * cossim_lastdim(p_pi, pP)

    return s_struct.mean(dim=1), s_sym.mean(dim=1)


@torch.no_grad()
def aggregate_scores(
    model, smiles_list: list[str], device: torch.device,
    num_perms: int = DEFAULT_NUM_PERMS,
    tau: float = DEFAULT_TAU,
    query_subsample: int | None = DEFAULT_QUERY_SUBSAMPLE,
    seed: int = 0,
    n_graphs: int | None = None,
    return_std: bool = False,
):
    """Average structural/symbolic scores over multiple graphs.

    Returns:
        When *return_std* is False (default):
            ``(mean_struct [L,H], mean_sym [L,H], mean_diff [L,H])``
        When *return_std* is True:
            ``(mean_struct, mean_sym, mean_diff, std_struct, std_sym)``
            where ``std_*`` are [L,H] arrays of per-graph standard deviations.
    """
    if n_graphs is not None:
        smiles_list = smiles_list[:n_graphs]

    sum_struct = sum_sym = None
    used = 0

    # Welford accumulators for optional std computation
    if return_std:
        welf_struct = WelfordAccumulator()
        welf_sym = WelfordAccumulator()

    for gi, smi in enumerate(smiles_list):
        batch, n, dist = build_batch_from_smiles(smi, model=model, device=device)
        dot_list, bias_list, A_list = extract_attention(model, batch)

        L = len(A_list)
        H = A_list[0].shape[0]

        sS = torch.empty((L, H), device="cpu")
        sY = torch.empty((L, H), device="cpu")

        for l_idx in range(L):
            sh, yh = score_layer_struct_sym(
                dot_all=dot_list[l_idx], bias_all=bias_list[l_idx], A_all=A_list[l_idx],
                num_perms=num_perms, tau=tau, seed=seed + 10_000 * gi + l_idx,
                query_subsample=query_subsample,
            )
            sS[l_idx] = sh.detach().cpu()
            sY[l_idx] = yh.detach().cpu()

        if sum_struct is None:
            sum_struct, sum_sym = sS.clone(), sY.clone()
        else:
            sum_struct += sS
            sum_sym += sY
        used += 1

        if return_std:
            welf_struct.update(sS.numpy())
            welf_sym.update(sY.numpy())

    mean_struct = (sum_struct / max(used, 1)).numpy()
    mean_sym = (sum_sym / max(used, 1)).numpy()

    if return_std:
        _, std_struct, _ = welf_struct.finalize()
        _, std_sym, _ = welf_sym.finalize()
        return mean_struct, mean_sym, mean_struct - mean_sym, std_struct, std_sym

    return mean_struct, mean_sym, mean_struct - mean_sym
