"""Hop-distance conditioned attention mass and statistics."""

import numpy as np
import torch

from ..extraction import extract_attention, node_only_conditional
from ..data import build_batch_from_smiles
from ..config import DEFAULT_D_CAP, DEFAULT_QUERY_SUBSAMPLE


def distance_conditioned_mass(
    A_all: torch.Tensor, dist_nn, head_idx: int,
    mode: str = "node_cond",
    query_subsample: int | None = DEFAULT_QUERY_SUBSAMPLE,
    seed: int = 0, d_cap: int | None = DEFAULT_D_CAP,
    drop_self: bool = True, eps: float = 1e-12,
):
    """Attention mass binned by hop distance for one head.

    Returns: M [B] where B = d_cap+2 (or dmax+1 if d_cap is None).
    """
    device = A_all.device
    H, T, _ = A_all.shape
    n = T - 1

    if isinstance(dist_nn, np.ndarray):
        dist_nn = torch.from_numpy(dist_nn)
    dist_nn = dist_nn.to(device=device).long()

    q_idx = torch.arange(n, device=device)
    if query_subsample is not None and n > query_subsample:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        q_idx = q_idx[torch.randperm(n, generator=g)[:query_subsample]].sort().values
    Q = int(q_idx.numel())

    if mode == "node_cond":
        P_all, _ = node_only_conditional(A_all, eps=eps)
        P = P_all[head_idx]
    else:
        P = A_all[head_idx, 1:, 1:]

    Pq = P.index_select(0, q_idx)
    Dq = dist_nn.index_select(0, q_idx)

    if drop_self:
        cols = q_idx.view(-1, 1)
        Pq = Pq.scatter(1, cols, torch.zeros((Q, 1), device=device, dtype=Pq.dtype))

    if d_cap is None:
        dmax = int(dist_nn.max().item())
        B = dmax + 1
        Db = Dq.clamp(0, dmax)
    else:
        B = d_cap + 2
        Db = torch.where(Dq > d_cap, torch.full_like(Dq, d_cap + 1), Dq).clamp_min(0)

    M = torch.zeros((B,), device=device, dtype=Pq.dtype)
    M.scatter_add_(0, Db.reshape(-1), Pq.reshape(-1))
    M = M / max(Q, 1)
    return M


def mu_sigma_from_M(M: torch.Tensor):
    """Mean and std of distance distribution M[d]."""
    d = torch.arange(M.numel(), device=M.device, dtype=M.dtype)
    mu = (d * M).sum()
    var = ((d - mu) ** 2 * M).sum().clamp_min(0.0)
    return mu, torch.sqrt(var)


@torch.no_grad()
def all_heads_distance_stats(
    A_all: torch.Tensor, dist_nn,
    mode: str = "node_cond",
    query_subsample: int | None = DEFAULT_QUERY_SUBSAMPLE,
    seed: int = 0, d_cap: int | None = DEFAULT_D_CAP,
    drop_self: bool = True,
):
    """Distance stats for all heads in one layer.

    Returns: (M_all [H,B], mu_all [H], sig_all [H])
    """
    H = A_all.shape[0]
    B = (d_cap + 2) if d_cap is not None else (int(torch.as_tensor(dist_nn).max().item()) + 1)

    M_all = torch.empty((H, B), device="cpu")
    mu_all = torch.empty((H,), device="cpu")
    sig_all = torch.empty((H,), device="cpu")

    for h in range(H):
        M = distance_conditioned_mass(
            A_all, dist_nn, h, mode=mode,
            query_subsample=query_subsample, seed=seed + 13 * h,
            d_cap=d_cap, drop_self=drop_self,
        )
        mu, sig = mu_sigma_from_M(M)
        M_all[h] = M.detach().cpu()
        mu_all[h] = mu.cpu()
        sig_all[h] = sig.cpu()

    return M_all, mu_all, sig_all


@torch.no_grad()
def aggregate_distance_stats(
    model, smiles_list: list[str], device: torch.device,
    mode: str = "node_cond",
    query_subsample: int | None = DEFAULT_QUERY_SUBSAMPLE,
    d_cap: int | None = DEFAULT_D_CAP,
    drop_self: bool = True, seed: int = 0,
    n_graphs: int | None = None,
):
    """Average distance stats over graphs.

    Returns: (mean_mu [L,H], mean_sigma [L,H], mean_M [L,H,B])
    """
    if n_graphs is not None:
        smiles_list = smiles_list[:n_graphs]

    sum_mu = sum_sig = sum_M = None
    used = 0

    for gi, smi in enumerate(smiles_list):
        batch, n, dist = build_batch_from_smiles(smi, model=model, device=device)
        _, _, A_list = extract_attention(model, batch)
        dist_t = torch.as_tensor(dist, device=device)

        L, H = len(A_list), A_list[0].shape[0]
        B = (d_cap + 2) if d_cap is not None else (int(dist_t.max().item()) + 1)

        mu_LH = torch.empty((L, H))
        sig_LH = torch.empty((L, H))
        M_LHB = torch.empty((L, H, B))

        for l_idx in range(L):
            M_h, mu_h, sig_h = all_heads_distance_stats(
                A_list[l_idx], dist_t, mode=mode,
                query_subsample=query_subsample, seed=seed + 10_000 * gi + 97 * l_idx,
                d_cap=d_cap, drop_self=drop_self,
            )
            mu_LH[l_idx] = mu_h
            sig_LH[l_idx] = sig_h
            M_LHB[l_idx] = M_h

        if sum_mu is None:
            sum_mu, sum_sig, sum_M = mu_LH.clone(), sig_LH.clone(), M_LHB.clone()
        else:
            sum_mu += mu_LH; sum_sig += sig_LH; sum_M += M_LHB
        used += 1

    d = max(used, 1)
    return (sum_mu / d).numpy(), (sum_sig / d).numpy(), (sum_M / d).numpy()
