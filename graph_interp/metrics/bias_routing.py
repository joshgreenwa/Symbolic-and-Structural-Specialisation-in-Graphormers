"""Bias-routes-dot-selects hypothesis testing: containment and cosine scores."""

import numpy as np
import torch

from ..extraction import extract_attention, node_only_conditional
from ..data import build_batch_from_smiles
from ..config import DEFAULT_NUM_PERMS, DEFAULT_TAU
from .scores import sample_transpositions, cossim_lastdim


@torch.no_grad()
def score_bias_routing(
    A_all, dot_all, bias_all,
    num_perms=DEFAULT_NUM_PERMS, tau=DEFAULT_TAU,
    top_k_frac=0.5, seed=0, eps=1e-12,
):
    """Containment and refinement metrics for one layer.

    Returns dict of [H] numpy arrays:
        containment, refinement_corr, bias_entropy, full_entropy,
        topk_mass_bias, topk_mass_full
    """
    device = A_all.device
    H, T, _ = A_all.shape
    n = T - 1

    out_keys = [
        "containment", "refinement_corr", "bias_entropy", "full_entropy",
        "topk_mass_bias", "topk_mass_full",
    ]
    if n < 3:
        return {k: np.zeros(H) for k in out_keys}

    A_nodes = A_all[:, 1:, 1:].float()
    A_nodes = A_nodes / A_nodes.sum(dim=-1, keepdim=True).clamp_min(eps)
    dot_nodes = dot_all[:, 1:, 1:].float()
    bias_nodes = bias_all[:, 1:, 1:].float()

    B_attn = torch.softmax(bias_nodes, dim=-1)
    B_attn = B_attn / B_attn.sum(dim=-1, keepdim=True).clamp_min(eps)

    k = max(int(n * top_k_frac), 1)
    _, topk_idx = B_attn.topk(k, dim=-1)
    topk_mask = torch.zeros(H, n, n, device=device, dtype=torch.bool)
    topk_mask.scatter_(2, topk_idx, True)

    bias_ent = -(B_attn * torch.log2(B_attn + eps)).sum(dim=-1).mean(dim=-1)
    full_ent = -(A_nodes * torch.log2(A_nodes + eps)).sum(dim=-1).mean(dim=-1)
    topk_mass_bias = (B_attn * topk_mask.float()).sum(dim=-1).mean(dim=-1)
    topk_mass_full = (A_nodes * topk_mask.float()).sum(dim=-1).mean(dim=-1)

    uv_list = sample_transpositions(n, num_perms, seed=seed)

    inv_tau = 1.0 / max(float(tau), 1e-12)
    delta = torch.empty((num_perms, H, n), device=device)
    for t, (u, v) in enumerate(uv_list):
        delta[t] = (A_nodes[:, :, u] - A_nodes[:, :, v]).abs() * inv_tau
    alpha = torch.softmax(delta, dim=0)

    containment_acc = torch.zeros(H, device=device)

    for t, (u, v) in enumerate(uv_list):
        w = alpha[t]
        perm = torch.arange(n, device=device)
        perm[u], perm[v] = v, u

        logits_pi = dot_nodes[:, :, perm] + bias_nodes
        A_pi = torch.softmax(logits_pi, dim=-1)
        A_pi = A_pi / A_pi.sum(dim=-1, keepdim=True).clamp_min(eps)

        dA = (A_pi - A_nodes).abs()
        dA_total = dA.sum(dim=-1).clamp_min(eps)
        dA_topk = (dA * topk_mask.float()).sum(dim=-1)
        containment_acc += (w * (dA_topk / dA_total)).mean(dim=-1)

    return {
        "containment": containment_acc.cpu().numpy(),
        "refinement_corr": np.zeros(H),  # simplified — full Pearson not needed for figures
        "bias_entropy": bias_ent.cpu().numpy(),
        "full_entropy": full_ent.cpu().numpy(),
        "topk_mass_bias": topk_mass_bias.cpu().numpy(),
        "topk_mass_full": topk_mass_full.cpu().numpy(),
    }


@torch.no_grad()
def score_bias_routing_cossim(
    A_all, dot_all, bias_all,
    num_perms=DEFAULT_NUM_PERMS, tau=DEFAULT_TAU,
    top_k_frac=0.5, seed=0, eps=1e-12,
):
    """Cosine-similarity scores restricted to bias top-k / bottom-k keys.

    Returns dict of [H] numpy arrays:
        struct_topk, symb_topk, struct_botk, symb_botk, struct_all, symb_all
    """
    device = A_all.device
    H, T, _ = A_all.shape
    n = T - 1

    keys = ["struct_topk", "symb_topk", "struct_botk", "symb_botk", "struct_all", "symb_all"]
    if n < 3:
        return {k: np.zeros(H) for k in keys}

    A_nodes = A_all[:, 1:, 1:].float()
    A_nodes = A_nodes / A_nodes.sum(dim=-1, keepdim=True).clamp_min(eps)
    dot_nodes = dot_all[:, 1:, 1:].float()
    bias_nodes = bias_all[:, 1:, 1:].float()

    B_attn = torch.softmax(bias_nodes, dim=-1)
    B_attn = B_attn / B_attn.sum(dim=-1, keepdim=True).clamp_min(eps)

    k = max(int(n * top_k_frac), 1)
    _, topk_idx = B_attn.topk(k, dim=-1)
    topk_mask = torch.zeros(H, n, n, device=device, dtype=torch.bool)
    topk_mask.scatter_(2, topk_idx, True)
    bottomk_mask = ~topk_mask

    uv_list = sample_transpositions(n, num_perms, seed=seed)

    inv_tau = 1.0 / max(float(tau), 1e-12)
    delta = torch.empty((num_perms, H, n), device=device)
    for t, (u, v) in enumerate(uv_list):
        delta[t] = (A_nodes[:, :, u] - A_nodes[:, :, v]).abs() * inv_tau
    alpha = torch.softmax(delta, dim=0)

    accum = {kk: torch.zeros(H, device=device) for kk in keys}

    for t, (u, v) in enumerate(uv_list):
        w = alpha[t]
        perm = torch.arange(n, device=device)
        perm[u], perm[v] = v, u

        logits_pi = dot_nodes[:, :, perm] + bias_nodes
        A_pi = torch.softmax(logits_pi, dim=-1)
        A_pi = A_pi / A_pi.sum(dim=-1, keepdim=True).clamp_min(eps)
        A_symb_ref = A_nodes[:, :, perm]

        for mask, s_key, y_key in [
            (topk_mask, "struct_topk", "symb_topk"),
            (bottomk_mask, "struct_botk", "symb_botk"),
            (None, "struct_all", "symb_all"),
        ]:
            if mask is not None:
                mf = mask.float()
                a_pi_m = A_pi * mf
                a_ref_m = A_nodes * mf
                a_sym_m = A_symb_ref * mf
            else:
                a_pi_m = A_pi
                a_ref_m = A_nodes
                a_sym_m = A_symb_ref

            cs_s = cossim_lastdim(a_pi_m, a_ref_m)  # [H, n]
            cs_y = cossim_lastdim(a_pi_m, a_sym_m)
            accum[s_key] += (w * cs_s).mean(dim=-1)
            accum[y_key] += (w * cs_y).mean(dim=-1)

    return {kk: v.cpu().numpy() for kk, v in accum.items()}


@torch.no_grad()
def aggregate_bias_routing(
    model, smiles_list: list[str], device: torch.device,
    num_perms=DEFAULT_NUM_PERMS, tau=DEFAULT_TAU,
    top_k_frac=0.5, seed=0,
    n_graphs: int | None = None,
):
    """Average bias routing metrics over graphs.

    Returns:
        containment_LH: [L, H]
        cossim_results: dict of [L, H] arrays (struct_topk, symb_topk, etc.)
    """
    if n_graphs is not None:
        smiles_list = smiles_list[:n_graphs]

    sum_cont = None
    sum_cossim: dict | None = None
    used = 0

    for gi, smi in enumerate(smiles_list):
        batch, n, dist = build_batch_from_smiles(smi, model=model, device=device)
        dot_list, bias_list, A_list = extract_attention(model, batch)

        L, H = len(A_list), A_list[0].shape[0]
        cont = np.empty((L, H))
        cs_dict: dict[str, np.ndarray] = {}

        for l_idx in range(L):
            r = score_bias_routing(
                A_list[l_idx], dot_list[l_idx], bias_list[l_idx],
                num_perms=num_perms, tau=tau, top_k_frac=top_k_frac,
                seed=seed + 10_000 * gi + l_idx,
            )
            cont[l_idx] = r["containment"]

            rc = score_bias_routing_cossim(
                A_list[l_idx], dot_list[l_idx], bias_list[l_idx],
                num_perms=num_perms, tau=tau, top_k_frac=top_k_frac,
                seed=seed + 10_000 * gi + l_idx + 5000,
            )
            for kk, v in rc.items():
                cs_dict.setdefault(kk, np.empty((L, H)))
                cs_dict[kk][l_idx] = v

        if sum_cont is None:
            sum_cont = cont
            sum_cossim = cs_dict
        else:
            sum_cont += cont
            for kk in sum_cossim:
                sum_cossim[kk] += cs_dict[kk]
        used += 1

    d = max(used, 1)
    return sum_cont / d, {kk: v / d for kk, v in sum_cossim.items()}
