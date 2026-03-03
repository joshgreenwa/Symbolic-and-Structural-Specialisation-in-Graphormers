"""KL divergence scores: full vs dot-only vs bias-only attention."""

import numpy as np
import torch

from ..extraction import extract_attention, node_only_conditional
from ..data import build_batch_from_smiles


def _safe_softmax(logits, dim=-1):
    return torch.softmax(logits.float(), dim=dim)


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12):
    """KL(p || q) along last dim. Returns [...] (non-negative)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def _normalized_kl(p, q, eps=1e-12):
    """KL normalized to [0, 1] via KL / (KL + 1)."""
    raw = kl_div(p, q, eps=eps)
    return raw / (raw + 1.0)


@torch.no_grad()
def kl_scores_one_layer(
    dot_all: torch.Tensor, bias_all: torch.Tensor, A_all: torch.Tensor,
    eps: float = 1e-12,
):
    """KL divergence of full attention from dot-only and bias-only.

    Returns:
        kl_dot_only: [H] — KL(full || dot-only), normalized
        kl_bias_only: [H] — KL(full || bias-only), normalized
    """
    H, T, _ = A_all.shape
    n = T - 1

    p_full, _ = node_only_conditional(A_all, eps=eps)  # [H, n, n]

    dot_nn = dot_all[:, 1:, 1:]
    bias_nn = bias_all[:, 1:, 1:]

    p_dot = _safe_softmax(dot_nn, dim=-1)
    p_dot = p_dot / p_dot.sum(dim=-1, keepdim=True).clamp_min(eps)

    p_bias = _safe_softmax(bias_nn, dim=-1)
    p_bias = p_bias / p_bias.sum(dim=-1, keepdim=True).clamp_min(eps)

    kl_dot = _normalized_kl(p_full, p_dot, eps=eps).mean(dim=-1)   # [H]
    kl_bias = _normalized_kl(p_full, p_bias, eps=eps).mean(dim=-1)  # [H]

    return kl_dot, kl_bias


@torch.no_grad()
def aggregate_kl_scores(
    model, smiles_list: list[str], device: torch.device,
    n_graphs: int | None = None,
):
    """Average KL scores over graphs.

    Returns: (mean_kl_dot [L,H], mean_kl_bias [L,H])
    """
    if n_graphs is not None:
        smiles_list = smiles_list[:n_graphs]

    sum_dot = sum_bias = None
    used = 0

    for smi in smiles_list:
        batch, n, dist = build_batch_from_smiles(smi, model=model, device=device)
        dot_list, bias_list, A_list = extract_attention(model, batch)

        L, H = len(A_list), A_list[0].shape[0]
        kd = torch.empty((L, H))
        kb = torch.empty((L, H))

        for l_idx in range(L):
            d, b = kl_scores_one_layer(dot_list[l_idx], bias_list[l_idx], A_list[l_idx])
            kd[l_idx] = d.cpu()
            kb[l_idx] = b.cpu()

        if sum_dot is None:
            sum_dot, sum_bias = kd.clone(), kb.clone()
        else:
            sum_dot += kd; sum_bias += kb
        used += 1

    d = max(used, 1)
    return (sum_dot / d).numpy(), (sum_bias / d).numpy()
