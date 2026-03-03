"""Centered cosine similarity, TV contrast, logit-level contrast, and other scoring variants.

This module implements several intervention-based metrics that compare
post-softmax attention distributions (and pre-softmax logits) under
key-permutation transpositions.  All metrics operate on node-only slices
(excluding the CLS token at index 0).
"""
from __future__ import annotations

import numpy as np
import torch

from ..extraction import extract_attention, node_only_conditional
from ..data import build_batch_from_smiles
from ..config import (
    DEFAULT_NUM_PERMS,
    DEFAULT_TAU,
    DEFAULT_QUERY_SUBSAMPLE,
    DEFAULT_N_GRAPHS,
)
from .scores import sample_transpositions, _swap_perm, cossim_lastdim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centered_cossim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cosine similarity after subtracting the row-wise mean from each vector.

    Parameters
    ----------
    a, b : Tensor  [..., K]
        Vectors whose last dimension is compared.

    Returns
    -------
    Tensor  [...]
        Per-row centered cosine similarity, clamped to [-1, 1].
    """
    a_c = a - a.mean(dim=-1, keepdim=True)
    b_c = b - b.mean(dim=-1, keepdim=True)
    return cossim_lastdim(a_c, b_c, eps=eps)


# ---------------------------------------------------------------------------
# Per-layer scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_layer_all_variants(
    dot_all: torch.Tensor,
    bias_all: torch.Tensor,
    A_all: torch.Tensor,
    num_perms: int = DEFAULT_NUM_PERMS,
    tau: float = DEFAULT_TAU,
    seed: int = 0,
    query_subsample: int | None = DEFAULT_QUERY_SUBSAMPLE,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Per-head scores for one layer under multiple metric variants.

    Parameters
    ----------
    dot_all : Tensor [H, T, T]
        Scaled dot-product logits (including CLS).
    bias_all : Tensor [H, T, T]
        Graph attention bias (including CLS).
    A_all : Tensor [H, T, T]
        Post-softmax attention (including CLS).
    num_perms : int
        Number of random transpositions to sample.
    tau : float
        Temperature for alpha weighting.
    seed : int
        RNG seed for transposition sampling.
    query_subsample : int or None
        If not None and *n* > this value, subsample query indices.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    dict mapping metric name -> numpy array of shape ``[H]``.
    Keys: ``tv_contrast_w``, ``tv_contrast_u``, ``tv_d_struct_w``,
    ``tv_d_symb_w``, ``ccossim_struct_w``, ``ccossim_symb_w``,
    ``ccossim_struct_u``, ``ccossim_symb_u``, ``logit_contrast_w``,
    ``logit_contrast_u``, ``orig_struct_w``, ``orig_symb_w``.
    """
    device = dot_all.device
    H, T, _ = dot_all.shape
    n = T - 1

    # Node-only slices
    dot = dot_all[:, 1:, 1:]        # [H, n, n]
    bias = bias_all[:, 1:, 1:]      # [H, n, n]
    p_base, _ = node_only_conditional(A_all)  # [H, n, n]

    # Query subsampling
    q_idx = torch.arange(n, device=device)
    if query_subsample is not None and n > query_subsample:
        g = torch.Generator(device=device)
        g.manual_seed(seed + 12345)
        q_idx = q_idx[torch.randperm(n, generator=g, device=device)[:query_subsample]].sort().values
    Q = q_idx.numel()

    dot_q = dot[:, q_idx, :]     # [H, Q, n]
    bias_q = bias[:, q_idx, :]   # [H, Q, n]
    p_q = p_base[:, q_idx, :]    # [H, Q, n]

    # Pre-softmax logits for query rows
    logits_ref = (dot_q + bias_q).float()  # [H, Q, n]

    # Transpositions
    uv_list = sample_transpositions(n=n, K=num_perms, seed=seed)

    # Alpha weights (softmax over permutations based on |p_u - p_v| / tau)
    delta = torch.empty((num_perms, H, Q), device=device, dtype=p_q.dtype)
    for t, (u, v) in enumerate(uv_list):
        delta[t] = (p_q[:, :, u] - p_q[:, :, v]).abs() / max(tau, 1e-12)
    alpha = torch.softmax(delta, dim=0)  # [num_perms, H, Q]

    # Accumulators (weighted and unweighted)
    tv_struct_w = torch.zeros((H, Q), device=device, dtype=torch.float32)
    tv_symb_w = torch.zeros((H, Q), device=device, dtype=torch.float32)
    tv_struct_u = torch.zeros((H, Q), device=device, dtype=torch.float32)
    tv_symb_u = torch.zeros((H, Q), device=device, dtype=torch.float32)

    ccossim_struct_w = torch.zeros((H, Q), device=device, dtype=torch.float32)
    ccossim_symb_w = torch.zeros((H, Q), device=device, dtype=torch.float32)
    ccossim_struct_u = torch.zeros((H, Q), device=device, dtype=torch.float32)
    ccossim_symb_u = torch.zeros((H, Q), device=device, dtype=torch.float32)

    logit_struct_w = torch.zeros((H, Q), device=device, dtype=torch.float32)
    logit_symb_w = torch.zeros((H, Q), device=device, dtype=torch.float32)
    logit_struct_u = torch.zeros((H, Q), device=device, dtype=torch.float32)
    logit_symb_u = torch.zeros((H, Q), device=device, dtype=torch.float32)

    orig_struct_w = torch.zeros((H, Q), device=device, dtype=torch.float32)
    orig_symb_w = torch.zeros((H, Q), device=device, dtype=torch.float32)

    for t, (u, v) in enumerate(uv_list):
        perm = _swap_perm(n, u, v, device=device)
        w = alpha[t]  # [H, Q]

        # Permuted logits and attention
        logits_pi = dot_q[:, :, perm] + bias_q      # [H, Q, n]
        p_pi = torch.softmax(logits_pi.float(), dim=-1).to(p_q.dtype)
        pP = p_q[:, :, perm]  # ideal permuted reference

        # --- TV distance ---
        tv_s = 0.5 * (p_pi - p_q).abs().sum(dim=-1)     # structural: compare to original
        tv_y = 0.5 * (p_pi - pP).abs().sum(dim=-1)       # symbolic: compare to permuted ref
        tv_struct_w += w * tv_s
        tv_symb_w += w * tv_y
        tv_struct_u += tv_s
        tv_symb_u += tv_y

        # --- Centered cosine similarity ---
        cc_s = _centered_cossim(p_pi, p_q, eps=eps)
        cc_y = _centered_cossim(p_pi, pP, eps=eps)
        ccossim_struct_w += w * cc_s
        ccossim_symb_w += w * cc_y
        ccossim_struct_u += cc_s
        ccossim_symb_u += cc_y

        # --- Logit-level L2 distance ---
        logits_pi_f = logits_pi.float()
        l2_s = ((logits_pi_f - logits_ref) ** 2).sum(dim=-1).sqrt()
        logits_ref_pi = logits_ref[:, :, perm]
        l2_y = ((logits_pi_f - logits_ref_pi) ** 2).sum(dim=-1).sqrt()
        logit_struct_w += w * l2_s
        logit_symb_w += w * l2_y
        logit_struct_u += l2_s
        logit_symb_u += l2_y

        # --- Original cosine similarity (standard, not centered) ---
        orig_struct_w += w * cossim_lastdim(p_pi, p_q, eps=eps)
        orig_symb_w += w * cossim_lastdim(p_pi, pP, eps=eps)

    # Normalise unweighted by num_perms
    inv_K = 1.0 / num_perms
    tv_struct_u *= inv_K
    tv_symb_u *= inv_K
    ccossim_struct_u *= inv_K
    ccossim_symb_u *= inv_K
    logit_struct_u *= inv_K
    logit_symb_u *= inv_K

    # TV contrast: (tv_symb - tv_struct) / (tv_symb + tv_struct + eps)
    def _contrast(s, y):
        return ((y - s) / (y + s + eps)).mean(dim=1).cpu().numpy()

    # Cosine/logit metrics: average over queries
    def _mean_q(t):
        return t.mean(dim=1).cpu().numpy()

    # Logit contrast: (l2_symb - l2_struct) / (l2_symb + l2_struct + eps)
    results = {
        "tv_contrast_w": _contrast(tv_struct_w, tv_symb_w),
        "tv_contrast_u": _contrast(tv_struct_u, tv_symb_u),
        "tv_d_struct_w": _mean_q(tv_struct_w),
        "tv_d_symb_w": _mean_q(tv_symb_w),
        "ccossim_struct_w": _mean_q(ccossim_struct_w),
        "ccossim_symb_w": _mean_q(ccossim_symb_w),
        "ccossim_struct_u": _mean_q(ccossim_struct_u),
        "ccossim_symb_u": _mean_q(ccossim_symb_u),
        "logit_contrast_w": _contrast(logit_struct_w, logit_symb_w),
        "logit_contrast_u": _contrast(logit_struct_u, logit_symb_u),
        "orig_struct_w": _mean_q(orig_struct_w),
        "orig_symb_w": _mean_q(orig_symb_w),
    }
    return results


# ---------------------------------------------------------------------------
# Multi-graph aggregation
# ---------------------------------------------------------------------------

@torch.no_grad()
def aggregate_all_variants(
    model,
    smiles_list: list[str],
    device: torch.device,
    n_graphs: int = DEFAULT_N_GRAPHS,
    num_perms: int = DEFAULT_NUM_PERMS,
    tau: float = DEFAULT_TAU,
    seed: int = 0,
    query_subsample: int | None = DEFAULT_QUERY_SUBSAMPLE,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], int]:
    """Average all scoring variants over multiple graphs.

    Parameters
    ----------
    model : GraphormerEncoder
        The pre-trained encoder (on *device*).
    smiles_list : list[str]
        SMILES strings to evaluate.
    device : torch.device
        Computation device.
    n_graphs : int
        Number of graphs to process (from the start of *smiles_list*).
    num_perms, tau, seed, query_subsample
        Forwarded to :func:`score_layer_all_variants`.
    verbose : bool
        If True, print progress every 10 graphs.

    Returns
    -------
    results_dict : dict[str, np.ndarray]
        Each value has shape ``[L, H]``.
    n_used : int
        Number of graphs successfully processed.
    """
    smiles_sub = smiles_list[:n_graphs]
    accum: dict[str, np.ndarray] | None = None
    n_used = 0

    for gi, smi in enumerate(smiles_sub):
        try:
            batch, n_atoms, dist = build_batch_from_smiles(smi, model=model, device=device)
            dot_list, bias_list, A_list = extract_attention(model, batch)
        except Exception:
            continue

        L = len(A_list)
        H = A_list[0].shape[0]

        # Collect per-layer results for this graph
        graph_results: dict[str, list[np.ndarray]] = {}
        for l_idx in range(L):
            layer_res = score_layer_all_variants(
                dot_all=dot_list[l_idx],
                bias_all=bias_list[l_idx],
                A_all=A_list[l_idx],
                num_perms=num_perms,
                tau=tau,
                seed=seed + 10_000 * gi + l_idx,
                query_subsample=query_subsample,
            )
            for key, val in layer_res.items():
                graph_results.setdefault(key, []).append(val)

        # Stack to [L, H] and accumulate
        graph_LH = {k: np.stack(v, axis=0) for k, v in graph_results.items()}

        if accum is None:
            accum = {k: v.copy() for k, v in graph_LH.items()}
        else:
            for k in accum:
                accum[k] += graph_LH[k]
        n_used += 1

        if verbose and (gi + 1) % 10 == 0:
            print(f"  [variants] {gi + 1}/{len(smiles_sub)} graphs processed ...")

    if accum is None:
        raise RuntimeError("No graphs were successfully processed.")

    results_dict = {k: v / max(n_used, 1) for k, v in accum.items()}

    if verbose:
        print(f"  [variants] Done. {n_used}/{len(smiles_sub)} graphs used.")

    return results_dict, n_used
