"""CLS-token specific scoring, entropy, and similarity metrics."""

import numpy as np
import torch
import torch.nn.functional as F

from ..extraction import extract_attention, extract_hidden_states, node_only_conditional
from ..data import build_batch_from_smiles
from ..config import DEFAULT_NUM_PERMS, DEFAULT_TAU
from .scores import sample_transpositions, cossim_lastdim, _swap_perm
from .entropy import entropy_bits
from .welford import WelfordAccumulator


# ── CLS-query conditional attention ──────────────────────────

def cls_conditional_over_nodes(A_all: torch.Tensor, eps: float = 1e-12):
    """CLS-query attention conditioned on node keys.

    Returns: (p_cls_nodes [H, n], cls_self [H], node_mass [H])
    """
    cls_self = A_all[:, 0, 0]
    p_nodes_raw = A_all[:, 0, 1:]
    node_mass = p_nodes_raw.sum(dim=-1)
    p_cls_nodes = p_nodes_raw / node_mass.unsqueeze(-1).clamp_min(eps)
    return p_cls_nodes, cls_self, node_mass


# ── CLS-query structural/symbolic scores ─────────────────────

@torch.no_grad()
def score_cls_struct_sym(
    dot_all, bias_all, A_all,
    num_perms=DEFAULT_NUM_PERMS, tau=DEFAULT_TAU,
    seed=0, eps=1e-12,
):
    """CLS-query version of structural/symbolic scoring.

    Returns: (s_struct [H], s_sym [H])
    """
    device = dot_all.device
    H, T, _ = dot_all.shape
    n = T - 1
    if n < 2:
        return torch.zeros(H, device=device), torch.zeros(H, device=device)

    dot_cls = dot_all[:, 0, 1:]
    bias_cls = bias_all[:, 0, 1:]
    p_base, _, _ = cls_conditional_over_nodes(A_all, eps=eps)

    uv_list = sample_transpositions(n=n, K=num_perms, seed=seed)

    inv_tau = 1.0 / max(float(tau), 1e-12)
    delta = torch.empty((num_perms, H), device=device, dtype=p_base.dtype)
    for t, (u, v) in enumerate(uv_list):
        delta[t] = (p_base[:, u] - p_base[:, v]).abs() * inv_tau
    alpha = torch.softmax(delta, dim=0)

    s_struct = torch.zeros(H, device=device, dtype=p_base.dtype)
    s_sym = torch.zeros(H, device=device, dtype=p_base.dtype)

    for t, (u, v) in enumerate(uv_list):
        perm = _swap_perm(n, u, v, device=device)
        logits = dot_cls[:, perm] + bias_cls
        p_pi = torch.softmax(logits.float(), dim=-1).to(p_base.dtype)
        pP = p_base[:, perm]
        w = alpha[t]
        s_struct += w * cossim_lastdim(p_pi, p_base)
        s_sym += w * cossim_lastdim(p_pi, pP)

    return s_struct, s_sym


# ── CLS entropy ──────────────────────────────────────────────

@torch.no_grad()
def cls_entropy(A_all: torch.Tensor, eps: float = 1e-12):
    """Entropy of CLS-query attention over node keys.

    Returns: (ent_bits [H], gini [H])
    """
    p_cls, _, _ = cls_conditional_over_nodes(A_all, eps=eps)
    ent = entropy_bits(p_cls, eps=eps)

    # Gini
    n = p_cls.shape[-1]
    sorted_p, _ = p_cls.sort(dim=-1)
    idx = torch.arange(1, n + 1, device=p_cls.device, dtype=p_cls.dtype)
    gini = (2.0 * (idx * sorted_p).sum(dim=-1) / (n * sorted_p.sum(dim=-1).clamp_min(eps))) - (n + 1.0) / n

    return ent, gini


# ── CLS similarity to mean of node representations ──────────

@torch.no_grad()
def cls_similarity_to_mean_nodes(hidden_list: list[torch.Tensor]):
    """cos(CLS, mean(nodes)) at each layer.

    Args:
        hidden_list: list[L] of [T, D] from extract_hidden_states.

    Returns: sim [L] numpy array
    """
    L = len(hidden_list)
    sim = np.empty(L)
    for l_idx in range(L):
        h = hidden_list[l_idx]
        cls_vec = h[0]
        node_mean = h[1:].mean(dim=0)
        cos = F.cosine_similarity(cls_vec.unsqueeze(0), node_mean.unsqueeze(0)).item()
        sim[l_idx] = cos
    return sim


# ── CLS dot-logit spread ────────────────────────────────────

@torch.no_grad()
def cls_dot_logit_spread(dot_list: list[torch.Tensor]):
    """Std of scaled dot-product logits from CLS query over node keys.

    Returns: cls_std [L], node_std [L] — averaged over heads
    """
    L = len(dot_list)
    cls_std = np.empty(L)
    node_std = np.empty(L)
    for l_idx in range(L):
        dot = dot_list[l_idx]  # [H, T, T]
        # CLS query -> node keys
        cls_logits = dot[:, 0, 1:]  # [H, n]
        cls_std[l_idx] = float(cls_logits.std(dim=-1).mean())
        # node queries -> node keys
        node_logits = dot[:, 1:, 1:]  # [H, n, n]
        node_std[l_idx] = float(node_logits.std(dim=-1).mean())
    return cls_std, node_std


# ── Aggregation ──────────────────────────────────────────────

@torch.no_grad()
def aggregate_cls_metrics(
    model, smiles_list: list[str], device: torch.device,
    num_perms=DEFAULT_NUM_PERMS, tau=DEFAULT_TAU, seed=0,
    n_graphs: int | None = None,
):
    """Average CLS metrics over graphs.

    Returns dict with keys:
        mean_struct, mean_sym, mean_diff: [L, H]
        mean_cls_ent, mean_cls_gini: [L, H]
        mean_cls_sim_to_mean: [L]
        mean_cls_dot_std, mean_node_dot_std: [L]
        mean_node_ent, mean_node_ent_norm: [L, H]
        std_struct, std_sym: [L, H]
        std_cls_ent, std_node_ent: [L, H]
        std_cls_sim: [L]
        std_cls_std, std_node_std: [L]
    """
    if n_graphs is not None:
        smiles_list = smiles_list[:n_graphs]

    sums: dict = {}
    used = 0

    # Welford accumulators for per-graph standard deviations
    welford: dict[str, WelfordAccumulator] = {
        "struct": WelfordAccumulator(),
        "sym": WelfordAccumulator(),
        "cls_ent": WelfordAccumulator(),
        "node_ent": WelfordAccumulator(),
        "cls_sim": WelfordAccumulator(),
        "cls_std": WelfordAccumulator(),
        "node_std": WelfordAccumulator(),
    }

    for gi, smi in enumerate(smiles_list):
        batch, n, dist = build_batch_from_smiles(smi, model=model, device=device)
        dot_list, bias_list, A_list = extract_attention(model, batch)
        _, hidden_list, _ = extract_hidden_states(model, batch)

        L, H = len(A_list), A_list[0].shape[0]

        struct_LH = torch.empty((L, H))
        sym_LH = torch.empty((L, H))
        cls_ent_LH = torch.empty((L, H))
        cls_gini_LH = torch.empty((L, H))
        node_ent_LH = torch.empty((L, H))

        for l_idx in range(L):
            ss, sy = score_cls_struct_sym(
                dot_list[l_idx], bias_list[l_idx], A_list[l_idx],
                num_perms=num_perms, tau=tau, seed=seed + 10_000 * gi + l_idx,
            )
            struct_LH[l_idx] = ss.cpu()
            sym_LH[l_idx] = sy.cpu()

            ce, cg = cls_entropy(A_list[l_idx])
            cls_ent_LH[l_idx] = ce.cpu()
            cls_gini_LH[l_idx] = cg.cpu()

            # node entropy for comparison
            p_node, _ = node_only_conditional(A_list[l_idx])
            node_ent_LH[l_idx] = entropy_bits(p_node).mean(dim=-1).cpu()

        cls_sim = cls_similarity_to_mean_nodes(hidden_list)
        cls_std, node_std = cls_dot_logit_spread(dot_list)

        vals = {
            "struct": struct_LH, "sym": sym_LH,
            "cls_ent": cls_ent_LH, "cls_gini": cls_gini_LH,
            "node_ent": node_ent_LH,
            "cls_sim": cls_sim, "cls_std": cls_std, "node_std": node_std,
        }

        # Feed Welford accumulators (convert to numpy for consistency)
        for k in welford:
            v = vals[k]
            welford[k].update(v.numpy() if hasattr(v, 'numpy') else np.asarray(v))

        if not sums:
            sums = {k: v.copy() if isinstance(v, np.ndarray) else v.clone() for k, v in vals.items()}
        else:
            for k in sums:
                sums[k] = sums[k] + vals[k]
        used += 1

    d = max(used, 1)
    out = {}
    for k, v in sums.items():
        arr = v / d
        out[k] = arr.numpy() if hasattr(arr, 'numpy') else arr

    out["diff"] = out["struct"] - out["sym"]

    # Add standard deviation arrays from Welford accumulators
    for k, acc in welford.items():
        _, std_arr, _ = acc.finalize()
        out["std_" + k] = std_arr

    return out
