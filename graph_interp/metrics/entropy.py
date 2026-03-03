"""Attention entropy, Gini coefficient, and effective-keys metrics."""

import numpy as np
import torch

from ..extraction import extract_attention, node_only_conditional
from ..data import build_batch_from_smiles
from .welford import WelfordAccumulator


def entropy_bits(p: torch.Tensor, eps: float = 1e-12):
    """Shannon entropy in bits along last dimension."""
    p = p.clamp_min(eps)
    return -(p * torch.log2(p)).sum(dim=-1)


def gini_coefficient(p: torch.Tensor):
    """Gini coefficient along last dimension. p: [..., n], sums to 1."""
    n = p.shape[-1]
    if n <= 1:
        return torch.zeros(p.shape[:-1], device=p.device)
    sorted_p, _ = p.sort(dim=-1)
    index = torch.arange(1, n + 1, device=p.device, dtype=p.dtype)
    return (2.0 * (index * sorted_p).sum(dim=-1) / (n * sorted_p.sum(dim=-1).clamp_min(1e-12))) - (n + 1.0) / n


@torch.no_grad()
def head_entropy(A_all: torch.Tensor, normalize: bool = True, eps: float = 1e-12):
    """Per-head entropy from attention matrix.

    Returns: (ent_bits [H], ent_norm [H] or None, eff_keys [H], cls_mass_mean [H])
    """
    H, T, _ = A_all.shape
    n = T - 1

    p_node, cls_mass = node_only_conditional(A_all, eps=eps)
    Hq_bits = entropy_bits(p_node, eps=eps)
    ent_bits = Hq_bits.mean(dim=-1)

    ent_norm = None
    if normalize:
        denom = float(np.log2(max(n, 2)))
        ent_norm = ent_bits / max(denom, 1e-12)

    eff_keys = torch.exp2(ent_bits)
    cls_mass_mean = cls_mass.mean(dim=-1)

    return ent_bits, ent_norm, eff_keys, cls_mass_mean


@torch.no_grad()
def head_gini(A_all: torch.Tensor, eps: float = 1e-12):
    """Per-head Gini coefficient. Returns [H]."""
    p_node, _ = node_only_conditional(A_all, eps=eps)
    return gini_coefficient(p_node).mean(dim=-1)  # mean over queries


@torch.no_grad()
def aggregate_entropy(
    model, smiles_list: list[str], device: torch.device,
    normalize: bool = True,
    n_graphs: int | None = None,
    return_std: bool = False,
):
    """Average entropy metrics over graphs.

    Returns:
        When *return_std* is False (default):
            ``(mean_ent_bits [L,H], mean_ent_norm [L,H], mean_eff_keys [L,H],
              mean_cls_mass [L,H], mean_gini [L,H])``
        When *return_std* is True:
            ``(mean_ent_bits, mean_ent_norm, mean_eff_keys, mean_cls_mass,
              mean_gini, std_ent_bits, std_gini)``
            where ``std_*`` are [L,H] arrays of per-graph standard deviations.
    """
    if n_graphs is not None:
        smiles_list = smiles_list[:n_graphs]

    sums = {}
    used = 0

    # Welford accumulators for optional std computation
    if return_std:
        welf_bits = WelfordAccumulator()
        welf_gini = WelfordAccumulator()

    for smi in smiles_list:
        batch, n, dist = build_batch_from_smiles(smi, model=model, device=device)
        _, _, A_list = extract_attention(model, batch)

        L, H = len(A_list), A_list[0].shape[0]
        eb = torch.empty((L, H))
        en = torch.empty((L, H)) if normalize else None
        ek = torch.empty((L, H))
        cm = torch.empty((L, H))
        gi = torch.empty((L, H))

        for l_idx in range(L):
            bits, norm, eff, cls_m = head_entropy(A_list[l_idx], normalize=normalize)
            eb[l_idx] = bits.cpu()
            ek[l_idx] = eff.cpu()
            cm[l_idx] = cls_m.cpu()
            if normalize:
                en[l_idx] = norm.cpu()
            gi[l_idx] = head_gini(A_list[l_idx]).cpu()

        if not sums:
            sums = {"bits": eb, "eff": ek, "cls": cm, "gini": gi}
            if normalize:
                sums["norm"] = en
        else:
            sums["bits"] += eb
            sums["eff"] += ek
            sums["cls"] += cm
            sums["gini"] += gi
            if normalize:
                sums["norm"] += en
        used += 1

        if return_std:
            welf_bits.update(eb.numpy())
            welf_gini.update(gi.numpy())

    d = max(used, 1)
    mean_bits = (sums["bits"] / d).numpy()
    mean_norm = (sums["norm"] / d).numpy() if normalize else None
    mean_eff = (sums["eff"] / d).numpy()
    mean_cls = (sums["cls"] / d).numpy()
    mean_gini = (sums["gini"] / d).numpy()

    if return_std:
        _, std_bits, _ = welf_bits.finalize()
        _, std_gini, _ = welf_gini.finalize()
        return mean_bits, mean_norm, mean_eff, mean_cls, mean_gini, std_bits, std_gini

    return mean_bits, mean_norm, mean_eff, mean_cls, mean_gini
