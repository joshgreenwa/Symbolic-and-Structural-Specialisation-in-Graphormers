"""Logit spread (dot vs bias standard deviation) aggregation with Welford stats."""
from __future__ import annotations

import numpy as np
import torch

from .welford import WelfordAccumulator
from ..extraction import extract_attention
from ..data import build_batch_from_smiles
from ..config import DEFAULT_N_GRAPHS


@torch.no_grad()
def aggregate_logit_spread(
    model,
    smiles_list: list[str],
    device: torch.device,
    n_graphs: int = DEFAULT_N_GRAPHS,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute per-layer dot and bias logit spread statistics with error bars.

    For each graph the function extracts pre-softmax logits and measures
    how spread out the dot-product and bias components are across key
    positions (standard deviation along the last axis, averaged over
    queries).  A Welford accumulator is used so the final statistics
    include both the mean and standard deviation across graphs.

    Parameters
    ----------
    model : GraphormerEncoder
        Pre-trained encoder (on *device*).
    smiles_list : list[str]
        SMILES strings to evaluate.
    device : torch.device
        Computation device.
    n_graphs : int
        Number of graphs to process.
    verbose : bool
        Print progress every 10 graphs.

    Returns
    -------
    dict with keys:
        ``dot_std_mean``  : [L] mean of dot-product logit spread across graphs
        ``dot_std_std``   : [L] std  of dot-product logit spread across graphs
        ``bias_std_mean`` : [L] mean of bias logit spread across graphs
        ``bias_std_std``  : [L] std  of bias logit spread across graphs
        ``var_ratio``     : [L] mean Var(dot) / Var(bias) across graphs
        ``log_r_mean``    : [L, H] mean log10(bias_std / dot_std) per head
        ``log_r_std``     : [L, H] std  log10(bias_std / dot_std) per head
        ``n_used``        : int  number of graphs successfully processed
    """
    smiles_sub = smiles_list[:n_graphs]

    # We need to know L (number of layers) from the first successful graph.
    # Initialise accumulators lazily after the first graph.
    welford_dot: list[WelfordAccumulator] | None = None   # one per layer
    welford_bias: list[WelfordAccumulator] | None = None
    welford_var_ratio: list[WelfordAccumulator] | None = None
    welford_log_r: list[WelfordAccumulator] | None = None  # [L], each accumulates [H]

    n_used = 0

    for gi, smi in enumerate(smiles_sub):
        try:
            batch, n_atoms, dist = build_batch_from_smiles(smi, model=model, device=device)
            dot_list, bias_list, A_list = extract_attention(model, batch)
        except Exception:
            continue

        L = len(dot_list)
        H = dot_list[0].shape[0]

        # Lazy initialisation
        if welford_dot is None:
            welford_dot = [WelfordAccumulator() for _ in range(L)]
            welford_bias = [WelfordAccumulator() for _ in range(L)]
            welford_var_ratio = [WelfordAccumulator() for _ in range(L)]
            welford_log_r = [WelfordAccumulator() for _ in range(L)]

        for l in range(L):
            # Node-only slices: exclude CLS (index 0)
            dot_nn = dot_list[l][:, 1:, 1:].float()   # [H, n, n]
            bias_nn = bias_list[l][:, 1:, 1:].float()  # [H, n, n]

            # Per-head spread: std over keys, mean over queries -> [H]
            ds = dot_nn.std(dim=-1).mean(dim=-1).cpu().numpy()   # [H]
            bs = bias_nn.std(dim=-1).mean(dim=-1).cpu().numpy()  # [H]

            # Accumulate layer-level aggregates (mean over heads for this graph)
            welford_dot[l].update(np.array(ds.mean()))
            welford_bias[l].update(np.array(bs.mean()))

            # Variance ratio: mean Var(dot) / mean Var(bias)
            var_dot = (ds ** 2).mean()
            var_bias = (bs ** 2).mean()
            welford_var_ratio[l].update(np.array(var_dot / max(var_bias, 1e-12)))

            # Per-head log ratio
            log_r = np.log10(np.clip(bs / np.clip(ds, 1e-12, None), 1e-12, None))
            welford_log_r[l].update(log_r)

        n_used += 1

        if verbose and (gi + 1) % 10 == 0:
            print(f"  [logit_spread] {gi + 1}/{len(smiles_sub)} graphs processed ...")

    if welford_dot is None:
        raise RuntimeError("No graphs were successfully processed.")

    L = len(welford_dot)

    # Finalize Welford accumulators
    dot_std_mean = np.empty(L)
    dot_std_std = np.empty(L)
    bias_std_mean = np.empty(L)
    bias_std_std = np.empty(L)
    var_ratio = np.empty(L)

    log_r_mean_list: list[np.ndarray] = []
    log_r_std_list: list[np.ndarray] = []

    for l in range(L):
        m, s, _ = welford_dot[l].finalize()
        dot_std_mean[l] = float(m)
        dot_std_std[l] = float(s)

        m, s, _ = welford_bias[l].finalize()
        bias_std_mean[l] = float(m)
        bias_std_std[l] = float(s)

        m, s, _ = welford_var_ratio[l].finalize()
        var_ratio[l] = float(m)

        m, s, _ = welford_log_r[l].finalize()
        log_r_mean_list.append(m)
        log_r_std_list.append(s)

    log_r_mean = np.stack(log_r_mean_list, axis=0)  # [L, H]
    log_r_std = np.stack(log_r_std_list, axis=0)     # [L, H]

    if verbose:
        print(f"  [logit_spread] Done. {n_used}/{len(smiles_sub)} graphs used.")

    return {
        "dot_std_mean": dot_std_mean,
        "dot_std_std": dot_std_std,
        "bias_std_mean": bias_std_mean,
        "bias_std_std": bias_std_std,
        "var_ratio": var_ratio,
        "log_r_mean": log_r_mean,
        "log_r_std": log_r_std,
        "n_used": n_used,
    }
