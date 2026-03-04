"""Section M: Top-k routing diagnostics (Figures 54-58).

This module ports four advanced notebook analyses into reusable functions:
1) Restricted permutation scores + logit interaction summaries.
2) Channel attribution within top-k (full vs dot-only vs bias-only).
3) D-selection diagnostics within top-k.
4) Entropy delta H[softmax(D+B)] - H[softmax(B)].
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

from ..data import build_batch_from_smiles
from ..extraction import extract_attention, node_only_conditional
from ..metrics.scores import sample_transpositions, cossim_lastdim
from ..metrics.variants import _centered_cossim

EPS = 1e-12


def _line_with_band(
    ax,
    x: np.ndarray,
    mat: np.ndarray,
    color: str,
    label: str,
    ls: str = "-",
    lw: float = 2.0,
):
    """Plot layerwise mean with 10-90 percentile band from [L, H] matrix."""
    mu = np.nanmean(mat, axis=1)
    lo = np.nanpercentile(mat, 10, axis=1)
    hi = np.nanpercentile(mat, 90, axis=1)
    ax.fill_between(x, lo, hi, color=color, alpha=0.10)
    ax.plot(x, mu, "o-", color=color, label=label, lw=lw, ms=4, linestyle=ls)


def _normalised_node_attention(
    A_layer: torch.Tensor,
    dot_layer: torch.Tensor,
    bias_layer: torch.Tensor,
    n: int,
    eps: float = EPS,
):
    """Return node-only A, D, B tensors on CPU with shape [H, n, n]."""
    A_node = node_only_conditional(A_layer, eps=eps)[0].cpu().float()
    A_n = A_node / A_node.sum(-1, keepdim=True).clamp_min(eps)
    D = dot_layer[:, 1:n + 1, 1:n + 1].cpu().float()
    B = bias_layer[:, 1:n + 1, 1:n + 1].cpu().float()
    return A_n, D, B


@torch.no_grad()
def restricted_scores_layer(
    A_layer: torch.Tensor,
    dot_layer: torch.Tensor,
    bias_layer: torch.Tensor,
    n: int,
    top_k_frac: float = 0.40,
    num_perms: int = 64,
    tau: float = 0.02,
    seed: int = 0,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Partitioned structural/symbolic scores for one layer.

    Returns [H] arrays for:
        struct/sym/ctr_struct/ctr_sym across partitions all/wt/cross,
        plus mass_capture.
    """
    H = dot_layer.shape[0]
    num_perms = max(int(num_perms), 1)

    A_n, D, B = _normalised_node_attention(A_layer, dot_layer, bias_layer, n, eps=eps)

    k = max(2, int(n * top_k_frac))
    _, tk_idx = B.topk(k, dim=-1)
    tk_mask = torch.zeros(H, n, n, dtype=torch.bool)
    tk_mask.scatter_(2, tk_idx, True)

    uv_list = sample_transpositions(n=n, K=num_perms, seed=seed)

    inv_tau = 1.0 / max(float(tau), eps)
    delta = torch.empty(num_perms, H, n)
    for t, (u, v) in enumerate(uv_list):
        delta[t] = (A_n[:, :, u] - A_n[:, :, v]).abs() * inv_tau
    alpha = torch.softmax(delta, dim=0)

    cat_wt = torch.zeros(num_perms, H, n, dtype=torch.bool)
    cat_cross = torch.zeros_like(cat_wt)
    for t, (u, v) in enumerate(uv_list):
        u_in = tk_mask[:, :, u]
        v_in = tk_mask[:, :, v]
        cat_wt[t] = u_in & v_in
        cat_cross[t] = u_in ^ v_in

    # Score arrays: [K, H, n]
    s_struct = torch.zeros(num_perms, H, n)
    s_sym = torch.zeros(num_perms, H, n)
    cs_struct = torch.zeros(num_perms, H, n)
    cs_sym = torch.zeros(num_perms, H, n)

    for t, (u, v) in enumerate(uv_list):
        perm = torch.arange(n)
        perm[u], perm[v] = v, u
        logits_pi = D[:, :, perm] + B
        p_pi = torch.softmax(logits_pi, dim=-1)
        p_perm_ref = A_n[:, :, perm]

        s_struct[t] = cossim_lastdim(p_pi, A_n)
        s_sym[t] = cossim_lastdim(p_pi, p_perm_ref)
        cs_struct[t] = _centered_cossim(p_pi, A_n, eps=eps)
        cs_sym[t] = _centered_cossim(p_pi, p_perm_ref, eps=eps)

    def _weighted_mean(scores: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor | None = None):
        w = weights.clone()
        if mask is not None:
            w = w * mask.float()
        denom = w.sum(0).clamp_min(eps)
        return ((w * scores).sum(0) / denom).mean(-1).numpy()

    score_names = ["struct", "sym", "ctr_struct", "ctr_sym"]
    score_tensors = [s_struct, s_sym, cs_struct, cs_sym]
    partitions = [("all", None), ("wt", cat_wt), ("cross", cat_cross)]

    out: dict[str, np.ndarray] = {}
    for score_name, score_tensor in zip(score_names, score_tensors):
        for partition_name, partition_mask in partitions:
            out[f"{score_name}_{partition_name}"] = _weighted_mean(
                score_tensor,
                alpha,
                partition_mask,
            )

    mass_in_topk = (A_n * tk_mask.float()).sum(-1)
    out["mass_capture"] = mass_in_topk.mean(-1).numpy()

    return out


@torch.no_grad()
def aggregate_restricted_scores(
    model,
    smiles_list: list[str],
    device: torch.device,
    n_graphs: int = 30,
    top_k_frac: float = 0.40,
    num_perms: int = 64,
    tau: float = 0.02,
    verbose: bool = True,
) -> dict[str, np.ndarray | int | float]:
    """Aggregate restricted permutation scores over multiple graphs.

    Returns dict with score arrays [L, H] and metadata keys:
    ``n_used``, ``L``, ``H``, ``top_k_frac``, ``num_perms``, ``tau``.
    """
    graphs = smiles_list[:n_graphs]

    keys: list[str] | None = None
    acc: dict[str, np.ndarray] | None = None
    cnt: np.ndarray | None = None
    used = 0
    L = 0
    H = 0

    for gi, smi in enumerate(graphs):
        try:
            batch, n_at, _ = build_batch_from_smiles(smi, model=model, device=device)
            if n_at < 6:
                continue
            dot_list, bias_list, A_list = extract_attention(model, batch)
        except Exception:
            continue

        if keys is None:
            L, H = len(dot_list), dot_list[0].shape[0]
            probe = restricted_scores_layer(
                A_list[0],
                dot_list[0],
                bias_list[0],
                n_at,
                top_k_frac=top_k_frac,
                num_perms=num_perms,
                tau=tau,
                seed=gi * 10000,
            )
            keys = list(probe.keys())
            acc = {k: np.zeros((L, H), dtype=float) for k in keys}
            cnt = np.zeros(L, dtype=float)

        for l in range(L):
            layer_scores = restricted_scores_layer(
                A_list[l],
                dot_list[l],
                bias_list[l],
                n_at,
                top_k_frac=top_k_frac,
                num_perms=num_perms,
                tau=tau,
                seed=gi * 10000 + l,
            )
            for k in keys:
                acc[k][l] += np.nan_to_num(layer_scores[k], nan=0.0)
            cnt[l] += 1.0

        used += 1
        if verbose and (gi + 1) % 10 == 0:
            print(f"  [restricted {gi + 1}/{len(graphs)}] ...")

    if keys is None or acc is None or cnt is None:
        raise RuntimeError("No valid graphs were processed for restricted scores.")

    out: dict[str, np.ndarray | int | float] = {
        k: acc[k] / np.maximum(cnt[:, None], 1.0)
        for k in keys
    }
    out["n_used"] = used
    out["L"] = L
    out["H"] = H
    out["top_k_frac"] = float(top_k_frac)
    out["num_perms"] = int(num_perms)
    out["tau"] = float(tau)
    return out


@torch.no_grad()
def aggregate_mass_capture_by_frac(
    model,
    smiles_list: list[str],
    device: torch.device,
    n_graphs: int = 30,
    top_k_fracs: Iterable[float] = (0.20, 0.30, 0.40, 0.50),
    verbose: bool = True,
) -> dict[str, object]:
    """Aggregate A-mass captured by B's top-k for multiple k fractions.

    Returns dict with keys:
    - ``mass_capture_by_frac``: {frac: [L, H]}
    - ``top_k_fracs``: list[float]
    - ``n_used``, ``L``, ``H``
    """
    frac_list = [float(f) for f in top_k_fracs]
    graphs = smiles_list[:n_graphs]

    acc_by_frac: dict[float, np.ndarray] | None = None
    cnt: np.ndarray | None = None
    used = 0
    L = 0
    H = 0

    for gi, smi in enumerate(graphs):
        try:
            batch, n_at, _ = build_batch_from_smiles(smi, model=model, device=device)
            if n_at < 6:
                continue
            dot_list, bias_list, A_list = extract_attention(model, batch)
        except Exception:
            continue

        if acc_by_frac is None:
            L, H = len(dot_list), dot_list[0].shape[0]
            acc_by_frac = {f: np.zeros((L, H), dtype=float) for f in frac_list}
            cnt = np.zeros(L, dtype=float)

        for l in range(L):
            A_n, _, B = _normalised_node_attention(A_list[l], dot_list[l], bias_list[l], n_at)
            for frac in frac_list:
                k = max(2, int(n_at * frac))
                _, idx = B.topk(k, dim=-1)
                mask = torch.zeros(H, n_at, n_at, dtype=torch.bool)
                mask.scatter_(2, idx, True)
                mass = (A_n * mask.float()).sum(-1).mean(-1).numpy()
                acc_by_frac[frac][l] += np.nan_to_num(mass, nan=0.0)
            cnt[l] += 1.0

        used += 1
        if verbose and (gi + 1) % 10 == 0:
            print(f"  [mass-capture {gi + 1}/{len(graphs)}] ...")

    if acc_by_frac is None or cnt is None:
        raise RuntimeError("No valid graphs were processed for mass-capture diagnostics.")

    out_vals = {f: arr / np.maximum(cnt[:, None], 1.0) for f, arr in acc_by_frac.items()}
    return {
        "mass_capture_by_frac": out_vals,
        "top_k_fracs": frac_list,
        "n_used": used,
        "L": L,
        "H": H,
    }


def fig54_partitioned_scores(restricted_results: dict[str, np.ndarray | int | float], N: int | None = None) -> Figure:
    """Fig 54: 4x3 panel of score partitions and score gaps across layers."""
    L = int(restricted_results["L"])
    layers = np.arange(L)

    score_types = [
        ("struct", r"$\bar{s}_{\mathrm{struct}}$"),
        ("sym", r"$\bar{s}_{\mathrm{sym}}$"),
        ("ctr_struct", r"$\tilde{s}_{\mathrm{struct}}$"),
        ("ctr_sym", r"$\tilde{s}_{\mathrm{sym}}$"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 18))

    for row, (score_name, score_label) in enumerate(score_types):
        # Col 0: absolute scores by partition
        ax = axes[row, 0]
        _line_with_band(ax, layers, restricted_results[f"{score_name}_all"], "#333333", "All")
        _line_with_band(ax, layers, restricted_results[f"{score_name}_cross"], "#c1121f", "Cross-boundary")
        _line_with_band(ax, layers, restricted_results[f"{score_name}_wt"], "#0077b6", "Within top-$k$")
        ax.set_ylabel(score_label, fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(layers)
        if row == 0:
            ax.set_title("Scores by partition")

        # Col 1: within-top-k minus all
        ax = axes[row, 1]
        gap_wt = restricted_results[f"{score_name}_wt"] - restricted_results[f"{score_name}_all"]
        _line_with_band(ax, layers, gap_wt, "#2a9d8f", "Within top-$k$ $-$ All")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_ylabel(f"$\\Delta$ {score_label}", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(layers)
        if row == 0:
            ax.set_title("Score gap (top-$k$ $-$ all)")

        # Col 2: cross minus all
        ax = axes[row, 2]
        gap_cross = restricted_results[f"{score_name}_cross"] - restricted_results[f"{score_name}_all"]
        _line_with_band(ax, layers, gap_cross, "#e76f51", "Cross $-$ All")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_ylabel(f"$\\Delta$ {score_label}", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(layers)
        if row == 0:
            ax.set_title("Score gap (cross $-$ all)")

    for ax in axes[-1, :]:
        ax.set_xlabel("Layer")

    n_show = int(restricted_results["n_used"]) if N is None else int(N)
    top_k_frac = float(restricted_results["top_k_frac"])
    fig.suptitle(
        f"Restricted permutation scores by partition ({n_show} graphs, top-$k$={top_k_frac:.2f})",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def fig55_mass_capture_multi(
    mass_capture_by_frac: dict[float, np.ndarray],
    N: int | None = None,
) -> Figure:
    """Fig 55: Layerwise top-k mass capture for several k fractions."""
    first = next(iter(mass_capture_by_frac.values()))
    L = first.shape[0]
    layers = np.arange(L)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#8d99ae", "#5e548e"]

    for i, frac in enumerate(sorted(mass_capture_by_frac.keys())):
        arr = mass_capture_by_frac[frac]
        _line_with_band(ax, layers, arr, colors[i % len(colors)], f"top-{int(frac * 100)}%")
        ax.axhline(frac, color=colors[i % len(colors)], ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of A mass captured")
    title = "Routing effectiveness at different top-$k$ thresholds"
    if N is not None:
        title += f" ({N} graphs)"
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(layers)
    fig.tight_layout()
    return fig


def top_restricted_heads(
    restricted_results: dict[str, np.ndarray | int | float],
    top_n: int = 10,
) -> list[dict[str, float | int]]:
    """Rank heads by mass-capture * positive symbolic gap."""
    H = int(restricted_results["H"])
    sym_gap = restricted_results["sym_wt"] - restricted_results["sym_all"]
    composite = restricted_results["mass_capture"] * np.maximum(sym_gap, 0.0)

    flat = np.argsort(composite.ravel())[::-1]
    ranked: list[dict[str, float | int]] = []
    for idx in flat[: max(top_n, 1)]:
        layer, head = int(idx // H), int(idx % H)
        ranked.append(
            {
                "layer": layer,
                "head": head,
                "mass_capture": float(restricted_results["mass_capture"][layer, head]),
                "sym_wt": float(restricted_results["sym_wt"][layer, head]),
                "sym_all": float(restricted_results["sym_all"][layer, head]),
                "sym_gap": float(sym_gap[layer, head]),
                "composite": float(composite[layer, head]),
            }
        )
    return ranked


@torch.no_grad()
def fig55b_restricted_logit_case(
    model,
    smiles_list: list[str],
    device: torch.device,
    layer: int,
    head: int,
    top_k_frac: float = 0.40,
    n_examples: int = 2,
    eps: float = EPS,
) -> list[Figure]:
    """Optional case-study figures for restricted-score interpretation.

    Returns a list of figures, one per example molecule.
    """
    n_examples = max(1, int(n_examples))
    figures: list[Figure] = []

    for ex in range(min(n_examples, len(smiles_list))):
        batch, n, dist = build_batch_from_smiles(smiles_list[ex], model=model, device=device)
        dot_list, bias_list, A_list = extract_attention(model, batch)

        A_n = node_only_conditional(A_list[layer], eps=eps)[0][head].cpu().float()
        A_n = A_n / A_n.sum(-1, keepdim=True).clamp_min(eps)
        D = dot_list[layer][head, 1:n + 1, 1:n + 1].cpu().float()
        B = bias_list[layer][head, 1:n + 1, 1:n + 1].cpu().float()
        sp = np.array(dist)[:n, :n]

        D_np, B_np, A_np = D.numpy(), B.numpy(), A_n.numpy()
        k = max(2, int(n * top_k_frac))

        # Choose high / median / low mass-capture queries.
        mc_per_q = np.array([A_np[q, np.argsort(B_np[q])[-k:]].sum() for q in range(n)])
        q_sorted = np.argsort(mc_per_q)
        q_indices = [int(q_sorted[-1]), int(q_sorted[len(q_sorted) // 2]), int(q_sorted[0])]

        fig = plt.figure(figsize=(22, 12))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.35)

        # Top row: D/B/A heatmaps.
        ax_d = fig.add_subplot(gs[0, 0])
        d_clip = np.percentile(np.abs(D_np), 95)
        norm_d = TwoSlopeNorm(vmin=-d_clip, vcenter=0, vmax=d_clip)
        im_d = ax_d.imshow(D_np, cmap="RdBu_r", norm=norm_d, aspect="auto")
        ax_d.set_title("$D$ (dot-product logits)", fontsize=11)
        ax_d.set_xlabel("Key")
        ax_d.set_ylabel("Query")
        fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04, label="Logit")

        ax_b = fig.add_subplot(gs[0, 1])
        b_clip = np.percentile(np.abs(B_np), 95)
        norm_b = TwoSlopeNorm(vmin=-b_clip, vcenter=0, vmax=b_clip)
        im_b = ax_b.imshow(B_np, cmap="PuOr_r", norm=norm_b, aspect="auto")
        ax_b.set_title("$B$ (bias logits)", fontsize=11)
        ax_b.set_xlabel("Key")
        fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04, label="Logit")

        ax_a = fig.add_subplot(gs[0, 2])
        a_plot = np.maximum(A_np, eps)
        a_clip_lo = np.percentile(a_plot[a_plot > eps], 5)
        a_clip_hi = np.percentile(a_plot, 99)
        im_a = ax_a.imshow(a_plot, cmap="YlOrRd", aspect="auto", vmin=a_clip_lo, vmax=a_clip_hi)
        ax_a.set_title("$A$ (attention weights)", fontsize=11)
        ax_a.set_xlabel("Key")
        fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04, label="Probability")

        for q in q_indices:
            for ax in (ax_d, ax_b, ax_a):
                ax.axhline(q - 0.5, color="lime", lw=1.2, ls="--", alpha=0.8)
                ax.axhline(q + 0.5, color="lime", lw=1.2, ls="--", alpha=0.8)

        # Bottom row: per-query decomposition.
        for panel_idx, q in enumerate(q_indices):
            ax = fig.add_subplot(gs[1, panel_idx])
            d_row = D_np[q]
            b_row = B_np[q]
            comb = d_row + b_row
            a_row = A_np[q]

            order = np.argsort(b_row)[::-1]
            x = np.arange(n)
            w = 0.22

            ax.bar(x - w, b_row[order], w, color="#0077b6", label="$B$ logits", alpha=0.85, zorder=3)
            ax.bar(x, d_row[order], w, color="#e63946", label="$D$ logits", alpha=0.85, zorder=3)
            ax.bar(x + w, comb[order], w, color="#6a4c93", label="$D+B$ logits", alpha=0.75, zorder=3)
            ax.axhline(0, color="k", lw=0.5, zorder=2)
            ax.set_ylabel("Logit value", fontsize=9)

            ax.axvspan(-0.5, k - 0.5, alpha=0.08, color="#0077b6", zorder=1, label=f"B top-{k}")
            ax.axvline(k - 0.5, color="#0077b6", ls=":", lw=1, alpha=0.5)

            ax2 = ax.twinx()
            markerline, stemlines, _ = ax2.stem(x, a_row[order], linefmt="-", markerfmt="o", basefmt=" ")
            plt.setp(stemlines, color="#333333", linewidth=0.8, alpha=0.6)
            plt.setp(markerline, color="#333333", markersize=4, zorder=5)
            ax2.set_ylabel("Attention $A$", fontsize=9, color="#555555")
            ax2.tick_params(axis="y", labelcolor="#555555")
            ax2.set_ylim(bottom=0)

            mc_q = mc_per_q[q]
            mean_hop = np.mean(sp[q, np.argsort(B_np[q])[-k:]])
            ax.set_title(f"Query {q} (top-k mass={mc_q:.2f}, mean hop={mean_hop:.2f})", fontsize=10)
            ax.set_xlabel("Keys sorted by $B$ logits")

            if panel_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=7, loc="upper right", ncol=2)

            if n <= 20:
                ax.set_xticks(x)
                ax.set_xticklabels(order, fontsize=6)
            else:
                ax.set_xticks(x[::3])
                ax.set_xticklabels(order[::3], fontsize=6)

        fig.suptitle(f"L{layer}, H{head} — example {ex} ({n} atoms)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        figures.append(fig)

    return figures


@torch.no_grad()
def channel_attribution_layer(
    A_layer: torch.Tensor,
    dot_layer: torch.Tensor,
    bias_layer: torch.Tensor,
    n: int,
    top_k_frac: float = 0.40,
    num_perms: int = 64,
    tau: float = 0.02,
    seed: int = 0,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Channel attribution for symbolic score within top-k at one layer."""
    H = dot_layer.shape[0]
    num_perms = max(int(num_perms), 1)

    A_n, D, B = _normalised_node_attention(A_layer, dot_layer, bias_layer, n, eps=eps)

    k = max(2, int(n * top_k_frac))
    _, tk_idx = B.topk(k, dim=-1)
    tk_mask = torch.zeros(H, n, n, dtype=torch.bool)
    tk_mask.scatter_(2, tk_idx, True)

    A_B = torch.softmax(B, dim=-1)
    A_B = A_B / A_B.sum(-1, keepdim=True).clamp_min(eps)
    A_D = torch.softmax(D, dim=-1)
    A_D = A_D / A_D.sum(-1, keepdim=True).clamp_min(eps)

    uv_list = sample_transpositions(n=n, K=num_perms, seed=seed)

    inv_tau = 1.0 / max(float(tau), eps)
    delta = torch.empty(num_perms, H, n)
    for t, (u, v) in enumerate(uv_list):
        delta[t] = (A_n[:, :, u] - A_n[:, :, v]).abs() * inv_tau
    alpha = torch.softmax(delta, dim=0)

    cat_wt = torch.zeros(num_perms, H, n, dtype=torch.bool)
    for t, (u, v) in enumerate(uv_list):
        cat_wt[t] = tk_mask[:, :, u] & tk_mask[:, :, v]

    # Score arrays: [K, H, n]
    sym_full = torch.zeros(num_perms, H, n)
    sym_bonly = torch.zeros(num_perms, H, n)
    sym_donly = torch.zeros(num_perms, H, n)
    csym_full = torch.zeros(num_perms, H, n)
    csym_bonly = torch.zeros(num_perms, H, n)
    csym_donly = torch.zeros(num_perms, H, n)

    for t, (u, v) in enumerate(uv_list):
        perm = torch.arange(n)
        perm[u], perm[v] = v, u

        # Full
        p_pi = torch.softmax(D[:, :, perm] + B, dim=-1)
        ref = A_n[:, :, perm]
        sym_full[t] = cossim_lastdim(p_pi, ref)
        csym_full[t] = _centered_cossim(p_pi, ref, eps=eps)

        # Bias-only (B unchanged under perm, but reference permutes)
        ref_b = A_B[:, :, perm]
        sym_bonly[t] = cossim_lastdim(A_B, ref_b)
        csym_bonly[t] = _centered_cossim(A_B, ref_b, eps=eps)

        # Dot-only
        p_pi_d = torch.softmax(D[:, :, perm], dim=-1)
        ref_d = A_D[:, :, perm]
        sym_donly[t] = cossim_lastdim(p_pi_d, ref_d)
        csym_donly[t] = _centered_cossim(p_pi_d, ref_d, eps=eps)

    def _weighted_mean(scores: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor):
        w = weights * mask.float()
        denom = w.sum(0).clamp_min(eps)
        return ((w * scores).sum(0) / denom).mean(-1).numpy()

    return {
        "sym_full_wt": _weighted_mean(sym_full, alpha, cat_wt),
        "sym_bonly_wt": _weighted_mean(sym_bonly, alpha, cat_wt),
        "sym_donly_wt": _weighted_mean(sym_donly, alpha, cat_wt),
        "csym_full_wt": _weighted_mean(csym_full, alpha, cat_wt),
        "csym_bonly_wt": _weighted_mean(csym_bonly, alpha, cat_wt),
        "csym_donly_wt": _weighted_mean(csym_donly, alpha, cat_wt),
    }


@torch.no_grad()
def aggregate_channel_attribution(
    model,
    smiles_list: list[str],
    device: torch.device,
    n_graphs: int = 30,
    top_k_frac: float = 0.40,
    num_perms: int = 64,
    tau: float = 0.02,
    verbose: bool = True,
) -> dict[str, np.ndarray | int | float]:
    """Aggregate channel attribution diagnostics over multiple graphs."""
    graphs = smiles_list[:n_graphs]
    keys = [
        "sym_full_wt",
        "sym_bonly_wt",
        "sym_donly_wt",
        "csym_full_wt",
        "csym_bonly_wt",
        "csym_donly_wt",
    ]

    acc: dict[str, np.ndarray] | None = None
    cnt: np.ndarray | None = None
    used = 0
    L = 0
    H = 0

    for gi, smi in enumerate(graphs):
        try:
            batch, n_at, _ = build_batch_from_smiles(smi, model=model, device=device)
            if n_at < 6:
                continue
            dot_list, bias_list, A_list = extract_attention(model, batch)
        except Exception:
            continue

        if acc is None:
            L, H = len(dot_list), dot_list[0].shape[0]
            acc = {k: np.zeros((L, H), dtype=float) for k in keys}
            cnt = np.zeros(L, dtype=float)

        for l in range(L):
            layer_scores = channel_attribution_layer(
                A_list[l],
                dot_list[l],
                bias_list[l],
                n_at,
                top_k_frac=top_k_frac,
                num_perms=num_perms,
                tau=tau,
                seed=gi * 10000 + l,
            )
            for k in keys:
                acc[k][l] += np.nan_to_num(layer_scores[k], nan=0.0)
            cnt[l] += 1.0

        used += 1
        if verbose and (gi + 1) % 10 == 0:
            print(f"  [channel {gi + 1}/{len(graphs)}] ...")

    if acc is None or cnt is None:
        raise RuntimeError("No valid graphs were processed for channel attribution.")

    out: dict[str, np.ndarray | int | float] = {
        k: acc[k] / np.maximum(cnt[:, None], 1.0)
        for k in keys
    }
    out["n_used"] = used
    out["L"] = L
    out["H"] = H
    out["top_k_frac"] = float(top_k_frac)
    return out


def fig56_channel_attribution(channel_results: dict[str, np.ndarray | int | float], N: int | None = None) -> Figure:
    """Fig 56: Channel attribution (full vs dot-only vs bias-only) within top-k."""
    L = int(channel_results["L"])
    layers = np.arange(L)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (prefix, ylabel) in enumerate([
        ("sym", r"$\bar{s}_{\mathrm{sym}}$ within top-$k$"),
        ("csym", r"$\bar{s}_{\mathrm{sym}}^{\mathrm{ctr}}$ within top-$k$"),
    ]):
        ax = axes[idx]
        for key, color, label in [
            (f"{prefix}_full_wt", "#333333", "Full (D+B)"),
            (f"{prefix}_donly_wt", "#e63946", "Dot-only (D)"),
            (f"{prefix}_bonly_wt", "#0077b6", "Bias-only (B)"),
        ]:
            _line_with_band(ax, layers, channel_results[key], color, label)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Channel attribution: {ylabel}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(layers)

    n_show = int(channel_results["n_used"]) if N is None else int(N)
    fig.suptitle(f"Dot vs bias attribution within top-$k$ ({n_show} graphs)", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def _rank_last(x: np.ndarray):
    idx = np.argsort(x, axis=-1)
    ranks = np.empty_like(idx, dtype=float)
    np.put_along_axis(ranks, idx, np.arange(x.shape[-1], dtype=float), axis=-1)
    return ranks


def _spearman_batch(x: np.ndarray, y: np.ndarray, eps: float = EPS):
    rx, ry = _rank_last(x), _rank_last(y)
    rx -= rx.mean(-1, keepdims=True)
    ry -= ry.mean(-1, keepdims=True)
    num = (rx * ry).sum(-1)
    den = np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1)) + eps
    return num / den


@torch.no_grad()
def d_selection_layer_diag(
    A_layer: torch.Tensor,
    dot_layer: torch.Tensor,
    bias_layer: torch.Tensor,
    n: int,
    top_k_frac: float = 0.40,
    n_shuf: int = 10,
    seed: int = 0,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Vectorized diagnostics for whether D selects within B's top-k."""
    H = dot_layer.shape[0]
    A_n, D, B = _normalised_node_attention(A_layer, dot_layer, bias_layer, n, eps=eps)

    k = max(2, int(n * top_k_frac))

    _, tk_idx = B.topk(k, dim=-1)   # [H, n, k]
    A_tk = A_n.gather(-1, tk_idx)   # [H, n, k]
    D_tk = D.gather(-1, tk_idx)
    B_tk = B.gather(-1, tk_idx)

    A_tk_r = A_tk / A_tk.sum(-1, keepdim=True).clamp_min(eps)

    # T1: Counterfactual TV
    real_logits = D_tk + B_tk
    real_p = torch.softmax(real_logits, dim=-1)
    bonly_p = torch.softmax(B_tk, dim=-1)

    tv_zero = 0.5 * (real_p - bonly_p).abs().sum(-1).mean(-1).numpy()

    # Shuffled D within top-k
    n_shuf = max(int(n_shuf), 1)
    rng = np.random.RandomState(seed)
    tv_shuf_acc = torch.zeros(H, n)
    for _ in range(n_shuf):
        perm_idx = torch.from_numpy(np.argsort(rng.rand(H, n, k), axis=-1).astype(np.int64))
        D_shuf = D_tk.gather(-1, perm_idx)
        shuf_p = torch.softmax(D_shuf + B_tk, dim=-1)
        tv_shuf_acc += 0.5 * (real_p - shuf_p).abs().sum(-1)
    tv_shuffled = (tv_shuf_acc / n_shuf).mean(-1).numpy()

    # Rank correlation within top-k
    A_np = A_tk_r.numpy()
    D_np = D_tk.numpy()
    B_np = B_tk.numpy()
    rho_DA = _spearman_batch(D_np, A_np, eps=eps).mean(1)
    rho_BA = _spearman_batch(B_np, A_np, eps=eps).mean(1)

    # Entropy within top-k
    max_ent = np.log(k) if k > 1 else 1.0
    ent_A = (-(A_tk_r * (A_tk_r + eps).log()).sum(-1) / max_ent).mean(-1).numpy()

    B_att_tk = torch.softmax(B_tk, dim=-1)
    ent_B = (-(B_att_tk * (B_att_tk + eps).log()).sum(-1) / max_ent).mean(-1).numpy()

    D_att_tk = torch.softmax(D_tk, dim=-1)
    ent_D = (-(D_att_tk * (D_att_tk + eps).log()).sum(-1) / max_ent).mean(-1).numpy()

    # Logit spread inside top-k
    D_std = D_tk.std(-1).mean(-1).numpy()
    B_std = B_tk.std(-1).mean(-1).numpy()

    return {
        "tv_zero": tv_zero,
        "tv_shuffled": tv_shuffled,
        "rho_DA": rho_DA,
        "rho_BA": rho_BA,
        "ent_A_topk": ent_A,
        "ent_B_topk": ent_B,
        "ent_D_topk": ent_D,
        "D_std_topk": D_std,
        "B_std_topk": B_std,
    }


@torch.no_grad()
def aggregate_d_selection(
    model,
    smiles_list: list[str],
    device: torch.device,
    n_graphs: int = 50,
    top_k_frac: float = 0.40,
    n_shuf: int = 10,
    verbose: bool = True,
) -> dict[str, np.ndarray | int | float]:
    """Aggregate D-selection diagnostics across graphs."""
    graphs = smiles_list[:n_graphs]
    keys = [
        "tv_zero",
        "tv_shuffled",
        "rho_DA",
        "rho_BA",
        "ent_A_topk",
        "ent_B_topk",
        "ent_D_topk",
        "D_std_topk",
        "B_std_topk",
    ]

    acc: dict[str, np.ndarray] | None = None
    cnt: np.ndarray | None = None
    used = 0
    L = 0
    H = 0

    for gi, smi in enumerate(graphs):
        try:
            batch, n_at, _ = build_batch_from_smiles(smi, model=model, device=device)
            if n_at < 6:
                continue
            dot_list, bias_list, A_list = extract_attention(model, batch)
        except Exception:
            continue

        if acc is None:
            L, H = len(dot_list), dot_list[0].shape[0]
            acc = {k: np.zeros((L, H), dtype=float) for k in keys}
            cnt = np.zeros(L, dtype=float)

        for l in range(L):
            layer_diag = d_selection_layer_diag(
                A_list[l],
                dot_list[l],
                bias_list[l],
                n_at,
                top_k_frac=top_k_frac,
                n_shuf=n_shuf,
                seed=gi * 10000 + l,
            )
            for k in keys:
                acc[k][l] += np.nan_to_num(layer_diag[k], nan=0.0)
            cnt[l] += 1.0

        used += 1
        if verbose and (gi + 1) % 10 == 0:
            print(f"  [d-select {gi + 1}/{len(graphs)}] ...")

    if acc is None or cnt is None:
        raise RuntimeError("No valid graphs were processed for D-selection diagnostics.")

    out: dict[str, np.ndarray | int | float] = {
        k: acc[k] / np.maximum(cnt[:, None], 1.0)
        for k in keys
    }
    out["n_used"] = used
    out["L"] = L
    out["H"] = H
    out["top_k_frac"] = float(top_k_frac)
    out["n_shuf"] = int(n_shuf)
    return out


def fig57_d_selection_summary(d_selection_results: dict[str, np.ndarray | int | float], N: int | None = None) -> Figure:
    """Fig 57: 2x3 summary panel for D-selection diagnostics."""
    L = int(d_selection_results["L"])
    H = int(d_selection_results["H"])
    layers = np.arange(L)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    _line_with_band(ax, layers, d_selection_results["tv_zero"], "#c1121f", "D zeroed (B only)")
    _line_with_band(ax, layers, d_selection_results["tv_shuffled"], "#e9c46a", "D shuffled")
    ax.set_title("(a) TV(real, counterfactual) within top-$k$")
    ax.set_ylabel("Total variation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")

    ax = axes[0, 1]
    _line_with_band(ax, layers, d_selection_results["rho_DA"], "#e63946", "Spearman(D, A)")
    _line_with_band(ax, layers, d_selection_results["rho_BA"], "#0077b6", "Spearman(B, A)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("(b) Ranking predictor for A within top-$k$")
    ax.set_ylabel("Spearman $\\rho$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")

    ax = axes[0, 2]
    _line_with_band(ax, layers, d_selection_results["ent_A_topk"], "#333333", "A")
    _line_with_band(ax, layers, d_selection_results["ent_B_topk"], "#0077b6", "softmax(B)")
    _line_with_band(ax, layers, d_selection_results["ent_D_topk"], "#e63946", "softmax(D)")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_title("(c) Normalised entropy within top-$k$")
    ax.set_ylabel("$H/H_{\\max}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")

    ax = axes[1, 0]
    _line_with_band(ax, layers, d_selection_results["D_std_topk"], "#e63946", "std(D logits)")
    _line_with_band(ax, layers, d_selection_results["B_std_topk"], "#0077b6", "std(B logits)")
    ax.set_title("(d) Logit spread within top-$k$")
    ax.set_ylabel("Std dev")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")

    layer_ids = np.repeat(np.arange(L), H)

    ax = axes[1, 1]
    sc = ax.scatter(
        d_selection_results["ent_A_topk"].ravel(),
        d_selection_results["tv_zero"].ravel(),
        c=layer_ids,
        cmap="viridis",
        s=18,
        alpha=0.6,
    )
    ax.set_xlabel("Entropy of A within top-$k$")
    ax.set_ylabel("TV(real, D-zeroed)")
    ax.set_title("(e) D impact vs A selectivity")
    ax.grid(True, alpha=0.2)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Layer")

    ax = axes[1, 2]
    sc2 = ax.scatter(
        d_selection_results["rho_DA"].ravel(),
        d_selection_results["tv_zero"].ravel(),
        c=layer_ids,
        cmap="viridis",
        s=18,
        alpha=0.6,
    )
    ax.set_xlabel("Spearman(D, A) within top-$k$")
    ax.set_ylabel("TV(real, D-zeroed)")
    ax.set_title("(f) Rank prediction vs causal effect")
    ax.grid(True, alpha=0.2)
    fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04, label="Layer")

    n_show = int(d_selection_results["n_used"]) if N is None else int(N)
    fig.suptitle(f"Does D select within top-$k$? ({n_show} graphs)", y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


def top_d_selection_heads(
    d_selection_results: dict[str, np.ndarray | int | float],
    min_layer: int = 3,
    top_n: int = 10,
) -> list[dict[str, float | int]]:
    """Rank heads by TV(real, D-zeroed), optionally ignoring early layers."""
    H = int(d_selection_results["H"])
    tv = np.array(d_selection_results["tv_zero"], copy=True)
    tv[: max(min_layer, 0)] = 0.0

    flat = np.argsort(tv.ravel())[::-1]
    ranked: list[dict[str, float | int]] = []
    for idx in flat[: max(top_n, 1)]:
        layer, head = int(idx // H), int(idx % H)
        ranked.append(
            {
                "layer": layer,
                "head": head,
                "tv_zero": float(d_selection_results["tv_zero"][layer, head]),
                "tv_shuffled": float(d_selection_results["tv_shuffled"][layer, head]),
                "rho_DA": float(d_selection_results["rho_DA"][layer, head]),
            }
        )
    return ranked


@torch.no_grad()
def fig57b_d_selection_deep_dive(
    model,
    smiles: str,
    device: torch.device,
    layer: int,
    head: int,
    top_k_frac: float = 0.40,
    eps: float = EPS,
) -> Figure:
    """Optional deep-dive figure for one (layer, head) on one molecule."""
    batch, n, dist = build_batch_from_smiles(smiles, model=model, device=device)
    dot_list, bias_list, A_list = extract_attention(model, batch)
    sp = np.array(dist)[:n, :n]

    A_n = node_only_conditional(A_list[layer], eps=eps)[0][head].cpu().float()
    A_n = A_n / A_n.sum(-1, keepdim=True).clamp_min(eps)
    D = dot_list[layer][head, 1:n + 1, 1:n + 1].cpu().float().numpy()
    B = bias_list[layer][head, 1:n + 1, 1:n + 1].cpu().float().numpy()
    A_np = A_n.numpy()
    k = max(2, int(n * top_k_frac))

    # Most selective and least selective query rows.
    selectivity = []
    for q in range(n):
        tk = np.argsort(B[q])[-k:]
        a_row = A_np[q, tk]
        a_row = a_row / (a_row.sum() + eps)
        h_norm = -(a_row * np.log(a_row + eps)).sum() / (np.log(k) if k > 1 else 1.0)
        selectivity.append(1.0 - h_norm)
    q_rows = [int(np.argmax(selectivity)), int(np.argmin(selectivity))]

    fig, axes = plt.subplots(len(q_rows), 3, figsize=(20, 5 * len(q_rows)))
    if len(q_rows) == 1:
        axes = axes[np.newaxis, :]

    for row, q in enumerate(q_rows):
        tk = np.sort(np.argsort(B[q])[-k:])
        hops = sp[q, tk]
        d_tk, b_tk, a_tk = D[q, tk], B[q, tk], A_np[q, tk]
        a_r = a_tk / (a_tk.sum() + eps)

        real_p = np.exp(d_tk + b_tk)
        real_p /= real_p.sum()
        bonly_p = np.exp(b_tk - b_tk.max())
        bonly_p /= bonly_p.sum()

        x = np.arange(len(tk))
        unique_hops = np.unique(hops)
        hop_colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_hops), 1)))
        hop_map = {hop: hop_colors[i] for i, hop in enumerate(unique_hops)}

        # Panel 1: logits
        w = 0.25
        ax = axes[row, 0]
        ax.bar(x - w, b_tk, w, color="#0077b6", alpha=0.7, label="B")
        ax.bar(x, d_tk, w, color=[hop_map[h] for h in hops], alpha=0.8, label="D (by hop)")
        ax.bar(x + w, d_tk + b_tk, w, color="#6a4c93", alpha=0.6, label="D+B")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{tk[i]}\\nh{int(hops[i])}" for i in range(len(tk))], fontsize=6)
        ax.set_ylabel("Logit")
        ax.set_title(f"Q{q}: logits (selectivity={selectivity[q]:.2f})")
        ax.legend(fontsize=7, ncol=3)

        # Panel 2: actual vs counterfactual probabilities
        ax = axes[row, 1]
        ax.plot(x, a_r, "ko-", lw=2, ms=6, label="A (actual)", zorder=5)
        ax.plot(x, real_p, "s--", color="#6a4c93", lw=1.5, ms=4, label="sm(D+B)")
        ax.plot(x, bonly_p, "^:", color="#0077b6", lw=1.5, ms=4, label="sm(B)")
        ax.fill_between(x, bonly_p, real_p, alpha=0.15, color="#e63946", label="D effect")
        ax.set_ylabel("Probability")
        ax.set_title("A vs B-only counterfactual")
        ax.legend(fontsize=7)

        # Panel 3: D-A scatter inside hop groups
        ax = axes[row, 2]
        for hop in unique_hops:
            mask = hops == hop
            if np.sum(mask) < 2:
                continue
            ax.scatter(
                d_tk[mask],
                a_r[mask],
                color=hop_map[hop],
                s=50,
                edgecolors="k",
                lw=0.5,
                label=f"Hop {int(hop)} ({np.sum(mask)})",
            )
        ax.set_xlabel("D logit")
        ax.set_ylabel("A weight")
        ax.set_title("D vs A within hop groups")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"L{layer}, H{head} — {n} atoms", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


@torch.no_grad()
def layer_entropy_delta(
    dot_layer: torch.Tensor,
    bias_layer: torch.Tensor,
    n: int,
    eps: float = EPS,
):
    """Per-head entropy delta H[softmax(D+B)] - H[softmax(B)] for one layer."""
    D = dot_layer[:, 1:n + 1, 1:n + 1].cpu().float()
    B = bias_layer[:, 1:n + 1, 1:n + 1].cpu().float()

    def _entropy(p: torch.Tensor):
        return -(p * (p + eps).log()).sum(-1).mean(-1)

    ent_bias = _entropy(torch.softmax(B, dim=-1))
    ent_full = _entropy(torch.softmax(D + B, dim=-1))
    return (ent_full - ent_bias).numpy()


@torch.no_grad()
def aggregate_entropy_delta(
    model,
    smiles_list: list[str],
    device: torch.device,
    n_graphs: int = 100,
    verbose: bool = True,
):
    """Aggregate entropy delta H[softmax(D+B)] - H[softmax(B)] over graphs.

    Returns ``(delta[L,H], n_used, L, H)``.
    """
    graphs = smiles_list[:n_graphs]

    acc: np.ndarray | None = None
    cnt: np.ndarray | None = None
    used = 0
    L = 0
    H = 0

    for gi, smi in enumerate(graphs):
        try:
            batch, n_at, _ = build_batch_from_smiles(smi, model=model, device=device)
            if n_at < 6:
                continue
            dot_list, bias_list, _ = extract_attention(model, batch)
        except Exception:
            continue

        if acc is None:
            L, H = len(dot_list), dot_list[0].shape[0]
            acc = np.zeros((L, H), dtype=float)
            cnt = np.zeros(L, dtype=float)

        for l in range(L):
            delta = layer_entropy_delta(dot_list[l], bias_list[l], n_at)
            acc[l] += np.nan_to_num(delta, nan=0.0)
            cnt[l] += 1.0

        used += 1
        if verbose and (gi + 1) % 20 == 0:
            print(f"  [entropy {gi + 1}/{len(graphs)}] ...")

    if acc is None or cnt is None:
        raise RuntimeError("No valid graphs were processed for entropy-delta diagnostics.")

    acc /= np.maximum(cnt[:, None], 1.0)
    return acc, used, L, H


def fig58_entropy_delta(delta: np.ndarray, n_used: int | None = None) -> Figure:
    """Fig 58: Bar chart of per-layer entropy delta with per-head scatter."""
    L, H = delta.shape
    layers = np.arange(L)
    mu = delta.mean(axis=1)
    colors = ["#0077b6" if v < 0 else "#c1121f" for v in mu]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(layers, mu, color=colors, width=0.7, zorder=3, alpha=0.85)

    # Per-head scatter to show spread
    for h_idx in range(H):
        jitter = np.random.default_rng(h_idx).uniform(-0.22, 0.22, L)
        ax.scatter(
            layers + jitter,
            delta[:, h_idx],
            s=12,
            color="#444444",
            alpha=0.18,
            zorder=2,
            edgecolors="none",
        )

    ax.axhline(0, color="k", lw=0.9, zorder=4)

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Entropy change (nats)", fontsize=13)
    title = "Does D reduce the entropy of B?"
    if n_used is not None:
        title += f" ({n_used} graphs)"
    ax.set_title(title, fontsize=14)
    ax.set_xticks(layers)
    ax.grid(True, axis="y", alpha=0.25)

    ax.legend(
        handles=[
            Patch(color="#0077b6", label="D sharpens (reduces entropy)"),
            Patch(color="#c1121f", label="D broadens (increases entropy)"),
        ],
        fontsize=11,
        framealpha=0.9,
        loc="best",
    )

    fig.tight_layout()
    return fig


def compute_section_m_results(
    model,
    smiles_list: list[str],
    device: torch.device,
    restricted_n_graphs: int = 30,
    restricted_top_k_frac: float = 0.40,
    restricted_num_perms: int = 64,
    restricted_tau: float = 0.02,
    mass_top_k_fracs: Iterable[float] = (0.20, 0.30, 0.40, 0.50),
    channel_n_graphs: int | None = None,
    channel_num_perms: int = 64,
    channel_tau: float = 0.02,
    d_select_n_graphs: int = 50,
    d_select_n_shuf: int = 10,
    entropy_n_graphs: int = 100,
    verbose: bool = True,
) -> dict[str, object]:
    """Compute all Section M diagnostics in one call."""
    if channel_n_graphs is None:
        channel_n_graphs = restricted_n_graphs

    if verbose:
        print("=" * 60)
        print("Section M: restricted permutation scores")
        print("=" * 60)
    restricted = aggregate_restricted_scores(
        model,
        smiles_list,
        device,
        n_graphs=restricted_n_graphs,
        top_k_frac=restricted_top_k_frac,
        num_perms=restricted_num_perms,
        tau=restricted_tau,
        verbose=verbose,
    )

    if verbose:
        print("\nComputing mass-capture curves across top-k fractions...")
    mass_capture = aggregate_mass_capture_by_frac(
        model,
        smiles_list,
        device,
        n_graphs=restricted_n_graphs,
        top_k_fracs=mass_top_k_fracs,
        verbose=verbose,
    )

    if verbose:
        print("\nComputing channel attribution...")
    channel = aggregate_channel_attribution(
        model,
        smiles_list,
        device,
        n_graphs=channel_n_graphs,
        top_k_frac=restricted_top_k_frac,
        num_perms=channel_num_perms,
        tau=channel_tau,
        verbose=verbose,
    )

    if verbose:
        print("\nComputing D-selection diagnostics...")
    d_select = aggregate_d_selection(
        model,
        smiles_list,
        device,
        n_graphs=d_select_n_graphs,
        top_k_frac=restricted_top_k_frac,
        n_shuf=d_select_n_shuf,
        verbose=verbose,
    )

    if verbose:
        print("\nComputing entropy delta diagnostics...")
    entropy_delta, entropy_used, entropy_L, entropy_H = aggregate_entropy_delta(
        model,
        smiles_list,
        device,
        n_graphs=entropy_n_graphs,
        verbose=verbose,
    )

    return {
        "restricted": restricted,
        "mass_capture": mass_capture,
        "channel": channel,
        "d_select": d_select,
        "entropy_delta": {
            "delta": entropy_delta,
            "n_used": entropy_used,
            "L": entropy_L,
            "H": entropy_H,
        },
    }
