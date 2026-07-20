#!/usr/bin/env python3
# # Graphormer specialisation methodology comparison
#
# This Colab-friendly script compares two per-head methodologies on the pretrained
# PCQM4Mv2 Graphormer:
#
# 1. **Current key-permutation score** from `graph_interp.metrics.scores`.
# 2. **Downstream transport score** with separate semantic and structural
#    interventions, adapted from the carriage/specialisation specification.
#
# The Graphormer transport site is the per-head context immediately before the
# attention output projection: `o[l,h,i] = sum_j A[l,h,i,j] V[l,h,j]`.
# Semantic interventions replace a complete atom-feature row with a real donor row
# from another PCQM molecule. Structural interventions transpose topology-derived
# payloads (degree, spatial position, direct-edge and multi-hop edge encodings) while
# holding atom content and the dense attention support fixed. The CLS token is included
# as a carrier because Graphormer reads its graph prediction from CLS.

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_URL = "https://github.com/joshgreenwa/Symbolic-and-Structural-Specialisation-in-Graphormers.git"
COLAB_REPO = Path("/content/repo")


def _running_in_colab() -> bool:
    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def _bootstrap_colab() -> None:
    """Clone the repository and install its pinned dependencies once per Colab runtime."""
    candidates = [Path.cwd(), COLAB_REPO]
    script_path = globals().get("__file__")
    if script_path:
        candidates.insert(0, Path(script_path).resolve().parents[1])

    repo_dir = next((path for path in candidates if (path / "graph_interp").is_dir()), None)
    if repo_dir is None:
        if not _running_in_colab():
            raise RuntimeError("Run this file from the repository root or inside Google Colab.")
        if not (COLAB_REPO / ".git").is_dir():
            subprocess.run(["git", "clone", REPO_URL, str(COLAB_REPO)], check=True)
        repo_dir = COLAB_REPO

    if _running_in_colab():
        # Preserve Colab's already-loaded NumPy/Torch ABI. Installing the repository's
        # full lock file here would replace NumPy in a live kernel and require a restart.
        marker = Path("/content/.graphormer_transport_comparison_deps_v3")
        if not marker.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "transformers==4.40.2",
                    "tokenizers==0.19.1",
                    "huggingface-hub==0.36.1",
                    "ogb==1.3.6",
                    "rdkit",
                ],
                check=True,
            )
            marker.touch()
            print("Graphormer dependencies installed; continuing without a runtime restart.", flush=True)
    sys.path.insert(0, str(repo_dir))


_bootstrap_colab()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GraphormerForGraphClassification

from graph_interp.config import MODEL_ID, MODEL_REVISION, apply_plot_defaults, set_seed
from graph_interp.data import build_batch_from_smiles, load_dataset, load_smiles_pool
from graph_interp.metrics.scores import aggregate_scores


# ## Configuration


@dataclass
class ComparisonConfig:
    num_graphs: int = 10
    donors: int = 4
    donor_pool_graphs: int = 128
    max_sources: int | None = 16
    legacy_num_perms: int = 96
    ablation_graphs: int = 10
    ablation_head_chunk: int = 16
    seed: int = 42
    verification_tol: float = 5e-4
    output_dir: str = "/content/graphormer_transport_comparison"
    device: str = "cuda"


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def load_full_pcqm_model(device: torch.device) -> GraphormerForGraphClassification:
    """Load the complete PCQM model, including the scalar graph-level readout."""
    model, loading_info = GraphormerForGraphClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_safetensors=True,
        output_loading_info=True,
    )
    if loading_info.get("mismatched_keys"):
        raise RuntimeError(f"Mismatched checkpoint keys: {loading_info['mismatched_keys']}")
    model = model.to(device).eval()
    return model


# ## Capture the per-head routed-value transport site


class HeadTransportCapture:
    """Capture or ablate the per-head input to every attention `out_proj`.

    Hugging Face Graphormer flattens `[tokens, batch, heads, head_dim]` immediately
    before `out_proj`. A forward-pre-hook therefore exposes the exact routed-value
    transport site and can zero individual heads before their downstream mixing.
    """

    def __init__(self, model: GraphormerForGraphClassification, ablations: dict[int, torch.Tensor] | None = None):
        self.model = model
        self.ablations = ablations or {}
        self.values: list[torch.Tensor | None] = [None] * len(model.encoder.graph_encoder.layers)
        self.handles: list[Any] = []

    def _hook(self, layer_idx: int):
        def hook(module, args):
            x = args[0]
            layer = self.model.encoder.graph_encoder.layers[layer_idx]
            heads = int(layer.self_attn.num_heads)
            tokens, batch_size, width = x.shape
            if width % heads:
                raise RuntimeError(f"Layer {layer_idx}: width {width} is not divisible by {heads} heads")

            head_for_replica = self.ablations.get(layer_idx)
            if head_for_replica is not None:
                if head_for_replica.numel() != batch_size:
                    raise ValueError(
                        f"Layer {layer_idx}: got {head_for_replica.numel()} ablation labels for batch {batch_size}"
                    )
                head_for_replica = head_for_replica.to(x.device)
                active = torch.nonzero(head_for_replica >= 0, as_tuple=False).flatten()
                if active.numel():
                    y = x.view(tokens, batch_size, heads, width // heads).clone()
                    y[:, active, head_for_replica[active], :] = 0.0
                    x = y.reshape_as(x)

            self.values[layer_idx] = x
            if x is not args[0]:
                return (x, *args[1:])
            return None

        return hook

    def __enter__(self):
        for layer_idx, layer in enumerate(self.model.encoder.graph_encoder.layers):
            self.handles.append(layer.self_attn.out_proj.register_forward_pre_hook(self._hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = (
        "input_nodes",
        "input_edges",
        "attn_bias",
        "in_degree",
        "out_degree",
        "spatial_pos",
        "attn_edge_type",
    )
    return {key: batch[key] for key in keys}


def forward_with_transport(
    model: GraphormerForGraphClassification,
    batch: dict[str, torch.Tensor],
    *,
    want_grad: bool,
    ablations: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    context = nullcontext() if want_grad else torch.no_grad()
    with HeadTransportCapture(model, ablations=ablations) as capture, context:
        output = model(**_model_inputs(batch), return_dict=True)
    if any(value is None for value in capture.values):
        raise RuntimeError("At least one Graphormer attention layer did not reach its output projection")
    return output.logits, [value for value in capture.values if value is not None]


def _transport_geometry(model: GraphormerForGraphClassification) -> tuple[int, int, int]:
    layers = model.encoder.graph_encoder.layers
    num_layers = len(layers)
    num_heads = int(layers[0].self_attn.num_heads)
    head_dim = int(layers[0].self_attn.head_dim)
    return num_layers, num_heads, head_dim


def clean_transport_gradients(
    model: GraphormerForGraphClassification,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return clean prediction and `d prediction_t / d o[l,h,i]` for every output."""
    model.zero_grad(set_to_none=True)
    logits, captures = forward_with_transport(model, batch, want_grad=True)
    prediction = logits[0].reshape(-1)
    num_outputs = int(prediction.numel())
    num_layers, num_heads, head_dim = _transport_geometry(model)

    grads_by_output: list[tuple[torch.Tensor, ...]] = []
    for output_idx in range(num_outputs):
        grads_by_output.append(
            torch.autograd.grad(
                prediction[output_idx],
                captures,
                retain_graph=output_idx < num_outputs - 1,
                allow_unused=False,
            )
        )

    phi: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
        layer_grads = []
        for output_idx in range(num_outputs):
            grad = grads_by_output[output_idx][layer_idx]
            tokens, batch_size, width = grad.shape
            if batch_size != 1 or width != num_heads * head_dim:
                raise RuntimeError(f"Unexpected clean transport gradient shape: {tuple(grad.shape)}")
            layer_grads.append(grad.view(tokens, 1, num_heads, head_dim)[:, 0])
        phi.append(torch.stack(layer_grads, dim=0).detach())
    return prediction.detach(), phi


def _captures_by_replica(
    captures: list[torch.Tensor],
    num_heads: int,
    head_dim: int,
) -> list[torch.Tensor]:
    result = []
    for value in captures:
        tokens, batch_size, width = value.shape
        if width != num_heads * head_dim:
            raise RuntimeError(f"Unexpected transport capture shape: {tuple(value.shape)}")
        result.append(value.view(tokens, batch_size, num_heads, head_dim).permute(1, 0, 2, 3))
    return result


def _functional_transport(phi: torch.Tensor, mean_delta: torch.Tensor) -> torch.Tensor:
    """`sum_carrier ||(phi_t dot mean_delta)_t||_2`, returned per head."""
    projected = (phi * mean_delta.unsqueeze(0)).sum(dim=-1)  # [outputs, carriers, heads]
    return torch.linalg.vector_norm(projected.float(), dim=0).sum(dim=0)


# ## Separate semantic and structural interventions


def repeat_batch(batch: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        reps = (repeats,) + (1,) * (value.ndim - 1)
        result[key] = value.repeat(reps)
    return result


def _node_permutation(n: int, u: int, v: int, device: torch.device) -> torch.Tensor:
    permutation = torch.arange(n, device=device)
    if u != v:
        permutation[u], permutation[v] = permutation[v].clone(), permutation[u].clone()
    return permutation


def transpose_structural_payload_(
    batch: dict[str, torch.Tensor],
    replica: int,
    u: int,
    v: int,
) -> None:
    """Mask-frozen Graphormer structural transposition.

    Atom content and `attn_bias` are unchanged. Graphormer's attention is dense, so
    freezing support is normally a no-op; retaining `attn_bias` also preserves that
    contract if a mask is added later.
    """
    n = int(batch["input_nodes"].shape[1])
    permutation = _node_permutation(n, u, v, batch["input_nodes"].device)

    for key in ("in_degree", "out_degree"):
        original = batch[key][replica].clone()
        batch[key][replica] = original.index_select(0, permutation)

    for key in ("spatial_pos", "attn_edge_type", "input_edges"):
        original = batch[key][replica].clone()
        batch[key][replica] = original.index_select(0, permutation).index_select(1, permutation)


def full_relabel_(batch: dict[str, torch.Tensor], replica: int, u: int, v: int) -> None:
    """Relabel content and every structural tensor; prediction should be invariant."""
    n = int(batch["input_nodes"].shape[1])
    permutation = _node_permutation(n, u, v, batch["input_nodes"].device)

    original_nodes = batch["input_nodes"][replica].clone()
    batch["input_nodes"][replica] = original_nodes.index_select(0, permutation)
    transpose_structural_payload_(batch, replica, u, v)

    token_permutation = torch.cat(
        [torch.zeros(1, device=permutation.device, dtype=permutation.dtype), permutation + 1]
    )
    original_bias = batch["attn_bias"][replica].clone()
    batch["attn_bias"][replica] = original_bias.index_select(0, token_permutation).index_select(
        1, token_permutation
    )


def sample_degree_matched_partners(
    degrees: np.ndarray,
    source: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.flatnonzero((degrees == degrees[source]) & (np.arange(len(degrees)) != source))
    if not len(candidates):
        others = np.flatnonzero(np.arange(len(degrees)) != source)
        if not len(others):
            return np.full(count, source, dtype=np.int64)
        gap = np.abs(degrees[others] - degrees[source])
        candidates = others[gap == gap.min()]
    return rng.choice(candidates, size=count, replace=True).astype(np.int64)


def build_donor_pool(
    smiles: list[str],
    encoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect on-manifold, complete Graphormer atom-feature rows on CPU."""
    rows = []
    graph_ids = []
    for graph_id, smi in enumerate(smiles):
        batch, n, _ = build_batch_from_smiles(smi, model=encoder, device=torch.device("cpu"))
        rows.append(batch["input_nodes"][0, :n].numpy().astype(np.int64))
        graph_ids.append(np.full(n, graph_id, dtype=np.int64))
    if not rows:
        raise RuntimeError("The donor pool is empty")
    return np.concatenate(rows, axis=0), np.concatenate(graph_ids, axis=0)


def _selected_sources(n: int, max_sources: int | None, rng: np.random.Generator) -> list[int]:
    if max_sources is None or max_sources >= n:
        return list(range(n))
    return sorted(rng.choice(n, size=max_sources, replace=False).astype(int).tolist())


def _verify_interventions(
    model: GraphormerForGraphClassification,
    base_batch: dict[str, torch.Tensor],
    tolerance: float,
) -> dict[str, float]:
    n = int(base_batch["input_nodes"].shape[1])
    num_layers, num_heads, head_dim = _transport_geometry(model)

    noop = repeat_batch(base_batch, 2)
    source = 0
    noop["input_nodes"][1, source] = noop["input_nodes"][0, source].clone()
    transpose_structural_payload_(noop, 1, source, source)
    _, captures = forward_with_transport(model, noop, want_grad=False)
    replica_caps = _captures_by_replica(captures, num_heads, head_dim)
    noop_max = max(float((layer[0] - layer[1]).abs().max().item()) for layer in replica_caps)

    relabel_delta = 0.0
    if n > 1:
        relabel = repeat_batch(base_batch, 2)
        full_relabel_(relabel, 1, 0, 1)
        logits, _ = forward_with_transport(model, relabel, want_grad=False)
        relabel_delta = float((logits[0] - logits[1]).abs().max().item())

    if noop_max > tolerance:
        raise RuntimeError(f"No-op transport moved by {noop_max:.3e}, exceeding {tolerance:.3e}")
    if relabel_delta > tolerance:
        raise RuntimeError(
            f"Full structure+content relabel changed prediction by {relabel_delta:.3e}, "
            f"exceeding {tolerance:.3e}"
        )
    return {"noop_transport_max": noop_max, "full_relabel_prediction_max": relabel_delta}


def score_transport_method(
    model: GraphormerForGraphClassification,
    smiles: list[str],
    graph_ids: list[int],
    donor_rows: np.ndarray,
    donor_graph_ids: np.ndarray,
    *,
    donors: int,
    max_sources: int | None,
    seed: int,
    verification_tol: float,
) -> dict[str, Any]:
    """Compute transport-based semantic and structural `[layer, head]` scores."""
    if donors < 1:
        raise ValueError("donors must be positive")
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    num_layers, num_heads, head_dim = _transport_geometry(model)
    semantic_sum = torch.zeros(num_layers, num_heads, dtype=torch.float64)
    structural_sum = torch.zeros_like(semantic_sum)
    total_semantic_sources = 0
    total_structural_sources = 0
    same_content_noop_max = 0.0
    checks: dict[str, float] = {}

    for graph_pos, (graph_id, smi) in enumerate(zip(graph_ids, smiles)):
        base, n, _ = build_batch_from_smiles(smi, model=model.encoder, device=device)
        if n < 2:
            continue
        if graph_pos == 0:
            checks = _verify_interventions(model, base, verification_tol)

        _, phi = clean_transport_gradients(model, base)
        sources = _selected_sources(n, max_sources, rng)
        candidate_donors = np.flatnonzero(donor_graph_ids != graph_id)
        if not len(candidate_donors):
            raise RuntimeError("No inter-graph donor rows are available")

        # Semantic intervention: replace complete atom-feature rows; structure stays fixed.
        for source in sources:
            donor_ids = rng.choice(candidate_donors, size=donors, replace=True)
            chosen_rows = donor_rows[donor_ids]
            intervention_batch = repeat_batch(base, donors + 1)
            intervention_batch["input_nodes"][1:, source, :] = torch.as_tensor(
                chosen_rows, device=device, dtype=intervention_batch["input_nodes"].dtype
            )
            _, captures = forward_with_transport(model, intervention_batch, want_grad=False)
            replica_caps = _captures_by_replica(captures, num_heads, head_dim)

            own = base["input_nodes"][0, source].detach().cpu().numpy()
            identical = np.all(chosen_rows == own[None, :], axis=1)
            for layer_idx, layer_caps in enumerate(replica_caps):
                mean_delta = layer_caps[0] - layer_caps[1:].mean(dim=0)
                semantic_sum[layer_idx] += _functional_transport(phi[layer_idx], mean_delta).double().cpu()
                if np.any(identical):
                    ids = torch.as_tensor(np.flatnonzero(identical) + 1, device=layer_caps.device)
                    same_content_noop_max = max(
                        same_content_noop_max,
                        float((layer_caps[0:1] - layer_caps.index_select(0, ids)).abs().max().item()),
                    )
            total_semantic_sources += 1

        # Structural intervention: transpose topology payload, degree-matching the partner.
        degrees = base["in_degree"][0, :n].detach().cpu().numpy()
        for source in sources:
            partners = sample_degree_matched_partners(degrees, source, donors, rng)
            intervention_batch = repeat_batch(base, donors + 1)
            for replica, partner in enumerate(partners, start=1):
                transpose_structural_payload_(intervention_batch, replica, source, int(partner))
            _, captures = forward_with_transport(model, intervention_batch, want_grad=False)
            replica_caps = _captures_by_replica(captures, num_heads, head_dim)
            for layer_idx, layer_caps in enumerate(replica_caps):
                mean_delta = layer_caps[0] - layer_caps[1:].mean(dim=0)
                structural_sum[layer_idx] += _functional_transport(phi[layer_idx], mean_delta).double().cpu()
            total_structural_sources += 1

        del phi
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[transport] graph {graph_pos + 1}/{len(smiles)}: n={n}, sources={len(sources)}",
            flush=True,
        )

    if not total_semantic_sources or not total_structural_sources:
        raise RuntimeError("No graph sources were scored")
    if same_content_noop_max > verification_tol:
        raise RuntimeError(
            f"Same-content semantic donor moved transport by {same_content_noop_max:.3e}, "
            f"exceeding {verification_tol:.3e}"
        )
    checks["same_content_transport_max"] = same_content_noop_max

    return {
        "semantic": (semantic_sum / total_semantic_sources).numpy(),
        "structural": (structural_sum / total_structural_sources).numpy(),
        "num_semantic_sources": total_semantic_sources,
        "num_structural_sources": total_structural_sources,
        "checks": checks,
    }


# ## Exact pre-output-projection head ablation


def compute_ablation_impact(
    model: GraphormerForGraphClassification,
    smiles: list[str],
    *,
    head_chunk: int,
) -> dict[str, np.ndarray]:
    """Mean label-free impact `||prediction_ablated - prediction_clean||_2` per head."""
    if head_chunk < 1:
        raise ValueError("head_chunk must be positive")
    device = next(model.parameters()).device
    num_layers, num_heads, _ = _transport_geometry(model)
    impact_per_graph = np.zeros((num_layers, num_heads, len(smiles)), dtype=np.float64)

    for graph_idx, smi in enumerate(smiles):
        base, _, _ = build_batch_from_smiles(smi, model=model.encoder, device=device)
        for layer_idx in range(num_layers):
            for start in range(0, num_heads, head_chunk):
                heads = torch.arange(start, min(start + head_chunk, num_heads), dtype=torch.long, device=device)
                batch = repeat_batch(base, int(heads.numel()) + 1)
                labels = torch.cat([torch.full((1,), -1, device=device, dtype=torch.long), heads])
                logits, _ = forward_with_transport(
                    model,
                    batch,
                    want_grad=False,
                    ablations={layer_idx: labels},
                )
                delta = torch.linalg.vector_norm((logits[1:] - logits[0]).float(), dim=-1)
                impact_per_graph[layer_idx, start : start + heads.numel(), graph_idx] = delta.cpu().numpy()
        print(f"[ablation] graph {graph_idx + 1}/{len(smiles)}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"mean": impact_per_graph.mean(axis=2), "per_graph": impact_per_graph}


# ## Alignment, score-impact tests, and figures


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return float("nan")
    return float(np.corrcoef(_rankdata(x[finite]), _rankdata(y[finite]))[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling, equivalent to SciPy's default `rankdata`."""
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(controls, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    finite = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    if finite.sum() < z.shape[1] + 3:
        return float("nan")
    xr = _rankdata(x[finite])
    yr = _rankdata(y[finite])
    zr = np.column_stack([np.ones(finite.sum())] + [_rankdata(z[finite, j]) for j in range(z.shape[1])])
    x_resid = xr - zr @ np.linalg.lstsq(zr, xr, rcond=None)[0]
    y_resid = yr - zr @ np.linalg.lstsq(zr, yr, rcond=None)[0]
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return float("nan")
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def _normalised_channel(score: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(score))
    return score / max(abs(mean), 1e-12)


def _rank_fraction(score: np.ndarray) -> np.ndarray:
    flat = np.asarray(score, dtype=float).reshape(-1)
    return _rankdata(flat) / len(flat)


def score_impact_correlations(
    semantic: np.ndarray,
    structural: np.ndarray,
    impact: np.ndarray,
) -> dict[str, float]:
    sem = semantic.reshape(-1)
    struct = structural.reshape(-1)
    outcome = impact.reshape(-1)
    layers, heads = semantic.shape
    depth = np.repeat(np.arange(layers, dtype=float), heads)
    return {
        "semantic_raw": _safe_spearman(sem, outcome),
        "structural_raw": _safe_spearman(struct, outcome),
        "semantic_controlling_structural": _partial_spearman(sem, outcome, struct),
        "structural_controlling_semantic": _partial_spearman(struct, outcome, sem),
        "semantic_controlling_depth": _partial_spearman(sem, outcome, depth),
        "structural_controlling_depth": _partial_spearman(struct, outcome, depth),
    }


def plot_method_scatter(
    semantic: np.ndarray,
    structural: np.ndarray,
    method_name: str,
    output_path: Path,
) -> None:
    sem = _normalised_channel(semantic).reshape(-1)
    struct = _normalised_channel(structural).reshape(-1)
    layers, heads = semantic.shape
    depth = np.repeat(np.arange(layers), heads)
    limit = max(float(np.nanmax(sem)), float(np.nanmax(struct)), 1.05)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    scatter = ax.scatter(struct, sem, c=depth, cmap="viridis", s=30, alpha=0.82, edgecolors="none")
    ax.plot([0, limit], [0, limit], "--", color="0.35", linewidth=1.2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Structural score / channel mean")
    ax.set_ylabel("Semantic score / channel mean")
    ax.set_title(f"{method_name}: per-head structural vs semantic specialisation")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Layer")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_method_alignment(
    legacy_semantic: np.ndarray,
    legacy_structural: np.ndarray,
    transport_semantic: np.ndarray,
    transport_structural: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    layers, heads = legacy_semantic.shape
    depth = np.repeat(np.arange(layers), heads)
    correlations = {
        "semantic": _safe_spearman(legacy_semantic, transport_semantic),
        "structural": _safe_spearman(legacy_structural, transport_structural),
    }
    legacy_lean = _normalised_channel(legacy_semantic) - _normalised_channel(legacy_structural)
    transport_lean = _normalised_channel(transport_semantic) - _normalised_channel(transport_structural)
    correlations["semantic_minus_structural"] = _safe_spearman(legacy_lean, transport_lean)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    comparisons = [
        (legacy_structural, transport_structural, "Structural", correlations["structural"]),
        (legacy_semantic, transport_semantic, "Semantic", correlations["semantic"]),
    ]
    for ax, (old, new, label, rho) in zip(axes[0], comparisons):
        sc = ax.scatter(old.reshape(-1), new.reshape(-1), c=depth, cmap="viridis", s=26, alpha=0.8)
        ax.set_xlabel(f"Current {label.lower()} score")
        ax.set_ylabel(f"Transport {label.lower()} score")
        ax.set_title(f"{label} alignment: Spearman rho={rho:.3f}")
        ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label="Layer")

    head_index = np.arange(layers * heads)
    for ax, old, new, label in (
        (axes[1, 0], legacy_structural, transport_structural, "Structural"),
        (axes[1, 1], legacy_semantic, transport_semantic, "Semantic"),
    ):
        ax.plot(head_index, _rank_fraction(old), linewidth=1.1, label="Current method")
        ax.plot(head_index, _rank_fraction(new), linewidth=1.1, label="Transport method")
        for boundary in range(heads, layers * heads, heads):
            ax.axvline(boundary - 0.5, color="0.85", linewidth=0.5)
        ax.set_xlabel("Flattened head index (layer-major)")
        ax.set_ylabel("Within-method percentile rank")
        ax.set_title(f"Per-head {label.lower()} rank profile")
        ax.legend()
        ax.grid(alpha=0.16)

    fig.suptitle(
        "Per-head agreement between key-permutation and downstream-transport scores\n"
        f"leaning alignment rho={correlations['semantic_minus_structural']:.3f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return correlations


def plot_score_vs_ablation(
    methods: dict[str, tuple[np.ndarray, np.ndarray]],
    impact: np.ndarray,
    correlations: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    layers, heads = impact.shape
    depth = np.repeat(np.arange(layers), heads)
    outcome = impact.reshape(-1)
    fig, axes = plt.subplots(len(methods), 3, figsize=(17, 5.2 * len(methods)), squeeze=False)

    bar_keys = [
        "semantic_raw",
        "semantic_controlling_structural",
        "semantic_controlling_depth",
        "structural_raw",
        "structural_controlling_semantic",
        "structural_controlling_depth",
    ]
    bar_labels = ["Sem raw", "Sem | str", "Sem | depth", "Str raw", "Str | sem", "Str | depth"]

    for row, (method_name, (semantic, structural)) in enumerate(methods.items()):
        for col, (score, channel) in enumerate(((semantic, "Semantic"), (structural, "Structural"))):
            rho = correlations[method_name][f"{channel.lower()}_raw"]
            sc = axes[row, col].scatter(
                score.reshape(-1), outcome, c=depth, cmap="viridis", s=25, alpha=0.8
            )
            axes[row, col].set_xlabel(f"{method_name} {channel.lower()} score")
            axes[row, col].set_ylabel("Mean functional head-ablation impact")
            axes[row, col].set_title(f"{channel}: Spearman rho={rho:.3f}")
            axes[row, col].grid(alpha=0.2)
            fig.colorbar(sc, ax=axes[row, col], label="Layer")

        values = [correlations[method_name][key] for key in bar_keys]
        colors = ["#d95f02"] * 3 + ["#1b9e77"] * 3
        axes[row, 2].bar(np.arange(len(values)), values, color=colors, alpha=0.85)
        axes[row, 2].axhline(0, color="0.25", linewidth=0.8)
        axes[row, 2].set_xticks(np.arange(len(values)))
        axes[row, 2].set_xticklabels(bar_labels, rotation=35, ha="right")
        axes[row, 2].set_ylim(-1, 1)
        axes[row, 2].set_ylabel("Spearman / partial-rank correlation")
        axes[row, 2].set_title(f"{method_name}: raw and controlled correlations")
        axes[row, 2].grid(axis="y", alpha=0.2)

    fig.suptitle("Do per-head specialisation scores predict causal ablation impact?", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ## End-to-end runner


def run(config: ComparisonConfig | None = None) -> dict[str, Any]:
    config = config or ComparisonConfig()
    if config.num_graphs < 1 or config.ablation_graphs < 1 or config.donor_pool_graphs < 2:
        raise ValueError("num_graphs/ablation_graphs must be positive and donor_pool_graphs must be at least 2")

    apply_plot_defaults()
    set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}@{MODEL_REVISION} on {device}...", flush=True)
    model = load_full_pcqm_model(device)
    pool_size = max(config.num_graphs, config.ablation_graphs, config.donor_pool_graphs)
    dataset = load_dataset()
    smiles_pool = load_smiles_pool(dataset, pool_size=pool_size)
    analysis_smiles = smiles_pool[: config.num_graphs]
    analysis_graph_ids = list(range(len(analysis_smiles)))

    print("Computing current key-permutation scores...", flush=True)
    legacy_structural, legacy_semantic, _, legacy_std_structural, legacy_std_semantic = aggregate_scores(
        model.encoder,
        analysis_smiles,
        device,
        n_graphs=len(analysis_smiles),
        num_perms=config.legacy_num_perms,
        seed=config.seed,
        return_std=True,
    )

    print(f"Building donor pool from {config.donor_pool_graphs} PCQM molecules...", flush=True)
    donor_rows, donor_graph_ids = build_donor_pool(
        smiles_pool[: config.donor_pool_graphs],
        model.encoder,
    )
    print(f"Donor pool contains {len(donor_rows)} atom-feature rows.", flush=True)

    transport = score_transport_method(
        model,
        analysis_smiles,
        analysis_graph_ids,
        donor_rows,
        donor_graph_ids,
        donors=config.donors,
        max_sources=config.max_sources,
        seed=config.seed,
        verification_tol=config.verification_tol,
    )
    transport_semantic = transport["semantic"]
    transport_structural = transport["structural"]

    ablation_smiles = smiles_pool[: config.ablation_graphs]
    ablation = compute_ablation_impact(
        model,
        ablation_smiles,
        head_chunk=config.ablation_head_chunk,
    )
    impact = ablation["mean"]

    plot_method_scatter(
        legacy_semantic,
        legacy_structural,
        "Current key-permutation method",
        output_dir / "scatter_current_method.png",
    )
    plot_method_scatter(
        transport_semantic,
        transport_structural,
        "Separate-intervention transport method",
        output_dir / "scatter_transport_method.png",
    )
    alignment = plot_method_alignment(
        legacy_semantic,
        legacy_structural,
        transport_semantic,
        transport_structural,
        output_dir / "per_head_method_alignment.png",
    )

    methods = {
        "Current method": (legacy_semantic, legacy_structural),
        "Transport method": (transport_semantic, transport_structural),
    }
    impact_correlations = {
        method_name: score_impact_correlations(semantic, structural, impact)
        for method_name, (semantic, structural) in methods.items()
    }
    plot_score_vs_ablation(
        methods,
        impact,
        impact_correlations,
        output_dir / "score_vs_ablation_impact.png",
    )

    np.savez_compressed(
        output_dir / "graphormer_specialisation_comparison.npz",
        legacy_semantic=legacy_semantic,
        legacy_structural=legacy_structural,
        legacy_std_semantic=legacy_std_semantic,
        legacy_std_structural=legacy_std_structural,
        transport_semantic=transport_semantic,
        transport_structural=transport_structural,
        ablation_impact_mean=impact,
        ablation_impact_per_graph=ablation["per_graph"],
    )
    summary = {
        "config": asdict(config),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transport_adaptation": {
            "transport_site": "per-head A@V immediately before attention out_proj",
            "carriers": "CLS plus molecular nodes",
            "semantic_intervention": "complete atom-feature donor row; structural tensors fixed",
            "structural_intervention": (
                "degree/spatial/direct-edge/multi-hop payload transposition; atom content and attention support fixed"
            ),
            "nuisance_average": "signed transport delta averaged over donors/partners before magnitude",
            "ablation": "zero the selected head at the same pre-out_proj transport site",
        },
        "transport_checks": transport["checks"],
        "alignment_spearman": alignment,
        "score_vs_ablation": impact_correlations,
        "outputs": {
            "current_scatter": "scatter_current_method.png",
            "transport_scatter": "scatter_transport_method.png",
            "method_alignment": "per_head_method_alignment.png",
            "score_vs_ablation": "score_vs_ablation_impact.png",
            "arrays": "graphormer_specialisation_comparison.npz",
        },
    }
    with (output_dir / "graphormer_specialisation_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)

    print("\nComparison complete. Outputs:", flush=True)
    for path in sorted(output_dir.iterdir()):
        print(f"  {path}", flush=True)
    print("\nPer-head method alignment:", json.dumps(_json_ready(alignment), indent=2), flush=True)
    print("Score-vs-ablation correlations:", json.dumps(_json_ready(impact_correlations), indent=2), flush=True)
    return {
        "legacy_semantic": legacy_semantic,
        "legacy_structural": legacy_structural,
        "transport_semantic": transport_semantic,
        "transport_structural": transport_structural,
        "ablation": ablation,
        "summary": summary,
        "output_dir": output_dir,
    }


def parse_args(argv: list[str] | None = None) -> ComparisonConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-graphs", type=int, default=10)
    parser.add_argument("--donors", type=int, default=4)
    parser.add_argument("--donor-pool-graphs", type=int, default=128)
    parser.add_argument("--max-sources", type=int, default=16, help="Use 0 to score every node")
    parser.add_argument("--legacy-num-perms", type=int, default=96)
    parser.add_argument("--ablation-graphs", type=int, default=10)
    parser.add_argument("--ablation-head-chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verification-tol", type=float, default=5e-4)
    parser.add_argument("--output-dir", default="/content/graphormer_transport_comparison")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    if argv is None:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)
    return ComparisonConfig(
        num_graphs=args.num_graphs,
        donors=args.donors,
        donor_pool_graphs=args.donor_pool_graphs,
        max_sources=None if args.max_sources == 0 else args.max_sources,
        legacy_num_perms=args.legacy_num_perms,
        ablation_graphs=args.ablation_graphs,
        ablation_head_chunk=args.ablation_head_chunk,
        seed=args.seed,
        verification_tol=args.verification_tol,
        output_dir=args.output_dir,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI/Colab entry point accepting command-line-style arguments."""
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
