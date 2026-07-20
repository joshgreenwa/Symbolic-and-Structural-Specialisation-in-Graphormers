"""Model loading and attention / hidden-state extraction (with optional ablation)."""

import math
from pathlib import Path
import numpy as np
import torch
from transformers import GraphormerForGraphClassification

from .config import MODEL_ID, MODEL_REVISION


# ── Model loading ────────────────────────────────────────────

def load_model(device: torch.device | str = "cuda"):
    """Load the pre-trained Graphormer encoder and return (model, device)."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    clf, _ = GraphormerForGraphClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_safetensors=True, output_loading_info=True,
    )
    model = clf.encoder.to(device).eval()
    return model, device


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    ckpt_dir: str | Path | None = None,
) -> Path:
    """Resolve a MolHIV Fairseq checkpoint path from a file, run dir, or ckpt dir."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return path

    if ckpt_dir is None:
        if run_dir is None:
            candidates = [
                Path("/content/drive/MyDrive/graphormer_molhiv_runs")
                / "public_pcqm_molhiv_pcqm4mv2_graphormer_base_seed1",
                Path("/content/graphormer_molhiv_runs")
                / "public_pcqm_molhiv_pcqm4mv2_graphormer_base_seed1",
                Path("/content/drive/MyDrive/graphormer_molhiv_runs")
                / "public_pcqm_molhiv_pcqm4mv1_graphormer_base_seed1",
                Path("/content/graphormer_molhiv_runs")
                / "public_pcqm_molhiv_pcqm4mv1_graphormer_base_seed1",
                Path("/content/drive/MyDrive/graphormer_molhiv_runs")
                / "official_molhiv_pcqm4mv1_graphormer_base_for_molhiv_seed1",
                Path("/content/graphormer_molhiv_runs")
                / "official_molhiv_pcqm4mv1_graphormer_base_for_molhiv_seed1",
                Path("/content/drive/MyDrive/graphormer_molhiv_runs")
                / "official_molhiv_pcqm4mv1_graphormer_base_seed1",
                Path("/content/graphormer_molhiv_runs")
                / "official_molhiv_pcqm4mv1_graphormer_base_seed1",
            ]
            for candidate in candidates:
                if candidate.exists():
                    run_dir = candidate
                    break
        if run_dir is None:
            raise FileNotFoundError(
                "Provide checkpoint_path, run_dir, or ckpt_dir for the MolHIV checkpoint."
            )
        ckpt_dir = Path(run_dir).expanduser() / "ckpts"
    else:
        ckpt_dir = Path(ckpt_dir).expanduser()

    preferred = [
        ckpt_dir / "checkpoint_best_valid_auc.pt",
        ckpt_dir / "checkpoint_best.pt",
        ckpt_dir / "checkpoint_last.pt",
    ]
    for path in preferred:
        if path.is_file():
            return path

    checkpoints = sorted(
        (
            path
            for path in ckpt_dir.glob("checkpoint*.pt")
            if "pretrained_step0" not in path.name
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if checkpoints:
        return checkpoints[0]
    raise FileNotFoundError(f"No checkpoint*.pt files found in {ckpt_dir}")


def _fairseq_encoder_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Convert official Fairseq Graphormer checkpoint keys to HF encoder keys."""
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a checkpoint dict or model state dict, got {type(state)}")

    encoder_state = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            encoder_state[key[len("encoder.") :]] = value
    if not encoder_state:
        raise RuntimeError("No 'encoder.' keys found in checkpoint model state.")
    return encoder_state


def _lookup_nested(obj, path: tuple[str, ...]):
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
        if cur is None:
            return None
    return cur


def _infer_pre_layernorm(checkpoint: dict, checkpoint_path: Path) -> bool:
    for path in (
        ("cfg", "model", "pre_layernorm"),
        ("args", "pre_layernorm"),
        ("extra_state", "pre_layernorm"),
    ):
        value = _lookup_nested(checkpoint, path)
        if value is not None:
            return bool(value)
    return "base_for_molhiv" in str(checkpoint_path)


def _set_pre_layernorm(model, enabled: bool) -> None:
    if hasattr(model, "config"):
        setattr(model.config, "pre_layernorm", enabled)
    graph_encoder = getattr(model, "graph_encoder", None)
    if graph_encoder is not None:
        for layer in getattr(graph_encoder, "layers", []):
            if hasattr(layer, "pre_layernorm"):
                layer.pre_layernorm = enabled


def load_molhiv_model(
    checkpoint_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    ckpt_dir: str | Path | None = None,
    device: torch.device | str = "cuda",
    base_model_id: str = MODEL_ID,
    base_revision: str = MODEL_REVISION,
):
    """Load a MolHIV fine-tuned Fairseq checkpoint into the HF Graphormer encoder.

    Returns:
        (model, device, checkpoint_path)

    The existing analysis code operates on the Hugging Face encoder object. The
    MolHIV training run saves official Microsoft/Fairseq checkpoints, whose keys
    are prefixed with ``encoder.``. This function instantiates the same HF
    Graphormer encoder used by the PCQM analysis, strips that prefix, and loads
    all shape-compatible encoder weights from the MolHIV checkpoint.
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    checkpoint_path = _resolve_checkpoint_path(checkpoint_path, run_dir, ckpt_dir)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    encoder_state = _fairseq_encoder_state(checkpoint)
    pre_layernorm = _infer_pre_layernorm(checkpoint, checkpoint_path)

    clf, _ = GraphormerForGraphClassification.from_pretrained(
        base_model_id,
        revision=base_revision,
        use_safetensors=True,
        output_loading_info=True,
    )
    model = clf.encoder
    _set_pre_layernorm(model, pre_layernorm)

    target_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in encoder_state.items():
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    critical_missing = [
        key for key in missing if key.startswith("graph_encoder.") and "embed_out" not in key
    ]
    if critical_missing:
        preview = ", ".join(critical_missing[:8])
        raise RuntimeError(f"Missing critical graph encoder keys after load: {preview}")
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise RuntimeError(f"Unexpected keys while loading MolHIV checkpoint: {preview}")
    if skipped:
        print(f"Skipped {len(skipped)} shape-mismatched/non-encoder keys; first few: {skipped[:5]}")

    model = model.to(device).eval()
    return model, device, checkpoint_path


# ── Internal: shared forward-pass preamble ───────────────────

def _forward_preamble(model, batch_device):
    ge = model.graph_encoder
    input_nodes = batch_device["input_nodes"]
    input_edges = batch_device["input_edges"]
    attn_bias = batch_device["attn_bias"]
    in_degree = batch_device["in_degree"]
    out_degree = batch_device["out_degree"]
    spatial_pos = batch_device["spatial_pos"]
    attn_edge_type = batch_device["attn_edge_type"]

    padding_mask = (input_nodes[:, :, 0] == 0)
    padding_mask_cls = torch.zeros(
        padding_mask.size(0), 1, device=padding_mask.device, dtype=padding_mask.dtype
    )
    padding_mask = torch.cat([padding_mask_cls, padding_mask], dim=1)

    graph_attn_bias = ge.graph_attn_bias(
        input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type
    )
    x = ge.graph_node_feature(input_nodes, in_degree, out_degree)
    if ge.embed_scale is not None:
        x = x * ge.embed_scale
    if ge.emb_layer_norm is not None:
        x = ge.emb_layer_norm(x)
    x = ge.dropout_module(x)
    x = x.transpose(0, 1)  # [T, B, D]
    return ge, x, padding_mask, graph_attn_bias


def _layer_attention(layer, y, graph_attn_bias, padding_mask):
    """Compute dot, bias, A for a single layer (no side effects on x)."""
    sa = layer.self_attn
    T, B, D = y.shape
    H = sa.num_heads
    Hd = D // H

    q = sa.q_proj(y) * getattr(sa, "scaling", 1.0 / math.sqrt(Hd))
    k = sa.k_proj(y)

    qh = q[:, 0].view(T, H, Hd).permute(1, 0, 2).contiguous()
    kh = k[:, 0].view(T, H, Hd).permute(1, 0, 2).contiguous()
    dot = torch.matmul(qh, kh.transpose(-1, -2))

    gab = graph_attn_bias
    if gab.dim() == 4:
        bias = gab[0]
    elif gab.dim() == 3:
        bias = gab[0].unsqueeze(0).repeat(H, 1, 1)
    else:
        raise RuntimeError(f"Unexpected graph_attn_bias shape {tuple(gab.shape)}")

    logits = (dot + bias).float()
    key_mask = padding_mask[0].bool().view(1, 1, T).expand(H, T, T)
    logits = logits.masked_fill(key_mask, float("-inf"))
    A = torch.softmax(logits, dim=-1).to(dot.dtype)

    return dot, bias, A, H, Hd


def _layer_ffn(layer, y_out):
    """Run the FFN sub-layer (post-attention residual)."""
    residual = y_out
    y2 = layer.final_layer_norm(y_out) if layer.pre_layernorm else y_out
    y2 = layer.activation_fn(layer.fc1(y2))
    y2 = layer.activation_dropout_module(y2)
    y2 = layer.fc2(y2)
    y2 = layer.dropout_module(y2)
    x = residual + y2
    if not layer.pre_layernorm:
        x = layer.final_layer_norm(x)
    return x


# ── Public: extract attention (dot, bias, A) ─────────────────

@torch.no_grad()
def extract_attention(model, batch_device: dict):
    """Extract attention components for all layers.

    Returns:
        dot_list:  list[L] of [H, T, T] scaled dot-product logits
        bias_list: list[L] of [H, T, T] graph attention bias
        A_list:    list[L] of [H, T, T] softmax(dot+bias)
    """
    ge, x, padding_mask, graph_attn_bias = _forward_preamble(model, batch_device)

    dot_list, bias_list, A_list = [], [], []

    for layer in ge.layers:
        residual = x
        y = layer.self_attn_layer_norm(x) if layer.pre_layernorm else x

        dot, bias, A, H, Hd = _layer_attention(layer, y, graph_attn_bias, padding_mask)
        dot_list.append(dot.detach())
        bias_list.append(bias.detach())
        A_list.append(A.detach())

        # advance x through the layer normally
        y_out, _ = layer.self_attn(
            query=y, key=y, value=y,
            attn_bias=graph_attn_bias, key_padding_mask=padding_mask,
            need_weights=False, need_head_weights=False, attn_mask=None,
        )
        y_out = layer.dropout_module(y_out)
        y_out = residual + y_out
        if not layer.pre_layernorm:
            y_out = layer.self_attn_layer_norm(y_out)
        x = _layer_ffn(layer, y_out)

    return dot_list, bias_list, A_list


# ── Public: extract attention with ablation ───────────────────

@torch.no_grad()
def extract_attention_with_ablation(
    model, batch_device: dict,
    ablate_heads: dict | None = None,
    ablation_mode: str = "zero",
):
    """Forward pass that optionally ablates specific heads.

    Args:
        ablate_heads: {layer_idx: [head_indices]} to ablate.
        ablation_mode: "zero" or "mean".

    Returns:
        dot_list, bias_list, A_list (same shapes as extract_attention)
    """
    if ablate_heads is None:
        ablate_heads = {}

    ge, x, padding_mask, graph_attn_bias = _forward_preamble(model, batch_device)
    dot_list, bias_list, A_list = [], [], []

    for layer_idx, layer in enumerate(ge.layers):
        residual = x
        y = layer.self_attn_layer_norm(x) if layer.pre_layernorm else x

        sa = layer.self_attn
        T, B, D = y.shape
        H = sa.num_heads
        Hd = D // H

        dot, bias, A, _, _ = _layer_attention(layer, y, graph_attn_bias, padding_mask)
        dot_list.append(dot.detach())
        bias_list.append(bias.detach())
        A_list.append(A.detach())

        # compute context with optional ablation
        v = sa.v_proj(y)
        vh = v[:, 0].view(T, H, Hd).permute(1, 0, 2).contiguous()
        context = torch.matmul(A, vh)

        if layer_idx in ablate_heads:
            for h in ablate_heads[layer_idx]:
                if ablation_mode == "zero":
                    context[h] = 0.0
                elif ablation_mode == "mean":
                    context[h] = context[h].mean(dim=0, keepdim=True).expand_as(context[h])

        context = context.permute(1, 0, 2).contiguous().view(T, 1, D)
        attn_out = sa.out_proj(context)
        attn_out = layer.dropout_module(attn_out)
        y_out = residual + attn_out
        if not layer.pre_layernorm:
            y_out = layer.self_attn_layer_norm(y_out)
        x = _layer_ffn(layer, y_out)

    return dot_list, bias_list, A_list


# ── Public: extract hidden states (with optional ablation) ────

@torch.no_grad()
def extract_hidden_states(
    model, batch_device: dict,
    ablate_heads: dict | None = None,
    ablation_mode: str = "zero",
    return_sublayers: bool = False,
):
    """Forward pass returning per-layer hidden states.

    Args:
        return_sublayers: if True, also return states after attention (before FFN).

    Returns:
        A_list:      list[L] of [H, T, T]
        hidden_list: list[L] of [T, D] hidden states after each full layer
        sublayer_list: (only if return_sublayers) list[L] of [T, D] after attention
        final_out:   [T, D]
    """
    if ablate_heads is None:
        ablate_heads = {}

    ge, x, padding_mask, graph_attn_bias = _forward_preamble(model, batch_device)

    A_list, hidden_list, sublayer_list = [], [], []

    for layer_idx, layer in enumerate(ge.layers):
        residual = x
        y = layer.self_attn_layer_norm(x) if layer.pre_layernorm else x

        sa = layer.self_attn
        T, B, D = y.shape
        H = sa.num_heads
        Hd = D // H

        _, _, A, _, _ = _layer_attention(layer, y, graph_attn_bias, padding_mask)
        A_list.append(A.detach())

        v = sa.v_proj(y)
        vh = v[:, 0].view(T, H, Hd).permute(1, 0, 2).contiguous()
        context = torch.matmul(A, vh)

        if layer_idx in ablate_heads:
            for h in ablate_heads[layer_idx]:
                if ablation_mode == "zero":
                    context[h] = 0.0
                elif ablation_mode == "mean":
                    context[h] = context[h].mean(dim=0, keepdim=True).expand_as(context[h])

        context = context.permute(1, 0, 2).contiguous().view(T, 1, D)
        attn_out = sa.out_proj(context)
        attn_out = layer.dropout_module(attn_out)
        y_out = residual + attn_out
        if not layer.pre_layernorm:
            y_out = layer.self_attn_layer_norm(y_out)

        if return_sublayers:
            sublayer_list.append(y_out[:, 0, :].detach().clone())

        x = _layer_ffn(layer, y_out)
        hidden_list.append(x[:, 0, :].detach().clone())

    final_out = x[:, 0, :].detach()
    if return_sublayers:
        return A_list, hidden_list, sublayer_list, final_out
    return A_list, hidden_list, final_out


# ── Public: extract attention with value outputs (A @ V) ──────

@torch.no_grad()
def extract_with_values(model, batch_device: dict):
    """Extract attention components plus per-head value outputs.

    Returns:
        dot_list, bias_list, A_list: as usual
        AV_list: list[L] of [H, T, Hd] — per-head context vectors (A @ V)
    """
    ge, x, padding_mask, graph_attn_bias = _forward_preamble(model, batch_device)

    dot_list, bias_list, A_list, AV_list = [], [], [], []

    for layer in ge.layers:
        residual = x
        y = layer.self_attn_layer_norm(x) if layer.pre_layernorm else x

        sa = layer.self_attn
        T, B, D = y.shape
        H = sa.num_heads
        Hd = D // H

        dot, bias, A, _, _ = _layer_attention(layer, y, graph_attn_bias, padding_mask)
        dot_list.append(dot.detach())
        bias_list.append(bias.detach())
        A_list.append(A.detach())

        v = sa.v_proj(y)
        vh = v[:, 0].view(T, H, Hd).permute(1, 0, 2).contiguous()
        context = torch.matmul(A, vh)  # [H, T, Hd]
        AV_list.append(context.detach())

        # advance x
        context_flat = context.permute(1, 0, 2).contiguous().view(T, 1, D)
        attn_out = sa.out_proj(context_flat)
        attn_out = layer.dropout_module(attn_out)
        y_out = residual + attn_out
        if not layer.pre_layernorm:
            y_out = layer.self_attn_layer_norm(y_out)
        x = _layer_ffn(layer, y_out)

    return dot_list, bias_list, A_list, AV_list


# ── Public: extract spatial bias profiles from model weights ──

def extract_spatial_bias_profiles(model):
    """Extract learned spatial (hop-distance) bias vectors from model weights.

    Returns:
        bias_profiles: [H, D_bins] numpy array
        dist_labels: list of str labels for each distance bin
    """
    ge = model.graph_encoder
    spatial_enc = None

    for attr_name in ["spatial_pos_encoder", "spatial_encoder"]:
        if hasattr(ge, attr_name):
            spatial_enc = getattr(ge, attr_name)
            break

    if spatial_enc is None:
        for name, module in ge.named_modules():
            if "spatial" in name.lower() and hasattr(module, "weight"):
                spatial_enc = module
                break

    if spatial_enc is None:
        raise RuntimeError("Could not find spatial position encoder in model.")

    W = spatial_enc.weight.detach().cpu().numpy()  # [D_bins, H]
    D_bins, H = W.shape
    bias_profiles = W.T  # [H, D_bins]

    spatial_pos_max = int(getattr(model.config, "spatial_pos_max", 20))
    dist_labels = []
    for d in range(D_bins):
        if d == 0:
            dist_labels.append("pad/0")
        elif d <= spatial_pos_max:
            dist_labels.append(str(d))
        else:
            dist_labels.append(f">{spatial_pos_max}")

    return bias_profiles, dist_labels


# ── Shared utility: node-only conditional attention ───────────

def node_only_conditional(A_all: torch.Tensor, eps: float = 1e-12):
    """Condition attention on node keys only (exclude CLS).

    Args:
        A_all: [H, T, T] post-softmax attention (CLS at index 0).

    Returns:
        p_node_cond: [H, n, n] rows sum to 1 over node keys
        cls_mass: [H, n] attention mass from node queries to CLS key
    """
    A_nn = A_all[:, 1:, 1:]
    row_mass = A_nn.sum(dim=-1, keepdim=True).clamp_min(eps)
    p_node_cond = A_nn / row_mass
    cls_mass = A_all[:, 1:, 0]
    return p_node_cond, cls_mass
