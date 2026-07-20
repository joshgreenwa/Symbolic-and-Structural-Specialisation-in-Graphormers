#!/usr/bin/env python3
"""Single-cell Colab runner for official Graphormer MolHIV fine-tuning.

Copy this entire file into one Google Colab code cell and run it. It mounts
Google Drive, creates a Python 3.9 virtual environment, installs the official
Microsoft Graphormer stack, clones Graphormer, fine-tunes on OGBG-MolHIV with
the Microsoft reproduction recipe, and evaluates ROC-AUC.

The symbolic/structural analysis notebooks in this repo use a separate, modern
Hugging Face stack to load the PCQM checkpoint:
  graph_interp.extraction.load_model()
  MODEL_ID = clefourrier/graphormer-base-pcqm4mv2
  MODEL_REVISION = refs/pr/4
That stack is intentionally separate from the official Microsoft Fairseq
training stack below. Set RUN_ANALYSIS_PCQM_SMOKE_TEST=1 to run the same
analysis-style PCQM load check before training.

Default runs target maximum OGBG-MolHIV AUC with the strongest public
Graphormer starting point available to this project: PCQM4Mv2. Microsoft's
actual `hiv_pre.sh` reproduction requires a
MolHIV-specific checkpoint that is listed in public registries but whose Azure
Blob URL is no longer publicly resolvable from current runtimes. The exact
Microsoft mode is therefore opt-in and will fail fast unless that checkpoint is
actually present in Drive/local cache.

Max-AUC public defaults:
  - Graphormer base, 12 layers, 768 hidden, 32 heads
  - Public PCQM4Mv2 checkpoint: pcqm4mv2_graphormer_base
  - Pre-layernorm disabled because public PCQM checkpoints are not pre-layernorm checkpoints
  - MolHIV fine-tune: 20 epochs, LR 2e-4, effective batch 128
  - warmup ratio 0.16, end LR 1e-5, Adam eps 1e-8, clip norm 5.0, weight decay 0
  - dropout 0.0, attention dropout 0.1, activation dropout 0.1
  - FLAG m=3, step size 0.01, magnitude 0

Exact Microsoft `hiv_pre.sh` mode:
  - Graphormer base, 12 layers, 768 hidden, 32 heads
  - MolHIV-specific public checkpoint: pcqm4mv1_graphormer_base_for_molhiv
  - Pre-layernorm enabled
  - MolHIV fine-tune: 4 epochs, LR 2e-4, effective batch 128
  - warmup ratio 0.16, end LR 1e-5, Adam eps 1e-8, clip norm 5.0, weight decay 0
  - dropout 0.0, attention dropout 0.1, activation dropout 0.1
  - FLAG m=3, step size 0.01, magnitude 0

By default this runner uses a Colab-safe batch implementation:
MICRO_BATCH_SIZE=32 and UPDATE_FREQ=4, preserving effective batch size 128. In
exact Microsoft mode it uses the literal script batch, MICRO_BATCH_SIZE=128 and
UPDATE_FREQ=1.

Strict Microsoft alignment is available only with CONFIG_PRESET=microsoft_hiv_pre.
Critical stale environment overrides are ignored while STRICT_MICROSOFT_CONFIG=1
and the run fails fast if the effective configuration drifts. The older NeurIPS
paper table recipe remains available by setting CONFIG_PRESET=paper_table.

Useful environment overrides before running:
  # Max-AUC public defaults:
  CONFIG_PRESET=public_pcqm_hiv_pre
  PRETRAINED_MODEL_NAME=pcqm4mv2_graphormer_base
  PRE_LAYERNORM_ARG=
  PAPER_EPOCHS=20
  WARMUP_RATIO=0.16
  WARMUP_RATIO_PERCENT=16
  END_LEARNING_RATE=1e-5
  DROPOUT=0.0
  FLAG_M=3
  FLAG_STEP_SIZE=0.01
  FLAG_MAG=0

  # Exact Microsoft hiv_pre.sh mode, only if the checkpoint is available:
  CONFIG_PRESET=microsoft_hiv_pre
  STRICT_MICROSOFT_CONFIG=1
  EXACT_PAPER_BATCH=1
  FORCE_RETRAIN=1
  PRETRAINED_MODEL_NAME=pcqm4mv1_graphormer_base_for_molhiv
  PAPER_EPOCHS=4
  WARMUP_RATIO=0.16
  WARMUP_RATIO_PERCENT=16
  END_LEARNING_RATE=1e-5
  DROPOUT=0.0
  FLAG_M=3
  FLAG_STEP_SIZE=0.01
  FLAG_MAG=0
  PRE_LAYERNORM_ARG=--pre-layernorm

  # Backup comparison only if PCQM4Mv2 underperforms:
  STRICT_MICROSOFT_CONFIG=0
  PRETRAINED_MODEL_NAME=pcqm4mv1_graphormer_base

  SEEDS=1,2,3
  MICRO_BATCH_SIZE=32
  UPDATE_FREQ=4
  DRIVE_RUN_ROOT=/content/drive/MyDrive/graphormer_molhiv_runs
  LOCAL_RUN_ROOT=/content/graphormer_molhiv_runs
  DISPLAY_FULL_FAIRSEQ_LOG=1
  BASH_XTRACE=1
  FORCE_RETRAIN=0
  SYNC_INTERVAL_SECONDS=120
  SAVE_INTERVAL_UPDATES=100
  KEEP_INTERVAL_UPDATES=-1
  TORCH_HOME=/content/graphormer_molhiv_runs/_torch_home
  PRETRAINED_CACHE_DRIVE=/content/drive/MyDrive/graphormer_molhiv_runs/_pretrained_cache
  MOLHIV_PRETRAINED_URL=https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_base_preln_pcqm4mv1_for_hiv.pt
  MOLHIV_PRETRAINED_LOCAL_PATH=/content/drive/MyDrive/graphormer_molhiv_runs/_pretrained_cache/checkpoint_base_preln_pcqm4mv1_for_hiv.pt
  HF_PCQM4MV1_PYTORCH_URL=https://huggingface.co/clefourrier/graphormer-base-pcqm4mv1/resolve/main/pytorch_model.bin
  HF_PCQM4MV2_PYTORCH_URL=https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/refs%2Fpr%2F4/pytorch_model.bin
  RUN_ANALYSIS_PCQM_SMOKE_TEST=1
  VALIDATE_PRETRAINED_LOAD=1
  REQUIRE_CUDA=1
  RESET_LOCAL_SETUP=1
  RUN_SETUP=0
  RUN_TRAIN=0
  RUN_EVAL=0

Default cell entry point:
  main()

Resume an interrupted Microsoft-recipe run instead of archiving it:
  main({"FORCE_RETRAIN": False})

Show full Fairseq stdout instead of the filtered notebook summary:
  main({"DISPLAY_FULL_FAIRSEQ_LOG": True})
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from collections import deque
from pathlib import Path
from urllib.parse import urlparse


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def run_bash(script: str, *, env: dict[str, str] | None = None, label: str = "bash") -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    process = subprocess.Popen(
        ["bash", "-lc", script],
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=80)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        tail.append(line.rstrip("\n"))
    return_code = process.wait()
    if return_code:
        print(f"\n{label} failed with exit code {return_code}. Last output lines:", flush=True)
        print("\n".join(tail), flush=True)
        raise RuntimeError(f"{label} failed with exit code {return_code}")


# Paths and reproducibility pins.
VENV_DIR = Path(os.environ.get("GRAPHORMER_VENV", "/content/graphormer-py39"))
GRAPHORMER_DIR = Path(os.environ.get("GRAPHORMER_DIR", "/content/Graphormer"))
DRIVE_MOUNT_POINT = Path(os.environ.get("DRIVE_MOUNT_POINT", "/content/drive"))
LOCAL_RUN_ROOT = Path(os.environ.get("LOCAL_RUN_ROOT", "/content/graphormer_molhiv_runs"))
DRIVE_RUN_ROOT = Path(
    os.environ.get(
        "DRIVE_RUN_ROOT",
        os.environ.get(
            "RUN_ROOT",
            str(DRIVE_MOUNT_POINT / "MyDrive" / "graphormer_molhiv_runs"),
        ),
    )
)
TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(LOCAL_RUN_ROOT / "_torch_home")))
PRETRAINED_CACHE_DRIVE = Path(
    os.environ.get("PRETRAINED_CACHE_DRIVE", str(DRIVE_RUN_ROOT / "_pretrained_cache"))
)
GRAPHORMER_COMMIT = os.environ.get(
    "GRAPHORMER_COMMIT", "a04573c40705fb174db261bb746a8258d00992f5"
)
PYTORCH_WHEEL_URL = os.environ.get(
    "PYTORCH_WHEEL_URL", "https" + "://download.pytorch.org/whl/torch_stable.html"
)
PYG_WHEEL_URL = os.environ.get(
    "PYG_WHEEL_URL", "https" + "://data.pyg.org/whl/torch-1.9.1+cu111.html"
)
DGL_WHEEL_URL = os.environ.get(
    "DGL_WHEEL_URL", "https" + "://data.dgl.ai/wheels/repo.html"
)
GRAPHORMER_GIT_URL = os.environ.get(
    "GRAPHORMER_GIT_URL", "https" + "://github.com/microsoft/Graphormer.git"
)
ANALYSIS_REPO_DIR = Path(os.environ.get("ANALYSIS_REPO_DIR", "/content/repo"))
ANALYSIS_REPO_URL = os.environ.get(
    "ANALYSIS_REPO_URL",
    "https"
    + "://github.com/joshgreenwa/Symbolic-and-Structural-Specialisation-in-Graphormers.git",
)
PYPI_SIMPLE_URL = os.environ.get("PYPI_SIMPLE_URL", "https" + "://pypi.org/simple")
HF_PCQM4MV1_PYTORCH_URL = os.environ.get(
    "HF_PCQM4MV1_PYTORCH_URL",
    "https"
    + "://huggingface.co/clefourrier/graphormer-base-pcqm4mv1/resolve/main/pytorch_model.bin",
)
HF_PCQM4MV2_PYTORCH_URL = os.environ.get(
    "HF_PCQM4MV2_PYTORCH_URL",
    "https"
    + "://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/refs%2Fpr%2F4/pytorch_model.bin",
)
MOLHIV_PRETRAINED_CANONICAL_FILENAME = "checkpoint_base_preln_pcqm4mv1_for_hiv.pt"
MOLHIV_PRETRAINED_URL = os.environ.get(
    "MOLHIV_PRETRAINED_URL",
    "https"
    + "://ml2md.blob.core.windows.net/graphormer-ckpts/"
    + MOLHIV_PRETRAINED_CANONICAL_FILENAME,
)
MOLHIV_PRETRAINED_ALT_URLS = os.environ.get(
    "MOLHIV_PRETRAINED_ALT_URLS",
    "https"
    + "://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv1/"
    + MOLHIV_PRETRAINED_CANONICAL_FILENAME,
)
MICROSOFT_PRESETS = {"microsoft_hiv_pre", "official_hiv_pre", "official_molhiv"}
PUBLIC_PCQM_PRESETS = {"public_hiv_pre_fallback", "public_pcqm_hiv_pre", "pcqm_hiv_pre"}
PAPER_TABLE_PRESETS = {"paper_table", "neurips_paper_table", "paper"}
MOLHIV_PRETRAINED_LOCAL_PATH = os.environ.get("MOLHIV_PRETRAINED_LOCAL_PATH", "")
PRETRAINED_MODEL_URL_OVERRIDE = os.environ.get("PRETRAINED_MODEL_URL_OVERRIDE", "")
SKIP_PRETRAINED_SOURCE_PREFLIGHT = env_bool("SKIP_PRETRAINED_SOURCE_PREFLIGHT", False)


CONFIG_PRESET = os.environ.get(
    "CONFIG_PRESET", os.environ.get("REPRODUCTION_CONFIG", "public_pcqm_hiv_pre")
).strip().lower()
STRICT_MICROSOFT_CONFIG = env_bool(
    "STRICT_MICROSOFT_CONFIG", CONFIG_PRESET in MICROSOFT_PRESETS
)
EXACT_PAPER_CONFIG = env_bool("EXACT_PAPER_CONFIG", False)
EXACT_PAPER_BATCH = env_bool("EXACT_PAPER_BATCH", CONFIG_PRESET in MICROSOFT_PRESETS)


def using_microsoft_hiv_pre() -> bool:
    return CONFIG_PRESET in MICROSOFT_PRESETS


def using_paper_table() -> bool:
    return CONFIG_PRESET in PAPER_TABLE_PRESETS


def using_public_pcqm_hiv_pre() -> bool:
    return CONFIG_PRESET in PUBLIC_PCQM_PRESETS


def config_value(
    name: str,
    fallback_default: str,
    microsoft_default: str,
    paper_table_default: str,
) -> str:
    """Return a config value while protecting strict Microsoft reproduction defaults."""
    if using_microsoft_hiv_pre():
        if STRICT_MICROSOFT_CONFIG:
            return microsoft_default
        return os.environ.get(name, microsoft_default)
    if using_public_pcqm_hiv_pre():
        return os.environ.get(name, fallback_default)
    if using_paper_table() and EXACT_PAPER_CONFIG:
        return paper_table_default
    return os.environ.get(name, fallback_default)


def batch_value(
    name: str,
    colab_default: str,
    microsoft_default: str,
    paper_table_default: str,
) -> str:
    if using_microsoft_hiv_pre():
        if STRICT_MICROSOFT_CONFIG:
            return microsoft_default
        return os.environ.get(name, microsoft_default if EXACT_PAPER_BATCH else colab_default)
    if using_public_pcqm_hiv_pre():
        return os.environ.get(name, colab_default)
    if using_paper_table() and EXACT_PAPER_CONFIG:
        return paper_table_default if EXACT_PAPER_BATCH else colab_default
    return os.environ.get(name, colab_default)


# Training defaults.
SEEDS = os.environ.get("SEEDS", "1")
N_GPU = config_value("N_GPU", "1", "1", "1")
PAPER_EPOCHS = config_value("PAPER_EPOCHS", "20", "4", "8")
TRAIN_SIZE_APPROX = config_value("TRAIN_SIZE_APPROX", "33000", "33000", "33000")
EFFECTIVE_BATCH_SIZE = config_value("EFFECTIVE_BATCH_SIZE", "128", "128", "128")
MICRO_BATCH_SIZE = batch_value("MICRO_BATCH_SIZE", "32", "128", "128")
UPDATE_FREQ = batch_value("UPDATE_FREQ", "4", "1", "1")
NUM_WORKERS = config_value("NUM_WORKERS", "2", "16", "2")
EVAL_BATCH_SIZE = os.environ.get("EVAL_BATCH_SIZE", "64")
WARMUP_RATIO = config_value("WARMUP_RATIO", "0.16", "0.16", "0.06")
WARMUP_RATIO_PERCENT = config_value("WARMUP_RATIO_PERCENT", "16", "16", "6")
PEAK_LEARNING_RATE = config_value("PEAK_LEARNING_RATE", "2e-4", "2e-4", "2e-4")
END_LEARNING_RATE = config_value("END_LEARNING_RATE", "1e-5", "1e-5", "1e-9")
ATTENTION_DROPOUT = config_value("ATTENTION_DROPOUT", "0.1", "0.1", "0.1")
ACT_DROPOUT = config_value("ACT_DROPOUT", "0.1", "0.1", "0.1")
DROPOUT = config_value("DROPOUT", "0.0", "0.0", "0.1")
FLAG_M = config_value("FLAG_M", "3", "3", "2")
FLAG_STEP_SIZE = config_value("FLAG_STEP_SIZE", "0.01", "0.01", "0.2")
FLAG_MAG = config_value("FLAG_MAG", "0", "0", "0")
SYNC_INTERVAL_SECONDS = os.environ.get("SYNC_INTERVAL_SECONDS", "120")
SAVE_INTERVAL_UPDATES = os.environ.get("SAVE_INTERVAL_UPDATES", "100")
KEEP_INTERVAL_UPDATES = os.environ.get("KEEP_INTERVAL_UPDATES", "-1")

# The Microsoft target uses the MolHIV-specific pre-layernorm PCQM4M checkpoint.
# Plain PCQM4Mv1/PCQM4Mv2 checkpoints remain opt-in analysis/ablation variants.
PRETRAINED_MODEL_NAME = config_value(
    "PRETRAINED_MODEL_NAME",
    "pcqm4mv2_graphormer_base",
    "pcqm4mv1_graphormer_base_for_molhiv",
    "pcqm4mv1_graphormer_base",
)
if using_microsoft_hiv_pre():
    if STRICT_MICROSOFT_CONFIG:
        PRE_LAYERNORM_ARG = "--pre-layernorm"
    else:
        PRE_LAYERNORM_ARG = os.environ.get("PRE_LAYERNORM_ARG", "--pre-layernorm")
elif using_paper_table() and EXACT_PAPER_CONFIG:
    PRE_LAYERNORM_ARG = ""
elif using_public_pcqm_hiv_pre():
    PRE_LAYERNORM_ARG = os.environ.get("PRE_LAYERNORM_ARG", "")
else:
    PRE_LAYERNORM_ARG = os.environ.get(
        "PRE_LAYERNORM_ARG",
        "--pre-layernorm"
        if PRETRAINED_MODEL_NAME == "pcqm4mv1_graphormer_base_for_molhiv"
        else "",
    )

RUN_SETUP = env_bool("RUN_SETUP", True)
RUN_TRAIN = env_bool("RUN_TRAIN", True)
RUN_EVAL = env_bool("RUN_EVAL", True)
RUN_ANALYSIS_PCQM_SMOKE_TEST = env_bool("RUN_ANALYSIS_PCQM_SMOKE_TEST", False)
VALIDATE_PRETRAINED_LOAD = env_bool("VALIDATE_PRETRAINED_LOAD", True)
MOUNT_DRIVE = env_bool("MOUNT_DRIVE", True)
FORCE_REMOUNT_DRIVE = env_bool("FORCE_REMOUNT_DRIVE", False)
RESET_LOCAL_SETUP = env_bool("RESET_LOCAL_SETUP", False)
REQUIRE_CUDA = env_bool("REQUIRE_CUDA", True)
FORCE_RETRAIN = env_bool("FORCE_RETRAIN", True)
ARCHIVE_EXISTING_RUN = env_bool("ARCHIVE_EXISTING_RUN", True)
DISPLAY_FULL_FAIRSEQ_LOG = env_bool("DISPLAY_FULL_FAIRSEQ_LOG", False)
BASH_XTRACE = env_bool("BASH_XTRACE", False)


def reload_config_from_env() -> None:
    global VENV_DIR, GRAPHORMER_DIR, DRIVE_MOUNT_POINT, LOCAL_RUN_ROOT, DRIVE_RUN_ROOT
    global TORCH_HOME, PRETRAINED_CACHE_DRIVE, GRAPHORMER_COMMIT, PYTORCH_WHEEL_URL
    global PYG_WHEEL_URL, DGL_WHEEL_URL, GRAPHORMER_GIT_URL, ANALYSIS_REPO_DIR
    global ANALYSIS_REPO_URL, PYPI_SIMPLE_URL, HF_PCQM4MV1_PYTORCH_URL
    global HF_PCQM4MV2_PYTORCH_URL, MOLHIV_PRETRAINED_URL, MOLHIV_PRETRAINED_ALT_URLS
    global MOLHIV_PRETRAINED_LOCAL_PATH, PRETRAINED_MODEL_URL_OVERRIDE
    global SKIP_PRETRAINED_SOURCE_PREFLIGHT, CONFIG_PRESET, STRICT_MICROSOFT_CONFIG
    global EXACT_PAPER_CONFIG, EXACT_PAPER_BATCH
    global SEEDS, N_GPU, PAPER_EPOCHS, TRAIN_SIZE_APPROX, EFFECTIVE_BATCH_SIZE
    global MICRO_BATCH_SIZE, UPDATE_FREQ
    global NUM_WORKERS, EVAL_BATCH_SIZE, WARMUP_RATIO, WARMUP_RATIO_PERCENT
    global PEAK_LEARNING_RATE, END_LEARNING_RATE, ATTENTION_DROPOUT, ACT_DROPOUT
    global DROPOUT, FLAG_M, FLAG_STEP_SIZE, FLAG_MAG, SYNC_INTERVAL_SECONDS
    global SAVE_INTERVAL_UPDATES, KEEP_INTERVAL_UPDATES, PRETRAINED_MODEL_NAME
    global PRE_LAYERNORM_ARG, RUN_SETUP, RUN_TRAIN, RUN_EVAL
    global RUN_ANALYSIS_PCQM_SMOKE_TEST, VALIDATE_PRETRAINED_LOAD, MOUNT_DRIVE
    global FORCE_REMOUNT_DRIVE, RESET_LOCAL_SETUP, REQUIRE_CUDA, FORCE_RETRAIN
    global ARCHIVE_EXISTING_RUN, DISPLAY_FULL_FAIRSEQ_LOG, BASH_XTRACE

    VENV_DIR = Path(os.environ.get("GRAPHORMER_VENV", "/content/graphormer-py39"))
    GRAPHORMER_DIR = Path(os.environ.get("GRAPHORMER_DIR", "/content/Graphormer"))
    DRIVE_MOUNT_POINT = Path(os.environ.get("DRIVE_MOUNT_POINT", "/content/drive"))
    LOCAL_RUN_ROOT = Path(os.environ.get("LOCAL_RUN_ROOT", "/content/graphormer_molhiv_runs"))
    DRIVE_RUN_ROOT = Path(
        os.environ.get(
            "DRIVE_RUN_ROOT",
            os.environ.get(
                "RUN_ROOT",
                str(DRIVE_MOUNT_POINT / "MyDrive" / "graphormer_molhiv_runs"),
            ),
        )
    )
    TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(LOCAL_RUN_ROOT / "_torch_home")))
    PRETRAINED_CACHE_DRIVE = Path(
        os.environ.get("PRETRAINED_CACHE_DRIVE", str(DRIVE_RUN_ROOT / "_pretrained_cache"))
    )
    GRAPHORMER_COMMIT = os.environ.get(
        "GRAPHORMER_COMMIT", "a04573c40705fb174db261bb746a8258d00992f5"
    )
    PYTORCH_WHEEL_URL = os.environ.get(
        "PYTORCH_WHEEL_URL", "https" + "://download.pytorch.org/whl/torch_stable.html"
    )
    PYG_WHEEL_URL = os.environ.get(
        "PYG_WHEEL_URL", "https" + "://data.pyg.org/whl/torch-1.9.1+cu111.html"
    )
    DGL_WHEEL_URL = os.environ.get(
        "DGL_WHEEL_URL", "https" + "://data.dgl.ai/wheels/repo.html"
    )
    GRAPHORMER_GIT_URL = os.environ.get(
        "GRAPHORMER_GIT_URL", "https" + "://github.com/microsoft/Graphormer.git"
    )
    ANALYSIS_REPO_DIR = Path(os.environ.get("ANALYSIS_REPO_DIR", "/content/repo"))
    ANALYSIS_REPO_URL = os.environ.get(
        "ANALYSIS_REPO_URL",
        "https"
        + "://github.com/joshgreenwa/Symbolic-and-Structural-Specialisation-in-Graphormers.git",
    )
    PYPI_SIMPLE_URL = os.environ.get("PYPI_SIMPLE_URL", "https" + "://pypi.org/simple")
    HF_PCQM4MV1_PYTORCH_URL = os.environ.get(
        "HF_PCQM4MV1_PYTORCH_URL",
        "https"
        + "://huggingface.co/clefourrier/graphormer-base-pcqm4mv1/resolve/main/pytorch_model.bin",
    )
    HF_PCQM4MV2_PYTORCH_URL = os.environ.get(
        "HF_PCQM4MV2_PYTORCH_URL",
        "https"
        + "://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/refs%2Fpr%2F4/pytorch_model.bin",
    )
    MOLHIV_PRETRAINED_URL = os.environ.get(
        "MOLHIV_PRETRAINED_URL",
        "https"
        + "://ml2md.blob.core.windows.net/graphormer-ckpts/"
        + MOLHIV_PRETRAINED_CANONICAL_FILENAME,
    )
    MOLHIV_PRETRAINED_ALT_URLS = os.environ.get(
        "MOLHIV_PRETRAINED_ALT_URLS",
        "https"
        + "://szheng.blob.core.windows.net/graphormer/modelzoo/pcqm4mv1/"
        + MOLHIV_PRETRAINED_CANONICAL_FILENAME,
    )
    MOLHIV_PRETRAINED_LOCAL_PATH = os.environ.get("MOLHIV_PRETRAINED_LOCAL_PATH", "")
    PRETRAINED_MODEL_URL_OVERRIDE = os.environ.get("PRETRAINED_MODEL_URL_OVERRIDE", "")
    SKIP_PRETRAINED_SOURCE_PREFLIGHT = env_bool("SKIP_PRETRAINED_SOURCE_PREFLIGHT", False)

    CONFIG_PRESET = os.environ.get(
        "CONFIG_PRESET", os.environ.get("REPRODUCTION_CONFIG", "public_pcqm_hiv_pre")
    ).strip().lower()
    STRICT_MICROSOFT_CONFIG = env_bool(
        "STRICT_MICROSOFT_CONFIG", CONFIG_PRESET in MICROSOFT_PRESETS
    )
    EXACT_PAPER_CONFIG = env_bool("EXACT_PAPER_CONFIG", False)
    EXACT_PAPER_BATCH = env_bool("EXACT_PAPER_BATCH", CONFIG_PRESET in MICROSOFT_PRESETS)
    SEEDS = os.environ.get("SEEDS", "1")
    N_GPU = config_value("N_GPU", "1", "1", "1")
    PAPER_EPOCHS = config_value("PAPER_EPOCHS", "20", "4", "8")
    TRAIN_SIZE_APPROX = config_value("TRAIN_SIZE_APPROX", "33000", "33000", "33000")
    EFFECTIVE_BATCH_SIZE = config_value("EFFECTIVE_BATCH_SIZE", "128", "128", "128")
    MICRO_BATCH_SIZE = batch_value("MICRO_BATCH_SIZE", "32", "128", "128")
    UPDATE_FREQ = batch_value("UPDATE_FREQ", "4", "1", "1")
    NUM_WORKERS = config_value("NUM_WORKERS", "2", "16", "2")
    EVAL_BATCH_SIZE = os.environ.get("EVAL_BATCH_SIZE", "64")
    WARMUP_RATIO = config_value("WARMUP_RATIO", "0.16", "0.16", "0.06")
    WARMUP_RATIO_PERCENT = config_value("WARMUP_RATIO_PERCENT", "16", "16", "6")
    PEAK_LEARNING_RATE = config_value("PEAK_LEARNING_RATE", "2e-4", "2e-4", "2e-4")
    END_LEARNING_RATE = config_value("END_LEARNING_RATE", "1e-5", "1e-5", "1e-9")
    ATTENTION_DROPOUT = config_value("ATTENTION_DROPOUT", "0.1", "0.1", "0.1")
    ACT_DROPOUT = config_value("ACT_DROPOUT", "0.1", "0.1", "0.1")
    DROPOUT = config_value("DROPOUT", "0.0", "0.0", "0.1")
    FLAG_M = config_value("FLAG_M", "3", "3", "2")
    FLAG_STEP_SIZE = config_value("FLAG_STEP_SIZE", "0.01", "0.01", "0.2")
    FLAG_MAG = config_value("FLAG_MAG", "0", "0", "0")
    SYNC_INTERVAL_SECONDS = os.environ.get("SYNC_INTERVAL_SECONDS", "120")
    SAVE_INTERVAL_UPDATES = os.environ.get("SAVE_INTERVAL_UPDATES", "100")
    KEEP_INTERVAL_UPDATES = os.environ.get("KEEP_INTERVAL_UPDATES", "-1")
    PRETRAINED_MODEL_NAME = config_value(
        "PRETRAINED_MODEL_NAME",
        "pcqm4mv2_graphormer_base",
        "pcqm4mv1_graphormer_base_for_molhiv",
        "pcqm4mv1_graphormer_base",
    )
    if using_microsoft_hiv_pre():
        if STRICT_MICROSOFT_CONFIG:
            PRE_LAYERNORM_ARG = "--pre-layernorm"
        else:
            PRE_LAYERNORM_ARG = os.environ.get("PRE_LAYERNORM_ARG", "--pre-layernorm")
    elif using_paper_table() and EXACT_PAPER_CONFIG:
        PRE_LAYERNORM_ARG = ""
    elif using_public_pcqm_hiv_pre():
        PRE_LAYERNORM_ARG = os.environ.get("PRE_LAYERNORM_ARG", "")
    else:
        PRE_LAYERNORM_ARG = os.environ.get(
            "PRE_LAYERNORM_ARG",
            "--pre-layernorm"
            if PRETRAINED_MODEL_NAME == "pcqm4mv1_graphormer_base_for_molhiv"
            else "",
        )

    RUN_SETUP = env_bool("RUN_SETUP", True)
    RUN_TRAIN = env_bool("RUN_TRAIN", True)
    RUN_EVAL = env_bool("RUN_EVAL", True)
    RUN_ANALYSIS_PCQM_SMOKE_TEST = env_bool("RUN_ANALYSIS_PCQM_SMOKE_TEST", False)
    VALIDATE_PRETRAINED_LOAD = env_bool("VALIDATE_PRETRAINED_LOAD", True)
    MOUNT_DRIVE = env_bool("MOUNT_DRIVE", True)
    FORCE_REMOUNT_DRIVE = env_bool("FORCE_REMOUNT_DRIVE", False)
    RESET_LOCAL_SETUP = env_bool("RESET_LOCAL_SETUP", False)
    REQUIRE_CUDA = env_bool("REQUIRE_CUDA", True)
    FORCE_RETRAIN = env_bool("FORCE_RETRAIN", True)
    ARCHIVE_EXISTING_RUN = env_bool("ARCHIVE_EXISTING_RUN", True)
    DISPLAY_FULL_FAIRSEQ_LOG = env_bool("DISPLAY_FULL_FAIRSEQ_LOG", False)
    BASH_XTRACE = env_bool("BASH_XTRACE", False)


def apply_main_overrides(overrides: dict[str, object] | None) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if isinstance(value, bool):
            os.environ[str(key)] = "1" if value else "0"
        else:
            os.environ[str(key)] = str(value)
    reload_config_from_env()


def mount_drive() -> None:
    if not MOUNT_DRIVE:
        print("Skipping Google Drive mount because MOUNT_DRIVE=0.", flush=True)
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception as exc:
        print(f"google.colab drive mount unavailable; continuing without mount: {exc}", flush=True)
        return
    drive.mount(str(DRIVE_MOUNT_POINT), force_remount=FORCE_REMOUNT_DRIVE)


def show_runtime() -> None:
    subprocess.run(["bash", "-lc", "command -v nvidia-smi >/dev/null && nvidia-smi || true"], check=False)
    print("Colab host Python:", sys.version, flush=True)
    print("Graphormer venv:", VENV_DIR, flush=True)
    print("Graphormer repo:", GRAPHORMER_DIR, flush=True)
    print("Local run root:", LOCAL_RUN_ROOT, flush=True)
    print("Drive run root:", DRIVE_RUN_ROOT, flush=True)
    print("Torch cache:", TORCH_HOME, flush=True)
    print("Drive pretrained cache:", PRETRAINED_CACHE_DRIVE, flush=True)
    if PRETRAINED_MODEL_NAME == "pcqm4mv1_graphormer_base_for_molhiv":
        print(
            "Expected MolHIV pretrained checkpoint:",
            PRETRAINED_CACHE_DRIVE / MOLHIV_PRETRAINED_CANONICAL_FILENAME,
            flush=True,
        )
        if MOLHIV_PRETRAINED_LOCAL_PATH:
            print("MolHIV pretrained local override:", MOLHIV_PRETRAINED_LOCAL_PATH, flush=True)
    print("Drive sync interval seconds:", SYNC_INTERVAL_SECONDS, flush=True)
    print("Config preset:", CONFIG_PRESET, flush=True)
    print("Strict Microsoft config:", STRICT_MICROSOFT_CONFIG, flush=True)
    print("Legacy exact paper-table config:", EXACT_PAPER_CONFIG, flush=True)
    print("Literal paper batch:", EXACT_PAPER_BATCH, flush=True)
    print("Force retrain:", FORCE_RETRAIN, flush=True)
    print("Archive existing run before retrain:", ARCHIVE_EXISTING_RUN, flush=True)
    print("Pretrained model:", PRETRAINED_MODEL_NAME, flush=True)
    print("Paper epochs:", PAPER_EPOCHS, flush=True)
    print("Effective batch size:", EFFECTIVE_BATCH_SIZE, flush=True)
    print("Micro batch size:", MICRO_BATCH_SIZE, flush=True)
    print("Update freq:", UPDATE_FREQ, flush=True)
    print("Warmup ratio:", WARMUP_RATIO or f"{WARMUP_RATIO_PERCENT}%", flush=True)
    print("Peak LR:", PEAK_LEARNING_RATE, flush=True)
    print("End LR:", END_LEARNING_RATE, flush=True)
    print("Dropout:", DROPOUT, flush=True)
    print("Attention dropout:", ATTENTION_DROPOUT, flush=True)
    print("Activation dropout:", ACT_DROPOUT, flush=True)
    print("FLAG m:", FLAG_M, flush=True)
    print("FLAG step size:", FLAG_STEP_SIZE, flush=True)
    print("FLAG magnitude:", FLAG_MAG, flush=True)
    print("Pre-layernorm arg:", PRE_LAYERNORM_ARG or "(disabled)", flush=True)
    print("Save interval updates:", SAVE_INTERVAL_UPDATES, flush=True)
    print("Keep interval updates:", KEEP_INTERVAL_UPDATES, flush=True)
    print("Display full Fairseq log:", DISPLAY_FULL_FAIRSEQ_LOG, flush=True)
    print("Bash xtrace:", BASH_XTRACE, flush=True)
    print("Analysis PCQM smoke test:", RUN_ANALYSIS_PCQM_SMOKE_TEST, flush=True)
    print("Validate pretrained load:", VALIDATE_PRETRAINED_LOAD, flush=True)
    print("Require CUDA:", REQUIRE_CUDA, flush=True)


def validate_config_alignment() -> None:
    """Fail fast when strict Microsoft MolHIV reproduction settings drift."""
    if not using_microsoft_hiv_pre():
        if using_public_pcqm_hiv_pre():
            expected = {
                "PRE_LAYERNORM_ARG": (PRE_LAYERNORM_ARG, ""),
                "PAPER_EPOCHS": (PAPER_EPOCHS, "20"),
                "EFFECTIVE_BATCH_SIZE": (EFFECTIVE_BATCH_SIZE, "128"),
                "MICRO_BATCH_SIZE": (MICRO_BATCH_SIZE, "32"),
                "UPDATE_FREQ": (UPDATE_FREQ, "4"),
                "WARMUP_RATIO": (WARMUP_RATIO, "0.16"),
                "PEAK_LEARNING_RATE": (PEAK_LEARNING_RATE, "2e-4"),
                "END_LEARNING_RATE": (END_LEARNING_RATE, "1e-5"),
                "DROPOUT": (DROPOUT, "0.0"),
                "FLAG_M": (FLAG_M, "3"),
                "FLAG_STEP_SIZE": (FLAG_STEP_SIZE, "0.01"),
                "FLAG_MAG": (FLAG_MAG, "0"),
            }
            mismatches = [
                f"{name}: got {actual!r}, expected {want!r}"
                for name, (actual, want) in expected.items()
                if str(actual) != str(want)
            ]
            if mismatches:
                raise RuntimeError(
                    "Public PCQM MolHIV fine-tune config drift detected:\n"
                    + "\n".join(mismatches)
                )
            allowed_pretrained = {"pcqm4mv2_graphormer_base", "pcqm4mv1_graphormer_base"}
            if PRETRAINED_MODEL_NAME not in allowed_pretrained:
                raise RuntimeError(
                    "Public PCQM MolHIV fine-tune must use pcqm4mv2_graphormer_base "
                    f"or backup pcqm4mv1_graphormer_base, got {PRETRAINED_MODEL_NAME!r}"
                )
            print(
                "Public PCQM MolHIV fine-tune config check passed. "
                "Primary max-AUC run uses PCQM4Mv2; PCQM4Mv1 is allowed only as backup. "
                "This is not exact Microsoft hiv_pre.sh initialization.",
                flush=True,
            )
            return
        print(f"Non-Microsoft config preset selected: {CONFIG_PRESET}", flush=True)
        return
    if not STRICT_MICROSOFT_CONFIG:
        print(
            "WARNING: STRICT_MICROSOFT_CONFIG=0; Microsoft hiv_pre.sh settings may be overridden.",
            flush=True,
        )
        return

    expected = {
        "PRETRAINED_MODEL_NAME": (PRETRAINED_MODEL_NAME, "pcqm4mv1_graphormer_base_for_molhiv"),
        "PRE_LAYERNORM_ARG": (PRE_LAYERNORM_ARG, "--pre-layernorm"),
        "PAPER_EPOCHS": (PAPER_EPOCHS, "4"),
        "EFFECTIVE_BATCH_SIZE": (EFFECTIVE_BATCH_SIZE, "128"),
        "MICRO_BATCH_SIZE": (MICRO_BATCH_SIZE, "128"),
        "UPDATE_FREQ": (UPDATE_FREQ, "1"),
        "WARMUP_RATIO": (WARMUP_RATIO, "0.16"),
        "WARMUP_RATIO_PERCENT": (WARMUP_RATIO_PERCENT, "16"),
        "PEAK_LEARNING_RATE": (PEAK_LEARNING_RATE, "2e-4"),
        "END_LEARNING_RATE": (END_LEARNING_RATE, "1e-5"),
        "DROPOUT": (DROPOUT, "0.0"),
        "ATTENTION_DROPOUT": (ATTENTION_DROPOUT, "0.1"),
        "ACT_DROPOUT": (ACT_DROPOUT, "0.1"),
        "FLAG_M": (FLAG_M, "3"),
        "FLAG_STEP_SIZE": (FLAG_STEP_SIZE, "0.01"),
        "FLAG_MAG": (FLAG_MAG, "0"),
        "NUM_WORKERS": (NUM_WORKERS, "16"),
    }
    mismatches = [
        f"{name}: got {actual!r}, expected {want!r}"
        for name, (actual, want) in expected.items()
        if str(actual) != str(want)
    ]
    if mismatches:
        raise RuntimeError(
            "Strict Microsoft MolHIV config drift detected:\n" + "\n".join(mismatches)
        )
    if not FORCE_RETRAIN:
        print(
            "WARNING: FORCE_RETRAIN=0; this will resume an existing run instead of starting fresh.",
            flush=True,
        )
    print("Strict Microsoft MolHIV config alignment check passed.", flush=True)


def _candidate_molhiv_pretrained_urls() -> list[str]:
    urls: list[str] = []
    for raw in [PRETRAINED_MODEL_URL_OVERRIDE, MOLHIV_PRETRAINED_URL, MOLHIV_PRETRAINED_ALT_URLS]:
        for url in str(raw or "").split(","):
            url = url.strip()
            if url and url not in urls:
                urls.append(url)
    return urls


def preflight_pretrained_source() -> None:
    """Catch dead public checkpoint URLs before spending time on Colab setup."""
    if (
        PRETRAINED_MODEL_NAME != "pcqm4mv1_graphormer_base_for_molhiv"
        or SKIP_PRETRAINED_SOURCE_PREFLIGHT
        or not (RUN_SETUP or RUN_TRAIN or VALIDATE_PRETRAINED_LOAD)
    ):
        return

    drive_path = PRETRAINED_CACHE_DRIVE / MOLHIV_PRETRAINED_CANONICAL_FILENAME
    local_cache_path = TORCH_HOME / "hub" / "checkpoints" / MOLHIV_PRETRAINED_CANONICAL_FILENAME
    manual_paths = [
        Path(path).expanduser()
        for path in [
            MOLHIV_PRETRAINED_LOCAL_PATH,
            os.environ.get("PRETRAINED_CHECKPOINT_LOCAL_PATH", ""),
            os.environ.get("PRETRAINED_CHECKPOINT_PATH", ""),
        ]
        if str(path).strip()
    ]
    for path in [*manual_paths, drive_path, local_cache_path]:
        if path.is_file() and path.stat().st_size > 1_000_000:
            print(f"Pretrained MolHIV checkpoint source found locally: {path}", flush=True)
            return

    errors: list[str] = []
    for url in _candidate_molhiv_pretrained_urls():
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=20) as response:
                status = getattr(response, "status", 200)
                if 200 <= status < 400:
                    print(f"Pretrained MolHIV checkpoint URL is reachable: {url}", flush=True)
                    return
                errors.append(f"{url} -> HTTP {status}")
        except Exception as exc:
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")

    error_lines = "\n".join(f"  - {line}" for line in errors)
    raise RuntimeError(
        "The Microsoft MolHIV pretrained checkpoint is not available from this runtime.\n"
        "This is the checkpoint required for strict Microsoft hiv_pre.sh reproduction:\n"
        f"  {MOLHIV_PRETRAINED_CANONICAL_FILENAME}\n\n"
        "Checked local/cache paths:\n"
        f"  - {drive_path}\n"
        f"  - {local_cache_path}\n"
        + "".join(f"  - {path}\n" for path in manual_paths)
        + "\nChecked public URLs:\n"
        + error_lines
        + "\n\n"
        "Fix: place the checkpoint file in the Drive cache path above, or set "
        "MOLHIV_PRETRAINED_LOCAL_PATH / PRETRAINED_CHECKPOINT_LOCAL_PATH to a valid copy. "
        "The runner will validate and mirror it before training."
    )


def analysis_pcqm_smoke_test() -> None:
    """Run the same PCQM Hugging Face loader path used by analysis notebooks."""
    run_bash(
        r"""
set -euo pipefail
if [ "${BASH_XTRACE:-0}" = "1" ]; then
  set -x
fi
filter_known_noise() {
  grep -v -E "DEPRECATION: omegaconf 2\\.0\\.6 has a non-standard dependency specifier|https://github.com/pypa/pip/issues/12063" || true
}
pip_install_host() {
  python3 -m pip install "$@" 2> >(filter_known_noise >&2)
}

if [ ! -d "$ANALYSIS_REPO_DIR/.git" ]; then
  git clone "$ANALYSIS_REPO_URL" "$ANALYSIS_REPO_DIR"
fi

cd "$ANALYSIS_REPO_DIR"
pip_install_host -q -r "$ANALYSIS_REPO_DIR/requirements.lock.txt"
pip_install_host -q -i "$PYPI_SIMPLE_URL" rdkit

python3 - <<'PY'
import os
import sys
import torch

repo = os.environ["ANALYSIS_REPO_DIR"]
if repo not in sys.path:
    sys.path.insert(0, repo)

from graph_interp.config import MODEL_ID, MODEL_REVISION
from graph_interp.extraction import load_model

print("analysis MODEL_ID", MODEL_ID)
print("analysis MODEL_REVISION", MODEL_REVISION)
model, device = load_model("cpu")
print("analysis PCQM model loaded on", device)
print("analysis encoder layers", len(model.graph_encoder.layers))
del model
torch.cuda.empty_cache()
PY
""",
        env={
            "ANALYSIS_REPO_DIR": str(ANALYSIS_REPO_DIR),
            "ANALYSIS_REPO_URL": ANALYSIS_REPO_URL,
            "PYPI_SIMPLE_URL": PYPI_SIMPLE_URL,
            "BASH_XTRACE": "1" if BASH_XTRACE else "0",
        },
        label="analysis PCQM smoke test",
    )


def setup() -> None:
    run_bash(
        r"""
set -euo pipefail
if [ "${BASH_XTRACE:-0}" = "1" ]; then
  set -x
fi
step() {
  echo
  echo "===== $* ====="
}
filter_known_noise() {
  grep -v -E \
    "DEPRECATION: omegaconf 2\\.0\\.6 has a non-standard dependency specifier|https://github.com/pypa/pip/issues/12063|Skipping acquire of configured file 'main/source/Sources'" \
    || true
}
pip_install_host() {
  python3 -m pip install "$@" 2> >(filter_known_noise >&2)
}
pip_install() {
  python -m pip install "$@" 2> >(filter_known_noise >&2)
}
export_cuda11_paths() {
  CUDA_LIB_PATHS=$(python - <<'PY'
import glob
import os
import site

paths = []
for site_dir in site.getsitepackages():
    for pattern in (
        os.path.join(site_dir, "torch", "lib"),
        os.path.join(site_dir, "nvidia", "*", "lib"),
    ):
        for path in glob.glob(pattern):
            if os.path.isdir(path) and path not in paths:
                paths.append(path)
print(":".join(paths))
PY
)
  if [ -n "$CUDA_LIB_PATHS" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "CUDA runtime library paths configured."
  fi
}

export DEBIAN_FRONTEND=noninteractive
export DGLBACKEND=pytorch
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME="$TORCH_HOME"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
step "Install OS packages"
apt-get update -qq 2> >(filter_known_noise >&2)
apt-get install -y -qq build-essential git wget curl ca-certificates rsync ninja-build

if [ "$RESET_LOCAL_SETUP" = "1" ]; then
  step "Reset local setup only"
  rm -rf "$VENV_DIR" "$GRAPHORMER_DIR"
fi

step "Install uv"
pip_install_host -q --upgrade uv

step "Create or validate Python 3.9 venv"
if [ -d "$VENV_DIR" ] && { [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 9):
    raise SystemExit(1)
try:
    import pip  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}; then
  echo "Removing malformed local venv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  uv python install 3.9
  uv venv "$VENV_DIR" --python 3.9 --seed
fi

source "$VENV_DIR/bin/activate"
python - <<'PY'
import sys
print("venv python", sys.version)
assert sys.version_info[:2] == (3, 9), sys.version
PY
if ! python -m pip --version; then
  python -m ensurepip --upgrade || true
fi
if ! python -m pip --version; then
  uv pip install --python "$VENV_DIR/bin/python" "pip==23.3.2" setuptools wheel
fi

step "Install Python build pins"
pip_install -q --upgrade "pip==23.3.2" "setuptools==59.5.0" wheel "Cython==0.29.36" "numpy==1.23.5"

step "Install PyTorch 1.9.1 CUDA 11.1"
pip_install -q \
  torch==1.9.1+cu111 torchaudio==0.9.1 \
  -f "$PYTORCH_WHEEL_URL"

step "Install CUDA 11 runtime libraries missing from modern Colab"
pip_install -q \
  nvidia-cuda-runtime-cu11==11.7.99 \
  nvidia-cublas-cu11==11.11.3.6 \
  nvidia-cusparse-cu11==11.7.5.86 \
  nvidia-cusolver-cu11==11.4.1.48 \
  nvidia-curand-cu11==10.2.10.91 \
  nvidia-cufft-cu11==10.9.0.58
export_cuda11_paths

step "Install molecule/data dependencies"
pip_install -q \
  lmdb tensorboard==2.11.2 tensorboardX==2.4.1 protobuf==3.20.3 \
  ogb==1.3.2 rdkit-pypi==2021.9.3 \
  scikit-learn==1.0.2 tqdm

step "Install PyG compiled dependencies"
pip_install -q \
  torch-scatter==2.0.9 torch-sparse==0.6.12 \
  -f "$PYG_WHEEL_URL"

step "Install PyG and DGL"
pip_install -q torch-geometric==1.7.2

step "Install PyG dataset import dependencies"
# Graphormer imports torch_geometric.datasets at task-registration time.
# PyG 1.7.2 imports many dataset modules eagerly, so these dataset-only extras
# are required even when the actual run uses only OGBG-MolHIV.
pip_install -q \
  googledrivedownloader==0.4 \
  python-louvain==0.16 \
  pandas==1.5.3 \
  networkx==2.8.8 \
  scipy==1.10.1
pip_install -q --upgrade "numpy==1.23.5" "protobuf==3.20.3"

pip_install -q dgl==0.7.2 -f "$DGL_WHEEL_URL"

step "Clone official Graphormer"
if [ -e "$GRAPHORMER_DIR" ] && [ ! -d "$GRAPHORMER_DIR/.git" ]; then
  echo "Removing malformed local Graphormer directory: $GRAPHORMER_DIR"
  rm -rf "$GRAPHORMER_DIR"
fi
if [ ! -d "$GRAPHORMER_DIR/.git" ]; then
  git clone "$GRAPHORMER_GIT_URL" "$GRAPHORMER_DIR"
fi
cd "$GRAPHORMER_DIR"
git fetch --depth 1 origin "$GRAPHORMER_COMMIT"
git checkout "$GRAPHORMER_COMMIT"
git submodule update --init --recursive

step "Install bundled Fairseq"
cd "$GRAPHORMER_DIR/fairseq"
pip_install -q -e .
pip_install -q --upgrade "Cython==0.29.36" "numpy==1.23.5"
pip_install -q --upgrade "protobuf==3.20.3" "tensorboard==2.11.2" "tensorboardX==2.4.1"
rm -rf build
rm -f fairseq/data/data_utils_fast.cpp fairseq/data/token_block_utils_fast.cpp
python -m cython -3 --cplus fairseq/data/data_utils_fast.pyx -o fairseq/data/data_utils_fast.cpp
python -m cython -3 --cplus fairseq/data/token_block_utils_fast.pyx -o fairseq/data/token_block_utils_fast.cpp
python - <<'PY'
from pathlib import Path

path = Path("examples/operators/alignment_train_cpu.cpp")
if path.exists():
    text = path.read_text()
    text = text.replace("free(cumprod_1mp);", "delete[] cumprod_1mp;")
    text = text.replace("free(cumprod_1mp_clamp);", "delete[] cumprod_1mp_clamp;")
    path.write_text(text)
PY
CFLAGS="${CFLAGS:-} -Wno-cpp -Wno-sign-compare -Wno-unknown-pragmas -Wno-mismatched-new-delete"
CXXFLAGS="${CXXFLAGS:-} -Wno-cpp -Wno-sign-compare -Wno-unknown-pragmas -Wno-mismatched-new-delete"
export CFLAGS CXXFLAGS
BUILD_LOG=/tmp/graphormer_fairseq_build_ext.log
if python setup.py build_ext --inplace > "$BUILD_LOG" 2>&1; then
  echo "Fairseq extensions built. Build log: $BUILD_LOG"
else
  echo "Fairseq extension build failed. Last 160 log lines:"
  tail -160 "$BUILD_LOG"
  exit 1
fi

step "Validate imports"
export_cuda11_paths
python - <<'PY'
import ctypes
import os
import sys
import tempfile

import google.protobuf
import numpy
import torch
import ogb
import torch_sparse
import torch_geometric
import dgl
import scipy
import pandas
import networkx
import community
from google_drive_downloader import GoogleDriveDownloader  # noqa: F401

required_libs = [
    "libcudart.so.11.0",
    "libcusparse.so.11",
    "libcublas.so.11",
    "libcusolver.so.11",
    "libcurand.so.10",
    "libcufft.so.10",
]
for lib in required_libs:
    ctypes.CDLL(lib)

if os.environ.get("REQUIRE_CUDA", "1") == "1" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. In Colab, switch to a GPU runtime before running training.")

from torch.utils.tensorboard import SummaryWriter

tb_dir = tempfile.mkdtemp(prefix="graphormer_tb_smoke_")
writer = SummaryWriter(tb_dir)
writer.add_scalar("smoke/value", 1.0, 0)
writer.close()

graphormer_root = os.environ["GRAPHORMER_DIR"]
fairseq_root = os.path.join(graphormer_root, "fairseq")
for path in (graphormer_root, fairseq_root):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, fairseq_root)
sys.path.insert(1, graphormer_root)

import fairseq
from fairseq import options, tasks  # noqa: F401
from fairseq.logging.progress_bar import progress_bar  # noqa: F401
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset  # noqa: F401

Evaluator("ogbg-molhiv")
from torch_geometric import datasets as pyg_datasets

for required_dataset_class in ("KarateClub", "Reddit2", "QM9", "MoleculeNet"):
    if not hasattr(pyg_datasets, required_dataset_class):
        raise RuntimeError(f"torch_geometric.datasets missing {required_dataset_class}")

import graphormer
from graphormer import criterions, models, tasks as graphormer_tasks  # noqa: F401
from graphormer.data.dataset import GraphormerDataset  # noqa: F401
from graphormer.data.ogb_datasets import OGBDatasetLookupTable
from graphormer.data.pyg_datasets import PYGDatasetLookupTable
from graphormer.pretrain import PRETRAINED_MODEL_URLS

if not hasattr(PYGDatasetLookupTable, "GetPYGDataset"):
    raise RuntimeError("Graphormer PYGDatasetLookupTable did not import correctly")
if not hasattr(OGBDatasetLookupTable, "GetOGBDataset"):
    raise RuntimeError("Graphormer OGBDatasetLookupTable did not import correctly")

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available(), "cuda", torch.version.cuda)
print("numpy", numpy.__version__)
print("protobuf", google.protobuf.__version__)
print("torch_sparse", torch_sparse.__version__)
print("ogb", ogb.__version__)
print("torch_geometric", torch_geometric.__version__)
print("dgl", dgl.__version__)
print("scipy", scipy.__version__)
print("pandas", pandas.__version__)
print("networkx", networkx.__version__)
print("community module", community.__name__)
print("fairseq", fairseq.__file__)
print("graphormer", graphormer.__file__)
print("available pretrained", sorted(PRETRAINED_MODEL_URLS))

pretrained_name = os.environ["PRETRAINED_MODEL_NAME"]
if pretrained_name not in PRETRAINED_MODEL_URLS:
    raise RuntimeError(f"Unknown pretrained model: {pretrained_name}")
PY

step "Validate fairseq-train CLI"
fairseq-train --user-dir "$GRAPHORMER_DIR/graphormer" --help >/tmp/fairseq_train_help.txt
head -5 /tmp/fairseq_train_help.txt
""",
        env={
            "VENV_DIR": str(VENV_DIR),
            "GRAPHORMER_DIR": str(GRAPHORMER_DIR),
            "GRAPHORMER_COMMIT": GRAPHORMER_COMMIT,
            "PYTORCH_WHEEL_URL": PYTORCH_WHEEL_URL,
            "PYG_WHEEL_URL": PYG_WHEEL_URL,
            "DGL_WHEEL_URL": DGL_WHEEL_URL,
            "GRAPHORMER_GIT_URL": GRAPHORMER_GIT_URL,
            "PRETRAINED_MODEL_NAME": PRETRAINED_MODEL_NAME,
            "TORCH_HOME": str(TORCH_HOME),
            "RESET_LOCAL_SETUP": "1" if RESET_LOCAL_SETUP else "0",
            "REQUIRE_CUDA": "1" if REQUIRE_CUDA else "0",
            "BASH_XTRACE": "1" if BASH_XTRACE else "0",
        },
        label="setup",
    )


def ensure_pretrained_checkpoint() -> None:
    run_bash(
        r"""
set -euo pipefail
if [ "${BASH_XTRACE:-0}" = "1" ]; then
  set -x
fi
source "$VENV_DIR/bin/activate"
export DGLBACKEND=pytorch
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME="$TORCH_HOME"
mkdir -p "$TORCH_HOME/hub/checkpoints" "$PRETRAINED_CACHE_DRIVE"

CUDA_LIB_PATHS=$(python - <<'PY'
import glob
import os
import site
paths = []
for site_dir in site.getsitepackages():
    for pattern in (os.path.join(site_dir, "torch", "lib"), os.path.join(site_dir, "nvidia", "*", "lib")):
        for path in glob.glob(pattern):
            if os.path.isdir(path) and path not in paths:
                paths.append(path)
print(":".join(paths))
PY
)
if [ -n "$CUDA_LIB_PATHS" ]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python - <<'PY'
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import torch
from torch.hub import download_url_to_file

graphormer_root = os.environ["GRAPHORMER_DIR"]
fairseq_root = os.path.join(graphormer_root, "fairseq")
for path in (graphormer_root, fairseq_root):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, fairseq_root)
sys.path.insert(1, graphormer_root)

import fairseq
print("fairseq import", fairseq.__file__)

from graphormer.pretrain import PRETRAINED_MODEL_URLS, load_pretrained_model

name = os.environ["PRETRAINED_MODEL_NAME"]
if name == "none":
    print("No pretrained checkpoint requested.")
    raise SystemExit(0)
if name not in PRETRAINED_MODEL_URLS:
    raise RuntimeError(f"Unknown pretrained model: {name}")

canonical_url = PRETRAINED_MODEL_URLS[name]
canonical_filename = os.path.basename(urlparse(canonical_url).path)
official_urls = []
for raw in [os.environ.get("PRETRAINED_MODEL_URL_OVERRIDE", ""), canonical_url]:
    raw = raw.strip()
    if raw and raw not in official_urls:
        official_urls.append(raw)
if name == "pcqm4mv1_graphormer_base_for_molhiv":
    official_urls = []
    for raw in [
        os.environ.get("PRETRAINED_MODEL_URL_OVERRIDE", ""),
        os.environ.get("MOLHIV_PRETRAINED_URL", canonical_url),
        os.environ.get("MOLHIV_PRETRAINED_ALT_URLS", ""),
        canonical_url,
    ]:
        for url in raw.split(","):
            url = url.strip()
            if url and url not in official_urls:
                official_urls.append(url)
    canonical_filename = "checkpoint_base_preln_pcqm4mv1_for_hiv.pt"
filename = canonical_filename
cache_dir = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
cache_dir.mkdir(parents=True, exist_ok=True)
local_path = cache_dir / filename
drive_dir = Path(os.environ["PRETRAINED_CACHE_DRIVE"])
drive_dir.mkdir(parents=True, exist_ok=True)
drive_path = drive_dir / filename

manual_paths = []
for raw in [
    os.environ.get("MOLHIV_PRETRAINED_LOCAL_PATH", ""),
    os.environ.get("PRETRAINED_CHECKPOINT_LOCAL_PATH", ""),
    os.environ.get("PRETRAINED_CHECKPOINT_PATH", ""),
]:
    raw = raw.strip()
    if raw:
        manual_paths.append(Path(raw).expanduser())

try:
    n_gpu = max(1, int(os.environ.get("N_GPU", "1")))
except ValueError:
    n_gpu = 1
rank_cache_paths = [cache_dir / f"{name}_{rank}" for rank in range(n_gpu)]
rank_drive_paths = [drive_dir / f"{name}_{rank}" for rank in range(n_gpu)]

required_keys = {
    "encoder.graph_encoder.graph_node_feature.atom_encoder.weight",
    "encoder.graph_encoder.graph_node_feature.graph_token.weight",
    "encoder.graph_encoder.graph_attn_bias.spatial_pos_encoder.weight",
    "encoder.graph_encoder.layers.0.self_attn.q_proj.weight",
    "encoder.graph_encoder.layers.11.fc2.weight",
    "encoder.layer_norm.weight",
    "encoder.lm_head_transform_weight.weight",
    "encoder.embed_out.weight",
    "encoder.lm_output_learned_bias",
    "encoder.masked_lm_pooler.weight",
}


def checkpoint_state(path: Path) -> dict:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj or not isinstance(obj["model"], dict):
        raise RuntimeError(f"{path} is not a Graphormer checkpoint with a model state")
    state = obj["model"]
    missing = sorted(required_keys - set(state))
    if missing:
        raise RuntimeError(f"{path} is missing required model keys: {missing[:8]}")
    return state


def validate_official_model_load(state: dict) -> None:
    from graphormer.models.graphormer import (
        GraphormerModel,
        graphormer_base_architecture,
    )

    args = SimpleNamespace(
        pretrained_model_name=name,
        max_nodes=512,
        num_atoms=512 * 9,
        num_edges=512 * 3,
        num_in_degree=512,
        num_out_degree=512,
        num_spatial=512,
        num_edge_dis=128,
        edge_type="multi_hop",
        multi_hop_max_dist=5,
        num_classes=1,
        load_pretrained_model_output_layer=True,
        remove_head=False,
        pre_layernorm=os.environ.get("PRE_LAYERNORM_ARG", "") == "--pre-layernorm",
    )
    graphormer_base_architecture(args)
    args.pretrained_model_name = "none"
    model = GraphormerModel.build_model(args, task=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Converted pretrained checkpoint does not match official Graphormer. "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    del model


def valid_checkpoint(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        state = checkpoint_state(path)
        validate_official_model_load(state)
        return True
    except Exception as exc:
        print(f"Invalid pretrained cache at {path}: {exc}")
        return False


def copy_if_valid(src: Path, dst: Path) -> bool:
    if valid_checkpoint(src):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return True
    return False


def mirror_cache_files() -> None:
    for path in rank_cache_paths:
        shutil.copy2(local_path, path)
    shutil.copy2(local_path, drive_path)
    for path in rank_drive_paths:
        shutil.copy2(local_path, path)


def download_with_retries(url: str, dst: Path, label: str, attempts: int = 3) -> bool:
    tmp = dst.with_name(dst.name + ".tmp")
    for attempt in range(1, attempts + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            print(f"Downloading {label}, attempt {attempt}/{attempts}: {url}")
            download_url_to_file(url, str(tmp), progress=True)
            if tmp.stat().st_size < 1_000_000:
                raise RuntimeError(f"Downloaded file is too small: {tmp.stat().st_size} bytes")
            tmp.replace(dst)
            return True
        except Exception as exc:
            print(f"{label} download attempt {attempt} failed: {exc}")
            if tmp.exists():
                tmp.unlink()
            time.sleep(min(30, 5 * attempt))
    return False


def convert_hf_graphormer_checkpoint(hf_path: Path, output_path: Path, hf_url: str) -> None:
    hf_obj = torch.load(str(hf_path), map_location="cpu")
    if isinstance(hf_obj, dict) and "state_dict" in hf_obj and isinstance(hf_obj["state_dict"], dict):
        hf_state = hf_obj["state_dict"]
    elif isinstance(hf_obj, dict) and "model" in hf_obj and isinstance(hf_obj["model"], dict):
        hf_state = hf_obj["model"]
    elif isinstance(hf_obj, dict):
        hf_state = hf_obj
    else:
        raise RuntimeError(f"Unexpected HF checkpoint object type: {type(hf_obj)}")

    state = {}
    for key, value in hf_state.items():
        if key.startswith("encoder."):
            state[key] = value
        elif key == "classifier.classifier.weight":
            state["encoder.embed_out.weight"] = value
        elif key == "classifier.lm_output_learned_bias":
            state["encoder.lm_output_learned_bias"] = value

    if "encoder.masked_lm_pooler.weight" not in state:
        state["encoder.masked_lm_pooler.weight"] = state[
            "encoder.lm_head_transform_weight.weight"
        ].clone()
    if "encoder.masked_lm_pooler.bias" not in state:
        state["encoder.masked_lm_pooler.bias"] = state[
            "encoder.lm_head_transform_weight.bias"
        ].clone()

    missing = sorted(required_keys - set(state))
    if missing:
        raise RuntimeError(f"HF conversion missing required keys: {missing[:8]}")

    validate_official_model_load(state)
    tmp = output_path.with_name(output_path.name + ".converted.tmp")
    torch.save(
        {
            "model": state,
            "extra_state": {
                "source": "huggingface",
                "hf_url": hf_url,
                "pretrained_model_name": name,
                "converted_for": "official_microsoft_graphormer_fairseq",
                "note": "HF GraphormerForGraphClassification keys converted to official Fairseq GraphormerModel keys.",
            },
        },
        str(tmp),
    )
    tmp.replace(output_path)


restored = False
for candidate in [*manual_paths, drive_path, *rank_drive_paths, local_path, *rank_cache_paths]:
    if candidate.exists() and copy_if_valid(candidate, local_path):
        print(f"Restored pretrained checkpoint from {candidate}")
        restored = True
        break

if not restored:
    official_ready = False
    for official_url in official_urls:
        if download_with_retries(official_url, local_path, "official Graphormer checkpoint"):
            official_ready = valid_checkpoint(local_path)
            if official_ready:
                break
            print(f"Official checkpoint downloaded but failed validation: {local_path}")
            local_path.unlink(missing_ok=True)
    hf_fallback_urls = {
        "pcqm4mv1_graphormer_base": os.environ["HF_PCQM4MV1_PYTORCH_URL"],
        "pcqm4mv2_graphormer_base": os.environ["HF_PCQM4MV2_PYTORCH_URL"],
    }
    if not official_ready and name in hf_fallback_urls:
        hf_url = hf_fallback_urls[name]
        hf_path = cache_dir / f"hf_{name}_pytorch_model.bin"
        if not hf_path.exists() or hf_path.stat().st_size < 1_000_000:
            if not download_with_retries(hf_url, hf_path, f"HF {name} checkpoint"):
                raise RuntimeError(
                    "Could not download the official Azure checkpoint or the HF fallback. "
                    "If Colab networking is restricted, place a valid checkpoint in "
                    f"{drive_path} and rerun."
                )
        convert_hf_graphormer_checkpoint(hf_path, local_path, hf_url)
        if not valid_checkpoint(local_path):
            raise RuntimeError(f"HF-converted checkpoint failed validation: {local_path}")
    elif not official_ready:
        raise RuntimeError(
            f"Could not download official checkpoint for {name}. The MolHIV-specific "
            "pre-layernorm checkpoint is required for strict Microsoft hiv_pre.sh reproduction "
            "and no official Hugging Face mirror is known. "
            f"Place a valid checkpoint at {drive_path}, set MOLHIV_PRETRAINED_LOCAL_PATH, "
            "or set PRETRAINED_CHECKPOINT_LOCAL_PATH to a valid copy."
        )

mirror_cache_files()
loaded = load_pretrained_model(name)
if not isinstance(loaded, dict) or not loaded:
    raise RuntimeError(f"load_pretrained_model({name!r}) returned an empty state")
validate_official_model_load(loaded)
print("pretrained checkpoint ready", name, "tensors", len(loaded), "cache", local_path)
print("rank cache files", [str(path) for path in rank_cache_paths])
print("drive cache", drive_path)
PY
""",
        env=common_env(),
        label="pretrained checkpoint",
    )


def train() -> None:
    run_bash(
        r"""
set -euo pipefail
if [ "${BASH_XTRACE:-0}" = "1" ]; then
  set -x
fi
source "$VENV_DIR/bin/activate"
export DGLBACKEND=pytorch
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME="$TORCH_HOME"
CUDA_LIB_PATHS=$(python - <<'PY'
import glob
import os
import site
paths = []
for site_dir in site.getsitepackages():
    for pattern in (os.path.join(site_dir, "torch", "lib"), os.path.join(site_dir, "nvidia", "*", "lib")):
        for path in glob.glob(pattern):
            if os.path.isdir(path) and path not in paths:
                paths.append(path)
print(":".join(paths))
PY
)
if [ -n "$CUDA_LIB_PATHS" ]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$GRAPHORMER_DIR/examples/property_prediction"
mkdir -p "$LOCAL_RUN_ROOT" "$DRIVE_RUN_ROOT"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
ACCUMULATED_BATCH=$((MICRO_BATCH_SIZE * UPDATE_FREQ))
if [ "$ACCUMULATED_BATCH" -ne "$EFFECTIVE_BATCH_SIZE" ]; then
  echo "WARNING: MICRO_BATCH_SIZE * UPDATE_FREQ = $ACCUMULATED_BATCH, but EFFECTIVE_BATCH_SIZE = $EFFECTIVE_BATCH_SIZE"
  echo "This changes the paper-target effective batch unless you intentionally adjusted EFFECTIVE_BATCH_SIZE too."
fi
TOT_UPDATES=$((TRAIN_SIZE_APPROX * PAPER_EPOCHS / EFFECTIVE_BATCH_SIZE / N_GPU))
export TOT_UPDATES WARMUP_RATIO WARMUP_RATIO_PERCENT
WARMUP_UPDATES=$(python - <<'PY'
import os

total = int(os.environ["TOT_UPDATES"])
raw_ratio = os.environ.get("WARMUP_RATIO", "").strip()
if raw_ratio:
    ratio = float(raw_ratio)
else:
    percent_or_ratio = float(os.environ["WARMUP_RATIO_PERCENT"])
    ratio = percent_or_ratio / 100.0 if percent_or_ratio > 1 else percent_or_ratio
if ratio < 0:
    raise SystemExit(f"Warmup ratio must be non-negative, got {ratio}")
updates = int(total * ratio)
if ratio > 0 and updates < 1:
    updates = 1
print(updates)
PY
)
MAX_EPOCH=$((PAPER_EPOCHS + 1))
if [ "$SAVE_INTERVAL_UPDATES" -gt 0 ]; then
  EXPECTED_INTERVAL_CKPTS=$((TOT_UPDATES / SAVE_INTERVAL_UPDATES))
else
  EXPECTED_INTERVAL_CKPTS=0
fi

echo "[graphormer-colab] Training target: $CONFIG_PRESET MolHIV Graphormer-FLAG"
echo "[graphormer-colab] pretrained=$PRETRAINED_MODEL_NAME epochs=$PAPER_EPOCHS batch=$MICRO_BATCH_SIZE update_freq=$UPDATE_FREQ effective_batch=$EFFECTIVE_BATCH_SIZE"
echo "[graphormer-colab] warmup_updates=$WARMUP_UPDATES total_updates=$TOT_UPDATES lr=$PEAK_LEARNING_RATE->$END_LEARNING_RATE dropout=$DROPOUT attn_dropout=$ATTENTION_DROPOUT flag_m=$FLAG_M flag_step=$FLAG_STEP_SIZE flag_mag=$FLAG_MAG"
echo "[graphormer-colab] config_preset=$CONFIG_PRESET strict_microsoft_config=$STRICT_MICROSOFT_CONFIG exact_paper_batch=$EXACT_PAPER_BATCH force_retrain=$FORCE_RETRAIN archive_existing_run=$ARCHIVE_EXISTING_RUN save_interval_updates=$SAVE_INTERVAL_UPDATES"

PRE_LAYERNORM_ARGS=()
if [ -n "$PRE_LAYERNORM_ARG" ]; then
  PRE_LAYERNORM_ARGS=("$PRE_LAYERNORM_ARG")
fi

for SEED in "${SEED_ARRAY[@]}"; do
  if [ "$CONFIG_PRESET" = "public_hiv_pre_fallback" ] || [ "$CONFIG_PRESET" = "public_pcqm_hiv_pre" ] || [ "$CONFIG_PRESET" = "pcqm_hiv_pre" ]; then
    RUN_NAME="public_pcqm_molhiv_${PRETRAINED_MODEL_NAME}_seed${SEED}"
  else
    RUN_NAME="official_molhiv_${PRETRAINED_MODEL_NAME}_seed${SEED}"
  fi
  LOCAL_RUN_DIR="$LOCAL_RUN_ROOT/${RUN_NAME}"
  DRIVE_RUN_DIR="$DRIVE_RUN_ROOT/${RUN_NAME}"
  SAVE_DIR="$LOCAL_RUN_DIR/ckpts"
  LOG_DIR="$LOCAL_RUN_DIR/logs"
  TB_DIR="$LOCAL_RUN_DIR/tb"

  if [ "$FORCE_RETRAIN" = "1" ]; then
    ARCHIVE_STAMP="$(date -u +"%Y%m%dT%H%M%SZ")_pid$$"
    for EXISTING_RUN_DIR in "$LOCAL_RUN_DIR" "$DRIVE_RUN_DIR"; do
      if [ -d "$EXISTING_RUN_DIR" ]; then
        if [ "$ARCHIVE_EXISTING_RUN" = "1" ]; then
          ARCHIVE_ROOT="$(dirname "$EXISTING_RUN_DIR")/_archived_fresh_runs"
          ARCHIVE_DIR="$ARCHIVE_ROOT/$(basename "$EXISTING_RUN_DIR")_$ARCHIVE_STAMP"
          mkdir -p "$ARCHIVE_ROOT"
          mv "$EXISTING_RUN_DIR" "$ARCHIVE_DIR"
          echo "[graphormer-colab] Archived existing run before fresh training: $EXISTING_RUN_DIR -> $ARCHIVE_DIR"
        else
          rm -rf "$EXISTING_RUN_DIR"
          echo "[graphormer-colab] Removed existing run before fresh training: $EXISTING_RUN_DIR"
        fi
      fi
    done
  elif [ -d "$DRIVE_RUN_DIR" ]; then
    echo "Restoring existing Drive run state from $DRIVE_RUN_DIR to $LOCAL_RUN_DIR"
    mkdir -p "$LOCAL_RUN_DIR"
    rsync -a "$DRIVE_RUN_DIR/" "$LOCAL_RUN_DIR/"
  fi
  mkdir -p "$SAVE_DIR" "$LOG_DIR" "$TB_DIR"

  cat > "$LOCAL_RUN_DIR/run_config.txt" <<EOF
run_name=$RUN_NAME
target=$CONFIG_PRESET
config_preset=$CONFIG_PRESET
strict_microsoft_config=$STRICT_MICROSOFT_CONFIG
exact_paper_config=$EXACT_PAPER_CONFIG
exact_paper_batch=$EXACT_PAPER_BATCH
force_retrain=$FORCE_RETRAIN
archive_existing_run=$ARCHIVE_EXISTING_RUN
pretrained_model_name=$PRETRAINED_MODEL_NAME
seed=$SEED
paper_epochs=$PAPER_EPOCHS
effective_batch_size=$EFFECTIVE_BATCH_SIZE
micro_batch_size=$MICRO_BATCH_SIZE
update_freq=$UPDATE_FREQ
accumulated_batch_size=$ACCUMULATED_BATCH
colab_gradient_accumulation_note=micro_batch_times_update_freq_should_equal_effective_batch_size
total_num_update=$TOT_UPDATES
warmup_ratio=$WARMUP_RATIO
warmup_ratio_percent=$WARMUP_RATIO_PERCENT
warmup_updates=$WARMUP_UPDATES
peak_learning_rate=$PEAK_LEARNING_RATE
end_learning_rate=$END_LEARNING_RATE
dropout=$DROPOUT
attention_dropout=$ATTENTION_DROPOUT
act_dropout=$ACT_DROPOUT
flag_m=$FLAG_M
flag_step_size=$FLAG_STEP_SIZE
flag_mag=$FLAG_MAG
pre_layernorm_arg=$PRE_LAYERNORM_ARG
save_interval_updates=$SAVE_INTERVAL_UPDATES
expected_interval_checkpoints=$EXPECTED_INTERVAL_CKPTS
keep_interval_updates=$KEEP_INTERVAL_UPDATES
local_run_dir=$LOCAL_RUN_DIR
drive_run_dir=$DRIVE_RUN_DIR
EOF

  STEP0_CKPT="$SAVE_DIR/pretrained_step0_${PRETRAINED_MODEL_NAME}.pt"
  if [ ! -f "$STEP0_CKPT" ]; then
    echo "Saving step-0 pretrained reference checkpoint to $STEP0_CKPT"
    export STEP0_CKPT
    python - <<'PY'
import os
import sys
import time
import torch

graphormer_root = os.environ["GRAPHORMER_DIR"]
fairseq_root = os.path.join(graphormer_root, "fairseq")
for path in (graphormer_root, fairseq_root):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, fairseq_root)
sys.path.insert(1, graphormer_root)

import fairseq
print("fairseq import", fairseq.__file__)
from graphormer.pretrain import load_pretrained_model

name = os.environ["PRETRAINED_MODEL_NAME"]
path = os.environ["STEP0_CKPT"]
state = {
    "model": load_pretrained_model(name),
    "extra_state": {
        "num_updates": 0,
        "pretrained_model_name": name,
        "note": "Step-0 PCQM pretrained baseline before OGBG-MolHIV fine-tuning.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    },
}
torch.save(state, path)
print(path)
PY
  fi

  sync_run_to_drive() {
    mkdir -p "$DRIVE_RUN_DIR"
    {
      echo -e "filename\tbytes\tmodified_utc"
      find "$SAVE_DIR" -maxdepth 1 -type f -name "*.pt" -printf "%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TSZ\n" | sort
    } > "$LOCAL_RUN_DIR/checkpoint_manifest.tsv"
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$LOCAL_RUN_DIR/last_drive_sync_utc.txt"
    rsync -a --delete --delay-updates "$LOCAL_RUN_DIR/" "$DRIVE_RUN_DIR/"
    sync
  }

  SYNC_PID=""
  if [ "$SYNC_INTERVAL_SECONDS" -gt 0 ]; then
    (
      while true; do
        sync_run_to_drive || true
        sleep "$SYNC_INTERVAL_SECONDS"
      done
    ) &
    SYNC_PID=$!
  fi

  filter_fairseq_train_stdout() {
    if [ "$DISPLAY_FULL_FAIRSEQ_LOG" = "1" ]; then
      cat
    else
      python -u -c '
import sys

needles = (
    "WARNING",
    "ERROR",
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "fairseq-train failed",
    "[graphormer-colab]",
    "INFO | train_inner |",
    "INFO | valid |",
    "INFO | train |",
    "INFO | fairseq_cli.train | task:",
    "INFO | fairseq_cli.train | model:",
    "INFO | fairseq_cli.train | criterion:",
    "INFO | fairseq_cli.train | num. shared model params",
    "INFO | fairseq_cli.train | training on",
    "INFO | fairseq_cli.train | max tokens",
    "INFO | fairseq_cli.train | begin validation",
    "INFO | fairseq_cli.train | end of epoch",
    "INFO | fairseq.trainer | Preparing to load checkpoint",
    "INFO | fairseq.trainer | No existing checkpoint",
    "INFO | fairseq.trainer | loading train data",
    "INFO | fairseq.trainer | begin training epoch",
    "INFO | fairseq.trainer | Saving checkpoint",
    "INFO | fairseq.trainer | Finished saving checkpoint",
    "INFO | fairseq.checkpoint_utils | Preparing to save checkpoint",
    "INFO | fairseq.checkpoint_utils | Saved checkpoint",
    "INFO | graphormer.tasks.graph_prediction | Loaded",
)
for line in sys.stdin:
    if any(needle in line for needle in needles):
        print(line, end="")
'
    fi
  }

  run_fairseq_train() {
    CUDA_VISIBLE_DEVICES=0 fairseq-train \
    --user-dir "$GRAPHORMER_DIR/graphormer" \
    --num-workers "$NUM_WORKERS" \
    --ddp-backend=legacy_ddp \
    --dataset-name ogbg-molhiv \
    --dataset-source ogb \
    --task graph_prediction_with_flag \
    --criterion binary_logloss_with_flag \
    --arch graphormer_base \
    --num-classes 1 \
    --attention-dropout "$ATTENTION_DROPOUT" --act-dropout "$ACT_DROPOUT" --dropout "$DROPOUT" \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 \
    --clip-norm 5.0 --weight-decay 0.0 \
    --lr-scheduler polynomial_decay --power 1 \
    --warmup-updates "$WARMUP_UPDATES" --total-num-update "$TOT_UPDATES" \
    --lr "$PEAK_LEARNING_RATE" --end-learning-rate "$END_LEARNING_RATE" \
    --batch-size "$MICRO_BATCH_SIZE" --update-freq "$UPDATE_FREQ" \
    --fp16 \
    --data-buffer-size 20 \
    --encoder-layers 12 \
    --encoder-embed-dim 768 \
    --encoder-ffn-embed-dim 768 \
    --encoder-attention-heads 32 \
    --max-epoch "$MAX_EPOCH" \
    --save-dir "$SAVE_DIR" \
    --save-interval 1 \
    --save-interval-updates "$SAVE_INTERVAL_UPDATES" \
    --keep-interval-updates "$KEEP_INTERVAL_UPDATES" \
    --pretrained-model-name "$PRETRAINED_MODEL_NAME" \
    --seed "$SEED" \
    --flag-m "$FLAG_M" \
    --flag-step-size "$FLAG_STEP_SIZE" \
    --flag-mag "$FLAG_MAG" \
    "${PRE_LAYERNORM_ARGS[@]}" \
    --tensorboard-logdir "$TB_DIR" \
    --log-format simple --log-interval 100 \
    --log-file "$LOG_DIR/train.log"
  }

  set +e
  if [ "$DISPLAY_FULL_FAIRSEQ_LOG" = "1" ]; then
    run_fairseq_train 2>&1 | tee "$LOG_DIR/train.stdout.log"
    TRAIN_RC=${PIPESTATUS[0]}
  else
    run_fairseq_train 2>&1 | tee "$LOG_DIR/train.full_stdout.log" | filter_fairseq_train_stdout | tee "$LOG_DIR/train.stdout.log"
    TRAIN_RC=${PIPESTATUS[0]}
  fi
  set -e

  if [ -n "$SYNC_PID" ]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  sync_run_to_drive

  if [ "$TRAIN_RC" -ne 0 ]; then
    echo "fairseq-train failed with exit code $TRAIN_RC"
    exit "$TRAIN_RC"
  fi
done
""",
        env=common_env(),
        label="train",
    )


def evaluate() -> None:
    run_bash(
        r"""
set -euo pipefail
if [ "${BASH_XTRACE:-0}" = "1" ]; then
  set -x
fi
source "$VENV_DIR/bin/activate"
export DGLBACKEND=pytorch
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME="$TORCH_HOME"
CUDA_LIB_PATHS=$(python - <<'PY'
import glob
import os
import site
paths = []
for site_dir in site.getsitepackages():
    for pattern in (os.path.join(site_dir, "torch", "lib"), os.path.join(site_dir, "nvidia", "*", "lib")):
        for path in glob.glob(pattern):
            if os.path.isdir(path) and path not in paths:
                paths.append(path)
print(":".join(paths))
PY
)
if [ -n "$CUDA_LIB_PATHS" ]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
PRE_LAYERNORM_ARGS=()
if [ -n "$PRE_LAYERNORM_ARG" ]; then
  PRE_LAYERNORM_ARGS=("$PRE_LAYERNORM_ARG")
fi

for SEED in "${SEED_ARRAY[@]}"; do
  if [ "$CONFIG_PRESET" = "public_hiv_pre_fallback" ] || [ "$CONFIG_PRESET" = "public_pcqm_hiv_pre" ] || [ "$CONFIG_PRESET" = "pcqm_hiv_pre" ]; then
    RUN_NAME="public_pcqm_molhiv_${PRETRAINED_MODEL_NAME}_seed${SEED}"
  else
    RUN_NAME="official_molhiv_${PRETRAINED_MODEL_NAME}_seed${SEED}"
  fi
  LOCAL_RUN_DIR="$LOCAL_RUN_ROOT/${RUN_NAME}"
  DRIVE_RUN_DIR="$DRIVE_RUN_ROOT/${RUN_NAME}"
  SAVE_DIR="$LOCAL_RUN_DIR/ckpts"
  LOG_DIR="$LOCAL_RUN_DIR/logs"

  if [ -d "$DRIVE_RUN_DIR" ]; then
    echo "Restoring Drive run state from $DRIVE_RUN_DIR to $LOCAL_RUN_DIR for evaluation"
    mkdir -p "$LOCAL_RUN_DIR"
    rsync -a "$DRIVE_RUN_DIR/" "$LOCAL_RUN_DIR/"
  fi
  mkdir -p "$LOG_DIR"

  for SPLIT in valid test; do
    CUDA_VISIBLE_DEVICES=0 python "$GRAPHORMER_DIR/graphormer/evaluate/evaluate.py" \
      --user-dir "$GRAPHORMER_DIR/graphormer" \
      --num-workers "$NUM_WORKERS" \
      --ddp-backend=legacy_ddp \
      --dataset-name ogbg-molhiv \
      --dataset-source ogb \
      --task graph_prediction \
      --arch graphormer_base \
      --num-classes 1 \
      --batch-size "$EVAL_BATCH_SIZE" \
      --save-dir "$SAVE_DIR" \
      --split "$SPLIT" \
      --metric auc \
      --seed "$SEED" \
      "${PRE_LAYERNORM_ARGS[@]}" \
      --log-format simple --log-interval 100 2>&1 | tee "$LOG_DIR/eval_${SPLIT}.log"
  done

  export SAVE_DIR LOG_DIR LOCAL_RUN_DIR
  python - <<'PY'
import os
import re
import shutil
from pathlib import Path

save_dir = Path(os.environ["SAVE_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
run_dir = Path(os.environ["LOCAL_RUN_DIR"])


def parse_eval_log(split: str):
    path = log_dir / f"eval_{split}.log"
    pairs = []
    current = None
    if not path.is_file():
        return pairs
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"evaluating checkpoint file (.*)", line)
        if m:
            current = m.group(1).strip()
            continue
        m = re.search(r"\bauc: ([0-9.eE+-]+)", line)
        if m and current:
            pairs.append((current, float(m.group(1))))
            current = None
    return pairs


valid_pairs = parse_eval_log("valid")
test_pairs = parse_eval_log("test")
rows = []
for split, pairs in (("valid", valid_pairs), ("test", test_pairs)):
    for checkpoint, auc in pairs:
        rows.append((split, checkpoint, Path(checkpoint).name, auc))

summary = run_dir / "checkpoint_auc_summary.tsv"
with summary.open("w") as handle:
    handle.write("split\tcheckpoint\tfilename\tauc\n")
    for split, checkpoint, filename, auc in rows:
        handle.write(f"{split}\t{checkpoint}\t{filename}\t{auc:.12g}\n")

if not valid_pairs:
    raise SystemExit("No validation AUC values found in eval_valid.log")

best_valid_checkpoint, best_valid_auc = max(valid_pairs, key=lambda item: item[1])
best_src = Path(best_valid_checkpoint)
if not best_src.is_file():
    raise SystemExit(f"Best validation checkpoint does not exist: {best_src}")

best_copy = save_dir / "checkpoint_best_valid_auc.pt"
if best_src.resolve() != best_copy.resolve():
    shutil.copy2(best_src, best_copy)

test_by_name = {Path(path).name: auc for path, auc in test_pairs}
best_test_auc = test_by_name.get(best_src.name)

(run_dir / "best_valid_auc_checkpoint.txt").write_text(
    f"{best_src}\n"
    f"best_valid_auc={best_valid_auc:.12g}\n"
    f"best_valid_auc_copy={best_copy}\n"
    + (f"matching_test_auc={best_test_auc:.12g}\n" if best_test_auc is not None else ""),
)

print("[graphormer-colab] Best validation AUC checkpoint:", best_src)
print("[graphormer-colab] Best validation AUC:", f"{best_valid_auc:.6f}")
if best_test_auc is not None:
    print("[graphormer-colab] Matching test AUC:", f"{best_test_auc:.6f}")
print("[graphormer-colab] Copied best-by-valid-AUC checkpoint to:", best_copy)
print("[graphormer-colab] Wrote AUC summary:", summary)
PY

  BEST_SINGLE_EVAL_DIR="$LOCAL_RUN_DIR/best_valid_auc_eval_ckpt"
  rm -rf "$BEST_SINGLE_EVAL_DIR"
  mkdir -p "$BEST_SINGLE_EVAL_DIR"
  cp "$SAVE_DIR/checkpoint_best_valid_auc.pt" "$BEST_SINGLE_EVAL_DIR/checkpoint_best_valid_auc.pt"

  for SPLIT in train valid test; do
    CUDA_VISIBLE_DEVICES=0 python "$GRAPHORMER_DIR/graphormer/evaluate/evaluate.py" \
      --user-dir "$GRAPHORMER_DIR/graphormer" \
      --num-workers "$NUM_WORKERS" \
      --ddp-backend=legacy_ddp \
      --dataset-name ogbg-molhiv \
      --dataset-source ogb \
      --task graph_prediction \
      --arch graphormer_base \
      --num-classes 1 \
      --batch-size "$EVAL_BATCH_SIZE" \
      --save-dir "$BEST_SINGLE_EVAL_DIR" \
      --split "$SPLIT" \
      --metric auc \
      --seed "$SEED" \
      "${PRE_LAYERNORM_ARGS[@]}" \
      --log-format simple --log-interval 100 2>&1 | tee "$LOG_DIR/eval_best_valid_auc_${SPLIT}.log"
  done

  export LOG_DIR LOCAL_RUN_DIR
  python - <<'PY'
import os
import re
from pathlib import Path

log_dir = Path(os.environ["LOG_DIR"])
run_dir = Path(os.environ["LOCAL_RUN_DIR"])
rows = []
for split in ("train", "valid", "test"):
    path = log_dir / f"eval_best_valid_auc_{split}.log"
    auc = None
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            match = re.search(r"\bauc: ([0-9.eE+-]+)", line)
            if match:
                auc = float(match.group(1))
    if auc is None:
        raise SystemExit(f"No AUC found in {path}")
    rows.append((split, auc))

summary = run_dir / "best_valid_auc_train_valid_test.tsv"
with summary.open("w") as handle:
    handle.write("split\tauc\n")
    for split, auc in rows:
        handle.write(f"{split}\t{auc:.12g}\n")

valid_auc = dict(rows)["valid"]
test_auc = dict(rows)["test"]
if valid_auc >= 0.78 and test_auc >= 0.78:
    interpretation = "strong_for_specialisation_analysis"
elif valid_auc >= 0.72 and test_auc >= 0.72:
    interpretation = "borderline_consider_pcqm4mv1_backup_or_more_epochs"
else:
    interpretation = "underperforming_stop_and_inspect_checkpoint_loading"
(run_dir / "best_valid_auc_interpretation.txt").write_text(
    f"valid_auc={valid_auc:.12g}\n"
    f"test_auc={test_auc:.12g}\n"
    f"interpretation={interpretation}\n"
)

print("[graphormer-colab] Best-valid checkpoint train/valid/test AUC:")
for split, auc in rows:
    print(f"[graphormer-colab]   {split}: {auc:.6f}")
print("[graphormer-colab] Interpretation:", interpretation)
print("[graphormer-colab] Wrote best checkpoint split summary:", summary)
PY

  mkdir -p "$DRIVE_RUN_DIR"
  {
    echo -e "filename\tbytes\tmodified_utc"
    find "$SAVE_DIR" -maxdepth 1 -type f -name "*.pt" -printf "%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TSZ\n" | sort
  } > "$LOCAL_RUN_DIR/checkpoint_manifest.tsv"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$LOCAL_RUN_DIR/last_drive_sync_utc.txt"
  rsync -a --delete --delay-updates "$LOCAL_RUN_DIR/" "$DRIVE_RUN_DIR/"
  sync
done
""",
        env=common_env(),
        label="evaluate",
    )


def list_outputs() -> None:
    run_bash(
        r"""
set -euo pipefail
echo "Drive outputs:"
if [ -d "$DRIVE_RUN_ROOT" ]; then
  find "$DRIVE_RUN_ROOT" -maxdepth 4 -type f | sort | sed -n '1,200p'
else
  echo "No Drive output directory yet: $DRIVE_RUN_ROOT"
fi
echo "Local outputs:"
if [ -d "$LOCAL_RUN_ROOT" ]; then
  find "$LOCAL_RUN_ROOT" -maxdepth 4 -type f | sort | sed -n '1,200p'
else
  echo "No local output directory yet: $LOCAL_RUN_ROOT"
fi
""",
        env={"DRIVE_RUN_ROOT": str(DRIVE_RUN_ROOT), "LOCAL_RUN_ROOT": str(LOCAL_RUN_ROOT)},
        label="list outputs",
    )


def common_env() -> dict[str, str]:
    return {
        "VENV_DIR": str(VENV_DIR),
        "GRAPHORMER_DIR": str(GRAPHORMER_DIR),
        "LOCAL_RUN_ROOT": str(LOCAL_RUN_ROOT),
        "DRIVE_RUN_ROOT": str(DRIVE_RUN_ROOT),
        "TORCH_HOME": str(TORCH_HOME),
        "PRETRAINED_CACHE_DRIVE": str(PRETRAINED_CACHE_DRIVE),
        "HF_PCQM4MV1_PYTORCH_URL": HF_PCQM4MV1_PYTORCH_URL,
        "HF_PCQM4MV2_PYTORCH_URL": HF_PCQM4MV2_PYTORCH_URL,
        "MOLHIV_PRETRAINED_URL": MOLHIV_PRETRAINED_URL,
        "MOLHIV_PRETRAINED_ALT_URLS": MOLHIV_PRETRAINED_ALT_URLS,
        "MOLHIV_PRETRAINED_LOCAL_PATH": MOLHIV_PRETRAINED_LOCAL_PATH,
        "PRETRAINED_MODEL_URL_OVERRIDE": PRETRAINED_MODEL_URL_OVERRIDE,
        "PRETRAINED_CHECKPOINT_LOCAL_PATH": os.environ.get(
            "PRETRAINED_CHECKPOINT_LOCAL_PATH", ""
        ),
        "PRETRAINED_CHECKPOINT_PATH": os.environ.get("PRETRAINED_CHECKPOINT_PATH", ""),
        "CONFIG_PRESET": CONFIG_PRESET,
        "STRICT_MICROSOFT_CONFIG": "1" if STRICT_MICROSOFT_CONFIG else "0",
        "EXACT_PAPER_CONFIG": "1" if EXACT_PAPER_CONFIG else "0",
        "EXACT_PAPER_BATCH": "1" if EXACT_PAPER_BATCH else "0",
        "FORCE_RETRAIN": "1" if FORCE_RETRAIN else "0",
        "ARCHIVE_EXISTING_RUN": "1" if ARCHIVE_EXISTING_RUN else "0",
        "DISPLAY_FULL_FAIRSEQ_LOG": "1" if DISPLAY_FULL_FAIRSEQ_LOG else "0",
        "BASH_XTRACE": "1" if BASH_XTRACE else "0",
        "SEEDS": SEEDS,
        "N_GPU": N_GPU,
        "PAPER_EPOCHS": PAPER_EPOCHS,
        "TRAIN_SIZE_APPROX": TRAIN_SIZE_APPROX,
        "EFFECTIVE_BATCH_SIZE": EFFECTIVE_BATCH_SIZE,
        "MICRO_BATCH_SIZE": MICRO_BATCH_SIZE,
        "UPDATE_FREQ": UPDATE_FREQ,
        "NUM_WORKERS": NUM_WORKERS,
        "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE,
        "WARMUP_RATIO": WARMUP_RATIO,
        "WARMUP_RATIO_PERCENT": WARMUP_RATIO_PERCENT,
        "PEAK_LEARNING_RATE": PEAK_LEARNING_RATE,
        "END_LEARNING_RATE": END_LEARNING_RATE,
        "ATTENTION_DROPOUT": ATTENTION_DROPOUT,
        "ACT_DROPOUT": ACT_DROPOUT,
        "DROPOUT": DROPOUT,
        "FLAG_M": FLAG_M,
        "FLAG_STEP_SIZE": FLAG_STEP_SIZE,
        "FLAG_MAG": FLAG_MAG,
        "SYNC_INTERVAL_SECONDS": SYNC_INTERVAL_SECONDS,
        "SAVE_INTERVAL_UPDATES": SAVE_INTERVAL_UPDATES,
        "KEEP_INTERVAL_UPDATES": KEEP_INTERVAL_UPDATES,
        "PRETRAINED_MODEL_NAME": PRETRAINED_MODEL_NAME,
        "PRE_LAYERNORM_ARG": PRE_LAYERNORM_ARG,
    }


def main(overrides: dict[str, object] | None = None) -> None:
    apply_main_overrides(overrides)
    mount_drive()
    show_runtime()
    validate_config_alignment()
    preflight_pretrained_source()
    if RUN_ANALYSIS_PCQM_SMOKE_TEST:
        analysis_pcqm_smoke_test()
    if RUN_SETUP:
        setup()
    if RUN_TRAIN or (RUN_SETUP and VALIDATE_PRETRAINED_LOAD):
        ensure_pretrained_checkpoint()
    if RUN_TRAIN:
        train()
    if RUN_EVAL:
        evaluate()
    list_outputs()


if __name__ == "__main__":
    main()
