"""Global constants, seed utility, and matplotlib defaults."""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Model constants ──────────────────────────────────────────
MODEL_ID = "clefourrier/graphormer-base-pcqm4mv2"
MODEL_REVISION = "refs/pr/4"
NUM_LAYERS = 12
NUM_HEADS = 32
MAX_DIST = 5

# ── Axis labels (LaTeX) ─────────────────────────────────────
LAYER_LABEL = r"Layer Index $\ell$"
HEAD_LABEL = r"Head Index $h$"

# ── Default computation parameters ──────────────────────────
DEFAULT_N_GRAPHS = 100
DEFAULT_NUM_PERMS = 96
DEFAULT_TAU = 0.02
DEFAULT_QUERY_SUBSAMPLE = 64
DEFAULT_D_CAP = 10
DEFAULT_POOL_SIZE = 2000

# ── Reproducibility ──────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Matplotlib defaults (called once at import) ─────────────

def apply_plot_defaults():
    plt.rcParams.update({
        "figure.dpi": 800,
        "savefig.dpi": 800,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "axes.grid": False,
    })

apply_plot_defaults()
