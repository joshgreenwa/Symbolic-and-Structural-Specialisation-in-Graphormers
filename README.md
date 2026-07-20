# Symbolic and Structural Specialisation in Graphormers

Utilities and notebooks for analysing Graphormer attention with symbolic vs structural diagnostics.

## Quick Start

```bash
git clone https://github.com/joshgreenwa/Symbolic-and-Structural-Specialisation-in-Graphormers.git
cd Symbolic-and-Structural-Specialisation-in-Graphormers
pip install -r requirements.lock.txt
```

## Main Notebook

- Open [`generate_figures.ipynb`](generate_figures.ipynb) to run the figure-generation pipeline.
- Open [`generate_molhiv_figures.ipynb`](generate_molhiv_figures.ipynb) to run the same figure pipeline on a fine-tuned OGBG-MolHIV checkpoint.
- Run [`colab/graphormer_transport_specialisation_comparison.py`](colab/graphormer_transport_specialisation_comparison.py) in Colab to compare the current key-permutation scores with separate semantic/structural interventions scored by downstream per-head transport, including method-alignment and head-ablation tests.

## Package

- Core code lives in `graph_interp/`.
