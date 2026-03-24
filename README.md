# Graph Sparsification

Research codebase for studying graph sparsification methods and their impact on epidemic dynamics. Implements the **Metric Backbone** (Dreveton et al., NeurIPS 2024), **Effective Resistance sparsification** (Spielman & Srivastava, 2011), and **SIR simulations** (Mercier et al., 2022) on weighted random graphs.

## Setup

### Prerequisites

- Python >= 3.9
- A C++ compiler with C++17 support (g++ or clang++)
- pip

### Installation

```bash
git clone https://github.com/GaelCondeLosada/graph-sparsification.git
cd graph-sparsification

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install in editable mode (compiles the C++ SIR module automatically)
pip install -e .
```

If the C++ compilation fails (usually a compiler detection issue), set `CXX` explicitly:

```bash
CXX=g++ pip install -e .
```

### Verify the installation

```bash
# Check the C++ SIR backend compiled
python -c "from graph_sparsification._sir_cpp import sir_simulation_cpp; print('C++ SIR: OK')"

# Check all modules load
python -c "from graph_sparsification import *; print('All imports: OK')"

# Run the test suite
pip install pytest
pytest tests/ -v
```

### Building the C++ module manually

If you need to rebuild the C++ extension without reinstalling:

```bash
CXX=g++ python setup.py build_ext --inplace
```

The compiled `.so` file is placed in `src/python/graph_sparsification/`. If the C++ module is unavailable, the package falls back to a pure Python SIR implementation automatically.

## Project structure

```
graph-sparsification/
├── src/
│   ├── python/graph_sparsification/
│   │   ├── generators.py      # Configuration model, wSBM
│   │   ├── sparsifiers.py     # Metric backbone, effective resistance
│   │   ├── sir.py             # SIR simulation + beta calibration
│   │   └── visualization.py   # Adjacency matrix + infection probability plots
│   └── cpp/
│       └── sir.cpp            # Fast SIR via pybind11
├── notebooks/
│   └── experiments.ipynb      # Main experiment notebook
├── tests/
│   └── test_basic.py
├── docs/                      # Reference papers
├── setup.py
└── pyproject.toml
```

## Weight conventions

The codebase distinguishes between two weight spaces:

| Space | Range | Used by |
|---|---|---|
| **Distance** (cost) | (0, ∞] | Graph generators, Metric Backbone (shortest paths) |
| **Proximity** | [0, 1] | Effective Resistance, SIR simulation |

Conversions (element-wise): `proximity = 1/(distance+1)` and `distance = 1/proximity - 1`.

## Quick start

### Generate a graph and sparsify it

```python
from graph_sparsification import *
import numpy as np

# Generate a distance-weighted stochastic block model
W_dist, z = wsbm(n=200, k=3,
                  pi=[1/3, 1/3, 1/3],
                  B=np.array([[10, 2, 2], [2, 10, 2], [2, 2, 10]], dtype=float),
                  rng=42)

# Metric Backbone (operates on distances)
W_mbb_dist = metric_backbone(W_dist)

# MBB Rescaled (rescales proximities to match original graph's total)
W_mbbr_prox = metric_backbone_rescaled(W_dist)

# Convert to proximity for EffR and SIR
W_prox = to_proximity(W_dist)
W_effr_prox = effective_resistance_sparsify(W_prox, fraction=0.1, rng=42)
```

### Run SIR and calibrate beta

The `calibrate_beta` function finds a transmission rate that produces a spread of infection probabilities across nodes (avoiding the degenerate cases of "nobody infected" or "everyone infected"):

```python
# Auto-calibrate beta on the proximity graph
beta, info = calibrate_beta(W_prox, gamma=1.0, target_mean_infection=0.5, rng=42)
print(f"beta={beta:.4f}, mean infection={info['mean_infection']:.2f}")

# Run Monte Carlo SIR (on proximity graph)
result = sir_monte_carlo(W_prox, beta=beta, gamma=1.0,
                         initial_infected=[0], n_runs=200, rng=42)
print(f"Infection probabilities: min={result['infection_prob'].min():.2f}, "
      f"max={result['infection_prob'].max():.2f}")
```

### Compare sparsifiers

```python
# Convert MBB to proximity for SIR comparison
W_mbb_prox = to_proximity(W_mbb_dist)

# Run SIR on original and all sparsified graphs (all proximity)
sir_orig = sir_monte_carlo(W_prox,      beta, 1.0, [0], n_runs=200, rng=42)
sir_mbb  = sir_monte_carlo(W_mbb_prox,  beta, 1.0, [0], n_runs=200, rng=42)
sir_mbbr = sir_monte_carlo(W_mbbr_prox, beta, 1.0, [0], n_runs=200, rng=42)
sir_effr = sir_monte_carlo(W_effr_prox, beta, 1.0, [0], n_runs=200, rng=42)

# Scatter plot: one point per node, 3-panel comparison
fig = plot_multi_infection_comparison(
    sir_orig['infection_prob'],
    [sir_mbb['infection_prob'], sir_mbbr['infection_prob'], sir_effr['infection_prob']],
    ['MBB', 'MBBr', 'Effective Resistance'],
)
```

### Run the full experiment notebook

```bash
jupyter notebook notebooks/experiments.ipynb
```

The notebook loops over 8 degree × 7 weight distributions on the configuration model, auto-calibrates beta for each, computes all 3 sparsifications (MBB, MBBr, EffR), runs SIR on proximity graphs, and produces comparison plots with R² summary.

## Algorithms

| Component | Method | Reference |
|---|---|---|
| **Metric Backbone (MBB)** | APSP on distance graph; keep edge (u,v) iff w(u,v) = d(u,v) | Dreveton et al., NeurIPS 2024 |
| **MBB Rescaled (MBBr)** | MBB sparsity pattern with proximity weights rescaled to match original total | — |
| **EffR Sparsification** | Sample edges proportional to w_e * R_e on proximity graph, reweight | Spielman & Srivastava, SIAM J. Comput. 2011 |
| **SIR Simulation** | Continuous-time Gillespie on proximity graph (rate = beta * proximity) | Mercier et al., PLoS Comp. Bio. 2022 |
| **Beta Calibration** | Bisection search targeting a mean infection probability of ~0.5 | — |

## Reference papers

- `docs/NeurIPS_WhyTheMetricBackbonePreservesCommunityStructure.pdf` — Dreveton et al., 2024
- `docs/mercier2022effective.pdf` — Mercier, Scarpino & Moore, 2022
- `docs/spielman and srivastava Rff.pdf` — Spielman & Srivastava, 2011
