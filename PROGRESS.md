# Progress Log

## 2026-03-24: Initial Implementation

### Completed
- **Repo structure**: `src/python/`, `src/cpp/`, `tests/`, `notebooks/`, `docs/`
- **Graph generators** (`generators.py`):
  - `configuration_model()`: Weighted configuration model with configurable degree sequence and weight distributions (exponential, uniform, lognormal)
  - `wsbm()` / `wsbm_fast()`: Weighted Stochastic Block Model — vectorized version for large graphs
- **Sparsifiers** (`sparsifiers.py`):
  - `metric_backbone()`: Computes MBB via APSP (scipy shortest_path), keeps only metric edges
  - `effective_resistance_sparsify()`: Spielman-Srivastava algorithm — computes exact effective resistances via Laplacian pseudoinverse, samples edges proportional to w_e * R_e, reweights to preserve Laplacian in expectation
- **SIR simulation** (`sir.py`):
  - `sir_simulation()`: Continuous-time Gillespie algorithm with heap-based priority queue
  - `sir_monte_carlo()`: Monte Carlo wrapper computing per-node infection probabilities
  - **C++ backend** (`src/cpp/sir.cpp`): pybind11 extension for fast SIR, auto-selected when available
- **Visualization** (`visualization.py`):
  - `plot_adjacency_comparison()`: Side-by-side adjacency matrices with community reordering
  - `plot_infection_comparison()` / `plot_multi_infection_comparison()`: Scatter plots of per-node infection probabilities (original vs sparsified) with R² annotation
- **Notebook** (`notebooks/experiments.ipynb`): 4 experiments — Configuration Model, wSBM k=3, Planted Partition k=2, Dense Configuration Model. All verified running end-to-end.
- **Tests**: 13 tests covering all modules, all passing

## 2026-03-24: Improvements

### Completed
- **Beta calibration** (`calibrate_beta()` in `sir.py`): bisection search to find beta that produces mean infection probability ~0.5 with good spread across nodes. Avoids degenerate regimes.
- **Notebook path fix**: added `sys.path` fallback so notebook works without `pip install -e .`
- **Notebook uses calibration**: each experiment auto-calibrates beta instead of hardcoded values
- **C++ build verified**: clean build from scratch works; documented `CXX=g++` workaround for compiler detection issues
- **README**: setup instructions, C++ compilation guide, quick start with code examples, algorithm summary table
- **Tests**: 15 tests (added calibrate_beta tests), all passing

### Decisions
- Used exact Laplacian pseudoinverse for EffR (sufficient for n < 2000 research-scale graphs; random projection path available for larger)
- Weights represent distances/costs (not similarities) throughout, consistent with MBB paper conventions
- SIR uses Gillespie algorithm matching Mercier et al. — each edge transmits at rate beta * w_e
- EffR sparsifier targets same edge count as MBB for fair comparison in experiments

## 2026-03-24: Distance/Proximity Distinction & MBBr

### Completed
- **Weight conventions**: clear separation between distance weights (costs in (0,∞], used for MBB shortest paths) and proximity weights ([0,1], used for EffR and SIR)
- **Conversion functions** in `sparsifiers.py`: `proximity_to_distance(p) = 1/p - 1`, `distance_to_proximity(d) = 1/(d+1)`, plus sparse-matrix helpers `to_proximity()` and `to_distance()`
- **MBBr** (`metric_backbone_rescaled()`): computes MBB on distances, converts to proximity, rescales so total proximity sum matches the original graph
- **Notebook updated**: main loop now generates distance graphs, converts to proximity for EffR/SIR, runs all 3 sparsifiers (MBB, MBBr, EffR), 3-panel infection comparison plots, summary table with all 3 R² columns, 3-bar aggregate chart
- **Tests**: 22 tests (added TestConversions 4 tests, TestMetricBackboneRescaled 3 tests), all passing

### Decisions
- Generators still produce distance weights — this is consistent with MBB operating on costs
- MBBr preserves the MBB sparsity pattern but rescales proximity weights so the total matches the original graph — this ensures fair comparison of SIR dynamics
- EffR and SIR both operate on proximity graphs — transmission rate beta * w_e uses proximity as weight
- Previous decision "Weights represent distances/costs throughout" is **superseded**: now explicitly dual-representation with conversions
