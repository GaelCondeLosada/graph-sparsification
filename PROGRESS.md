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

### Decisions
- Used exact Laplacian pseudoinverse for EffR (sufficient for n < 2000 research-scale graphs; random projection path available for larger)
- Weights represent distances/costs (not similarities) throughout, consistent with MBB paper conventions
- SIR uses Gillespie algorithm matching Mercier et al. — each edge transmits at rate beta * w_e
- EffR sparsifier targets same edge count as MBB for fair comparison in experiments
