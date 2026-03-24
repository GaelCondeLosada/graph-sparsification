"""Graph sparsification research package."""

from .generators import configuration_model, wsbm
from .sparsifiers import metric_backbone, effective_resistance_sparsify
from .sir import sir_simulation, sir_monte_carlo
from .visualization import plot_adjacency_comparison, plot_infection_comparison
