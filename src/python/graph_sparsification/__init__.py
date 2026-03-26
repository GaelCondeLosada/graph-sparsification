"""Graph sparsification research package."""

from .generators import configuration_model, wsbm
from .sparsifiers import (
    metric_backbone,
    metric_backbone_rescaled,
    effective_resistance_sparsify,
    proximity_to_distance,
    distance_to_proximity,
    to_proximity,
    to_distance,
)
from .sir import sir_simulation, sir_monte_carlo, calibrate_beta
from .visualization import plot_adjacency_comparison, plot_infection_comparison
from .neumann_sparsifier import neumann_sparsify
