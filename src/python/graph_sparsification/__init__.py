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

try:
    from .heat_kernel_gd import heat_kernel_gd_sparsify
except ImportError:  # pragma: no cover
    heat_kernel_gd_sparsify = None  # type: ignore[misc, assignment]
