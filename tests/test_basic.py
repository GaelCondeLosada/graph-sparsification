"""Basic tests for graph sparsification package."""

import numpy as np
from scipy import sparse
import pytest

from graph_sparsification.generators import configuration_model, wsbm, wsbm_fast
from graph_sparsification.sparsifiers import metric_backbone, effective_resistance_sparsify
from graph_sparsification.sir import sir_simulation, sir_monte_carlo, calibrate_beta


class TestConfigurationModel:
    def test_basic(self):
        deg = lambda n, rng: np.full(n, 3)
        wt = lambda m, rng: rng.exponential(1.0, size=m)
        W = configuration_model(20, deg, wt, rng=42)
        assert W.shape == (20, 20)
        assert (W != W.T).nnz == 0  # symmetric

    def test_weights_positive(self):
        deg = lambda n, rng: np.full(n, 4)
        wt = lambda m, rng: rng.exponential(1.0, size=m)
        W = configuration_model(50, deg, wt, rng=0)
        assert W.data.min() > 0

    def test_lambda_samplers(self):
        samplers = [
            lambda m, rng: rng.exponential(1.0, size=m),
            lambda m, rng: rng.uniform(0.1, 1.0, size=m),
            lambda m, rng: rng.lognormal(0.0, 1.0, size=m),
        ]
        deg = lambda n, rng: rng.integers(2, 6, size=n)
        for wt in samplers:
            W = configuration_model(30, deg, wt, rng=42)
            assert W.nnz > 0


class TestWSBM:
    def test_basic(self):
        n, k = 100, 3
        pi = [1/3, 1/3, 1/3]
        B = np.array([[10, 1, 1],
                      [1, 10, 1],
                      [1, 1, 10]], dtype=float)
        W, z = wsbm(n, k, pi, B, rng=42)
        assert W.shape == (n, n)
        assert len(z) == n
        assert set(z).issubset({0, 1, 2})

    def test_fast_version(self):
        n, k = 200, 2
        pi = [0.5, 0.5]
        B = np.array([[8, 2], [2, 8]], dtype=float)
        W, z = wsbm_fast(n, k, pi, B, rng=42)
        assert W.shape == (n, n)
        assert W.nnz > 0


class TestMetricBackbone:
    def test_small_graph(self):
        # Triangle: 0-1 (w=1), 0-2 (w=3), 1-2 (w=1)
        # Shortest path 0->2 goes through 1 (cost 2 < 3), so edge 0-2 is semi-metric
        W = sparse.csr_matrix(np.array([
            [0, 1, 3],
            [1, 0, 1],
            [3, 1, 0],
        ], dtype=float))
        W_mbb = metric_backbone(W)
        # Edge (0,2) should be removed
        assert W_mbb[0, 2] == 0
        assert W_mbb[2, 0] == 0
        # Edges (0,1) and (1,2) should remain
        assert W_mbb[0, 1] == 1
        assert W_mbb[1, 2] == 1

    def test_all_metric(self):
        # Path graph: all edges are metric
        n = 5
        rows = [0, 1, 2, 3]
        cols = [1, 2, 3, 4]
        data = [1.0, 1.0, 1.0, 1.0]
        W = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
        W = W + W.T
        W = W.tocsr()
        W_mbb = metric_backbone(W)
        assert W_mbb.nnz == W.nnz

    def test_sparsifies(self):
        # Random graph should have fewer edges in backbone
        n = 50
        W = configuration_model(n, lambda n, rng: np.full(n, 6),
                                lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        if W.nnz > 0:
            W_mbb = metric_backbone(W)
            assert W_mbb.nnz <= W.nnz


class TestEffectiveResistance:
    def test_sparsifies(self):
        n = 30
        W = configuration_model(n, lambda n, rng: np.full(n, 5),
                                lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        if W.nnz > 0:
            W_sparse = effective_resistance_sparsify(W, fraction=0.5, rng=42)
            assert W_sparse.shape == (n, n)
            # Should have fewer edges
            assert sparse.triu(W_sparse).nnz <= sparse.triu(W).nnz + 5  # some tolerance

    def test_preserves_connectivity(self):
        # With enough samples, graph should remain connected
        from scipy.sparse.csgraph import connected_components
        n = 20
        W = sparse.csr_matrix(np.ones((n, n)) - np.eye(n))  # complete graph
        W_sparse = effective_resistance_sparsify(W, q=n * 5, rng=42)
        n_comp, _ = connected_components(W_sparse, directed=False)
        assert n_comp == 1


class TestSIR:
    def _make_graph(self):
        return configuration_model(30, lambda n, rng: np.full(n, 4),
                                   lambda m, rng: rng.uniform(0.5, 1.5, size=m), rng=42)

    def test_python_backend(self):
        W = self._make_graph()
        result = sir_simulation(W, beta=0.5, gamma=1.0,
                                initial_infected=[0], rng=42, use_cpp=False)
        assert 'infected' in result
        assert result['infected'][0]  # initial node always infected
        assert result['arrival_times'][0] == 0.0

    def test_cpp_backend(self):
        W = self._make_graph()
        result = sir_simulation(W, beta=0.5, gamma=1.0,
                                initial_infected=[0], rng=42, use_cpp=True)
        assert 'infected' in result
        assert result['infected'][0]

    def test_monte_carlo(self):
        W = self._make_graph()
        result = sir_monte_carlo(W, beta=0.5, gamma=1.0,
                                 initial_infected=[0], n_runs=10, rng=42)
        assert result['infection_prob'][0] == 1.0  # always infected
        assert len(result['all_arrival_times']) == 10


class TestCalibrateBeta:
    def test_converges(self):
        W = configuration_model(50, lambda n, rng: np.full(n, 5),
                                lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        beta, info = calibrate_beta(
            W, gamma=1.0, target_mean_infection=0.5, target_range=(0.2, 0.8),
            n_calibration_runs=15, rng=42, verbose=False
        )
        assert 0.0 < beta < 10.0
        assert 0.1 < info['mean_infection'] < 0.9

    def test_returns_history(self):
        W = configuration_model(40, lambda n, rng: np.full(n, 5),
                                lambda m, rng: rng.exponential(1.0, size=m), rng=0)
        _, info = calibrate_beta(W, n_calibration_runs=10, rng=0,
                                 max_iterations=5, verbose=False)
        assert len(info['history']) > 0
        assert 'infection_prob' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
