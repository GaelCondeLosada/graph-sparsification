"""Basic tests for graph sparsification package."""

import numpy as np
from scipy import sparse
import pytest

from graph_sparsification.generators import configuration_model, wsbm, wsbm_fast
from graph_sparsification.sparsifiers import (
    metric_backbone,
    metric_backbone_rescaled,
    effective_resistance_sparsify,
    proximity_to_distance,
    distance_to_proximity,
    to_proximity,
    to_distance,
)
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


class TestConversions:
    def test_roundtrip_scalars(self):
        distances = np.array([0.0, 0.5, 1.0, 5.0, 100.0])
        proxs = distance_to_proximity(distances)
        assert np.all(proxs >= 0) and np.all(proxs <= 1)
        distances_back = proximity_to_distance(proxs)
        np.testing.assert_allclose(distances_back, distances, atol=1e-12)

    def test_roundtrip_sparse(self):
        W = sparse.random(10, 10, density=0.3, format='csr', random_state=42)
        W = W + W.T
        W.data = np.abs(W.data) + 0.1  # ensure positive
        W_prox = to_proximity(W)
        W_back = to_distance(W_prox)
        np.testing.assert_allclose(W_back.data, W.data, atol=1e-12)

    def test_proximity_bounds(self):
        distances = np.array([0.0, 1e-6, 1.0, 1e6])
        proxs = distance_to_proximity(distances)
        assert np.all(proxs > 0)
        assert np.all(proxs <= 1)

    def test_distance_monotonic(self):
        # Larger distance -> smaller proximity
        d = np.array([1.0, 2.0, 10.0])
        p = distance_to_proximity(d)
        assert np.all(np.diff(p) < 0)


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


class TestMetricBackboneRescaled:
    def test_preserves_sparsity(self):
        n = 50
        W_dist = configuration_model(n, lambda n, rng: np.full(n, 6),
                                     lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        W_mbb_dist = metric_backbone(W_dist)
        W_mbbr = metric_backbone_rescaled(W_dist)
        # Same sparsity pattern as MBB
        assert sparse.triu(W_mbbr).nnz == sparse.triu(W_mbb_dist).nnz

    def test_rescales_proximity_sum(self):
        n = 50
        W_dist = configuration_model(n, lambda n, rng: np.full(n, 6),
                                     lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        if W_dist.nnz == 0:
            return
        W_prox = to_proximity(W_dist)
        W_mbbr = metric_backbone_rescaled(W_dist)
        # Sum of proximities should match original
        np.testing.assert_allclose(W_mbbr.data.sum(), W_prox.data.sum(), rtol=1e-10)

    def test_returns_proximity_weights(self):
        n = 30
        W_dist = configuration_model(n, lambda n, rng: np.full(n, 5),
                                     lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        W_mbbr = metric_backbone_rescaled(W_dist)
        # MBBr weights are rescaled proximities — they should be positive
        if W_mbbr.nnz > 0:
            assert W_mbbr.data.min() > 0


class TestEffectiveResistance:
    def test_sparsifies(self):
        n = 30
        W = configuration_model(n, lambda n, rng: np.full(n, 5),
                                lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        if W.nnz > 0:
            W_prox = to_proximity(W)
            W_sparse = effective_resistance_sparsify(W_prox, fraction=0.5, rng=42)
            assert W_sparse.shape == (n, n)
            assert sparse.triu(W_sparse).nnz <= sparse.triu(W_prox).nnz + 5

    def test_preserves_connectivity(self):
        from scipy.sparse.csgraph import connected_components
        n = 20
        W = sparse.csr_matrix(np.ones((n, n)) - np.eye(n))
        W_sparse = effective_resistance_sparsify(W, q=n * 5, rng=42)
        n_comp, _ = connected_components(W_sparse, directed=False)
        assert n_comp == 1

    def test_exact_edges_mode(self):
        n = 40
        W = configuration_model(n, lambda n, rng: np.full(n, 5),
                                lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        if W.nnz == 0:
            return
        W_prox = to_proximity(W)
        target = 20
        W_sparse = effective_resistance_sparsify(W_prox, n_edges=target)
        assert sparse.triu(W_sparse).nnz == target

    def test_exact_edges_reweights(self):
        """Reweighted edges should be >= original (upweighted to compensate)."""
        n = 40
        W = configuration_model(n, lambda n, rng: np.full(n, 6),
                                lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        if W.nnz == 0:
            return
        W_prox = to_proximity(W)
        n_orig = sparse.triu(W_prox).nnz
        target = max(n_orig // 3, 5)
        W_sparse = effective_resistance_sparsify(W_prox, n_edges=target, rng=42)
        # With fewer edges kept, reweighted values should generally be larger
        # than original (w̃ = w / (k*p) and k < m so upscaling happens)
        assert W_sparse.data.mean() > W_prox.data.mean()


class TestSIR:
    def _make_graph(self):
        W_dist = configuration_model(30, lambda n, rng: np.full(n, 4),
                                     lambda m, rng: rng.uniform(0.5, 1.5, size=m), rng=42)
        return to_proximity(W_dist)

    def test_python_backend(self):
        W = self._make_graph()
        result = sir_simulation(W, beta=0.5, gamma=1.0,
                                initial_infected=[0], rng=42, use_cpp=False)
        assert 'infected' in result
        assert result['infected'][0]
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
        assert result['infection_prob'][0] == 1.0
        assert len(result['all_arrival_times']) == 10


class TestCalibrateBeta:
    def test_converges(self):
        W_dist = configuration_model(50, lambda n, rng: np.full(n, 5),
                                     lambda m, rng: rng.exponential(1.0, size=m), rng=42)
        W_prox = to_proximity(W_dist)
        beta, info = calibrate_beta(
            W_prox, gamma=1.0, target_mean_infection=0.5, target_range=(0.2, 0.8),
            n_calibration_runs=15, rng=42, verbose=False
        )
        assert 0.0 < beta < 10.0
        assert 0.1 < info['mean_infection'] < 0.9

    def test_returns_history(self):
        W_dist = configuration_model(40, lambda n, rng: np.full(n, 5),
                                     lambda m, rng: rng.exponential(1.0, size=m), rng=0)
        W_prox = to_proximity(W_dist)
        _, info = calibrate_beta(W_prox, n_calibration_runs=10, rng=0,
                                 max_iterations=5, verbose=False)
        assert len(info['history']) > 0
        assert 'infection_prob' in info


class TestHeatKernelGD:
    def test_rescales_total_weight_and_runs(self):
        pytest.importorskip("torch")
        from graph_sparsification.heat_kernel_gd import heat_kernel_gd_sparsify

        n = 8
        rows, cols, data = [], [], []
        for i in range(n - 1):
            rows.append(i)
            cols.append(i + 1)
            data.append(0.5 + 0.1 * i)
        rows_sym = rows + cols
        cols_sym = cols + rows
        data_sym = data + data
        W = sparse.csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n, n))

        m = 5
        W_s, info = heat_kernel_gd_sparsify(
            W,
            m=m,
            t=1.0,
            n_steps=40,
            lr=0.05,
            rng_seed=0,
            return_history=True,
        )
        assert W_s.shape == (n, n)
        assert np.isfinite(info["final_loss"])
        np.testing.assert_allclose(info["sum_sparse"], info["sum_orig"], rtol=1e-5, atol=1e-8)
        assert sparse.triu(W_s, 1).nnz == m

    def test_t_from_gamma_over_beta(self):
        pytest.importorskip("torch")
        from graph_sparsification.heat_kernel_gd import heat_kernel_gd_sparsify

        n = 5
        rows, cols, data = [0, 1], [1, 0], [0.5, 0.5]
        W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        W_s, info = heat_kernel_gd_sparsify(
            W, m=1, beta=2.0, gamma=1.0, n_steps=2, rng_seed=0
        )
        assert info["t"] == 0.5
        assert W_s.shape == (n, n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
