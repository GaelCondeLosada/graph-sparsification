"""Quick experiment: does using proximities instead of distances for the
Neumann weight transform fix the catastrophic failure on LogLogN(2,0.8)?

The hypothesis: proximity p = 1/(d+1) maps any distance to [0,1], so the
exp(-p/tau) transform won't collapse when the distance distribution has
infinite mean. The tau calibration based on mean proximity should be stable.
"""
import sys, os, time
import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'python'))

from graph_sparsification.generators import configuration_model
from graph_sparsification.sparsifiers import (
    metric_backbone, metric_backbone_rescaled,
    effective_resistance_sparsify, to_proximity,
)
from graph_sparsification.sir import sir_monte_carlo, calibrate_beta
from graph_sparsification.neumann_sparsifier import (
    neumann_sparsify, _build_symmetric_prox, _global_rescale,
    _partial_sinkhorn, _sinkhorn_rescale,
)


def _calibrate_from_prox(prox, edge_i, edge_j, n, max_row_sum,
                          target_mean=0.3, n_samples=5, seed=0):
    """Calibrate Neumann weights directly from proximities.

    Instead of exp(-dist/tau), use prox directly as Neumann weights,
    scaled so the max row sum is bounded. Since prox is already in [0,1],
    this avoids the infinite-mean distance problem entirely.
    """
    # Use proximity values directly as Neumann weights (already in [0,1])
    a = prox.copy()

    # Scale down if max row sum too large
    rng = np.random.default_rng(seed)
    max_rs = 0.0
    for _ in range(n_samples):
        perm = rng.permutation(n)
        rows = np.minimum(perm[edge_i], perm[edge_j])
        row_sums = np.zeros(n)
        np.add.at(row_sums, rows, a)
        max_rs = max(max_rs, row_sums.max())

    if max_rs > max_row_sum:
        a *= max_row_sum / max_rs
    return a


def _compute_importance_prox(edge_i, edge_j, a_weights, n, n_perms=50, seed=42):
    """Compute Neumann importance, same as original but takes any weights."""
    n_edges = len(a_weights)
    importance = np.zeros(n_edges)
    structural = np.zeros(n_edges)
    rng = np.random.default_rng(seed)
    I_n = np.eye(n)

    for _ in range(n_perms):
        perm = rng.permutation(n)
        pi_i, pi_j = perm[edge_i], perm[edge_j]
        rows = np.minimum(pi_i, pi_j)
        cols = np.maximum(pi_i, pi_j)

        A = np.zeros((n, n))
        A[rows, cols] = a_weights
        S = solve_triangular(I_n - A, I_n, lower=False, unit_diagonal=True)

        col_norms_sq = (S * S).sum(axis=0)
        row_norms_sq = (S * S).sum(axis=1)
        path_score = col_norms_sq[rows] * row_norms_sq[cols]
        importance += a_weights ** 2 * path_score
        structural += path_score

    return importance / n_perms, structural / n_perms


def neumann_sparsify_prox(W_dist, n_target_edges, max_row_sum=3.0,
                           n_perms=50, seed=42, verbose=True):
    """Neumann sparsification using proximity-based weights."""
    n = W_dist.shape[0]
    W_triu = sparse.triu(W_dist, k=1).tocoo()
    edge_i = W_triu.row.copy()
    edge_j = W_triu.col.copy()
    dists = W_triu.data.copy()
    n_edges = len(dists)
    orig_prox = 1.0 / (dists + 1.0)

    if n_target_edges >= n_edges:
        return _build_symmetric_prox(edge_i, edge_j, orig_prox, n)

    retention = n_target_edges / n_edges

    # Key difference: use proximities directly as Neumann weights
    a = _calibrate_from_prox(orig_prox, edge_i, edge_j, n, max_row_sum, seed=seed)

    if verbose:
        print(f"  Proximity-Neumann: a range [{a.min():.4f}, {a.max():.4f}], "
              f"mean={a.mean():.4f}")

    full_imp, structural_imp = _compute_importance_prox(
        edge_i, edge_j, a, n, n_perms=n_perms, seed=seed)

    # Same adaptive scoring as original
    if retention > 0.25:
        s_max = structural_imp.max()
        struct_norm = structural_imp / s_max if s_max > 0 else np.ones(n_edges)
        score = struct_norm * orig_prox
        mode = 'high-retention'
    else:
        score = full_imp
        mode = 'low-retention'

    top_k = np.argsort(score)[-n_target_edges:]
    sel_i = edge_i[top_k]
    sel_j = edge_j[top_k]
    sel_prox = orig_prox[top_k]

    if verbose:
        print(f"  Selected {n_target_edges}/{n_edges} ({mode}, ret={100*retention:.1f}%)")

    # Same rescaling as original
    W_orig_prox = _build_symmetric_prox(edge_i, edge_j, orig_prox, n)
    target_rs = np.array(W_orig_prox.sum(axis=1)).ravel()

    if retention > 0.25:
        W_sparse = _build_symmetric_prox(sel_i, sel_j, sel_prox, n)
        W_sparse = _sinkhorn_rescale(W_sparse, target_rs, max_inflation=3.0)
    else:
        rescaled = _global_rescale(orig_prox, sel_prox)
        W_sparse = _build_symmetric_prox(sel_i, sel_j, rescaled, n)
        W_sparse = _partial_sinkhorn(W_sparse, target_rs, alpha=0.1)

    return W_sparse


def _mse(po, ps):
    m = np.isfinite(po) & np.isfinite(ps)
    return float(np.mean((po[m] - ps[m]) ** 2))


# ── Test configs ─────────────────────────────────────────────────────
configs = [
    # Catastrophic failures
    ("Unif(1,50)", "LogLogN(2,0.8)",
     lambda n, rng: rng.integers(1, 51, size=n),
     lambda m, rng: np.exp(rng.lognormal(2.0, 0.8, size=m))),
    ("Exp(60)", "LogLogN(2,0.8)",
     lambda n, rng: np.ceil(rng.exponential(60, size=n)).astype(int),
     lambda m, rng: np.exp(rng.lognormal(2.0, 0.8, size=m))),
    ("Exp(60)", "LogLogN(1.2,0.8)",
     lambda n, rng: np.ceil(rng.exponential(60, size=n)).astype(int),
     lambda m, rng: np.exp(rng.lognormal(1.2, 0.8, size=m))),
    # A config where original works well, to check we don't regress
    ("Unif(1,50)", "LogN(2,1)",
     lambda n, rng: rng.integers(1, 51, size=n),
     lambda m, rng: rng.lognormal(2.0, 1.0, size=m)),
    ("Pareto(2.5,20)", "LogLogN(1.2,0.4)",
     lambda n, rng: np.ceil((rng.pareto(2.5, size=n) + 1) * 20).astype(int),
     lambda m, rng: np.exp(rng.lognormal(1.2, 0.4, size=m))),
]


if __name__ == '__main__':
    N = 500
    N_SIR = 200
    SEED = 42

    print(f"{'Config':<35} {'MBBr':>10} {'EffR':>10} {'Orig':>10} {'Prox':>10} {'Improvement':>12}")
    print("-" * 95)

    for deg_name, wt_name, deg_fn, wt_fn in configs:
        label = f"{deg_name} | {wt_name}"
        rng = np.random.default_rng(42)

        W_dist = configuration_model(N, deg_fn, wt_fn, rng=rng)
        n_edges = sparse.triu(W_dist).nnz
        W_prox = to_proximity(W_dist)

        beta, _ = calibrate_beta(W_prox, gamma=1.0, n_calibration_runs=30,
                                  rng=rng, verbose=False)

        W_mbb = metric_backbone(W_dist)
        n_mbb = sparse.triu(W_mbb).nnz
        W_mbbr = metric_backbone_rescaled(W_dist)
        W_effr = effective_resistance_sparsify(W_prox, n_edges=n_mbb,
                                                rng=np.random.default_rng(SEED))

        # Original Neumann (distance-based)
        t0 = time.time()
        W_orig = neumann_sparsify(W_dist, n_mbb, seed=SEED, verbose=False)
        t_orig = time.time() - t0

        # Proximity-based Neumann
        t0 = time.time()
        W_prox_neur = neumann_sparsify_prox(W_dist, n_mbb, seed=SEED, verbose=True)
        t_prox = time.time() - t0

        initial = [int(np.argmax(np.array(W_prox.sum(axis=1)).ravel()))]

        sir_orig_graph = sir_monte_carlo(W_prox, beta, 1.0, initial,
                                          n_runs=N_SIR, rng=np.random.default_rng(100))
        sir_mbbr = sir_monte_carlo(W_mbbr, beta, 1.0, initial,
                                    n_runs=N_SIR, rng=np.random.default_rng(100))
        sir_effr = sir_monte_carlo(W_effr, beta, 1.0, initial,
                                    n_runs=N_SIR, rng=np.random.default_rng(100))
        sir_n_orig = sir_monte_carlo(W_orig, beta, 1.0, initial,
                                      n_runs=N_SIR, rng=np.random.default_rng(100))
        sir_n_prox = sir_monte_carlo(W_prox_neur, beta, 1.0, initial,
                                      n_runs=N_SIR, rng=np.random.default_rng(100))

        p_ref = sir_orig_graph['infection_prob']
        mse_mbbr = _mse(p_ref, sir_mbbr['infection_prob'])
        mse_effr = _mse(p_ref, sir_effr['infection_prob'])
        mse_orig = _mse(p_ref, sir_n_orig['infection_prob'])
        mse_prox = _mse(p_ref, sir_n_prox['infection_prob'])

        best = min(mse_mbbr, mse_effr)
        improve = f"{mse_orig/mse_prox:.1f}x better" if mse_prox < mse_orig else f"{mse_prox/mse_orig:.1f}x worse"

        print(f"{label:<35} {mse_mbbr:>10.6f} {mse_effr:>10.6f} "
              f"{mse_orig:>10.6f} {mse_prox:>10.6f} {improve:>12}")

    print("\nOrig = distance-based exp(-d/tau), Prox = proximity-based Neumann weights")
