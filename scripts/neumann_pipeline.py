"""Pipeline: compare Neumann sparsification vs MBBr and EffR."""

import time
import numpy as np
from scipy import sparse

from graph_sparsification import (
    configuration_model, metric_backbone, metric_backbone_rescaled,
    effective_resistance_sparsify, to_proximity, calibrate_beta, sir_monte_carlo,
)
from graph_sparsification.neumann_sparsifier import neumann_sparsify

N = 500
N_SIR = 200
SEED = 42

CONFIGS = [
    {
        'name': 'Unif(1,50) + Exp(1)',
        'degree_sampler': lambda n, rng: rng.integers(1, 51, size=n),
        'weight_sampler': lambda m, rng: rng.exponential(1.0, size=m),
    },
    {
        'name': 'Exp(30) + LogN(2,1)',
        'degree_sampler': lambda n, rng: np.clip(rng.exponential(30, size=n), 1, None).astype(int),
        'weight_sampler': lambda m, rng: rng.lognormal(2.0, 1.0, size=m),
    },
    {
        'name': 'Unif(1,100) + Exp(1/30)',
        'degree_sampler': lambda n, rng: rng.integers(1, 101, size=n),
        'weight_sampler': lambda m, rng: rng.exponential(1.0 / 30, size=m),
    },
    {
        'name': 'Pareto(2.5,20) + Exp(30)',
        'degree_sampler': lambda n, rng: np.clip((rng.pareto(2.5, size=n) + 1) * 20, 1, None).astype(int),
        'weight_sampler': lambda m, rng: rng.exponential(30.0, size=m),
    },
    # Additional test configs
    {
        'name': 'Unif(10,80) + Gamma(2,5)',
        'degree_sampler': lambda n, rng: rng.integers(10, 81, size=n),
        'weight_sampler': lambda m, rng: rng.gamma(2.0, 5.0, size=m),
    },
    {
        'name': 'Poisson(40) + Exp(10)',
        'degree_sampler': lambda n, rng: np.clip(rng.poisson(40, size=n), 1, None),
        'weight_sampler': lambda m, rng: rng.exponential(10.0, size=m),
    },
    {
        'name': 'Pareto(3,10) + LogN(1,0.5)',
        'degree_sampler': lambda n, rng: np.clip((rng.pareto(3, size=n) + 1) * 10, 1, None).astype(int),
        'weight_sampler': lambda m, rng: rng.lognormal(1.0, 0.5, size=m),
    },
    {
        'name': 'Unif(5,30) + Weibull(2)*10',
        'degree_sampler': lambda n, rng: rng.integers(5, 31, size=n),
        'weight_sampler': lambda m, rng: rng.weibull(2.0, size=m) * 10,
    },
]


def compute_mse(p_orig, p_sparse):
    mask = np.isfinite(p_orig) & np.isfinite(p_sparse)
    return np.mean((p_orig[mask] - p_sparse[mask]) ** 2)


def run_config(config, graph_seed=SEED):
    print(f"\n{'='*70}")
    print(f"Config: {config['name']}  (n={N})")
    print(f"{'='*70}")

    rng = np.random.default_rng(graph_seed)
    t0 = time.time()

    W_dist = configuration_model(N, config['degree_sampler'],
                                 config['weight_sampler'], rng=rng)
    W_prox = to_proximity(W_dist)
    n_orig = sparse.triu(W_dist, k=1).nnz
    print(f"Graph: {N} nodes, {n_orig} edges (triu)")

    W_mbb_dist = metric_backbone(W_dist)
    n_mbb = sparse.triu(W_mbb_dist, k=1).nnz
    W_mbbr = metric_backbone_rescaled(W_dist)
    print(f"MBB edges: {n_mbb}  ({100*n_mbb/n_orig:.1f}%)")

    W_effr = effective_resistance_sparsify(W_prox, n_edges=n_mbb,
                                           rng=np.random.default_rng(SEED))

    W_neumann = neumann_sparsify(W_dist, n_mbb, seed=SEED, verbose=True)
    n_neur = sparse.triu(W_neumann, k=1).nnz
    print(f"Neumann edges: {n_neur}")

    degrees = np.array(W_prox.sum(axis=1)).ravel()
    initial = [int(np.argmax(degrees))]
    beta, info = calibrate_beta(W_prox, gamma=1.0, target_range=(0.6, 0.7),
                                n_calibration_runs=30,
                                rng=np.random.default_rng(SEED), verbose=False)
    print(f"Beta={beta:.4f}, mean_inf={info['mean_infection']:.3f}")

    print(f"Running SIR ({N_SIR} runs)...")
    sir_orig = sir_monte_carlo(W_prox, beta, 1.0, initial, n_runs=N_SIR,
                               rng=np.random.default_rng(100))
    sir_mbbr = sir_monte_carlo(W_mbbr, beta, 1.0, initial, n_runs=N_SIR,
                               rng=np.random.default_rng(100))
    sir_effr = sir_monte_carlo(W_effr, beta, 1.0, initial, n_runs=N_SIR,
                               rng=np.random.default_rng(100))
    sir_neur = sir_monte_carlo(W_neumann, beta, 1.0, initial, n_runs=N_SIR,
                               rng=np.random.default_rng(100))

    p_orig = sir_orig['infection_prob']
    mse_mbbr = compute_mse(p_orig, sir_mbbr['infection_prob'])
    mse_effr = compute_mse(p_orig, sir_effr['infection_prob'])
    mse_neur = compute_mse(p_orig, sir_neur['infection_prob'])

    elapsed = time.time() - t0
    print(f"\nResults ({elapsed:.0f}s):")
    print(f"  MBBr    MSE: {mse_mbbr:.6f}")
    print(f"  EffR    MSE: {mse_effr:.6f}")
    print(f"  Neumann MSE: {mse_neur:.6f}")

    best_baseline = min(mse_mbbr, mse_effr)
    status = 'WIN' if mse_neur <= best_baseline else f'{mse_neur/best_baseline:.2f}x'
    print(f"  Status: {status}")

    return {
        'config': config['name'],
        'n_orig': n_orig, 'n_mbb': n_mbb,
        'mse_mbbr': mse_mbbr, 'mse_effr': mse_effr, 'mse_neur': mse_neur,
    }


if __name__ == '__main__':
    results = []
    for config in CONFIGS:
        res = run_config(config)
        results.append(res)

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'MBBr':>10} {'EffR':>10} {'Neumann':>10} {'Status':>10}")
    print("-" * 75)
    n_wins = 0
    for r in results:
        best = min(r['mse_mbbr'], r['mse_effr'])
        status = 'WIN' if r['mse_neur'] <= best else f"{r['mse_neur']/best:.2f}x"
        if r['mse_neur'] <= best:
            n_wins += 1
        print(f"{r['config']:<30} {r['mse_mbbr']:>10.6f} {r['mse_effr']:>10.6f} "
              f"{r['mse_neur']:>10.6f} {status:>10}")
    print(f"\nWins: {n_wins}/{len(results)}")
