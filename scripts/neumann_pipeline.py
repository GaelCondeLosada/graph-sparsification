#!/usr/bin/env python3
"""Neumann sparsification evaluation pipeline.

Runs Neumann, MBBr, and EffR on a set of graph configurations and reports
comparative MSE of SIR infection probabilities. Results are saved to JSON
for downstream plotting.

Usage:
    python scripts/neumann_pipeline.py                     # 5 diverse regimes
    python scripts/neumann_pipeline.py --n-nodes 1000      # larger graphs
    python scripts/neumann_pipeline.py --configs all        # all 56 configs
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from graph_sparsification.generators import configuration_model
from graph_sparsification.sparsifiers import (
    metric_backbone,
    metric_backbone_rescaled,
    effective_resistance_sparsify,
    to_proximity,
)
from graph_sparsification.sir import sir_monte_carlo, calibrate_beta
from graph_sparsification.neumann_sparsifier import neumann_sparsify


# ── Configuration registry ─────────────────────────────────────────

DEGREE_DISTRIBUTIONS = {
    'Unif(1,50)':      lambda n, rng: rng.integers(1, 51, size=n),
    'Unif(1,100)':     lambda n, rng: rng.integers(1, 101, size=n),
    'Exp(30)':         lambda n, rng: np.ceil(rng.exponential(30, size=n)).astype(int),
    'Exp(60)':         lambda n, rng: np.ceil(rng.exponential(60, size=n)).astype(int),
    'LogN(3.26,0.66)': lambda n, rng: np.ceil(rng.lognormal(3.26, 0.66, size=n)).astype(int),
    'LogN(3.26,2)':    lambda n, rng: np.ceil(rng.lognormal(3.26, 2.0, size=n)).astype(int),
    'Pareto(2.5,20)':  lambda n, rng: np.ceil((rng.pareto(2.5, size=n) + 1) * 20).astype(int),
    'Pareto(1.5,30)':  lambda n, rng: np.ceil((rng.pareto(1.5, size=n) + 1) * 30).astype(int),
}

WEIGHT_DISTRIBUTIONS = {
    'Exp(1/30)':        lambda m, rng: rng.exponential(30.0, size=m),
    'Exp(1)':           lambda m, rng: rng.exponential(1.0, size=m),
    'Exp(30)':          lambda m, rng: rng.exponential(1 / 30, size=m),
    'LogN(2,1)':        lambda m, rng: rng.lognormal(2.0, 1.0, size=m),
    'LogLogN(1.2,0.4)': lambda m, rng: np.exp(rng.lognormal(1.2, 0.4, size=m)),
    'LogLogN(1.2,0.8)': lambda m, rng: np.exp(rng.lognormal(1.2, 0.8, size=m)),
    'LogLogN(2,0.8)':   lambda m, rng: np.exp(rng.lognormal(2.0, 0.8, size=m)),
}

# 5 diverse regimes covering different retention levels and weight tails
DIVERSE_5 = [
    ('Unif(1,50)',      'Exp(1)'),           # ~22% retention, standard
    ('Exp(60)',         'LogLogN(1.2,0.4)'),  # ~28% retention, moderate tail
    ('LogN(3.26,0.66)', 'LogN(2,1)'),         # ~36% retention, lognormal
    ('Pareto(2.5,20)',  'Exp(30)'),           # ~18% retention, high degree
    ('Pareto(1.5,30)',  'LogLogN(2,0.8)'),    # ~11% retention, extreme tail
]


def build_config_list(config_name):
    """Return list of (deg_name, wt_name) tuples for the given config set."""
    if config_name == 'diverse5':
        return DIVERSE_5
    elif config_name == 'all':
        return [(d, w) for d in DEGREE_DISTRIBUTIONS for w in WEIGHT_DISTRIBUTIONS]
    else:
        raise ValueError(f"Unknown config set: {config_name}")


def _mse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean((a[m] - b[m]) ** 2))


def run_config(deg_name, wt_name, n_nodes, n_sir, seed, n_perms,
               neumann_kwargs=None):
    """Run one configuration, return result dict."""
    rng = np.random.default_rng(seed)

    deg_fn = DEGREE_DISTRIBUTIONS[deg_name]
    wt_fn = WEIGHT_DISTRIBUTIONS[wt_name]

    W_dist = configuration_model(n_nodes, deg_fn, wt_fn, rng=rng)
    n_edges = sparse.triu(W_dist).nnz

    if n_edges < 10:
        return None

    W_prox = to_proximity(W_dist)

    beta, cal_info = calibrate_beta(
        W_prox, gamma=1.0, n_calibration_runs=30, rng=rng, verbose=False)

    # Sparsify
    W_mbb_dist = metric_backbone(W_dist)
    n_mbb = sparse.triu(W_mbb_dist).nnz
    retention = n_mbb / n_edges

    W_mbbr = metric_backbone_rescaled(W_dist)
    W_effr = effective_resistance_sparsify(
        W_prox, n_edges=n_mbb, rng=np.random.default_rng(seed))

    kw = dict(n_perms=n_perms, seed=seed, verbose=False)
    if neumann_kwargs:
        kw.update(neumann_kwargs)
    t0 = time.time()
    W_neumann = neumann_sparsify(W_dist, n_mbb, **kw)
    t_neur = time.time() - t0

    # SIR evaluation
    initial = [int(np.argmax(np.array(W_prox.sum(axis=1)).ravel()))]
    sir_kw = dict(n_runs=n_sir, rng=np.random.default_rng(100))

    p_orig = sir_monte_carlo(W_prox, beta, 1.0, initial, **sir_kw)['infection_prob']
    mse_mbbr = _mse(p_orig, sir_monte_carlo(W_mbbr, beta, 1.0, initial, **sir_kw)['infection_prob'])
    mse_effr = _mse(p_orig, sir_monte_carlo(W_effr, beta, 1.0, initial, **sir_kw)['infection_prob'])
    mse_neur = _mse(p_orig, sir_monte_carlo(W_neumann, beta, 1.0, initial, **sir_kw)['infection_prob'])

    return {
        'deg': deg_name, 'wt': wt_name,
        'n_nodes': n_nodes, 'n_edges': n_edges,
        'n_mbb': n_mbb, 'retention': retention,
        'beta': beta, 't_neumann': t_neur,
        'mse_mbbr': mse_mbbr, 'mse_effr': mse_effr, 'mse_neur': mse_neur,
    }


def print_results(results):
    """Print a summary table."""
    print(f"\n{'Degree':<18} {'Weights':<18} {'Ret%':>5} "
          f"{'MBBr':>10} {'EffR':>10} {'Neumann':>10} {'Status':>8} {'Time':>5}")
    print('-' * 100)

    n_wins = n_beat_effr = n_beat_mbbr = 0

    for r in results:
        best = min(r['mse_mbbr'], r['mse_effr'])
        if r['mse_neur'] <= best:
            status = 'WIN'
            n_wins += 1
        else:
            status = f"{r['mse_neur'] / best:.2f}x"
        if r['mse_neur'] <= r['mse_effr']:
            n_beat_effr += 1
        if r['mse_neur'] <= r['mse_mbbr']:
            n_beat_mbbr += 1

        print(f"{r['deg']:<18} {r['wt']:<18} {r['retention'] * 100:>4.1f}% "
              f"{r['mse_mbbr']:>10.6f} {r['mse_effr']:>10.6f} "
              f"{r['mse_neur']:>10.6f} {status:>8} {r['t_neumann']:>4.0f}s")

    n = len(results)
    print(f"\nWins vs best: {n_wins}/{n}  |  "
          f"Beats EffR: {n_beat_effr}/{n}  |  "
          f"Beats MBBr: {n_beat_mbbr}/{n}")


def main():
    parser = argparse.ArgumentParser(description='Neumann sparsification pipeline')
    parser.add_argument('--configs', default='diverse5',
                        choices=['diverse5', 'all'],
                        help='Config set to run (default: diverse5)')
    parser.add_argument('--n-nodes', type=int, default=500)
    parser.add_argument('--n-sir', type=int, default=200)
    parser.add_argument('--n-perms', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: auto-named)')
    args = parser.parse_args()

    configs = build_config_list(args.configs)
    print(f"Running {len(configs)} configs with N={args.n_nodes}, "
          f"SIR={args.n_sir}, perms={args.n_perms}")

    results = []
    for i, (deg_name, wt_name) in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] {deg_name} | {wt_name}")
        r = run_config(deg_name, wt_name, args.n_nodes, args.n_sir,
                       args.seed, args.n_perms)
        if r is None:
            print("  SKIP (too few edges)")
            continue
        results.append(r)

        best = min(r['mse_mbbr'], r['mse_effr'])
        status = 'WIN' if r['mse_neur'] <= best else f"{r['mse_neur'] / best:.2f}x"
        print(f"  MBBr={r['mse_mbbr']:.6f}  EffR={r['mse_effr']:.6f}  "
              f"Neumann={r['mse_neur']:.6f}  [{status}]  ({r['t_neumann']:.0f}s)")

    print_results(results)

    # Save results
    out_path = args.output
    if out_path is None:
        out_path = f"neumann_results_{args.configs}_n{args.n_nodes}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
