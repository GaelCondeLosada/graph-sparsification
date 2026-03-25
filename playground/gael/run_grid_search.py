"""Run grid search configs in batches, saving results incrementally."""
import sys, os, json, time
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'python'))

from graph_sparsification.generators import configuration_model
from graph_sparsification.sparsifiers import (
    metric_backbone, metric_backbone_rescaled,
    effective_resistance_sparsify, to_proximity,
)
from graph_sparsification.sir import sir_monte_carlo, calibrate_beta
from graph_sparsification.neumann_sparsifier import neumann_sparsify

N_NODES = 500
N_SIR_RUNS = 200
N_CAL_RUNS = 30
GAMMA = 1.0
SEED = 42
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'grid_search_results.json')

degree_distributions = {
    'Unif(1,50)':      lambda n, rng: rng.integers(1, 51, size=n),
    'Unif(1,100)':     lambda n, rng: rng.integers(1, 101, size=n),
    'Exp(30)':         lambda n, rng: np.ceil(rng.exponential(30, size=n)).astype(int),
    'Exp(60)':         lambda n, rng: np.ceil(rng.exponential(60, size=n)).astype(int),
    'LogN(3.26,0.66)': lambda n, rng: np.ceil(rng.lognormal(3.26, 0.66, size=n)).astype(int),
    'LogN(3.26,2)':    lambda n, rng: np.ceil(rng.lognormal(3.26, 2.0, size=n)).astype(int),
    'Pareto(2.5,20)':  lambda n, rng: np.ceil((rng.pareto(2.5, size=n) + 1) * 20).astype(int),
    'Pareto(1.5,30)':  lambda n, rng: np.ceil((rng.pareto(1.5, size=n) + 1) * 30).astype(int),
}

weight_distributions = {
    'Exp(1/30)':        lambda m, rng: rng.exponential(30.0, size=m),
    'Exp(1)':           lambda m, rng: rng.exponential(1.0, size=m),
    'Exp(30)':          lambda m, rng: rng.exponential(1/30, size=m),
    'LogN(2,1)':        lambda m, rng: rng.lognormal(2.0, 1.0, size=m),
    'LogLogN(1.2,0.4)': lambda m, rng: np.exp(rng.lognormal(1.2, 0.4, size=m)),
    'LogLogN(1.2,0.8)': lambda m, rng: np.exp(rng.lognormal(1.2, 0.8, size=m)),
    'LogLogN(2,0.8)':   lambda m, rng: np.exp(rng.lognormal(2.0, 0.8, size=m)),
}

def _mse(po, ps):
    m = np.isfinite(po) & np.isfinite(ps)
    return float(np.mean((po[m] - ps[m])**2))

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []

def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def run_one(deg_name, deg_sampler, wt_name, wt_sampler):
    label = f'{deg_name}  |  {wt_name}'
    rng = np.random.default_rng(42)

    W_dist = configuration_model(N_NODES, deg_sampler, wt_sampler, rng=rng)
    n_edges = sparse.triu(W_dist).nnz
    if n_edges < 10:
        print(f'  SKIP: only {n_edges} edges')
        return None

    W_prox = to_proximity(W_dist)

    beta, cal_info = calibrate_beta(
        W_prox, gamma=GAMMA,
        n_calibration_runs=N_CAL_RUNS, rng=rng, verbose=False)

    W_mbb_dist = metric_backbone(W_dist)
    n_mbb = sparse.triu(W_mbb_dist).nnz
    retention = n_mbb / n_edges
    W_mbbr = metric_backbone_rescaled(W_dist)
    W_effr = effective_resistance_sparsify(
        W_prox, n_edges=n_mbb, rng=np.random.default_rng(SEED))

    t0 = time.time()
    W_neumann = neumann_sparsify(W_dist, n_mbb, seed=SEED, verbose=False)
    t_neur = time.time() - t0

    initial = [int(np.argmax(np.array(W_prox.sum(axis=1)).ravel()))]

    sir_orig = sir_monte_carlo(W_prox, beta, GAMMA, initial,
                                n_runs=N_SIR_RUNS, rng=np.random.default_rng(100))
    sir_mbbr = sir_monte_carlo(W_mbbr, beta, GAMMA, initial,
                                n_runs=N_SIR_RUNS, rng=np.random.default_rng(100))
    sir_effr = sir_monte_carlo(W_effr, beta, GAMMA, initial,
                                n_runs=N_SIR_RUNS, rng=np.random.default_rng(100))
    sir_neur = sir_monte_carlo(W_neumann, beta, GAMMA, initial,
                                n_runs=N_SIR_RUNS, rng=np.random.default_rng(100))

    p_orig = sir_orig['infection_prob']
    mse_mbbr = _mse(p_orig, sir_mbbr['infection_prob'])
    mse_effr = _mse(p_orig, sir_effr['infection_prob'])
    mse_neur = _mse(p_orig, sir_neur['infection_prob'])

    best = min(mse_mbbr, mse_effr)
    status = 'WIN' if mse_neur <= best else f'{mse_neur/best:.2f}x'
    print(f'  MBBr={mse_mbbr:.6f}  EffR={mse_effr:.6f}  Neumann={mse_neur:.6f}  [{status}]  ({t_neur:.0f}s)')

    return {
        'deg': deg_name, 'wt': wt_name, 'label': label,
        'n_edges': n_edges, 'n_mbb': n_mbb,
        'retention': retention, 'beta': beta,
        'mse_mbbr': mse_mbbr, 'mse_effr': mse_effr, 'mse_neur': mse_neur,
    }


if __name__ == '__main__':
    # Parse optional batch args: python run_grid_search.py [start] [end]
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 999

    existing = load_results()
    done_keys = {(r['deg'], r['wt']) for r in existing}
    print(f'Already completed: {len(done_keys)} configs')

    configs = []
    for deg_name, deg_sampler in degree_distributions.items():
        for wt_name, wt_sampler in weight_distributions.items():
            configs.append((deg_name, deg_sampler, wt_name, wt_sampler))

    batch = configs[start:end]
    print(f'Running configs {start}-{min(end, len(configs))-1} ({len(batch)} total)\n')

    for i, (deg_name, deg_sampler, wt_name, wt_sampler) in enumerate(batch):
        if (deg_name, wt_name) in done_keys:
            print(f'[{start+i}] {deg_name} | {wt_name} — already done, skipping')
            continue

        print(f'[{start+i}] {deg_name} | {wt_name}')
        t0 = time.time()
        result = run_one(deg_name, deg_sampler, wt_name, wt_sampler)
        if result is not None:
            existing.append(result)
            save_results(existing)
            done_keys.add((deg_name, wt_name))
        print(f'  Total: {time.time()-t0:.0f}s\n')

    print(f'\nCompleted: {len(existing)} configs saved to {RESULTS_FILE}')
