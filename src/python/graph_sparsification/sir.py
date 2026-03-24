"""SIR epidemic simulation on weighted graphs.

Implements a continuous-time, event-driven Gillespie algorithm using a heap-based
priority queue. Each edge e transmits the infection at rate beta * w_e, and each
infected node recovers at rate gamma.
"""

import numpy as np
from scipy import sparse
import heapq

# Try to import the C++ backend for faster simulation
try:
    from graph_sparsification._sir_cpp import sir_simulation_cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


def sir_simulation(W, beta, gamma, initial_infected=None, t_max=20.0,
                   rng=None, use_cpp=True):
    """Run a single SIR simulation on a weighted graph.

    Continuous-time, event-driven Gillespie algorithm. Each edge e=(i,j) with
    weight w_e transmits infection at rate beta*w_e. Infected nodes recover at
    rate gamma.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix.
    beta : float
        Infection rate per unit edge weight.
    gamma : float
        Recovery rate.
    initial_infected : array-like or None
        Indices of initially infected nodes. If None, a single random node.
    t_max : float
        Maximum simulation time.
    rng : np.random.Generator or int or None
    use_cpp : bool
        Use C++ backend if available.

    Returns
    -------
    result : dict
        - 'infected': bool array, whether each node was ever infected
        - 'arrival_times': float array, time of infection (inf if never)
        - 'recovery_times': float array, time of recovery (inf if never)
        - 'S_t', 'I_t', 'R_t': lists of (time, count) for SIR curves
    """
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)
    else:
        rng = np.random.default_rng(rng)

    W = sparse.csr_matrix(W, dtype=float)
    n = W.shape[0]

    if initial_infected is None:
        initial_infected = [rng.integers(n)]
    initial_infected = np.asarray(initial_infected, dtype=int)

    if use_cpp and _HAS_CPP:
        return _sir_cpp_wrapper(W, beta, gamma, initial_infected, t_max, rng)

    return _sir_python(W, beta, gamma, initial_infected, t_max, rng)


def _sir_python(W, beta, gamma, initial_infected, t_max, rng):
    """Pure Python SIR implementation with Gillespie algorithm."""
    n = W.shape[0]

    # States: 0=S, 1=I, 2=R
    state = np.zeros(n, dtype=int)
    arrival_times = np.full(n, np.inf)
    recovery_times = np.full(n, np.inf)

    # Priority queue: (time, event_type, node_i, node_j)
    # event_type: 0=infection (i infects j), 1=recovery (i recovers)
    heap = []
    event_id = 0

    S_t = [(0.0, n - len(initial_infected))]
    I_t = [(0.0, len(initial_infected))]
    R_t = [(0.0, 0)]
    n_S = n - len(initial_infected)
    n_I = len(initial_infected)
    n_R = 0

    def schedule_events(node, t_infect):
        """Schedule infection and recovery events for a newly infected node."""
        nonlocal event_id

        # Schedule recovery
        dt_recover = rng.exponential(1.0 / gamma)
        heapq.heappush(heap, (t_infect + dt_recover, 1, node, -1, event_id))
        event_id += 1

        # Schedule infection attempts to neighbors
        start, end = W.indptr[node], W.indptr[node + 1]
        neighbors = W.indices[start:end]
        edge_weights = W.data[start:end]

        for nbr, w in zip(neighbors, edge_weights):
            rate = beta * w
            if rate > 0:
                dt_infect = rng.exponential(1.0 / rate)
                heapq.heappush(heap, (t_infect + dt_infect, 0, node, nbr, event_id))
                event_id += 1

    # Initialize
    for node in initial_infected:
        state[node] = 1
        arrival_times[node] = 0.0
        schedule_events(node, 0.0)

    # Process events
    while heap:
        t, etype, src, dst, _ = heapq.heappop(heap)

        if t > t_max:
            break

        if etype == 1:
            # Recovery event
            if state[src] == 1:
                state[src] = 2
                recovery_times[src] = t
                n_I -= 1
                n_R += 1
                S_t.append((t, n_S))
                I_t.append((t, n_I))
                R_t.append((t, n_R))

        elif etype == 0:
            # Infection event: src tries to infect dst
            if state[src] == 1 and state[dst] == 0:
                state[dst] = 1
                arrival_times[dst] = t
                n_S -= 1
                n_I += 1
                S_t.append((t, n_S))
                I_t.append((t, n_I))
                R_t.append((t, n_R))
                schedule_events(dst, t)

    return {
        'infected': arrival_times < np.inf,
        'arrival_times': arrival_times,
        'recovery_times': recovery_times,
        'S_t': S_t,
        'I_t': I_t,
        'R_t': R_t,
    }


def _sir_cpp_wrapper(W, beta, gamma, initial_infected, t_max, rng):
    """Wrapper around C++ SIR implementation."""
    W = sparse.csr_matrix(W, dtype=float)
    seed = int(rng.integers(0, 2**31))

    arrival_times, recovery_times = sir_simulation_cpp(
        W.indptr.astype(np.int32),
        W.indices.astype(np.int32),
        W.data.astype(np.float64),
        W.shape[0],
        beta, gamma,
        initial_infected.astype(np.int32),
        t_max, seed
    )

    return {
        'infected': arrival_times < np.inf,
        'arrival_times': arrival_times,
        'recovery_times': recovery_times,
        'S_t': [],  # C++ version doesn't track curves for speed
        'I_t': [],
        'R_t': [],
    }


def sir_monte_carlo(W, beta, gamma, initial_infected=None, n_runs=100,
                    t_max=20.0, rng=None, use_cpp=True):
    """Run multiple SIR simulations and compute infection probabilities.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix.
    beta, gamma : float
        SIR parameters.
    initial_infected : array-like or None
        Initially infected nodes.
    n_runs : int
        Number of independent simulations.
    t_max : float
        Maximum simulation time.
    rng : np.random.Generator or int or None
    use_cpp : bool
        Use C++ backend if available.

    Returns
    -------
    result : dict
        - 'infection_prob': float array of shape (n,), P(node infected).
        - 'mean_arrival_time': float array, mean arrival time (conditioned
          on infection).
        - 'all_arrival_times': list of arrays, per-run arrival times.
    """
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)
    else:
        rng = np.random.default_rng(rng)

    n = W.shape[0]
    infection_counts = np.zeros(n)
    arrival_time_sums = np.zeros(n)
    arrival_time_counts = np.zeros(n)
    all_arrival_times = []

    for _ in range(n_runs):
        result = sir_simulation(W, beta, gamma, initial_infected, t_max,
                                rng=rng, use_cpp=use_cpp)
        infected = result['infected']
        arrivals = result['arrival_times']

        infection_counts += infected
        finite_mask = np.isfinite(arrivals)
        arrival_time_sums[finite_mask] += arrivals[finite_mask]
        arrival_time_counts[finite_mask] += 1
        all_arrival_times.append(arrivals.copy())

    infection_prob = infection_counts / n_runs

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_arrival_time = np.where(
            arrival_time_counts > 0,
            arrival_time_sums / arrival_time_counts,
            np.inf
        )

    return {
        'infection_prob': infection_prob,
        'mean_arrival_time': mean_arrival_time,
        'all_arrival_times': all_arrival_times,
    }


def calibrate_beta(W, gamma=1.0, target_mean_infection=0.6,
                   target_range=(0.45, 0.7), initial_infected=None,
                   n_calibration_runs=20, t_max=100.0, rng=None,
                   beta_min=1e-4, beta_max=10.0, max_iterations=30,
                   verbose=True):
    """Find a beta that produces an interesting spread of infection probabilities.

    Uses bisection search on beta to target a mean infection probability near
    `target_mean_infection`. The goal is to avoid the two degenerate regimes:
    - beta too low: almost no one gets infected (boring)
    - beta too high: everyone gets infected with probability ~1 (no variance)

    A good range is when mean infection probability is around 0.4-0.6 and
    the standard deviation across nodes is high (heterogeneous spread).

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix.
    gamma : float
        Recovery rate.
    target_mean_infection : float
        Desired mean infection probability across nodes (default 0.5).
    target_range : tuple of float
        Acceptable range for mean infection probability.
    initial_infected : array-like or None
        Initially infected nodes. If None, picks highest-degree node.
    n_calibration_runs : int
        Number of SIR runs per beta evaluation (fewer = faster but noisier).
    t_max : float
        Maximum simulation time.
    rng : np.random.Generator or int or None
    beta_min, beta_max : float
        Search bounds for beta.
    max_iterations : int
        Maximum bisection iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    beta : float
        Calibrated beta value.
    info : dict
        - 'infection_prob': infection probabilities at calibrated beta
        - 'mean_infection': mean infection probability
        - 'std_infection': std of infection probabilities
        - 'history': list of (beta, mean_infection) tried
    """
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)
    else:
        rng = np.random.default_rng(rng)

    W = sparse.csr_matrix(W, dtype=float)
    n = W.shape[0]

    if initial_infected is None:
        degrees = np.array(W.sum(axis=1)).ravel()
        initial_infected = [int(np.argmax(degrees))]

    lo, hi = beta_min, beta_max
    history = []
    best_beta = (lo + hi) / 2
    best_probs = None
    best_distance = float('inf')

    for iteration in range(max_iterations):
        beta = (lo + hi) / 2

        result = sir_monte_carlo(W, beta, gamma, initial_infected,
                                 n_runs=n_calibration_runs, t_max=t_max, rng=rng)
        probs = result['infection_prob']
        mean_inf = probs.mean()
        std_inf = probs.std()

        history.append((beta, mean_inf))
        distance = abs(mean_inf - target_mean_infection)

        if verbose:
            print(f"  iter {iteration+1:2d}: beta={beta:.6f}, "
                  f"mean_inf={mean_inf:.3f}, std={std_inf:.3f}")

        if distance < best_distance:
            best_distance = distance
            best_beta = beta
            best_probs = probs.copy()

        # Check if we're in the target range
        if target_range[0] <= mean_inf <= target_range[1]:
            if verbose:
                print(f"  -> Converged: beta={beta:.6f}")
            return beta, {
                'infection_prob': probs,
                'mean_infection': mean_inf,
                'std_infection': std_inf,
                'history': history,
            }

        # Bisection: if mean infection is too high, reduce beta
        if mean_inf > target_mean_infection:
            hi = beta
        else:
            lo = beta

        # Early stop if bounds are very tight
        if (hi - lo) / max(hi, 1e-10) < 0.01:
            break

    if verbose:
        mean_inf = best_probs.mean() if best_probs is not None else 0.0
        print(f"  -> Best found: beta={best_beta:.6f}, mean_inf={mean_inf:.3f}")

    return best_beta, {
        'infection_prob': best_probs,
        'mean_infection': best_probs.mean() if best_probs is not None else 0.0,
        'std_infection': best_probs.std() if best_probs is not None else 0.0,
        'history': history,
    }
