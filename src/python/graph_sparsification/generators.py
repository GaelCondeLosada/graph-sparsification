"""Graph generators: Configuration Model and weighted Stochastic Block Model."""

import numpy as np
from scipy import sparse


def configuration_model(degree_sequence, weight_distribution="exponential",
                        weight_params=None, rng=None):
    """Generate a weighted graph from the configuration model.

    Each node gets a number of half-edges equal to its degree. Half-edges are
    paired uniformly at random. Self-loops and multi-edges are collapsed (weights
    summed). Edge weights are drawn iid from the specified distribution.

    Parameters
    ----------
    degree_sequence : array-like of int
        Desired degree for each node.
    weight_distribution : str
        "exponential", "uniform", or "lognormal".
    weight_params : dict or None
        Parameters for the weight distribution. Defaults depend on distribution.
    rng : np.random.Generator or None

    Returns
    -------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix (symmetric).
    """
    rng = np.random.default_rng(rng)
    degree_sequence = np.asarray(degree_sequence, dtype=int)
    n = len(degree_sequence)

    # Ensure even sum of degrees
    total = degree_sequence.sum()
    if total % 2 == 1:
        degree_sequence = degree_sequence.copy()
        idx = rng.integers(n)
        degree_sequence[idx] += 1
        total += 1

    # Build stub list and shuffle
    stubs = np.repeat(np.arange(n), degree_sequence)
    rng.shuffle(stubs)

    # Pair stubs
    src = stubs[0::2]
    dst = stubs[1::2]

    # Remove self-loops
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    # Ensure canonical ordering for dedup
    swap = src > dst
    src[swap], dst[swap] = dst[swap].copy(), src[swap].copy()

    # Assign weights
    weight_params = weight_params or {}
    m = len(src)
    if weight_distribution == "exponential":
        scale = weight_params.get("scale", 1.0)
        weights = rng.exponential(scale, size=m)
    elif weight_distribution == "uniform":
        low = weight_params.get("low", 0.1)
        high = weight_params.get("high", 1.0)
        weights = rng.uniform(low, high, size=m)
    elif weight_distribution == "lognormal":
        mean = weight_params.get("mean", 0.0)
        sigma = weight_params.get("sigma", 1.0)
        weights = rng.lognormal(mean, sigma, size=m)
    else:
        raise ValueError(f"Unknown weight distribution: {weight_distribution}")

    # Build sparse matrix, summing duplicate edges
    W = sparse.coo_matrix((weights, (src, dst)), shape=(n, n))
    W = W + W.T  # symmetrize
    W = W.tocsr()
    W.eliminate_zeros()

    return W


def wsbm(n, k, pi, B, weight_distribution="exponential", Lambda=None, rng=None):
    """Generate a weighted Stochastic Block Model graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Number of communities.
    pi : array-like of float, shape (k,)
        Community membership probabilities (must sum to 1).
    B : array-like of float, shape (k, k)
        Edge probability matrix. B[a,b] = probability of edge between
        a node in community a and a node in community b (before scaling).
        Actual probability is B[a,b] * rho_n where rho_n = log(n)/n.
    weight_distribution : str
        "exponential", "uniform", or "lognormal".
    Lambda : array-like of float, shape (k, k) or None
        Rate parameters for edge weights between communities.
        If None, defaults to ones matrix.
    rng : np.random.Generator or None

    Returns
    -------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix (symmetric, distance/cost graph).
    z : np.ndarray of int, shape (n,)
        Community assignments.
    """
    rng = np.random.default_rng(rng)
    pi = np.asarray(pi, dtype=float)
    B = np.asarray(B, dtype=float)

    if Lambda is None:
        Lambda = np.ones((k, k))
    Lambda = np.asarray(Lambda, dtype=float)

    # Assign communities
    z = rng.choice(k, size=n, p=pi)

    # Scaling factor: rho_n = log(n)/n for sparse regime
    rho_n = np.log(n) / n

    # Generate edges
    rows, cols, weights = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = z[i], z[j]
            p = B[a, b] * rho_n
            p = min(p, 1.0)
            if rng.random() < p:
                # Sample weight (cost/distance)
                lam = Lambda[a, b]
                if weight_distribution == "exponential":
                    w = rng.exponential(1.0 / lam)
                elif weight_distribution == "uniform":
                    w = rng.uniform(0, 1.0 / lam)
                elif weight_distribution == "lognormal":
                    w = rng.lognormal(0, 1.0 / lam)
                else:
                    raise ValueError(f"Unknown: {weight_distribution}")

                rows.append(i)
                cols.append(j)
                weights.append(w)

    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    weights = np.array(weights, dtype=float)

    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n))
    W = W + W.T
    W = W.tocsr()

    return W, z


def wsbm_fast(n, k, pi, B, weight_distribution="exponential", Lambda=None, rng=None):
    """Vectorized wSBM generator — much faster for large n.

    Same interface as wsbm() but avoids the O(n^2) Python loop by
    vectorizing edge sampling per block pair.
    """
    rng = np.random.default_rng(rng)
    pi = np.asarray(pi, dtype=float)
    B = np.asarray(B, dtype=float)

    if Lambda is None:
        Lambda = np.ones((k, k))
    Lambda = np.asarray(Lambda, dtype=float)

    z = rng.choice(k, size=n, p=pi)
    rho_n = np.log(n) / n

    # Group nodes by community
    communities = [np.where(z == a)[0] for a in range(k)]

    rows_all, cols_all, weights_all = [], [], []

    for a in range(k):
        for b in range(a, k):
            nodes_a = communities[a]
            nodes_b = communities[b]
            p = min(B[a, b] * rho_n, 1.0)
            if p <= 0 or len(nodes_a) == 0 or len(nodes_b) == 0:
                continue

            if a == b:
                # Intra-community: upper triangle only
                na = len(nodes_a)
                num_possible = na * (na - 1) // 2
                if num_possible == 0:
                    continue
                num_edges = rng.binomial(num_possible, p)
                if num_edges == 0:
                    continue
                # Sample edge indices
                idx = rng.choice(num_possible, size=num_edges, replace=False)
                # Convert linear index to (i,j) in upper triangle
                ii = np.floor((-1 + np.sqrt(1 + 8 * idx)) / 2).astype(int)
                # Adjust for off-by-one
                ii = np.clip(ii, 0, na - 2)
                while True:
                    tri_start = ii * (ii + 1) // 2
                    overflow = idx < tri_start
                    if not overflow.any():
                        break
                    ii[overflow] -= 1
                tri_start = ii * (ii + 1) // 2
                jj = idx - tri_start
                ii += 1  # shift to get proper upper triangle
                r = nodes_a[ii]
                c = nodes_a[jj]
            else:
                # Inter-community: all pairs
                num_possible = len(nodes_a) * len(nodes_b)
                num_edges = rng.binomial(num_possible, p)
                if num_edges == 0:
                    continue
                idx = rng.choice(num_possible, size=num_edges, replace=False)
                ii = idx // len(nodes_b)
                jj = idx % len(nodes_b)
                r = nodes_a[ii]
                c = nodes_b[jj]

            # Ensure r < c for upper triangle
            swap = r > c
            r[swap], c[swap] = c[swap].copy(), r[swap].copy()

            # Sample weights
            lam = Lambda[a, b]
            if weight_distribution == "exponential":
                w = rng.exponential(1.0 / lam, size=num_edges)
            elif weight_distribution == "uniform":
                w = rng.uniform(0, 1.0 / lam, size=num_edges)
            elif weight_distribution == "lognormal":
                w = rng.lognormal(0, 1.0 / lam, size=num_edges)
            else:
                raise ValueError(f"Unknown: {weight_distribution}")

            rows_all.append(r)
            cols_all.append(c)
            weights_all.append(w)

    if len(rows_all) == 0:
        return sparse.csr_matrix((n, n)), z

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    weights = np.concatenate(weights_all)

    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n))
    W = W + W.T
    W = W.tocsr()

    return W, z
