"""Graph sparsification: Metric Backbone, MBB Rescaled, and Effective Resistance.

Weight conventions
------------------
- **Distance** (cost) weights live in (0, inf]. Used for shortest-path
  computations (Metric Backbone).
- **Proximity** weights live in [0, 1]. Used for Effective Resistance
  sparsification and SIR simulation (transmission rate ~ proximity).

Conversion (element-wise on nonzero entries):
    proximity = 1 / (distance + 1)
    distance  = 1 / proximity - 1
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


# ── Weight-space conversions ─────────────────────────────────────────────

def proximity_to_distance(proximity):
    """Convert proximity weights to distances.

    proximity in [0, 1]  ->  distance in (0, inf]
    Formula: distance = (1 / proximity) - 1
    """
    return (1.0 / proximity) - 1.0


def distance_to_proximity(distance):
    """Convert distance weights to proximities.

    distance in (0, inf]  ->  proximity in [0, 1]
    Formula: proximity = 1 / (distance + 1)
    """
    return 1.0 / (distance + 1.0)


def _convert_sparse_weights(W, convert_fn):
    """Apply *convert_fn* element-wise to every nonzero entry of *W*.

    Returns a new CSR matrix with the same sparsity pattern.
    """
    W = sparse.csr_matrix(W, dtype=float).copy()
    W.data = convert_fn(W.data)
    return W


def to_proximity(W_dist):
    """Convert a distance-weighted sparse matrix to proximity weights."""
    return _convert_sparse_weights(W_dist, distance_to_proximity)


def to_distance(W_prox):
    """Convert a proximity-weighted sparse matrix to distance weights."""
    return _convert_sparse_weights(W_prox, proximity_to_distance)


# ── Metric Backbone ──────────────────────────────────────────────────────

def metric_backbone(W_dist):
    """Compute the metric backbone of a distance-weighted graph.

    The metric backbone is the union of all shortest paths. An edge (u,v) is
    kept iff it lies on some shortest path, i.e., its weight equals the shortest
    path distance between u and v.

    Parameters
    ----------
    W_dist : scipy.sparse.csr_matrix
        Weighted adjacency matrix with **distance** weights (costs > 0).

    Returns
    -------
    W_mbb : scipy.sparse.csr_matrix
        Metric backbone — same distance weights, fewer edges.
    """
    W_dist = sparse.csr_matrix(W_dist)
    n = W_dist.shape[0]

    dist = shortest_path(W_dist, directed=False)

    W_coo = sparse.triu(W_dist, format="coo")
    rows, cols, data = W_coo.row, W_coo.col, W_coo.data

    sp_dists = dist[rows, cols]
    tol = 1e-10 * np.maximum(np.abs(data), np.abs(sp_dists))
    tol = np.maximum(tol, 1e-14)
    is_metric = np.abs(data - sp_dists) <= tol

    metric_rows = rows[is_metric]
    metric_cols = cols[is_metric]
    metric_data = data[is_metric]

    W_mbb = sparse.coo_matrix((metric_data, (metric_rows, metric_cols)), shape=(n, n))
    W_mbb = W_mbb + W_mbb.T
    W_mbb = W_mbb.tocsr()

    return W_mbb


# ── Metric Backbone Rescaled (MBBr) ──────────────────────────────────────

def metric_backbone_rescaled(W_dist):
    """Metric Backbone with rescaled proximity weights.

    1. Compute the MBB on the distance graph.
    2. Convert both the original graph and MBB to proximity weights.
    3. Rescale MBB proximities so that their sum equals the original
       graph's proximity sum.

    This preserves the MBB sparsity pattern while ensuring that the
    total "connectivity budget" matches the original graph.

    Parameters
    ----------
    W_dist : scipy.sparse.csr_matrix
        Weighted adjacency matrix with **distance** weights.

    Returns
    -------
    W_mbbr : scipy.sparse.csr_matrix
        MBBr graph with **proximity** weights (rescaled).
    """
    W_mbb_dist = metric_backbone(W_dist)

    W_prox = to_proximity(W_dist)
    W_mbb_prox = to_proximity(W_mbb_dist)

    sum_orig = W_prox.data.sum()
    sum_mbb = W_mbb_prox.data.sum()

    if sum_mbb > 0:
        scale = sum_orig / sum_mbb
        W_mbbr = W_mbb_prox.copy()
        W_mbbr.data *= scale
    else:
        W_mbbr = W_mbb_prox.copy()

    return W_mbbr


def _compute_effective_resistances(W, n_projections=None, epsilon=0.1):
    """Compute approximate effective resistances for all edges.

    Uses the Johnson-Lindenstrauss random projection approach:
    R_e = ||Z(chi_u - chi_v)||^2 where Z approximates W^{1/2} B L^+ .

    For moderate-sized graphs, we compute exact resistances via L^+.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix.
    n_projections : int or None
        Number of random projections for approximation. If None, uses exact.
    epsilon : float
        Approximation parameter (only used if n_projections is set).

    Returns
    -------
    resistances : dict
        Maps (i, j) with i < j to effective resistance R_e.
    """
    W = sparse.csr_matrix(W, dtype=float)
    n = W.shape[0]

    # Build graph Laplacian L = D - A
    degrees = np.array(W.sum(axis=1)).ravel()
    L = sparse.diags(degrees) - W
    L = L.toarray()

    if n_projections is None or n <= 2000:
        # Exact: compute pseudoinverse of L
        # L is singular (rank n-1), so we use pinv
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Pseudoinverse: invert nonzero eigenvalues
        tol = 1e-10 * eigenvalues.max()
        nonzero = eigenvalues > tol
        L_pinv = (eigenvectors[:, nonzero]
                  * (1.0 / eigenvalues[nonzero])[np.newaxis, :]
                  ) @ eigenvectors[:, nonzero].T

        # Extract effective resistances for existing edges
        W_coo = sparse.triu(W, format="coo")
        resistances = {}
        for idx in range(len(W_coo.row)):
            i, j = W_coo.row[idx], W_coo.col[idx]
            R_e = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
            resistances[(i, j)] = max(R_e, 0.0)

    else:
        # Approximate via random projection
        k = int(np.ceil(24 * np.log(n) / epsilon**2))
        k = min(k, n_projections) if n_projections else k

        eigenvalues, eigenvectors = np.linalg.eigh(L)
        tol = 1e-10 * eigenvalues.max()
        nonzero = eigenvalues > tol

        # Z = Q @ diag(1/sqrt(lambda)) @ U^T where Q is random projection
        rng = np.random.default_rng(42)
        Q = rng.choice([-1, 1], size=(k, nonzero.sum())) / np.sqrt(k)
        Z = Q @ np.diag(1.0 / np.sqrt(eigenvalues[nonzero])) @ eigenvectors[:, nonzero].T

        W_coo = sparse.triu(W, format="coo")
        resistances = {}
        for idx in range(len(W_coo.row)):
            i, j = W_coo.row[idx], W_coo.col[idx]
            diff = Z[:, i] - Z[:, j]
            R_e = np.dot(diff, diff)
            resistances[(i, j)] = max(R_e, 0.0)

    return resistances


def effective_resistance_sparsify(W, q=None, fraction=0.1, rng=None):
    """Sparsify a graph using the Spielman-Srivastava algorithm.

    Samples edges with probability proportional to w_e * R_e (effective
    resistance importance), then reweights to preserve the Laplacian in
    expectation.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix.
    q : int or None
        Number of edge samples. If None, computed from fraction.
    fraction : float
        Fraction of total edges to target (used if q is None).
    rng : np.random.Generator or None

    Returns
    -------
    W_sparse : scipy.sparse.csr_matrix
        Sparsified weighted adjacency matrix.
    """
    rng = np.random.default_rng(rng)
    W = sparse.csr_matrix(W, dtype=float)
    n = W.shape[0]

    # Get edges and weights (upper triangle)
    W_coo = sparse.triu(W, format="coo")
    edges_i = W_coo.row
    edges_j = W_coo.col
    weights = W_coo.data
    m = len(weights)

    if m == 0:
        return W.copy()

    if q is None:
        q = max(int(fraction * m), n)

    # Compute effective resistances
    resistances = _compute_effective_resistances(W)

    # Edge importances: u_e = w_e * R_e
    R_vals = np.array([resistances.get((i, j), 0.0)
                       for i, j in zip(edges_i, edges_j)])
    importances = weights * R_vals

    # Sampling probabilities
    total_importance = importances.sum()
    if total_importance <= 0:
        # Fallback to uniform
        probs = np.ones(m) / m
    else:
        probs = importances / total_importance

    # Sample q edges with replacement
    sampled_indices = rng.choice(m, size=q, replace=True, p=probs)

    # Count how many times each edge was sampled
    counts = np.bincount(sampled_indices, minlength=m)

    # Reweight: w_tilde_e = w_e * count / (q * p_e)
    new_rows, new_cols, new_weights = [], [], []
    for idx in range(m):
        if counts[idx] > 0:
            w_new = weights[idx] * counts[idx] / (q * probs[idx])
            new_rows.append(edges_i[idx])
            new_cols.append(edges_j[idx])
            new_weights.append(w_new)

    new_rows = np.array(new_rows, dtype=int)
    new_cols = np.array(new_cols, dtype=int)
    new_weights = np.array(new_weights, dtype=float)

    W_sparse = sparse.coo_matrix((new_weights, (new_rows, new_cols)), shape=(n, n))
    W_sparse = W_sparse + W_sparse.T
    W_sparse = W_sparse.tocsr()

    return W_sparse
