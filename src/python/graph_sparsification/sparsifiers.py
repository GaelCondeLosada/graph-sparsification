"""Graph sparsification: Metric Backbone and Effective Resistance."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


def metric_backbone(W):
    """Compute the metric backbone of a weighted graph.

    The metric backbone is the union of all shortest paths. An edge (u,v) is
    kept iff it lies on some shortest path, i.e., its weight equals the shortest
    path distance between u and v.

    For a distance/cost graph, edge (u,v) with cost c(u,v) is metric if
    c(u,v) = d(u,v) where d is the shortest-path distance.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix (costs/distances, not similarities).

    Returns
    -------
    W_mbb : scipy.sparse.csr_matrix
        Metric backbone adjacency matrix (same weights, fewer edges).
    """
    W = sparse.csr_matrix(W)
    n = W.shape[0]

    # Compute all-pairs shortest paths
    dist = shortest_path(W, directed=False)

    # An edge (i,j) is metric iff its weight equals the shortest path distance
    W_coo = sparse.triu(W, format="coo")
    rows, cols, data = W_coo.row, W_coo.col, W_coo.data

    # Edge is metric if w(i,j) == d(i,j) (within floating point tolerance)
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
