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


def effective_resistance_sparsify(W, q=None, fraction=0.1, n_edges=None,
                                   rng=None):
    """Sparsify a graph using the Spielman-Srivastava algorithm.

    Two modes of operation (both follow the Spielman-Srivastava reweighting
    scheme from Mercier et al. 2022, so that E[w̃_e] = w_e):

    **Sampling mode** (default): sample *q* edges with replacement,
    probability p_e proportional to w_e * R_e, reweight
    w̃_e = w_e * count / (q * p_e).  The number of *distinct* edges in
    the result is random.

    **Exact-edges mode** (``n_edges`` is set): sample exactly *n_edges*
    distinct edges without replacement, probability proportional to
    w_e * R_e, reweight w̃_e = w_e / (n_edges * p_e).  Guarantees an
    exact edge count for fair comparison with other sparsifiers.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Weighted adjacency matrix.
    q : int or None
        Number of edge samples (sampling mode). If None, computed from
        *fraction*.
    fraction : float
        Fraction of total edges to target (sampling mode, used if *q* is
        None).
    n_edges : int or None
        If set, switch to exact-edges mode and sample exactly this many
        distinct edges (upper-triangle count).  Overrides *q* and *fraction*.
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

    # Compute effective resistances
    resistances = _compute_effective_resistances(W)
    R_vals = np.array([resistances.get((i, j), 0.0)
                       for i, j in zip(edges_i, edges_j)])
    importances = weights * R_vals

    # Sampling probabilities (shared by both modes)
    total_importance = importances.sum()
    if total_importance <= 0:
        probs = np.ones(m) / m
    else:
        probs = importances / total_importance

    # ── Exact-edges mode ──────────────────────────────────────────────
    if n_edges is not None:
        # Edges with zero importance get p=0 and cannot be drawn with
        # replace=False; k must not exceed the number of positive probs.
        n_pos = int(np.count_nonzero(probs > 0))
        if n_pos == 0:
            probs = np.ones(m) / m
            n_pos = m
        k = int(min(n_edges, m, n_pos))
        if k == 0:
            return sparse.csr_matrix((n, n), dtype=float)
        # Sample k distinct edges without replacement, prob ∝ w_e * R_e
        selected = rng.choice(m, size=k, replace=False, p=probs)

        # Reweight: w̃_e = w_e / (k * p_e)  so that E[w̃_e] = w_e
        sel_weights = weights[selected] / (k * probs[selected])

        W_sparse = sparse.coo_matrix(
            (sel_weights, (edges_i[selected], edges_j[selected])),
            shape=(n, n))
        W_sparse = W_sparse + W_sparse.T
        return W_sparse.tocsr()

    # ── Sampling mode (Spielman-Srivastava) ───────────────────────────
    if q is None:
        q = max(int(fraction * m), n)

    sampled_indices = rng.choice(m, size=q, replace=True, p=probs)
    counts = np.bincount(sampled_indices, minlength=m)

    # Reweight: w_tilde_e = w_e * count / (q * p_e)
    selected = counts > 0
    w_new = weights[selected] * counts[selected] / (q * probs[selected])

    W_sparse = sparse.coo_matrix(
        (w_new, (edges_i[selected], edges_j[selected])), shape=(n, n))
    W_sparse = W_sparse + W_sparse.T
    return W_sparse.tocsr()
