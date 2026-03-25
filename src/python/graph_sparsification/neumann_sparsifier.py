"""Neumann series-based graph sparsification.

Core idea: preserve the Neumann series S = (I - A)^{-1} of the original graph
under sparsification. Uses random node permutations to handle the upper-triangular
ordering constraint (each permutation gives a different DAG).

Two operating modes based on edge retention ratio:
- Low retention (<25%): Score by full Neumann importance (a^2 * structural),
  rescale with global + 10% Sinkhorn correction.
- High retention (>=25%): Score by structural Neumann importance * proximity,
  rescale with capped Sinkhorn.
"""

import numpy as np
from scipy import sparse


def _calibrate_transform(dists, edge_i, edge_j, n, max_row_sum,
                          target_mean=0.1, n_samples=5, seed=0):
    """Transform distances to Neumann weights a = exp(-d/tau), then scale
    so the max row sum of the upper-triangular A is bounded.

    tau is calibrated so that the mean weight equals target_mean, giving
    controlled dynamic range regardless of the distance distribution.
    """
    mean_d = dists.mean()
    tau = -mean_d / np.log(np.clip(target_mean, 1e-6, 1 - 1e-6))
    a = np.exp(-dists / tau)

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


def _build_symmetric_prox(edge_i, edge_j, prox_weights, n):
    """Build a symmetric sparse matrix from upper-triangular edges."""
    W = sparse.coo_matrix(
        (np.concatenate([prox_weights, prox_weights]),
         (np.concatenate([edge_i, edge_j]),
          np.concatenate([edge_j, edge_i]))),
        shape=(n, n),
    ).tocsr()
    return W


def _global_rescale(orig_prox_all, selected_prox):
    """Rescale selected proximity weights to match original total."""
    total_orig = orig_prox_all.sum() * 2
    total_sel = selected_prox.sum() * 2
    if total_sel > 0:
        return selected_prox * (total_orig / total_sel)
    return selected_prox.copy()


def _sinkhorn_rescale(W_sparse, target_row_sums, max_inflation=None,
                       n_iters=50):
    """Diagonal scaling D*W*D to match target row sums."""
    W = W_sparse.copy().tocsr().astype(float)
    n = W.shape[0]
    d = np.ones(n)

    for _ in range(n_iters):
        Wd = np.array(W.dot(d)).ravel()
        rs = d * Wd
        safe_rs = np.maximum(rs, 1e-10)
        ratio = np.where(safe_rs > 1e-10,
                         np.maximum(target_row_sums, 0) / safe_rs, 1.0)
        d *= np.sqrt(np.clip(ratio, 0.01, 100.0))

    W_coo = W.tocoo()
    scale = d[W_coo.row] * d[W_coo.col]

    if max_inflation is not None:
        base_factor = target_row_sums.sum() / max(W_sparse.data.sum(), 1e-10)
        scale = np.clip(scale, 0, max_inflation * base_factor)

    W_coo.data = W_coo.data * scale
    return W_coo.tocsr()


def _partial_sinkhorn(W_sparse, target_row_sums, alpha=0.1, n_iters=50):
    """Blend between current weights and Sinkhorn-corrected weights.

    alpha=0: no correction (return input as-is)
    alpha=1: full Sinkhorn
    """
    if alpha <= 0:
        return W_sparse.copy()
    W_sink = _sinkhorn_rescale(W_sparse, target_row_sums)
    result = W_sparse.multiply(1 - alpha) + W_sink.multiply(alpha)
    return result.tocsr()


def _compute_importance(edge_i, edge_j, a_weights, n, n_perms=30, seed=42):
    """Compute Neumann importance for each edge via Sherman-Morrison.

    Returns both the full importance (includes a_e^2 weight factor) and
    the structural importance (path-centrality only, without a_e^2).

    Full:       imp_e = E_perm[ a_e^2 * ||S[:,r_e]||^2 * ||S[c_e,:]||^2 ]
    Structural: str_e = E_perm[ ||S[:,r_e]||^2 * ||S[c_e,:]||^2 ]
    """
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
        S = np.linalg.inv(I_n - A)

        col_norms_sq = (S * S).sum(axis=0)
        row_norms_sq = (S * S).sum(axis=1)
        path_score = col_norms_sq[rows] * row_norms_sq[cols]
        importance += a_weights ** 2 * path_score
        structural += path_score

    return importance / n_perms, structural / n_perms


def neumann_sparsify(W_dist, n_target_edges, max_row_sum=3.0,
                     n_perms=50, seed=42, verbose=True):
    """Neumann series sparsification.

    Computes edge importance via the Sherman-Morrison formula on the
    Neumann resolvent S = (I - A)^{-1}, averaged over random node
    permutations. The algorithm adapts its scoring and weight
    rescaling strategy based on the edge retention ratio.

    Parameters
    ----------
    W_dist : scipy.sparse matrix
        Distance-weighted adjacency matrix.
    n_target_edges : int
        Number of upper-triangular edges to keep.
    max_row_sum : float
        Bound on max row sum of the Neumann A matrix.
    n_perms : int
        Permutations for importance scoring.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    W_sparse : scipy.sparse.csr_matrix
        Sparsified graph with proximity weights.
    """
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

    # Phase 1: Neumann importance scoring
    a = _calibrate_transform(dists, edge_i, edge_j, n, max_row_sum,
                              seed=seed)
    if verbose:
        print(f"  Phase 1: Neumann importance ({n_perms} permutations)...")
    full_imp, structural_imp = _compute_importance(
        edge_i, edge_j, a, n, n_perms=n_perms, seed=seed)

    # Adaptive scoring based on retention
    if retention > 0.25:
        # High retention: structural importance * proximity
        # Structural importance captures path-centrality without weight bias;
        # combined with proximity for SIR relevance.
        s_max = structural_imp.max()
        struct_norm = structural_imp / s_max if s_max > 0 else np.ones(n_edges)
        score = struct_norm * orig_prox
        mode = 'high-retention'
    else:
        # Low retention: full Neumann importance (a^2 * structural)
        # The a^2 factor naturally prioritizes high-proximity edges,
        # which is critical when keeping few edges.
        score = full_imp
        mode = 'low-retention'

    # Phase 2: Edge selection
    top_k = np.argsort(score)[-n_target_edges:]
    sel_i = edge_i[top_k]
    sel_j = edge_j[top_k]
    sel_prox = orig_prox[top_k]

    if verbose:
        print(f"  Selected {n_target_edges} / {n_edges} edges "
              f"({mode}, retention={100*retention:.1f}%)")

    # Phase 3: Weight assignment with adaptive rescaling
    W_orig_prox = _build_symmetric_prox(edge_i, edge_j, orig_prox, n)
    target_rs = np.array(W_orig_prox.sum(axis=1)).ravel()

    if retention > 0.25:
        # Capped Sinkhorn for high retention
        W_sparse = _build_symmetric_prox(sel_i, sel_j, sel_prox, n)
        W_sparse = _sinkhorn_rescale(W_sparse, target_rs, max_inflation=3.0)
        if verbose:
            print(f"  Rescaling: capped Sinkhorn")
    else:
        # Global rescale + light Sinkhorn correction for low retention
        rescaled = _global_rescale(orig_prox, sel_prox)
        W_sparse = _build_symmetric_prox(sel_i, sel_j, rescaled, n)
        W_sparse = _partial_sinkhorn(W_sparse, target_rs, alpha=0.1)
        if verbose:
            print(f"  Rescaling: global + 10% Sinkhorn")

    return W_sparse
