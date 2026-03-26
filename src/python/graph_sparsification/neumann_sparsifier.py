"""Neumann series-based graph sparsification.

Core idea
---------
For a strictly upper-triangular matrix A, the Neumann series
S = (I - A)^{-1} = I + A + A^2 + ... encodes *all paths* in the DAG.
Edge importance is derived from the Sherman-Morrison sensitivity of S:

    imp_e = a_e^2 * ||S[:,src_e]||^2 * ||S[dst_e,:]||^2

Since DAG ordering is arbitrary, we average over random node permutations.

The pipeline has three pluggable stages:
  1. **Transform**: distances -> Neumann weights a_e in [0, 1)
  2. **Scoring**: Neumann importance -> edge score for selection
  3. **Rescaling**: adjust proximity weights after edge selection

Each stage is a plain function; swap them to experiment.
"""

import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Distance → Neumann weight transforms
# ═══════════════════════════════════════════════════════════════════════

def transform_proximity(dists, edge_i, edge_j, n, max_row_sum=3.0,
                        n_samples=5, seed=0):
    """Use proximity 1/(d+1) directly as Neumann weights.

    Robust to arbitrary distance distributions (including infinite-mean)
    because proximity is bounded in [0, 1).
    """
    a = 1.0 / (dists + 1.0)
    return _cap_row_sum(a, edge_i, edge_j, n, max_row_sum, n_samples, seed)


def transform_exp(dists, edge_i, edge_j, n, max_row_sum=3.0,
                  target_mean=0.1, n_samples=5, seed=0):
    """Classic exp(-d/tau) transform with mean-calibrated tau.

    Works well when mean distance is finite. Collapses for heavy-tailed
    distributions (LogLogN(2,0.8)) where mean >> median.
    """
    mean_d = dists.mean()
    tau = -mean_d / np.log(np.clip(target_mean, 1e-6, 1 - 1e-6))
    a = np.exp(-dists / tau)
    return _cap_row_sum(a, edge_i, edge_j, n, max_row_sum, n_samples, seed)


def _cap_row_sum(a, edge_i, edge_j, n, max_row_sum, n_samples, seed):
    """Scale weights so the max row sum of the upper-triangular A is bounded."""
    rng = np.random.default_rng(seed)
    max_rs = 0.0
    for _ in range(n_samples):
        perm = rng.permutation(n)
        rows = np.minimum(perm[edge_i], perm[edge_j])
        row_sums = np.zeros(n)
        np.add.at(row_sums, rows, a)
        max_rs = max(max_rs, row_sums.max())
    if max_rs > max_row_sum:
        a = a * (max_row_sum / max_rs)
    return a


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Scoring functions (importance → edge score for selection)
# ═══════════════════════════════════════════════════════════════════════

def score_adaptive(full_imp, structural_imp, orig_prox, retention):
    """Default adaptive scoring based on edge retention ratio.

    - Low retention (<25%): full Neumann importance (a^2 * structural).
      The a^2 factor naturally prioritizes high-proximity edges.
    - High retention (>=25%): normalized structural * proximity.
      Structural captures path-centrality; proximity adds SIR relevance.
    """
    if retention > 0.25:
        s_max = structural_imp.max()
        struct_norm = structural_imp / s_max if s_max > 0 else np.ones_like(structural_imp)
        return struct_norm * orig_prox
    return full_imp


def score_full_importance(full_imp, structural_imp, orig_prox, retention):
    """Always use full Neumann importance a^2 * structural."""
    return full_imp


def score_structural_x_prox(full_imp, structural_imp, orig_prox, retention):
    """Always use structural * proximity (ignores a^2 weight factor)."""
    s_max = structural_imp.max()
    struct_norm = structural_imp / s_max if s_max > 0 else np.ones_like(structural_imp)
    return struct_norm * orig_prox


def score_structural_only(full_imp, structural_imp, orig_prox, retention):
    """Pure structural importance (path centrality only)."""
    return structural_imp


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Weight rescaling functions
# ═══════════════════════════════════════════════════════════════════════

def rescale_adaptive(W_sparse, target_row_sums, orig_prox_all, sel_prox,
                     retention):
    """Default adaptive rescaling.

    - Low retention: global rescale + 10% Sinkhorn correction.
    - High retention: capped Sinkhorn.
    """
    if retention > 0.25:
        return sinkhorn_rescale(W_sparse, target_row_sums, max_inflation=3.0)
    rescaled = global_rescale(orig_prox_all, sel_prox)
    # Rebuild with rescaled weights
    W_coo = W_sparse.tocoo()
    W_coo.data[:] = np.concatenate([rescaled, rescaled]) if len(rescaled) * 2 == len(W_coo.data) else W_coo.data
    # Simpler: just use partial sinkhorn on already-built matrix
    W_global = _rebuild_with_global_rescale(W_sparse, orig_prox_all)
    return partial_sinkhorn(W_global, target_row_sums, alpha=0.1)


def rescale_global_only(W_sparse, target_row_sums, orig_prox_all, sel_prox,
                        retention):
    """Global rescaling only (multiply all weights by total_orig / total_sel)."""
    return _rebuild_with_global_rescale(W_sparse, orig_prox_all)


def rescale_sinkhorn_only(W_sparse, target_row_sums, orig_prox_all, sel_prox,
                          retention):
    """Full Sinkhorn rescaling."""
    return sinkhorn_rescale(W_sparse, target_row_sums)


def rescale_global_plus_sinkhorn(W_sparse, target_row_sums, orig_prox_all,
                                  sel_prox, retention, alpha=0.1):
    """Global rescale + partial Sinkhorn blend."""
    W_global = _rebuild_with_global_rescale(W_sparse, orig_prox_all)
    return partial_sinkhorn(W_global, target_row_sums, alpha=alpha)


# ═══════════════════════════════════════════════════════════════════════
# Core computation (not typically swapped)
# ═══════════════════════════════════════════════════════════════════════

def compute_importance(edge_i, edge_j, a_weights, n, n_perms=50, seed=42):
    """Compute Neumann edge importance via random permutations.

    For each permutation, builds S = (I - A)^{-1} using a triangular solve
    and computes the Sherman-Morrison sensitivity for each edge.

    Returns
    -------
    full_importance : array
        a_e^2 * ||S[:,src]||^2 * ||S[dst,:]||^2, averaged over perms.
    structural_importance : array
        ||S[:,src]||^2 * ||S[dst,:]||^2, averaged over perms.
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
        S = solve_triangular(I_n - A, I_n, lower=False, unit_diagonal=True)

        col_norms_sq = (S * S).sum(axis=0)
        row_norms_sq = (S * S).sum(axis=1)
        path_score = col_norms_sq[rows] * row_norms_sq[cols]
        importance += a_weights ** 2 * path_score
        structural += path_score

    return importance / n_perms, structural / n_perms


def build_symmetric_prox(edge_i, edge_j, prox_weights, n):
    """Build a symmetric sparse proximity matrix from upper-triangular edges."""
    W = sparse.coo_matrix(
        (np.concatenate([prox_weights, prox_weights]),
         (np.concatenate([edge_i, edge_j]),
          np.concatenate([edge_j, edge_i]))),
        shape=(n, n),
    ).tocsr()
    return W


def global_rescale(orig_prox_all, selected_prox):
    """Rescale selected proximity weights to match original total."""
    total_orig = orig_prox_all.sum() * 2
    total_sel = selected_prox.sum() * 2
    if total_sel > 0:
        return selected_prox * (total_orig / total_sel)
    return selected_prox.copy()


def sinkhorn_rescale(W_sparse, target_row_sums, max_inflation=None,
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


def partial_sinkhorn(W_sparse, target_row_sums, alpha=0.1, n_iters=50):
    """Blend: (1-alpha)*W + alpha*sinkhorn(W)."""
    if alpha <= 0:
        return W_sparse.copy()
    W_sink = sinkhorn_rescale(W_sparse, target_row_sums, n_iters=n_iters)
    result = W_sparse.multiply(1 - alpha) + W_sink.multiply(alpha)
    return result.tocsr()


def _rebuild_with_global_rescale(W_sparse, orig_prox_all):
    """Apply global rescaling to an already-built symmetric matrix."""
    total_orig = orig_prox_all.sum() * 2
    total_sel = W_sparse.data.sum()
    if total_sel > 0:
        W_out = W_sparse.copy()
        W_out.data[:] = W_out.data * (total_orig / total_sel)
        return W_out
    return W_sparse.copy()


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def neumann_sparsify(W_dist, n_target_edges, *,
                     transform_fn=None,
                     score_fn=None,
                     rescale_fn=None,
                     max_row_sum=3.0,
                     n_perms=50,
                     seed=42,
                     verbose=True):
    """Neumann series sparsification with pluggable stages.

    Parameters
    ----------
    W_dist : scipy.sparse matrix
        Distance-weighted adjacency matrix.
    n_target_edges : int
        Number of upper-triangular edges to keep.
    transform_fn : callable, optional
        (dists, edge_i, edge_j, n, max_row_sum, ...) -> a_weights.
        Default: transform_proximity.
    score_fn : callable, optional
        (full_imp, structural_imp, orig_prox, retention) -> scores.
        Default: score_adaptive.
    rescale_fn : callable, optional
        (W_sparse, target_rs, orig_prox, sel_prox, retention) -> W_out.
        Default: rescale_adaptive.
    max_row_sum : float
        Bound on max row sum of the Neumann A matrix.
    n_perms : int
        Number of random permutations for importance averaging.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    W_sparse : scipy.sparse.csr_matrix
        Sparsified graph with proximity weights.
    """
    if transform_fn is None:
        transform_fn = transform_proximity
    if score_fn is None:
        score_fn = score_adaptive
    if rescale_fn is None:
        rescale_fn = rescale_adaptive

    # ── Extract edges ──────────────────────────────────────────────
    n = W_dist.shape[0]
    W_triu = sparse.triu(W_dist, k=1).tocoo()
    edge_i = W_triu.row.copy()
    edge_j = W_triu.col.copy()
    dists = W_triu.data.copy()
    n_edges = len(dists)
    orig_prox = 1.0 / (dists + 1.0)

    if n_target_edges >= n_edges:
        return build_symmetric_prox(edge_i, edge_j, orig_prox, n)

    retention = n_target_edges / n_edges

    # ── Stage 1: Transform ─────────────────────────────────────────
    a = transform_fn(dists, edge_i, edge_j, n, max_row_sum, seed=seed)
    if verbose:
        print(f"  Transform: a in [{a.min():.4f}, {a.max():.4f}], "
              f"mean={a.mean():.4f}")

    # ── Compute importance ─────────────────────────────────────────
    if verbose:
        print(f"  Computing importance ({n_perms} permutations)...")
    full_imp, structural_imp = compute_importance(
        edge_i, edge_j, a, n, n_perms=n_perms, seed=seed)

    # ── Stage 2: Score & select ────────────────────────────────────
    score = score_fn(full_imp, structural_imp, orig_prox, retention)

    top_k = np.argsort(score)[-n_target_edges:]
    sel_i = edge_i[top_k]
    sel_j = edge_j[top_k]
    sel_prox = orig_prox[top_k]

    if verbose:
        print(f"  Selected {n_target_edges}/{n_edges} edges "
              f"(retention={100 * retention:.1f}%)")

    # ── Stage 3: Rescale ───────────────────────────────────────────
    W_orig_prox = build_symmetric_prox(edge_i, edge_j, orig_prox, n)
    target_rs = np.array(W_orig_prox.sum(axis=1)).ravel()

    W_sparse = build_symmetric_prox(sel_i, sel_j, sel_prox, n)
    W_sparse = rescale_fn(W_sparse, target_rs, orig_prox, sel_prox, retention)

    if verbose:
        actual_rs = np.array(W_sparse.sum(axis=1)).ravel()
        rs_ratio = actual_rs.sum() / max(target_rs.sum(), 1e-10)
        print(f"  Rescaling: total weight ratio = {rs_ratio:.3f}")

    return W_sparse
