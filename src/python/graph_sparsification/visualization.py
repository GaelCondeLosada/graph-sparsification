"""Visualization utilities for graph sparsification research."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ── Community detection (for plotting only) ───────────────────────────

def detect_communities(W, n_clusters=None, max_clusters=10):
    """Detect communities via spectral clustering on the graph Laplacian.

    This is a lightweight, sklearn-free implementation used exclusively
    for reordering nodes in adjacency-matrix plots.  It does NOT modify
    the graph or any other data structure.

    Parameters
    ----------
    W : scipy.sparse matrix
        Weighted adjacency matrix (symmetric).
    n_clusters : int or None
        Number of communities.  If None, auto-selected by the largest
        eigengap in the first *max_clusters* eigenvalues of the
        normalized Laplacian.
    max_clusters : int
        Upper bound when auto-selecting the number of clusters.

    Returns
    -------
    labels : np.ndarray of int, shape (n,)
        Community assignment for each node.
    """
    W = sparse.csr_matrix(W, dtype=float)
    n = W.shape[0]

    if n <= 2:
        return np.zeros(n, dtype=int)

    # Normalized Laplacian: L_sym = D^{-1/2} L D^{-1/2}
    degrees = np.array(W.sum(axis=1)).ravel()
    degrees = np.maximum(degrees, 1e-12)  # avoid division by zero
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L = laplacian(W, normed=False)
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # Compute a few smallest eigenvalues/vectors
    k_compute = min(max_clusters + 1, n - 1)
    if n <= 500:
        # Dense eigen for small graphs
        L_dense = L_sym.toarray() if sparse.issparse(L_sym) else L_sym
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        eigenvalues = eigenvalues[:k_compute]
        eigenvectors = eigenvectors[:, :k_compute]
    else:
        from scipy.sparse.linalg import eigsh
        eigenvalues, eigenvectors = eigsh(
            L_sym, k=k_compute, which='SM', tol=1e-6)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    # Auto-select k via largest eigengap
    if n_clusters is None:
        gaps = np.diff(eigenvalues[:max_clusters])
        # Skip the first eigenvalue (always ~0); look at gaps[1:]
        if len(gaps) > 1:
            n_clusters = int(np.argmax(gaps[1:]) + 2)
        else:
            n_clusters = 2
        n_clusters = max(2, min(n_clusters, max_clusters))

    # Spectral embedding: first k eigenvectors (skip constant eigenvector)
    X = eigenvectors[:, :n_clusters].copy()
    # Row-normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = X / norms

    # K-means clustering (simple Lloyd's algorithm)
    labels = _kmeans(X, n_clusters, max_iter=50, n_init=5)
    return labels


def _kmeans(X, k, max_iter=50, n_init=5):
    """Minimal k-means (Lloyd's algorithm) for spectral clustering."""
    n, d = X.shape
    rng = np.random.default_rng(0)
    best_labels = np.zeros(n, dtype=int)
    best_inertia = np.inf

    for _ in range(n_init):
        # k-means++ initialization
        centers = np.empty((k, d))
        centers[0] = X[rng.integers(n)]
        for c in range(1, k):
            dists = np.min(
                np.sum((X[:, None, :] - centers[None, :c, :]) ** 2, axis=2),
                axis=1)
            probs = dists / dists.sum()
            centers[c] = X[rng.choice(n, p=probs)]

        for _ in range(max_iter):
            # Assign
            dists = np.sum(
                (X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            # Update
            new_centers = np.empty_like(centers)
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    new_centers[c] = members.mean(axis=0)
                else:
                    new_centers[c] = X[rng.integers(n)]
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        inertia = sum(
            np.sum((X[labels == c] - centers[c]) ** 2)
            for c in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


# ── Adjacency matrix plots ────────────────────────────────────────────

def plot_adjacency_comparison(W_original, W_sparse, labels=("Original", "Sparsified"),
                              communities="auto", figsize=(14, 6), log_scale=True):
    """Plot adjacency matrices of two graphs side by side.

    Parameters
    ----------
    W_original, W_sparse : scipy.sparse matrix
        Adjacency matrices.
    labels : tuple of str
        Labels for the two graphs.
    communities : array-like, "auto", or None
        Community assignments for node reordering.
        - ``"auto"``: detect communities via spectral clustering on
          ``W_original`` (default).
        - ``None``: no reordering.
        - array-like: explicit community labels.
    figsize : tuple
    log_scale : bool
        Use log scale for colorbar.
    """
    if isinstance(communities, str) and communities == "auto":
        communities = detect_communities(W_original)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if communities is not None:
        order = np.argsort(communities)
    else:
        order = np.arange(W_original.shape[0])

    for ax, W, label in zip(axes, [W_original, W_sparse], labels):
        A = W.toarray() if sparse.issparse(W) else np.asarray(W)
        A = A[np.ix_(order, order)]

        # Count edges
        nnz = np.count_nonzero(A) // 2  # symmetric, so divide by 2

        mask = A > 0
        vmin = A[mask].min() if mask.any() else 1e-10
        vmax = A[mask].max() if mask.any() else 1

        display = np.where(mask, A, np.nan)

        if log_scale and mask.any():
            im = ax.imshow(display, cmap='viridis', aspect='equal',
                           norm=LogNorm(vmin=vmin, vmax=vmax),
                           interpolation='none')
        else:
            im = ax.imshow(display, cmap='viridis', aspect='equal',
                           interpolation='none')

        ax.set_title(f"{label}\n({nnz} edges)", fontsize=12)
        ax.set_xlabel("Node index")
        ax.set_ylabel("Node index")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Edge weight")

    plt.tight_layout()
    return fig


# ── Infection probability plots ───────────────────────────────────────

def plot_infection_comparison(prob_original, prob_sparse,
                              labels=("Original", "Sparsified"),
                              figsize=(7, 7)):
    """Scatter plot comparing infection probabilities (one point per node).

    Reproduces the style of Fig. 2 in Mercier et al. (2022).

    Parameters
    ----------
    prob_original : array-like
        Infection probability per node on the original graph.
    prob_sparse : array-like
        Infection probability per node on the sparsified graph.
    labels : tuple of str
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    prob_original = np.asarray(prob_original)
    prob_sparse = np.asarray(prob_sparse)

    # Mean squared error on finite pairs
    mask = np.isfinite(prob_original) & np.isfinite(prob_sparse)
    po = prob_original[mask]
    ps = prob_sparse[mask]

    if len(po) > 0:
        mse = float(np.mean((po - ps) ** 2))
    else:
        mse = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle="-", alpha=0.4)

    ax.scatter(po, ps, alpha=0.4, s=40, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='y = x')

    ax.set_xlabel(f"{labels[0]} Node Infection Probability", fontsize=12)
    ax.set_ylabel(f"{labels[1]} Node Infection Probability", fontsize=12)
    ax.set_title(f"Infection Probability Comparison\nMSE = {mse:.5f}", fontsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)

    plt.tight_layout()
    return fig


def plot_multi_infection_comparison(prob_original, sparsified_probs,
                                    sparsified_labels, figsize=None):
    """Compare infection probabilities across multiple sparsifiers.

    Parameters
    ----------
    prob_original : array-like
        Infection probabilities on the original graph.
    sparsified_probs : list of array-like
        Infection probabilities for each sparsifier.
    sparsified_labels : list of str
        Labels for each sparsifier.
    figsize : tuple or None

    Returns
    -------
    fig : matplotlib Figure
    """
    n_sparse = len(sparsified_probs)
    if figsize is None:
        figsize = (6 * n_sparse, 6)

    fig, axes = plt.subplots(1, n_sparse, figsize=figsize)
    if n_sparse == 1:
        axes = [axes]

    prob_original = np.asarray(prob_original)

    for ax, ps, label in zip(axes, sparsified_probs, sparsified_labels):
        ps = np.asarray(ps)
        mask = np.isfinite(prob_original) & np.isfinite(ps)
        po_m = prob_original[mask]
        ps_m = ps[mask]

        if len(po_m) > 0:
            mse = float(np.mean((po_m - ps_m) ** 2))
        else:
            mse = 0.0

        ax.scatter(po_m, ps_m, alpha=0.4, s=40, edgecolors='none')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1)

        ax.set_xlabel("Original Infection Prob.", fontsize=11)
        ax.set_ylabel(f"{label} Infection Prob.", fontsize=11)
        ax.set_title(f"{label}\nMSE = {mse:.5f}", fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.grid(True, linestyle="-", alpha=0.4)

    plt.tight_layout()
    return fig
