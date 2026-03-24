"""Visualization utilities for graph sparsification research."""

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_adjacency_comparison(W_original, W_sparse, labels=("Original", "Sparsified"),
                              communities=None, figsize=(14, 6), log_scale=True):
    """Plot adjacency matrices of two graphs side by side.

    Parameters
    ----------
    W_original, W_sparse : scipy.sparse matrix
        Adjacency matrices.
    labels : tuple of str
        Labels for the two graphs.
    communities : array-like or None
        Community assignments for node reordering.
    figsize : tuple
    log_scale : bool
        Use log scale for colorbar.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Reorder nodes by community if provided
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

    # Compute R^2
    mask = np.isfinite(prob_original) & np.isfinite(prob_sparse)
    po = prob_original[mask]
    ps = prob_sparse[mask]

    if len(po) > 1:
        ss_res = np.sum((po - ps) ** 2)
        ss_tot = np.sum((po - po.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r2 = 0.0

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(po, ps, alpha=0.3, s=10, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='y = x')

    ax.set_xlabel(f"{labels[0]} Node Infection Probability", fontsize=12)
    ax.set_ylabel(f"{labels[1]} Node Infection Probability", fontsize=12)
    ax.set_title(f"Infection Probability Comparison\n$R^2 = {r2:.3f}$", fontsize=13)
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

        if len(po_m) > 1:
            ss_res = np.sum((po_m - ps_m) ** 2)
            ss_tot = np.sum((po_m - po_m.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            r2 = 0.0

        ax.scatter(po_m, ps_m, alpha=0.3, s=10, edgecolors='none')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1)

        ax.set_xlabel("Original Infection Prob.", fontsize=11)
        ax.set_ylabel(f"{label} Infection Prob.", fontsize=11)
        ax.set_title(f"{label}\n$R^2 = {r2:.3f}$", fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig
