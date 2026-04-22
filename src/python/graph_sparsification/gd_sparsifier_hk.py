"""GD sparsifier: heat kernel diagonal loss (unified).

Optimizes edge selection via gradient descent to preserve the heat kernel
diagonal of the original graph. The heat kernel diagonal [exp(-tL)]_ii
encodes per-node return probability under continuous-time random walk and
subsumes degree (order-1 Taylor), 2-hop self-return (order-2 Taylor), and
higher-order spectral structure into a single quantity.

Loss = mean( [(H_diag_t - H_diag_sparse_t) / clamp(H_diag_t, min=mean(H_diag_t)/10)]^2 )

where H = exp(-t * L) with L = D - W (unnormalized Laplacian, G1).
Single time scale t_auto = 2.0 / mean_weighted_degree (clamped to [0.01, 10.0]).
Denominator uses per-node H_diag with an ADAPTIVE floor = mean(H_diag_target)/10.
This caps gradient amplification at 10x (relative to average) for extreme hubs,
while preserving the per-node relative normalization for typical nodes.
  - Fixed floor=0.01: 21x amplification for Exp(30) hubs → config 20 fails
  - Fixed floor=0.05: inflates denominator for Pareto hubs (H_diag~0.03) → configs 44-51 fail
  - Adaptive floor mean/10: ~10x cap, scales with each graph's H_diag distribution

Key design choices:
  G1:  Unnormalized Laplacian L = D - W. Normalized L_sym has eigenvalues
       in [0,2] giving std ~0.008 at N=500 -- near-zero gradients.
  G2:  Hard top-k with DETACHED mask (STE). All soft pruning fails.
  G3:  Proximity rescaling mandatory after extraction.
  G4:  No row-sum loss. Row sums of exp(-tL) are identically 1.
  G6:  Loss uses ONLY heat kernel diagonal -- no SIS, degree, or 2-hop.

Auto-scaling: t = 2.0 / mean_weighted_degree (clamped to [0.01, 10.0]).
"""

import numpy as np
import torch
from scipy import sparse


def _find_mbb_edges(W_dist, rows_idx, cols_idx, n_existing):
    """Identify which upper-triangle edges belong to the metric backbone.

    Returns a boolean tensor of length n_existing.
    """
    from .sparsifiers import metric_backbone

    W_mbb_dist = metric_backbone(W_dist)
    W_mbb_upper = sparse.triu(W_mbb_dist, k=1).tocoo()
    mbb_set = set(zip(W_mbb_upper.row.tolist(), W_mbb_upper.col.tolist()))
    return torch.tensor(
        [
            (rows_idx[i].item(), cols_idx[i].item()) in mbb_set
            for i in range(n_existing)
        ],
        dtype=torch.bool,
    )


def heat_kernel_sparsify_v2(
    W_dist,
    n_edges,
    t=None,
    n_steps=150,
    lr=0.1,
    verbose=False,
):
    """Sparsify a graph using GD on the heat kernel diagonal loss.

    Minimizes normalized MSE of the heat kernel diagonal [exp(-tL)]_ii
    between the original and sparsified graph, where L = D - W is the
    unnormalized Laplacian.

    The heat kernel diagonal subsumes degree, 2-hop, and higher-order
    spectral structure into a single unified loss via its Taylor expansion:
        [exp(-tL)]_ii = 1 - t*d_i + (t^2/2)(d_i^2 + [W^2]_ii) + O(t^3)

    Parameters
    ----------
    W_dist : scipy.sparse matrix
        Weighted adjacency with **distance** weights (costs > 0).
    n_edges : int
        Target number of edges in the upper triangle.
    t : float or None
        Diffusion time parameter. If None (default), auto-scaled to
        2.0 / mean_weighted_degree, clamped to [0.01, 10.0]. Pass a
        float to override the auto-scaling.
    n_steps : int
        Number of gradient descent iterations.
    lr : float
        Adam learning rate.
    verbose : bool
        Print diagnostics at initialization and loss every 50 steps.

    Returns
    -------
    W_result : scipy.sparse.csr_matrix
        Sparsified graph with **proximity** weights, symmetric.
    """
    # -- 1. Convert distance -> proximity ----------------------------------------
    W_dist_csr = sparse.csr_matrix(W_dist, dtype=np.float64)
    n = W_dist_csr.shape[0]

    W_prox = W_dist_csr.copy()
    W_prox.data = 1.0 / (W_prox.data + 1.0)

    W_prox_dense = torch.tensor(W_prox.toarray(), dtype=torch.float64)

    # Upper-triangle mask (existing edges only)
    mask_upper = torch.triu(W_prox_dense > 0, diagonal=1)
    orig_vals = W_prox_dense[mask_upper]
    orig_total = orig_vals.sum().item()  # upper-triangle sum

    rows_idx, cols_idx = torch.where(mask_upper)
    n_existing = rows_idx.shape[0]

    if n_existing == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)

    # Clamp n_edges to available edges
    n_edges = min(n_edges, n_existing)

    # -- 2. Auto-scale t ---------------------------------------------------------
    with torch.no_grad():
        mean_wd = float(W_prox_dense.sum()) / n  # mean weighted degree

    if t is None:
        t_auto = 2.0 / max(mean_wd, 1e-6)
        t_auto = max(0.01, min(10.0, t_auto))
    else:
        t_auto = float(t)

    # -- 3. Compute degree heterogeneity and MBB bias ----------------------------
    with torch.no_grad():
        degree_target = W_prox_dense.sum(dim=1)
        deg_cv = degree_target.std().item() / (degree_target.mean().item() + 1e-10)
        # Adaptive MBB bias: stronger when degree heterogeneity is high
        mbb_bias = max(0.5, min(3.0, deg_cv * 2.0))

    # Find metric backbone edges for initialization bias
    is_mbb = _find_mbb_edges(W_dist_csr, rows_idx, cols_idx, n_existing)

    # -- 4. Precompute heat kernel diagonal target (constant) --------------------
    with torch.no_grad():
        L_target = torch.diag(W_prox_dense.sum(dim=1)) - W_prox_dense
        H_diag_target = torch.matrix_exp(-t_auto * L_target).diagonal().detach()

    if verbose:
        n_mbb_edges = int(is_mbb.sum().item())
        hk_cv = float(H_diag_target.std() / (H_diag_target.mean() + 1e-10))
        print(
            f"  HKv2: n={n}, edges={n_existing}, budget={n_edges}, "
            f"t_auto={t_auto:.4f}, HK_diag std/mean={hk_cv:.4f}, "
            f"deg_cv={deg_cv:.2f}, mbb_bias={mbb_bias:.2f}, "
            f"n_mbb_in_pool={n_mbb_edges}"
        )

    # -- 5. Initialize logits with MBB bias --------------------------------------
    with torch.no_grad():
        p_init = orig_vals.clamp(1e-6, 1 - 1e-6)
        logit_init = torch.logit(p_init)
        logit_init[is_mbb] += mbb_bias

    param_logit = logit_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([param_logit], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps
    )

    # -- 6. GD loop --------------------------------------------------------------
    for step in range(n_steps):
        optimizer.zero_grad()

        prox_vals = torch.sigmoid(param_logit)

        # Hard top-k with STE (G2: mask is DETACHED)
        if prox_vals.numel() > n_edges:
            threshold = torch.topk(prox_vals, n_edges).values[-1].item()
            hard_mask = (prox_vals >= threshold).float().detach()
        else:
            hard_mask = torch.ones_like(prox_vals)

        prox_sparse = prox_vals * hard_mask

        # Rescale to preserve total proximity (G3 in-loop rescaling)
        sparse_total = prox_sparse.sum() + 1e-10
        prox_rescaled = prox_sparse * (orig_total / sparse_total)

        # Build symmetric dense matrix
        W_sparse = torch.zeros((n, n), dtype=torch.float64)
        W_sparse[rows_idx, cols_idx] = prox_rescaled
        W_sparse = W_sparse + W_sparse.T

        # -- Loss: heat kernel diagonal (G6: ONLY HK diagonal)
        L_sparse = torch.diag(W_sparse.sum(dim=1)) - W_sparse
        H_diag_sparse = torch.matrix_exp(-t_auto * L_sparse).diagonal()

        floor = (H_diag_target.mean() / 10.0).clamp(min=1e-6)
        denom = H_diag_target.clamp(min=floor.item())
        loss = (
            ((H_diag_target - H_diag_sparse) / denom)
            .pow(2)
            .mean()
        )

        loss.backward()

        # Gradient clipping
        with torch.no_grad():
            if param_logit.grad is not None:
                param_logit.grad.clamp_(-1.0, 1.0)

        optimizer.step()
        scheduler.step()

        if verbose and (step % 50 == 0 or step == n_steps - 1):
            with torch.no_grad():
                n_active = int((hard_mask > 0.5).sum().item())
            print(
                f"  HKv2 step {step:4d}: loss={loss.item():.6f}, "
                f"n_active={n_active}"
            )

    # -- 7. Extract top-k edges and rescale proximity (G3) -----------------------
    with torch.no_grad():
        prox_final = torch.sigmoid(param_logit)

        if prox_final.numel() > n_edges:
            thresh = torch.topk(prox_final, n_edges).values[-1].item()
            keep = prox_final >= thresh
        else:
            keep = torch.ones(prox_final.numel(), dtype=torch.bool)

        # Handle ties: trim to exact n_edges
        keep_idx = torch.where(keep)[0]
        if keep_idx.numel() > n_edges:
            vals_at_keep = prox_final[keep_idx]
            _, sorted_order = vals_at_keep.sort(descending=True)
            drop = sorted_order[n_edges:]
            keep[keep_idx[drop]] = False

        prox_kept = prox_final[keep].numpy()

        # Rescale: 2 * sum(prox_out_rescaled) == W_prox.data.sum()
        kept_sum = prox_kept.sum() + 1e-10
        scale = orig_total / kept_sum
        prox_out = prox_kept * scale

        rows_out = rows_idx[keep].numpy()
        cols_out = cols_idx[keep].numpy()

    # -- 8. Build symmetric sparse result ----------------------------------------
    W_result = sparse.coo_matrix(
        (prox_out, (rows_out, cols_out)), shape=(n, n)
    )
    W_result = W_result + W_result.T
    W_result = W_result.tocsr()

    # Correctness checks
    result_nnz = sparse.triu(W_result, k=1).nnz
    result_total = W_result.data.sum()
    expected_total = W_prox.data.sum()

    if result_nnz != n_edges:
        raise RuntimeError(
            f"Edge count mismatch: got {result_nnz}, expected {n_edges}"
        )

    rel_err = abs(result_total - expected_total) / (expected_total + 1e-10)
    if rel_err > 0.01:
        raise RuntimeError(
            f"Proximity sum mismatch: relative error {rel_err:.4f} > 0.01 "
            f"(got {result_total:.4f}, expected {expected_total:.4f})"
        )

    return W_result
