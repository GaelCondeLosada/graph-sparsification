"""Heat-kernel matching sparsification via gradient descent.

Matches the C2 implementation in ``heat_kernel_GD_example.py`` (``run_c2_heat_kernel``):
hard top-k by threshold with detached mask (STE), proximity rescaling to preserve
upper-triangle mass, ``H(t)=exp(-tL)`` diagonal MSE, Adam + cosine LR schedule,
and logit initialization from current edge proximities.

Heat-kernel time: pass ``t`` explicitly, or set ``t = gamma / beta`` from ``beta`` and
``gamma``, or default ``t = 1.0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "heat_kernel_gd requires PyTorch. Install with: pip install torch"
    ) from e


def compute_heat_kernel(W: torch.Tensor, t: float) -> torch.Tensor:
    """H(t) = exp(-t * L) where L = D - W (same as ``heat_kernel_GD_example``)."""
    d = W.sum(dim=1)
    D = torch.diag(d)
    L = D - W
    return torch.matrix_exp(-t * L)


def heat_kernel_gd_sparsify(
    W_orig: sparse.spmatrix | np.ndarray,
    m: int,
    t: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    n_steps: int = 300,
    lr: float = 0.05,
    edge_candidates: np.ndarray | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    rng_seed: int | None = None,
    return_history: bool = False,
    grad_clip: float = 1.0,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Sparsify using the same GD loop as ``run_c2_heat_kernel`` in ``heat_kernel_GD_example.py``.

    Parameters
    ----------
    W_orig
        Symmetric nonnegative proximity weights.
    m
        Target number of undirected edges (upper-triangle count); clamped to available candidates.
    t
        Heat time for ``exp(-tL)``. If ``None``, use ``gamma/beta`` when both given, else ``1.0``.
    beta, gamma
        When ``t`` is ``None`` and both set: ``t = gamma / beta`` (requires ``beta > 0``).
    n_steps, lr
        Adam steps and learning rate (example often uses 100 and 0.05).
    edge_candidates
        Optional ``(2, E)`` or ``(E, 2)`` with ``i < j``; restricts optimization to those pairs
        that also have positive weight in ``W_orig``.
    device, dtype
        Torch device and floating dtype (``float64`` recommended).
    rng_seed
        Passed to ``torch.manual_seed`` before optimization (example does not use this).
    return_history
        If True, ``info['loss_history']`` lists the loss each step.
    grad_clip
        Clamp ``param_logit.grad`` to ``[-grad_clip, grad_clip]`` after each backward (default 1.0).
    """
    if m < 0:
        raise ValueError("m must be nonnegative")

    if t is not None:
        t_use = float(t)
    elif beta is not None and gamma is not None:
        b, g = float(beta), float(gamma)
        if b <= 0:
            raise ValueError("beta must be positive when using t = gamma / beta")
        t_use = g / b
    else:
        t_use = 1.0

    W_csr = sparse.csr_matrix(W_orig, dtype=float)
    n = W_csr.shape[0]
    if W_csr.shape[0] != W_csr.shape[1]:
        raise ValueError("W_orig must be square")

    W_csr = (W_csr + W_csr.T) * 0.5
    W_csr.eliminate_zeros()

    W_full_np = np.asarray(W_csr.toarray(), dtype=np.float64)
    sum_orig_full_triu = float(sparse.triu(W_csr, k=1).sum())

    if m == 0:
        W_out = sparse.csr_matrix((n, n), dtype=float)
        info: dict[str, Any] = {
            "final_loss": float("nan"),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "sum_orig": sum_orig_full_triu,
            "sum_sparse": 0.0,
            "t": t_use,
        }
        if return_history:
            info["loss_history"] = []
        return W_out, info

    dev = torch.device(device or "cpu")
    tor_dtype = dtype

    if rng_seed is not None:
        torch.manual_seed(int(rng_seed))

    W_prox_t = torch.tensor(W_full_np, dtype=tor_dtype, device=dev)
    mask_upper = torch.triu(W_prox_t > 0, diagonal=1)

    if edge_candidates is not None:
        ec = np.asarray(edge_candidates, dtype=np.int64)
        if ec.ndim != 2:
            raise ValueError("edge_candidates must be 2D")
        if ec.shape[0] == 2:
            ei_np, ej_np = ec[0], ec[1]
        elif ec.shape[1] == 2:
            ei_np, ej_np = ec[:, 0], ec[:, 1]
        else:
            raise ValueError("edge_candidates must have shape (2, E) or (E, 2)")
        cand_mask = torch.zeros((n, n), dtype=torch.bool, device=dev)
        for ii in range(ei_np.size):
            cand_mask[int(ei_np[ii]), int(ej_np[ii])] = True
        mask_upper = mask_upper & cand_mask

    n_upper = int(mask_upper.sum().item())
    if n_upper == 0:
        raise ValueError("no candidate edges (upper triangle with positive weight)")

    n_budget = min(int(m), n_upper)

    with torch.no_grad():
        H_target = compute_heat_kernel(W_prox_t, t=t_use)
        diag_target = H_target.diagonal()

    with torch.no_grad():
        p_init = W_prox_t[mask_upper].clamp(1e-6, 1.0 - 1e-6)
        logit_init = torch.logit(p_init)

    param_logit = logit_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([param_logit], lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(n_steps))

    rows_idx, cols_idx = torch.where(mask_upper)
    orig_total = float(W_prox_t[mask_upper].sum().item())

    loss_history: list[float] = []
    for _step in range(int(n_steps)):
        opt.zero_grad(set_to_none=True)

        prox_vals = torch.sigmoid(param_logit)

        if prox_vals.numel() > n_budget:
            threshold = torch.topk(prox_vals, n_budget).values[-1].item()
            hard_mask = (prox_vals >= threshold).float().detach()
        else:
            hard_mask = torch.ones_like(prox_vals)

        prox_sparse = prox_vals * hard_mask

        sparse_total = prox_sparse.sum() + 1e-10
        prox_rescaled = prox_sparse * (orig_total / float(sparse_total.item()))

        W_sparse = torch.zeros((n, n), dtype=tor_dtype, device=dev)
        W_sparse[rows_idx, cols_idx] = prox_rescaled
        W_sparse = W_sparse + W_sparse.T

        H_sparse = compute_heat_kernel(W_sparse, t=t_use)
        diag_sparse = H_sparse.diagonal()
        loss = (diag_target - diag_sparse).pow(2).sum()

        loss.backward()

        with torch.no_grad():
            if param_logit.grad is not None:
                param_logit.grad.clamp_(-float(grad_clip), float(grad_clip))

        opt.step()
        sched.step()
        loss_history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        prox_final = torch.sigmoid(param_logit)
        if prox_final.numel() > n_budget:
            thresh = torch.topk(prox_final, n_budget).values[-1].item()
            keep = prox_final >= thresh
        else:
            keep = torch.ones(prox_final.numel(), dtype=torch.bool, device=dev)

        prox_kept = prox_final * keep.float()
        kept_total = prox_kept.sum().item() + 1e-10
        scale = orig_total / kept_total
        prox_out = (prox_kept * scale).cpu().numpy()

        rows_idx_np = rows_idx.cpu().numpy()
        cols_idx_np = cols_idx.cpu().numpy()
        keep_np = keep.cpu().numpy()
        rows_out = rows_idx_np[keep_np]
        cols_out = cols_idx_np[keep_np]
        prox_out = prox_out[keep_np]

    if rows_out.size == 0:
        W_out = sparse.csr_matrix((n, n), dtype=float)
        info = {
            "final_loss": loss_history[-1] if loss_history else float("nan"),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "sum_orig": orig_total,
            "sum_sparse": 0.0,
            "t": t_use,
        }
        if return_history:
            info["loss_history"] = loss_history
        return W_out, info

    data_sym = np.concatenate([prox_out, prox_out])
    rows_sym = np.concatenate([rows_out, cols_out])
    cols_sym = np.concatenate([cols_out, rows_out])
    W_out = sparse.csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n, n))
    W_out.eliminate_zeros()

    sum_sparse = float(sparse.triu(W_out, k=1).sum())

    info = {
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "edge_index": np.stack([rows_out, cols_out], axis=0),
        "sum_orig": orig_total,
        "sum_sparse": sum_sparse,
        "t": t_use,
    }
    if return_history:
        info["loss_history"] = loss_history

    return W_out, info
