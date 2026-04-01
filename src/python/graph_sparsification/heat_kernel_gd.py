"""Heat-kernel matching sparsification via gradient descent (hard top-k + STE).

Implements the procedure from §2.3–2.6: minimize diagonal MSE between
``exp(-t L_orig)`` and ``exp(-t L_sparse)`` using Adam on edge logits, hard
top-m selection with a straight-through estimator (STE), and proximity
rescaling so the sparse graph preserves total edge weight.

By default, if ``t`` is not passed, use ``t = gamma / beta`` when both SIR-style
``beta`` and ``gamma`` are provided (otherwise ``t = 1``).
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


def _laplacian(W: torch.Tensor) -> torch.Tensor:
    """Unnormalized Laplacian L = D - W (symmetric W)."""
    d = W.sum(dim=-1)
    return torch.diag(d) - W


def _adjacency_from_upper_edges(
    n: int,
    ei: torch.Tensor,
    ej: torch.Tensor,
    w_upper: torch.Tensor,
) -> torch.Tensor:
    """Build symmetric adjacency from upper-triangle edge list (one entry per undirected edge)."""
    W = torch.zeros((n, n), dtype=w_upper.dtype, device=w_upper.device)
    W[ei, ej] = w_upper
    W[ej, ei] = w_upper
    return W


def _ste_hard_topk_weights(
    pi: torch.Tensor,
    m: int,
    sum_orig: torch.Tensor,
) -> torch.Tensor:
    """Hard top-m by proximity, then rescale so sum of sparse weights equals sum_orig.

    Forward uses hard selection + rescaling. STE: ``pi + (w_fwd - pi).detach()`` so
    backward flows through soft ``pi`` as if the selection were identity.
    """
    e = pi.numel()
    k = min(int(m), e)
    if k <= 0:
        return torch.zeros_like(pi)

    _, idx = torch.topk(pi, k)
    w_hard = torch.zeros_like(pi)
    w_hard.scatter_(0, idx, pi[idx])

    denom = w_hard.sum().clamp_min(torch.finfo(pi.dtype).eps)
    scale = sum_orig / denom
    w_fwd = w_hard * scale

    return pi + (w_fwd - pi).detach()


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
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Sparsify an undirected weighted graph by GD on the heat-kernel diagonal.

    Parameters
    ----------
    W_orig
        Symmetric nonnegative proximity / adjacency (``scipy.sparse`` or dense ``numpy``).
    m
        Number of undirected edges to keep (upper-triangle count).
    t
        Heat-kernel time. If ``None``, set from SIR parameters as ``t = gamma / beta``
        when both ``beta`` and ``gamma`` are given; otherwise defaults to ``1.0``.
    beta, gamma
        SIR-style rates; when ``t`` is ``None`` and both are set, ``t = gamma / beta``
        (mean infectious period in units where infection is scaled by ``beta`` on edge weights).
    n_steps
        Adam steps (default 300 as in the write-up).
    lr
        Adam learning rate (default 0.05).
    edge_candidates
        Optional ``(2, E)`` int array of candidate undirected edges ``(i, j)`` with ``i < j``.
        Default: all upper-triangular positions where ``W_orig > 0``.
    device
        Torch device; default CPU.
    dtype
        Floating dtype for optimization (``float64`` recommended for ``matrix_exp``).
    rng_seed
        RNG seed for initial logits.
    return_history
        If True, ``info['loss_history']`` lists the loss each step.

    Returns
    -------
    W_sparse : scipy.sparse.csr_matrix
        Symmetric sparse weights on the selected top-``m`` edges after training,
        **rescaled** so total edge weight matches the original (sum over upper triangle).
    info : dict
        ``final_loss``, ``loss_history`` (optional), ``edge_index`` ``(2, m)`` numpy int64
        for chosen edges, ``sum_orig``, ``sum_sparse`` (post-rescale, should match),
        and ``t`` (heat-kernel time used).
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

    if edge_candidates is None:
        tri = sparse.triu(W_csr, k=1, format="coo")
        ei_np = tri.row.astype(np.int64, copy=False)
        ej_np = tri.col.astype(np.int64, copy=False)
    else:
        ec = np.asarray(edge_candidates, dtype=np.int64)
        if ec.ndim != 2:
            raise ValueError("edge_candidates must be 2D")
        if ec.shape[0] == 2:
            ei_np, ej_np = ec[0], ec[1]
        elif ec.shape[1] == 2:
            ei_np, ej_np = ec[:, 0], ec[:, 1]
        else:
            raise ValueError("edge_candidates must have shape (2, E) or (E, 2)")
        if np.any(ei_np >= n) or np.any(ej_np >= n):
            raise ValueError("edge_candidates indices out of range")
        if np.any(ei_np >= ej_np):
            raise ValueError("edge_candidates must use i < j for each undirected edge")

    if ei_np.size == 0:
        raise ValueError("no candidate edges")

    w0 = np.asarray(W_csr[ei_np, ej_np]).ravel().astype(np.float64, copy=False)
    sum_orig = float(np.sum(w0))

    if m == 0:
        W_out = sparse.csr_matrix((n, n), dtype=float)
        info: dict[str, Any] = {
            "final_loss": float("nan"),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "sum_orig": sum_orig,
            "sum_sparse": 0.0,
            "t": t_use,
        }
        if return_history:
            info["loss_history"] = []
        return W_out, info

    if m > ei_np.size:
        raise ValueError(f"m={m} exceeds number of candidate edges {ei_np.size}")

    dev = torch.device(device or "cpu")
    tor_dtype = dtype

    ei = torch.from_numpy(ei_np).to(dev)
    ej = torch.from_numpy(ej_np).to(dev)
    E = ei.numel()

    W_full_np = W_csr.toarray()
    W_target = torch.tensor(W_full_np, dtype=tor_dtype, device=dev)
    L_target = _laplacian(W_target)
    with torch.no_grad():
        H_target = torch.linalg.matrix_exp(-t_use * L_target)

    sum_orig_t = torch.tensor(sum_orig, dtype=tor_dtype, device=dev)

    g = torch.Generator(device=dev)
    if rng_seed is not None:
        g.manual_seed(int(rng_seed))
    theta = torch.randn(E, dtype=tor_dtype, device=dev, generator=g) * 0.01
    theta.requires_grad_(True)

    opt = torch.optim.Adam([theta], lr=float(lr))

    loss_history: list[float] = []
    for _ in range(int(n_steps)):
        opt.zero_grad(set_to_none=True)
        pi = torch.sigmoid(theta)
        w_edge = _ste_hard_topk_weights(pi, m, sum_orig_t)
        W_s = _adjacency_from_upper_edges(n, ei, ej, w_edge)
        L_s = _laplacian(W_s)
        H_s = torch.linalg.matrix_exp(-t_use * L_s)
        loss = ((torch.diag(H_target) - torch.diag(H_s)) ** 2).sum()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        pi = torch.sigmoid(theta)
        _, idx = torch.topk(pi, min(m, E))
        w_hard = torch.zeros(E, dtype=tor_dtype, device=dev)
        w_hard.scatter_(0, idx, pi[idx])
        denom = w_hard.sum().clamp_min(torch.finfo(tor_dtype).eps)
        scale = sum_orig_t / denom
        w_final = w_hard * scale

    w_np = w_final.cpu().numpy()
    idx_np = idx.cpu().numpy()
    sel_i = ei_np[idx_np]
    sel_j = ej_np[idx_np]
    w_sel = w_np[idx_np]

    rows = np.concatenate([sel_i, sel_j])
    cols = np.concatenate([sel_j, sel_i])
    data = np.concatenate([w_sel, w_sel])
    W_out = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W_out.eliminate_zeros()

    sum_sparse = float(sparse.triu(W_out, k=1).sum())

    info = {
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "edge_index": np.stack([sel_i, sel_j], axis=0),
        "sum_orig": sum_orig,
        "sum_sparse": sum_sparse,
        "t": t_use,
    }
    if return_history:
        info["loss_history"] = loss_history

    return W_out, info
