"""
GD Sparsification v2 -- Iteration 3
=====================================
Testing the most promising remaining approaches:
  C1: Reproduce v1's R3 (hard top-k + raw proximity + row-sum loss + P=5) — reference baseline
  C2: Hard top-k + heat kernel loss (no permutations!) — exp(-tL) diffusion kernel
  C3: Hard top-k + learnable alpha + resolvent row-sum — jointly learn edge selection + scaling
  C4: Hard-concrete L0 relaxation (Louizos et al. 2018) — last soft pruning attempt

Guardrails:
  G1: NO 1/n in A matrix
  G2: d_max from valid edges only (not 1e30 sentinels)
  G3: Proximity rescaling ESSENTIAL after extraction
  G5: Random permutation before upper-triangular
  G8: F1 kernel compresses A too much — DO NOT USE
  G9: L1 is WRONG for pruning — DO NOT USE
  G10: Hard top-k in forward pass (STE for backward), mask is DETACHED
  G11: Soft mask (Gumbel, budget penalty, temperature) all fail — budget overwhelmed by task loss
"""

import sys
import importlib.util
import json
import time
import traceback
from pathlib import Path

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import igraph as ig
from scipy import sparse

# ── repo root ──────────────────────────────────────────────────────────────────

def _find_repo_root():
    cwd = Path(__file__).resolve().parent
    for base in (cwd, cwd.parent, cwd.parent.parent, cwd.parent.parent.parent,
                 *cwd.parents):
        if (base / "src" / "python" / "graph_sparsification").is_dir():
            return base
    raise RuntimeError("Could not locate src/python/graph_sparsification.")

_ROOT = _find_repo_root()
sys.path.insert(0, str(_ROOT / "src" / "python"))

from graph_sparsification.generators import configuration_model, wsbm
from graph_sparsification.sir import sir_monte_carlo
from graph_sparsification.sparsifiers import (
    distance_to_proximity,
    effective_resistance_sparsify,
    metric_backbone,
    metric_backbone_rescaled,
    proximity_to_distance,
    to_proximity,
)

OUTPUT_DIR = Path(__file__).resolve().parent


# ── igraph helpers ─────────────────────────────────────────────────────────────

def is_sir_cpp_available():
    return importlib.util.find_spec("graph_sparsification._sir_cpp") is not None


def igraph_to_sparse_distance(G):
    n = G.vcount()
    W = sparse.lil_matrix((n, n), dtype=np.float64)
    for e in G.es:
        i, j = e.tuple
        d = float(e["distance_weight"])
        W[i, j] = d
        W[j, i] = d
    return sparse.csr_matrix(W)


def igraph_to_sparse_proximity(G):
    n = G.vcount()
    W = sparse.lil_matrix((n, n), dtype=np.float64)
    for e in G.es:
        i, j = e.tuple
        p = float(e["proximity_weight"])
        W[i, j] = p
        W[j, i] = p
    return sparse.csr_matrix(W)


def sparse_distance_to_igraph(W_dist):
    W_dist = sparse.triu(sparse.csr_matrix(W_dist, dtype=float), k=1)
    coo = W_dist.tocoo()
    n = int(W_dist.shape[0])
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    if not edges:
        return ig.Graph(n=n, directed=False)
    G = ig.Graph(n=n, edges=edges, directed=False)
    d = np.asarray(coo.data, dtype=float)
    p = distance_to_proximity(d)
    G.es["distance_weight"] = d.tolist()
    G.es["proximity_weight"] = p.tolist()
    return normalize_proximity_weights(G)


def normalize_proximity_weights(G):
    prev_scale = 1.0
    try:
        prev_scale = float(G["beta_scale"])
    except (KeyError, TypeError, ValueError):
        pass
    if G.ecount() == 0:
        G["beta_scale"] = prev_scale
        return G
    p = np.asarray(G.es["proximity_weight"], dtype=float)
    mx = float(np.max(p))
    if mx <= 0.0:
        G["beta_scale"] = prev_scale
        return G
    p = p / mx
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    d = proximity_to_distance(p)
    G.es["proximity_weight"] = p.tolist()
    G.es["distance_weight"] = d.tolist()
    G["beta_scale"] = prev_scale * mx
    return G


def get_beta_scale(G):
    try:
        s = float(G["beta_scale"])
        return s if s > 0.0 else 1.0
    except (KeyError, TypeError, ValueError):
        return 1.0


# ── graph generators ───────────────────────────────────────────────────────────

def generate_wsbm(n, pi, B, rho_N=None, Lambda=None, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    pi = np.asarray(pi, dtype=float)
    k = len(pi)
    B = np.asarray(B, dtype=float).copy()
    if rho_N is not None:
        base = np.log(n) / n
        B *= rho_N / base
    W, _z = wsbm(n, k, pi, B, weight_distribution="exponential", Lambda=Lambda, rng=rng)
    return sparse_distance_to_igraph(W)


def generate_config_model(n, degree_distribution, weight_distribution, seed=None):
    rng = np.random.default_rng(seed)

    def deg_sampler(n_, rng_):
        d = np.asarray(degree_distribution(n_), dtype=int)
        return np.maximum(d, 1)

    def weight_sampler(m, rng_):
        return np.asarray(weight_distribution(m), dtype=float)

    W = configuration_model(n, deg_sampler, weight_sampler, rng=rng)
    return sparse_distance_to_igraph(W)


# ── MSE + SIR helpers ─────────────────────────────────────────────────────────

def compute_mse(p_orig, p_sparse, n_bootstrap=2000, rng=None):
    rng = np.random.default_rng(rng)
    p_orig = np.asarray(p_orig, dtype=float).ravel()
    p_sparse = np.asarray(p_sparse, dtype=float).ravel()
    diff2 = (p_orig - p_sparse) ** 2
    mse = float(np.mean(diff2))
    n = diff2.size
    if n == 0:
        return mse, (mse, mse)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boots = diff2[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mse, (float(lo), float(hi))


def edge_dist_stats(dists_sel, label):
    d = np.asarray(dists_sel)
    d = d[(d > 0) & (d < 1e29)]
    if len(d) == 0:
        print(f"      [{label}] edge_dist: EMPTY")
        return
    print(f"      [{label}] n={len(d)}, d_min={d.min():.4f}, d_mean={d.mean():.4f}, d_max={d.max():.4f}")


# ── Proximity selection helper ─────────────────────────────────────────────────

def select_edges_and_build_prox(rows_sel, cols_sel, dists_sel, n, sum_prox_full):
    rows_sel = np.asarray(rows_sel)
    cols_sel = np.asarray(cols_sel)
    dists_sel = np.asarray(dists_sel, dtype=float)
    valid = (dists_sel > 0) & (dists_sel < 1e29)
    rows_sel, cols_sel, dists_sel = rows_sel[valid], cols_sel[valid], dists_sel[valid]
    if len(dists_sel) == 0:
        return sparse.csr_matrix((n, n), dtype=float)
    p_sel = distance_to_proximity(dists_sel)
    scale = sum_prox_full / (2.0 * p_sel.sum())
    p_rescaled = p_sel * scale
    W = sparse.csr_matrix(
        (np.concatenate([p_rescaled, p_rescaled]),
         (np.concatenate([rows_sel, cols_sel]),
          np.concatenate([cols_sel, rows_sel]))),
        shape=(n, n)
    )
    return W


# ══════════════════════════════════════════════════════════════════════════════
# RESOLVENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_resolvent_permuted(A_sym, n_perms, rng_seed=None):
    """
    Compute averaged resolvent over random permutations.
    R = (I - triu(A_perm))^{-1} - I, averaged over random permutations.
    """
    n = A_sym.shape[0]
    I = torch.eye(n, dtype=A_sym.dtype)
    R_accum = torch.zeros_like(A_sym)

    gen = torch.Generator()
    if rng_seed is not None:
        gen.manual_seed(rng_seed)

    for _ in range(n_perms):
        perm = torch.randperm(n, generator=gen)
        inv_perm = torch.argsort(perm)

        A_perm = A_sym[perm][:, perm]
        A_upper = torch.triu(A_perm, diagonal=1)

        X = torch.linalg.solve_triangular(I - A_upper, I, upper=True)
        R_perm = X - I

        R_unperm = R_perm[inv_perm][:, inv_perm]
        R_accum = R_accum + R_unperm

    R_avg = R_accum / n_perms
    return R_avg


# ══════════════════════════════════════════════════════════════════════════════
# C2: Hard top-k + heat kernel loss (NO permutations!)
# ══════════════════════════════════════════════════════════════════════════════

def compute_heat_kernel(W, t=1.0):
    """H(t) = exp(-t * L) where L = D - W is the Laplacian."""
    D = torch.diag(W.sum(dim=1))
    L = D - W
    return torch.matrix_exp(-t * L)


def run_c2_heat_kernel(
    W_prox_np, sir_beta, gamma, n_edges, sum_prox_full,
    n_steps=100, lr=0.05, t_heat=1.0,
):
    """
    Hard top-k STE + heat kernel H(t) = exp(-tL) diagonal loss.
    Heat kernel is symmetric -> no permutation trick needed.
    NOTE: Row sums of exp(-tL) are always exactly 1 (Markov property),
    so we use DIAGONAL entries H_ii (return probabilities) which vary by node.
    """
    W_prox_t = torch.tensor(W_prox_np, dtype=torch.float64)
    n = W_prox_t.shape[0]

    mask_upper = torch.triu(W_prox_t > 0, diagonal=1)
    orig_total = W_prox_t[mask_upper].sum().item()

    print(f"    C2: n={n}, budget={n_edges}, t_heat={t_heat}")

    # Precompute target heat kernel diagonal (return probabilities)
    with torch.no_grad():
        H_target = compute_heat_kernel(W_prox_t, t=t_heat)
        diag_target = H_target.diagonal()
    print(f"    C2: H_target diag: min={diag_target.min():.4f}, mean={diag_target.mean():.4f}, "
          f"max={diag_target.max():.4f}")

    with torch.no_grad():
        p_init = W_prox_t[mask_upper].clamp(1e-6, 1 - 1e-6)
        logit_init = torch.logit(p_init)

    param_logit = logit_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([param_logit], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    rows_idx, cols_idx = torch.where(mask_upper)
    loss_hist = []

    for step in range(n_steps):
        opt.zero_grad()

        prox_vals = torch.sigmoid(param_logit)

        # Hard top-k STE
        if prox_vals.numel() > n_edges:
            threshold = torch.topk(prox_vals, n_edges).values[-1].item()
            hard_mask = (prox_vals >= threshold).float().detach()
        else:
            hard_mask = torch.ones_like(prox_vals)

        prox_sparse = prox_vals * hard_mask

        # Rescale to match original total
        sparse_total = prox_sparse.sum() + 1e-10
        prox_rescaled = prox_sparse * (orig_total / sparse_total)

        # Build symmetric matrix
        W_sparse = torch.zeros((n, n), dtype=torch.float64)
        W_sparse[rows_idx, cols_idx] = prox_rescaled
        W_sparse = W_sparse + W_sparse.T

        # Heat kernel diagonal loss (no permutations needed!)
        H_sparse = compute_heat_kernel(W_sparse, t=t_heat)
        diag_sparse = H_sparse.diagonal()
        loss = (diag_target - diag_sparse).pow(2).sum()

        loss.backward()

        with torch.no_grad():
            if param_logit.grad is not None:
                param_logit.grad.clamp_(-1.0, 1.0)

        opt.step()
        sched.step()
        loss_hist.append(loss.item())

        if step % 50 == 0 or step == n_steps - 1:
            with torch.no_grad():
                n_active = int((hard_mask > 0.5).sum().item())
            print(f"    C2 step {step:4d}: loss={loss.item():.4f}, n_active={n_active}")

    # Extract
    with torch.no_grad():
        prox_final = torch.sigmoid(param_logit)
        if prox_final.numel() > n_edges:
            thresh = torch.topk(prox_final, n_edges).values[-1].item()
            keep = prox_final >= thresh
        else:
            keep = torch.ones(prox_final.numel(), dtype=torch.bool)

        prox_kept = prox_final * keep.float()
        kept_total = prox_kept.sum().item() + 1e-10
        scale = orig_total / kept_total
        prox_out = (prox_kept * scale).numpy()

        rows_out = rows_idx[keep].numpy()
        cols_out = cols_idx[keep].numpy()
        prox_out = prox_out[keep.numpy()]

    return rows_out, cols_out, prox_out, loss_hist


# ══════════════════════════════════════════════════════════════════════════════
# C3: Hard top-k + learnable alpha + resolvent row-sum
# ══════════════════════════════════════════════════════════════════════════════

def run_c3_learnable_alpha(
    W_prox_np, sir_beta, gamma, n_edges, sum_prox_full,
    n_steps=300, lr=0.05, n_perms_target=20, n_perms_step=5,
):
    """
    Like C1/R3 but with learnable scaling: A = sigmoid(alpha_raw) * W_sparse.
    Jointly optimizes edge selection + scaling factor.
    """
    W_prox_t = torch.tensor(W_prox_np, dtype=torch.float64)
    n = W_prox_t.shape[0]

    mask_upper = torch.triu(W_prox_t > 0, diagonal=1)
    orig_total = W_prox_t[mask_upper].sum().item()

    print(f"    C3: n={n}, budget={n_edges}")

    # Compute target with raw proximity (same as C1 for fair comparison)
    with torch.no_grad():
        R_target = compute_resolvent_permuted(W_prox_t, n_perms_target, rng_seed=42)
        rowsum_target = R_target.sum(dim=1)
    print(f"    C3: rowsum_target: min={rowsum_target.min():.4f}, mean={rowsum_target.mean():.4f}, "
          f"max={rowsum_target.max():.4f}")

    with torch.no_grad():
        p_init = W_prox_t[mask_upper].clamp(1e-6, 1 - 1e-6)
        logit_init = torch.logit(p_init)

    param_logit = logit_init.clone().detach().requires_grad_(True)
    # Initialize alpha_raw=0 -> sigmoid(0)=0.5, so A starts at ~half of raw proximity
    alpha_raw = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.Adam([param_logit, alpha_raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    rows_idx, cols_idx = torch.where(mask_upper)
    loss_hist = []
    perm_seed_counter = 5000

    for step in range(n_steps):
        opt.zero_grad()

        alpha = torch.sigmoid(alpha_raw)
        prox_vals = torch.sigmoid(param_logit)

        # Hard top-k STE
        if prox_vals.numel() > n_edges:
            threshold = torch.topk(prox_vals, n_edges).values[-1].item()
            hard_mask = (prox_vals >= threshold).float().detach()
        else:
            hard_mask = torch.ones_like(prox_vals)

        prox_sparse = prox_vals * hard_mask

        # Rescale to match original total
        sparse_total = prox_sparse.sum() + 1e-10
        prox_rescaled = prox_sparse * (orig_total / sparse_total)

        # Build symmetric A with learnable alpha
        W_sparse = torch.zeros((n, n), dtype=torch.float64)
        W_sparse[rows_idx, cols_idx] = prox_rescaled
        W_sparse = W_sparse + W_sparse.T
        A = alpha * W_sparse

        R_sparse = compute_resolvent_permuted(A, n_perms_step, rng_seed=perm_seed_counter + step)

        rowsum_sparse = R_sparse.sum(dim=1)
        loss = (rowsum_target - rowsum_sparse).pow(2).sum()

        loss.backward()

        with torch.no_grad():
            if param_logit.grad is not None:
                param_logit.grad.clamp_(-1.0, 1.0)
            if alpha_raw.grad is not None:
                alpha_raw.grad.clamp_(-1.0, 1.0)

        opt.step()
        sched.step()
        loss_hist.append(loss.item())

        if step % 50 == 0 or step == n_steps - 1:
            with torch.no_grad():
                n_active = int((hard_mask > 0.5).sum().item())
                alpha_val = torch.sigmoid(alpha_raw).item()
            print(f"    C3 step {step:4d}: loss={loss.item():.4f}, n_active={n_active}, alpha={alpha_val:.4f}")

    # Extract — note: we extract edges based on prox_vals, then rescale using proximity rescaling
    # The alpha was only for the resolvent computation during training
    with torch.no_grad():
        prox_final = torch.sigmoid(param_logit)
        if prox_final.numel() > n_edges:
            thresh = torch.topk(prox_final, n_edges).values[-1].item()
            keep = prox_final >= thresh
        else:
            keep = torch.ones(prox_final.numel(), dtype=torch.bool)

        prox_kept = prox_final * keep.float()
        kept_total = prox_kept.sum().item() + 1e-10
        scale = orig_total / kept_total
        prox_out = (prox_kept * scale).numpy()

        rows_out = rows_idx[keep].numpy()
        cols_out = cols_idx[keep].numpy()
        prox_out = prox_out[keep.numpy()]

    final_alpha = torch.sigmoid(alpha_raw).item()
    print(f"    C3: final alpha={final_alpha:.4f}")

    return rows_out, cols_out, prox_out, loss_hist

def build_graphs():
    seed = 42
    n = 500

    # G1: wSBM 3-block
    k1, B_diag, B_off = 3, 8.0, 1.5
    B1 = np.full((k1, k1), B_off)
    np.fill_diagonal(B1, B_diag)
    Lambda_diag, Lambda_off = 0.1, 200.0
    Lambda1 = np.full((k1, k1), Lambda_off)
    np.fill_diagonal(Lambda1, Lambda_diag)
    G1 = generate_wsbm(n, np.ones(k1) / k1, B1, rho_N=4 * np.log(n) / n,
                       Lambda=Lambda1, seed=seed)

    # G2: Configuration model
    G2 = generate_config_model(
        n=n,
        degree_distribution=lambda sz: np.random.default_rng(42).exponential(
            scale=30, size=sz).astype(int),
        weight_distribution=lambda m: np.random.default_rng(42).exponential(
            scale=30, size=m),
        seed=42,
    )

    # G3: wSBM 4-block strong
    k3, B_diag3, B_off3 = 4, 10.0, 1.0
    B3 = np.full((k3, k3), B_off3)
    np.fill_diagonal(B3, B_diag3)
    Lambda_diag3, Lambda_off3 = 0.01, 100.0
    Lambda3 = np.full((k3, k3), Lambda_off3)
    np.fill_diagonal(Lambda3, Lambda_diag3)
    G3 = generate_wsbm(n, np.ones(k3) / k3, B3, rho_N=4 * np.log(n) / n,
                       Lambda=Lambda3, seed=seed)

    return [("G1_wSBM3", G1), ("G2_CM", G2), ("G3_wSBM4", G3)]


# ── SIR parameters ────────────────────────────────────────────────────────────

SIR_GAMMA = 1.0
SIR_T_MAX = 100.0
SIR_N_RUNS = 500
SIR_PATIENT_ZEROS = 0.01
SIR_SEED = 42
SIR_BETAS = {"G1_wSBM3": 0.05, "G2_CM": 1.0, "G3_wSBM4": 0.1}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("Building graphs...")
    graphs = build_graphs()
    for gname, G in graphs:
        print(f"  {gname}: {G.vcount()} nodes, {G.ecount()} edges, "
              f"beta_scale={get_beta_scale(G):.4f}")

    results = {}

    for gname, G_orig in graphs:
        sir_beta = SIR_BETAS[gname]
        print(f"\n{'='*70}")
        print(f"Graph: {gname}  (beta={sir_beta}, gamma={SIR_GAMMA})")

        # Baseline pipeline
        print("  Running baseline pipeline (MBBr + EffR)...")
        W_dist = igraph_to_sparse_distance(G_orig)
        W_prox = to_proximity(W_dist)
        W_mbb_dist = metric_backbone(W_dist)
        n_edges = sparse.triu(W_mbb_dist, k=1).nnz
        if n_edges == 0:
            n_edges = max(1, sparse.triu(W_prox, k=1).nnz // 2)
        W_mbbr_prox = metric_backbone_rescaled(W_dist)
        W_effr = effective_resistance_sparsify(W_prox, n_edges=n_edges, rng=SIR_SEED)

        beta_eff = sir_beta * get_beta_scale(G_orig)

        mc_kw = dict(
            gamma=SIR_GAMMA,
            initial_infected=SIR_PATIENT_ZEROS,
            n_runs=SIR_N_RUNS,
            t_max=SIR_T_MAX,
            rng=SIR_SEED,
            use_cpp=is_sir_cpp_available(),
        )

        def probs(Wp):
            return sir_monte_carlo(Wp, beta_eff, **mc_kw)["infection_prob"]

        p_orig = np.array(probs(W_prox))
        p_mbbr = probs(W_mbbr_prox)
        p_effr = probs(W_effr)

        br = np.random.default_rng(SIR_SEED)
        mse_mbbr, mse_mbbr_CI = compute_mse(p_orig, p_mbbr, rng=br)
        mse_effr, mse_effr_CI = compute_mse(p_orig, p_effr, rng=br)
        sum_prox_full = float(W_prox.data.sum())

        print(f"  n_edges budget: {n_edges}")
        print(f"  sum_prox_full: {sum_prox_full:.4f}")
        print(f"  MBBr MSE: {mse_mbbr:.6f}  EffR MSE: {mse_effr:.6f}")

        results[gname] = {
            "mse_mbbr": mse_mbbr,
            "mse_mbbr_CI": list(mse_mbbr_CI),
            "mse_effr": mse_effr,
            "mse_effr_CI": list(mse_effr_CI),
            "approaches": {},
        }

        # Dense proximity matrix for GD
        W_prox_dense = np.array(W_prox.todense(), dtype=np.float64)

        dist_orig_dense = np.array(
            G_orig.get_adjacency(attribute="distance_weight", default=1e30).data,
            dtype=np.float64,
        )
        n = G_orig.vcount()

        # ── Eval helpers ──────────────────────────────────────────────────
        def eval_approach(name, W_prox_sel):
            if W_prox_sel is None:
                print(f"    [{name}] SKIPPED (None)")
                return {"mse": float("nan"), "status": "skipped"}
            try:
                p_sel = sir_monte_carlo(W_prox_sel, beta_eff, **mc_kw)["infection_prob"]
                mse, mse_CI = compute_mse(p_orig, p_sel, rng=np.random.default_rng(SIR_SEED))
                beats_mbbr = mse < mse_mbbr
                beats_effr = mse < mse_effr
                tag = ""
                if beats_mbbr and beats_effr:
                    tag = " << WIN"
                elif beats_mbbr:
                    tag = " (beats MBBr)"
                elif beats_effr:
                    tag = " (beats EffR)"
                print(f"    [{name}] MSE={mse:.6f}  beats_MBBr={beats_mbbr}  beats_EffR={beats_effr}{tag}")
                return {
                    "mse": mse,
                    "mse_CI": list(mse_CI),
                    "beats_mbbr": beats_mbbr,
                    "beats_effr": beats_effr,
                    "status": "ok",
                }
            except Exception as e:
                msg = str(e)
                print(f"    [{name}] ERROR: {msg[:200]}")
                traceback.print_exc()
                return {"mse": float("nan"), "status": "error", "error": msg}

        def eval_gd_result(name, rows, cols, prox_vals, loss_hist):
            """Build scipy sparse from GD output, eval with SIR."""
            try:
                prox_vals = np.asarray(prox_vals, dtype=float)
                if len(prox_vals) == 0:
                    print(f"    [{name}] ERROR: 0 edges extracted")
                    return {"mse": float("nan"), "status": "error", "error": "0 edges", "n_edges": 0}
                edge_dist_stats(dist_orig_dense[rows, cols], name)
                print(f"    [{name}] prox: min={prox_vals.min():.4f}, "
                      f"mean={prox_vals.mean():.4f}, max={prox_vals.max():.4f}, "
                      f"total={prox_vals.sum():.2f}")

                W_sel = sparse.csr_matrix(
                    (np.concatenate([prox_vals, prox_vals]),
                     (np.concatenate([rows, cols]),
                      np.concatenate([cols, rows]))),
                    shape=(n, n)
                )
                r = eval_approach(name, W_sel)
                r["final_loss"] = loss_hist[-1] if loss_hist else None
                r["n_edges"] = len(rows)
                return r
            except Exception as e:
                msg = str(e)
                print(f"    [{name}] ERROR: {msg[:200]}")
                traceback.print_exc()
                return {"mse": float("nan"), "status": "error", "error": msg}

        # ── C2a: Heat kernel (t=1.0) ────────────────────────────────────
        print(f"\n  --- C2a: Heat kernel (t=1.0, no perms) ---")
        t0 = time.time()
        try:
            rows_c2a, cols_c2a, prox_c2a, loss_c2a = run_c2_heat_kernel(
                W_prox_dense, sir_beta, SIR_GAMMA, n_edges, sum_prox_full,
                n_steps=100, lr=0.05, t_heat=1.0,
            )
            dt = time.time() - t0
            print(f"    C2a time: {dt:.1f}s")
            r = eval_gd_result("C2a", rows_c2a, cols_c2a, prox_c2a, loss_c2a)
            r["time_s"] = dt
            results[gname]["approaches"]["C2a_heat_t1"] = r
        except Exception as e:
            print(f"  [C2a] FAILED: {e}"); traceback.print_exc()
            results[gname]["approaches"]["C2a_heat_t1"] = {"status": "error", "error": str(e)}

        # ── C2b: Heat kernel (t=0.5) ────────────────────────────────────
        print(f"\n  --- C2b: Heat kernel (t=0.5, no perms) ---")
        t0 = time.time()
        try:
            rows_c2b, cols_c2b, prox_c2b, loss_c2b = run_c2_heat_kernel(
                W_prox_dense, sir_beta, SIR_GAMMA, n_edges, sum_prox_full,
                n_steps=100, lr=0.05, t_heat=0.5,
            )
            dt = time.time() - t0
            print(f"    C2b time: {dt:.1f}s")
            r = eval_gd_result("C2b", rows_c2b, cols_c2b, prox_c2b, loss_c2b)
            r["time_s"] = dt
            results[gname]["approaches"]["C2b_heat_t05"] = r
        except Exception as e:
            print(f"  [C2b] FAILED: {e}"); traceback.print_exc()
            results[gname]["approaches"]["C2b_heat_t05"] = {"status": "error", "error": str(e)}

        # ── C3: Learnable alpha ──────────────────────────────────────────
        print(f"\n  --- C3: Hard top-k + learnable alpha + resolvent ---")
        t0 = time.time()
        try:
            rows_c3, cols_c3, prox_c3, loss_c3 = run_c3_learnable_alpha(
                W_prox_dense, sir_beta, SIR_GAMMA, n_edges, sum_prox_full,
                n_steps=300, lr=0.05, n_perms_target=20, n_perms_step=5,
            )
            dt = time.time() - t0
            print(f"    C3 time: {dt:.1f}s")
            r = eval_gd_result("C3", rows_c3, cols_c3, prox_c3, loss_c3)
            r["time_s"] = dt
            results[gname]["approaches"]["C3_learnable_alpha"] = r
        except Exception as e:
            print(f"  [C3] FAILED: {e}"); traceback.print_exc()
            results[gname]["approaches"]["C3_learnable_alpha"] = {"status": "error", "error": str(e)}


    # ── Summary table ──────────────────────────────────────────────────────────
    approach_names = [
        "A1_top_prox",
        "C1_r3_reproduce",
        "C2a_heat_t1",
        "C2b_heat_t05",
        "C3_learnable_alpha",
        "C4_hard_concrete",
    ]
    approach_labels = [
        "A1: TopProx+resc",
        "C1: R3 reproduce (P=5)",
        "C2a: HeatKernel t=1.0",
        "C2b: HeatKernel t=0.5",
        "C3: LearnableAlpha+Res",
        "C4: HardConcrete L0",
    ]

    dt_total = time.time() - t_total
    print(f"\n{'='*120}")
    print(f"RESULTS TABLE (v2 iter3)")
    print(f"Total time: {dt_total:.0f}s")
    print(f"{'='*120}")
    header = f"{'Approach':<32} | {'G1_wSBM3':>10} | {'G2_CM':>10} | {'G3_wSBM4':>10} | {'Wins':>5} | {'Time':>8}"
    print(header)
    print("-"*120)

    mbbr_vals = {gn: results[gn]["mse_mbbr"] for gn in ["G1_wSBM3", "G2_CM", "G3_wSBM4"]}
    effr_vals = {gn: results[gn]["mse_effr"] for gn in ["G1_wSBM3", "G2_CM", "G3_wSBM4"]}
    print(f"{'MBBr (ref)':<32} | {mbbr_vals['G1_wSBM3']:>10.4f} | {mbbr_vals['G2_CM']:>10.4f} | {mbbr_vals['G3_wSBM4']:>10.4f} |  --- |      ---")
    print(f"{'EffR (ref)':<32} | {effr_vals['G1_wSBM3']:>10.4f} | {effr_vals['G2_CM']:>10.4f} | {effr_vals['G3_wSBM4']:>10.4f} |  --- |      ---")
    print("-"*120)

    for aname, alabel in zip(approach_names, approach_labels):
        mses = {}
        times = {}
        for gn in ["G1_wSBM3", "G2_CM", "G3_wSBM4"]:
            ap = results.get(gn, {}).get("approaches", {}).get(aname, {})
            mses[gn] = ap.get("mse", float("nan"))
            times[gn] = ap.get("time_s", None)

        wins = sum(
            1 for gn in ["G1_wSBM3", "G2_CM", "G3_wSBM4"]
            if (not np.isnan(mses[gn]) and
                mses[gn] < results[gn]["mse_mbbr"] and
                mses[gn] < results[gn]["mse_effr"])
        )

        def fmt(v, gn):
            if np.isnan(v):
                return "  ERROR"
            tag = ""
            if v < results[gn]["mse_mbbr"] and v < results[gn]["mse_effr"]:
                tag = "*"
            return f"{v:.4f}{tag}"

        avg_time = np.nanmean([t for t in times.values() if t is not None]) if any(t is not None for t in times.values()) else float("nan")
        time_str = f"{avg_time:.0f}s" if not np.isnan(avg_time) else "---"

        print(f"{alabel:<32} | {fmt(mses['G1_wSBM3'], 'G1_wSBM3'):>10} | "
              f"{fmt(mses['G2_CM'], 'G2_CM'):>10} | "
              f"{fmt(mses['G3_wSBM4'], 'G3_wSBM4'):>10} | {wins:>5} | {time_str:>8}")

    print("="*120)
    print("  * = beats BOTH MBBr and EffR")

    # Wins detail
    print("\nDetailed wins (beats BOTH MBBr and EffR):")
    for aname, alabel in zip(approach_names, approach_labels):
        for gn in ["G1_wSBM3", "G2_CM", "G3_wSBM4"]:
            ap = results.get(gn, {}).get("approaches", {}).get(aname, {})
            mse = ap.get("mse", float("nan"))
            mbbr = results[gn]["mse_mbbr"]
            effr = results[gn]["mse_effr"]
            if not np.isnan(mse) and mse < mbbr and mse < effr:
                print(f"  WIN: {alabel} on {gn}: {mse:.4f} < MBBr={mbbr:.4f}, EffR={effr:.4f}")

    # Save results
    out_path = OUTPUT_DIR / "results_v2_iter3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
