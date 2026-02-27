"""Spectral radius and block-level diagnostics."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


def spectral_radius(W: sparse.csr_matrix) -> float:
    """Compute spectral radius max|lambda_i| of sparse matrix W.

    Primary: ARPACK via scipy.sparse.linalg.eigs(W, k=1, which='LM').
    Fallback: power iteration on |W| (non-negative matrix → Perron-Frobenius
    eigenvalue = spectral radius of |W|, which upper-bounds rho(W)).
    """
    if W.nnz == 0:
        return 0.0
    n = W.shape[0]
    if n < 3:
        # For tiny matrices, use dense eigenvalue computation
        eigvals = np.linalg.eigvals(W.toarray())
        return float(np.max(np.abs(eigvals)))
    try:
        # Deterministic start vector for reproducibility
        v0 = np.ones(n, dtype=np.float64)
        v0 /= np.linalg.norm(v0)
        vals = eigs(W.astype(np.float64), k=1, which='LM', v0=v0,
                    return_eigenvectors=False, maxiter=1000, tol=1e-8)
        rho = float(np.max(np.abs(vals)))
        if np.isfinite(rho):
            return rho
    except Exception:
        pass
    # Fallback: power iteration on |W|
    return _power_iter_abs(W, n_iter=300)


def _power_iter_abs(W: sparse.csr_matrix, n_iter: int = 300) -> float:
    """Power iteration on |W| (element-wise absolute value).

    For a non-negative matrix, power iteration converges to the
    Perron-Frobenius eigenvalue = spectral radius.
    """
    W_abs = W.copy()
    W_abs.data = np.abs(W_abs.data)
    n = W_abs.shape[0]
    x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)
    for _ in range(n_iter):
        y = W_abs @ x
        norm = np.linalg.norm(y)
        if norm < 1e-15:
            return 0.0
        x = y / norm
    # Rayleigh quotient on |W|
    y = W_abs @ x
    return float(np.linalg.norm(y))


def block_stats(W: sparse.csr_matrix, pop: dict) -> dict:
    """Compute block-level diagnostics.

    Returns dict with:
        rho_full: spectral radius of full W
        rho_EE: spectral radius of E→E block
        norm_EE, norm_EI, norm_IE, norm_II: Frobenius norms
        balance: |sum_I| / |sum_E|
    """
    N = W.shape[0]
    E_idx = pop["E_idx"]
    I_idx = pop["I_idx"]

    rho_full = spectral_radius(W)

    # E→E sub-block
    W_EE = _extract_block(W, E_idx, E_idx, N)
    rho_EE = spectral_radius(W_EE) if W_EE.nnz > 0 else 0.0

    # Frobenius norms
    W_EI = _extract_block(W, E_idx, I_idx, N)
    W_IE = _extract_block(W, I_idx, E_idx, N)
    W_II = _extract_block(W, I_idx, I_idx, N)

    norm_EE = float(sparse.linalg.norm(W_EE, 'fro'))
    norm_EI = float(sparse.linalg.norm(W_EI, 'fro'))
    norm_IE = float(sparse.linalg.norm(W_IE, 'fro'))
    norm_II = float(sparse.linalg.norm(W_II, 'fro'))

    # E/I balance
    d = W.data
    sum_E = d[d > 0].sum() if len(d[d > 0]) > 0 else 0.0
    sum_I = np.abs(d[d < 0]).sum() if len(d[d < 0]) > 0 else 0.0
    balance = float(sum_I / (sum_E + 1e-15))

    return {
        "rho_full": float(rho_full),
        "rho_EE": float(rho_EE),
        "norm_EE": norm_EE,
        "norm_EI": norm_EI,
        "norm_IE": norm_IE,
        "norm_II": norm_II,
        "balance": balance,
    }


def _extract_block(W: sparse.csr_matrix, src_idx, tgt_idx, N) -> sparse.csr_matrix:
    """Extract sub-block: rows=tgt_idx, cols=src_idx, keep full NxN shape."""
    mask_row = np.zeros(N, dtype=np.bool_)
    mask_col = np.zeros(N, dtype=np.bool_)
    mask_row[tgt_idx] = True
    mask_col[src_idx] = True

    coo = W.tocoo()
    keep = mask_row[coo.row] & mask_col[coo.col]
    return sparse.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=(N, N),
    )
