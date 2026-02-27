"""Heterogeneous synapse parameter sampling + sparse W construction."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from .config import GeneratorConfig, BLOCK_NAMES, BLOCK_SIGN, STPParams
from .graph_gen import BLOCK_ID_MAP


def sample_synapse_params(cfg: GeneratorConfig, edges: dict, pop: dict,
                          seed: int) -> dict:
    """Sample per-synapse parameters.

    Returns dict with:
        w: float64[] — signed weights (positive for E→*, negative for I→*)
        U: float64[] — STP utilization
        tau_d_ms: float64[] — STP depression time constant
        tau_f_ms: float64[] — STP facilitation time constant
        delay_ms: float64[] — axonal delay
    """
    pre = edges["pre"]
    block_id = edges["block_id"]
    n_syn = len(pre)
    rng = np.random.default_rng(seed)

    w = np.zeros(n_syn, dtype=np.float64)
    U = np.zeros(n_syn, dtype=np.float64)
    tau_d = np.zeros(n_syn, dtype=np.float64)
    tau_f = np.zeros(n_syn, dtype=np.float64)

    block_bases = {
        "EE": cfg.w_EE_base, "EI": cfg.w_EI_base,
        "IE": cfg.w_IE_base, "II": cfg.w_II_base,
    }

    for bname in BLOCK_NAMES:
        bid = BLOCK_ID_MAP[bname]
        mask = block_id == bid
        n_b = mask.sum()
        if n_b == 0:
            continue

        # ── Weights: lognormal around base ──
        sign = BLOCK_SIGN[bname]
        base = block_bases[bname]
        if cfg.w_sigma > 0:
            # lognormal: mean = base, sigma_log = w_sigma
            mag = rng.lognormal(
                mean=np.log(base) - 0.5 * cfg.w_sigma**2,
                sigma=cfg.w_sigma,
                size=n_b,
            )
        else:
            mag = np.full(n_b, base, dtype=np.float64)
        w[mask] = sign * mag

        # ── STP ──
        stp: STPParams = cfg.stp_params[bname]
        U[mask] = np.clip(
            rng.normal(stp.U, stp.sigma * stp.U, n_b), 1e-3, 1.0)
        tau_d[mask] = np.clip(
            rng.normal(stp.tau_d_ms, stp.sigma * stp.tau_d_ms, n_b), 1.0, None)
        tau_f[mask] = np.clip(
            rng.normal(stp.tau_f_ms, stp.sigma * stp.tau_f_ms, n_b), 0.1, None)

    # ── Delays: uniform [delay_min, delay_max] ──
    delay = rng.uniform(cfg.delay_min_ms, cfg.delay_max_ms, n_syn)

    return {
        "w": w,
        "U": U,
        "tau_d_ms": tau_d,
        "tau_f_ms": tau_f,
        "delay_ms": delay,
    }


def rescale_for_balance(syn_params: dict, target: float = 1.0) -> float:
    """Rescale inhibitory weights so |sum_I|/|sum_E| ≈ target.

    Modifies syn_params["w"] in place. Returns actual balance after rescaling.
    """
    w = syn_params["w"]
    sum_E = w[w > 0].sum()
    sum_I_abs = np.abs(w[w < 0]).sum()

    if sum_I_abs < 1e-15 or sum_E < 1e-15:
        return 0.0

    current_balance = sum_I_abs / sum_E
    scale_I = target / current_balance

    w[w < 0] *= scale_I
    return float(np.abs(w[w < 0]).sum() / (w[w > 0].sum() + 1e-15))


def build_sparse_W(N: int, edges: dict, w: np.ndarray) -> sparse.csr_matrix:
    """Build N×N sparse weight matrix from edge list + signed weights."""
    return sparse.csr_matrix(
        (w, (edges["post"], edges["pre"])),
        shape=(N, N),
    )
