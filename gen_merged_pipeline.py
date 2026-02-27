
########################################
# FILE: gen/__init__.py
########################################

# gen — frozen reservoir bundle generator

########################################
# FILE: gen/config.py
########################################

"""GeneratorConfig — all parameters for frozen bundle generation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


@dataclass
class HeteroParam:
    """One heterogeneous neuron parameter."""
    base: float
    sigma: float          # relative σ (e.g. 0.05 = 5%)
    clamp_lo: float       # multiplier lower bound (e.g. 0.8)
    clamp_hi: float       # multiplier upper bound (e.g. 1.2)


# ── Default neuron heterogeneity (HH model) ──
DEFAULT_NEURON_HETERO: Dict[str, HeteroParam] = {
    "Cm":  HeteroParam(base=1.0,   sigma=0.05, clamp_lo=0.8, clamp_hi=1.2),
    "gL":  HeteroParam(base=5e-5,  sigma=0.05, clamp_lo=0.8, clamp_hi=1.2),
    "gNa": HeteroParam(base=100.0, sigma=0.05, clamp_lo=0.8, clamp_hi=1.2),
    "gK":  HeteroParam(base=30.0,  sigma=0.05, clamp_lo=0.8, clamp_hi=1.2),
    "Vt":  HeteroParam(base=-63.0, sigma=0.03, clamp_lo=0.9, clamp_hi=1.1),
    "Ib":  HeteroParam(base=0.05,  sigma=0.10, clamp_lo=0.5, clamp_hi=1.5),
}


@dataclass
class STPParams:
    """STP parameters per block (mean ± jitter)."""
    U: float
    tau_d_ms: float
    tau_f_ms: float
    sigma: float = 0.3     # relative jitter


# ── Default STP per block ──
DEFAULT_STP: Dict[str, STPParams] = {
    "EE": STPParams(U=0.50, tau_d_ms=1100.0, tau_f_ms=50.0),
    "EI": STPParams(U=0.05, tau_d_ms=125.0,  tau_f_ms=1200.0),
    "IE": STPParams(U=0.25, tau_d_ms=700.0,  tau_f_ms=20.0),
    "II": STPParams(U=0.32, tau_d_ms=144.0,  tau_f_ms=60.0),
}


@dataclass
class GeneratorConfig:
    """All parameters for one frozen bundle."""
    # ── Identity ──
    seed: int = 100
    bundle_version: str = "2.0"

    # ── Network size ──
    N: int = 100
    frac_I: float = 0.2

    # ── Graph ──
    graph_type: str = "ER"             # "ER" or "fixed_indegree"
    p_conn: float = 0.2               # for ER
    k_in: int = 20                     # for fixed_indegree
    allow_self: bool = False

    # ── Base weights (magnitudes, sign applied by E/I type) ──
    w_EE_base: float = 3.0
    w_EI_base: float = 6.0
    w_IE_base: float = 11.2
    w_II_base: float = 11.2
    w_sigma: float = 0.3              # relative weight jitter (lognormal)

    # ── Neuron heterogeneity ──
    neuron_hetero: Dict[str, HeteroParam] = field(
        default_factory=lambda: dict(DEFAULT_NEURON_HETERO))

    # ── Synapse STP ──
    stp_params: Dict[str, STPParams] = field(
        default_factory=lambda: dict(DEFAULT_STP))

    # ── Balance control ──
    target_balance: float = 1.0        # target |sum_I|/|sum_E| ratio
    balance_tol: float = 0.1           # tolerance (±10%)

    # ── Delays ──
    delay_min_ms: float = 0.5
    delay_max_ms: float = 2.0

    # ── Poisson input ──
    poisson_T_s: float = 3.0
    poisson_dt_ms: float = 0.025
    poisson_rate_hz: float = 10.0
    poisson_n_channels: int = -1       # -1 = use N

    # ── Regimes (5 regimes centered on edge of chaos) ──
    rho_eff_targets: List[float] = field(
        default_factory=lambda: [0.40, 0.70, 1.00, 1.40, 2.00])
    edge_multipliers: List[float] = field(
        default_factory=lambda: [0.40, 0.70, 1.00, 1.40, 2.00])

    # ── Edge-of-chaos detection ──
    edge_cv_thr: float = 0.45         # CV_ISI_E threshold for chaos onset
    edge_fano_thr: float = 0.50       # Fano_E threshold for chaos onset
    edge_n_scan: int = 16             # coarse scan points (logspace)
    edge_n_refine: int = 6            # refinement points around edge
    edge_rho_eff_min: float = 0.1     # min effective ρ for grid
    edge_rho_eff_max: float = 3.0     # max effective ρ for grid
    edge_silent_thr: float = 40.0     # %silent_E threshold for candidate filtering
    edge_cv_trend_tol: float = 0.02   # CV trend tolerance (Step B)
    edge_fano_trend_tol: float = 0.03 # Fano trend tolerance (Step B)

    # ── Output ──
    bundle_dir: str = "bundles"


BLOCK_NAMES = ("EE", "EI", "IE", "II")
BLOCK_SIGN = {"EE": +1.0, "EI": +1.0, "IE": -1.0, "II": -1.0}

REGIME_NAMES = [
    "R1_deep_stable", "R2_stable", "R3_edge",
    "R4_weak_chaos", "R5_strong_chaos",
]


def validate_config(cfg: GeneratorConfig) -> None:
    """Raise ValueError on invalid config."""
    if cfg.N <= 0:
        raise ValueError(f"N must be > 0, got {cfg.N}")
    if not (0.0 < cfg.frac_I < 1.0):
        raise ValueError(f"frac_I must be in (0,1), got {cfg.frac_I}")

    if cfg.graph_type not in ("ER", "fixed_indegree"):
        raise ValueError(f"graph_type must be 'ER' or 'fixed_indegree', got {cfg.graph_type!r}")
    if cfg.graph_type == "ER" and not (0.0 <= cfg.p_conn <= 1.0):
        raise ValueError(f"p_conn must be in [0,1], got {cfg.p_conn}")
    if cfg.graph_type == "fixed_indegree" and cfg.k_in <= 0:
        raise ValueError(f"k_in must be > 0, got {cfg.k_in}")

    if cfg.poisson_T_s <= 0:
        raise ValueError(f"poisson_T_s must be > 0, got {cfg.poisson_T_s}")
    if cfg.poisson_dt_ms <= 0:
        raise ValueError(f"poisson_dt_ms must be > 0, got {cfg.poisson_dt_ms}")

    n_reg = len(cfg.rho_eff_targets)
    if n_reg != 5:
        raise ValueError(f"rho_eff_targets must have 5 elements, got {n_reg}")
    if cfg.rho_eff_targets != sorted(cfg.rho_eff_targets):
        raise ValueError(f"rho_eff_targets must be sorted ascending")
    if len(set(cfg.rho_eff_targets)) != 5:
        raise ValueError(f"rho_eff_targets must be unique")

    n_mult = len(cfg.edge_multipliers)
    if n_mult != 5:
        raise ValueError(f"edge_multipliers must have 5 elements, got {n_mult}")
    if cfg.edge_multipliers != sorted(cfg.edge_multipliers):
        raise ValueError(f"edge_multipliers must be sorted ascending")
    if cfg.edge_cv_thr <= 0:
        raise ValueError(f"edge_cv_thr must be > 0, got {cfg.edge_cv_thr}")
    if cfg.edge_fano_thr <= 0:
        raise ValueError(f"edge_fano_thr must be > 0, got {cfg.edge_fano_thr}")
    if cfg.edge_n_scan < 4:
        raise ValueError(f"edge_n_scan must be >= 4, got {cfg.edge_n_scan}")

    if cfg.delay_min_ms <= 0 or cfg.delay_max_ms <= cfg.delay_min_ms:
        raise ValueError(f"delay range invalid: [{cfg.delay_min_ms}, {cfg.delay_max_ms}]")

    for name, hp in cfg.neuron_hetero.items():
        if hp.sigma < 0:
            raise ValueError(f"neuron_hetero[{name}].sigma must be >= 0")
        if hp.clamp_lo >= hp.clamp_hi:
            raise ValueError(f"neuron_hetero[{name}] clamp_lo >= clamp_hi")

########################################
# FILE: gen/seeding.py
########################################

"""Deterministic seed hierarchy from one master seed."""
from __future__ import annotations

import numpy as np

SEED_KEYS = ("graph", "weights", "neurons", "synapses", "poisson", "regimes")


def split_seeds(master_seed: int) -> dict[str, int]:
    """Derive deterministic sub-seeds from master seed.

    Uses numpy default_rng → integers for reproducibility.
    Returns dict with keys: graph, weights, neurons, synapses, poisson, regimes.
    """
    rng = np.random.default_rng(master_seed)
    vals = rng.integers(0, 2**63, size=len(SEED_KEYS), dtype=np.int64)
    return {k: int(v) for k, v in zip(SEED_KEYS, vals)}

########################################
# FILE: gen/population.py
########################################

"""E/I population assignment."""
from __future__ import annotations

import numpy as np


def make_population(N: int, frac_I: float, seed: int) -> dict:
    """Assign neurons to E and I populations.

    Returns dict with:
        N_E, N_I: int
        E_idx: int32 array (sorted)
        I_idx: int32 array (sorted)
        is_I: bool array length N
    """
    N_I = int(round(N * frac_I))
    N_E = N - N_I

    rng = np.random.default_rng(seed)
    perm = np.arange(N, dtype=np.int32)
    rng.shuffle(perm)

    I_idx = np.sort(perm[:N_I])
    E_idx = np.sort(perm[N_I:])

    is_I = np.zeros(N, dtype=np.bool_)
    is_I[I_idx] = True

    return {
        "N_E": N_E, "N_I": N_I,
        "E_idx": E_idx, "I_idx": I_idx,
        "is_I": is_I,
    }

########################################
# FILE: gen/graph_gen.py
########################################

"""Graph generation — ER or fixed-indegree."""
from __future__ import annotations

import numpy as np
from .config import GeneratorConfig

# Block IDs: EE=0, EI=1, IE=2, II=3
BLOCK_ID_MAP = {"EE": 0, "EI": 1, "IE": 2, "II": 3}


def generate_edges(cfg: GeneratorConfig, pop: dict, seed: int) -> dict:
    """Generate directed edges for the reservoir.

    Returns dict with:
        pre: int32[] — pre-synaptic neuron indices
        post: int32[] — post-synaptic neuron indices
        block_id: uint8[] — block type (0=EE, 1=EI, 2=IE, 3=II)
    """
    E_idx = pop["E_idx"]
    I_idx = pop["I_idx"]
    is_I = pop["is_I"]

    # Block definitions: (src_indices, tgt_indices, block_name, same_pop)
    blocks = [
        (E_idx, E_idx, "EE", True),
        (E_idx, I_idx, "EI", False),
        (I_idx, E_idx, "IE", False),
        (I_idx, I_idx, "II", True),
    ]

    rng = np.random.default_rng(seed)
    pre_all, post_all, bid_all = [], [], []

    for src, tgt, bname, same_pop in blocks:
        bid = BLOCK_ID_MAP[bname]

        if cfg.graph_type == "ER":
            _gen_er(src, tgt, cfg.p_conn, same_pop, cfg.allow_self,
                    bid, rng, pre_all, post_all, bid_all)
        else:
            _gen_fixed_indegree(src, tgt, cfg.k_in, same_pop, cfg.allow_self,
                                bid, rng, pre_all, post_all, bid_all)

    return {
        "pre": np.concatenate(pre_all).astype(np.int32) if pre_all else np.zeros(0, np.int32),
        "post": np.concatenate(post_all).astype(np.int32) if post_all else np.zeros(0, np.int32),
        "block_id": np.concatenate(bid_all).astype(np.uint8) if bid_all else np.zeros(0, np.uint8),
    }


def _gen_er(src, tgt, p, same_pop, allow_self, bid, rng, pre_all, post_all, bid_all):
    """ER random graph for one block."""
    pre_list, post_list = [], []
    for si in range(len(src)):
        for ti in range(len(tgt)):
            if same_pop and si == ti and not allow_self:
                continue
            if rng.random() < p:
                pre_list.append(src[si])
                post_list.append(tgt[ti])
    if pre_list:
        pre_all.append(np.array(pre_list, dtype=np.int32))
        post_all.append(np.array(post_list, dtype=np.int32))
        bid_all.append(np.full(len(pre_list), bid, dtype=np.uint8))


def _gen_fixed_indegree(src, tgt, k_in, same_pop, allow_self, bid, rng,
                        pre_all, post_all, bid_all):
    """Fixed in-degree graph: each target neuron gets exactly k_in inputs from src."""
    pre_list, post_list = [], []
    for ti in range(len(tgt)):
        candidates = list(range(len(src)))
        if same_pop and not allow_self:
            candidates = [c for c in candidates if c != ti]
        k = min(k_in, len(candidates))
        chosen = rng.choice(candidates, size=k, replace=False)
        for ci in chosen:
            pre_list.append(src[ci])
            post_list.append(tgt[ti])
    if pre_list:
        pre_all.append(np.array(pre_list, dtype=np.int32))
        post_all.append(np.array(post_list, dtype=np.int32))
        bid_all.append(np.full(len(pre_list), bid, dtype=np.uint8))

########################################
# FILE: gen/hetero_neurons.py
########################################

"""Heterogeneous neuron parameter sampling."""
from __future__ import annotations

import numpy as np
from .config import GeneratorConfig, HeteroParam


def sample_neuron_params(cfg: GeneratorConfig, pop: dict, seed: int) -> dict[str, np.ndarray]:
    """Sample per-neuron parameters with heterogeneity.

    For each parameter in cfg.neuron_hetero:
        value[i] = base * clip(1 + sigma * N(0,1), clamp_lo, clamp_hi)

    Returns dict mapping param_name -> float64 array of length N.
    """
    N = cfg.N
    rng = np.random.default_rng(seed)
    params = {}

    for name, hp in cfg.neuron_hetero.items():
        if hp.sigma > 0:
            mult = 1.0 + hp.sigma * rng.standard_normal(N)
            mult = np.clip(mult, hp.clamp_lo, hp.clamp_hi)
        else:
            mult = np.ones(N, dtype=np.float64)
        params[name] = (hp.base * mult).astype(np.float64)

    return params

########################################
# FILE: gen/hetero_synapses.py
########################################

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

########################################
# FILE: gen/spectral.py
########################################

"""Spectral radius and block-level diagnostics."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from numba import njit


@njit(cache=True)
def _power_iter(data, indices, indptr, n, n_iter=200):
    """Spectral radius via power iteration on CSR (absolute value)."""
    x = np.ones(n, dtype=np.float64)
    x /= np.sqrt(np.dot(x, x))
    for _ in range(n_iter):
        y = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                s += data[k] * x[indices[k]]
            y[i] = s
        norm = np.sqrt(np.dot(y, y))
        if norm < 1e-15:
            return 0.0
        x = y / norm
    # Rayleigh quotient
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        y[i] = s
    return np.sqrt(np.dot(y, y))


def spectral_radius(W: sparse.csr_matrix, n_iter: int = 200) -> float:
    """Compute spectral radius of sparse matrix W via power iteration."""
    if W.nnz == 0:
        return 0.0
    return float(_power_iter(W.data.astype(np.float64),
                             W.indices, W.indptr, W.shape[0], n_iter))


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
    is_I = pop["is_I"]

    rho_full = spectral_radius(W)

    # E→E sub-block (same full matrix shape, zero out non-E)
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

########################################
# FILE: gen/poisson_bank.py
########################################

"""Frozen Poisson spike train bank."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def generate_poisson_trains(
    n_channels: int,
    T_s: float,
    rate_hz: Union[float, np.ndarray],
    seed: int,
) -> list[np.ndarray]:
    """Generate deterministic Poisson spike trains.

    Args:
        n_channels: number of independent channels
        T_s: duration in seconds
        rate_hz: scalar or array of length n_channels
        seed: RNG seed

    Returns:
        List of n_channels arrays, each containing sorted spike times in seconds.
    """
    rng = np.random.default_rng(seed)

    if np.isscalar(rate_hz):
        rates = np.full(n_channels, rate_hz, dtype=np.float64)
    else:
        rates = np.asarray(rate_hz, dtype=np.float64)
        assert len(rates) == n_channels

    trains = []
    for ch in range(n_channels):
        lam = rates[ch] * T_s
        n_spikes = rng.poisson(lam)
        if n_spikes > 0:
            times = np.sort(rng.uniform(0.0, T_s, n_spikes))
        else:
            times = np.zeros(0, dtype=np.float64)
        trains.append(times)

    return trains


def save_poisson(trains: list[np.ndarray], path: Path,
                 T_s: float, rate_hz: float, seed: int) -> None:
    """Save Poisson trains to compressed NPZ."""
    data = {
        "n_channels": len(trains),
        "T_s": T_s,
        "rate_hz": rate_hz,
        "seed": seed,
    }
    for ch, t in enumerate(trains):
        data[f"ch{ch}"] = t
    np.savez_compressed(path, **data)


def load_poisson(path: Path) -> tuple[list[np.ndarray], dict]:
    """Load Poisson trains from NPZ.

    Returns (trains_list, metadata_dict).
    """
    raw = np.load(path, allow_pickle=False)
    n_channels = int(raw["n_channels"])
    meta = {
        "n_channels": n_channels,
        "T_s": float(raw["T_s"]),
        "rate_hz": float(raw["rate_hz"]),
        "seed": int(raw["seed"]),
    }
    trains = [raw[f"ch{ch}"] for ch in range(n_channels)]
    return trains, meta

########################################
# FILE: gen/regimes.py
########################################

"""Regime definitions: simple α from ρ targets + algorithmic edge-of-chaos detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import REGIME_NAMES, GeneratorConfig


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE MODE — 5 regimes from ρ_eff targets (no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

def make_regimes(rho_base: float, targets: list[float]) -> list[dict]:
    """Create 5 regime definitions from ρ targets.

    Each regime: alpha_i = target_i / rho_base.

    Returns list of dicts with: name, index, rho_target, alpha.
    """
    regimes = []
    for i, rho_t in enumerate(targets):
        alpha = rho_t / rho_base
        regimes.append({
            "name": REGIME_NAMES[i],
            "index": i,
            "rho_target": float(rho_t),
            "alpha": float(alpha),
        })
    return regimes


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE MODE — 5 regimes from α_edge × multipliers
# ═══════════════════════════════════════════════════════════════════════════════

def make_regimes_from_edge(
    alpha_edge: float,
    rho_base: float,
    multipliers: Optional[List[float]] = None,
) -> list[dict]:
    """Create 5 regimes centered on α_edge.

    R1..R5 = α_edge × multipliers[0..4].
    Default multipliers: [0.40, 0.70, 1.00, 1.40, 2.00].
    """
    if multipliers is None:
        multipliers = [0.40, 0.70, 1.00, 1.40, 2.00]

    regimes = []
    for i, (name, mult) in enumerate(zip(REGIME_NAMES, multipliers)):
        alpha = alpha_edge * mult
        regimes.append({
            "name": name,
            "index": i,
            "alpha": float(alpha),
            "rho_target": float(alpha * rho_base),
            "multiplier": float(mult),
            "alpha_edge": float(alpha_edge),
        })
    return regimes


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE FINDING — scan α grid via Brian2 (requires Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_alpha_grid(rho_base: float, cfg: GeneratorConfig) -> np.ndarray:
    """Build deterministic logspace α grid from ρ_eff range.

    alpha_min = rho_eff_min / rho_base
    alpha_max = rho_eff_max / rho_base
    K = edge_n_scan points in logspace.
    """
    alpha_min = cfg.edge_rho_eff_min / rho_base
    alpha_max = cfg.edge_rho_eff_max / rho_base
    return np.geomspace(alpha_min, alpha_max, cfg.edge_n_scan)


def _run_scan_points(
    bundle_dir: Path,
    alphas: np.ndarray,
    rho_base: float,
    warmup_ms: float,
    measure_ms: float,
    dt_ms: float,
    label: str = "scan",
) -> List[dict]:
    """Run Brian2 for each α and return list of scan entry dicts."""
    from .brian_smoke import run_smoke_one_regime, evaluate_scan_liveness

    results = []
    n = len(alphas)
    for k, alpha in enumerate(alphas):
        regime_stub = {
            "name": f"{label}_{k:02d}",
            "index": k,
            "rho_target": float(alpha * rho_base),
            "alpha": float(alpha),
        }
        try:
            m = run_smoke_one_regime(
                bundle_dir, regime_stub, warmup_ms, measure_ms, dt_ms)
            gate = evaluate_scan_liveness(
                m.rate_E, m.rate_I, m.pct_silent_E, m.sync_E, m.has_nan)
            entry = {
                "alpha": float(alpha),
                "cv_isi_E": m.cv_isi_E,
                "fano_E": m.fano_E,
                "rate_E": m.rate_E,
                "rate_I": m.rate_I,
                "pct_silent_E": m.pct_silent_E,
                "sync_E": m.sync_E,
                "status": m.status,
                "gate": gate,
            }
        except Exception as exc:
            entry = {
                "alpha": float(alpha),
                "cv_isi_E": 0.0,
                "fano_E": 0.0,
                "rate_E": 0.0,
                "rate_I": 0.0,
                "pct_silent_E": 100.0,
                "sync_E": 0.0,
                "status": f"ERROR: {exc}",
                "gate": "FAIL",
            }
        results.append(entry)
        print(f"    [{k+1}/{n}] α={alpha:.6f}  "
              f"cvE={entry['cv_isi_E']:.3f}  fanoE={entry['fano_E']:.3f}  "
              f"rateE={entry['rate_E']:.1f}  silent={entry['pct_silent_E']:.0f}%  "
              f"[{entry['gate']}]")
    return results


def _build_refine_grid(
    alphas_coarse: np.ndarray,
    k_star: int,
    n_refine: int,
) -> np.ndarray:
    """Build refinement α grid around k_star.

    - k_star == len-1  → extend upward:  linspace(α[K-1], α[K-1]*2, n+1)[1:]
    - k_star == 0      → extend downward: linspace(α[0]/2, α[0], n+1)[:-1]
    - otherwise        → zoom in:  linspace(α[k-1], α[k+1], n+2)[1:-1]
    """
    K = len(alphas_coarse)
    if k_star == K - 1:
        # Extend upward
        return np.linspace(alphas_coarse[-1], alphas_coarse[-1] * 2, n_refine + 1)[1:]
    elif k_star == 0:
        # Extend downward
        return np.linspace(alphas_coarse[0] / 2, alphas_coarse[0], n_refine + 1)[:-1]
    else:
        # Zoom in between neighbors
        return np.linspace(
            alphas_coarse[k_star - 1], alphas_coarse[k_star + 1],
            n_refine + 2)[1:-1]


def find_alpha_edge(
    bundle_dir: Path,
    rho_base: float,
    cfg: Optional[GeneratorConfig] = None,
    warmup_ms: float = 1000.0,
    measure_ms: float = 2000.0,
    dt_ms: float = 0.025,
) -> Tuple[float, dict]:
    """Two-stage scan to find edge of chaos (requires Brian2).

    Stage 1 — coarse grid (K=edge_n_scan, logspace):
        Run all K points. Find k* via Step A+B or fallback.

    Stage 2 — refinement (K_refine=edge_n_refine, linspace around k*):
        - k* at last  → extend upward
        - k* at first → extend downward
        - k* in middle → zoom between α[k*-1] and α[k*+1]
        Merge coarse + refine, sort by α, re-run Step A+B on combined.

    Returns (α_edge, scan_info) where scan_info is a dict with:
        coarse_grid, refine_grid, merged_grid, edge_rule,
        alpha_edge_source, refine_direction.
    """
    bundle_dir = Path(bundle_dir)
    if cfg is None:
        cfg = GeneratorConfig()

    cv_thr = cfg.edge_cv_thr
    fano_thr = cfg.edge_fano_thr
    silent_thr = cfg.edge_silent_thr
    cv_trend_tol = cfg.edge_cv_trend_tol
    fano_trend_tol = cfg.edge_fano_trend_tol
    n_refine = cfg.edge_n_refine

    edge_rule = {
        "cv_thr": cv_thr,
        "fano_thr": fano_thr,
        "silent_thr": silent_thr,
        "cv_trend_tol": cv_trend_tol,
        "fano_trend_tol": fano_trend_tol,
    }

    # ── Stage 1: coarse grid ──
    alphas_coarse = _build_alpha_grid(rho_base, cfg)
    alpha_lo, alpha_hi = float(alphas_coarse[0]), float(alphas_coarse[-1])

    print(f"  Stage 1 — coarse: {cfg.edge_n_scan} points in "
          f"[{alpha_lo:.6f}, {alpha_hi:.6f}]  "
          f"(ρ_eff=[{cfg.edge_rho_eff_min}, {cfg.edge_rho_eff_max}])")

    coarse_results = _run_scan_points(
        bundle_dir, alphas_coarse, rho_base,
        warmup_ms, measure_ms, dt_ms, label="coarse")

    # Find k* on coarse grid
    coarse_edge = _find_edge_ab(
        coarse_results, cv_thr, fano_thr, silent_thr,
        cv_trend_tol, fano_trend_tol)

    if coarse_edge is not None:
        k_star = next(i for i, r in enumerate(coarse_results)
                      if r["alpha"] == coarse_edge)
        coarse_method = "A+B"
    else:
        # Try fallback to get k*
        fb_edge = _find_edge_fallback(coarse_results, silent_thr)
        if fb_edge is not None:
            k_star = next(i for i, r in enumerate(coarse_results)
                          if r["alpha"] == fb_edge)
            coarse_method = "fallback"
        else:
            raise RuntimeError(
                "Edge not found — all coarse scan points FAIL or "
                "%silent_E > threshold. "
                "Network may be degenerate or scan range too narrow.")

    print(f"  Coarse k*={k_star} (α={alphas_coarse[k_star]:.6f}, {coarse_method})")

    # ── Stage 2: refinement ──
    if n_refine <= 0:
        edge_alpha = coarse_results[k_star]["alpha"]
        print(f"  α_edge = {edge_alpha:.6f}  (coarse only, {coarse_method})")
        scan_info = {
            "coarse_grid": coarse_results,
            "refine_grid": [],
            "merged_grid": coarse_results,
            "edge_rule": edge_rule,
            "alpha_edge_source": f"coarse_k{k_star}_{coarse_method}",
            "refine_direction": None,
        }
        return edge_alpha, scan_info

    refine_alphas = _build_refine_grid(alphas_coarse, k_star, n_refine)
    direction = ("up" if k_star == len(alphas_coarse) - 1
                 else "down" if k_star == 0
                 else "zoom")
    print(f"  Stage 2 — refine: {n_refine} points ({direction}) in "
          f"[{refine_alphas[0]:.6f}, {refine_alphas[-1]:.6f}]")

    refine_results = _run_scan_points(
        bundle_dir, refine_alphas, rho_base,
        warmup_ms, measure_ms, dt_ms, label="refine")

    # Merge and sort by alpha
    all_results = sorted(coarse_results + refine_results, key=lambda r: r["alpha"])

    # Re-run Step A+B on merged set
    alpha_edge = _find_edge_ab(
        all_results, cv_thr, fano_thr, silent_thr,
        cv_trend_tol, fano_trend_tol)

    if alpha_edge is not None:
        # Determine source: was it from coarse or refine?
        refine_alpha_set = set(float(a) for a in refine_alphas)
        source_stage = "refine" if alpha_edge in refine_alpha_set else "coarse"
        merged_idx = next(i for i, r in enumerate(all_results)
                          if r["alpha"] == alpha_edge)
        source_label = f"{source_stage}_A+B_merged_k{merged_idx}"
        print(f"  α_edge = {alpha_edge:.6f}  (A+B on merged, from {source_stage})")
    else:
        # Fallback on merged
        alpha_edge = _find_edge_fallback(all_results, silent_thr)
        if alpha_edge is not None:
            source_label = "fallback_merged"
            print(f"  WARNING: A+B never met on merged, "
                  f"fallback α_edge={alpha_edge:.6f}  (max chaos_score)")
        else:
            raise RuntimeError(
                "Edge not found — all scan points FAIL or %silent_E > threshold. "
                "Network may be degenerate or scan range too narrow.")

    scan_info = {
        "coarse_grid": coarse_results,
        "refine_grid": refine_results,
        "merged_grid": all_results,
        "edge_rule": edge_rule,
        "alpha_edge_source": source_label,
        "refine_direction": direction,
    }
    return alpha_edge, scan_info


def _find_edge_ab(
    scan_results: List[dict],
    cv_thr: float, fano_thr: float, silent_thr: float,
    cv_trend_tol: float, fano_trend_tol: float,
) -> Optional[float]:
    """Step A + B: find first α passing candidate + trend criteria."""
    for k, r in enumerate(scan_results):
        # Must pass liveness gate
        if r["gate"] == "FAIL":
            continue

        # Step A: candidate criteria
        if r["pct_silent_E"] > silent_thr:
            continue
        if r["cv_isi_E"] < cv_thr or r["fano_E"] < fano_thr:
            continue

        # Step B: trend check (skip for k=0)
        if k > 0:
            prev = scan_results[k - 1]
            if r["cv_isi_E"] < prev["cv_isi_E"] - cv_trend_tol:
                continue
            if r["fano_E"] < prev["fano_E"] - fano_trend_tol:
                continue

        return r["alpha"]
    return None


def _find_edge_fallback(
    scan_results: List[dict], silent_thr: float
) -> Optional[float]:
    """Fallback: argmax of z(CV_ISI_E) + z(Fano_E) among eligible points."""
    eligible = [r for r in scan_results
                if r["gate"] in ("PASS", "WARN")
                and r["pct_silent_E"] <= silent_thr]
    if not eligible:
        return None

    cvs = np.array([r["cv_isi_E"] for r in eligible])
    fanos = np.array([r["fano_E"] for r in eligible])

    # z-score normalization
    cv_std = cvs.std()
    fano_std = fanos.std()
    z_cv = (cvs - cvs.mean()) / cv_std if cv_std > 1e-12 else np.zeros_like(cvs)
    z_fano = (fanos - fanos.mean()) / fano_std if fano_std > 1e-12 else np.zeros_like(fanos)

    scores = z_cv + z_fano
    best_idx = int(np.argmax(scores))
    return eligible[best_idx]["alpha"]


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY CHECK — extra sim at α_edge with longer T_eval
# ═══════════════════════════════════════════════════════════════════════════════

def run_stability_check(
    bundle_dir: Path,
    alpha_edge: float,
    rho_base: float,
    scan_info: dict,
    warmup_ms: float = 1000.0,
    measure_ms: float = 4000.0,
    dt_ms: float = 0.025,
    tol: float = 0.10,
) -> dict:
    """Run extra simulation at α_edge with longer T_eval.

    Compares CV_ISI_E and Fano_E from the scan (2s) against a new
    run with measure_ms (default 4s). If both are within ±tol (10%),
    the edge is considered stable.

    Returns dict with: stable, cv_scan, cv_long, fano_scan, fano_long,
    cv_rel_diff, fano_rel_diff, tol, measure_ms.
    """
    from .brian_smoke import run_smoke_one_regime

    # Get scan CV/Fano at α_edge from merged grid
    merged = scan_info.get("merged_grid", [])
    edge_entry = None
    for r in merged:
        if abs(r["alpha"] - alpha_edge) < 1e-12:
            edge_entry = r
            break

    if edge_entry is None:
        return {"stable": None, "reason": "alpha_edge not found in merged_grid"}

    cv_scan = edge_entry["cv_isi_E"]
    fano_scan = edge_entry["fano_E"]

    print(f"  Stability check: α={alpha_edge:.6f}, "
          f"measure={measure_ms:.0f}ms (vs scan 2000ms)")

    regime_stub = {
        "name": "stability_check",
        "index": -1,
        "rho_target": float(alpha_edge * rho_base),
        "alpha": float(alpha_edge),
    }

    try:
        m = run_smoke_one_regime(
            bundle_dir, regime_stub, warmup_ms, measure_ms, dt_ms)
        cv_long = m.cv_isi_E
        fano_long = m.fano_E

        cv_rel = abs(cv_long - cv_scan) / max(cv_scan, 1e-9)
        fano_rel = abs(fano_long - fano_scan) / max(fano_scan, 1e-9)
        stable = cv_rel <= tol and fano_rel <= tol

        result = {
            "stable": stable,
            "cv_scan": float(cv_scan),
            "cv_long": float(cv_long),
            "cv_rel_diff": float(cv_rel),
            "fano_scan": float(fano_scan),
            "fano_long": float(fano_long),
            "fano_rel_diff": float(fano_rel),
            "tol": float(tol),
            "measure_ms": float(measure_ms),
            "rate_E_long": float(m.rate_E),
            "status_long": m.status,
        }

        verdict = "STABLE" if stable else "UNSTABLE"
        print(f"    CV: scan={cv_scan:.3f} → long={cv_long:.3f} "
              f"(Δ={cv_rel:.1%})  "
              f"Fano: scan={fano_scan:.3f} → long={fano_long:.3f} "
              f"(Δ={fano_rel:.1%})  [{verdict}]")
        return result

    except Exception as exc:
        print(f"    Stability check FAILED: {exc}")
        return {"stable": None, "reason": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE + R5 CLAMP
# ═══════════════════════════════════════════════════════════════════════════════

def save_edge_regimes(
    bundle_dir: Path,
    alpha_edge: float,
    rho_base: float,
    scan_info: dict,
    multipliers: Optional[List[float]] = None,
    edge_method: str = "metrics_scan",
    r5_clamp_steps: int = 2,
) -> list[dict]:
    """Generate 5 regimes from α_edge and save to bundle.

    R5 clamp logic: if R5 would be at mult=2.00 but that α caused FAIL
    in the scan, reduce to 1.80 then 1.60 (max r5_clamp_steps reductions).

    scan_info is the dict returned by find_alpha_edge() with keys:
        coarse_grid, refine_grid, merged_grid, edge_rule,
        alpha_edge_source, refine_direction.

    Writes regimes/regimes.json with structured metadata.
    """
    if multipliers is None:
        multipliers = [0.40, 0.70, 1.00, 1.40, 2.00]

    # Use merged_grid for R5 clamp logic
    merged_grid = scan_info.get("merged_grid", [])

    # R5 clamp: check if the R5 α would be in FAIL territory
    r5_clamped = False
    r5_final_mult = multipliers[4]
    r5_alpha = alpha_edge * r5_final_mult

    fail_alphas = [r["alpha"] for r in merged_grid if r.get("gate") == "FAIL"]
    if fail_alphas:
        pass_warn_alphas = [r["alpha"] for r in merged_grid
                            if r.get("gate") in ("PASS", "WARN")]
        if pass_warn_alphas:
            max_ok_alpha = max(pass_warn_alphas)
            clamp_reductions = [0.20, 0.40]
            for step in range(r5_clamp_steps):
                if r5_alpha > max_ok_alpha * 1.1:
                    r5_final_mult -= clamp_reductions[step] if step < len(clamp_reductions) else 0.20
                    r5_alpha = alpha_edge * r5_final_mult
                    r5_clamped = True
                else:
                    break

    final_multipliers = multipliers[:4] + [r5_final_mult]
    regimes = make_regimes_from_edge(alpha_edge, rho_base, final_multipliers)

    # Build output with structured metadata
    out_data = {
        "alpha_edge": float(alpha_edge),
        "rho_base": float(rho_base),
        "edge_method": edge_method,
        "alpha_edge_source": scan_info.get("alpha_edge_source", "unknown"),
        "refine_direction": scan_info.get("refine_direction"),
        "edge_rule": scan_info.get("edge_rule", {}),
        "r5_clamped": r5_clamped,
        "r5_final_multiplier": float(r5_final_mult),
        "multipliers": [float(m) for m in final_multipliers],
        "coarse_grid": scan_info.get("coarse_grid", []),
        "refine_grid": scan_info.get("refine_grid", []),
        "regimes": regimes,
    }

    # Add stability check results if present
    if "stability_check" in scan_info:
        out_data["stability_check"] = scan_info["stability_check"]

    bundle_dir = Path(bundle_dir)
    reg_dir = bundle_dir / "regimes"
    reg_dir.mkdir(exist_ok=True)
    out_path = reg_dir / "regimes.json"
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Wrote edge regimes: {out_path}")
    if r5_clamped:
        print(f"    R5 clamped: mult {multipliers[4]:.2f} → {r5_final_mult:.2f}")
    for r in regimes:
        print(f"    {r['name']:20s}  α={r['alpha']:.6f}  "
              f"mult={r['multiplier']:.2f}")
    return regimes

########################################
# FILE: gen/io_bundle.py
########################################

"""Bundle I/O — write, hash, manifest."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import GeneratorConfig


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bundle(
    out_dir: Path,
    cfg: GeneratorConfig,
    seeds: dict[str, int],
    pop: dict,
    edges: dict,
    neuron_params: dict[str, np.ndarray],
    syn_params: dict,
    base_stats: dict,
    regimes: list[dict],
    poisson_trains: list[np.ndarray],
) -> Path:
    """Write all bundle artifacts to disk.

    Directory structure:
        out_dir/
            config.json
            network/
                population.json
                edges.npz
                neuron_params.npz
                synapse_params.npz
                base_stats.json
            regimes/
                regimes.json
            poisson/
                trains_3s.npz
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── config.json ──
    cfg_dict = _config_to_dict(cfg)
    cfg_dict["seeds"] = seeds
    _write_json(out_dir / "config.json", cfg_dict)

    # ── network/ ──
    net_dir = out_dir / "network"
    net_dir.mkdir(exist_ok=True)

    _write_json(net_dir / "population.json", {
        "N_E": int(pop["N_E"]),
        "N_I": int(pop["N_I"]),
        "E_idx": pop["E_idx"].tolist(),
        "I_idx": pop["I_idx"].tolist(),
    })

    np.savez_compressed(
        net_dir / "edges.npz",
        pre=edges["pre"], post=edges["post"], block_id=edges["block_id"],
    )

    np.savez_compressed(net_dir / "neuron_params.npz", **neuron_params)

    np.savez_compressed(
        net_dir / "synapse_params.npz",
        w=syn_params["w"],
        U=syn_params["U"],
        tau_d_ms=syn_params["tau_d_ms"],
        tau_f_ms=syn_params["tau_f_ms"],
        delay_ms=syn_params["delay_ms"],
    )

    _write_json(net_dir / "base_stats.json", base_stats)

    # ── regimes/ ──
    reg_dir = out_dir / "regimes"
    reg_dir.mkdir(exist_ok=True)
    _write_json(reg_dir / "regimes.json", regimes)

    # ── poisson/ ──
    poi_dir = out_dir / "poisson"
    poi_dir.mkdir(exist_ok=True)
    poi_data = {"n_channels": len(poisson_trains),
                "T_s": cfg.poisson_T_s,
                "rate_hz": cfg.poisson_rate_hz,
                "seed": seeds["poisson"]}
    for ch, t in enumerate(poisson_trains):
        poi_data[f"ch{ch}"] = t
    np.savez_compressed(poi_dir / "trains_3s.npz", **poi_data)

    return out_dir


def write_manifest(bundle_dir: Path) -> Path:
    """Create manifest.json with SHA-256 hashes for all data files."""
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": {},
    }

    for ext in ("*.json", "*.npz"):
        for p in sorted(bundle_dir.rglob(ext)):
            if p.name == "manifest.json":
                continue
            rel = str(p.relative_to(bundle_dir))
            manifest["files"][rel] = sha256_file(p)

    manifest_path = bundle_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def verify_manifest(bundle_dir: Path) -> list[str]:
    """Verify bundle integrity against manifest. Returns list of errors."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return ["manifest.json not found"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    for rel, expected in manifest.get("files", {}).items():
        full = bundle_dir / rel
        if not full.exists():
            errors.append(f"missing: {rel}")
            continue
        actual = sha256_file(full)
        if actual != expected:
            errors.append(f"hash mismatch: {rel}")

    return errors


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _config_to_dict(cfg: GeneratorConfig) -> dict:
    """Convert GeneratorConfig to a JSON-safe dict."""
    d = {}
    for k, v in cfg.__dict__.items():
        if k == "neuron_hetero":
            d[k] = {name: {"base": hp.base, "sigma": hp.sigma,
                           "clamp_lo": hp.clamp_lo, "clamp_hi": hp.clamp_hi}
                    for name, hp in v.items()}
        elif k == "stp_params":
            d[k] = {name: {"U": sp.U, "tau_d_ms": sp.tau_d_ms,
                           "tau_f_ms": sp.tau_f_ms, "sigma": sp.sigma}
                    for name, sp in v.items()}
        else:
            d[k] = v
    return d

########################################
# FILE: gen/brian_smoke.py
########################################

#!/usr/bin/env python3
"""Brian2 smoke test — liveness + separation metrics for a frozen bundle.

Usage:
    python -m gen.brian_smoke bundles/bundle_seed_123
    python -m gen.brian_smoke bundles/bundle_seed_123 --regime 0 --warmup-ms 1000 --measure-ms 2000
    python -m gen.brian_smoke bundles/bundle_seed_123 --regime all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .io_bundle import verify_manifest


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SmokeMetrics:
    """Metrics for one regime smoke test."""
    regime_name: str
    regime_index: int
    alpha: float
    rate_E: float
    rate_I: float
    cv_isi_E: float
    cv_isi_I: float
    fano_E: float
    fano_I: float
    sync_E: float
    sync_I: float
    pct_silent_E: float    # % of E neurons with 0 spikes in measure window
    pct_silent_I: float    # % of I neurons with 0 spikes
    spike_count_E: int     # total E spikes in measure window
    spike_count_I: int     # total I spikes in measure window
    has_nan: bool
    status: str        # OK, SILENT, RUNAWAY, NAN


def compute_rate(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                 total_ms: float, b) -> float:
    """Mean firing rate (Hz) over measurement window."""
    dur_s = (total_ms - warmup_ms) / 1000.0
    counts = []
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        counts.append(np.sum((t >= warmup_ms) & (t < total_ms)))
    return float(np.mean(counts) / dur_s) if counts else 0.0


def compute_cv_isi(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                   total_ms: float, b) -> float:
    """Median CV of ISI across neurons with >= 3 spikes in T_eval."""
    cvs = []
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        t = t[(t >= warmup_ms) & (t < total_ms)]
        if len(t) >= 3:
            isi = np.diff(t)
            m = isi.mean()
            if m > 1e-9:
                cvs.append(float(isi.std() / m))
    return float(np.median(cvs)) if cvs else 0.0


def compute_fano(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                 total_ms: float, b, bin_ms: float = 50.0) -> float:
    """Median Fano factor = var(counts)/mean(counts) per neuron in bins."""
    n_bins = max(1, int((total_ms - warmup_ms) / bin_ms))
    fanos = []
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        t = t[(t >= warmup_ms) & (t < total_ms)]
        hist, _ = np.histogram(t, bins=n_bins, range=(warmup_ms, total_ms))
        m = hist.mean()
        if m > 1e-9:
            fanos.append(float(hist.var() / m))
    return float(np.median(fanos)) if fanos else 0.0


def compute_sync(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                 total_ms: float, b, bin_ms: float = 5.0) -> float:
    """Population synchrony = var(pop_signal) / (N * mean(var_individual))."""
    n_bins = max(1, int((total_ms - warmup_ms) / bin_ms))
    pop_hist = np.zeros(n_bins, dtype=np.float64)
    ind_vars = []
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        t = t[(t >= warmup_ms) & (t < total_ms)]
        hist, _ = np.histogram(t, bins=n_bins, range=(warmup_ms, total_ms))
        pop_hist += hist
        ind_vars.append(float(np.var(hist)))
    pop_var = float(np.var(pop_hist))
    mean_iv = float(np.mean(ind_vars)) if ind_vars else 0.0
    return pop_var / (len(indices) * mean_iv + 1e-15)


def compute_pct_silent(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                       total_ms: float, b) -> float:
    """Percentage of neurons with 0 spikes in measurement window."""
    n_silent = 0
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        count = np.sum((t >= warmup_ms) & (t < total_ms))
        if count == 0:
            n_silent += 1
    return 100.0 * n_silent / len(indices) if len(indices) > 0 else 100.0


def compute_spike_count(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                        total_ms: float, b) -> int:
    """Total spike count across all neurons in measurement window."""
    total = 0
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        total += int(np.sum((t >= warmup_ms) & (t < total_ms)))
    return total


def evaluate_status(rate_E: float, rate_I: float, has_nan: bool) -> str:
    """Determine liveness status from metrics."""
    if has_nan:
        return "NAN"
    if rate_E < 0.5 and rate_I < 0.5:
        return "SILENT"
    if rate_E > 200 or rate_I > 300:
        return "RUNAWAY"
    return "OK"


def evaluate_scan_liveness(rate_E: float, rate_I: float, pct_silent_E: float,
                           sync_E: float, has_nan: bool) -> str:
    """Liveness gate for edge scan points.

    Returns: 'FAIL', 'WARN', or 'PASS'.

    FAIL: NaN/Inf, rate_E > 200 (runaway), rate_E < 0.1 or %silent > 80 (silent).
    WARN: 40 < %silent_E <= 80, or syncE > 1.3.
    PASS: everything else.
    """
    if has_nan:
        return "FAIL"
    if rate_E > 200 or rate_I > 300:
        return "FAIL"
    if rate_E < 0.1 or pct_silent_E > 80:
        return "FAIL"
    if pct_silent_E > 40:
        return "WARN"
    if sync_E > 1.3:
        return "WARN"
    return "PASS"


def check_separation(results: List[SmokeMetrics], rel_thr: float = 0.15
                     ) -> Tuple[bool, List[dict]]:
    """Check if regimes are separated by at least one metric.

    Returns (any_pass, details_list).
    """
    ok = [r for r in results if r.status == "OK"]
    if len(ok) < 2:
        return True, []   # can't judge with < 2

    metric_names = [
        ("rate_E",       [r.rate_E       for r in ok]),
        ("rate_I",       [r.rate_I       for r in ok]),
        ("cv_isi_E",     [r.cv_isi_E     for r in ok]),
        ("cv_isi_I",     [r.cv_isi_I     for r in ok]),
        ("fano_E",       [r.fano_E       for r in ok]),
        ("fano_I",       [r.fano_I       for r in ok]),
        ("sync_E",       [r.sync_E       for r in ok]),
        ("sync_I",       [r.sync_I       for r in ok]),
        ("pct_silent_E", [r.pct_silent_E for r in ok]),
    ]

    details = []
    any_pass = False
    for name, vals in metric_names:
        mn, mx = min(vals), max(vals)
        med = float(np.median(vals))
        span = mx - mn
        thr = rel_thr * med if med > 1e-12 else rel_thr
        passed = span >= thr
        if passed:
            any_pass = True
        details.append({
            "metric": name, "min": mn, "max": mx, "median": med,
            "span": span, "threshold": thr, "pass": passed,
        })
    return any_pass, details


def evaluate_5regime_quality(results: List[SmokeMetrics],
                             silent_thr: float = 40.0) -> dict:
    """Evaluate quality of 5-regime set.

    Per-regime liveness:
        FAIL: NaN, RUNAWAY (any regime), SILENT on R1..R4
        WARN: R5 SILENT/RUNAWAY, R1 SILENT, %silent_E > silent_thr on R3..R5
        PASS: otherwise

    Separation criteria (R1 vs R5):
        FAIL: any liveness FAIL
        WARN: CV(R5) < 1.10*CV(R1) or Fano(R5) < 1.10*Fano(R1)
        PASS: CV(R5) >= 1.15*CV(R1) AND Fano(R5) >= 1.15*Fano(R1)
              AND %silent_E <= silent_thr for all regimes
    """
    regime_status = {}

    for r in results:
        idx = r.regime_index
        name = r.regime_name

        if r.status == "NAN":
            regime_status[idx] = {"name": name, "verdict": "FAIL",
                                  "reason": "NaN detected"}
            continue

        if idx == 0:  # R1 deep stable
            if r.status == "RUNAWAY":
                regime_status[idx] = {"name": name, "verdict": "FAIL",
                                      "reason": "runaway at low α"}
            elif r.status == "SILENT":
                regime_status[idx] = {"name": name, "verdict": "WARN",
                                      "reason": "silent"}
            else:
                regime_status[idx] = {"name": name, "verdict": "PASS",
                                      "reason": ""}

        elif idx in (1, 2, 3):  # R2, R3, R4 — must be alive
            if r.status != "OK":
                regime_status[idx] = {"name": name, "verdict": "FAIL",
                                      "reason": f"status={r.status}"}
            elif r.pct_silent_E > silent_thr:
                regime_status[idx] = {"name": name, "verdict": "WARN",
                                      "reason": f"pct_silent_E={r.pct_silent_E:.0f}%"
                                                 f" > {silent_thr:.0f}%"}
            else:
                regime_status[idx] = {"name": name, "verdict": "PASS",
                                      "reason": ""}

        elif idx == 4:  # R5 strong chaos — tolerate RUNAWAY/SILENT
            if r.status == "NAN":
                regime_status[idx] = {"name": name, "verdict": "FAIL",
                                      "reason": "NaN"}
            elif r.status in ("SILENT", "RUNAWAY"):
                regime_status[idx] = {"name": name, "verdict": "WARN",
                                      "reason": f"status={r.status}"}
            elif r.pct_silent_E > silent_thr:
                regime_status[idx] = {"name": name, "verdict": "WARN",
                                      "reason": f"pct_silent_E={r.pct_silent_E:.0f}%"
                                                 f" > {silent_thr:.0f}%"}
            else:
                regime_status[idx] = {"name": name, "verdict": "PASS",
                                      "reason": ""}

    # ── Separation criteria (R1 vs R5) ──
    verdicts = [v["verdict"] for v in regime_status.values()]
    if "FAIL" in verdicts:
        overall = "FAIL"
    else:
        # Check CV/Fano ratio between R1 and R5
        r1 = next((r for r in results if r.regime_index == 0), None)
        r5 = next((r for r in results if r.regime_index == 4), None)

        sep_ok = True
        sep_warn = False
        if r1 and r5 and r1.status == "OK" and r5.status == "OK":
            cv1, cv5 = r1.cv_isi_E, r5.cv_isi_E
            fano1, fano5 = r1.fano_E, r5.fano_E

            # WARN thresholds
            if cv1 > 1e-9 and cv5 < 1.10 * cv1:
                sep_warn = True
            if fano1 > 1e-9 and fano5 < 1.10 * fano1:
                sep_warn = True

            # PASS thresholds
            if cv1 > 1e-9 and cv5 < 1.15 * cv1:
                sep_ok = False
            if fano1 > 1e-9 and fano5 < 1.15 * fano1:
                sep_ok = False

        # Check %silent across all regimes
        for r in results:
            if r.pct_silent_E > silent_thr and r.regime_index in (2, 3, 4):
                sep_ok = False

        if "WARN" in verdicts or sep_warn or not sep_ok:
            overall = "WARN" if "WARN" in verdicts or sep_warn else "PASS"
            if not sep_ok and "WARN" not in verdicts:
                overall = "WARN"
        else:
            overall = "PASS"

    return {
        "overall": overall,
        "per_regime": regime_status,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BRIAN2 NETWORK BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _import_brian2():
    """Lazy Brian2 import."""
    from brian2 import (
        start_scope, NeuronGroup, Synapses, SpikeGeneratorGroup,
        SpikeMonitor, Network, Equations, defaultclock,
    )
    from brian2 import ms as b_ms, mV as b_mV, nA as b_nA
    from brian2 import umetre, ufarad, siemens, msiemens, cm

    _area = 20000 * umetre ** 2
    constants = {
        "Cm": 1 * ufarad * cm ** -2 * _area,
        "gl": 5e-5 * siemens * cm ** -2 * _area,
        "El": -65 * b_mV,
        "EK": -90 * b_mV,
        "ENa": 50 * b_mV,
        "g_na": 100 * msiemens * cm ** -2 * _area,
        "g_kd": 30 * msiemens * cm ** -2 * _area,
        "VT": -63 * b_mV,
    }
    return {
        "start_scope": start_scope,
        "NeuronGroup": NeuronGroup,
        "Synapses": Synapses,
        "SpikeGeneratorGroup": SpikeGeneratorGroup,
        "SpikeMonitor": SpikeMonitor,
        "Network": Network,
        "Equations": Equations,
        "defaultclock": defaultclock,
        "ms": b_ms, "mV": b_mV, "nA": b_nA,
        **constants,
    }


HH_EQUATIONS = '''
dv/dt = (gl*(El-v)
        - g_na*(m**3)*h*(v-ENa)
        - g_kd*(n**4)*(v-EK)
        + I_total) / Cm : volt

dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)
        -0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1

dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1-n)
        -.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1

dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1-h)
        -4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1

I_b : ampere (shared, constant)
tau_stimulus : second (constant)
dI_stimulus/dt = -I_stimulus/tau_stimulus : ampere

I_syn_ee_syn : ampere
I_syn_ei_syn : ampere
I_syn_ie_syn : ampere
I_syn_ii_syn : ampere

I_total = I_b + I_stimulus
          + I_syn_ee_syn + I_syn_ei_syn
          + I_syn_ie_syn + I_syn_ii_syn : ampere
'''

STP_MODEL_TEMPLATE = """
A : ampere (constant)
U : 1 (constant)
tau_I : second (shared, constant)
D : second (constant)
dx/dt =  z/D       : 1 (clock-driven)
dy/dt = -y/tau_I   : 1 (clock-driven)
z = 1 - x - y      : 1
I_syn_{name}_post = A*y : ampere (summed)
du/dt = -u/F : 1 (clock-driven)
F : second (constant)
"""

STP_ON_PRE = """
u += U*(1-u)
y += u*x
x += -u*x
"""

BLOCK_TO_SYN_NAME = {"EE": "ee_syn", "EI": "ei_syn", "IE": "ie_syn", "II": "ii_syn"}
BLOCK_SIGN = {"EE": +1.0, "EI": +1.0, "IE": -1.0, "II": -1.0}
TAU_I_MAP = {"EE": 3.0, "EI": 3.0, "IE": 6.0, "II": 6.0}


def _jitter_spike_collisions(
    indices: np.ndarray, times_ms: np.ndarray, dt_ms: float,
) -> tuple:
    """Shift spikes that collide within the same dt bin for a given neuron.

    Brian2 SpikeGeneratorGroup forbids the same neuron spiking twice in one
    dt bin. For each collision, shift the later spike forward by dt_ms.
    Returns (indices, times_ms) sorted by time.
    """
    if len(times_ms) == 0:
        return indices, times_ms

    # Work per-neuron: bin times and shift collisions
    times_ms = times_ms.copy()
    unique_neurons = np.unique(indices)
    for nid in unique_neurons:
        mask = indices == nid
        t = times_ms[mask]
        if len(t) < 2:
            continue
        # Sort this neuron's spikes
        order = np.argsort(t)
        t = t[order]
        # Shift collisions: if two consecutive spikes are in the same dt bin
        for j in range(1, len(t)):
            if t[j] - t[j - 1] < dt_ms:
                t[j] = t[j - 1] + dt_ms
        times_ms[mask] = t[np.argsort(order)]  # restore original order

    # Re-sort globally by time
    order = np.argsort(times_ms)
    return indices[order], times_ms[order]


def build_brian2_from_bundle(bundle_dir: Path, alpha: float,
                             warmup_ms: float, measure_ms: float,
                             dt_ms: float = 0.025,
                             poisson_realization: int = 0):
    """Build and run a Brian2 HH+STP simulation from bundle data.

    Returns (spike_trains_dict, E_idx, I_idx, neurons, b).
    """
    b = _import_brian2()
    b["start_scope"]()

    # ── Load bundle data ──
    pop_data = json.loads((bundle_dir / "network" / "population.json").read_text())
    E_idx = np.array(pop_data["E_idx"], dtype=np.int32)
    I_idx = np.array(pop_data["I_idx"], dtype=np.int32)
    N = pop_data["N_E"] + pop_data["N_I"]

    edges_raw = np.load(bundle_dir / "network" / "edges.npz")
    pre = edges_raw["pre"]
    post = edges_raw["post"]
    block_id = edges_raw["block_id"]

    syn_raw = np.load(bundle_dir / "network" / "synapse_params.npz")
    w_all = syn_raw["w"]
    U_all = syn_raw["U"]
    tau_d_all = syn_raw["tau_d_ms"]
    tau_f_all = syn_raw["tau_f_ms"]
    delay_all = syn_raw["delay_ms"]

    neuron_raw = np.load(bundle_dir / "network" / "neuron_params.npz")

    poi_raw = np.load(bundle_dir / "poisson" / "trains_3s.npz")
    n_ch = int(poi_raw["n_channels"])

    total_ms = warmup_ms + measure_ms

    # ── Neurons ──
    neurons = b["NeuronGroup"](
        N, b["Equations"](HH_EQUATIONS),
        threshold='v > -40*mV',
        refractory='v > -40*mV',
        method='exponential_euler',
        name='neurons',
        namespace={k: b[k] for k in ("Cm", "gl", "El", "EK", "ENa", "g_na", "g_kd", "VT")},
    )
    neurons.v = b["El"]
    neurons.m = 0
    neurons.h = 1
    neurons.n = 0.5
    neurons.I_b = float(np.median(neuron_raw["Ib"])) * b["nA"]
    neurons.I_stimulus = 0 * b["nA"]
    neurons.tau_stimulus = 3 * b["ms"]
    for ii in I_idx:
        neurons.tau_stimulus[int(ii)] = 6 * b["ms"]

    objects = [neurons]

    # ── STP Synapses (per block) ──
    block_id_map = {"EE": 0, "EI": 1, "IE": 2, "II": 3}
    for bname, bid in block_id_map.items():
        mask = block_id == bid
        if not np.any(mask):
            continue

        syn_name = BLOCK_TO_SYN_NAME[bname]
        sign = BLOCK_SIGN[bname]
        eqs = STP_MODEL_TEMPLATE.format(name=syn_name)

        syn = b["Synapses"](
            neurons, neurons, model=eqs, on_pre=STP_ON_PRE,
            method='exact', name=syn_name,
        )
        syn.connect(i=pre[mask], j=post[mask])

        syn.A = sign * alpha * np.abs(w_all[mask]) * b["nA"]
        syn.U = U_all[mask]
        syn.D = tau_d_all[mask] * b["ms"]
        syn.F = np.maximum(tau_f_all[mask], 0.1) * b["ms"]
        syn.delay = delay_all[mask] * b["ms"]
        syn.tau_I = TAU_I_MAP[bname] * b["ms"]
        syn.x = 1
        syn.u = U_all[mask]
        objects.append(syn)

    # ── Poisson spike generators ──
    # Map channels to neurons (first N_E → E, next N_I → I)
    N_E = len(E_idx)
    N_I = len(I_idx)

    # E trains
    e_indices, e_times = [], []
    for ch_i in range(min(N_E, n_ch)):
        t_s = poi_raw[f"ch{ch_i}"]
        t_ms = t_s * 1000.0
        t_ms = t_ms[t_ms < total_ms]
        if len(t_ms) > 0:
            e_indices.append(np.full(len(t_ms), ch_i, dtype=np.int32))
            e_times.append(t_ms)

    if e_indices:
        e_idx_arr = np.concatenate(e_indices)
        e_t_arr = np.concatenate(e_times)
        order = np.argsort(e_t_arr)
        e_idx_arr, e_t_arr = e_idx_arr[order], e_t_arr[order]
    else:
        e_idx_arr = np.zeros(0, dtype=np.int32)
        e_t_arr = np.zeros(0, dtype=np.float64)

    # Jitter collisions: if same neuron spikes twice in same dt bin, shift +dt
    e_idx_arr, e_t_arr = _jitter_spike_collisions(e_idx_arr, e_t_arr, dt_ms)

    sg_E = b["SpikeGeneratorGroup"](N_E, e_idx_arr, e_t_arr * b["ms"], name='sg_E')
    syn_sg_E = b["Synapses"](sg_E, neurons, on_pre='I_stimulus += 1.5*nA', name='syn_sg_E')
    syn_sg_E.connect(i=np.arange(N_E), j=E_idx)
    objects.extend([sg_E, syn_sg_E])

    # I trains
    i_indices, i_times = [], []
    for ch_i in range(N_I):
        ch_global = N_E + ch_i
        if ch_global >= n_ch:
            break
        t_s = poi_raw[f"ch{ch_global}"]
        t_ms = t_s * 1000.0
        t_ms = t_ms[t_ms < total_ms]
        if len(t_ms) > 0:
            i_indices.append(np.full(len(t_ms), ch_i, dtype=np.int32))
            i_times.append(t_ms)

    if i_indices:
        i_idx_arr = np.concatenate(i_indices)
        i_t_arr = np.concatenate(i_times)
        order = np.argsort(i_t_arr)
        i_idx_arr, i_t_arr = i_idx_arr[order], i_t_arr[order]
    else:
        i_idx_arr = np.zeros(0, dtype=np.int32)
        i_t_arr = np.zeros(0, dtype=np.float64)

    # Jitter collisions for I trains too
    i_idx_arr, i_t_arr = _jitter_spike_collisions(i_idx_arr, i_t_arr, dt_ms)

    sg_I = b["SpikeGeneratorGroup"](N_I, i_idx_arr, i_t_arr * b["ms"], name='sg_I')
    syn_sg_I = b["Synapses"](sg_I, neurons, on_pre='I_stimulus += 0.75*nA', name='syn_sg_I')
    syn_sg_I.connect(i=np.arange(N_I), j=I_idx)
    objects.extend([sg_I, syn_sg_I])

    spike_mon = b["SpikeMonitor"](neurons, name='spike_monitor')
    objects.append(spike_mon)

    b["defaultclock"].dt = dt_ms * b["ms"]
    net = b["Network"](objects)
    net.store()

    # ── Run ──
    rng_v = np.random.default_rng(12345)
    net.restore()
    neurons.v = (float(b["El"] / b["mV"]) + rng_v.uniform(-2, 2, N)) * b["mV"]
    net.run(total_ms * b["ms"])

    return spike_mon.spike_trains(), E_idx, I_idx, neurons, b


def run_smoke_one_regime(bundle_dir: Path, regime: dict,
                         warmup_ms: float, measure_ms: float,
                         dt_ms: float = 0.025) -> SmokeMetrics:
    """Run one regime and compute all metrics."""
    alpha = regime["alpha"]
    total_ms = warmup_ms + measure_ms

    st, E_idx, I_idx, neurons, b = build_brian2_from_bundle(
        bundle_dir, alpha, warmup_ms, measure_ms, dt_ms)

    rate_E = compute_rate(st, E_idx, warmup_ms, total_ms, b)
    rate_I = compute_rate(st, I_idx, warmup_ms, total_ms, b)
    cv_E = compute_cv_isi(st, E_idx, warmup_ms, total_ms, b)
    cv_I = compute_cv_isi(st, I_idx, warmup_ms, total_ms, b)
    fano_E = compute_fano(st, E_idx, warmup_ms, total_ms, b)
    fano_I = compute_fano(st, I_idx, warmup_ms, total_ms, b)
    sync_E = compute_sync(st, E_idx, warmup_ms, total_ms, b)
    sync_I = compute_sync(st, I_idx, warmup_ms, total_ms, b)
    pct_silent_E = compute_pct_silent(st, E_idx, warmup_ms, total_ms, b)
    pct_silent_I = compute_pct_silent(st, I_idx, warmup_ms, total_ms, b)
    spike_count_E = compute_spike_count(st, E_idx, warmup_ms, total_ms, b)
    spike_count_I = compute_spike_count(st, I_idx, warmup_ms, total_ms, b)

    v_arr = np.array(neurons.v / b["mV"])
    has_nan = bool(np.any(np.isnan(v_arr)) or np.any(np.isinf(v_arr)))
    status = evaluate_status(rate_E, rate_I, has_nan)

    return SmokeMetrics(
        regime_name=regime["name"], regime_index=regime["index"],
        alpha=alpha,
        rate_E=rate_E, rate_I=rate_I,
        cv_isi_E=cv_E, cv_isi_I=cv_I,
        fano_E=fano_E, fano_I=fano_I,
        sync_E=sync_E, sync_I=sync_I,
        pct_silent_E=pct_silent_E, pct_silent_I=pct_silent_I,
        spike_count_E=spike_count_E, spike_count_I=spike_count_I,
        has_nan=has_nan, status=status,
    )


def run_smoke(bundle_dir: Path, regime_indices: Optional[List[int]] = None,
              warmup_ms: float = 1000.0, measure_ms: float = 2000.0,
              dt_ms: float = 0.025) -> List[SmokeMetrics]:
    """Run smoke test for selected regimes (or all).

    Returns list of SmokeMetrics + saves validation/metrics.json to bundle.
    """
    bundle_dir = Path(bundle_dir)

    # Verify integrity
    errors = verify_manifest(bundle_dir)
    if errors:
        print(f"WARNING: bundle integrity issues: {errors}")

    regimes_raw = json.loads(
        (bundle_dir / "regimes" / "regimes.json").read_text())
    # Support both formats: list (simple) or dict with "regimes" key (edge)
    if isinstance(regimes_raw, list):
        regimes = regimes_raw
    else:
        regimes = regimes_raw.get("regimes", regimes_raw)

    if regime_indices is None:
        regime_indices = list(range(len(regimes)))

    print(f"{'='*60}")
    print(f"  BRIAN2 SMOKE TEST")
    print(f"  bundle: {bundle_dir}")
    print(f"  regimes: {regime_indices}")
    print(f"  warmup={warmup_ms}ms  measure={measure_ms}ms  dt={dt_ms}ms")
    print(f"{'='*60}")

    results = []
    for ri in regime_indices:
        reg = regimes[ri] if isinstance(regimes[ri], dict) else regimes[ri]
        t0 = time.time()
        m = run_smoke_one_regime(bundle_dir, reg, warmup_ms, measure_ms, dt_ms)
        dt_run = time.time() - t0
        results.append(m)
        print(f"  {m.regime_name:20s}  α={m.alpha:.6f}  "
              f"rE={m.rate_E:5.1f}  rI={m.rate_I:5.1f}  "
              f"cvE={m.cv_isi_E:.2f}  fanoE={m.fano_E:.2f}  "
              f"syncE={m.sync_E:.2f}  silent_E={m.pct_silent_E:.0f}%  "
              f"[{m.status}]  {dt_run:.1f}s")

    # Separation check
    sep_pass, sep_details = check_separation(results)
    if sep_pass:
        print(f"\n  Separation: PASS")
    else:
        print(f"\n  Separation: WARN (no metric >=15% spread)")
        for d in sep_details:
            print(f"    {d['metric']:12s}: span={d['span']:.3f} "
                  f"thr={d['threshold']:.3f} {'PASS' if d['pass'] else 'FAIL'}")

    # 5-regime quality check
    quality = evaluate_5regime_quality(results)
    print(f"\n  5-regime quality: {quality['overall']}")
    for idx, info in sorted(quality["per_regime"].items()):
        suffix = f" ({info['reason']})" if info["reason"] else ""
        print(f"    {info['name']:20s}  {info['verdict']}{suffix}")

    # Save results
    val_dir = bundle_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    out_path = val_dir / "metrics.json"
    out_data = {
        "warmup_ms": warmup_ms,
        "measure_ms": measure_ms,
        "dt_ms": dt_ms,
        "results": [asdict(r) for r in results],
        "separation": {"pass": sep_pass, "details": sep_details},
        "quality_5regime": quality,
    }
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Saved: {out_path}")

    # Summary
    fails = [r for r in results if r.status in ("NAN", "RUNAWAY")]
    if fails:
        print(f"\n  FAIL: {len(fails)} regimes failed liveness")
        for f in fails:
            print(f"    {f.regime_name}: {f.status}")
    else:
        print(f"\n  All {len(results)} regimes: LIVENESS OK")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Brian2 smoke test for frozen bundle")
    ap.add_argument("bundle_dir", type=Path)
    ap.add_argument("--regime", default="all",
                    help="Regime index (0-4) or 'all'")
    ap.add_argument("--warmup-ms", type=float, default=1000.0)
    ap.add_argument("--measure-ms", type=float, default=2000.0)
    ap.add_argument("--dt-ms", type=float, default=0.025)
    args = ap.parse_args()

    if args.regime == "all":
        indices = None
    else:
        indices = [int(args.regime)]

    run_smoke(args.bundle_dir, indices,
              args.warmup_ms, args.measure_ms, args.dt_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

########################################
# FILE: gen/calibrate_edge.py
########################################

#!/usr/bin/env python3
"""Calibrate edge of chaos for a frozen bundle (requires Brian2).

Scans α values via Brian2, finds α_edge using CV_ISI_E + Fano_E thresholds
with liveness gating and trend check, then writes 5 regimes to regimes.json.

Usage:
    python -m gen.calibrate_edge bundles/bundle_seed_123
    python -m gen.calibrate_edge bundles/bundle_seed_123 --n-scan 20 --warmup-ms 1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import GeneratorConfig
from .regimes import find_alpha_edge, save_edge_regimes, run_stability_check


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Find edge of chaos and generate 5 regimes")
    ap.add_argument("bundle_dir", type=Path,
                    help="Path to existing bundle directory")
    ap.add_argument("--cv-thr", type=float, default=0.45,
                    help="CV_ISI_E threshold for chaos onset (default: 0.45)")
    ap.add_argument("--fano-thr", type=float, default=0.50,
                    help="Fano_E threshold for chaos onset (default: 0.50)")
    ap.add_argument("--n-scan", type=int, default=16,
                    help="Number of coarse α scan points (default: 16)")
    ap.add_argument("--n-refine", type=int, default=6,
                    help="Number of refinement points around edge (default: 6, 0=disable)")
    ap.add_argument("--warmup-ms", type=float, default=1000.0,
                    help="Warmup duration per scan point (default: 1000)")
    ap.add_argument("--measure-ms", type=float, default=2000.0,
                    help="Measurement duration per scan point (default: 2000)")
    ap.add_argument("--dt-ms", type=float, default=0.025)
    ap.add_argument("--multipliers", type=float, nargs=5,
                    default=[0.40, 0.70, 1.00, 1.40, 2.00],
                    help="5 multipliers for α_edge "
                         "(default: 0.40 0.70 1.00 1.40 2.00)")
    ap.add_argument("--rho-eff-min", type=float, default=0.1,
                    help="Min effective ρ for grid (default: 0.1)")
    ap.add_argument("--rho-eff-max", type=float, default=3.0,
                    help="Max effective ρ for grid (default: 3.0)")
    ap.add_argument("--stability-check", action="store_true", default=True,
                    help="Run stability check at α_edge (default: on)")
    ap.add_argument("--no-stability-check", dest="stability_check",
                    action="store_false",
                    help="Skip stability check")
    ap.add_argument("--stability-ms", type=float, default=4000.0,
                    help="Measurement duration for stability check (default: 4000)")
    ap.add_argument("--stability-tol", type=float, default=0.10,
                    help="Relative tolerance for stability (default: 0.10)")
    args = ap.parse_args()

    bundle_dir = args.bundle_dir
    if not bundle_dir.exists():
        print(f"ERROR: bundle not found: {bundle_dir}")
        return 1

    # Load rho_base from bundle
    stats_path = bundle_dir / "network" / "base_stats.json"
    if not stats_path.exists():
        print(f"ERROR: base_stats.json not found in {bundle_dir}")
        return 1
    stats = json.loads(stats_path.read_text())
    rho_base = stats["rho_full"]

    # Build cfg with user overrides
    cfg = GeneratorConfig(
        edge_cv_thr=args.cv_thr,
        edge_fano_thr=args.fano_thr,
        edge_n_scan=args.n_scan,
        edge_n_refine=args.n_refine,
        edge_rho_eff_min=args.rho_eff_min,
        edge_rho_eff_max=args.rho_eff_max,
    )

    print(f"{'='*60}")
    print(f"  EDGE-OF-CHAOS CALIBRATION")
    print(f"  bundle: {bundle_dir}")
    print(f"  rho_base: {rho_base:.4f}")
    print(f"  cv_thr={cfg.edge_cv_thr}  fano_thr={cfg.edge_fano_thr}")
    print(f"  n_scan={cfg.edge_n_scan}  n_refine={cfg.edge_n_refine}  "
          f"ρ_eff=[{cfg.edge_rho_eff_min}, {cfg.edge_rho_eff_max}]")
    print(f"  warmup={args.warmup_ms}ms  measure={args.measure_ms}ms")
    print(f"  multipliers={args.multipliers}")
    print(f"{'='*60}")

    t0 = time.time()

    alpha_edge, scan_info = find_alpha_edge(
        bundle_dir, rho_base, cfg,
        warmup_ms=args.warmup_ms,
        measure_ms=args.measure_ms,
        dt_ms=args.dt_ms,
    )

    # Stability check
    if args.stability_check:
        stab = run_stability_check(
            bundle_dir, alpha_edge, rho_base, scan_info,
            warmup_ms=args.warmup_ms,
            measure_ms=args.stability_ms,
            dt_ms=args.dt_ms,
            tol=args.stability_tol,
        )
        scan_info["stability_check"] = stab

    edge_method = ("metrics_scan_2stage"
                    if scan_info.get("refine_grid")
                    else "metrics_scan")

    regimes = save_edge_regimes(
        bundle_dir, alpha_edge, rho_base, scan_info,
        multipliers=args.multipliers,
        edge_method=edge_method,
    )

    elapsed = time.time() - t0
    print(f"\n  α_edge = {alpha_edge:.6f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

########################################
# FILE: gen/generate_one_bundle.py
########################################

#!/usr/bin/env python3
"""Generate one frozen reservoir bundle.

Usage:
    python -m gen.generate_one_bundle --seed 123 --N 100
    python -m gen.generate_one_bundle --seed 123 --N 100 --rho-targets 0.3 0.7 1.1 1.6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import GeneratorConfig, validate_config
from .seeding import split_seeds
from .population import make_population
from .graph_gen import generate_edges
from .hetero_neurons import sample_neuron_params
from .hetero_synapses import sample_synapse_params, build_sparse_W, rescale_for_balance
from .spectral import spectral_radius, block_stats
from .regimes import make_regimes
from .poisson_bank import generate_poisson_trains
from .io_bundle import write_bundle, write_manifest


def generate_bundle(cfg: GeneratorConfig, out_dir: Path | None = None) -> Path:
    """Run the full generation pipeline. Returns bundle directory path."""
    t0 = time.time()

    # ── Step 0: Validate ──
    validate_config(cfg)
    print(f"[0] Config OK  (N={cfg.N}, frac_I={cfg.frac_I}, "
          f"targets={cfg.rho_eff_targets})")

    # ── Step 1: Seeds ──
    seeds = split_seeds(cfg.seed)
    print(f"[1] Seeds: {seeds}")

    # ── Step 2: Population E/I ──
    pop = make_population(cfg.N, cfg.frac_I, seeds["graph"])
    print(f"[2] Population: N_E={pop['N_E']}, N_I={pop['N_I']}")

    # ── Step 3: Graph ──
    edges = generate_edges(cfg, pop, seeds["graph"])
    n_syn = len(edges["pre"])
    print(f"[3] Graph: {n_syn} synapses ({cfg.graph_type})")

    # ── Step 4: Neuron heterogeneity ──
    neuron_params = sample_neuron_params(cfg, pop, seeds["neurons"])
    print(f"[4] Neuron params: {len(neuron_params)} parameters, "
          f"{cfg.N} neurons each")

    # ── Step 5: Synapse params + balance rescaling + W ──
    syn_params = sample_synapse_params(cfg, edges, pop, seeds["synapses"])
    bal_before = float(np.abs(syn_params["w"][syn_params["w"] < 0]).sum() /
                       (syn_params["w"][syn_params["w"] > 0].sum() + 1e-15))
    bal_after = rescale_for_balance(syn_params, target=cfg.target_balance)
    W = build_sparse_W(cfg.N, edges, syn_params["w"])
    print(f"[5] Synapse params: {n_syn} synapses, W shape={W.shape}")
    print(f"    balance: {bal_before:.3f} → {bal_after:.3f} "
          f"(target={cfg.target_balance})")

    # ── Step 6: Spectral radius + stats ──
    stats = block_stats(W, pop)
    rho_base = stats["rho_full"]
    print(f"[6] rho_base={rho_base:.4f}  rho_EE={stats['rho_EE']:.4f}  "
          f"balance={stats['balance']:.3f}")

    if rho_base < 1e-10:
        raise RuntimeError(f"rho_base={rho_base} ~ 0, cannot define regimes")
    if rho_base > 1e4:
        print(f"  WARNING: rho_base={rho_base} extremely large")

    # ── Step 7: 5 regimes (simple mode from ρ targets) ──
    regimes = make_regimes(rho_base, cfg.rho_eff_targets)
    for r in regimes:
        print(f"  {r['name']:20s}  α={r['alpha']:.6f}  ρ_target={r['rho_target']:.2f}")
    print(f"    (use `python -m gen.calibrate_edge` for edge-of-chaos calibration)")

    # ── Step 8: Poisson bank ──
    n_ch = cfg.poisson_n_channels if cfg.poisson_n_channels > 0 else cfg.N
    poisson = generate_poisson_trains(
        n_ch, cfg.poisson_T_s, cfg.poisson_rate_hz, seeds["poisson"])
    total_spikes = sum(len(t) for t in poisson)
    print(f"[8] Poisson: {n_ch} channels, {cfg.poisson_T_s}s, "
          f"{total_spikes} total spikes")

    # ── Step 9: Write bundle ──
    if out_dir is None:
        out_dir = Path(cfg.bundle_dir) / f"bundle_seed_{cfg.seed}"

    bundle_path = write_bundle(
        out_dir, cfg, seeds, pop, edges, neuron_params,
        syn_params, stats, regimes, poisson)

    manifest_path = write_manifest(bundle_path)
    elapsed = time.time() - t0
    print(f"[9] Bundle written: {bundle_path}/")
    print(f"    manifest: {manifest_path}")
    print(f"    time: {elapsed:.1f}s")

    return bundle_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate one frozen reservoir bundle")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--frac-I", type=float, default=0.2)
    ap.add_argument("--graph-type", default="ER", choices=["ER", "fixed_indegree"])
    ap.add_argument("--p-conn", type=float, default=0.2)
    ap.add_argument("--k-in", type=int, default=20)
    ap.add_argument("--poisson-T-s", type=float, default=3.0)
    ap.add_argument("--poisson-rate-hz", type=float, default=10.0)
    ap.add_argument("--rho-targets", type=float, nargs=5,
                    default=[0.50, 0.75, 1.00, 1.25, 1.60])
    ap.add_argument("--bundle-dir", default="bundles")
    args = ap.parse_args()

    cfg = GeneratorConfig(
        seed=args.seed,
        N=args.N,
        frac_I=args.frac_I,
        graph_type=args.graph_type,
        p_conn=args.p_conn,
        k_in=args.k_in,
        poisson_T_s=args.poisson_T_s,
        poisson_rate_hz=args.poisson_rate_hz,
        rho_eff_targets=args.rho_targets,
        bundle_dir=args.bundle_dir,
    )

    generate_bundle(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

########################################
# FILE: gen/tests/__init__.py
########################################


########################################
# FILE: gen/tests/test_all.py
########################################

"""Comprehensive tests for all gen modules — no Brian2 required."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from gen.config import GeneratorConfig, validate_config, HeteroParam
from gen.seeding import split_seeds
from gen.population import make_population
from gen.graph_gen import generate_edges, BLOCK_ID_MAP
from gen.hetero_neurons import sample_neuron_params
from gen.hetero_synapses import sample_synapse_params, build_sparse_W, rescale_for_balance
from gen.spectral import spectral_radius, block_stats
from gen.regimes import make_regimes, make_regimes_from_edge
from gen.poisson_bank import generate_poisson_trains, save_poisson, load_poisson
from gen.io_bundle import write_bundle, write_manifest, verify_manifest, sha256_file
from gen.generate_one_bundle import generate_bundle


# ═══════════════════════════════════════════════════════════════════════════════
# 0) config
# ═══════════════════════════════════════════════════════════════════════════════

def test_validate_config_ok():
    cfg = GeneratorConfig()
    validate_config(cfg)  # should not raise


@pytest.mark.parametrize("field,value,match", [
    ("N", 0, "N must be > 0"),
    ("N", -5, "N must be > 0"),
    ("frac_I", 0.0, "frac_I must be in"),
    ("frac_I", 1.0, "frac_I must be in"),
    ("frac_I", -0.1, "frac_I must be in"),
    ("p_conn", 1.5, "p_conn must be in"),
    ("poisson_T_s", 0, "poisson_T_s must be > 0"),
    ("poisson_dt_ms", -1, "poisson_dt_ms must be > 0"),
])
def test_validate_config_rejects_bad_values(field, value, match):
    cfg = GeneratorConfig(**{field: value})
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_validate_config_rejects_wrong_target_count():
    cfg = GeneratorConfig(rho_eff_targets=[0.3, 0.7, 1.1])
    with pytest.raises(ValueError, match="5 elements"):
        validate_config(cfg)


def test_validate_config_rejects_unsorted_targets():
    cfg = GeneratorConfig(rho_eff_targets=[1.6, 0.7, 0.3, 1.1, 2.0])
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_config(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# 1) seeding
# ═══════════════════════════════════════════════════════════════════════════════

def test_split_seeds_deterministic():
    s1 = split_seeds(42)
    s2 = split_seeds(42)
    assert s1 == s2


def test_split_seeds_unique():
    s = split_seeds(42)
    vals = list(s.values())
    assert len(set(vals)) == len(vals), "sub-seeds must be unique"


def test_split_seeds_different_master():
    s1 = split_seeds(42)
    s2 = split_seeds(43)
    assert s1 != s2


# ═══════════════════════════════════════════════════════════════════════════════
# 2) population
# ═══════════════════════════════════════════════════════════════════════════════

def test_population_sizes():
    pop = make_population(100, 0.2, seed=0)
    assert pop["N_E"] == 80
    assert pop["N_I"] == 20
    assert len(pop["E_idx"]) == 80
    assert len(pop["I_idx"]) == 20


def test_population_disjoint_complete():
    pop = make_population(100, 0.2, seed=0)
    all_idx = np.concatenate([pop["E_idx"], pop["I_idx"]])
    assert len(np.unique(all_idx)) == 100
    assert set(all_idx) == set(range(100))


def test_population_deterministic():
    p1 = make_population(100, 0.2, seed=7)
    p2 = make_population(100, 0.2, seed=7)
    assert np.array_equal(p1["E_idx"], p2["E_idx"])
    assert np.array_equal(p1["I_idx"], p2["I_idx"])


def test_population_is_I_consistent():
    pop = make_population(100, 0.2, seed=0)
    for idx in pop["I_idx"]:
        assert pop["is_I"][idx] is np.True_
    for idx in pop["E_idx"]:
        assert pop["is_I"][idx] is np.False_


# ═══════════════════════════════════════════════════════════════════════════════
# 3) graph
# ═══════════════════════════════════════════════════════════════════════════════

def _make_edges(seed=0, graph_type="ER", allow_self=False):
    cfg = GeneratorConfig(N=50, frac_I=0.2, graph_type=graph_type,
                          p_conn=0.3, k_in=10, allow_self=allow_self)
    pop = make_population(50, 0.2, seed)
    return generate_edges(cfg, pop, seed), pop, cfg


def test_edges_in_range():
    edges, pop, cfg = _make_edges()
    assert edges["pre"].min() >= 0
    assert edges["pre"].max() < cfg.N
    assert edges["post"].min() >= 0
    assert edges["post"].max() < cfg.N


def test_edges_no_self_loops_if_disabled():
    edges, _, _ = _make_edges(allow_self=False)
    self_loops = np.sum(edges["pre"] == edges["post"])
    assert self_loops == 0


def test_edges_block_id_valid():
    edges, _, _ = _make_edges()
    assert set(np.unique(edges["block_id"])).issubset({0, 1, 2, 3})


def test_edges_deterministic():
    e1, _, _ = _make_edges(seed=42)
    e2, _, _ = _make_edges(seed=42)
    assert np.array_equal(e1["pre"], e2["pre"])
    assert np.array_equal(e1["post"], e2["post"])
    assert np.array_equal(e1["block_id"], e2["block_id"])


def test_edges_fixed_indegree():
    cfg = GeneratorConfig(N=50, frac_I=0.2, graph_type="fixed_indegree",
                          k_in=5, allow_self=False)
    pop = make_population(50, 0.2, 0)
    edges = generate_edges(cfg, pop, 0)
    assert len(edges["pre"]) > 0
    assert edges["pre"].min() >= 0
    assert edges["pre"].max() < 50


# ═══════════════════════════════════════════════════════════════════════════════
# 4) neuron heterogeneity
# ═══════════════════════════════════════════════════════════════════════════════

def test_neuron_params_shape():
    cfg = GeneratorConfig(N=50)
    pop = make_population(50, 0.2, 0)
    params = sample_neuron_params(cfg, pop, seed=0)
    for name, arr in params.items():
        assert arr.shape == (50,), f"{name} shape wrong"


def test_neuron_params_clamped():
    cfg = GeneratorConfig(N=500)
    pop = make_population(500, 0.2, 0)
    params = sample_neuron_params(cfg, pop, seed=0)
    for name, arr in params.items():
        hp = cfg.neuron_hetero[name]
        lo = hp.base * hp.clamp_lo
        hi = hp.base * hp.clamp_hi
        if hp.base > 0:
            assert arr.min() >= lo - 1e-12, f"{name} below clamp"
            assert arr.max() <= hi + 1e-12, f"{name} above clamp"


def test_neuron_params_no_nan_inf():
    cfg = GeneratorConfig(N=100)
    pop = make_population(100, 0.2, 0)
    params = sample_neuron_params(cfg, pop, seed=0)
    for name, arr in params.items():
        assert not np.any(np.isnan(arr)), f"{name} has NaN"
        assert not np.any(np.isinf(arr)), f"{name} has Inf"


def test_neuron_params_deterministic():
    cfg = GeneratorConfig(N=50)
    pop = make_population(50, 0.2, 0)
    p1 = sample_neuron_params(cfg, pop, seed=7)
    p2 = sample_neuron_params(cfg, pop, seed=7)
    for name in p1:
        assert np.array_equal(p1[name], p2[name])


# ═══════════════════════════════════════════════════════════════════════════════
# 5) synapse params
# ═══════════════════════════════════════════════════════════════════════════════

def _make_syn(seed=0):
    cfg = GeneratorConfig(N=50, frac_I=0.2, p_conn=0.3)
    pop = make_population(50, 0.2, seed)
    edges = generate_edges(cfg, pop, seed)
    syn = sample_synapse_params(cfg, edges, pop, seed)
    return syn, edges, pop, cfg


def test_syn_params_lengths_match_edges():
    syn, edges, _, _ = _make_syn()
    n = len(edges["pre"])
    for key in ("w", "U", "tau_d_ms", "tau_f_ms", "delay_ms"):
        assert len(syn[key]) == n, f"{key} length mismatch"


def test_weight_signs_match_pre_type():
    syn, edges, pop, _ = _make_syn()
    is_I = pop["is_I"]
    for i in range(len(edges["pre"])):
        pre_idx = edges["pre"][i]
        w = syn["w"][i]
        if is_I[pre_idx]:
            assert w <= 0, f"I→? synapse {i} has positive weight"
        else:
            assert w >= 0, f"E→? synapse {i} has negative weight"


def test_delay_in_range():
    syn, _, _, cfg = _make_syn()
    assert syn["delay_ms"].min() >= cfg.delay_min_ms - 1e-12
    assert syn["delay_ms"].max() <= cfg.delay_max_ms + 1e-12


def test_build_W_shape():
    syn, edges, _, cfg = _make_syn()
    W = build_sparse_W(cfg.N, edges, syn["w"])
    assert W.shape == (cfg.N, cfg.N)


def test_build_W_nonzero_matches_edges():
    syn, edges, _, cfg = _make_syn()
    W = build_sparse_W(cfg.N, edges, syn["w"])
    # nnz should equal number of edges (some might overlap → csr sums them)
    # but typically no overlap in ER: just check > 0 and <= n_edges
    assert W.nnz > 0
    assert W.nnz <= len(edges["pre"])


# ═══════════════════════════════════════════════════════════════════════════════
# 6) spectral
# ═══════════════════════════════════════════════════════════════════════════════

def test_spectral_radius_positive():
    syn, edges, pop, cfg = _make_syn()
    W = build_sparse_W(cfg.N, edges, syn["w"])
    rho = spectral_radius(W)
    assert rho > 0


def test_spectral_radius_deterministic():
    syn, edges, pop, cfg = _make_syn(seed=42)
    W = build_sparse_W(cfg.N, edges, syn["w"])
    r1 = spectral_radius(W)
    r2 = spectral_radius(W)
    assert r1 == r2


def test_spectral_radius_known_small():
    # 2x2: [[0, 2], [2, 0]] → eigenvalues ±2 → ρ=2
    W = sparse.csr_matrix(np.array([[0, 2], [2, 0]], dtype=np.float64))
    rho = spectral_radius(W, n_iter=200)
    assert abs(rho - 2.0) < 0.01


def test_block_stats_keys():
    syn, edges, pop, cfg = _make_syn()
    W = build_sparse_W(cfg.N, edges, syn["w"])
    stats = block_stats(W, pop)
    for key in ("rho_full", "rho_EE", "norm_EE", "norm_EI", "norm_IE",
                "norm_II", "balance"):
        assert key in stats


# ═══════════════════════════════════════════════════════════════════════════════
# 7) regimes
# ═══════════════════════════════════════════════════════════════════════════════

def test_regimes_len_5():
    regs = make_regimes(10.0, [0.3, 0.7, 1.0, 1.3, 1.6])
    assert len(regs) == 5


def test_regimes_alpha_monotonic_if_targets_monotonic():
    regs = make_regimes(10.0, [0.3, 0.7, 1.0, 1.3, 1.6])
    alphas = [r["alpha"] for r in regs]
    assert alphas == sorted(alphas)


def test_regimes_alpha_computation():
    regs = make_regimes(5.0, [0.3, 0.7, 1.0, 1.3, 1.6])
    for r in regs:
        expected = r["rho_target"] / 5.0
        assert abs(r["alpha"] - expected) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 8) poisson
# ═══════════════════════════════════════════════════════════════════════════════

def test_poisson_times_in_range():
    trains = generate_poisson_trains(10, 3.0, 10.0, seed=0)
    for ch, t in enumerate(trains):
        if len(t) > 0:
            assert t.min() >= 0, f"ch{ch} times < 0"
            assert t.max() < 3.0, f"ch{ch} times >= T"


def test_poisson_sorted():
    trains = generate_poisson_trains(10, 3.0, 10.0, seed=0)
    for ch, t in enumerate(trains):
        assert np.all(np.diff(t) >= 0), f"ch{ch} not sorted"


def test_poisson_deterministic():
    t1 = generate_poisson_trains(10, 3.0, 10.0, seed=42)
    t2 = generate_poisson_trains(10, 3.0, 10.0, seed=42)
    for ch in range(10):
        assert np.array_equal(t1[ch], t2[ch])


def test_poisson_rate_sanity():
    n_ch = 100
    T = 3.0
    rate = 10.0
    trains = generate_poisson_trains(n_ch, T, rate, seed=0)
    counts = [len(t) for t in trains]
    mean_count = np.mean(counts)
    expected = rate * T
    # Within 6σ: σ = sqrt(rate*T) per channel, mean over 100 → σ_mean ~ σ/10
    tol = 6 * np.sqrt(rate * T) / np.sqrt(n_ch)
    assert abs(mean_count - expected) < tol, \
        f"mean_count={mean_count:.1f} vs expected={expected:.1f}"


def test_poisson_save_load(tmp_path):
    trains = generate_poisson_trains(5, 2.0, 10.0, seed=7)
    path = tmp_path / "trains.npz"
    save_poisson(trains, path, T_s=2.0, rate_hz=10.0, seed=7)
    loaded, meta = load_poisson(path)
    assert meta["n_channels"] == 5
    assert meta["T_s"] == 2.0
    for ch in range(5):
        assert np.array_equal(trains[ch], loaded[ch])


# ═══════════════════════════════════════════════════════════════════════════════
# 9) io_bundle
# ═══════════════════════════════════════════════════════════════════════════════

def test_bundle_files_exist(tmp_path):
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "test_bundle"
    generate_bundle(cfg, out_dir=out)

    assert (out / "config.json").exists()
    assert (out / "network" / "population.json").exists()
    assert (out / "network" / "edges.npz").exists()
    assert (out / "network" / "neuron_params.npz").exists()
    assert (out / "network" / "synapse_params.npz").exists()
    assert (out / "network" / "base_stats.json").exists()
    assert (out / "regimes" / "regimes.json").exists()
    assert (out / "poisson" / "trains_3s.npz").exists()
    assert (out / "manifest.json").exists()


def test_manifest_has_hashes_for_all_files(tmp_path):
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "test_bundle"
    generate_bundle(cfg, out_dir=out)

    manifest = json.loads((out / "manifest.json").read_text())
    # All npz and json (except manifest) should be in files
    for p in out.rglob("*.npz"):
        rel = str(p.relative_to(out))
        assert rel in manifest["files"], f"{rel} not in manifest"
    for p in out.rglob("*.json"):
        if p.name != "manifest.json":
            rel = str(p.relative_to(out))
            assert rel in manifest["files"], f"{rel} not in manifest"


def test_reload_npz_roundtrip(tmp_path):
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "test_bundle"
    generate_bundle(cfg, out_dir=out)

    edges = np.load(out / "network" / "edges.npz")
    assert "pre" in edges
    assert "post" in edges
    assert "block_id" in edges


def test_manifest_verify_ok(tmp_path):
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "test_bundle"
    generate_bundle(cfg, out_dir=out)
    errors = verify_manifest(out)
    assert errors == []


# ═══════════════════════════════════════════════════════════════════════════════
# 10) end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

def test_generate_bundle_end_to_end(tmp_path):
    cfg = GeneratorConfig(N=50, frac_I=0.2, p_conn=0.2, poisson_T_s=1.0,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "e2e"
    path = generate_bundle(cfg, out_dir=out)
    assert path.exists()
    assert (path / "manifest.json").exists()

    # Regimes check
    regs = json.loads((path / "regimes" / "regimes.json").read_text())
    assert len(regs) == 5
    alphas = [r["alpha"] for r in regs]
    assert alphas == sorted(alphas)


def test_generate_bundle_reproducible(tmp_path):
    """Same seed → identical file hashes."""
    cfg = GeneratorConfig(N=30, frac_I=0.2, p_conn=0.2, poisson_T_s=0.5,
                          seed=42, rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])

    out1 = tmp_path / "run1"
    generate_bundle(cfg, out_dir=out1)

    out2 = tmp_path / "run2"
    generate_bundle(cfg, out_dir=out2)

    # Compare all npz file hashes
    for p1 in sorted(out1.rglob("*.npz")):
        rel = p1.relative_to(out1)
        p2 = out2 / rel
        assert p2.exists(), f"{rel} missing in run2"
        h1 = sha256_file(p1)
        h2 = sha256_file(p2)
        assert h1 == h2, f"hash mismatch for {rel}"


# ═══════════════════════════════════════════════════════════════════════════════
# 11) balance control
# ═══════════════════════════════════════════════════════════════════════════════

def test_balance_close_to_target():
    cfg = GeneratorConfig(N=100, frac_I=0.2, p_conn=0.2, target_balance=1.0)
    pop = make_population(100, 0.2, 0)
    edges = generate_edges(cfg, pop, 0)
    syn = sample_synapse_params(cfg, edges, pop, 0)
    bal_after = rescale_for_balance(syn, target=1.0)
    assert abs(bal_after - 1.0) < 0.01, f"balance={bal_after:.3f}, expected ~1.0"


def test_balance_different_targets():
    cfg = GeneratorConfig(N=100, frac_I=0.2, p_conn=0.2)
    pop = make_population(100, 0.2, 0)
    edges = generate_edges(cfg, pop, 0)
    for target in [0.8, 1.0, 1.2]:
        syn = sample_synapse_params(cfg, edges, pop, 0)
        bal = rescale_for_balance(syn, target=target)
        assert abs(bal - target) < 0.02, f"target={target}, got {bal:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 12) regimes spread
# ═══════════════════════════════════════════════════════════════════════════════

def test_regimes_spread():
    """Consecutive α ratio should be > 1.25 for default 5-regime targets."""
    regs = make_regimes(10.0, [0.40, 0.70, 1.00, 1.40, 2.00])
    alphas = [r["alpha"] for r in regs]
    for i in range(len(alphas) - 1):
        ratio = alphas[i+1] / alphas[i]
        assert ratio >= 1.25, f"ratio[{i}→{i+1}]={ratio:.2f} < 1.25"


# ═══════════════════════════════════════════════════════════════════════════════
# 13) smoke pass/fail logic (no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

from gen.brian_smoke import (evaluate_status, evaluate_scan_liveness,
                             check_separation, SmokeMetrics,
                             evaluate_5regime_quality)


def test_evaluate_status_ok():
    assert evaluate_status(20.0, 30.0, False) == "OK"


def test_evaluate_status_nan():
    assert evaluate_status(20.0, 30.0, True) == "NAN"


def test_evaluate_status_silent():
    assert evaluate_status(0.1, 0.1, False) == "SILENT"


def test_evaluate_status_runaway():
    assert evaluate_status(250.0, 30.0, False) == "RUNAWAY"
    assert evaluate_status(20.0, 350.0, False) == "RUNAWAY"


def _make_smoke_metrics(rate_E, cv_isi_E, fano_E, sync_E, idx=0,
                        pct_silent_E=0.0, status="OK"):
    return SmokeMetrics(
        regime_name=f"R{idx}", regime_index=idx, alpha=0.01*(idx+1),
        rate_E=rate_E, rate_I=rate_E*1.2,
        cv_isi_E=cv_isi_E, cv_isi_I=cv_isi_E,
        fano_E=fano_E, fano_I=fano_E,
        sync_E=sync_E, sync_I=sync_E,
        pct_silent_E=pct_silent_E, pct_silent_I=pct_silent_E,
        spike_count_E=int(rate_E * 100), spike_count_I=int(rate_E * 25),
        has_nan=False, status=status,
    )


def test_check_separation_pass():
    """Metrics with clear spread should PASS."""
    results = [
        _make_smoke_metrics(10.0, 0.4, 0.4, 1.0, 0),
        _make_smoke_metrics(20.0, 0.5, 0.6, 1.3, 1),
        _make_smoke_metrics(35.0, 0.7, 0.8, 1.6, 2),
        _make_smoke_metrics(50.0, 0.9, 1.1, 2.0, 3),
        _make_smoke_metrics(70.0, 1.1, 1.5, 2.5, 4),
    ]
    passed, details = check_separation(results)
    assert passed


def test_check_separation_fail():
    """Nearly identical metrics should FAIL separation."""
    results = [
        _make_smoke_metrics(20.0, 0.50, 0.50, 1.00, 0),
        _make_smoke_metrics(20.1, 0.50, 0.50, 1.01, 1),
        _make_smoke_metrics(20.0, 0.51, 0.50, 1.00, 2),
        _make_smoke_metrics(20.1, 0.50, 0.51, 1.01, 3),
        _make_smoke_metrics(20.0, 0.50, 0.50, 1.00, 4),
    ]
    passed, details = check_separation(results)
    assert not passed


def test_smoke_loads_bundle(tmp_path):
    """Verify bundle can be loaded (without Brian2 simulation)."""
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "smoke_bundle"
    generate_bundle(cfg, out_dir=out)

    # Verify all files loadable
    import json
    regs = json.loads((out / "regimes" / "regimes.json").read_text())
    assert len(regs) == 5
    edges = np.load(out / "network" / "edges.npz")
    assert "pre" in edges
    poi = np.load(out / "poisson" / "trains_3s.npz")
    assert int(poi["n_channels"]) == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 14) make_regimes_from_edge
# ═══════════════════════════════════════════════════════════════════════════════

def test_regimes_from_edge_len_5():
    regs = make_regimes_from_edge(0.01, rho_base=100.0)
    assert len(regs) == 5


def test_regimes_from_edge_center_is_alpha_edge():
    alpha_edge = 0.015
    regs = make_regimes_from_edge(alpha_edge, rho_base=50.0)
    # R3 (index 2) should have multiplier 1.00 → alpha == alpha_edge
    r3 = regs[2]
    assert abs(r3["alpha"] - alpha_edge) < 1e-12
    assert r3["multiplier"] == 1.0


def test_regimes_from_edge_multipliers_applied():
    alpha_edge = 0.02
    mults = [0.40, 0.70, 1.00, 1.40, 2.00]
    regs = make_regimes_from_edge(alpha_edge, rho_base=50.0, multipliers=mults)
    for r, m in zip(regs, mults):
        expected_alpha = alpha_edge * m
        assert abs(r["alpha"] - expected_alpha) < 1e-12
        assert abs(r["multiplier"] - m) < 1e-12


def test_regimes_from_edge_alpha_monotonic():
    regs = make_regimes_from_edge(0.01, rho_base=100.0)
    alphas = [r["alpha"] for r in regs]
    assert alphas == sorted(alphas)


def test_regimes_from_edge_has_metadata():
    regs = make_regimes_from_edge(0.01, rho_base=100.0)
    for r in regs:
        assert "alpha_edge" in r
        assert "multiplier" in r
        assert r["alpha_edge"] == 0.01


def test_regimes_from_edge_custom_multipliers():
    mults = [0.3, 0.6, 1.0, 1.5, 2.0]
    regs = make_regimes_from_edge(0.01, rho_base=50.0, multipliers=mults)
    assert len(regs) == 5
    assert regs[0]["alpha"] == pytest.approx(0.003)
    assert regs[4]["alpha"] == pytest.approx(0.02)


# ═══════════════════════════════════════════════════════════════════════════════
# 15) scan liveness gate
# ═══════════════════════════════════════════════════════════════════════════════

def test_scan_liveness_pass():
    assert evaluate_scan_liveness(20.0, 30.0, 5.0, 0.9, False) == "PASS"


def test_scan_liveness_fail_nan():
    assert evaluate_scan_liveness(20.0, 30.0, 5.0, 0.9, True) == "FAIL"


def test_scan_liveness_fail_runaway():
    assert evaluate_scan_liveness(250.0, 30.0, 5.0, 0.9, False) == "FAIL"


def test_scan_liveness_fail_silent():
    assert evaluate_scan_liveness(0.05, 10.0, 85.0, 0.9, False) == "FAIL"


def test_scan_liveness_warn_high_silent():
    assert evaluate_scan_liveness(10.0, 15.0, 55.0, 0.9, False) == "WARN"


def test_scan_liveness_warn_high_sync():
    assert evaluate_scan_liveness(20.0, 30.0, 5.0, 1.5, False) == "WARN"


# ═══════════════════════════════════════════════════════════════════════════════
# 16) 5-regime quality evaluation (no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_5regime_quality_all_ok():
    """5 healthy regimes with good separation → overall PASS."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3),
        _make_smoke_metrics(50.0, 0.80, 1.00, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "PASS"


def test_5regime_quality_r5_runaway_is_warn():
    """R5 runaway → WARN, not FAIL."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3),
        _make_smoke_metrics(250.0, 1.00, 1.30, 2.0, 4, status="RUNAWAY"),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "WARN"
    assert q["per_regime"][4]["verdict"] == "WARN"


def test_5regime_quality_r2_nan_is_fail():
    """R2 NaN → overall FAIL."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.3, 0.3, 0.8, 0),
        SmokeMetrics(
            regime_name=REGIME_NAMES[1], regime_index=1, alpha=0.02,
            rate_E=0.0, rate_I=0.0,
            cv_isi_E=0.0, cv_isi_I=0.0,
            fano_E=0.0, fano_I=0.0,
            sync_E=0.0, sync_I=0.0,
            pct_silent_E=100.0, pct_silent_I=100.0,
            spike_count_E=0, spike_count_I=0,
            has_nan=True, status="NAN",
        ),
        _make_smoke_metrics(25.0, 0.6, 0.7, 1.2, 2),
        _make_smoke_metrics(35.0, 0.8, 1.0, 1.5, 3),
        _make_smoke_metrics(50.0, 1.0, 1.3, 2.0, 4),
    ]
    for i, name in enumerate(REGIME_NAMES):
        results[i].regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "FAIL"


def test_5regime_quality_separation_r1_vs_r5():
    """CV(R5) >= 1.15*CV(R1) and Fano(R5) >= 1.15*Fano(R1) → PASS."""
    from gen.config import REGIME_NAMES
    # R5 CV=0.80 vs R1 CV=0.30 → ratio=2.67 >> 1.15  PASS
    # R5 Fano=1.00 vs R1 Fano=0.30 → ratio=3.33 >> 1.15  PASS
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3),
        _make_smoke_metrics(50.0, 0.80, 1.00, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "PASS"


def test_5regime_quality_separation_warn_low_ratio():
    """CV(R5) < 1.10*CV(R1) → WARN."""
    from gen.config import REGIME_NAMES
    # R1 cv=0.50, R5 cv=0.52 → ratio=1.04 < 1.10 → WARN
    results = [
        _make_smoke_metrics(5.0, 0.50, 0.50, 0.8, 0),
        _make_smoke_metrics(15.0, 0.51, 0.51, 1.0, 1),
        _make_smoke_metrics(25.0, 0.51, 0.51, 1.2, 2),
        _make_smoke_metrics(35.0, 0.52, 0.52, 1.5, 3),
        _make_smoke_metrics(50.0, 0.52, 0.52, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "WARN"


def test_5regime_quality_high_silent_on_r3_is_warn():
    """pct_silent_E > 40% on R3 → WARN."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2, pct_silent_E=50.0),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3),
        _make_smoke_metrics(50.0, 0.80, 1.00, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "WARN"
    assert q["per_regime"][2]["verdict"] == "WARN"


# ═══════════════════════════════════════════════════════════════════════════════
# 17) edge config validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_validate_config_edge_multipliers_ok():
    cfg = GeneratorConfig()
    validate_config(cfg)  # default multipliers should be valid


def test_validate_config_rejects_wrong_multiplier_count():
    cfg = GeneratorConfig(edge_multipliers=[0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match="edge_multipliers must have 5"):
        validate_config(cfg)


def test_validate_config_rejects_unsorted_multipliers():
    cfg = GeneratorConfig(edge_multipliers=[1.60, 1.25, 1.00, 0.75, 0.50])
    with pytest.raises(ValueError, match="edge_multipliers must be sorted"):
        validate_config(cfg)


def test_validate_config_rejects_bad_edge_thresholds():
    cfg = GeneratorConfig(edge_cv_thr=0)
    with pytest.raises(ValueError, match="edge_cv_thr"):
        validate_config(cfg)

    cfg = GeneratorConfig(edge_fano_thr=-1)
    with pytest.raises(ValueError, match="edge_fano_thr"):
        validate_config(cfg)


def test_validate_config_rejects_too_few_scan_points():
    cfg = GeneratorConfig(edge_n_scan=2)
    with pytest.raises(ValueError, match="edge_n_scan"):
        validate_config(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# 18) edge algorithm unit tests (no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

from gen.regimes import (_find_edge_ab, _find_edge_fallback,
                         _build_alpha_grid, _build_refine_grid,
                         save_edge_regimes)


def test_find_edge_ab_basic():
    """Step A+B finds first point meeting thresholds with trend OK."""
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.30, "fano_E": 0.30,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.02, "cv_isi_E": 0.40, "fano_E": 0.45,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.03, "cv_isi_E": 0.50, "fano_E": 0.55,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.04, "cv_isi_E": 0.60, "fano_E": 0.65,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge == 0.03  # first point crossing both thresholds


def test_find_edge_ab_skips_fail_gate():
    """Points with gate=FAIL are skipped even if thresholds met."""
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.50, "fano_E": 0.60,
         "pct_silent_E": 5.0, "gate": "FAIL"},
        {"alpha": 0.02, "cv_isi_E": 0.50, "fano_E": 0.60,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge == 0.02


def test_find_edge_ab_skips_high_silent():
    """Points with %silent > thr are not candidates."""
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.50, "fano_E": 0.60,
         "pct_silent_E": 50.0, "gate": "PASS"},
        {"alpha": 0.02, "cv_isi_E": 0.50, "fano_E": 0.60,
         "pct_silent_E": 10.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge == 0.02


def test_find_edge_ab_trend_rejects_dip():
    """A CV dip > trend_tol rejects the candidate (trend uses prev regardless of gate)."""
    scan = [
        # k=0: high CV but gate=FAIL → skipped by Step A, but CV used for trend at k=1
        {"alpha": 0.01, "cv_isi_E": 0.60, "fano_E": 0.60,
         "pct_silent_E": 5.0, "gate": "FAIL"},
        # k=1: passes Step A (CV≥0.45, Fano≥0.50) but trend rejects:
        #   CV[1]=0.55 < CV[0]-tol = 0.60-0.02 = 0.58 → REJECT
        {"alpha": 0.02, "cv_isi_E": 0.55, "fano_E": 0.60,
         "pct_silent_E": 5.0, "gate": "PASS"},
        # k=2: passes Step A but trend rejects:
        #   CV[2]=0.50 < CV[1]-tol = 0.55-0.02 = 0.53 → REJECT
        {"alpha": 0.03, "cv_isi_E": 0.50, "fano_E": 0.65,
         "pct_silent_E": 5.0, "gate": "PASS"},
        # k=3: passes Step A AND trend OK:
        #   CV[3]=0.56 >= CV[2]-tol = 0.50-0.02 = 0.48 ✓
        #   Fano[3]=0.70 >= Fano[2]-tol = 0.65-0.03 = 0.62 ✓
        {"alpha": 0.04, "cv_isi_E": 0.56, "fano_E": 0.70,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    # k=0 FAIL gate, k=1 trend reject, k=2 trend reject, k=3 passes
    assert edge == 0.04


def test_find_edge_ab_returns_none_when_nothing_qualifies():
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.20, "fano_E": 0.20,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.02, "cv_isi_E": 0.30, "fano_E": 0.30,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge is None


def test_find_edge_fallback_uses_chaos_score():
    """Fallback picks argmax(z(CV) + z(Fano))."""
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.20, "fano_E": 0.20,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.02, "cv_isi_E": 0.35, "fano_E": 0.40,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.03, "cv_isi_E": 0.30, "fano_E": 0.35,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_fallback(scan, silent_thr=40.0)
    assert edge == 0.02  # highest CV + Fano combined


def test_find_edge_fallback_returns_none_all_fail():
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.20, "fano_E": 0.20,
         "pct_silent_E": 5.0, "gate": "FAIL"},
    ]
    assert _find_edge_fallback(scan, silent_thr=40.0) is None


def test_build_alpha_grid_deterministic():
    """Same rho_base + cfg → same grid."""
    cfg = GeneratorConfig()
    g1 = _build_alpha_grid(50.0, cfg)
    g2 = _build_alpha_grid(50.0, cfg)
    assert np.array_equal(g1, g2)


def test_build_alpha_grid_range():
    """Grid spans from rho_eff_min/rho_base to rho_eff_max/rho_base."""
    cfg = GeneratorConfig(edge_rho_eff_min=0.1, edge_rho_eff_max=3.0,
                          edge_n_scan=16)
    rho = 50.0
    grid = _build_alpha_grid(rho, cfg)
    assert len(grid) == 16
    assert abs(grid[0] - 0.1 / 50.0) < 1e-12
    assert abs(grid[-1] - 3.0 / 50.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 18b) refinement grid tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_refine_grid_middle():
    """k* in middle → linspace between neighbors, excluding endpoints."""
    alphas = np.array([0.01, 0.02, 0.04, 0.08])
    grid = _build_refine_grid(alphas, k_star=2, n_refine=4)
    assert len(grid) == 4
    # Should be strictly between alpha[1]=0.02 and alpha[3]=0.08
    assert grid[0] > 0.02
    assert grid[-1] < 0.08
    # Should be sorted ascending
    assert np.all(np.diff(grid) > 0)


def test_refine_grid_last_point():
    """k* at K-1 → extend upward from last coarse point."""
    alphas = np.array([0.01, 0.02, 0.04, 0.08])
    grid = _build_refine_grid(alphas, k_star=3, n_refine=6)
    assert len(grid) == 6
    # All points above last coarse point
    assert grid[0] > 0.08
    assert grid[-1] <= 0.08 * 2
    assert np.all(np.diff(grid) > 0)


def test_refine_grid_first_point():
    """k* at 0 → extend downward from first coarse point."""
    alphas = np.array([0.01, 0.02, 0.04, 0.08])
    grid = _build_refine_grid(alphas, k_star=0, n_refine=6)
    assert len(grid) == 6
    # All points below first coarse point
    assert grid[-1] < 0.01
    assert grid[0] >= 0.01 / 2
    assert np.all(np.diff(grid) > 0)


def test_refine_grid_no_overlap_with_coarse():
    """Refinement grid should not contain the coarse endpoints."""
    alphas = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
    # Middle case
    grid_mid = _build_refine_grid(alphas, k_star=2, n_refine=6)
    for a in grid_mid:
        assert a != 0.02 and a != 0.08  # no overlap with neighbors
    # Last case
    grid_last = _build_refine_grid(alphas, k_star=4, n_refine=6)
    assert 0.16 not in grid_last
    # First case
    grid_first = _build_refine_grid(alphas, k_star=0, n_refine=6)
    assert 0.01 not in grid_first


def test_merged_results_sorted_by_alpha():
    """Merging coarse + refine and sorting by alpha works correctly."""
    coarse = [
        {"alpha": 0.01, "cv_isi_E": 0.3, "fano_E": 0.3,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.04, "cv_isi_E": 0.5, "fano_E": 0.6,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    refine = [
        {"alpha": 0.02, "cv_isi_E": 0.4, "fano_E": 0.45,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.03, "cv_isi_E": 0.46, "fano_E": 0.52,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    merged = sorted(coarse + refine, key=lambda r: r["alpha"])
    alphas = [r["alpha"] for r in merged]
    assert alphas == sorted(alphas)
    assert len(merged) == 4
    # A+B on merged should find 0.03 (first passing cv≥0.45, fano≥0.50)
    edge = _find_edge_ab(merged, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge == 0.03


# ═══════════════════════════════════════════════════════════════════════════════
# 19) User-specified test: extreme alpha → FAIL
# ═══════════════════════════════════════════════════════════════════════════════

def test_extreme_alpha_silent():
    """Very low α → SILENT → scan gate FAIL."""
    assert evaluate_scan_liveness(
        rate_E=0.05, rate_I=0.1, pct_silent_E=90.0,
        sync_E=0.5, has_nan=False) == "FAIL"


def test_extreme_alpha_runaway():
    """Very high α → RUNAWAY → scan gate FAIL."""
    assert evaluate_scan_liveness(
        rate_E=300.0, rate_I=400.0, pct_silent_E=0.0,
        sync_E=0.5, has_nan=False) == "FAIL"


# ═══════════════════════════════════════════════════════════════════════════════
# 20) User-specified test: 5-regime monotonic trend (soft)
# ═══════════════════════════════════════════════════════════════════════════════

def test_monotonic_trend_soft():
    """CV(R5) - CV(R1) >= 0.05 OR CV(R5) >= 1.15*CV(R1).
    Analogously for Fano."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.35, 0.40, 1.0, 1),
        _make_smoke_metrics(25.0, 0.42, 0.50, 1.2, 2),
        _make_smoke_metrics(35.0, 0.50, 0.60, 1.5, 3),
        _make_smoke_metrics(50.0, 0.60, 0.75, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name

    cv_r1 = results[0].cv_isi_E
    cv_r5 = results[4].cv_isi_E
    fano_r1 = results[0].fano_E
    fano_r5 = results[4].fano_E

    # Check "net increase" criteria
    cv_ok = (cv_r5 - cv_r1 >= 0.05) or (cv_r5 >= 1.15 * cv_r1)
    fano_ok = (fano_r5 - fano_r1 >= 0.05) or (fano_r5 >= 1.15 * fano_r1)
    assert cv_ok, f"CV trend fail: R1={cv_r1:.2f}, R5={cv_r5:.2f}"
    assert fano_ok, f"Fano trend fail: R1={fano_r1:.2f}, R5={fano_r5:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 21) User-specified test: silent guard
# ═══════════════════════════════════════════════════════════════════════════════

def test_silent_guard_r3_r5():
    """If %silent_E > 40% in R3, R4, or R5 → quality WARN or FAIL."""
    from gen.config import REGIME_NAMES
    # R4 has 50% silent
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3, pct_silent_E=50.0),
        _make_smoke_metrics(50.0, 0.80, 1.00, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["per_regime"][3]["verdict"] == "WARN"
    assert q["overall"] in ("WARN", "FAIL")


# ═══════════════════════════════════════════════════════════════════════════════
# 22) save_edge_regimes with scan_info dict
# ═══════════════════════════════════════════════════════════════════════════════

def test_save_edge_regimes_writes_structured_metadata(tmp_path):
    """save_edge_regimes writes coarse_grid, refine_grid, edge_rule, source."""
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "regimes").mkdir(parents=True)

    scan_info = {
        "coarse_grid": [
            {"alpha": 0.01, "cv_isi_E": 0.3, "fano_E": 0.3,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.02, "cv_isi_E": 0.5, "fano_E": 0.6,
             "pct_silent_E": 5.0, "gate": "PASS"},
        ],
        "refine_grid": [
            {"alpha": 0.015, "cv_isi_E": 0.46, "fano_E": 0.52,
             "pct_silent_E": 5.0, "gate": "PASS"},
        ],
        "merged_grid": [
            {"alpha": 0.01, "cv_isi_E": 0.3, "fano_E": 0.3,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.015, "cv_isi_E": 0.46, "fano_E": 0.52,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.02, "cv_isi_E": 0.5, "fano_E": 0.6,
             "pct_silent_E": 5.0, "gate": "PASS"},
        ],
        "edge_rule": {
            "cv_thr": 0.45, "fano_thr": 0.50,
            "silent_thr": 40.0, "cv_trend_tol": 0.02,
            "fano_trend_tol": 0.03,
        },
        "alpha_edge_source": "refine_A+B_merged_k1",
        "refine_direction": "zoom",
    }

    regimes = save_edge_regimes(
        bundle_dir, alpha_edge=0.015, rho_base=100.0,
        scan_info=scan_info, edge_method="metrics_scan_2stage",
    )
    assert len(regimes) == 5

    data = json.loads((bundle_dir / "regimes" / "regimes.json").read_text())
    assert data["alpha_edge"] == 0.015
    assert data["edge_method"] == "metrics_scan_2stage"
    assert data["alpha_edge_source"] == "refine_A+B_merged_k1"
    assert data["refine_direction"] == "zoom"
    assert "edge_rule" in data
    assert data["edge_rule"]["cv_thr"] == 0.45
    assert len(data["coarse_grid"]) == 2
    assert len(data["refine_grid"]) == 1
    assert len(data["regimes"]) == 5


def test_save_edge_regimes_r5_clamp_uses_merged(tmp_path):
    """R5 clamp logic reads gate from merged_grid in scan_info."""
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "regimes").mkdir(parents=True)

    scan_info = {
        "coarse_grid": [],
        "refine_grid": [],
        "merged_grid": [
            {"alpha": 0.01, "cv_isi_E": 0.5, "fano_E": 0.5,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.02, "cv_isi_E": 0.6, "fano_E": 0.6,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.03, "cv_isi_E": 0.0, "fano_E": 0.0,
             "pct_silent_E": 90.0, "gate": "FAIL"},
        ],
        "edge_rule": {},
        "alpha_edge_source": "coarse_k1_A+B",
        "refine_direction": None,
    }

    # alpha_edge=0.015, R5 mult=2.00 → R5 α=0.030
    # max_ok=0.02, R5>max_ok*1.1=0.022 → should clamp
    regimes = save_edge_regimes(
        bundle_dir, alpha_edge=0.015, rho_base=100.0,
        scan_info=scan_info,
    )
    data = json.loads((bundle_dir / "regimes" / "regimes.json").read_text())
    assert data["r5_clamped"] is True
    assert data["r5_final_multiplier"] < 2.00


def test_save_edge_regimes_stability_check_passthrough(tmp_path):
    """Stability check result is written to regimes.json when present."""
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "regimes").mkdir(parents=True)

    scan_info = {
        "coarse_grid": [],
        "refine_grid": [],
        "merged_grid": [
            {"alpha": 0.02, "cv_isi_E": 0.5, "fano_E": 0.5,
             "pct_silent_E": 5.0, "gate": "PASS"},
        ],
        "edge_rule": {},
        "alpha_edge_source": "coarse_k0_A+B",
        "refine_direction": None,
        "stability_check": {
            "stable": True,
            "cv_scan": 0.50, "cv_long": 0.52,
            "cv_rel_diff": 0.04,
            "fano_scan": 0.50, "fano_long": 0.48,
            "fano_rel_diff": 0.04,
            "tol": 0.10,
            "measure_ms": 4000.0,
        },
    }

    save_edge_regimes(
        bundle_dir, alpha_edge=0.02, rho_base=100.0,
        scan_info=scan_info,
    )
    data = json.loads((bundle_dir / "regimes" / "regimes.json").read_text())
    assert "stability_check" in data
    assert data["stability_check"]["stable"] is True
    assert data["stability_check"]["cv_scan"] == 0.50
