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
    bundle_version: str = "2.1"

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
