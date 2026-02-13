"""
Edge-of-chaos calibration: Dense Scan + Brentq + Recovery.

Finds inh_scaling values where λ≈0, λ<0 (stable), and λ>0 (chaotic).
Constrained by plausibility checks.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from .lyapunov import estimate_lyapunov, LyapunovResult
from .plausibility import run_plausibility_checks, PlausibilityResult
from .build_reservoir import Reservoir

logger = logging.getLogger(__name__)


@dataclass
class ScanPoint:
    """Result at a single inh_scaling value."""
    inh_scaling: float
    lambda_est: float
    lambda_std: float
    kernel_quality: float = 0.0     # Legenstein-Maass KQ (rank of state matrix)
    plausibility_ok: bool = True
    plausibility_result: Optional[PlausibilityResult] = None
    lyapunov_result: Optional[LyapunovResult] = None


@dataclass
class CalibrationResult:
    """Result of edge-of-chaos calibration."""
    inh_scaling_edge: float
    inh_scaling_stable: float
    inh_scaling_chaos: float
    lambda_edge: float
    lambda_stable: float
    lambda_chaos: float
    edge_plausibility_ok: bool
    method_used: str           # "brentq" or "recovery"
    scan_points: List[ScanPoint] = field(default_factory=list)
    brentq_converged: bool = False
    n_invalid_biology: int = 0
    notes: str = ""


def dense_scan(reservoir: Reservoir, cfg: dict, seed_input: int,
               seed_lyapunov: int) -> List[ScanPoint]:
    """
    Dense grid scan of inh_scaling to map λ landscape.

    Args:
        reservoir: Reservoir with frozen topology
        cfg: configuration dict
        seed_input: input seed
        seed_lyapunov: lyapunov perturbation seed

    Returns:
        List of ScanPoints sorted by inh_scaling
    """
    cal_cfg = cfg.get("calibration", {})
    n_points = cal_cfg.get("dense_n_points", 100)
    range_min = cal_cfg.get("dense_range_min", 0.5)
    range_max = cal_cfg.get("dense_range_max", 12.0)

    inh_values = np.linspace(range_min, range_max, n_points)
    scan_points = []

    for i, inh_s in enumerate(inh_values):
        logger.info(f"Dense scan [{i+1}/{n_points}]: inh_scaling={inh_s:.3f}")

        # Estimate λ
        lyap_result = estimate_lyapunov(
            reservoir, cfg, inh_s, seed_lyapunov, seed_input
        )

        # Plausibility check
        plaus_result = run_plausibility_checks(
            reservoir, cfg, inh_s, seed_input
        )

        # Compute Kernel Quality (Maass 2007)
        kq = compute_kernel_quality(reservoir, cfg, inh_s, seed_input)
        
        logger.info(f"  Result: λ={lyap_result.lambda_estimate:.4f}, KQ={kq:.1f}, "
                    f"FR={plaus_result.firing_stats.mean_rate_hz:.2f} Hz, "
                    f"Plausible={plaus_result.is_plausible}")

        point = ScanPoint(
            inh_scaling=inh_s,
            lambda_est=lyap_result.lambda_estimate if not np.isnan(lyap_result.lambda_estimate) else 999.0,
            lambda_std=lyap_result.lambda_std if not np.isnan(lyap_result.lambda_std) else 999.0,
            kernel_quality=kq,
            plausibility_ok=plaus_result.is_plausible,
            plausibility_result=plaus_result,
            lyapunov_result=lyap_result,
        )
        scan_points.append(point)

    return scan_points


def find_sign_change_interval(scan_points: List[ScanPoint]
                               ) -> Optional[Tuple[ScanPoint, ScanPoint]]:
    """
    Find interval [a, b] where λ changes sign and both ends are plausible.

    Returns:
        Tuple of (point_a, point_b) or None if not found
    """
    valid = [p for p in scan_points if p.plausibility_ok and
             not np.isnan(p.lambda_est) and abs(p.lambda_est) < 100]

    for i in range(len(valid) - 1):
        if valid[i].lambda_est * valid[i+1].lambda_est < 0:
            return (valid[i], valid[i+1])

    return None


def brentq_refinement(reservoir: Reservoir, cfg: dict,
                       a_scaling: float, b_scaling: float,
                       seed_input: int, seed_lyapunov: int
                       ) -> Tuple[float, float, bool]:
    """
    Brentq-style bisection to find inh_scaling where λ≈0.

    Simplified Brent's method (bisection) targeting λ=0.

    Args:
        reservoir: Reservoir
        cfg: configuration
        a_scaling, b_scaling: bracket endpoints
        seed_input, seed_lyapunov: seeds

    Returns:
        (best_inh_scaling, best_lambda, converged)
    """
    cal_cfg = cfg.get("calibration", {})
    xtol = cal_cfg.get("brentq_xtol", 0.01)
    maxiter = cal_cfg.get("brentq_maxiter", 50)

    # Evaluate endpoints
    la = estimate_lyapunov(reservoir, cfg, a_scaling, seed_lyapunov, seed_input)
    lb = estimate_lyapunov(reservoir, cfg, b_scaling, seed_lyapunov, seed_input)

    fa = la.lambda_estimate
    fb = lb.lambda_estimate

    if np.isnan(fa) or np.isnan(fb):
        return (a_scaling + b_scaling) / 2, np.nan, False

    if fa * fb > 0:
        # Same sign — no zero crossing guaranteed
        logger.warning("Brentq: endpoints have same sign, trying bisection anyway")

    best_scaling = (a_scaling + b_scaling) / 2
    best_lambda = (fa + fb) / 2

    a, b = a_scaling, b_scaling
    f_a, f_b = fa, fb

    for iteration in range(maxiter):
        mid = (a + b) / 2.0
        if (b - a) < xtol:
            break

        lyap_mid = estimate_lyapunov(reservoir, cfg, mid, seed_lyapunov, seed_input)
        f_mid = lyap_mid.lambda_estimate

        if np.isnan(f_mid):
            # Try slightly offset
            mid += xtol * 0.1
            lyap_mid = estimate_lyapunov(reservoir, cfg, mid, seed_lyapunov, seed_input)
            f_mid = lyap_mid.lambda_estimate

        if np.isnan(f_mid):
            break

        logger.info(f"Brentq [{iteration+1}/{maxiter}]: "
                    f"inh={mid:.4f}, λ={f_mid:.4f}, interval=[{a:.4f}, {b:.4f}]")

        if abs(f_mid) < abs(best_lambda):
            best_scaling = mid
            best_lambda = f_mid

        if f_a * f_mid < 0:
            b = mid
            f_b = f_mid
        else:
            a = mid
            f_a = f_mid

    converged = abs(best_lambda) < 0.1
    return best_scaling, best_lambda, converged


def calibrate_edge_of_chaos(reservoir: Reservoir, cfg: dict,
                             seed_input: int, seed_lyapunov: int
                             ) -> CalibrationResult:
    """
    Full hybrid calibration: Dense Scan + Brentq + Recovery.

    Steps:
    1. Dense scan → map λ landscape
    2. Find sign-change interval (plausibility-constrained)
    3. Brentq refinement → λ≈0
    4. Recovery: if brentq fails, pick min |λ| from plausible points
    5. Select stable/edge/chaos points algorithmically

    Args:
        reservoir: Reservoir with frozen topology
        cfg: configuration
        seed_input: input seed (shared for comparisons)
        seed_lyapunov: lyapunov perturbation seed

    Returns:
        CalibrationResult
    """
    cal_cfg = cfg.get("calibration", {})
    stable_target = cal_cfg.get("stable_target_lambda", -1.0)
    chaos_target = cal_cfg.get("chaos_target_lambda", 1.0)

    # Step 1: Dense scan
    logger.info("=== Phase 1: Dense Scan ===")
    scan_points = dense_scan(reservoir, cfg, seed_input, seed_lyapunov)

    # Count invalid biology
    n_invalid = sum(1 for p in scan_points if not p.plausibility_ok)

    # Step 2: Find sign-change interval
    logger.info("=== Phase 2: Finding sign-change interval ===")
    interval = find_sign_change_interval(scan_points)

    inh_edge = np.nan
    lambda_edge = np.nan
    brentq_converged = False
    method_used = "recovery"

    if interval is not None:
        a_point, b_point = interval
        logger.info(f"Sign change found: [{a_point.inh_scaling:.3f}, {b_point.inh_scaling:.3f}]")
        logger.info(f"λ values: [{a_point.lambda_est:.4f}, {b_point.lambda_est:.4f}]")

        # Step 3: Brentq refinement
        logger.info("=== Phase 3: Brentq Refinement ===")
        inh_edge, lambda_edge, brentq_converged = brentq_refinement(
            reservoir, cfg, a_point.inh_scaling, b_point.inh_scaling,
            seed_input, seed_lyapunov
        )

        if brentq_converged:
            # Verify plausibility at edge point
            plaus_edge = run_plausibility_checks(
                reservoir, cfg, inh_edge, seed_input
            )
            if plaus_edge.is_plausible:
                method_used = "brentq"
            else:
                logger.warning("Edge point failed plausibility, falling back to recovery")
                brentq_converged = False

    # Step 4: Recovery (if needed)
    if not brentq_converged:
        logger.info("=== Phase 4: Recovery ===")
        plausible_points = [p for p in scan_points
                            if p.plausibility_ok and not np.isnan(p.lambda_est)
                            and abs(p.lambda_est) < 100]

        if len(plausible_points) > 0:
            # Pick min |λ| point
            best_point = min(plausible_points, key=lambda p: abs(p.lambda_est))
            inh_edge = best_point.inh_scaling
            lambda_edge = best_point.lambda_est
            method_used = "recovery"
        else:
            # No plausible points at all
            logger.error("No biologically plausible points found!")
            return CalibrationResult(
                inh_scaling_edge=np.nan,
                inh_scaling_stable=np.nan,
                inh_scaling_chaos=np.nan,
                lambda_edge=np.nan,
                lambda_stable=np.nan,
                lambda_chaos=np.nan,
                edge_plausibility_ok=False,
                method_used="failed",
                scan_points=scan_points,
                brentq_converged=False,
                n_invalid_biology=n_invalid,
                notes="No plausible points found in entire scan range",
            )

    # Step 5: Select stable and chaotic points
    plausible_points = [p for p in scan_points
                        if p.plausibility_ok and not np.isnan(p.lambda_est)
                        and abs(p.lambda_est) < 100]

    # Stable: closest to stable_target among negative-λ plausible points
    neg_points = [p for p in plausible_points if p.lambda_est < -0.1]
    if neg_points:
        stable_point = min(neg_points, key=lambda p: abs(p.lambda_est - stable_target))
        inh_stable = stable_point.inh_scaling
        lambda_stable = stable_point.lambda_est
    else:
        # Fallback: most negative λ
        inh_stable = plausible_points[0].inh_scaling if plausible_points else np.nan
        lambda_stable = plausible_points[0].lambda_est if plausible_points else np.nan

    # Chaotic: closest to chaos_target among positive-λ plausible points
    pos_points = [p for p in plausible_points if p.lambda_est > 0.1]
    if pos_points:
        chaos_point = min(pos_points, key=lambda p: abs(p.lambda_est - chaos_target))
        inh_chaos = chaos_point.inh_scaling
        lambda_chaos = chaos_point.lambda_est
    else:
        # Fallback: most positive λ
        inh_chaos = plausible_points[-1].inh_scaling if plausible_points else np.nan
        lambda_chaos = plausible_points[-1].lambda_est if plausible_points else np.nan

    # Edge plausibility check
    edge_plaus = run_plausibility_checks(reservoir, cfg, inh_edge, seed_input)

    return CalibrationResult(
        inh_scaling_edge=inh_edge,
        inh_scaling_stable=inh_stable,
        inh_scaling_chaos=inh_chaos,
        lambda_edge=lambda_edge,
        lambda_stable=lambda_stable,
        lambda_chaos=lambda_chaos,
        edge_plausibility_ok=edge_plaus.is_plausible,
        method_used=method_used,
        scan_points=scan_points,
        brentq_converged=brentq_converged,
        n_invalid_biology=n_invalid,
    )


def compute_kernel_quality(reservoir: Reservoir, cfg: dict, inh_scaling: float,
                          seed_input: int) -> float:
    """
    Compute Kernel Quality (KQ) as the numerical rank of the state matrix.
    Based on Legenstein & Maass (2007).
    
    We present M different input sequences and measure the rank of the 
    resulting state matrix.
    """
    from .states import simulate_reservoir
    from .encoding import generate_poisson_input
    
    # Use N+1 inputs to check for full rank potential
    N = reservoir.N
    M = 30  # Optimized for quick research
    
    input_cfg = cfg.get("input", {})
    dt = cfg.get("integration", {}).get("dt", 0.01)
    symbol_dur = input_cfg.get("symbol_duration", 50.0)
    base_rate = input_cfg.get("base_rate", 20.0)
    input_gain = input_cfg.get("input_gain", 50.0)
    n_input = len(reservoir.input_neurons)

    states_list = []
    rng = np.random.default_rng(seed_input)
    
    short_cfg = cfg.copy()
    short_cfg['simulation']['duration'] = 50.0 # short bursts for KQ (SPEED!)
    
    for i in range(M):
        if (i+1) % 10 == 0:
            logger.info(f"    KQ probe [{i+1}/{M}]...")
        # Generate different random input sequence (0 or 1 bits)
        u = rng.uniform(0, 1, 5) # fewer bits for speed
        spikes = generate_poisson_input(
            u, symbol_dur, dt, n_input, base_rate, input_gain, rng
        )
        
        res = simulate_reservoir(reservoir, spikes, cfg, inh_scaling)
        if not res.aborted:
            # Use the mean firing rate as the state vector for this input
            states_list.append(np.mean(res.firing_rates_filtered, axis=0))
    
    if len(states_list) < 2:
        return 0.0
        
    # State matrix S: rows are state vectors
    S = np.stack(states_list)
    
    # Kernel Quality = numerical rank
    rank = np.linalg.matrix_rank(S, tol=1e-5)
    return float(rank)
