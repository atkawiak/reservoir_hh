"""
Biological plausibility checks for HH reservoir simulations.

Validates that network dynamics remain within biologically realistic bounds.
Distinguishes biological chaos from numerical artifacts.

References:
    Brunel (2000), J. Comput. Neurosci., 8(3), 183-208.
    Softky & Koch (1993), J. Neurosci., 13(1), 334-350.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hh import HHParams, HHState, rk4_step, clip_gating
from .synapses import SynapseParams, SynapticState
from .build_reservoir import Reservoir

logger = logging.getLogger(__name__)


@dataclass
class FiringStats:
    """Firing rate statistics for a simulation."""
    mean_rate_hz: float             # mean firing rate across neurons
    std_rate_hz: float              # std of firing rates
    min_rate_hz: float              # minimum firing rate
    max_rate_hz: float              # maximum firing rate
    fraction_active: float          # fraction of neurons that fired at least once
    rates_per_neuron: Optional[np.ndarray] = None  # (N,) rates in Hz


@dataclass
class SynchronyStats:
    """Synchrony measurements."""
    synchrony_index: float          # population synchrony (0=async, 1=sync)
    mean_cv_isi: float              # mean coefficient of variation of ISI
    std_cv_isi: float               # std of CV(ISI)


@dataclass
class PlausibilityResult:
    """Result of plausibility checks."""
    is_plausible: bool
    firing_stats: FiringStats
    synchrony_stats: SynchronyStats
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""


def compute_firing_stats(spike_trains: np.ndarray, dt: float,
                          washout_steps: int) -> FiringStats:
    """
    Compute firing rate statistics from spike trains.

    Args:
        spike_trains: (n_steps, N) binary
        dt: time step (ms)
        washout_steps: initial steps to discard

    Returns:
        FiringStats
    """
    data = spike_trains[washout_steps:]
    n_steps, N = data.shape
    duration_ms = n_steps * dt
    duration_s = duration_ms / 1000.0

    spikes_per_neuron = data.sum(axis=0)
    rates_hz = spikes_per_neuron / duration_s

    active = spikes_per_neuron > 0
    fraction_active = np.mean(active)

    return FiringStats(
        mean_rate_hz=float(np.mean(rates_hz)),
        std_rate_hz=float(np.std(rates_hz)),
        min_rate_hz=float(np.min(rates_hz)),
        max_rate_hz=float(np.max(rates_hz)),
        fraction_active=float(fraction_active),
        rates_per_neuron=rates_hz,
    )


def compute_synchrony(spike_trains: np.ndarray, dt: float,
                       washout_steps: int, bin_ms: float = 5.0) -> SynchronyStats:
    """
    Compute synchrony index and ISI statistics.

    Synchrony index: variance of population rate / mean of individual variances.
    CV of ISI: computed per neuron, then averaged.

    Args:
        spike_trains: (n_steps, N) binary
        dt: time step (ms)
        washout_steps: initial steps to discard
        bin_ms: bin size for population rate (ms)

    Returns:
        SynchronyStats
    """
    data = spike_trains[washout_steps:]
    n_steps, N = data.shape

    # Population synchrony index
    bin_steps = max(1, int(bin_ms / dt))
    n_bins = n_steps // bin_steps

    if n_bins < 2:
        return SynchronyStats(0.0, 1.0, 0.0)

    # Population rate in bins
    pop_rate = np.zeros(n_bins)
    for b in range(n_bins):
        start = b * bin_steps
        end = start + bin_steps
        pop_rate[b] = data[start:end].sum() / (bin_steps * dt * N / 1000.0)

    # Individual neuron rates in bins
    neuron_rates = np.zeros((n_bins, N))
    for b in range(n_bins):
        start = b * bin_steps
        end = start + bin_steps
        neuron_rates[b] = data[start:end].sum(axis=0) / (bin_steps * dt / 1000.0)

    # Synchrony index = var(pop_rate) / mean(var(individual_rates))
    var_pop = np.var(pop_rate)
    individual_vars = np.var(neuron_rates, axis=0)
    mean_individual_var = np.mean(individual_vars)

    if mean_individual_var > 1e-10:
        synchrony_index = var_pop / mean_individual_var
    else:
        synchrony_index = 0.0

    synchrony_index = min(synchrony_index, 1.0)

    # CV of ISI per neuron
    cv_isis = []
    for i in range(N):
        spike_times = np.where(data[:, i] > 0)[0] * dt
        if len(spike_times) >= 3:
            isis = np.diff(spike_times)
            if np.mean(isis) > 1e-10:
                cv = np.std(isis) / np.mean(isis)
                cv_isis.append(cv)

    if len(cv_isis) > 0:
        mean_cv = float(np.mean(cv_isis))
        std_cv = float(np.std(cv_isis))
    else:
        mean_cv = 0.0
        std_cv = 0.0

    return SynchronyStats(
        synchrony_index=float(synchrony_index),
        mean_cv_isi=mean_cv,
        std_cv_isi=std_cv,
    )


def run_plausibility_checks(reservoir: Reservoir, cfg: dict,
                             inh_scaling: float, seed_input: int
                             ) -> PlausibilityResult:
    """
    Run a short simulation and check biological plausibility.

    Runs a reduced-duration simulation (washout + 500 ms) to assess:
    - Firing rates within biological range
    - Sufficient fraction of active neurons
    - Synchrony not pathological
    - No voltage blow-ups

    Args:
        reservoir: Reservoir
        cfg: configuration
        inh_scaling: inhibitory scaling
        seed_input: input seed

    Returns:
        PlausibilityResult
    """
    from .states import simulate_reservoir

    plaus_cfg = cfg.get("plausibility", {})
    min_fr = plaus_cfg.get("min_firing_rate", 0.5)
    max_fr = plaus_cfg.get("max_firing_rate", 200.0)
    min_active = plaus_cfg.get("min_active_fraction", 0.10)
    max_sync = plaus_cfg.get("max_synchrony", 0.85)
    min_cv = plaus_cfg.get("min_cv_isi", 0.1)
    max_cv = plaus_cfg.get("max_cv_isi", 5.0)

    dt = cfg.get("integration", {}).get("dt", 0.01)
    washout_ms = cfg.get("simulation", {}).get("washout", 500.0)

    # Short simulation: washout + 500 ms for plausibility
    plaus_duration_ms = washout_ms + 500.0
    n_steps = int(plaus_duration_ms / dt)
    washout_steps = int(washout_ms / dt)

    N = reservoir.N
    input_neurons = reservoir.input_neurons
    n_input = len(input_neurons)

    # Generate background input
    input_rng = np.random.default_rng(seed_input)
    base_rate = cfg.get("input", {}).get("base_rate", 2.0)
    rate_per_ms = base_rate / 1000.0
    p_spike = min(rate_per_ms * dt, 1.0)
    input_spikes = (input_rng.random((n_steps, n_input)) < p_spike).astype(np.float64)

    # Simulate
    result = simulate_reservoir(
        reservoir, input_spikes, cfg, inh_scaling, return_all_traces=False
    )

    if result.aborted:
        return PlausibilityResult(
            is_plausible=False,
            firing_stats=FiringStats(0, 0, 0, 0, 0),
            synchrony_stats=SynchronyStats(0, 0, 0),
            violations=["Simulation aborted: " + result.abort_reason],
            aborted=True,
            abort_reason=result.abort_reason,
        )

    # Compute statistics
    firing_stats = compute_firing_stats(result.spike_trains, dt, washout_steps)
    synchrony_stats = compute_synchrony(result.spike_trains, dt, washout_steps)

    violations = []
    warnings = []

    # Check firing rates
    if firing_stats.mean_rate_hz < min_fr:
        violations.append(f"Mean firing rate too low: {firing_stats.mean_rate_hz:.2f} Hz < {min_fr}")
    if firing_stats.mean_rate_hz > max_fr:
        violations.append(f"Mean firing rate too high: {firing_stats.mean_rate_hz:.2f} Hz > {max_fr}")

    # Check active fraction
    if firing_stats.fraction_active < min_active:
        violations.append(f"Too few active neurons: {firing_stats.fraction_active:.2%} < {min_active:.0%}")

    # Check synchrony
    if synchrony_stats.synchrony_index > max_sync:
        warnings.append(f"High synchrony: {synchrony_stats.synchrony_index:.3f} > {max_sync}")

    # Check CV of ISI
    if synchrony_stats.mean_cv_isi > 0:
        if synchrony_stats.mean_cv_isi < min_cv:
            warnings.append(f"CV(ISI) too regular: {synchrony_stats.mean_cv_isi:.3f} < {min_cv}")
        if synchrony_stats.mean_cv_isi > max_cv:
            warnings.append(f"CV(ISI) too irregular: {synchrony_stats.mean_cv_isi:.3f} > {max_cv}")

    # Health incidents
    if len(result.health_incidents) > 0:
        for inc in result.health_incidents:
            violations.append(f"Health incident: {inc.incident_type} at t={inc.time_ms:.1f} ms")

    is_plausible = len(violations) == 0

    return PlausibilityResult(
        is_plausible=is_plausible,
        firing_stats=firing_stats,
        synchrony_stats=synchrony_stats,
        violations=violations,
        warnings=warnings,
    )


def dt_sensitivity_test(reservoir: Reservoir, cfg: dict,
                         inh_scaling: float, seed_input: int,
                         seed_lyapunov: int) -> Dict:
    """
    Test if chaos is a numerical artifact by comparing λ at dt and dt/2.

    Returns:
        dict with lambda_dt, lambda_dt_half, difference, is_artifact
    """
    from .lyapunov import estimate_lyapunov

    dt_main = cfg.get("integration", {}).get("dt", 0.01)
    dt_half = cfg.get("integration", {}).get("dt_verification", dt_main / 2)

    # λ at main dt
    lyap_main = estimate_lyapunov(reservoir, cfg, inh_scaling, seed_lyapunov, seed_input)

    # λ at half dt
    cfg_half = cfg.copy()
    if "integration" not in cfg_half:
        cfg_half["integration"] = {}
    cfg_half["integration"]["dt"] = dt_half

    lyap_half = estimate_lyapunov(reservoir, cfg_half, inh_scaling, seed_lyapunov, seed_input)

    lambda_main = lyap_main.lambda_estimate
    lambda_half = lyap_half.lambda_estimate

    if np.isnan(lambda_main) or np.isnan(lambda_half):
        return {
            "lambda_dt": lambda_main,
            "lambda_dt_half": lambda_half,
            "difference": np.nan,
            "is_artifact": True,
            "notes": "One or both estimates are NaN",
        }

    diff = abs(lambda_main - lambda_half)
    # If sign changes between dt values, likely artifact
    sign_change = (lambda_main > 0) != (lambda_half > 0)
    # If difference is large relative to magnitude
    relative_diff = diff / max(abs(lambda_main), abs(lambda_half), 0.01)

    is_artifact = sign_change or relative_diff > 0.5

    return {
        "lambda_dt": lambda_main,
        "lambda_dt_half": lambda_half,
        "difference": diff,
        "relative_difference": relative_diff,
        "sign_change": sign_change,
        "is_artifact": is_artifact,
    }
