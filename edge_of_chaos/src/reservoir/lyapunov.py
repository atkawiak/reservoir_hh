"""
Lyapunov exponent estimation via finite-difference perturbation method.

Measures the maximal Lyapunov exponent (MLE) of an HH reservoir.
Uses perturbation + renormalization with multiple repeats for stability.

References:
    Wolf et al. (1985), Physica D, 16(3), 285-317.
    Sprott (2003), Chaos and time-series analysis. Oxford University Press.
    Benettin et al. (1980), Meccanica, 15(1), 9-20.
"""

import numpy as np
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

from .hh import HHParams, HHState, rk4_step, clip_gating
from .synapses import SynapseParams, SynapticState
from .build_reservoir import Reservoir

logger = logging.getLogger(__name__)


@dataclass
class LyapunovResult:
    """Result of Lyapunov exponent estimation."""
    lambda_estimate: float          # estimated MLE (1/ms)
    lambda_per_repeat: list         # λ from each repeat
    lambda_std: float               # std across repeats
    is_stable_estimate: bool        # True if std < threshold
    n_renorm_steps: int             # total renormalization steps used
    aborted: bool = False
    abort_reason: str = ""


def estimate_lyapunov(reservoir: Reservoir, cfg: dict, inh_scaling: float,
                      seed_lyapunov: int, seed_input: int) -> LyapunovResult:
    """
    Estimate maximal Lyapunov exponent using perturbation method.

    Protocol:
    1. Simulate reference network for washout period
    2. Apply small perturbation δ₀ to all V
    3. Simulate both networks in parallel
    4. Periodically measure divergence and renormalize
    5. Average log(divergence_ratio) over time
    6. Repeat n_repeats times with different perturbation seeds
    7. Report median of repeats

    Args:
        reservoir: Reservoir with frozen topology
        cfg: configuration dict
        inh_scaling: inhibitory weight scaling
        seed_lyapunov: base seed for perturbation RNG
        seed_input: seed for input generation

    Returns:
        LyapunovResult
    """
    lyap_cfg = cfg.get("lyapunov", {})
    delta_0 = lyap_cfg.get("delta_0", 1e-6)
    renorm_period = lyap_cfg.get("renorm_period", 2.0)
    washout_ms = lyap_cfg.get("washout", 500.0)
    measurement_ms = lyap_cfg.get("measurement_window", 500.0)
    n_repeats = lyap_cfg.get("n_repeats", 3)
    stability_threshold = lyap_cfg.get("stability_threshold", 0.3)

    dt = cfg.get("integration", {}).get("dt", 0.01)
    N = reservoir.N

    hh_params = HHParams.from_config(cfg)
    syn_params = SynapseParams.from_config(cfg)
    W = reservoir.get_scaled_weights(inh_scaling)

    washout_steps = int(washout_ms / dt)
    measurement_steps = int(measurement_ms / dt)
    renorm_steps = int(renorm_period / dt)
    total_steps = washout_steps + measurement_steps

    # Generate input (background Poisson)
    input_rng = np.random.default_rng(seed_input)
    input_cfg = cfg.get("input", {})
    base_rate = input_cfg.get("base_rate", 2.0)
    rate_per_ms = base_rate / 1000.0
    p_spike = min(rate_per_ms * dt, 1.0)

    input_neurons = reservoir.input_neurons
    n_input = len(input_neurons)
    input_weight = input_cfg.get("input_weight", 10.0)

    # Input PSC decay constant (matches simulate_reservoir)
    decay_input = np.exp(-dt / syn_params.tau_exc)

    # Pre-generate all input spikes
    input_spikes_full = (input_rng.random((total_steps, n_input)) < p_spike).astype(np.float64)

    lambdas_per_repeat = []

    for rep in range(n_repeats):
        rep_rng = np.random.default_rng(seed_lyapunov + rep * 1000)

        # Initialize reference state
        init_rng = np.random.default_rng(0)
        state_ref = HHState(N, hh_params, rng=init_rng)
        syn_ref = SynapticState(N, syn_params)

        V_old_ref = state_ref.V.copy()

        # Input PSC variable (reset per repeat)
        s_input = np.zeros(n_input, dtype=np.float64)

        # --- Washout phase (only reference) ---
        for step in range(washout_steps):
            # Exponential PSC for input
            s_input *= decay_input
            s_input += input_spikes_full[step, :]

            I_input = np.zeros(N)
            I_input[input_neurons] = input_weight * s_input

            I_syn = syn_ref.compute_synaptic_current(W, state_ref.V,
                                                      reservoir.is_excitatory)
            vec = state_ref.get_vector()
            vec = rk4_step(vec, N, hh_params, I_input + I_syn, dt)
            vec, _ = clip_gating(vec, N)

            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                lambdas_per_repeat.append(np.nan)
                break
            state_ref.set_from_vector(vec)

            spikes = (V_old_ref < 0.0) & (state_ref.V >= 0.0)
            syn_ref.update(dt, spikes, reservoir.is_excitatory)
            V_old_ref = state_ref.V.copy()
        else:
            # --- Apply perturbation ---
            state_pert = state_ref.copy()
            syn_pert = syn_ref.copy()

            perturbation = rep_rng.normal(0, delta_0, N)
            state_pert.V += perturbation

            initial_dist = np.linalg.norm(
                state_ref.get_vector() - state_pert.get_vector()
            )
            if initial_dist < 1e-15:
                initial_dist = delta_0

            V_old_ref = state_ref.V.copy()
            V_old_pert = state_pert.V.copy()

            # --- Measurement phase ---
            log_ratios = []
            steps_since_renorm = 0
            aborted = False

            for step in range(measurement_steps):
                global_step = washout_steps + step

                # Exponential PSC for input (continued from washout)
                s_input *= decay_input
                s_input += input_spikes_full[global_step, :]

                I_input = np.zeros(N)
                I_input[input_neurons] = input_weight * s_input

                # Reference
                I_syn_ref = syn_ref.compute_synaptic_current(
                    W, state_ref.V, reservoir.is_excitatory
                )
                vec_ref = rk4_step(state_ref.get_vector(), N, hh_params,
                                    I_input + I_syn_ref, dt)
                vec_ref, _ = clip_gating(vec_ref, N)

                # Perturbed
                I_syn_pert = syn_pert.compute_synaptic_current(
                    W, state_pert.V, reservoir.is_excitatory
                )
                vec_pert = rk4_step(state_pert.get_vector(), N, hh_params,
                                     I_input + I_syn_pert, dt)
                vec_pert, _ = clip_gating(vec_pert, N)

                if (np.any(np.isnan(vec_ref)) or np.any(np.isinf(vec_ref)) or
                    np.any(np.isnan(vec_pert)) or np.any(np.isinf(vec_pert))):
                    aborted = True
                    break

                state_ref.set_from_vector(vec_ref)
                state_pert.set_from_vector(vec_pert)

                sp_ref = (V_old_ref < 0.0) & (state_ref.V >= 0.0)
                sp_pert = (V_old_pert < 0.0) & (state_pert.V >= 0.0)
                syn_ref.update(dt, sp_ref, reservoir.is_excitatory)
                syn_pert.update(dt, sp_pert, reservoir.is_excitatory)
                V_old_ref = state_ref.V.copy()
                V_old_pert = state_pert.V.copy()

                steps_since_renorm += 1

                # Renormalize periodically
                if steps_since_renorm >= renorm_steps:
                    current_dist = np.linalg.norm(
                        state_ref.get_vector() - state_pert.get_vector()
                    )

                    if current_dist > 1e-15:
                        log_ratio = np.log(current_dist / initial_dist)
                        log_ratios.append(log_ratio)

                        # Renormalize: scale perturbation back to δ₀
                        diff = state_pert.get_vector() - state_ref.get_vector()
                        direction = diff / current_dist
                        state_pert.set_from_vector(
                            state_ref.get_vector() + direction * initial_dist
                        )
                        # syn_pert is NOT reset, allowing synaptic divergence tracking
                    else:
                        # Distance collapsed to zero — perturbation died
                        log_ratios.append(-20.0)  # very negative λ

                    steps_since_renorm = 0

            if aborted or len(log_ratios) == 0:
                lambdas_per_repeat.append(np.nan)
            else:
                # λ = (1/T) * Σ log(d_i/d_0) where T is in ms
                T_total = len(log_ratios) * renorm_period
                lambda_est = np.sum(log_ratios) / T_total
                lambdas_per_repeat.append(float(lambda_est))

    # Aggregate: median of repeats
    valid_lambdas = [l for l in lambdas_per_repeat if not np.isnan(l)]
    if len(valid_lambdas) == 0:
        return LyapunovResult(
            lambda_estimate=np.nan,
            lambda_per_repeat=lambdas_per_repeat,
            lambda_std=np.nan,
            is_stable_estimate=False,
            n_renorm_steps=0,
            aborted=True,
            abort_reason="All repeats aborted (NaN/Inf)",
        )

    lambda_median = float(np.median(valid_lambdas))
    lambda_std = float(np.std(valid_lambdas))
    is_stable = lambda_std < stability_threshold

    return LyapunovResult(
        lambda_estimate=lambda_median,
        lambda_per_repeat=lambdas_per_repeat,
        lambda_std=lambda_std,
        is_stable_estimate=is_stable,
        n_renorm_steps=int(measurement_ms / renorm_period),
    )
