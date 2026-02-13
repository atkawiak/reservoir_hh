"""
Reservoir simulation engine: runs the HH network forward in time.

Integrates the HH ODEs with RK4, handles synaptic dynamics, spike detection,
and health checks. Collects state trajectories for readout.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .hh import HHParams, HHState, rk4_step, clip_gating
from .synapses import SynapseParams, SynapticState
from .build_reservoir import Reservoir

logger = logging.getLogger(__name__)


@dataclass
class HealthIncident:
    """Record of a health check violation."""
    timestep: int
    time_ms: float
    incident_type: str
    details: str


@dataclass
class SimulationResult:
    """
    Result of a reservoir simulation.

    Contains spike trains, voltage traces, and health information.
    """
    spike_trains: np.ndarray           # (n_steps, N) binary
    firing_rates_filtered: np.ndarray  # (n_steps, N) filtered firing rates
    voltage_traces: Optional[np.ndarray] = None  # (n_steps, N) mV
    synaptic_traces: Optional[np.ndarray] = None # (n_steps, N)
    downsample_factor: int = 1
    health_incidents: List[HealthIncident] = field(default_factory=list)
    n_clipped_total: int = 0
    aborted: bool = False
    abort_reason: str = ""


def detect_spikes(V_old: np.ndarray, V_new: np.ndarray,
                  threshold: float = 0.0) -> np.ndarray:
    """
    Detect spikes using rising-edge detection.

    A spike is detected when V crosses threshold from below.

    Args:
        V_old: voltage at previous timestep (N,)
        V_new: voltage at current timestep (N,)
        threshold: spike threshold (mV)

    Returns:
        spikes: boolean array (N,)
    """
    return (V_old < threshold) & (V_new >= threshold)


def simulate_reservoir(reservoir: Reservoir, input_spikes: np.ndarray,
                       cfg: dict, inh_scaling: float,
                       input_weight: Optional[float] = None,
                       return_all_traces: bool = False) -> SimulationResult:
    """
    Simulate HH reservoir with given input.

    Input spikes are injected via exponential postsynaptic current (PSC),
    not as raw delta-function currents. This is the standard methodology
    in spiking reservoir computing (Maass et al. 2002, Destexhe et al. 1994).

    For each input neuron j:
      s_input_j(t+dt) = s_input_j(t) * exp(-dt/tau_exc) + spike_j(t)
      I_input_j(t) = w_input * s_input_j(t)

    This delivers charge Q ≈ w_input * tau_exc per spike, sufficient to
    trigger HH action potentials with biologically reasonable weights.

    Args:
        reservoir: Reservoir object with frozen topology
        input_spikes: spike input matrix (n_steps, N_input_neurons)
        cfg: full configuration dict
        inh_scaling: inhibitory weight scaling factor
        input_weight: synaptic weight for input PSC (µA/cm²)
        return_all_traces: if True, store full voltage traces

    Returns:
        SimulationResult
    """
    N = reservoir.N
    dt = cfg.get("integration", {}).get("dt", 0.01)
    max_V = cfg.get("plausibility", {}).get("max_V", 100.0)
    tau_readout = cfg.get("readout", {}).get("tau_readout", 20.0)

    input_cfg = cfg.get("input", {})
    if input_weight is None:
        input_weight = input_cfg.get("input_weight", 10.0)
    hh_params = HHParams.from_config(cfg)
    syn_params = SynapseParams.from_config(cfg)

    # Initialize states
    init_rng = np.random.default_rng(0)  # deterministic init
    hh_state = HHState(N, hh_params, rng=init_rng)
    syn_state = SynapticState(N, syn_params)

    # Get scaled weight matrix
    W = reservoir.get_scaled_weights(inh_scaling)

    n_steps = input_spikes.shape[0]
    input_neurons = reservoir.input_neurons
    n_input = len(input_neurons)

    # Input synaptic variables — exponential PSC per input neuron
    s_input = np.zeros(n_input, dtype=np.float64)
    decay_input = np.exp(-dt / syn_params.tau_exc)

    # Storage optimization: Filtered rates can be massive (GBs).
    # We downsample the storage to save memory. For readout, ~0.1-1ms resolution is enough.
    downsample = cfg.get("simulation", {}).get("downsample_rates", 10)  # default: store every 10th step
    n_steps_stored = (n_steps + downsample - 1) // downsample
    
    spike_trains = np.zeros((n_steps, N), dtype=np.bool_)
    voltage_traces = np.zeros((n_steps, N), dtype=np.float32) if return_all_traces else None
    synaptic_traces = np.zeros((n_steps, N), dtype=np.float32) if return_all_traces else None
    filtered_rates_stored = np.zeros((n_steps_stored, N), dtype=np.float32)
    
    # Running temporary rate vector
    current_filtered_rate = np.zeros(N, dtype=np.float64)

    incidents: List[HealthIncident] = []
    n_clipped_total = 0
    decay_readout = np.exp(-dt / tau_readout)

    V_old = hh_state.V.copy()

    for step in range(n_steps):
        time_ms = step * dt

        # --- SIGNAL INJECTION PROTOCOL ---
        I_input = np.zeros(N)
        protocol = input_cfg.get("protocol", "poisson")

        if protocol == "poisson":
            # 1. Poisson Spike Injection (Standard Maass/Legenstein)
            s_input *= decay_input
            if step < input_spikes.shape[0]:
                s_input += input_spikes[step]
            I_input[input_neurons] = input_weight * s_input
        
        elif protocol == "current":
            # 2. Direct Current Injection (Alternative Literature Standard)
            # In 'current' mode, input_spikes actually contains the normalized analog values
            # pre-sampled at every timestep (n_steps, N_input).
            if step < input_spikes.shape[0]:
                analog_signal = input_spikes[step] # Current value per channel
                # Map [0, 1] to [base_I, base_I + gain_I]
                # Default: I_base=0.0, I_gain=5.0
                i_base = input_cfg.get("i_base", 0.0)
                i_gain = input_cfg.get("i_gain", 5.0)
                I_input[input_neurons] = i_base + i_gain * analog_signal

        # Synaptic current from recurrent network
        I_syn = syn_state.compute_synaptic_current(
            W, hh_state.V, reservoir.is_excitatory
        )

        # Total external current
        I_ext = I_input + I_syn

        # RK4 step
        state_vec = hh_state.get_vector()
        new_state_vec = rk4_step(state_vec, N, hh_params, I_ext, dt)

        # Clip gating variables
        new_state_vec, n_clipped = clip_gating(new_state_vec, N)
        n_clipped_total += n_clipped

        # Health check: NaN/Inf
        if np.any(np.isnan(new_state_vec)) or np.any(np.isinf(new_state_vec)):
            incidents.append(HealthIncident(
                step, time_ms, "nan_inf", "NaN or Inf detected in state vector"
            ))
            return SimulationResult(
                spike_trains=spike_trains,
                voltage_traces=voltage_traces if voltage_traces is not None else np.array([]),
                synaptic_traces=synaptic_traces if synaptic_traces is not None else np.array([]),
                firing_rates_filtered=filtered_rates_stored, # downsampled
                downsample_factor=downsample,
                health_incidents=incidents,
                n_clipped_total=n_clipped_total,
                aborted=True,
                abort_reason="NaN/Inf in state vector",
            )

        hh_state.set_from_vector(new_state_vec)

        # Health check: voltage blow-up
        if np.any(np.abs(hh_state.V) > max_V):
            n_blowup = int(np.sum(np.abs(hh_state.V) > max_V))
            incidents.append(HealthIncident(
                step, time_ms, "voltage_blowup",
                f"{n_blowup} neurons exceeded |V| > {max_V} mV"
            ))
            return SimulationResult(
                spike_trains=spike_trains,
                voltage_traces=voltage_traces if voltage_traces is not None else np.array([]),
                synaptic_traces=synaptic_traces if synaptic_traces is not None else np.array([]),
                firing_rates_filtered=filtered_rates_stored, # downsampled
                downsample_factor=downsample,
                health_incidents=incidents,
                n_clipped_total=n_clipped_total,
                aborted=True,
                abort_reason=f"Voltage blow-up: {n_blowup} neurons > {max_V} mV",
            )

        # Spike detection (rising edge)
        spikes = detect_spikes(V_old, hh_state.V)
        spike_trains[step] = spikes.astype(np.float64)
        V_old = hh_state.V.copy()

        # Update recurrent synaptic state
        syn_state.update(dt, spikes, reservoir.is_excitatory)

        # Record traces
        if voltage_traces is not None:
            voltage_traces[step] = hh_state.V.astype(np.float32)
        if synaptic_traces is not None:
            synaptic_traces[step] = syn_state.s.astype(np.float32)

        # Update filtered firing rate (exponential filter) - incremental
        current_filtered_rate = current_filtered_rate * decay_readout + \
                               spikes.astype(np.float64) / tau_readout
        
        # Store periodically (downsample)
        if step % downsample == 0:
            filtered_rates_stored[step // downsample] = current_filtered_rate.astype(np.float32)

    return SimulationResult(
        spike_trains=spike_trains,
        voltage_traces=voltage_traces if voltage_traces is not None else np.array([]),
        synaptic_traces=synaptic_traces if synaptic_traces is not None else np.array([]),
        firing_rates_filtered=filtered_rates_stored,
        downsample_factor=downsample,
        health_incidents=incidents,
        n_clipped_total=n_clipped_total,
    )


def simulate_reservoir_dual(reservoir: Reservoir, input_spikes: np.ndarray,
                            cfg: dict, inh_scaling: float,
                            perturbation: np.ndarray,
                            input_weight: Optional[float] = None
                            ) -> Tuple[SimulationResult, SimulationResult]:
    """
    Simulate two copies of the reservoir in parallel (for Lyapunov).

    The second copy starts with a perturbed initial state.
    Both copies receive identical input via exponential PSC model.

    Args:
        reservoir: Reservoir
        input_spikes: input spike matrix
        cfg: configuration
        inh_scaling: inhibitory scaling
        perturbation: perturbation to add to initial V (N,)
        input_weight: input synaptic weight (µA/cm²)

    Returns:
        (result_ref, result_pert)
    """
    N = reservoir.N
    dt = cfg.get("integration", {}).get("dt", 0.01)
    max_V = cfg.get("plausibility", {}).get("max_V", 100.0)
    tau_readout = cfg.get("readout", {}).get("tau_readout", 20.0)

    input_cfg = cfg.get("input", {})
    if input_weight is None:
        input_weight = input_cfg.get("input_weight", 10.0)

    hh_params = HHParams.from_config(cfg)
    syn_params = SynapseParams.from_config(cfg)

    init_rng_ref = np.random.default_rng(0)
    init_rng_pert = np.random.default_rng(0)

    state_ref = HHState(N, hh_params, rng=init_rng_ref)
    state_pert = HHState(N, hh_params, rng=init_rng_pert)
    # Apply perturbation to V only
    state_pert.V = state_pert.V + perturbation

    syn_ref = SynapticState(N, syn_params)
    syn_pert = SynapticState(N, syn_params)

    W = reservoir.get_scaled_weights(inh_scaling)
    n_steps = input_spikes.shape[0]
    input_neurons = reservoir.input_neurons
    n_input = len(input_neurons)

    # Input PSC: shared between ref and pert (same external stimulus)
    s_input = np.zeros(n_input, dtype=np.float64)
    decay_input = np.exp(-dt / syn_params.tau_exc)

    # Storage (minimal — just spike trains and filtered rates)
    spikes_ref = np.zeros((n_steps, N), dtype=np.float64)
    spikes_pert = np.zeros((n_steps, N), dtype=np.float64)
    rates_ref = np.zeros((n_steps, N), dtype=np.float64)
    rates_pert = np.zeros((n_steps, N), dtype=np.float64)

    V_old_ref = state_ref.V.copy()
    V_old_pert = state_pert.V.copy()

    # We also track distance for Lyapunov
    distances = np.zeros(n_steps)
    decay_readout = np.exp(-dt / tau_readout)

    for step in range(n_steps):
        # --- Input PSC (shared, deterministic) ---
        s_input *= decay_input
        if step < input_spikes.shape[0]:
            s_input += input_spikes[step]

        I_input = np.zeros(N)
        I_input[input_neurons] = input_weight * s_input

        # Reference
        I_syn_ref = syn_ref.compute_synaptic_current(W, state_ref.V, reservoir.is_excitatory)
        vec_ref = state_ref.get_vector()
        new_ref = rk4_step(vec_ref, N, hh_params, I_input + I_syn_ref, dt)
        new_ref, _ = clip_gating(new_ref, N)

        if np.any(np.isnan(new_ref)) or np.any(np.isinf(new_ref)):
            break
        state_ref.set_from_vector(new_ref)

        # Perturbed
        I_syn_pert = syn_pert.compute_synaptic_current(W, state_pert.V, reservoir.is_excitatory)
        vec_pert = state_pert.get_vector()
        new_pert = rk4_step(vec_pert, N, hh_params, I_input + I_syn_pert, dt)
        new_pert, _ = clip_gating(new_pert, N)

        if np.any(np.isnan(new_pert)) or np.any(np.isinf(new_pert)):
            break
        state_pert.set_from_vector(new_pert)

        # Spike detection
        sp_ref = detect_spikes(V_old_ref, state_ref.V)
        sp_pert = detect_spikes(V_old_pert, state_pert.V)
        spikes_ref[step] = sp_ref.astype(np.float64)
        spikes_pert[step] = sp_pert.astype(np.float64)
        V_old_ref = state_ref.V.copy()
        V_old_pert = state_pert.V.copy()

        # Update recurrent synapses
        syn_ref.update(dt, sp_ref, reservoir.is_excitatory)
        syn_pert.update(dt, sp_pert, reservoir.is_excitatory)

        # Filtered rates
        if step > 0:
            rates_ref[step] = rates_ref[step-1] * decay_readout + sp_ref / tau_readout
            rates_pert[step] = rates_pert[step-1] * decay_readout + sp_pert / tau_readout
        else:
            rates_ref[step] = sp_ref / tau_readout
            rates_pert[step] = sp_pert / tau_readout

        # Distance (L2 of state vectors)
        distances[step] = np.linalg.norm(state_ref.get_vector() - state_pert.get_vector())

    result_ref = SimulationResult(
        spike_trains=spikes_ref,
        voltage_traces=np.array([]),
        synaptic_traces=np.array([]),
        firing_rates_filtered=rates_ref,
    )
    result_pert = SimulationResult(
        spike_trains=spikes_pert,
        voltage_traces=np.array([]),
        synaptic_traces=np.array([]),
        firing_rates_filtered=rates_pert,
    )

    # Store distances as attribute
    result_ref._distances = distances
    return result_ref, result_pert
