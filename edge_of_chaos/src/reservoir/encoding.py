"""
Poisson input encoding: continuous signal → spike trains.

Converts continuous input signals to Poisson spike trains for HH reservoir.
Uses deterministic RNG seeds for reproducibility.
"""

import numpy as np


def poisson_encode_signal(signal: np.ndarray, dt: float, base_rate: float,
                          input_gain: float, rng: np.random.Generator) -> np.ndarray:
    """
    Encode a continuous signal as a Poisson spike train.

    Args:
        signal: continuous signal values, shape (T_symbols,)
        dt: integration timestep (ms)
        base_rate: background firing rate (Hz)
        input_gain: Hz per unit of signal
        rng: deterministic random generator

    Returns:
        spike_train: binary array, shape (n_timesteps,)
            where n_timesteps is determined by signal length and dt
    """
    # Convert rates from Hz to spikes/ms
    rates = (base_rate + input_gain * signal) / 1000.0  # Hz → per ms
    rates = np.clip(rates, 0.0, None)  # no negative rates

    # For each timestep, probability of spike
    p_spike = rates * dt
    p_spike = np.clip(p_spike, 0.0, 1.0)

    spikes = (rng.random(len(signal)) < p_spike).astype(np.float64)
    return spikes


def generate_poisson_input(signal_per_symbol: np.ndarray, symbol_duration_ms: float,
                           dt: float, N_input: int, base_rate: float,
                           input_gain: float, rng: np.random.Generator,
                           n_fibers_per_input: int = 4) -> np.ndarray:
    """
    Generate Poisson spike trains using Maass methodology.
    Each logical input signal is mapped to several independent Poisson fibers.
    
    Args:
        signal_per_symbol: (n_symbols,) signal in [0, 1]
        ...
        n_fibers_per_input: Number of independent fibers per input neuron (Maass standard = 4-20)
    """
    steps_per_symbol = int(round(symbol_duration_ms / dt))
    n_symbols = len(signal_per_symbol)
    n_total_steps = steps_per_symbol * n_symbols

    # Final matrix shape (n_steps, N_input)
    # We treat each of the N_input neurons as receiving a unique fiber bundle
    spike_matrix = np.zeros((n_total_steps, N_input), dtype=np.float64)

    for sym_idx in range(n_symbols):
        val = signal_per_symbol[sym_idx]
        # Linear frequency encoding: f = f_base + f_gain * val
        rate_hz = base_rate + input_gain * val
        rate_per_ms = rate_hz / 1000.0
        p_spike = min(rate_per_ms * dt, 1.0)

        start = sym_idx * steps_per_symbol
        end = start + steps_per_symbol

        # Each of N_input neurons samples independently from the same underlying rate.
        # This is equivalent to having N_input fibers.
        # If we had multiple input DIMENSIONS, we would split N_input into groups.
        spike_matrix[start:end, :] = (
            rng.random((steps_per_symbol, N_input)) < p_spike
        ).astype(np.float64)

    return spike_matrix


def generate_analog_input(signal_per_symbol: np.ndarray, symbol_duration_ms: float,
                          dt: float, N_input: int) -> np.ndarray:
    """
    Generate an analog (continuous) signal for direct current injection.
    Samples the symbol sequence at simulation integration steps.
    
    Returns:
        matrix: (n_total_steps, N_input) where each column is the symbol value.
    """
    steps_per_symbol = int(round(symbol_duration_ms / dt))
    n_symbols = len(signal_per_symbol)
    n_total_steps = steps_per_symbol * n_symbols

    analog_matrix = np.zeros((n_total_steps, N_input), dtype=np.float64)

    for sym_idx in range(n_symbols):
        val = signal_per_symbol[sym_idx]
        start = sym_idx * steps_per_symbol
        end = start + steps_per_symbol
        analog_matrix[start:end, :] = val

    return analog_matrix


def generate_background_spikes(n_steps: int, N: int, background_rate_hz: float,
                                dt: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate independent background Poisson spike trains for all neurons.

    Args:
        n_steps: number of time steps
        N: number of neurons
        background_rate_hz: background firing rate (Hz)
        dt: time step (ms)
        rng: deterministic RNG

    Returns:
        spike_matrix: shape (n_steps, N)
    """
    rate_per_ms = background_rate_hz / 1000.0
    p_spike = min(rate_per_ms * dt, 1.0)
    return (rng.random((n_steps, N)) < p_spike).astype(np.float64)


def generate_input_data(cfg: dict, signal_per_symbol: np.ndarray, 
                        N_input: int, rng: np.random.Generator) -> np.ndarray:
    """
    Unified entry point for input generation.
    Handles 'poisson' and 'current' protocols.
    """
    input_cfg = cfg.get("input", {})
    protocol = input_cfg.get("protocol", "poisson")
    
    symbol_duration = input_cfg.get("symbol_duration", 20.0)
    dt = cfg.get("integration", {}).get("dt", 0.01)

    if protocol == "poisson":
        base_rate = input_cfg.get("base_rate", 5.0)
        input_gain = input_cfg.get("input_gain", 40.0)
        n_fibers = input_cfg.get("n_fibers_per_input", 4)
        return generate_poisson_input(
            signal_per_symbol, symbol_duration, dt, 
            N_input, base_rate, input_gain, rng,
            n_fibers_per_input=n_fibers
        )
    elif protocol == "current":
        return generate_analog_input(
            signal_per_symbol, symbol_duration, dt, N_input
        )
    else:
        raise ValueError(f"Unknown input protocol: {protocol}")
