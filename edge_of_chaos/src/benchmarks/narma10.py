"""
NARMA-10 benchmark runner for HH reservoir.
"""

import numpy as np
import logging
from typing import Dict

from ..reservoir.build_reservoir import Reservoir
from ..reservoir.encoding import generate_poisson_input
from ..reservoir.states import simulate_reservoir
from ..reservoir.readout import (
    extract_symbol_features, split_data, StandardScaler, ridge_regression
)
from .protocols import generate_narma10_data

logger = logging.getLogger(__name__)


def run_narma10_benchmark(reservoir: Reservoir, cfg: dict, inh_scaling: float,
                           seed_input: int) -> Dict:
    """
    Run NARMA-10 benchmark.

    Protocol:
    1. Generate NARMA-10 sequence
    2. Encode input as Poisson spike trains
    3. Simulate reservoir
    4. Extract features (one per symbol)
    5. Train ridge regression to predict NARMA-10 output
    6. Evaluate NRMSE on test set

    Args:
        reservoir: Reservoir
        cfg: configuration
        inh_scaling: inhibitory scaling
        seed_input: input seed

    Returns:
        dict with nrmse, train_nrmse, readout details
    """
    rng = np.random.default_rng(seed_input)
    narma_data = generate_narma10_data(cfg, rng)
    signal = narma_data["signal"]
    target = narma_data["target"]

    # Encode
    # Encode
    input_cfg = cfg.get("input", {})
    narma_cfg = cfg.get("benchmarks", {}).get("narma10", {})
    
    # Check for override inside benchmark config
    override = narma_cfg.get("input_override", {})
    symbol_duration = override.get("symbol_duration", input_cfg.get("symbol_duration", 50.0))
    input_gain = override.get("input_gain", input_cfg.get("input_gain", 40.0))
    base_rate = input_cfg.get("base_rate", 5.0)

    dt = cfg.get("integration", {}).get("dt", 0.01)
    steps_per_symbol = int(round(symbol_duration / dt))

    input_rng = np.random.default_rng(seed_input + 2000)
    protocol = input_cfg.get("protocol", "poisson")

    if protocol == "poisson":
        input_spikes = generate_poisson_input(
            signal, symbol_duration, dt,
            len(reservoir.input_neurons),
            base_rate,
            input_gain,
            input_rng
        )
    elif protocol == "current":
        from ..reservoir.encoding import generate_analog_input
        input_spikes = generate_analog_input(
            signal, symbol_duration, dt,
            len(reservoir.input_neurons)
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    # Simulate
    result = simulate_reservoir(reservoir, input_spikes, cfg, inh_scaling)

    if result.aborted:
        return {"nrmse": 999.0, "aborted": True, "reason": result.abort_reason}

    # Extract features (Align washout with symbol boundaries)
    washout_ms = cfg.get("simulation", {}).get("washout", 500.0)
    washout_symbols = int(np.ceil(washout_ms / symbol_duration))
    washout_steps = washout_symbols * steps_per_symbol

    features = extract_symbol_features(
        result.firing_rates_filtered, washout_steps, steps_per_symbol, 
        method="last", downsample=result.downsample_factor
    )

    n_symbols = features.shape[0]
    target_trimmed = target[washout_symbols:washout_symbols + n_symbols]

    if len(target_trimmed) != n_symbols:
        n_symbols = min(n_symbols, len(target_trimmed))
        features = features[:n_symbols]
        target_trimmed = target_trimmed[:n_symbols]

    # Readout
    readout_cfg = cfg.get("readout", {})
    alpha_lo, alpha_hi = readout_cfg.get("alpha_range", [-6, 6])
    n_alphas = readout_cfg.get("n_alphas", 20)
    alphas = np.logspace(alpha_lo, alpha_hi, n_alphas)

    train_frac = readout_cfg.get("train_fraction", 0.6)
    val_frac = readout_cfg.get("val_fraction", 0.2)

    scaler = StandardScaler()
    n_train = int(n_symbols * train_frac)
    scaler.fit(features[:n_train])
    features_scaled = scaler.transform(features)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        features_scaled, target_trimmed, train_frac, val_frac
    )

    result_readout = ridge_regression(
        X_train, y_train, X_val, y_val, X_test, y_test, alphas
    )

    return {
        "nrmse": result_readout.test_score,
        "train_nrmse": result_readout.train_score,
        "val_nrmse": result_readout.val_score,
        "best_alpha": result_readout.best_hparam,
        "n_symbols": n_symbols,
        "aborted": False,
    }
