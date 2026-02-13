"""
Memory Capacity benchmark runner for HH reservoir.
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
from .protocols import generate_mc_data, evaluate_mc

logger = logging.getLogger(__name__)


def run_mc_benchmark(reservoir: Reservoir, cfg: dict, inh_scaling: float,
                     seed_input: int) -> Dict:
    """
    Run Memory Capacity benchmark.

    Protocol:
    1. Generate MC signal (uniform i.i.d.)
    2. Encode as Poisson spike trains
    3. Simulate reservoir
    4. Extract features (filtered firing rate, one per symbol)
    5. For each delay k: train ridge regression, compute MC_k
    6. Sum MC_k = MC_total

    Args:
        reservoir: Reservoir with frozen topology
        cfg: configuration
        inh_scaling: inhibitory weight scaling
        seed_input: input RNG seed

    Returns:
        dict with mc_total, mc_per_k, readout details
    """
    rng = np.random.default_rng(seed_input)
    mc_data = generate_mc_data(cfg, rng)
    signal = mc_data["signal"]
    targets = mc_data["targets"]
    K_max = mc_data["K_max"]

    # Encode signal → Poisson spikes (with per-benchmark overrides)
    input_cfg = cfg.get("input", {})
    mc_cfg = cfg.get("benchmarks", {}).get("mc", {})
    
    # Check for override inside benchmark config
    override = mc_cfg.get("input_override", {})
    symbol_duration = override.get("symbol_duration", input_cfg.get("symbol_duration", 20.0))
    input_gain = override.get("input_gain", input_cfg.get("input_gain", 40.0))
    base_rate = input_cfg.get("base_rate", 5.0)

    dt = cfg.get("integration", {}).get("dt", 0.01)
    steps_per_symbol = int(round(symbol_duration / dt))

    input_rng = np.random.default_rng(seed_input + 1000)
    from ..reservoir.encoding import generate_input_data
    input_spikes = generate_input_data(cfg, signal, len(reservoir.input_neurons), input_rng)

    # Simulate
    result = simulate_reservoir(reservoir, input_spikes, cfg, inh_scaling)

    if result.aborted:
        return {"mc_total": 0.0, "aborted": True, "reason": result.abort_reason}

    # Extract features (Align washout with symbol boundaries)
    washout_ms = cfg.get("simulation", {}).get("washout", 500.0)
    washout_symbols = int(np.ceil(washout_ms / symbol_duration))
    washout_steps = washout_symbols * steps_per_symbol

    features = extract_symbol_features(
        result.firing_rates_filtered, washout_steps, steps_per_symbol, 
        method="last", downsample=result.downsample_factor
    )

    # Align with targets (trim washout symbols)
    washout_symbols = int(washout_ms / symbol_duration)
    n_symbols = features.shape[0]

    # Hyperparameter range for ridge
    readout_cfg = cfg.get("readout", {})
    alpha_lo, alpha_hi = readout_cfg.get("alpha_range", [-6, 6])
    n_alphas = readout_cfg.get("n_alphas", 20)
    alphas = np.logspace(alpha_lo, alpha_hi, n_alphas)

    train_frac = readout_cfg.get("train_fraction", 0.6)
    val_frac = readout_cfg.get("val_fraction", 0.2)

    # Scaler: fit only on training portion
    n_train = int(n_symbols * train_frac)
    scaler = StandardScaler()
    scaler.fit(features[:n_train])
    features_scaled = scaler.transform(features)

    predictions = {}

    for k in range(1, K_max + 1):
        target = targets[k]
        # Trim to match features
        target_trimmed = target[washout_symbols:washout_symbols + n_symbols]

        if len(target_trimmed) != n_symbols:
            break

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            features_scaled, target_trimmed, train_frac, val_frac
        )

        result_readout = ridge_regression(
            X_train, y_train, X_val, y_val, X_test, y_test, alphas
        )

        # Store test predictions for MC calculation
        predictions[k] = result_readout.predictions
        # Keep track of actual test targets too
        test_y_true = y_test
        test_signal = X_test[:, 0] # Unused but for structure

    # Evaluate MC on test portion only
    mc_total = 0.0
    mc_per_k = {}
    for k, y_pred in predictions.items():
        # Get true test target for delay k
        # Re-split to be 100% sure we have exact same indices
        target_k = targets[k][washout_symbols:washout_symbols + n_symbols]
        _, _, _, _, _, y_true_test = split_data(
            features_scaled, target_k, train_frac, val_frac
        )
        
        # Pearson correlation squared
        if np.var(y_true_test) > 1e-12 and np.var(y_pred) > 1e-12:
            corr = np.corrcoef(y_true_test, y_pred)[0, 1]
            mc_k = corr**2
        else:
            mc_k = 0.0
        
        mc_per_k[k] = mc_k
        mc_total += mc_k

    return {
        "mc_total": mc_total,
        "mc_per_k": mc_per_k,
        "n_symbols": n_symbols,
        "aborted": False,
    }
