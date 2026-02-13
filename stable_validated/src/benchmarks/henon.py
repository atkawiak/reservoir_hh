
"""
Henon Map Prediction benchmark runner for HH reservoir.
"""

import numpy as np
import logging
from typing import Dict

from ..reservoir.build_reservoir import Reservoir
from ..reservoir.encoding import generate_input_data
from ..reservoir.states import simulate_reservoir
from ..reservoir.readout import (
    extract_symbol_features, split_data, StandardScaler, ridge_regression
)
from .protocols import generate_henon_data, evaluate_henon

logger = logging.getLogger(__name__)

def run_henon_benchmark(reservoir: Reservoir, cfg: dict,
                        inh_scaling: float, seed_input: int) -> Dict:
    """
    Run Henon Map prediction benchmark.
    
    Predict x(n+1) from history.
    """
    rng = np.random.default_rng(seed_input)
    data = generate_henon_data(cfg, rng)
    signal = data["signal"]
    target = data["target"]

    # Encode
    input_rng = np.random.default_rng(seed_input + 5000)
    input_spikes = generate_input_data(cfg, signal, len(reservoir.input_neurons), input_rng)

    # Simulate
    result = simulate_reservoir(reservoir, input_spikes, cfg, inh_scaling)

    if result.aborted:
        return {"nrmse": 999.0, "aborted": True, "reason": result.abort_reason}

    # Extract Features
    dt = cfg.get("integration", {}).get("dt", 0.01)
    input_cfg = cfg.get("input", {})
    symbol_duration = input_cfg.get("symbol_duration", 25.0)
    steps_per_symbol = int(round(symbol_duration / dt))
    
    washout_ms = cfg.get("simulation", {}).get("washout", 500.0)
    washout_symbols = int(np.ceil(washout_ms / symbol_duration))
    washout_steps = washout_symbols * steps_per_symbol

    features = extract_symbol_features(
        result.firing_rates_filtered, washout_steps, steps_per_symbol, 
        method="last", downsample=result.downsample_factor
    )

    n_symbols = features.shape[0]
    target_trimmed = target[washout_symbols:washout_symbols + n_symbols]

    # Readout
    readout_cfg = cfg.get("readout", {})
    train_frac = readout_cfg.get("train_fraction", 0.6)
    val_frac = readout_cfg.get("val_fraction", 0.2)
    
    alphas = np.logspace(-6, 6, 20)
    
    n_train = int(n_symbols * train_frac)
    scaler = StandardScaler()
    scaler.fit(features[:n_train])
    features_scaled = scaler.transform(features)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        features_scaled, target_trimmed, train_frac, val_frac
    )

    result_readout = ridge_regression(
        X_train, y_train, X_val, y_val, X_test, y_test, alphas
    )

    eval_res = evaluate_henon(result_readout.predictions, y_test, washout=0)

    return {
        "nrmse": eval_res["nrmse"],
        "mse": eval_res["mse"],
        "n_symbols": n_symbols,
        "aborted": False
    }
