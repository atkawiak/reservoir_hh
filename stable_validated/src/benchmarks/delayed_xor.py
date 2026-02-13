"""
Delayed XOR benchmark runner for HH reservoir.
"""

import numpy as np
import logging
from typing import Dict

from ..reservoir.build_reservoir import Reservoir
from ..reservoir.encoding import generate_poisson_input
from ..reservoir.states import simulate_reservoir
from ..reservoir.readout import (
    extract_symbol_features, split_data, StandardScaler,
    logistic_regression, linear_svm
)
from .protocols import generate_delayed_xor_data

logger = logging.getLogger(__name__)


def run_delayed_xor_benchmark(reservoir: Reservoir, cfg: dict,
                                inh_scaling: float, seed_input: int,
                                readout_type: str = "logreg") -> Dict:
    """
    Run Delayed XOR benchmark.

    Protocol:
    1. Generate random bit sequence
    2. Encode as Poisson spike trains
    3. Simulate reservoir
    4. Extract features (one per symbol)
    5. Train classifier (logreg or linear SVM) for XOR(u(t), u(t-τ))
    6. Evaluate accuracy on test set

    Args:
        reservoir: Reservoir
        cfg: configuration
        inh_scaling: inhibitory scaling
        seed_input: input seed
        readout_type: "logreg" or "linear_svm"

    Returns:
        dict with accuracy, train_accuracy, readout details
    """
    rng = np.random.default_rng(seed_input)
    xor_data = generate_delayed_xor_data(cfg, rng)
    signal = xor_data["signal"]
    target = xor_data["target"]
    delay = xor_data["delay"]

    # Encode
    # Encode (with override)
    input_cfg = cfg.get("input", {})
    xor_cfg = cfg.get("benchmarks", {}).get("delayed_xor", {})
    
    override = xor_cfg.get("input_override", {})
    symbol_duration = override.get("symbol_duration", input_cfg.get("symbol_duration", 30.0))
    input_gain = override.get("input_gain", input_cfg.get("input_gain", 40.0))
    base_rate = input_cfg.get("base_rate", 5.0)

    dt = cfg.get("integration", {}).get("dt", 0.01)
    steps_per_symbol = int(round(symbol_duration / dt))

    input_rng = np.random.default_rng(seed_input + 3000)
    from ..reservoir.encoding import generate_input_data
    input_spikes = generate_input_data(cfg, signal, len(reservoir.input_neurons), input_rng)

    # Simulate
    result = simulate_reservoir(reservoir, input_spikes, cfg, inh_scaling)

    if result.aborted:
        return {"accuracy": 0.0, "aborted": True, "reason": result.abort_reason}

    # Extract features (Align washout with symbol boundaries)
    washout_ms = cfg.get("simulation", {}).get("washout", 500.0)
    washout_symbols = max(int(np.ceil(washout_ms / symbol_duration)), delay + 1)
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
    C_lo, C_hi = readout_cfg.get("C_range", [-4, 4])
    n_alphas = readout_cfg.get("n_alphas", 20)
    C_values = np.logspace(C_lo, C_hi, n_alphas)

    train_frac = readout_cfg.get("train_fraction", 0.6)
    val_frac = readout_cfg.get("val_fraction", 0.2)

    scaler = StandardScaler()
    n_train = int(n_symbols * train_frac)
    scaler.fit(features[:n_train])
    features_scaled = scaler.transform(features)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        features_scaled, target_trimmed, train_frac, val_frac
    )

    # Readout: Using Logistic Regression for binary XOR classification
    C_values = np.logspace(-4, 4, 10)
    result_readout = logistic_regression(
        X_train, y_train, X_val, y_val, X_test, y_test, C_values
    )

    return {
        "accuracy": result_readout.test_score, 
        "train_accuracy": result_readout.train_score,
        "val_accuracy": result_readout.val_score,
        "best_C": result_readout.best_hparam,
        "n_symbols": n_symbols,
        "aborted": False,
    }
