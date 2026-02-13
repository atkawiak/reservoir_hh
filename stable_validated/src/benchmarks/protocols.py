"""
Benchmark protocols for HH reservoir computing.

Provides standardized data generation and evaluation for:
- Memory Capacity (MC)
- NARMA-10
- Delayed XOR
"""

import numpy as np
from typing import Dict


def generate_mc_data(cfg: dict, rng: np.random.Generator) -> Dict:
    """Generate Memory Capacity benchmark data."""
    mc_cfg = cfg.get("benchmarks", {}).get("mc", {})
    seq_len = mc_cfg.get("sequence_length", 3000)
    input_range = mc_cfg.get("input_range", [-0.5, 0.5])
    K_max = mc_cfg.get("K_max", 100)

    lo, hi = input_range
    signal = rng.uniform(lo, hi, seq_len)

    # Targets: delayed copies
    targets = {}
    for k in range(1, K_max + 1):
        target = np.zeros(seq_len)
        target[k:] = signal[:-k]
        targets[k] = target

    return {"signal": signal, "targets": targets, "K_max": K_max}


def generate_narma10_data(cfg: dict, rng: np.random.Generator) -> Dict:
    """Generate NARMA-10 benchmark data."""
    narma_cfg = cfg.get("benchmarks", {}).get("narma10", {})
    seq_len = narma_cfg.get("sequence_length", 3000)
    input_range = narma_cfg.get("input_range", [0.0, 0.5])
    order = narma_cfg.get("order", 10)

    lo, hi = input_range
    u = rng.uniform(lo, hi, seq_len)

    y = np.zeros(seq_len)
    for t in range(order, seq_len - 1):
        sum_past = np.sum(y[t - order:t])
        y[t + 1] = (0.3 * y[t] +
                     0.05 * y[t] * sum_past +
                     1.5 * u[t - order + 1] * u[t] +
                     0.1)
        # Clip to prevent divergence
        y[t + 1] = np.clip(y[t + 1], -10.0, 10.0)

    return {"signal": u, "target": y}


def generate_delayed_xor_data(cfg: dict, rng: np.random.Generator) -> Dict:
    """Generate Delayed XOR benchmark data."""
    xor_cfg = cfg.get("benchmarks", {}).get("delayed_xor", {})
    seq_len = xor_cfg.get("sequence_length", 3000)
    delay = xor_cfg.get("delay", 3)

    bits = rng.integers(0, 2, size=seq_len).astype(np.float64)

    # XOR(u(t), u(t - delay))
    target = np.zeros(seq_len)
    for t in range(delay, seq_len):
        target[t] = float(int(bits[t]) ^ int(bits[t - delay]))

    return {"signal": bits, "target": target, "delay": delay}


def evaluate_mc(predictions: Dict[int, np.ndarray], targets: Dict[int, np.ndarray],
                signal: np.ndarray, washout: int) -> Dict:
    """
    Evaluate Memory Capacity.

    Returns:
        dict with mc_total, mc_per_k
    """
    mc_per_k = {}
    mc_total = 0.0
    threshold = 0.01

    for k in sorted(targets.keys()):
        if k not in predictions:
            break

        y_pred = predictions[k]
        y_true = targets[k][washout:]
        u_delayed = signal[washout:]

        if len(y_pred) != len(u_delayed):
            # Fallback for double-washout if somehow applied
            min_len = min(len(y_pred), len(u_delayed))
            y_pred = y_pred[:min_len]
            y_true = y_true[:min_len]
            u_delayed = u_delayed[:min_len]

        cov_uy = np.cov(u_delayed, y_pred)[0, 1]
        var_u = np.var(u_delayed)
        var_y = np.var(y_pred)

        if var_u > 1e-12 and var_y > 1e-12:
            mc_k = cov_uy ** 2 / (var_u * var_y)
        else:
            mc_k = 0.0

        mc_k = min(mc_k, 1.0)
        mc_per_k[k] = mc_k
        mc_total += mc_k

        if mc_k < threshold:
            break

    return {"mc_total": mc_total, "mc_per_k": mc_per_k}


def evaluate_narma10(y_pred: np.ndarray, y_true: np.ndarray, washout: int,
                      metric: str = "nrmse") -> Dict:
    """Evaluate NARMA-10 prediction."""
    pred = y_pred[washout:]
    true = y_true[washout:]

    mse = np.mean((pred - true) ** 2)
    var_true = np.var(true)

    if metric == "nrmse":
        score = float(np.sqrt(mse / max(var_true, 1e-12)))
    elif metric == "nmse":
        score = float(mse / max(var_true, 1e-12))
    else:
        score = float(np.sqrt(mse / max(var_true, 1e-12)))

    return {"score": score, "metric": metric, "mse": float(mse)}


def evaluate_delayed_xor(y_pred: np.ndarray, y_true: np.ndarray,
                          washout: int) -> Dict:
    """Evaluate Delayed XOR classification."""
    pred = (y_pred[washout:] >= 0.5).astype(int)
    true = y_true[washout:].astype(int)

    accuracy = float(np.mean(pred == true))
    return {"accuracy": accuracy}


def generate_henon_data(cfg: dict, rng: np.random.Generator) -> Dict:
    """
    Generate Henon map chaotic data.
    Equation: x(n+1) = 1 - a*x(n)^2 + b*x(n-1)
    Standard: a=1.4, b=0.3
    """
    henon_cfg = cfg.get("benchmarks", {}).get("henon", {})
    seq_len = henon_cfg.get("sequence_length", 3000)
    a, b = 1.4, 0.3
    
    x = np.zeros(seq_len + 1)
    y = np.zeros(seq_len + 1)
    
    # Init
    x[0], y[0] = 0.1, 0.1
    
    for i in range(seq_len):
        x[i+1] = 1 - a * x[i]**2 + y[i]
        y[i+1] = b * x[i]

    # Signal is the series of x values
    # Rescale from ~[-1.5, 1.5] to [0, 1] for Poisson
    x_min, x_max = -1.5, 1.5
    signal = (x[:seq_len] - x_min) / (x_max - x_min)
    signal = np.clip(signal, 0, 1)
    
    # Target is x[n+1] (the next values in sequence)
    target = x[1:seq_len+1]
    
    return {"signal": signal, "target": target}


def evaluate_henon(y_pred: np.ndarray, y_true: np.ndarray, washout: int) -> Dict:
    """Evaluate Henon map prediction via NRMSE."""
    pred = y_pred[washout:]
    true = y_true[washout:]
    
    mse = np.mean((pred - true) ** 2)
    var_true = np.var(true)
    nrmse = np.sqrt(mse / max(var_true, 1e-12))
    
    return {"nrmse": float(nrmse), "mse": float(mse)}
