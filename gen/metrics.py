"""Benchmark metrics for reservoir computing evaluation.

Primary metric: NRMSE (Normalized Root Mean Squared Error).
"""
from __future__ import annotations

import numpy as np


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Root Mean Squared Error.

    NRMSE = sqrt(MSE / var(y_true)) = RMSE / std(y_true)

    Returns
    -------
    float
        NRMSE value. 0 = perfect, 1 = mean baseline, >1 = worse than mean.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    var_y = np.var(y_true)
    if var_y < 1e-15:
        return float('inf')

    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse / var_y))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
