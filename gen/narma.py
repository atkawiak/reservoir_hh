"""NARMA-10 signal generator for reservoir computing benchmarks.

Generates the canonical NARMA-10 time series:
    y(k+1) = α·y(k) + β·y(k)·Σ_{i=0}^{9} y(k-i) + γ·u(k-9)·u(k) + δ

with u(k) ~ Uniform[0, 0.5].

Reference: Atiya & Parlos (2000).
"""
from __future__ import annotations

import numpy as np


# Canonical NARMA-10 parameters
NARMA_ALPHA = 0.3
NARMA_BETA = 0.05
NARMA_GAMMA = 1.5
NARMA_DELTA = 0.1
NARMA_ORDER = 10
NARMA_U_LO = 0.0
NARMA_U_HI = 0.5
NARMA_DIVERGE_THR = 1e6


def generate_narma10(
    K: int,
    seed: int,
    *,
    alpha: float = NARMA_ALPHA,
    beta: float = NARMA_BETA,
    gamma: float = NARMA_GAMMA,
    delta: float = NARMA_DELTA,
    u_lo: float = NARMA_U_LO,
    u_hi: float = NARMA_U_HI,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate NARMA-10 input/target pair of length K.

    Parameters
    ----------
    K : int
        Total number of time steps (including warmup).
    seed : int
        RNG seed for input signal u.
    alpha, beta, gamma, delta : float
        NARMA-10 equation coefficients.
    u_lo, u_hi : float
        Uniform input range.

    Returns
    -------
    u : ndarray, shape (K,)
        Input signal, u[k] ~ U[u_lo, u_hi].
    y : ndarray, shape (K,)
        Target signal, computed recursively.

    Raises
    ------
    ValueError
        If K < NARMA_ORDER or if the series diverges.
    """
    if K < NARMA_ORDER:
        raise ValueError(f"K={K} must be >= NARMA_ORDER={NARMA_ORDER}")

    rng = np.random.default_rng(seed)
    u = rng.uniform(u_lo, u_hi, size=K)

    y = np.zeros(K, dtype=np.float64)

    for k in range(NARMA_ORDER - 1, K - 1):
        y_sum = 0.0
        for i in range(NARMA_ORDER):
            y_sum += y[k - i]

        y[k + 1] = (
            alpha * y[k]
            + beta * y[k] * y_sum
            + gamma * u[k - NARMA_ORDER + 1] * u[k]
            + delta
        )

        if abs(y[k + 1]) > NARMA_DIVERGE_THR:
            raise ValueError(
                f"NARMA-10 diverged at step k={k+1}: y={y[k+1]:.2e}. "
                f"Check input range or parameters."
            )

    return u, y
