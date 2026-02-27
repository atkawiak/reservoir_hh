"""Heterogeneous neuron parameter sampling."""
from __future__ import annotations

import numpy as np
from .config import GeneratorConfig, HeteroParam


def sample_neuron_params(cfg: GeneratorConfig, pop: dict, seed: int) -> dict[str, np.ndarray]:
    """Sample per-neuron parameters with heterogeneity.

    For each parameter in cfg.neuron_hetero:
        value[i] = base * clip(1 + sigma * N(0,1), clamp_lo, clamp_hi)

    Returns dict mapping param_name -> float64 array of length N.
    """
    N = cfg.N
    rng = np.random.default_rng(seed)
    params = {}

    for name, hp in cfg.neuron_hetero.items():
        if hp.sigma > 0:
            mult = 1.0 + hp.sigma * rng.standard_normal(N)
            mult = np.clip(mult, hp.clamp_lo, hp.clamp_hi)
        else:
            mult = np.ones(N, dtype=np.float64)
        params[name] = (hp.base * mult).astype(np.float64)

    return params
