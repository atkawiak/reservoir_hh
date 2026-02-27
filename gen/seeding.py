"""Deterministic seed hierarchy from one master seed."""
from __future__ import annotations

import numpy as np

SEED_KEYS = ("population", "graph", "weights", "neurons", "synapses", "poisson", "regimes")


def split_seeds(master_seed: int) -> dict[str, int]:
    """Derive deterministic sub-seeds from master seed.

    Uses numpy default_rng → integers for reproducibility.
    Returns dict with keys: population, graph, weights, neurons, synapses,
    poisson, regimes.

    NOTE: v2.1 added 'population' key. All sub-seed values differ from v2.0.
    """
    rng = np.random.default_rng(master_seed)
    vals = rng.integers(0, 2**63, size=len(SEED_KEYS), dtype=np.int64)
    return {k: int(v) for k, v in zip(SEED_KEYS, vals)}
