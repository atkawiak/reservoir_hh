"""E/I population assignment."""
from __future__ import annotations

import numpy as np


def make_population(N: int, frac_I: float, seed: int) -> dict:
    """Assign neurons to E and I populations.

    Returns dict with:
        N_E, N_I: int
        E_idx: int32 array (sorted)
        I_idx: int32 array (sorted)
        is_I: bool array length N
    """
    N_I = int(round(N * frac_I))
    N_E = N - N_I

    rng = np.random.default_rng(seed)
    perm = np.arange(N, dtype=np.int32)
    rng.shuffle(perm)

    I_idx = np.sort(perm[:N_I])
    E_idx = np.sort(perm[N_I:])

    is_I = np.zeros(N, dtype=np.bool_)
    is_I[I_idx] = True

    return {
        "N_E": N_E, "N_I": N_I,
        "E_idx": E_idx, "I_idx": I_idx,
        "is_I": is_I,
    }
