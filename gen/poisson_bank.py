"""Frozen Poisson spike train bank."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def generate_poisson_trains(
    n_channels: int,
    T_s: float,
    rate_hz: Union[float, np.ndarray],
    seed: int,
) -> list[np.ndarray]:
    """Generate deterministic Poisson spike trains.

    Args:
        n_channels: number of independent channels
        T_s: duration in seconds
        rate_hz: scalar or array of length n_channels
        seed: RNG seed

    Returns:
        List of n_channels arrays, each containing sorted spike times in seconds.
    """
    rng = np.random.default_rng(seed)

    if np.isscalar(rate_hz):
        rates = np.full(n_channels, rate_hz, dtype=np.float64)
    else:
        rates = np.asarray(rate_hz, dtype=np.float64)
        assert len(rates) == n_channels

    trains = []
    for ch in range(n_channels):
        lam = rates[ch] * T_s
        n_spikes = rng.poisson(lam)
        if n_spikes > 0:
            times = np.sort(rng.uniform(0.0, T_s, n_spikes))
        else:
            times = np.zeros(0, dtype=np.float64)
        trains.append(times)

    return trains


def save_poisson(trains: list[np.ndarray], path: Path,
                 T_s: float, rate_hz: float, seed: int) -> None:
    """Save Poisson trains to compressed NPZ."""
    data = {
        "n_channels": len(trains),
        "T_s": T_s,
        "rate_hz": rate_hz,
        "seed": seed,
    }
    for ch, t in enumerate(trains):
        data[f"ch{ch}"] = t
    np.savez_compressed(path, **data)


def load_poisson(path: Path) -> tuple[list[np.ndarray], dict]:
    """Load Poisson trains from NPZ.

    Returns (trains_list, metadata_dict).
    """
    raw = np.load(path, allow_pickle=False)
    n_channels = int(raw["n_channels"])
    meta = {
        "n_channels": n_channels,
        "T_s": float(raw["T_s"]),
        "rate_hz": float(raw["rate_hz"]),
        "seed": int(raw["seed"]),
    }
    trains = [raw[f"ch{ch}"] for ch in range(n_channels)]
    return trains, meta
