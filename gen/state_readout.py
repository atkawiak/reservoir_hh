"""Spike count state extraction for reservoir computing readout.

Converts raw spike trains from Brian2 SpikeMonitor into a (K, N) state
matrix where each row is the spike count vector for one time bin.
"""
from __future__ import annotations

import numpy as np


def extract_spike_counts(
    spike_trains: dict,
    n_neurons: int,
    dt_task_ms: float,
    total_ms: float,
    warmup_ms: float = 0.0,
    b_ms_unit=None,
) -> np.ndarray:
    """Convert spike trains to binned spike count matrix.

    Parameters
    ----------
    spike_trains : dict
        Brian2 spike_trains() dict: {neuron_id: array_of_spike_times}.
        Times are in Brian2 units (seconds) unless b_ms_unit is None.
    n_neurons : int
        Total neuron count in the reservoir.
    dt_task_ms : float
        Bin width in ms (one NARMA step = one bin).
    total_ms : float
        Total simulation duration in ms (warmup + measure).
    warmup_ms : float
        Warmup to discard from the start (ms).
    b_ms_unit : optional
        Brian2 ms unit for converting spike times. If None, spike times
        are assumed to already be in ms.

    Returns
    -------
    X : ndarray, shape (K, n_neurons)
        Spike counts per bin per neuron. K = int((total_ms - warmup_ms) / dt_task_ms).
    """
    measure_ms = total_ms - warmup_ms
    K = int(measure_ms / dt_task_ms)
    if K <= 0:
        raise ValueError(
            f"No bins: total_ms={total_ms}, warmup_ms={warmup_ms}, "
            f"dt_task_ms={dt_task_ms}"
        )

    X = np.zeros((K, n_neurons), dtype=np.int32)

    for nid in range(n_neurons):
        if nid not in spike_trains:
            continue
        t = spike_trains[nid]
        if b_ms_unit is not None:
            t = np.array(t / b_ms_unit)
        else:
            t = np.asarray(t, dtype=np.float64)

        # Keep only spikes in measurement window
        t = t[(t >= warmup_ms) & (t < total_ms)]
        if len(t) == 0:
            continue

        # Bin index for each spike
        bins = ((t - warmup_ms) / dt_task_ms).astype(np.int64)
        bins = np.clip(bins, 0, K - 1)

        # Count spikes per bin
        counts = np.bincount(bins, minlength=K)
        X[:, nid] = counts[:K]

    return X


def _zscore_columns(X: np.ndarray) -> np.ndarray:
    """Z-score normalize each column (neuron) independently."""
    X = X.astype(np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0  # avoid div-by-zero for silent neurons
    return (X - mu) / sigma


def extract_spike_counts_ei(
    spike_trains: dict,
    n_neurons: int,
    E_idx: np.ndarray,
    I_idx: np.ndarray,
    dt_task_ms: float,
    total_ms: float,
    warmup_ms: float = 0.0,
    b_ms_unit=None,
    normalize: bool = False,
) -> np.ndarray:
    """Extract spike counts with E/I split: returns [X_E | X_I] concatenation.

    Parameters
    ----------
    spike_trains : dict
        Brian2 spike_trains() dict.
    n_neurons : int
        Total neuron count.
    E_idx : ndarray
        Indices of excitatory neurons.
    I_idx : ndarray
        Indices of inhibitory neurons.
    dt_task_ms, total_ms, warmup_ms, b_ms_unit :
        Same as extract_spike_counts.
    normalize : bool
        If True, z-score normalize E and I blocks independently.

    Returns
    -------
    X_ei : ndarray, shape (K, N_E + N_I)
        Concatenated [X_E, X_I] where columns are ordered by E_idx then I_idx.
    """
    X_full = extract_spike_counts(
        spike_trains, n_neurons, dt_task_ms, total_ms,
        warmup_ms=warmup_ms, b_ms_unit=b_ms_unit,
    )
    X_E = X_full[:, E_idx]
    X_I = X_full[:, I_idx]

    if normalize:
        X_E = _zscore_columns(X_E)
        X_I = _zscore_columns(X_I)

    return np.hstack([X_E, X_I])


def delay_embed(X: np.ndarray, D: int) -> np.ndarray:
    """Augment state matrix with time-delayed copies.

    Creates feature vector [X[k], X[k-1], ..., X[k-D]] at each time step k.
    This gives the linear readout explicit access to past reservoir states,
    compensating for limited fading memory in small reservoirs.

    Parameters
    ----------
    X : ndarray, shape (K, N)
        Original spike count matrix.
    D : int
        Number of delay steps. D=0 returns X unchanged.

    Returns
    -------
    X_aug : ndarray, shape (K - D, N * (D + 1))
        Augmented state matrix. Row k contains [X[k], X[k-1], ..., X[k-D]].
    """
    if D <= 0:
        return X
    K, N = X.shape
    X_aug = np.zeros((K - D, N * (D + 1)), dtype=X.dtype)
    for d in range(D + 1):
        X_aug[:, d * N:(d + 1) * N] = X[D - d:K - d]
    return X_aug
