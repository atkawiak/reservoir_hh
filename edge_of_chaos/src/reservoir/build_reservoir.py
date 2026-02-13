"""
Reservoir builder: generate HH networks with Dale's law, frozen topology.

Creates weight matrices with the specified E/I ratio, connection probability,
and weight distributions. Supports saving and loading frozen topologies.
"""

import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class ReservoirConfig:
    """Network topology configuration."""
    N: int = 100
    connection_prob: float = 0.2
    ei_ratio: float = 0.8
    w_exc_mean: float = 0.05
    w_exc_std: float = 0.01
    w_inh_mean: float = 0.2
    w_inh_std: float = 0.04
    inh_scaling: float = 1.0
    spectral_radius: Optional[float] = None  # target spectral radius for normalization

    @classmethod
    def from_config(cls, cfg: dict) -> "ReservoirConfig":
        net_cfg = cfg.get("network", {})
        return cls(
            N=net_cfg.get("N", 100),
            connection_prob=net_cfg.get("connection_prob", 0.2),
            ei_ratio=net_cfg.get("ei_ratio", 0.8),
            w_exc_mean=net_cfg.get("w_exc_mean", 0.05),
            w_exc_std=net_cfg.get("w_exc_std", 0.01),
            w_inh_mean=net_cfg.get("w_inh_mean", 0.2),
            w_inh_std=net_cfg.get("w_inh_std", 0.04),
            inh_scaling=net_cfg.get("inh_scaling", 1.0),
            spectral_radius=net_cfg.get("spectral_radius", None),
        )


class Reservoir:
    """
    HH Reservoir with frozen topology support.

    Attributes:
        N: number of neurons
        W_base: base weight matrix (N, N) — frozen
        is_excitatory: boolean array (N,) — neuron types (Dale's law)
        connection_mask: boolean array (N, N) — connection structure
        input_neurons: indices of neurons receiving external input
        config: ReservoirConfig used to generate this reservoir
        seed: topology seed
    """

    def __init__(self, N: int, W_base: np.ndarray, is_excitatory: np.ndarray,
                 connection_mask: np.ndarray, input_neurons: np.ndarray,
                 config: ReservoirConfig, seed: int):
        self.N = N
        self.W_base = W_base
        self.is_excitatory = is_excitatory
        self.connection_mask = connection_mask
        self.input_neurons = input_neurons
        self.config = config
        self.seed = seed

    def get_scaled_weights(self, inh_scaling: float) -> np.ndarray:
        """
        Get weight matrix with inhibitory weights scaled.

        Dale's law enforced: excitatory weights positive, inhibitory negative.
        Only the magnitude of inhibitory weights is scaled.

        Args:
            inh_scaling: scaling factor for inhibitory weights

        Returns:
            W: scaled weight matrix (N, N)
        """
        W = self.W_base.copy()
        # Inhibitory neurons have negative weights in W_base
        inh_mask = ~self.is_excitatory
        # Scale columns corresponding to inhibitory neurons
        W[:, inh_mask] *= inh_scaling
        return W

    def get_spectral_radius(self) -> float:
        """Compute the spectral radius (max absolute eigenvalue) of the base weights."""
        eigenvalues = np.linalg.eigvals(self.W_base)
        return float(np.max(np.abs(eigenvalues)))

    def save(self, path: str):
        """Save frozen topology to disk."""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "W_base.npy"), self.W_base)
        np.save(os.path.join(path, "is_excitatory.npy"), self.is_excitatory)
        np.save(os.path.join(path, "connection_mask.npy"), self.connection_mask)
        np.save(os.path.join(path, "input_neurons.npy"), self.input_neurons)

        meta = {
            "N": self.N,
            "seed": self.seed,
            "config": asdict(self.config),
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Reservoir":
        """Load frozen topology from disk."""
        W_base = np.load(os.path.join(path, "W_base.npy"))
        is_excitatory = np.load(os.path.join(path, "is_excitatory.npy"))
        connection_mask = np.load(os.path.join(path, "connection_mask.npy"))
        input_neurons = np.load(os.path.join(path, "input_neurons.npy"))

        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        config = ReservoirConfig(**meta["config"])
        return cls(
            N=meta["N"],
            W_base=W_base,
            is_excitatory=is_excitatory,
            connection_mask=connection_mask,
            input_neurons=input_neurons,
            config=config,
            seed=meta["seed"],
        )


def build_reservoir(cfg: dict, seed: int) -> Reservoir:
    """
    Build an HH reservoir from configuration.

    Enforces Dale's law: each neuron is either excitatory or inhibitory.
    Excitatory neurons have positive outgoing weights.
    Inhibitory neurons have negative outgoing weights.

    Args:
        cfg: full configuration dict
        seed: topology seed (seed_topology)

    Returns:
        Reservoir object with frozen W_base
    """
    rng = np.random.default_rng(seed)
    rc = ReservoirConfig.from_config(cfg)
    N = rc.N

    # Assign neuron types (Dale's law)
    n_exc = int(round(N * rc.ei_ratio))
    types = np.zeros(N, dtype=bool)
    types[:n_exc] = True
    rng.shuffle(types)
    is_excitatory = types

    # Connection mask (sparse, no self-connections)
    connection_mask = rng.random((N, N)) < rc.connection_prob
    np.fill_diagonal(connection_mask, False)

    # Generate base weights
    W_base = np.zeros((N, N), dtype=np.float64)
    for j in range(N):
        connected = connection_mask[:, j]
        n_conn = np.sum(connected)
        if n_conn == 0:
            continue

        if is_excitatory[j]:
            # Excitatory neuron → positive weights
            weights = rng.normal(rc.w_exc_mean, rc.w_exc_std, int(n_conn))
            weights = np.abs(weights)  # enforce positive
        else:
            # Inhibitory neuron → negative weights
            weights = rng.normal(rc.w_inh_mean, rc.w_inh_std, int(n_conn))
            weights = -np.abs(weights)  # enforce negative

        W_base[connected, j] = weights

    # Optional spectral radius normalization
    if rc.spectral_radius is not None:
        eigenvalues = np.linalg.eigvals(W_base)
        current_rho = np.max(np.abs(eigenvalues))
        if current_rho > 1e-10:
            W_base = W_base * (rc.spectral_radius / current_rho)

    # Select input neurons
    input_cfg = cfg.get("input", {})
    input_fraction = input_cfg.get("input_fraction", 0.3)
    n_input = max(1, int(round(N * input_fraction)))
    input_neurons = rng.choice(N, size=n_input, replace=False)
    input_neurons.sort()

    return Reservoir(
        N=N,
        W_base=W_base,
        is_excitatory=is_excitatory,
        connection_mask=connection_mask,
        input_neurons=input_neurons,
        config=rc,
        seed=seed,
    )
