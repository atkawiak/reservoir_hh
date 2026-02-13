"""
Synaptic models for HH reservoir: current-based and conductance-based.

Implements exponential decay synapses with spike-triggered updates.

References:
    Destexhe, Mainen & Sejnowski (1994), Neural Computation, 6(1), 14-18.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SynapseParams:
    """Synapse configuration."""
    model: str = "current"      # "current" or "conductance"
    tau_exc: float = 5.0        # ms — excitatory decay
    tau_inh: float = 10.0       # ms — inhibitory decay
    E_syn_exc: float = 0.0      # mV — excitatory reversal (conductance only)
    E_syn_inh: float = -80.0    # mV — inhibitory reversal (conductance only)

    @classmethod
    def from_config(cls, cfg: dict) -> "SynapseParams":
        syn_cfg = cfg.get("synapse", {})
        return cls(
            model=syn_cfg.get("model", "current"),
            tau_exc=syn_cfg.get("tau_exc", 5.0),
            tau_inh=syn_cfg.get("tau_inh", 10.0),
            E_syn_exc=syn_cfg.get("E_syn_exc", 0.0),
            E_syn_inh=syn_cfg.get("E_syn_inh", -80.0),
        )


class SynapticState:
    """
    Synaptic state for N neurons — tracks synaptic activation variables.

    s_exc[j] and s_inh[j] represent post-synaptic activation from
    excitatory and inhibitory pre-synaptic neurons respectively.
    """

    def __init__(self, N: int, params: SynapseParams):
        self.N = N
        self.params = params
        # Per-neuron synaptic variable (exponential decay)
        self.s = np.zeros(N, dtype=np.float64)

    def update(self, dt: float, spikes: np.ndarray, is_excitatory: np.ndarray):
        """
        Update synaptic variables with exponential decay + spike input.

        Args:
            dt: time step (ms)
            spikes: boolean array (N,) — which neurons spiked this step
            is_excitatory: boolean array (N,) — neuron type
        """
        # Separate decay by neuron type
        tau = np.where(is_excitatory, self.params.tau_exc, self.params.tau_inh)
        decay = np.exp(-dt / tau)
        self.s = self.s * decay + spikes.astype(np.float64)

    def compute_synaptic_current(self, W: np.ndarray, V: np.ndarray,
                                  is_excitatory: np.ndarray) -> np.ndarray:
        """
        Compute total synaptic current into each neuron.

        Args:
            W: weight matrix (N, N) — W[i, j] = weight from j to i
            V: membrane potentials (N,) — needed for conductance model
            is_excitatory: boolean array (N,) — neuron types

        Returns:
            I_syn: synaptic current per neuron (N,)
        """
        if self.params.model == "current":
            # I_syn_i = sum_j W_ij * s_j
            I_syn = W @ self.s
        elif self.params.model == "conductance":
            # Separate E and I contributions
            # g_syn_i = sum_j |W_ij| * s_j  (separately for E and I pre-neurons)
            exc_mask = is_excitatory.astype(np.float64)
            inh_mask = (1.0 - exc_mask)

            # Excitatory conductance
            W_exc = np.abs(W) * exc_mask[np.newaxis, :]
            g_exc = W_exc @ self.s
            I_exc = g_exc * (self.params.E_syn_exc - V)

            # Inhibitory conductance
            W_inh = np.abs(W) * inh_mask[np.newaxis, :]
            g_inh = W_inh @ self.s
            I_inh = g_inh * (self.params.E_syn_inh - V)

            I_syn = I_exc + I_inh
        else:
            raise ValueError(f"Unknown synapse model: {self.params.model}")

        return I_syn

    def copy(self) -> "SynapticState":
        new = SynapticState.__new__(SynapticState)
        new.N = self.N
        new.params = self.params
        new.s = self.s.copy()
        return new
