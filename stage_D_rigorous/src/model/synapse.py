import numpy as np

class SynapseGroup:
    """
    Vectorized exponential conductance-based synapse model.
    Handles temporal decay and spike-triggered conductance increases.
    """
    def __init__(self, size, dt=0.05, tau=5.0, E_rev=0.0):
        self.dt = dt
        self.tau_decay = tau
        self.reversal_potential = E_rev
        self.conductance = np.zeros(size)

    def update(self):
        """Applies exponential decay to the synaptic conductance."""
        decay_factor = np.exp(-self.dt / self.tau_decay)
        self.conductance *= decay_factor

    def add_spikes(self, spike_active_mask, weight_matrix):
        """
        Increases conductance based on incoming spikes.
        spike_active_mask: boolean array (size_from)
        weight_matrix: (size_to, size_from)
        """
        if np.any(spike_active_mask):
            spike_strengths = spike_active_mask.astype(float)
            conductance_increase = np.dot(weight_matrix, spike_strengths)
            self.conductance += conductance_increase

    def get_current(self, membrane_potential):
        """Calculates driving current: I = g * (E_rev - V_mem)."""
        driving_force = self.reversal_potential - membrane_potential
        return self.conductance * driving_force
