import numpy as np

class SynapseGroup:
    """
    Vectorized exponential conductance-based synapse group.
    """
    def __init__(self, size, dt=0.05, tau=5.0, E_rev=0.0):
        self.dt = dt
        self.tau = tau
        self.E_rev = E_rev
        self.g = np.zeros(size)  # Current conductances

    def update(self):
        # Exponential decay
        self.g *= np.exp(-self.dt / self.tau)

    def add_spikes(self, spike_vector, weight_matrix):
        """
        spike_vector: boolean array of which neurons fired
        weight_matrix: (size_to, size_from)
        """
        # Conductance increase: g_i = sum_j (W_ij * spike_j)
        self.g += np.dot(weight_matrix, spike_vector.astype(float))

    def get_current(self, V_mem):
        # I = g * (V_rev - V_mem)
        return self.g * (self.E_rev - V_mem)
