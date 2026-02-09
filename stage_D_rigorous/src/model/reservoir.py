import numpy as np
from .neuron_hh import HHGroup
from .synapse import SynapseGroup

class Reservoir:
    """
    Bio-realistic Reservoir (Echo State Network) composed of Hodgkin-Huxley neurons.
    Dynamics are controlled by E/I balance and synaptic kinetics.
    """
    def __init__(self, n_neurons=100, ei_ratio=0.8, connectivity=0.1, dt=0.05, config=None):
        self.n_neurons = n_neurons
        self.dt = dt
        self.config = config or {}
        
        # 1. Initialize Network Topology
        self.weight_matrix = self._initialize_random_topology(n_neurons, ei_ratio, connectivity)
        
        # 2. Apply Dynamical Constraints
        target_rho = self.config.get('dynamics_control', {}).get('target_spectral_radius', 0.95)
        self.normalize_spectral_radius(target_rho)
        
        # 3. Partition weights for Synapse Groups
        self.weights_exc = np.where(self.weight_matrix > 0, self.weight_matrix, 0)
        self.weights_inh = np.where(self.weight_matrix < 0, np.abs(self.weight_matrix), 0)
        
        # 4. Initialize Neuron Population
        neuron_config = self.config.get('neuron_hh', {})
        self.neuron_group = HHGroup(n_neurons, dt=dt, config=neuron_config)
        
        # 5. Initialize Input Mapping
        self._initialize_input_layer(n_neurons)
        
        # 6. Initialize Synapse Groups (Internal & External)
        self._initialize_synapses(n_neurons, dt)
        
        # 7. Initialize Readout State
        self.tau_readout = self.config.get('dynamics_control', {}).get('tau_readout', 50.0)
        self.instantaneous_rates = np.zeros(n_neurons)
        self.prev_spikes = np.zeros(n_neurons, dtype=bool)

    def _initialize_random_topology(self, n_neurons, ei_ratio, connectivity):
        """Creates a weight matrix with random E/I connections."""
        weights = np.zeros((n_neurons, n_neurons))
        n_exc = int(n_neurons * ei_ratio)
        inh_scaling = self.config.get('synapse', {}).get('inh_scaling', 4.0)
        
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and np.random.rand() < connectivity:
                    is_excitatory = j < n_exc
                    if is_excitatory:
                        weights[i, j] = np.random.uniform(0.1, 1.0)
                    else:
                        weights[i, j] = -np.random.uniform(0.1, 1.0) * inh_scaling
        return weights

    def _initialize_input_layer(self, n_neurons):
        """Sets up the mapping from external signal units to reservoir neurons."""
        input_density = self.config.get('input', {}).get('density', 0.1)
        n_input_units = int(n_neurons * input_density)
        self.input_indices = np.random.choice(n_neurons, n_input_units, replace=False)
        self.input_weights = np.random.uniform(0.0, 1.0, size=n_input_units)

    def _initialize_synapses(self, n_neurons, dt):
        """Initializes internal excitatory/inhibitory and external input synapses."""
        syn_config = self.config.get('synapse', {})
        
        # Reusable parameters
        tau_exc = syn_config.get('tau_exc', 5.0)
        tau_inh = syn_config.get('tau_inh', 10.0)
        e_exc = syn_config.get('E_exc', 0.0)
        e_inh = syn_config.get('E_inh', -80.0)
        
        self.syn_exc = SynapseGroup(n_neurons, dt=dt, tau=tau_exc, E_rev=e_exc)
        self.syn_inh = SynapseGroup(n_neurons, dt=dt, tau=tau_inh, E_rev=e_inh)
        self.syn_in = SynapseGroup(n_neurons, dt=dt, tau=tau_exc, E_rev=e_exc)

    def normalize_spectral_radius(self, target_radius):
        """Scales the weight matrix so its spectral radius matches the target."""
        if target_radius <= 0:
            return
            
        try:
            eigenvalues = np.linalg.eigvals(self.weight_matrix)
            current_radius = np.max(np.abs(eigenvalues))
            if current_radius > 0:
                self.weight_matrix *= (target_radius / current_radius)
        except np.linalg.LinAlgError:
            print("Warning: Eigenvalue calculation failed. Skipping normalization.")

    def step(self, input_spikes, I_bias=0.0):
        """
        Advances the reservoir by one time-step (dt).
        
        1. Updates synaptic conductances.
        2. Integrates synaptic currents.
        3. Updates neuron state variables.
        4. Updates filtered readout rates.
        """
        # Synaptic decay
        self.syn_exc.update()
        self.syn_inh.update()
        self.syn_in.update()
        
        # Distribute internal and external spikes
        self.syn_exc.add_spikes(self.prev_spikes, self.weights_exc)
        self.syn_inh.add_spikes(self.prev_spikes, self.weights_inh)
        self.syn_in.conductance[self.input_indices] += input_spikes * self.input_weights
        
        # Calculate driving currents
        I_syn = (self.syn_exc.get_current(self.neuron_group.V) + 
                 self.syn_inh.get_current(self.neuron_group.V) + 
                 self.syn_in.get_current(self.neuron_group.V))
        
        # Neuron dynamics
        v_membrane, spikes = self.neuron_group.step(I_syn + I_bias)
        
        # Readout smoothing (exponential filter)
        self._update_readout_filter(spikes)
        
        self.prev_spikes = spikes
        return v_membrane, spikes, self.instantaneous_rates

    def _update_readout_filter(self, spikes):
        """Applies exponential decay readout to the firing rates."""
        decay_factor = np.exp(-self.dt / self.tau_readout)
        self.instantaneous_rates = (self.instantaneous_rates * decay_factor + 
                                    spikes.astype(float))
