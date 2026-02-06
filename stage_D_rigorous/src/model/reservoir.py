import numpy as np
from .neuron_hh import HHGroup
from .synapse import SynapseGroup

class Reservoir:
    def __init__(self, n_neurons=100, ei_ratio=0.8, connectivity=0.1, dt=0.05, config=None):
        self.n_neurons = n_neurons
        self.dt = dt
        
        # Matrix initialization
        self.W = np.zeros((n_neurons, n_neurons))
        n_exc = int(n_neurons * ei_ratio)
        
        scaling = config.get('synapse', {}).get('inh_scaling', 4.0)
        
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and np.random.rand() < connectivity:
                    if j < n_exc:
                        self.W[i, j] = np.random.uniform(0.1, 1.0)
                    else:
                        self.W[i, j] = -np.random.uniform(0.1, 1.0) * scaling

        self.normalize_spectral_radius(config.get('dynamics_control', {}).get('target_spectral_radius', 0.95))
        
        # Split W into excitatory and inhibitory parts for SynapseGroup
        self.W_exc = np.where(self.W > 0, self.W, 0)
        self.W_inh = np.where(self.W < 0, np.abs(self.W), 0)
        
        # Neurons
        self.neuron_group = HHGroup(n_neurons, dt=dt, config=config.get('neuron_hh', {}))
        
        # Input to Reservoir Mapping (Fixed Random Weights)
        # Assuming input is already scaled by gain in the benchmark script
        n_input_units = int(n_neurons * config['input']['density'])
        self.input_indices = np.random.choice(n_neurons, n_input_units, replace=False)
        
        # W_in: weight from input unit to selected reservoir neuron
        self.W_in = np.random.uniform(0.0, 1.0, size=n_input_units)

        # Synapses (Internal)
        syn_config = config.get('synapse', {})
        self.syn_exc = SynapseGroup(n_neurons, dt=dt, tau=syn_config.get('tau_exc', 5.0), E_rev=syn_config.get('E_exc', 0.0))
        self.syn_inh = SynapseGroup(n_neurons, dt=dt, tau=syn_config.get('tau_inh', 10.0), E_rev=syn_config.get('E_inh', -80.0))
        
        # Synapses (Input) - Excitatory entry for external signals
        self.syn_in = SynapseGroup(n_neurons, dt=dt, tau=syn_config.get('tau_exc', 5.0), E_rev=syn_config.get('E_exc', 0.0))

        # Readout filter (filtering spikes for regression)
        self.tau_readout = config.get('dynamics_control', {}).get('tau_readout', 50.0) # ms
        self.r = np.zeros(n_neurons) # filtered firing rates
        self.prev_spikes = np.zeros(n_neurons, dtype=bool)

    def normalize_spectral_radius(self, target_rho):
        if target_rho <= 0: return
        try:
            eigenvalues = np.linalg.eigvals(self.W)
            current_rho = np.max(np.abs(eigenvalues))
            if current_rho > 0:
                self.W *= (target_rho / current_rho)
        except np.linalg.LinAlgError:
            print("Warning: Eigenvalue calculation failed. Skipping normalization.")

    def step(self, input_spikes, I_bias=0.0):
        """
        input_spikes: binary array of spikes from input units (size: n_input_units)
        I_bias: small background current (optional)
        """
        # 1. Update synapse decay (Internal & Input)
        self.syn_exc.update()
        self.syn_inh.update()
        self.syn_in.update()
        
        # 2. Propagate internal spikes
        self.syn_exc.add_spikes(self.prev_spikes, self.W_exc)
        self.syn_inh.add_spikes(self.prev_spikes, self.W_inh)
        
        # 3. Propagate external input spikes through W_in
        # Create a full-size spike vector for the input synapses
        input_term = np.zeros(self.n_neurons)
        # Each input spike j is multiplied by its weight W_in[j] and hits neuron input_indices[j]
        input_term[self.input_indices] = input_spikes * self.W_in
        self.syn_in.add_spikes(np.ones(self.n_neurons, dtype=bool), np.diag(input_term))
        
        # 4. Calculate total synaptic currents
        I_syn = self.syn_exc.get_current(self.neuron_group.V) + \
                self.syn_inh.get_current(self.neuron_group.V) + \
                self.syn_in.get_current(self.neuron_group.V)
        
        # 5. Step neurons
        V, spikes = self.neuron_group.step(I_syn + I_bias)


        
        # 5. Update readout filter (exponential decay + spikes)
        # dr/dt = -r/tau + sum(spikes)
        decay = np.exp(-self.dt / self.tau_readout)
        self.r = self.r * decay + spikes.astype(float)
        
        self.prev_spikes = spikes
        return V, spikes, self.r

