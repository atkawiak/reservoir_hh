import numpy as np

class HHGroup:
    """
    Vectorized Hodgkin-Huxley neuron group population.
    Includes standard Na/K currents and an optional A-type current (Maass et al. 2002).
    """
    def __init__(self, size, dt=0.05, config=None):
        self.size = size
        self.dt = dt
        config = config or {}
        
        # Physical constants
        self.C_m = config.get('C_m', 1.0)
        self.g_Na = config.get('g_Na', 120.0)
        self.g_K = config.get('g_K', 36.0)
        self.g_L = config.get('g_L', 0.3)
        self.E_Na = config.get('E_Na', 50.0)
        self.E_K = config.get('E_K', -77.0)
        self.E_L = config.get('E_L', -54.4)
        self.V_rest = config.get('V_rest', -65.0)
        
        # A-type potassium current (fast activation, slow inactivation)
        self.g_A = config.get('g_A', 0.0)
        self.tau_a = config.get('tau_a', 2.0)
        self.tau_b = config.get('tau_b', 150.0)
        
        self._initialize_state_variables(size)

    def _initialize_state_variables(self, size):
        """Initializes gating variables to steady-state at V_rest."""
        self.V = np.full(size, self.V_rest)
        self.V_old = np.copy(self.V)
        
        # Sodium and Potassium steady states
        self.m = self._alpha_m(self.V) / (self._alpha_m(self.V) + self._beta_m(self.V))
        self.h = self._alpha_h(self.V) / (self._alpha_h(self.V) + self._beta_h(self.V))
        self.n = self._alpha_n(self.V) / (self._alpha_n(self.V) + self._beta_n(self.V))
        
        # A-type steady states
        if self.g_A > 0:
            self.a = self._a_inf(self.V)
            self.b = self._b_inf(self.V)
        else:
            self.a, self.b = np.zeros(size), np.zeros(size)
            
        self.spike = np.zeros(size, dtype=bool)

    # --- Hodgkin-Huxley Rate Functions ---

    def _alpha_m(self, V):
        """Sodium activation alpha."""
        denom = (1.0 - np.exp(-(V + 40.0) / 10.0))
        return np.where(np.abs(V + 40.0) < 1e-7, 1.0, 0.1 * (V + 40.0) / denom)
    
    def _beta_m(self, V): 
        """Sodium activation beta."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def _alpha_h(self, V): 
        """Sodium inactivation alpha."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def _beta_h(self, V): 
        """Sodium inactivation beta."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def _alpha_n(self, V):
        """Potassium activation alpha."""
        denom = (1.0 - np.exp(-(V + 55.0) / 10.0))
        return np.where(np.abs(V + 55.0) < 1e-7, 0.1, 0.01 * (V + 55.0) / denom)
    
    def _beta_n(self, V): 
        """Potassium activation beta."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    # --- A-type Steady State Functions ---

    def _a_inf(self, V): 
        """A-type activation steady state."""
        return 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
    
    def _b_inf(self, V): 
        """A-type inactivation steady state."""
        return 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))

    # --- Dynamics ---

    def step(self, I_total):
        """Advances the neuron group by dt using Euler integration."""
        # 1. Currents
        I_ion = self._calculate_ionic_currents()
        
        # 2. Membrane Voltage Update
        dVdt = (I_total - I_ion) / self.C_m
        self.V_old = np.copy(self.V)
        self.V += dVdt * self.dt
        
        # 3. Gating Variable Updates
        self._update_gates()
        
        # 4. Spike Detection
        self.spike = (self.V >= -30.0) & (self.V_old < -30.0)
        
        return self.V, self.spike

    def _calculate_ionic_currents(self):
        """Aggregates all active and leak currents."""
        I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)
        
        I_A = 0.0
        if self.g_A > 0:
            I_A = self.g_A * self.a**3 * self.b * (self.V - self.E_K)
            
        return I_Na + I_K + I_L + I_A

    def _update_gates(self):
        """Updates Na/K/A gating particles using first-order ODEs."""
        # Standard Na/K
        dmdt = self._alpha_m(self.V) * (1.0 - self.m) - self._beta_m(self.V) * self.m
        dhdt = self._alpha_h(self.V) * (1.0 - self.h) - self._beta_h(self.V) * self.h
        dndt = self._alpha_n(self.V) * (1.0 - self.n) - self._beta_n(self.V) * self.n
        
        self.m += dmdt * self.dt
        self.h += dhdt * self.dt
        self.n += dndt * self.dt
        
        # A-type
        if self.g_A > 0:
            dadt = (self._a_inf(self.V) - self.a) / self.tau_a
            dbdt = (self._b_inf(self.V) - self.b) / self.tau_b
            self.a += dadt * self.dt
            self.b += dbdt * self.dt
