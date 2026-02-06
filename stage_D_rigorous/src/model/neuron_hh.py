import numpy as np

class HHGroup:
    """
    Vectorized Hodgkin-Huxley neuron group with optional A-type current.
    Significant performance boost over individual objects.
    """
    def __init__(self, size, dt=0.05, config=None):
        self.size = size
        self.dt = dt
        
        # Standard HH parameters
        self.C_m = config.get('C_m', 1.0)
        self.g_Na = config.get('g_Na', 120.0)
        self.g_K = config.get('g_K', 36.0)
        self.g_L = config.get('g_L', 0.3)
        self.E_Na = config.get('E_Na', 50.0)
        self.E_K = config.get('E_K', -77.0)
        self.E_L = config.get('E_L', -54.4)
        self.V_rest = config.get('V_rest', -65.0)
        
        # A-type potassium current (optional, default OFF)
        self.g_A = config.get('g_A', 0.0)  # Set to ~47.7 to enable (Maass et al.)
        self.tau_a = config.get('tau_a', 2.0)  # ms, fast activation
        self.tau_b = config.get('tau_b', 150.0)  # ms, slow inactivation
        
        # State variables (arrays)
        self.V = np.full(size, self.V_rest)
        
        # Initial values for standard HH gates
        m_inf = self.alpha_m(self.V) / (self.alpha_m(self.V) + self.beta_m(self.V))
        h_inf = self.alpha_h(self.V) / (self.alpha_h(self.V) + self.beta_h(self.V))
        n_inf = self.alpha_n(self.V) / (self.alpha_n(self.V) + self.beta_n(self.V))
        
        self.m = m_inf
        self.h = h_inf
        self.n = n_inf
        
        # A-type gates (if g_A > 0)
        if self.g_A > 0:
            self.a = self.a_inf(self.V)
            self.b = self.b_inf(self.V)
        else:
            self.a = np.zeros(size)
            self.b = np.zeros(size)
        
        self.spike = np.zeros(size, dtype=bool)
        self.V_old = np.copy(self.V)

    # Standard HH rate functions
    def alpha_m(self, V):
        return np.where(np.abs(V + 40.0) < 1e-7, 1.0, 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)))
    
    def beta_m(self, V): 
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V): 
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V): 
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        return np.where(np.abs(V + 55.0) < 1e-7, 0.1, 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)))
    
    def beta_n(self, V): 
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    # A-type current steady-state functions (Maass et al. 2002)
    def a_inf(self, V):
        # Activation gate (fast, voltage-dependent)
        # Sigmoid centered around -50 mV
        return 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
    
    def b_inf(self, V):
        # Inactivation gate (slow, voltage-dependent)
        # Sigmoid centered around -80 mV
        return 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))

    def step(self, I_total):
        # Calculate ionic currents
        I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)
        
        # A-type current (if enabled)
        if self.g_A > 0:
            I_A = self.g_A * self.a**3 * self.b * (self.V - self.E_K)
        else:
            I_A = 0.0
        
        # Membrane voltage derivative
        dVdt = (I_total - I_Na - I_K - I_A - I_L) / self.C_m
        
        # Standard HH gate dynamics
        dmdt = self.alpha_m(self.V) * (1.0 - self.m) - self.beta_m(self.V) * self.m
        dhdt = self.alpha_h(self.V) * (1.0 - self.h) - self.beta_h(self.V) * self.h
        dndt = self.alpha_n(self.V) * (1.0 - self.n) - self.beta_n(self.V) * self.n
        
        # A-type gate dynamics (if enabled)
        if self.g_A > 0:
            dadt = (self.a_inf(self.V) - self.a) / self.tau_a
            dbdt = (self.b_inf(self.V) - self.b) / self.tau_b
        else:
            dadt = 0.0
            dbdt = 0.0
        
        # Update all state variables
        self.V_old = np.copy(self.V)
        self.V += dVdt * self.dt
        self.m += dmdt * self.dt
        self.h += dhdt * self.dt
        self.n += dndt * self.dt
        
        if self.g_A > 0:
            self.a += dadt * self.dt
            self.b += dbdt * self.dt
        
        # Rising edge spike detection (threshold -30mV)
        self.spike = (self.V >= -30.0) & (self.V_old < -30.0)
        return self.V, self.spike
