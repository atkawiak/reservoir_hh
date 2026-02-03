
import numpy as np
import scipy.sparse as sp
import os
import joblib
import hashlib
from typing import Tuple, Dict, Any, Optional

# Stable HH Functions
def alpha_m_safe(V):
    num = 0.1 * (V + 40.0); denom = 1.0 - np.exp(-(V + 40.0) / 10.0)
    mask = np.abs(V + 40.0) < 1e-6
    res = np.zeros_like(V); res[~mask] = num[~mask] / denom[~mask]; res[mask] = 1.0
    return res

def alpha_n_safe(V):
    num = 0.01 * (V + 55.0); denom = 1.0 - np.exp(-(V + 55.0) / 10.0)
    mask = np.abs(V + 55.0) < 1e-6
    res = np.zeros_like(V); res[~mask] = num[~mask] / denom[~mask]; res[mask] = 0.1
    return res

def beta_m_safe(V): return 4.0 * np.exp(-(V + 65.0) / 18.0)
def alpha_h_safe(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h_safe(V): return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def beta_n_safe(V): return 0.125 * np.exp(-(V + 65.0) / 80.0)

class HHModel:
    def __init__(self, config, trial_generators: Dict, seeds_tuple: Tuple = (0,0,0,0)):
        self.cfg = config
        self.seeds_tuple = seeds_tuple
        self.rng_rec = trial_generators['rec']
        self.rng_inmask = trial_generators['inmask']
        
        # Generation Logic
        # Generation Logic
        self.W_rec = self._generate_dale_weights()
        self.W_in = (self.rng_inmask.random(self.cfg.hh.N) < self.cfg.hh.in_density).astype(float)

    def _generate_dale_weights(self):
        N = self.cfg.hh.N
        density = self.cfg.hh.density
        n_exc = int(0.8 * N)
        ei_vec = np.ones(N); ei_vec[n_exc:] = -1
        
        mask = sp.random(N, N, density=density, random_state=self.rng_rec).toarray() > 0
        W = np.zeros((N, N))
        W[mask] = self.rng_rec.uniform(0, 1.0, size=np.sum(mask))
        W *= ei_vec
        
        # Spectral Radius Scaling (Initial)
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 1e-10: W /= radius
        return W

    def get_steady_state(self, V):
        am, bm = alpha_m_safe(V), beta_m_safe(V)
        ah, bh = alpha_h_safe(V), beta_h_safe(V)
        an, bn = alpha_n_safe(V), beta_n_safe(V)
        return am/(am+bm), ah/(ah+bh), an/(an+bn)

    def get_cache_key(self, rho: float, bias: float, task_input_id: str, len_input: int, gL: Optional[float] = None) -> str:
        """Cache key including seeds, params, and input length.
        
        Args:
            gL: Leak conductance (Phase 1). If None, uses cfg.hh.gL.
        """
        gL_eff = gL if gL is not None else self.cfg.hh.gL
        key = f"{rho}_{bias}_{gL_eff}_{self.seeds_tuple}_{self.cfg.hh.N}_{task_input_id}_{self.cfg.task.dt}_{len_input}"
        return hashlib.sha1(key.encode()).hexdigest()

    def simulate(self, rho: float, bias: float, spikes_in: np.ndarray, 
                 task_input_id: str, trim_steps: int = 500,
                 full_state: Optional[Dict[str, Any]] = None,
                 gL: Optional[float] = None) -> Dict[str, Any]:
        """Runs HH simulation with full state control for rigorous branching.
        
        Args:
            gL: Leak conductance override (for Phase 1 sweep). If None, uses cfg.hh.gL
        """
        W_eff = self.W_rec * rho
        N, dt = self.cfg.hh.N, self.cfg.task.dt
        p = self.cfg.hh
        
        # Override gL if provided (Phase 1 sweep)
        gL_eff = gL if gL is not None else p.gL
        
        # State Initialization
        if full_state is not None:
             V = full_state['V'].copy()
             m = full_state['m'].copy()
             h = full_state['h'].copy()
             n = full_state['n'].copy()
             b_gate = full_state['b_gate'].copy()
             s_trace = full_state['s_trace'].copy()
             s_in_trace = full_state['s_in_trace']
        else:
             V = np.full(N, -65.0)
             m, h, n = self.get_steady_state(V)
             b_gate = 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
             s_trace = np.zeros(N)
             s_in_trace = 0.0
        
        decay_in = np.exp(-dt / p.tau_in)
        decay_syn = np.exp(-dt / p.tau_syn)
        
        res_spikes = np.zeros((len(spikes_in), N), dtype=np.bool_)
        syn_stats = {'sum_I': 0.0, 'sq_sum': 0.0, 'count': 0}
        saturation_count = 0
        
        # Bio-plausibility tracking
        last_spike_time = np.full(N, -np.inf)  # Track ISI for CV calculation
        isi_list = [[] for _ in range(N)]  # Store ISIs per neuron
        v_trace = np.zeros(len(spikes_in)) # To be used for Lyapunov
        I_exc_total, I_inh_total = 0.0, 0.0  # E/I balance
        
        for t in range(len(spikes_in)):
            V_old = V.copy()
            s_in_trace = s_in_trace * decay_in + spikes_in[t]
            I_pulse = (self.cfg.hh.in_gain * s_in_trace) * self.W_in + bias
            
            # Gating
            a_inf = 1.0 / (1.0 + np.exp(-(V + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + np.exp((V + 80.0) / 6.0))
            b_gate += (dt / p.tauA) * (b_inf - b_gate)
            
            am, bm = alpha_m_safe(V), beta_m_safe(V); m += dt * (am * (1 - m) - bm * m)
            ah, bh = alpha_h_safe(V), beta_h_safe(V); h += dt * (ah * (1 - h) - bh * h)
            an, bn = alpha_n_safe(V), beta_n_safe(V); n += dt * (an * (1 - n) - bn * n)
            
            # Synaptic Current
            syn = W_eff @ s_trace
            I_syn = np.maximum(0, syn)*(V-p.Eexc) + np.maximum(0, -syn)*(V-p.Einh)
            
            # Integration
            dV = (-p.gNa*(m**3)*h*(V-p.ENa) - p.gK*(n**4)*(V-p.EK) - gL_eff*(V-p.EL) 
                  - p.gA*(a_inf**3)*b_gate*(V-p.EA) - I_syn + I_pulse) / p.C
            V += dt * dV
            v_trace[t] = np.mean(V) # Collection for Lyapunov
            
            # Saturation Check
            if np.any(np.abs(V) > 100): 
                saturation_count += 1
                V = np.clip(V, -100, 100)
            
            # Spikes
            spks = (V > -20.0) & (V_old <= -20.0)
            res_spikes[t] = spks
            s_trace = s_trace * decay_syn + spks.astype(float)
            
            if t >= trim_steps:
                syn_stats['sum_I'] += np.mean(I_syn)
                syn_stats['sq_sum'] += np.mean(I_syn**2)
                syn_stats['count'] += 1
                
                # Bio-metrics: E/I currents, voltage sampling
                I_exc = np.maximum(0, syn) * (V - p.Eexc)
                I_inh = np.maximum(0, -syn) * (V - p.Einh)
                I_exc_total += np.abs(np.mean(I_exc))
                I_inh_total += np.abs(np.mean(I_inh))
                
                if (t - trim_steps) % 50 == 0:  # Sample every 50 steps
                    pass # voltage_samples was removed in favor of v_trace
            
            # Track ISIs for CV calculation
            for neuron_idx in np.where(spks)[0]:
                if last_spike_time[neuron_idx] > -np.inf:
                    isi = (t - last_spike_time[neuron_idx]) * dt  # ms
                    isi_list[neuron_idx].append(isi)
                last_spike_time[neuron_idx] = t

        # Trimming
        trimmed_spikes = res_spikes[trim_steps:]
        valid_steps = syn_stats['count']
        
        mean_I = syn_stats['sum_I'] / valid_steps if valid_steps > 0 else 0
        mean_I2 = syn_stats['sq_sum'] / valid_steps if valid_steps > 0 else 0
        var_I = mean_I2 - (mean_I)**2
        
        # Capture Final State
        final_state = {
            'V': V.copy(), 'm': m.copy(), 'h': h.copy(), 'n': n.copy(),
            'b_gate': b_gate.copy(), 's_trace': s_trace.copy(), 's_in_trace': s_in_trace
        }

        # Compute biological plausibility metrics
        # 1. CV of ISI (Coefficient of Variation)
        cv_values = []
        for neuron_isis in isi_list:
            if len(neuron_isis) > 1:
                cv = np.std(neuron_isis) / (np.mean(neuron_isis) + 1e-9)
                cv_values.append(cv)
        mean_cv = np.mean(cv_values) if len(cv_values) > 0 else 0.0
        
        # compute biological plausibility metrics
        # ... CV ...
        mean_cv = np.mean(cv_values) if len(cv_values) > 0 else 0.0
        ei_balance = I_exc_total / (I_inh_total + 1e-9) if valid_steps > 0 else 0.0
        mean_voltage = np.mean(v_trace[trim_steps:]) if valid_steps > 0 else -65.0
        
        result = {
            'spikes': trimmed_spikes,
            'v_trace': v_trace[trim_steps:],  # NEW: Continuous trace for Lyapunov
            'mean_rate': (np.sum(trimmed_spikes) / (valid_steps * dt * 1e-3)) / N if valid_steps > 0 else 0.0,
            'mean_I_syn': mean_I,
            'var_I_syn': var_I,
            'saturation_flag': (saturation_count / len(spikes_in)) > 0.01,
            'final_state': final_state,
            'cv_isi': mean_cv,
            'ei_balance_ratio': ei_balance,
            'mean_voltage': mean_voltage
        }
        
        return result
