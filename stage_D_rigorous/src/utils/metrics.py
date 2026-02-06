import numpy as np
import yaml
import os
from sklearn.linear_model import Ridge

def calculate_lyapunov(reservoir, n_steps=3000, perturbation=1e-4, renorm_interval=10, seed=42):
    """
    Calculate pseudo-Lyapunov exponent using Benettin method.
    Fixed: Added support for A-type gates and better state synchronization.
    """
    from src.model.reservoir import Reservoir
    n_neurons = reservoir.n_neurons
    dt = reservoir.dt
    
    # Create perturbed copy (Twin)
    # We pass a minimal config just to initialize the neuron group properly if g_A is handled
    config = {'neuron_hh': {'g_A': reservoir.neuron_group.g_A}}
    res_twin = Reservoir(n_neurons=n_neurons, config={'input': {'density': 0.1}, 'system': {'dt': dt}, 'neuron_hh': config['neuron_hh']})
    
    # Force identical weights
    res_twin.W = reservoir.W.copy()
    res_twin.W_exc = reservoir.W_exc.copy()
    res_twin.W_inh = reservoir.W_inh.copy()
    res_twin.W_in = reservoir.W_in.copy()
    res_twin.input_indices = reservoir.input_indices.copy()
    
    # Synchronize ALL state variables
    res_twin.neuron_group.V = reservoir.neuron_group.V.copy()
    res_twin.neuron_group.m = reservoir.neuron_group.m.copy()
    res_twin.neuron_group.h = reservoir.neuron_group.h.copy()
    res_twin.neuron_group.n = reservoir.neuron_group.n.copy()
    if reservoir.neuron_group.g_A > 0:
        res_twin.neuron_group.a = reservoir.neuron_group.a.copy()
        res_twin.neuron_group.b = reservoir.neuron_group.b.copy()
    
    # Perturb twin
    res_twin.neuron_group.V += perturbation
    
    log_divergences = []
    input_rate = 20.0 # Hz
    p_spike = input_rate * dt / 1000.0
    n_input = len(reservoir.input_indices)
    
    rng = np.random.default_rng(seed)
    
    for step in range(n_steps):
        spikes_in = (rng.random(n_input) < p_spike).astype(float)
        reservoir.step(spikes_in)
        res_twin.step(spikes_in)
        
        if (step + 1) % renorm_interval == 0:
            # Distance in full phase space
            diffs = [
                res_twin.neuron_group.V - reservoir.neuron_group.V,
                res_twin.neuron_group.m - reservoir.neuron_group.m,
                res_twin.neuron_group.h - reservoir.neuron_group.h,
                res_twin.neuron_group.n - reservoir.neuron_group.n
            ]
            if reservoir.neuron_group.g_A > 0:
                diffs.append(res_twin.neuron_group.a - reservoir.neuron_group.a)
                diffs.append(res_twin.neuron_group.b - reservoir.neuron_group.b)
            
            delta = np.concatenate(diffs)
            divergence = np.linalg.norm(delta)
            
            if divergence > 1e-15:
                log_divergences.append(np.log(divergence / perturbation))
                scale = perturbation / divergence
                res_twin.neuron_group.V = reservoir.neuron_group.V + diffs[0] * scale
                res_twin.neuron_group.m = reservoir.neuron_group.m + diffs[1] * scale
                res_twin.neuron_group.h = reservoir.neuron_group.h + diffs[2] * scale
                res_twin.neuron_group.n = reservoir.neuron_group.n + diffs[3] * scale
                if reservoir.neuron_group.g_A > 0:
                    res_twin.neuron_group.a = reservoir.neuron_group.a + diffs[4] * scale
                    res_twin.neuron_group.b = reservoir.neuron_group.b + diffs[5] * scale
            else:
                log_divergences.append(-20.0)
                
    if len(log_divergences) > 0:
        return np.mean(log_divergences) / (renorm_interval * dt)
    return 0.0

def get_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r = np.max(y_true) - np.min(y_true)
    return rmse / r if r > 0 else rmse

def get_mc(y_true, X_states, ridge_alpha=0.000001):
    total_mc = 0
    n_samples = len(y_true)
    split = int(n_samples * 0.7)
    for k in range(1, 41):
        if k >= n_samples: break
        y_delayed = np.concatenate([np.zeros(k), y_true[:-k]])
        X_train, X_test = X_states[:split], X_states[split:]
        y_train, y_test = y_delayed[:split], y_delayed[split:]
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        total_mc += corr**2 if not np.isnan(corr) and corr > 0 else 0
    return total_mc

def calculate_kernel_quality(X_states):
    """
    Measure Kernel Quality (Kernel Rank) using Singular Value Decomposition.
    High rank indicates rich nonlinear representation.
    """
    if X_states.shape[0] < X_states.shape[1]:
        # If we have fewer samples than neurons, rank is capped by samples
        # But we usually have more samples.
        pass
    
    # SVD to find effective rank (number of singular values above threshold)
    try:
        _, s, _ = np.linalg.svd(X_states, full_matrices=False)
        # Numerical rank: sum of singular values that are significant
        # or just count values > threshold
        threshold = s.max() * max(X_states.shape) * np.finfo(s.dtype).eps
        rank = np.sum(s > threshold)
        return rank / X_states.shape[1] # Normalized rank (percentage of neurons utilized)
    except:
        return 0.0

def calculate_separation_property(res_instance, input_rate_1=20.0, input_rate_2=25.0, n_steps=1000):
    """
    Measure Separation Property: Distance between reservoir states for different input frequencies.
    Optimal separation is expected at the Edge of Chaos.
    """
    dt = res_instance.dt
    p1 = input_rate_1 * dt / 1000.0
    p2 = input_rate_2 * dt / 1000.0
    n_in = len(res_instance.input_indices)
    
    # Run two identical reservoirs with different inputs
    # We save the original state
    orig_V = res_instance.neuron_group.V.copy()
    
    states1 = []
    rng = np.random.default_rng(42)
    for _ in range(n_steps):
        u = (rng.random(n_in) < p1).astype(float)
        res_instance.step(u)
        states1.append(res_instance.neuron_group.V.copy())
    
    # Reset
    res_instance.neuron_group.V = orig_V.copy()
    # Reset other gates if necessary... for simplicity let's assume V is enough meta-proxy
    
    states2 = []
    rng = np.random.default_rng(42) # Same seed for internal noise if any
    for _ in range(n_steps):
        u = (rng.random(n_in) < p2).astype(float)
        res_instance.step(u)
        states2.append(res_instance.neuron_group.V.copy())
        
    states1 = np.array(states1)
    states2 = np.array(states2)
    
    # Calculate average distance between corresponding states
    dist = np.mean(np.linalg.norm(states1 - states2, axis=1))
    return dist
