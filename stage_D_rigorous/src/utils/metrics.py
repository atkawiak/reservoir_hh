import numpy as np
from sklearn.linear_model import Ridge

# ============================================================================
# DYNAMICAL SYSTEM METRICS
# ============================================================================

def calculate_lyapunov(reservoir, n_steps=3000, perturbation=1e-4, renorm_interval=10, seed=42):
    """
    Calculate pseudo-Lyapunov exponent using the Benettin method.
    Estimates the sensitive dependence on initial conditions.
    """
    # Create and synchronize a twin reservoir
    twin_reservoir = _create_synchronized_twin(reservoir)
    
    # Apply initial perturbation to the twin
    twin_reservoir.neuron_group.V += perturbation
    
    log_divergences = _evolve_and_measure_divergence(
        reservoir, twin_reservoir, n_steps, perturbation, renorm_interval, seed
    )
    
    dt = reservoir.dt
    if not log_divergences:
        return 0.0
        
    return np.mean(log_divergences) / (renorm_interval * dt)

def _create_synchronized_twin(original_reservoir):
    """Creates a twin reservoir with identical weights and state."""
    from src.model.reservoir import Reservoir
    
    # Initialize twin with same configuration
    config = {'neuron_hh': {'g_A': original_reservoir.neuron_group.g_A}}
    twin = Reservoir(
        n_neurons=original_reservoir.n_neurons, 
        config={'input': {'density': 0.1}, 'system': {'dt': original_reservoir.dt}, 'neuron_hh': config['neuron_hh']}
    )
    
    # Snapshot weights
    twin.weight_matrix = original_reservoir.weight_matrix.copy()
    twin.weights_exc = original_reservoir.weights_exc.copy()
    twin.weights_inh = original_reservoir.weights_inh.copy()
    twin.input_weights = original_reservoir.input_weights.copy()
    twin.input_indices = original_reservoir.input_indices.copy()
    
    # Sync all dynamical gates (A-type supported)
    _sync_neuron_states(original_reservoir.neuron_group, twin.neuron_group)
    
    return twin

def _sync_neuron_states(source, target):
    """Copies all HH gating variables from source to target group."""
    target.V = source.V.copy()
    target.m = source.m.copy()
    target.h = source.h.copy()
    target.n = source.n.copy()
    if hasattr(source, 'a'):
        target.a = source.a.copy()
        target.b = source.b.copy()

def _evolve_and_measure_divergence(res, twin, steps, eps, interval, seed):
    """Main loop for Benettin's algorithm: evolve both systems and renormalize."""
    log_divs = []
    dt = res.dt
    p_spike = 20.0 * dt / 1000.0 # 20Hz base input
    n_in = len(res.input_indices)
    rng = np.random.default_rng(seed)
    
    for step in range(steps):
        spikes_in = (rng.random(n_in) < p_spike).astype(float)
        res.step(spikes_in)
        twin.step(spikes_in)
        
        if (step + 1) % interval == 0:
            divergence = _calculate_phase_space_distance(res.neuron_group, twin.neuron_group)
            
            if divergence > 1e-15:
                log_divs.append(np.log(divergence / eps))
                _renormalize_twin_state(res.neuron_group, twin.neuron_group, divergence, eps)
            else:
                log_divs.append(-20.0) # Massive convergence fallback
                
    return log_divs

def _calculate_phase_space_distance(ng1, ng2):
    """Computes Euclidean distance between two neuron groups in full phase space."""
    diffs = [ng2.V - ng1.V, ng2.m - ng1.m, ng2.h - ng1.h, ng2.n - ng1.n]
    if hasattr(ng1, 'a'):
        diffs.extend([ng2.a - ng1.a, ng2.b - ng1.b])
    
    return np.linalg.norm(np.concatenate(diffs))

def _renormalize_twin_state(source, target, current_dist, target_dist):
    """Rescales the difference between systems to keep them within linear regime."""
    scale = target_dist / current_dist
    
    target.V = source.V + (target.V - source.V) * scale
    target.m = source.m + (target.m - source.m) * scale
    target.h = source.h + (target.h - source.h) * scale
    target.n = source.n + (target.n - source.n) * scale
    if hasattr(source, 'a'):
        target.a = source.a + (target.a - source.a) * scale
        target.b = source.b + (target.b - source.b) * scale

# ============================================================================
# PERFORMANCE BENCHMARK TOOLS
# ============================================================================

def get_nrmse(y_true, y_pred):
    """Normalized Root Mean Square Error."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    target_range = np.max(y_true) - np.min(y_true)
    return rmse / target_range if target_range > 0 else rmse

def get_mc(y_values, states, ridge_alpha=1e-6):
    """
    Memory Capacity (MC) measurement.
    Iteratively predicts past inputs using reservoir states.
    """
    total_capacity = 0
    n_samples = len(y_values)
    split_idx = int(n_samples * 0.7)
    
    # Max lag capped at 40 or partial samples
    max_lag = min(40, n_samples - 1)
    
    for delay in range(1, max_lag + 1):
        y_delayed = np.concatenate([np.zeros(delay), y_values[:-delay]])
        
        X_train, X_test = states[:split_idx], states[split_idx:]
        y_train, y_test = y_delayed[:split_idx], y_delayed[split_idx:]
        
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        
        correlation = np.corrcoef(y_test, prediction)[0, 1]
        
        if not np.isnan(correlation) and correlation > 0:
            total_capacity += correlation**2
            
    return total_capacity

# ============================================================================
# STRUCTURAL PROPERTIES
# ============================================================================

def calculate_kernel_quality(states):
    """
    Measure Kernel Quality (Rank of the state matrix).
    Evaluates the dimensionality of the reservoir's neural representation.
    """
    if states.size == 0:
        return 0.0
    
    try:
        _, singular_values, _ = np.linalg.svd(states, full_matrices=False)
        # Numerical rank calculation: count singular values above threshold
        machine_eps = np.finfo(singular_values.dtype).eps
        threshold = singular_values.max() * max(states.shape) * machine_eps
        
        # Using a slightly higher threshold (10x) for robustness to numeric noise
        rank = np.sum(singular_values > (threshold * 10))
        return rank / states.shape[1]
    except (np.linalg.LinAlgError, ValueError):
        return 0.0

def calculate_separation_property(states_set_1, states_set_2):
    """Distance between reservoir state means for two distinct input signals."""
    if states_set_1.size == 0 or states_set_2.size == 0:
        return 0.0
        
    mean_state_1 = np.mean(states_set_1, axis=0)
    mean_state_2 = np.mean(states_set_2, axis=0)
    
    dist = np.linalg.norm(mean_state_1 - mean_state_2)
    return dist / np.sqrt(states_set_1.shape[1]) # Scale by dimension

def lempel_ziv_complexity(binary_sequence):
    """
    Computes Lempel-Ziv Complexity for a categorical sequence (e.g., spike train).
    Measures the amount of non-redundant information in neural activity.
    """
    sequence_data = binary_sequence.astype(int).tolist()
    n_length = len(sequence_data)
    if n_length == 0:
        return 0.0
    
    unique_words = set()
    unique_words.add(tuple([sequence_data[0]]))
    
    word_count = 1
    current_word = []
    
    for i in range(1, n_length):
        current_word.append(sequence_data[i])
        if tuple(current_word) not in unique_words:
            unique_words.add(tuple(current_word))
            current_word = []
            word_count += 1
            
    # Normalized complexity: word_count / (n / log2(n))
    return (word_count * np.log2(n_length)) / n_length if n_length > 0 else 0.0

def calculate_lz_complexity_population(spike_matrix):
    """Average Lempel-Ziv complexity across all neurons in the population."""
    if spike_matrix.size == 0:
        return 0.0
    
    neuron_complexities = [
        lempel_ziv_complexity(spike_matrix[:, i]) 
        for i in range(spike_matrix.shape[1])
    ]
    return np.mean(neuron_complexities)
