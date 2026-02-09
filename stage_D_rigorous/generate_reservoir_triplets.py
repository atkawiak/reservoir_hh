import numpy as np
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import brentq
from src.model.reservoir import Reservoir
from src.utils.metrics import (calculate_lyapunov, calculate_kernel_quality, 
                               calculate_separation_property, calculate_lz_complexity_population)
from benchmark_mc import run_mc_benchmark
from benchmark_narma10 import run_narma_benchmark
from repro_working_reservoir import run_xor_benchmark
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# CORE UTILITIES
# ============================================================================

def load_task_config():
    with open('task_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def calculate_median_lyapunov(inh_scaling_val, config, n_neurons, seed, repetitions=1):
    """
    Measure Lyapunov exponent by taking the median of multiple runs to reduce noise.
    Uses E/I balance (inh_scaling) as the primary control parameter.
    """
    np.random.seed(seed)
    context_config = config.copy()
    context_config['synapse']['inh_scaling'] = float(inh_scaling_val)
    
    lyapunov_values = []
    for i in range(repetitions):
        reservoir = Reservoir(n_neurons=n_neurons, config=context_config)
        # Normalize to baseline (4.0) to ensure E/I balance is the only variable
        reservoir.normalize_spectral_radius(4.0)
        
        # 1200 steps used for stable convergence
        exponent = calculate_lyapunov(reservoir, n_steps=1200, seed=seed + i*100)
        lyapunov_values.append(exponent)
        
    return np.median(lyapunov_values)

# ============================================================================
# CALIBRATION LOGIC
# ============================================================================

def perform_surgical_calibration(n_neurons, seed, config):
    """
    Locates the dynamical regimes (Stable, Edge, Chaotic) using a dense scan
    followed by precise root-finding for the Edge of Chaos (Lambda ≈ 0).
    """
    def lyapunov_function_wrapper(inh):
        return calculate_median_lyapunov(inh, config, n_neurons, seed, repetitions=1)

    # 1. DENSE SCAN to map the Lyapunov landscape
    scan_points = np.linspace(0.5, 12.0, 100)
    scan_lambdas = np.array([lyapunov_function_wrapper(inh) for inh in scan_points])
    
    peak_idx = np.argmax(scan_lambdas)
    chaotic_inh = scan_points[peak_idx]
    
    # 2. FIND EDGE OF CHAOS (zero crossing on the right-side slope)
    edge_inh = _find_zero_crossing_right_of_peak(lyapunov_function_wrapper, scan_points, scan_lambdas, peak_idx)
    
    # 3. DEFINE REGIMES
    stable_inh = min(edge_inh + 4.0, 12.0) # Deep stability is further right
    
    print(f"  [CALIB] CHAOTIC={chaotic_inh:.2f}, EDGE={edge_inh:.2f}, STABLE={stable_inh:.2f}")
    
    return [
        ('STABLE', stable_inh),
        ('EDGE', edge_inh),
        ('CHAOTIC', chaotic_inh)
    ]

def _find_zero_crossing_right_of_peak(target_func, x_points, y_points, peak_idx):
    """Internal helper to locate Lambda=0 with surgical precision."""
    interval = None
    for i in range(peak_idx, len(y_points) - 1):
        if y_points[i] * y_points[i+1] <= 0:
            interval = (x_points[i], x_points[i+1])
            break
            
    if not interval:
        # Fallback to closest point if no crossing found
        best_idx = peak_idx + np.argmin(np.abs(y_points[peak_idx:]))
        return x_points[best_idx]
        
    try:
        return brentq(target_func, interval[0], interval[1], xtol=0.01)
    except ValueError:
        # Precision recovery if sign mismatch occurs due to noise
        val_start, val_end = target_func(interval[0]), target_func(interval[1])
        return interval[0] if abs(val_start) < abs(val_end) else interval[1]

# ============================================================================
# BENCHMARK COORDINATION
# ============================================================================

def compute_regime_metrics(regime_name, inh_val, n_neurons, seed, config):
    """Runs all benchmarks and structural metrics for a specific dynamical state."""
    print(f"  [BENCH] Evaluating {regime_name}...")
    
    context_config = config.copy()
    context_config['synapse']['inh_scaling'] = float(inh_val)
    context_config['system']['n_neurons'] = n_neurons
    
    # Baseline Lambda for verification
    actual_lambda = calculate_median_lyapunov(inh_val, config, n_neurons, seed, repetitions=1)
    
    try:
        # 1. Performance tasks
        mc = run_mc_benchmark(context_config, n_samples=800, max_lag=25)
        narma = run_narma_benchmark(context_config, length=800)
        xor = run_xor_benchmark(context_config, n_samples=200)
        
        # 2. Structural/Dynamical properties
        reservoir = Reservoir(n_neurons=n_neurons, config=context_config)
        reservoir.normalize_spectral_radius(4.0)
        
        structural_results = _compute_structural_properties(reservoir, seed, config)
        
        return {
            'N': n_neurons,
            'Seed': seed,
            'Regime': regime_name,
            'Inh_Scaling': inh_val,
            'Lambda': actual_lambda,
            'MC': mc,
            'NARMA': 1.0 - narma, # Use 1-NRMSE where higher is better
            'XOR': xor,
            **structural_results
        }
    except Exception as error:
        print(f"  [ERROR] Benchmark failed for {regime_name}: {error}")
        return _provide_default_error_results(n_neurons, seed, regime_name, inh_val, actual_lambda)

def _compute_structural_properties(reservoir, seed, config):
    """Analyzes inner dynamics: Rank, LZ-Complexity, and Separation."""
    states, spikes = _collect_reservoir_activity(reservoir, seed, config)
    
    # Alternative states for separation measurement
    states_alt = _collect_alternative_activity(reservoir, seed, config)
    
    return {
        'KernelRank': calculate_kernel_quality(np.array(states)),
        'LZ_Complexity': calculate_lz_complexity_population(np.array(spikes)),
        'Separation': calculate_separation_property(np.array(states), np.array(states_alt))
    }

def _collect_reservoir_activity(reservoir, seed, config, steps=500):
    """Collects Vm states and Spikes from the reservoir under standard stimulation."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dt = config['system']['dt']
    p_spike = config['input']['rate_signal'] * dt / 1000.0
    n_inputs = len(reservoir.input_indices)
    
    states, spikes_history = [], []
    for _ in range(steps):
        input_vector = (rng.random(n_inputs) < p_spike).astype(float)
        _, spikes, r = reservoir.step(input_vector)
        states.append(r.copy())
        spikes_history.append(spikes.copy())
        
    return states, spikes_history

def _collect_alternative_activity(reservoir, seed, config, warmup=300, steps=200):
    """Collects states under a different input rate for Separation Property."""
    np.random.seed(seed + 1)
    rng = np.random.default_rng(seed + 1)
    p_alt = 10.0 * config['system']['dt'] / 1000.0 # Distinct 10Hz rate
    n_inputs = len(reservoir.input_indices)
    
    # Partial reset
    reservoir.neuron_group.V[:] = config['neuron_hh']['V_rest']
    
    for _ in range(warmup):
        reservoir.step((rng.random(n_inputs) < p_alt).astype(float))
        
    states_alt = []
    for _ in range(steps):
        _, _, r = reservoir.step((rng.random(n_inputs) < p_alt).astype(float))
        states_alt.append(r.copy())
        
    return states_alt

def _provide_default_error_results(n_neurons, seed, regime, inh, lambda_val):
    """Returns a safe data structure if benchmarks fail."""
    return {
        'N': n_neurons, 'Seed': seed, 'Regime': regime, 'Inh_Scaling': inh, 'Lambda': lambda_val,
        'MC': 0, 'NARMA': 0, 'XOR': 0, 'KernelRank': 0, 'LZ_Complexity': 0, 'Separation': 0
    }

# ============================================================================
# ORCHESTRATION
# ============================================================================

def process_triplet_for_seed(n_neurons, seed, config):
    """Coordinates calibration and testing for a specific network realization."""
    print(f"--- Processing N={n_neurons}, Seed={seed} ---")
    
    try:
        regime_triplet = perform_surgical_calibration(n_neurons, seed, config)
    except Exception as error:
        print(f"  [CRITICAL] Calibration failed for Seed {seed}: {error}")
        return None

    seed_results = []
    for regime_name, inh_scaling in regime_triplet:
        metric_data = compute_regime_metrics(regime_name, inh_scaling, n_neurons, seed, config)
        seed_results.append(metric_data)
        
    return seed_results

def run_rigorous_study(n_neurons=100, n_seeds=30):
    config = load_task_config()
    output_path = f'RESULTS_RIGOROUS_N{n_neurons}.csv'
    
    seeds = list(range(101, 101 + n_seeds))
    master_results = []
    
    print(f"Starting Rigorous E/I Balance Study: N={n_neurons}, {n_seeds} seeds.")
    
    # Utilize parallel processing for efficiency
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_seed = {executor.submit(process_triplet_for_seed, n_neurons, s, config): s for s in seeds}
        
        for future in as_completed(future_to_seed):
            seed_id = future_to_seed[future]
            try:
                result = future.result()
                if result:
                    master_results.extend(result)
                    # Periodic save to prevent data loss
                    pd.DataFrame(master_results).to_csv(output_path, index=False)
                print(f"DONE: Seed {seed_id}")
            except Exception as e:
                print(f"  [SYSTEM ERROR] Future execution failed for Seed {seed_id}: {e}")

    if master_results:
        _display_final_summary(pd.DataFrame(master_results), n_neurons)

def _display_final_summary(dataframe, n_neurons):
    """Prints statistical averages and saves summary plot."""
    print("\n--- STATISTICAL SUMMARY ---")
    metrics = ['Lambda', 'MC', 'NARMA', 'XOR', 'KernelRank', 'LZ_Complexity', 'Separation']
    print(dataframe.groupby('Regime')[metrics].agg(['mean', 'std']))
    
    # Visual analysis
    plt.figure(figsize=(18, 10))
    for i, metric in enumerate(metrics[1:]): # Skip Lambda for the bar plots
        plt.subplot(2, 3, i+1)
        sns.barplot(data=dataframe, x='Regime', y=metric, order=['STABLE', 'EDGE', 'CHAOTIC'], palette='muted')
        plt.title(f'{metric} (N={n_neurons})')
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_name = f'PLOT_RIGOROUS_N{n_neurons}.png'
    plt.savefig(plot_name, dpi=200)
    print(f"Final analysis visualization saved to {plot_name}")

if __name__ == "__main__":
    import sys
    target_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    target_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    run_rigorous_study(n_neurons=target_n, n_seeds=target_seeds)
