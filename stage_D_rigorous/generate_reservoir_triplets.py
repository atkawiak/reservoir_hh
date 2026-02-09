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

def load_config():
    with open('task_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def measure_lambda_robust(inh_val, config, n_neurons, seed, repetitions=3):
    """
    Measure Lyapunov exponent robustly by taking the median of multiple runs.
    Ensures E/I balance (inh_scaling) is the control parameter.
    """
    np.random.seed(seed)
    ctx_config = config.copy()
    ctx_config['synapse']['inh_scaling'] = float(inh_val)
    
    lambdas = []
    for i in range(repetitions):
        res = Reservoir(n_neurons=n_neurons, config=ctx_config)
        # We normalize spectral radius to a baseline (4.0) so that 
        # the ONLY thing that moves the system is the E/I balance (inh_scaling).
        res.normalize_spectral_radius(4.0)
        l_val = calculate_lyapunov(res, n_steps=1200, seed=seed + i*100)
        lambdas.append(l_val)
        
    return np.median(lambdas)

# ============================================================================
# CALIBRATION & BENCHMARKING
# ============================================================================

def process_triplet_for_seed(n_neurons, seed, config):
    """
    Surgical Precision: Dense Scan followed by Brentq refinement.
    Ensures that EDGE is exactly Lambda=0 with error recovery.
    """
    print(f"--- Processing N={n_neurons}, Seed={seed} (Surgical Mode) ---")
    
    # 1. DENSE SCAN (100 points) to map the territory
    try:
        def get_l(inh):
            np.random.seed(seed) # Strict local determinism
            config_copy = config.copy()
            config_copy['synapse']['inh_scaling'] = float(inh)
            res = Reservoir(n_neurons=n_neurons, config=config_copy)
            res.normalize_spectral_radius(4.0)
            # 1200 steps for better convergence
            return calculate_lyapunov(res, n_steps=1200, seed=seed)

        scan_inhs = np.linspace(0.5, 12.0, 100)
        scan_lambdas = []
        for inh in scan_inhs:
            scan_lambdas.append(get_l(inh))
        
        scan_lambdas = np.array(scan_lambdas)
        peak_idx = np.argmax(scan_lambdas)
        peak_inh = scan_inhs[peak_idx]
        
        # 2. FIND CROSSING POINT (on the right of the peak)
        brent_interval = None
        for i in range(peak_idx, len(scan_lambdas)-1):
            if scan_lambdas[i] * scan_lambdas[i+1] <= 0:
                brent_interval = (scan_inhs[i], scan_inhs[i+1])
                break
        
        if brent_interval:
            print(f"  [ZOOM] Found crossing in interval {brent_interval}. Refining...")
            try:
                edge_inh = brentq(get_l, brent_interval[0], brent_interval[1], xtol=0.01)
            except ValueError:
                # Handle unexpected sign issues due to noise
                la, lb = get_l(brent_interval[0]), get_l(brent_interval[1])
                edge_inh = brent_interval[0] if abs(la) < abs(lb) else brent_interval[1]
                print(f"  [RECOVERY] Brent sign mismatch, using best scan point: Inh={edge_inh:.2f}")
        else:
            right_side = scan_lambdas[peak_idx:]
            edge_idx = peak_idx + np.argmin(np.abs(right_side))
            edge_inh = scan_inhs[edge_idx]
            print(f"  [FALLBACK] No crossing, using closest Lambda={scan_lambdas[edge_idx]:.4f}")

        stable_inh = min(edge_inh + 4.0, 12.0)
        print(f"  [SURGICAL] CHAOTIC(Peak)={peak_inh:.2f}, EDGE(Zero)={edge_inh:.2f}, STABLE={stable_inh:.2f}")

        triplets = [
            ('STABLE', stable_inh),
            ('EDGE', edge_inh),
            ('CHAOTIC', peak_inh)
        ]
        
    except Exception as e:
        print(f"  [ERROR] Surgical Calibration failed for Seed {seed}: {e}")
        return None

    results = []
    for regime, inh_val in triplets:
        print(f"  [BENCH] Regime: {regime} (inh_scaling={inh_val:.2f})...")
        
        ctx_config = config.copy()
        ctx_config['synapse']['inh_scaling'] = float(inh_val)
        ctx_config['system']['n_neurons'] = n_neurons
        
        # Verify Lambda for this state
        l_verify = measure_lambda_robust(inh_val, config, n_neurons, seed, repetitions=1)
        
        # Run suite
        try:
            mc = run_mc_benchmark(ctx_config, n_samples=800, max_lag=25)
            narma = run_narma_benchmark(ctx_config, length=800)
            xor = run_xor_benchmark(ctx_config, n_samples=200)
            
            # --- NEW METRICS ---
            # 1. Get States for Kernel Rank and LZ Complexity
            np.random.seed(seed)
            res = Reservoir(n_neurons=n_neurons, config=ctx_config)
            res.normalize_spectral_radius(4.0)
            
            states = []
            spikes_history = []
            p_in = config['input']['rate_signal'] * config['system']['dt'] / 1000.0
            n_in = len(res.input_indices)
            rng = np.random.default_rng(seed)
            
            for _ in range(500): # 500 steps for structural analysis
                u = (rng.random(n_in) < p_in).astype(float)
                _, spikes, r = res.step(u)
                states.append(r.copy())
                spikes_history.append(spikes.copy())
            
            kernel_rank = calculate_kernel_quality(np.array(states))
            lz_complexity = calculate_lz_complexity_population(np.array(spikes_history))
            
            # 2. Separation Property (Reuse current instance)
            # We already have states for rate_signal. Let's get them for a different rate.
            states_alt = []
            p_alt = 10.0 * config['system']['dt'] / 1000.0 # Alternative low rate
            
            # Reset just V to get a fresh-ish trajectory from the same topology
            res.neuron_group.V[:] = config['neuron_hh']['V_rest']
            for _ in range(300): # Warmup alt
                u = (rng.random(n_in) < p_alt).astype(float)
                res.step(u)
            for _ in range(200): # Measure alt
                u = (rng.random(n_in) < p_alt).astype(float)
                _, _, r = res.step(u)
                states_alt.append(r.copy())
            
            separation = calculate_separation_property(np.array(states), np.array(states_alt))
            
        except Exception as e:
            import traceback
            print(f"  [ERROR] Benchmark failed for Seed {seed}: {e}")
            traceback.print_exc()
            mc, narma, xor, kernel_rank, lz_complexity, separation = 0, 1.0, 0, 0, 0, 0
            
        results.append({
            'N': n_neurons,
            'Seed': seed,
            'Regime': regime,
            'Inh_Scaling': inh_val,
            'Lambda': l_verify,
            'MC': mc,
            'NARMA': 1.0 - narma, 
            'XOR': xor,
            'KernelRank': kernel_rank,
            'LZ_Complexity': lz_complexity,
            'Separation': separation
        })
        
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_rigorous_study(n_neurons=100, n_seeds=30):
    config = load_config()
    output_file = f'RESULTS_RIGOROUS_N{n_neurons}.csv'
    
    seeds = list(range(101, 101 + n_seeds))
    all_results = []
    
    print(f"Starting Rigorous E/I Balance Study: N={n_neurons}, {n_seeds} seeds.")
    print(f"Results will be saved to: {output_file}")
    
    num_workers = 4 # Full power
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_triplet_for_seed, n_neurons, s, config): s for s in seeds}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_results.extend(res)
                pd.DataFrame(all_results).to_csv(output_file, index=False)
            completed += 1
            print(f"DONE [{completed}/{n_seeds}] Seed {futures[future]}")

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)
    
    # Statistical Summary
    print("\n--- STATISTICAL SUMMARY ---")
    metrics_list = ['Lambda', 'MC', 'NARMA', 'XOR', 'KernelRank', 'LZ_Complexity', 'Separation']
    summary = df.groupby('Regime')[metrics_list].agg(['mean', 'std'])
    print(summary)
    
    # Plotting
    plt.figure(figsize=(20, 10))
    plot_metrics = ['MC', 'NARMA', 'XOR', 'KernelRank', 'LZ_Complexity', 'Separation']
    for i, m in enumerate(plot_metrics):
        plt.subplot(2, 3, i+1)
        sns.barplot(data=df, x='Regime', y=m, order=['STABLE', 'EDGE', 'CHAOTIC'], palette='muted')
        plt.title(f'{m} across Regimes (N={n_neurons})')
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = f'PLOT_RIGOROUS_N{n_neurons}.png'
    plt.savefig(plot_file, dpi=200)
    print(f"Final analysis saved to {plot_file}")

if __name__ == "__main__":
    import sys
    n = 100
    seeds = 30
    if len(sys.argv) > 1: n = int(sys.argv[1])
    if len(sys.argv) > 2: seeds = int(sys.argv[2])
    
    run_rigorous_study(n_neurons=n, n_seeds=seeds)
