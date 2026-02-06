import numpy as np
import yaml
import pandas as pd
import os
import time
from multiprocessing import Pool, cpu_count
from scipy.optimize import brentq
from src.model.reservoir import Reservoir
from src.utils.metrics import calculate_lyapunov
from benchmark_narma10 import run_narma_benchmark
from repro_working_reservoir import run_xor_benchmark
from benchmark_mc import run_mc_benchmark

def load_config():
    with open('task_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_lambda(inh_val, config, n_neurons, seed):
    """Safety-wrapped lambda measurement"""
    try:
        np.random.seed(seed)
        ctx_config = config.copy()
        ctx_config['synapse']['inh_scaling'] = float(inh_val)
        res = Reservoir(n_neurons=n_neurons, config=ctx_config)
        res.normalize_spectral_radius(0.95)
        return calculate_lyapunov(res, n_steps=2000)
    except:
        return 999.0 # Signal error

def process_single_reservoir(seed):
    """Robust pipeline for mass production"""
    config = load_config()
    n_neurons = config['system']['n_neurons']
    
    # 1. Broad Search for Root Bracket
    # Some seeds are naturally very stable, some very chaotic. 
    # We check a wider range.
    test_points = [1.0, 2.5, 4.5, 7.0]
    lambdas = [get_lambda(p, config, n_neurons, seed) for p in test_points]
    
    # Find where it crosses zero
    bracket = None
    for i in range(len(lambdas)-1):
        if lambdas[i] > 0 and lambdas[i+1] < 0:
            bracket = (test_points[i], test_points[i+1])
            break
        elif lambdas[i] < 0 and lambdas[i+1] > 0: # Unusual but possible
            bracket = (test_points[i], test_points[i+1])
            break
            
    if not bracket:
        return {'seed': seed, 'status': 'FAILED: No zero found in range 1.0-7.0'}

    try:
        # 2. Precision targeting of Edge of Chaos
        critical_inh = brentq(get_lambda, bracket[0], bracket[1], 
                             args=(config, n_neurons, seed), xtol=0.1)
        
        # 3. Setup critical config & benchmarks
        crit_config = config.copy()
        crit_config['synapse']['inh_scaling'] = float(critical_inh)
        np.random.seed(seed) 
        
        # Reduced length for mass testing to keep it feasible
        narma = run_narma_benchmark(crit_config, length=800)
        xor = run_xor_benchmark(crit_config, n_samples=200)
        mc = run_mc_benchmark(crit_config, n_samples=500, max_lag=20)
        
        return {
            'seed': seed,
            'critical_inh': critical_inh,
            'narma_nrmse': narma,
            'xor_accuracy': xor,
            'mc_bits': mc,
            'status': 'SUCCESS'
        }
    except Exception as e:
        return {'seed': seed, 'status': f'FAILED: {str(e)}'}

def run_mass_production(total_n=100):
    output_file = f'mass_results_{total_n}.csv'
    
    # Generate unique seeds
    np.random.seed(42)
    all_seeds = np.random.randint(1000, 99999, total_n).tolist()
    
    print(f"🏭 Starting MASS PRODUCTION of {total_n} reservoirs...")
    print(f"System: {cpu_count()} cores detected.")
    
    start_time = time.time()
    results = []
    
    # Use Pool to parallelize
    with Pool(processes=cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(process_single_reservoir, all_seeds)):
            results.append(res)
            # Periodic saving
            if len(results) % 5 == 0:
                pd.DataFrame(results).to_csv(output_file, index=False)
                elapsed = time.time() - start_time
                print(f"🚀 Progress: {len(results)}/{total_n} | Time: {elapsed:.1f}s | Success Rate: {len([r for r in results if r['status'] == 'SUCCESS'])}/{len(results)}")

    print(f"\n✅ FULL BATCH COMPLETE. Total time: {time.time() - start_time:.1f}s")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Let's start with 100 as the first big wave
    run_mass_production(total_n=100)
