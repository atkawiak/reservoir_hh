import numpy as np
import yaml
import pandas as pd
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

def find_lambda_zero(inh_val, config, n_neurons, seed):
    """Target function for root finding: returns lambda for given inhibition"""
    np.random.seed(seed)
    ctx_config = config.copy()
    ctx_config['synapse']['inh_scaling'] = float(inh_val)
    
    res = Reservoir(n_neurons=n_neurons, config=ctx_config)
    res.normalize_spectral_radius(0.95)
    return calculate_lyapunov(res, n_steps=1500)

def process_single_reservoir(seed):
    """Full pipeline for one reservoir seed"""
    config = load_config()
    n_neurons = config['system']['n_neurons']
    
    print(f"-> Seed {seed}: Locating Edge of Chaos...")
    
    try:
        # 1. Find Critical Inhibition (lambda approx 0)
        # We search between 1.5 (chaos) and 4.5 (stable)
        critical_inh = brentq(find_lambda_zero, 1.5, 4.5, args=(config, n_neurons, seed), xtol=0.1)
        
        # 2. Setup critical config
        crit_config = config.copy()
        crit_config['synapse']['inh_scaling'] = float(critical_inh)
        np.random.seed(seed) # Ensure benchmarks use same weight realization
        
        # 3. Run Benchmarks
        print(f"-> Seed {seed}: Running benchmarks at Inh={critical_inh:.2f}...")
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

def run_experiment(n_batch=10):
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    
    print(f"🚀 Starting protocol for {n_batch} reservoirs...")
    print(f"Using {cpu_count()} cores.")
    
    results = []
    # Use Pool for parallel processing
    with Pool(processes=min(cpu_count(), n_batch)) as pool:
        results = pool.map(process_single_reservoir, seeds)
    
    df = pd.DataFrame(results)
    df.to_csv('batch_10_results.csv', index=False)
    
    # Summary Statistics
    success_df = df[df['status'] == 'SUCCESS']
    print("\n" + "="*50)
    print("BATCH 10 RESULTS SUMMARY")
    print("="*50)
    print(f"Successful: {len(success_df)}/10")
    if len(success_df) > 0:
        print(f"Avg Critical Inh: {success_df['critical_inh'].mean():.3f} ± {success_df['critical_inh'].std():.3f}")
        print(f"Avg NARMA NRMSE:  {success_df['narma_nrmse'].mean():.4f}")
        print(f"Avg XOR Accuracy: {success_df['xor_accuracy'].mean():.2%}")
        print(f"Avg MC bits:      {success_df['mc_bits'].mean():.2f}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
