import numpy as np
import yaml
import pandas as pd
import os
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

def benchmark_regime(name, inh_val, config, seed):
    """Run full benchmark suite for a specific regime"""
    print(f"   -> [{name}] Running suite at inh_scaling={inh_val:.2f}")
    
    ctx_config = config.copy()
    ctx_config['synapse']['inh_scaling'] = float(inh_val)
    
    # 1. Verify Lambda
    n_neurons = config['system']['n_neurons']
    np.random.seed(seed)
    res = Reservoir(n_neurons=n_neurons, config=ctx_config)
    res.normalize_spectral_radius(0.95)
    lambda_val = calculate_lyapunov(res, n_steps=1500)
    
    # 2. Run Benchmarks
    narma = run_narma_benchmark(ctx_config, length=800)
    xor = run_xor_benchmark(ctx_config, n_samples=200)
    mc = run_mc_benchmark(ctx_config, n_samples=500, max_lag=20)
    
    return {
        'regime': name,
        'inh_scaling': inh_val,
        'lambda': lambda_val,
        'narma_nrmse': narma,
        'xor_accuracy': xor,
        'mc_bits': mc
    }

def process_triplet(seed):
    """Generates Stable, Edge, and Chaotic states for a single seed"""
    config = load_config()
    n_neurons = config['system']['n_neurons']
    
    print(f"🚀 Processing Seed {seed}...")
    
    try:
        # 1. Locate Edge of Chaos
        critical_inh = brentq(find_lambda_zero, 1.5, 5.0, args=(config, n_neurons, seed), xtol=0.1)
        
        # 2. Define Triplets
        # Stable: More inhibition (+1.0)
        # Edge: At critical
        # Chaotic: Less inhibition (-1.0)
        triplets = [
            ('STABLE', critical_inh + 1.0),
            ('EDGE', critical_inh),
            ('CHAOTIC', critical_inh - 1.0)
        ]
        
        results = []
        for name, val in triplets:
            res_data = benchmark_regime(name, val, config, seed)
            res_data['seed'] = seed
            results.append(res_data)
            
        return results
    except Exception as e:
        print(f"❌ Seed {seed} failed: {e}")
        return None

def run_triplet_pipeline(n_seeds=5):
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010][:n_seeds]
    
    print(f"🏗️  Starting Reservoir Triplet Generation (N={n_seeds} seeds)")
    print(f"Using {cpu_count()} cores.")
    
    all_results = []
    
    # Use Pool to handle multi-seed generation
    with Pool(processes=min(cpu_count(), n_seeds)) as pool:
        triplet_results = pool.map(process_triplet, seeds)
        
    for res in triplet_results:
        if res:
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    output_file = 'reservoir_triplets_results.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("💎 TRIPLET GENERATION COMPLETE")
    print(f"Results saved to: {output_file}")
    print("="*50)
    
    # Quick Summary Table
    summary = df.groupby('regime')[['narma_nrmse', 'xor_accuracy', 'mc_bits', 'lambda']].mean()
    print(summary)

if __name__ == "__main__":
    run_triplet_pipeline(n_seeds=3) # Small batch for demo
