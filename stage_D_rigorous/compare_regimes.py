import numpy as np
import yaml
import pandas as pd
from src.model.reservoir import Reservoir
from src.utils.metrics import calculate_lyapunov
from benchmark_mc import run_mc_benchmark_full
from benchmark_narma10 import run_narma_benchmark
from repro_working_reservoir import run_xor_benchmark
import matplotlib.pyplot as plt
import seaborn as sns

def find_regimes(seed=101):
    with open('task_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    n_neurons = config['system']['n_neurons']
    
    # We will vary Rho to find the regimes
    # Rho < 1.0 (usually Stable)
    # Rho ~ 2.0-5.0 (usually Edge/Chaos for HH)
    rho_range = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    results = []
    
    print("Searching for Stable, Edge, and Chaotic regimes...")
    for rho in rho_range:
        np.random.seed(seed)
        config['dynamics_control']['target_spectral_radius'] = rho
        res = Reservoir(n_neurons=n_neurons, config=config)
        res.normalize_spectral_radius(rho)
        l = calculate_lyapunov(res, n_steps=2000, seed=seed)
        print(f"  Rho={rho:.1f} -> Lambda={l:.4f}")
        results.append((rho, l))
    
    return results

def run_regime_benchmarks(regimes_dict, seed=101):
    """
    regimes_dict: { 'Stable': rho_val, 'Edge': rho_val, 'Chaotic': rho_val }
    """
    with open('task_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    final_results = []
    
    for label, rho in regimes_dict.items():
        print(f"\nBenchmarking {label} regime (Rho={rho})...")
        config['dynamics_control']['target_spectral_radius'] = rho
        np.random.seed(seed)
        
        # 1. Performance
        mc = run_mc_benchmark_full(config, n_samples=2000, max_lag=40)[0]
        narma = run_narma_benchmark(config, length=1500)
        xor = run_xor_benchmark(config, n_samples=500)
        
        final_results.append({
            'Regime': label,
            'Rho': rho,
            'MC (bits)': mc,
            'NARMA (1-NRMSE)': 1.0 - narma, # Higher is better
            'XOR Accuracy': xor
        })
        
    return pd.DataFrame(final_results)

def plot_triplet_results(df):
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['MC (bits)', 'NARMA (1-NRMSE)', 'XOR Accuracy']
    colors = ['#34495e', '#e74c3c', '#2ecc71']
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x='Regime', y=metric, ax=axes[i], order=['Stable', 'Edge', 'Chaotic'], palette="viridis")
        axes[i].set_title(f'{metric} across Regimes', fontsize=14, fontweight='bold')
        axes[i].set_ylim(0, df[metric].max() * 1.2)
        
    plt.tight_layout()
    plt.savefig('REGIME_COMPARISON.png', dpi=300)
    print("\nSaved: REGIME_COMPARISON.png")

if __name__ == "__main__":
    # 1. First find the rhos
    # Based on previous tests and general HH knowledge:
    # Stable: Rho=0.5
    # Edge: Rho=1.5 - 2.5
    # Chaotic: Rho=10.0
    
    # Let's verify with find_regimes or just use these educated guesses for speed
    # or even better, let the script find them.
    
    # For now, let's use fixed targets based on previous 'check_rho' and 'check_lambda' outputs
    # Let's adjust based on the -0.01 to -0.05 typical values we saw.
    
    triplet = {
        'Stable': 0.5,
        'Edge': 2.5,   # We saw Lambda ~ -0.04 here
        'Chaotic': 15.0 # Let's go high to ensure positive lambda
    }
    
    results_df = run_regime_benchmarks(triplet)
    plot_triplet_results(results_df)
    
    print("\nFinal Results Table:")
    print(results_df)
