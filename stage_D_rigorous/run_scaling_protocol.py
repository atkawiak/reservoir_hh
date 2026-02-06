import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.model.reservoir import Reservoir
from src.utils.metrics import calculate_lyapunov, get_mc, calculate_kernel_quality, calculate_separation_property
from benchmark_mc import run_mc_benchmark_full
from benchmark_mackey_glass import run_mackey_glass_benchmark

def run_scaling_experiment(n_neurons_list=[100, 200], rho_values=np.linspace(0.5, 10.0, 8), seed=101):
    results = []
    
    with open('task_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
        
    for n_neurons in n_neurons_list:
        print(f"\nRunning Scaling Experiment for N={n_neurons}...")
        base_config['system']['n_neurons'] = n_neurons
        
        for rho in rho_values:
            print(f"  Testing Rho={rho:.2f}...", end=" ", flush=True)
            np.random.seed(seed)
            base_config['dynamics_control']['target_spectral_radius'] = rho
            
            # Initialize reservoir
            res = Reservoir(n_neurons=n_neurons, config=base_config)
            res.normalize_spectral_radius(rho)
            
            # 1. Lyapunov
            l_val = calculate_lyapunov(res, n_steps=2000, seed=seed)
            
            # Performance (MC & Mackey-Glass)
            # MC
            try:
                mc_val, _, X_states = run_mc_benchmark_full(base_config, n_samples=1500, max_lag=40)
                # Handle potential NaNs from numerical instability in chaotic regime
                if np.any(np.isnan(X_states)):
                    X_states = np.nan_to_num(X_states, 0.0)
            except Exception as e:
                print(f"Error in MC: {e}")
                mc_val = 0
                X_states = np.zeros((1500, n_neurons))
            
            # Mackey-Glass
            try:
                mg_perf = run_mackey_glass_benchmark(base_config, length=1500)
            except Exception as e:
                print(f"Error in MG: {e}")
                mg_perf = 0
            
            # 3. Mechanistic Metrics
            # Kernel Quality (Rank)
            rank_val = calculate_kernel_quality(X_states)
            
            # Separation Property
            sep_val = calculate_separation_property(res, n_steps=1000)
            
            print(f"Lambda={l_val:.4f}, MC={mc_val:.2f}, MG={mg_perf:.2f}")
            
            results.append({
                'N': n_neurons,
                'Rho': rho,
                'Lambda': l_val,
                'MC': mc_val,
                'MackeyGlass': mg_perf,
                'Rank': rank_val,
                'Separation': sep_val
            })
            
    return pd.DataFrame(results)

def plot_comprehensive_proof(df):
    sns.set_theme(style="whitegrid")
    # 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Performance vs Lambda (The Proof)
    for n in df['N'].unique():
        sub = df[df['N'] == n]
        axes[0,0].plot(sub['Lambda'], sub['MC'], 'o-', label=f'N={n} (MC)')
        axes[0,1].plot(sub['Lambda'], sub['MackeyGlass'], 's--', label=f'N={n} (MG)')
        
    axes[0,0].set_title('Memory Capacity vs Lambda', fontweight='bold')
    axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[0,0].set_xlabel('Lyapunov Exponent (Lambda)')
    axes[0,0].legend()
    
    axes[0,1].set_title('Mackey-Glass vs Lambda', fontweight='bold')
    axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('Lyapunov Exponent (Lambda)')
    axes[0,1].legend()
    
    # 2. Mechanistic Metrics vs Lambda
    for n in df['N'].unique():
        sub = df[df['N'] == n]
        axes[1,0].plot(sub['Lambda'], sub['Rank'], '^-', label=f'N={n}')
        axes[1,1].plot(sub['Lambda'], sub['Separation'], 'x-', label=f'N={n}')
        
    axes[1,0].set_title('Kernel Quality (Rank) vs Lambda', fontweight='bold')
    axes[1,0].set_xlabel('Lyapunov Exponent (Lambda)')
    axes[1,0].legend()
    
    axes[1,1].set_title('Separation Property vs Lambda', fontweight='bold')
    axes[1,1].set_xlabel('Lyapunov Exponent (Lambda)')
    axes[1,1].legend()
    
    # 3. Phase Diagram: Lambda vs Rho
    for n in df['N'].unique():
        sub = df[df['N'] == n]
        axes[0,2].plot(sub['Rho'], sub['Lambda'], 'D-', label=f'N={n}')
        
    axes[0,2].set_title('Dynamics Control: Lambda vs Rho', fontweight='bold')
    axes[0,2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0,2].set_xlabel('Spectral Radius (Rho)')
    axes[0,2].legend()
    
    # Delete unused subplot
    fig.delaxes(axes[1,2])
    
    plt.tight_layout()
    plt.savefig('COMPREHENSIVE_PROOF_SCALING.png', dpi=300)
    print("\nSaved: COMPREHENSIVE_PROOF_SCALING.png")

if __name__ == "__main__":
    df_results = run_scaling_experiment(n_neurons_list=[100, 200], rho_values=np.linspace(0.5, 8.0, 6))
    plot_comprehensive_proof(df_results)
    df_results.to_csv('SCALING_EXPERIMENT_RESULTS.csv', index=False)
