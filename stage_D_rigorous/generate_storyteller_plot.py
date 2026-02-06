import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.model.reservoir import Reservoir
from src.utils.metrics import calculate_lyapunov
from benchmark_mc import run_mc_benchmark_full
from benchmark_narma10 import run_narma_benchmark

def generate_storyteller_data(seed=101):
    with open('task_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    n_neurons = config['system']['n_neurons']
    inh_range = np.linspace(1.0, 5.0, 10)
    results = []
    
    print(f"Generating Storyteller data for Seed {seed}...")
    
    for inh in inh_range:
        print(f"  Testing Inh Scaling: {inh:.2f}")
        np.random.seed(seed)
        config['synapse']['inh_scaling'] = float(inh)
        
        # 1. Reservoir with correct rho
        res = Reservoir(n_neurons=n_neurons, config=config)
        res.normalize_spectral_radius(0.95)
        
        # 2. Measures
        l = calculate_lyapunov(res, n_steps=1500, seed=seed)
        mc, mc_curve = run_mc_benchmark_full(config, n_samples=1500, max_lag=40)
        narma = run_narma_benchmark(config, length=1000)
        
        results.append({
            'inh': inh,
            'lambda': l,
            'mc': mc,
            'narma': narma
        })
        
    return pd.DataFrame(results), mc_curve

def plot_storyteller(df, mc_curve):
    sns.set_theme(style="white", palette="muted")
    fig = plt.figure(figsize=(18, 5))
    
    # Color definition
    color_mc = '#2c3e50'
    color_narma = '#e74c3c'
    color_lambda = '#3498db'
    
    # Panel A: Memory Capacity Curve (The Fix)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(range(1, len(mc_curve)+1), mc_curve, 'o-', color=color_mc, linewidth=2, markersize=6)
    ax1.fill_between(range(1, len(mc_curve)+1), mc_curve, color=color_mc, alpha=0.1)
    ax1.set_title('A. Fading Memory Capacity (Corrected)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Lag (k)', fontsize=12)
    ax1.set_ylabel('Recall Accuracy $R^2(k)$', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel B: Performance vs Lyapunov (The Proof)
    ax2 = fig.add_subplot(1, 3, 2)
    # MC Axis
    ax2_mc = ax2.twinx()
    sns.lineplot(data=df, x='lambda', y='mc', ax=ax2, color=color_mc, marker='o', linewidth=3, label='MC (Linear)')
    # NARMA Axis (NRMSE - lower is better, so we plot 1-NRMSE or just NRMSE)
    sns.lineplot(data=df, x='lambda', y='narma', ax=ax2_mc, color=color_narma, marker='s', linewidth=3, label='NARMA (Nonlinear)')
    
    ax2.set_title('B. Edge of Chaos Hypothesis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lyapunov Exponent ($\lambda$)', fontsize=12)
    ax2.set_ylabel('MC [bits]', color=color_mc, fontsize=12)
    ax2_mc.set_ylabel('NARMA NRMSE', color=color_narma, fontsize=12)
    
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax2.text(0.02, ax2.get_ylim()[1]*0.9, 'Edge of Chaos', rotation=90, verticalalignment='top')
    
    # Panel C: A-Current Tuning (The Mechanism)
    ax3 = fig.add_subplot(1, 3, 3)
    # Here we show Lambda vs Inhalation Scaling
    sns.lineplot(data=df, x='inh', y='lambda', ax=ax3, color=color_lambda, linewidth=3, marker='D')
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('C. Balancing Global Dynamics', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Inhibitory Scaling factor', fontsize=12)
    ax3.set_ylabel('Chaos Level ($\lambda$)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('STORYTELLER_RESULTS.png', dpi=300)
    print("Updated: STORYTELLER_RESULTS.png")

if __name__ == "__main__":
    df, mc_curve = generate_storyteller_data()
    plot_storyteller(df, mc_curve)
