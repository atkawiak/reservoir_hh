import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from src.model.reservoir import Reservoir
from src.benchmarks.trainer import ReservoirTrainer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_mc_benchmark(config=None, n_samples=2000, max_lag=60):
    if config is None:
        config = load_config('task_config.yaml')
    elif isinstance(config, str):
        config = load_config(config)
        
    n_neurons = config['system']['n_neurons']
    dt = config['system']['dt']
    
    # Setup Reservoir
    res = Reservoir(n_neurons=n_neurons, config=config)
    n_input = len(res.input_indices)
    
    # Task: MC
    symbol_duration = 50.0 
    steps_per_symbol = int(symbol_duration / dt)
    
    # Generate input signal (i.i.d. uniform random)
    u_discrete = np.random.uniform(0.0, 1.0, n_samples)
    base_rate = config['input']['rate_background']
    max_rate = config['input']['rate_signal']
    
    # --- WARM-UP PHASE (Washout) ---
    warmup_samples = 200
    for s in range(warmup_samples):
        input_val = np.random.uniform(0, 1)
        target_rate = base_rate + input_val * (max_rate - base_rate)
        p_spike = target_rate * dt / 1000.0
        for t in range(steps_per_symbol):
            spikes_in = (np.random.rand(n_input) < p_spike).astype(float)
            res.step(spikes_in)

    # --- MAIN SIMULATION ---
    features = []
    for s in range(n_samples):
        input_val = u_discrete[s]
        target_rate = base_rate + input_val * (max_rate - base_rate)
        p_spike = target_rate * dt / 1000.0
        
        for t in range(steps_per_symbol):
            spikes_in = (np.random.rand(n_input) < p_spike).astype(float)
            _, spikes_out, r = res.step(spikes_in)
            
        features.append(r.copy())
        
    X = np.array(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split train/test
    split = int(n_samples * 0.7)
    X_train, X_test = X[:split], X[split:]
    u_train, u_test = u_discrete[:split], u_discrete[split:]
    
    total_mc = 0
    mc_per_lag = []
    
    # Use small alpha for MC as it's a linear memory task
    ridge_alpha = config['benchmarks']['mc']['ridge_alpha']
    
    for k in range(1, max_lag + 1):
        # CORRECT DELAYED TARGET: y[n] = u[n-k]
        y_delayed = np.concatenate([np.zeros(k), u_discrete[:-k]])
        
        y_train_k = y_delayed[:split]
        y_test_k = y_delayed[split:]
        
        # Train
        trainer = ReservoirTrainer(alpha=ridge_alpha)
        trainer.train(X_train, y_train_k)
        
        # Predict
        y_pred = trainer.predict(X_test)
        
        # Corr^2 (standard MC definition)
        corr = np.corrcoef(y_test_k, y_pred)[0, 1]
        mc_k = corr**2 if not np.isnan(corr) and corr > 0 else 0
        
        total_mc += mc_k
        mc_per_lag.append(mc_k)
            
    # Return curve only if requested or if running standalone
    return total_mc

def run_mc_benchmark_full(config=None, n_samples=2000, max_lag=60):
    """Version that returns the curve for plotting"""
    if config is None:
        config = load_config('task_config.yaml')
    elif isinstance(config, str):
        config = load_config(config)
        
    n_neurons = config['system']['n_neurons']
    dt = config['system']['dt']
    
    res = Reservoir(n_neurons=n_neurons, config=config)
    n_input = len(res.input_indices)
    symbol_duration = 50.0 
    steps_per_symbol = int(symbol_duration / dt)
    u_discrete = np.random.uniform(0.0, 1.0, n_samples)
    base_rate = config['input']['rate_background']
    max_rate = config['input']['rate_signal']
    
    warmup_samples = 200
    for s in range(warmup_samples):
        input_val = np.random.uniform(0, 1)
        target_rate = base_rate + input_val * (max_rate - base_rate)
        p_spike = target_rate * dt / 1000.0
        for t in range(steps_per_symbol):
            spikes_in = (np.random.rand(n_input) < p_spike).astype(float)
            res.step(spikes_in)

    features = []
    for s in range(n_samples):
        input_val = u_discrete[s]
        target_rate = base_rate + input_val * (max_rate - base_rate)
        p_spike = target_rate * dt / 1000.0
        for t in range(steps_per_symbol):
            spikes_in = (np.random.rand(n_input) < p_spike).astype(float)
            _, spikes_out, r = res.step(spikes_in)
        features.append(r.copy())
        
    X = np.array(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    split = int(n_samples * 0.7)
    X_train, X_test = X[:split], X[split:]
    total_mc = 0
    mc_per_lag = []
    ridge_alpha = config['benchmarks']['mc']['ridge_alpha']
    
    for k in range(1, max_lag + 1):
        y_delayed = np.concatenate([np.zeros(k), u_discrete[:-k]])
        y_train_k = y_delayed[:split]
        y_test_k = y_delayed[split:]
        trainer = ReservoirTrainer(alpha=ridge_alpha)
        trainer.train(X_train, y_train_k)
        y_pred = trainer.predict(X_test)
        corr = np.corrcoef(y_test_k, y_pred)[0, 1]
        mc_k = corr**2 if not np.isnan(corr) and corr > 0 else 0
        total_mc += mc_k
        mc_per_lag.append(mc_k)
            
    return total_mc, mc_per_lag, X

if __name__ == "__main__":
    final_mc, curve, _ = run_mc_benchmark_full(n_samples=2000, max_lag=60)
    print(f"\nFinal Memory Capacity: {final_mc:.2f} bits")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(curve)+1), curve, 'o-', linewidth=2, markersize=8, color='#2c3e50')
    plt.fill_between(range(1, len(curve)+1), curve, color='#3498db', alpha=0.2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Lag (k)', fontsize=12)
    plt.ylabel('Memory Capacity $R^2(k)$', fontsize=12)
    plt.title(f'Memory Capacity Curve (Total MC = {final_mc:.2f} bits)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    
    plt.text(0.95, 0.95, f'Total MC: {final_mc:.2f} bits', 
             transform=plt.gca().transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plot_path = 'mc_curve_fixed.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Figure saved to: {plot_path}")
    plt.show()
