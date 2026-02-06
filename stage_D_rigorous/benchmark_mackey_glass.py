import numpy as np
import yaml
from src.model.reservoir import Reservoir
from src.benchmarks.trainer import ReservoirTrainer
from src.utils.metrics import get_nrmse
import matplotlib.pyplot as plt

def generate_mackey_glass(length=2000, beta=0.2, gamma=0.1, n=10, tau=17, dt=1.0):
    """
    Generate Mackey-Glass chaotic time series.
    """
    n_total = length + tau + 100 # Extra for warmup
    x = np.zeros(n_total)
    x[:tau+1] = 0.5 + 0.1 * np.random.rand(tau+1)
    
    for i in range(tau, n_total - 1):
        x[i+1] = x[i] + dt * (beta * x[i-tau] / (1 + x[i-tau]**n) - gamma * x[i])
        
    return x[-(length+1):-1] # Return requested length

def run_mackey_glass_benchmark(config=None, length=2000, prediction_horizon=10):
    if config is None:
        with open('task_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
    n_neurons = config['system']['n_neurons']
    rho = config['dynamics_control']['target_spectral_radius']
    
    # Generate data
    data = generate_mackey_glass(length=length + prediction_horizon + 100)
    
    # Target is prediction_horizon steps ahead
    u = data[:-prediction_horizon]
    y_target = data[prediction_horizon:]
    
    # Initialize Reservoir
    res = Reservoir(n_neurons=n_neurons, config=config)
    res.normalize_spectral_radius(rho)
    
    # Warmup
    warmup = 100
    for i in range(warmup):
        u_val = u[i]
        # Encode as periodic/poisson if needed, but for MG often we use direct injection
        # or simplified poisson. Let's use poisson for consistency with HH experiments.
        p = (u_val + 1.0) * 20.0 * res.dt / 1000.0 # Map to freq
        u_spike = (np.random.rand(len(res.input_indices)) < p).astype(float)
        res.step(u_spike)
        
    # Training collection
    X_states = []
    for i in range(warmup, len(u)):
        u_val = u[i]
        p = (u_val + 1.0) * 20.0 * res.dt / 1000.0
        u_spike = (np.random.rand(len(res.input_indices)) < p).astype(float)
        res.step(u_spike)
        X_states.append(res.neuron_group.V.copy())
        
    X_states = np.array(X_states)
    y_target = y_target[warmup:]
    
    # Split
    split = int(len(X_states) * 0.7)
    X_train, X_test = X_states[:split], X_states[split:]
    y_train, y_test = y_target[:split], y_target[split:]
    
    # Train
    trainer = ReservoirTrainer(alpha=1e-6)
    trainer.train(X_train, y_train)
    
    # Predict
    y_pred = trainer.predict(X_test)
    
    nrmse = get_nrmse(y_test, y_pred)
    return 1.0 - nrmse # Return 'performance' (higher is better)

if __name__ == "__main__":
    perf = run_mackey_glass_benchmark()
    print(f"Mackey-Glass Performance (1-NRMSE): {perf:.4f}")
