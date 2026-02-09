import numpy as np
import yaml
from src.model.reservoir import Reservoir
from src.benchmarks.trainer import ReservoirTrainer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_narma10(length):
    u = np.random.uniform(0.0, 0.5, length)
    y = np.zeros(length)
    for t in range(10, length-1):
        sum_y = np.sum(y[t-9:t+1])
        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * sum_y + 1.5 * u[t-9] * u[t] + 0.1
    return u, y

def run_narma_benchmark(config=None, length=2000):
    if config is None:
        config = load_config('task_config.yaml')
        
    n_neurons = config['system']['n_neurons']
    dt = config['system']['dt']
    
    # Reservoir Setup
    res = Reservoir(n_neurons=n_neurons, config=config)
    n_input = len(res.input_indices)
    
    # Rate Coding Parameters
    base_rate = config['input']['rate_background']
    rate_scale = (config['input']['rate_signal'] - base_rate) / 0.5
    
    symbol_duration = 50.0 
    steps_per_symbol = int(symbol_duration / dt)

    # --- WARM-UP PHASE (Washout) ---
    warmup_samples = 100
    for s in range(warmup_samples):
        input_val = np.random.uniform(0, 0.5)
        target_rate = base_rate + input_val * rate_scale
        p_spike = target_rate * dt / 1000.0
        for t in range(steps_per_symbol):
            spikes_in = (np.random.rand(n_input) < p_spike).astype(float)
            res.step(spikes_in)

    # --- MAIN SIMULATION ---
    u, target = generate_narma10(length)
    features = []
    for k in range(length):
        input_val = u[k]
        target_rate = base_rate + input_val * rate_scale
        p_spike = target_rate * dt / 1000.0
        
        for t in range(steps_per_symbol):
            spikes_in = (np.random.rand(n_input) < p_spike).astype(float)
            _, _, r = res.step(spikes_in)
            
        features.append(r.copy())
            
    X = np.array(features)
    y = target
    
    # Additional washout from NARMA sequence
    X = X[10:]
    y = y[10:]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    split = int(len(y) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    trainer = ReservoirTrainer(alpha=config['benchmarks']['narma10']['ridge_alpha'])
    trainer.train(X_train, y_train)
    y_pred = trainer.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    n_range = np.max(y_test) - np.min(y_test)
    nrmse = rmse / n_range if n_range > 0 else rmse
    
    return nrmse

if __name__ == "__main__":
    score = run_narma_benchmark()
    print(f"\nFinal NARMA NRMSE: {score:.4f}")
