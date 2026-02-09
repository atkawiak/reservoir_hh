import numpy as np
import yaml
from src.model.reservoir import Reservoir
from src.benchmarks.trainer import ReservoirTrainer

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_xor_benchmark(config, n_samples=500):
    """
    XOR benchmark with proper train/test split.
    Task: Predict bit[n] ^ bit[n-1] from Reservoir State at symbol n.
    """
    n_neurons = config['system']['n_neurons']
    dt = config['system']['dt']
    
    # Setup Reservoir
    res = Reservoir(n_neurons=n_neurons, config=config)
    n_input = len(res.input_indices)
    
    # Task Parameters
    symbol_duration = config['benchmarks']['xor']['symbol_duration']
    steps_per_symbol = int(symbol_duration / dt)
    
    # --- WARM-UP PHASE (Washout) ---
    warmup_samples = 100
    for s in range(warmup_samples):
        bit = np.random.randint(0, 2)
        target_rate = 2.0 if bit == 0 else 40.0
        p_spike = target_rate * dt / 1000.0
        for t in range(steps_per_symbol):
            spikes_in = (np.random.rand(n_input) < p_spike).astype(float)
            res.step(spikes_in)

    # --- MAIN SIMULATION ---
    raw_bits = np.random.randint(0, 2, n_samples)
    # Target: Bit[n] XOR Bit[n-1]
    targets = np.bitwise_xor(raw_bits[:-1], raw_bits[1:])
    
    features = []
    for s in range(n_samples):
        # We take the state at the end of the symbol
        bit = raw_bits[s]
        target_rate = 2.0 if bit == 0 else 40.0
        p_spike = target_rate * dt / 1000.0
        
        for t in range(steps_per_symbol):
            spikes_in = (np.random.rand(n_input) < p_spike).astype(float)
            v, spikes, r = res.step(spikes_in)
            
        features.append(r.copy())

    # Features X[n] should depend on u[n] and u[n-1] (memory)
    X = np.array(features)
    # y[n] = XOR(u[n], u[n-1])
    # So we align X[1:] with targets
    X_aligned = X[1:]
    y_aligned = targets
    
    split = int(len(y_aligned) * 0.7)
    X_train, X_test = X_aligned[:split], X_aligned[split:]
    y_train, y_test = y_aligned[:split], y_aligned[split:]
    
    trainer = ReservoirTrainer(alpha=config['benchmarks']['xor']['ridge_alpha'])
    trainer.train(X_train, y_train)
    
    # TEST Accuracy
    y_pred = trainer.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = np.mean(y_pred_binary == y_test)
    return accuracy

def main():
    config = load_config('task_config.yaml')
    acc = run_xor_benchmark(config, n_samples=1000)
    print(f"\nFinal XOR Accuracy (Test): {acc:.2%}")

if __name__ == "__main__":
    main()
