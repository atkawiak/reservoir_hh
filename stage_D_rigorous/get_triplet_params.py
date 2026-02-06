import numpy as np
import yaml
import pandas as pd
from scipy.optimize import brentq
from src.model.reservoir import Reservoir
from src.utils.metrics import calculate_lyapunov

def load_config():
    with open('task_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def find_lambda_zero(inh_val, config, n_neurons, seed):
    """Mierzy lambda dla zadanej inhibicji - cel dla algorytmu brentq"""
    np.random.seed(seed)
    ctx_config = config.copy()
    ctx_config['synapse']['inh_scaling'] = float(inh_val)
    
    res = Reservoir(n_neurons=n_neurons, config=ctx_config)
    res.normalize_spectral_radius(0.95)
    # Krótszy krok dla szybkości generowania
    return calculate_lyapunov(res, n_steps=1200)

def generate_triplets(n_seeds=5):
    config = load_config()
    n_neurons = config['system']['n_neurons']
    results = []

    print(f"🎯 Rozpoczynam generowanie tripletów dla {n_seeds} ziaren...")

    for seed in [101, 202, 303, 404, 505][:n_seeds]:
        print(f"--- Seed {seed} ---")
        try:
            # 1. Znajdź punkt krytyczny (Edge)
            print("Searching for Edge of Chaos (lambda=0)...")
            critical_inh = brentq(find_lambda_zero, 1.5, 5.0, args=(config, n_neurons, seed), xtol=0.1)
            
            # 2. Zdefiniuj triplet
            triplet = {
                'seed': seed,
                'stable_inh': round(critical_inh + 1.0, 3),
                'edge_inh': round(critical_inh, 3),
                'chaotic_inh': round(critical_inh - 1.0, 3)
            }
            results.append(triplet)
            print(f"✅ Sukces: Edge at {triplet['edge_inh']}")
            
        except Exception as e:
            print(f"❌ Błąd dla seed {seed}: {e}")

    # Zapisz do CSV
    df = pd.DataFrame(results)
    df.to_csv('generated_triplets.csv', index=False)
    print(f"\n✨ Gotowe! Parametry tripletów zapisane w 'generated_triplets.csv'")

if __name__ == "__main__":
    generate_triplets(n_seeds=3)
