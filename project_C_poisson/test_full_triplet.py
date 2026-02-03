import os
import sys
import numpy as np
import pandas as pd

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import load_config
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from tasks.xor import DelayedXOR
from readout import ReadoutModule
from utils import filter_and_downsample

def run_full_triplet_test():
    cfg = load_config('configs/production_config.yaml')
    
    # === SENSITIVE REGIME SETTINGS ===
    # Lower input, higher bias - network is "ready to fire"
    cfg.hh.in_gain = 0.05  # 10x weaker than before
    cfg.task.poisson_rate_min = 5.0
    cfg.task.poisson_rate_max = 15.0  # Much calmer input
    
    seed_id = 46
    rng_mgr = RNGManager(seed_id)
    tg = rng_mgr.get_trial_generators(seed_id)
    
    print("=" * 60)
    print("FULL TRIPLET TEST: Edge of Chaos + XOR Classification")
    print("=" * 60)
    
    # Define triplet configurations
    # Key insight: gA controls stability (high=stable), rho controls coupling
    triplet_configs = {
        'STABLE':   {'gA': 25.0, 'gL': 0.3, 'rho': 1.0,  'bias': 8.0},
        'EDGE':     {'gA': 10.0, 'gL': 0.1, 'rho': 5.0,  'bias': 8.0},
        'CHAOTIC':  {'gA': 2.0,  'gL': 0.05, 'rho': 8.0, 'bias': 8.0},
    }
    
    results = []
    
    for regime, params in triplet_configs.items():
        print(f"\n--- Testing {regime} ---")
        cfg.hh.gA = params['gA']
        cfg.hh.gL = params['gL']
        
        hh = HHModel(cfg, tg, rng_mgr.get_trial_seeds_tuple(seed_id))
        
        # === 1. MEASURE LYAPUNOV ===
        dt = cfg.task.dt
        steps_per_symbol = int(cfg.task.symbol_ms / dt)
        
        # Weak probe signal for Lyapunov
        len_lyap = 8000
        spikes_lyap = (tg['in'].random(len_lyap) < (10.0 * dt * 1e-3)).astype(float)
        
        s1 = hh.simulate(params['rho'], params['bias'], spikes_lyap, "L1", trim_steps=500, gL=params['gL'])
        
        pert = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in s1['final_state'].items()}
        pert['V'][0] += 0.01
        s2 = hh.simulate(params['rho'], params['bias'], spikes_lyap, "L2", trim_steps=500, full_state=pert, gL=params['gL'])
        
        # Calculate lambda from voltage traces
        diff = np.abs(s1['v_trace'] - s2['v_trace'])
        log_d = np.log(np.maximum(diff[500:2500], 1e-12))
        slope, _ = np.polyfit(np.arange(len(log_d)), log_d, 1)
        lam = (slope / dt) * 1000.0
        fr = s1['mean_rate']
        
        print(f"   Lambda: {lam:.4f} | FR: {fr:.1f} Hz")
        
        # === 2. XOR CLASSIFICATION ===
        xor_task = DelayedXOR(tg['in'])
        readout = ReadoutModule(tg['readout'], cv_folds=3, cv_gap=5)
        
        length = 1000
        u, y = xor_task.generate_data(length, delay=3)
        
        # Convert to Poisson spikes (same weak input regime)
        rates = cfg.task.poisson_rate_min + u * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_xor = (tg['in'].random(length * steps_per_symbol) < 
                      (np.repeat(rates, steps_per_symbol) * dt * 1e-3)).astype(float)
        
        # Simulate
        res_xor = hh.simulate(params['rho'], params['bias'], spikes_xor, "XOR", trim_steps=500, gL=params['gL'])
        
        if res_xor['mean_rate'] < 0.5:
            print(f"   XOR: FAILED (no spikes)")
            acc = 0.0
        else:
            # Process states
            phi = filter_and_downsample(res_xor['spikes'], steps_per_symbol, dt, cfg.task.tau_trace)
            
            # Align targets
            y_aligned = y[-len(phi):]
            phi_aligned = phi[-len(y_aligned):]
            
            # Train classifier
            metrics = readout.train_ridge_cv(phi_aligned, y_aligned, task_type='classification')
            acc = metrics.get('accuracy', 0.0)
            print(f"   XOR Accuracy: {acc:.2%}")
        
        results.append({'regime': regime, 'lambda': lam, 'fr': fr, 'xor_acc': acc, **params})
    
    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Check hypothesis
    edge_acc = df[df['regime'] == 'EDGE']['xor_acc'].values[0]
    stable_acc = df[df['regime'] == 'STABLE']['xor_acc'].values[0]
    chaotic_acc = df[df['regime'] == 'CHAOTIC']['xor_acc'].values[0]
    
    if edge_acc > stable_acc and edge_acc > chaotic_acc:
        print("\n✅ HYPOTHESIS CONFIRMED: Edge of Chaos gives best XOR accuracy!")
    else:
        print("\n⚠️ HYPOTHESIS NOT CONFIRMED in this test.")

if __name__ == "__main__":
    run_full_triplet_test()
