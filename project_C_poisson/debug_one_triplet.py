
import os
import sys
import numpy as np
import pandas as pd

# Set paths
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import load_config
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

def test_sensitive_regime():
    cfg = load_config('configs/production_config.yaml')
    # Use Seed 46
    df_crit = pd.read_csv('results/critical_ensemble_params.csv')
    row = df_crit.iloc[0]
    seed_id = int(row['seed_id'])
    
    # 10x weaker input, but higher base bias to keep it sensitive
    cfg.hh.in_gain = 0.01 
    cfg.task.poisson_rate_max = 10.0 # 5x lower rate
    
    print(f"🔬 TESTING SENSITIVE REGIME FOR SEED: {seed_id}")
    print(f"Params: InGain={cfg.hh.in_gain}, MaxRate={cfg.task.poisson_rate_max}Hz")
    print("-" * 60)

    # Search for chaos in this quiet but sensitive network
    for gA in [20.0, 5.0, 1.0]:
        for rho in [1.0, 5.0, 10.0, 20.0]:
            p = {'gA': gA, 'gL': 0.05, 'rho': rho, 'bias': 5.0} # Fixed bias for sensitivity
            
            # Simplified clean lambda measure
            rng_mgr = RNGManager(seed_id)
            tg = rng_mgr.get_trial_generators(seed_id)
            hh = HHModel(cfg, tg, rng_mgr.get_trial_seeds_tuple(seed_id))
            lyap_mod = LyapunovModule(tg['in'])
            dt = cfg.task.dt
            
            # Simulate
            spikes_test = (tg['in'].random(6000) < (5.0 * dt * 1e-3)).astype(float)
            s1 = hh.simulate(rho, p['bias'], spikes_test, "L1", trim_steps=500, gL=p['gL'])
            
            pert = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in s1['final_state'].items()}
            pert['V'][0] += 0.01
            s2 = hh.simulate(rho, p['bias'], spikes_test, "L2", trim_steps=500, full_state=pert, gL=p['gL'])
            
            # Check for divergence relative to initial perturbation
            diff = np.abs(s1['v_trace'] - s2['v_trace'])
            log_d = np.log(np.maximum(diff[1000:3000], 1e-12))
            slope, _ = np.polyfit(np.arange(len(log_d)), log_d, 1)
            lam = (slope / dt) * 1000.0
            
            fr = s1['mean_rate']
            status = "OK" if fr < 200 else "SAT"
            print(f"gA={gA:4.1f} | rho={rho:4.1f} -> lam={lam:9.4f}, fr={fr:5.1f} Hz | {status}")
            
            if status == "OK" and lam > 1.0:
                 print(f"   🌟 SUCCESS! Found real chaos: lam={lam:.4f}")
                 return

if __name__ == "__main__":
    test_sensitive_regime()
