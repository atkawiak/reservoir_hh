
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

def get_clean_lambda(params, seed_id, cfg):
    """Rigorous lambda calculation using filtered traces and Z-scoring."""
    try:
        rng_mgr = RNGManager(seed_id)
        tg = rng_mgr.get_trial_generators(seed_id)
        seeds_tuple = rng_mgr.get_trial_seeds_tuple(seed_id)
        
        # Override params in config object for instantiation
        cfg.hh.gA = float(params['gA'])
        cfg.hh.gL = float(params['gL'])
        rho = float(params['rho'])
        bias = float(params['bias'])
        
        hh = HHModel(cfg, tg, seeds_tuple)
        lyap_mod = LyapunovModule(tg['in'])
        
        dt = cfg.task.dt
        steps_per_symbol = int(cfg.task.symbol_ms / dt)
        
        # 1. Warm-up
        steps_wu = 1000
        spikes_wu = (tg['in'].random(steps_wu) < (50.0 * dt * 1e-3)).astype(float)
        res_wu = hh.simulate(rho, bias, spikes_wu, "WU", trim_steps=0, gL=params['gL'])
        
        # 2. Parallel trajectories
        len_test = 5000
        spikes_test = (tg['in'].random(len_test) < (50.0 * dt * 1e-3)).astype(float)
        
        # Ref
        s1 = hh.simulate(rho, bias, spikes_test, "L1", trim_steps=0, full_state=res_wu['final_state'], gL=params['gL'])
        
        # Perturbed
        pert = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in res_wu['final_state'].items()}
        pert['V'][0] += 1e-5
        s2 = hh.simulate(rho, bias, spikes_test, "L2", trim_steps=0, full_state=pert, gL=params['gL'])
        
        # Process to phi (filtered traces)
        tau = cfg.task.tau_trace
        phi1 = filter_and_downsample(s1['spikes'], steps_per_symbol, dt, tau)
        phi2 = filter_and_downsample(s2['spikes'], steps_per_symbol, dt, tau)
        
        # Calculate slope
        # compute_lambda returns raw slope per step of phi
        slope = lyap_mod.compute_lambda(phi1, phi2, window_range=(20, 100))
        
        # phi is downsampled by 'steps_per_symbol'
        step_ms = steps_per_symbol * dt
        lambda_sec = (slope / step_ms) * 1000.0
        
        return lambda_sec, s1['mean_rate']
    except Exception as e:
        return None, str(e)

def find_perfect_triplet_v2():
    cfg = load_config('configs/production_config.yaml')
    df_crit = pd.read_csv('results/critical_ensemble_params.csv')
    row = df_crit.iloc[0]
    seed_id = int(row['seed_id'])
    
    print(f"🔬 RIGOROUS TRIPLET SEARCH FOR SEED: {seed_id}")
    print("-" * 50)

    # 1. THE EDGE
    p_edge = {'gA': row['gA'], 'gL': row['gL'], 'rho': row['rho'], 'bias': row['bias']}
    lam_e, fr_e = get_clean_lambda(p_edge, seed_id, cfg)
    print(f"📍 EDGE:    lam={lam_e:8.4f}, fr={fr_e:6.1f} Hz")

    # 2. THE STABLE
    p_stable = p_edge.copy()
    p_stable['gA'] = 30.0 # High adaptation
    p_stable['gL'] = 0.4 # High leak
    p_stable['rho'] = 0.5 # Low recurrence
    lam_s, fr_s = get_clean_lambda(p_stable, seed_id, cfg)
    print(f"❄️  STABLE:  lam={lam_s:8.4f}, fr={fr_s:6.1f} Hz")

    # 3. THE CHAOTIC (Tuning gA)
    print("\n🔍 Searching for CHAOS by reducing gA (releasing the brake)...")
    for gA_val in [2.0, 1.0, 0.0]:
        p_c = p_edge.copy()
        p_c['gA'] = gA_val
        p_c['rho'] = 10.0 # Higher recurrence
        p_c['gL'] = 0.05
        p_c['bias'] = 2.0
        
        l, f = get_clean_lambda(p_c, seed_id, cfg)
        print(f"   Probe: gA={gA_val:4.1f}, rho={p_c['rho']:4.1f} -> lam={l:8.4f}, fr={f:6.1f} Hz")
        if l > 0.1 and f < 200:
            print(f"\n🔥 CHAOTIC: lam={l:8.4f}, fr={f:6.1f} Hz | gA={gA_val:.2f}")
            print("-" * 50)
            print("✅ SUCCESS: Found Chaos by reducing A-current!")
            return

    print("\n❌ FAILED to find chaos even without A-current.")

if __name__ == "__main__":
    find_perfect_triplet_v2()
