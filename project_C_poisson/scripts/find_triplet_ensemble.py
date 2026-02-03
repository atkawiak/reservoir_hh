#!/usr/bin/env python3
"""
STAGE A.2: TRIPLET ENSEMBLE GENERATION (High Speed Search)
"""
import sys
import os
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set project paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import load_config
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

# CONFIGURATION
OUT_CSV = os.path.join(ROOT, 'results/triplet_ensemble_params.csv')
LOCAL_CFG = os.path.join(ROOT, 'configs/local_test.yaml')
MAX_WORKERS = 52

def get_lambda(params: dict, trial_idx: int) -> tuple:
    """Calculates lambda. Returns (lambda, fr)."""
    try:
        cfg = load_config(LOCAL_CFG)
        cfg.hh.N = 100
        cfg.hh.gA = float(params['gA'])
        cfg.hh.gL = float(params['gL'])
        rho = float(params['rho'])
        bias = float(params['bias'])
        
        rng_mgr = RNGManager(5000 + trial_idx)
        trial_gens = rng_mgr.get_trial_generators(trial_idx)
        seeds_tuple = rng_mgr.get_trial_seeds_tuple(trial_idx)
        
        hh = HHModel(cfg, trial_gens, seeds_tuple)
        lyap = LyapunovModule(trial_gens['in'])
        steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
        
        # FAST PROBE MODE
        steps_wu = 1000 # 50ms warmup
        u_wu = trial_gens['in'].uniform(0, 1, steps_wu // steps_per_symbol + 1)
        rates_wu = cfg.task.poisson_rate_min + u_wu * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_wu = (trial_gens['in'].random(steps_wu) < (np.repeat(rates_wu, steps_per_symbol)[:steps_wu] * cfg.task.dt * 1e-3)).astype(np.float32)
        res_wu = hh.simulate(rho, bias, spikes_wu, "WU_S", trim_steps=0, gL=cfg.hh.gL)
        start_state = res_wu['final_state']
        
        # 5000 steps = 250ms. Plenty to see trend.
        len_steps = 5000
        u_l = trial_gens['in'].uniform(0, 1, len_steps // steps_per_symbol + 1)
        rates_l = cfg.task.poisson_rate_min + u_l * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_l = (trial_gens['in'].random(len_steps) < (np.repeat(rates_l, steps_per_symbol)[:len_steps] * cfg.task.dt * 1e-3)).astype(np.float32)
        
        s1 = hh.simulate(rho, bias, spikes_l, "L1_V", trim_steps=100, full_state=start_state, gL=cfg.hh.gL)
        v1 = s1['v_trace']
        
        pert = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in start_state.items()}
        pert['V'][0] += 1e-4 # Larger perturbation for V-base
        s2 = hh.simulate(rho, bias, spikes_l, "L2_V", trim_steps=100, full_state=pert, gL=cfg.hh.gL)
        v2 = s2['v_trace']
        
        # Calculate slope on V traces (no z-score needed for simple divergence)
        diff = np.abs(v1 - v2)
        log_diff = np.log(np.maximum(diff, 1e-12))
        
        # Linear fit on log distance
        t_vals = np.arange(len(log_diff))
        # Use mid-simulation window to avoid initial transient or saturation
        t_start, t_end = 1000, 4000
        slope, _ = np.polyfit(t_vals[t_start:t_end], log_diff[t_start:t_end], 1)
        
        # result is slope per step. convert to s^-1.
        return slope / (cfg.task.dt / 1000.0), s1['mean_rate']
    except Exception as e:
        return None, str(e)

def process_triplet(row_data: dict, seed_id: int, triplet_id: int):
    p_crit = {'gA': row_data['gA'], 'gL': row_data['gL'], 'rho': row_data['rho'], 'bias': row_data['bias']}
    results = [{'triplet_id': triplet_id, 'seed_id': seed_id, 'regime': 'critical', **p_crit, 'lambda': row_data['lambda'], 'fr': row_data['fr']}]
    
    # 1. Gradient Estimation
    eps = {'gA': 2.0, 'gL': 0.05, 'rho': 0.2, 'bias': 1.0}
    grad_list = []
    param_names = ['gA', 'gL', 'rho', 'bias']
    
    for p_name in param_names:
        p_t = p_crit.copy()
        p_t[p_name] += eps[p_name]
        lam, err = get_lambda(p_t, seed_id) # Use seed_id for consistent RNG
        if lam is not None:
            grad_list.append((lam - row_data['lambda']) / eps[p_name])
        else:
            grad_list.append(0.0)
            
    grad_vec = np.array(grad_list)
    norm = np.linalg.norm(grad_vec)
    grad_unit = grad_vec / norm if norm > 1e-7 else np.array([0, -1.0, 0, 0])
    print(f"   [Triplet {triplet_id}] Grad Norm: {norm:.4f}, Unit: {grad_unit}")
        
    # 2. Step and Verify
    for direction, regime, target_sign in [(-1, 'stable', -1), (+1, 'chaotic', 1)]:
        alpha = 20.0 # Aggressive search
        for multiplier in [1.0, 5.0, 20.0]:
            p_n = p_crit.copy()
            for i, name in enumerate(param_names):
                p_n[name] += direction * multiplier * alpha * grad_unit[i]
                if name == 'gL': p_n[name] = np.clip(p_n[name], 0.005, 0.5)
                if name == 'rho': p_n[name] = np.clip(p_n[name], 0.1, 15.0)
                if name == 'gA': p_n[name] = np.max([p_n[name], 0.0])
                if name == 'bias': p_n[name] = np.clip(p_n[name], 0.0, 100.0)
            
            lam_v, fr_v = get_lambda(p_n, seed_id)
            if lam_v is not None:
                 print(f"      [Triplet {triplet_id}] {regime} probe (mult={multiplier}): lambda={lam_v:.4f}")
                 if (target_sign == -1 and lam_v < -0.05) or (target_sign == 1 and lam_v > 0.05):
                    results.append({'triplet_id': triplet_id, 'seed_id': seed_id, 'regime': regime, **p_n, 'lambda': lam_v, 'fr': fr_v})
                    print(f"      ✅ Found {regime} for {triplet_id}!")
                    break 
    return results

def main():
    in_csv = os.path.join(ROOT, 'results/critical_ensemble_params.csv')
    if not os.path.exists(in_csv):
        print(f"❌ Input {in_csv} not found.")
        return
    
    df = pd.read_csv(in_csv)
    # Limit to 1000 triplets if more exist
    df = df.iloc[:1000] 
    
    processed_triplets = set()
    if os.path.exists(OUT_CSV):
        try:
            old_df = pd.read_csv(OUT_CSV)
            counts = old_df.groupby('triplet_id').size()
            processed_triplets = set(counts[counts >= 3].index)
        except: pass
    else:
        # Init CSV with proper header
        with open(OUT_CSV, 'w') as f:
            f.write("triplet_id,seed_id,regime,gA,gL,rho,bias,lambda,fr\n")

    print(f"🚀 Resuming: {len(processed_triplets)} done. Monitoring {in_csv} for new candidates...")
    sys.stdout.flush()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    start_time = time.time()
    generated_so_far = len(processed_triplets)
    
    # Continuous processing loop
    while generated_so_far < 1000:
        try:
            # Re-read DataFrame to get new rows
            try:
                # Use engine='python' for more robust parsing and skip bad lines
                df = pd.read_csv(in_csv, on_bad_lines='skip', engine='python')
                
                # Validation: Check if columns exist
                if 'seed_id' not in df.columns:
                    # Maybe it's the old format? Skip for now.
                    print("⚠️ Warning: 'seed_id' column missing. Waiting...")
                    time.sleep(5)
                    continue
                    
            except pd.errors.EmptyDataError:
                time.sleep(5)
                continue
            except Exception as e:
                print(f"⚠️ CSV Read Error: {e}")
                time.sleep(5)
                continue
            # Filter already processed
            new_candidates = [
                (idx, row) for idx, row in df.iterrows() 
                if idx not in processed_triplets and (row.get('seed_id', idx) is not None)
            ]
            
            if not new_candidates:
                print("   ... Waiting for more critical points ...")
                time.sleep(10)
                continue
                
            print(f"   ► Found {len(new_candidates)} new candidates to process.")
            
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_triplet, row.to_dict(), int(row.get('seed_id', idx)), idx): idx 
                    for idx, row in new_candidates
                }
                
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        res = future.result()
                        if len(res) >= 1:
                            with open(OUT_CSV, 'a') as f:
                                for r in res:
                                    f.write(f"{r['triplet_id']},{r['seed_id']},{r['regime']},{r['gA']:.4f},{r['gL']:.4f},{r['rho']:.4f},{r['bias']:.4f},{r['lambda']:.4f},{r['fr']:.2f}\n")
                                f.flush()
                            processed_triplets.add(idx)
                            generated_so_far += 1
                            print(f"✅ Triplet {idx} completed! Total: {generated_so_far}/1000")
                    except Exception as e:
                        print(f"❌ Error processing triplet {idx}: {e}")
                        
        except Exception as e:
            print(f"⚠️ Main loop error (retrying): {e}")
            time.sleep(5)

    print(f"✅ MISSION ACCOMPLISHED. Generated {generated_so_far} triplets.")
    sys.stdout.flush()

    print("✨ ALL DONE STAGE A.2.")

if __name__ == "__main__":
    main()
