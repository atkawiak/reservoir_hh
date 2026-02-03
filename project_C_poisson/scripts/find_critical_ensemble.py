#!/usr/bin/env python3
"""
CRITICAL ENSEMBLE SEARCH (1000 Candidates)
Searches for 1000 reservoir configurations at the "Edge of Chaos" (lambda approx 0).

Method:
- Random Sampling of parameters (gA, gL, rho, bias)
- Parallel execution (HPC optimized)
- Quick Lyapunov estimation
- Saves valid candidates to 'results/critical_ensemble_params.csv'

Target:
- Lambda in [-0.05, +0.05]
- Bio-plausible firing rate (1 < FR < 100 Hz)
"""

import sys
import os
import numpy as np
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import load_config
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

def evaluate_candidate(cfg_path: str, params: dict, trial_idx: int) -> dict:
    try:
        # Load minimal config
        cfg = load_config(cfg_path)
        cfg.hh.N = 100 # Small probing network
        
        # Apply Candidate Params
        cfg.hh.gA = params['gA']
        cfg.hh.gL = params['gL']
        
        rho = params['rho']
        bias = params['bias']
        
        # Setup
        base_seed = 3000 + trial_idx
        rng_mgr = RNGManager(base_seed)
        trial_gens = rng_mgr.get_trial_generators(trial_idx)
        seeds_tuple = rng_mgr.get_trial_seeds_tuple(trial_idx)
        
        hh = HHModel(cfg, trial_gens, seeds_tuple)
        lyap = LyapunovModule(trial_gens['in'])
        steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
        
        # 1. Warm-up
        steps_wu = 1000
        u_wu = trial_gens['in'].uniform(0, 1, steps_wu // steps_per_symbol + 1)
        rates_wu = cfg.task.poisson_rate_min + u_wu * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_wu = (trial_gens['in'].random(steps_wu) < (np.repeat(rates_wu, steps_per_symbol)[:steps_wu] * cfg.task.dt * 1e-3)).astype(np.float32)
        
        res_wu = hh.simulate(rho, bias, spikes_wu, "WU_probe", trim_steps=0, gL=params['gL'])
        start_state = res_wu['final_state']
        
        # 2. Lyapunov Trace
        len_steps = 10000 # 500ms simulation
        u_l = trial_gens['in'].uniform(0, 1, len_steps // steps_per_symbol + 1)
        rates_l = cfg.task.poisson_rate_min + u_l * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_l = (trial_gens['in'].random(len_steps) < (np.repeat(rates_l, steps_per_symbol)[:len_steps] * cfg.task.dt * 1e-3)).astype(np.float32)
        
        state_l1 = hh.simulate(rho, bias, spikes_l, "L_ref", trim_steps=100, full_state=start_state, gL=params['gL'])
        v1 = state_l1['v_trace']
        
        perturbed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in start_state.items()}
        perturbed['V'][0] += 1e-4
        state_l2 = hh.simulate(rho, bias, spikes_l, "L_pert", trim_steps=100, full_state=perturbed, gL=params['gL'])
        v2 = state_l2['v_trace']
        
        # Fast Slope via V-trace divergence
        diff = np.abs(v1 - v2)
        log_diff = np.log(np.maximum(diff, 1e-12))
        t_vals = np.arange(len(log_diff))
        # Use middle 80% to avoid edge effects
        t_s, t_e = int(0.1*len(log_diff)), int(0.9*len(log_diff))
        slope, _ = np.polyfit(t_vals[t_s:t_e], log_diff[t_s:t_e], 1)
        lambda_sec = slope / (cfg.task.dt / 1000.0)
        
        return {
            'valid': True, 'seed_id': trial_idx,
            'gA': params['gA'], 'gL': params['gL'], 'rho': rho, 'bias': bias,
            'lambda': lambda_sec, 'fr': state_l1['mean_rate']
        }
        
    except Exception as e:
        import traceback
        print(f"DEBUG WORKER ERROR: {e}")
        traceback.print_exc()
        return {'valid': False, 'error': str(e)}

def generate_random_params():
    """Smart sampling from 'Danger Zone' where chaos is likely."""
    return {
        'gA': np.random.uniform(0.0, 30.0),    # Wide range of adaptation
        'gL': np.random.uniform(0.01, 0.4),    # Bias towards low leak (instability)
        'rho': np.random.uniform(0.1, 5.0),    # Recurrence strength
        'bias': np.random.uniform(0.0, 20.0)   # External drive
    }

def main():
    try:
        _main_logic()
    except Exception as e:
        print(f"FATAL ERROR IN MAIN: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

def _main_logic():
    target_count = 1000
    out_csv = os.path.join(ROOT, 'results/critical_ensemble_params.csv')
    
    if not os.path.exists(out_csv):
        with open(out_csv, 'w') as f:
            f.write("seed_id,gA,gL,rho,bias,lambda,fr\n")
    
    max_workers = 52
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    print(f"🔎 STARTING CRITICAL ENSEMBLE SEARCH (Target: {target_count})")
    sys.stdout.flush()
    
    found_count = 0
    total_scanned = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        batch_size = max_workers * 3 # Larger buffer
        tasks = {}
        
        while found_count < target_count:
            # Replenish
            while len(tasks) < batch_size:
                params = generate_random_params()
                cfg_abs = os.path.join(ROOT, 'configs/local_test.yaml')
                future = executor.submit(evaluate_candidate, cfg_abs, params, total_scanned)
                tasks[future] = params
                total_scanned += 1
            
            # Wait for any to complete
            done_futures = as_completed(tasks)
            # Process one by one to keep logging alive
            for future in done_futures:
                res = future.result()
                del tasks[future]
                
                if res['valid']:
                    lam, fr = res['lambda'], res['fr']
                    if -0.05 <= lam <= 0.05 and 1.0 < fr < 100.0:
                        found_count += 1
                        print(f"🔥 FOUND #{found_count}: λ={lam:.4f} | gA={res['gA']:.2f} gL={res['gL']:.3f} ρ={res['rho']:.2f}")
                        sys.stdout.flush()
                        with open(out_csv, 'a') as f:
                            f.write(f"{res['seed_id']},{res['gA']},{res['gL']},{res['rho']},{res['bias']},{lam:.4f},{fr:.2f}\n")
                else:
                    if total_scanned % 100 == 0:
                        print(f"❌ Worker Error: {res.get('error', 'unknown')}")
                        sys.stdout.flush()

                processed = total_scanned - len(tasks)
                if processed % 50 == 0:
                    rate = processed / (time.time() - start_time + 1e-9)
                    print(f"   [Scan: {processed} | Found: {found_count} | Rate: {rate:.1f} trials/s]")
                    sys.stdout.flush()

                if found_count >= target_count: break
                
                # Check if we need more tasks to keep the pipeline full
                if len(tasks) < max_workers:
                    break # Break inner loop to replenish
                
    print(f"✅ MISSION ACCOMPLISHED. Found {found_count} critical reservoirs.")
                
    print(f"✅ MISSION ACCOMPLISHED. Found {found_count} critical reservoirs.")

if __name__ == "__main__":
    main()
