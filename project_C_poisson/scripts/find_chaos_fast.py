#!/usr/bin/env python3
"""
CHAOS SEARCH FAST (Phase 1)
Aggressively searches for a chaotic regime. Writes LIVE results to CSV.

Strategy:
1.  Iterate over "Destabilization Grids":
    - gA (A-current): [10.0, 5.0, 0.0] 
    - gL (Leak): [0.05, 0.01] 
    - rho (Spectral Radius): [1.0, 5.0, 10.0]
    - bias (External Current): [5.0, 10.0]

2.  Compute Lambda & Write to CSV immediately.
"""

import sys
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import load_config, ExperimentConfig
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

def run_chaos_probe(cfg_path: str, gA: float, gL: float, rho: float, bias: float, trial_idx: int) -> dict:
    try:
        cfg = load_config(cfg_path)
        cfg.hh.N = 100
        cfg.hh.gA = gA
        cfg.hh.gL = gL
        base_seed = 2025 + trial_idx
        
        rng_mgr = RNGManager(base_seed)
        trial_gens = rng_mgr.get_trial_generators(trial_idx)
        seeds_tuple = rng_mgr.get_trial_seeds_tuple(trial_idx)
        
        hh = HHModel(cfg, trial_gens, seeds_tuple)
        lyap = LyapunovModule(trial_gens['in'])
        
        steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
        
        # Warm-up 
        steps_wu = 1000
        u_wu = trial_gens['in'].uniform(0, 1, steps_wu // steps_per_symbol + 1)
        rates_wu = cfg.task.poisson_rate_min + u_wu * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_wu = (trial_gens['in'].random(steps_wu) < (np.repeat(rates_wu, steps_per_symbol)[:steps_wu] * cfg.task.dt * 1e-3)).astype(np.float32)
        
        res_wu = hh.simulate(rho, bias, spikes_wu, "WU", trim_steps=0, gL=gL)
        start_state = res_wu['final_state']
        
        # Lyapunov Trace (Short)
        len_lyap = 500 
        u_l = trial_gens['in'].uniform(0, 1, len_lyap)
        rates_l = cfg.task.poisson_rate_min + u_l * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_l = (trial_gens['in'].random(len_lyap * steps_per_symbol) < (np.repeat(rates_l, steps_per_symbol) * cfg.task.dt * 1e-3)).astype(np.float32)
        
        state_l1 = hh.simulate(rho, bias, spikes_l, "L_ref", trim_steps=50, full_state=start_state, gL=gL)
        phi1 = filter_and_downsample(state_l1['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        perturbed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in start_state.items()}
        perturbed['V'][0] += 1e-6
        
        state_l2 = hh.simulate(rho, bias, spikes_l, "L_pert", trim_steps=50, full_state=perturbed, gL=gL)
        phi2 = filter_and_downsample(state_l2['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        step_s = (steps_per_symbol * cfg.task.dt) / 1000.0
        slope = lyap.compute_lambda(phi1, phi2, window_range=[20, 100])
        lambda_sec = slope / step_s
        
        return {
            'gA': gA, 'gL': gL, 'rho': rho, 'bias': bias,
            'lambda': lambda_sec,
            'fr': state_l1['mean_rate'],
            'sat': state_l1['saturation_flag']
        }
    except Exception as e:
        return {'error': str(e), 'gA': gA, 'gL': gL, 'rho': rho, 'bias': bias}

def main():
    print("MATCHING CHAOS - FAST LIVE MODE")
    # Reduced Grid
    gA_grid = [10.0, 5.0, 0.0]
    gL_grid = [0.05, 0.01]
    rho_grid = [1.0, 5.0, 10.0]
    bias_grid = [5.0, 10.0]
    
    tasks = []
    trial_idx = 0
    for gA in gA_grid:
        for gL in gL_grid:
            for rho in rho_grid:
                for bias in bias_grid:
                    tasks.append((gA, gL, rho, bias, trial_idx))
                    trial_idx += 1
    
    print(f"Total Probes: {len(tasks)}")
    
    out_csv = 'results/live_chaos.csv'
    with open(out_csv, 'w') as f:
        f.write("gA,gL,rho,bias,lambda,fr,sat\n")

    max_workers = min(60, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_chaos_probe, 'configs/local_test.yaml', *t): t for t in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if 'error' not in res:
                with open(out_csv, 'a') as f:
                    f.write(f"{res['gA']},{res['gL']},{res['rho']},{res['bias']},{res['lambda']:.4f},{res['fr']:.2f},{res['sat']}\n")
                
                if res['lambda'] > 0.01 and res['fr'] < 200:
                     print(f"🔥 CHAOS FOUND! gA={res['gA']} gL={res['gL']} ρ={res['rho']} -> λ={res['lambda']:.4f}")
            
    print("Done.")

if __name__ == "__main__":
    main()
