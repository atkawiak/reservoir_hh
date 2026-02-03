#!/usr/bin/env python3
"""
CHAOS SEARCH PROTOCOL (Phase 1)
Aggressively searches for a chaotic regime by neutralizing stability mechanisms.

Strategy:
1.  Iterate over "Destabilization Grids":
    - gA (A-current): [20.0, 10.0, 5.0, 0.0] -> 0.0 removes adaptation
    - gL (Leak): [0.3, 0.1, 0.05, 0.01] -> 0.01 removes forgetting
    - rho (Spectral Radius): [1.0, 2.0, 5.0, 10.0, 20.0] -> High rho drives chaos
    - bias (External Current): [5.0, 10.0, 20.0] -> Drives spiking

2.  Compute Lambda (Univariate Lyapunov):
    - Uses short simulation with perturbation to estimate LE.

3.  Filter & Report:
    - Target: lambda_sec > 0.01 (Chaos) AND Firing Rate < 200 Hz (Not Saturation)
"""

import sys
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import load_config, ExperimentConfig
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

def run_chaos_probe(cfg_path: str, gA: float, gL: float, rho: float, bias: float, trial_idx: int) -> dict:
    """Runs a quick probe to estimate stability."""
    try:
        # Load minimal config
        cfg = load_config(cfg_path)
        
        # Override parameters
        cfg.hh.N = 100
        cfg.hh.gA = gA
        cfg.hh.gL = gL
        # Ensure we use new weights for each trial to avoid structural bias
        base_seed = 2025 + trial_idx
        
        # Setup
        rng_mgr = RNGManager(base_seed)
        trial_gens = rng_mgr.get_trial_generators(trial_idx)
        seeds_tuple = rng_mgr.get_trial_seeds_tuple(trial_idx)
        
        hh = HHModel(cfg, trial_gens, seeds_tuple)
        lyap = LyapunovModule(trial_gens['in'])
        
        steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
        
        # 1. Warm-up (Short but sufficient)
        steps_wu = 1000
        u_wu = trial_gens['in'].uniform(0, 1, steps_wu // steps_per_symbol + 1)
        rates_wu = cfg.task.poisson_rate_min + u_wu * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_wu = (trial_gens['in'].random(steps_wu) < (np.repeat(rates_wu, steps_per_symbol)[:steps_wu] * cfg.task.dt * 1e-3)).astype(np.float32)
        
        res_wu = hh.simulate(rho, bias, spikes_wu, "WU", trim_steps=0, gL=gL)
        start_state = res_wu['final_state']
        
        # 2. Lyapunov Trace
        len_lyap = 500 # Short trace for speed
        u_l = trial_gens['in'].uniform(0, 1, len_lyap)
        rates_l = cfg.task.poisson_rate_min + u_l * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_l = (trial_gens['in'].random(len_lyap * steps_per_symbol) < (np.repeat(rates_l, steps_per_symbol) * cfg.task.dt * 1e-3)).astype(np.float32)
        
        # Ref trajectory
        state_l1 = hh.simulate(rho, bias, spikes_l, "L_ref", trim_steps=50, full_state=start_state, gL=gL)
        phi1 = filter_and_downsample(state_l1['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        # Perturbed trajectory
        perturbed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in start_state.items()}
        perturbed['V'][0] += 1e-6
        
        state_l2 = hh.simulate(rho, bias, spikes_l, "L_pert", trim_steps=50, full_state=perturbed, gL=gL)
        phi2 = filter_and_downsample(state_l2['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        # Compute Lambda
        step_s = (steps_per_symbol * cfg.task.dt) / 1000.0
        slope = lyap.compute_lambda(phi1, phi2, window_range=[20, 100]) # Quick slope
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
    print("=" * 80)
    print("  CHAOS SEARCH PROBE v1.0")
    print("  Objective: Find params where λ > 0.02 and FR < 200 Hz")
    print("=" * 80)
    
    # PARAMETER GRID (Aggressive Destabilization)
    gA_grid = [20.0, 10.0, 5.0, 0.0]    # Reducing Adaptation
    gL_grid = [0.3, 0.1, 0.05, 0.01]    # Reducing Leak
    rho_grid = [1.0, 2.0, 5.0, 10.0, 20.0]  # Driving force
    bias_grid = [5.0, 10.0, 20.0] # Excitation
    
    tasks = []
    trial_idx = 0
    for gA in gA_grid:
        for gL in gL_grid:
            for rho in rho_grid:
                for bias in bias_grid:
                    tasks.append((gA, gL, rho, bias, trial_idx))
                    trial_idx += 1
    
    print(f"Total Probes: {len(tasks)}")
    
    results = []
    start_time = time.time()
    
    # Parallel Execution
    max_workers = min(60, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_chaos_probe, 'configs/local_test.yaml', *t): t for t in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if 'error' not in res:
                results.append(res)
                
                # Live Reporting for interesting hits
                regime = "STABLE"
                if res['lambda'] > 0.01: regime = "CHAOS"
                if res['fr'] > 200: regime = "SATURATION"
                
                if regime == "CHAOS":
                    print(f"🔥 CHAOS FOUND! gA={res['gA']} gL={res['gL']} ρ={res['rho']} bias={res['bias']} -> λ={res['lambda']:.4f}, FR={res['fr']:.1f}")
            else:
                print(f"ERROR: {res['error']}")
                
            if i % 50 == 0:
                print(f"Progress: {i}/{len(tasks)}...")

    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    out_path = f"results/chaos_search_{int(time.time())}.parquet"
    df.to_parquet(out_path)
    print(f"\nSaved results to {out_path}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("  CHAOS REPORT CARD")
    print("=" * 80)
    
    chaos = df[(df['lambda'] > 0.01) & (df['fr'] < 200)]
    print(f"CHAOTIC REGIMES FOUND: {len(chaos)}")
    
    if len(chaos) > 0:
        print("\nTop Chaotic Configurations (Maximize λ):")
        print(chaos.sort_values('lambda', ascending=False).head(10)[['gA', 'gL', 'rho', 'bias', 'lambda', 'fr']])
        
        print("\nOptimal Destruction Parameters (Most frequent in chaotic set):")
        print(f"Best gA: {chaos['gA'].mode()[0]}")
        print(f"Best gL: {chaos['gL'].mode()[0]}")
    else:
        print("❌ No valid chaotic regimes found. Increase destruction!")

if __name__ == "__main__":
    main()
