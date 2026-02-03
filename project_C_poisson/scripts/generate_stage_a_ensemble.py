#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = '/home/retrai/reservoir_hh'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from scripts.find_triplet_ensemble import get_lambda
from config import load_config
from hh_model import HHModel
from rng_manager import RNGManager
from utils import filter_and_downsample
from tasks.mc import MemoryCapacity
from readout import ReadoutModule

CRIT_CSV = os.path.join(ROOT, 'results/critical_ensemble_params.csv')
OUT_PARQUET = os.path.join(ROOT, 'results/ensemble_stage_a.parquet')
MAX_WORKERS = 52

def measure_intrinsic_metrics(params, seed_id, cfg):
    """Measures KR and MC for a given configuration using a dedicated simulation."""
    try:
        rng_mgr = RNGManager(8000 + seed_id) # Consistent seed for metrics
        trial_gens = rng_mgr.get_trial_generators(seed_id)
        seeds_tuple = rng_mgr.get_trial_seeds_tuple(seed_id)
        
        hh = HHModel(cfg, trial_gens, seeds_tuple)
        mc_task = MemoryCapacity(trial_gens['in'])
        readout = ReadoutModule(trial_gens['readout'], cv_folds=3, cv_gap=5)
        
        # 1. Generate Uniform Noise Input
        length = 2000
        u = mc_task.generate_data(length)
        steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
        rates = cfg.task.poisson_rate_min + u * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        rates_up = np.repeat(rates, steps_per_symbol)
        spikes_in = (trial_gens['in'].random(len(rates_up)) < (rates_up * cfg.task.dt * 1e-3)).astype(float)
        
        # 2. Simulate
        res = hh.simulate(params['rho'], params['bias'], spikes_in, "METRICS", gL=params['gL'])
        if res['mean_rate'] == 0: return 0, 0
        
        # 3. Process States
        phi = filter_and_downsample(res['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        # 4. Calculate Kernel Rank (Rank of state matrix)
        phi_slice = phi[-1000:] 
        kr = np.linalg.matrix_rank(phi_slice, tol=1e-4)
        
        # 5. Calculate Memory Capacity
        u_tgt = u[-len(phi):]
        mc_res = mc_task.run_mc_analysis(phi, u_tgt, readout, max_lag=40)
        
        return float(mc_res['mc']), int(kr)
    except Exception as e:
        print(f"Error measuring metrics for {seed_id}: {e}")
        return 0, 0

def bifurcation_tracker(p_edge, seed_id, cfg, target='chaotic'):
    """Finds a neighbor by tracking bifurcation along rho/gA/gL."""
    p = p_edge.copy()
    if target == 'chaotic':
        rho_low, rho_high = p_edge['rho'], 15.0
        p['gA'] = p_edge['gA'] * 0.2
        p['gL'] = np.clip(p_edge['gL'] * 0.5, 0.005, 0.4)
    else: # stable
        rho_low, rho_high = 0.5, p_edge['rho']
        p['gL'] = np.clip(p_edge['gL'] * 2.0, 0.005, 0.4)

    for _ in range(6): # Fast binary search
        p['rho'] = (rho_low + rho_high) / 2
        lam, fr = get_lambda(p, seed_id)
        if lam is None: break
        
        if target == 'chaotic':
            if 0.08 < lam < 0.5 and 1 < fr < 150: return p, lam, fr
            elif lam > 0.5: rho_high = p['rho']
            else: rho_low = p['rho']
        else: # stable
            if -0.4 < lam < -0.08 and 1 < fr < 150: return p, lam, fr
            elif lam < -0.4: rho_low = p['rho']
            else: rho_high = p['rho']
            
    return None, None, None

def process_full_triplet(idx, row, cfg):
    seed_id = int(row.get('seed_id', idx))
    p_edge = {'gA': row['gA'], 'gL': row['gL'], 'rho': row['rho'], 'bias': row['bias']}
    results = []
    
    R = {'rho': 15.0, 'gL': 0.4, 'gA': 30.0, 'bias': 25.0}

    variants = [('critical', p_edge, row['lambda'], row['fr'])]
    
    p_s, lam_s, fr_s = bifurcation_tracker(p_edge, seed_id, cfg, 'stable')
    if p_s: variants.append(('stable', p_s, lam_s, fr_s))
    
    p_c, lam_c, fr_c = bifurcation_tracker(p_edge, seed_id, cfg, 'chaotic')
    if p_c: variants.append(('chaotic', p_c, lam_c, fr_c))
    
    for regime, p, lam, fr in variants:
        mc, kr = measure_intrinsic_metrics(p, seed_id, cfg)
        dist_4d = np.sqrt(sum(((p[k]-p_edge[k])/R[k])**2 for k in p))
        results.append({
            'triplet_id': idx, 'seed_id': seed_id, 'regime': regime,
            **p, 'lambda': lam, 'fr': fr,
            'mc_intrinsic': mc, 'kernel_rank': kr,
            'dist_lam': abs(lam - row['lambda']),
            'dist_4d': dist_4d
        })
    return results

from tqdm import tqdm

def main():
    cfg = load_config('configs/production_config.yaml')
    df_crit = pd.read_csv(CRIT_CSV)
    print(f"🚀 Stage A Production: 1000 Triplets | 52 Workers")
    
    final_data = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_full_triplet, i, row, cfg): i for i, row in df_crit.head(1000).iterrows()}
        
        # tqdm progress bar
        with tqdm(total=len(futures), desc="Generating Triplets", unit="triplet") as pbar:
            for i, f in enumerate(as_completed(futures)):
                try:
                    res = f.result()
                    final_data.extend(res)
                except Exception as e:
                    print(f"\n❌ Error in triplet {i}: {e}")
                
                pbar.update(1)
                
                # Periodic save and log progress to file for user
                if (i+1) % 10 == 0:
                    pd.DataFrame(final_data).to_parquet(OUT_PARQUET)
                    with open(os.path.join(ROOT, 'results/stage_a_progress.log'), 'a') as logf:
                        logf.write(f"{time.strftime('%H:%M:%S')} - Progress: {i+1}/1000 triplets ({pbar.format_dict['rate']:.2f} trip/s)\n")
    
    pd.DataFrame(final_data).to_parquet(OUT_PARQUET)
    print(f"✨ ALL DONE. Result saved to {OUT_PARQUET}")

if __name__ == "__main__":
    main()
