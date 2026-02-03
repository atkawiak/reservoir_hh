#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT, 'scripts')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from find_triplet_ensemble import get_lambda

CRIT_CSV = os.path.join(ROOT, 'results/critical_ensemble_params.csv')
TRIP_CSV = os.path.join(ROOT, 'results/triplet_ensemble_params.csv')
MAX_WORKERS = 52

def find_chaos_aggressive(triplet_id, seed_id, p_crit):
    """Try several 'chaotic' guesses if gradient failed."""
    # Guesses: [gA_mult, gL_mult, rho_mult, bias_mult]
    # We want LESS leak, MORE recurrence, LESS adaptation
    guesses = [
        {'gA': p_crit['gA']*0.5, 'gL': p_crit['gL']*0.5, 'rho': p_crit['rho']*2.0, 'bias': p_crit['bias']},
        {'gA': 0.0, 'gL': 0.02, 'rho': 4.0, 'bias': p_crit['bias']},
        {'gA': p_crit['gA'], 'gL': 0.01, 'rho': 5.0, 'bias': p_crit['bias']},
        {'gA': 0.0, 'gL': p_crit['gL'], 'rho': 3.0, 'bias': 15.0},
    ]
    
    for p_guess in guesses:
        # Clip to safe ranges
        p_guess['gA'] = np.max([p_guess['gA'], 0.0])
        p_guess['gL'] = np.clip(p_guess['gL'], 0.005, 0.4)
        p_guess['rho'] = np.clip(p_guess['rho'], 0.1, 10.0)
        
        lam, fr = get_lambda(p_guess, seed_id)
        if lam is not None and lam > 0.1 and 1.0 < fr < 150.0:
            return {'triplet_id': triplet_id, 'seed_id': seed_id, 'regime': 'chaotic', **p_guess, 'lambda': lam, 'fr': fr}
    
    return None

def main():
    if not os.path.exists(CRIT_CSV) or not os.path.exists(TRIP_CSV):
        print("Missing input files.")
        return

    df_crit = pd.read_csv(CRIT_CSV)
    df_trip = pd.read_csv(TRIP_CSV)
    
    # Find IDs missing chaos
    has_chaos = set(df_trip[df_trip['regime'] == 'chaotic']['triplet_id'])
    missing_ids = [i for i in range(len(df_crit)) if i not in has_chaos]
    
    print(f"🔥 CHAOS BOOST: Retrying {len(missing_ids)} missing triplets...")
    
    found_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx in missing_ids:
            row = df_crit.iloc[idx].to_dict()
            f = executor.submit(find_chaos_aggressive, idx, int(row.get('seed_id', idx)), row)
            futures[f] = idx
            
        for future in as_completed(futures):
            res = future.result()
            if res:
                with open(TRIP_CSV, 'a') as f:
                    f.write(f"{res['triplet_id']},{res['seed_id']},{res['regime']},{res['gA']:.4f},{res['gL']:.4f},{res['rho']:.4f},{res['bias']:.4f},{res['lambda']:.4f},{res['fr']:.2f}\n")
                found_count += 1
                if found_count % 10 == 0:
                    print(f"✅ Found {found_count} new chaotic neighbors...")

    print(f"✨ DONE. Found {found_count} additional chaotic networks.")

if __name__ == "__main__":
    main()
