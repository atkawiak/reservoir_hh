#!/usr/bin/env python3
"""
STAGE B: TRIPLET FUNCTIONAL EVALUATION
Takes the 3000 networks (1000 triplets) from Stage A.2 and evaluates 
their computational performance (XOR, NARMA, MC).

HPC Optimized: Uses ProcessPoolExecutor with 52 workers.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import load_config
from run_experiment import run_trial

def compute_distances(p_edge, p_neighbor):
    """Compute various distance metrics between edge and neighbor in parameter space."""
    # Parameter ranges for normalization
    ranges = {'gA': 30.0, 'gL': 0.39, 'rho': 9.9, 'bias': 20.0}
    
    # Raw deltas
    deltas = {k: p_neighbor[k] - p_edge[k] for k in ['gA', 'gL', 'rho', 'bias']}
    
    # Normalized deltas
    norm_deltas = {k: deltas[k] / ranges[k] for k in deltas}
    
    # Distance metrics
    euclidean = np.sqrt(sum(d**2 for d in norm_deltas.values()))
    manhattan = sum(abs(d) for d in norm_deltas.values())
    chebyshev = max(abs(d) for d in norm_deltas.values())
    
    return {
        'distance_euclidean': euclidean,
        'distance_manhattan': manhattan,
        'distance_chebyshev': chebyshev,
        'delta_gA': deltas['gA'],
        'delta_gL': deltas['gL'],
        'delta_rho': deltas['rho'],
        'delta_bias': deltas['bias']
    }

def process_triplet_row(row: dict, cfg, N_override: int, triplet_data: dict):
    """Wraps run_trial for a single row in the ensemble CSV."""
    # trial_idx in the ensemble csv is the 'triplet_id'
    # We use (triplet_id * 3 + regime_offset) to ensure different seeds if desired,
    # but run_trial uses seeds based on trial_idx.
    # To keep seeds consistent across a triplet, we use the SAME trial_idx for all 3 members.
    # This is crucial! We want to compare the SAME connectivity matrix with different parameters.
    
    # Map 'regime' for logging
    regime = row['regime']
    
    try:
        results = run_trial(
            N=N_override,
            gL=row['gL'],
            rho=row['rho'],
            bias=row['bias'],
            difficulty=difficulty,
            trial_idx=int(row['seed_id']), # Use consistent seed for all triplet members
            cfg=cfg,
            base_seed=8000 # Different base seed from Stage A (params), but same relative seed inside
        )
        # Add metadata
        for r in results:
            r['triplet_id'] = row['triplet_id']
            r['seed_id'] = row['seed_id']
            r['regime'] = regime
            r['ensemble_lambda'] = row['lambda'] # The lambda measured during generation
            
        return results
    except Exception as e:
        print(f"❌ Error in Row {row['triplet_id']}-{regime}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default="results/triplet_ensemble_params.csv")
    parser.add_argument("--out_parquet", type=str, default="results/triplet_scores.parquet")
    parser.add_argument("--workers", type=int, default=52)
    parser.add_argument("--N", type=int, default=200, help="Reservoir size for evaluation")
    args = parser.parse_args()
    
    if not os.path.exists(args.in_csv):
        print(f"❌ Input {args.in_csv} not found.")
        return

    df = pd.read_csv(args.in_csv)
    cfg = load_config('configs/local_test.yaml')
    
    print(f"🚀 Starting STAGE B: Functional Evaluation (N={args.N})")
    print(f"🔥 Evaluating {len(df)} configurations in {args.workers} workers...")
    
    # 1. HPC Thread Settings
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    start_time = time.time()
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_triplet_row, row.to_dict(), cfg, args.N): i 
                   for i, row in df.iterrows()}
        
        for count, future in enumerate(as_completed(futures)):
            res = future.result()
            all_results.extend(res)
            
            if (count + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (count + 1) / elapsed
                print(f"✅ Processed {count+1}/{len(df)} rows. Speed: {rate:.2f} rows/s. Est. remaining: {(len(df)-(count+1))/rate/60:.1f} min")
                
            # Periodic Save
            if (count + 1) % 100 == 0:
                temp_df = pd.DataFrame(all_results)
                temp_df.to_parquet(args.out_parquet)

    # Final Save
    final_df = pd.DataFrame(all_results)
    final_df.to_parquet(args.out_parquet)
    print(f"✨ ALL DONE. Results saved to {args.out_parquet}")
    print(f"📊 Final rows: {len(final_df)}")

if __name__ == "__main__":
    main()
