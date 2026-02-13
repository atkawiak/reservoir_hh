"""
Script to run simulation for one seed across three regimes (stable, edge, chaos).

Uses a frozen topology and performs:
1. Edge-of-chaos calibration
2. Identification of stable and chaotic points
3. Benchmark execution for each regime
4. Results logging
"""

import os
import argparse
import yaml
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from ..reservoir.build_reservoir import Reservoir
from ..reservoir.calibration import calibrate_edge_of_chaos
from ..benchmarks.mc import run_mc_benchmark
from ..benchmarks.narma10 import run_narma10_benchmark
from ..benchmarks.delayed_xor import run_delayed_xor_benchmark
from ..reservoir.plausibility import run_plausibility_checks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_benchmarks(res, cfg, inh_scaling, regime_name, seed_input):
    """Run all benchmarks for a given regime."""
    logger.info(f"  Running benchmarks for regime: {regime_name} (scaling={inh_scaling:.3f})")
    
    # Reliability: check plausibility again for the specific scaling
    plaus = run_plausibility_checks(res, cfg, inh_scaling, seed_input)
    
    # MC
    mc_res = run_mc_benchmark(res, cfg, inh_scaling, seed_input)
    
    # NARMA-10
    narma_res = run_narma10_benchmark(res, cfg, inh_scaling, seed_input)
    
    # Delayed XOR
    xor_res = run_delayed_xor_benchmark(res, cfg, inh_scaling, seed_input)
    
    return {
        "regime": regime_name,
        "inh_scaling": inh_scaling,
        "plausibility_ok": plaus.is_plausible,
        "mc": mc_res.get("mc_total", 0.0),
        "narma10": narma_res.get("nrmse", 999.0),
        "xor_acc": xor_res.get("accuracy", 0.0),
        "firing_rate": plaus.firing_stats.mean_rate_hz,
        "synchrony": plaus.synchrony_stats.synchrony_index,
        "cv_isi": plaus.synchrony_stats.mean_cv_isi,
        "aborted": mc_res.get("aborted", False) or narma_res.get("aborted", False) or xor_res.get("aborted", False)
    }

def main():
    parser = argparse.ArgumentParser(description="Run one seed across 3 regimes.")
    parser.add_argument("--reservoir_path", type=str, required=True, help="Path to frozen reservoir")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load reservoir
    res = Reservoir.load(args.reservoir_path)
    N = res.N
    seed = res.seed
    
    logger.info(f"Starting Seed {seed}, N={N}")
    
    seed_input = cfg.get("experiment", {}).get("seed_input", 42)
    seed_lyap = seed  # use topology seed for lyapunov perturbation too
    
    # 1. Calibrate edge of chaos
    cal_res = calibrate_edge_of_chaos(res, cfg, seed_input, seed_lyap)
    
    regimes = {
        "stable": (cal_res.inh_scaling_stable, cal_res.lambda_stable),
        "edge": (cal_res.inh_scaling_edge, cal_res.lambda_edge),
        "chaos": (cal_res.inh_scaling_chaos, cal_res.lambda_chaos)
    }
    
    results_list = []
    
    # 2. Run benchmarks for each regime
    for name, (scaling, lam) in regimes.items():
        if np.isnan(scaling):
            logger.warning(f"Regime {name} unavailable for seed {seed}")
            continue
            
        res_bench = run_benchmarks(res, cfg, scaling, name, seed_input)
        res_bench["seed"] = seed
        res_bench["N"] = N
        res_bench["lambda"] = lam
        res_bench["calibration_method"] = cal_res.method_used
        results_list.append(res_bench)
    
    # Save results
    df = pd.DataFrame(results_list)
    out_file = os.path.join(args.output_dir, f"results_N{N}_seed{seed}.csv")
    df.to_csv(out_file, index=False)
    
    # Save full calibration metadata for debugging
    cal_file = os.path.join(args.output_dir, f"cal_N{N}_seed{seed}.json")
    with open(cal_file, "w") as f:
        # Convert scan points to serializable format
        points = []
        for p in cal_res.scan_points:
            points.append({
                "inh": p.inh_scaling,
                "lambda": p.lambda_est,
                "plaus": p.plausibility_ok
            })
        json.dump({
            "seed": seed,
            "N": N,
            "edge_scaling": cal_res.inh_scaling_edge,
            "edge_lambda": cal_res.lambda_edge,
            "method": cal_res.method_used,
            "points": points
        }, f, indent=2)

    logger.info(f"Finished Seed {seed}, N={N}. Results saved to {out_file}")

if __name__ == "__main__":
    main()
