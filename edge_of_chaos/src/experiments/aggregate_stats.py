"""
Script to aggregate results and perform statistical analysis.

Implements:
1. Friedman test for paired comparison of three regimes.
2. Wilcoxon signed-rank test for post-hoc analysis.
3. Bootstrap confidence intervals for main metrics.
4. Summary reports and rejection rate analysis.
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
import logging
from scipy import stats
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, [(1-ci)/2 * 100, (1+ci)/2 * 100])

def perform_statistical_tests(df, metric_cols):
    """Run Friedman and Wilcoxon tests for paired regimes."""
    results = {}
    
    # We need seeds that have all 3 regimes
    for metric in metric_cols:
        logger.info(f"Analyzing metric: {metric}")
        pivot = df.pivot(index="seed", columns="regime", values=metric).dropna()
        
        if len(pivot) < 5:
            logger.warning(f"Not enough complete seeds for {metric}")
            continue
            
        # Friedman test (non-parametric paired)
        try:
            f_stat, f_p = stats.friedmanchisquare(
                pivot["stable"], pivot["edge"], pivot["chaos"]
            )
        except ValueError:
            f_stat, f_p = 0.0, 1.0
            
        # Post-hoc Wilcoxon
        w_edge_stable_stat, w_edge_stable_p = stats.wilcoxon(pivot["edge"], pivot["stable"])
        w_edge_chaos_stat, w_edge_chaos_p = stats.wilcoxon(pivot["edge"], pivot["chaos"])
        w_stable_chaos_stat, w_stable_chaos_p = stats.wilcoxon(pivot["stable"], pivot["chaos"])
        
        results[metric] = {
            "n_seeds": len(pivot),
            "friedman_p": f_p,
            "wilcoxon_edge_vs_stable_p": w_edge_stable_p,
            "wilcoxon_edge_vs_chaos_p": w_edge_chaos_p,
            "wilcoxon_stable_vs_chaos_p": w_stable_chaos_p,
            "means": pivot.mean().to_dict(),
            "stds": pivot.std().to_dict()
        }
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Aggregate and analyze experimental results.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with CSV results")
    parser.add_argument("--output_file", type=str, default="final_stats.csv", help="Output statistics file")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "results_N*_seed*.csv"))
    if not files:
        logger.error("No result files found.")
        return

    all_data = pd.concat([pd.read_csv(f) for f in files])
    
    # Save all combined data
    all_data.to_csv(os.path.join(args.results_dir, "all_results_combined.csv"), index=False)
    
    metrics = ["mc", "narma10", "xor_acc"]
    
    # Analysis per N
    final_rows = []
    
    for N in all_data["N"].unique():
        logger.info(f"Processing size N={N}")
        df_n = all_data[all_data["N"] == N]
        
        # Stats
        stats_res = perform_statistical_tests(df_n, metrics)
        
        # Aggregate stats for report
        for regime in ["stable", "edge", "chaos"]:
            df_reg = df_n[df_n["regime"] == regime]
            row = {"N": N, "regime": regime, "n_seeds": len(df_reg)}
            
            for m in metrics:
                vals = df_reg[m].dropna()
                row[f"{m}_mean"] = vals.mean()
                row[f"{m}_std"] = vals.std()
                ci_low, ci_high = bootstrap_ci(vals.values)
                row[f"{m}_ci_low"] = ci_low
                row[f"{m}_ci_high"] = ci_high
                
            # Add p-values to the edge row for convenience
            if regime == "edge":
                for m in metrics:
                    if m in stats_res:
                        row[f"{m}_friedman_p"] = stats_res[m]["friedman_p"]
                        row[f"{m}_vs_stable_p"] = stats_res[m]["wilcoxon_edge_vs_stable_p"]
                        row[f"{m}_vs_chaos_p"] = stats_res[m]["wilcoxon_edge_vs_chaos_p"]
            
            final_rows.append(row)

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(args.output_file, index=False)
    
    # Summary report
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    for N in all_data["N"].unique():
        print(f"\nSize N={N}:")
        n_df = final_df[final_df["N"] == N]
        for m in metrics:
            edge_val = n_df[n_df["regime"] == "edge"][f"{m}_mean"].values[0]
            stable_val = n_df[n_df["regime"] == "stable"][f"{m}_mean"].values[0]
            chaos_val = n_df[n_df["regime"] == "chaos"][f"{m}_mean"].values[0]
            
            p_stable = n_df[n_df["regime"] == "edge"][f"{m}_vs_stable_p"].values[0]
            p_chaos = n_df[n_df["regime"] == "edge"][f"{m}_vs_chaos_p"].values[0]
            
            sig_s = "*" if p_stable < 0.05 else "ns"
            sig_c = "*" if p_chaos < 0.05 else "ns"
            
            print(f"  {m.upper():7}: Stable={stable_val:6.3f} | Edge={edge_val:6.3f} | Chaos={chaos_val:6.3f}")
            print(f"             (Edge vs Stable p={p_stable:6.4f} [{sig_s}], Edge vs Chaos p={p_chaos:6.4f} [{sig_c}])")
    
    # Rejection rate
    total_seeds = len(all_data["seed"].unique())
    completed_seeds = len(all_data.groupby("seed").filter(lambda x: len(x) == 3)["seed"].unique())
    print(f"\nTotal seeds: {total_seeds}")
    print(f"Completed seeds (all 3 regimes): {completed_seeds}")
    print(f"Success rate: {completed_seeds/total_seeds:.1%}")

if __name__ == "__main__":
    main()
