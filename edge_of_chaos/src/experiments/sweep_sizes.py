"""
Master script to sweep through multiple network sizes N.
"""

import os
import argparse
import yaml
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Sweep across multiple N sizes.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 200], help="Sizes to test")
    args = parser.parse_args()

    # 1. Generate all topologies first
    logger.info("Step 1: Generating topologies...")
    gen_cmd = [
        "python3", "-m", "src.experiments.generate_reservoirs",
        "--config", args.config,
        "--sizes"
    ] + [str(s) for s in args.sizes]
    subprocess.run(gen_cmd, check=True)

    # 2. Run sweep for each N
    for N in args.sizes:
        logger.info(f"Step 2: Processing size N={N}...")
        sweep_cmd = [
            "python3", "-m", "src.experiments.sweep_seeds",
            "--N", str(N),
            "--config", args.config
        ]
        subprocess.run(sweep_cmd, check=True)

    # 3. Aggregate results
    logger.info("Step 3: Aggregating statistics...")
    agg_cmd = [
        "python3", "-m", "src.experiments.aggregate_stats",
        "--output_dir", "results"
    ]
    subprocess.run(agg_cmd, check=True)

    logger.info("All experiments completed.")

if __name__ == "__main__":
    main()
