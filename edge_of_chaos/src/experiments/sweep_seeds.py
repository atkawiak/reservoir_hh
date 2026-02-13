"""
Script to sweep through all seeds for a given network size N.
"""

import os
import argparse
import yaml
import subprocess
import logging
import glob
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Sweep seeds for a given N.")
    parser.add_argument("--N", type=int, required=True, help="Network size")
    parser.add_argument("--frozen_dir", type=str, default="frozen", help="Dir with frozen reservoirs")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--output_dir", type=str, default="results", help="Output dir")
    args = parser.parse_args()

    n_dir = os.path.join(args.frozen_dir, f"N_{args.N}")
    if not os.path.exists(n_dir):
        logger.error(f"Directory {n_dir} does not exist. Run generate_reservoirs first.")
        return

    seed_dirs = sorted(glob.glob(os.path.join(n_dir, "seed_*")))
    logger.info(f"Found {len(seed_dirs)} seeds for N={args.N}")

    for s_dir in seed_dirs:
        seed = os.path.basename(s_dir).split("_")[-1]
        logger.info(f"Processing seed {seed}...")
        
        cmd = [
            "python3", "-m", "src.experiments.run_one_seed",
            "--reservoir_path", s_dir,
            "--config", args.config,
            "--output_dir", args.output_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running seed {seed}: {e}")

if __name__ == "__main__":
    main()
