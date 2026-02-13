"""
Script to generate and freeze topology for multiple seeds and network sizes.

Ensures that we have exactly the same base networks for comparison across regimes.
"""

import os
import argparse
import yaml
import logging
from ..reservoir.build_reservoir import build_reservoir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate and freeze reservoir topologies.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="frozen", help="Directory to save frozen reservoirs")
    parser.add_argument("--seeds_count", type=int, default=30, help="Number of seeds to generate")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 200], help="Network sizes (N)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    for N in args.sizes:
        logger.info(f"Generating reservoirs for N={N}")
        cfg["network"]["N"] = N
        size_dir = os.path.join(args.output_dir, f"N_{N}")
        os.makedirs(size_dir, exist_ok=True)

        for i in range(args.seeds_count):
            seed = cfg.get("experiment", {}).get("seed_base", 1000) + i
            res_path = os.path.join(size_dir, f"seed_{seed}")
            
            if os.path.exists(res_path):
                logger.info(f"Seed {seed} for N={N} already exists, skipping.")
                continue

            logger.info(f"Building reservoir seed {seed}...")
            res = build_reservoir(cfg, seed)
            res.save(res_path)
            logger.info(f"Saved to {res_path}")

if __name__ == "__main__":
    main()
