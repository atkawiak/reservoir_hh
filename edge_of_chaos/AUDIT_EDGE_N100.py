
import yaml
import logging
import pandas as pd
import time
from src.reservoir.build_reservoir import build_reservoir
from src.benchmarks.mc import run_mc_benchmark
from src.benchmarks.narma10 import run_narma10_benchmark
from src.benchmarks.delayed_xor import run_delayed_xor_benchmark
from src.benchmarks.henon import run_henon_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_audit():
    tasks = [
        ("MC", "configs/scientific/EDGE_mc.yaml", run_mc_benchmark),
        ("NARMA", "configs/scientific/EDGE_narma.yaml", run_narma10_benchmark),
        ("XOR", "configs/scientific/EDGE_xor.yaml", run_delayed_xor_benchmark),
        ("HENON", "configs/scientific/EDGE_henon.yaml", run_henon_benchmark)
    ]
    
    final_output = []
    for name, config_path, runner in tasks:
        logger.info(f"Odpalam EDGE {name}...")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        res = build_reservoir(cfg, seed=100)
        start = time.time()
        res_dict = runner(res, cfg, inh_scaling=1.0, seed_input=100)
        if name == "MC": val = res_dict['mc_total']
        elif name == "NARMA": val = res_dict['nrmse']
        elif name == "XOR": val = res_dict['accuracy']
        elif name == "HENON": val = res_dict['nrmse']
        final_output.append({"Zadanie": name, "Wynik EDGE": val})

    print("\nRAPORT EDGE (N=100)")
    print(pd.DataFrame(final_output).to_string(index=False))

if __name__ == "__main__":
    run_audit()
