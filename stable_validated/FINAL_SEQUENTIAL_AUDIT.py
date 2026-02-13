
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
        ("MC", "configs/scientific/FULL_mc_maass.yaml", run_mc_benchmark),
        ("NARMA", "configs/scientific/FULL_narma_legenstein.yaml", run_narma10_benchmark),
        ("XOR", "configs/scientific/FULL_xor_standard.yaml", run_delayed_xor_benchmark),
        ("HENON", "configs/scientific/FULL_henon_atlas.yaml", run_henon_benchmark)
    ]
    
    final_output = []
    
    print("\n" + "#"*60, flush=True)
    print("RYGORYSTYCZNY AUDYT SEKWENCYJNY (Brak Przerwań)", flush=True)
    print("#"*60 + "\n", flush=True)

    for name, config_path, runner in tasks:
        logger.info(f"Odpalam {name}...")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # Fresh reservoir for each task to be 100% clean
        res = build_reservoir(cfg, seed=1000)
        
        start = time.time()
        try:
            res_dict = runner(res, cfg, inh_scaling=1.0, seed_input=1000)
            elapsed = time.time() - start
            
            if name == "MC": val = res_dict['mc_total']
            elif name == "NARMA": val = res_dict['nrmse']
            elif name == "XOR": val = res_dict['accuracy']
            elif name == "HENON": val = res_dict['nrmse']
            
            logger.info(f"ZAKOŃCZONO {name}: {val:.4f} (Czas: {elapsed:.1f}s)")
            final_output.append({"Zadanie": name, "Wynik": val})
        except Exception as e:
            logger.error(f"BŁĄD w {name}: {e}")

    print("\n" + "="*40)
    print("OSTATECZNY RAPORT NAUKOWY")
    print("="*40)
    df = pd.DataFrame(final_output)
    print(df.to_string(index=False))
    print("="*40)

if __name__ == "__main__":
    run_audit()
