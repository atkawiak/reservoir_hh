#!/usr/bin/env python3
"""
PRECYZYJNE POMIARY λ
Sprawdzamy czy są SUBTELNE różnice w λ między parametrami.
Może różnica jest na poziomie 1e-5 zamiast 1e-2.
"""

import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

import numpy as np
from config import load_config
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

def precise_lyapunov(cfg, rho: float, gL: float, bias: float) -> dict:
    """Precyzyjne obliczenie λ z więcej cyfr."""
    
    cfg.hh.N = 100
    
    rng_mgr = RNGManager(2025)
    trial_gens = rng_mgr.get_trial_generators(0)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(0)
    
    hh = HHModel(cfg, trial_gens, seeds_tuple)
    lyap = LyapunovModule(trial_gens['in'])
    
    steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
    
    # Dłuższy warm-up
    steps_wu = 1000
    u_wu = trial_gens['in'].uniform(0, 1, steps_wu // steps_per_symbol + 1)
    rates_wu = cfg.task.poisson_rate_min + u_wu * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
    spikes_wu = (trial_gens['in'].random(steps_wu) < (np.repeat(rates_wu, steps_per_symbol)[:steps_wu] * cfg.task.dt * 1e-3)).astype(np.float32)
    
    res_wu = hh.simulate(rho, bias, spikes_wu, "WU", trim_steps=0, gL=gL)
    start_state = res_wu['final_state']
    
    # Dłuższa trajektoria dla lepszej precyzji
    len_lyap = 500
    u_l = trial_gens['in'].uniform(0, 1, len_lyap)
    rates_l = cfg.task.poisson_rate_min + u_l * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
    spikes_l = (trial_gens['in'].random(len_lyap * steps_per_symbol) < (np.repeat(rates_l, steps_per_symbol) * cfg.task.dt * 1e-3)).astype(np.float32)
    
    state_l1 = hh.simulate(rho, bias, spikes_l, "L_ref", trim_steps=0, full_state=start_state, gL=gL)
    phi1 = filter_and_downsample(state_l1['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
    
    perturbed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in start_state.items()}
    perturbed['V'][0] += 1e-6
    
    state_l2 = hh.simulate(rho, bias, spikes_l, "L_pert", trim_steps=0, full_state=perturbed, gL=gL)
    phi2 = filter_and_downsample(state_l2['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
    
    step_s = (steps_per_symbol * cfg.task.dt) / 1000.0
    slope = lyap.compute_lambda(phi1, phi2, window_range=[50, 300])
    lambda_sec = slope / step_s
    
    return {
        'rho': rho, 'gL': gL, 'bias': bias,
        'lambda_sec': lambda_sec,
        'lambda_precise': f"{lambda_sec:.8f}",  # Więcej cyfr
        'firing_rate': state_l1['mean_rate'],
    }

def main():
    print("=" * 80)
    print("  PRECYZYJNE POMIARY λ - szukamy subtelnych różnic")
    print("=" * 80)
    
    cfg = load_config('configs/local_test.yaml')
    
    results = []
    
    # Gęstszy grid ρ w biologicznym zakresie
    rho_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    gL = 0.3
    bias = 5.0
    
    print(f"\n--- gL={gL}, bias={bias} ---")
    print(f"{'ρ':>6} | {'λ (precyzyjne)':>16} | {'FR':>8} | Ranking")
    print("-" * 50)
    
    for rho in rho_values:
        try:
            res = precise_lyapunov(cfg, rho=rho, gL=gL, bias=bias)
            results.append(res)
            print(f"{rho:6.1f} | {res['lambda_precise']:>16} | {res['firing_rate']:8.1f} Hz |")
        except Exception as e:
            print(f"{rho:6.1f} | ERROR: {e}")
    
    # Sortowanie po λ
    print("\n" + "=" * 80)
    print("  RANKING: od najbardziej stabilnego do najbliżej chaosu")
    print("=" * 80)
    
    sorted_results = sorted(results, key=lambda x: x['lambda_sec'])
    
    for i, r in enumerate(sorted_results):
        stability = "NAJBARDZIEJ STABILNY" if i == 0 else "NAJMNIEJ STABILNY" if i == len(sorted_results)-1 else ""
        print(f"  {i+1}. ρ={r['rho']:.1f} | λ={r['lambda_precise']} | FR={r['firing_rate']:.1f} Hz {stability}")
    
    # Różnica między min i max
    if len(sorted_results) >= 2:
        lam_min = sorted_results[0]['lambda_sec']
        lam_max = sorted_results[-1]['lambda_sec']
        diff = lam_max - lam_min
        print(f"\n  Różnica λ: {diff:.8f} s^-1")
        print(f"  Zakres: [{lam_min:.8f}, {lam_max:.8f}]")
        
        if abs(diff) < 1e-6:
            print("\n  ⚠️ Różnica < 1e-6 - prawdopodobnie NUMERYCZNY NOISE")
        elif abs(diff) < 1e-4:
            print("\n  ℹ️ Różnica < 1e-4 - SUBTELNA ale może istotna dla performance")
        else:
            print("\n  ✅ Różnica > 1e-4 - ZNACZĄCA zmiana stabilności")

if __name__ == "__main__":
    main()
