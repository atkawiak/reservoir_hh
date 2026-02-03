#!/usr/bin/env python3
"""
CHAOS THRESHOLD FINDER
Szybko skanuje ρ żeby znaleźć gdzie λ przechodzi przez 0 (Edge of Chaos).
"""

import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

import numpy as np
from config import load_config, ExperimentConfig
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

def quick_lyapunov(rho: float, gL: float, N: int = 100, bias: float = 5.0) -> dict:
    """Szybkie obliczenie λ dla danego ρ."""
    
    # Minimal config
    cfg = load_config('configs/local_test.yaml')
    cfg.hh.N = N
    
    rng_mgr = RNGManager(2025)
    trial_gens = rng_mgr.get_trial_generators(0)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(0)
    
    hh = HHModel(cfg, trial_gens, seeds_tuple)
    lyap = LyapunovModule(trial_gens['in'])
    
    steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
    
    # Warm-up
    steps_wu = 500
    u_wu = trial_gens['in'].uniform(0, 1, steps_wu // steps_per_symbol + 1)
    rates_wu = cfg.task.poisson_rate_min + u_wu * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
    spikes_wu = (trial_gens['in'].random(steps_wu) < (np.repeat(rates_wu, steps_per_symbol)[:steps_wu] * cfg.task.dt * 1e-3)).astype(np.float32)
    
    res_wu = hh.simulate(rho, bias, spikes_wu, "WU", trim_steps=0, gL=gL)
    start_state = res_wu['final_state']
    
    # Lyapunov trajectories
    len_lyap = 300
    u_l = trial_gens['in'].uniform(0, 1, len_lyap)
    rates_l = cfg.task.poisson_rate_min + u_l * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
    spikes_l = (trial_gens['in'].random(len_lyap * steps_per_symbol) < (np.repeat(rates_l, steps_per_symbol) * cfg.task.dt * 1e-3)).astype(np.float32)
    
    # Reference
    state_l1 = hh.simulate(rho, bias, spikes_l, "L_ref", trim_steps=0, full_state=start_state, gL=gL)
    phi1 = filter_and_downsample(state_l1['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
    
    # Perturbed
    perturbed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in start_state.items()}
    perturbed['V'][0] += 1e-6
    
    state_l2 = hh.simulate(rho, bias, spikes_l, "L_pert", trim_steps=0, full_state=perturbed, gL=gL)
    phi2 = filter_and_downsample(state_l2['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
    
    step_s = (steps_per_symbol * cfg.task.dt) / 1000.0
    slope = lyap.compute_lambda(phi1, phi2, window_range=[30, 150])
    lambda_sec = slope / step_s
    
    return {
        'rho': rho,
        'gL': gL,
        'lambda_sec': lambda_sec,
        'firing_rate': state_l1['mean_rate'],
        'saturation': state_l1['saturation_flag']
    }

def main():
    print("=" * 70)
    print("  CHAOS THRESHOLD FINDER")
    print("  Szukam ρ gdzie λ przechodzi przez 0")
    print("=" * 70)
    
    # Parametry do testowania
    gL_values = [0.3, 0.15, 0.075]
    rho_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    results = []
    
    for gL in gL_values:
        print(f"\n--- gL = {gL} ---")
        chaos_threshold = None
        
        for rho in rho_values:
            try:
                res = quick_lyapunov(rho, gL, N=100, bias=5.0)
                regime = "CHAOS" if res['lambda_sec'] > 0.01 else "EDGE" if abs(res['lambda_sec']) < 0.01 else "STABLE"
                marker = "🔥" if regime == "CHAOS" else "⚡" if regime == "EDGE" else "❄️"
                
                print(f"  ρ={rho:5.1f} | λ={res['lambda_sec']:+.4f} s⁻¹ | FR={res['firing_rate']:5.1f} Hz | {marker} {regime}")
                
                res['gL'] = gL
                res['regime'] = regime
                results.append(res)
                
                # Znajdź próg
                if chaos_threshold is None and res['lambda_sec'] > 0:
                    chaos_threshold = rho
                    
            except Exception as e:
                print(f"  ρ={rho:5.1f} | ERROR: {e}")
        
        if chaos_threshold:
            print(f"  → PRÓG CHAOSU dla gL={gL}: ρ ≈ {chaos_threshold}")
        else:
            print(f"  → BRAK CHAOSU do ρ=100 dla gL={gL}")
    
    print("\n" + "=" * 70)
    print("PODSUMOWANIE: Zalecane zakresy ρ dla Phase 1")
    print("=" * 70)
    
    for gL in gL_values:
        gL_results = [r for r in results if r['gL'] == gL]
        edges = [r for r in gL_results if r['regime'] == 'EDGE']
        chaos = [r for r in gL_results if r['regime'] == 'CHAOS']
        
        if edges:
            print(f"  gL={gL}: Edge of Chaos przy ρ ∈ [{min(r['rho'] for r in edges)}, {max(r['rho'] for r in edges)}]")
        elif chaos:
            min_chaos = min(r['rho'] for r in chaos)
            print(f"  gL={gL}: Chaos zaczyna się od ρ ≈ {min_chaos}, sweep zalecany [1, {min_chaos*2}]")
        else:
            print(f"  gL={gL}: Zbyt stabilne, potrzeba wyższych ρ lub niższych gL")

if __name__ == "__main__":
    main()
