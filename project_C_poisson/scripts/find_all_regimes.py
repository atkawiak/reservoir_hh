#!/usr/bin/env python3
"""
FIND ALL REGIMES
Szukamy parametrów które dają:
- SUBCRITICAL: λ < -0.01 (stabilny)
- CRITICAL: |λ| < 0.01 (edge of chaos)
- CHAOTIC: λ > +0.01 (chaos)

Testujemy: gL, gA (prąd A), bias
"""

import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

import numpy as np
from dataclasses import replace
from config import load_config
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

def quick_lyapunov(cfg, rho: float, gL: float, bias: float, gA: float = 20.0) -> dict:
    """Szybkie obliczenie λ."""
    
    cfg.hh.N = 100
    cfg.hh.gA = gA  # Override A-current
    
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
    
    # Lyapunov 
    len_lyap = 300
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
    slope = lyap.compute_lambda(phi1, phi2, window_range=[30, 150])
    lambda_sec = slope / step_s
    
    return {
        'rho': rho, 'gL': gL, 'bias': bias, 'gA': gA,
        'lambda_sec': lambda_sec,
        'firing_rate': state_l1['mean_rate'],
        'saturation': state_l1['saturation_flag']
    }

def classify_regime(lam):
    if lam < -0.01:
        return "SUBCRIT", "❄️"
    elif lam > 0.01:
        return "CHAOTIC", "🔥"
    else:
        return "EDGE", "⚡"

def main():
    print("=" * 80)
    print("  SZUKANIE WSZYSTKICH REŻIMÓW (subcritical, edge, chaotic)")
    print("=" * 80)
    
    cfg = load_config('configs/local_test.yaml')
    
    results = []
    
    # Test 1: Różne gA (prąd A) - wyłączenie powinno destabilizować
    print("\n" + "=" * 60)
    print("TEST 1: Wpływ prądu A (gA)")
    print("  Hipoteza: gA=0 → mniej stabilna sieć → chaos przy niższym ρ")
    print("=" * 60)
    
    for gA in [20.0, 10.0, 5.0, 0.0]:  # Od pełnego do wyłączonego
        print(f"\n--- gA = {gA} (A-current {'ON' if gA > 0 else 'OFF'}) ---")
        for rho in [1.0, 2.0, 5.0]:
            try:
                res = quick_lyapunov(cfg, rho=rho, gL=0.3, bias=5.0, gA=gA)
                regime, marker = classify_regime(res['lambda_sec'])
                bio = "✅" if 1 < res['firing_rate'] < 50 else "❌"
                print(f"  ρ={rho:.1f} | λ={res['lambda_sec']:+.4f} | FR={res['firing_rate']:5.1f} Hz {bio} | {marker} {regime}")
                results.append(res)
            except Exception as e:
                print(f"  ρ={rho:.1f} | ERROR: {e}")
    
    # Test 2: Bardzo niski gL
    print("\n" + "=" * 60)
    print("TEST 2: Bardzo niski gL (mniej stabilizacji)")
    print("  Hipoteza: gL→0 → wolniejszy powrót do rest → łatwiej chaos")
    print("=" * 60)
    
    for gL in [0.3, 0.1, 0.05, 0.01]:
        print(f"\n--- gL = {gL} ---")
        for rho in [1.0, 2.0, 5.0]:
            try:
                res = quick_lyapunov(cfg, rho=rho, gL=gL, bias=5.0, gA=20.0)
                regime, marker = classify_regime(res['lambda_sec'])
                bio = "✅" if 1 < res['firing_rate'] < 50 else "❌"
                print(f"  ρ={rho:.1f} | λ={res['lambda_sec']:+.4f} | FR={res['firing_rate']:5.1f} Hz {bio} | {marker} {regime}")
                results.append(res)
            except Exception as e:
                print(f"  ρ={rho:.1f} | ERROR: {e}")
    
    # Test 3: Wysoki bias (więcej aktywności spontanicznej)
    print("\n" + "=" * 60)
    print("TEST 3: Wysoki bias (więcej aktywności spontanicznej)")
    print("  Hipoteza: bias↑ → więcej spikes → łatwiej interakcje → chaos")
    print("=" * 60)
    
    for bias in [0.0, 5.0, 10.0, 20.0, 50.0]:
        print(f"\n--- bias = {bias} pA ---")
        for rho in [1.0, 2.0, 5.0]:
            try:
                res = quick_lyapunov(cfg, rho=rho, gL=0.3, bias=bias, gA=20.0)
                regime, marker = classify_regime(res['lambda_sec'])
                bio = "✅" if 1 < res['firing_rate'] < 50 else "❌"
                print(f"  ρ={rho:.1f} | λ={res['lambda_sec']:+.4f} | FR={res['firing_rate']:5.1f} Hz {bio} | {marker} {regime}")
                results.append(res)
            except Exception as e:
                print(f"  ρ={rho:.1f} | ERROR: {e}")
    
    # Podsumowanie
    print("\n" + "=" * 80)
    print("  PODSUMOWANIE: Znalezione parametry dla każdego reżimu")
    print("=" * 80)
    
    subcrit = [r for r in results if r['lambda_sec'] < -0.01 and 1 < r['firing_rate'] < 50]
    edge = [r for r in results if abs(r['lambda_sec']) < 0.01 and 1 < r['firing_rate'] < 50]
    chaotic = [r for r in results if r['lambda_sec'] > 0.01 and 1 < r['firing_rate'] < 50]
    
    print(f"\n❄️ SUBCRITICAL (λ < -0.01, FR ∈ [1,50]): {len(subcrit)} punktów")
    if subcrit:
        for r in subcrit[:3]:
            print(f"   gL={r['gL']}, gA={r['gA']}, bias={r['bias']}, ρ={r['rho']} → λ={r['lambda_sec']:.4f}")
    
    print(f"\n⚡ EDGE OF CHAOS (|λ| < 0.01, FR ∈ [1,50]): {len(edge)} punktów")
    if edge:
        for r in edge[:3]:
            print(f"   gL={r['gL']}, gA={r['gA']}, bias={r['bias']}, ρ={r['rho']} → λ={r['lambda_sec']:.4f}")
    
    print(f"\n🔥 CHAOTIC (λ > +0.01, FR ∈ [1,50]): {len(chaotic)} punktów")
    if chaotic:
        for r in chaotic[:3]:
            print(f"   gL={r['gL']}, gA={r['gA']}, bias={r['bias']}, ρ={r['rho']} → λ={r['lambda_sec']:.4f}")
    
    if not subcrit or not chaotic:
        print("\n⚠️  UWAGA: Nie znaleziono kontrastu między reżimami!")
        print("   Model HH z A-current jest zbyt stabilny.")
        print("   Rozważ: wyłączenie A-current (gA=0) lub inne parametry.")

if __name__ == "__main__":
    main()
