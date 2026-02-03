#!/usr/bin/env python3
"""
LOCAL VALIDATION SCRIPT
Uruchamia minimalny test i sprawdza czy wszystkie wyniki są poprawne.
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np

# Change to project directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

def run_single_trial():
    """Uruchamia pojedynczy trial i zwraca wyniki."""
    print("=" * 60)
    print("KROK 1: Uruchamianie pojedynczego trialu...")
    print("=" * 60)
    
    result = subprocess.run(
        ['python', 'src/run_experiment.py', '--config', 'configs/local_test.yaml'],
        capture_output=True, text=True, timeout=300
    )
    
    if result.returncode != 0:
        print(f"BŁĄD: {result.stderr}")
        return None
    
    print(result.stdout)
    return True

def validate_results():
    """Sprawdza czy wyniki są kompletne i sensowne."""
    print("\n" + "=" * 60)
    print("KROK 2: Walidacja wyników...")
    print("=" * 60)
    
    # Znajdź najnowszy plik
    results_dir = 'results'
    files = sorted([f for f in os.listdir(results_dir) if f.endswith('.parquet')])
    if not files:
        print("BŁĄD: Brak plików wyników!")
        return False
    
    latest = os.path.join(results_dir, files[-1])
    print(f"Wczytuję: {latest}")
    
    df = pd.read_parquet(latest)
    print(f"Liczba wierszy: {len(df)}")
    print(f"Kolumny: {list(df.columns)}")
    
    # WALIDACJA SCHEMATU
    required_cols = ['N', 'gL', 'rho', 'bias', 'difficulty', 'task', 'metric', 'value',
                     'firing_rate', 'cv_isi', 'saturation_flag', 'seed_tuple_id']
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ BRAK KOLUMN: {missing}")
        return False
    print("✅ Wszystkie wymagane kolumny obecne")
    
    # WALIDACJA TASKÓW
    expected_tasks = {'NARMA', 'XOR', 'MC', 'Lyapunov'}
    actual_tasks = set(df['task'].unique())
    if not expected_tasks.issubset(actual_tasks):
        print(f"❌ BRAK TASKÓW: {expected_tasks - actual_tasks}")
        return False
    print(f"✅ Wszystkie taski: {actual_tasks}")
    
    # WALIDACJA METRYK
    print("\n--- Metryki per task ---")
    for task in expected_tasks:
        task_df = df[df['task'] == task]
        metrics = task_df['metric'].unique()
        values = task_df['value'].values
        print(f"  {task}: metrics={list(metrics)}, value range=[{values.min():.4f}, {values.max():.4f}]")
    
    # WALIDACJA BIO
    print("\n--- Bio-plausibility ---")
    fr = df['firing_rate'].iloc[0]
    cv = df['cv_isi'].iloc[0]
    sat = df['saturation_flag'].iloc[0]
    
    print(f"  Firing Rate: {fr:.2f} Hz {'✅' if 1 < fr < 50 else '⚠️ (poza zakresem biologicznym)'}")
    print(f"  CV ISI: {cv:.3f} {'✅' if 0.3 < cv < 2.0 else '⚠️'}")
    print(f"  Saturation: {sat} {'✅' if not sat else '⚠️'}")
    
    # WALIDACJA LYAPUNOV
    lyap_df = df[(df['task'] == 'Lyapunov') & (df['metric'] == 'lambda_sec')]
    if len(lyap_df) > 0:
        lam = lyap_df['value'].iloc[0]
        print(f"  Lambda: {lam:.4f} s^-1 {'(subcritical)' if lam < 0 else '(chaotic)' if lam > 0.01 else '(edge of chaos!)'}")
    
    # WALIDACJA XOR
    xor_df = df[(df['task'] == 'XOR') & (df['metric'] == 'accuracy')]
    if len(xor_df) > 0:
        acc = xor_df['value'].iloc[0]
        print(f"  XOR Accuracy: {acc:.3f} {'✅ (> baseline)' if acc > 0.55 else '⚠️ (≈ random)'}")
    
    print("\n" + "=" * 60)
    print("WALIDACJA ZAKOŃCZONA")
    print("=" * 60)
    
    return True

def main():
    print("\n" + "🧪" * 30)
    print("  LOCAL VALIDATION TEST")
    print("🧪" * 30 + "\n")
    
    # Run
    success = run_single_trial()
    if not success:
        print("\n❌ TRIAL FAILED")
        sys.exit(1)
    
    # Validate
    valid = validate_results()
    if not valid:
        print("\n❌ VALIDATION FAILED")
        sys.exit(1)
    
    print("\n✅ ALL TESTS PASSED!")
    sys.exit(0)

if __name__ == "__main__":
    main()
