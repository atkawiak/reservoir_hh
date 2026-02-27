"""
Empirical rho -> lambda scan for seed 100.
Measures true Lyapunov exponent at 16 spectral radius values.
"""
import time
import numpy as np
from regime_calibrator import load_regime_npz, measure_lambda_benettin, DT_MS

RHO_BASE = 53.482092   # measured for seed=100 in regime_builder
SEED = 100
NPZ  = "regimes/R3_near_critical_seed100.npz"

regime_data = load_regime_npz(NPZ)

rho_targets = np.array([
    0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95,
    1.00, 1.07, 1.15, 1.30, 1.50, 1.80, 2.20, 2.80
])
alphas = rho_targets / RHO_BASE

N_REPEATS  = 3
WARMUP_MS  = 200.0
MEASURE_MS = 400.0
RENORM_MS  = 10.0

print(f"{'rho':>6s}  {'alpha':>10s}  {'lam_mean':>10s}  {'lam_std':>8s}  vals", flush=True)
print("-" * 70, flush=True)

t0 = time.time()
rows = []
for rho, alpha in zip(rho_targets, alphas):
    lams = []
    for r in range(N_REPEATS):
        rng = np.random.default_rng(SEED + 888 + r * 100)
        lam = measure_lambda_benettin(
            regime_data, alpha, 0.05, rng,
            warmup_ms=WARMUP_MS, measure_ms=MEASURE_MS,
            renorm_ms=RENORM_MS, dt_ms=DT_MS,
        )
        lams.append(lam)
    m, s = np.mean(lams), np.std(lams)
    vals = ", ".join(f"{l:.1f}" for l in lams)
    print(f"{rho:6.2f}  {alpha:10.6f}  {m:10.2f}  {s:8.2f}  {vals}", flush=True)
    rows.append((rho, alpha, m, s))

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)

# Save CSV
import csv
with open("/tmp/rho_lambda_scan_seed100.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rho", "alpha", "lam_mean", "lam_std"])
    w.writerows(rows)
print("CSV saved: /tmp/rho_lambda_scan_seed100.csv", flush=True)
