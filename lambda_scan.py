"""
Lambda scan: measure lambda(alpha) at multiple points for one regime NPZ.
Uses the same SpikeGeneratorGroup deterministic Poisson as regime_calibrator.

Usage:
  python lambda_scan.py              # full scan at default dt=0.05ms
  python lambda_scan.py --dt-test    # numerical stability: compare dt=0.05 vs 0.025
"""
import sys
import time
import numpy as np

# Import everything we need from regime_calibrator
from regime_calibrator import (
    load_regime_npz, measure_lambda_benettin, build_network,
    Metrics, El, mV, ms, DT_MS,
)


def scan_lambda(npz_path, alphas, seed=42, n_repeats=3,
                warmup_ms=200.0, measure_ms=500.0, renorm_ms=10.0,
                dt_ms=None):
    """Measure lambda at each alpha, repeating n_repeats times with different seeds."""
    regime_data = load_regime_npz(npz_path)
    dt_label = f"{dt_ms}" if dt_ms is not None else f"{DT_MS} (default)"
    print(f"Scanning {npz_path}: N={regime_data['N_total']}, "
          f"{len(alphas)} alpha points, {n_repeats} repeats, dt={dt_label}ms\n")
    print(f"{'alpha':>12s}  {'lam_mean':>10s}  {'lam_std':>10s}  {'CV%':>5s}  {'lam_vals'}")
    print("-" * 80)

    results = []
    for alpha in alphas:
        lams = []
        for r in range(n_repeats):
            rng = np.random.default_rng(seed + 888 + r * 100)
            lam = measure_lambda_benettin(
                regime_data, alpha, 0.05, rng,
                warmup_ms=warmup_ms, measure_ms=measure_ms,
                renorm_ms=renorm_ms, dt_ms=dt_ms,
            )
            lams.append(lam)
        mean_l = np.mean(lams)
        std_l = np.std(lams)
        cv = 100 * std_l / (abs(mean_l) + 1e-10)
        vals_str = ", ".join(f"{l:.1f}" for l in lams)
        print(f"{alpha:12.6f}  {mean_l:10.2f}  {std_l:10.2f}  {cv:4.0f}%  {vals_str}")
        results.append((alpha, mean_l, std_l, lams))

    return results


if __name__ == "__main__":
    t0 = time.time()

    # Use R3_near_critical as representative (mid-range alpha)
    npz_path = "regimes/R3_near_critical_seed42.npz"

    if "--dt-test" in sys.argv:
        # ── Numerical stability test: dt=0.05 vs dt=0.025 ──
        # Fewer alpha points, 3 repeats each — enough to compare means.
        alphas = np.array([0.005, 0.012, 0.03, 0.06])

        print("=" * 80)
        print("DT STABILITY TEST: dt=0.05ms")
        print("=" * 80)
        r1 = scan_lambda(npz_path, alphas, n_repeats=3, measure_ms=500.0,
                          dt_ms=0.05)

        print(f"\n{'=' * 80}")
        print("DT STABILITY TEST: dt=0.025ms")
        print("=" * 80)
        r2 = scan_lambda(npz_path, alphas, n_repeats=3, measure_ms=500.0,
                          dt_ms=0.025)

        print(f"\n{'=' * 80}")
        print("COMPARISON: |mean_050 - mean_025| / mean_050")
        print("=" * 80)
        print(f"{'alpha':>12s}  {'mean_050':>10s}  {'mean_025':>10s}  {'rel_diff':>10s}")
        print("-" * 50)
        for (a1, m1, _, _), (a2, m2, _, _) in zip(r1, r2):
            rel = abs(m1 - m2) / (abs(m1) + 1e-10)
            flag = "  OK" if rel < 0.10 else "  WARN" if rel < 0.20 else "  BAD"
            print(f"{a1:12.6f}  {m1:10.2f}  {m2:10.2f}  {rel:9.1%}{flag}")

    else:
        # ── Standard full scan ──
        alphas = np.array([
            0.001, 0.005, 0.008, 0.012, 0.02,
            0.04, 0.06, 0.08, 0.10,
        ])

        print("=" * 80)
        print("LAMBDA SCAN: measure_ms=500, n_repeats=3")
        print("=" * 80)
        results = scan_lambda(npz_path, alphas, n_repeats=3, measure_ms=500.0)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
