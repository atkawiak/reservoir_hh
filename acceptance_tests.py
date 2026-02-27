"""
Acceptance tests for calibrated regimes.

A) Regime separation test — verify λ distributions don't overlap (P10–P90).
B) Cross-check deterministic (SpikeGenerator) vs stochastic (PoissonGroup).
C) Report with λ recommendations.
D) dt-stability — compare dt=0.025 vs dt=0.0125 (rel_diff < 15%).
E) eps-invariance — compare λ at eps=0.02, 0.05, 0.1 mV.
F) time-length convergence — compare median λ at measure_ms=500, 1000, 2000.

Usage:
  python acceptance_tests.py --seed 100
  python acceptance_tests.py --seed 100 --test A        # only separation test
  python acceptance_tests.py --seed 100 --test D        # only dt-stability
  python acceptance_tests.py --seed 100 --test DEF      # all 3 validation tests
  python acceptance_tests.py --seed 100 --n-seeds 20    # more Poisson seeds
"""
import os
import sys
import time
import argparse
import csv
import numpy as np

from regime_calibrator import (
    load_regime_npz, measure_lambda_benettin, build_network,
    generate_poisson_trains, Metrics,
    LAMBDA_WINDOWS_RC, LAMBDA_TARGETS_RC,
    El, mV, ms, DT_MS, DT_LAMBDA_MS,
    # HH constants must be in caller namespace for Brian2 net.run()
    EK, ENa, g_na, g_kd, gl, Cm, VT,
)
from brian2 import start_scope

REGIMES = [
    "R1_super_stable",
    "R2_stable",
    "R3_near_critical",
    "R4_edge_of_chaos",
    "R5_chaotic",
]

# ── Golden setup defaults ──
GOLDEN_EPS_MV = 0.05                # perturbation amplitude (mV)
GOLDEN_DT_MS = DT_MS                # 0.025 ms (metrics)
GOLDEN_DT_LAMBDA_MS = DT_LAMBDA_MS  # 0.0125 ms (Benettin)
GOLDEN_WARMUP_MS = 200.0
GOLDEN_MEASURE_MS = 2000.0          # final report: 2000ms
GOLDEN_RENORM_MS = 5.0              # renormalization interval
GOLDEN_N_SEEDS = 7                  # final report: 7 repeats per regime


# ═════════════════════════════════════════════════════════════════════════════
# A) REGIME SEPARATION TEST
# ═════════════════════════════════════════════════════════════════════════════

def test_separation(seed, npz_dir="regimes_calibrated", n_seeds=GOLDEN_N_SEEDS,
                    warmup_ms=GOLDEN_WARMUP_MS, measure_ms=GOLDEN_MEASURE_MS,
                    renorm_ms=GOLDEN_RENORM_MS):
    """Measure λ on n_seeds fresh Poisson realizations per regime.

    For each calibrated regime NPZ, run n_seeds independent Benettin
    measurements using Poisson seeds disjoint from calibration seeds.
    Report median, IQR, P10, P90 and check for overlap.
    """
    print("\n" + "=" * 70)
    print("  TEST A: REGIME SEPARATION (fresh Poisson seeds)")
    print("=" * 70)
    print(f"  n_seeds={n_seeds}  warmup={warmup_ms}ms  measure={measure_ms}ms"
          f"  eps={GOLDEN_EPS_MV}mV  dt_lambda={GOLDEN_DT_LAMBDA_MS}ms\n")

    all_results = {}

    for regime_name in REGIMES:
        npz_path = os.path.join(npz_dir, f"{regime_name}_seed{seed}.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP {regime_name}: {npz_path} not found")
            continue

        regime_data = load_regime_npz(npz_path)
        alpha = float(regime_data["alpha_final"])

        lams = []
        for k in range(n_seeds):
            # Use seed range 5000+ to avoid overlap with calibration seeds
            rng = np.random.default_rng(seed + 5000 + k * 37)
            lam = measure_lambda_benettin(
                regime_data, alpha, 0.05, rng,
                warmup_ms=warmup_ms, measure_ms=measure_ms,
                renorm_ms=renorm_ms, eps_mV=GOLDEN_EPS_MV,
                dt_ms=GOLDEN_DT_LAMBDA_MS,
            )
            lams.append(lam)

        lams = np.array(lams)
        med = np.median(lams)
        q1, q3 = np.percentile(lams, [25, 75])
        p10, p90 = np.percentile(lams, [10, 90])
        iqr = q3 - q1

        all_results[regime_name] = dict(
            alpha=alpha, lams=lams, median=med,
            q1=q1, q3=q3, p10=p10, p90=p90, iqr=iqr,
        )

        print(f"  {regime_name:25s}  alpha={alpha:.6f}  "
              f"med={med:6.1f}  IQR=[{q1:.1f}, {q3:.1f}]  "
              f"P10-P90=[{p10:.1f}, {p90:.1f}]")

    # ── Check overlap ──
    print(f"\n  {'--- OVERLAP CHECK ---':^70s}")
    names = [r for r in REGIMES if r in all_results]
    overlap_found = False

    for i in range(len(names) - 1):
        r_lo = all_results[names[i]]
        r_hi = all_results[names[i + 1]]

        # Overlap if P90 of lower regime > P10 of higher regime
        if r_lo["p90"] > r_hi["p10"]:
            overlap_found = True
            gap = r_hi["p10"] - r_lo["p90"]
            print(f"  OVERLAP: {names[i]} P90={r_lo['p90']:.1f} > "
                  f"{names[i+1]} P10={r_hi['p10']:.1f}  (gap={gap:.1f})")
        else:
            gap = r_hi["p10"] - r_lo["p90"]
            print(f"  OK:      {names[i]} P90={r_lo['p90']:.1f} < "
                  f"{names[i+1]} P10={r_hi['p10']:.1f}  (gap={gap:.1f})")

    status = "FAIL — regimes overlap" if overlap_found else "PASS — no overlap"
    print(f"\n  SEPARATION: {status}")

    # ── Save CSV ──
    csv_path = f"acceptance_A_separation_seed{seed}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "alpha", "median", "q1", "q3", "p10", "p90",
                     "iqr", "n_seeds"])
        for rn in names:
            r = all_results[rn]
            w.writerow([rn, r["alpha"], r["median"], r["q1"], r["q3"],
                        r["p10"], r["p90"], r["iqr"], n_seeds])
    print(f"  Saved: {csv_path}\n")

    return all_results, not overlap_found


# ═════════════════════════════════════════════════════════════════════════════
# B) CROSS-CHECK: DETERMINISTIC vs POISSON GROUP
# ═════════════════════════════════════════════════════════════════════════════

def test_crosscheck(seed, npz_dir="regimes_calibrated", n_runs=5,
                    warmup_ms=GOLDEN_WARMUP_MS, measure_ms=GOLDEN_MEASURE_MS):
    """Compare metrics under deterministic SpikeGenerator vs PoissonGroup.

    For each regime, run n_runs with each input mode and report median
    metrics (rate_E, rate_I, CV_ISI, sync).
    """
    print("\n" + "=" * 70)
    print("  TEST B: CROSS-CHECK (SpikeGenerator vs PoissonGroup)")
    print("=" * 70)
    print(f"  n_runs={n_runs}  warmup={warmup_ms}ms  measure={measure_ms}ms\n")

    total_ms = warmup_ms + measure_ms
    all_results = {}

    for regime_name in REGIMES:
        npz_path = os.path.join(npz_dir, f"{regime_name}_seed{seed}.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP {regime_name}")
            continue

        regime_data = load_regime_npz(npz_path)
        alpha = float(regime_data["alpha_final"])
        N = int(regime_data["N_total"])
        idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
        idx_I = np.array(regime_data["idx_I"], dtype=np.int32)
        n_E, n_I = len(idx_E), len(idx_I)

        modes = {}
        for mode in ["spike_generator", "poisson_group"]:
            rates_E, rates_I, cvs, syncs = [], [], [], []
            for k in range(n_runs):
                rng_k = np.random.default_rng(seed + 3000 + k * 100)

                kw = dict(
                    alpha_override=alpha, I_b_nA=0.05,
                    input_mode=mode, dt_ms=GOLDEN_DT_MS,
                )
                if mode == "spike_generator":
                    trains_E = generate_poisson_trains(
                        n_E, 20.0, total_ms, rng_k, dt_ms=GOLDEN_DT_MS)
                    trains_I = generate_poisson_trains(
                        n_I, 20.0, total_ms, rng_k, dt_ms=GOLDEN_DT_MS)
                    kw["spike_trains_E"] = trains_E
                    kw["spike_trains_I"] = trains_I

                net, spike_mon, neurons, ie, ii = build_network(
                    regime_data, **kw)
                v_init = El / mV + rng_k.uniform(-2, 2, N)
                net.store()
                neurons.v = v_init * mV
                net.run(total_ms * ms)

                spike_trains = spike_mon.spike_trains()
                rE, rI = Metrics.firing_rates(
                    spike_trains, ie, ii, warmup_ms, total_ms)
                cv = Metrics.cv_isi(spike_trains, ie, warmup_ms, total_ms)
                syn = Metrics.sync_index(
                    spike_trains, N, warmup_ms, total_ms)

                rates_E.append(rE)
                rates_I.append(rI)
                cvs.append(cv)
                syncs.append(syn)

            modes[mode] = dict(
                rate_E=np.median(rates_E), rate_I=np.median(rates_I),
                cv_isi=np.median(cvs), sync=np.median(syncs),
            )

        det = modes["spike_generator"]
        poi = modes["poisson_group"]
        dr_E = abs(det["rate_E"] - poi["rate_E"])
        dr_I = abs(det["rate_I"] - poi["rate_I"])

        ok = dr_E < 5.0 and dr_I < 5.0  # allow ±5 Hz difference

        all_results[regime_name] = dict(det=det, poi=poi, ok=ok)

        print(f"  {regime_name:25s}")
        print(f"    SpikeGen:  rE={det['rate_E']:5.1f}  rI={det['rate_I']:5.1f}  "
              f"cv={det['cv_isi']:.2f}  sync={det['sync']:.2f}")
        print(f"    PoissonGr: rE={poi['rate_E']:5.1f}  rI={poi['rate_I']:5.1f}  "
              f"cv={poi['cv_isi']:.2f}  sync={poi['sync']:.2f}")
        print(f"    ΔrE={dr_E:.1f}  ΔrI={dr_I:.1f}  {'OK' if ok else 'WARN'}")

    # ── Save CSV ──
    csv_path = f"acceptance_B_crosscheck_seed{seed}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "mode", "rate_E", "rate_I", "cv_isi", "sync"])
        for rn, data in all_results.items():
            for mode_name, m in [("spike_generator", data["det"]),
                                  ("poisson_group", data["poi"])]:
                w.writerow([rn, mode_name, m["rate_E"], m["rate_I"],
                            m["cv_isi"], m["sync"]])
    print(f"\n  Saved: {csv_path}\n")

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# C) LAMBDA TARGETS REPORT
# ═════════════════════════════════════════════════════════════════════════════

def test_report(seed, separation_results=None, crosscheck_results=None):
    """Generate report with λ recommendations based on test A and B."""
    print("\n" + "=" * 70)
    print("  TEST C: LAMBDA TARGETS REPORT")
    print("=" * 70)

    if separation_results is None:
        print("  (run test A first for data-driven recommendations)\n")
        print("  Current RC targets:")
        for rn in REGIMES:
            lo, hi = LAMBDA_WINDOWS_RC[rn]
            tgt = LAMBDA_TARGETS_RC[rn]
            print(f"    {rn:25s}  target={tgt:5.1f}  window=[{lo}, {hi}]")
        print()
        return

    print(f"\n  Empirical λ distributions (n seeds from test A):\n")
    print(f"  {'Regime':25s}  {'Median':>7s}  {'P10':>6s}  {'P90':>6s}  "
          f"{'RC target':>9s}  {'In window?':>10s}")
    print(f"  {'-'*70}")

    recommendations = []
    for rn in REGIMES:
        if rn not in separation_results:
            continue
        r = separation_results[rn]
        tgt = LAMBDA_TARGETS_RC[rn]
        lo, hi = LAMBDA_WINDOWS_RC[rn]
        med = r["median"]
        in_window = lo <= med <= hi

        print(f"  {rn:25s}  {med:7.1f}  {r['p10']:6.1f}  {r['p90']:6.1f}  "
              f"{tgt:9.1f}  {'YES' if in_window else 'NO'}")

        # Suggest adjusted window based on empirical P10-P90
        margin = 0.15 * r["iqr"]  # 15% of IQR as margin
        new_lo = max(0, r["p10"] - margin)
        new_hi = r["p90"] + margin
        new_tgt = med
        recommendations.append((rn, new_tgt, new_lo, new_hi))

    print(f"\n  Data-driven recommendations (based on empirical distributions):\n")
    print(f"  {'Regime':25s}  {'New target':>10s}  {'New window':>16s}")
    print(f"  {'-'*55}")
    for rn, tgt, lo, hi in recommendations:
        print(f"  {rn:25s}  {tgt:10.1f}  [{lo:6.1f}, {hi:6.1f}]")

    # Check monotonicity of new targets
    tgts = [t for _, t, _, _ in recommendations]
    mono = all(tgts[i] < tgts[i + 1] for i in range(len(tgts) - 1))
    print(f"\n  New targets monotonic: {'PASS' if mono else 'FAIL'}")

    if crosscheck_results:
        det_ok = all(d.get("ok", False) for d in crosscheck_results.values())
        print(f"  SpikeGen vs PoissonGroup consistent: "
              f"{'PASS' if det_ok else 'WARN — check test B details'}")

    print()
    return recommendations


# ═════════════════════════════════════════════════════════════════════════════
# D) DT-STABILITY TEST
# ═════════════════════════════════════════════════════════════════════════════

def test_dt_stability(seed, npz_dir="regimes_calibrated", n_repeats=7,
                      dt_fine=0.0125, dt_coarse=0.025):
    """Compare λ at two dt_sim values using IDENTICAL spike trains.

    Spike trains are generated once at dt_train_ms = dt_coarse (safe for
    both dt_sim values).  The SAME trains are fed to Benettin at both
    dt_sim = dt_coarse and dt_sim = dt_fine.

    PASS criterion:
      λ < 10:  abs_diff ≤ 2.0
      λ ≥ 10:  rel_diff ≤ 15%
    """
    print("\n" + "=" * 70)
    print("  TEST D: DT-STABILITY (same trains, different dt_sim)")
    print("=" * 70)
    dt_train = dt_coarse
    print(f"  dt_coarse={dt_coarse}ms  dt_fine={dt_fine}ms  n_repeats={n_repeats}")
    print(f"  dt_train_ms={dt_train}ms (fixed for both — safe for coarser dt)")
    print(f"  PASS criterion: abs_diff≤2.0 when λ<10, rel_diff≤15% when λ≥10\n")

    test_regimes = ["R3_near_critical", "R5_chaotic"]
    all_pass = True

    for regime_name in test_regimes:
        npz_path = os.path.join(npz_dir, f"{regime_name}_seed{seed}.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP {regime_name}: not found")
            continue

        regime_data = load_regime_npz(npz_path)
        alpha = float(regime_data["alpha_final"])
        N = int(regime_data["N_total"])
        idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
        idx_I = np.array(regime_data["idx_I"], dtype=np.int32)
        n_E, n_I = len(idx_E), len(idx_I)

        test_alphas = [0.5 * alpha, alpha, 2.0 * alpha]
        total_ms = GOLDEN_WARMUP_MS + 1000.0

        print(f"  {regime_name} (alpha_cal={alpha:.6f}):")
        print(f"    {'alpha':>12s}  {'lam_coarse':>10s}  {'lam_fine':>10s}  "
              f"{'criterion':>18s}  {'status':>6s}")
        print(f"    {'-'*62}")

        for a in test_alphas:
            lams_coarse, lams_fine = [], []
            for r in range(n_repeats):
                rng_train = np.random.default_rng(seed + 7000 + r * 100)
                trains_E = generate_poisson_trains(
                    n_E, 20.0, total_ms, rng_train, dt_train_ms=dt_train)
                trains_I = generate_poisson_trains(
                    n_I, 20.0, total_ms, rng_train, dt_train_ms=dt_train)

                rng_c = np.random.default_rng(seed + 7500 + r * 100)
                lc = measure_lambda_benettin(
                    regime_data, a, 0.05, rng_c,
                    warmup_ms=GOLDEN_WARMUP_MS, measure_ms=1000.0,
                    renorm_ms=GOLDEN_RENORM_MS, eps_mV=GOLDEN_EPS_MV,
                    dt_ms=dt_coarse,
                    trains_E=trains_E, trains_I=trains_I,
                )
                lams_coarse.append(lc)

                rng_f = np.random.default_rng(seed + 7500 + r * 100)
                lf = measure_lambda_benettin(
                    regime_data, a, 0.05, rng_f,
                    warmup_ms=GOLDEN_WARMUP_MS, measure_ms=1000.0,
                    renorm_ms=GOLDEN_RENORM_MS, eps_mV=GOLDEN_EPS_MV,
                    dt_ms=dt_fine,
                    trains_E=trains_E, trains_I=trains_I,
                )
                lams_fine.append(lf)

            mc = np.median(lams_coarse)
            mf = np.median(lams_fine)
            abs_d = abs(mc - mf)
            ref_lam = min(abs(mc), abs(mf))

            if ref_lam < 10:
                ok = abs_d <= 2.0
                crit_str = f"abs_diff={abs_d:.2f}≤2.0"
            else:
                rel = abs_d / (abs(mf) + 1e-10)
                ok = rel <= 0.15
                crit_str = f"rel_diff={rel:.1%}≤15%"

            if not ok:
                all_pass = False

            print(f"    {a:12.6f}  {mc:10.2f}  {mf:10.2f}  "
                  f"{crit_str:>18s}  {'OK' if ok else 'FAIL'}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  DT-STABILITY: {status}\n")
    return all_pass


# ═════════════════════════════════════════════════════════════════════════════
# E) EPS-INVARIANCE TEST
# ═════════════════════════════════════════════════════════════════════════════

def test_eps_invariance(seed, npz_dir="regimes_calibrated", n_repeats=7):
    """Compare λ at eps=0.02, 0.05, 0.1 mV — PAIRED.

    For each repeat, generates v_init and delta_dir ONCE, then measures
    lambda with all three eps values using the same initial conditions,
    same trains, and same perturbation direction.  This isolates the
    effect of perturbation amplitude from initial-condition variance.

    PASS criterion: spread / median ≤ 25%.
    """
    print("\n" + "=" * 70)
    print("  TEST E: EPS-INVARIANCE (paired: same v_init + delta_dir + trains)")
    print("=" * 70)

    eps_values = [0.02, 0.05, 0.1]
    print(f"  eps values: {eps_values} mV  n_repeats={n_repeats}")
    print(f"  dt_lambda={GOLDEN_DT_LAMBDA_MS}ms  renorm={GOLDEN_RENORM_MS}ms")
    print(f"  PASS criterion: spread / median ≤ 25%\n")

    test_regimes = ["R3_near_critical", "R5_chaotic"]
    all_pass = True

    for regime_name in test_regimes:
        npz_path = os.path.join(npz_dir, f"{regime_name}_seed{seed}.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP {regime_name}: not found")
            continue

        regime_data = load_regime_npz(npz_path)
        alpha = float(regime_data["alpha_final"])
        N = int(regime_data["N_total"])
        idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
        idx_I = np.array(regime_data["idx_I"], dtype=np.int32)
        n_E, n_I = len(idx_E), len(idx_I)
        total_ms = GOLDEN_WARMUP_MS + GOLDEN_MEASURE_MS

        print(f"  {regime_name} (alpha={alpha:.6f}):")
        print(f"    {'eps_mV':>8s}  {'median_lam':>10s}  {'std':>8s}  {'values'}")
        print(f"    {'-'*55}")

        # Pre-generate trains, v_init, delta_dir ONCE per repeat
        all_trains = []
        all_v_init = []
        all_delta_dir = []
        for r in range(n_repeats):
            rng_train = np.random.default_rng(seed + 8000 + r * 100)
            tE = generate_poisson_trains(n_E, 20.0, total_ms, rng_train,
                                         dt_train_ms=GOLDEN_DT_MS)
            tI = generate_poisson_trains(n_I, 20.0, total_ms, rng_train,
                                         dt_train_ms=GOLDEN_DT_MS)
            all_trains.append((tE, tI))

            rng_ic = np.random.default_rng(seed + 8500 + r * 100)
            v_init = El / mV + rng_ic.uniform(-2, 2, N)
            delta_dir = rng_ic.normal(0, 1.0, N)
            all_v_init.append(v_init)
            all_delta_dir.append(delta_dir)

        medians = []
        for eps in eps_values:
            lams = []
            for r in range(n_repeats):
                trains_E, trains_I = all_trains[r]
                rng_unused = np.random.default_rng(0)
                lam = measure_lambda_benettin(
                    regime_data, alpha, 0.05, rng_unused,
                    warmup_ms=GOLDEN_WARMUP_MS, measure_ms=GOLDEN_MEASURE_MS,
                    renorm_ms=GOLDEN_RENORM_MS, eps_mV=eps,
                    dt_ms=GOLDEN_DT_LAMBDA_MS,
                    trains_E=trains_E, trains_I=trains_I,
                    v_init_mV=all_v_init[r],
                    delta_dir=all_delta_dir[r],
                )
                lams.append(lam)

            med = np.median(lams)
            std = np.std(lams)
            medians.append(med)
            vals = ", ".join(f"{l:.1f}" for l in lams)
            print(f"    {eps:8.3f}  {med:10.2f}  {std:8.2f}  {vals}")

        # Check: spread / median ≤ 25%
        med_of_meds = np.median(medians)
        spread = max(medians) - min(medians)
        rel_spread = spread / (abs(med_of_meds) + 1e-10)
        ok = rel_spread <= 0.25
        if not ok:
            all_pass = False

        print(f"    spread={spread:.2f}  rel={rel_spread:.1%}  "
              f"{'OK' if ok else 'FAIL'}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  EPS-INVARIANCE: {status}\n")
    return all_pass


# ═════════════════════════════════════════════════════════════════════════════
# F) TIME-LENGTH CONVERGENCE TEST
# ═════════════════════════════════════════════════════════════════════════════

def test_time_convergence(seed, npz_dir="regimes_calibrated", n_repeats=7):
    """Compare median λ at measure_ms=500, 1000, 2000 using SAME trains.

    Trains are generated for the longest duration (2000ms).  Shorter
    durations use the same trains (Benettin just runs fewer windows).

    PASS criterion: |λ(1000) - λ(2000)| / |λ(2000)| ≤ 15%.
    """
    print("\n" + "=" * 70)
    print("  TEST F: TIME-LENGTH CONVERGENCE (same trains)")
    print("=" * 70)

    durations = [500.0, 1000.0, 2000.0]
    max_dur = max(durations)
    print(f"  measure_ms values: {durations}  n_repeats={n_repeats}")
    print(f"  dt_lambda={GOLDEN_DT_LAMBDA_MS}ms  renorm={GOLDEN_RENORM_MS}ms")
    print(f"  PASS criterion: |λ(1000) - λ(2000)| / |λ(2000)| ≤ 15%\n")

    test_regimes = ["R3_near_critical", "R5_chaotic"]
    all_pass = True

    for regime_name in test_regimes:
        npz_path = os.path.join(npz_dir, f"{regime_name}_seed{seed}.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP {regime_name}: not found")
            continue

        regime_data = load_regime_npz(npz_path)
        alpha = float(regime_data["alpha_final"])
        N = int(regime_data["N_total"])
        idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
        idx_I = np.array(regime_data["idx_I"], dtype=np.int32)
        n_E, n_I = len(idx_E), len(idx_I)
        total_ms_max = GOLDEN_WARMUP_MS + max_dur

        # Pre-generate trains for the longest duration
        all_trains = []
        for r in range(n_repeats):
            rng_train = np.random.default_rng(seed + 9000 + r * 100)
            tE = generate_poisson_trains(n_E, 20.0, total_ms_max, rng_train,
                                         dt_train_ms=GOLDEN_DT_MS)
            tI = generate_poisson_trains(n_I, 20.0, total_ms_max, rng_train,
                                         dt_train_ms=GOLDEN_DT_MS)
            all_trains.append((tE, tI))

        print(f"  {regime_name} (alpha={alpha:.6f}):")
        print(f"    {'measure_ms':>10s}  {'median_lam':>10s}  {'std':>8s}  {'values'}")
        print(f"    {'-'*55}")

        medians = {}
        for dur in durations:
            lams = []
            for r in range(n_repeats):
                trains_E, trains_I = all_trains[r]
                rng_pert = np.random.default_rng(seed + 9500 + r * 100)
                lam = measure_lambda_benettin(
                    regime_data, alpha, 0.05, rng_pert,
                    warmup_ms=GOLDEN_WARMUP_MS, measure_ms=dur,
                    renorm_ms=GOLDEN_RENORM_MS, eps_mV=GOLDEN_EPS_MV,
                    dt_ms=GOLDEN_DT_LAMBDA_MS,
                    trains_E=trains_E, trains_I=trains_I,
                )
                lams.append(lam)

            med = np.median(lams)
            std = np.std(lams)
            medians[dur] = med
            vals = ", ".join(f"{l:.1f}" for l in lams)
            print(f"    {dur:10.0f}  {med:10.2f}  {std:8.2f}  {vals}")

        # Primary check: 1000 vs 2000 (rel_diff ≤ 15%)
        if 2000.0 in medians and 1000.0 in medians:
            ref = medians[2000.0]
            rel_1k = abs(medians[1000.0] - ref) / (abs(ref) + 1e-10)
            ok_1k = rel_1k <= 0.15
            print(f"    |λ(1000)-λ(2000)|/λ(2000) = {rel_1k:.1%}  "
                  f"{'OK' if ok_1k else 'FAIL'}")
            if not ok_1k:
                all_pass = False

        # Info: 500 vs 2000
        if 2000.0 in medians and 500.0 in medians:
            ref = medians[2000.0]
            rel_500 = abs(medians[500.0] - ref) / (abs(ref) + 1e-10)
            ok_500 = rel_500 <= 0.15
            print(f"    |λ(500)-λ(2000)|/λ(2000)  = {rel_500:.1%}  "
                  f"{'OK' if ok_500 else 'INFO — 500ms may be too short'}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  TIME-LENGTH CONVERGENCE: {status}\n")
    return all_pass


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Acceptance tests for calibrated regimes")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--test", default="all",
                    help="Which test(s) to run: A,B,C,D,E,F or 'all' (default: all)")
    ap.add_argument("--n-seeds", type=int, default=GOLDEN_N_SEEDS,
                    help="Number of fresh Poisson seeds for test A")
    ap.add_argument("--npz-dir", default="regimes_calibrated")
    args = ap.parse_args()

    tests = args.test.upper()

    t0 = time.time()

    sep_results = None
    cross_results = None

    if tests == "ALL" or "A" in tests:
        sep_results, sep_pass = test_separation(
            args.seed, npz_dir=args.npz_dir, n_seeds=args.n_seeds)

    if tests == "ALL" or "B" in tests:
        cross_results = test_crosscheck(
            args.seed, npz_dir=args.npz_dir)

    if tests == "ALL" or "C" in tests:
        test_report(args.seed, sep_results, cross_results)

    if tests == "ALL" or "D" in tests:
        test_dt_stability(args.seed, npz_dir=args.npz_dir)

    if tests == "ALL" or "E" in tests:
        test_eps_invariance(args.seed, npz_dir=args.npz_dir)

    if tests == "ALL" or "F" in tests:
        test_time_convergence(args.seed, npz_dir=args.npz_dir)

    print(f"Total time: {time.time() - t0:.0f}s")
