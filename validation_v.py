#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation V runner — adapted for HH+STP reservoir pipeline.

Modes:
  - full:  Phase 1 (n_validation × validation_eval_ms) + Phase 2 (final_reps × final_eval_ms)
  - fast:  DT-stability + TIME-length ONLY at alpha_calibrated (or alpha_final fallback),
           with fixed trains + fixed IC + fixed delta_dir per seed (low-variance numeric gate)

FAST is intended to be "benchmark-ready gate" without multi-hour acceptance suites.

Usage:
  python validation_v.py --regime R3_near_critical --seed 100 --mode fast
  python validation_v.py --regime all --seed 100 --mode full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Imports from our HH+STP reservoir codebase ──
from regime_calibrator import (
    load_regime_npz,
    measure_lambda_benettin as _measure_lambda_raw,
    generate_poisson_trains,
    DT_MS, DT_LAMBDA_MS, El, mV, ms,
    # HH constants must be in caller namespace for Brian2 net.run()
    EK, ENa, g_na, g_kd, gl, Cm, VT,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Thresholds:
    """PASS/FAIL criteria — adapted for HH+STP λ range (1–60 /s)."""
    lambda_small_thr: float = 10.0   # |λ_ref| < 10 → abs_diff
    abs_diff_thr: float = 2.0        # PASS if |Δ| ≤ 2.0 (small λ)
    rel_diff_thr: float = 0.15       # PASS if |Δ|/|ref| ≤ 15% (large λ)
    rel_eps: float = 1e-12           # division guard


@dataclass(frozen=True)
class RunParams:
    dt_train_ms: float = 0.025       # Poisson binning (≥ max dt_sim for safety)
    dt_lambda_ms: float = 0.0125     # Benettin integration step (fine)
    renorm_ms: float = 5.0           # renormalization interval
    warmup_ms: float = 200.0
    eps_mV: float = 0.05
    n_validation: int = 7
    final_reps: int = 9
    final_eval_ms: float = 3000.0
    validation_eval_ms: float = 2000.0


@dataclass
class RunResult:
    seeds: List[int]
    lambdas: List[float]
    median: float
    mean: float
    std: float
    stderr: float


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _compute_stats(seeds: List[int], vals: List[float]) -> RunResult:
    arr = np.asarray(vals, dtype=float)
    return RunResult(
        seeds=seeds,
        lambdas=[float(v) for v in arr],
        median=float(np.median(arr)) if arr.size else float("nan"),
        mean=float(arr.mean()) if arr.size else float("nan"),
        std=float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
        stderr=float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size >= 2 else 0.0,
    )


def _pass_fail(
    lambda_ref: float, lambda_new: float, thr: Thresholds
) -> Tuple[bool, Dict[str, float], str]:
    abs_diff = abs(lambda_new - lambda_ref)
    ref_abs = abs(lambda_ref)

    if ref_abs < thr.lambda_small_thr:
        ok = abs_diff <= thr.abs_diff_thr
        rule = "abs_diff"
        metric = abs_diff
        limit = thr.abs_diff_thr
    else:
        rel_diff = abs_diff / max(ref_abs, thr.rel_eps)
        ok = rel_diff <= thr.rel_diff_thr
        rule = "rel_diff"
        metric = rel_diff
        limit = thr.rel_diff_thr

    details = {
        "lambda_ref": float(lambda_ref),
        "lambda_new": float(lambda_new),
        "abs_diff": float(abs_diff),
        "metric": float(metric),
        "limit": float(limit),
    }
    return ok, details, rule


def _load_ref(ref_path: Optional[str]) -> Optional[float]:
    if not ref_path:
        return None
    p = Path(ref_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    for k in ("lambda_ref", "median", "mean", "lambda"):
        if k in data:
            return float(data[k])
    raise ValueError(f"No lambda_ref/median/mean/lambda key in: {p}")


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_alpha_from_npz(regime_data: dict) -> float:
    # Prefer calibrated alpha if present, otherwise fallback.
    if "alpha_calibrated" in regime_data:
        return float(regime_data["alpha_calibrated"])
    return float(regime_data["alpha_final"])


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic preparation (kills variance in numeric tests)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PreparedInputs:
    trains_E: Tuple[np.ndarray, np.ndarray]
    trains_I: Tuple[np.ndarray, np.ndarray]
    v_init_mV: np.ndarray
    delta_dir: np.ndarray


def prepare_inputs(
    regime_data: dict,
    seed: int,
    rp: RunParams,
    eval_ms_max: float,
    *,
    delta_dir_seed_base: int = 777,
) -> PreparedInputs:
    """
    Prepare deterministic trains + initial state + perturbation direction.
    This is the core trick for fast numeric validation: dt/time comparisons
    use identical inputs → much lower noise.
    """
    N = int(regime_data["N_total"])
    idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
    idx_I = np.array(regime_data["idx_I"], dtype=np.int32)
    n_E, n_I = len(idx_E), len(idx_I)

    total_ms = rp.warmup_ms + float(eval_ms_max)

    # Trains: deterministic per seed
    rng_train = np.random.default_rng(seed)
    trains_E = generate_poisson_trains(n_E, 20.0, total_ms, rng_train, dt_train_ms=rp.dt_train_ms)
    trains_I = generate_poisson_trains(n_I, 20.0, total_ms, rng_train, dt_train_ms=rp.dt_train_ms)

    # IC + delta_dir: deterministic per seed (NOT per index i)
    rng_ic = np.random.default_rng(delta_dir_seed_base + seed)
    v_init_mV = float(El / mV) + rng_ic.uniform(-2, 2, N)
    delta_dir = rng_ic.normal(0.0, 1.0, N)

    return PreparedInputs(
        trains_E=trains_E,
        trains_I=trains_I,
        v_init_mV=v_init_mV.astype(np.float64),
        delta_dir=delta_dir.astype(np.float64),
    )


def measure_lambda_prepared(
    regime_data: dict,
    alpha: float,
    seed: int,
    rp: RunParams,
    eval_ms: float,
    *,
    dt_sim_ms: float,
    prepared: PreparedInputs,
) -> float:
    """
    Single Benettin measurement using pre-generated trains + IC + delta_dir.
    """
    rng_pert = np.random.default_rng(seed + 500_000)  # kept for API compatibility
    return float(_measure_lambda_raw(
        regime_data, float(alpha), 0.05, rng_pert,
        warmup_ms=float(rp.warmup_ms),
        measure_ms=float(eval_ms),
        renorm_ms=float(rp.renorm_ms),
        eps_mV=float(rp.eps_mV),
        dt_ms=float(dt_sim_ms),
        dt_train_ms=float(rp.dt_train_ms),
        trains_E=prepared.trains_E, trains_I=prepared.trains_I,
        v_init_mV=prepared.v_init_mV, delta_dir=prepared.delta_dir,
    ))


# ═══════════════════════════════════════════════════════════════════════════
# Full runner (kept mostly as-is)
# ═══════════════════════════════════════════════════════════════════════════

def run_batch_full(
    regime_data: dict,
    alpha: float,
    seeds: List[int],
    rp: RunParams,
    eval_ms: float,
    *,
    paired: bool = True,
    delta_dir_seed_base: int = 777,
) -> RunResult:
    """
    FULL mode batch. Uses pre-generated trains per seed but does not compare dt/time.
    Kept for completeness, but for benchmark gate use FAST mode instead.
    """
    lambdas: List[float] = []
    eval_ms = float(eval_ms)

    for seed in seeds:
        prepared = prepare_inputs(regime_data, seed, rp, eval_ms, delta_dir_seed_base=delta_dir_seed_base)
        lam = measure_lambda_prepared(
            regime_data, alpha, seed, rp, eval_ms,
            dt_sim_ms=rp.dt_lambda_ms,
            prepared=prepared,
        )
        lambdas.append(lam)
        print(f"      seed={seed}  λ={lam:.2f}")

    return _compute_stats(seeds, lambdas)


def validate_regime_full(
    regime_name: str,
    regime_seed: int,
    npz_dir: str,
    rp: RunParams,
    thr: Thresholds,
    seed_base: int,
    delta_dir_seed_base: int,
    lambda_ref: Optional[float],
    run_dir: Path,
) -> Dict[str, Any]:
    npz_path = os.path.join(npz_dir, f"{regime_name}_seed{regime_seed}.npz")
    if not os.path.exists(npz_path):
        print(f"  SKIP {regime_name}: {npz_path} not found")
        return {"regime": regime_name, "status": "SKIP"}

    regime_data = load_regime_npz(npz_path)
    alpha = _get_alpha_from_npz(regime_data)

    print(f"\n{'='*60}")
    print(f"  {regime_name}  alpha={alpha:.6f}  (FULL)")
    print(f"{'='*60}")

    seeds_val = [seed_base + i for i in range(rp.n_validation)]
    print(f"\n  Phase 1: validation  n={rp.n_validation}  eval_ms={rp.validation_eval_ms}")
    val_stats = run_batch_full(
        regime_data, alpha, seeds_val, rp,
        eval_ms=rp.validation_eval_ms,
        paired=True,
        delta_dir_seed_base=delta_dir_seed_base,
    )
    print(f"  → median={val_stats.median:.2f}  mean={val_stats.mean:.2f}  "
          f"std={val_stats.std:.2f}  stderr={val_stats.stderr:.2f}")

    seeds_final = [seed_base + 10_000 + i for i in range(rp.final_reps)]
    print(f"\n  Phase 2: final verification  n={rp.final_reps}  eval_ms={rp.final_eval_ms}")
    final_stats = run_batch_full(
        regime_data, alpha, seeds_final, rp,
        eval_ms=rp.final_eval_ms,
        paired=True,
        delta_dir_seed_base=delta_dir_seed_base + 10_000,
    )
    print(f"  → median={final_stats.median:.2f}  mean={final_stats.mean:.2f}  "
          f"std={final_stats.std:.2f}  stderr={final_stats.stderr:.2f}")

    verdict: Optional[Dict[str, Any]] = None
    if lambda_ref is not None:
        ok, details, rule = _pass_fail(lambda_ref, final_stats.median, thr)
        verdict = {"passed": bool(ok), "rule": rule, **details}
        status = "PASS" if ok else "FAIL"
        print(f"\n  vs ref (λ_ref={lambda_ref:.2f}): {status}  "
              f"rule={rule}  metric={details['metric']:.4f}  limit={details['limit']}")
    else:
        print(f"\n  No ref → PASS/FAIL skipped (stats only).")

    payload = {
        "mode": "full",
        "regime": regime_name,
        "alpha": alpha,
        "run_params": asdict(rp),
        "thresholds": asdict(thr),
        "lambda_ref": lambda_ref,
        "validation": asdict(val_stats),
        "final": asdict(final_stats),
        "verdict": verdict,
    }
    _save_json(run_dir / f"{regime_name}_summary.json", payload)

    for name, stats in [("validation", val_stats), ("final", final_stats)]:
        csv_path = run_dir / f"{regime_name}_{name}_lambdas.csv"
        csv_path.write_text(
            "seed,lambda\n" +
            "\n".join(f"{s},{l}" for s, l in zip(stats.seeds, stats.lambdas)),
            encoding="utf-8",
        )

    return payload


# ═══════════════════════════════════════════════════════════════════════════
# FAST numeric gate (dt + time only at alpha_calibrated)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FastConfig:
    dt_coarse_ms: float = 0.0125     # was 0.025 — too coarse for HH+STP
    dt_fine_ms: float = 0.00625      # real convergence test: 0.0125 vs 0.00625
    eval_ms_short: float = 800.0
    eval_ms_long: float = 1600.0
    n_seeds: int = 2
    # criterion uses Thresholds, with ref = "better" (fine dt or longer time)


def validate_regime_fast(
    regime_name: str,
    regime_seed: int,
    npz_dir: str,
    rp: RunParams,
    thr: Thresholds,
    seed_base: int,
    delta_dir_seed_base: int,
    run_dir: Path,
    fast: FastConfig,
) -> Dict[str, Any]:
    npz_path = os.path.join(npz_dir, f"{regime_name}_seed{regime_seed}.npz")
    if not os.path.exists(npz_path):
        print(f"  SKIP {regime_name}: {npz_path} not found")
        return {"regime": regime_name, "status": "SKIP"}

    # In FAST mode, skip R1 as a gate: λ≈0 and sign flips are expected.
    if regime_name == "R1_super_stable":
        print(f"\n{'='*60}")
        print(f"  {regime_name}: FAST gate skipped (λ≈0 regime)")
        print(f"{'='*60}")
        return {"mode": "fast", "regime": regime_name, "status": "SKIP_FAST_R1"}

    regime_data = load_regime_npz(npz_path)
    alpha = _get_alpha_from_npz(regime_data)

    seeds = [seed_base + i for i in range(int(fast.n_seeds))]

    print(f"\n{'='*60}")
    print(f"  {regime_name}  alpha={alpha:.6f}  (FAST)")
    print(f"{'='*60}")
    print(f"  Seeds: {seeds}")
    print(f"  DT test: coarse={fast.dt_coarse_ms}  fine={fast.dt_fine_ms}  eval={fast.eval_ms_short}ms")
    print(f"  TIME test: short={fast.eval_ms_short}  long={fast.eval_ms_long}  dt={fast.dt_fine_ms}")

    # Prepare once per seed for max duration needed
    eval_ms_max = float(max(fast.eval_ms_long, fast.eval_ms_short))
    prepared_per_seed: Dict[int, PreparedInputs] = {}
    for s in seeds:
        prepared_per_seed[s] = prepare_inputs(
            regime_data, s, rp, eval_ms_max,
            delta_dir_seed_base=delta_dir_seed_base
        )

    # ── Test D': DT stability at alpha_cal ──
    dt_rows = []
    dt_pass = 0
    for s in seeds:
        prep = prepared_per_seed[s]
        lam_fine = measure_lambda_prepared(
            regime_data, alpha, s, rp, fast.eval_ms_short,
            dt_sim_ms=fast.dt_fine_ms, prepared=prep
        )
        lam_coarse = measure_lambda_prepared(
            regime_data, alpha, s, rp, fast.eval_ms_short,
            dt_sim_ms=fast.dt_coarse_ms, prepared=prep
        )
        ok, details, rule = _pass_fail(lam_fine, lam_coarse, thr)  # ref = fine dt
        dt_pass += int(ok)
        dt_rows.append({
            "seed": s,
            "lambda_fine": lam_fine,
            "lambda_coarse": lam_coarse,
            "passed": bool(ok),
            "rule": rule,
            **details,
        })
        print(f"    [DT] seed={s}  fine={lam_fine:.2f}  coarse={lam_coarse:.2f}  "
              f"{'PASS' if ok else 'FAIL'}  rule={rule}  metric={details['metric']:.4f}")

    # ── Test F': TIME-length convergence at alpha_cal (dt=fine) ──
    time_rows = []
    time_pass = 0
    for s in seeds:
        prep = prepared_per_seed[s]
        lam_long = measure_lambda_prepared(
            regime_data, alpha, s, rp, fast.eval_ms_long,
            dt_sim_ms=fast.dt_fine_ms, prepared=prep
        )
        lam_short = measure_lambda_prepared(
            regime_data, alpha, s, rp, fast.eval_ms_short,
            dt_sim_ms=fast.dt_fine_ms, prepared=prep
        )
        ok, details, rule = _pass_fail(lam_long, lam_short, thr)  # ref = longer time
        time_pass += int(ok)
        time_rows.append({
            "seed": s,
            "lambda_long": lam_long,
            "lambda_short": lam_short,
            "passed": bool(ok),
            "rule": rule,
            **details,
        })
        print(f"    [TIME] seed={s}  long={lam_long:.2f}  short={lam_short:.2f}  "
              f"{'PASS' if ok else 'FAIL'}  rule={rule}  metric={details['metric']:.4f}")

    # Majority rule for FAST gate
    dt_ok = dt_pass >= (len(seeds) // 2 + 1)
    time_ok = time_pass >= (len(seeds) // 2 + 1)
    overall_ok = dt_ok and time_ok

    print(f"\n  FAST gate summary: DT {'PASS' if dt_ok else 'FAIL'} ({dt_pass}/{len(seeds)}), "
          f"TIME {'PASS' if time_ok else 'FAIL'} ({time_pass}/{len(seeds)}), "
          f"OVERALL {'PASS' if overall_ok else 'FAIL'}")

    payload = {
        "mode": "fast",
        "regime": regime_name,
        "alpha": alpha,
        "run_params": asdict(rp),
        "thresholds": asdict(thr),
        "fast_config": asdict(fast),
        "dt_test": {
            "passed": bool(dt_ok),
            "passed_count": int(dt_pass),
            "total": int(len(seeds)),
            "rows": dt_rows,
        },
        "time_test": {
            "passed": bool(time_ok),
            "passed_count": int(time_pass),
            "total": int(len(seeds)),
            "rows": time_rows,
        },
        "overall_passed": bool(overall_ok),
    }

    _save_json(run_dir / f"{regime_name}_summary.json", payload)

    # CSV exports
    dt_csv = run_dir / f"{regime_name}_DT.csv"
    dt_csv.write_text(
        "seed,lambda_fine,lambda_coarse,passed,rule,metric,limit,abs_diff,lambda_ref,lambda_new\n" +
        "\n".join(
            f"{r['seed']},{r['lambda_fine']:.6f},{r['lambda_coarse']:.6f},{int(r['passed'])},{r['rule']},"
            f"{r['metric']:.6f},{r['limit']:.6f},{r['abs_diff']:.6f},{r['lambda_ref']:.6f},{r['lambda_new']:.6f}"
            for r in dt_rows
        ),
        encoding="utf-8",
    )
    time_csv = run_dir / f"{regime_name}_TIME.csv"
    time_csv.write_text(
        "seed,lambda_long,lambda_short,passed,rule,metric,limit,abs_diff,lambda_ref,lambda_new\n" +
        "\n".join(
            f"{r['seed']},{r['lambda_long']:.6f},{r['lambda_short']:.6f},{int(r['passed'])},{r['rule']},"
            f"{r['metric']:.6f},{r['limit']:.6f},{r['abs_diff']:.6f},{r['lambda_ref']:.6f},{r['lambda_new']:.6f}"
            for r in time_rows
        ),
        encoding="utf-8",
    )

    return payload


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

ALL_REGIMES = [
    "R1_super_stable", "R2_stable", "R3_near_critical",
    "R4_edge_of_chaos", "R5_chaotic",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validation V runner for HH+STP reservoir")

    ap.add_argument("--mode", choices=["full", "fast"], default="fast",
                    help="Validation mode: fast numeric gate (default) or full long run")

    ap.add_argument("--regime", required=True,
                    help="Regime name (e.g. R3_near_critical) or 'all'")
    ap.add_argument("--seed", type=int, default=100,
                    help="Regime seed (for NPZ file lookup)")

    ap.add_argument("--npz-dir", default="regimes_calibrated")
    ap.add_argument("--outdir", default="validation_out")

    # Core parameters
    ap.add_argument("--dt-train-ms", type=float,
                    default=float(os.getenv("DT_TRAIN_MS", "0.025")))
    ap.add_argument("--dt-lambda-ms", type=float,
                    default=float(os.getenv("DT_LAMBDA_MS", "0.0125")))
    ap.add_argument("--renorm-ms", type=float,
                    default=float(os.getenv("RENORM_MS", "5")))
    ap.add_argument("--eps-mV", type=float, default=0.05)
    ap.add_argument("--warmup-ms", type=float, default=200.0)

    # FULL repetitions
    ap.add_argument("--n-validation", type=int, default=7)
    ap.add_argument("--final-reps", type=int, default=9)
    ap.add_argument("--final-eval-ms", type=float, default=3000.0)
    ap.add_argument("--validation-eval-ms", type=float, default=2000.0)

    # Thresholds
    ap.add_argument("--lambda-small-thr", type=float, default=10.0)
    ap.add_argument("--abs-diff-thr", type=float, default=2.0)
    ap.add_argument("--rel-diff-thr", type=float, default=0.15)

    # Reference (FULL only; FAST uses internal ref = "better")
    ap.add_argument("--ref-json", default=None,
                    help="JSON with lambda_ref (for FULL PASS/FAIL comparison)")

    # Seed control
    ap.add_argument("--seed-base", type=int, default=12345)
    ap.add_argument("--delta-dir-seed-base", type=int, default=777)

    # FAST config
    ap.add_argument("--fast-n-seeds", type=int, default=2)
    ap.add_argument("--fast-eval-ms-short", type=float, default=800.0)
    ap.add_argument("--fast-eval-ms-long", type=float, default=1600.0)
    ap.add_argument("--fast-dt-coarse-ms", type=float, default=0.0125)
    ap.add_argument("--fast-dt-fine-ms", type=float, default=0.00625)

    args = ap.parse_args()

    thr = Thresholds(
        lambda_small_thr=args.lambda_small_thr,
        abs_diff_thr=args.abs_diff_thr,
        rel_diff_thr=args.rel_diff_thr,
    )
    rp = RunParams(
        dt_train_ms=args.dt_train_ms,
        dt_lambda_ms=args.dt_lambda_ms,
        renorm_ms=args.renorm_ms,
        warmup_ms=args.warmup_ms,
        eps_mV=args.eps_mV,
        n_validation=args.n_validation,
        final_reps=args.final_reps,
        final_eval_ms=args.final_eval_ms,
        validation_eval_ms=args.validation_eval_ms,
    )

    outdir = Path(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"V_{stamp}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    lambda_ref = _load_ref(args.ref_json)

    fast = FastConfig(
        dt_coarse_ms=args.fast_dt_coarse_ms,
        dt_fine_ms=args.fast_dt_fine_ms,
        eval_ms_short=args.fast_eval_ms_short,
        eval_ms_long=args.fast_eval_ms_long,
        n_seeds=args.fast_n_seeds,
    )

    print("═" * 60)
    print("  VALIDATION V")
    print(f"  mode={args.mode}  regime={args.regime}  seed={args.seed}")
    print(f"  params: {asdict(rp)}")
    print(f"  thresholds: {asdict(thr)}")
    if args.mode == "fast":
        print(f"  fast: {asdict(fast)}")
    print("═" * 60)

    t0 = time.time()

    regimes = ALL_REGIMES if args.regime.lower() == "all" else [args.regime]
    all_results = []

    for regime_name in regimes:
        if args.mode == "fast":
            result = validate_regime_fast(
                regime_name=regime_name,
                regime_seed=args.seed,
                npz_dir=args.npz_dir,
                rp=rp,
                thr=thr,
                seed_base=args.seed_base,
                delta_dir_seed_base=args.delta_dir_seed_base,
                run_dir=run_dir,
                fast=fast,
            )
        else:
            result = validate_regime_full(
                regime_name=regime_name,
                regime_seed=args.seed,
                npz_dir=args.npz_dir,
                rp=rp,
                thr=thr,
                seed_base=args.seed_base,
                delta_dir_seed_base=args.delta_dir_seed_base,
                lambda_ref=lambda_ref,
                run_dir=run_dir,
            )

        all_results.append(result)

    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    _save_json(run_dir / "run_index.json", {"results": all_results})

    print(f"\n{'═'*60}")
    print(f"  VALIDATION V COMPLETE  time={mins}m{secs}s")
    print(f"  Results in: {run_dir}/")
    print(f"{'═'*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
