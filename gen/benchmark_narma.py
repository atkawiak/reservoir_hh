#!/usr/bin/env python3
"""NARMA-10 benchmark for frozen LSM reservoir bundles.

End-to-end pipeline:
    1. Generate NARMA-10 input/target signal
    2. Run Brian2 simulation with NARMA current injection
    3. Extract spike count states from reservoir
    4. Train ridge regression readout
    5. Evaluate NRMSE on held-out test set
    6. Save results to bundle/benchmarks/narma10/

Usage:
    python -m gen.benchmark_narma bundles/bundle_seed_14896 --regime 2
    python -m gen.benchmark_narma bundles/bundle_seed_14896 --regime 0 1 2 3 4 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .narma import generate_narma10, NARMA_ORDER
from .state_readout import extract_spike_counts, extract_spike_counts_ei, delay_embed
from .ridge import ridge_cv_fit, ridge_predict
from .metrics import nrmse, rmse, r_squared


@dataclass
class NarmaConfig:
    """Configuration for one NARMA-10 benchmark run."""
    # Signal generation
    K_total: int = 500         # total NARMA steps (warmup + train + test)
    K_warmup: int = 100        # steps to discard (reservoir + NARMA transient)
    K_train: int = 300         # training steps
    K_test: int = 100          # test steps
    narma_seed: int = 42       # seed for NARMA input u

    # Brian2 simulation
    dt_task_ms: float = 10.0   # NARMA step duration in Brian2 time (ms)
    brian_warmup_ms: float = 0.0  # extra Brian2 warmup before NARMA starts
    narma_scale_nA: float = 0.5   # current scaling: I = u * scale (nA)
    bg_scale: float = 1.0      # Poisson background current multiplier
    dt_ms: float = 0.025      # Brian2 integration timestep

    # State extraction
    n_delays: int = 10         # time-delay embedding depth (0 = no delays)
    ei_split: bool = False     # if True, use [X_E, X_I] with separate normalization

    # Readout
    ridge_alphas: list = None  # regularization grid (None → default)
    ridge_cv_folds: int = 5
    ridge_seed: int = 0


def _tdl_baseline(
    u: np.ndarray,
    y: np.ndarray,
    D: int,
    K_warmup: int,
    K_train: int,
    K_test: int,
    ridge_cv_folds: int = 5,
    ridge_seed: int = 0,
) -> dict:
    """Tapped Delay Line baseline: ridge on [u(t-1),..,u(t-D)].

    Uses exactly the same train/test split and standardization as the RC pipeline.
    Computed once per (narma_seed, n_delays, splits) — independent of brian seed.

    Returns dict with nrmse_tdl_u, r2_tdl_u.
    """
    # Build TDL feature matrix from input u only
    K = len(u)
    X_tdl = np.zeros((K, D), dtype=np.float64)
    for d in range(1, D + 1):
        X_tdl[d:, d - 1] = u[:K - d]

    # Same split as RC (after delay embedding the target also shifts by D)
    y_shifted = y[D:]
    X_shifted = X_tdl[D:]

    i_start = K_warmup
    i_split = i_start + K_train
    i_end   = i_split + K_test
    K_eff   = len(y_shifted)
    i_end   = min(i_end, K_eff)

    X_tr = X_shifted[i_start:i_split]
    X_te = X_shifted[i_split:i_end]
    y_tr = y_shifted[i_start:i_split]
    y_te = y_shifted[i_split:i_end]

    # Standardize on train only
    mu = X_tr.mean(axis=0, keepdims=True)
    sigma = X_tr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    X_tr = (X_tr - mu) / sigma
    X_te = (X_te - mu) / sigma

    model, _ = ridge_cv_fit(X_tr, y_tr, n_folds=ridge_cv_folds, seed=ridge_seed)
    y_pred = ridge_predict(model, X_te)

    return {
        "nrmse_tdl_u": nrmse(y_te, y_pred),
        "r2_tdl_u":    r_squared(y_te, y_pred),
    }


def run_narma_benchmark(
    bundle_dir: Path,
    regime_index: int,
    cfg: NarmaConfig,
    *,
    seed_offset: int = 0,
) -> dict:
    """Run one NARMA-10 benchmark: one regime, one seed.

    Parameters
    ----------
    bundle_dir : Path
        Bundle directory.
    regime_index : int
        Regime index (0-4) to use.
    cfg : NarmaConfig
        Benchmark configuration.
    seed_offset : int
        Added to narma_seed for multi-seed runs.

    Returns
    -------
    dict with keys: nrmse, rmse, r2, alpha_ridge, regime_index,
    regime_name, alpha, narma_seed, cfg, etc.
    """
    from .brian_smoke import build_brian2_from_bundle

    bundle_dir = Path(bundle_dir)

    # Load regime
    regimes_raw = json.loads(
        (bundle_dir / "regimes" / "regimes.json").read_text())
    if isinstance(regimes_raw, list):
        regimes = regimes_raw
    else:
        regimes = regimes_raw.get("regimes", regimes_raw)
    regime = regimes[regime_index]
    alpha_syn = regime["alpha"]

    # 1. Generate NARMA-10 signal
    narma_seed = cfg.narma_seed + seed_offset
    u, y = generate_narma10(cfg.K_total, seed=narma_seed)

    # 1b. TDL baseline (deterministic, no Brian2 needed, same splits)
    tdl_baseline = _tdl_baseline(
        u, y,
        D=cfg.n_delays,
        K_warmup=cfg.K_warmup,
        K_train=cfg.K_train,
        K_test=cfg.K_test,
        ridge_cv_folds=cfg.ridge_cv_folds,
        ridge_seed=cfg.ridge_seed,
    )

    # 2. Prepare NARMA drive for Brian2
    # Brian2 sim duration: brian_warmup + K_total * dt_task_ms
    brian_total_ms = cfg.brian_warmup_ms + cfg.K_total * cfg.dt_task_ms
    # The drive array covers the entire sim: zeros during brian_warmup, then u[k]
    n_warmup_bins = int(cfg.brian_warmup_ms / cfg.dt_task_ms) if cfg.brian_warmup_ms > 0 else 0
    narma_drive = np.concatenate([
        np.zeros(n_warmup_bins),  # zero current during Brian2 warmup
        u,                        # NARMA input for actual simulation
    ])

    # Effective warmup/measure for build_brian2_from_bundle
    # We pass 0 warmup and full duration as measure, since we handle
    # warmup ourselves via K_warmup steps
    warmup_for_brian = 0.0
    measure_for_brian = brian_total_ms

    # 3. Run Brian2 simulation
    pop_data = json.loads((bundle_dir / "network" / "population.json").read_text())
    N_E = pop_data["N_E"]
    N_I = pop_data["N_I"]
    N = N_E + N_I

    t0 = time.time()
    spike_trains, E_idx, I_idx, neurons, b = build_brian2_from_bundle(
        bundle_dir, alpha_syn,
        warmup_ms=warmup_for_brian,
        measure_ms=measure_for_brian,
        dt_ms=cfg.dt_ms,
        narma_drive=narma_drive,
        narma_dt_ms=cfg.dt_task_ms,
        narma_scale_nA=cfg.narma_scale_nA,
        bg_scale=cfg.bg_scale,
    )
    brian_time = time.time() - t0

    # 4. Extract spike count states
    # Always extract raw counts first (for diagnostics)
    X_full_raw = extract_spike_counts(
        spike_trains, n_neurons=N,
        dt_task_ms=cfg.dt_task_ms,
        total_ms=brian_total_ms,
        warmup_ms=0.0,
        b_ms_unit=b["ms"],
    )
    X_raw_diag = X_full_raw[n_warmup_bins:, :]  # for diagnostics

    if cfg.ei_split:
        X_full = extract_spike_counts_ei(
            spike_trains, n_neurons=N,
            E_idx=E_idx, I_idx=I_idx,
            dt_task_ms=cfg.dt_task_ms,
            total_ms=brian_total_ms,
            warmup_ms=0.0,
            b_ms_unit=b["ms"],
            normalize=False,  # standardization done after split (Rule 6)
        )
    else:
        X_full = X_full_raw

    # X_full shape: (n_warmup_bins + K_total, N)
    # Discard brian warmup bins and NARMA warmup steps
    X_raw = X_full[n_warmup_bins:, :]  # (K_total, N) — feature matrix

    # Diagnostics on raw spike counts (before delay embedding, before normalization)
    i_start_raw = cfg.K_warmup
    i_end_raw = i_start_raw + cfg.K_train + cfg.K_test
    total_spikes = int(X_raw_diag[i_start_raw:i_end_raw].sum())
    mean_rate_per_bin = float(X_raw_diag[i_start_raw:i_end_raw].mean())
    active_neurons = int(np.sum(X_raw_diag[i_start_raw:i_end_raw].sum(axis=0) > 0))

    # Time-delay embedding (augment with past bins)
    D = cfg.n_delays
    if D > 0:
        X_narma = delay_embed(X_raw, D)
        # After embedding, X_narma is (K_total - D, N*(D+1))
        # Targets y are shifted accordingly
        y = y[D:]
        K_effective = cfg.K_total - D
    else:
        X_narma = X_raw
        K_effective = cfg.K_total

    # Split into train/test (skip K_warmup)
    i_start = cfg.K_warmup
    i_split = i_start + cfg.K_train
    i_end = i_split + cfg.K_test

    if i_end > K_effective:
        i_end = K_effective

    X_train = X_narma[i_start:i_split].astype(np.float64)
    X_test = X_narma[i_split:i_end].astype(np.float64)
    y_train = y[i_start:i_split]
    y_test = y[i_split:i_end]

    # Standardize features using train-only statistics (Rule 6: no leakage)
    if cfg.ei_split:
        # Population-level normalization: one shared mean/std for all E neurons,
        # another for all I neurons. This preserves within-population variance
        # structure while equalizing E vs I scales.
        n_per_block = N_E + N_I
        D_plus_1 = (cfg.n_delays + 1) if cfg.n_delays > 0 else 1
        # Gather all E and I column indices across delay taps
        e_cols = []
        i_cols = []
        for d in range(D_plus_1):
            base = d * n_per_block
            e_cols.extend(range(base, base + N_E))
            i_cols.extend(range(base + N_E, base + n_per_block))
        # Single mean/std per population (computed from train only)
        mu_e = float(X_train[:, e_cols].mean())
        s_e = float(X_train[:, e_cols].std())
        mu_i = float(X_train[:, i_cols].mean())
        s_i = float(X_train[:, i_cols].std())
        if s_e < 1e-12:
            s_e = 1.0
        if s_i < 1e-12:
            s_i = 1.0
        X_train[:, e_cols] = (X_train[:, e_cols] - mu_e) / s_e
        X_test[:, e_cols] = (X_test[:, e_cols] - mu_e) / s_e
        X_train[:, i_cols] = (X_train[:, i_cols] - mu_i) / s_i
        X_test[:, i_cols] = (X_test[:, i_cols] - mu_i) / s_i
    else:
        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

    # Sanity checks (Rule 9)
    has_nan = bool(np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)))
    pct_silent = float(np.mean(X_raw_diag[i_start_raw:i_end_raw].sum(axis=0) == 0))
    status = "OK"
    if has_nan:
        status = "NaN"
    elif total_spikes == 0:
        status = "SILENT"
    elif pct_silent > 0.5:
        status = "MOSTLY_SILENT"

    # 5. Ridge regression readout
    model, alpha_ridge = ridge_cv_fit(
        X_train, y_train,
        alphas=cfg.ridge_alphas,
        n_folds=cfg.ridge_cv_folds,
        seed=cfg.ridge_seed,
    )
    y_pred = ridge_predict(model, X_test)

    # 6. Metrics
    nrmse_val = nrmse(y_test, y_pred)
    rmse_val = rmse(y_test, y_pred)
    r2_val = r_squared(y_test, y_pred)

    result = {
        "nrmse": nrmse_val,
        "rmse": rmse_val,
        "r2": r2_val,
        "alpha_ridge": alpha_ridge,
        "regime_index": regime_index,
        "regime_name": regime.get("name", f"R{regime_index}"),
        "alpha_syn": alpha_syn,
        "narma_seed": narma_seed,
        "seed_offset": seed_offset,
        "brian_time_s": brian_time,
        "total_spikes": total_spikes,
        "mean_rate_per_bin": mean_rate_per_bin,
        "active_neurons": active_neurons,
        "N": N,
        "K_total": cfg.K_total,
        "K_warmup": cfg.K_warmup,
        "K_train": cfg.K_train,
        "K_test": cfg.K_test,
        "dt_task_ms": cfg.dt_task_ms,
        "narma_scale_nA": cfg.narma_scale_nA,
        "bg_scale": cfg.bg_scale,
        "ei_split": cfg.ei_split,
        "n_delays": cfg.n_delays,
        "n_features": X_train.shape[1],
        "dt_ms": cfg.dt_ms,
        "ridge_cv_folds": cfg.ridge_cv_folds,
        "ridge_seed": cfg.ridge_seed,
        "status": status,
        "pct_silent": pct_silent,
        # TDL baseline: ridge on [u(t-1),..,u(t-D)], same splits/normalization
        **tdl_baseline,
    }

    return result


def save_narma_results(
    bundle_dir: Path,
    results: list[dict],
    cfg: NarmaConfig,
) -> Path:
    """Save benchmark results to bundle/benchmarks/narma10/.

    Saves:
      - results.json: all run results
      - summary.json: aggregated per-regime stats
      - report.md: human-readable report
    """
    out_dir = Path(bundle_dir) / "benchmarks" / "narma10"
    out_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # summary.json: per-regime aggregation (OK-only for stats, all for counts)
    regime_groups = {}
    for r in results:
        ri = r["regime_index"]
        if ri not in regime_groups:
            regime_groups[ri] = []
        regime_groups[ri].append(r)

    summary = {}
    for ri, runs in sorted(regime_groups.items()):
        ok_runs = [r for r in runs if r.get("status") == "OK"]
        pool = ok_runs if ok_runs else runs
        nrmses = [r["nrmse"] for r in pool]
        r2s = [r["r2"] for r in pool]
        summary[str(ri)] = {
            "regime_name": runs[0]["regime_name"],
            "alpha_syn": runs[0]["alpha_syn"],
            "n_seeds": len(runs),
            "n_ok": len(ok_runs),
            "nrmse_mean": float(np.mean(nrmses)),
            "nrmse_std": float(np.std(nrmses)),
            "nrmse_min": float(np.min(nrmses)),
            "nrmse_max": float(np.max(nrmses)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # report.md
    n_ok_total = sum(s["n_ok"] for s in summary.values())
    n_total = len(results)
    lines = ["# NARMA-10 Benchmark Report\n"]
    lines.append(f"Bundle: `{bundle_dir}`\n")
    if n_ok_total < n_total:
        lines.append(f"**Note:** {n_total - n_ok_total}/{n_total} non-OK runs "
                     f"excluded from aggregation.\n")
    lines.append(f"| Regime | α_syn | n_ok/total | NRMSE mean±std | R² mean |")
    lines.append(f"|--------|-------|------------|----------------|---------|")
    for ri, s in sorted(summary.items()):
        lines.append(
            f"| {s['regime_name']} | {s['alpha_syn']:.6f} | "
            f"{s['n_ok']}/{s['n_seeds']} | {s['nrmse_mean']:.4f}±{s['nrmse_std']:.4f} | "
            f"{s['r2_mean']:.4f} |"
        )
    lines.append("")
    lines.append(f"Config: K_total={cfg.K_total}, K_warmup={cfg.K_warmup}, "
                 f"K_train={cfg.K_train}, K_test={cfg.K_test}, "
                 f"dt_task_ms={cfg.dt_task_ms}, narma_scale_nA={cfg.narma_scale_nA}, "
                 f"bg_scale={cfg.bg_scale}, ei_split={cfg.ei_split}")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  Results saved to {out_dir}/")
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="NARMA-10 benchmark for frozen LSM reservoir")
    ap.add_argument("bundle_dir", type=Path,
                    help="Path to frozen bundle directory")
    ap.add_argument("--regime", type=int, nargs="+", default=[2],
                    help="Regime indices (0-4), default: [2] (edge)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Number of NARMA seeds per regime (default: 1)")
    ap.add_argument("--K-total", type=int, default=500,
                    help="Total NARMA steps (default: 500)")
    ap.add_argument("--K-warmup", type=int, default=100,
                    help="Warmup steps to discard (default: 100)")
    ap.add_argument("--K-train", type=int, default=300,
                    help="Training steps (default: 300)")
    ap.add_argument("--K-test", type=int, default=100,
                    help="Test steps (default: 100)")
    ap.add_argument("--dt-task-ms", type=float, default=10.0,
                    help="NARMA step duration in ms (default: 10)")
    ap.add_argument("--narma-scale-nA", type=float, default=0.5,
                    help="Current scaling factor (default: 0.5)")
    ap.add_argument("--narma-seed", type=int, default=42,
                    help="Base NARMA seed (default: 42)")
    ap.add_argument("--n-delays", type=int, default=10,
                    help="Time-delay embedding depth (default: 10)")
    ap.add_argument("--bg-scale", type=float, default=1.0,
                    help="Poisson background current multiplier (default: 1.0)")
    ap.add_argument("--ei-split", action="store_true", default=False,
                    help="Use E/I split features with separate normalization")
    args = ap.parse_args()

    if not args.bundle_dir.exists():
        print(f"ERROR: bundle not found: {args.bundle_dir}")
        return 1

    cfg = NarmaConfig(
        K_total=args.K_total,
        K_warmup=args.K_warmup,
        K_train=args.K_train,
        K_test=args.K_test,
        dt_task_ms=args.dt_task_ms,
        narma_scale_nA=args.narma_scale_nA,
        narma_seed=args.narma_seed,
        n_delays=args.n_delays,
        bg_scale=args.bg_scale,
        ei_split=args.ei_split,
    )

    # Validate
    assert cfg.K_warmup + cfg.K_train + cfg.K_test <= cfg.K_total, \
        f"K_warmup + K_train + K_test = {cfg.K_warmup + cfg.K_train + cfg.K_test} > K_total = {cfg.K_total}"

    print(f"{'='*60}")
    print(f"  NARMA-10 BENCHMARK")
    print(f"  bundle: {args.bundle_dir}")
    print(f"  regimes: {args.regime}")
    print(f"  seeds: {args.seeds}")
    print(f"  K_total={cfg.K_total}  K_warmup={cfg.K_warmup}  "
          f"K_train={cfg.K_train}  K_test={cfg.K_test}")
    print(f"  dt_task_ms={cfg.dt_task_ms}  narma_scale_nA={cfg.narma_scale_nA}  "
          f"bg_scale={cfg.bg_scale}  ei_split={cfg.ei_split}")
    print(f"{'='*60}")

    all_results = []
    for ri in args.regime:
        for s in range(args.seeds):
            print(f"\n  Running regime={ri}, seed_offset={s}...")
            t0 = time.time()
            result = run_narma_benchmark(
                args.bundle_dir, ri, cfg, seed_offset=s)
            elapsed = time.time() - t0
            all_results.append(result)
            print(f"    NRMSE={result['nrmse']:.4f}  R²={result['r2']:.4f}  "
                  f"α_ridge={result['alpha_ridge']:.2e}  "
                  f"spikes={result['total_spikes']}  "
                  f"active={result['active_neurons']}/{result['N']}  "
                  f"time={elapsed:.1f}s")

    out_dir = save_narma_results(args.bundle_dir, all_results, cfg)

    print(f"\n  Done. Results in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
