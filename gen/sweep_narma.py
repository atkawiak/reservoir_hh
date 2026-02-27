#!/usr/bin/env python3
"""Parameter sweep runner for NARMA-10 benchmark.

Runs parameter sweeps efficiently by caching Brian2 spike trains
and reusing them for readout-only parameter variations.

Brian2-affecting parameters (each combo = new simulation):
    bg_scale, narma_scale_nA, dt_task_ms, regime, seed

Readout-only parameters (reuse cached spike trains):
    n_delays, ei_split

Usage:
    python -m gen.sweep_narma bundles/bundle_seed_14896 --regime 2 \
        --bg-scales 0.25 0.5 1.0
    python -m gen.sweep_narma bundles/bundle_seed_14896 --regime 2 \
        --narma-scales-nA 0.5 1.0 1.5 2.0
    python -m gen.sweep_narma bundles/bundle_seed_14896 --regime 0 1 2 3 4 \
        --seeds 3 --full-sweep
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .narma import generate_narma10
from .state_readout import extract_spike_counts, extract_spike_counts_ei, delay_embed
from .ridge import ridge_cv_fit, ridge_predict
from .metrics import nrmse, rmse, r_squared


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""
    # Signal generation
    K_total: int = 800
    K_warmup: int = 100
    K_train: int = 500
    K_test: int = 100
    narma_seed: int = 42
    n_seeds: int = 1
    dt_ms: float = 0.025           # Brian2 integration timestep

    # Brian2-affecting parameters (each combo = new simulation)
    bg_scales: list[float] = field(default_factory=lambda: [1.0])
    narma_scales_nA: list[float] = field(default_factory=lambda: [0.5])
    regime_indices: list[int] = field(default_factory=lambda: [2])
    dt_task_ms_values: list[float] = field(default_factory=lambda: [10.0])

    # Readout-only parameters (reuse cached spike trains)
    n_delays_values: list[int] = field(default_factory=lambda: [10])
    ei_split_values: list[bool] = field(default_factory=lambda: [False])

    # Readout
    ridge_alphas: list | None = None
    ridge_cv_folds: int = 5
    ridge_seed: int = 0


def _run_brian2_cached(
    bundle_dir: Path,
    regime_index: int,
    narma_seed: int,
    K_total: int,
    dt_task_ms: float,
    narma_scale_nA: float,
    bg_scale: float,
    dt_ms: float,
) -> dict:
    """Run Brian2 simulation and return cached data for readout sweeps."""
    from .brian_smoke import build_brian2_from_bundle

    u, y = generate_narma10(K_total, seed=narma_seed)

    brian_total_ms = K_total * dt_task_ms
    narma_drive = u  # no extra warmup

    pop_data = json.loads((bundle_dir / "network" / "population.json").read_text())
    N = pop_data["N_E"] + pop_data["N_I"]

    regimes_raw = json.loads(
        (bundle_dir / "regimes" / "regimes.json").read_text())
    if isinstance(regimes_raw, list):
        regimes = regimes_raw
    else:
        regimes = regimes_raw.get("regimes", regimes_raw)
    regime = regimes[regime_index]
    alpha_syn = regime["alpha"]

    spike_trains, E_idx, I_idx, neurons, b = build_brian2_from_bundle(
        bundle_dir, alpha_syn,
        warmup_ms=0.0,
        measure_ms=brian_total_ms,
        dt_ms=dt_ms,
        narma_drive=narma_drive,
        narma_dt_ms=dt_task_ms,
        narma_scale_nA=narma_scale_nA,
        bg_scale=bg_scale,
    )

    return {
        "spike_trains": spike_trains,
        "E_idx": E_idx,
        "I_idx": I_idx,
        "N": N,
        "b_ms_unit": b["ms"],
        "brian_total_ms": brian_total_ms,
        "u": u,
        "y": y,
        "alpha_syn": alpha_syn,
        "regime_name": regime.get("name", f"R{regime_index}"),
        "dt_task_ms": dt_task_ms,
    }


def _compute_liveness(
    spike_trains: list,
    E_idx: np.ndarray,
    b_ms_unit,
    t_start_ms: float,
    t_end_ms: float,
) -> dict:
    """Post-warmup liveness check for E population.

    Thresholds (Korekta 1):
        FAIL : pct_silent_E ≥ 40%,  or rate_E ≥ 200 Hz (runaway),  or no spikes
        WARN : pct_silent_E 10–40%, or rate_E < 2 Hz,               or CV_ISI_E < 0.5
        PASS : pct_silent_E < 10%,  2 ≤ rate_E < 200 Hz,            CV_ISI_E ≥ 0.5

    Measured in the post-warmup window [t_start_ms, t_end_ms).
    """
    window_s = (t_end_ms - t_start_ms) / 1000.0
    e_counts = np.zeros(len(E_idx), dtype=np.float64)
    cvs: list[float] = []

    for k, idx in enumerate(E_idx):
        times_ms = np.asarray(spike_trains[idx]) / float(b_ms_unit)
        in_win = times_ms[(times_ms >= t_start_ms) & (times_ms < t_end_ms)]
        e_counts[k] = len(in_win)
        if len(in_win) >= 2:
            isis = np.diff(np.sort(in_win))
            m = float(isis.mean())
            if m > 0:
                cvs.append(float(isis.std() / m))

    rate_E_hz = float(e_counts.mean() / window_s) if window_s > 0 else 0.0
    pct_silent_E = float(np.mean(e_counts == 0))
    cv_isi_E = float(np.mean(cvs)) if cvs else 0.0

    if e_counts.sum() == 0:
        liveness, reason = "FAIL", "no E spikes"
    elif pct_silent_E >= 0.40:
        liveness, reason = "FAIL", f"pct_silent_E={pct_silent_E:.0%} ≥ 40%"
    elif rate_E_hz >= 200.0:
        liveness, reason = "FAIL", f"runaway: rate_E={rate_E_hz:.0f} Hz"
    elif pct_silent_E >= 0.10:
        liveness, reason = "WARN", f"pct_silent_E={pct_silent_E:.0%} (10–40%)"
    elif rate_E_hz < 2.0:
        liveness, reason = "WARN", f"rate_E={rate_E_hz:.2f} Hz < 2 Hz"
    elif cv_isi_E < 0.5:
        liveness, reason = "WARN", f"CV_ISI_E={cv_isi_E:.2f} < 0.5"
    else:
        liveness, reason = "PASS", "ok"

    return {
        "liveness": liveness,
        "liveness_reason": reason,
        "rate_E_hz": rate_E_hz,
        "pct_silent_E": pct_silent_E,
        "cv_isi_E": cv_isi_E,
    }


def _evaluate_readout(
    cached: dict,
    n_delays: int,
    ei_split: bool,
    K_warmup: int,
    K_train: int,
    K_test: int,
    ridge_alphas,
    ridge_cv_folds: int,
    ridge_seed: int,
) -> dict:
    """Evaluate one readout configuration using cached spike trains."""
    N = cached["N"]
    dt_task_ms = cached["dt_task_ms"]
    y = cached["y"].copy()

    # Always extract raw counts for diagnostics
    X_raw_full = extract_spike_counts(
        cached["spike_trains"], n_neurons=N,
        dt_task_ms=dt_task_ms,
        total_ms=cached["brian_total_ms"],
        warmup_ms=0.0,
        b_ms_unit=cached["b_ms_unit"],
    )

    # Diagnostics on raw spike counts (before normalization)
    i_start_raw = K_warmup
    i_end_raw = min(i_start_raw + K_train + K_test, X_raw_full.shape[0])
    active_window = X_raw_full[i_start_raw:i_end_raw]
    total_spikes = int(active_window.sum())
    active_neurons = int(np.sum(active_window.sum(axis=0) > 0))
    # mean spikes per (bin, neuron) — Krok 0 criterion: > 0.2 means
    # each neuron fires on average at least once every 5 bins
    mean_spikes_per_bin = float(active_window.mean())

    # Feature matrix (possibly E/I split; no normalization here — Rule 6)
    if ei_split:
        X_raw = extract_spike_counts_ei(
            cached["spike_trains"], n_neurons=N,
            E_idx=cached["E_idx"], I_idx=cached["I_idx"],
            dt_task_ms=dt_task_ms,
            total_ms=cached["brian_total_ms"],
            warmup_ms=0.0,
            b_ms_unit=cached["b_ms_unit"],
            normalize=False,
        )
    else:
        X_raw = X_raw_full

    # Time-delay embedding
    D = n_delays
    if D > 0:
        X_narma = delay_embed(X_raw, D)
        y = y[D:]
        K_effective = X_raw.shape[0] - D
    else:
        X_narma = X_raw
        K_effective = X_raw.shape[0]

    i_start = K_warmup
    i_split = i_start + K_train
    i_end = min(i_split + K_test, K_effective)

    X_train = X_narma[i_start:i_split].astype(np.float64)
    X_test = X_narma[i_split:i_end].astype(np.float64)
    y_train = y[i_start:i_split]
    y_test = y[i_split:i_end]

    # Standardize features using train-only statistics (Rule 6: no leakage)
    if ei_split:
        # Population-level normalization: shared mean/std per population
        N_E = len(cached["E_idx"])
        N_I = len(cached["I_idx"])
        n_per_block = N_E + N_I
        D_plus_1 = (n_delays + 1) if n_delays > 0 else 1
        e_cols = []
        i_cols = []
        for d in range(D_plus_1):
            base = d * n_per_block
            e_cols.extend(range(base, base + N_E))
            i_cols.extend(range(base + N_E, base + n_per_block))
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
    pct_silent = float(np.mean(
        X_raw_full[i_start_raw:i_end_raw].sum(axis=0) == 0))
    status = "OK"
    if has_nan:
        status = "NaN"
    elif total_spikes == 0:
        status = "SILENT"
    elif pct_silent > 0.5:
        status = "MOSTLY_SILENT"

    model, alpha_ridge = ridge_cv_fit(
        X_train, y_train,
        alphas=ridge_alphas,
        n_folds=ridge_cv_folds,
        seed=ridge_seed,
    )
    y_pred = ridge_predict(model, X_test)

    return {
        "nrmse": nrmse(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "r2": r_squared(y_test, y_pred),
        "alpha_ridge": alpha_ridge,
        "total_spikes": total_spikes,
        "active_neurons": active_neurons,
        "mean_spikes_per_bin": mean_spikes_per_bin,
        "n_features": X_train.shape[1],
        "K_test_actual": len(y_test),
        "status": status,
        "pct_silent": pct_silent,
    }


def _run_one_group(args: tuple) -> list[dict]:
    """Worker function: one Brian2 sim + all readout configs.

    Top-level (not nested) so it is picklable by ProcessPoolExecutor.
    Each process has its own Brian2 global state; start_scope() inside
    build_brian2_from_bundle() ensures clean isolation per call.
    """
    (bundle_dir, regime_idx, bg_scale, narma_scale, dt_task,
     narma_seed, cfg, readout_grid, group_id, total_groups) = args

    prefix = f"[{group_id:2d}/{total_groups}]"
    print(f"  {prefix} Brian2: regime={regime_idx}, bg={bg_scale}, "
          f"narma={narma_scale}nA, dt={dt_task}ms, seed={narma_seed}",
          flush=True)

    t0 = time.time()
    cached = _run_brian2_cached(
        bundle_dir=Path(bundle_dir),
        regime_index=regime_idx,
        narma_seed=narma_seed,
        K_total=cfg.K_total,
        dt_task_ms=dt_task,
        narma_scale_nA=narma_scale,
        bg_scale=bg_scale,
        dt_ms=cfg.dt_ms,
    )
    brian_time = time.time() - t0

    # Liveness check (Korekta 1): post-warmup window
    liveness_info = _compute_liveness(
        spike_trains=cached["spike_trains"],
        E_idx=cached["E_idx"],
        b_ms_unit=cached["b_ms_unit"],
        t_start_ms=cfg.K_warmup * dt_task,
        t_end_ms=cached["brian_total_ms"],
    )
    lv = liveness_info["liveness"]
    print(f"  {prefix} done {brian_time:.0f}s | "
          f"{lv} ({liveness_info['liveness_reason']}) | "
          f"rate_E={liveness_info['rate_E_hz']:.1f}Hz | "
          f"CV_ISI_E={liveness_info['cv_isi_E']:.2f}",
          flush=True)

    results: list[dict] = []
    for n_delays, ei_split in readout_grid:
        readout_result = _evaluate_readout(
            cached=cached,
            n_delays=n_delays,
            ei_split=ei_split,
            K_warmup=cfg.K_warmup,
            K_train=cfg.K_train,
            K_test=cfg.K_test,
            ridge_alphas=cfg.ridge_alphas,
            ridge_cv_folds=cfg.ridge_cv_folds,
            ridge_seed=cfg.ridge_seed,
        )
        results.append({
            # Brian2 params
            "regime_index": regime_idx,
            "regime_name": cached["regime_name"],
            "alpha_syn": cached["alpha_syn"],
            "bg_scale": bg_scale,
            "narma_scale_nA": narma_scale,
            "narma_seed": narma_seed,
            "brian_time_s": brian_time,
            # Readout params
            "dt_task_ms": dt_task,
            "n_delays": n_delays,
            "ei_split": ei_split,
            # Config
            "N": cached["N"],
            "K_total": cfg.K_total,
            "K_warmup": cfg.K_warmup,
            "K_train": cfg.K_train,
            "K_test": cfg.K_test,
            "dt_ms": cfg.dt_ms,
            "ridge_cv_folds": cfg.ridge_cv_folds,
            "ridge_seed": cfg.ridge_seed,
            # Liveness + metrics
            **liveness_info,
            **readout_result,
        })
        print(f"  {prefix} D={n_delays:2d} ei={'Y' if ei_split else 'N'} "
              f"NRMSE={readout_result['nrmse']:.4f} "
              f"R2={readout_result['r2']:.4f} "
              f"spikes={readout_result['total_spikes']} "
              f"active={readout_result['active_neurons']}/{cached['N']}",
              flush=True)

    return results


def run_sweep(
    bundle_dir: Path,
    cfg: SweepConfig,
    max_workers: int = 1,
) -> list[dict]:
    """Run the full parameter sweep (sequential or parallel).

    Parameters
    ----------
    max_workers : int
        Number of parallel Brian2 processes.  1 = sequential (original
        behaviour, caches spike trains for readout-only reuse).
        >1 = parallel via ProcessPoolExecutor; each worker runs an
        independent Brian2 sim + all readout configs.
        Tip: set to int(os.cpu_count() * 0.8) to leave 20% headroom.
    """
    bundle_dir = Path(bundle_dir)

    # Build parameter grids
    brian_grid = list(itertools.product(
        cfg.regime_indices,
        cfg.bg_scales,
        cfg.narma_scales_nA,
        cfg.dt_task_ms_values,
        range(cfg.n_seeds),
    ))
    readout_grid = list(itertools.product(
        cfg.n_delays_values,
        cfg.ei_split_values,
    ))

    total_brian = len(brian_grid)
    total_readout = len(readout_grid)
    print(f"  Sweep: {total_brian} Brian2 sims x {total_readout} readout configs "
          f"= {total_brian * total_readout} total evaluations"
          f"  (workers={max_workers})")

    if max_workers == 1:
        # ── Sequential (original) ─────────────────────────────────────────────
        all_results: list[dict] = []
        for b_idx, (regime_idx, bg_scale, narma_scale, dt_task, seed_offset) \
                in enumerate(brian_grid):
            narma_seed = cfg.narma_seed + seed_offset
            print(f"\n  [{b_idx+1}/{total_brian}] Brian2: regime={regime_idx}, "
                  f"bg={bg_scale}, narma={narma_scale}nA, dt={dt_task}ms, "
                  f"seed={narma_seed}")

            t0 = time.time()
            cached = _run_brian2_cached(
                bundle_dir=bundle_dir,
                regime_index=regime_idx,
                narma_seed=narma_seed,
                K_total=cfg.K_total,
                dt_task_ms=dt_task,
                narma_scale_nA=narma_scale,
                bg_scale=bg_scale,
                dt_ms=cfg.dt_ms,
            )
            brian_time = time.time() - t0
            print(f"    Brian2 done in {brian_time:.1f}s")

            liveness_info = _compute_liveness(
                spike_trains=cached["spike_trains"],
                E_idx=cached["E_idx"],
                b_ms_unit=cached["b_ms_unit"],
                t_start_ms=cfg.K_warmup * dt_task,
                t_end_ms=cached["brian_total_ms"],
            )
            lv = liveness_info["liveness"]
            print(f"    Liveness: {lv} ({liveness_info['liveness_reason']})  "
                  f"rate_E={liveness_info['rate_E_hz']:.1f}Hz  "
                  f"pct_silent_E={liveness_info['pct_silent_E']:.0%}  "
                  f"CV_ISI_E={liveness_info['cv_isi_E']:.2f}")

            for n_delays, ei_split in readout_grid:
                readout_result = _evaluate_readout(
                    cached=cached,
                    n_delays=n_delays,
                    ei_split=ei_split,
                    K_warmup=cfg.K_warmup,
                    K_train=cfg.K_train,
                    K_test=cfg.K_test,
                    ridge_alphas=cfg.ridge_alphas,
                    ridge_cv_folds=cfg.ridge_cv_folds,
                    ridge_seed=cfg.ridge_seed,
                )
                all_results.append({
                    "regime_index": regime_idx,
                    "regime_name": cached["regime_name"],
                    "alpha_syn": cached["alpha_syn"],
                    "bg_scale": bg_scale,
                    "narma_scale_nA": narma_scale,
                    "narma_seed": narma_seed,
                    "brian_time_s": brian_time,
                    "dt_task_ms": dt_task,
                    "n_delays": n_delays,
                    "ei_split": ei_split,
                    "N": cached["N"],
                    "K_total": cfg.K_total,
                    "K_warmup": cfg.K_warmup,
                    "K_train": cfg.K_train,
                    "K_test": cfg.K_test,
                    "dt_ms": cfg.dt_ms,
                    "ridge_cv_folds": cfg.ridge_cv_folds,
                    "ridge_seed": cfg.ridge_seed,
                    **liveness_info,
                    **readout_result,
                })
                print(f"    D={n_delays:2d}  ei={'Y' if ei_split else 'N'}  "
                      f"NRMSE={readout_result['nrmse']:.4f}  "
                      f"R2={readout_result['r2']:.4f}  "
                      f"spikes={readout_result['total_spikes']}  "
                      f"active={readout_result['active_neurons']}/{cached['N']}")
        return all_results

    else:
        # ── Parallel via ProcessPoolExecutor ──────────────────────────────────
        # Build one job tuple per Brian2 sim (picklable — no lambdas/closures).
        jobs = [
            (
                str(bundle_dir),          # Path → str for safe pickling
                regime_idx,
                bg_scale,
                narma_scale,
                dt_task,
                cfg.narma_seed + seed_offset,
                cfg,
                readout_grid,
                b_idx + 1,               # 1-based group_id for logging
                total_brian,
            )
            for b_idx, (regime_idx, bg_scale, narma_scale, dt_task, seed_offset)
            in enumerate(brian_grid)
        ]

        all_results = []
        n_done = 0
        t_start = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(_run_one_group, job): job for job in jobs
            }
            for fut in as_completed(future_to_job):
                n_done += 1
                try:
                    group_results = fut.result()
                    all_results.extend(group_results)
                    elapsed = time.time() - t_start
                    rate = elapsed / n_done
                    eta = rate * (total_brian - n_done)
                    print(f"  ✓ {n_done}/{total_brian} groups done  "
                          f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                          flush=True)
                except Exception as exc:
                    job = future_to_job[fut]
                    print(f"  ✗ ERROR in group (regime={job[1]}, "
                          f"bg={job[2]}, seed={job[5]}): {exc}",
                          flush=True)

        return all_results


def save_sweep_results(
    bundle_dir: Path,
    results: list[dict],
    cfg: SweepConfig,
) -> Path:
    """Save sweep results to bundle/benchmarks/narma10/sweep/."""
    out_dir = Path(bundle_dir) / "benchmarks" / "narma10" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # sweep_table.csv
    if results:
        fieldnames = list(results[0].keys())
        with open(out_dir / "sweep_table.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # sweep_report.md
    _write_sweep_report(out_dir, results, cfg)

    print(f"  Results saved to {out_dir}/")
    return out_dir


def _write_sweep_report(
    out_dir: Path,
    results: list[dict],
    cfg: SweepConfig,
) -> None:
    """Generate markdown report with summary tables."""
    lines = ["# NARMA-10 Parameter Sweep Report\n"]

    if not results:
        lines.append("No results.\n")
        (out_dir / "sweep_report.md").write_text(
            "\n".join(lines), encoding="utf-8")
        return

    # Filter OK-only for aggregation (Rule 9: SILENT/MOSTLY_SILENT distort averages)
    ok_results = [r for r in results if r.get("status") == "OK"]
    n_excluded = len(results) - len(ok_results)

    # Overall best (from OK results only)
    best_pool = ok_results if ok_results else results
    best = min(best_pool, key=lambda r: r["nrmse"])
    lines.append("## Best Configuration (status=OK only)\n")
    lines.append(f"- NRMSE: **{best['nrmse']:.4f}**")
    lines.append(f"- R2: **{best['r2']:.4f}**")
    lines.append(f"- Regime: {best['regime_name']} (alpha_syn={best['alpha_syn']:.6f})")
    lines.append(f"- bg_scale={best['bg_scale']}, narma_scale={best['narma_scale_nA']}nA, "
                 f"dt_task={best['dt_task_ms']}ms, n_delays={best['n_delays']}, "
                 f"ei_split={best['ei_split']}")
    if n_excluded > 0:
        lines.append(f"- Excluded {n_excluded}/{len(results)} non-OK runs "
                     f"(SILENT/MOSTLY_SILENT/NaN)")
    lines.append("")

    # Group by swept dimension and compute mean NRMSE (OK-only)
    pivot_results = ok_results if ok_results else results

    def _pivot_table(results, row_key, col_key, val_key="nrmse"):
        """Build pivot table: row_key x col_key -> mean(val_key)."""
        groups = {}
        for r in results:
            rk = r[row_key]
            ck = r[col_key]
            groups.setdefault((rk, ck), []).append(r[val_key])
        rows = sorted(set(rk for rk, _ in groups))
        cols = sorted(set(ck for _, ck in groups))
        return rows, cols, {(rk, ck): np.mean(groups.get((rk, ck), [np.nan]))
                           for rk in rows for ck in cols}

    def _format_pivot(title, rows, cols, data, row_label, col_fmt=str):
        lines = [f"## {title}\n"]
        header = f"| {row_label} | " + " | ".join(col_fmt(c) for c in cols) + " |"
        sep = "|" + "---|" * (len(cols) + 1)
        lines.append(header)
        lines.append(sep)
        for r in rows:
            vals = " | ".join(f"{data.get((r, c), float('nan')):.4f}" for c in cols)
            lines.append(f"| {r} | {vals} |")
        lines.append("")
        return lines

    # ── Liveness table (Korekta 1) ────────────────────────────────────────────
    if results and "liveness" in results[0]:
        lines.append("## Liveness Check (E-population, post-warmup)\n")
        lines.append(
            "PASS: pct_silent_E<10%, 2≤rate_E<200 Hz, CV_ISI_E≥0.5 | "
            "WARN: 10–40% silent lub rate/CV graniczne | "
            "FAIL: ≥40% silent lub runaway\n")

        if len(cfg.bg_scales) > 1:
            # Pivot: regime × bg_scale → worst liveness across seeds
            regime_names_lv = sorted(set(r["regime_name"] for r in results))
            bg_scales_lv = sorted(cfg.bg_scales)
            lv_groups: dict[tuple, list[str]] = {}
            for r in results:
                key = (r["regime_name"], r["bg_scale"])
                lv_groups.setdefault(key, []).append(r["liveness"])

            def _worst(vals: list[str]) -> str:
                if "FAIL" in vals:
                    return "FAIL"
                if "WARN" in vals:
                    return "WARN"
                return "PASS" if vals else "-"

            header_lv = ("| Regime | "
                         + " | ".join(f"bg={b}" for b in bg_scales_lv) + " |")
            sep_lv = "|" + "---|" * (len(bg_scales_lv) + 1)
            lines.append(header_lv)
            lines.append(sep_lv)
            for rn in regime_names_lv:
                row_vals = [_worst(lv_groups.get((rn, b), [])) for b in bg_scales_lv]
                lines.append("| " + rn + " | " + " | ".join(row_vals) + " |")
            lines.append("")

            # bg_scale recommendation: lowest with all-PASS, else all-not-FAIL
            bg_worst_all: dict[float, str] = {}
            for b in bg_scales_lv:
                vals_b = [r["liveness"] for r in results if r["bg_scale"] == b]
                bg_worst_all[b] = _worst(vals_b)

            recommended_bg = None
            for b in bg_scales_lv:
                if bg_worst_all[b] == "PASS":
                    recommended_bg = b
                    break
            if recommended_bg is None:
                for b in bg_scales_lv:
                    if bg_worst_all[b] == "WARN":
                        recommended_bg = b
                        break

            if recommended_bg is not None:
                tier = bg_worst_all[recommended_bg]
                lines.append(
                    f"**Zalecany bg_scale:** `{recommended_bg}` "
                    f"(najniższy spełniający liveness={tier} we wszystkich reżimach)\n")
            else:
                lines.append("**Uwaga:** żaden bg_scale nie spełnia PASS ani WARN!\n")
        else:
            # Single bg_scale — just list per-regime liveness
            regime_names_lv = sorted(set(r["regime_name"] for r in results))
            lv_by_regime: dict[str, list[str]] = {}
            for r in results:
                lv_by_regime.setdefault(r["regime_name"], []).append(r["liveness"])
            lines.append("| Regime | Liveness | rate_E (Hz) | pct_silent_E | CV_ISI_E |")
            lines.append("|--------|----------|-------------|-------------|----------|")
            regime_rows = {}
            for r in results:
                rn = r["regime_name"]
                if rn not in regime_rows:
                    regime_rows[rn] = r
            for rn in sorted(regime_rows):
                r = regime_rows[rn]
                lines.append(
                    f"| {rn} | {r['liveness']} ({r['liveness_reason']}) "
                    f"| {r['rate_E_hz']:.1f} | {r['pct_silent_E']:.0%} "
                    f"| {r['cv_isi_E']:.2f} |")
            lines.append("")

    # ── Tuning note (Korekta 2) ───────────────────────────────────────────────
    tuning_regimes = sorted(set(r["regime_name"] for r in results))
    lines.append(
        f"> **Nota metodologiczna (Korekta 2):** parametry dobrane na reżimach: "
        f"{', '.join(tuning_regimes)}. "
        f"Wyniki dla tych samych reżimów są lekko uprzywilejowane.\n")

    # ── NRMSE pivot tables ────────────────────────────────────────────────────
    # bg_scale pivot (if swept)
    if len(cfg.bg_scales) > 1:
        rows, cols, data = _pivot_table(pivot_results, "regime_name", "bg_scale")
        lines.extend(_format_pivot(
            "NRMSE by Regime x bg_scale", rows, cols, data,
            "Regime", lambda c: f"bg={c}"))

    # narma_scale_nA pivot (if swept)
    if len(cfg.narma_scales_nA) > 1:
        rows, cols, data = _pivot_table(pivot_results, "regime_name", "narma_scale_nA")
        lines.extend(_format_pivot(
            "NRMSE by Regime x narma_scale_nA", rows, cols, data,
            "Regime", lambda c: f"{c}nA"))

    # dt_task_ms pivot (if swept)
    if len(cfg.dt_task_ms_values) > 1:
        rows, cols, data = _pivot_table(pivot_results, "regime_name", "dt_task_ms")
        lines.extend(_format_pivot(
            "NRMSE by Regime x dt_task_ms", rows, cols, data,
            "Regime", lambda c: f"{c}ms"))

    # n_delays pivot (if swept)
    if len(cfg.n_delays_values) > 1:
        rows, cols, data = _pivot_table(pivot_results, "regime_name", "n_delays")
        lines.extend(_format_pivot(
            "NRMSE by Regime x n_delays", rows, cols, data,
            "Regime", lambda c: f"D={c}"))

    # E/I split comparison
    if True in cfg.ei_split_values and False in cfg.ei_split_values:
        rows, cols, data = _pivot_table(pivot_results, "regime_name", "ei_split")
        lines.extend(_format_pivot(
            "NRMSE by Regime x ei_split", rows, cols, data,
            "Regime", lambda c: "E/I" if c else "all"))

    # Status summary
    if n_excluded > 0:
        lines.append("## Status Summary\n")
        status_counts = {}
        for r in results:
            s = r.get("status", "UNKNOWN")
            status_counts[s] = status_counts.get(s, 0) + 1
        for s, cnt in sorted(status_counts.items()):
            lines.append(f"- {s}: {cnt}")
        lines.append("")

    # Config summary
    lines.append("## Sweep Configuration\n")
    lines.append(f"- K_total={cfg.K_total}, K_warmup={cfg.K_warmup}, "
                 f"K_train={cfg.K_train}, K_test={cfg.K_test}")
    lines.append(f"- bg_scales: {cfg.bg_scales}")
    lines.append(f"- narma_scales_nA: {cfg.narma_scales_nA}")
    lines.append(f"- dt_task_ms: {cfg.dt_task_ms_values}")
    lines.append(f"- n_delays: {cfg.n_delays_values}")
    lines.append(f"- ei_split: {cfg.ei_split_values}")
    lines.append(f"- n_seeds: {cfg.n_seeds}")
    lines.append(f"- Total evaluations: {len(results)} ({len(ok_results)} OK)")
    lines.append("")

    (out_dir / "sweep_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="NARMA-10 parameter sweep for frozen LSM reservoir")
    ap.add_argument("bundle_dir", type=Path,
                    help="Path to frozen bundle directory")
    ap.add_argument("--regime", type=int, nargs="+", default=[2],
                    help="Regime indices (0-4)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Number of NARMA seeds per combo (default: 1)")
    ap.add_argument("--K-total", type=int, default=800)
    ap.add_argument("--K-warmup", type=int, default=100)
    ap.add_argument("--K-train", type=int, default=500)
    ap.add_argument("--K-test", type=int, default=100)
    ap.add_argument("--narma-seed", type=int, default=42)

    # Sweep grids
    ap.add_argument("--bg-scales", type=float, nargs="+", default=[1.0],
                    help="Background current multipliers (default: [1.0])")
    ap.add_argument("--narma-scales-nA", type=float, nargs="+", default=[0.5],
                    help="NARMA current scales in nA (default: [0.5])")
    ap.add_argument("--dt-task-ms", type=float, nargs="+", default=[10.0],
                    help="Binning widths in ms (default: [10.0])")
    ap.add_argument("--n-delays", type=int, nargs="+", default=[10],
                    help="Delay embedding depths (default: [10])")
    ap.add_argument("--ei-split", action="store_true", default=False,
                    help="Include E/I split in sweep (sweeps both True and False)")

    # Parallelism
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel Brian2 workers. Use -1 for 80%% of CPU cores "
                         "(e.g. 40 on a 50-core server). Default: 1 (sequential).")

    # Preset
    ap.add_argument("--full-sweep", action="store_true",
                    help="Full sweep: bg={0.25,0.5,1.0}, narma={0.5,1.0,1.5,2.0}, "
                         "dt={5,10,20,50}, delays={0,5,10,15}, ei={F,T}")

    args = ap.parse_args()

    if not args.bundle_dir.exists():
        print(f"ERROR: bundle not found: {args.bundle_dir}")
        return 1

    ei_split_values = [False, True] if args.ei_split else [False]

    if args.full_sweep:
        cfg = SweepConfig(
            K_total=args.K_total,
            K_warmup=args.K_warmup,
            K_train=args.K_train,
            K_test=args.K_test,
            narma_seed=args.narma_seed,
            n_seeds=args.seeds,
            regime_indices=args.regime,
            bg_scales=[0.25, 0.5, 1.0],
            narma_scales_nA=[0.5, 1.0, 1.5, 2.0],
            dt_task_ms_values=[5.0, 10.0, 20.0, 50.0],
            n_delays_values=[0, 5, 10, 15],
            ei_split_values=[False, True],
        )
    else:
        cfg = SweepConfig(
            K_total=args.K_total,
            K_warmup=args.K_warmup,
            K_train=args.K_train,
            K_test=args.K_test,
            narma_seed=args.narma_seed,
            n_seeds=args.seeds,
            regime_indices=args.regime,
            bg_scales=args.bg_scales,
            narma_scales_nA=args.narma_scales_nA,
            dt_task_ms_values=args.dt_task_ms,
            n_delays_values=args.n_delays,
            ei_split_values=ei_split_values,
        )

    # Validate
    assert cfg.K_warmup + cfg.K_train + cfg.K_test <= cfg.K_total, \
        f"K_warmup + K_train + K_test > K_total"

    # Count
    n_brian = (len(cfg.regime_indices) * len(cfg.bg_scales) *
               len(cfg.narma_scales_nA) * len(cfg.dt_task_ms_values) *
               cfg.n_seeds)
    n_readout = len(cfg.n_delays_values) * len(cfg.ei_split_values)

    print(f"{'='*60}")
    print(f"  NARMA-10 PARAMETER SWEEP")
    print(f"  bundle: {args.bundle_dir}")
    print(f"  regimes: {cfg.regime_indices}")
    print(f"  bg_scales: {cfg.bg_scales}")
    print(f"  narma_scales_nA: {cfg.narma_scales_nA}")
    print(f"  dt_task_ms: {cfg.dt_task_ms_values}")
    print(f"  n_delays: {cfg.n_delays_values}")
    print(f"  ei_split: {cfg.ei_split_values}")
    print(f"  seeds: {cfg.n_seeds}")
    print(f"  → {n_brian} Brian2 sims x {n_readout} readout = "
          f"{n_brian * n_readout} evaluations")
    print(f"{'='*60}")

    # Resolve --jobs -1 → 80 % of available cores
    import os
    if args.jobs == -1:
        n_cores = os.cpu_count() or 1
        max_workers = max(1, int(n_cores * 0.8))
        print(f"  --jobs -1 → {max_workers} workers ({n_cores} cores × 80%)")
    else:
        max_workers = args.jobs

    results = run_sweep(args.bundle_dir, cfg, max_workers=max_workers)
    out_dir = save_sweep_results(args.bundle_dir, results, cfg)

    # Print best result
    if results:
        best = min(results, key=lambda r: r["nrmse"])
        print(f"\n  Best: NRMSE={best['nrmse']:.4f}  R2={best['r2']:.4f}  "
              f"bg={best['bg_scale']}  narma={best['narma_scale_nA']}nA  "
              f"dt={best['dt_task_ms']}ms  D={best['n_delays']}  "
              f"ei={'Y' if best['ei_split'] else 'N'}  "
              f"regime={best['regime_name']}")

    print(f"\n  Done. {len(results)} evaluations saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
