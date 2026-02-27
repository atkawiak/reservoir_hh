#!/usr/bin/env python3
"""Calibrate edge of chaos for a frozen bundle (requires Brian2).

Scans α values via Brian2, finds α_edge using CV_ISI_E + Fano_E thresholds
with liveness gating and trend check, then writes 5 regimes to regimes.json.

Usage:
    python -m gen.calibrate_edge bundles/bundle_seed_123
    python -m gen.calibrate_edge bundles/bundle_seed_123 --n-scan 20 --warmup-ms 1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import GeneratorConfig
from .io_bundle import write_manifest
from .regimes import find_alpha_edge, save_edge_regimes, run_stability_check


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Find edge of chaos and generate 5 regimes")
    ap.add_argument("bundle_dir", type=Path,
                    help="Path to existing bundle directory")
    ap.add_argument("--cv-thr", type=float, default=0.45,
                    help="CV_ISI_E threshold for chaos onset (default: 0.45)")
    ap.add_argument("--fano-thr", type=float, default=0.50,
                    help="Fano_E threshold for chaos onset (default: 0.50)")
    ap.add_argument("--n-scan", type=int, default=16,
                    help="Number of coarse α scan points (default: 16)")
    ap.add_argument("--n-refine", type=int, default=6,
                    help="Number of refinement points around edge (default: 6, 0=disable)")
    ap.add_argument("--warmup-ms", type=float, default=1000.0,
                    help="Warmup duration per scan point (default: 1000)")
    ap.add_argument("--measure-ms", type=float, default=2000.0,
                    help="Measurement duration per scan point (default: 2000)")
    ap.add_argument("--dt-ms", type=float, default=0.025)
    ap.add_argument("--multipliers", type=float, nargs=5,
                    default=[0.40, 0.70, 1.00, 1.40, 2.00],
                    help="5 multipliers for α_edge "
                         "(default: 0.40 0.70 1.00 1.40 2.00)")
    ap.add_argument("--rho-eff-min", type=float, default=0.1,
                    help="Min effective ρ for grid (default: 0.1)")
    ap.add_argument("--rho-eff-max", type=float, default=3.0,
                    help="Max effective ρ for grid (default: 3.0)")
    ap.add_argument("--stability-check", action="store_true", default=True,
                    help="Run stability check at α_edge (default: on)")
    ap.add_argument("--no-stability-check", dest="stability_check",
                    action="store_false",
                    help="Skip stability check")
    ap.add_argument("--stability-ms", type=float, default=4000.0,
                    help="Measurement duration for stability check (default: 4000)")
    ap.add_argument("--stability-tol", type=float, default=0.10,
                    help="Relative tolerance for stability (default: 0.10)")
    args = ap.parse_args()

    bundle_dir = args.bundle_dir
    if not bundle_dir.exists():
        print(f"ERROR: bundle not found: {bundle_dir}")
        return 1

    # Load rho_base from bundle
    stats_path = bundle_dir / "network" / "base_stats.json"
    if not stats_path.exists():
        print(f"ERROR: base_stats.json not found in {bundle_dir}")
        return 1
    stats = json.loads(stats_path.read_text())
    rho_base = stats["rho_full"]

    # Build cfg with user overrides
    cfg = GeneratorConfig(
        edge_cv_thr=args.cv_thr,
        edge_fano_thr=args.fano_thr,
        edge_n_scan=args.n_scan,
        edge_n_refine=args.n_refine,
        edge_rho_eff_min=args.rho_eff_min,
        edge_rho_eff_max=args.rho_eff_max,
    )

    print(f"{'='*60}")
    print(f"  EDGE-OF-CHAOS CALIBRATION")
    print(f"  bundle: {bundle_dir}")
    print(f"  rho_base: {rho_base:.4f}")
    print(f"  cv_thr={cfg.edge_cv_thr}  fano_thr={cfg.edge_fano_thr}")
    print(f"  n_scan={cfg.edge_n_scan}  n_refine={cfg.edge_n_refine}  "
          f"ρ_eff=[{cfg.edge_rho_eff_min}, {cfg.edge_rho_eff_max}]")
    print(f"  warmup={args.warmup_ms}ms  measure={args.measure_ms}ms")
    print(f"  multipliers={args.multipliers}")
    print(f"{'='*60}")

    t0 = time.time()

    alpha_edge, scan_info = find_alpha_edge(
        bundle_dir, rho_base, cfg,
        warmup_ms=args.warmup_ms,
        measure_ms=args.measure_ms,
        dt_ms=args.dt_ms,
    )

    # Stability check
    if args.stability_check:
        stab = run_stability_check(
            bundle_dir, alpha_edge, rho_base, scan_info,
            warmup_ms=args.warmup_ms,
            measure_ms=args.stability_ms,
            dt_ms=args.dt_ms,
            tol=args.stability_tol,
        )
        scan_info["stability_check"] = stab

    edge_method = ("metrics_scan_2stage"
                    if scan_info.get("refine_grid")
                    else "metrics_scan")

    regimes = save_edge_regimes(
        bundle_dir, alpha_edge, rho_base, scan_info,
        multipliers=args.multipliers,
        edge_method=edge_method,
    )

    # Refresh manifest after regimes.json was overwritten
    manifest_path = write_manifest(bundle_dir)
    print(f"  Manifest refreshed: {manifest_path}")

    elapsed = time.time() - t0
    print(f"\n  α_edge = {alpha_edge:.6f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
