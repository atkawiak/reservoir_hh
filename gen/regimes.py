"""Regime definitions: simple α from ρ targets + algorithmic edge-of-chaos detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import REGIME_NAMES, GeneratorConfig


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE MODE — 5 regimes from ρ_eff targets (no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

def make_regimes(rho_base: float, targets: list[float]) -> list[dict]:
    """Create 5 regime definitions from ρ targets.

    Each regime: alpha_i = target_i / rho_base.

    Returns list of dicts with: name, index, rho_target, alpha.
    """
    regimes = []
    for i, rho_t in enumerate(targets):
        alpha = rho_t / rho_base
        regimes.append({
            "name": REGIME_NAMES[i],
            "index": i,
            "rho_target": float(rho_t),
            "alpha": float(alpha),
        })
    return regimes


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE MODE — 5 regimes from α_edge × multipliers
# ═══════════════════════════════════════════════════════════════════════════════

def make_regimes_from_edge(
    alpha_edge: float,
    rho_base: float,
    multipliers: Optional[List[float]] = None,
) -> list[dict]:
    """Create 5 regimes centered on α_edge.

    R1..R5 = α_edge × multipliers[0..4].
    Default multipliers: [0.40, 0.70, 1.00, 1.40, 2.00].
    """
    if multipliers is None:
        multipliers = [0.40, 0.70, 1.00, 1.40, 2.00]

    regimes = []
    for i, (name, mult) in enumerate(zip(REGIME_NAMES, multipliers)):
        alpha = alpha_edge * mult
        regimes.append({
            "name": name,
            "index": i,
            "alpha": float(alpha),
            "rho_target": float(alpha * rho_base),
            "multiplier": float(mult),
            "alpha_edge": float(alpha_edge),
        })
    return regimes


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE FINDING — scan α grid via Brian2 (requires Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_alpha_grid(rho_base: float, cfg: GeneratorConfig) -> np.ndarray:
    """Build deterministic logspace α grid from ρ_eff range.

    alpha_min = rho_eff_min / rho_base
    alpha_max = rho_eff_max / rho_base
    K = edge_n_scan points in logspace.
    """
    alpha_min = cfg.edge_rho_eff_min / rho_base
    alpha_max = cfg.edge_rho_eff_max / rho_base
    return np.geomspace(alpha_min, alpha_max, cfg.edge_n_scan)


def _run_scan_points(
    bundle_dir: Path,
    alphas: np.ndarray,
    rho_base: float,
    warmup_ms: float,
    measure_ms: float,
    dt_ms: float,
    label: str = "scan",
) -> List[dict]:
    """Run Brian2 for each α and return list of scan entry dicts."""
    from .brian_smoke import run_smoke_one_regime, evaluate_scan_liveness

    results = []
    n = len(alphas)
    for k, alpha in enumerate(alphas):
        regime_stub = {
            "name": f"{label}_{k:02d}",
            "index": k,
            "rho_target": float(alpha * rho_base),
            "alpha": float(alpha),
        }
        try:
            m = run_smoke_one_regime(
                bundle_dir, regime_stub, warmup_ms, measure_ms, dt_ms)
            gate = evaluate_scan_liveness(
                m.rate_E, m.rate_I, m.pct_silent_E, m.sync_E, m.has_nan)
            entry = {
                "alpha": float(alpha),
                "cv_isi_E": m.cv_isi_E,
                "fano_E": m.fano_E,
                "rate_E": m.rate_E,
                "rate_I": m.rate_I,
                "pct_silent_E": m.pct_silent_E,
                "sync_E": m.sync_E,
                "status": m.status,
                "gate": gate,
            }
        except Exception as exc:
            entry = {
                "alpha": float(alpha),
                "cv_isi_E": 0.0,
                "fano_E": 0.0,
                "rate_E": 0.0,
                "rate_I": 0.0,
                "pct_silent_E": 100.0,
                "sync_E": 0.0,
                "status": f"ERROR: {exc}",
                "gate": "FAIL",
            }
        results.append(entry)
        print(f"    [{k+1}/{n}] α={alpha:.6f}  "
              f"cvE={entry['cv_isi_E']:.3f}  fanoE={entry['fano_E']:.3f}  "
              f"rateE={entry['rate_E']:.1f}  silent={entry['pct_silent_E']:.0f}%  "
              f"[{entry['gate']}]")
    return results


def _build_refine_grid(
    alphas_coarse: np.ndarray,
    k_star: int,
    n_refine: int,
) -> np.ndarray:
    """Build refinement α grid around k_star.

    - k_star == len-1  → extend upward:  linspace(α[K-1], α[K-1]*2, n+1)[1:]
    - k_star == 0      → extend downward: linspace(α[0]/2, α[0], n+1)[:-1]
    - otherwise        → zoom in:  linspace(α[k-1], α[k+1], n+2)[1:-1]
    """
    K = len(alphas_coarse)
    if k_star == K - 1:
        # Extend upward
        return np.linspace(alphas_coarse[-1], alphas_coarse[-1] * 2, n_refine + 1)[1:]
    elif k_star == 0:
        # Extend downward
        return np.linspace(alphas_coarse[0] / 2, alphas_coarse[0], n_refine + 1)[:-1]
    else:
        # Zoom in between neighbors
        return np.linspace(
            alphas_coarse[k_star - 1], alphas_coarse[k_star + 1],
            n_refine + 2)[1:-1]


def find_alpha_edge(
    bundle_dir: Path,
    rho_base: float,
    cfg: Optional[GeneratorConfig] = None,
    warmup_ms: float = 1000.0,
    measure_ms: float = 2000.0,
    dt_ms: float = 0.025,
) -> Tuple[float, dict]:
    """Two-stage scan to find edge of chaos (requires Brian2).

    Stage 1 — coarse grid (K=edge_n_scan, logspace):
        Run all K points. Find k* via Step A+B or fallback.

    Stage 2 — refinement (K_refine=edge_n_refine, linspace around k*):
        - k* at last  → extend upward
        - k* at first → extend downward
        - k* in middle → zoom between α[k*-1] and α[k*+1]
        Merge coarse + refine, sort by α, re-run Step A+B on combined.

    Returns (α_edge, scan_info) where scan_info is a dict with:
        coarse_grid, refine_grid, merged_grid, edge_rule,
        alpha_edge_source, refine_direction.
    """
    bundle_dir = Path(bundle_dir)
    if cfg is None:
        cfg = GeneratorConfig()

    cv_thr = cfg.edge_cv_thr
    fano_thr = cfg.edge_fano_thr
    silent_thr = cfg.edge_silent_thr
    cv_trend_tol = cfg.edge_cv_trend_tol
    fano_trend_tol = cfg.edge_fano_trend_tol
    n_refine = cfg.edge_n_refine

    edge_rule = {
        "cv_thr": cv_thr,
        "fano_thr": fano_thr,
        "silent_thr": silent_thr,
        "cv_trend_tol": cv_trend_tol,
        "fano_trend_tol": fano_trend_tol,
    }

    # ── Stage 1: coarse grid ──
    alphas_coarse = _build_alpha_grid(rho_base, cfg)
    alpha_lo, alpha_hi = float(alphas_coarse[0]), float(alphas_coarse[-1])

    print(f"  Stage 1 — coarse: {cfg.edge_n_scan} points in "
          f"[{alpha_lo:.6f}, {alpha_hi:.6f}]  "
          f"(ρ_eff=[{cfg.edge_rho_eff_min}, {cfg.edge_rho_eff_max}])")

    coarse_results = _run_scan_points(
        bundle_dir, alphas_coarse, rho_base,
        warmup_ms, measure_ms, dt_ms, label="coarse")

    # Find k* on coarse grid
    coarse_edge = _find_edge_ab(
        coarse_results, cv_thr, fano_thr, silent_thr,
        cv_trend_tol, fano_trend_tol)

    if coarse_edge is not None:
        k_star = next(i for i, r in enumerate(coarse_results)
                      if r["alpha"] == coarse_edge)
        coarse_method = "A+B"
    else:
        # Try fallback to get k*
        fb_edge = _find_edge_fallback(coarse_results, silent_thr)
        if fb_edge is not None:
            k_star = next(i for i, r in enumerate(coarse_results)
                          if r["alpha"] == fb_edge)
            coarse_method = "fallback"
        else:
            raise RuntimeError(
                "Edge not found — all coarse scan points FAIL or "
                "%silent_E > threshold. "
                "Network may be degenerate or scan range too narrow.")

    print(f"  Coarse k*={k_star} (α={alphas_coarse[k_star]:.6f}, {coarse_method})")

    # ── Stage 2: refinement ──
    if n_refine <= 0:
        edge_alpha = coarse_results[k_star]["alpha"]
        print(f"  α_edge = {edge_alpha:.6f}  (coarse only, {coarse_method})")
        scan_info = {
            "coarse_grid": coarse_results,
            "refine_grid": [],
            "merged_grid": coarse_results,
            "edge_rule": edge_rule,
            "alpha_edge_source": f"coarse_k{k_star}_{coarse_method}",
            "refine_direction": None,
        }
        return edge_alpha, scan_info

    refine_alphas = _build_refine_grid(alphas_coarse, k_star, n_refine)
    direction = ("up" if k_star == len(alphas_coarse) - 1
                 else "down" if k_star == 0
                 else "zoom")
    print(f"  Stage 2 — refine: {n_refine} points ({direction}) in "
          f"[{refine_alphas[0]:.6f}, {refine_alphas[-1]:.6f}]")

    refine_results = _run_scan_points(
        bundle_dir, refine_alphas, rho_base,
        warmup_ms, measure_ms, dt_ms, label="refine")

    # Merge and sort by alpha
    all_results = sorted(coarse_results + refine_results, key=lambda r: r["alpha"])

    # Re-run Step A+B on merged set
    alpha_edge = _find_edge_ab(
        all_results, cv_thr, fano_thr, silent_thr,
        cv_trend_tol, fano_trend_tol)

    if alpha_edge is not None:
        # Determine source: was it from coarse or refine?
        refine_alpha_set = set(float(a) for a in refine_alphas)
        source_stage = "refine" if alpha_edge in refine_alpha_set else "coarse"
        merged_idx = next(i for i, r in enumerate(all_results)
                          if r["alpha"] == alpha_edge)
        source_label = f"{source_stage}_A+B_merged_k{merged_idx}"
        print(f"  α_edge = {alpha_edge:.6f}  (A+B on merged, from {source_stage})")
    else:
        # Fallback on merged
        alpha_edge = _find_edge_fallback(all_results, silent_thr)
        if alpha_edge is not None:
            source_label = "fallback_merged"
            print(f"  WARNING: A+B never met on merged, "
                  f"fallback α_edge={alpha_edge:.6f}  (max chaos_score)")
        else:
            raise RuntimeError(
                "Edge not found — all scan points FAIL or %silent_E > threshold. "
                "Network may be degenerate or scan range too narrow.")

    scan_info = {
        "coarse_grid": coarse_results,
        "refine_grid": refine_results,
        "merged_grid": all_results,
        "edge_rule": edge_rule,
        "alpha_edge_source": source_label,
        "refine_direction": direction,
    }
    return alpha_edge, scan_info


def _find_edge_ab(
    scan_results: List[dict],
    cv_thr: float, fano_thr: float, silent_thr: float,
    cv_trend_tol: float, fano_trend_tol: float,
) -> Optional[float]:
    """Step A + B: find first α passing candidate + trend criteria."""
    for k, r in enumerate(scan_results):
        # Must pass liveness gate
        if r["gate"] == "FAIL":
            continue

        # Step A: candidate criteria
        if r["pct_silent_E"] > silent_thr:
            continue
        if r["cv_isi_E"] < cv_thr or r["fano_E"] < fano_thr:
            continue

        # Step B: trend check (skip for k=0)
        if k > 0:
            prev = scan_results[k - 1]
            if r["cv_isi_E"] < prev["cv_isi_E"] - cv_trend_tol:
                continue
            if r["fano_E"] < prev["fano_E"] - fano_trend_tol:
                continue

        return r["alpha"]
    return None


def _find_edge_fallback(
    scan_results: List[dict], silent_thr: float
) -> Optional[float]:
    """Fallback: argmax of z(CV_ISI_E) + z(Fano_E) among eligible points."""
    eligible = [r for r in scan_results
                if r["gate"] in ("PASS", "WARN")
                and r["pct_silent_E"] <= silent_thr]
    if not eligible:
        return None

    cvs = np.array([r["cv_isi_E"] for r in eligible])
    fanos = np.array([r["fano_E"] for r in eligible])

    # z-score normalization
    cv_std = cvs.std()
    fano_std = fanos.std()
    z_cv = (cvs - cvs.mean()) / cv_std if cv_std > 1e-12 else np.zeros_like(cvs)
    z_fano = (fanos - fanos.mean()) / fano_std if fano_std > 1e-12 else np.zeros_like(fanos)

    scores = z_cv + z_fano
    best_idx = int(np.argmax(scores))
    return eligible[best_idx]["alpha"]


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY CHECK — extra sim at α_edge with longer T_eval
# ═══════════════════════════════════════════════════════════════════════════════

def run_stability_check(
    bundle_dir: Path,
    alpha_edge: float,
    rho_base: float,
    scan_info: dict,
    warmup_ms: float = 1000.0,
    measure_ms: float = 4000.0,
    dt_ms: float = 0.025,
    tol: float = 0.10,
) -> dict:
    """Run extra simulation at α_edge with longer T_eval.

    Compares CV_ISI_E and Fano_E from the scan (2s) against a new
    run with measure_ms (default 4s). If both are within ±tol (10%),
    the edge is considered stable.

    Returns dict with: stable, cv_scan, cv_long, fano_scan, fano_long,
    cv_rel_diff, fano_rel_diff, tol, measure_ms.
    """
    from .brian_smoke import run_smoke_one_regime

    # Get scan CV/Fano at α_edge from merged grid
    merged = scan_info.get("merged_grid", [])
    edge_entry = None
    for r in merged:
        if abs(r["alpha"] - alpha_edge) < 1e-12:
            edge_entry = r
            break

    if edge_entry is None:
        return {"stable": None, "reason": "alpha_edge not found in merged_grid"}

    cv_scan = edge_entry["cv_isi_E"]
    fano_scan = edge_entry["fano_E"]

    print(f"  Stability check: α={alpha_edge:.6f}, "
          f"measure={measure_ms:.0f}ms (vs scan 2000ms)")

    regime_stub = {
        "name": "stability_check",
        "index": -1,
        "rho_target": float(alpha_edge * rho_base),
        "alpha": float(alpha_edge),
    }

    try:
        m = run_smoke_one_regime(
            bundle_dir, regime_stub, warmup_ms, measure_ms, dt_ms)
        cv_long = m.cv_isi_E
        fano_long = m.fano_E

        cv_rel = abs(cv_long - cv_scan) / max(cv_scan, 1e-9)
        fano_rel = abs(fano_long - fano_scan) / max(fano_scan, 1e-9)
        stable = cv_rel <= tol and fano_rel <= tol

        result = {
            "stable": stable,
            "cv_scan": float(cv_scan),
            "cv_long": float(cv_long),
            "cv_rel_diff": float(cv_rel),
            "fano_scan": float(fano_scan),
            "fano_long": float(fano_long),
            "fano_rel_diff": float(fano_rel),
            "tol": float(tol),
            "measure_ms": float(measure_ms),
            "rate_E_long": float(m.rate_E),
            "status_long": m.status,
        }

        verdict = "STABLE" if stable else "UNSTABLE"
        print(f"    CV: scan={cv_scan:.3f} → long={cv_long:.3f} "
              f"(Δ={cv_rel:.1%})  "
              f"Fano: scan={fano_scan:.3f} → long={fano_long:.3f} "
              f"(Δ={fano_rel:.1%})  [{verdict}]")
        return result

    except Exception as exc:
        print(f"    Stability check FAILED: {exc}")
        return {"stable": None, "reason": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE + R5 CLAMP
# ═══════════════════════════════════════════════════════════════════════════════

def save_edge_regimes(
    bundle_dir: Path,
    alpha_edge: float,
    rho_base: float,
    scan_info: dict,
    multipliers: Optional[List[float]] = None,
    edge_method: str = "metrics_scan",
    r5_clamp_steps: int = 2,
) -> list[dict]:
    """Generate 5 regimes from α_edge and save to bundle.

    R5 clamp logic: if R5 would be at mult=2.00 but that α caused FAIL
    in the scan, reduce to 1.80 then 1.60 (max r5_clamp_steps reductions).

    scan_info is the dict returned by find_alpha_edge() with keys:
        coarse_grid, refine_grid, merged_grid, edge_rule,
        alpha_edge_source, refine_direction.

    Writes regimes/regimes.json with structured metadata.
    """
    if multipliers is None:
        multipliers = [0.40, 0.70, 1.00, 1.40, 2.00]

    # Use merged_grid for R5 clamp logic
    merged_grid = scan_info.get("merged_grid", [])

    # R5 clamp: check if R5 α lands in or beyond FAIL territory
    r5_clamped = False
    r5_final_mult = multipliers[4]
    r5_alpha = alpha_edge * r5_final_mult

    fail_alphas = sorted(
        [r["alpha"] for r in merged_grid if r.get("gate") == "FAIL"])
    pass_warn_alphas = sorted(
        [r["alpha"] for r in merged_grid
         if r.get("gate") in ("PASS", "WARN")])

    if fail_alphas and pass_warn_alphas:
        max_ok_alpha = max(pass_warn_alphas)

        def _should_clamp(r5_a: float) -> bool:
            """Clamp if R5 is beyond all known-good points or if a FAIL
            point lies between alpha_edge and R5."""
            if r5_a > max_ok_alpha:
                return True
            # Any FAIL between alpha_edge and r5_a?
            return any(f > alpha_edge and f <= r5_a for f in fail_alphas)

        clamp_reductions = [0.20, 0.40]
        for step in range(r5_clamp_steps):
            if not _should_clamp(r5_alpha):
                break
            reduction = clamp_reductions[step] if step < len(clamp_reductions) else 0.20
            r5_final_mult -= reduction
            r5_alpha = alpha_edge * r5_final_mult
            r5_clamped = True

    final_multipliers = multipliers[:4] + [r5_final_mult]
    regimes = make_regimes_from_edge(alpha_edge, rho_base, final_multipliers)

    # Build output with structured metadata
    out_data = {
        "alpha_edge": float(alpha_edge),
        "rho_base": float(rho_base),
        "edge_method": edge_method,
        "alpha_edge_source": scan_info.get("alpha_edge_source", "unknown"),
        "refine_direction": scan_info.get("refine_direction"),
        "edge_rule": scan_info.get("edge_rule", {}),
        "r5_clamped": r5_clamped,
        "r5_final_multiplier": float(r5_final_mult),
        "multipliers": [float(m) for m in final_multipliers],
        "coarse_grid": scan_info.get("coarse_grid", []),
        "refine_grid": scan_info.get("refine_grid", []),
        "regimes": regimes,
    }

    # Add stability check results if present
    if "stability_check" in scan_info:
        out_data["stability_check"] = scan_info["stability_check"]

    bundle_dir = Path(bundle_dir)
    reg_dir = bundle_dir / "regimes"
    reg_dir.mkdir(exist_ok=True)
    out_path = reg_dir / "regimes.json"
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Wrote edge regimes: {out_path}")
    if r5_clamped:
        print(f"    R5 clamped: mult {multipliers[4]:.2f} → {r5_final_mult:.2f}")
    for r in regimes:
        print(f"    {r['name']:20s}  α={r['alpha']:.6f}  "
              f"mult={r['multiplier']:.2f}")
    return regimes
