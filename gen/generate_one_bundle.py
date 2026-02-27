#!/usr/bin/env python3
"""Generate one frozen reservoir bundle.

Usage:
    python -m gen.generate_one_bundle --seed 123 --N 100
    python -m gen.generate_one_bundle --seed 123 --N 100 --rho-targets 0.3 0.7 1.1 1.6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import GeneratorConfig, validate_config
from .seeding import split_seeds
from .population import make_population
from .graph_gen import generate_edges
from .hetero_neurons import sample_neuron_params
from .hetero_synapses import sample_synapse_params, build_sparse_W, rescale_for_balance
from .spectral import spectral_radius, block_stats
from .regimes import make_regimes
from .poisson_bank import generate_poisson_trains
from .io_bundle import write_bundle, write_manifest


def generate_bundle(cfg: GeneratorConfig, out_dir: Path | None = None) -> Path:
    """Run the full generation pipeline. Returns bundle directory path."""
    t0 = time.time()

    # ── Step 0: Validate ──
    validate_config(cfg)
    print(f"[0] Config OK  (N={cfg.N}, frac_I={cfg.frac_I}, "
          f"targets={cfg.rho_eff_targets})")

    # ── Step 1: Seeds ──
    seeds = split_seeds(cfg.seed)
    print(f"[1] Seeds: {seeds}")

    # ── Step 2: Population E/I ──
    pop = make_population(cfg.N, cfg.frac_I, seeds["population"])
    print(f"[2] Population: N_E={pop['N_E']}, N_I={pop['N_I']}")

    # ── Step 3: Graph ──
    edges = generate_edges(cfg, pop, seeds["graph"])
    n_syn = len(edges["pre"])
    print(f"[3] Graph: {n_syn} synapses ({cfg.graph_type})")

    # ── Step 4: Neuron heterogeneity ──
    neuron_params = sample_neuron_params(cfg, pop, seeds["neurons"])
    print(f"[4] Neuron params: {len(neuron_params)} parameters, "
          f"{cfg.N} neurons each")

    # ── Step 5: Synapse params + balance rescaling + W ──
    syn_params = sample_synapse_params(cfg, edges, pop, seeds["synapses"])
    bal_before = float(np.abs(syn_params["w"][syn_params["w"] < 0]).sum() /
                       (syn_params["w"][syn_params["w"] > 0].sum() + 1e-15))
    bal_after = rescale_for_balance(syn_params, target=cfg.target_balance)
    W = build_sparse_W(cfg.N, edges, syn_params["w"])
    print(f"[5] Synapse params: {n_syn} synapses, W shape={W.shape}")
    print(f"    balance: {bal_before:.3f} → {bal_after:.3f} "
          f"(target={cfg.target_balance})")

    # ── Step 6: Spectral radius + stats ──
    stats = block_stats(W, pop)
    rho_base = stats["rho_full"]
    print(f"[6] rho_base={rho_base:.4f}  rho_EE={stats['rho_EE']:.4f}  "
          f"balance={stats['balance']:.3f}")

    if rho_base < 1e-10:
        raise RuntimeError(f"rho_base={rho_base} ~ 0, cannot define regimes")
    if rho_base > 1e4:
        print(f"  WARNING: rho_base={rho_base} extremely large")

    # ── Step 7: 5 regimes (simple mode from ρ targets) ──
    regimes = make_regimes(rho_base, cfg.rho_eff_targets)
    for r in regimes:
        print(f"  {r['name']:20s}  α={r['alpha']:.6f}  ρ_target={r['rho_target']:.2f}")
    print(f"    (use `python -m gen.calibrate_edge` for edge-of-chaos calibration)")

    # ── Step 8: Poisson bank ──
    n_ch = cfg.poisson_n_channels if cfg.poisson_n_channels > 0 else cfg.N
    poisson = generate_poisson_trains(
        n_ch, cfg.poisson_T_s, cfg.poisson_rate_hz, seeds["poisson"])
    total_spikes = sum(len(t) for t in poisson)
    print(f"[8] Poisson: {n_ch} channels, {cfg.poisson_T_s}s, "
          f"{total_spikes} total spikes")

    # ── Step 9: Write bundle ──
    if out_dir is None:
        out_dir = Path(cfg.bundle_dir) / f"bundle_seed_{cfg.seed}"

    bundle_path = write_bundle(
        out_dir, cfg, seeds, pop, edges, neuron_params,
        syn_params, stats, regimes, poisson)

    manifest_path = write_manifest(bundle_path)
    elapsed = time.time() - t0
    print(f"[9] Bundle written: {bundle_path}/")
    print(f"    manifest: {manifest_path}")
    print(f"    time: {elapsed:.1f}s")

    return bundle_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate one frozen reservoir bundle")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--frac-I", type=float, default=0.2)
    ap.add_argument("--graph-type", default="ER", choices=["ER", "fixed_indegree"])
    ap.add_argument("--p-conn", type=float, default=0.2)
    ap.add_argument("--k-in", type=int, default=20)
    ap.add_argument("--poisson-T-s", type=float, default=3.0)
    ap.add_argument("--poisson-rate-hz", type=float, default=10.0)
    ap.add_argument("--rho-targets", type=float, nargs=5,
                    default=[0.50, 0.75, 1.00, 1.25, 1.60])
    ap.add_argument("--bundle-dir", default="bundles")
    args = ap.parse_args()

    cfg = GeneratorConfig(
        seed=args.seed,
        N=args.N,
        frac_I=args.frac_I,
        graph_type=args.graph_type,
        p_conn=args.p_conn,
        k_in=args.k_in,
        poisson_T_s=args.poisson_T_s,
        poisson_rate_hz=args.poisson_rate_hz,
        rho_eff_targets=args.rho_targets,
        bundle_dir=args.bundle_dir,
    )

    generate_bundle(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
