#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frozen Reservoir Pipeline — generates deterministic benchmark bundles.

For a given seed: generates frozen Poisson inputs + frozen network topology,
then defines 5 stability regimes (alpha scaling) on the same frozen base.
Everything is saved to disk for reproducible benchmarks.

Steps:
  0) Input validation
  1) Seed hierarchy (deterministic RNG tree)
  2) Poisson spike train generation & save  (K realizations × N channels)
  3) Frozen network base (topology + raw weights + STP + delays)
  4) 5 regime configurations (alpha scaling from ρ targets)
  5) Per-regime liveness validation (Brian2 HH+STP simulation)
  6) Bundle manifest & integrity

Usage:
  python pipeline.py --seed 100
  python pipeline.py --seed 100 --step 0123456     # all steps (default)
  python pipeline.py --seed 100 --step 0123         # freeze only (no Brian2)
  python pipeline.py --seed 100 --step 56           # validate + bundle existing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from numba import njit


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — PARAMETER DEFINITIONS & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """All pipeline parameters. Frozen after validation."""
    # Identity
    seed: int = 100

    # Network size
    N: int = 135
    frac_E: float = 0.8

    # Connectivity (ER probabilities)
    p_EE: float = 0.3
    p_EI: float = 0.2
    p_IE: float = 0.4
    p_II: float = 0.1

    # Base weight amplitudes (arbitrary units, sign applied by type)
    A_base_EE: float = 3.0
    A_base_EI: float = 6.0
    A_base_IE: float = 11.2
    A_base_II: float = 11.2

    # STP means
    U_EE: float = 0.50;  D_EE: float = 1100.0;  F_EE: float = 50.0;   tau_I_EE: float = 3.0
    U_EI: float = 0.05;  D_EI: float = 125.0;   F_EI: float = 1200.0; tau_I_EI: float = 3.0
    U_IE: float = 0.25;  D_IE: float = 700.0;   F_IE: float = 20.0;   tau_I_IE: float = 6.0
    U_II: float = 0.32;  D_II: float = 144.0;   F_II: float = 60.0;   tau_I_II: float = 6.0

    # Delays (ms)
    delay_EE: float = 1.5; delay_EI: float = 0.8
    delay_IE: float = 0.8; delay_II: float = 0.8
    delay_jitter: float = 0.1

    # Poisson input
    poisson_rate_hz: float = 20.0
    poisson_duration_ms: float = 6000.0    # warmup + max eval (1000+3000 + margin)
    poisson_K: int = 7                     # number of realizations
    dt_train_ms: float = 0.025             # binning resolution

    # Simulation
    dt_sim_ms: float = 0.025               # metrics integration step
    I_b_nA: float = 0.05                   # background current

    # Regime targets (5 spectral radius targets, logspace-like spread)
    rho_targets: List[float] = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.5])

    # Validation
    warmup_ms: float = 1000.0              # 1 second warmup
    validation_ms: float = 3000.0          # 3 second measurement window

    # Balance constraints
    balance_lo: float = 0.6
    balance_hi: float = 1.4

    # Output
    bundle_dir: str = "bundles"


SYN_TYPES = ("EE", "EI", "IE", "II")
SYN_SIGN = {"EE": +1.0, "EI": +1.0, "IE": -1.0, "II": -1.0}
REGIME_NAMES = [
    "R1_super_stable", "R2_stable", "R3_near_critical",
    "R4_edge_of_chaos", "R5_chaotic",
]


def validate_config(cfg: PipelineConfig) -> List[str]:
    """Step 0 tests: return list of error strings (empty = OK)."""
    errors = []

    if cfg.N < 2:
        errors.append(f"N={cfg.N} must be >= 2")
    if not (0.0 < cfg.frac_E < 1.0):
        errors.append(f"frac_E={cfg.frac_E} must be in (0,1)")

    for p_name in ("p_EE", "p_EI", "p_IE", "p_II"):
        val = getattr(cfg, p_name)
        if not (0.0 <= val <= 1.0):
            errors.append(f"{p_name}={val} must be in [0,1]")

    if cfg.dt_train_ms <= 0:
        errors.append(f"dt_train_ms={cfg.dt_train_ms} must be > 0")
    if cfg.dt_sim_ms <= 0:
        errors.append(f"dt_sim_ms={cfg.dt_sim_ms} must be > 0")
    if cfg.dt_train_ms < cfg.dt_sim_ms:
        errors.append(f"dt_train_ms={cfg.dt_train_ms} < dt_sim_ms={cfg.dt_sim_ms} "
                       f"(trains must be >= max dt to avoid multi-spike bins)")

    if cfg.poisson_duration_ms < cfg.warmup_ms + cfg.validation_ms:
        errors.append(f"duration={cfg.poisson_duration_ms} < warmup+validation="
                       f"{cfg.warmup_ms + cfg.validation_ms}")

    if len(cfg.rho_targets) != 5:
        errors.append(f"rho_targets must have exactly 5 elements, got {len(cfg.rho_targets)}")
    elif cfg.rho_targets != sorted(cfg.rho_targets):
        errors.append(f"rho_targets must be sorted ascending: {cfg.rho_targets}")
    elif len(set(cfg.rho_targets)) != 5:
        errors.append(f"rho_targets must be unique: {cfg.rho_targets}")

    if cfg.poisson_K < 1:
        errors.append(f"poisson_K={cfg.poisson_K} must be >= 1")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SEED HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SeedManifest:
    """Deterministic sub-seeds derived from master seed."""
    master: int
    graph: int
    weights: int
    stp: int
    delays: int
    poisson: int
    ei_split: int
    validation: int


def build_seed_manifest(seed: int) -> SeedManifest:
    """Derive deterministic sub-seeds from master seed.

    Uses SplitMix-style derivation: master → hash → sub-seeds.
    Guarantees: same seed → same sub-seeds, different seed → different sub-seeds.
    """
    rng = np.random.default_rng(seed)
    # Draw 8 independent sub-seeds (uint64 range)
    sub = rng.integers(0, 2**63, size=8, dtype=np.int64)
    return SeedManifest(
        master=seed,
        graph=int(sub[0]),
        weights=int(sub[1]),
        stp=int(sub[2]),
        delays=int(sub[3]),
        poisson=int(sub[4]),
        ei_split=int(sub[5]),
        validation=int(sub[6]),
    )


def test_seed_hierarchy(seed: int) -> List[str]:
    """Step 1 tests."""
    errors = []

    # Same seed → identical
    m1 = build_seed_manifest(seed)
    m2 = build_seed_manifest(seed)
    if m1 != m2:
        errors.append("FAIL: same seed gave different sub-seeds")

    # Different seed → different
    m3 = build_seed_manifest(seed + 1)
    if m1.graph == m3.graph and m1.weights == m3.weights:
        errors.append("FAIL: different seeds gave identical sub-seeds")

    # Serialization round-trip
    d = asdict(m1)
    m4 = SeedManifest(**d)
    if m1 != m4:
        errors.append("FAIL: serialization round-trip failed")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — POISSON SPIKE TRAINS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_poisson_trains(N: int, rate_hz: float, duration_ms: float,
                            rng: np.random.Generator, dt_train_ms: float
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate independent Poisson spike trains for N channels.

    Returns (indices, times_ms) globally sorted by time.
    Deduplication: max 1 spike per channel per dt_train_ms bin.
    """
    indices_list = []
    times_list = []
    lam = rate_hz * duration_ms / 1000.0

    for i in range(N):
        n_spikes = rng.poisson(lam)
        if n_spikes > 0:
            t = rng.uniform(0.0, duration_ms, n_spikes)
            bins = np.floor(t / dt_train_ms).astype(np.int64)
            bins = np.unique(bins)
            t = (bins.astype(np.float64) + rng.random(len(bins))) * dt_train_ms
            t = np.minimum(t, duration_ms - 1e-9)
            indices_list.append(np.full(len(t), i, dtype=np.int32))
            times_list.append(t)

    if not indices_list:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    indices = np.concatenate(indices_list)
    times = np.concatenate(times_list)
    order = np.argsort(times, kind="mergesort")
    return indices[order], times[order]


def generate_all_poisson(cfg: PipelineConfig, seeds: SeedManifest
                          ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate K realizations of (E_indices, E_times, I_indices, I_times).

    Each realization gets a deterministic sub-seed derived from seeds.poisson + k.
    """
    N_E = int(cfg.N * cfg.frac_E)
    N_I = cfg.N - N_E
    all_trains = []

    for k in range(cfg.poisson_K):
        rng_k = np.random.default_rng(seeds.poisson + k * 1000)
        idx_E, t_E = generate_poisson_trains(
            N_E, cfg.poisson_rate_hz, cfg.poisson_duration_ms,
            rng_k, cfg.dt_train_ms)
        idx_I, t_I = generate_poisson_trains(
            N_I, cfg.poisson_rate_hz, cfg.poisson_duration_ms,
            rng_k, cfg.dt_train_ms)
        all_trains.append((idx_E, t_E, idx_I, t_I))

    return all_trains


def save_poisson(trains: list, path: Path, cfg: PipelineConfig) -> None:
    """Save K Poisson realizations to NPZ."""
    data = {
        "K": len(trains),
        "N_E": int(cfg.N * cfg.frac_E),
        "N_I": cfg.N - int(cfg.N * cfg.frac_E),
        "rate_hz": cfg.poisson_rate_hz,
        "duration_ms": cfg.poisson_duration_ms,
        "dt_train_ms": cfg.dt_train_ms,
    }
    for k, (idx_E, t_E, idx_I, t_I) in enumerate(trains):
        data[f"E_idx_{k}"] = idx_E
        data[f"E_times_{k}"] = t_E
        data[f"I_idx_{k}"] = idx_I
        data[f"I_times_{k}"] = t_I
    np.savez_compressed(path, **data)


def load_poisson(path: Path) -> Tuple[List, dict]:
    """Load Poisson trains from NPZ. Returns (trains_list, metadata)."""
    raw = np.load(path, allow_pickle=False)
    K = int(raw["K"])
    meta = {
        "K": K,
        "N_E": int(raw["N_E"]),
        "N_I": int(raw["N_I"]),
        "rate_hz": float(raw["rate_hz"]),
        "duration_ms": float(raw["duration_ms"]),
        "dt_train_ms": float(raw["dt_train_ms"]),
    }
    trains = []
    for k in range(K):
        trains.append((
            raw[f"E_idx_{k}"], raw[f"E_times_{k}"],
            raw[f"I_idx_{k}"], raw[f"I_times_{k}"],
        ))
    return trains, meta


def test_poisson(trains: list, cfg: PipelineConfig, seeds: SeedManifest) -> List[str]:
    """Step 2 tests."""
    errors = []
    N_E = int(cfg.N * cfg.frac_E)
    N_I = cfg.N - N_E

    # Determinism: regenerate and compare bitwise
    trains2 = generate_all_poisson(cfg, seeds)
    for k in range(cfg.poisson_K):
        for j, label in enumerate(("E_idx", "E_times", "I_idx", "I_times")):
            if not np.array_equal(trains[k][j], trains2[k][j]):
                errors.append(f"FAIL: realization k={k} {label} not bitwise identical")
                break

    for k in range(cfg.poisson_K):
        idx_E, t_E, idx_I, t_I = trains[k]

        # Indices in range
        if len(idx_E) > 0 and (idx_E.min() < 0 or idx_E.max() >= N_E):
            errors.append(f"FAIL: k={k} E indices out of [0, {N_E})")
        if len(idx_I) > 0 and (idx_I.min() < 0 or idx_I.max() >= N_I):
            errors.append(f"FAIL: k={k} I indices out of [0, {N_I})")

        # Times in range
        for t, label in [(t_E, "E"), (t_I, "I")]:
            if len(t) > 0:
                if t.min() < 0:
                    errors.append(f"FAIL: k={k} {label} times < 0")
                if t.max() >= cfg.poisson_duration_ms:
                    errors.append(f"FAIL: k={k} {label} times >= duration")
                # Monotonicity (global sort)
                if not np.all(np.diff(t) >= 0):
                    errors.append(f"FAIL: k={k} {label} times not sorted")

        # Statistical sanity: spike count ~ rate * duration * N
        dur_s = cfg.poisson_duration_ms / 1000.0
        expected_E = cfg.poisson_rate_hz * dur_s * N_E
        actual_E = len(t_E)
        if actual_E < 0.3 * expected_E or actual_E > 3.0 * expected_E:
            errors.append(f"WARN: k={k} E spike count {actual_E} vs expected ~{expected_E:.0f}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FROZEN NETWORK BASE
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _power_iteration(data, indices, indptr, n, n_iter=80):
    """Spectral radius via power iteration on CSR matrix."""
    x = np.ones(n, dtype=np.float64)
    x /= np.sqrt((x * x).sum())
    for _ in range(n_iter):
        y = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                s += data[k] * x[indices[k]]
            y[i] = s
        norm = np.sqrt((y * y).sum())
        if norm < 1e-15:
            return 0.0
        x = y / norm
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        y[i] = s
    return np.sqrt((y * y).sum())


@dataclass
class FrozenNetwork:
    """Complete frozen reservoir base (topology + weights + STP + delays).

    All arrays are deterministic for a given seed. Alpha scaling is NOT applied
    here — that's per-regime (Step 4).
    """
    N: int
    N_E: int
    N_I: int
    idx_E: np.ndarray          # (N_E,) int32
    idx_I: np.ndarray          # (N_I,) int32

    # Per synapse type: pre/post indices, raw magnitudes, STP params, delays
    edges: Dict[str, Dict[str, np.ndarray]]   # {type: {pre, post, raw_mag, U, D, F, delay, tau_I}}

    # CSR template (alpha=1, for spectral radius computation)
    _csr_template: sparse.csr_matrix


def build_frozen_network(cfg: PipelineConfig, seeds: SeedManifest) -> FrozenNetwork:
    """Build the frozen network base from seed."""
    N = cfg.N
    N_E = int(N * cfg.frac_E)
    N_I = N - N_E

    # ── E/I split ──
    rng_ei = np.random.default_rng(seeds.ei_split)
    idx = np.arange(N, dtype=np.int32)
    rng_ei.shuffle(idx)
    idx_E = np.sort(idx[:N_E])
    idx_I = np.sort(idx[N_E:])

    is_E = np.zeros(N, dtype=np.bool_)
    is_E[idx_E] = True

    # ── Edges ──
    rng_graph = np.random.default_rng(seeds.graph)
    syn_specs = [
        ("EE", idx_E, idx_E, cfg.p_EE, True),
        ("EI", idx_E, idx_I, cfg.p_EI, False),
        ("IE", idx_I, idx_E, cfg.p_IE, False),
        ("II", idx_I, idx_I, cfg.p_II, True),
    ]

    edges = {}
    for syn_type, src_idx, tgt_idx, prob, same_pop in syn_specs:
        # Deterministic sub-rng per type
        type_seed = seeds.graph + hash(syn_type) % (2**31)
        rng_edge = np.random.default_rng(type_seed)

        pre_list, post_list = [], []
        for si in range(len(src_idx)):
            for ti in range(len(tgt_idx)):
                if same_pop and si == ti:
                    continue
                if rng_edge.random() < prob:
                    pre_list.append(src_idx[si])
                    post_list.append(tgt_idx[ti])

        pre = np.array(pre_list, dtype=np.int32)
        post = np.array(post_list, dtype=np.int32)
        n_syn = len(pre)

        # ── Raw magnitudes (positive, unscaled) ──
        rng_w = np.random.default_rng(seeds.weights + hash(syn_type) % (2**31))
        A_base = getattr(cfg, f"A_base_{syn_type}")
        raw_mag = rng_w.gamma(shape=1.0, scale=A_base, size=n_syn).astype(np.float64)

        # ── STP params ──
        rng_stp = np.random.default_rng(seeds.stp + hash(syn_type) % (2**31))
        U_mean = getattr(cfg, f"U_{syn_type}")
        D_mean = getattr(cfg, f"D_{syn_type}")
        F_mean = getattr(cfg, f"F_{syn_type}")
        tau_I_val = getattr(cfg, f"tau_I_{syn_type}")

        U = np.clip(rng_stp.normal(U_mean, 0.5 * U_mean, n_syn), 1e-3, 1.0)
        D = np.clip(rng_stp.normal(D_mean, 0.5 * D_mean, n_syn), 1.0, None)
        F = np.clip(rng_stp.normal(F_mean, 0.5 * F_mean, n_syn), 0.0, None)
        tau_I = np.full(n_syn, tau_I_val, dtype=np.float64)

        # ── Delays ──
        rng_del = np.random.default_rng(seeds.delays + hash(syn_type) % (2**31))
        delay_mean = getattr(cfg, f"delay_{syn_type}")
        delay = np.clip(
            rng_del.normal(delay_mean, cfg.delay_jitter, n_syn),
            0.1, None
        )

        edges[syn_type] = {
            "pre": pre, "post": post, "raw_mag": raw_mag,
            "U": U, "D": D, "F": F, "delay": delay, "tau_I": tau_I,
        }

    # ── CSR template (alpha=1, for spectral radius) ──
    rows, cols, vals = [], [], []
    for syn_type in SYN_TYPES:
        e = edges[syn_type]
        if len(e["pre"]) == 0:
            continue
        rows.append(e["post"])
        cols.append(e["pre"])
        sign = SYN_SIGN[syn_type]
        vals.append(sign * e["raw_mag"])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    csr = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    return FrozenNetwork(
        N=N, N_E=N_E, N_I=N_I,
        idx_E=idx_E, idx_I=idx_I,
        edges=edges,
        _csr_template=csr,
    )


def spectral_radius(net: FrozenNetwork, alpha: float = 1.0) -> float:
    """Compute spectral radius of alpha * W."""
    W = net._csr_template.copy()
    W.data = W.data * alpha
    return float(_power_iteration(W.data, W.indices, W.indptr, net.N, n_iter=80))


def ei_balance(net: FrozenNetwork, alpha: float = 1.0) -> float:
    """Compute |sum_I| / |sum_E| ratio."""
    d = net._csr_template.data * alpha
    sum_E = d[d > 0].sum()
    sum_I = np.abs(d[d < 0]).sum()
    return float(sum_I / (sum_E + 1e-15))


def save_network(net: FrozenNetwork, path: Path) -> None:
    """Save frozen network to NPZ."""
    data = {
        "N": net.N, "N_E": net.N_E, "N_I": net.N_I,
        "idx_E": net.idx_E, "idx_I": net.idx_I,
    }
    for syn_type in SYN_TYPES:
        e = net.edges[syn_type]
        for key, arr in e.items():
            data[f"{syn_type}_{key}"] = arr
    np.savez_compressed(path, **data)


def load_network(path: Path) -> FrozenNetwork:
    """Load frozen network from NPZ."""
    raw = np.load(path, allow_pickle=False)
    N = int(raw["N"])
    N_E = int(raw["N_E"])
    N_I = int(raw["N_I"])
    idx_E = raw["idx_E"]
    idx_I = raw["idx_I"]

    edges = {}
    for syn_type in SYN_TYPES:
        edges[syn_type] = {
            "pre": raw[f"{syn_type}_pre"],
            "post": raw[f"{syn_type}_post"],
            "raw_mag": raw[f"{syn_type}_raw_mag"],
            "U": raw[f"{syn_type}_U"],
            "D": raw[f"{syn_type}_D"],
            "F": raw[f"{syn_type}_F"],
            "delay": raw[f"{syn_type}_delay"],
            "tau_I": raw[f"{syn_type}_tau_I"],
        }

    # Rebuild CSR template
    rows, cols, vals = [], [], []
    for syn_type in SYN_TYPES:
        e = edges[syn_type]
        if len(e["pre"]) == 0:
            continue
        rows.append(e["post"])
        cols.append(e["pre"])
        vals.append(SYN_SIGN[syn_type] * e["raw_mag"])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    csr = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    return FrozenNetwork(N=N, N_E=N_E, N_I=N_I,
                         idx_E=idx_E, idx_I=idx_I,
                         edges=edges, _csr_template=csr)


def test_network(net: FrozenNetwork, cfg: PipelineConfig, seeds: SeedManifest) -> List[str]:
    """Step 3 tests."""
    errors = []

    # Determinism
    net2 = build_frozen_network(cfg, seeds)
    if not np.array_equal(net.idx_E, net2.idx_E):
        errors.append("FAIL: idx_E not deterministic")
    if not np.array_equal(net.idx_I, net2.idx_I):
        errors.append("FAIL: idx_I not deterministic")
    for syn_type in SYN_TYPES:
        for key in ("pre", "post", "raw_mag"):
            if not np.array_equal(net.edges[syn_type][key], net2.edges[syn_type][key]):
                errors.append(f"FAIL: {syn_type}.{key} not deterministic")

    # Size consistency
    if net.N_E + net.N_I != net.N:
        errors.append(f"FAIL: N_E({net.N_E}) + N_I({net.N_I}) != N({net.N})")

    # No self-loops (for same-population types)
    for syn_type in ("EE", "II"):
        e = net.edges[syn_type]
        self_loops = np.sum(e["pre"] == e["post"])
        if self_loops > 0:
            errors.append(f"FAIL: {syn_type} has {self_loops} self-loops")

    # All indices in [0, N)
    for syn_type in SYN_TYPES:
        e = net.edges[syn_type]
        if len(e["pre"]) > 0:
            if e["pre"].min() < 0 or e["pre"].max() >= net.N:
                errors.append(f"FAIL: {syn_type} pre indices out of [0, {net.N})")
            if e["post"].min() < 0 or e["post"].max() >= net.N:
                errors.append(f"FAIL: {syn_type} post indices out of [0, {net.N})")

    # Non-zero connectivity
    total_syn = sum(len(net.edges[s]["pre"]) for s in SYN_TYPES)
    if total_syn == 0:
        errors.append("FAIL: zero total synapses")

    # Spectral radius sanity
    rho = spectral_radius(net, alpha=1.0)
    if rho < 1e-10:
        errors.append(f"FAIL: rho(alpha=1)={rho:.6f} ~ 0")

    # Scaling linearity check
    rho_half = spectral_radius(net, alpha=0.5)
    rho_double = spectral_radius(net, alpha=2.0)
    err_half = abs(rho_half - 0.5 * rho) / (rho + 1e-15)
    err_double = abs(rho_double - 2.0 * rho) / (rho + 1e-15)
    if err_half > 0.02 or err_double > 0.02:
        errors.append(f"WARN: non-linear scaling (err_half={err_half:.4f}, "
                       f"err_double={err_double:.4f})")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — 5 REGIME CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockDiagnostics:
    """Per-regime block-level matrix diagnostics."""
    rho_full: float
    rho_EE: float
    norm_EE: float    # Frobenius norm
    norm_EI: float
    norm_IE: float
    norm_II: float
    ei_incoming_ratio: float   # mean(|I_incoming|) / mean(|E_incoming|) on E neurons


@dataclass
class RegimeConfig:
    """Single regime configuration."""
    name: str
    index: int
    alpha: float
    rho_target: float
    rho_actual: float
    balance: float
    block_diag: Optional[BlockDiagnostics] = None


def compute_block_diagnostics(net: FrozenNetwork, alpha: float) -> BlockDiagnostics:
    """Compute per-block spectral/norm diagnostics for a given alpha."""
    N = net.N

    # ── Build block sub-matrices ──
    def _block_csr(syn_type):
        e = net.edges[syn_type]
        if len(e["pre"]) == 0:
            return sparse.csr_matrix((N, N))
        sign = SYN_SIGN[syn_type]
        return sparse.csr_matrix(
            (sign * alpha * e["raw_mag"], (e["post"], e["pre"])),
            shape=(N, N))

    W_EE = _block_csr("EE")
    W_EI = _block_csr("EI")
    W_IE = _block_csr("IE")
    W_II = _block_csr("II")

    # ── ρ_EE (spectral radius of E→E block only) ──
    rho_EE = float(_power_iteration(W_EE.data, W_EE.indices, W_EE.indptr, N, n_iter=80))

    # ── ρ_full ──
    rho_full = spectral_radius(net, alpha=alpha)

    # ── Frobenius norms ──
    norm_EE = float(sparse.linalg.norm(W_EE, 'fro'))
    norm_EI = float(sparse.linalg.norm(W_EI, 'fro'))
    norm_IE = float(sparse.linalg.norm(W_IE, 'fro'))
    norm_II = float(sparse.linalg.norm(W_II, 'fro'))

    # ── E/I incoming ratio on E neurons ──
    # For each E neuron: sum of |excitatory incoming| vs |inhibitory incoming|
    W_full = net._csr_template.copy()
    W_full.data = W_full.data * alpha
    e_incoming_exc = []
    e_incoming_inh = []
    for idx in net.idx_E:
        row = W_full.getrow(int(idx)).toarray().ravel()
        e_incoming_exc.append(row[row > 0].sum())
        e_incoming_inh.append(np.abs(row[row < 0]).sum())
    mean_exc = float(np.mean(e_incoming_exc)) if e_incoming_exc else 0.0
    mean_inh = float(np.mean(e_incoming_inh)) if e_incoming_inh else 0.0
    ei_ratio = mean_inh / (mean_exc + 1e-15)

    return BlockDiagnostics(
        rho_full=rho_full, rho_EE=rho_EE,
        norm_EE=norm_EE, norm_EI=norm_EI,
        norm_IE=norm_IE, norm_II=norm_II,
        ei_incoming_ratio=ei_ratio,
    )


def compute_regimes(net: FrozenNetwork, cfg: PipelineConfig) -> List[RegimeConfig]:
    """Compute alpha for each ρ target via direct scaling, with block diagnostics."""
    rho_base = spectral_radius(net, alpha=1.0)
    regimes = []

    for i, rho_target in enumerate(cfg.rho_targets):
        alpha = rho_target / rho_base
        rho_actual = spectral_radius(net, alpha=alpha)
        bal = ei_balance(net, alpha=alpha)
        diag = compute_block_diagnostics(net, alpha)

        regimes.append(RegimeConfig(
            name=REGIME_NAMES[i],
            index=i,
            alpha=alpha,
            rho_target=rho_target,
            rho_actual=rho_actual,
            balance=bal,
            block_diag=diag,
        ))

    return regimes


def save_regimes(regimes: List[RegimeConfig], path: Path) -> None:
    """Save regime configs to JSON."""
    data = [asdict(r) for r in regimes]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_regimes(path: Path) -> List[RegimeConfig]:
    """Load regime configs from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    regimes = []
    for d in data:
        bd = d.pop("block_diag", None)
        if bd is not None:
            bd = BlockDiagnostics(**bd)
        regimes.append(RegimeConfig(**d, block_diag=bd))
    return regimes


def test_regimes(regimes: List[RegimeConfig], cfg: PipelineConfig) -> List[str]:
    """Step 4 tests."""
    errors = []

    if len(regimes) != 5:
        errors.append(f"FAIL: expected 5 regimes, got {len(regimes)}")
        return errors

    # Each regime references valid alpha
    for r in regimes:
        if r.alpha <= 0:
            errors.append(f"FAIL: {r.name} alpha={r.alpha} <= 0")

    # Alpha monotonically increasing (since rho_targets are sorted and scaling is linear)
    alphas = [r.alpha for r in regimes]
    if alphas != sorted(alphas):
        errors.append(f"WARN: alphas not monotonic: {alphas}")

    # ρ actual close to target
    for r in regimes:
        err = abs(r.rho_actual - r.rho_target) / (r.rho_target + 1e-15)
        if err > 0.05:
            errors.append(f"WARN: {r.name} rho_actual={r.rho_actual:.4f} vs "
                           f"target={r.rho_target:.4f} (err={err:.2%})")

    # Balance in range
    for r in regimes:
        if not (cfg.balance_lo <= r.balance <= cfg.balance_hi):
            errors.append(f"WARN: {r.name} balance={r.balance:.3f} "
                           f"out of [{cfg.balance_lo}, {cfg.balance_hi}]")

    # Serialization round-trip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as f:
        p = Path(f.name)
        save_regimes(regimes, p)
        loaded = load_regimes(p)
        for orig, load in zip(regimes, loaded):
            if abs(orig.alpha - load.alpha) > 1e-12:
                errors.append("FAIL: regime serialization round-trip failed")
                break

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PER-REGIME LIVENESS VALIDATION (Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

def _import_brian2():
    """Lazy import of Brian2 (heavy, only needed for step 5)."""
    from brian2 import (
        start_scope, NeuronGroup, Synapses, SpikeGeneratorGroup,
        SpikeMonitor, Network, Equations, defaultclock,
    )
    from brian2 import ms as b_ms, mV as b_mV, nA as b_nA, Hz as b_Hz
    from brian2 import umetre, ufarad, siemens, msiemens, cm

    # HH biophysical constants
    _area = 20000 * umetre ** 2
    constants = {
        "Cm": 1 * ufarad * cm ** -2 * _area,
        "gl": 5e-5 * siemens * cm ** -2 * _area,
        "El": -65 * b_mV,
        "EK": -90 * b_mV,
        "ENa": 50 * b_mV,
        "g_na": 100 * msiemens * cm ** -2 * _area,
        "g_kd": 30 * msiemens * cm ** -2 * _area,
        "VT": -63 * b_mV,
    }
    return {
        "start_scope": start_scope,
        "NeuronGroup": NeuronGroup,
        "Synapses": Synapses,
        "SpikeGeneratorGroup": SpikeGeneratorGroup,
        "SpikeMonitor": SpikeMonitor,
        "Network": Network,
        "Equations": Equations,
        "defaultclock": defaultclock,
        "ms": b_ms, "mV": b_mV, "nA": b_nA, "Hz": b_Hz,
        **constants,
    }


HH_EQUATIONS_STR = '''
dv/dt = (gl*(El-v)
        - g_na*(m**3)*h*(v-ENa)
        - g_kd*(n**4)*(v-EK)
        + I_total) / Cm : volt

dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)
        -0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1

dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1-n)
        -.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1

dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1-h)
        -4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1

I_b : ampere (shared, constant)
tau_stimulus : second (constant)
dI_stimulus/dt = -I_stimulus/tau_stimulus : ampere

I_syn_ee_synapses : ampere
I_syn_ei_synapses : ampere
I_syn_ie_synapses : ampere
I_syn_ii_synapses : ampere

I_total = I_b + I_stimulus
          + I_syn_ee_synapses + I_syn_ei_synapses
          + I_syn_ie_synapses + I_syn_ii_synapses : ampere
'''

STP_EQUATIONS_TEMPLATE = """
A : ampere (constant)
U : 1 (constant)
tau_I : second (shared, constant)
D : second (constant)
dx/dt =  z/D       : 1 (clock-driven)
dy/dt = -y/tau_I   : 1 (clock-driven)
z = 1 - x - y      : 1
I_syn_{name}_post = A*y : ampere (summed)
du/dt = -u/F : 1 (clock-driven)
F : second (constant)
"""

STP_ON_PRE = """
u += U*(1-u)
y += u*x
x += -u*x
"""

SYN_BRIAN_NAMES = {"EE": "ee_synapses", "EI": "ei_synapses",
                   "IE": "ie_synapses", "II": "ii_synapses"}


def build_brian2_network(net: FrozenNetwork, alpha: float,
                         trains_E: Tuple[np.ndarray, np.ndarray],
                         trains_I: Tuple[np.ndarray, np.ndarray],
                         cfg: PipelineConfig):
    """Build Brian2 HH+STP network from frozen network + Poisson trains."""
    b = _import_brian2()
    b["start_scope"]()

    N = net.N
    neurons = b["NeuronGroup"](
        N, b["Equations"](HH_EQUATIONS_STR),
        threshold='v > -40*mV',
        refractory='v > -40*mV',
        method='exponential_euler',
        name='neurons',
        namespace={k: b[k] for k in ("Cm", "gl", "El", "EK", "ENa", "g_na", "g_kd", "VT")},
    )
    neurons.v = b["El"]
    neurons.m = 0
    neurons.h = 1
    neurons.n = 0.5
    neurons.I_b = cfg.I_b_nA * b["nA"]
    neurons.I_stimulus = 0 * b["nA"]
    neurons.tau_stimulus = 3 * b["ms"]
    for ii in net.idx_I:
        neurons.tau_stimulus[int(ii)] = 6 * b["ms"]

    objects = [neurons]

    # ── STP Synapses ──
    for syn_type in SYN_TYPES:
        syn_name = SYN_BRIAN_NAMES[syn_type]
        sign = SYN_SIGN[syn_type]
        e = net.edges[syn_type]

        if len(e["pre"]) == 0:
            continue

        eqs_syn = STP_EQUATIONS_TEMPLATE.format(name=syn_name)
        syn = b["Synapses"](
            neurons, neurons, model=eqs_syn, on_pre=STP_ON_PRE,
            method='exact', name=syn_name,
        )
        syn.connect(i=e["pre"], j=e["post"])

        syn.A = sign * alpha * e["raw_mag"] * b["nA"]
        syn.U = e["U"]
        syn.D = e["D"] * b["ms"]
        syn.F = np.maximum(e["F"], 0.1) * b["ms"]
        syn.delay = e["delay"] * b["ms"]
        syn.tau_I = e["tau_I"][0] * b["ms"]
        syn.x = 1
        syn.u = e["U"]
        objects.append(syn)

    # ── Spike generators ──
    sg_E_idx, sg_E_times = trains_E
    sg_I_idx, sg_I_times = trains_I
    sg_E = b["SpikeGeneratorGroup"](net.N_E, sg_E_idx, sg_E_times * b["ms"], name='sg_E')
    sg_I = b["SpikeGeneratorGroup"](net.N_I, sg_I_idx, sg_I_times * b["ms"], name='sg_I')

    syn_sg_E = b["Synapses"](sg_E, neurons, on_pre='I_stimulus += 1.5*nA', name='syn_sg_E')
    syn_sg_E.connect(i=np.arange(net.N_E), j=net.idx_E)
    syn_sg_I = b["Synapses"](sg_I, neurons, on_pre='I_stimulus += 0.75*nA', name='syn_sg_I')
    syn_sg_I.connect(i=np.arange(net.N_I), j=net.idx_I)
    objects.extend([sg_E, sg_I, syn_sg_E, syn_sg_I])

    spike_mon = b["SpikeMonitor"](neurons, name='spike_monitor')
    objects.append(spike_mon)

    b["defaultclock"].dt = cfg.dt_sim_ms * b["ms"]
    brian_net = b["Network"](objects)
    brian_net.store()

    return brian_net, spike_mon, neurons, b


@dataclass
class ValidationResult:
    """Liveness validation result for one regime."""
    regime: str
    alpha: float
    rate_E: float
    rate_I: float
    sync_E: float             # population synchrony (E neurons)
    sync_I: float             # population synchrony (I neurons)
    cv_isi_E: float           # median CV of ISI (E neurons)
    cv_isi_I: float           # median CV of ISI (I neurons)
    fano_E: float             # Fano factor (E neurons)
    fano_I: float             # Fano factor (I neurons)
    has_nan: bool
    status: str               # OK, SILENT, RUNAWAY, NAN


def _compute_sync(spike_trains_dict, indices, warmup_ms, total_ms, b,
                   bin_ms: float = 5.0) -> float:
    """Population synchrony = var(pop_signal) / (N * mean(var_individual))."""
    n_bins = max(1, int((total_ms - warmup_ms) / bin_ms))
    pop_hist = np.zeros(n_bins, dtype=np.float64)
    ind_vars = []
    for idx in indices:
        t_ms_arr = np.array(spike_trains_dict[int(idx)] / b["ms"])
        t_ms_arr = t_ms_arr[(t_ms_arr >= warmup_ms) & (t_ms_arr < total_ms)]
        hist, _ = np.histogram(t_ms_arr, bins=n_bins,
                               range=(warmup_ms, total_ms))
        pop_hist += hist
        ind_vars.append(float(np.var(hist)))
    pop_var = float(np.var(pop_hist))
    mean_ind_var = float(np.mean(ind_vars)) if ind_vars else 0.0
    return pop_var / (len(indices) * mean_ind_var + 1e-15)


def _compute_cv_isi(spike_trains_dict, indices, warmup_ms, total_ms,
                    b) -> float:
    """Median CV of ISI across neurons with >= 2 spikes."""
    cvs = []
    for idx in indices:
        t_ms_arr = np.array(spike_trains_dict[int(idx)] / b["ms"])
        t_ms_arr = t_ms_arr[(t_ms_arr >= warmup_ms) & (t_ms_arr < total_ms)]
        if len(t_ms_arr) >= 2:
            isi = np.diff(t_ms_arr)
            mean_isi = isi.mean()
            if mean_isi > 1e-9:
                cvs.append(float(isi.std() / mean_isi))
    return float(np.median(cvs)) if cvs else 0.0


def _compute_fano(spike_trains_dict, indices, warmup_ms, total_ms,
                  b, bin_ms: float = 50.0) -> float:
    """Fano factor = var(counts) / mean(counts) across neurons, in bins."""
    n_bins = max(1, int((total_ms - warmup_ms) / bin_ms))
    all_counts = []
    for idx in indices:
        t_ms_arr = np.array(spike_trains_dict[int(idx)] / b["ms"])
        t_ms_arr = t_ms_arr[(t_ms_arr >= warmup_ms) & (t_ms_arr < total_ms)]
        hist, _ = np.histogram(t_ms_arr, bins=n_bins,
                               range=(warmup_ms, total_ms))
        all_counts.append(hist)
    if not all_counts:
        return 0.0
    counts = np.array(all_counts)       # (N_pop, n_bins)
    per_neuron_fano = []
    for row in counts:
        m = row.mean()
        if m > 1e-9:
            per_neuron_fano.append(float(row.var() / m))
    return float(np.median(per_neuron_fano)) if per_neuron_fano else 0.0


def validate_regime_liveness(
    regime: RegimeConfig,
    net: FrozenNetwork,
    trains: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cfg: PipelineConfig,
    seeds: SeedManifest,
) -> ValidationResult:
    """Run short Brian2 sim and check liveness + compute separation metrics."""
    idx_E_t, t_E, idx_I_t, t_I = trains
    total_ms = cfg.warmup_ms + cfg.validation_ms

    brian_net, spike_mon, neurons, b = build_brian2_network(
        net, regime.alpha, (idx_E_t, t_E), (idx_I_t, t_I), cfg)

    rng_v = np.random.default_rng(seeds.validation + regime.index)
    brian_net.restore()
    neurons.v = (float(b["El"] / b["mV"]) + rng_v.uniform(-2, 2, net.N)) * b["mV"]
    brian_net.run(total_ms * b["ms"])

    st = spike_mon.spike_trains()

    # ── Firing rates ──
    def _rate(indices):
        counts = []
        dur_s = cfg.validation_ms / 1000.0
        for idx in indices:
            t_ms_arr = np.array(st[int(idx)] / b["ms"])
            counts.append(np.sum((t_ms_arr >= cfg.warmup_ms) & (t_ms_arr < total_ms)))
        return float(np.mean(counts) / dur_s) if counts else 0.0

    rate_E = _rate(net.idx_E)
    rate_I = _rate(net.idx_I)

    # ── Synchrony (E and I separately) ──
    sync_E = _compute_sync(st, net.idx_E, cfg.warmup_ms, total_ms, b)
    sync_I = _compute_sync(st, net.idx_I, cfg.warmup_ms, total_ms, b)

    # ── CV ISI (E and I separately) ──
    cv_isi_E = _compute_cv_isi(st, net.idx_E, cfg.warmup_ms, total_ms, b)
    cv_isi_I = _compute_cv_isi(st, net.idx_I, cfg.warmup_ms, total_ms, b)

    # ── Fano factor (E and I separately, 50ms bins) ──
    fano_E = _compute_fano(st, net.idx_E, cfg.warmup_ms, total_ms, b, bin_ms=50.0)
    fano_I = _compute_fano(st, net.idx_I, cfg.warmup_ms, total_ms, b, bin_ms=50.0)

    # ── NaN check ──
    v_arr = np.array(neurons.v / b["mV"])
    has_nan = bool(np.any(np.isnan(v_arr)) or np.any(np.isinf(v_arr)))

    # ── Status ──
    if has_nan:
        status = "NAN"
    elif rate_E < 0.01 and rate_I < 0.01:
        status = "SILENT"
    elif rate_E > 200 or rate_I > 300:
        status = "RUNAWAY"
    else:
        status = "OK"

    return ValidationResult(
        regime=regime.name, alpha=regime.alpha,
        rate_E=rate_E, rate_I=rate_I,
        sync_E=sync_E, sync_I=sync_I,
        cv_isi_E=cv_isi_E, cv_isi_I=cv_isi_I,
        fano_E=fano_E, fano_I=fano_I,
        has_nan=has_nan, status=status,
    )


def test_validation(results: List[ValidationResult]) -> List[str]:
    """Step 5 tests: liveness + regime separation.

    Separation criterion: for at least 1 metric M across the 5 regimes,
      max(M) - min(M) >= rel_thr * median(M)
    where rel_thr = 0.15 (15%).
    """
    REL_THR = 0.15
    errors = []

    # ── Liveness checks ──
    for r in results:
        if r.status == "NAN":
            errors.append(f"FAIL: {r.regime} has NaN/Inf in state")
        elif r.status == "SILENT":
            errors.append(f"WARN: {r.regime} is silent (rate_E={r.rate_E:.1f})")
        elif r.status == "RUNAWAY":
            errors.append(f"FAIL: {r.regime} runaway (rate_E={r.rate_E:.1f})")

    # ── Regime separation check ──
    ok_results = [r for r in results if r.status == "OK"]
    if len(ok_results) < 2:
        return errors   # can't test separation with < 2 OK regimes

    metric_names = [
        ("rate_E",   [r.rate_E   for r in ok_results]),
        ("rate_I",   [r.rate_I   for r in ok_results]),
        ("sync_E",   [r.sync_E   for r in ok_results]),
        ("sync_I",   [r.sync_I   for r in ok_results]),
        ("cv_isi_E", [r.cv_isi_E for r in ok_results]),
        ("cv_isi_I", [r.cv_isi_I for r in ok_results]),
        ("fano_E",   [r.fano_E   for r in ok_results]),
        ("fano_I",   [r.fano_I   for r in ok_results]),
    ]

    sep_details = []
    any_pass = False
    for name, vals in metric_names:
        mn, mx = min(vals), max(vals)
        med = float(np.median(vals))
        span = mx - mn
        thr = REL_THR * med if med > 1e-12 else REL_THR
        passed = span >= thr
        if passed:
            any_pass = True
        sep_details.append((name, mn, mx, med, span, thr, passed))

    if not any_pass:
        errors.append(
            f"WARN: regimes too similar — no metric exceeds {REL_THR:.0%} "
            f"relative spread across {len(ok_results)} OK regimes")
        for name, mn, mx, med, span, thr, passed in sep_details:
            errors.append(f"  {name:10s}: min={mn:.3f} max={mx:.3f} "
                          f"span={span:.3f} thr={thr:.3f} {'PASS' if passed else 'FAIL'}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — BUNDLE MANIFEST & INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

def _file_hash(path: Path) -> str:
    """SHA256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle_manifest(bundle_dir: Path, cfg: PipelineConfig, seeds: SeedManifest,
                          regimes: List[RegimeConfig],
                          validation: Optional[List[ValidationResult]]) -> Path:
    """Write manifest.json with hashes and full config."""
    manifest = {
        "version": "1.0",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": cfg.seed,
        "config": asdict(cfg),
        "seeds": asdict(seeds),
        "regimes": [asdict(r) for r in regimes],
        "validation": [asdict(v) for v in validation] if validation else None,
        "files": {},
    }

    # Hash all data files
    for p in sorted(bundle_dir.rglob("*.npz")):
        rel = str(p.relative_to(bundle_dir))
        manifest["files"][rel] = _file_hash(p)
    for p in sorted(bundle_dir.rglob("*.json")):
        if p.name != "manifest.json":
            rel = str(p.relative_to(bundle_dir))
            manifest["files"][rel] = _file_hash(p)

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest_path


def test_bundle(bundle_dir: Path) -> List[str]:
    """Step 6 tests: integrity verification."""
    errors = []

    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append("FAIL: manifest.json not found")
        return errors

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Check all files exist and hashes match
    for rel_path, expected_hash in manifest.get("files", {}).items():
        full_path = bundle_dir / rel_path
        if not full_path.exists():
            errors.append(f"FAIL: file missing: {rel_path}")
            continue
        actual_hash = _file_hash(full_path)
        if actual_hash != expected_hash:
            errors.append(f"FAIL: hash mismatch for {rel_path}")

    # Check essential files
    for required in ["poisson/trains.npz", "network/base.npz", "regimes/regimes.json"]:
        if required not in manifest.get("files", {}):
            errors.append(f"FAIL: required file missing from manifest: {required}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: PipelineConfig, steps: str = "0123456") -> Path:
    """Run the full pipeline, returning the bundle directory path."""
    bundle_dir = Path(cfg.bundle_dir) / f"bundle_seed_{cfg.seed}"

    print("═" * 65)
    print("  FROZEN RESERVOIR PIPELINE")
    print(f"  seed={cfg.seed}  N={cfg.N}  steps={steps}")
    print(f"  bundle: {bundle_dir}/")
    print("═" * 65)

    t0 = time.time()
    seeds = None
    trains = None
    net = None
    regimes = None
    val_results = None

    # ── Step 0: Validate parameters ──
    if "0" in steps:
        print("\n▶ STEP 0: Parameter validation")
        errs = validate_config(cfg)
        _report("Step 0", errs)

    # ── Step 1: Seed hierarchy ──
    if "1" in steps:
        print("\n▶ STEP 1: Seed hierarchy")
        seeds = build_seed_manifest(cfg.seed)
        errs = test_seed_hierarchy(cfg.seed)
        _report("Step 1", errs)
        print(f"  master={seeds.master}  graph={seeds.graph}  "
              f"weights={seeds.weights}  poisson={seeds.poisson}")

    if seeds is None:
        seeds = build_seed_manifest(cfg.seed)

    # ── Step 2: Poisson trains ──
    if "2" in steps:
        print(f"\n▶ STEP 2: Poisson trains (K={cfg.poisson_K}, N={cfg.N}, "
              f"dur={cfg.poisson_duration_ms}ms)")
        trains = generate_all_poisson(cfg, seeds)
        errs = test_poisson(trains, cfg, seeds)
        _report("Step 2", errs)

        poisson_dir = bundle_dir / "poisson"
        poisson_dir.mkdir(parents=True, exist_ok=True)
        save_poisson(trains, poisson_dir / "trains.npz", cfg)
        print(f"  Saved: {poisson_dir}/trains.npz")

    # ── Step 3: Frozen network ──
    if "3" in steps:
        print(f"\n▶ STEP 3: Frozen network (N={cfg.N}, frac_E={cfg.frac_E})")
        net = build_frozen_network(cfg, seeds)
        errs = test_network(net, cfg, seeds)
        _report("Step 3", errs)

        rho_base = spectral_radius(net, alpha=1.0)
        bal_base = ei_balance(net, alpha=1.0)
        total_syn = sum(len(net.edges[s]["pre"]) for s in SYN_TYPES)
        print(f"  N_E={net.N_E}  N_I={net.N_I}  synapses={total_syn}")
        print(f"  rho(α=1)={rho_base:.6f}  balance={bal_base:.3f}")

        net_dir = bundle_dir / "network"
        net_dir.mkdir(parents=True, exist_ok=True)
        save_network(net, net_dir / "base.npz")
        print(f"  Saved: {net_dir}/base.npz")

    # ── Step 4: Regime definitions ──
    if "4" in steps:
        if net is None:
            net = load_network(bundle_dir / "network" / "base.npz")

        print(f"\n▶ STEP 4: 5 regime configurations (ρ targets: {cfg.rho_targets})")
        regimes = compute_regimes(net, cfg)
        errs = test_regimes(regimes, cfg)
        _report("Step 4", errs)

        for r in regimes:
            d = r.block_diag
            diag_str = ""
            if d:
                diag_str = (f"  ρEE={d.rho_EE:.3f}  "
                            f"‖EE‖={d.norm_EE:.1f} ‖EI‖={d.norm_EI:.1f} "
                            f"‖IE‖={d.norm_IE:.1f} ‖II‖={d.norm_II:.1f}  "
                            f"I/E={d.ei_incoming_ratio:.2f}")
            print(f"  {r.name:25s}  α={r.alpha:.6f}  ρ={r.rho_actual:.4f}  "
                  f"bal={r.balance:.3f}")
            if diag_str:
                print(f"    {diag_str}")

        reg_dir = bundle_dir / "regimes"
        reg_dir.mkdir(parents=True, exist_ok=True)
        save_regimes(regimes, reg_dir / "regimes.json")
        print(f"  Saved: {reg_dir}/regimes.json")

    # ── Step 5: Per-regime liveness validation ──
    if "5" in steps:
        if trains is None:
            trains, _ = load_poisson(bundle_dir / "poisson" / "trains.npz")
        if net is None:
            net = load_network(bundle_dir / "network" / "base.npz")
        if regimes is None:
            regimes = load_regimes(bundle_dir / "regimes" / "regimes.json")

        print(f"\n▶ STEP 5: Per-regime liveness validation "
              f"(warmup={cfg.warmup_ms}ms, sim={cfg.validation_ms}ms)")
        val_results = []
        for regime in regimes:
            vr = validate_regime_liveness(regime, net, trains[0], cfg, seeds)
            val_results.append(vr)
            print(f"  {vr.regime:25s}  rateE={vr.rate_E:5.1f}  rateI={vr.rate_I:5.1f}  "
                  f"syncE={vr.sync_E:5.2f}  cvE={vr.cv_isi_E:.2f}  "
                  f"fanoE={vr.fano_E:.2f}  [{vr.status}]")

        errs = test_validation(val_results)
        _report("Step 5", errs)

        val_dir = bundle_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        val_path = val_dir / "summary.json"
        val_path.write_text(
            json.dumps([asdict(v) for v in val_results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Saved: {val_path}")

    # ── Step 6: Bundle manifest ──
    if "6" in steps:
        if regimes is None:
            regimes = load_regimes(bundle_dir / "regimes" / "regimes.json")

        print(f"\n▶ STEP 6: Bundle manifest & integrity")
        manifest_path = save_bundle_manifest(bundle_dir, cfg, seeds, regimes, val_results)
        errs = test_bundle(bundle_dir)
        _report("Step 6", errs)
        print(f"  Saved: {manifest_path}")

    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    print(f"\n{'═'*65}")
    print(f"  PIPELINE COMPLETE  time={mins}m{secs}s")
    print(f"  Bundle: {bundle_dir}/")
    print(f"{'═'*65}")

    return bundle_dir


def _report(step_name: str, errors: List[str]) -> None:
    """Print test results for a step."""
    fails = [e for e in errors if e.startswith("FAIL")]
    warns = [e for e in errors if e.startswith("WARN")]

    if fails:
        print(f"  ✗ {step_name}: {len(fails)} FAIL, {len(warns)} WARN")
        for e in fails:
            print(f"    {e}")
        for e in warns:
            print(f"    {e}")
        raise SystemExit(1)
    elif warns:
        print(f"  ~ {step_name}: PASS with {len(warns)} warnings")
        for e in warns:
            print(f"    {e}")
    else:
        print(f"  ✓ {step_name}: PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen Reservoir Pipeline")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--step", default="0123456",
                    help="Which steps to run (default: all)")
    ap.add_argument("--N", type=int, default=135)
    ap.add_argument("--frac-E", type=float, default=0.8)
    ap.add_argument("--K", type=int, default=7,
                    help="Number of Poisson realizations")
    ap.add_argument("--duration-ms", type=float, default=6000.0,
                    help="Poisson train duration")
    ap.add_argument("--dt-train-ms", type=float, default=0.025)
    ap.add_argument("--dt-sim-ms", type=float, default=0.025)
    ap.add_argument("--rho-targets", type=float, nargs=5,
                    default=[0.6, 0.8, 1.0, 1.2, 1.5],
                    help="5 spectral radius targets (logspace-like)")
    ap.add_argument("--bundle-dir", default="bundles")

    args = ap.parse_args()

    cfg = PipelineConfig(
        seed=args.seed,
        N=args.N,
        frac_E=args.frac_E,
        poisson_K=args.K,
        poisson_duration_ms=args.duration_ms,
        dt_train_ms=args.dt_train_ms,
        dt_sim_ms=args.dt_sim_ms,
        rho_targets=args.rho_targets,
        bundle_dir=args.bundle_dir,
    )

    run_pipeline(cfg, steps=args.step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
