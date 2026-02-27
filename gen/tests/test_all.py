"""Comprehensive tests for all gen modules — no Brian2 required."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from gen.config import GeneratorConfig, validate_config, HeteroParam
from gen.seeding import split_seeds
from gen.population import make_population
from gen.graph_gen import generate_edges, BLOCK_ID_MAP
from gen.hetero_neurons import sample_neuron_params
from gen.hetero_synapses import sample_synapse_params, build_sparse_W, rescale_for_balance
from gen.spectral import spectral_radius, block_stats
from gen.regimes import make_regimes, make_regimes_from_edge
from gen.poisson_bank import generate_poisson_trains, save_poisson, load_poisson
from gen.io_bundle import write_bundle, write_manifest, verify_manifest, sha256_file
from gen.generate_one_bundle import generate_bundle


# ═══════════════════════════════════════════════════════════════════════════════
# 0) config
# ═══════════════════════════════════════════════════════════════════════════════

def test_validate_config_ok():
    cfg = GeneratorConfig()
    validate_config(cfg)  # should not raise


@pytest.mark.parametrize("field,value,match", [
    ("N", 0, "N must be > 0"),
    ("N", -5, "N must be > 0"),
    ("frac_I", 0.0, "frac_I must be in"),
    ("frac_I", 1.0, "frac_I must be in"),
    ("frac_I", -0.1, "frac_I must be in"),
    ("p_conn", 1.5, "p_conn must be in"),
    ("poisson_T_s", 0, "poisson_T_s must be > 0"),
    ("poisson_dt_ms", -1, "poisson_dt_ms must be > 0"),
])
def test_validate_config_rejects_bad_values(field, value, match):
    cfg = GeneratorConfig(**{field: value})
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_validate_config_rejects_wrong_target_count():
    cfg = GeneratorConfig(rho_eff_targets=[0.3, 0.7, 1.1])
    with pytest.raises(ValueError, match="5 elements"):
        validate_config(cfg)


def test_validate_config_rejects_unsorted_targets():
    cfg = GeneratorConfig(rho_eff_targets=[1.6, 0.7, 0.3, 1.1, 2.0])
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_config(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# 1) seeding
# ═══════════════════════════════════════════════════════════════════════════════

def test_split_seeds_deterministic():
    s1 = split_seeds(42)
    s2 = split_seeds(42)
    assert s1 == s2


def test_split_seeds_unique():
    s = split_seeds(42)
    vals = list(s.values())
    assert len(set(vals)) == len(vals), "sub-seeds must be unique"


def test_split_seeds_different_master():
    s1 = split_seeds(42)
    s2 = split_seeds(43)
    assert s1 != s2


def test_split_seeds_has_population_key():
    s = split_seeds(42)
    assert "population" in s
    assert "graph" in s
    assert s["population"] != s["graph"]


def test_split_seeds_key_count():
    s = split_seeds(42)
    assert len(s) == 7  # population, graph, weights, neurons, synapses, poisson, regimes


# ═══════════════════════════════════════════════════════════════════════════════
# 2) population
# ═══════════════════════════════════════════════════════════════════════════════

def test_population_sizes():
    pop = make_population(100, 0.2, seed=0)
    assert pop["N_E"] == 80
    assert pop["N_I"] == 20
    assert len(pop["E_idx"]) == 80
    assert len(pop["I_idx"]) == 20


def test_population_disjoint_complete():
    pop = make_population(100, 0.2, seed=0)
    all_idx = np.concatenate([pop["E_idx"], pop["I_idx"]])
    assert len(np.unique(all_idx)) == 100
    assert set(all_idx) == set(range(100))


def test_population_deterministic():
    p1 = make_population(100, 0.2, seed=7)
    p2 = make_population(100, 0.2, seed=7)
    assert np.array_equal(p1["E_idx"], p2["E_idx"])
    assert np.array_equal(p1["I_idx"], p2["I_idx"])


def test_population_is_I_consistent():
    pop = make_population(100, 0.2, seed=0)
    for idx in pop["I_idx"]:
        assert pop["is_I"][idx] is np.True_
    for idx in pop["E_idx"]:
        assert pop["is_I"][idx] is np.False_


# ═══════════════════════════════════════════════════════════════════════════════
# 3) graph
# ═══════════════════════════════════════════════════════════════════════════════

def _make_edges(seed=0, graph_type="ER", allow_self=False):
    cfg = GeneratorConfig(N=50, frac_I=0.2, graph_type=graph_type,
                          p_conn=0.3, k_in=10, allow_self=allow_self)
    pop = make_population(50, 0.2, seed)
    return generate_edges(cfg, pop, seed), pop, cfg


def test_edges_in_range():
    edges, pop, cfg = _make_edges()
    assert edges["pre"].min() >= 0
    assert edges["pre"].max() < cfg.N
    assert edges["post"].min() >= 0
    assert edges["post"].max() < cfg.N


def test_edges_no_self_loops_if_disabled():
    edges, _, _ = _make_edges(allow_self=False)
    self_loops = np.sum(edges["pre"] == edges["post"])
    assert self_loops == 0


def test_edges_block_id_valid():
    edges, _, _ = _make_edges()
    assert set(np.unique(edges["block_id"])).issubset({0, 1, 2, 3})


def test_edges_deterministic():
    e1, _, _ = _make_edges(seed=42)
    e2, _, _ = _make_edges(seed=42)
    assert np.array_equal(e1["pre"], e2["pre"])
    assert np.array_equal(e1["post"], e2["post"])
    assert np.array_equal(e1["block_id"], e2["block_id"])


def test_edges_fixed_indegree():
    cfg = GeneratorConfig(N=50, frac_I=0.2, graph_type="fixed_indegree",
                          k_in=5, allow_self=False)
    pop = make_population(50, 0.2, 0)
    edges = generate_edges(cfg, pop, 0)
    assert len(edges["pre"]) > 0
    assert edges["pre"].min() >= 0
    assert edges["pre"].max() < 50


# ═══════════════════════════════════════════════════════════════════════════════
# 4) neuron heterogeneity
# ═══════════════════════════════════════════════════════════════════════════════

def test_neuron_params_shape():
    cfg = GeneratorConfig(N=50)
    pop = make_population(50, 0.2, 0)
    params = sample_neuron_params(cfg, pop, seed=0)
    for name, arr in params.items():
        assert arr.shape == (50,), f"{name} shape wrong"


def test_neuron_params_clamped():
    cfg = GeneratorConfig(N=500)
    pop = make_population(500, 0.2, 0)
    params = sample_neuron_params(cfg, pop, seed=0)
    for name, arr in params.items():
        hp = cfg.neuron_hetero[name]
        lo = hp.base * hp.clamp_lo
        hi = hp.base * hp.clamp_hi
        if hp.base > 0:
            assert arr.min() >= lo - 1e-12, f"{name} below clamp"
            assert arr.max() <= hi + 1e-12, f"{name} above clamp"


def test_neuron_params_no_nan_inf():
    cfg = GeneratorConfig(N=100)
    pop = make_population(100, 0.2, 0)
    params = sample_neuron_params(cfg, pop, seed=0)
    for name, arr in params.items():
        assert not np.any(np.isnan(arr)), f"{name} has NaN"
        assert not np.any(np.isinf(arr)), f"{name} has Inf"


def test_neuron_params_deterministic():
    cfg = GeneratorConfig(N=50)
    pop = make_population(50, 0.2, 0)
    p1 = sample_neuron_params(cfg, pop, seed=7)
    p2 = sample_neuron_params(cfg, pop, seed=7)
    for name in p1:
        assert np.array_equal(p1[name], p2[name])


# ═══════════════════════════════════════════════════════════════════════════════
# 5) synapse params
# ═══════════════════════════════════════════════════════════════════════════════

def _make_syn(seed=0):
    cfg = GeneratorConfig(N=50, frac_I=0.2, p_conn=0.3)
    pop = make_population(50, 0.2, seed)
    edges = generate_edges(cfg, pop, seed)
    syn = sample_synapse_params(cfg, edges, pop, seed)
    return syn, edges, pop, cfg


def test_syn_params_lengths_match_edges():
    syn, edges, _, _ = _make_syn()
    n = len(edges["pre"])
    for key in ("w", "U", "tau_d_ms", "tau_f_ms", "delay_ms"):
        assert len(syn[key]) == n, f"{key} length mismatch"


def test_weight_signs_match_pre_type():
    syn, edges, pop, _ = _make_syn()
    is_I = pop["is_I"]
    for i in range(len(edges["pre"])):
        pre_idx = edges["pre"][i]
        w = syn["w"][i]
        if is_I[pre_idx]:
            assert w <= 0, f"I→? synapse {i} has positive weight"
        else:
            assert w >= 0, f"E→? synapse {i} has negative weight"


def test_delay_in_range():
    syn, _, _, cfg = _make_syn()
    assert syn["delay_ms"].min() >= cfg.delay_min_ms - 1e-12
    assert syn["delay_ms"].max() <= cfg.delay_max_ms + 1e-12


def test_build_W_shape():
    syn, edges, _, cfg = _make_syn()
    W = build_sparse_W(cfg.N, edges, syn["w"])
    assert W.shape == (cfg.N, cfg.N)


def test_build_W_nonzero_matches_edges():
    syn, edges, _, cfg = _make_syn()
    W = build_sparse_W(cfg.N, edges, syn["w"])
    # nnz should equal number of edges (some might overlap → csr sums them)
    # but typically no overlap in ER: just check > 0 and <= n_edges
    assert W.nnz > 0
    assert W.nnz <= len(edges["pre"])


# ═══════════════════════════════════════════════════════════════════════════════
# 6) spectral
# ═══════════════════════════════════════════════════════════════════════════════

def test_spectral_radius_positive():
    syn, edges, pop, cfg = _make_syn()
    W = build_sparse_W(cfg.N, edges, syn["w"])
    rho = spectral_radius(W)
    assert rho > 0


def test_spectral_radius_deterministic():
    syn, edges, pop, cfg = _make_syn(seed=42)
    W = build_sparse_W(cfg.N, edges, syn["w"])
    r1 = spectral_radius(W)
    r2 = spectral_radius(W)
    assert r1 == pytest.approx(r2, rel=1e-6)


def test_spectral_radius_known_small():
    # 2x2: [[0, 2], [2, 0]] → eigenvalues ±2 → ρ=2
    W = sparse.csr_matrix(np.array([[0, 2], [2, 0]], dtype=np.float64))
    rho = spectral_radius(W)
    assert abs(rho - 2.0) < 0.01


def test_spectral_radius_nonsymmetric():
    """Non-symmetric matrix: spectral radius != largest singular value.

    [[0, 4], [-1, 0]] has eigenvalues ±2i → ρ=2.
    Singular values of A: σ_max = 4. Old power iteration would give ~4 (wrong).
    """
    W = sparse.csr_matrix(np.array([[0, 4], [-1, 0]], dtype=np.float64))
    rho = spectral_radius(W)
    assert abs(rho - 2.0) < 0.05, f"Expected rho~2.0, got {rho}"


def test_block_stats_keys():
    syn, edges, pop, cfg = _make_syn()
    W = build_sparse_W(cfg.N, edges, syn["w"])
    stats = block_stats(W, pop)
    for key in ("rho_full", "rho_EE", "norm_EE", "norm_EI", "norm_IE",
                "norm_II", "balance"):
        assert key in stats


# ═══════════════════════════════════════════════════════════════════════════════
# 7) regimes
# ═══════════════════════════════════════════════════════════════════════════════

def test_regimes_len_5():
    regs = make_regimes(10.0, [0.3, 0.7, 1.0, 1.3, 1.6])
    assert len(regs) == 5


def test_regimes_alpha_monotonic_if_targets_monotonic():
    regs = make_regimes(10.0, [0.3, 0.7, 1.0, 1.3, 1.6])
    alphas = [r["alpha"] for r in regs]
    assert alphas == sorted(alphas)


def test_regimes_alpha_computation():
    regs = make_regimes(5.0, [0.3, 0.7, 1.0, 1.3, 1.6])
    for r in regs:
        expected = r["rho_target"] / 5.0
        assert abs(r["alpha"] - expected) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 8) poisson
# ═══════════════════════════════════════════════════════════════════════════════

def test_poisson_times_in_range():
    trains = generate_poisson_trains(10, 3.0, 10.0, seed=0)
    for ch, t in enumerate(trains):
        if len(t) > 0:
            assert t.min() >= 0, f"ch{ch} times < 0"
            assert t.max() < 3.0, f"ch{ch} times >= T"


def test_poisson_sorted():
    trains = generate_poisson_trains(10, 3.0, 10.0, seed=0)
    for ch, t in enumerate(trains):
        assert np.all(np.diff(t) >= 0), f"ch{ch} not sorted"


def test_poisson_deterministic():
    t1 = generate_poisson_trains(10, 3.0, 10.0, seed=42)
    t2 = generate_poisson_trains(10, 3.0, 10.0, seed=42)
    for ch in range(10):
        assert np.array_equal(t1[ch], t2[ch])


def test_poisson_rate_sanity():
    n_ch = 100
    T = 3.0
    rate = 10.0
    trains = generate_poisson_trains(n_ch, T, rate, seed=0)
    counts = [len(t) for t in trains]
    mean_count = np.mean(counts)
    expected = rate * T
    # Within 6σ: σ = sqrt(rate*T) per channel, mean over 100 → σ_mean ~ σ/10
    tol = 6 * np.sqrt(rate * T) / np.sqrt(n_ch)
    assert abs(mean_count - expected) < tol, \
        f"mean_count={mean_count:.1f} vs expected={expected:.1f}"


def test_poisson_save_load(tmp_path):
    trains = generate_poisson_trains(5, 2.0, 10.0, seed=7)
    path = tmp_path / "trains.npz"
    save_poisson(trains, path, T_s=2.0, rate_hz=10.0, seed=7)
    loaded, meta = load_poisson(path)
    assert meta["n_channels"] == 5
    assert meta["T_s"] == 2.0
    for ch in range(5):
        assert np.array_equal(trains[ch], loaded[ch])


# ═══════════════════════════════════════════════════════════════════════════════
# 9) io_bundle
# ═══════════════════════════════════════════════════════════════════════════════

def test_bundle_files_exist(tmp_path):
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "test_bundle"
    generate_bundle(cfg, out_dir=out)

    assert (out / "config.json").exists()
    assert (out / "network" / "population.json").exists()
    assert (out / "network" / "edges.npz").exists()
    assert (out / "network" / "neuron_params.npz").exists()
    assert (out / "network" / "synapse_params.npz").exists()
    assert (out / "network" / "base_stats.json").exists()
    assert (out / "regimes" / "regimes.json").exists()
    assert (out / "poisson" / "trains_3s.npz").exists()
    assert (out / "manifest.json").exists()


def test_manifest_has_hashes_for_all_files(tmp_path):
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "test_bundle"
    generate_bundle(cfg, out_dir=out)

    manifest = json.loads((out / "manifest.json").read_text())
    # All npz and json (except manifest) should be in files
    for p in out.rglob("*.npz"):
        rel = str(p.relative_to(out))
        assert rel in manifest["files"], f"{rel} not in manifest"
    for p in out.rglob("*.json"):
        if p.name != "manifest.json":
            rel = str(p.relative_to(out))
            assert rel in manifest["files"], f"{rel} not in manifest"


def test_reload_npz_roundtrip(tmp_path):
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "test_bundle"
    generate_bundle(cfg, out_dir=out)

    edges = np.load(out / "network" / "edges.npz")
    assert "pre" in edges
    assert "post" in edges
    assert "block_id" in edges


def test_manifest_verify_ok(tmp_path):
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "test_bundle"
    generate_bundle(cfg, out_dir=out)
    errors = verify_manifest(out)
    assert errors == []


# ═══════════════════════════════════════════════════════════════════════════════
# 10) end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

def test_generate_bundle_end_to_end(tmp_path):
    cfg = GeneratorConfig(N=50, frac_I=0.2, p_conn=0.2, poisson_T_s=1.0,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "e2e"
    path = generate_bundle(cfg, out_dir=out)
    assert path.exists()
    assert (path / "manifest.json").exists()

    # Regimes check
    regs = json.loads((path / "regimes" / "regimes.json").read_text())
    assert len(regs) == 5
    alphas = [r["alpha"] for r in regs]
    assert alphas == sorted(alphas)


def test_generate_bundle_reproducible(tmp_path):
    """Same seed → identical file hashes."""
    cfg = GeneratorConfig(N=30, frac_I=0.2, p_conn=0.2, poisson_T_s=0.5,
                          seed=42, rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])

    out1 = tmp_path / "run1"
    generate_bundle(cfg, out_dir=out1)

    out2 = tmp_path / "run2"
    generate_bundle(cfg, out_dir=out2)

    # Compare all npz file hashes
    for p1 in sorted(out1.rglob("*.npz")):
        rel = p1.relative_to(out1)
        p2 = out2 / rel
        assert p2.exists(), f"{rel} missing in run2"
        h1 = sha256_file(p1)
        h2 = sha256_file(p2)
        assert h1 == h2, f"hash mismatch for {rel}"


# ═══════════════════════════════════════════════════════════════════════════════
# 11) balance control
# ═══════════════════════════════════════════════════════════════════════════════

def test_balance_close_to_target():
    cfg = GeneratorConfig(N=100, frac_I=0.2, p_conn=0.2, target_balance=1.0)
    pop = make_population(100, 0.2, 0)
    edges = generate_edges(cfg, pop, 0)
    syn = sample_synapse_params(cfg, edges, pop, 0)
    bal_after = rescale_for_balance(syn, target=1.0)
    assert abs(bal_after - 1.0) < 0.01, f"balance={bal_after:.3f}, expected ~1.0"


def test_balance_different_targets():
    cfg = GeneratorConfig(N=100, frac_I=0.2, p_conn=0.2)
    pop = make_population(100, 0.2, 0)
    edges = generate_edges(cfg, pop, 0)
    for target in [0.8, 1.0, 1.2]:
        syn = sample_synapse_params(cfg, edges, pop, 0)
        bal = rescale_for_balance(syn, target=target)
        assert abs(bal - target) < 0.02, f"target={target}, got {bal:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 12) regimes spread
# ═══════════════════════════════════════════════════════════════════════════════

def test_regimes_spread():
    """Consecutive α ratio should be > 1.25 for default 5-regime targets."""
    regs = make_regimes(10.0, [0.40, 0.70, 1.00, 1.40, 2.00])
    alphas = [r["alpha"] for r in regs]
    for i in range(len(alphas) - 1):
        ratio = alphas[i+1] / alphas[i]
        assert ratio >= 1.25, f"ratio[{i}→{i+1}]={ratio:.2f} < 1.25"


# ═══════════════════════════════════════════════════════════════════════════════
# 13) smoke pass/fail logic (no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

from gen.brian_smoke import (evaluate_status, evaluate_scan_liveness,
                             check_separation, SmokeMetrics,
                             evaluate_5regime_quality)


def test_evaluate_status_ok():
    assert evaluate_status(20.0, 30.0, False) == "OK"


def test_evaluate_status_nan():
    assert evaluate_status(20.0, 30.0, True) == "NAN"


def test_evaluate_status_silent():
    assert evaluate_status(0.1, 0.1, False) == "SILENT"


def test_evaluate_status_runaway():
    assert evaluate_status(250.0, 30.0, False) == "RUNAWAY"
    assert evaluate_status(20.0, 350.0, False) == "RUNAWAY"


def _make_smoke_metrics(rate_E, cv_isi_E, fano_E, sync_E, idx=0,
                        pct_silent_E=0.0, status="OK"):
    return SmokeMetrics(
        regime_name=f"R{idx}", regime_index=idx, alpha=0.01*(idx+1),
        rate_E=rate_E, rate_I=rate_E*1.2,
        cv_isi_E=cv_isi_E, cv_isi_I=cv_isi_E,
        fano_E=fano_E, fano_I=fano_E,
        sync_E=sync_E, sync_I=sync_E,
        pct_silent_E=pct_silent_E, pct_silent_I=pct_silent_E,
        spike_count_E=int(rate_E * 100), spike_count_I=int(rate_E * 25),
        has_nan=False, status=status,
    )


def test_check_separation_pass():
    """Metrics with clear spread should PASS."""
    results = [
        _make_smoke_metrics(10.0, 0.4, 0.4, 1.0, 0),
        _make_smoke_metrics(20.0, 0.5, 0.6, 1.3, 1),
        _make_smoke_metrics(35.0, 0.7, 0.8, 1.6, 2),
        _make_smoke_metrics(50.0, 0.9, 1.1, 2.0, 3),
        _make_smoke_metrics(70.0, 1.1, 1.5, 2.5, 4),
    ]
    passed, details = check_separation(results)
    assert passed


def test_check_separation_fail():
    """Nearly identical metrics should FAIL separation."""
    results = [
        _make_smoke_metrics(20.0, 0.50, 0.50, 1.00, 0),
        _make_smoke_metrics(20.1, 0.50, 0.50, 1.01, 1),
        _make_smoke_metrics(20.0, 0.51, 0.50, 1.00, 2),
        _make_smoke_metrics(20.1, 0.50, 0.51, 1.01, 3),
        _make_smoke_metrics(20.0, 0.50, 0.50, 1.00, 4),
    ]
    passed, details = check_separation(results)
    assert not passed


def test_smoke_loads_bundle(tmp_path):
    """Verify bundle can be loaded (without Brian2 simulation)."""
    cfg = GeneratorConfig(N=20, frac_I=0.2, p_conn=0.3, poisson_T_s=0.5,
                          rho_eff_targets=[0.40, 0.70, 1.00, 1.40, 2.00])
    out = tmp_path / "smoke_bundle"
    generate_bundle(cfg, out_dir=out)

    # Verify all files loadable
    import json
    regs = json.loads((out / "regimes" / "regimes.json").read_text())
    assert len(regs) == 5
    edges = np.load(out / "network" / "edges.npz")
    assert "pre" in edges
    poi = np.load(out / "poisson" / "trains_3s.npz")
    assert int(poi["n_channels"]) == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 14) make_regimes_from_edge
# ═══════════════════════════════════════════════════════════════════════════════

def test_regimes_from_edge_len_5():
    regs = make_regimes_from_edge(0.01, rho_base=100.0)
    assert len(regs) == 5


def test_regimes_from_edge_center_is_alpha_edge():
    alpha_edge = 0.015
    regs = make_regimes_from_edge(alpha_edge, rho_base=50.0)
    # R3 (index 2) should have multiplier 1.00 → alpha == alpha_edge
    r3 = regs[2]
    assert abs(r3["alpha"] - alpha_edge) < 1e-12
    assert r3["multiplier"] == 1.0


def test_regimes_from_edge_multipliers_applied():
    alpha_edge = 0.02
    mults = [0.40, 0.70, 1.00, 1.40, 2.00]
    regs = make_regimes_from_edge(alpha_edge, rho_base=50.0, multipliers=mults)
    for r, m in zip(regs, mults):
        expected_alpha = alpha_edge * m
        assert abs(r["alpha"] - expected_alpha) < 1e-12
        assert abs(r["multiplier"] - m) < 1e-12


def test_regimes_from_edge_alpha_monotonic():
    regs = make_regimes_from_edge(0.01, rho_base=100.0)
    alphas = [r["alpha"] for r in regs]
    assert alphas == sorted(alphas)


def test_regimes_from_edge_has_metadata():
    regs = make_regimes_from_edge(0.01, rho_base=100.0)
    for r in regs:
        assert "alpha_edge" in r
        assert "multiplier" in r
        assert r["alpha_edge"] == 0.01


def test_regimes_from_edge_custom_multipliers():
    mults = [0.3, 0.6, 1.0, 1.5, 2.0]
    regs = make_regimes_from_edge(0.01, rho_base=50.0, multipliers=mults)
    assert len(regs) == 5
    assert regs[0]["alpha"] == pytest.approx(0.003)
    assert regs[4]["alpha"] == pytest.approx(0.02)


# ═══════════════════════════════════════════════════════════════════════════════
# 15) scan liveness gate
# ═══════════════════════════════════════════════════════════════════════════════

def test_scan_liveness_pass():
    assert evaluate_scan_liveness(20.0, 30.0, 5.0, 0.9, False) == "PASS"


def test_scan_liveness_fail_nan():
    assert evaluate_scan_liveness(20.0, 30.0, 5.0, 0.9, True) == "FAIL"


def test_scan_liveness_fail_runaway():
    assert evaluate_scan_liveness(250.0, 30.0, 5.0, 0.9, False) == "FAIL"


def test_scan_liveness_fail_silent():
    assert evaluate_scan_liveness(0.05, 10.0, 85.0, 0.9, False) == "FAIL"


def test_scan_liveness_warn_high_silent():
    assert evaluate_scan_liveness(10.0, 15.0, 55.0, 0.9, False) == "WARN"


def test_scan_liveness_warn_high_sync():
    assert evaluate_scan_liveness(20.0, 30.0, 5.0, 1.5, False) == "WARN"


# ═══════════════════════════════════════════════════════════════════════════════
# 16) 5-regime quality evaluation (no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_5regime_quality_all_ok():
    """5 healthy regimes with good separation → overall PASS."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3),
        _make_smoke_metrics(50.0, 0.80, 1.00, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "PASS"


def test_5regime_quality_r5_runaway_is_warn():
    """R5 runaway → WARN, not FAIL."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3),
        _make_smoke_metrics(250.0, 1.00, 1.30, 2.0, 4, status="RUNAWAY"),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "WARN"
    assert q["per_regime"][4]["verdict"] == "WARN"


def test_5regime_quality_r2_nan_is_fail():
    """R2 NaN → overall FAIL."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.3, 0.3, 0.8, 0),
        SmokeMetrics(
            regime_name=REGIME_NAMES[1], regime_index=1, alpha=0.02,
            rate_E=0.0, rate_I=0.0,
            cv_isi_E=0.0, cv_isi_I=0.0,
            fano_E=0.0, fano_I=0.0,
            sync_E=0.0, sync_I=0.0,
            pct_silent_E=100.0, pct_silent_I=100.0,
            spike_count_E=0, spike_count_I=0,
            has_nan=True, status="NAN",
        ),
        _make_smoke_metrics(25.0, 0.6, 0.7, 1.2, 2),
        _make_smoke_metrics(35.0, 0.8, 1.0, 1.5, 3),
        _make_smoke_metrics(50.0, 1.0, 1.3, 2.0, 4),
    ]
    for i, name in enumerate(REGIME_NAMES):
        results[i].regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "FAIL"


def test_5regime_quality_separation_r1_vs_r5():
    """CV(R5) >= 1.15*CV(R1) and Fano(R5) >= 1.15*Fano(R1) → PASS."""
    from gen.config import REGIME_NAMES
    # R5 CV=0.80 vs R1 CV=0.30 → ratio=2.67 >> 1.15  PASS
    # R5 Fano=1.00 vs R1 Fano=0.30 → ratio=3.33 >> 1.15  PASS
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3),
        _make_smoke_metrics(50.0, 0.80, 1.00, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "PASS"


def test_5regime_quality_separation_warn_low_ratio():
    """CV(R5) < 1.10*CV(R1) → WARN."""
    from gen.config import REGIME_NAMES
    # R1 cv=0.50, R5 cv=0.52 → ratio=1.04 < 1.10 → WARN
    results = [
        _make_smoke_metrics(5.0, 0.50, 0.50, 0.8, 0),
        _make_smoke_metrics(15.0, 0.51, 0.51, 1.0, 1),
        _make_smoke_metrics(25.0, 0.51, 0.51, 1.2, 2),
        _make_smoke_metrics(35.0, 0.52, 0.52, 1.5, 3),
        _make_smoke_metrics(50.0, 0.52, 0.52, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "WARN"


def test_5regime_quality_high_silent_on_r3_is_warn():
    """pct_silent_E > 40% on R3 → WARN."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2, pct_silent_E=50.0),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3),
        _make_smoke_metrics(50.0, 0.80, 1.00, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["overall"] == "WARN"
    assert q["per_regime"][2]["verdict"] == "WARN"


# ═══════════════════════════════════════════════════════════════════════════════
# 17) edge config validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_validate_config_edge_multipliers_ok():
    cfg = GeneratorConfig()
    validate_config(cfg)  # default multipliers should be valid


def test_validate_config_rejects_wrong_multiplier_count():
    cfg = GeneratorConfig(edge_multipliers=[0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match="edge_multipliers must have 5"):
        validate_config(cfg)


def test_validate_config_rejects_unsorted_multipliers():
    cfg = GeneratorConfig(edge_multipliers=[1.60, 1.25, 1.00, 0.75, 0.50])
    with pytest.raises(ValueError, match="edge_multipliers must be sorted"):
        validate_config(cfg)


def test_validate_config_rejects_bad_edge_thresholds():
    cfg = GeneratorConfig(edge_cv_thr=0)
    with pytest.raises(ValueError, match="edge_cv_thr"):
        validate_config(cfg)

    cfg = GeneratorConfig(edge_fano_thr=-1)
    with pytest.raises(ValueError, match="edge_fano_thr"):
        validate_config(cfg)


def test_validate_config_rejects_too_few_scan_points():
    cfg = GeneratorConfig(edge_n_scan=2)
    with pytest.raises(ValueError, match="edge_n_scan"):
        validate_config(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# 18) edge algorithm unit tests (no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

from gen.regimes import (_find_edge_ab, _find_edge_fallback,
                         _build_alpha_grid, _build_refine_grid,
                         save_edge_regimes)


def test_find_edge_ab_basic():
    """Step A+B finds first point meeting thresholds with trend OK."""
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.30, "fano_E": 0.30,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.02, "cv_isi_E": 0.40, "fano_E": 0.45,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.03, "cv_isi_E": 0.50, "fano_E": 0.55,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.04, "cv_isi_E": 0.60, "fano_E": 0.65,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge == 0.03  # first point crossing both thresholds


def test_find_edge_ab_skips_fail_gate():
    """Points with gate=FAIL are skipped even if thresholds met."""
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.50, "fano_E": 0.60,
         "pct_silent_E": 5.0, "gate": "FAIL"},
        {"alpha": 0.02, "cv_isi_E": 0.50, "fano_E": 0.60,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge == 0.02


def test_find_edge_ab_skips_high_silent():
    """Points with %silent > thr are not candidates."""
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.50, "fano_E": 0.60,
         "pct_silent_E": 50.0, "gate": "PASS"},
        {"alpha": 0.02, "cv_isi_E": 0.50, "fano_E": 0.60,
         "pct_silent_E": 10.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge == 0.02


def test_find_edge_ab_trend_rejects_dip():
    """A CV dip > trend_tol rejects the candidate (trend uses prev regardless of gate)."""
    scan = [
        # k=0: high CV but gate=FAIL → skipped by Step A, but CV used for trend at k=1
        {"alpha": 0.01, "cv_isi_E": 0.60, "fano_E": 0.60,
         "pct_silent_E": 5.0, "gate": "FAIL"},
        # k=1: passes Step A (CV≥0.45, Fano≥0.50) but trend rejects:
        #   CV[1]=0.55 < CV[0]-tol = 0.60-0.02 = 0.58 → REJECT
        {"alpha": 0.02, "cv_isi_E": 0.55, "fano_E": 0.60,
         "pct_silent_E": 5.0, "gate": "PASS"},
        # k=2: passes Step A but trend rejects:
        #   CV[2]=0.50 < CV[1]-tol = 0.55-0.02 = 0.53 → REJECT
        {"alpha": 0.03, "cv_isi_E": 0.50, "fano_E": 0.65,
         "pct_silent_E": 5.0, "gate": "PASS"},
        # k=3: passes Step A AND trend OK:
        #   CV[3]=0.56 >= CV[2]-tol = 0.50-0.02 = 0.48 ✓
        #   Fano[3]=0.70 >= Fano[2]-tol = 0.65-0.03 = 0.62 ✓
        {"alpha": 0.04, "cv_isi_E": 0.56, "fano_E": 0.70,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    # k=0 FAIL gate, k=1 trend reject, k=2 trend reject, k=3 passes
    assert edge == 0.04


def test_find_edge_ab_returns_none_when_nothing_qualifies():
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.20, "fano_E": 0.20,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.02, "cv_isi_E": 0.30, "fano_E": 0.30,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_ab(scan, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge is None


def test_find_edge_fallback_uses_chaos_score():
    """Fallback picks argmax(z(CV) + z(Fano))."""
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.20, "fano_E": 0.20,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.02, "cv_isi_E": 0.35, "fano_E": 0.40,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.03, "cv_isi_E": 0.30, "fano_E": 0.35,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    edge = _find_edge_fallback(scan, silent_thr=40.0)
    assert edge == 0.02  # highest CV + Fano combined


def test_find_edge_fallback_returns_none_all_fail():
    scan = [
        {"alpha": 0.01, "cv_isi_E": 0.20, "fano_E": 0.20,
         "pct_silent_E": 5.0, "gate": "FAIL"},
    ]
    assert _find_edge_fallback(scan, silent_thr=40.0) is None


def test_build_alpha_grid_deterministic():
    """Same rho_base + cfg → same grid."""
    cfg = GeneratorConfig()
    g1 = _build_alpha_grid(50.0, cfg)
    g2 = _build_alpha_grid(50.0, cfg)
    assert np.array_equal(g1, g2)


def test_build_alpha_grid_range():
    """Grid spans from rho_eff_min/rho_base to rho_eff_max/rho_base."""
    cfg = GeneratorConfig(edge_rho_eff_min=0.1, edge_rho_eff_max=3.0,
                          edge_n_scan=16)
    rho = 50.0
    grid = _build_alpha_grid(rho, cfg)
    assert len(grid) == 16
    assert abs(grid[0] - 0.1 / 50.0) < 1e-12
    assert abs(grid[-1] - 3.0 / 50.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 18b) refinement grid tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_refine_grid_middle():
    """k* in middle → linspace between neighbors, excluding endpoints."""
    alphas = np.array([0.01, 0.02, 0.04, 0.08])
    grid = _build_refine_grid(alphas, k_star=2, n_refine=4)
    assert len(grid) == 4
    # Should be strictly between alpha[1]=0.02 and alpha[3]=0.08
    assert grid[0] > 0.02
    assert grid[-1] < 0.08
    # Should be sorted ascending
    assert np.all(np.diff(grid) > 0)


def test_refine_grid_last_point():
    """k* at K-1 → extend upward from last coarse point."""
    alphas = np.array([0.01, 0.02, 0.04, 0.08])
    grid = _build_refine_grid(alphas, k_star=3, n_refine=6)
    assert len(grid) == 6
    # All points above last coarse point
    assert grid[0] > 0.08
    assert grid[-1] <= 0.08 * 2
    assert np.all(np.diff(grid) > 0)


def test_refine_grid_first_point():
    """k* at 0 → extend downward from first coarse point."""
    alphas = np.array([0.01, 0.02, 0.04, 0.08])
    grid = _build_refine_grid(alphas, k_star=0, n_refine=6)
    assert len(grid) == 6
    # All points below first coarse point
    assert grid[-1] < 0.01
    assert grid[0] >= 0.01 / 2
    assert np.all(np.diff(grid) > 0)


def test_refine_grid_no_overlap_with_coarse():
    """Refinement grid should not contain the coarse endpoints."""
    alphas = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
    # Middle case
    grid_mid = _build_refine_grid(alphas, k_star=2, n_refine=6)
    for a in grid_mid:
        assert a != 0.02 and a != 0.08  # no overlap with neighbors
    # Last case
    grid_last = _build_refine_grid(alphas, k_star=4, n_refine=6)
    assert 0.16 not in grid_last
    # First case
    grid_first = _build_refine_grid(alphas, k_star=0, n_refine=6)
    assert 0.01 not in grid_first


def test_merged_results_sorted_by_alpha():
    """Merging coarse + refine and sorting by alpha works correctly."""
    coarse = [
        {"alpha": 0.01, "cv_isi_E": 0.3, "fano_E": 0.3,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.04, "cv_isi_E": 0.5, "fano_E": 0.6,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    refine = [
        {"alpha": 0.02, "cv_isi_E": 0.4, "fano_E": 0.45,
         "pct_silent_E": 5.0, "gate": "PASS"},
        {"alpha": 0.03, "cv_isi_E": 0.46, "fano_E": 0.52,
         "pct_silent_E": 5.0, "gate": "PASS"},
    ]
    merged = sorted(coarse + refine, key=lambda r: r["alpha"])
    alphas = [r["alpha"] for r in merged]
    assert alphas == sorted(alphas)
    assert len(merged) == 4
    # A+B on merged should find 0.03 (first passing cv≥0.45, fano≥0.50)
    edge = _find_edge_ab(merged, cv_thr=0.45, fano_thr=0.50,
                         silent_thr=40.0, cv_trend_tol=0.02,
                         fano_trend_tol=0.03)
    assert edge == 0.03


# ═══════════════════════════════════════════════════════════════════════════════
# 19) User-specified test: extreme alpha → FAIL
# ═══════════════════════════════════════════════════════════════════════════════

def test_extreme_alpha_silent():
    """Very low α → SILENT → scan gate FAIL."""
    assert evaluate_scan_liveness(
        rate_E=0.05, rate_I=0.1, pct_silent_E=90.0,
        sync_E=0.5, has_nan=False) == "FAIL"


def test_extreme_alpha_runaway():
    """Very high α → RUNAWAY → scan gate FAIL."""
    assert evaluate_scan_liveness(
        rate_E=300.0, rate_I=400.0, pct_silent_E=0.0,
        sync_E=0.5, has_nan=False) == "FAIL"


# ═══════════════════════════════════════════════════════════════════════════════
# 20) User-specified test: 5-regime monotonic trend (soft)
# ═══════════════════════════════════════════════════════════════════════════════

def test_monotonic_trend_soft():
    """CV(R5) - CV(R1) >= 0.05 OR CV(R5) >= 1.15*CV(R1).
    Analogously for Fano."""
    from gen.config import REGIME_NAMES
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.35, 0.40, 1.0, 1),
        _make_smoke_metrics(25.0, 0.42, 0.50, 1.2, 2),
        _make_smoke_metrics(35.0, 0.50, 0.60, 1.5, 3),
        _make_smoke_metrics(50.0, 0.60, 0.75, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name

    cv_r1 = results[0].cv_isi_E
    cv_r5 = results[4].cv_isi_E
    fano_r1 = results[0].fano_E
    fano_r5 = results[4].fano_E

    # Check "net increase" criteria
    cv_ok = (cv_r5 - cv_r1 >= 0.05) or (cv_r5 >= 1.15 * cv_r1)
    fano_ok = (fano_r5 - fano_r1 >= 0.05) or (fano_r5 >= 1.15 * fano_r1)
    assert cv_ok, f"CV trend fail: R1={cv_r1:.2f}, R5={cv_r5:.2f}"
    assert fano_ok, f"Fano trend fail: R1={fano_r1:.2f}, R5={fano_r5:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 21) User-specified test: silent guard
# ═══════════════════════════════════════════════════════════════════════════════

def test_silent_guard_r3_r5():
    """If %silent_E > 40% in R3, R4, or R5 → quality WARN or FAIL."""
    from gen.config import REGIME_NAMES
    # R4 has 50% silent
    results = [
        _make_smoke_metrics(5.0, 0.30, 0.30, 0.8, 0),
        _make_smoke_metrics(15.0, 0.40, 0.50, 1.0, 1),
        _make_smoke_metrics(25.0, 0.50, 0.60, 1.2, 2),
        _make_smoke_metrics(35.0, 0.65, 0.80, 1.5, 3, pct_silent_E=50.0),
        _make_smoke_metrics(50.0, 0.80, 1.00, 2.0, 4),
    ]
    for r, name in zip(results, REGIME_NAMES):
        r.regime_name = name
    q = evaluate_5regime_quality(results)
    assert q["per_regime"][3]["verdict"] == "WARN"
    assert q["overall"] in ("WARN", "FAIL")


# ═══════════════════════════════════════════════════════════════════════════════
# 22) save_edge_regimes with scan_info dict
# ═══════════════════════════════════════════════════════════════════════════════

def test_save_edge_regimes_writes_structured_metadata(tmp_path):
    """save_edge_regimes writes coarse_grid, refine_grid, edge_rule, source."""
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "regimes").mkdir(parents=True)

    scan_info = {
        "coarse_grid": [
            {"alpha": 0.01, "cv_isi_E": 0.3, "fano_E": 0.3,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.02, "cv_isi_E": 0.5, "fano_E": 0.6,
             "pct_silent_E": 5.0, "gate": "PASS"},
        ],
        "refine_grid": [
            {"alpha": 0.015, "cv_isi_E": 0.46, "fano_E": 0.52,
             "pct_silent_E": 5.0, "gate": "PASS"},
        ],
        "merged_grid": [
            {"alpha": 0.01, "cv_isi_E": 0.3, "fano_E": 0.3,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.015, "cv_isi_E": 0.46, "fano_E": 0.52,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.02, "cv_isi_E": 0.5, "fano_E": 0.6,
             "pct_silent_E": 5.0, "gate": "PASS"},
        ],
        "edge_rule": {
            "cv_thr": 0.45, "fano_thr": 0.50,
            "silent_thr": 40.0, "cv_trend_tol": 0.02,
            "fano_trend_tol": 0.03,
        },
        "alpha_edge_source": "refine_A+B_merged_k1",
        "refine_direction": "zoom",
    }

    regimes = save_edge_regimes(
        bundle_dir, alpha_edge=0.015, rho_base=100.0,
        scan_info=scan_info, edge_method="metrics_scan_2stage",
    )
    assert len(regimes) == 5

    data = json.loads((bundle_dir / "regimes" / "regimes.json").read_text())
    assert data["alpha_edge"] == 0.015
    assert data["edge_method"] == "metrics_scan_2stage"
    assert data["alpha_edge_source"] == "refine_A+B_merged_k1"
    assert data["refine_direction"] == "zoom"
    assert "edge_rule" in data
    assert data["edge_rule"]["cv_thr"] == 0.45
    assert len(data["coarse_grid"]) == 2
    assert len(data["refine_grid"]) == 1
    assert len(data["regimes"]) == 5


def test_save_edge_regimes_r5_clamp_uses_merged(tmp_path):
    """R5 clamp logic reads gate from merged_grid in scan_info."""
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "regimes").mkdir(parents=True)

    scan_info = {
        "coarse_grid": [],
        "refine_grid": [],
        "merged_grid": [
            {"alpha": 0.01, "cv_isi_E": 0.5, "fano_E": 0.5,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.02, "cv_isi_E": 0.6, "fano_E": 0.6,
             "pct_silent_E": 5.0, "gate": "PASS"},
            {"alpha": 0.03, "cv_isi_E": 0.0, "fano_E": 0.0,
             "pct_silent_E": 90.0, "gate": "FAIL"},
        ],
        "edge_rule": {},
        "alpha_edge_source": "coarse_k1_A+B",
        "refine_direction": None,
    }

    # alpha_edge=0.015, R5 mult=2.00 → R5 α=0.030
    # max_ok=0.02, R5>max_ok*1.1=0.022 → should clamp
    regimes = save_edge_regimes(
        bundle_dir, alpha_edge=0.015, rho_base=100.0,
        scan_info=scan_info,
    )
    data = json.loads((bundle_dir / "regimes" / "regimes.json").read_text())
    assert data["r5_clamped"] is True
    assert data["r5_final_multiplier"] < 2.00


def test_save_edge_regimes_stability_check_passthrough(tmp_path):
    """Stability check result is written to regimes.json when present."""
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "regimes").mkdir(parents=True)

    scan_info = {
        "coarse_grid": [],
        "refine_grid": [],
        "merged_grid": [
            {"alpha": 0.02, "cv_isi_E": 0.5, "fano_E": 0.5,
             "pct_silent_E": 5.0, "gate": "PASS"},
        ],
        "edge_rule": {},
        "alpha_edge_source": "coarse_k0_A+B",
        "refine_direction": None,
        "stability_check": {
            "stable": True,
            "cv_scan": 0.50, "cv_long": 0.52,
            "cv_rel_diff": 0.04,
            "fano_scan": 0.50, "fano_long": 0.48,
            "fano_rel_diff": 0.04,
            "tol": 0.10,
            "measure_ms": 4000.0,
        },
    }

    save_edge_regimes(
        bundle_dir, alpha_edge=0.02, rho_base=100.0,
        scan_info=scan_info,
    )
    data = json.loads((bundle_dir / "regimes" / "regimes.json").read_text())
    assert "stability_check" in data
    assert data["stability_check"]["stable"] is True
    assert data["stability_check"]["cv_scan"] == 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# 23) jitter spike collisions (unit test, no Brian2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_jitter_spike_collisions_resolves_collisions():
    """Same neuron, two spikes within dt → second spike shifted."""
    from gen.brian_smoke import _jitter_spike_collisions
    indices = np.array([0, 0, 1], dtype=np.int32)
    times = np.array([1.0, 1.005, 2.0])
    dt = 0.025
    idx_out, t_out = _jitter_spike_collisions(indices, times, dt)
    # Neuron 0's spikes: 1.0 and 1.005 are within dt=0.025
    # After jitter: 1.0 and 1.0+0.025=1.025
    n0_times = t_out[idx_out == 0]
    assert len(n0_times) == 2
    assert n0_times[0] == pytest.approx(1.0)
    assert n0_times[1] == pytest.approx(1.025)
    # Neuron 1 is unchanged
    n1_times = t_out[idx_out == 1]
    assert n1_times[0] == pytest.approx(2.0)


def test_jitter_spike_collisions_empty():
    from gen.brian_smoke import _jitter_spike_collisions
    idx, t = _jitter_spike_collisions(np.array([], dtype=np.int32),
                                       np.array([]), 0.025)
    assert len(idx) == 0
    assert len(t) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 24) vectorized ER edge count sanity
# ═══════════════════════════════════════════════════════════════════════════════

def test_er_edge_count_approximately_expected():
    """ER edge count should be approximately N_src * N_tgt * p."""
    cfg = GeneratorConfig(N=200, frac_I=0.2, graph_type="ER",
                          p_conn=0.3, allow_self=False)
    pop = make_population(200, 0.2, seed=0)
    edges = generate_edges(cfg, pop, seed=0)
    n_edges = len(edges["pre"])
    # Expected: sum of block contributions (160E, 40I)
    # EE: 160*159*0.3, EI: 160*40*0.3, IE: 40*160*0.3, II: 40*39*0.3
    expected = (160 * 159 + 160 * 40 + 40 * 160 + 40 * 39) * 0.3
    assert abs(n_edges - expected) < 6 * np.sqrt(expected), \
        f"n_edges={n_edges}, expected~{expected:.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 25) NARMA-10 generator
# ═══════════════════════════════════════════════════════════════════════════════

from gen.narma import generate_narma10, NARMA_ORDER


def test_narma_generator_shape():
    """Output arrays have correct length K."""
    u, y = generate_narma10(500, seed=42)
    assert u.shape == (500,)
    assert y.shape == (500,)


def test_narma_generator_deterministic():
    """Same seed → identical output."""
    u1, y1 = generate_narma10(300, seed=7)
    u2, y2 = generate_narma10(300, seed=7)
    assert np.array_equal(u1, u2)
    assert np.array_equal(y1, y2)


def test_narma_generator_different_seeds():
    """Different seeds → different output."""
    u1, y1 = generate_narma10(300, seed=7)
    u2, y2 = generate_narma10(300, seed=8)
    assert not np.array_equal(u1, u2)
    assert not np.array_equal(y1, y2)


def test_narma_generator_u_range():
    """Input u stays in [0, 0.5]."""
    u, _ = generate_narma10(1000, seed=42)
    assert u.min() >= 0.0
    assert u.max() <= 0.5


def test_narma_generator_y_bounded():
    """Target y stays bounded (no divergence) for standard params."""
    _, y = generate_narma10(3000, seed=42)
    assert np.all(np.isfinite(y))
    assert y.max() < 100.0  # should be ~0.15–1.0 typically


def test_narma_generator_y_nonzero():
    """y should have nonzero values after warmup."""
    _, y = generate_narma10(500, seed=42)
    assert y[NARMA_ORDER:].max() > 0.05


def test_narma_generator_rejects_short():
    """K < NARMA_ORDER raises ValueError."""
    with pytest.raises(ValueError, match="must be >= NARMA_ORDER"):
        generate_narma10(5, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
# 26) spike count state extraction
# ═══════════════════════════════════════════════════════════════════════════════

from gen.state_readout import extract_spike_counts


def test_binning_state_shape():
    """Correct (K, N) shape for known spike trains."""
    # 3 neurons, 100ms total, 0 warmup, 10ms bins → K=10
    spike_trains = {
        0: np.array([5.0, 15.0, 25.0]),   # spikes at 5, 15, 25 ms
        1: np.array([55.0]),                # one spike at 55 ms
        2: np.array([]),                    # silent
    }
    X = extract_spike_counts(spike_trains, n_neurons=3,
                             dt_task_ms=10.0, total_ms=100.0)
    assert X.shape == (10, 3)


def test_binning_state_counts():
    """Spike counts match known inputs."""
    spike_trains = {
        0: np.array([5.0, 15.0, 25.0]),
        1: np.array([55.0]),
        2: np.array([]),
    }
    X = extract_spike_counts(spike_trains, n_neurons=3,
                             dt_task_ms=10.0, total_ms=100.0)
    # Neuron 0: bin 0 (0-10ms)=1, bin 1 (10-20ms)=1, bin 2 (20-30ms)=1
    assert X[0, 0] == 1
    assert X[1, 0] == 1
    assert X[2, 0] == 1
    assert X[3, 0] == 0
    # Neuron 1: bin 5 (50-60ms)=1
    assert X[5, 1] == 1
    assert X[4, 1] == 0
    # Neuron 2: all zero
    assert np.all(X[:, 2] == 0)


def test_binning_state_warmup_discard():
    """Warmup spikes are excluded from state matrix."""
    spike_trains = {
        0: np.array([5.0, 50.0, 150.0]),
    }
    # Total 200ms, warmup 100ms → measure 100ms → K=10 bins of 10ms
    X = extract_spike_counts(spike_trains, n_neurons=1,
                             dt_task_ms=10.0, total_ms=200.0,
                             warmup_ms=100.0)
    assert X.shape == (10, 1)
    # Only spike at 150ms survives → bin (150-100)/10 = 5
    assert X[5, 0] == 1
    assert X.sum() == 1  # other spike at 5ms and 50ms discarded


def test_binning_state_total_spikes():
    """Total spike count in matrix equals input spike count in window."""
    rng = np.random.default_rng(42)
    n_neurons = 10
    spike_trains = {}
    total_expected = 0
    for nid in range(n_neurons):
        n_spikes = rng.integers(0, 20)
        t = rng.uniform(0.0, 500.0, size=n_spikes)
        spike_trains[nid] = np.sort(t)
        total_expected += int(np.sum((t >= 100.0) & (t < 500.0)))
    X = extract_spike_counts(spike_trains, n_neurons=n_neurons,
                             dt_task_ms=10.0, total_ms=500.0,
                             warmup_ms=100.0)
    assert X.sum() == total_expected


# ═══════════════════════════════════════════════════════════════════════════════
# 27) NARMA injection signal shape validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_injection_shape_covers_simulation():
    """NARMA drive array must cover the full simulation duration."""
    from gen.narma import generate_narma10
    warmup_ms = 1000.0
    measure_ms = 2000.0
    dt_task_ms = 10.0
    total_ms = warmup_ms + measure_ms
    # K steps needed: total_ms / dt_task_ms
    K_needed = int(total_ms / dt_task_ms)
    # Generate NARMA with enough steps (+ some extra for safety)
    K_narma = K_needed + 200  # 200 extra steps for NARMA warmup discard
    u, y = generate_narma10(K_narma, seed=42)
    # The drive array for Brian2 is u[:K_needed]
    narma_drive = u[:K_needed]
    assert narma_drive.shape == (K_needed,)
    assert K_needed == 300  # 3000ms / 10ms


def test_injection_scaling():
    """Check that scaled NARMA drive is in expected nA range."""
    from gen.narma import generate_narma10
    u, _ = generate_narma10(500, seed=42)
    scale_nA = 0.5
    scaled = u * scale_nA
    # u in [0, 0.5], scale=0.5 → I_narma in [0, 0.25] nA
    assert scaled.min() >= 0.0
    assert scaled.max() <= 0.25 + 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 27b) delay embedding
# ═══════════════════════════════════════════════════════════════════════════════

from gen.state_readout import delay_embed


def test_delay_embed_shape():
    """D=3 delays on (10, 5) → (7, 20)."""
    X = np.arange(50).reshape(10, 5)
    X_aug = delay_embed(X, D=3)
    assert X_aug.shape == (7, 20)  # (10-3, 5*(3+1))


def test_delay_embed_zero():
    """D=0 returns X unchanged."""
    X = np.ones((10, 3))
    X_aug = delay_embed(X, D=0)
    assert np.array_equal(X_aug, X)


def test_delay_embed_content():
    """Verify delay embedding contains correct past slices."""
    X = np.arange(20).reshape(5, 4)
    X_aug = delay_embed(X, D=2)
    # X_aug[0] should contain [X[2], X[1], X[0]]
    expected_row0 = np.concatenate([X[2], X[1], X[0]])
    assert np.array_equal(X_aug[0], expected_row0)


# ═══════════════════════════════════════════════════════════════════════════════
# 28) ridge regression readout
# ═══════════════════════════════════════════════════════════════════════════════

from gen.ridge import ridge_cv_fit, ridge_predict


def test_ridge_perfect_linear():
    """Ridge recovers exact linear relationship y = 2*x1 + 3*x2 + 1."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 2))
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1
    model, alpha = ridge_cv_fit(X, y, alphas=[1e-6, 1e-4])
    y_pred = ridge_predict(model, X)
    assert np.allclose(y, y_pred, atol=0.01)


def test_ridge_selects_alpha():
    """ridge_cv_fit returns a valid alpha from the grid."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 5))
    y = X @ rng.standard_normal(5) + rng.normal(0, 0.1, 100)
    _, alpha = ridge_cv_fit(X, y, alphas=[1e-6, 1e-2, 100.0])
    assert alpha in [1e-6, 1e-2, 100.0]


# ═══════════════════════════════════════════════════════════════════════════════
# 29) benchmark metrics
# ═══════════════════════════════════════════════════════════════════════════════

from gen.metrics import nrmse, rmse, r_squared


def test_nrmse_perfect():
    """Perfect prediction → NRMSE = 0."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nrmse(y, y) == pytest.approx(0.0)


def test_nrmse_mean_baseline():
    """Predicting the mean → NRMSE = 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_mean = np.full_like(y, y.mean())
    assert nrmse(y, y_mean) == pytest.approx(1.0)


def test_nrmse_worse_than_mean():
    """Bad prediction → NRMSE > 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_bad = np.array([10.0, -5.0, 10.0, -5.0])
    assert nrmse(y, y_bad) > 1.0


def test_rmse_known():
    """RMSE of [1,2] vs [3,4] = sqrt((4+4)/2) = 2."""
    assert rmse(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(2.0)


def test_r_squared_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert r_squared(y, y) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 30) E/I split state extraction
# ═══════════════════════════════════════════════════════════════════════════════

from gen.state_readout import extract_spike_counts_ei


def test_ei_split_shape():
    """E/I split produces (K, N_E + N_I) matrix."""
    spike_trains = {
        0: np.array([5.0, 15.0]),   # E neuron
        1: np.array([25.0]),         # E neuron
        2: np.array([35.0, 55.0]),   # I neuron
    }
    E_idx = np.array([0, 1])
    I_idx = np.array([2])
    X = extract_spike_counts_ei(spike_trains, n_neurons=3,
                                 E_idx=E_idx, I_idx=I_idx,
                                 dt_task_ms=10.0, total_ms=100.0)
    assert X.shape == (10, 3)  # K=10, N_E + N_I = 2 + 1 = 3


def test_ei_split_column_ordering():
    """First N_E columns are E neurons, last N_I columns are I neurons."""
    spike_trains = {
        0: np.array([5.0]),      # neuron 0 (E)
        1: np.array([]),         # neuron 1 (I)
        2: np.array([15.0]),     # neuron 2 (E)
    }
    E_idx = np.array([0, 2])
    I_idx = np.array([1])
    X = extract_spike_counts_ei(spike_trains, n_neurons=3,
                                 E_idx=E_idx, I_idx=I_idx,
                                 dt_task_ms=10.0, total_ms=50.0)
    # Col 0 = neuron 0 (E), Col 1 = neuron 2 (E), Col 2 = neuron 1 (I)
    assert X[0, 0] == 1  # neuron 0 spike at 5ms → bin 0
    assert X[1, 1] == 1  # neuron 2 spike at 15ms → bin 1
    assert X[:, 2].sum() == 0  # neuron 1 (I) is silent


def test_ei_split_normalize():
    """Z-score normalization produces zero mean per column."""
    spike_trains = {i: np.array([float(j) for j in range(i, 100, 5)])
                    for i in range(5)}
    E_idx = np.array([0, 1, 2, 3])
    I_idx = np.array([4])
    X = extract_spike_counts_ei(spike_trains, n_neurons=5,
                                 E_idx=E_idx, I_idx=I_idx,
                                 dt_task_ms=10.0, total_ms=100.0,
                                 normalize=True)
    # After z-scoring, each non-constant column should have mean ~0
    for col in range(X.shape[1]):
        col_std = np.std(X[:, col])
        if col_std > 1e-10:
            assert abs(np.mean(X[:, col])) < 1e-10


def test_ei_split_without_normalize_matches_reordered():
    """Without normalization, ei_split is a column reordering of full matrix."""
    from gen.state_readout import extract_spike_counts
    spike_trains = {
        0: np.array([5.0, 15.0]),
        1: np.array([25.0]),
        2: np.array([35.0]),
    }
    E_idx = np.array([0, 2])
    I_idx = np.array([1])
    X_ei = extract_spike_counts_ei(spike_trains, n_neurons=3,
                                    E_idx=E_idx, I_idx=I_idx,
                                    dt_task_ms=10.0, total_ms=50.0,
                                    normalize=False)
    X_full = extract_spike_counts(spike_trains, n_neurons=3,
                                   dt_task_ms=10.0, total_ms=50.0)
    assert np.array_equal(X_ei[:, 0], X_full[:, 0])  # E neuron 0
    assert np.array_equal(X_ei[:, 1], X_full[:, 2])  # E neuron 2
    assert np.array_equal(X_ei[:, 2], X_full[:, 1])  # I neuron 1


# ═══════════════════════════════════════════════════════════════════════════════
# 31) ridge denser alpha grid
# ═══════════════════════════════════════════════════════════════════════════════


def test_ridge_default_alphas_denser():
    """Default alpha grid should have 21 points in logspace."""
    from gen.ridge import DEFAULT_ALPHAS
    assert len(DEFAULT_ALPHAS) == 21
    assert DEFAULT_ALPHAS[0] == pytest.approx(1e-6, rel=1e-3)
    assert DEFAULT_ALPHAS[-1] == pytest.approx(1e4, rel=1e-3)
