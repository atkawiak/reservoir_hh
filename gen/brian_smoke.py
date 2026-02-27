#!/usr/bin/env python3
"""Brian2 smoke test — liveness + separation metrics for a frozen bundle.

Usage:
    python -m gen.brian_smoke bundles/bundle_seed_123
    python -m gen.brian_smoke bundles/bundle_seed_123 --regime 0 --warmup-ms 1000 --measure-ms 2000
    python -m gen.brian_smoke bundles/bundle_seed_123 --regime all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .io_bundle import verify_manifest


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SmokeMetrics:
    """Metrics for one regime smoke test."""
    regime_name: str
    regime_index: int
    alpha: float
    rate_E: float
    rate_I: float
    cv_isi_E: float
    cv_isi_I: float
    fano_E: float
    fano_I: float
    sync_E: float
    sync_I: float
    pct_silent_E: float    # % of E neurons with 0 spikes in measure window
    pct_silent_I: float    # % of I neurons with 0 spikes
    spike_count_E: int     # total E spikes in measure window
    spike_count_I: int     # total I spikes in measure window
    has_nan: bool
    status: str        # OK, SILENT, RUNAWAY, NAN


def compute_rate(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                 total_ms: float, b) -> float:
    """Mean firing rate (Hz) over measurement window."""
    dur_s = (total_ms - warmup_ms) / 1000.0
    counts = []
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        counts.append(np.sum((t >= warmup_ms) & (t < total_ms)))
    return float(np.mean(counts) / dur_s) if counts else 0.0


def compute_cv_isi(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                   total_ms: float, b) -> float:
    """Median CV of ISI across neurons with >= 3 spikes in T_eval."""
    cvs = []
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        t = t[(t >= warmup_ms) & (t < total_ms)]
        if len(t) >= 3:
            isi = np.diff(t)
            m = isi.mean()
            if m > 1e-9:
                cvs.append(float(isi.std() / m))
    return float(np.median(cvs)) if cvs else 0.0


def compute_fano(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                 total_ms: float, b, bin_ms: float = 50.0) -> float:
    """Median Fano factor = var(counts)/mean(counts) per neuron in bins."""
    n_bins = max(1, int((total_ms - warmup_ms) / bin_ms))
    fanos = []
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        t = t[(t >= warmup_ms) & (t < total_ms)]
        hist, _ = np.histogram(t, bins=n_bins, range=(warmup_ms, total_ms))
        m = hist.mean()
        if m > 1e-9:
            fanos.append(float(hist.var() / m))
    return float(np.median(fanos)) if fanos else 0.0


def compute_sync(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                 total_ms: float, b, bin_ms: float = 5.0) -> float:
    """Population synchrony = var(pop_signal) / (N * mean(var_individual))."""
    n_bins = max(1, int((total_ms - warmup_ms) / bin_ms))
    pop_hist = np.zeros(n_bins, dtype=np.float64)
    ind_vars = []
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        t = t[(t >= warmup_ms) & (t < total_ms)]
        hist, _ = np.histogram(t, bins=n_bins, range=(warmup_ms, total_ms))
        pop_hist += hist
        ind_vars.append(float(np.var(hist)))
    pop_var = float(np.var(pop_hist))
    mean_iv = float(np.mean(ind_vars)) if ind_vars else 0.0
    return pop_var / (len(indices) * mean_iv + 1e-15)


def compute_pct_silent(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                       total_ms: float, b) -> float:
    """Percentage of neurons with 0 spikes in measurement window."""
    n_silent = 0
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        count = np.sum((t >= warmup_ms) & (t < total_ms))
        if count == 0:
            n_silent += 1
    return 100.0 * n_silent / len(indices) if len(indices) > 0 else 100.0


def compute_spike_count(spike_trains: dict, indices: np.ndarray, warmup_ms: float,
                        total_ms: float, b) -> int:
    """Total spike count across all neurons in measurement window."""
    total = 0
    for idx in indices:
        t = np.array(spike_trains[int(idx)] / b["ms"])
        total += int(np.sum((t >= warmup_ms) & (t < total_ms)))
    return total


def evaluate_status(rate_E: float, rate_I: float, has_nan: bool) -> str:
    """Determine liveness status from metrics."""
    if has_nan:
        return "NAN"
    if rate_E < 0.5 and rate_I < 0.5:
        return "SILENT"
    if rate_E > 200 or rate_I > 300:
        return "RUNAWAY"
    return "OK"


def evaluate_scan_liveness(rate_E: float, rate_I: float, pct_silent_E: float,
                           sync_E: float, has_nan: bool) -> str:
    """Liveness gate for edge scan points.

    Returns: 'FAIL', 'WARN', or 'PASS'.

    FAIL: NaN/Inf, rate_E > 200 (runaway), rate_E < 0.1 or %silent > 80 (silent).
    WARN: 40 < %silent_E <= 80, or syncE > 1.3.
    PASS: everything else.
    """
    if has_nan:
        return "FAIL"
    if rate_E > 200 or rate_I > 300:
        return "FAIL"
    if rate_E < 0.1 or pct_silent_E > 80:
        return "FAIL"
    if pct_silent_E > 40:
        return "WARN"
    if sync_E > 1.3:
        return "WARN"
    return "PASS"


def check_separation(results: List[SmokeMetrics], rel_thr: float = 0.15
                     ) -> Tuple[bool, List[dict]]:
    """Check if regimes are separated by at least one metric.

    Returns (any_pass, details_list).
    """
    ok = [r for r in results if r.status == "OK"]
    if len(ok) < 2:
        return True, []   # can't judge with < 2

    metric_names = [
        ("rate_E",       [r.rate_E       for r in ok]),
        ("rate_I",       [r.rate_I       for r in ok]),
        ("cv_isi_E",     [r.cv_isi_E     for r in ok]),
        ("cv_isi_I",     [r.cv_isi_I     for r in ok]),
        ("fano_E",       [r.fano_E       for r in ok]),
        ("fano_I",       [r.fano_I       for r in ok]),
        ("sync_E",       [r.sync_E       for r in ok]),
        ("sync_I",       [r.sync_I       for r in ok]),
        ("pct_silent_E", [r.pct_silent_E for r in ok]),
    ]

    details = []
    any_pass = False
    for name, vals in metric_names:
        mn, mx = min(vals), max(vals)
        med = float(np.median(vals))
        span = mx - mn
        thr = rel_thr * med if med > 1e-12 else rel_thr
        passed = span >= thr
        if passed:
            any_pass = True
        details.append({
            "metric": name, "min": mn, "max": mx, "median": med,
            "span": span, "threshold": thr, "pass": passed,
        })
    return any_pass, details


def evaluate_5regime_quality(results: List[SmokeMetrics],
                             silent_thr: float = 40.0) -> dict:
    """Evaluate quality of 5-regime set.

    Per-regime liveness:
        FAIL: NaN, RUNAWAY (any regime), SILENT on R1..R4
        WARN: R5 SILENT/RUNAWAY, R1 SILENT, %silent_E > silent_thr on R3..R5
        PASS: otherwise

    Separation criteria (R1 vs R5):
        FAIL: any liveness FAIL
        WARN: CV(R5) < 1.10*CV(R1) or Fano(R5) < 1.10*Fano(R1)
        PASS: CV(R5) >= 1.15*CV(R1) AND Fano(R5) >= 1.15*Fano(R1)
              AND %silent_E <= silent_thr for all regimes
    """
    regime_status = {}

    for r in results:
        idx = r.regime_index
        name = r.regime_name

        if r.status == "NAN":
            regime_status[idx] = {"name": name, "verdict": "FAIL",
                                  "reason": "NaN detected"}
            continue

        if idx == 0:  # R1 deep stable
            if r.status == "RUNAWAY":
                regime_status[idx] = {"name": name, "verdict": "FAIL",
                                      "reason": "runaway at low α"}
            elif r.status == "SILENT":
                regime_status[idx] = {"name": name, "verdict": "WARN",
                                      "reason": "silent"}
            else:
                regime_status[idx] = {"name": name, "verdict": "PASS",
                                      "reason": ""}

        elif idx in (1, 2, 3):  # R2, R3, R4 — must be alive
            if r.status != "OK":
                regime_status[idx] = {"name": name, "verdict": "FAIL",
                                      "reason": f"status={r.status}"}
            elif r.pct_silent_E > silent_thr:
                regime_status[idx] = {"name": name, "verdict": "WARN",
                                      "reason": f"pct_silent_E={r.pct_silent_E:.0f}%"
                                                 f" > {silent_thr:.0f}%"}
            else:
                regime_status[idx] = {"name": name, "verdict": "PASS",
                                      "reason": ""}

        elif idx == 4:  # R5 strong chaos — tolerate RUNAWAY/SILENT
            if r.status == "NAN":
                regime_status[idx] = {"name": name, "verdict": "FAIL",
                                      "reason": "NaN"}
            elif r.status in ("SILENT", "RUNAWAY"):
                regime_status[idx] = {"name": name, "verdict": "WARN",
                                      "reason": f"status={r.status}"}
            elif r.pct_silent_E > silent_thr:
                regime_status[idx] = {"name": name, "verdict": "WARN",
                                      "reason": f"pct_silent_E={r.pct_silent_E:.0f}%"
                                                 f" > {silent_thr:.0f}%"}
            else:
                regime_status[idx] = {"name": name, "verdict": "PASS",
                                      "reason": ""}

    # ── Separation criteria (R1 vs R5) ──
    verdicts = [v["verdict"] for v in regime_status.values()]
    if "FAIL" in verdicts:
        overall = "FAIL"
    else:
        # Check CV/Fano ratio between R1 and R5
        r1 = next((r for r in results if r.regime_index == 0), None)
        r5 = next((r for r in results if r.regime_index == 4), None)

        sep_ok = True
        sep_warn = False
        if r1 and r5 and r1.status == "OK" and r5.status == "OK":
            cv1, cv5 = r1.cv_isi_E, r5.cv_isi_E
            fano1, fano5 = r1.fano_E, r5.fano_E

            # WARN thresholds
            if cv1 > 1e-9 and cv5 < 1.10 * cv1:
                sep_warn = True
            if fano1 > 1e-9 and fano5 < 1.10 * fano1:
                sep_warn = True

            # PASS thresholds
            if cv1 > 1e-9 and cv5 < 1.15 * cv1:
                sep_ok = False
            if fano1 > 1e-9 and fano5 < 1.15 * fano1:
                sep_ok = False

        # Check %silent across all regimes
        for r in results:
            if r.pct_silent_E > silent_thr and r.regime_index in (2, 3, 4):
                sep_ok = False

        if "WARN" in verdicts or sep_warn or not sep_ok:
            overall = "WARN" if "WARN" in verdicts or sep_warn else "PASS"
            if not sep_ok and "WARN" not in verdicts:
                overall = "WARN"
        else:
            overall = "PASS"

    return {
        "overall": overall,
        "per_regime": regime_status,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BRIAN2 NETWORK BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _import_brian2():
    """Lazy Brian2 import."""
    from brian2 import (
        start_scope, NeuronGroup, Synapses, SpikeGeneratorGroup,
        SpikeMonitor, Network, Equations, defaultclock,
    )
    from brian2 import ms as b_ms, mV as b_mV, nA as b_nA
    from brian2 import umetre, ufarad, siemens, msiemens, cm

    _area = 20000 * umetre ** 2
    # Unit conversion factors for per-neuron parameters
    constants = {
        "El": -65 * b_mV,
        "EK": -90 * b_mV,
        "ENa": 50 * b_mV,
        # Conversion factors: bundle stores dimensionless multipliers
        # relative to default HH values; these are the base units.
        "_Cm_unit": 1 * ufarad * cm ** -2 * _area,
        "_gl_unit": 1 * siemens * cm ** -2 * _area,
        "_gna_unit": 1 * msiemens * cm ** -2 * _area,
        "_gkd_unit": 1 * msiemens * cm ** -2 * _area,
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
        "ms": b_ms, "mV": b_mV, "nA": b_nA,
        **constants,
    }


HH_EQUATIONS = '''
Cm : farad (constant)
gl : siemens (constant)
g_na : siemens (constant)
g_kd : siemens (constant)
VT : volt (constant)

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

I_b : ampere (constant)
tau_stimulus : second (constant)
dI_stimulus/dt = -I_stimulus/tau_stimulus : ampere

I_syn_ee_syn : ampere
I_syn_ei_syn : ampere
I_syn_ie_syn : ampere
I_syn_ii_syn : ampere

w_input : 1 (constant)
I_narma : ampere

I_total = I_b + I_stimulus + I_narma
          + I_syn_ee_syn + I_syn_ei_syn
          + I_syn_ie_syn + I_syn_ii_syn : ampere
'''

STP_MODEL_TEMPLATE = """
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

BLOCK_TO_SYN_NAME = {"EE": "ee_syn", "EI": "ei_syn", "IE": "ie_syn", "II": "ii_syn"}
BLOCK_SIGN = {"EE": +1.0, "EI": +1.0, "IE": -1.0, "II": -1.0}
TAU_I_MAP = {"EE": 3.0, "EI": 3.0, "IE": 6.0, "II": 6.0}


def _jitter_spike_collisions(
    indices: np.ndarray, times_ms: np.ndarray, dt_ms: float,
) -> tuple:
    """Shift spikes that collide within the same dt bin for a given neuron.

    Brian2 SpikeGeneratorGroup forbids the same neuron spiking twice in one
    dt bin. For each collision, shift the later spike forward by dt_ms.
    Returns (indices, times_ms) sorted by time.
    """
    if len(times_ms) == 0:
        return indices, times_ms

    times_ms = times_ms.copy()
    unique_neurons = np.unique(indices)

    for nid in unique_neurons:
        pos = np.flatnonzero(indices == nid)
        if len(pos) < 2:
            continue

        # Sort this neuron's spikes by time
        t = times_ms[pos]
        sort_order = np.argsort(t)
        t_sorted = t[sort_order]

        # Shift collisions: if two consecutive spikes are in the same dt bin
        for j in range(1, len(t_sorted)):
            if t_sorted[j] - t_sorted[j - 1] < dt_ms:
                t_sorted[j] = t_sorted[j - 1] + dt_ms

        # Write back: pos[sort_order[k]] is the original flat index for the
        # k-th spike in sorted order
        times_ms[pos[sort_order]] = t_sorted

    # Re-sort globally by time
    order = np.argsort(times_ms)
    return indices[order], times_ms[order]


def build_brian2_from_bundle(bundle_dir: Path, alpha: float,
                             warmup_ms: float, measure_ms: float,
                             dt_ms: float = 0.025,
                             poisson_realization: int = 0,
                             narma_drive: np.ndarray | None = None,
                             narma_dt_ms: float = 10.0,
                             narma_scale_nA: float = 0.5,
                             bg_scale: float = 1.0):
    """Build and run a Brian2 HH+STP simulation from bundle data.

    Parameters
    ----------
    narma_drive : ndarray or None
        1-D array of input signal u[k] values. Each value is held constant
        for narma_dt_ms milliseconds, applied as current injection
        I_narma = u[k] * narma_scale_nA to ALL reservoir neurons.
        If None, I_narma = 0 (smoke test mode).
    narma_dt_ms : float
        Duration per NARMA step in ms (default: 10.0).
    narma_scale_nA : float
        Scaling factor: I_narma = u[k] * narma_scale_nA (nA).
    bg_scale : float
        Multiplier for Poisson background current. Default 1.0 preserves
        existing behavior (1.5 nA for E, 0.75 nA for I).

    Returns (spike_trains_dict, E_idx, I_idx, neurons, b).
    """
    b = _import_brian2()
    b["start_scope"]()

    # ── Load bundle data ──
    pop_data = json.loads((bundle_dir / "network" / "population.json").read_text())
    E_idx = np.array(pop_data["E_idx"], dtype=np.int32)
    I_idx = np.array(pop_data["I_idx"], dtype=np.int32)
    N = pop_data["N_E"] + pop_data["N_I"]

    edges_raw = np.load(bundle_dir / "network" / "edges.npz")
    pre = edges_raw["pre"]
    post = edges_raw["post"]
    block_id = edges_raw["block_id"]

    syn_raw = np.load(bundle_dir / "network" / "synapse_params.npz")
    w_all = syn_raw["w"]
    U_all = syn_raw["U"]
    tau_d_all = syn_raw["tau_d_ms"]
    tau_f_all = syn_raw["tau_f_ms"]
    delay_all = syn_raw["delay_ms"]

    neuron_raw = np.load(bundle_dir / "network" / "neuron_params.npz")

    poi_raw = np.load(bundle_dir / "poisson" / "trains_3s.npz")
    n_ch = int(poi_raw["n_channels"])

    total_ms = warmup_ms + measure_ms

    # ── Neurons (per-neuron heterogeneity from bundle) ──
    neurons = b["NeuronGroup"](
        N, b["Equations"](HH_EQUATIONS),
        threshold='v > -40*mV',
        refractory='v > -40*mV',
        method='exponential_euler',
        name='neurons',
        namespace={k: b[k] for k in ("El", "EK", "ENa")},
    )
    neurons.v = b["El"]
    neurons.m = 0
    neurons.h = 1
    neurons.n = 0.5

    # Per-neuron HH parameters from neuron_params.npz
    # Bundle stores: Cm (µF/cm² × area), gL (S/cm² × area),
    # gNa (mS/cm² × area), gK (mS/cm² × area), Vt (mV), Ib (nA)
    neurons.Cm = neuron_raw["Cm"] * b["_Cm_unit"]
    neurons.gl = neuron_raw["gL"] * b["_gl_unit"]
    neurons.g_na = neuron_raw["gNa"] * b["_gna_unit"]
    neurons.g_kd = neuron_raw["gK"] * b["_gkd_unit"]
    neurons.VT = neuron_raw["Vt"] * b["mV"]
    neurons.I_b = neuron_raw["Ib"] * b["nA"]

    neurons.I_stimulus = 0 * b["nA"]
    neurons.tau_stimulus = 3 * b["ms"]
    for ii in I_idx:
        neurons.tau_stimulus[int(ii)] = 6 * b["ms"]

    objects = [neurons]

    # ── STP Synapses (per block) ──
    block_id_map = {"EE": 0, "EI": 1, "IE": 2, "II": 3}
    for bname, bid in block_id_map.items():
        mask = block_id == bid
        if not np.any(mask):
            continue

        syn_name = BLOCK_TO_SYN_NAME[bname]
        sign = BLOCK_SIGN[bname]
        eqs = STP_MODEL_TEMPLATE.format(name=syn_name)

        syn = b["Synapses"](
            neurons, neurons, model=eqs, on_pre=STP_ON_PRE,
            method='exact', name=syn_name,
        )
        syn.connect(i=pre[mask], j=post[mask])

        syn.A = sign * alpha * np.abs(w_all[mask]) * b["nA"]
        syn.U = U_all[mask]
        syn.D = tau_d_all[mask] * b["ms"]
        syn.F = np.maximum(tau_f_all[mask], 0.1) * b["ms"]
        syn.delay = delay_all[mask] * b["ms"]
        syn.tau_I = TAU_I_MAP[bname] * b["ms"]
        syn.x = 1
        syn.u = U_all[mask]
        objects.append(syn)

    # ── Poisson spike generators ──
    # Map channels to neurons (first N_E → E, next N_I → I)
    N_E = len(E_idx)
    N_I = len(I_idx)

    # E trains
    e_indices, e_times = [], []
    for ch_i in range(min(N_E, n_ch)):
        t_s = poi_raw[f"ch{ch_i}"]
        t_ms = t_s * 1000.0
        t_ms = t_ms[t_ms < total_ms]
        if len(t_ms) > 0:
            e_indices.append(np.full(len(t_ms), ch_i, dtype=np.int32))
            e_times.append(t_ms)

    if e_indices:
        e_idx_arr = np.concatenate(e_indices)
        e_t_arr = np.concatenate(e_times)
        order = np.argsort(e_t_arr)
        e_idx_arr, e_t_arr = e_idx_arr[order], e_t_arr[order]
    else:
        e_idx_arr = np.zeros(0, dtype=np.int32)
        e_t_arr = np.zeros(0, dtype=np.float64)

    # Jitter collisions: if same neuron spikes twice in same dt bin, shift +dt
    e_idx_arr, e_t_arr = _jitter_spike_collisions(e_idx_arr, e_t_arr, dt_ms)

    sg_E = b["SpikeGeneratorGroup"](N_E, e_idx_arr, e_t_arr * b["ms"], name='sg_E')
    bg_E_nA = 1.5 * bg_scale
    syn_sg_E = b["Synapses"](sg_E, neurons, on_pre=f'I_stimulus += {bg_E_nA}*nA', name='syn_sg_E')
    syn_sg_E.connect(i=np.arange(N_E), j=E_idx)
    objects.extend([sg_E, syn_sg_E])

    # I trains
    i_indices, i_times = [], []
    for ch_i in range(N_I):
        ch_global = N_E + ch_i
        if ch_global >= n_ch:
            break
        t_s = poi_raw[f"ch{ch_global}"]
        t_ms = t_s * 1000.0
        t_ms = t_ms[t_ms < total_ms]
        if len(t_ms) > 0:
            i_indices.append(np.full(len(t_ms), ch_i, dtype=np.int32))
            i_times.append(t_ms)

    if i_indices:
        i_idx_arr = np.concatenate(i_indices)
        i_t_arr = np.concatenate(i_times)
        order = np.argsort(i_t_arr)
        i_idx_arr, i_t_arr = i_idx_arr[order], i_t_arr[order]
    else:
        i_idx_arr = np.zeros(0, dtype=np.int32)
        i_t_arr = np.zeros(0, dtype=np.float64)

    # Jitter collisions for I trains too
    i_idx_arr, i_t_arr = _jitter_spike_collisions(i_idx_arr, i_t_arr, dt_ms)

    sg_I = b["SpikeGeneratorGroup"](N_I, i_idx_arr, i_t_arr * b["ms"], name='sg_I')
    bg_I_nA = 0.75 * bg_scale
    syn_sg_I = b["Synapses"](sg_I, neurons, on_pre=f'I_stimulus += {bg_I_nA}*nA', name='syn_sg_I')
    syn_sg_I.connect(i=np.arange(N_I), j=I_idx)
    objects.extend([sg_I, syn_sg_I])

    # ── NARMA current injection via TimedArray ──
    if narma_drive is not None:
        from brian2 import TimedArray
        # Sparse random input weights (standard LSM approach):
        # Only ~30% of neurons receive the input, with N(0,1) weights.
        # This creates diverse initial responses; the recurrent network
        # transforms this into a rich high-dimensional state.
        rng_input = np.random.default_rng(98765)
        w_in = np.zeros(N, dtype=np.float64)
        input_mask = rng_input.random(N) < 0.3  # ~30% input neurons
        n_input = max(1, input_mask.sum())
        w_in[input_mask] = rng_input.standard_normal(n_input)
        neurons.w_input = w_in

        # Build piecewise-constant current signal: one value per narma_dt_ms
        # narma_drive is the raw u[k] array; scale to nA
        narma_signal_nA = narma_drive * narma_scale_nA
        narma_ta = TimedArray(narma_signal_nA * b["nA"], dt=narma_dt_ms * b["ms"])
        # I_narma = w_input * narma_ta(t) — each neuron gets different weight
        neurons.run_regularly(
            'I_narma = w_input * narma_ta(t)',
            dt=dt_ms * b["ms"],
            name='narma_update',
        )
        neurons.namespace['narma_ta'] = narma_ta
    else:
        neurons.w_input = 0.0
        neurons.I_narma = 0 * b["nA"]

    spike_mon = b["SpikeMonitor"](neurons, name='spike_monitor')
    objects.append(spike_mon)

    b["defaultclock"].dt = dt_ms * b["ms"]
    net = b["Network"](objects)
    net.store()

    # ── Run ──
    rng_v = np.random.default_rng(12345)
    net.restore()
    neurons.v = (float(b["El"] / b["mV"]) + rng_v.uniform(-2, 2, N)) * b["mV"]
    if narma_drive is not None:
        # Re-apply I_narma after restore (restore resets it to 0)
        neurons.I_narma = 0 * b["nA"]
    net.run(total_ms * b["ms"])

    return spike_mon.spike_trains(), E_idx, I_idx, neurons, b


def run_smoke_one_regime(bundle_dir: Path, regime: dict,
                         warmup_ms: float, measure_ms: float,
                         dt_ms: float = 0.025) -> SmokeMetrics:
    """Run one regime and compute all metrics."""
    alpha = regime["alpha"]
    total_ms = warmup_ms + measure_ms

    st, E_idx, I_idx, neurons, b = build_brian2_from_bundle(
        bundle_dir, alpha, warmup_ms, measure_ms, dt_ms)

    rate_E = compute_rate(st, E_idx, warmup_ms, total_ms, b)
    rate_I = compute_rate(st, I_idx, warmup_ms, total_ms, b)
    cv_E = compute_cv_isi(st, E_idx, warmup_ms, total_ms, b)
    cv_I = compute_cv_isi(st, I_idx, warmup_ms, total_ms, b)
    fano_E = compute_fano(st, E_idx, warmup_ms, total_ms, b)
    fano_I = compute_fano(st, I_idx, warmup_ms, total_ms, b)
    sync_E = compute_sync(st, E_idx, warmup_ms, total_ms, b)
    sync_I = compute_sync(st, I_idx, warmup_ms, total_ms, b)
    pct_silent_E = compute_pct_silent(st, E_idx, warmup_ms, total_ms, b)
    pct_silent_I = compute_pct_silent(st, I_idx, warmup_ms, total_ms, b)
    spike_count_E = compute_spike_count(st, E_idx, warmup_ms, total_ms, b)
    spike_count_I = compute_spike_count(st, I_idx, warmup_ms, total_ms, b)

    v_arr = np.array(neurons.v / b["mV"])
    has_nan = bool(np.any(np.isnan(v_arr)) or np.any(np.isinf(v_arr)))
    status = evaluate_status(rate_E, rate_I, has_nan)

    return SmokeMetrics(
        regime_name=regime["name"], regime_index=regime["index"],
        alpha=alpha,
        rate_E=rate_E, rate_I=rate_I,
        cv_isi_E=cv_E, cv_isi_I=cv_I,
        fano_E=fano_E, fano_I=fano_I,
        sync_E=sync_E, sync_I=sync_I,
        pct_silent_E=pct_silent_E, pct_silent_I=pct_silent_I,
        spike_count_E=spike_count_E, spike_count_I=spike_count_I,
        has_nan=has_nan, status=status,
    )


def run_smoke(bundle_dir: Path, regime_indices: Optional[List[int]] = None,
              warmup_ms: float = 1000.0, measure_ms: float = 2000.0,
              dt_ms: float = 0.025) -> List[SmokeMetrics]:
    """Run smoke test for selected regimes (or all).

    Returns list of SmokeMetrics + saves validation/metrics.json to bundle.
    """
    bundle_dir = Path(bundle_dir)

    # Verify integrity
    errors = verify_manifest(bundle_dir)
    if errors:
        print(f"WARNING: bundle integrity issues: {errors}")

    regimes_raw = json.loads(
        (bundle_dir / "regimes" / "regimes.json").read_text())
    # Support both formats: list (simple) or dict with "regimes" key (edge)
    if isinstance(regimes_raw, list):
        regimes = regimes_raw
    else:
        regimes = regimes_raw.get("regimes", regimes_raw)

    if regime_indices is None:
        regime_indices = list(range(len(regimes)))

    print(f"{'='*60}")
    print(f"  BRIAN2 SMOKE TEST")
    print(f"  bundle: {bundle_dir}")
    print(f"  regimes: {regime_indices}")
    print(f"  warmup={warmup_ms}ms  measure={measure_ms}ms  dt={dt_ms}ms")
    print(f"{'='*60}")

    results = []
    for ri in regime_indices:
        reg = regimes[ri] if isinstance(regimes[ri], dict) else regimes[ri]
        t0 = time.time()
        m = run_smoke_one_regime(bundle_dir, reg, warmup_ms, measure_ms, dt_ms)
        dt_run = time.time() - t0
        results.append(m)
        print(f"  {m.regime_name:20s}  α={m.alpha:.6f}  "
              f"rE={m.rate_E:5.1f}  rI={m.rate_I:5.1f}  "
              f"cvE={m.cv_isi_E:.2f}  fanoE={m.fano_E:.2f}  "
              f"syncE={m.sync_E:.2f}  silent_E={m.pct_silent_E:.0f}%  "
              f"[{m.status}]  {dt_run:.1f}s")

    # Separation check
    sep_pass, sep_details = check_separation(results)
    if sep_pass:
        print(f"\n  Separation: PASS")
    else:
        print(f"\n  Separation: WARN (no metric >=15% spread)")
        for d in sep_details:
            print(f"    {d['metric']:12s}: span={d['span']:.3f} "
                  f"thr={d['threshold']:.3f} {'PASS' if d['pass'] else 'FAIL'}")

    # 5-regime quality check
    quality = evaluate_5regime_quality(results)
    print(f"\n  5-regime quality: {quality['overall']}")
    for idx, info in sorted(quality["per_regime"].items()):
        suffix = f" ({info['reason']})" if info["reason"] else ""
        print(f"    {info['name']:20s}  {info['verdict']}{suffix}")

    # Save results
    val_dir = bundle_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    out_path = val_dir / "metrics.json"
    out_data = {
        "warmup_ms": warmup_ms,
        "measure_ms": measure_ms,
        "dt_ms": dt_ms,
        "results": [asdict(r) for r in results],
        "separation": {"pass": sep_pass, "details": sep_details},
        "quality_5regime": quality,
    }
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Saved: {out_path}")

    # Summary
    fails = [r for r in results if r.status in ("NAN", "RUNAWAY")]
    if fails:
        print(f"\n  FAIL: {len(fails)} regimes failed liveness")
        for f in fails:
            print(f"    {f.regime_name}: {f.status}")
    else:
        print(f"\n  All {len(results)} regimes: LIVENESS OK")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Brian2 smoke test for frozen bundle")
    ap.add_argument("bundle_dir", type=Path)
    ap.add_argument("--regime", default="all",
                    help="Regime index (0-4) or 'all'")
    ap.add_argument("--warmup-ms", type=float, default=1000.0)
    ap.add_argument("--measure-ms", type=float, default=2000.0)
    ap.add_argument("--dt-ms", type=float, default=0.025)
    args = ap.parse_args()

    if args.regime == "all":
        indices = None
    else:
        indices = [int(args.regime)]

    run_smoke(args.bundle_dir, indices,
              args.warmup_ms, args.measure_ms, args.dt_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
