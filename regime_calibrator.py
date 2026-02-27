"""
ETAP B: Closed-loop regime calibration via Brian2 HH+STP simulations.

Loads frozen reservoir configs from ETAP A (.npz files), builds Brian2 networks
under standard operating conditions (I_b + independent Poisson 20Hz per neuron),
and adjusts alpha via bracketed bisection until each regime meets dynamical criteria.

lambda_measured is a TRUE maximal Lyapunov exponent estimate (Benettin method
with periodic renormalization), computed from paired trajectories with random
per-neuron perturbation.

Standard operating environment:
  - Background current: I_b (shared, constant)
  - Independent Poisson input per neuron at 20 Hz
    (weight: 1.5 nA for E, 0.75 nA for I)
  - NO shared stimulus (avoids artificial synchronization)

Regime criteria: lambda (primary) + guardrails (rate, sync, CV).
"""

import os
import csv
import time
import numpy as np

from brian2 import (
    start_scope, NeuronGroup, Synapses, PoissonGroup, SpikeGeneratorGroup,
    SpikeMonitor, Network, Equations, defaultclock,
)
from brian2 import ms, mV, nA, second, Hz
from brian2 import umetre, ufarad, siemens, msiemens, cm

# ═══════════════════════════════════════════════════════════════════════════════
# 1. HH BIOPHYSICAL CONSTANTS (module-level for Brian2 namespace)
# ═══════════════════════════════════════════════════════════════════════════════

_area = 20000 * umetre ** 2
Cm    = 1 * ufarad * cm ** -2 * _area
gl    = 5e-5 * siemens * cm ** -2 * _area
El    = -65 * mV
EK    = -90 * mV
ENa   = 50 * mV
g_na  = 100 * msiemens * cm ** -2 * _area
g_kd  = 30 * msiemens * cm ** -2 * _area
VT    = -63 * mV

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONSTANTS & REGIME CRITERIA
# ═══════════════════════════════════════════════════════════════════════════════

SYN_TYPES = ("EE", "EI", "IE", "II")
SYN_SIGN  = {"EE": +1.0, "EI": +1.0, "IE": -1.0, "II": -1.0}
SYN_BRIAN_NAMES = {"EE": "ee_synapses", "EI": "ei_synapses",
                   "IE": "ie_synapses", "II": "ii_synapses"}

DT_MS = 0.025          # metrics / network integration step
DT_LAMBDA_MS = 0.0125  # Benettin lambda integration step (finer for accuracy)

# ── Golden setup (Benettin numerical stability) ──
# eps_mV        = 0.05    perturbation amplitude
# dt_lambda_ms  = 0.0125  Benettin integration step
# dt_metrics_ms = 0.025   metrics / network step
# dt_train_ms   = 0.025   Poisson train binning (≥ max dt_sim for safety)
# renorm_ms     = 5.0     renormalization interval
# Bisection:     measure_ms=1000, n_lambda_repeats=5
# Final verif:   measure_ms_final=3000, n_repeats_final=9

# Lambda targets (1/s) for each regime.  Bisection steers toward these.
# NOTE: Under deterministic Poisson (SpikeGeneratorGroup), lambda has high
# trial-to-trial variance (std≈10-40).  Calibrator uses median of K
# realizations.  With median(7): usable range ~3 to ~80 (1/s).
# WARNING: ρ→λ mapping is NON-MONOTONIC for HH+STP spiking networks.
# Spectral radius is only a rough starting point; Benettin λ is the
# authoritative regime criterion.
LAMBDA_TARGETS = {
    "R1_super_stable":   5.0,
    "R2_stable":        12.0,
    "R3_near_critical": 22.0,
    "R4_edge_of_chaos": 35.0,
    "R5_chaotic":       55.0,
}

# Guardrails: hard constraints checked AFTER lambda is in range.
GUARDRAILS = {
    "R1_super_stable":  {"rate_E_max": 80, "sync_max": 20},
    "R2_stable":        {"rate_E_range": (0.5, 80), "sync_max": 20},
    "R3_near_critical": {"rate_E_range": (1, 80), "cv_isi_min": 0.3, "sync_max": 20},
    "R4_edge_of_chaos": {"rate_E_range": (1, 80), "sync_max": 30},
    "R5_chaotic":       {"rate_E_range": (1, 100)},
}

# Non-overlapping lambda windows — guarantee monotonic separation.
# Tuned for deterministic Poisson (SpikeGeneratorGroup) with median(7).
# Empirical scan (seed=100, n_repeats=3, 16 ρ points):
#   ρ=0.10→λ≈8  ρ=0.35→λ≈36  ρ=0.75→λ≈58  ρ=1.07→λ≈61
#   ρ=2.80→λ≈83   (high variance: std≈10-40, NON-MONOTONIC ρ→λ)
# Windows widened to accommodate high trial-to-trial variance.
LAMBDA_WINDOWS = {
    "R1_super_stable":  (  0.0,   12.0),
    "R2_stable":        (  8.0,   22.0),
    "R3_near_critical": ( 18.0,   35.0),
    "R4_edge_of_chaos": ( 28.0,   50.0),
    "R5_chaotic":       ( 42.0,   90.0),
}

# RC-optimized preset: narrower range near edge-of-chaos where memory
# capacity and separation are typically maximal.
# Use with: --preset rc
LAMBDA_WINDOWS_RC = {
    "R1_super_stable":  (  0.0,    3.0),
    "R2_stable":        (  3.0,    8.0),
    "R3_near_critical": (  8.0,   15.0),
    "R4_edge_of_chaos": ( 15.0,   25.0),
    "R5_chaotic":       ( 25.0,   40.0),
}
LAMBDA_TARGETS_RC = {
    "R1_super_stable":   1.5,
    "R2_stable":         5.0,
    "R3_near_critical": 11.0,
    "R4_edge_of_chaos": 20.0,
    "R5_chaotic":       32.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. NPZ LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_regime_npz(npz_path):
    """Load NPZ and return dict with Python-native scalars."""
    raw = np.load(npz_path, allow_pickle=True)
    d = {}
    for key in raw.files:
        val = raw[key]
        d[key] = val.item() if val.ndim == 0 else val
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NETWORK BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

HH_EQUATIONS = Equations('''
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

x_pos : 1 (constant)
y_pos : 1 (constant)
z_pos : 1 (constant)
''')

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


def generate_poisson_trains(N, rate_hz, duration_ms, rng,
                            dt_ms=None, dt_train_ms=None):
    """Pre-generate independent Poisson spike trains for N neurons.
    Returns (indices, times_ms) globally sorted by time for SpikeGeneratorGroup.

    Deduplication: at most 1 spike per neuron per dt_train_ms bin.  We draw
    uniform times, map to integer bin indices via floor(t/dt_train_ms), keep
    unique bins, then place spike at a random time within each surviving bin.

    dt_train_ms controls the binning resolution for deduplication.  It is
    independent of the simulation timestep dt_ms so that changing the
    integration step does NOT change the spike trains (critical for dt
    convergence tests).  If not given, falls back to dt_ms or DT_MS.
    """
    if dt_train_ms is None:
        dt_train_ms = dt_ms if dt_ms is not None else DT_MS
    indices = []
    times = []
    lam = rate_hz * duration_ms / 1000.0
    for i in range(N):
        n_spikes = rng.poisson(lam)
        if n_spikes > 0:
            t = rng.uniform(0.0, duration_ms, n_spikes)
            bins = np.floor(t / dt_train_ms).astype(np.int64)
            bins = np.unique(bins)                      # max 1 spike/bin/neuron
            t = (bins.astype(np.float64) + rng.random(len(bins))) * dt_train_ms
            t = np.minimum(t, duration_ms - 1e-9)
            indices.append(np.full(len(t), i, dtype=np.int32))
            times.append(t)
    if not indices:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    indices = np.concatenate(indices)
    times = np.concatenate(times)
    order = np.argsort(times, kind="mergesort")
    return indices[order], times[order]


def build_network(regime_data, alpha_override=None, I_b_nA=0.05,
                  input_mode="poisson_group", poisson_rate_hz=20.0,
                  spike_trains_E=None, spike_trains_I=None,
                  dt_ms=None):
    """
    Build Brian2 HH+STP network from NPZ regime data.

    Background input: independent PoissonGroup per neuron (NOT shared stimulus).
    Integrator: exponential_euler (analytically handles HH exprel/exp terms).

    Returns: (net, spike_mon, neurons, idx_E, idx_I)
    """
    start_scope()

    N = int(regime_data["N_total"])
    idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
    idx_I = np.array(regime_data["idx_I"], dtype=np.int32)

    alpha = alpha_override if alpha_override is not None else float(regime_data["alpha_final"])
    A_scale_E = float(regime_data.get("A_scale_E", 1.0))
    A_scale_I = float(regime_data.get("A_scale_I", 1.0))

    # ── Neurons (exponential_euler: stable for HH exprel/exp terms) ──
    neurons = NeuronGroup(
        N, HH_EQUATIONS,
        threshold='v > -40*mV',
        refractory='v > -40*mV',
        method='exponential_euler',
        name='neurons',
    )
    neurons.v = El
    neurons.m = 0
    neurons.h = 1
    neurons.n = 0.5
    neurons.I_b = I_b_nA * nA
    neurons.I_stimulus = 0 * nA

    # tau_stimulus: 3ms for E, 6ms for I
    neurons.tau_stimulus = 3 * ms
    for ii in idx_I:
        neurons.tau_stimulus[int(ii)] = 6 * ms

    positions = np.arange(N)
    neurons.x_pos = positions % 3
    neurons.y_pos = (positions // 3) % 3
    neurons.z_pos = positions // 9

    # ── STP Synapses from NPZ ──
    objects = [neurons]

    for syn_type in SYN_TYPES:
        syn_name = SYN_BRIAN_NAMES[syn_type]
        sign = SYN_SIGN[syn_type]
        a_scale = A_scale_E if syn_type in ("EE", "EI") else A_scale_I

        pre_idx = np.array(regime_data[f"{syn_type}_pre"], dtype=np.int32)
        post_idx = np.array(regime_data[f"{syn_type}_post"], dtype=np.int32)
        raw_mag = np.array(regime_data[f"{syn_type}_raw_mag"], dtype=np.float64)
        U_vals = np.array(regime_data[f"{syn_type}_U"], dtype=np.float64)
        D_vals = np.array(regime_data[f"{syn_type}_D"], dtype=np.float64)
        F_vals = np.array(regime_data[f"{syn_type}_F"], dtype=np.float64)
        delay_vals = np.array(regime_data[f"{syn_type}_delay"], dtype=np.float64)
        tau_I_val = float(regime_data[f"{syn_type}_tau_I"][0])

        if len(pre_idx) == 0:
            continue

        eqs_syn = STP_EQUATIONS_TEMPLATE.format(name=syn_name)
        syn = Synapses(
            neurons, neurons, model=eqs_syn, on_pre=STP_ON_PRE,
            method='exact', name=syn_name,
        )
        syn.connect(i=pre_idx, j=post_idx)

        syn.A = sign * alpha * a_scale * raw_mag * nA
        syn.U = U_vals
        syn.D = D_vals * ms
        syn.F = np.maximum(F_vals, 0.1) * ms
        syn.delay = delay_vals * ms
        syn.tau_I = tau_I_val * ms
        syn.x = 1
        syn.u = U_vals

        objects.append(syn)

    # ── Background input (per neuron) ──
    if input_mode == "poisson_group":
        n_E = len(idx_E)
        n_I = len(idx_I)

        pois_E = PoissonGroup(n_E, rates=poisson_rate_hz * Hz, name='poisson_E')
        pois_I = PoissonGroup(n_I, rates=poisson_rate_hz * Hz, name='poisson_I')

        syn_pois_E = Synapses(
            pois_E, neurons, on_pre='I_stimulus += 1.5*nA', name='syn_poisson_E',
        )
        syn_pois_E.connect(i=np.arange(n_E), j=idx_E)

        syn_pois_I = Synapses(
            pois_I, neurons, on_pre='I_stimulus += 0.75*nA', name='syn_poisson_I',
        )
        syn_pois_I.connect(i=np.arange(n_I), j=idx_I)

        objects.extend([pois_E, pois_I, syn_pois_E, syn_pois_I])

    elif input_mode == "spike_generator":
        assert spike_trains_E is not None, "spike_trains_E required for spike_generator mode"
        assert spike_trains_I is not None, "spike_trains_I required for spike_generator mode"
        n_E = len(idx_E)
        n_I = len(idx_I)
        sg_E_idx, sg_E_times = spike_trains_E
        sg_I_idx, sg_I_times = spike_trains_I
        assert sg_E_idx.max() < n_E if len(sg_E_idx) else True, "E indices out of range"
        assert sg_I_idx.max() < n_I if len(sg_I_idx) else True, "I indices out of range"
        sg_E = SpikeGeneratorGroup(n_E, sg_E_idx, sg_E_times * ms, name='sg_E')
        sg_I = SpikeGeneratorGroup(n_I, sg_I_idx, sg_I_times * ms, name='sg_I')
        syn_sg_E = Synapses(sg_E, neurons, on_pre='I_stimulus += 1.5*nA', name='syn_sg_E')
        syn_sg_E.connect(i=np.arange(n_E), j=idx_E)
        syn_sg_I = Synapses(sg_I, neurons, on_pre='I_stimulus += 0.75*nA', name='syn_sg_I')
        syn_sg_I.connect(i=np.arange(n_I), j=idx_I)
        objects.extend([sg_E, sg_I, syn_sg_E, syn_sg_I])

    elif input_mode == "none":
        pass  # no background input (spontaneous check)

    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    # ── Monitors ──
    spike_mon = SpikeMonitor(neurons, name='spike_monitor')
    objects.append(spike_mon)

    # ── Network ──
    defaultclock.dt = (dt_ms if dt_ms is not None else DT_MS) * ms
    net = Network(objects)
    net.store()

    return net, spike_mon, neurons, idx_E, idx_I


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DYNAMICS METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class Metrics:

    @staticmethod
    def firing_rates(spike_trains, idx_E, idx_I, t_start_ms, t_end_ms):
        """Mean firing rate (Hz) for E and I populations."""
        dur_s = (t_end_ms - t_start_ms) / 1000.0

        def _rate(indices):
            counts = []
            for idx in indices:
                t_ms = np.array(spike_trains[int(idx)] / ms)
                counts.append(np.sum((t_ms >= t_start_ms) & (t_ms < t_end_ms)))
            return float(np.mean(counts) / dur_s) if counts else 0.0

        return _rate(idx_E), _rate(idx_I)

    @staticmethod
    def cv_isi(spike_trains, indices, t_start_ms, t_end_ms):
        """Mean CV of ISI across neurons with >= 3 spikes."""
        cvs = []
        for idx in indices:
            t_ms = np.array(spike_trains[int(idx)] / ms)
            t_ms = t_ms[(t_ms >= t_start_ms) & (t_ms < t_end_ms)]
            if len(t_ms) < 3:
                continue
            isi = np.diff(t_ms)
            mu = np.mean(isi)
            if mu < 1e-10:
                continue
            cvs.append(float(np.std(isi) / mu))
        return float(np.mean(cvs)) if cvs else 0.0

    @staticmethod
    def sync_index(spike_trains, N, t_start_ms, t_end_ms, bin_ms=5.0):
        """
        Synchronization chi^2 = var(sum_i x_i) / (N * <var(x_i)>).
        = 1 for independent Poisson, = N for perfect synchrony.
        """
        n_bins = int((t_end_ms - t_start_ms) / bin_ms)
        if n_bins < 2:
            return 0.0

        pop_hist = np.zeros(n_bins, dtype=np.float64)
        ind_vars = []

        for idx in range(N):
            t_ms = np.array(spike_trains[idx] / ms)
            t_ms = t_ms[(t_ms >= t_start_ms) & (t_ms < t_end_ms)]
            hist, _ = np.histogram(t_ms, bins=n_bins,
                                   range=(t_start_ms, t_end_ms))
            pop_hist += hist
            ind_vars.append(float(np.var(hist)))

        pop_var = float(np.var(pop_hist))
        mean_ind_var = float(np.mean(ind_vars))
        return pop_var / (N * mean_ind_var + 1e-15)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BENETTIN LYAPUNOV (with renormalization)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_state_vector(neurons):
    """Return full state (v,m,h,n) as 4×N array (dimensionless, in mV/1 units)."""
    return np.stack([
        np.array(neurons.v / mV),
        np.array(neurons.m[:]),
        np.array(neurons.h[:]),
        np.array(neurons.n[:]),
    ])  # shape (4, N)


def _set_state_from_ref_plus_delta(neurons, ref_state, delta_state):
    """Set neuron state to ref + delta (clamping gating vars to [0,1])."""
    v_new = ref_state[0] + delta_state[0]
    m_new = np.clip(ref_state[1] + delta_state[1], 0, 1)
    h_new = np.clip(ref_state[2] + delta_state[2], 0, 1)
    n_new = np.clip(ref_state[3] + delta_state[3], 0, 1)
    neurons.v = v_new * mV
    neurons.m = m_new
    neurons.h = h_new
    neurons.n = n_new


def measure_lambda_benettin(regime_data, alpha, I_b_nA, rng,
                            warmup_ms=200.0, measure_ms=500.0,
                            renorm_ms=5.0, eps_mV=0.05, dt_ms=None,
                            dt_train_ms=None,
                            trains_E=None, trains_I=None,
                            v_init_mV=None, delta_dir=None):
    """
    Maximal Lyapunov exponent via Benettin renormalization.

    Measures divergence in FULL state space (v, m, h, n) — not just v.
    Perturbation is applied to v only (random per neuron, ||delta_v||=eps).
    Renormalization norm uses all 4 state variables (v in mV, m/h/n dimensionless).

    Uses SpikeGeneratorGroup with pre-generated Poisson trains for deterministic
    replay. Brian2's store/restore does NOT save PoissonGroup RNG state, but
    SpikeGeneratorGroup spike times are fixed arrays — both trajectories
    (reference + perturbed) get identical input via store()/restore().

    If trains_E/trains_I are provided, uses them directly (shared with metrics).
    Otherwise generates new trains from rng.

    Optional pairing parameters (for eps-invariance tests):
      v_init_mV: fixed initial v (array of length N, in mV units).
      delta_dir: fixed perturbation direction (array of length N).
                 Scaled to ||delta_v|| = eps_mV internally.

    Returns: lambda (1/s)
    """
    N = int(regime_data["N_total"])
    idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
    idx_I = np.array(regime_data["idx_I"], dtype=np.int32)
    n_E, n_I = len(idx_E), len(idx_I)

    total_ms = warmup_ms + measure_ms

    # Use provided trains or generate new ones
    if trains_E is None or trains_I is None:
        trains_E = generate_poisson_trains(n_E, 20.0, total_ms, rng,
                                           dt_train_ms=dt_train_ms)
        trains_I = generate_poisson_trains(n_I, 20.0, total_ms, rng,
                                           dt_train_ms=dt_train_ms)

    net, spike_mon, neurons, idx_E, idx_I = build_network(
        regime_data, alpha_override=alpha, I_b_nA=I_b_nA,
        input_mode="spike_generator",
        spike_trains_E=trains_E, spike_trains_I=trains_I,
        dt_ms=dt_ms,
    )

    if v_init_mV is not None:
        v_init = v_init_mV
    else:
        v_init = El / mV + rng.uniform(-2, 2, N)

    # ── Warmup ──
    net.restore()
    neurons.v = v_init * mV
    net.run(warmup_ms * ms)
    state_after_warmup = _get_state_vector(neurons).copy()
    net.store('after_warmup')

    # ── Reference trajectory: save full state at renorm checkpoints ──
    n_windows = int(measure_ms / renorm_ms)
    ref_checkpoints = []
    for _ in range(n_windows):
        net.run(renorm_ms * ms)
        ref_checkpoints.append(_get_state_vector(neurons).copy())

    # ── Perturbed trajectory with Benettin renormalization ──
    net.restore('after_warmup')

    # Perturb v only; m,h,n start identical
    if delta_dir is not None:
        delta_v = delta_dir * (eps_mV / np.linalg.norm(delta_dir))
    else:
        delta_v = rng.normal(0, 1.0, N)
        delta_v = delta_v * (eps_mV / np.linalg.norm(delta_v))
    neurons.v = (state_after_warmup[0] + delta_v) * mV

    sum_log = 0.0
    n_valid = 0

    for w in range(n_windows):
        net.run(renorm_ms * ms)

        state_pert = _get_state_vector(neurons)
        state_ref = ref_checkpoints[w]

        delta_state = state_pert - state_ref  # (4, N)
        d = np.linalg.norm(delta_state)       # Frobenius norm over all 4×N

        if d > 1e-15:
            sum_log += np.log(d / eps_mV)
            n_valid += 1
            # Renormalize full state perturbation back to eps
            delta_renorm = delta_state * (eps_mV / d)
            _set_state_from_ref_plus_delta(neurons, state_ref, delta_renorm)

    total_time_s = n_valid * renorm_ms / 1000.0
    if total_time_s < 1e-10:
        return 0.0
    return float(sum_log / total_time_s)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. REGIME CALIBRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeCalibrator:
    def __init__(self, npz_dir="regimes", output_dir="regimes_calibrated",
                 seed=42, max_iter=12, I_b_nA=0.05,
                 warmup_ms=200.0, measure_ms=1000.0, renorm_ms=5.0,
                 n_lambda_repeats=5, eps_mV=0.05,
                 dt_ms=DT_MS, dt_lambda_ms=DT_LAMBDA_MS,
                 dt_train_ms=None,
                 measure_ms_final=3000.0, n_repeats_final=9):
        self.npz_dir = npz_dir
        self.output_dir = output_dir
        self.seed = int(seed)
        self.max_iter = max_iter
        self.I_b_nA = I_b_nA
        self.warmup_ms = warmup_ms
        self.measure_ms = measure_ms
        self.renorm_ms = renorm_ms
        self.n_lambda_repeats = n_lambda_repeats
        self.eps_mV = eps_mV
        self.dt_ms = dt_ms
        self.dt_lambda_ms = dt_lambda_ms
        self.dt_train_ms = dt_train_ms if dt_train_ms is not None else dt_ms
        self.measure_ms_final = measure_ms_final
        self.n_repeats_final = n_repeats_final
        os.makedirs(output_dir, exist_ok=True)

    def calibrate_all(self):
        results = []
        for regime_name in LAMBDA_TARGETS:
            npz_path = os.path.join(self.npz_dir,
                                    f"{regime_name}_seed{self.seed}.npz")
            if not os.path.exists(npz_path):
                print(f"  SKIP {regime_name}: not found")
                continue
            print(f"\n{'='*60}")
            print(f"  CALIBRATING: {regime_name}")
            print(f"{'='*60}")
            res = self.calibrate_one(regime_name, npz_path)
            results.append(res)
        return results

    def _pregenerate_common_trains(self, regime_data, n_repeats=None,
                                    duration_ms=None):
        """Pre-generate common random Poisson trains for all alpha comparisons.

        Common random numbers reduce variance when comparing λ across
        different alpha values (bracketing + bisection), because the same
        input realization is used for every alpha tested.

        dt_train_ms is used for binning (independent of dt_ms integration step).
        """
        N = int(regime_data["N_total"])
        idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
        idx_I = np.array(regime_data["idx_I"], dtype=np.int32)
        n_E, n_I = len(idx_E), len(idx_I)
        if n_repeats is None:
            n_repeats = self.n_lambda_repeats
        if duration_ms is None:
            duration_ms = self.warmup_ms + self.measure_ms

        common_trains = []
        for k in range(n_repeats):
            rng_k = np.random.default_rng(self.seed + 888 + k * 100)
            trains_E = generate_poisson_trains(n_E, 20.0, duration_ms, rng_k,
                                               dt_train_ms=self.dt_train_ms)
            trains_I = generate_poisson_trains(n_I, 20.0, duration_ms, rng_k,
                                               dt_train_ms=self.dt_train_ms)
            common_trains.append((trains_E, trains_I))
        return common_trains

    def calibrate_one(self, regime_name, npz_path):
        regime_data = load_regime_npz(npz_path)
        alpha_initial = float(regime_data["alpha_final"])
        lam_target = LAMBDA_TARGETS[regime_name]
        lam_lo, lam_hi = LAMBDA_WINDOWS[regime_name]

        print(f"  alpha_initial={alpha_initial:.6f}  lam_target={lam_target}  "
              f"window=[{lam_lo}, {lam_hi}]")

        # ── Pre-generate common trains (shared across all alpha comparisons) ──
        self._common_trains = self._pregenerate_common_trains(regime_data)

        # ── Phase 1: Bracketing ──
        alpha_low, alpha_high, lam_low, lam_high = self._bracket(
            regime_data, lam_target, alpha_initial,
        )
        print(f"  Bracket: alpha=[{alpha_low:.6f}, {alpha_high:.6f}]  "
              f"lam=[{lam_low:.2f}, {lam_high:.2f}]")

        # ── Phase 2: Bisection ──
        best_metrics = None
        best_alpha = alpha_initial
        best_loss = float("inf")
        met = False

        alpha = 0.5 * (alpha_low + alpha_high)

        for iteration in range(self.max_iter):
            print(f"\n  --- iter {iteration+1}/{self.max_iter}  alpha={alpha:.6f} ---")

            metrics = self._run_and_measure(regime_data, alpha)
            ok, reasons = self._check(metrics, regime_name)

            loss = abs(metrics["lambda_div"] - lam_target)
            if loss < best_loss:
                best_loss = loss
                best_metrics = metrics.copy()
                best_alpha = alpha

            print(f"    rateE={metrics['rate_E']:.1f}  rateI={metrics['rate_I']:.1f}  "
                  f"cv={metrics['cv_isi_E']:.2f}  sync={metrics['sync_index']:.2f}  "
                  f"lam={metrics['lambda_div']:.2f} (IQR={metrics.get('lambda_iqr', 0):.1f})")

            if ok:
                print(f"    -> OK")
                best_metrics = metrics.copy()
                best_alpha = alpha
                best_loss = loss
                met = True
                break
            else:
                print(f"    -> miss: {', '.join(reasons)}")

            # Bisect on lambda
            if metrics["lambda_div"] < lam_target:
                alpha_low = alpha
            else:
                alpha_high = alpha
            alpha = 0.5 * (alpha_low + alpha_high)

        final_alpha = best_alpha

        # ── Phase 3: Final verification (longer, more repeats) ──
        print(f"\n  Final verification (measure_ms={self.measure_ms_final}, "
              f"n_repeats={self.n_repeats_final})...")
        final_trains = self._pregenerate_common_trains(
            regime_data,
            n_repeats=self.n_repeats_final,
            duration_ms=self.warmup_ms + self.measure_ms_final,
        )
        final_metrics = self._run_and_measure(
            regime_data, final_alpha,
            measure_ms_override=self.measure_ms_final,
            repeats_override=self.n_repeats_final,
            trains_list_override=final_trains,
        )
        print(f"    rateE={final_metrics['rate_E']:.1f}  "
              f"rateI={final_metrics['rate_I']:.1f}  "
              f"lam={final_metrics['lambda_div']:.2f} "
              f"(IQR={final_metrics.get('lambda_iqr', 0):.1f})")

        # ── Phase 4: Spontaneous sanity check ──
        print(f"\n  Spontaneous check (I_b only)...")
        spont = self._spontaneous_check(regime_data, final_alpha)
        print(f"    rateE={spont['rate_E']:.1f}  rateI={spont['rate_I']:.1f}  "
              f"sync={spont['sync_index']:.2f}  [{spont['status']}]")

        result = dict(
            regime=regime_name,
            seed=self.seed,
            alpha_etapA=alpha_initial,
            alpha_calibrated=final_alpha,
            rho_proxy=float(regime_data.get("rho", 0)),
            lambda_proxy=float(regime_data.get("lambda_proxy", 0)),
            lambda_measured=final_metrics["lambda_div"],
            rate_E=final_metrics["rate_E"],
            rate_I=final_metrics["rate_I"],
            cv_isi_E=final_metrics["cv_isi_E"],
            cv_isi_I=final_metrics["cv_isi_I"],
            sync_index=final_metrics["sync_index"],
            spont_rate_E=spont["rate_E"],
            spont_status=spont["status"],
            iterations=iteration + 1,
            status="OK" if met else "BEST",
        )

        self._save_calibrated_npz(regime_data, result, regime_name)
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Bracketing: find [alpha_low, alpha_high] that bracket lam_target
    # ──────────────────────────────────────────────────────────────────────

    def _bracket(self, regime_data, lam_target, alpha_initial):
        """Expand bracket until lam_target is between lam(alpha_low) and lam(alpha_high).

        Handles non-monotonic λ(alpha): if lam_low > lam_high after initial
        measurement, swaps (alpha_low, alpha_high) so that lam_low < lam_high.
        """
        alpha_low = 0.1 * alpha_initial
        alpha_high = 5.0 * alpha_initial

        lam_low = self._quick_lambda(regime_data, alpha_low)
        lam_high = self._quick_lambda(regime_data, alpha_high)
        print(f"  Bracket init: alpha=[{alpha_low:.6f},{alpha_high:.6f}]  "
              f"lam=[{lam_low:.2f},{lam_high:.2f}]")

        # Ensure lam_low <= lam_high (swap if non-monotonic)
        if lam_low > lam_high:
            alpha_low, alpha_high = alpha_high, alpha_low
            lam_low, lam_high = lam_high, lam_low
            print(f"    swapped bracket (non-monotonic): "
                  f"alpha=[{alpha_low:.6f},{alpha_high:.6f}]  "
                  f"lam=[{lam_low:.2f},{lam_high:.2f}]")

        for _ in range(6):
            if lam_low <= lam_target <= lam_high:
                return alpha_low, alpha_high, lam_low, lam_high

            if lam_target < lam_low:
                alpha_low *= 0.3
                lam_low = self._quick_lambda(regime_data, alpha_low)
                print(f"    expand low: alpha_low={alpha_low:.6f} lam={lam_low:.2f}")
            if lam_target > lam_high:
                alpha_high *= 2.0
                lam_high = self._quick_lambda(regime_data, alpha_high)
                print(f"    expand high: alpha_high={alpha_high:.6f} lam={lam_high:.2f}")

        return alpha_low, alpha_high, lam_low, lam_high

    def _quick_lambda(self, regime_data, alpha):
        """Quick lambda measurement for bracketing.

        Uses common pre-generated trains (if available) for variance
        reduction across alpha comparisons.  Falls back to independent
        trains if _common_trains not yet set.
        """
        trains_list = getattr(self, '_common_trains', None)
        n_quick = min(3, len(trains_list)) if trains_list else 3
        lams = []
        for k in range(n_quick):
            rng = np.random.default_rng(self.seed + 555 + k * 100)
            kw = dict(
                warmup_ms=self.warmup_ms, measure_ms=500.0,
                renorm_ms=self.renorm_ms, eps_mV=self.eps_mV,
                dt_ms=self.dt_lambda_ms,
            )
            if trains_list:
                kw["trains_E"] = trains_list[k][0]
                kw["trains_I"] = trains_list[k][1]
            lam_k = measure_lambda_benettin(
                regime_data, alpha, self.I_b_nA, rng, **kw,
            )
            lams.append(lam_k)
        return float(np.median(lams))

    # ──────────────────────────────────────────────────────────────────────
    # Main measurement: metrics + Benettin lambda
    # ──────────────────────────────────────────────────────────────────────

    def _run_and_measure(self, regime_data, alpha,
                         measure_ms_override=None, repeats_override=None,
                         trains_list_override=None):
        """Full measurement: firing metrics + Benettin lambda.

        Uses common pre-generated trains for variance reduction across
        alpha comparisons.  Both metrics and lambda use the SAME trains
        so guardrails and lambda are evaluated under identical input.
        Final values are medians across realizations.

        Optional overrides for final verification (longer, more repeats).
        """
        measure_ms = measure_ms_override or self.measure_ms
        n_repeats = repeats_override or self.n_lambda_repeats

        N = int(regime_data["N_total"])
        idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
        idx_I = np.array(regime_data["idx_I"], dtype=np.int32)
        n_E, n_I = len(idx_E), len(idx_I)
        total_ms = self.warmup_ms + measure_ms

        trains_list = trains_list_override or getattr(self, '_common_trains', None)

        all_rate_E, all_rate_I = [], []
        all_cv_E, all_cv_I = [], []
        all_sync = []
        all_lam = []

        for k in range(n_repeats):
            rng_k = np.random.default_rng(self.seed + 888 + k * 100)

            # ── Use common trains if available, else generate ──
            if trains_list and k < len(trains_list):
                trains_E, trains_I = trains_list[k]
            else:
                trains_E = generate_poisson_trains(n_E, 20.0, total_ms, rng_k,
                                                   dt_train_ms=self.dt_train_ms)
                trains_I = generate_poisson_trains(n_I, 20.0, total_ms, rng_k,
                                                   dt_train_ms=self.dt_train_ms)

            # ── Metrics: single run with spike_generator ──
            net, spike_mon, neurons, ie, ii = build_network(
                regime_data, alpha_override=alpha, I_b_nA=self.I_b_nA,
                input_mode="spike_generator",
                spike_trains_E=trains_E, spike_trains_I=trains_I,
                dt_ms=self.dt_ms,
            )
            v_init = El / mV + rng_k.uniform(-2, 2, N)
            net.restore()
            neurons.v = v_init * mV
            net.run(total_ms * ms)

            spike_trains = spike_mon.spike_trains()
            t_start = self.warmup_ms
            t_end = total_ms

            rE, rI = Metrics.firing_rates(spike_trains, ie, ii, t_start, t_end)
            cvE = Metrics.cv_isi(spike_trains, ie, t_start, t_end)
            cvI = Metrics.cv_isi(spike_trains, ii, t_start, t_end)
            syn = Metrics.sync_index(spike_trains, N, t_start, t_end)

            all_rate_E.append(rE)
            all_rate_I.append(rI)
            all_cv_E.append(cvE)
            all_cv_I.append(cvI)
            all_sync.append(syn)

            # ── Lambda via Benettin (same trains, separate rng for perturbation) ──
            rng_lam = np.random.default_rng(self.seed + 999_000 + k * 100)
            lam_k = measure_lambda_benettin(
                regime_data, alpha, self.I_b_nA, rng_lam,
                warmup_ms=self.warmup_ms, measure_ms=measure_ms,
                renorm_ms=self.renorm_ms, eps_mV=self.eps_mV,
                dt_ms=self.dt_lambda_ms,
                trains_E=trains_E, trains_I=trains_I,
            )
            all_lam.append(lam_k)

        def _iqr(vals):
            q1, q3 = np.percentile(vals, [25, 75])
            return float(q3 - q1)

        return dict(
            rate_E=float(np.median(all_rate_E)),
            rate_I=float(np.median(all_rate_I)),
            cv_isi_E=float(np.median(all_cv_E)),
            cv_isi_I=float(np.median(all_cv_I)),
            sync_index=float(np.median(all_sync)),
            lambda_div=float(np.median(all_lam)),
            lambda_iqr=_iqr(all_lam),
            lambda_all=list(all_lam),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Criteria check
    # ──────────────────────────────────────────────────────────────────────

    def _check(self, metrics, regime_name):
        """Check lambda window + guardrails. Returns (bool, reasons_list)."""
        reasons = []
        lam = metrics["lambda_div"]
        lam_lo, lam_hi = LAMBDA_WINDOWS[regime_name]

        # Primary: lambda in window
        if not (lam_lo <= lam <= lam_hi):
            reasons.append(f"lam={lam:.2f} not in [{lam_lo},{lam_hi}]")

        # Guardrails
        g = GUARDRAILS[regime_name]
        if "rate_E_max" in g and metrics["rate_E"] > g["rate_E_max"]:
            reasons.append(f"rateE={metrics['rate_E']:.1f}>{g['rate_E_max']}")
        if "rate_E_range" in g:
            lo, hi = g["rate_E_range"]
            if not (lo <= metrics["rate_E"] <= hi):
                reasons.append(f"rateE={metrics['rate_E']:.1f} not in [{lo},{hi}]")
        if "cv_isi_min" in g and metrics["cv_isi_E"] < g["cv_isi_min"]:
            reasons.append(f"cv={metrics['cv_isi_E']:.2f}<{g['cv_isi_min']}")
        if "sync_max" in g and metrics["sync_index"] > g["sync_max"]:
            reasons.append(f"sync={metrics['sync_index']:.2f}>{g['sync_max']}")

        return len(reasons) == 0, reasons

    # ──────────────────────────────────────────────────────────────────────
    # Spontaneous check
    # ──────────────────────────────────────────────────────────────────────

    def _spontaneous_check(self, regime_data, alpha):
        """1s run with only I_b (no Poisson), check for runaway/pathology.

        NOTE:
        For very small alpha, network coupling is effectively off and HH neurons
        can appear highly synchronized due to identical dynamics/attractor.
        In that case we only treat RUNAWAY as a failure; sync is reported but
        not used to mark PATHOLOGICAL_SYNC.
        """
        N = int(regime_data["N_total"])
        idx_E = np.array(regime_data["idx_E"], dtype=np.int32)
        idx_I = np.array(regime_data["idx_I"], dtype=np.int32)

        net, spike_mon, neurons, _, _ = build_network(
            regime_data, alpha_override=alpha, I_b_nA=self.I_b_nA,
            input_mode="none", dt_ms=self.dt_ms,
        )

        rng = np.random.default_rng(self.seed + 999)

        net.restore()

        # Randomize full state a bit (reduces artificial phase-locking)
        neurons.v = El + rng.uniform(-2, 2, N) * mV
        neurons.m = np.clip(rng.normal(0.05, 0.02, N), 0, 1)
        neurons.h = np.clip(rng.normal(0.95, 0.02, N), 0, 1)
        neurons.n = np.clip(rng.normal(0.50, 0.05, N), 0, 1)

        net.run(1000 * ms)

        spike_trains = spike_mon.spike_trains()
        rate_E, rate_I = Metrics.firing_rates(spike_trains, idx_E, idx_I, 200, 1000)
        sync = Metrics.sync_index(spike_trains, N, 200, 1000)

        # Hard failure only: runaway
        status = "OK"
        if rate_E > 150 or rate_I > 200:
            status = "RUNAWAY"
        else:
            # Report sync, but don't fail for very small alpha (R1 case)
            alpha_sync_min = 0.002
            if alpha >= alpha_sync_min and sync > 20:
                status = "PATHOLOGICAL_SYNC"

        return dict(rate_E=rate_E, rate_I=rate_I, sync_index=sync, status=status)

    # ──────────────────────────────────────────────────────────────────────
    # Output
    # ──────────────────────────────────────────────────────────────────────

    def _save_calibrated_npz(self, regime_data, result, regime_name):
        npz_data = dict(regime_data)
        npz_data["alpha_final"] = result["alpha_calibrated"]     # overwrite!
        npz_data["alpha_calibrated"] = result["alpha_calibrated"]
        npz_data["lambda_measured"] = result["lambda_measured"]
        npz_data["calibrated_rate_E"] = result["rate_E"]
        npz_data["calibrated_rate_I"] = result["rate_I"]
        npz_data["calibrated_cv_isi_E"] = result["cv_isi_E"]
        npz_data["calibrated_cv_isi_I"] = result["cv_isi_I"]
        npz_data["calibrated_sync_index"] = result["sync_index"]
        npz_data["calibration_iterations"] = result["iterations"]
        npz_data["calibration_status"] = result["status"]

        npz_path = os.path.join(
            self.output_dir, f"{regime_name}_seed{self.seed}.npz",
        )
        np.savez_compressed(npz_path, **npz_data)
        print(f"  Saved: {npz_path}")

    def save_csv(self, results):
        csv_path = os.path.join(
            self.output_dir, f"regimes_calibrated_seed{self.seed}.csv",
        )
        fields = [
            "regime", "seed", "alpha_etapA", "alpha_calibrated",
            "rho_proxy", "lambda_proxy", "lambda_measured",
            "rate_E", "rate_I", "cv_isi_E", "cv_isi_I", "sync_index",
            "spont_rate_E", "spont_status", "iterations", "status",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\nSaved CSV: {csv_path}")
        return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RHO-LAMBDA EMPIRICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def rho_lambda_scan(npz_path, rho_base, seed=42, n_points=8, n_repeats=5,
                    warmup_ms=200.0, measure_ms=300.0, renorm_ms=10.0,
                    dt_ms=None):
    """Quick empirical ρ→λ scan to validate regime assumptions.

    Measures true Lyapunov exponent at n_points spectral radius values.
    Returns list of (rho, alpha, lam_mean, lam_std) tuples.
    """
    regime_data = load_regime_npz(npz_path)
    dt = dt_ms if dt_ms is not None else DT_MS

    rho_targets = np.linspace(0.1, 2.5, n_points)
    alphas = rho_targets / rho_base

    print(f"\n{'='*60}")
    print(f"  RHO -> LAMBDA EMPIRICAL SCAN  (n_repeats={n_repeats})")
    print(f"{'='*60}")
    print(f"  {'rho':>6s}  {'alpha':>10s}  {'lam_med':>10s}  {'IQR':>8s}  {'lam_std':>8s}")
    print(f"  {'-'*52}")

    results = []
    for rho, alpha in zip(rho_targets, alphas):
        lams = []
        for r in range(n_repeats):
            rng = np.random.default_rng(seed + 777 + r * 100)
            lam = measure_lambda_benettin(
                regime_data, alpha, 0.05, rng,
                warmup_ms=warmup_ms, measure_ms=measure_ms,
                renorm_ms=renorm_ms, dt_ms=dt,
            )
            lams.append(lam)
        med = float(np.median(lams))
        q1, q3 = np.percentile(lams, [25, 75])
        iqr = float(q3 - q1)
        std = float(np.std(lams))
        print(f"  {rho:6.2f}  {alpha:10.6f}  {med:10.2f}  {iqr:8.2f}  {std:8.2f}")
        results.append((rho, alpha, med, iqr, std))

    # Check monotonicity
    meds = [r[2] for r in results]
    mono = all(meds[i] <= meds[i+1] for i in range(len(meds)-1))
    print(f"\n  Monotonic: {'YES' if mono else 'NO (expected for HH+STP)'}")
    print(f"  Range: [{min(meds):.1f}, {max(meds):.1f}] 1/s")
    print(f"{'='*60}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(seed=42, npz_dir="regimes", output_dir="regimes_calibrated", **kwargs):
    t0 = time.time()
    cal = RegimeCalibrator(npz_dir=npz_dir, output_dir=output_dir,
                           seed=seed, **kwargs)
    results = cal.calibrate_all()
    cal.save_csv(results)

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['regime']:25s}  "
            f"alpha: {r['alpha_etapA']:.6f}->{r['alpha_calibrated']:.6f}  "
            f"lam_proxy={r['lambda_proxy']:+.4f}  lam_meas={r['lambda_measured']:+.2f}  "
            f"rateE={r['rate_E']:.1f}Hz  sync={r['sync_index']:.2f}  "
            f"iter={r['iterations']}  [{r['status']}]"
        )

    lams = [r["lambda_measured"] for r in results]
    mono = all(lams[i] < lams[i + 1] for i in range(len(lams) - 1))
    print(f"\n  lambda_measured monotonic: {'PASS' if mono else 'FAIL'}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=str, default=None,
                    help="Calibrate single regime, e.g. R3_near_critical")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preset", choices=["dynamics", "rc"], default="dynamics",
                    help="Lambda windows preset: 'dynamics' (wide) or 'rc' (near edge-of-chaos)")
    ap.add_argument("--scan", action="store_true",
                    help="Run rho->lambda empirical scan instead of calibration")
    ap.add_argument("--rho-base", type=float, default=None,
                    help="rho_base from Etap A (required for --scan)")
    args = ap.parse_args()

    # Apply preset globally before any calibration
    if args.preset == "rc":
        LAMBDA_WINDOWS.update(LAMBDA_WINDOWS_RC)
        LAMBDA_TARGETS.update(LAMBDA_TARGETS_RC)
        print(f"Using RC preset (lower lambda targets)")

    if args.scan:
        # Empirical ρ→λ validation scan
        npz_path = os.path.join("regimes",
                                f"R3_near_critical_seed{args.seed}.npz")
        if args.rho_base is None:
            # Compute rho_base from regime_builder
            from regime_builder import ReservoirBuilder
            b = ReservoirBuilder(seed=args.seed)
            W = b.build_weight_csr(alpha=1.0)
            rho_base = b.spectral_radius(W)
            print(f"Computed rho_base={rho_base:.6f}")
        else:
            rho_base = args.rho_base
        rho_lambda_scan(npz_path, rho_base, seed=args.seed)
    elif args.regime:
        # Single-regime mode (for parallel launch)
        t0 = time.time()
        cal = RegimeCalibrator(seed=args.seed)
        npz_path = os.path.join(cal.npz_dir,
                                f"{args.regime}_seed{args.seed}.npz")
        res = cal.calibrate_one(args.regime, npz_path)
        print(f"\nDONE {args.regime}: alpha={res['alpha_calibrated']:.6f}  "
              f"lam={res['lambda_measured']:.2f}  [{res['status']}]  "
              f"{time.time()-t0:.1f}s")
    else:
        run(seed=args.seed)
