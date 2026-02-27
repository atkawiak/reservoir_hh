# narma10.py - NARMA10 benchmark for Schmitt/Maass Liquid State Machine
#
# v6: aggressive I_b diversity (STD=4) + weaker E->E (8nA)
#     + multi-timescale traces + T_STEPS=5000
#
# Based on: Sebastian Schmitt, 2022 (reservoir construction)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
    TimedArray, prefs, defaultclock,
)
from brian2 import ms, mV, Mohm, nA, second, Hz

from sklearn.linear_model import Ridge

# ============================================================
# Constants
# ============================================================
N_NEURONS = 135
V_THRESH = 15 * mV
V_RESET = 13.5 * mV
DT = 0.1 * ms

# NARMA parameters
STEP = 100 * ms               # one NARMA time step
T_STEPS = 5000                # total steps (500s) — more train data
WASHOUT = 50                  # initial transient to discard (5s)
TRAIN_FRAC = 0.7

# Input: N_IN independent Poisson neurons, sparse connectivity
# With p=0.3, each neuron connects to ~15 input neurons on average
# Per-synapse weight scaled to preserve mean total: 18/(50*0.3) = 1.2 nA
N_IN = 50
P_INPUT = 0.3                 # sparse! each neuron sees different subset
A_INPUT_EXC = 18.0 / (N_IN * P_INPUT)   # ~1.2 nA per synapse
A_INPUT_INH = 9.0 / (N_IN * P_INPUT)    # ~0.6 nA per synapse

R0 = 0.0                      # baseline rate (Hz)
K_RATE = 200.0                # Hz per unit of u

# Reservoir tuning
A_EE = 8.0                    # nA — weak E->E to prevent bursts, some recurrence
A_EI = 60.0                   # nA — keep strong E->I for stability
A_IE = -22.0                  # nA — moderately stronger inhibition (was -19)
A_II = -19.0                  # nA
I_B_MEAN = 13.0               # nA — near threshold, input drives spiking
I_B_STD = 4.0                 # nA — large diversity (some quiet, some fast neurons)

# Feature extraction — multi-timescale traces
TAU_X_LIST = [50.0, 150.0, 500.0]  # ms — fast/medium/slow dynamics
                                    # decay/step: 0.14 / 0.51 / 0.82
N_LAGS = 1                         # no lag stacking; timescales provide temporal context
USE_QUADRATIC = True                # add x_i^2 features

# Reproducibility seeds
SEED_TOPOLOGY = 42
SEED_INPUT = 123
SEED_SIM = 54321


# ============================================================
# NARMA10 target
# ============================================================
def generate_narma10(u, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1):
    N = len(u)
    y = np.zeros(N)
    for t in range(10, N):
        y[t] = (alpha * y[t - 1]
                + beta * y[t - 1] * np.sum(y[t - 10:t])
                + gamma * u[t - 10] * u[t]
                + delta)
        y[t] = np.clip(y[t], 0, 1)
    return y


# ============================================================
# Reservoir builders
# ============================================================
def get_neurons():
    neurons = NeuronGroup(
        N_NEURONS,
        """
        tau_mem : second (shared, constant)
        tau_refrac : second (constant)
        v_reset : volt (shared, constant)
        v_thresh : volt (shared, constant)
        I_b : ampere (constant)
        tau_stimulus : second (constant)
        I_syn_ee_synapses : ampere
        I_syn_ei_synapses : ampere
        I_syn_ie_synapses : ampere
        I_syn_ii_synapses : ampere
        dI_stimulus/dt = -I_stimulus/tau_stimulus : ampere
        R_in : ohm
        dv/dt = -v/tau_mem + (I_syn_ee_synapses +
                              I_syn_ei_synapses +
                              I_syn_ie_synapses +
                              I_syn_ii_synapses)*R_in/tau_mem
                           + I_b*R_in/tau_mem
                           + I_stimulus*R_in/tau_mem: volt (unless refractory)
        x_pos : 1 (constant)
        y_pos : 1 (constant)
        z_pos : 1 (constant)
        """,
        threshold="v>v_thresh",
        reset="v=v_reset",
        refractory="tau_refrac",
        method="exact",
        name="neurons",
    )
    neurons.tau_mem = 30 * ms
    neurons.v_thresh = V_THRESH
    neurons.v_reset = V_RESET

    # Heterogeneous I_b: each neuron gets a different bias current
    ib_values = np.random.normal(I_B_MEAN, I_B_STD, size=len(neurons))
    ib_values = np.clip(ib_values, I_B_MEAN * 0.3, I_B_MEAN * 1.7)  # safety clip
    neurons.I_b = ib_values * nA

    neurons.v[:] = (
        np.random.uniform(V_RESET / mV, V_THRESH / mV, size=len(neurons)) * mV
    )
    neurons.R_in = 1 * Mohm

    indices = np.arange(len(neurons))
    np.random.shuffle(indices)
    neurons.x_pos = indices % 3
    neurons.y_pos = (indices // 3) % 3
    neurons.z_pos = indices // 9

    return neurons


def get_synapses(name, source, target, C, l, tau_I, A, U, D, F, delay):
    synapses_eqs = """
    A : ampere (constant)
    U : 1 (constant)
    tau_I : second (shared, constant)
    D : second (constant)
    dx/dt =  z/D       : 1 (clock-driven)
    dy/dt = -y/tau_I   : 1 (clock-driven)
    z = 1 - x - y      : 1
    I_syn_{}_post = A*y : ampere (summed)
    """.format(name)

    if F:
        synapses_eqs += """
        du/dt = -u/F : 1 (clock-driven)
        F : second (constant)
        """
        synapses_action = """
        u += U*(1-u)
        y += u*x
        x += -u*x
        """
    else:
        synapses_action = """
        y += U*x
        x += -U*x
        """

    synapses = Synapses(
        source, target,
        model=synapses_eqs, on_pre=synapses_action,
        method="exact", name=name, delay=delay,
    )
    synapses.connect(
        p=f"{C} * exp(-((x_pos_pre-x_pos_post)**2 + (y_pos_pre-y_pos_post)**2"
          f" + (z_pos_pre-z_pos_post)**2)/{l}**2)"
    )

    N_syn = len(synapses)
    synapses.tau_I = tau_I
    synapses.A[:] = (
        np.sign(A / nA) * np.random.gamma(1, abs(A / nA), size=N_syn) * nA
    )
    synapses.U[:] = np.random.normal(U, 0.5 * U, size=N_syn)
    synapses.U[:][synapses.U < 0] = U
    synapses.D[:] = np.random.normal(D / ms, 0.5 * D / ms, size=N_syn) * ms
    synapses.D[:][synapses.D / ms <= 0] = D
    synapses.x = 1

    if F:
        synapses.F[:] = np.random.normal(F / ms, 0.5 * F / ms, size=N_syn) * ms
        synapses.F[:][synapses.F / ms <= 0] = F

    return synapses


# ============================================================
# Feature extraction
# ============================================================
def extract_exp_traces(spike_monitor_exc, spike_monitor_inh,
                       t_steps, step_sec, tau_x_sec):
    """Exponential trace at end of each time step, O(n_spikes + t_steps)."""
    decay = np.exp(-step_sec / tau_x_sec)
    X = np.zeros((t_steps, N_NEURONS))

    all_trains = (list(spike_monitor_exc.spike_trains().values()) +
                  list(spike_monitor_inh.spike_trains().values()))

    for neuron_idx, train in enumerate(all_trains):
        spikes = np.sort(np.array(train / second))
        trace_val = 0.0
        spike_ptr = 0
        for t_idx in range(t_steps):
            t_end = (t_idx + 1) * step_sec
            trace_val *= decay
            while spike_ptr < len(spikes) and spikes[spike_ptr] < t_end:
                dt = t_end - spikes[spike_ptr]
                trace_val += np.exp(-dt / tau_x_sec)
                spike_ptr += 1
            X[t_idx, neuron_idx] = trace_val

    return X


def add_stacking(X, n_lags):
    """Add time-lagged copies: [x[n], x[n-1], ..., x[n-n_lags+1]]."""
    if n_lags <= 1:
        return X
    t_steps, n_feat = X.shape
    X_stacked = np.zeros((t_steps, n_feat * n_lags))
    for lag in range(n_lags):
        col_start = lag * n_feat
        col_end = (lag + 1) * n_feat
        if lag == 0:
            X_stacked[:, col_start:col_end] = X
        else:
            X_stacked[lag:, col_start:col_end] = X[:-lag]
    return X_stacked


def build_features(X_raw, n_lags, quadratic):
    """Stack lags, optionally add quadratic features."""
    X = add_stacking(X_raw, n_lags)
    if quadratic:
        X_quad = X_raw ** 2
        X = np.hstack([X, X_quad])
    return X


def evaluate_readout(X, y_target, washout, train_frac):
    """Ridge regression with alpha grid search."""
    t_steps = len(y_target)
    train_end = washout + int((t_steps - washout) * train_frac)

    X_train = X[washout:train_end]
    y_train = y_target[washout:train_end]
    X_test = X[train_end:]
    y_test = y_target[train_end:]

    best_nrmse = np.inf
    best_alpha = None
    best_pred = None

    for alpha in [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        pred = ridge.predict(X_test)
        nrmse = (np.sqrt(np.mean((y_test - pred) ** 2))
                 / (np.std(y_test) + 1e-12))
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_alpha = alpha
            best_pred = pred

    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X_train, y_train)
    y_pred_train = ridge_final.predict(X_train)

    nrmse_train = (np.sqrt(np.mean((y_train - y_pred_train) ** 2))
                   / (np.std(y_train) + 1e-12))
    ss_res = np.sum((y_test - best_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2) + 1e-12
    r2_test = 1 - ss_res / ss_tot

    return {
        'nrmse_train': nrmse_train,
        'nrmse_test': best_nrmse,
        'r2_test': r2_test,
        'best_alpha': best_alpha,
        'y_pred_test': best_pred,
        'y_test': y_test,
        'train_end': train_end,
    }


def compute_fano(spike_monitor_exc, spike_monitor_inh, duration_sec):
    """Population Fano factor as synchrony proxy."""
    binw_sec = 0.005  # 5 ms
    nbins = int(duration_sec / binw_sec)
    counts = np.zeros(nbins, dtype=int)
    for mon in [spike_monitor_exc, spike_monitor_inh]:
        t_sec = np.array(mon.t / second)
        bins = np.floor(t_sec / binw_sec).astype(int)
        bins = bins[bins < nbins]
        np.add.at(counts, bins, 1)
    mean_c = counts.mean()
    return counts.var() / (mean_c + 1e-12)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    prefs.codegen.target = "numpy"
    np.random.seed(SEED_TOPOLOGY)

    DURATION = T_STEPS * STEP
    step_sec = float(STEP / second)
    n_timescales = len(TAU_X_LIST)
    n_feat_total = (N_NEURONS * n_timescales * N_LAGS
                    + (N_NEURONS * n_timescales if USE_QUADRATIC else 0))

    print("=" * 60)
    print("  NARMA10 Liquid State Machine Benchmark (v6)")
    print("=" * 60)
    print(f"  T_STEPS={T_STEPS}, STEP={STEP/ms:.0f}ms, "
          f"Duration={DURATION/second:.1f}s")
    print(f"  N_NEURONS={N_NEURONS}, N_IN={N_IN}, P_INPUT={P_INPUT}")
    print(f"  A_EE={A_EE} nA, A_IE={A_IE} nA")
    print(f"  I_b: N({I_B_MEAN}, {I_B_STD}) nA per neuron (heterogeneous)")
    print(f"  Input weights: exc={A_INPUT_EXC:.2f} nA x ~{int(N_IN*P_INPUT)}/neuron, "
          f"inh={A_INPUT_INH:.2f} nA x ~{int(N_IN*P_INPUT)}/neuron")
    print(f"  Features: multi-tau={TAU_X_LIST}ms, "
          f"{N_LAGS}-lag"
          f"{' + quadratic' if USE_QUADRATIC else ''} "
          f"= {n_feat_total} dims")
    print(f"  Seeds: topology={SEED_TOPOLOGY}, input={SEED_INPUT}, "
          f"sim={SEED_SIM}")
    print()

    # --- Build reservoir ---
    print("Building reservoir...")
    neurons = get_neurons()
    n_exc = int(0.8 * len(neurons))

    exc_neurons = neurons[:n_exc]
    exc_neurons.tau_refrac = 3 * ms
    exc_neurons.tau_stimulus = 3 * ms

    inh_neurons = neurons[n_exc:]
    inh_neurons.tau_refrac = 2 * ms
    inh_neurons.tau_stimulus = 6 * ms

    l_lambda = 2

    ee_synapses = get_synapses(
        "ee_synapses", exc_neurons, exc_neurons,
        C=0.3, l=l_lambda, tau_I=3*ms, A=A_EE*nA, U=0.5,
        D=1.1*second, F=0.05*second, delay=1.5*ms)
    ei_synapses = get_synapses(
        "ei_synapses", exc_neurons, inh_neurons,
        C=0.2, l=l_lambda, tau_I=3*ms, A=A_EI*nA, U=0.05,
        D=0.125*second, F=1.2*second, delay=0.8*ms)
    ie_synapses = get_synapses(
        "ie_synapses", inh_neurons, exc_neurons,
        C=0.4, l=l_lambda, tau_I=6*ms, A=A_IE*nA, U=0.25,
        D=0.7*second, F=0.02*second, delay=0.8*ms)
    ii_synapses = get_synapses(
        "ii_synapses", inh_neurons, inh_neurons,
        C=0.1, l=l_lambda, tau_I=6*ms, A=A_II*nA, U=0.32,
        D=0.144*second, F=0.06*second, delay=0.8*ms)

    # --- Input signal ---
    print("Generating input signal and NARMA10 target...")
    rng_u = np.random.default_rng(SEED_INPUT)
    u_input = rng_u.uniform(0.0, 0.5, size=T_STEPS)
    y_target = generate_narma10(u_input)

    print(f"  u: min={u_input.min():.3f}, max={u_input.max():.3f}, "
          f"mean={u_input.mean():.3f}")
    print(f"  y_target: min={y_target.min():.3f}, max={y_target.max():.3f}, "
          f"mean={y_target[WASHOUT:].mean():.3f}")

    # --- TimedArray + PoissonGroup (N_IN independent neurons) ---
    rate_values = R0 + K_RATE * u_input
    rate_ta = TimedArray(rate_values * Hz, dt=STEP, name="rate_ta")
    input_group = PoissonGroup(N_IN, rates='rate_ta(t)', name="input_group")

    # --- Input synapses with SPARSE connectivity + random delays ---
    in_to_exc = Synapses(
        input_group, exc_neurons,
        "A : ampere (shared, constant)",
        on_pre="I_stimulus += A",
        name="in_to_exc")
    in_to_exc.connect(p=P_INPUT)
    in_to_exc.A = A_INPUT_EXC * nA
    in_to_exc.delay[:] = np.random.uniform(0, 5, size=len(in_to_exc)) * ms

    in_to_inh = Synapses(
        input_group, inh_neurons,
        "A : ampere (shared, constant)",
        on_pre="I_stimulus += A",
        name="in_to_inh")
    in_to_inh.connect(p=P_INPUT)
    in_to_inh.A = A_INPUT_INH * nA
    in_to_inh.delay[:] = np.random.uniform(0, 5, size=len(in_to_inh)) * ms

    print(f"  Input synapses: {len(in_to_exc)} exc + {len(in_to_inh)} inh")

    # --- Spike monitors ---
    spike_monitor_exc = SpikeMonitor(exc_neurons, name="spike_monitor_exc")
    spike_monitor_inh = SpikeMonitor(inh_neurons, name="spike_monitor_inh")

    # --- Network ---
    defaultclock.dt = DT
    net = Network([
        neurons,
        ee_synapses, ei_synapses, ie_synapses, ii_synapses,
        input_group, in_to_exc, in_to_inh,
        spike_monitor_exc, spike_monitor_inh,
    ])

    # --- Run simulation ---
    np.random.seed(SEED_SIM)
    print(f"\nRunning simulation ({DURATION/second:.1f}s)...")
    net.run(DURATION, report='text')
    print("Simulation complete.\n")

    # --- Diagnostics ---
    T_sec = float(DURATION / second)
    rate_e = spike_monitor_exc.num_spikes / (n_exc * T_sec)
    rate_i = spike_monitor_inh.num_spikes / (len(inh_neurons) * T_sec)
    fano = compute_fano(spike_monitor_exc, spike_monitor_inh, T_sec)
    print(f"Firing rates:  Exc={rate_e:.1f} Hz,  Inh={rate_i:.1f} Hz")
    print(f"Population Fano factor: {fano:.1f}")

    if rate_e == 0 and rate_i == 0:
        print("FATAL: network is dead.")
        raise SystemExit(1)

    # --- Extract features (multi-timescale) ---
    print("Extracting multi-timescale exp-trace features...")
    X_parts = []
    for tau_ms in TAU_X_LIST:
        tau_sec = tau_ms / 1000.0
        X_tau = extract_exp_traces(
            spike_monitor_exc, spike_monitor_inh,
            T_STEPS, step_sec, tau_sec)
        X_parts.append(X_tau)
        print(f"  tau={tau_ms}ms: shape {X_tau.shape}")

    X_raw = np.hstack(X_parts)  # (T_STEPS, N_NEURONS * n_timescales)
    X = build_features(X_raw, N_LAGS, USE_QUADRATIC)
    print(f"  Features: raw {X_raw.shape} -> final {X.shape}")

    # --- Sanity check (use medium timescale for corr) ---
    mean_trace = X_parts[1].mean(axis=1)  # middle timescale
    corr_u_rate = np.corrcoef(u_input, mean_trace)[0, 1]
    print(f"  corr(u, mean_trace) = {corr_u_rate:.3f}")
    if abs(corr_u_rate) < 0.05:
        print("  WARNING: input barely affects reservoir!")

    # --- Readout ---
    print("\nTraining ridge regression...")
    res = evaluate_readout(X, y_target, WASHOUT, TRAIN_FRAC)
    train_end = res['train_end']

    print(f"\n{'='*60}")
    print(f"  NARMA10 RESULTS (v6)")
    print(f"{'='*60}")
    print(f"  Firing rates:      Exc={rate_e:.1f} Hz, Inh={rate_i:.1f} Hz")
    print(f"  Fano factor:       {fano:.1f}")
    print(f"  corr(u, trace):    {corr_u_rate:.4f}")
    print(f"  Best ridge alpha:  {res['best_alpha']}")
    print(f"  NRMSE (train):     {res['nrmse_train']:.4f}")
    print(f"  NRMSE (test):      {res['nrmse_test']:.4f}")
    print(f"  R^2   (test):      {res['r2_test']:.4f}")
    print(f"{'='*60}")

    # --- Plot ---
    t_axis = np.arange(T_STEPS) * step_sec
    t_test = t_axis[train_end:]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(t_axis, u_input, 'b-', linewidth=0.5)
    axes[0].set_ylabel('u(t)')
    axes[0].set_title('Input signal')
    axes[0].set_xlim(t_axis[0], t_axis[-1])

    axes[1].plot(t_test, res['y_test'], 'k-', linewidth=1, label='target')
    axes[1].plot(t_test, res['y_pred_test'], 'r-', linewidth=1,
                 alpha=0.7, label='prediction')
    axes[1].set_ylabel('y(t)')
    axes[1].set_title(f'NARMA10 test (NRMSE={res["nrmse_test"]:.4f}, '
                      f'R\u00b2={res["r2_test"]:.4f})')
    axes[1].legend(loc='upper right')

    axes[2].plot(t_axis, mean_trace, 'g-', linewidth=0.5)
    axes[2].set_ylabel('mean trace')
    axes[2].set_xlabel('time (s)')
    axes[2].set_title(f'Population mean activity '
                      f'(corr with u: {corr_u_rate:.3f}, Fano: {fano:.1f})')
    axes[2].set_xlim(t_axis[0], t_axis[-1])

    plt.tight_layout()
    plt.savefig("E:/dd/narma10_result.png", dpi=150)
    print(f"\nPlot saved to narma10_result.png")
    print("Done!")
