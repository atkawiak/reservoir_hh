"""
Reservoir computing benchmark (Schmitt separation test)
Hodgkin-Huxley neurons + Short-Term Plasticity (STP) synapses

Based on: Maass, Natschlaeger, Markram (2002)

NOTE: HH is ~10-100x slower than LIF. Consider reducing N_PAIRS for
initial testing. Synapse amplitudes (A) are scaled down ~100x vs LIF
because HH membrane is much more sensitive to current injection
(effective input resistance ~100 MOhm vs LIF's 1 MOhm).

Tuning strategy:
  1. First tune I_b so that mean firing rate is 5-15 Hz
  2. Then sweep alpha (synapse scale) for stable -> edge -> chaos
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeGeneratorGroup,
    SpikeMonitor,
    Network,
    Equations,
    defaultclock,
)
from brian2 import ms, mV, nA, second, Hz
from brian2 import umetre, ufarad, siemens, msiemens, cm

# ── Constants ─────────────────────────────────────────────────────────────────

N_NEURONS = 135

STIMULUS_POISSON_RATE = 20 * Hz
TARGET_DISTANCES = [0.4, 0.2, 0.1]
N_PAIRS = 200

DT = 0.1 * ms
DURATION = 500 * ms
TS = np.arange(0, DURATION / ms, DT / ms)


# ── Helper functions ──────────────────────────────────────────────────────────

def exponential_convolution(t, spikes, tau):
    """Convolve spikes with exponential kernel."""
    if len(spikes):
        return sum(np.exp(-((t - st) / tau)) * (t >= st) for st in spikes)
    return np.zeros(len(t))


def gaussian_convolution(t, spikes, tau):
    """Convolve spikes with Gaussian kernel."""
    if len(spikes):
        return sum(np.exp(-(((t - st) / tau) ** 2)) for st in spikes)
    return np.zeros(len(t))


def euclidian_distance(liquid_states_u, liquid_states_v):
    """Euclidian distance between liquid states (no sqrt, as in paper)."""
    return np.mean((liquid_states_u - liquid_states_v) ** 2, axis=0)


def distance(conv_a, conv_b, dt):
    """L2 distance between convolutions (no sqrt, as in paper)."""
    return sum((conv_a - conv_b) ** 2) * dt


def generate_poisson(duration, rate):
    """Generate Poisson spike train with no double-spikes per time bin."""
    while True:
        N = np.random.poisson(rate * duration)
        spikes = np.sort(np.random.uniform(0, duration, N))
        shift = 1e-3 * (DT / ms)
        timebins = ((spikes + shift) / (DT / ms)).astype(np.int32)
        if not any(np.diff(timebins) == 0):
            return spikes


def collect_stimulus_pairs():
    """Collect pairs of Poisson stimuli at target distances."""
    DELTA_DISTANCE = 0.01
    collected = defaultdict(list)

    while True:
        su = generate_poisson(DURATION / ms, STIMULUS_POISSON_RATE / Hz / 1e3)
        sv = generate_poisson(DURATION / ms, STIMULUS_POISSON_RATE / Hz / 1e3)

        cu = gaussian_convolution(TS, su, tau=5)
        cv = gaussian_convolution(TS, sv, tau=5)

        nd = distance(cu, cv, DT / ms) / (DURATION / ms)

        for td in TARGET_DISTANCES:
            if abs(nd - td) < DELTA_DISTANCE and len(collected[td]) < N_PAIRS:
                collected[td].append((su, sv))

        if len(collected) == len(TARGET_DISTANCES) and all(
            len(v) == N_PAIRS for v in collected.values()
        ):
            break

    return collected


# ── HH biophysical constants (module-level for Brian2 namespace resolution) ──

_area = 20000 * umetre ** 2
Cm    = 1 * ufarad * cm ** -2 * _area
gl    = 5e-5 * siemens * cm ** -2 * _area
El    = -65 * mV
EK    = -90 * mV
ENa   = 50 * mV
g_na  = 100 * msiemens * cm ** -2 * _area
g_kd  = 30 * msiemens * cm ** -2 * _area
VT    = -63 * mV


# ── Neuron model (Hodgkin-Huxley) ────────────────────────────────────────────

def get_neurons():
    eqs = Equations('''
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

    neurons = NeuronGroup(
        N_NEURONS, eqs,
        threshold='v > -40*mV',
        refractory='v > -40*mV',
        method='exponential_euler',
        name='neurons',
    )

    # Initial conditions
    neurons.v = El
    neurons.m = 0
    neurons.h = 1
    neurons.n = 0.5

    # Bias current (tuned: alpha=10x sweep gave exc~25Hz, inh~32Hz)
    neurons.I_b = 0.05 * nA
    neurons.tau_stimulus = 3 * ms
    neurons.I_stimulus = 0 * nA

    # Column topology (15 x 3 x 3)
    indices = np.arange(N_NEURONS)
    np.random.shuffle(indices)
    neurons.x_pos = indices % 3
    neurons.y_pos = (indices // 3) % 3
    neurons.z_pos = indices // 9

    return neurons


# ── Synapse model (STP) ──────────────────────────────────────────────────────

def get_synapses(name, source, target, C, l, tau_I, A, U, D, F, delay):
    eqs = """
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
        eqs += """
        du/dt = -u/F : 1 (clock-driven)
        F : second (constant)
        """
        action = """
        u += U*(1-u)
        y += u*x
        x += -u*x
        """
    else:
        action = """
        y += U*x
        x += -U*x
        """

    syn = Synapses(
        source, target,
        model=eqs,
        on_pre=action,
        method="exact",
        name=name,
        delay=delay,
    )
    syn.connect(
        p=f"{C} * exp(-((x_pos_pre-x_pos_post)**2"
          f" + (y_pos_pre-y_pos_post)**2"
          f" + (z_pos_pre-z_pos_post)**2)/{l}**2)"
    )

    N_syn = len(syn)
    syn.tau_I = tau_I

    syn.A[:] = np.sign(A / nA) * np.random.gamma(1, abs(A / nA), size=N_syn) * nA

    syn.U[:] = np.random.normal(U, 0.5 * U, size=N_syn)
    syn.U[:][syn.U < 0] = U

    syn.D[:] = np.random.normal(D / ms, 0.5 * D / ms, size=N_syn) * ms
    syn.D[:][syn.D / ms <= 0] = D

    syn.x = 1

    if F:
        syn.F[:] = np.random.normal(F / ms, 0.5 * F / ms, size=N_syn) * ms
        syn.F[:][syn.F / ms <= 0] = F

    return syn


# ── Simulation ────────────────────────────────────────────────────────────────

def sim(net, spike_times):
    """Run one trial: restore, set stimulus, run, extract liquid states."""
    net.restore()

    # Small random perturbation around resting potential
    net["neurons"].v = El + np.random.uniform(-2, 2, size=N_NEURONS) * mV

    net["stimulus"].set_spikes([0] * len(spike_times), spike_times * ms)
    net.run(DURATION)

    spikes = (
        list(net["spike_monitor_exc"].spike_trains().values())
        + list(net["spike_monitor_inh"].spike_trains().values())
    )
    return np.array(
        [exponential_convolution(TS, st / ms, tau=30) for st in spikes]
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    if not os.environ.get("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")

    neurons = get_neurons()

    N_exc = int(0.8 * N_NEURONS)
    exc = neurons[:N_exc]
    inh = neurons[N_exc:]

    # HH has natural refractory — only set stimulus decay constants
    exc.tau_stimulus = 3 * ms
    inh.tau_stimulus = 6 * ms

    L = 2

    # Synapse amplitudes: ~10x down vs LIF (tuned via parameter sweep)
    ee = get_synapses(
        "ee_synapses", exc, exc,
        C=0.3, l=L, tau_I=3 * ms,
        A=3.0 * nA, U=0.5, D=1.1 * second, F=0.05 * second, delay=1.5 * ms,
    )
    ei = get_synapses(
        "ei_synapses", exc, inh,
        C=0.2, l=L, tau_I=3 * ms,
        A=6.0 * nA, U=0.05, D=0.125 * second, F=1.2 * second, delay=0.8 * ms,
    )
    ie = get_synapses(
        "ie_synapses", inh, exc,
        C=0.4, l=L, tau_I=6 * ms,
        A=-1.9 * nA, U=0.25, D=0.7 * second, F=0.02 * second, delay=0.8 * ms,
    )
    ii = get_synapses(
        "ii_synapses", inh, inh,
        C=0.1, l=L, tau_I=6 * ms,
        A=-1.9 * nA, U=0.32, D=0.144 * second, F=0.06 * second, delay=0.8 * ms,
    )

    # Stimulus input
    stimulus = SpikeGeneratorGroup(1, [], [] * ms, name="stimulus")

    stim_syn_exc = Synapses(
        stimulus, exc,
        "A : ampere (shared, constant)",
        on_pre="I_stimulus += A",
    )
    stim_syn_exc.connect(p=1)
    stim_syn_exc.A = 1.5 * nA

    stim_syn_inh = Synapses(
        stimulus, inh,
        "A : ampere (shared, constant)",
        on_pre="I_stimulus += A",
    )
    stim_syn_inh.connect(p=1)
    stim_syn_inh.A = 0.75 * nA

    # Monitors
    sm_exc = SpikeMonitor(exc, name="spike_monitor_exc")
    sm_inh = SpikeMonitor(inh, name="spike_monitor_inh")

    # Network
    defaultclock.dt = DT

    net = Network([
        neurons,
        ee, ei, ie, ii,
        stim_syn_exc, stim_syn_inh,
        stimulus,
        sm_exc, sm_inh,
    ])
    net.store()

    # Collect stimulus pairs
    print("Collecting stimulus pairs...")
    collected_pairs = collect_stimulus_pairs()
    # d=0 pairs: identical stimuli
    collected_pairs[0] = [
        [generate_poisson(DURATION / ms, STIMULUS_POISSON_RATE / Hz / 1e3)] * 2
        for _ in range(N_PAIRS)
    ]

    # Run separation test
    result = defaultdict(list)
    for d, pairs in collected_pairs.items():
        print(f"  d={d}: running {len(pairs)} pairs...")
        for i, (su, sv) in enumerate(pairs):
            if (i + 1) % 20 == 0:
                print(f"    pair {i+1}/{len(pairs)}")
            lu = sim(net, su)
            lv = sim(net, sv)
            result[d].append(euclidian_distance(lu, lv))

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    linestyles = ["dashed", (0, (8, 6, 1, 6)), (0, (5, 10)), "solid"]

    for d, ls in zip(TARGET_DISTANCES + [0], linestyles):
        eds = np.array(result[d])
        ax.plot(
            TS / 1000, np.mean(eds, axis=0),
            label=f"d(u,v)={d}", linestyle=ls, color="k",
        )

    ax.set_xlabel("time [sec]")
    ax.set_ylabel("state distance")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 2.5)
    ax.set_title("HH + STP  (Schmitt separation)")
    ax.legend(loc="upper center", fontsize="x-large", frameon=False)

    plt.tight_layout()
    plt.savefig("hh_separation.png", dpi=150)
    print("Saved hh_separation.png")
    plt.show()
