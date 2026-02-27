# Sebastian Schmitt, 2022

from collections import defaultdict
import multiprocessing

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeGeneratorGroup,
    SpikeMonitor,
    Network,
    prefs,
)
from brian2 import ms, mV, Mohm, nA, second, Hz
from brian2 import defaultclock, prefs

N_NEURONS = 135
V_THRESH = 15 * mV
V_RESET = 13.5 * mV

STIMULUS_POISSON_RATE = 20 * Hz
TARGET_DISTANCES = [0.4, 0.2, 0.1]
N_PAIRS = 200

DT = 0.1 * ms
DURATION = 500 * ms
TS = np.arange(0, DURATION / ms, DT / ms)


def exponential_convolution(t, spikes, tau):
    """Convolute spikes with exponential kernel
    t -- numpy array of times to evaluate the convolution
    spikes -- iterable of spike times
    tau -- exponential decay constant
    """
    if len(spikes):
        return sum([np.exp(-((t - st) / tau)) * (t >= st) for st in spikes])
    else:
        return np.zeros(len(TS))


def gaussian_convolution(t, spikes, tau):
    """Convolute spikes with Gaussian kernel
    t -- numpy array of times to evaluate the convolution
    spikes -- iterable of spike times
    tau -- exponential decay constant
    """
    if len(spikes):
        return sum([np.exp(-(((t - st) / tau) ** 2)) for st in spikes])
    else:
        return np.zeros(len(TS))


def euclidian_distance(liquid_states_u, liquid_states_v):
    """Euclidian distance between liquid states
    liquid_states_u -- liquid states
    liquid_states_v -- other liquid states

    To match the numbers in the paper, the square root is omitted
    """

    return np.mean((liquid_states_u - liquid_states_v) ** 2, axis=0)


def distance(conv_a, conv_b, dt):
    """Difference of convolutions in the L2-norm
    conv_a -- convolutions
    conv_b -- other convolutions
    dt -- time step

    To match the numbers in the paper, the square root is omitted
    """

    return sum((conv_a - conv_b) ** 2) * dt


def generate_poisson(duration, rate):
    """Generate Poisson spike train
    duration -- duration of spike train
    rate -- rate of spike train

    Return only spike trains that do not have multiple spikes per time bin
    """
    while True:
        N = np.random.poisson(rate * duration)
        spikes = np.random.uniform(0, duration, N)

        spikes_orig = np.sort(spikes)
        shift = 1e-3 * (DT / ms)
        timebins = ((spikes_orig + shift) / (DT / ms)).astype(np.int32)

        if not any(np.diff(timebins) == 0):
            return spikes_orig


def collect_stimulus_pairs():
    """Collect pairs of input stimuli close in target distance"""
    DELTA_DISTANCE = 0.01
    collected_pairs = defaultdict(list)

    while True:

        spikes_u = generate_poisson(DURATION / ms, STIMULUS_POISSON_RATE / Hz / 1e3)
        spikes_v = generate_poisson(DURATION / ms, STIMULUS_POISSON_RATE / Hz / 1e3)

        conv_u = gaussian_convolution(TS, spikes_u, tau=5)
        conv_v = gaussian_convolution(TS, spikes_v, tau=5)

        normed_distance = distance(conv_u, conv_v, DT / ms) / (DURATION / ms)

        for target_distance in TARGET_DISTANCES:
            if (
                abs(normed_distance - target_distance) < DELTA_DISTANCE
                and len(collected_pairs[target_distance]) < N_PAIRS
            ):
                collected_pairs[target_distance].append((spikes_u, spikes_v))

        # stop if we have enough pairs collected
        if len(collected_pairs) == len(TARGET_DISTANCES) and all(
            np.array(list(map(len, collected_pairs.values()))) == N_PAIRS
        ):
            break

    return collected_pairs


def get_neurons():
    neurons = NeuronGroup(
        N_NEURONS,
        """
        tau_mem : second (shared, constant)
        tau_refrac : second (constant)
        v_reset : volt (shared, constant)
        v_thresh : volt (shared, constant)
        I_b : ampere (shared, constant)
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

    neurons.I_b = 13.5 * nA

    neurons.v[:] = (
        np.random.uniform(V_RESET / mV, V_THRESH / mV, size=len(neurons)) * mV
    )

    neurons.R_in = 1 * Mohm

    # to randomly assign excitatory and inhibitory neurons later
    indices = np.arange(len(neurons))
    np.random.shuffle(indices)

    # a column of 15x3x3 neurons
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
    dx/dt =  z/D       : 1 (clock-driven) # recovered
    dy/dt = -y/tau_I   : 1 (clock-driven) # active
    z = 1 - x - y      : 1                # inactive
    I_syn_{}_post = A*y : ampere (summed)
    """.format(name)

    if F:
        synapses_eqs += """
        du/dt = -u/F : 1 (clock-driven)
        F : second (constant)
        """

        synapses_action = """
        u += U*(1-u)
        y += u*x # important: update y first
        x += -u*x
        """
    else:
        synapses_action = """
        y += U*x # important: update y first
        x += -U*x
        """

    synapses = Synapses(
        source,
        target,
        model=synapses_eqs,
        on_pre=synapses_action,
        method="exact",
        name=name,
        delay=delay,
    )

    synapses.connect(
        p=f"{C} * exp(-((x_pos_pre-x_pos_post)**2 + (y_pos_pre-y_pos_post)**2 + (z_pos_pre-z_pos_post)**2)/{l}**2)"
    )

    N_syn = len(synapses)

    synapses.tau_I = tau_I

    synapses.A[:] = np.sign(A / nA) * np.random.gamma(1, abs(A / nA), size=N_syn) * nA

    synapses.U[:] = np.random.normal(U, 0.5 * U, size=N_syn)
    # paper samples from uniform, we take the mean
    synapses.U[:][synapses.U < 0] = U

    synapses.D[:] = np.random.normal(D / ms, 0.5 * D / ms, size=N_syn) * ms
    # paper samples from uniform, we take the mean
    synapses.D[:][synapses.D / ms <= 0] = D

    # start fully recovered
    synapses.x = 1

    if F:
        synapses.F[:] = np.random.normal(F / ms, 0.5 * F / ms, size=N_syn) * ms
        # paper samples from uniform, we take the mean
        synapses.F[:][synapses.F / ms <= 0] = F

    return synapses


def run_diagnostics(net, neurons, exc_neurons, inh_neurons,
                    spike_monitor_exc, spike_monitor_inh):
    """Run network once with Poisson stimulus and report regime diagnostics.

    Checks: firing rates, synchrony (Fano factor), irregularity (CV ISI),
    and stationarity (rate drift).
    """
    net.restore()

    net["neurons"].v = (
        np.random.uniform(V_RESET / mV, V_THRESH / mV, size=len(neurons)) * mV
    )

    # inject a simple Poisson stimulus
    spike_times_diag = generate_poisson(DURATION / ms,
                                        STIMULUS_POISSON_RATE / Hz / 1e3)
    net["stimulus"].set_spikes([0] * len(spike_times_diag),
                               spike_times_diag * ms)
    net.run(DURATION)

    T = float(DURATION / second)
    n_exc = len(exc_neurons)
    n_inh = len(inh_neurons)

    print("\n" + "=" * 60)
    print("  NETWORK REGIME DIAGNOSTICS")
    print("=" * 60)

    # --- 1) Mean firing rates ---
    rate_exc = spike_monitor_exc.num_spikes / (n_exc * T)
    rate_inh = spike_monitor_inh.num_spikes / (n_inh * T)

    def rate_verdict(r, label):
        if r == 0:
            return f"  {label}: {r:.1f} Hz  [FAIL - network dead]"
        elif r > 100:
            return f"  {label}: {r:.1f} Hz  [WARN - possible explosion]"
        elif r > 80:
            return f"  {label}: {r:.1f} Hz  [WARN - high activity]"
        else:
            return f"  {label}: {r:.1f} Hz  [OK]"

    print("\n1) Firing rates:")
    print(rate_verdict(rate_exc, "Excitatory"))
    print(rate_verdict(rate_inh, "Inhibitory"))

    # --- 2) Population Fano factor (synchrony proxy) ---
    binw = 5 * ms
    nbins = int(DURATION / binw)
    counts = np.zeros(nbins, dtype=int)

    for mon in [spike_monitor_exc, spike_monitor_inh]:
        bins = np.floor(mon.t / binw).astype(int)
        bins = bins[bins < nbins]
        np.add.at(counts, bins, 1)

    mean_count = counts.mean()
    fano = counts.var() / (mean_count + 1e-12)

    if fano > 5:
        fano_tag = "[WARN - bursty/synchronous]"
    elif fano < 0.3:
        fano_tag = "[WARN - too regular/clock-like]"
    else:
        fano_tag = "[OK]"

    print(f"\n2) Population Fano factor: {fano:.2f}  {fano_tag}")
    print(f"   (mean bin count: {mean_count:.1f} spikes / 5 ms)")

    # --- 3) CV of ISI ---
    def cv_isi(spike_trains):
        cvs = []
        for st in spike_trains:
            st_arr = np.array(st / second)
            if len(st_arr) >= 5:
                isi = np.diff(st_arr)
                if isi.mean() > 0:
                    cvs.append(isi.std() / isi.mean())
        return np.array(cvs)

    tr_exc = spike_monitor_exc.spike_trains()
    tr_inh = spike_monitor_inh.spike_trains()
    cvs_exc = cv_isi(list(tr_exc.values()))
    cvs_inh = cv_isi(list(tr_inh.values()))
    cvs_all = np.concatenate([cvs_exc, cvs_inh]) if len(cvs_exc) + len(cvs_inh) > 0 else np.array([])

    if len(cvs_all) > 0:
        med_cv = np.median(cvs_all)
        if med_cv < 0.5:
            cv_tag = "[WARN - too regular]"
        elif med_cv > 2.0:
            cv_tag = "[WARN - very bursty]"
        else:
            cv_tag = "[OK - AI-like]"
        print(f"\n3) CV of ISI: median={med_cv:.2f}  {cv_tag}")
        print(f"   (n={len(cvs_all)} neurons with >=5 spikes)")
        if len(cvs_exc) > 0:
            print(f"   Exc: median={np.median(cvs_exc):.2f}, "
                  f"range=[{cvs_exc.min():.2f}, {cvs_exc.max():.2f}]")
        if len(cvs_inh) > 0:
            print(f"   Inh: median={np.median(cvs_inh):.2f}, "
                  f"range=[{cvs_inh.min():.2f}, {cvs_inh.max():.2f}]")
    else:
        print("\n3) CV of ISI: [FAIL - no neurons with >=5 spikes]")

    # --- 4) Stationarity (rate drift) ---
    n_windows = 5
    window_dur = T / n_windows
    all_spike_times = np.array([])
    for mon in [spike_monitor_exc, spike_monitor_inh]:
        all_spike_times = np.concatenate([all_spike_times,
                                          np.array(mon.t / second)])

    n_all = n_exc + n_inh
    window_rates = np.zeros(n_windows)
    for w in range(n_windows):
        t_lo = w * window_dur
        t_hi = (w + 1) * window_dur
        n_spikes = np.sum((all_spike_times >= t_lo) & (all_spike_times < t_hi))
        window_rates[w] = n_spikes / (n_all * window_dur)

    # linear regression: rate vs window index
    x = np.arange(n_windows)
    slope = np.polyfit(x, window_rates, 1)[0]
    # slope is Hz change per window; convert to Hz/s
    slope_per_sec = slope / window_dur

    mean_rate = window_rates.mean()
    if mean_rate > 0:
        rel_drift = abs(slope_per_sec) / mean_rate
        if rel_drift > 0.5:
            drift_tag = "[WARN - strong drift]"
        elif rel_drift > 0.2:
            drift_tag = "[WARN - moderate drift]"
        else:
            drift_tag = "[OK - stable]"
    else:
        rel_drift = 0
        drift_tag = "[N/A - no spikes]"

    print(f"\n4) Stationarity:")
    print(f"   Window rates (Hz): {['%.1f' % r for r in window_rates]}")
    print(f"   Slope: {slope_per_sec:+.1f} Hz/s  "
          f"(relative: {rel_drift:.2f})  {drift_tag}")

    print("\n" + "=" * 60)
    print("  END DIAGNOSTICS")
    print("=" * 60 + "\n")


def sim(net, neurons, spike_times):
    """Run network with given stimulus

    Redraws initial membrane voltages

    net -- the network to simulate
    neurons -- the neuron group
    spike_times -- the stimulus to inject
    """
    net.restore()

    net["neurons"].v = (
        np.random.uniform(V_RESET / mV, V_THRESH / mV, size=len(neurons)) * mV
    )
    net["stimulus"].set_spikes([0] * len(spike_times), spike_times * ms)

    net.run(DURATION)

    spikes = list(net["spike_monitor_exc"].spike_trains().values()) + list(
        net["spike_monitor_inh"].spike_trains().values()
    )

    liquid_states = np.array(
        [exponential_convolution(TS, st / ms, tau=30) for st in spikes]
    )

    return liquid_states


if __name__ == '__main__':
    prefs.codegen.target = "numpy"

    neurons = get_neurons()

    N_exc = int(0.8 * len(neurons))

    exc_neurons = neurons[:N_exc]
    exc_neurons.tau_refrac = 3 * ms
    exc_neurons.tau_stimulus = 3 * ms

    inh_neurons = neurons[N_exc:]
    inh_neurons.tau_refrac = 2 * ms
    inh_neurons.tau_stimulus = 6 * ms

    l_lambda = 2

    ee_synapses = get_synapses(
        "ee_synapses",
        exc_neurons,
        exc_neurons,
        C=0.3,
        l=l_lambda,
        tau_I=3 * ms,
        A=30 * nA,
        U=0.5,
        D=1.1 * second,
        F=0.05 * second,
        delay=1.5 * ms,
    )
    ei_synapses = get_synapses(
        "ei_synapses",
        exc_neurons,
        inh_neurons,
        C=0.2,
        l=l_lambda,
        tau_I=3 * ms,
        A=60 * nA,
        U=0.05,
        D=0.125 * second,
        F=1.2 * second,
        delay=0.8 * ms,
    )
    ie_synapses = get_synapses(
        "ie_synapses",
        inh_neurons,
        exc_neurons,
        C=0.4,
        l=l_lambda,
        tau_I=6 * ms,
        A=-19 * nA,
        U=0.25,
        D=0.7 * second,
        F=0.02 * second,
        delay=0.8 * ms,
    )
    ii_synapses = get_synapses(
        "ii_synapses",
        inh_neurons,
        inh_neurons,
        C=0.1,
        l=l_lambda,
        tau_I=6 * ms,
        A=-19 * nA,
        U=0.32,
        D=0.144 * second,
        F=0.06 * second,
        delay=0.8 * ms,
    )

    # place holder for stimulus
    stimulus = SpikeGeneratorGroup(1, [], [] * ms, name="stimulus")

    spike_monitor_stimulus = SpikeMonitor(stimulus)

    static_synapses_exc = Synapses(
        stimulus,
        exc_neurons,
        "A : ampere (shared, constant)",
        on_pre="I_stimulus += A"
    )
    static_synapses_exc.connect(p=1)
    static_synapses_exc.A = 18 * nA

    static_synapses_inh = Synapses(
        stimulus,
        inh_neurons,
        "A : ampere (shared, constant)",
        on_pre="I_stimulus += A"
    )
    static_synapses_inh.connect(p=1)
    static_synapses_inh.A = 9 * nA

    spike_monitor_exc = SpikeMonitor(exc_neurons, name="spike_monitor_exc")
    spike_monitor_inh = SpikeMonitor(inh_neurons, name="spike_monitor_inh")

    defaultclock.dt = DT

    net = Network(
        [
            neurons,
            ee_synapses,
            ei_synapses,
            ie_synapses,
            ii_synapses,
            static_synapses_exc,
            static_synapses_inh,
            stimulus,
            spike_monitor_exc,
            spike_monitor_inh,
        ]
    )
    net.store()

    print("Running network diagnostics...")
    run_diagnostics(net, neurons, exc_neurons, inh_neurons,
                    spike_monitor_exc, spike_monitor_inh)

    print("Collecting stimulus pairs...")
    collected_pairs = collect_stimulus_pairs()
    print("Done collecting stimulus pairs.")

    # add only jittered pairs (distance = 0)
    collected_pairs[0] = [
        [generate_poisson(DURATION / ms, STIMULUS_POISSON_RATE / Hz / 1e3)] * 2
        for _ in range(N_PAIRS)
    ]

    result = defaultdict(list)
    # loop over all distances and Poisson stimulus pairs
    for d, pairs in collected_pairs.items():
        print(f"Processing distance d={d}...")
        for i, (spikes_u, spikes_v) in enumerate(pairs):
            if i % 20 == 0:
                print(f"  Pair {i}/{len(pairs)}")
            liquid_states_u = sim(net, neurons, spikes_u)
            liquid_states_v = sim(net, neurons, spikes_v)
            ed = euclidian_distance(liquid_states_u, liquid_states_v)
            result[d].append(ed)

    # plot
    fig, ax = plt.subplots(figsize=(5, 5))

    linestyles = ["dashed", (0, (8, 6, 1, 6)), (0, (5, 10)), "solid"]

    for d, ls in zip(TARGET_DISTANCES + [0], linestyles):

        eds = result[d]
        eds = np.array(eds)

        ax.plot(
            TS / 1000, np.mean(eds, axis=0), label=f"d(u,v)={d}", linestyle=ls, color="k"
        )

    ax.set_xlabel("time [sec]")
    ax.set_ylabel("state distance")

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 2.5)

    ax.legend(loc="upper center", fontsize="x-large", frameon=False)

    plt.savefig("E:/dd/liquid_state_machine_result.png", dpi=150)
    plt.show()
    print("Done!")
