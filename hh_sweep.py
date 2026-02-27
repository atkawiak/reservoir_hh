"""HH parameter sweep — test multiple alpha/I_b configurations."""
import sys
sys.path.insert(0, ".")

import reservoir_hh_stp as r
import matplotlib
matplotlib.use("Agg")
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import time

from brian2 import (
    NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor, Network,
    defaultclock, start_scope,
    ms, mV, nA, second, Hz,
)

_sum = sum  # save builtin before any overwrite

# (alpha_scale, I_b_nA, stim_exc_nA, stim_inh_nA, label)
CONFIGS = [
    (3.0,  0.1,  0.5,  0.25, "alpha=3x, Ib=0.1"),
    (5.0,  0.1,  0.9,  0.45, "alpha=5x, Ib=0.1"),
    (5.0,  0.2,  0.9,  0.45, "alpha=5x, Ib=0.2"),
    (10.0, 0.1,  1.5,  0.75, "alpha=10x, Ib=0.1"),
]

r.N_PAIRS = 15

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for idx, (alpha, ib, stim_e, stim_i, label) in enumerate(CONFIGS):
    ax = axes[idx // 2][idx % 2]
    print(f"\n=== Config {idx+1}: {label} ===")
    t0 = time.time()

    start_scope()
    neurons = r.get_neurons()
    neurons.I_b = ib * nA

    N_exc = int(0.8 * r.N_NEURONS)
    exc = neurons[:N_exc]
    inh = neurons[N_exc:]
    exc.tau_stimulus = 3 * ms
    inh.tau_stimulus = 6 * ms

    L = 2
    A_ee = 0.30 * alpha
    A_ei = 0.60 * alpha
    A_ie = -0.19 * alpha
    A_ii = -0.19 * alpha

    ee = r.get_synapses("ee_synapses", exc, exc, C=0.3, l=L, tau_I=3*ms,
                        A=A_ee*nA, U=0.5, D=1.1*second, F=0.05*second, delay=1.5*ms)
    ei = r.get_synapses("ei_synapses", exc, inh, C=0.2, l=L, tau_I=3*ms,
                        A=A_ei*nA, U=0.05, D=0.125*second, F=1.2*second, delay=0.8*ms)
    ie = r.get_synapses("ie_synapses", inh, exc, C=0.4, l=L, tau_I=6*ms,
                        A=A_ie*nA, U=0.25, D=0.7*second, F=0.02*second, delay=0.8*ms)
    ii = r.get_synapses("ii_synapses", inh, inh, C=0.1, l=L, tau_I=6*ms,
                        A=A_ii*nA, U=0.32, D=0.144*second, F=0.06*second, delay=0.8*ms)

    stimulus = SpikeGeneratorGroup(1, [], [] * ms, name="stimulus")
    se = Synapses(stimulus, exc, "A : ampere (shared, constant)", on_pre="I_stimulus += A")
    se.connect(p=1)
    se.A = stim_e * nA
    si = Synapses(stimulus, inh, "A : ampere (shared, constant)", on_pre="I_stimulus += A")
    si.connect(p=1)
    si.A = stim_i * nA

    sm_exc = SpikeMonitor(exc, name="spike_monitor_exc")
    sm_inh = SpikeMonitor(inh, name="spike_monitor_inh")
    defaultclock.dt = r.DT

    net = Network([neurons, ee, ei, ie, ii, se, si, stimulus, sm_exc, sm_inh])
    net.store()

    # Quick rate check
    spk = r.generate_poisson(r.DURATION / ms, r.STIMULUS_POISSON_RATE / Hz / 1e3)
    r.sim(net, spk)
    n_exc_spk = _sum(len(s) for s in net["spike_monitor_exc"].spike_trains().values())
    n_inh_spk = _sum(len(s) for s in net["spike_monitor_inh"].spike_trains().values())
    rate_e = n_exc_spk / N_exc / (r.DURATION / second)
    rate_i = n_inh_spk / (r.N_NEURONS - N_exc) / (r.DURATION / second)
    print(f"  Rates: exc={rate_e:.1f} Hz, inh={rate_i:.1f} Hz")

    if rate_e > 200:
        print("  SKIP - exploding!")
        ax.set_title(f"{label}\nexploding (exc={rate_e:.0f} Hz)", color="red")
        continue

    collected_pairs = r.collect_stimulus_pairs()
    collected_pairs[0] = [
        [r.generate_poisson(r.DURATION / ms, r.STIMULUS_POISSON_RATE / Hz / 1e3)] * 2
        for _ in range(r.N_PAIRS)
    ]

    result = defaultdict(list)
    for d, pairs in collected_pairs.items():
        for su, sv in pairs:
            lu = r.sim(net, su)
            lv = r.sim(net, sv)
            result[d].append(r.euclidian_distance(lu, lv))

    styles = ["dashed", (0, (8, 6, 1, 6)), (0, (5, 10)), "solid"]
    for d, ls in zip(r.TARGET_DISTANCES + [0], styles):
        eds = np.array(result[d])
        ax.plot(r.TS / 1000, np.mean(eds, axis=0), label=f"d={d}", linestyle=ls, color="k")

    ax.set_xlabel("time [sec]")
    ax.set_ylabel("state distance")
    ax.set_xlim(0, 0.5)
    ax.set_title(f"{label}\nexc={rate_e:.1f}Hz inh={rate_i:.1f}Hz")
    ax.legend(fontsize="small", frameon=False)
    print(f"  Done in {time.time() - t0:.0f}s")

plt.suptitle("HH + STP parameter sweep", fontsize=14)
plt.tight_layout()
plt.savefig("hh_sweep.png", dpi=150)
print("\nSaved hh_sweep.png")
