# R2: Spiking Reservoir State Extraction — Research Notes

## Problem

Convert continuous spiking activity of N reservoir neurons into a
discrete state vector x[k] ∈ R^N for each NARMA time step k.

## Method: Spike Count Binning

Most common and simplest approach for LSM readout:

1. Divide simulation into bins of width ΔT_task (ms)
2. For each bin k and each neuron j: count spikes in [k·ΔT, (k+1)·ΔT)
3. State vector: x[k] = [count_1(k), count_2(k), ..., count_N(k)]

### Bin Size (ΔT_task)

- Literature uses 10–50 ms typically
- Shorter bins → finer temporal resolution but sparser counts
- For HH neurons with firing rates ~5–50 Hz:
  - 10 ms → ~0.05–0.5 spikes/bin on average (very sparse)
  - 20 ms → ~0.1–1.0 spikes/bin
  - 50 ms → ~0.25–2.5 spikes/bin
- **Our default: ΔT_task = 10 ms** (matches user spec)
- Can sweep ΔT_task as hyperparameter

### Which Neurons

- Count ALL reservoir neurons (both E and I)
- State dimension = N_total (e.g., 100)
- Could also use only E neurons (N_E=80) to reduce dimensionality

## Alternative: Exponential Decay Filter

Instead of hard bins, convolve spike trains with exponential kernel:

```
r_j(t) = Σ_spikes exp(-(t - t_spike) / τ_filter)
```

- τ_filter = 10–50 ms
- Sample r_j at each ΔT_task → smoother state
- More biologically motivated but more complex
- **Not using for v1** — spike counts are simpler and standard

## Alternative: Membrane Voltage Snapshot

Sample V_m of each neuron at specific times:

- Higher information content (continuous values)
- But: computationally expensive to store all V_m traces
- Brian2 StateMonitor needed (memory-heavy for long sims)
- **Not using for v1** — spike counts sufficient

## Readout: Ridge Regression

Standard linear readout for reservoir computing:

```
y_pred = X @ w + b
```

Training via ridge regression (L2-regularized least squares):

```
w* = (X^T X + λI)^{-1} X^T y
```

- λ (alpha): regularization strength
- Cross-validate λ over [1e-6, 1e-4, 1e-2, 1, 100, 1e4]
- sklearn.linear_model.Ridge or manual closed-form

### Train/Test Split

- Temporal split (NOT random): first K_train steps for training, last K_test for testing
- No shuffling — preserves temporal structure
- Warmup steps discarded from both NARMA and reservoir

## State Matrix Assembly

For K time steps with N neurons:

```
X shape: (K, N)    — each row is one time step's state
y shape: (K,)      — NARMA target at that step
```

After warmup discard:
- X_train = X[warmup:warmup+K_train]
- X_test  = X[warmup+K_train:warmup+K_train+K_test]

## Our Implementation Plan

1. `state_readout.py`: function `extract_spike_counts(spike_times, spike_indices, n_neurons, dt_task_ms, total_ms)` → X matrix
2. Uses np.histogram or manual binning
3. Returns (K, N) array of integer spike counts
4. Optional: normalize counts (divide by ΔT_task to get rates)

## References

- Maass, Natschläger, Markram (2002): original LSM paper
- Verstraeten et al. (2007): reservoir computing survey, spike count readout
- Norton & Ventura (2010): LSM with spike count features
