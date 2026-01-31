# Implementation Plan: Project C - Hodgkin-Huxley Reservoir "Edge of Chaos" Study

## Research Objective

**Hypothesis**: Biological neural networks, through evolutionary optimization, operate at the **"Edge of Chaos"** (Lyapunov exponent λ ≈ 0), where computational performance is maximized.

**Goal**: Rigorously test whether a biologically-inspired Hodgkin-Huxley reservoir with A-current dynamics demonstrates:
1. A performance peak at λ ≈ 0 across multiple tasks
2. Superior performance compared to mathematical baselines (ESN) at this boundary
3. Biologically plausible dynamics (firing rates 1-50 Hz)

---

## Directory Structure

```
project_C_poisson/
├── src/
│   ├── hh_model.py          # HH neuron with A-current (Shriki et al.)
│   ├── run_experiment.py    # Main sweep runner
│   ├── run_esn.py           # ESN baseline
│   ├── run_analysis.py      # Statistical analysis
│   ├── config.py            # Configuration schema
│   ├── rng_manager.py       # 4-stream deterministic RNG
│   ├── readout.py           # Ridge regression + Blocked CV
│   ├── cv.py                # Cross-validation with temporal gap
│   ├── utils.py             # Filtering, downsampling
│   ├── tasks/
│   │   ├── narma.py         # NARMA-10 regression
│   │   ├── xor.py           # Delayed XOR classification
│   │   ├── mc.py            # Memory Capacity
│   │   └── lyapunov_task.py # Lyapunov exponent
│   └── baselines_rc/
│       └── esn.py           # Echo State Network
├── configs/
│   ├── production_config.yaml   # Full sweep (1000 trials)
│   └── validation_config.yaml   # Quick validation (8 trials)
├── tests/
│   ├── test_repro.py        # Reproducibility tests
│   └── test_leakage.py      # Data leakage tests
├── deploy.sh                # Docker deployment
├── Dockerfile
└── docker-compose.yml
```

---

## Core Methodology

### 1. Reproducibility (Uncompromising)

**RNG Management** (`rng_manager.py`):
- **4 independent streams**:
  1. `seed_rec`: Network weights (W_rec)
  2. `seed_inmask`: Input mask (W_in)
  3. `seed_in`: Input spike trains
  4. `seed_readout`: CV splits
- **Deterministic seeding**: `base_seed=2025`, trial-specific offsets
- **No global state**: Each trial gets isolated generators

**Caching** (`hh_model.py`):
- Cache key: `SHA1(rho, bias, seeds_tuple, N, task_input_id, dt, len_input)`
- Cached data: Filtered spike trains, firing rates, synaptic stats
- Cache invalidation: Automatic on parameter change

**Version tracking**:
- Git hash embedded in all result files
- Docker image SHA256 logged

### 2. Biological Realism

**Hodgkin-Huxley Model** (`hh_model.py`):
```python
# Channels
gNa = 120.0  # Sodium
gK = 36.0    # Potassium (delayed rectifier)
gL = 0.3     # Leak
gA = 20.0    # A-current (Shriki et al.)

# Reversal potentials
ENa = 50.0 mV
EK = -77.0 mV
EL = -54.4 mV
EA = -80.0 mV

# A-current dynamics
a_inf = 1 / (1 + exp(-(V + 50) / 20))
b_inf = 1 / (1 + exp((V + 80) / 6))
tau_b = 20 ms
```

**Network Architecture**:
- N = 100 neurons
- Density = 0.2 (sparse connectivity)
- Dale's principle: 80% excitatory, 20% inhibitory
- Spectral radius scaling: W_rec *= ρ

**Input Encoding**:
- Poisson spike trains: rate ∈ [10, 150] Hz
- Symbol duration: 20 ms
- Integration timestep: dt = 0.05 ms (400 steps/symbol)

### 3. Lyapunov Exponent Calculation (Critical Fix)

**Problem with naive approach**:
- Perturbing at t=0 from steady-state is unphysical
- Transient dynamics contaminate measurement

**Rigorous solution** (Attractor Branching):
```python
# 1. Warm-up phase (1000 steps)
res_warmup = hh.simulate(rho, bias, spikes_warmup, trim_steps=0)
attractor_state = res_warmup['final_state']  # Capture V, m, h, n, b, s_trace

# 2. Branch trajectories from attractor
# Reference
traj_ref = hh.simulate(rho, bias, spikes_test, trim_steps=0, full_state=attractor_state)

# Perturbed (ε = 1e-6 mV on neuron 0)
perturbed_state = copy.deepcopy(attractor_state)
perturbed_state['V'][0] += 1e-6
traj_pert = hh.simulate(rho, bias, spikes_test, trim_steps=0, full_state=perturbed_state)

# 3. Measure divergence (skip initial window)
phi_ref = filter(traj_ref['spikes'])
phi_pert = filter(traj_pert['spikes'])
lambda_step = lyap.compute_lambda(phi_ref, phi_pert, window_range=[50, 250])
lambda_sec = lambda_step / step_duration_s  # Convert to s^-1
```

**Key improvements**:
- Start from **true attractor** (not artificial steady-state)
- Window range [50, 250] skips any residual transient
- Units: s⁻¹ (physically interpretable)

### 4. Cross-Validation (Temporal Data)

**Blocked CV** (`cv.py`):
- 5 folds
- 10-step gap between train/test to prevent leakage
- Ridge regression with α ∈ [1e-6, 1e-4, 1e-2, 1, 10, 100]

**Baseline comparisons**:
- **NARMA**: AR(10) model
- **XOR**: Class priors (50%)
- **MC**: Shuffled input
- **Lyapunov**: Baseline = 0 (reference point)

### 5. Statistical Analysis

**Hypothesis Test** (`run_analysis.py`):
```
H0: Performance at λ ≈ 0 equals performance at λ << 0 or λ >> 0
H1: Performance at λ ≈ 0 is significantly higher
```

**Method**:
- Identify "Peak" region: |λ| < 0.01 s⁻¹
- Identify "Neighbors": 0.01 < |λ| < 0.1 s⁻¹
- Wilcoxon signed-rank test (paired, non-parametric)
- Cohen's d for effect size
- Bonferroni correction for multiple tasks

**Bio-plausibility filter**:
- Only analyze trials with 1 < firing_rate < 50 Hz
- Flag outliers (saturation, silence)

---

## Experimental Parameters

### Sweep Grid (Production)
```yaml
rho_grid: [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
bias_grid: [0.0, 2.0, 4.0, 6.0, 8.0]  # pA
seeds: 20
```

**Total trials**: 10 × 5 × 20 = **1000**

### Tasks
1. **NARMA-10**: T=2000, order=10, metric=NRMSE
2. **XOR**: T=2000, delay=2, metric=Accuracy
3. **MC**: T=2000, max_lag=20, metric=Capacity
4. **Lyapunov**: T_warmup=1000, T_test=500, metric=λ (s⁻¹)

### ESN Baseline
```yaml
rho_esn: [0.1, 0.2, ..., 1.2]  # 12 values
seeds: 20
leaking_rate: 1.0  # No leak (fair comparison)
```

---

## Execution Plan

### Phase 1: Validation (Local)
```bash
python validate_local.py
```
- Quick sweep: 2×2×2 = 8 trials
- Verify: No crashes, reasonable metrics
- Time: ~2 minutes

### Phase 2: Production (Docker)
```bash
./deploy.sh
```
- Validation sweep (8 trials)
- Production sweep (1000 trials)
- ESN baseline (240 trials)
- Resource limits: 80% CPU/RAM
- Time: ~2 hours on 60-core server

### Phase 3: Analysis
```bash
python src/run_analysis.py --results results/*.parquet
```
- Load all `.parquet` files
- Filter by bio-plausibility
- Statistical tests
- Generate plots:
  - λ vs Performance (scatter + heatmap)
  - Firing Rate distribution
  - HH vs ESN comparison

---

## Expected Outcomes

### Success Criteria
1. **Peak at λ ≈ 0**: Statistically significant (p < 0.05) performance maximum
2. **HH > ESN**: At least one task shows superior performance at the peak
3. **Biological plausibility**: Peak occurs within 1-50 Hz firing rate range

### Potential Results
- **Strong support**: Peak in all 3 tasks (NARMA, XOR, MC)
- **Moderate support**: Peak in 2/3 tasks
- **Weak support**: Peak in 1/3 tasks
- **Null result**: No peak (uniform or monotonic performance)

---

## Risk Mitigation

### Numerical Stability
- **Issue**: Ridge regression explodes with near-singular matrices
- **Solution**: Strong regularization (α ≥ 1e-6), condition number checks

### Memory Overflow
- **Issue**: 1000 trials × 100 neurons × 800k timesteps
- **Solution**: Streaming filters, limited workers (60), cache cleanup

### Reproducibility Failure
- **Issue**: Non-deterministic results across runs
- **Solution**: Strict RNG isolation, Docker pinning, tests in `test_repro.py`

---

## Deliverables

1. **Code**: Clean, documented, tested
2. **Results**: `.parquet` files with all trials
3. **Analysis**: Statistical report + plots
4. **Documentation**: This plan + README
5. **Publication**: Draft manuscript (if hypothesis confirmed)

---

## Timeline

- **Day 1**: Code finalization, local validation
- **Day 2**: Production deployment (2h run)
- **Day 3**: Analysis, interpretation
- **Day 4**: Manuscript draft (if positive results)

---

## Adherence to Research Protocol

This plan follows strict academic standards:
- ✅ Reproducibility (deterministic RNG, Docker, Git)
- ✅ Fair baselines (ESN, AR, priors)
- ✅ Statistical rigor (Wilcoxon, effect size, correction)
- ✅ Biological realism (HH+A-current, Dale's principle)
- ✅ Methodological transparency (all code public)

**No shortcuts. No p-hacking. Publication-grade only.**
