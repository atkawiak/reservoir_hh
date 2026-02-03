# Implementation Plan: Project C - Hodgkin-Huxley Reservoir "Edge of Chaos" Study

## Research Objective

**Central Research Question**: Do **maximally biologically realistic** Hodgkin-Huxley neural networks, operating as reservoirs, achieve optimal computational performance at the **"Edge of Chaos"** ($\lambda \approx 0$), thereby validating the **Critical Brain Hypothesis** in conductance-based spiking neural networks?

**Hypothesis**: Conductance-based Hodgkin-Huxley reservoirs with A-current (IA potassium channel), Dale's principle (E/I segregation), and biologically plausible parameters exhibit:
1. **Maximal classification accuracy** and **minimal prediction error** at the **Edge of Chaos** ($|\lambda| < 0.01$ s⁻¹)
2. **Superior generalization** (lowest Train-Test gap) in the critical regime
3. **Peak Memory Capacity** at $\lambda \to 0^-$ (below chaos threshold)
4. **Biological plausibility**: Firing rates 1-50 Hz, power-law dynamics reminiscent of neuronal avalanches

---

## Theoretical Foundation (Literature-Driven)

### 1. Critical Brain Hypothesis (Beggs & Plenz, 2003)

**Neuronal Avalanches as Evidence:**
- Spontaneous activity in biological cortical networks exhibits **neuronal avalanches** with **power-law size distributions** [Beggs & Plenz, 2003; *J. Neuroscience*]
- This is a hallmark of **self-organized criticality (SOC)** - systems naturally tune themselves to phase transitions
- **Criticality is the ONLY known computational regime** that inherently optimizes:
  - Information transmission [Shew et al., 2011]
  - Dynamic range [Kinouchi & Copelli, 2006]
  - Memory capacity [Haldeman & Beggs, 2005]

**Implication for Our Study:**
✅ If HH reservoirs naturally exhibit Edge of Chaos behavior, they replicate fundamental brain dynamics.

---

### 2. Edge of Chaos in Reservoir Computing

**Empirical Evidence:**
- **Echo State Networks (ESN)**: Optimal MNIST classification "just below chaos" ($\lambda < 0$ but close to 0) [Verstraeten et al., 2007]
- **Maximum Memory Capacity**: Achieved at $\lambda \to 0^-$ (longest memory without instability) [Jaeger, 2001; Bertschinger & Natschläger, 2004]
- **Information Transfer**: Peaks at critical transition between order and chaos [Lizier et al., 2014; *Chaos*]
- **Neural Microcircuits**: "Edge of Chaos" **predicts** circuit parameters yielding maximal computational performance [Maass et al., 2002]

**Controversy:**
Some studies show performance degradation *at* the edge in specific contexts [Legenstein & Maass, 2007]. However, consensus supports edge-of-chaos optimality for:
- Temporal integration tasks (NARMA)
- Classification requiring memory (XOR with delays)
- Pattern recognition in noisy environments

---

### 3. Biological Realism: HH vs. Simplified Models

**Why Hodgkin-Huxley?**

| **Criterion**              | **Hodgkin-Huxley (HH)**                          | **Leaky Integrate-and-Fire (LIF)**       |
|----------------------------|--------------------------------------------------|------------------------------------------|
| **Biological Accuracy**    | ✅ **Gold standard**: Captures all 20 neurocomputational features | ⚠️ Captures only 3 basic firing patterns |
| **Ion Channel Detail**     | ✅ Explicit $g_{Na}$, $g_K$, $g_L$, $g_A$ dynamics | ❌ Abstract "leak" only                  |
| **Action Potential Shape** | ✅ Complete waveform (spike width, afterhyperpolarization) | ❌ Delta function (no shape)             |
| **Bifurcation Richness**   | ✅ Hopf, SNIC, saddle-node bifurcations          | ⚠️ Limited bifurcation structure         |
| **Computational Cost**     | ⚠️ 8x slower than LIF (on neuromorphic hardware) | ✅ 95% less computation than HH          |
| **Transcriptomic Correlation** | ✅ Parameters correlate with gene expression [Gouwens et al., 2018] | ❌ No molecular mapping                  |

**Trade-off:**
- LIF: Efficient for large-scale simulations (millions of neurons)
- **HH: Necessary for studying biologically realistic dynamics** where ion channel mechanisms matter

**Our Choice:**
We accept the computational cost to test whether **biological realism itself** predicts edge-of-chaos behavior.

---

### 4. The A-Current (IA Potassium Channel)

**Role in Neural Dynamics:**
- **Transient outward K+ current** (fast activation, slower inactivation)
- **Delays spiking** and **reduces firing frequency** [Connor & Stevens, 1971]
- **Can trigger bursting** by influencing spike trajectory near unstable fixed points [Shriki et al., 2003]
- **Spike-frequency adaptation**: Crucial for temporal coding

**Bifurcation Impact:**
- Increasing $g_A$ (A-current conductance) acts as **bifurcation parameter**
- Can shift system from tonic spiking → bursting → quiescence
- Interacts with $g_L$ (leak) to determine **chaos onset**

**Literature Gap:**
While Shriki et al. (2003) used HH+A-current for rate model validation, **no study has explicitly tested A-current's role in reservoir computing at the Edge of Chaos**. 

**Our Contribution:**
✅ First systematic study of $g_A$ as control parameter for criticality in biological reservoirs.

---

### 5. Dale's Principle and Network Stability

**Dale's Principle**: Each neuron is either purely **excitatory** (E) or **inhibitory** (I) across all synapses.

**Benefits in Artificial Networks (Recent Findings):**
- **Functional robustness** against synaptic noise [Sussex et al., 2023]
- **Simpler learning**: Adjusting neuron-level properties (not individual weights) suffices [NeurIPS 2022]
- **Biological design efficiency**: Fewer neurotransmitter/receptor types per neuron

**In Biological Networks:**
- Establishes **balanced dynamics** (E/I balance) crucial for irregular firing [van Vreeswijk & Sompolinsky, 1996]
- **Prevents runaway excitation** (epileptiform activity in supercritical regime)

**Implementation:**
- 80% Excitatory, 20% Inhibitory (cortical ratio)
- Sign constraints on $W_{rec}$ during initialization

---

### 6. Readout Layer: Ridge Regression as Standard

**Why Linear Readout?**
- **Reservoir Computing paradigm**: Fixed random recurrent weights, **only readout is trained**
- High-dimensional reservoir states become **linearly separable** via kernel trick [Maass et al., 2002]
- **Ridge Regression** prevents overfitting in high-dimensional space [Lukoševičius & Jaeger, 2009]

**Literature Consensus:**
✅ Ridge is the de facto standard for reservoir readout [Verstraeten et al., 2007; Jaeger, 2001]
✅ For classification: Logistic regression or linear SVM on reservoir states
✅ Cross-validation essential for temporal data (blocked CV with gap to prevent leakage)

**Our Implementation:**
- Ridge with $\alpha \in \{10^{-6}, 10^{-4}, 10^{-2}, 1, 10, 100\}$
- 5-fold Blocked CV with 10-step gap
- Z-score normalization of reservoir states

---

## Research Gap and Contribution

**What is Known:**
1. Simplified reservoirs (ESN, LIF) show edge-of-chaos benefits [Jaeger, Verstraeten]
2. Biological brains operate at criticality (neuronal avalanches) [Beggs & Plenz]
3. HH models are biologically accurate but computationally expensive [Gerstner & Kistler]

**What is UNKNOWN:**
❓ Do **full conductance-based HH models** with A-current exhibit edge-of-chaos optimization?  
❓ Does **biological realism** (Dale's principle, realistic $g_{Na}/g_K/g_A$) alter the critical regime?  
❓ Is the "Edge of Chaos" in HH networks the **same** as in abstract ESNs, or does it emerge at different $\rho$ values?

**Our Hypothesis:**
✅ HH reservoirs will require **higher $\rho$** (stronger recurrence) to reach chaos compared to ESNs, due to **intrinsic ion channel stabilization** ($g_K$, $g_A$ act as "brakes").  
✅ Performance will peak at $|\lambda| \approx 0$, validating Critical Brain Hypothesis in conductance-based models.  
✅ **Generalization gap** (Train - Test error) will be **minimized** at the edge, demonstrating criticality's role in preventing overfitting

---

## Experimental Design: Dynamic Ensemble Protocol

Instead of a traditional brute-force grid search, we use a **Targeted Ensemble Approach** to isolate the effect of the "Edge of Chaos". We generate triplets of networks that are architecturally identical but dynamically distinct.

### **STAGE A: Critical Ensemble Generation**
**Objective**: Build a high-quality dataset of 1000 "triplets" (3000 networks total).

1.  **Stage A.1: Critical Point Localizer**
    - Sample parameter space $(g_A, g_L, \rho, bias)$ using random search.
    - Identify 1000 **Critical Points** (The Edge) where $\lambda \in [-0.05, 0.05]$ and Firing Rate $\in [1, 100]$ Hz.
    - Status: ✅ Complete (1009 points found).

2.  **Stage A.2: Neighbor Generation via Bifurcation Tracking**
    - For each Critical Point, find two neighbors by systematically tracking bifurcations:
        - **Stable Neighbor**: Push $\lambda \to \approx -0.2$ by increasing $g_L$ or decreasing $\rho$.
        - **Chaotic Neighbor**: Push $\lambda \to \approx +0.2$ by decreasing $g_A/g_L$ or increasing $\rho$.
    - **Bifurcation Tracking**: Use binary search in parameter space to find the exact point where the desired $\lambda$ is reached, ensuring small, controlled distances $d$ from the edge.

3.  **Metrological Characterization (Intrinsic Metrics)**:
    - For every network in the ensemble, calculate task-independent properties:
        - **Kernel Rank (KR)**: Richness of the high-dimensional projection (measured via SVD rank of state matrix).
        - **Memory Capacity (MC_intrinsic)**: Inherent memory retention using uniform noise input.
        - **Distance Metrics**: 
            - $\Delta \lambda$ (Dynamic distance)
            - $d_{4D}$ (Normalized Euclidean distance in parameter space).

### **STAGE B: Functional Evaluation**
**Objective**: Test the 3000 networks on benchmark tasks to test the Edge of Chaos hypothesis.

1.  **Benchmark Tasks**:
    - **Delayed XOR**: Non-linear logic with memory (Delays 1-5).
    - **NARMA-10**: Non-linear time-series prediction.
    - **Memory Capacity (Full)**: Linear memory duration.

2.  **Analysis of Performance vs. Distance**:
    - Plot Task Performance $P = f(\lambda)$ and $P = f(d_{4D})$.
    - Quantify the "width" of the critical regime.
    - Correlate intrinsic metrics (KR, MC) with task performance to explain **why** the edge is optimal.

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
- ✅ **Reproducibility**: Deterministic RNG (4 independent streams), Docker containerization, Git versioning
- ✅ **Fair Baselines**: ESN (abstract reservoir), AR(10) (NARMA), Class priors (XOR), Shuffled input (MC)
- ✅ **Statistical Rigor**: Wilcoxon signed-rank, Cohen's d, Bonferroni correction, bio-plausibility filtering
- ✅ **Biological Realism**: Full conductance-based HH model, Dale's principle, A-current dynamics
- ✅ **Methodological Transparency**: All code public, detailed theoretical foundation, contingency plans

**No shortcuts. No p-hacking. Publication-grade only.**

---

## Biological Realism: Validation Metrics

To claim this is a "**maximally biological**" reservoir study, we must quantify bio-plausibility beyond firing rates:

### 1. **Firing Rate Distribution**
- **Target:** 1-50 Hz (cortical range)
- **Analysis:** Histogram of per-neuron firing rates across all trials
- **Criterion:** >80% of neurons in biologically plausible range

### 2. **Coefficient of Variation (CV) of ISI**
- **Target:** CV ∈ [0.5, 1.5] (irregular spiking, not Poisson-like)
- **Biological Reference:** Cortical neurons exhibit irregular firing with CV ~1.0 [Softky & Koch, 1993]
- **Analysis:** Compute CV of inter-spike intervals per neuron

### 3. **E/I Balance Ratio**
- **Target:** Excitatory synaptic current ≈ Inhibitory synaptic current (balanced regime)
- **Measurement:** $|I_{syn}^E| / |I_{syn}^I|$ should be near 1.0
- **Biological Reference:** Balanced networks show cancellation of E/I currents [van Vreeswijk & Sompolinsky, 1996]

### 4. **Power-Law Activity Dynamics** (Optional, if time permits)
- **Target:** Avalanche size distribution follows $P(s) \sim s^{-\alpha}$ with $\alpha \approx 1.5$
- **Analysis:** Detect "avalanches" (bursts of activity), fit power-law using Maximum Likelihood Estimation
- **Significance:** Direct replication of Beggs & Plenz (2003) neuronal avalanches

### 5. **Membrane Potential Statistics**
- **Target:** Resting potential near -65 mV, spike threshold ~-55 mV
- **Analysis:** Mean and std of $V(t)$ during non-spiking periods
- **Deviation Check:** Flag trials with abnormal depolarization (e.g., mean V > -50 mV)

**Reporting:**
Create a "**Bio-Plausibility Score**" (0-100) combining these metrics:
```python
bio_score = (
    0.3 * firing_rate_in_range +
    0.2 * cv_in_range +
    0.2 * ei_balance_score +
    0.15 * voltage_stats_score +
    0.15 * (1 if power_law_detected else 0)
) * 100
```

Include in publication: "Trials with bio_score > 70 were retained for analysis."

---

## Publication Strategy

### Target Journals (Ranked by Fit)

1.  **Nature Communications** (IF ~17.7)
    - **Angle:** "Conductance-based reservoirs validate Critical Brain Hypothesis"
    - **Strength:** Interdisciplinary (neuroscience + ML), open access
    - **Challenge:** Requires strong novelty - emphasize **first HH+A-current edge-of-chaos study**

2.  **eLife** (Computational Neuroscience)
    - **Angle:** "Biological realism predicts edge-of-chaos optimization in spiking reservoirs"
    - **Strength:** Favors rigorous computational work, transparent peer review
    - **Challenge:** Strong competition in reservoir computing

3.  **Neural Computation** (IF ~2.9, but high prestige in field)
    - **Angle:** "Lyapunov analysis of Hodgkin-Huxley reservoirs with A-current"
    - **Strength:** Technical depth welcome, reservoir computing core audience
    - **Challenge:** May want comparison to real neural data

4.  **PLOS Computational Biology**
    - **Angle:** "Self-organized criticality in biologically detailed spiking networks"
    - **Strength:** Open access, favors biological motivation
    - **Challenge:** Broad readership - must make critical brain hypothesis accessible

### Key Citations to Include

**Theoretical Foundation:**
- Beggs & Plenz (2003) - Neuronal avalanches (J. Neuroscience)
- Jaeger (2001) - Echo State Property (GMD Report)
- Bertschinger & Natschläger (2004) - Edge of Chaos in LSM (Neural Comput.)
- Lizier et al. (2014) - Information transfer at criticality (Chaos)
- Shriki et al. (2003) - A-current rate model (Network)

**Biological Realism:**
- Hodgkin & Huxley (1952) - Original conductance-based model
- Connor & Stevens (1971) - A-current discovery
- van Vreeswijk & Sompolinsky (1996) - Balanced networks
- Maass et al. (2002) - Liquid State Machines (Neural Comput.)
- Verstraeten et al. (2007) - Reservoir computing review

### Manuscript Outline (Tentative)

**Title:** "Edge of Chaos in Conductance-Based Hodgkin-Huxley Reservoirs: Validating the Critical Brain Hypothesis with A-Current Dynamics"

**Abstract (250 words):**
- Problem: Unknown if biological neurons operate at edge of chaos
- Gap: Simplified models (ESN, LIF) show benefits, but lack ion channel realism
- Method: Full HH with A-current, sweep $\rho$ and $g_L$ to locate critical regime
- Result: Performance peaks at $|\lambda| \approx 0$ (if confirmed)
- Significance: First evidence that **biological detail** predicts criticality

**Main Sections:**
1. **Introduction**
   - Critical Brain Hypothesis (Beggs & Plenz)
   - Edge of Chaos in abstract reservoirs
   - Gap: No conductance-based validation
   
2. **Methods**
   - HH+A-current model details
   - Reservoir architecture (Dale's principle)
   - Tasks (NARMA, XOR, MC, Lyapunov)
   - Statistical analysis protocol
   
3. **Results**
   - Phase 1: Locating criticality (bifurcation diagram)
   - Phase 2: Performance vs. $\lambda$ (scatter plots, bin analysis)
   - Generalization gap minimization at edge
   - Biological plausibility metrics
   
4. **Discussion**
   - HH vs. ESN: Does biological realism matter?
   - Role of A-current in criticality
   - Connection to neuronal avalanches
   - Limitations and future work

**Supplementary Material:**
- Full parameter table
- Reproducibility checklist (Docker hash, Git commit)
- Extended bifurcation analysis ($g_L$ vs. $\rho$)
- Comparison with LIF reservoirs (if time permits)

---

## Success Criteria (Final Checklist)

✅ **Phase 1 Complete:** Functional regime identified (XOR > 0.6)  
✅ **Phase 2 Complete:** High-statistics sweep (n≥50 per bin)  
✅ **Statistical Significance:** Wilcoxon test p < 0.05 (Bonferroni-corrected)  
✅ **Effect Size:** Cohen's d > 0.5 (medium to large effect)  
✅ **Biological Plausibility:** Bio-score > 70 for >80% of analyzed trials  
✅ **Reproducibility:** Independent run on different machine yields same peak λ  
✅ **Code Quality:** Passes all tests in `tests/`, lints clean, documented  

**Publication Decision Tree:**
- **Strong Support (all metrics peak at edge):** Submit to Nature Communications / eLife
- **Moderate Support (2/3 tasks):** Submit to Neural Computation / PLOS Comp Bio
- **Null Result:** Write methods paper for Journal of Open Source Software, Document "why HH doesn't show edge-of-chaos" (still publishable as negative result)

---

## Final Note: Biological Realism as Core Contribution

**What makes this study unique:**

Most reservoir computing studies use:
- ⚠️ Echo State Networks (abstract, rate-coded)
- ⚠️ Leaky Integrate-and-Fire (simplified spiking)
- ⚠️ Random connectivity (no Dale's principle)

**We use:**
- ✅ **Full Hodgkin-Huxley** (4 conductances: Na, K, L, A)
- ✅ **A-current** (biologically realistic spike-frequency adaptation)
- ✅ **Dale's principle** (E/I segregation)
- ✅ **Biologically plausible firing rates** (1-50 Hz cortical range)

**The question is not "does edge of chaos exist?"** (known from ESN literature)  
**The question is:** "Do **ion channel mechanisms** naturally self-organize to this regime?"

If YES → Strong evidence that evolution "tuned" $g_{Na}$, $g_K$, $g_A$ for criticality  
If NO → Suggests abstract reservoirs differ fundamentally from biological neurons

**Either outcome is scientifically valuable.**

