# Hodgkin-Huxley Reservoir Computing: Edge of Chaos Hypothesis

## Core Thesis
**Experimental proof that biologically grounded neural networks achieve maximal computational performance (Memory Capacity, XOR, NARMA) at the "Edge of Chaos" ($\lambda \approx 0$).**

This project demonstrates that biological adaptation mechanisms (specifically the **A-current**) automatically tune the network dynamics towards this critical boundary, balancing stability with computational flexibility.

## Overview
This repository implements a high-performance **Hodgkin-Huxley Reservoir** using `scipy` and sparse matrices. It is designed to investigate the relationship between:
1.  **Biological Realism:** Ion channel dynamics ($I_{Na}, I_K, I_A, I_L$) and Dale's principle.
2.  **Dynamical Regime:** Chaotic vs. Stable dynamics (quantified by Lyapunov Exponent $\lambda$).
3.  **Computational Power:** Performance on benchmark tasks (XOR, NARMA-10, Memory Capacity).

## Scientific Motivation

**Central Hypothesis**: Biological neurons, through evolutionary optimization, operate at the critical boundary where the Lyapunov exponent λ ≈ 0 (the "Edge of Chaos"). At this point, the network maximizes:
- **Memory capacity** (ability to retain temporal information)
- **Nonlinear processing** (ability to solve complex tasks like XOR, NARMA)
- **Biological plausibility** (realistic firing rates ~1-50 Hz)

This project rigorously tests this hypothesis using:
1. **Biologically realistic HH neurons** with A-current dynamics (Shriki et al.)
2. **Attractor-based Lyapunov exponent calculation** (avoiding transient artifacts)
3. **Fair baselines** (Echo State Networks, autoregressive models)
4. **Strict reproducibility** (deterministic RNG, Docker containerization)

## Project Structure

```
project_C_poisson/
├── src/
│   ├── hh_model.py          # Hodgkin-Huxley neuron dynamics with A-current
│   ├── run_experiment.py    # Main experimental runner (60-core parallelized)
│   ├── run_esn.py           # ESN baseline for comparison
│   ├── run_analysis.py      # Statistical analysis (Wilcoxon, Cohen's d)
│   ├── config.py            # Experiment configuration schema
│   ├── rng_manager.py       # 4-stream deterministic RNG
│   ├── readout.py           # Ridge regression with Blocked CV
│   ├── cv.py                # Cross-validation with temporal gap
│   ├── utils.py             # Filtering, downsampling utilities
│   ├── tasks/
│   │   ├── narma.py         # NARMA-10 nonlinear regression
│   │   ├── xor.py           # Delayed XOR classification
│   │   ├── mc.py            # Memory Capacity analysis
│   │   └── lyapunov_task.py # Lyapunov exponent calculation
│   └── baselines_rc/
│       └── esn.py           # Echo State Network baseline
├── configs/
│   ├── production_config.yaml   # Full sweep (10×5×20 = 1000 trials)
│   └── validation_config.yaml   # Quick validation (2×2×2 = 8 trials)
├── tests/
│   ├── test_repro.py        # Reproducibility tests
│   └── test_leakage.py      # Data leakage prevention tests
├── deploy.sh                # Docker deployment script
├── Dockerfile               # Containerized environment
└── docker-compose.yml       # Resource-limited execution
```

## Key Features

### 1. Biological Realism
- **Hodgkin-Huxley dynamics**: Full Na⁺, K⁺, leak, and A-current channels
- **Dale's principle**: 80% excitatory, 20% inhibitory neurons
- **Poisson input encoding**: Realistic spike train generation (10-150 Hz)

### 2. Scientific Rigor
- **Attractor-based Lyapunov**: 1000-step warm-up before trajectory branching
- **Blocked Cross-Validation**: 10-step temporal gap to prevent data leakage
- **Deterministic RNG**: 4 independent streams (network, input mask, input, readout)
- **Git hash tracking**: All results linked to exact code version

### 3. Computational Efficiency
- **60-core parallelization**: Optimized for high-memory servers (2TB RAM)
- **Intelligent caching**: Simulation states cached by (ρ, bias, seeds, input_id)
- **Memory-efficient filtering**: Streaming exponential filter (no O(T×N) arrays)

### 4. Live Diagnostics
During execution, the system reports every 5 trials:
```
[INTERMEDIATE REPORT 245/1000]
 > Params: rho=0.10, bias=2.00
 > Bio: Firing Rate = 12.45 Hz (OK)
 > Chaos: Mean Lambda = -0.001 s^-1
 > Task Score: NARMA=1.02, XOR=0.52, MC=0.15
```

## Experimental Protocol

See `implementation_plan.md` for the complete methodology.

### Parameter Sweep
- **Spectral radius (ρ)**: [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
- **Input bias**: [0.0, 2.0, 4.0, 6.0, 8.0] pA
- **Seeds**: 20 independent trials per (ρ, bias) pair

### Benchmark Tasks
1. **NARMA-10**: Nonlinear autoregressive moving average (regression)
   - Baseline: AR(10) model
2. **Delayed XOR**: Memory-dependent classification
   - Baseline: Class priors
3. **Memory Capacity**: Linear memory retention
   - Baseline: Shuffled input
4. **Lyapunov Exponent**: Stability/chaos quantification
   - Units: s⁻¹ (per second)

### Statistical Analysis
- **Hypothesis test**: Wilcoxon signed-rank (Peak vs Neighbors)
- **Effect size**: Cohen's d
- **Significance**: α = 0.05 (Bonferroni-corrected)

## Quick Start

### Local Validation (2 minutes)
```bash
cd project_C_poisson
python validate_local.py
```

### Production Deployment (Docker)
```bash
cd project_C_poisson
./deploy.sh
```

This will:
1. Build Docker image with all dependencies
2. Run validation sweep (8 trials)
3. Run production sweep (1000 trials)
4. Run ESN baseline sweep
5. Save results to `results/*.parquet`

### Analysis
```bash
python src/run_analysis.py --results results/results_coarse_*.parquet
```

## Requirements

- **Python**: 3.12+
- **Dependencies**: numpy, scipy, pandas, scikit-learn, joblib, pyarrow
- **Hardware (recommended)**: 32+ cores, 64GB+ RAM
- **Docker**: For reproducible deployment

## Results Schema

Each trial produces 8 rows (4 tasks × 2 metrics):

| Column | Description |
|--------|-------------|
| `rho` | Spectral radius |
| `bias` | Input bias (pA) |
| `seed_tuple_id` | Trial index (0-19) |
| `task` | NARMA / XOR / MC / Lyapunov |
| `metric` | nrmse / accuracy / capacity / lambda_sec |
| `value` | Task performance |
| `baseline` | Baseline performance |
| `improvement` | value - baseline |
| `firing_rate` | Mean firing rate (Hz) |
| `lambda_sec` | Lyapunov exponent (s⁻¹) |
| `git_hash` | Code version |

## Citation

If you use this code, please cite:

```bibtex
@software{kawiak2026reservoir,
  author = {Kawiak, Adam},
  title = {Hodgkin-Huxley Reservoir Computing: Edge of Chaos Hypothesis},
  year = {2026},
  url = {https://github.com/atkawiak/reservoir_hh}
}
```

## License

MIT License - See LICENSE file for details

## Contact

Adam Kawiak - [GitHub](https://github.com/atkawiak)
