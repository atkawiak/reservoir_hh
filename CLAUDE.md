# Reservoir Computing — Hodgkin-Huxley + STP (HH+STP LSM)

## Project overview

Liquid State Machine (LSM) research using biologically-realistic Hodgkin-Huxley neurons with Short-Term Plasticity (STP) synapses, simulated in Brian2. Goal: evaluate reservoir computing performance (NARMA-10 benchmark) across 5 edge-of-chaos regimes.

**Key metric**: NRMSE (lower = better; 0=perfect, 1=mean baseline). R² also reported.

## Repository structure

```
gen/                   ← main code (Python module)
  benchmark_narma.py   ← single-config NARMA-10 benchmark runner
  sweep_narma.py       ← multi-param sweep runner (parallel, bg_scale, narma_scale, dt)
  brian_smoke.py       ← Brian2 network builder from frozen bundle
  state_readout.py     ← spike count extraction + E/I split
  ridge.py             ← ridge regression with KFold CV
  narma.py             ← NARMA-10 task generator
  regimes.py           ← regime definitions (R1-R5: deep stable → strong chaos)
  tests/test_all.py    ← pytest test suite (138 tests)
  ...
bundles/               ← frozen LSM network bundles (HH+STP, different seeds/sizes)
  bundle_seed_100/     ← N=100 network (E/I split, 5 regimes)
  bundle_seed_500/     ← N=500 network (E/I split, 5 regimes)
  bundle_seed_14896/   ← another N=? network
  ...
docs/                  ← LaTeX/PDF documentation
notes/                 ← research notes
```

## 5 Regimes (edge-of-chaos calibration)

| Regime | Name           | α_syn characteristic       |
|--------|----------------|-----------------------------|
| R1     | deep_stable    | lowest (most stable)        |
| R2     | stable         |                             |
| R3     | edge           | edge-of-chaos (optimal RC)  |
| R4     | weak_chaos     |                             |
| R5     | strong_chaos   | highest (most chaotic)      |

## Key methodology decisions (from Claude sessions)

### Liveness criteria (post-warmup)
- **PASS**: pct_silent_E < 10% AND rate_E ∈ [2, 200] Hz AND CV_ISI_E ≥ 0.5
- **WARN**: pct_silent_E 10-40% OR rate_E < 2 Hz OR CV_ISI_E < 0.5
- **FAIL**: pct_silent_E ≥ 40% OR rate_E ≥ 200 Hz OR no spikes at all
- Mean spikes/bin criterion: > 0.2 for valid signal

### Tuning bias avoidance (Korekta 2)
- Parameters tuned on **R2 + R4** (indices 1, 3), NOT R3
- R3 held out as validation / final report
- Always note in report which regimes were used for tuning

### Fixed seeds for reproducibility (Korekta 3)
- Use 3-5 fixed seeds in all sweeps (not random)
- Default: seeds 42, 43, 44

### 4-step methodology
- **Krok 0 (dt sweep)**: Find optimal dt_task_ms (criterion: spikes/bin > 0.2)
  - Result for bundle_seed_500: **dt=20ms** (spikes/bin=0.279)
  - dt=10ms fails (0.153), dt=50ms too regular (CV_ISI_E=0.15)
- **Krok 1 (bg_scale sweep)**: Tune Poisson background on R2+R4, 3 seeds, dt=20ms
  - bg_scales tested: [0.1, 0.25, 0.5, 1.0]
  - Select lowest all-PASS bg_scale
- **Krok 2 (narma_scale sweep)**: Tune NARMA injection strength on R3, 3 seeds
  - narma_scales tested: [0.5, 1.0, 2.0, 4.0] nA
- **Krok 3 (full benchmark)**: All 5 regimes, 5 seeds, optimal (bg_scale, narma_scale), dt=20ms

## How to run

### Run single benchmark
```bash
python -m gen.benchmark_narma bundles/bundle_seed_500 \
  --regime 2 --bg-scale 0.5 --ei-split --dt-task-ms 20
```

### Run parameter sweep (parallel)
```bash
# Krok 1: bg_scale sweep (tune on R2+R4)
python -m gen.sweep_narma bundles/bundle_seed_500 \
  --regime 1 3 --seeds 3 --bg-scales 0.1 0.25 0.5 1.0 \
  --narma-scales-nA 1.0 --dt-task-ms 20 --jobs -1

# Krok 2: narma_scale sweep (on R3 only)
python -m gen.sweep_narma bundles/bundle_seed_500 \
  --regime 2 --seeds 3 --bg-scales <optimal_from_krok1> \
  --narma-scales-nA 0.5 1.0 2.0 4.0 --dt-task-ms 20 --jobs -1

# Krok 3: full benchmark
python -m gen.sweep_narma bundles/bundle_seed_500 \
  --regime 0 1 2 3 4 --seeds 5 \
  --bg-scales <optimal> --narma-scales-nA <optimal> \
  --dt-task-ms 20 --ei-split --jobs -1
```

### --jobs flag
- `--jobs 1`: sequential (default)
- `--jobs N`: N parallel Brian2 workers (ProcessPoolExecutor)
- `--jobs -1`: auto = 80% of CPU cores

### Run tests
```bash
pytest gen/tests/test_all.py -v
```

## Current experiment state (as of 2026-02-27)

- **bundle_seed_500** (N=500 HH+STP network) is the primary test network
- **Krok 0 completed**: dt=20ms selected for bundle_seed_500
- **Krok 1 in progress**: bg_scale sweep (R2+R4, 3 seeds, 22 parallel workers)
  - Command: `python -m gen.sweep_narma bundles/bundle_seed_500 --regime 1 3 --seeds 3 --narma-seed 42 --K-total 800 --K-warmup 100 --K-train 500 --K-test 100 --bg-scales 0.1 0.25 0.5 1.0 --narma-scales-nA 1.0 --dt-task-ms 20 --n-delays 10 --jobs -1`
  - 24 Brian2 sims, ~30-40 min on 28-core laptop
- **Krok 2**: narma_scale sweep on R3, pending Krok 1 results
- **Krok 3**: full 5-regime benchmark, pending Krok 2

## Key design rules

1. **Frozen bundles**: Network weights/topology never change during experiments (runtime-only sweeps)
2. **No data leakage**: Standardization fitted on train set only, applied to test
3. **Separate E/I readout**: `--ei-split` flag uses separate z-scored E and I spike count columns
4. **Brian2 caching**: In sequential mode, spike trains cached across readout configs
5. **Process isolation**: Each parallel Brian2 job runs in separate process (different PID = independent Brian2 global state)
6. **Denser ridge alpha grid**: 21-point logspace from 1e-6 to 1e4

## Brian2 simulation parameters (bundle_seed_500)

| Param       | Value  | Notes                             |
|-------------|--------|-----------------------------------|
| N neurons   | 500    | 400E + 100I                       |
| K_warmup    | 100    | warmup bins (not used for ridge)  |
| K_train     | 500    | training bins                     |
| K_test      | 100    | test bins                         |
| dt_task_ms  | 20     | bin width (from Krok 0)           |
| bg_scale    | TBD    | from Krok 1                       |
| narma_scale | TBD    | from Krok 2                       |

## Previous N=100 results (bundle_seed_100)

NRMSE ~0.79-0.82 across 5 regimes with narma_scale=2.0, ei_split=True, dt=10ms.
Regime separation ~0.03 NRMSE (R1→R3 direction).

## N=500 preliminary results (dt=10ms, no bg tuning)

NRMSE ~0.842-0.859 — worse than N=100 (Poisson background too dominant at dt=10ms).
After switching to dt=20ms: NRMSE ~0.877 on R3 (single seed, narma_scale=1.0, bg=1.0).
Expecting improvement after Krok 1-2 tuning.
