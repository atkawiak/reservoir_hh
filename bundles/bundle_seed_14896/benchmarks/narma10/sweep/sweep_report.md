# NARMA-10 Parameter Sweep Report

## Best Configuration

- NRMSE: **0.7970**
- R2: **0.3648**
- Regime: R4_weak_chaos (alpha_syn=0.047389)
- bg_scale=1.0, narma_scale=2.0nA, dt_task=10.0ms, n_delays=10, ei_split=True

## NRMSE by Regime x narma_scale_nA

| Regime | 1.0nA | 2.0nA |
|---|---|---|
| R1_deep_stable | 0.9194 | 0.8241 |
| R2_stable | 0.9187 | 0.8135 |
| R3_edge | 0.9538 | 0.8168 |
| R4_weak_chaos | 0.9137 | 0.8125 |
| R5_strong_chaos | 0.9109 | 0.8272 |

## NRMSE by Regime x ei_split

| Regime | all | E/I |
|---|---|---|
| R1_deep_stable | 0.9068 | 0.8368 |
| R2_stable | 0.8988 | 0.8334 |
| R3_edge | 0.8964 | 0.8743 |
| R4_weak_chaos | 0.8946 | 0.8316 |
| R5_strong_chaos | 0.9036 | 0.8345 |

## Sweep Configuration

- K_total=800, K_warmup=100, K_train=500, K_test=100
- bg_scales: [1.0]
- narma_scales_nA: [1.0, 2.0]
- dt_task_ms: [10.0]
- n_delays: [10]
- ei_split: [False, True]
- n_seeds: 1
- Total evaluations: 20

