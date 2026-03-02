# NARMA-10 Parameter Sweep Report

## Best Configuration (status=OK only)

- NRMSE: **0.6030**
- R2: **0.6364**
- Regime: R3_edge (alpha_syn=0.007114)
- bg_scale=0.25, narma_scale=4.0nA, dt_task=20.0ms, n_delays=10, ei_split=True

## Liveness Check (E-population, post-warmup)

PASS: pct_silent_E<10%, 2≤rate_E<200 Hz, CV_ISI_E≥0.5 | WARN: 10–40% silent lub rate/CV graniczne | FAIL: ≥40% silent lub runaway

| Regime | Liveness | rate_E (Hz) | pct_silent_E | CV_ISI_E |
|--------|----------|-------------|-------------|----------|
| R1_deep_stable | WARN (pct_silent_E=12% (10–40%)) | 20.3 | 12% | 0.15 |
| R2_stable | WARN (pct_silent_E=11% (10–40%)) | 19.5 | 11% | 0.16 |
| R3_edge | WARN (pct_silent_E=12% (10–40%)) | 19.4 | 12% | 0.17 |
| R4_weak_chaos | WARN (pct_silent_E=12% (10–40%)) | 18.9 | 12% | 0.20 |
| R5_strong_chaos | WARN (pct_silent_E=13% (10–40%)) | 18.2 | 13% | 0.20 |

> **Nota metodologiczna (Korekta 2):** parametry dobrane na reżimach: R1_deep_stable, R2_stable, R3_edge, R4_weak_chaos, R5_strong_chaos. Wyniki dla tych samych reżimów są lekko uprzywilejowane.

## NRMSE by Regime x ei_split

| Regime | all | E/I |
|---|---|---|
| R1_deep_stable | 0.7537 | 0.7104 |
| R2_stable | 0.7492 | 0.7092 |
| R3_edge | 0.7444 | 0.7057 |
| R4_weak_chaos | 0.7496 | 0.7072 |
| R5_strong_chaos | 0.7474 | 0.7063 |

## Sweep Configuration

- K_total=800, K_warmup=100, K_train=500, K_test=100
- bg_scales: [0.25]
- narma_scales_nA: [4.0]
- dt_task_ms: [20.0]
- n_delays: [10]
- ei_split: [False, True]
- n_seeds: 5
- Total evaluations: 50 (50 OK)

