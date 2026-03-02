# NARMA-10 Parameter Sweep Report

## Best Configuration (status=OK only)

- NRMSE: **0.6366**
- R2: **0.5948**
- Regime: R3_edge (alpha_syn=0.007114)
- bg_scale=0.25, narma_scale=4.0nA, dt_task=20.0ms, n_delays=10, ei_split=False

## Liveness Check (E-population, post-warmup)

PASS: pct_silent_E<10%, 2≤rate_E<200 Hz, CV_ISI_E≥0.5 | WARN: 10–40% silent lub rate/CV graniczne | FAIL: ≥40% silent lub runaway

| Regime | Liveness | rate_E (Hz) | pct_silent_E | CV_ISI_E |
|--------|----------|-------------|-------------|----------|
| R1_deep_stable | WARN (pct_silent_E=12% (10–40%)) | 20.2 | 12% | 0.16 |
| R2_stable | WARN (pct_silent_E=12% (10–40%)) | 20.0 | 12% | 0.16 |
| R3_edge | WARN (pct_silent_E=11% (10–40%)) | 19.1 | 11% | 0.17 |
| R4_weak_chaos | WARN (pct_silent_E=12% (10–40%)) | 19.1 | 12% | 0.19 |
| R5_strong_chaos | WARN (pct_silent_E=13% (10–40%)) | 18.1 | 13% | 0.20 |

> **Nota metodologiczna (Korekta 2):** parametry dobrane na reżimach: R1_deep_stable, R2_stable, R3_edge, R4_weak_chaos, R5_strong_chaos. Wyniki dla tych samych reżimów są lekko uprzywilejowane.

## Sweep Configuration

- K_total=800, K_warmup=100, K_train=500, K_test=100
- bg_scales: [0.25]
- narma_scales_nA: [4.0]
- dt_task_ms: [20.0]
- n_delays: [10]
- ei_split: [False]
- n_seeds: 5
- Total evaluations: 25 (25 OK)

