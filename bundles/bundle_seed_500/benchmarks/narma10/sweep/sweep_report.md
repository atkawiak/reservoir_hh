# NARMA-10 Parameter Sweep Report

## Best Configuration (status=OK only)

- NRMSE: **0.6451**
- R2: **0.5838**
- Regime: R3_edge (alpha_syn=0.007114)
- bg_scale=0.25, narma_scale=1.0nA, dt_task=20.0ms, n_delays=10, ei_split=False

## Liveness Check (E-population, post-warmup)

PASS: pct_silent_E<10%, 2≤rate_E<200 Hz, CV_ISI_E≥0.5 | WARN: 10–40% silent lub rate/CV graniczne | FAIL: ≥40% silent lub runaway

| Regime | Liveness | rate_E (Hz) | pct_silent_E | CV_ISI_E |
|--------|----------|-------------|-------------|----------|
| R3_edge | WARN (pct_silent_E=11% (10–40%)) | 19.1 | 11% | 0.17 |

> **Nota metodologiczna (Korekta 2):** parametry dobrane na reżimach: R3_edge. Wyniki dla tych samych reżimów są lekko uprzywilejowane.

## NRMSE by Regime x narma_scale_nA

| Regime | 0.5nA | 1.0nA | 2.0nA | 4.0nA |
|---|---|---|---|---|
| R3_edge | 0.8446 | 0.7606 | 0.7427 | 0.7203 |

## Sweep Configuration

- K_total=800, K_warmup=100, K_train=500, K_test=100
- bg_scales: [0.25]
- narma_scales_nA: [0.5, 1.0, 2.0, 4.0]
- dt_task_ms: [20.0]
- n_delays: [10]
- ei_split: [False]
- n_seeds: 3
- Total evaluations: 12 (12 OK)

