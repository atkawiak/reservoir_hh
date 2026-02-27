# NARMA-10 Parameter Sweep Report

## Best Configuration (status=OK only)

- NRMSE: **0.8282**
- R2: **0.3141**
- Regime: R3_edge (alpha_syn=0.007114)
- bg_scale=1.0, narma_scale=1.0nA, dt_task=50.0ms, n_delays=10, ei_split=False

## Liveness Check (E-population, post-warmup)

PASS: pct_silent_E<10%, 2≤rate_E<200 Hz, CV_ISI_E≥0.5 | WARN: 10–40% silent lub rate/CV graniczne | FAIL: ≥40% silent lub runaway

| Regime | Liveness | rate_E (Hz) | pct_silent_E | CV_ISI_E |
|--------|----------|-------------|-------------|----------|
| R3_edge | WARN (CV_ISI_E=0.41 < 0.5) | 14.7 | 1% | 0.41 |

> **Nota metodologiczna (Korekta 2):** parametry dobrane na reżimach: R3_edge. Wyniki dla tych samych reżimów są lekko uprzywilejowane.

## NRMSE by Regime x dt_task_ms

| Regime | 10.0ms | 20.0ms | 50.0ms |
|---|---|---|---|
| R3_edge | 0.9612 | 0.8773 | 0.8282 |

## Sweep Configuration

- K_total=800, K_warmup=100, K_train=500, K_test=100
- bg_scales: [1.0]
- narma_scales_nA: [1.0]
- dt_task_ms: [10.0, 20.0, 50.0]
- n_delays: [10]
- ei_split: [False]
- n_seeds: 1
- Total evaluations: 3 (3 OK)

