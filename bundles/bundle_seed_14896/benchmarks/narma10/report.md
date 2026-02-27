# NARMA-10 Benchmark Report

Bundle: `bundles/bundle_seed_14896`

| Regime | α_syn | n_seeds | NRMSE mean±std | R² mean |
|--------|-------|---------|----------------|---------|
| R1_deep_stable | 0.013540 | 5 | 0.8146±0.0768 | 0.3306 |
| R2_stable | 0.023695 | 5 | 0.7999±0.0923 | 0.3517 |
| R3_edge | 0.033849 | 5 | 0.8014±0.0868 | 0.3502 |
| R4_weak_chaos | 0.047389 | 5 | 0.7956±0.0941 | 0.3582 |
| R5_strong_chaos | 0.067699 | 5 | 0.7897±0.0882 | 0.3686 |

Config: K_total=800, K_warmup=100, K_train=500, K_test=100, dt_task_ms=10.0, narma_scale_nA=1.0
