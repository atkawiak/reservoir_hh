# NARMA-10 Benchmark Report

Bundle: `bundles/bundle_seed_1500`

| Regime | α_syn | n_seeds | NRMSE mean±std | R² mean |
|--------|-------|---------|----------------|---------|
| R1_deep_stable | 0.011705 | 5 | 0.8080±0.0816 | 0.3405 |
| R2_stable | 0.020484 | 5 | 0.8048±0.0750 | 0.3467 |
| R3_edge | 0.029263 | 5 | 0.8000±0.0844 | 0.3529 |
| R4_weak_chaos | 0.040968 | 5 | 0.7924±0.0742 | 0.3667 |
| R5_strong_chaos | 0.058526 | 5 | 0.8035±0.0847 | 0.3472 |

Config: K_total=800, K_warmup=100, K_train=500, K_test=100, dt_task_ms=10.0, narma_scale_nA=1.0
