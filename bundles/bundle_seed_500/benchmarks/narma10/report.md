# NARMA-10 Benchmark Report

Bundle: `bundles/bundle_seed_500`

| Regime | α_syn | n_ok/total | NRMSE mean±std | R² mean |
|--------|-------|------------|----------------|---------|
| R1_deep_stable | 0.002845 | 1/1 | 0.8588±0.0000 | 0.2625 |
| R2_stable | 0.004980 | 1/1 | 0.8500±0.0000 | 0.2775 |
| R3_edge | 0.007114 | 1/1 | 0.8432±0.0000 | 0.2890 |
| R4_weak_chaos | 0.009959 | 1/1 | 0.8418±0.0000 | 0.2914 |
| R5_strong_chaos | 0.014227 | 1/1 | 0.8444±0.0000 | 0.2870 |

Config: K_total=800, K_warmup=100, K_train=500, K_test=100, dt_task_ms=10.0, narma_scale_nA=2.0, bg_scale=1.0, ei_split=True
