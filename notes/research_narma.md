# R1: NARMA10 Benchmark — Research Notes

## Canonical Equation

NARMA-10 (Nonlinear Auto-Regressive Moving Average, order 10):

```
y(k+1) = α·y(k) + β·y(k)·Σ_{i=0}^{9} y(k-i) + γ·u(k-9)·u(k) + δ
```

Standard parameters: α=0.3, β=0.05, γ=1.5, δ=0.1

## Input Signal

- u(k) ~ Uniform[0, 0.5]  (most common in literature)
- Some papers use U[0, 0.2] — we use **U[0, 0.5]** for canonical comparison
- Generated with deterministic RNG seed for reproducibility

## Initial Conditions

- y(k) = 0 for k < 0  (or first 10 steps zeroed)
- Discard initial transient (warmup): first 100–200 steps typically

## Divergence Risk

- NARMA-10 with standard params can diverge for certain input ranges
- With u ∈ [0, 0.5], output stays bounded (typical range ~0.15–1.0)
- Must verify: if any y(k) > 10 or NaN → flag as diverged

## Sequence Length

- Literature uses 1000–5000 total steps
- Our plan: K_total = K_train + K_test (e.g., 3000 total)
- K_train = 2000, K_test = 1000 (after discarding warmup)
- Warmup (discarded): first ~200 steps

## Metric: NRMSE

Normalized Root Mean Squared Error:

```
NRMSE = sqrt( MSE(y_pred, y_true) / var(y_true) )
         = RMSE / σ_y
```

- NRMSE = 0 → perfect prediction
- NRMSE = 1 → predicts as well as mean baseline
- NRMSE > 1 → worse than mean
- Good reservoir: NRMSE < 0.4 typically
- State-of-art ESN: NRMSE ~0.05–0.2

## Known Pitfalls (from Kodali et al. 2024)

1. Different parameter sets across literature make comparison hard
2. Must report exact (α, β, γ, δ) and u range
3. Must distinguish trainable vs. fixed parameters
4. Must separate reservoir cost from readout cost

## Implementation Plan for Our Pipeline

1. Generate u[0..K-1] from seed → deterministic
2. Compute y[0..K-1] recursively (pure NumPy)
3. Discretize: each NARMA step = ΔT_task ms of Brian2 time
4. Map u(k) → I_narma(t) current injection (scaled, via TimedArray)
5. After Brian2 sim: extract spike counts per bin → state X[k]
6. Ridge regression: X_train → y_train, evaluate on X_test → y_test
7. Report NRMSE

## References

- Atiya & Parlos (2000): original NARMA definition
- Kodali et al. (2024): "Sustainable NARMA-10 Benchmarking for Quantum RC" (arXiv:2510.25183)
- "Reservoir Computing Benchmarks: a tutorial review and critique" (arXiv:2405.06561)
