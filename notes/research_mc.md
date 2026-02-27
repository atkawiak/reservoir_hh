# R3: Memory Capacity — Research Notes (Optional Benchmark)

## Definition

Linear Memory Capacity (MC) measures a reservoir's ability to
reconstruct past inputs from current state:

```
MC = Σ_{d=1}^{D_max} r²(d)
```

where r²(d) is the squared Pearson correlation between:
- True: u(k - d)  (input delayed by d steps)
- Predicted: y_d(k) = w_d^T · x(k)  (linear readout for delay d)

## Properties

- MC ≤ N (upper bound = number of reservoir nodes)
- MC measures linear memory only
- Higher MC → better at remembering past inputs
- Edge-of-chaos typically maximizes MC (Bertschinger & Natschläger 2004)

## Test Protocol

1. Generate random input u(k) ~ U[-1, 1] or N(0, 1), length K
2. For each delay d = 1, 2, ..., D_max:
   a. Target: y_d(k) = u(k - d)
   b. Train ridge regression: x(k) → y_d(k)
   c. Compute r²(d) on test set
3. MC = sum of all r²(d)
4. D_max typically = 2*N or until r²(d) < threshold (e.g., 0.01)

## Relevance to Our Project

- Complementary to NARMA10 (which tests nonlinear processing + memory)
- MC is easier to interpret: higher = more memory
- Can validate that edge-of-chaos regime has highest MC
- Lower priority than NARMA10 for initial implementation

## Implementation (Future)

- Reuse state_readout.py for spike count extraction
- Reuse ridge.py for per-delay regression
- New: gen/benchmark_mc.py with MC-specific logic
- Output: MC value + MC profile plot (r² vs delay)

## References

- Jaeger (2001): short-term memory capacity in ESN
- Bertschinger & Natschläger (2004): MC at edge of chaos
- Dambre et al. (2012): information processing capacity
