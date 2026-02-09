# Reference: HH Reservoir Triplet Generation Pipeline
## State: Stable, Edge of Chaos (Λ≈0), Chaotic

This document captures the distilled knowledge and methodology for the "Surgical Precision" calibration of Hodgkin-Huxley (HH) resonators.

---

### 1. Core Methodology: Hybrid Calibration
To precisely identify the Edge of Chaos (EoC) in HH neurons, which exhibit non-monotonic Lyapunov landscapes, we use a hybrid approach:

1.  **Dense Grid Scan (100 points)**: Maps the Lyapunov exponent (Λ) across the inhibitory scaling range (0.5 to 12.0). 
    - **Key Detail**: We use a high baseline energy (`target_spectral_radius=4.0`) to ensure the system reaches chaos before being "tamed" by inhibition.
2.  **Peak Detection**: Identify the point of maximum Λ (Max Chaos).
3.  **Brentq Refinement**: Perform a root-finding search for Λ=0 in the narrow interval to the right of the peak.
4.  **Regime Assignment**:
    - **CHAOTIC**: Set at `Peak_Inh` (Maximum Λ).
    - **EDGE**: Set at `Λ=0` (Zero crossing).
    - **STABLE**: Set at `Edge_Inh + 4.0` (Deep stability).

---

### 2. Experimental Proof (N=100, 30 seeds)
The rigorous study on 30 independent realizations for N=100 confirmed:
- **NARMA-10 and XOR Accuracy**: Statistically highest at the **EDGE** of chaos.
- **Memory Capacity (MC)**: Decreases monotonically as the system enters chaos.
- **Complexity (Kernel Rank/Separation)**: Increases monotonically with chaos.

**Key Finding**: The "Edge of Chaos" is the unique point where the system balances enough memory with high enough complexity to solve nonlinear temporal tasks.

---

### 3. Files in the Reference Pipeline
- `generate_reservoir_triplets.py`: The main execution script.
- `task_config.yaml`: Configuration for HH neurons and benchmarks.
- `src/`: Core library containing `Reservoir` and `NeuronGroupHH`.

---

### 4. How to Use
To generate a new set of triplets:
```bash
py generate_reservoir_triplets.py [N_NEURONS] [N_SEEDS]
```
Example for production:
```bash
py generate_reservoir_triplets.py 500 30
```

---

### 5. Architectural Diagram (Logical Flow)
1. Initialize Reservoir (ρ=4.0)
2. Scan Inh (0.5 -> 12.0) -> calculate_lyapunov(100 times)
3. Find Λ=0 via brentq zoom.
4. Scale regimes: Stable (+4), Edge (+0), Chaotic (at Peak).
5. Run benchmarks (NARMA, XOR, MC, Rank) for all three.
6. Output to CSV.
