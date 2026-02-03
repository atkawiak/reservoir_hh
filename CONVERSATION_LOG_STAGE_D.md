# Conversation History & Progress Log: Stage C & D

## Project: Edge of Chaos Verification in Hodgkin-Huxley Neural Networks
**Date:** 2026-02-03
**Status:** Stage D (Rigorous Implementation) Initiated

### 1. Summary of Insights & Problems Identified
During Stage B/C, we observed that the "Stable" state was paradoxically outperforming the "Edge" state in XOR tasks. Multi-agent brainstorming and Research Engineer analysis identified the following "Success Blockers":
- **Overdriving:** High input gain (10.0) and high Poisson rates (80Hz) were forcing the network into a driver-responder mode, bypassing recurrent dynamics.
- **Lack of Sparsity:** Uniform input across all neurons synchronized the reservoir, reducing linear separability.
- **Task Mismatch:** XOR Delay=1 was too simple and favored high-fidelity transmission (Stable) over complex memory (Edge).

### 2. Actions Taken
- **Model Refactoring:** Updated `hh_model.py` to support **Spatial Sparse Input**. `W_in @ s_in_trace` ensures each neuron gets a unique (or no) input signal.
- **E/I Balance (Dale's Law):** Implemented separate scaling for inhibitory weights (`inh_scale=4.0`). This enables "Balanced Chaos" and prevents saturation at high spectral radii.
- **Deep Research Review:** Verified literature parameters:
    - Input Sparsity: 20-30%
    - Symbol Duration: 50 ms
    - Rates: Background 2Hz, Active 45Hz.
    - Input Gain: Optimized for perturbation ($\sim 3.0$).

### 3. Repository Restructuring
- Created `stage_D_rigorous/` as a clean slate for the final verification.
- Moved all experimental scripts to `project_C_poisson/`.
- Updated `.gitignore` to exclude large result files (`.csv`, `.log`).

### 4. Current State
The code in `stage_D_rigorous/` is ready for execution. It implements:
- **Delayed XOR (Delay 1, 2, 3)**
- **Memory Capacity (MC)**
- **NARMA-10** (to be tested)

### 5. Git Status
All changes committed and pushed to `https://github.com/atkawiak/reservoir_hh`.

**Note:** The agent is instructed NOT to run the code until further confirmation, only to maintain the repository state and conversation history.
