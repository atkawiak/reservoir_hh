"""
Automatic reservoir regime finder (build-only, no Brian2).

ETAP A: Builds ONE frozen E/I Erdos-Renyi reservoir (topology + raw weights +
STP params) and finds 5 operating regimes (super_stable -> chaotic) by tuning
the scaling factor alpha on the SAME frozen weight matrix.

  lambda_proxy = log(rho) is a linearized proxy for ordering regimes.
  It is NOT a true Lyapunov exponent of the HH+STP dynamics.
  True Lyapunov will be computed in ETAP B (Brian2 simulation) via
  trajectory divergence / tangent dynamics.

ETAP B (separate module, Brian2): validates actual dynamics per regime:
  - firing rates E/I, CV ISI, synchronization
  - trajectory divergence (true Lyapunov)
  - memory capacity / separability (RC metrics)

Outputs:
  - 5 x .npz files with full connectivity + STP params + build metadata
  - 1 x .csv summary table

Key design rules:
  - Topology and raw weights are generated ONCE per seed.
  - Regime tuning only rescales the SAME reservoir (no re-sampling).
  - alpha_final is per-seed, not a global constant (depends on rho_base).

Sanity checks (run once per seed):
  - Scaling linearity: rho(k*alpha) ~ k*rho(alpha) for frozen weights.
  - NNZ consistency: nnz_raw (sum of per-type synapses) vs nnz_csr (after
    CSR deduplication). Large discrepancy = duplicate (post,pre) pairs.
"""

import os
import csv
import time
import numpy as np
from scipy import sparse
from numba import njit

# ═══════════════════════════════════════════════════════════════════════════════
# 1. REGIME DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

REGIMES = {
    "R1_super_stable":  {"rho_lo": 0.35, "rho_hi": 0.55},
    "R2_stable":        {"rho_lo": 0.65, "rho_hi": 0.85},
    "R3_near_critical": {"rho_lo": 0.90, "rho_hi": 1.00},
    "R4_edge_of_chaos": {"rho_lo": 1.02, "rho_hi": 1.12},
    "R5_chaotic":       {"rho_lo": 1.20, "rho_hi": 1.40},
}

# Stable, deterministic mapping (NO hash()).
SYN_TYPES = ("EE", "EI", "IE", "II")
SYN_ID = {name: i for i, name in enumerate(SYN_TYPES)}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEFAULT SYNAPSE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = dict(
    N_total=135,
    frac_E=0.8,

    # ER connection probabilities
    p_EE=0.3,
    p_EI=0.2,
    p_IE=0.4,
    p_II=0.1,

    # Base amplitude scales (magnitudes). Signs applied by syn type.
    A_base_EE=3.0,     # arbitrary units (kept unitless here)
    A_base_EI=6.0,
    A_base_IE=11.2,    # inhibitory magnitudes
    A_base_II=11.2,

    # STP means
    U_EE=0.50,  D_EE=1100.0,  F_EE=50.0,    tau_I_EE=3.0,
    U_EI=0.05,  D_EI=125.0,   F_EI=1200.0,  tau_I_EI=3.0,
    U_IE=0.25,  D_IE=700.0,   F_IE=20.0,    tau_I_IE=6.0,
    U_II=0.32,  D_II=144.0,   F_II=60.0,    tau_I_II=6.0,

    # Delays (ms)
    delay_EE=1.5, delay_EI=0.8, delay_IE=0.8, delay_II=0.8,
    delay_jitter=0.1,

    # Balance constraints
    balance_lo=0.6,
    balance_hi=1.4,

    # Sweep
    sweep_trials=600,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. NUMBA: POWER ITERATION ON CSR
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _power_iteration(data, indices, indptr, n, n_iter=80):
    """
    Spectral radius approximation via power iteration on CSR matrix.
    data/indices/indptr define matrix A in CSR row-major form.
    """
    x = np.ones(n, dtype=np.float64)
    x /= np.sqrt((x * x).sum())

    for _ in range(n_iter):
        y = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                s += data[k] * x[indices[k]]
            y[i] = s
        norm = np.sqrt((y * y).sum())
        if norm < 1e-15:
            return 0.0
        x = y / norm

    # One more multiply to estimate |A x|
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        y[i] = s
    rho = np.sqrt((y * y).sum())
    return rho

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RESERVOIR BUILDER (FROZEN TOPOLOGY + RAW WEIGHTS + STP)
# ═══════════════════════════════════════════════════════════════════════════════

class ReservoirBuilder:
    """
    Build reservoir once:
      - E/I split
      - ER edges for EE/EI/IE/II
      - raw positive magnitudes for each synapse population (gamma)
      - STP params (U/D/F/delay/tauI) for each synapse population

    Then allow fast rescaling:
      W(alpha, A_scale_E, A_scale_I) without re-sampling.
    """

    def __init__(self, seed=42, **kwargs):
        self.params = {**DEFAULT_PARAMS, **kwargs}
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        p = self.params
        self.N = int(p["N_total"])
        self.N_E = int(self.N * p["frac_E"])
        self.N_I = self.N - self.N_E

        # Deterministic E/I assignment
        idx = np.arange(self.N, dtype=np.int32)
        self.rng.shuffle(idx)
        self.idx_E = np.sort(idx[: self.N_E])
        self.idx_I = np.sort(idx[self.N_E :])

        self.is_E = np.zeros(self.N, dtype=np.bool_)
        self.is_E[self.idx_E] = True

        self.edges = {}      # per type: pre/post
        self.raw_mag = {}    # per type: raw positive magnitudes
        self.stp = {}        # per type: U/D/F/delay/tauI

        self._build_all()

        # Build a CSR "template" once: (rows, cols) fixed.
        self._csr_template = self._build_csr_template()

    # ──────────────────────────────────────────────────────────────────────
    # Build phase
    # ──────────────────────────────────────────────────────────────────────

    def _build_all(self):
        self._build_edges_all()
        self._build_raw_magnitudes_all()
        self._build_stp_all()

    def _build_edges_all(self):
        p = self.params
        syn_specs = [
            ("EE", self.idx_E, self.idx_E, p["p_EE"], True),
            ("EI", self.idx_E, self.idx_I, p["p_EI"], False),
            ("IE", self.idx_I, self.idx_E, p["p_IE"], False),
            ("II", self.idx_I, self.idx_I, p["p_II"], True),
        ]

        for name, src_idx, tgt_idx, prob, same_pop in syn_specs:
            rng_edge = np.random.default_rng(self.seed * 1000 + SYN_ID[name])

            pre_list = []
            post_list = []
            n_src = len(src_idx)
            n_tgt = len(tgt_idx)

            for si in range(n_src):
                for ti in range(n_tgt):
                    if same_pop and si == ti:
                        continue
                    if rng_edge.random() < prob:
                        pre_list.append(src_idx[si])
                        post_list.append(tgt_idx[ti])

            pre = np.array(pre_list, dtype=np.int32)
            post = np.array(post_list, dtype=np.int32)
            self.edges[name] = {"pre": pre, "post": post, "n_syn": pre.size}

    def _build_raw_magnitudes_all(self):
        p = self.params
        # One RNG stream for raw magnitudes (deterministic)
        rng_w = np.random.default_rng(self.seed * 200 + 7)

        for name in SYN_TYPES:
            n_syn = self.edges[name]["n_syn"]
            if n_syn == 0:
                self.raw_mag[name] = np.zeros(0, dtype=np.float64)
                continue

            A_base = float(p[f"A_base_{name}"])
            # Gamma(shape=1, scale=A_base) => exponential-like
            mag = rng_w.gamma(shape=1.0, scale=A_base, size=n_syn).astype(np.float64)
            self.raw_mag[name] = mag

    def _build_stp_all(self):
        p = self.params
        for name in SYN_TYPES:
            n_syn = self.edges[name]["n_syn"]
            rng_stp = np.random.default_rng(self.seed * 300 + 19 + SYN_ID[name])

            if n_syn == 0:
                self.stp[name] = dict(
                    U=np.zeros(0), D=np.zeros(0), F=np.zeros(0),
                    delay=np.zeros(0), tau_I=np.zeros(0)
                )
                continue

            U_mean = float(p[f"U_{name}"])
            D_mean = float(p[f"D_{name}"])
            F_mean = float(p[f"F_{name}"])

            U = rng_stp.normal(U_mean, 0.5 * U_mean, size=n_syn)
            U = np.clip(U, 1e-3, 1.0)

            D = rng_stp.normal(D_mean, 0.5 * D_mean, size=n_syn)
            D = np.clip(D, 1.0, None)

            F = rng_stp.normal(F_mean, 0.5 * F_mean, size=n_syn)
            F = np.clip(F, 0.0, None)

            delay_mean = float(p[f"delay_{name}"])
            jitter = float(p["delay_jitter"])
            delay = rng_stp.normal(delay_mean, jitter, size=n_syn)
            delay = np.clip(delay, 0.1, None)

            tau_I = np.full(n_syn, float(p[f"tau_I_{name}"]), dtype=np.float64)

            self.stp[name] = dict(U=U, D=D, F=F, delay=delay, tau_I=tau_I)

    def _build_csr_template(self):
        """
        Build CSR structure once: indices/indptr fixed for the whole reservoir.
        We build an empty CSR with zeros, then later we will overwrite .data.
        """
        rows = []
        cols = []
        for name in SYN_TYPES:
            e = self.edges[name]
            if e["n_syn"] == 0:
                continue
            rows.append(e["post"])
            cols.append(e["pre"])

        if not rows:
            raise ValueError("No edges generated; check p_* probabilities.")

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)

        data = np.zeros(rows.size, dtype=np.float64)
        W = sparse.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        return W

    # ──────────────────────────────────────────────────────────────────────
    # Fast rescaling API
    # ──────────────────────────────────────────────────────────────────────

    def build_weight_csr(self, alpha=1.0, A_scale_E=1.0, A_scale_I=1.0):
        """
        Return CSR W for given scaling without resampling anything.
        """
        alpha = float(alpha)
        A_scale_E = float(A_scale_E)
        A_scale_I = float(A_scale_I)

        # Recreate data array in the same concat order as template edges.
        vals = []
        for name in SYN_TYPES:
            mag = self.raw_mag[name]
            if mag.size == 0:
                continue
            if name in ("EE", "EI"):
                sign = +1.0
                scale = A_scale_E
            else:
                sign = -1.0
                scale = A_scale_I
            vals.append(alpha * scale * sign * mag)

        data = np.concatenate(vals).astype(np.float64)

        # Copy template (cheap) and assign data
        W = self._csr_template.copy()
        W.data = data
        return W

    # ──────────────────────────────────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────────────────────────────────

    @property
    def nnz_raw(self):
        """Total synapse count before CSR deduplication."""
        return sum(self.edges[name]["n_syn"] for name in SYN_TYPES)

    def spectral_radius(self, W):
        return float(_power_iteration(W.data, W.indices, W.indptr, W.shape[0], n_iter=80))

    def ei_balance(self, W):
        d = W.data
        sum_E = d[d > 0].sum()
        sum_I = np.abs(d[d < 0]).sum()
        return float(sum_I / (sum_E + 1e-15))

    def lambda_proxy(self, rho):
        """Linearized proxy: log(rho). NOT a true Lyapunov exponent of HH+STP."""
        if rho <= 1e-15:
            return float("-inf")
        return float(np.log(rho))

    def mean_outdegree(self):
        """
        Mean outdegree based on topology only (independent of scaling).
        """
        pre_all = []
        for name in SYN_TYPES:
            pre_all.append(self.edges[name]["pre"])
        pre_all = np.concatenate(pre_all)
        deg = np.bincount(pre_all, minlength=self.N).astype(np.float64)

        mean_E = float(deg[self.is_E].mean()) if self.N_E > 0 else 0.0
        mean_I = float(deg[~self.is_E].mean()) if self.N_I > 0 else 0.0
        return mean_E, mean_I

# ═══════════════════════════════════════════════════════════════════════════════
# 5. REGIME FINDER
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeFinder:
    def __init__(self, seed=42, output_dir="regimes", **kwargs):
        self.seed = int(seed)
        self.output_dir = output_dir
        self.builder = ReservoirBuilder(seed=self.seed, **kwargs)
        os.makedirs(output_dir, exist_ok=True)

    def find_all(self):
        b = self.builder
        p = b.params

        print("Measuring base rho (alpha=1.0)...")
        W_base = b.build_weight_csr(alpha=1.0, A_scale_E=1.0, A_scale_I=1.0)
        rho_base = b.spectral_radius(W_base)
        bal_base = b.ei_balance(W_base)
        print(f"  rho_base = {rho_base:.6f}  balance={bal_base:.3f}")

        if rho_base < 1e-8:
            raise ValueError("rho_base ~ 0. Check connectivity probabilities.")

        # ── Sanity check A: scaling linearity ──
        rho_half = b.spectral_radius(b.build_weight_csr(alpha=0.5))
        rho_double = b.spectral_radius(b.build_weight_csr(alpha=2.0))
        err_half = abs(rho_half - 0.5 * rho_base) / rho_base
        err_double = abs(rho_double - 2.0 * rho_base) / rho_base
        lin_ok = err_half < 0.02 and err_double < 0.02
        print(f"\n  Linearity check:")
        print(f"    rho(0.5)={rho_half:.6f}  expected={0.5*rho_base:.6f}  err={err_half:.4f}")
        print(f"    rho(2.0)={rho_double:.6f}  expected={2.0*rho_base:.6f}  err={err_double:.4f}")
        print(f"    -> {'PASS' if lin_ok else 'WARN: non-linear scaling'}")

        # ── Sanity check B: NNZ consistency (duplicate detection) ──
        nnz_raw = b.nnz_raw
        nnz_csr = W_base.nnz
        nnz_ok = nnz_csr == nnz_raw
        print(f"\n  NNZ check: raw={nnz_raw}  csr={nnz_csr}  "
              f"{'PASS' if nnz_ok else f'WARN: {nnz_raw - nnz_csr} duplicates merged'}")

        results = []
        for regime_name, bounds in REGIMES.items():
            print(f"\n--- {regime_name} ---")
            res = self._find_one(regime_name, bounds, rho_base)
            results.append(res)

        return results

    def _find_one(self, regime_name, bounds, rho_base):
        b = self.builder
        p = b.params

        rho_target = 0.5 * (bounds["rho_lo"] + bounds["rho_hi"])

        # Method A: direct scaling (works because weights are frozen)
        alpha0 = rho_target / rho_base

        W = b.build_weight_csr(alpha=alpha0)
        rho = b.spectral_radius(W)
        bal = b.ei_balance(W)
        print(f"  Method A: alpha={alpha0:.6f} rho={rho:.6f} bal={bal:.3f}")

        in_rho = bounds["rho_lo"] <= rho <= bounds["rho_hi"]
        in_bal = p["balance_lo"] <= bal <= p["balance_hi"]
        if in_rho and in_bal:
            return self._package(regime_name, bounds, alpha0, 1.0, 1.0, rho, bal, status="OK_A")

        # Method B: sweep around alpha0 + small EI scaling
        print(f"  Method A miss (rho_in={in_rho}, bal_in={in_bal}). Sweep...")

        best = None
        best_loss = float("inf")
        rng = np.random.default_rng(self.seed * 500 + 100 + SYN_ID.get("EE", 0) + (sum(map(ord, regime_name)) % 997))

        trials = int(p.get("sweep_trials", 600))
        for _ in range(trials):
            a  = alpha0 * rng.uniform(0.85, 1.15)
            ae = rng.uniform(0.90, 1.10)
            ai = rng.uniform(0.90, 1.10)

            Wc = b.build_weight_csr(alpha=a, A_scale_E=ae, A_scale_I=ai)
            rc = b.spectral_radius(Wc)
            bc = b.ei_balance(Wc)

            loss = abs(rc - rho_target)
            if not (p["balance_lo"] <= bc <= p["balance_hi"]):
                loss += 5.0  # penalty

            if loss < best_loss:
                best_loss = loss
                best = (a, ae, ai, rc, bc)

        if best is None:
            return self._package(regime_name, bounds, alpha0, 1.0, 1.0, rho, bal, status="FAIL")

        a, ae, ai, rho, bal = best
        in_rho = bounds["rho_lo"] <= rho <= bounds["rho_hi"]
        in_bal = p["balance_lo"] <= bal <= p["balance_hi"]
        status = "OK_S" if (in_rho and in_bal) else "APPROX"
        print(f"  Sweep: alpha={a:.6f} ae={ae:.3f} ai={ai:.3f} rho={rho:.6f} bal={bal:.3f} [{status}]")

        return self._package(regime_name, bounds, a, ae, ai, rho, bal, status=status)

    def _package(self, regime_name, bounds, alpha_final, A_scale_E, A_scale_I, rho, balance, status):
        b = self.builder
        lam = b.lambda_proxy(rho)
        degE, degI = b.mean_outdegree()

        # Save NPZ: topology + raw mags + STP + final scaling + build metadata
        p = b.params
        npz_data = dict(
            seed=self.seed,
            regime=regime_name,
            N_total=b.N,
            frac_E=p["frac_E"],
            idx_E=b.idx_E,
            idx_I=b.idx_I,

            alpha_final=float(alpha_final),
            A_scale_E=float(A_scale_E),
            A_scale_I=float(A_scale_I),

            rho=float(rho),
            lambda_proxy=float(lam),
            balance=float(balance),

            rho_lo=float(bounds["rho_lo"]),
            rho_hi=float(bounds["rho_hi"]),

            # Build metadata for traceability (point 6)
            raw_mag_unit="arb",
            p_EE=float(p["p_EE"]),
            p_EI=float(p["p_EI"]),
            p_IE=float(p["p_IE"]),
            p_II=float(p["p_II"]),
            A_base_EE=float(p["A_base_EE"]),
            A_base_EI=float(p["A_base_EI"]),
            A_base_IE=float(p["A_base_IE"]),
            A_base_II=float(p["A_base_II"]),
            U_mean_EE=float(p["U_EE"]),
            U_mean_EI=float(p["U_EI"]),
            U_mean_IE=float(p["U_IE"]),
            U_mean_II=float(p["U_II"]),
            D_mean_EE=float(p["D_EE"]),
            D_mean_EI=float(p["D_EI"]),
            D_mean_IE=float(p["D_IE"]),
            D_mean_II=float(p["D_II"]),
            F_mean_EE=float(p["F_EE"]),
            F_mean_EI=float(p["F_EI"]),
            F_mean_IE=float(p["F_IE"]),
            F_mean_II=float(p["F_II"]),
        )

        # Store per synapse type: pre/post + raw_mag + STP (all frozen)
        for name in SYN_TYPES:
            e = b.edges[name]
            npz_data[f"{name}_pre"] = e["pre"]
            npz_data[f"{name}_post"] = e["post"]
            npz_data[f"{name}_raw_mag"] = b.raw_mag[name]

            stp = b.stp[name]
            npz_data[f"{name}_U"] = stp["U"]
            npz_data[f"{name}_D"] = stp["D"]
            npz_data[f"{name}_F"] = stp["F"]
            npz_data[f"{name}_delay"] = stp["delay"]
            npz_data[f"{name}_tau_I"] = stp["tau_I"]

        npz_path = os.path.join(self.output_dir, f"{regime_name}_seed{self.seed}.npz")
        np.savez_compressed(npz_path, **npz_data)
        print(f"  Saved NPZ: {npz_path}")

        rho_target = 0.5 * (bounds["rho_lo"] + bounds["rho_hi"])
        return dict(
            regime=regime_name,
            seed=self.seed,
            rho_target=rho_target,
            alpha_final=float(alpha_final),
            A_scale_E=float(A_scale_E),
            A_scale_I=float(A_scale_I),
            rho=float(rho),
            lambda_proxy=float(lam),
            balance=float(balance),
            mean_outdeg_E=float(degE),
            mean_outdeg_I=float(degI),
            status=status,
        )

    def save_csv(self, results):
        csv_path = os.path.join(self.output_dir, f"regimes_seed{self.seed}.csv")
        fields = [
            "regime","seed","rho_target","alpha_final","A_scale_E","A_scale_I",
            "rho","lambda_proxy","balance","mean_outdeg_E","mean_outdeg_I","status"
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\nSaved CSV: {csv_path}")
        return csv_path

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(seed=42, output_dir="regimes", **kwargs):
    t0 = time.time()
    finder = RegimeFinder(seed=seed, output_dir=output_dir, **kwargs)
    results = finder.find_all()
    finder.save_csv(results)

    # Validation: monotonic rho & lambda
    rhos = [r["rho"] for r in results]
    lams = [r["lambda_proxy"] for r in results]
    mono_rho = all(rhos[i] < rhos[i+1] for i in range(len(rhos)-1))
    mono_lam = all(lams[i] < lams[i+1] for i in range(len(lams)-1))

    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    for r in results:
        bounds = REGIMES[r["regime"]]
        in_rho = bounds["rho_lo"] <= r["rho"] <= bounds["rho_hi"]
        print(
            f"  {r['regime']:25s} rho={r['rho']:.6f} {'OK' if in_rho else 'MISS':4s} "
            f"lam_proxy={r['lambda_proxy']:+.4f} bal={r['balance']:.3f} "
            f"alpha={r['alpha_final']:.6f} ae={r['A_scale_E']:.3f} ai={r['A_scale_I']:.3f} [{r['status']}]"
        )
    print(f"\n  rho monotonic:        {'PASS' if mono_rho else 'FAIL'}")
    print(f"  lam_proxy monotonic:  {'PASS' if mono_lam else 'FAIL'}")
    print(f"\n  Total time: {time.time() - t0:.2f}s")

    return results


if __name__ == "__main__":
    run(seed=42, output_dir="regimes")
