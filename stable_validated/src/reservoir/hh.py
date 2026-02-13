"""
Hodgkin-Huxley neuron model with optional A-current (Connor-Stevens).

Implements the full HH system of ODEs with RK4 integration.
All parameters are configurable. Dale's law is enforced externally.

References:
    Hodgkin & Huxley (1952), J. Physiol., 117(4), 500-544.
    Connor & Stevens (1971), J. Physiol., 213(1), 31-53.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HHParams:
    """Standard HH neuron parameters."""
    C_m: float = 1.0       # µF/cm²
    g_Na: float = 120.0    # mS/cm²
    g_K: float = 36.0      # mS/cm²
    g_L: float = 0.3       # mS/cm²
    E_Na: float = 50.0     # mV
    E_K: float = -77.0     # mV
    E_L: float = -54.4     # mV
    V_rest: float = -65.0  # mV
    # A-current (Connor-Stevens) — active only if use_A_current is True
    use_A_current: bool = True
    g_A: float = 47.7      # mS/cm²
    E_A: float = -75.0     # mV

    @classmethod
    def from_config(cls, cfg: dict) -> "HHParams":
        neuron_cfg = cfg.get("neuron", {})
        use_a = neuron_cfg.get("model", "hh") == "hh_a"
        return cls(
            C_m=neuron_cfg.get("C_m", 1.0),
            g_Na=neuron_cfg.get("g_Na", 120.0),
            g_K=neuron_cfg.get("g_K", 36.0),
            g_L=neuron_cfg.get("g_L", 0.3),
            E_Na=neuron_cfg.get("E_Na", 50.0),
            E_K=neuron_cfg.get("E_K", -77.0),
            E_L=neuron_cfg.get("E_L", -54.4),
            V_rest=neuron_cfg.get("V_rest", -65.0),
            use_A_current=use_a,
            g_A=neuron_cfg.get("g_A", 47.7),
            E_A=neuron_cfg.get("E_A", -75.0),
        )


# ─── Gating variable rate functions (Hodgkin-Huxley original) ───

def _safe_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Handle 0/0 in alpha/beta rate functions via L'Hôpital limit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(y) > 1e-7, x / y, np.ones_like(x))
    return result


def alpha_m(V: np.ndarray) -> np.ndarray:
    dv = V + 40.0
    return _safe_div(0.1 * dv, 1.0 - np.exp(-dv / 10.0))


def beta_m(V: np.ndarray) -> np.ndarray:
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h(V: np.ndarray) -> np.ndarray:
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def alpha_n(V: np.ndarray) -> np.ndarray:
    dv = V + 55.0
    return _safe_div(0.01 * dv, 1.0 - np.exp(-dv / 10.0))


def beta_n(V: np.ndarray) -> np.ndarray:
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


# ─── A-current gating (Connor-Stevens style) ───

def a_inf(V: np.ndarray) -> np.ndarray:
    """Steady-state activation for A-current."""
    return (0.0761 * np.exp((V + 94.22) / 31.84) /
            (1.0 + np.exp((V + 1.17) / 28.93))) ** (1.0 / 3.0)


def tau_a(V: np.ndarray) -> np.ndarray:
    """Time constant for A-current activation (ms)."""
    return 0.3632 + 1.158 / (1.0 + np.exp((V + 55.96) / 20.12))


def b_inf(V: np.ndarray) -> np.ndarray:
    """Steady-state inactivation for A-current."""
    return 1.0 / (1.0 + np.exp((V + 53.3) / 14.54)) ** 4


def tau_b(V: np.ndarray) -> np.ndarray:
    """Time constant for A-current inactivation (ms)."""
    return 1.24 + 2.678 / (1.0 + np.exp((V + 50.0) / 16.027))


# ─── HH State Vector ───

class HHState:
    """
    State of N Hodgkin-Huxley neurons.

    State variables per neuron: V, m, h, n, [a, b if A-current]
    """

    def __init__(self, N: int, params: HHParams, rng: Optional[np.random.Generator] = None):
        self.N = N
        self.params = params
        if rng is None:
            rng = np.random.default_rng(0)

        # Initialize at rest with small random perturbation
        self.V = params.V_rest + rng.normal(0, 0.1, N)
        # Steady-state gating at V_rest
        V0 = self.V
        am, bm = alpha_m(V0), beta_m(V0)
        self.m = am / (am + bm)
        ah, bh = alpha_h(V0), beta_h(V0)
        self.h = ah / (ah + bh)
        an, bn = alpha_n(V0), beta_n(V0)
        self.n = an / (an + bn)

        if params.use_A_current:
            self.a = a_inf(V0)
            self.b_gate = b_inf(V0)
        else:
            self.a = np.zeros(N)
            self.b_gate = np.zeros(N)

    def get_vector(self) -> np.ndarray:
        """Return flat state vector [V, m, h, n, a, b] shape (6*N,)."""
        return np.concatenate([self.V, self.m, self.h, self.n, self.a, self.b_gate])

    def set_from_vector(self, vec: np.ndarray):
        """Set state from flat vector."""
        N = self.N
        self.V = vec[0:N].copy()
        self.m = vec[N:2*N].copy()
        self.h = vec[2*N:3*N].copy()
        self.n = vec[3*N:4*N].copy()
        self.a = vec[4*N:5*N].copy()
        self.b_gate = vec[5*N:6*N].copy()

    def copy(self) -> "HHState":
        """Deep copy of state."""
        new = HHState.__new__(HHState)
        new.N = self.N
        new.params = self.params
        new.V = self.V.copy()
        new.m = self.m.copy()
        new.h = self.h.copy()
        new.n = self.n.copy()
        new.a = self.a.copy()
        new.b_gate = self.b_gate.copy()
        return new


def hh_derivatives(state_vec: np.ndarray, N: int, params: HHParams,
                   I_ext: np.ndarray) -> np.ndarray:
    """
    Compute derivatives for the HH system.

    Args:
        state_vec: [V, m, h, n, a, b] flattened, shape (6*N,)
        N: number of neurons
        params: HH parameters
        I_ext: external current per neuron, shape (N,)

    Returns:
        derivatives vector, same shape as state_vec
    """
    V = state_vec[0:N]
    m = state_vec[N:2*N]
    h = state_vec[2*N:3*N]
    n = state_vec[3*N:4*N]
    a = state_vec[4*N:5*N]
    b = state_vec[5*N:6*N]

    # Clip V for numerical stability in exponential rate functions
    # Prevents NaN blow-up at large V
    V_safe = np.clip(V, -100.0, 100.0)

    # Ionic currents
    I_Na = params.g_Na * m**3 * h * (V - params.E_Na)
    I_K = params.g_K * n**4 * (V - params.E_K)
    I_L = params.g_L * (V - params.E_L)

    # A-current (if enabled)
    I_A = np.zeros(N)
    if params.use_A_current:
        I_A = params.g_A * a**3 * b * (V - params.E_A)

    # Membrane equation
    dVdt = (-I_Na - I_K - I_L - I_A + I_ext) / params.C_m

    # Gating variables (use V_safe for rate functions)
    am, bm = alpha_m(V_safe), beta_m(V_safe)
    dmdt = am * (1.0 - m) - bm * m

    ah, bh = alpha_h(V_safe), beta_h(V_safe)
    dhdt = ah * (1.0 - h) - bh * h

    an, bn = alpha_n(V_safe), beta_n(V_safe)
    dndt = an * (1.0 - n) - bn * n

    # A-current gating (use V_safe for rate functions)
    if params.use_A_current:
        dadt = (a_inf(V_safe) - a) / tau_a(V_safe)
        dbdt = (b_inf(V_safe) - b) / tau_b(V_safe)
    else:
        dadt = np.zeros(N)
        dbdt = np.zeros(N)

    return np.concatenate([dVdt, dmdt, dhdt, dndt, dadt, dbdt])


def rk4_step(state_vec: np.ndarray, N: int, params: HHParams,
             I_ext: np.ndarray, dt: float) -> np.ndarray:
    """
    Single RK4 integration step for HH system.

    Args:
        state_vec: current state vector (6*N,)
        N: number of neurons
        params: HH parameters
        I_ext: external current (N,) — assumed constant during step
        dt: time step in ms

    Returns:
        new state vector (6*N,)
    """
    k1 = hh_derivatives(state_vec, N, params, I_ext)
    k2 = hh_derivatives(state_vec + 0.5 * dt * k1, N, params, I_ext)
    k3 = hh_derivatives(state_vec + 0.5 * dt * k2, N, params, I_ext)
    k4 = hh_derivatives(state_vec + dt * k3, N, params, I_ext)

    new_state = state_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return new_state


def clip_gating(state_vec: np.ndarray, N: int) -> tuple:
    """
    Clip gating variables to [0, 1] and return count of clipped values.

    Returns:
        (clipped_state_vec, n_clipped)
    """
    n_clipped = 0
    for start in [N, 2*N, 3*N, 4*N, 5*N]:
        end = start + N
        segment = state_vec[start:end]
        below = np.sum(segment < 0)
        above = np.sum(segment > 1)
        n_clipped += int(below + above)
        state_vec[start:end] = np.clip(segment, 0.0, 1.0)
    return state_vec, n_clipped
