import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

@dataclass
class ChannelParams:
    """Rectangular, prismatic channel. Bed slope comes from bed elevation array."""
    b: float       # bottom width [m]
    n: float       # Manning n [-]
    Q: float       # discharge [m^3/s]
    g: float = 9.81


Method = Literal["euler", "rk2", "rk4"]


def rectangular_section(y: float, p: ChannelParams):
    if y <= 0:
        raise ValueError("Depth y must be positive.")
    A = p.b * y
    P = p.b + 2.0 * y
    R = A / P
    T = p.b
    V = p.Q / A
    Fr2 = V**2 / (p.g * (A / T))  # Fr^2 = V^2/(g*D), D=A/T
    return A, R, Fr2


def manning_Sf(y: float, p: ChannelParams) -> float:
    A, R, _ = rectangular_section(y, p)
    return (p.Q * p.n / (A * (R ** (2.0 / 3.0)))) ** 2


def compute_S0_from_bed(x: np.ndarray, zb: np.ndarray) -> np.ndarray:
    """
    Local bed slope S0(x) = -dzb/dx (positive if bed falls downstream).
    """
    x = np.asarray(x, float)
    zb = np.asarray(zb, float)

    if x.ndim != 1 or zb.ndim != 1 or x.size != zb.size:
        raise ValueError("x and zb must be 1D arrays with the same length.")
    if not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError("x must be strictly monotonic.")

    dzdx = np.empty_like(zb)
    dzdx[1:-1] = (zb[2:] - zb[:-2]) / (x[2:] - x[:-2])
    dzdx[0]    = (zb[1] - zb[0]) / (x[1] - x[0])
    dzdx[-1]   = (zb[-1] - zb[-2]) / (x[-1] - x[-2])

    return -dzdx


def gvf_rhs(y: float, S0_local: float, p: ChannelParams, denom_eps: float = 1e-6) -> float:
    """
    RHS of GVF ODE: dy/dx = (S0(x) - Sf(y)) / (1 - Fr^2(y))
    """
    _, _, Fr2 = rectangular_section(y, p)
    Sf = manning_Sf(y, p)

    denom = 1.0 - Fr2
    if abs(denom) < denom_eps:
        denom = np.sign(denom) * denom_eps

    return (S0_local - Sf) / denom


def step_ode(
    y: float,
    dx: float,
    rhs_at_x,
    method: Method,
) -> float:
    """
    One ODE step for y' = f(x, y) using fixed-step explicit methods.
    rhs_at_x(k, y_val) should return f at a "stage" k=0.. with whatever
    local x info you embed (we use piecewise-constant S0 per segment).
    """
    if method == "euler":
        k1 = rhs_at_x(0, y)
        return y + dx * k1

    if method == "rk2":
        k1 = rhs_at_x(0, y)
        k2 = rhs_at_x(1, y + 0.5 * dx * k1)
        return y + dx * k2

    if method == "rk4":
        k1 = rhs_at_x(0, y)
        k2 = rhs_at_x(1, y + 0.5 * dx * k1)
        k3 = rhs_at_x(2, y + 0.5 * dx * k2)
        k4 = rhs_at_x(3, y + dx * k3)
        return y + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    raise ValueError(f"Unknown method: {method}")


def gvf_from_downstream(
    x: np.ndarray,
    zb: np.ndarray,
    wse_downstream: float,
    params: ChannelParams,
    method: Method = "rk4",
    denom_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Route GVF from downstream -> upstream with selectable numerical method.

    Boundary conditions:
      - Discharge Q (in params)
      - Downstream WSE (wse_downstream)

    Notes
    -----
    - Works whether x increases downstream (common) or decreases downstream.
    - Uses a piecewise-constant S0 on each segment for RK stages (stable + simple).
      If you want S0(x) interpolated for stages, you can extend rhs_at_x.

    Returns
    -------
    y   : depth [m]
    wse : water surface elevation [m]
    """
    x = np.asarray(x, float)
    zb = np.asarray(zb, float)
    if x.size != zb.size or x.size < 2:
        raise ValueError("x and zb must have same length >= 2.")

    # Determine downstream end based on x direction:
    x_increases = (x[1] > x[0])
    idx_ds = -1 if x_increases else 0

    # Downstream depth from downstream WSE
    y_ds = wse_downstream - zb[idx_ds]
    if y_ds <= 0:
        raise ValueError(
            f"Downstream depth non-positive: y_ds={y_ds:.4f} m "
            f"(wse_ds={wse_downstream}, zb_ds={zb[idx_ds]})."
        )

    S0 = compute_S0_from_bed(x, zb)
    y = np.full_like(zb, np.nan, dtype=float)
    y[idx_ds] = y_ds

    # Integration order: downstream -> upstream
    if idx_ds == -1:
        indices = range(x.size - 2, -1, -1)  # ... -> 0
    else:
        indices = range(1, x.size)           # 1 -> ...

    for i in indices:
        j = i + 1 if idx_ds == -1 else i - 1  # neighbor closer to downstream boundary

        dx = x[i] - x[j]
        yj = y[j]

        # Use slope from the known side (j) for the whole segment (j -> i)
        S0_seg = float(S0[j])

        def rhs_at_x(stage: int, y_val: float) -> float:
            return gvf_rhs(y_val, S0_seg, params, denom_eps=denom_eps)

        yi = step_ode(yj, dx, rhs_at_x, method=method)

        if yi <= 0:
            raise RuntimeError(
                f"Depth became non-positive at x={x[i]:.2f} m: y={yi:.4f} m "
                f"(last valid y={yj:.4f} m at x={x[j]:.2f} m)."
            )

        y[i] = yi

    wse = zb + y
    return y, wse