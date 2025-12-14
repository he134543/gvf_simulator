import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

def normal_depth_rectangular(Q, b, n, S_mean, g=9.81, tol=1e-8, max_iter=60):
    """
    Normal depth for a rectangular channel using Manning's equation
    with a constant mean slope S_mean.
    """
    # initial guess (wide-channel approximation)
    y = (Q * n / (b * np.sqrt(S_mean))) ** (3.0 / 5.0)
    y = max(y, 1e-3)

    for _ in range(max_iter):
        A = b * y
        P = b + 2.0 * y
        R = A / P
        Sf = (Q * n / (A * R ** (2.0 / 3.0))) ** 2

        f = Sf - S_mean
        if abs(f) < tol:
            return y

        # finite-difference derivative
        dy = 1e-4 * y
        y2 = y + dy
        A2 = b * y2
        P2 = b + 2.0 * y2
        R2 = A2 / P2
        Sf2 = (Q * n / (A2 * R2 ** (2.0 / 3.0))) ** 2

        dfdy = (Sf2 - Sf) / dy
        y -= f / dfdy
        y = max(y, 1e-3)

    raise RuntimeError("Normal depth did not converge")



def generate_bed_profile_from_downstream(
    x_dist: np.ndarray,
    z_ds: float,
    S_mean: float,
    A: float,
    wavelength: float,
):
    """
    Generate a bed profile referenced to the downstream bed elevation.

    Parameters
    ----------
    x_dist : np.ndarray
        Distance array (monotonic). Can increase or decrease downstream.
    z_ds : float
        Bed elevation at the downstream end [m].
    S_mean : float
        Mean bed slope (positive, m/m).
    A : float
        Amplitude of sine-wave bed undulations [m].
    wavelength : float
        Wavelength of sine-wave bed undulations [m].

    Returns
    -------
    bed : np.ndarray
        Bed elevation profile [m].
    """

    x = np.asarray(x_dist, float)

    # Identify downstream index
    if x[1] > x[0]:
        idx_ds = -1   # x increases downstream
    else:
        idx_ds = 0    # x decreases downstream

    # Distance measured upstream from downstream (>= 0)
    x_up = np.abs(x - x[idx_ds])

    # Bed profile: rise upstream against slope
    bed = (
        z_ds
        + S_mean * x_up
        + A * np.sin(2.0 * np.pi * x_up / wavelength)
    )

    return bed
