import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
import pandas as pd


@dataclass
class GVFZoneResult:
    idx_tr: Optional[int]
    idx_L: Optional[int]
    x_tr: Optional[float]
    x_L: Optional[float]
    offset: np.ndarray          # wse - wse_normal
    slope_m_per_m: np.ndarray   # d(wse)/dx in m/m
    is_gvf: np.ndarray          # True between xTr and xL (inclusive)


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    if window % 2 == 0:
        raise ValueError("smooth_window must be odd.")
    k = window // 2
    ypad = np.pad(y, (k, k), mode="edge")
    return np.convolve(ypad, np.ones(window) / window, mode="valid")


def _first_persistent_true(mask: np.ndarray, start: int, stop: int, min_run: int) -> Optional[int]:
    """First index i in [start, stop) where mask is True for >= min_run consecutive points starting at i."""
    if min_run <= 1:
        idx = np.argmax(mask[start:stop]) if np.any(mask[start:stop]) else None
        return (start + idx) if idx is not None else None

    run = 0
    for i in range(start, stop):
        if mask[i]:
            run += 1
            if run >= min_run:
                return i - min_run + 1
        else:
            run = 0
    return None


def _xtr_from_xL_upstream(
    within_mask: np.ndarray,
    idx_L: int,
    min_run: int,
) -> Optional[int]:
    """
    Scan from idx_L upstream (decreasing index) to find the FIRST point that
    converges to normal depth (within_mask True) with persistence.

    Returns idx_tr (closest to idx_L but upstream) such that:
      - within_mask is True for min_run points ending at idx_tr (in upstream direction)
      - and the point immediately downstream of idx_tr (idx_tr+1) is NOT within_mask
        (so this is the boundary nearest to xL)
    """
    if idx_L is None:
        return None
    if min_run < 1:
        min_run = 1

    # If xL is very close to upstream boundary
    lo = max(0, min_run - 1)

    for i in range(idx_L, lo - 1, -1):
        if not within_mask[i]:
            continue

        j0 = i - min_run + 1
        if j0 < 0:
            continue

        if np.all(within_mask[j0:i + 1]):
            # ensure this is the FIRST convergence point when coming from xL upstream
            if i == idx_L or (i + 1 <= idx_L and not within_mask[i + 1]):
                return i

    return None


def determine_gvf_zone_xL_first(
    x: np.ndarray,
    wse: np.ndarray,
    wse_normal: np.ndarray,
    *,
    offset_thresh_m: float = 0.2,
    slope_thresh_m_per_km: float = 0.01,
    min_run_tr: int = 5,
    min_run_L: int = 5,
    smooth_window: int = 9,
) -> GVFZoneResult:
    """
    xL-first GVF zone detection:

      1) Find xL: first point (upstream->downstream) where |dWSE/dx| <= slope_thresh
         persistently for min_run_L points.

      2) Recompute xTr: from xL moving upstream, find the first point where
         |WSE - WSE_normal| <= offset_thresh_m persistently for min_run_tr points.

    Assumes x is strictly monotonic. Internally we work in upstream->downstream order.
    """
    x = np.asarray(x, float)
    wse = np.asarray(wse, float)
    wse_normal = np.asarray(wse_normal, float)

    if not (x.ndim == wse.ndim == wse_normal.ndim == 1):
        raise ValueError("x, wse, wse_normal must be 1D arrays.")
    if not (len(x) == len(wse) == len(wse_normal)):
        raise ValueError("x, wse, wse_normal must have the same length.")
    if len(x) < 3:
        raise ValueError("Need at least 3 points.")
    if not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError("x must be strictly monotonic.")

    # Ensure upstream->downstream index order (x increasing)
    flipped = False
    if x[1] < x[0]:
        flipped = True
        x = x[::-1]
        wse = wse[::-1]
        wse_normal = wse_normal[::-1]

    # Diagnostics
    offset = wse - wse_normal

    # Slope criterion on smoothed WSE
    wse_s = _moving_average(wse, smooth_window)
    slope_m_per_m = np.gradient(wse_s, x)
    slope_thresh_m_per_m = slope_thresh_m_per_km / 1000.0
    flat_mask = np.abs(slope_m_per_m) <= slope_thresh_m_per_m

    # 1) xL: first persistent flat point
    idx_L = _first_persistent_true(flat_mask, start=0, stop=len(x), min_run=min_run_L)

    # 2) xTr: from xL upstream, first point that converges to normal depth
    within_mask = np.abs(offset) <= offset_thresh_m
    idx_tr = None
    if idx_L is not None:
        idx_tr = _xtr_from_xL_upstream(within_mask, idx_L, min_run=min_run_tr)

    # Build GVF mask (between xTr and xL)
    is_gvf = np.zeros_like(x, dtype=bool)
    if idx_tr is not None and idx_L is not None and idx_tr <= idx_L:
        is_gvf[idx_tr:idx_L + 1] = True

    # Flip back to original indexing if needed
    if flipped:
        def flip_idx(i):
            return None if i is None else (len(x) - 1 - i)

        idx_tr_out = flip_idx(idx_tr)
        idx_L_out = flip_idx(idx_L)

        # also flip diagnostic arrays back
        offset_out = offset[::-1]
        slope_out = slope_m_per_m[::-1]
        is_gvf_out = is_gvf[::-1]

        x_tr = None if idx_tr_out is None else float(np.asarray(x[::-1])[idx_tr_out])
        x_L  = None if idx_L_out  is None else float(np.asarray(x[::-1])[idx_L_out])

        return GVFZoneResult(idx_tr_out, idx_L_out, x_tr, x_L, offset_out, slope_out, is_gvf_out)

    # Not flipped
    x_tr = None if idx_tr is None else float(x[idx_tr])
    x_L  = None if idx_L  is None else float(x[idx_L])

    return GVFZoneResult(idx_tr, idx_L, x_tr, x_L, offset, slope_m_per_m, is_gvf)