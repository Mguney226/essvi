"""Gridded surface + per-point provenance.

Grid k in [-2, 2] step 0.025 (161 points). Per grid value: flag (observed/interpolated/
extrapolated), dist_k (to nearest informing quote), dist_t (Act/365F to nearest fitted expiry).
"""
from __future__ import annotations

import numpy as np

from ..constants import (
    CM_TENORS,
    DAYCOUNT,
    GRID_K_MAX,
    GRID_K_MIN,
    GRID_K_STEP,
    OBSERVED_DK,
    OBSERVED_WEIGHT_MIN,
)
from ..ssvi.model import theta_phi, w_of_k


def k_grid() -> np.ndarray:
    n = int(round((GRID_K_MAX - GRID_K_MIN) / GRID_K_STEP)) + 1
    return np.round(GRID_K_MIN + GRID_K_STEP * np.arange(n), 6)


def slice_grid(theta, rho, eta, gamma, t, quote_k, quote_w_norm, dist_t=0.0):
    """Build the grid rows for one expiry slice.

    quote_k: log-moneyness of the surviving fitted quotes; quote_w_norm: their normalized fit
    weights in [0,1]. Returns dict of arrays: k, w, iv, flag, dist_k, dist_t.
    """
    k = k_grid()
    w = w_of_k(k, theta, rho, eta, gamma)
    iv = np.sqrt(np.maximum(w, 0.0) / t)

    obs_k = np.asarray(quote_k, dtype=float)
    obs_w = np.asarray(quote_w_norm, dtype=float)
    informing = obs_k[obs_w >= OBSERVED_WEIGHT_MIN] if obs_k.size else np.array([])
    lo = obs_k.min() if obs_k.size else 0.0
    hi = obs_k.max() if obs_k.size else 0.0

    if obs_k.size:
        dist_k = np.min(np.abs(k[:, None] - obs_k[None, :]), axis=1)
    else:
        dist_k = np.full_like(k, np.nan)

    flag = np.full(k.shape, "interpolated", dtype=object)
    # extrapolated beyond the outermost surviving quote (either wing)
    flag[(k < lo) | (k > hi)] = "extrapolated"
    # observed if an informing quote (weight>=0.25) within OBSERVED_DK and within the quoted span
    if informing.size:
        near = np.min(np.abs(k[:, None] - informing[None, :]), axis=1) <= OBSERVED_DK
        in_span = (k >= lo) & (k <= hi)
        flag[near & in_span] = "observed"

    return {
        "k": k,
        "w": w,
        "iv": iv,
        "flag": flag.astype(str),
        "dist_k": dist_k,
        "dist_t": np.full_like(k, float(dist_t)),
    }


def cm_tenor_grid(slice_ts, thetas, rhos, eta, gamma, quote_ks):
    """Constant-maturity grid nodes: for each CM tenor *within the fitted span*,
    interpolate (theta, rho) in time - rho in (rho*psi)/psi space to preserve the HM coupling - and
    emit the full k-grid at that fixed maturity. Never extrapolated in time. dist_t is the Act/365F
    distance to the nearest listed expiry, so a buyer can tell CM nodes (dist_t>0) from listed (==0).

    Returns a list of dicts: {tenor, t_cm, k, w, iv, flag, dist_k, dist_t}.
    """
    slice_ts = np.asarray(slice_ts, float)
    thetas = np.asarray(thetas, float)
    rho_arr = np.full_like(thetas, float(rhos)) if np.isscalar(rhos) else np.asarray(rhos, float)
    psi = theta_phi(thetas, eta, gamma)
    rho_psi = rho_arr * psi
    tmin, tmax = slice_ts.min(), slice_ts.max()
    k = k_grid()
    allq = np.concatenate([np.asarray(q, float) for q in quote_ks if len(q)]) \
        if any(len(q) for q in quote_ks) else np.array([])
    lo, hi = (allq.min(), allq.max()) if allq.size else (0.0, 0.0)
    dist_k = np.min(np.abs(k[:, None] - allq[None, :]), axis=1) if allq.size else np.full_like(k, np.nan)

    out = []
    for tenor in CM_TENORS:
        t_cm = tenor / DAYCOUNT
        if t_cm < tmin or t_cm > tmax:
            continue                                       # never extrapolate in time
        theta_cm = float(np.interp(t_cm, slice_ts, thetas))
        psi_cm = float(theta_phi(np.array([theta_cm]), eta, gamma)[0])
        rho_cm = float(np.clip(np.interp(t_cm, slice_ts, rho_psi) / max(psi_cm, 1e-12), -0.999, 0.999))
        w = w_of_k(k, theta_cm, rho_cm, eta, gamma)
        iv = np.sqrt(np.maximum(w, 0.0) / t_cm)
        dist_t = float(np.min(np.abs(t_cm - slice_ts)))
        flag = np.where((k < lo) | (k > hi), "extrapolated", "interpolated")  # CM is always time-interp
        for j in range(k.size):
            out.append({"tenor": int(tenor), "t_cm": float(t_cm), "k": float(k[j]), "w": float(w[j]),
                        "iv": float(iv[j]), "flag": str(flag[j]),
                        "dist_k": float(dist_k[j]), "dist_t": dist_t})
    return out


def cm_calendar_min(cm_rows, eps=1e-6):
    """Calendar no-arb check on the CM grid: total variance w(k, t_cm) must be non-decreasing in t_cm
    at each k. Returns (ok, min_delta_w) - ok if min_delta_w >= -eps over all (k, adjacent-tenor) pairs."""
    by_k: dict = {}
    for r in cm_rows:
        t = r.get("t_cm", r.get("t"))                # accepts cm_tenor_grid output OR emitted grid rows
        by_k.setdefault(round(r["k"], 6), []).append((t, r["w"]))
    cal_min = float("inf")
    for series in by_k.values():
        series.sort()
        for i in range(1, len(series)):
            cal_min = min(cal_min, series[i][1] - series[i - 1][1])
    if cal_min == float("inf"):
        cal_min = 0.0
    return cal_min >= -eps, cal_min
