"""Constant-maturity iv_history series.

For each CM tenor (days): interpolate theta monotonically in t, then read off the fitted SSVI:
atm_iv, 25-delta risk reversal & butterfly, term-structure slope, and the interpolated forward.
25-delta strikes are located by forward delta on the fitted surface.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri  # inverse standard-normal CDF

from ..constants import CM_TENORS, DAYCOUNT
from ..ssvi.model import w_of_k

_D1_25 = ndtri(0.75)   # ~0.6745: 25-delta call has d1 = -ndtri(0.75)? see below


def _sigma_at_k(k, theta, rho, eta, gamma, t):
    return float(np.sqrt(max(w_of_k(np.array([k]), theta, rho, eta, gamma)[0], 0.0) / t))


def _solve_k_for_call_delta(target_delta, theta, rho, eta, gamma):
    """Find k where forward call delta N(d1)=target. d1=(-k+w/2)/sqrt(w). Scan+interp (monotone)."""
    ks = np.linspace(-1.5, 1.5, 601)
    w = w_of_k(ks, theta, rho, eta, gamma)
    sw = np.sqrt(np.maximum(w, 1e-12))
    d1 = (-ks + 0.5 * w) / sw
    from scipy.special import ndtr
    dlt = ndtr(d1)                          # decreasing in k
    # find crossing of dlt - target
    f = dlt - target_delta
    idx = np.where(np.diff(np.sign(f)) != 0)[0]
    if idx.size == 0:
        return float(ks[np.argmin(np.abs(f))])
    i = idx[0]
    x0, x1, y0, y1 = ks[i], ks[i + 1], f[i], f[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def build_iv_history(slice_ts: np.ndarray, thetas: np.ndarray, rhos, forwards: np.ndarray, eta, gamma):
    """slice_ts/thetas/forwards ordered by increasing t; rhos scalar (SSVI) or per-slice (eSSVI).
    Returns list of dict rows per CM tenor within the fitted span (no time extrapolation).
    rho at a CM tenor is interpolated in (rho*psi) / psi space, which preserves the HM coupling."""
    from ..ssvi.model import theta_phi

    slice_ts = np.asarray(slice_ts, float)
    thetas = np.asarray(thetas, float)
    forwards = np.asarray(forwards, float)
    rho_arr = np.full_like(thetas, float(rhos)) if np.isscalar(rhos) else np.asarray(rhos, float)
    psi = theta_phi(thetas, eta, gamma)
    rho_psi = rho_arr * psi
    tmin, tmax = slice_ts.min(), slice_ts.max()

    rows = []
    atm_by_tenor = {}
    for tenor in CM_TENORS:
        t_cm = tenor / DAYCOUNT
        if t_cm < tmin or t_cm > tmax:
            continue                         # never extrapolate in time
        theta_cm = float(np.interp(t_cm, slice_ts, thetas))
        fwd_cm = float(np.interp(t_cm, slice_ts, forwards))
        psi_cm = float(theta_phi(np.array([theta_cm]), eta, gamma)[0])
        rho_cm = float(np.clip(np.interp(t_cm, slice_ts, rho_psi) / max(psi_cm, 1e-12), -0.999, 0.999))
        atm_iv = float(np.sqrt(theta_cm / t_cm))
        kc = _solve_k_for_call_delta(0.25, theta_cm, rho_cm, eta, gamma)   # 25-delta call
        kp = _solve_k_for_call_delta(0.75, theta_cm, rho_cm, eta, gamma)   # 25-delta put (call delta 0.75)
        sc = _sigma_at_k(kc, theta_cm, rho_cm, eta, gamma, t_cm)
        sp = _sigma_at_k(kp, theta_cm, rho_cm, eta, gamma, t_cm)
        rr = sc - sp
        bf = 0.5 * (sc + sp) - atm_iv
        atm_by_tenor[tenor] = atm_iv
        rows.append({"tenor": int(tenor), "atm_iv": atm_iv, "rr_25d": rr, "bf_25d": bf,
                     "slope": np.nan, "fwd": fwd_cm, "t_cm": t_cm})

    # slope = d(atm_iv)/d(ln t), central difference across present tenors
    for i, r in enumerate(rows):
        lo = rows[i - 1] if i > 0 else None
        hi = rows[i + 1] if i < len(rows) - 1 else None
        if lo and hi:
            r["slope"] = (hi["atm_iv"] - lo["atm_iv"]) / (np.log(hi["t_cm"]) - np.log(lo["t_cm"]))
        elif hi:
            r["slope"] = (hi["atm_iv"] - r["atm_iv"]) / (np.log(hi["t_cm"]) - np.log(r["t_cm"]))
        elif lo:
            r["slope"] = (r["atm_iv"] - lo["atm_iv"]) / (np.log(r["t_cm"]) - np.log(lo["t_cm"]))
        else:
            r["slope"] = 0.0
    return rows
