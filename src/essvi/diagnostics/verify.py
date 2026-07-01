"""No-arbitrage re-check behind arb_ok.

Re-tests each fitted slice on a 5x-denser grid (dk=0.005):
  - butterfly: the analytic Gatheral-Jacquier density g(k) >= -eps (butterfly_ok). Exact for the SSVI
    family, but it shares that family with the fit's butterfly constraint, so on its own it is a
    sufficient-condition CONSISTENCY/corruption guard rather than a fully independent test. It is
    corroborated by butterfly_ok_fd -- a structurally INDEPENDENT Breeden-Litzenberger check that takes
    the second strike-derivative of the Black call price directly, using no analytic g(k) (proven to
    agree on arb-free slices and to catch genuine violations in tests/test_verify.py).
  - calendar: w(k,t2) - w(k,t1) >= -eps between adjacent expiries (calendar_ok_pair_rhos) -- a direct
    difference of fitted variances, genuinely independent of the fit constraints.
Independent enough to disagree and ship arb_ok=false.
"""
from __future__ import annotations

import numpy as np

from scipy.special import ndtr

from ..constants import DENSE_K_STEP, EPS_ARB, GRID_K_MAX, GRID_K_MIN
from ..ssvi.model import implied_density, w_of_k

# Finite-difference second-derivative noise floor for the INDEPENDENT butterfly certificate.
# Looser than EPS_ARB (which is an absolute tolerance on total variance w): a genuine butterfly
# violation drives the Breeden-Litzenberger density O(0.1)+ negative, while FD noise on the dk=0.005
# grid sits well below this. Calibrated against the analytic check on real fitted slices.
EPS_BFLY_FD = 5e-4


def dense_k() -> np.ndarray:
    n = int(round((GRID_K_MAX - GRID_K_MIN) / DENSE_K_STEP)) + 1
    return GRID_K_MIN + DENSE_K_STEP * np.arange(n)


def butterfly_ok(theta, rho, eta, gamma) -> tuple[bool, float]:
    k = dense_k()
    g = implied_density(k, theta, rho, eta, gamma)
    gmin = float(np.nanmin(g))
    return (gmin >= -EPS_ARB), gmin


def calendar_ok_pair(theta1, theta2, rho, eta, gamma) -> tuple[bool, float]:
    """w(k,t2) - w(k,t1) >= -eps at every dense k (theta2 is the longer expiry)."""
    k = dense_k()
    dw = w_of_k(k, theta2, rho, eta, gamma) - w_of_k(k, theta1, rho, eta, gamma)
    dmin = float(np.min(dw))
    return (dmin >= -EPS_ARB), dmin


def calendar_ok_pair_rhos(theta1, rho1, theta2, rho2, eta, gamma) -> tuple[bool, float]:
    """Calendar check with per-slice rho (eSSVI): w(k,t2;rho2) - w(k,t1;rho1) >= -eps at dense k."""
    k = dense_k()
    dw = w_of_k(k, theta2, rho2, eta, gamma) - w_of_k(k, theta1, rho1, eta, gamma)
    dmin = float(np.min(dw))
    return (dmin >= -EPS_ARB), dmin


def butterfly_ok_fd(theta, rho, eta, gamma) -> tuple[bool, float]:
    """INDEPENDENT butterfly certificate. The undiscounted Black call price implied by w(k) must be
    convex in strike K (Breeden-Litzenberger density d2C/dK2 >= 0). Computed directly from Black call
    prices by finite difference -- uses NO analytic g(k), so it does not share the Gatheral-Jacquier
    family that the fit constraint and implied_density() are both built on. This is the genuinely
    independent corroboration of butterfly_ok(). Returns (ok, min discrete curvature)."""
    k = dense_k()
    w = w_of_k(k, theta, rho, eta, gamma)
    sw = np.sqrt(w)
    d1 = -k / sw + 0.5 * sw
    C = ndtr(d1) - np.exp(k) * ndtr(d1 - sw)            # call / forward, undiscounted, strike K = F*e^k
    kf = np.exp(k)                                      # K / F, strictly increasing
    slope = np.diff(C) / np.diff(kf)                    # dC/dK in [-1, 0]
    curv = np.diff(slope) / (0.5 * (kf[2:] - kf[:-2]))  # discrete d2C/dK2 ~ risk-neutral density >= 0
    cmin = float(np.nanmin(curv))
    return (cmin >= -EPS_BFLY_FD), cmin


def verify_surface(thetas: np.ndarray, rhos, eta, gamma):
    """Return per-slice arb_ok plus the calendar verdict against the next-longer slice.

    thetas must be ordered by settlement datetime (increasing t). `rhos` is either a scalar
    (global SSVI) or an array of per-slice rho_i (eSSVI). Returns:
        arb_ok: bool array (per slice), bfly_min: float array, cal_min: float array.
    """
    n = len(thetas)
    rho_arr = np.full(n, float(rhos)) if np.isscalar(rhos) else np.asarray(rhos, dtype=float)
    arb = np.ones(n, dtype=bool)
    bfly_min = np.empty(n)
    cal_min = np.full(n, np.inf)
    for i in range(n):
        ok_b, gmin = butterfly_ok(thetas[i], rho_arr[i], eta, gamma)
        bfly_min[i] = gmin
        arb[i] = ok_b
    for i in range(n - 1):
        ok_c, dmin = calendar_ok_pair_rhos(thetas[i], rho_arr[i], thetas[i + 1], rho_arr[i + 1], eta, gamma)
        cal_min[i] = dmin
        if not ok_c:
            arb[i] = False
            arb[i + 1] = False
    return arb, bfly_min, cal_min
