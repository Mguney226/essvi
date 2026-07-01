"""Black-76 (options on a forward) pricing, vega, and a deterministic implied-variance inverter.

All prices are NORMALIZED by D*F (the discount factor times forward), so the only state is
log-forward moneyness k = log(K / F) and total implied variance w = sigma^2 * t. Working in
normalized-OTM-price space keeps the deep wings (vega -> 0) numerically conditioned.

Determinism: vectorized numpy + scipy.special.ndtr (no transcendental branch that varies by
input magnitude), fixed-iteration safeguarded Newton. No randomness, no early stop on wall clock.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorized & deterministic

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _npdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def norm_price(k: np.ndarray, w: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Normalized Black price (= price / (D*F)). k=log(K/F), w=sigma^2*t, is_call bool array."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    sw = np.sqrt(np.maximum(w, 0.0))
    ek = np.exp(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (-k + 0.5 * w) / sw
        d2 = d1 - sw
    # w -> 0 limit: intrinsic
    call = np.where(sw > 0, ndtr(d1) - ek * ndtr(d2), np.maximum(1.0 - ek, 0.0))
    put = np.where(sw > 0, ek * ndtr(-d2) - ndtr(-d1), np.maximum(ek - 1.0, 0.0))
    return np.where(is_call, call, put)


def norm_vega_sw(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """d(norm_price)/d(sqrt(w)) = n(d1). Proportional to true vega; used as a fit weight.
    Independent of call/put (puts and calls share vega)."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = np.sqrt(np.maximum(w, 1e-300))
    d1 = (-k + 0.5 * w) / sw
    return _npdf(d1)


def otm_is_call(k: np.ndarray) -> np.ndarray:
    """OTM convention: calls for k>=0 (strike above forward), puts for k<0."""
    return np.asarray(k, dtype=float) >= 0.0


def implied_total_variance(
    k: np.ndarray,
    price_norm: np.ndarray,
    is_call: np.ndarray,
    *,
    max_iter: int = 64,
    tol: float = 1e-12,
    w_lo: float = 1e-10,
    w_hi: float = 50.0,
) -> np.ndarray:
    """Invert normalized Black price -> total variance w. Vectorized, deterministic.

    Returns NaN where the price is below intrinsic / above the no-arb bound (e.g. NBBO mid <
    intrinsic) or where the bracket does not contain a root. Safeguarded Newton in sqrt(w) with
    bisection fallback, fixed iteration count.
    """
    k = np.asarray(k, dtype=float)
    p = np.asarray(price_norm, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    n = k.shape[0]

    # No-arb price bounds (normalized): call in (max(1-e^k,0), 1); put in (max(e^k-1,0), e^k).
    ek = np.exp(k)
    lo_price = np.where(is_call, np.maximum(1.0 - ek, 0.0), np.maximum(ek - 1.0, 0.0))
    hi_price = np.where(is_call, 1.0, ek)
    valid = np.isfinite(p) & (p > lo_price + 1e-14) & (p < hi_price - 1e-14)

    s_lo = np.full(n, np.sqrt(w_lo))
    s_hi = np.full(n, np.sqrt(w_hi))
    # initial guess: Brenner-Subrahmanyam style, clamped into the bracket
    s = np.clip(np.abs(p) * 2.5066282746310002, s_lo * 1.0001, s_hi * 0.9999)

    for _ in range(max_iter):
        w = s * s
        f = norm_price(k, w, is_call) - p            # monotone increasing in s
        v = norm_vega_sw(k, w)                        # df/ds > 0
        # maintain bracket
        too_high = f > 0
        s_hi = np.where(too_high, np.minimum(s_hi, s), s_hi)
        s_lo = np.where(~too_high, np.maximum(s_lo, s), s_lo)
        # Newton step, fall back to bisection if it leaves the bracket or vega ~ 0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s_newton = s - f / v
        ok = np.isfinite(s_newton) & (s_newton > s_lo) & (s_newton < s_hi) & (v > 1e-12)
        s = np.where(ok, s_newton, 0.5 * (s_lo + s_hi))

    w = s * s
    f = norm_price(k, w, is_call) - p
    converged = valid & (np.abs(f) < tol * 10) & (np.abs(s_hi - s_lo) < 1e-9 + 1e-7 * s)
    out = np.where(valid, w, np.nan)
    # if didn't converge tightly, still return the bracketed value but only if price residual small
    out = np.where(valid & (np.abs(f) < 1e-8), w, out)
    out = np.where(valid, out, np.nan)
    return out
