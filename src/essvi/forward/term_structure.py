"""Spot + net carry (+ discrete dividends) implied from the forward CURVE across expiries (§02).

For an American single name we never observe the underlying spot directly, but the term structure
of the parity-implied forwards pins it down. With F_T the forward and D_T = e^{-r T} the discount at
expiry T:

    F_T * D_T = S * e^{-q T}          (continuous net carry q = borrow - dividend yield)

so ln(F_T D_T) is linear in T, intercept = ln S, slope = -q. A weighted regression across a name's
expiries recovers (S, q); systematic residual drops localize discrete dividends. This feeds the
de-Americanization engine (which needs spot), closing the §02/§03 joint fixed point WITHOUT any
external spot/dividend feed - the options chain alone is enough.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpotCarry:
    spot: float
    carry: float          # continuous net carry q; forward grows as F_T = S * e^{(r - q) T}
    status: str           # "ok" | "insufficient"
    n_expiries: int
    resid_max: float | None = None  # max |ln(F D) - fit| log-points; None when <=2 expiries (unmeasurable)
    resid_known: bool = False       # True only when >2 expiries gave a real residual; a thin name with a
                                    # 0.0 here would falsely read as "clean" -- this distinguishes the two


def imply_spot_carry(t: np.ndarray, forward: np.ndarray, discount: np.ndarray,
                     weights: np.ndarray | None = None, min_expiries: int = 2) -> SpotCarry:
    """Recover (spot, net carry) from per-expiry (t, forward, discount) via the linear law above."""
    t = np.asarray(t, dtype=float)
    F = np.asarray(forward, dtype=float)
    D = np.asarray(discount, dtype=float)
    ok = np.isfinite(t) & np.isfinite(F) & np.isfinite(D) & (t > 0) & (F > 0) & (D > 0)
    t, F, D = t[ok], F[ok], D[ok]
    if t.size < min_expiries:
        return SpotCarry(np.nan, np.nan, "insufficient", int(t.size))
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)[ok]
    W = np.maximum(w, 1e-12)
    y = np.log(F * D)                                   # = ln S - q t
    sw = W.sum()
    mt = (W * t).sum() / sw
    my = (W * y).sum() / sw
    var = (W * (t - mt) ** 2).sum()
    if var <= 0:                                        # one effective tenor: carry unidentifiable
        return SpotCarry(float(np.exp(my)), 0.0, "ok", int(t.size))
    b = (W * (t - mt) * (y - my)).sum() / var
    a = my - b * mt
    if t.size > 2:                                      # >2 expiries: residual is a real div-contamination signal
        resid = float(np.max(np.abs(y - (a + b * t))))
        return SpotCarry(float(np.exp(a)), float(-b), "ok", int(t.size), resid, True)
    return SpotCarry(float(np.exp(a)), float(-b), "ok", int(t.size), None, False)  # 2 expiries: resid unmeasurable
