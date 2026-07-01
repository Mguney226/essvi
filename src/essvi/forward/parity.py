"""Forward & discount construction via put-call parity.

For a European chain at one expiry: C(K) - P(K) = D*(F - K). Regress (C-P) on K over near-ATM
pairs (inverse-variance weighted) -> slope = -D, intercept = D*F. The FRED-seeded discount anchors
a sanity band. Returns (F, D, status).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForwardResult:
    forward: float
    discount: float
    status: str          # "ok" | "no_forward"
    n_pairs: int


def imply_forward_discount(
    strikes: np.ndarray,
    call_minus_put: np.ndarray,
    weights: np.ndarray,
    d_seed: float,
    *,
    band_frac: float = 0.05,
    min_pairs: int = 3,
) -> ForwardResult:
    strikes = np.asarray(strikes, dtype=float)
    y = np.asarray(call_minus_put, dtype=float)
    w = np.asarray(weights, dtype=float)
    if strikes.size < min_pairs:
        return ForwardResult(np.nan, np.nan, "no_forward", strikes.size)

    order = np.argsort(strikes)
    strikes, y, w = strikes[order], y[order], w[order]

    # rough forward: strike where (C-P) crosses zero
    sign = np.sign(y)
    cross = np.where(np.diff(sign) != 0)[0]
    if cross.size:
        i = cross[0]
        x0, x1, y0, y1 = strikes[i], strikes[i + 1], y[i], y[i + 1]
        f_rough = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else 0.5 * (x0 + x1)
    else:
        f_rough = strikes[np.argmin(np.abs(y))]

    band = band_frac * f_rough
    sel = np.abs(strikes - f_rough) <= band
    if sel.sum() < min_pairs:
        # widen once
        sel = np.abs(strikes - f_rough) <= 2 * band
    if sel.sum() < min_pairs:
        return ForwardResult(np.nan, np.nan, "no_forward", int(sel.sum()))

    K, Y, W = strikes[sel], y[sel], np.maximum(w[sel], 1e-12)
    # weighted least squares: Y = a + b*K
    sw = W.sum()
    mk = (W * K).sum() / sw
    my = (W * Y).sum() / sw
    cov = (W * (K - mk) * (Y - my)).sum()
    var = (W * (K - mk) ** 2).sum()
    if var <= 0:
        return ForwardResult(np.nan, np.nan, "no_forward", int(sel.sum()))
    b = cov / var
    a = my - b * mk
    D = -b
    if D <= 0:
        return ForwardResult(np.nan, np.nan, "no_forward", int(sel.sum()))
    F = a / D

    # sanity vs FRED seed and the rough forward
    if not (0.80 <= D <= 1.05) or not np.isfinite(F) or abs(F - f_rough) > 0.10 * f_rough:
        # fall back to seeded discount, forward from ATM pair
        D = d_seed
        F = f_rough + float(np.average((Y / D + K - f_rough), weights=W)) if D > 0 else f_rough
        if not np.isfinite(F) or F <= 0:
            return ForwardResult(np.nan, np.nan, "no_forward", int(sel.sum()))
    return ForwardResult(float(F), float(D), "ok", int(sel.sum()))
