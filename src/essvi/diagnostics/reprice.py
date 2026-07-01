"""Per-slice repricing error: reprice_rmse = 100*sqrt(mean((sig_fit-sig_mid)^2)),
in Black vol points, in-sample, unweighted, over the fitted (OTM-side) quotes."""
from __future__ import annotations

import numpy as np

from ..ssvi.model import w_of_k


def reprice_rmse(theta, rho, eta, gamma, t, quote_k, sig_mid) -> float:
    quote_k = np.asarray(quote_k, dtype=float)
    sig_mid = np.asarray(sig_mid, dtype=float)
    if quote_k.size == 0:
        return float("nan")
    w_fit = w_of_k(quote_k, theta, rho, eta, gamma)
    sig_fit = np.sqrt(np.maximum(w_fit, 0.0) / t)
    return float(100.0 * np.sqrt(np.mean((sig_fit - sig_mid) ** 2)))
