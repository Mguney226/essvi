"""Borrow/dividend sensitivities.

For the EUROPEAN index layer there is no de-Americanization: borrow/dividends enter only through
the parity-implied forward, so these are forward-level sensitivities, not separable de-Am terms.
  dvol_dborrow = ATM-IV change (vol points) for a +/-25bp carry bump on the forward.
  dvol_ddiv    = 0.0 (cash index: dividends are not separately identified from the forward).
They become genuinely meaningful in the American single-name (de-Am) layer (M4).
"""
from __future__ import annotations

import numpy as np

from ..ssvi.model import w_of_k


def dvol_dborrow(theta, rho, eta, gamma, t) -> float:
    dk = 0.0025 * t                       # 25bp carry over t shifts ln F by ~0.0025*t
    sig = lambda kk: float(np.sqrt(w_of_k(np.array([kk]), theta, rho, eta, gamma)[0] / t))
    return 100.0 * (sig(-dk) - sig(+dk))   # vol points across the +/-25bp bump


def dvol_ddiv(theta, rho, eta, gamma, t) -> float:
    return 0.0
