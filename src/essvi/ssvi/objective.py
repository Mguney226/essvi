"""Robust weighted objective for the joint SSVI fit.

Per quote: total-variance observation w_obs at log-moneyness k, weight omega = vega*(mid/spread)*quote_weight.
Per slice: residuals (w_model - w_obs) are standardized by a fixed per-slice scale (set once from the
ATM-seeded residual MAD, held constant during SLSQP so the objective stays smooth) and run through a
Huber loss (delta=1.345). Total loss is summed over slices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import HUBER_DELTA
from .model import dw_drho, dw_dtheta, w_of_k


@dataclass
class Slice:
    expiry: str
    t: float                      # Act/365F years to expiry (settlement-time aware)
    k: np.ndarray                 # log-forward moneyness of fitted quotes
    w_obs: np.ndarray             # observed total variance = iv^2 * t
    omega: np.ndarray             # fit weights
    forward: float
    discount: float
    theta_atm: float              # ATM total variance seed (w_obs interpolated to k=0)
    median_spread: float          # for the eSSVI trigger
    scale: float = 1.0            # per-slice residual scale (set in finalize)
    settle: str = "pm"            # 'am' or 'pm'

    def finalize_scale(self):
        wm = w_of_k(self.k, self.theta_atm, -0.3, 0.7, 0.45)   # rough seed surface
        resid = np.abs(self.w_obs - wm)
        mad = np.median(resid) if resid.size else 1.0
        self.scale = float(max(mad, 1e-6))
        return self


def huber(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def slice_loss(theta, rho, eta, gamma, sl: Slice, delta: float) -> float:
    wm = w_of_k(sl.k, theta, rho, eta, gamma)
    r = (wm - sl.w_obs) / sl.scale
    return float(np.sum(sl.omega * huber(r, delta)))


def slice_loss_grad(theta, rho, eta, gamma, sl: Slice, delta: float) -> np.ndarray:
    """Gradient of slice_loss w.r.t. (theta, rho), eta/gamma held fixed (the warm-path params).
    huber is C1, so d huber/d r = clip(r, -delta, delta) is continuous -> the objective gradient is smooth."""
    wm = w_of_k(sl.k, theta, rho, eta, gamma)
    r = (wm - sl.w_obs) / sl.scale
    dloss_dw = sl.omega * np.clip(r, -delta, delta) / sl.scale     # d loss / d w_j per quote
    g_theta = float(np.sum(dloss_dw * dw_dtheta(sl.k, theta, rho, eta, gamma)))
    g_rho = float(np.sum(dloss_dw * dw_drho(sl.k, theta, rho, eta, gamma)))
    return np.array([g_theta, g_rho])


def total_loss(x: np.ndarray, slices: list[Slice], delta: float = HUBER_DELTA) -> float:
    n = len(slices)
    thetas = x[:n]
    rho, eta, gamma = x[n], x[n + 1], x[n + 2]
    return sum(slice_loss(thetas[i], rho, eta, gamma, slices[i], delta) for i in range(n))
