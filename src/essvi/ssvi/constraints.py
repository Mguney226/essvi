"""Constraints for the joint SSVI fit, as scipy SLSQP inequality dicts.

Parameter vector x = [theta_0, ..., theta_{n-1}, rho, eta, gamma] for n listed expiries.
All inequality functions return values that must be >= 0.
"""
from __future__ import annotations

import numpy as np

from ..constants import ETA_MIN, GAMMA_MAX, GAMMA_MIN, RHO_ABS_MAX, RHO_SMOOTH, ETA_MIN
from .model import d_theta_phi, phi


def smooth_abs(rho: float) -> float:
    return float(np.sqrt(rho * rho + RHO_SMOOTH * RHO_SMOOTH))


def unpack(x: np.ndarray, n: int):
    return x[:n], x[n], x[n + 1], x[n + 2]


def make_bounds(n: int):
    return (
        [(1e-8, None)] * n
        + [(-RHO_ABS_MAX, RHO_ABS_MAX), (ETA_MIN, None), (GAMMA_MIN, GAMMA_MAX)]
    )


def butterfly_ineq(x: np.ndarray, n: int) -> float:
    """2 - eta*(1+|rho|) >= 0  -> sufficient (Gatheral-Jacquier) for butterfly-free at every theta."""
    _, rho, eta, _ = unpack(x, n)
    return 2.0 - eta * (1.0 + smooth_abs(rho))


def calendar_monotone_ineq(x: np.ndarray, n: int) -> np.ndarray:
    """theta_{i+1} - theta_i >= 0 (linear; no calendar-spread crossing)."""
    return np.diff(x[:n])


def slope_cap_ineq(x: np.ndarray, n: int) -> np.ndarray:
    """cap_i - d_theta[theta*phi]_i >= 0 at each theta, cap = (1/rho^2)(1+sqrt(1-rho^2))*phi (§04)."""
    thetas, rho, eta, gamma = unpack(x, n)
    ph = phi(thetas, eta, gamma)
    r2 = max(rho * rho, 1e-12)
    cap = (1.0 / r2) * (1.0 + np.sqrt(max(1.0 - rho * rho, 0.0))) * ph
    return cap - d_theta_phi(thetas, eta, gamma)


def constraint_dicts(n: int):
    return [
        {"type": "ineq", "fun": lambda x: np.atleast_1d(butterfly_ineq(x, n))},
        {"type": "ineq", "fun": lambda x: calendar_monotone_ineq(x, n)},
        {"type": "ineq", "fun": lambda x: slope_cap_ineq(x, n)},
    ]
