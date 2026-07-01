"""Joint constrained-SSVI calibration.

Fits (theta_0..theta_{n-1}, rho, eta, gamma) over all listed expiries at once with SLSQP, from a
deterministic 16-point cold-start grid; the lowest converged loss wins (ties broken by start index).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import minimize

from ..constants import (
    COLD_START_ETA,
    COLD_START_GAMMA,
    COLD_START_RHO,
    HUBER_DELTA,
    SLSQP_FTOL,
    SLSQP_MAXITER,
)
from .constraints import butterfly_ineq, constraint_dicts, make_bounds, slope_cap_ineq, smooth_abs
from .objective import Slice, total_loss


@dataclass
class SSVIFit:
    thetas: np.ndarray
    rho: float
    eta: float
    gamma: float
    loss: float
    success: bool
    n_starts_converged: int
    rho_at_bound: bool
    fit_method: str = "ssvi"


def fit_joint(slices: list[Slice]) -> SSVIFit | None:
    """Return the best SSVI fit, or None if no start converges."""
    n = len(slices)
    if n == 0:
        return None
    for sl in slices:
        sl.finalize_scale()

    # monotone ATM-variance seed
    theta0 = np.array([sl.theta_atm for sl in slices], dtype=float)
    theta0 = np.maximum.accumulate(np.maximum(theta0, 1e-7))

    bounds = make_bounds(n)
    cons = constraint_dicts(n)

    best = None
    best_start = None
    converged = 0
    for idx, (rho0, eta0, gamma0) in enumerate(product(COLD_START_RHO, COLD_START_ETA, COLD_START_GAMMA)):
        # keep the start feasible w.r.t. the butterfly bound: eta*(1+|rho|) <= 2
        eta_start = min(eta0, 2.0 / (1.0 + abs(rho0)) - 1e-3)
        x0 = np.concatenate([theta0, [rho0, eta_start, gamma0]])
        try:
            res = minimize(
                total_loss,
                x0,
                args=(slices, HUBER_DELTA),
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER},
            )
        except Exception:
            continue
        if not res.success:
            continue
        converged += 1
        if best is None or res.fun < best.fun - 1e-15:
            best = res
            best_start = idx

    if best is None:
        return None

    x = best.x
    thetas = x[:n]
    rho, eta, gamma = float(x[n]), float(x[n + 1]), float(x[n + 2])
    rho_at_bound = abs(abs(rho) - 0.999) < 1e-4
    return SSVIFit(
        thetas=thetas,
        rho=rho,
        eta=eta,
        gamma=gamma,
        loss=float(best.fun),
        success=True,
        n_starts_converged=converged,
        rho_at_bound=rho_at_bound,
    )
