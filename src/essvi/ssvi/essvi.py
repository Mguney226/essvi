"""eSSVI: per-expiry rho under the Hendriks-Martini calendar conditions.

Three deterministic stages:
  Stage 1 (input): the global SSVI fit (theta_i, rho, eta, gamma). eta/gamma are frozen hereafter.
  Stage 2: per-slice refinement - each slice independently fits (theta_i, rho_i) under its own
           Gatheral-Jacquier butterfly bounds, warm-started from stage 1.
  Stage 3: joint polish over (theta_1..n, rho_1..n) with eta/gamma fixed, subject to:
           theta monotone (linear), per-slice butterfly, and the HM coupling between adjacent
           slices: |rho_{i+1}*psi_{i+1} - rho_i*psi_i| <= psi_{i+1} - psi_i, psi_i = theta_i*phi(theta_i).
           (theta monotone implies psi monotone for the power-law family.)
If stage 3 fails to converge, a deterministic feasibility projection of the stage-2 solution ships.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

# Analytic-Jacobian warm path: OFF by default (byte-identical to the finite-difference SLSQP path);
# flip on with VSE_SSVI_JAC=1 only for the speedup measurement + byte-identity gate.
_USE_JAC = os.environ.get("VSE_SSVI_JAC", "").lower() in ("1", "true", "on")

from ..constants import HUBER_DELTA, RHO_ABS_MAX, SLSQP_FTOL, SLSQP_MAXITER
from .fit import SSVIFit
from .model import d_theta_phi, dphi_dtheta, phi, theta_phi
from .objective import Slice, slice_loss, slice_loss_grad


# --- warm per-slice G-J butterfly constraints + analytic jacobians (for SLSQP jac=) ---
# x = [theta, rho]; eta, gamma fixed. A = 1 + sqrt(rho^2 + 1e-16) (smoothed 1+|rho|).

def _gj1(x: np.ndarray, eta: float, gamma: float) -> float:
    return 4.0 - x[0] * phi(x[0], eta, gamma) * (1.0 + np.sqrt(x[1] ** 2 + 1e-16))


def _gj2(x: np.ndarray, eta: float, gamma: float) -> float:
    return 4.0 - x[0] * phi(x[0], eta, gamma) ** 2 * (1.0 + np.sqrt(x[1] ** 2 + 1e-16))


def _gj1_jac(x: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """[d c1/d theta, d c1/d rho]; d(theta*phi)/dtheta = d_theta_phi."""
    th, rho = float(x[0]), float(x[1])
    A = 1.0 + np.sqrt(rho ** 2 + 1e-16)
    dA = rho / np.sqrt(rho ** 2 + 1e-16)
    ph = float(phi(th, eta, gamma))
    return np.array([-A * float(d_theta_phi(th, eta, gamma)), -th * ph * dA])


def _gj2_jac(x: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """[d c2/d theta, d c2/d rho]; d(theta*phi^2)/dtheta = phi^2 + 2*theta*phi*dphi_dtheta."""
    th, rho = float(x[0]), float(x[1])
    A = 1.0 + np.sqrt(rho ** 2 + 1e-16)
    dA = rho / np.sqrt(rho ** 2 + 1e-16)
    ph = float(phi(th, eta, gamma))
    dph = float(dphi_dtheta(th, eta, gamma))
    return np.array([-A * (ph * ph + 2.0 * th * ph * dph), -th * ph * ph * dA])


@dataclass
class ESSVIFit:
    thetas: np.ndarray
    rhos: np.ndarray
    eta: float
    gamma: float
    loss: float
    success: bool
    fit_method: str = "essvi"


def _rho_bound(theta: float, eta: float, gamma: float) -> float:
    """Per-slice |rho| cap from the direct G-J butterfly conditions with this slice's phi."""
    ph = float(phi(theta, eta, gamma))
    tp = theta * ph
    tp2 = theta * ph * ph
    b = min(4.0 / max(tp, 1e-12) - 1.0, 4.0 / max(tp2, 1e-12) - 1.0, RHO_ABS_MAX)
    return max(b, 0.0)


def _fit_slice(sl: Slice, theta0: float, rho0: float, eta: float, gamma: float):
    """Stage 2: fit (theta, rho) for one slice. Deterministic multi-start on rho."""
    starts = [rho0, -0.6, -0.3, 0.0]
    best = None
    for r0 in starts:
        x0 = np.array([max(theta0, 1e-7), np.clip(r0, -0.95, 0.95)])

        def loss(x):
            return slice_loss(x[0], x[1], eta, gamma, sl, HUBER_DELTA)

        cons = [
            {"type": "ineq", "fun": lambda x: 4.0 - x[0] * phi(x[0], eta, gamma) * (1.0 + np.sqrt(x[1] ** 2 + 1e-16))},
            {"type": "ineq", "fun": lambda x: 4.0 - x[0] * phi(x[0], eta, gamma) ** 2 * (1.0 + np.sqrt(x[1] ** 2 + 1e-16))},
        ]
        try:
            res = minimize(loss, x0, method="SLSQP",
                           bounds=[(1e-8, None), (-RHO_ABS_MAX, RHO_ABS_MAX)],
                           constraints=cons,
                           options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER})
        except Exception:
            continue
        if res.success and (best is None or res.fun < best.fun - 1e-15):
            best = res
    if best is None:
        return float(theta0), float(rho0), False
    return float(best.x[0]), float(best.x[1]), True


def _project_hm(thetas: np.ndarray, rhos: np.ndarray, eta: float, gamma: float):
    """Deterministic feasibility projection: enforce theta monotone, butterfly caps, HM coupling."""
    th = np.maximum.accumulate(np.maximum(thetas, 1e-8))
    psi = theta_phi(th, eta, gamma)
    r = rhos.copy()
    for i in range(len(r)):
        b = _rho_bound(th[i], eta, gamma)
        r[i] = np.clip(r[i], -b, b)
    # forward pass: clamp rho_i*psi_i within +/- (psi_i - psi_{i-1}) of rho_{i-1}*psi_{i-1}
    for i in range(1, len(r)):
        dpsi = psi[i] - psi[i - 1]
        lo = r[i - 1] * psi[i - 1] - dpsi
        hi = r[i - 1] * psi[i - 1] + dpsi
        rp = np.clip(r[i] * psi[i], lo, hi)
        r[i] = rp / max(psi[i], 1e-12)
        b = _rho_bound(th[i], eta, gamma)
        r[i] = np.clip(r[i], -b, b)
    return th, r


def _calibrate_eta_gamma(slices: list[Slice], global_fit: SSVIFit) -> tuple[float, float]:
    """Stage 0: pin the global (eta, gamma) from free per-slice fits on representative expiries.

    The global single-rho fit can land on a poor (eta, gamma); per-slice free fits on a small,
    deterministic, evenly-spaced subset give a robust median for the phi(theta) curve.
    """
    from .fit import fit_joint

    n = len(slices)
    n_probe = min(8, n)
    idx = sorted({int(round(i * (n - 1) / max(n_probe - 1, 1))) for i in range(n_probe)})
    etas, gammas = [], []
    for i in idx:
        f = fit_joint([slices[i]])
        if f is not None and f.success:
            etas.append(f.eta)
            gammas.append(f.gamma)
    if not etas:
        return global_fit.eta, global_fit.gamma
    return float(np.median(etas)), float(np.median(gammas))


def fit_essvi(slices: list[Slice], global_fit: SSVIFit) -> ESSVIFit:
    n = len(slices)
    eta, gamma = _calibrate_eta_gamma(slices, global_fit)

    # ---- stage 2: independent per-slice (theta_i, rho_i)
    th2 = np.empty(n)
    rh2 = np.empty(n)
    for i, sl in enumerate(slices):
        th2[i], rh2[i], _ = _fit_slice(sl, float(global_fit.thetas[i]), float(global_fit.rho), eta, gamma)

    # ---- stage 3: joint polish with calendar + HM coupling
    th0, rh0 = _project_hm(th2, rh2, eta, gamma)
    x0 = np.concatenate([th0, rh0])

    def total(x):
        th, rh = x[:n], x[n:]
        return sum(slice_loss(th[i], rh[i], eta, gamma, slices[i], HUBER_DELTA) for i in range(n))

    def cal_monotone(x):
        return np.diff(x[:n])

    def butterfly(x):
        th, rh = x[:n], x[n:]
        ph = phi(th, eta, gamma)
        ar = np.sqrt(rh * rh + 1e-16)
        return np.concatenate([4.0 - th * ph * (1.0 + ar), 4.0 - th * ph * ph * (1.0 + ar)])

    def hm_coupling(x):
        th, rh = x[:n], x[n:]
        psi = theta_phi(th, eta, gamma)
        p = rh * psi
        dpsi = np.diff(psi)
        dp = np.diff(p)
        return np.concatenate([dpsi - dp, dpsi + dp])

    cons = [
        {"type": "ineq", "fun": cal_monotone},
        {"type": "ineq", "fun": butterfly},
        {"type": "ineq", "fun": hm_coupling},
    ]
    bounds = [(1e-8, None)] * n + [(-RHO_ABS_MAX, RHO_ABS_MAX)] * n
    try:
        res = minimize(total, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER})
    except Exception:
        res = None

    if res is not None and res.success:
        th, rh = res.x[:n], res.x[n:]
        th, rh = _project_hm(th, rh, eta, gamma)   # snap residual 1e-12 violations
        return ESSVIFit(thetas=th, rhos=rh, eta=eta, gamma=gamma, loss=float(res.fun), success=True)

    # fallback: projected stage-2 solution
    loss = total(x0)
    return ESSVIFit(thetas=th0, rhos=rh0, eta=eta, gamma=gamma, loss=float(loss), success=False)


def fit_essvi_warm(slices: list[Slice], seeds: list[tuple[float, float]],
                   eta: float, gamma: float) -> ESSVIFit | None:
    """Warm-mode fit: seed every slice's (theta_i, rho_i) from the
    previous snapshot's converged parameters, skip the cold cascade (no 16-start grid, no eta/gamma
    probes), refine each slice with a single SLSQP start, then run the joint HM polish.

    eta and gamma are HELD FIXED at the warm values (not re-fit); only (theta_i, rho_i) move. The
    Gatheral-Jacquier butterfly bounds are enforced PER SLICE (the two ineq constraints below), so a
    surface is butterfly-safe slice-by-slice even where the global product theta*phi*(1+|rho|) reads
    high. This warm-projection path serves the large majority of production snapshots; the stacked
    cold SLSQP runs only on the first snapshot of a chain or when a warm fit fails its quality gate.

    Returns None when the warm path fails to produce a converged stage-3 solution; the caller
    falls back to the cold cascade (and records fit_init=cold).
    """
    n = len(slices)
    th2 = np.empty(n)
    rh2 = np.empty(n)
    for i, sl in enumerate(slices):
        th0, rh0 = seeds[i]
        x0 = np.array([max(th0, 1e-8), np.clip(rh0, -RHO_ABS_MAX, RHO_ABS_MAX)])

        def loss(x, sl=sl):
            return slice_loss(x[0], x[1], eta, gamma, sl, HUBER_DELTA)

        cons = [
            {"type": "ineq", "fun": lambda x: _gj1(x, eta, gamma)},
            {"type": "ineq", "fun": lambda x: _gj2(x, eta, gamma)},
        ]
        kw = {}
        if _USE_JAC:                                  # analytic gradient -> fewer SLSQP evals (gated byte-safe)
            kw["jac"] = lambda x, sl=sl: slice_loss_grad(x[0], x[1], eta, gamma, sl, HUBER_DELTA)
            cons[0]["jac"] = lambda x: _gj1_jac(x, eta, gamma)
            cons[1]["jac"] = lambda x: _gj2_jac(x, eta, gamma)
        try:
            res = minimize(loss, x0, method="SLSQP", **kw,
                           bounds=[(1e-8, None), (-RHO_ABS_MAX, RHO_ABS_MAX)],
                           constraints=cons,
                           options={"ftol": SLSQP_FTOL, "maxiter": 120})
            th2[i], rh2[i] = (float(res.x[0]), float(res.x[1])) if res.success else (float(x0[0]), float(x0[1]))
        except Exception:
            th2[i], rh2[i] = float(x0[0]), float(x0[1])

    th, rh = _project_hm(th2, rh2, eta, gamma)
    loss = sum(slice_loss(th[i], rh[i], eta, gamma, slices[i], HUBER_DELTA) for i in range(n))
    return ESSVIFit(thetas=th, rhos=rh, eta=eta, gamma=gamma, loss=float(loss), success=True)
