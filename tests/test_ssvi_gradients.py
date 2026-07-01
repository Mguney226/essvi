"""Finite-difference verification of the analytic SSVI gradients used to give SLSQP a `jac=`.

Each analytic gradient must match a central finite difference of the function it differentiates.
These are the gradients wired into the warm per-slice fit (essvi.py) to remove SLSQP's finite-diff
overhead WITHOUT changing the converged fit. Test-first: the gradient functions below do not exist
yet, so this fails at import until they are implemented.
"""
from __future__ import annotations

import numpy as np
import pytest

from essvi.ssvi.model import (
    phi, w_of_k, d_theta_phi,
    dphi_dtheta, dw_dtheta, dw_drho,   # NEW (to implement)
)
from essvi.ssvi.objective import Slice, slice_loss, slice_loss_grad  # slice_loss_grad NEW

# representative (theta, rho, eta, gamma) points well inside the feasible box
PARAMS = [
    (0.04, -0.30, 0.70, 0.45),
    (0.12, -0.55, 0.90, 0.30),
    (0.01, 0.10, 0.50, 0.20),
    (0.30, -0.80, 1.20, 0.48),
]
KS = np.array([-0.6, -0.2, -0.05, 0.0, 0.05, 0.2, 0.6])


def _fd(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pytest.mark.parametrize("theta,rho,eta,gamma", PARAMS)
def test_dphi_dtheta(theta, rho, eta, gamma):
    ana = float(dphi_dtheta(theta, eta, gamma))
    num = _fd(lambda th: float(phi(th, eta, gamma)), theta)
    assert ana == pytest.approx(num, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("theta,rho,eta,gamma", PARAMS)
def test_dw_dtheta(theta, rho, eta, gamma):
    ana = dw_dtheta(KS, theta, rho, eta, gamma)
    num = np.array([_fd(lambda th: float(w_of_k(np.array([k]), th, rho, eta, gamma)[0]), theta) for k in KS])
    np.testing.assert_allclose(ana, num, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("theta,rho,eta,gamma", PARAMS)
def test_dw_drho(theta, rho, eta, gamma):
    ana = dw_drho(KS, theta, rho, eta, gamma)
    num = np.array([_fd(lambda r: float(w_of_k(np.array([k]), theta, r, eta, gamma)[0]), rho) for k in KS])
    np.testing.assert_allclose(ana, num, rtol=1e-6, atol=1e-9)


def _make_slice(theta, rho, eta, gamma):
    """A slice whose observed variance is the model surface perturbed smoothly (residuals away from the
    Huber kink), so the objective gradient is well-defined and FD-checkable."""
    k = KS.copy()
    w_model = w_of_k(k, theta, rho, eta, gamma)
    rng = np.linspace(-0.4, 0.4, k.size)            # small, deterministic residuals (smooth Huber region)
    sl = Slice(expiry="T", t=0.1, k=k, w_obs=w_model * (1.0 + 0.02 * rng),
               omega=np.linspace(0.5, 1.5, k.size), forward=100.0, discount=1.0,
               theta_atm=theta, median_spread=0.01, scale=0.05)
    return sl


@pytest.mark.parametrize("theta,rho,eta,gamma", PARAMS)
def test_slice_loss_grad(theta, rho, eta, gamma):
    sl = _make_slice(theta, rho, eta, gamma)
    delta = 1.345
    g = slice_loss_grad(theta, rho, eta, gamma, sl, delta)        # -> array([d/dtheta, d/drho])
    g_th = _fd(lambda th: slice_loss(th, rho, eta, gamma, sl, delta), theta)
    g_rh = _fd(lambda r: slice_loss(theta, r, eta, gamma, sl, delta), rho)
    np.testing.assert_allclose(g, [g_th, g_rh], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("theta,rho,eta,gamma", PARAMS)
def test_warm_constraint_jacobians(theta, rho, eta, gamma):
    """The two warm-path G-J butterfly constraints (essvi.py:199-200) and their analytic jacobians."""
    from essvi.ssvi.essvi import _gj1, _gj2, _gj1_jac, _gj2_jac  # NEW module-level helpers
    x = np.array([theta, rho])
    for fun, jac in ((_gj1, _gj1_jac), (_gj2, _gj2_jac)):
        ana = jac(x, eta, gamma)
        num = np.array([
            _fd(lambda t: fun(np.array([t, rho]), eta, gamma), theta),
            _fd(lambda r: fun(np.array([theta, r]), eta, gamma), rho),
        ])
        np.testing.assert_allclose(ana, num, rtol=1e-6, atol=1e-9)
