"""SSVI total-variance surface and the power-law phi(theta) family.

    w(k, theta) = (theta/2) * { 1 + rho*phi*k + sqrt[ (phi*k + rho)^2 + (1 - rho^2) ] }
    phi(theta)  = eta / [ theta^gamma * (1 + theta)^(1 - gamma) ]

theta_i (ATM total variance, one per listed expiry) are fitted; (rho, eta, gamma) are the three
global skew/curvature parameters shared across the whole term structure.
"""
from __future__ import annotations

import numpy as np


def phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return eta / (np.power(theta, gamma) * np.power(1.0 + theta, 1.0 - gamma))


def phi_simple(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """Simplified power law eta/theta^gamma. Used only by the methodology §10 hand-arithmetic
    worked example; production uses phi() (the (1+theta)^(1-gamma) form), within ~0.5% at these theta."""
    theta = np.asarray(theta, dtype=float)
    return eta / np.power(theta, gamma)


def w_of_k(k: np.ndarray, theta: float, rho: float, eta: float, gamma: float) -> np.ndarray:
    """Total variance w(k) for a single slice at ATM variance `theta`."""
    k = np.asarray(k, dtype=float)
    ph = float(phi(theta, eta, gamma))
    disc = np.sqrt((ph * k + rho) ** 2 + (1.0 - rho * rho))
    return 0.5 * theta * (1.0 + rho * ph * k + disc)


def total_variance_grid(
    k: np.ndarray, thetas: np.ndarray, rho: float, eta: float, gamma: float
) -> np.ndarray:
    """w for each (expiry, k): returns array shape (n_expiry, n_k)."""
    thetas = np.asarray(thetas, dtype=float)
    k = np.asarray(k, dtype=float)
    out = np.empty((thetas.shape[0], k.shape[0]), dtype=float)
    for i, th in enumerate(thetas):
        out[i] = w_of_k(k, th, rho, eta, gamma)
    return out


def theta_phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """theta * phi(theta) = eta * (theta / (1 + theta))^(1 - gamma). Monotone increasing in theta."""
    theta = np.asarray(theta, dtype=float)
    return eta * np.power(theta / (1.0 + theta), 1.0 - gamma)


def d_theta_phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """d/dtheta [ theta*phi(theta) ]. With g(theta)=theta/(1+theta), d(theta*phi)=eta*(1-gamma)*g^(-gamma)*g'."""
    theta = np.asarray(theta, dtype=float)
    g = theta / (1.0 + theta)
    gp = 1.0 / (1.0 + theta) ** 2
    return eta * (1.0 - gamma) * np.power(g, -gamma) * gp


# --- analytic gradients of w(k) w.r.t. the fitted params (for SLSQP jac=, byte-safe speedup) ---

def dphi_dtheta(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """d phi / d theta. ln phi = ln eta - gamma*ln(theta) - (1-gamma)*ln(1+theta)
    => d phi/d theta = -phi * ( gamma/theta + (1-gamma)/(1+theta) ).  (Note: PLUS inside, not minus.)"""
    theta = np.asarray(theta, dtype=float)
    ph = phi(theta, eta, gamma)
    return -ph * (gamma / theta + (1.0 - gamma) / (1.0 + theta))


def dw_dtheta(k: np.ndarray, theta: float, rho: float, eta: float, gamma: float) -> np.ndarray:
    """d w(k) / d theta (eta, gamma fixed). w = 0.5*theta*S, S = 1 + rho*ph*k + disc,
    disc = sqrt((ph*k+rho)^2 + (1-rho^2)); only ph=phi(theta) carries the theta-dependence inside S."""
    k = np.asarray(k, dtype=float)
    ph = float(phi(theta, eta, gamma))
    dph = float(dphi_dtheta(theta, eta, gamma))
    disc = np.sqrt((ph * k + rho) ** 2 + (1.0 - rho * rho))
    S = 1.0 + rho * ph * k + disc
    dS = dph * k * (rho + (ph * k + rho) / disc)
    return 0.5 * S + 0.5 * theta * dS


def dw_drho(k: np.ndarray, theta: float, rho: float, eta: float, gamma: float) -> np.ndarray:
    """d w(k) / d rho (theta, eta, gamma fixed). d disc/d rho = ph*k/disc."""
    k = np.asarray(k, dtype=float)
    ph = float(phi(theta, eta, gamma))
    disc = np.sqrt((ph * k + rho) ** 2 + (1.0 - rho * rho))
    return 0.5 * theta * ph * k * (1.0 + 1.0 / disc)


# --- Gatheral-Jacquier no-arbitrage quantities (used by constraints.py and verify.py) ---

def gj_butterfly_values(thetas: np.ndarray, rho: float, eta: float, gamma: float):
    """Return the two G-J butterfly quantities per expiry: theta*phi*(1+|rho|) and theta*phi^2*(1+|rho|).
    Butterfly-free requires the first < 4 and the second <= 4 at every theta."""
    thetas = np.asarray(thetas, dtype=float)
    ph = phi(thetas, eta, gamma)
    a = thetas * ph * (1.0 + abs(rho))
    b = thetas * ph * ph * (1.0 + abs(rho))
    return a, b


def calendar_slope_cap(theta: np.ndarray, rho: float, eta: float, gamma: float) -> np.ndarray:
    """Upper bound for d_theta[theta*phi]: (1/rho^2)*(1 + sqrt(1 - rho^2)) * phi(theta) (§04).
    The lower bound (>= 0) holds automatically for this monotone family."""
    ph = phi(theta, eta, gamma)
    r2 = max(rho * rho, 1e-12)
    return (1.0 / r2) * (1.0 + np.sqrt(max(1.0 - rho * rho, 0.0))) * ph


def implied_density(k: np.ndarray, theta: float, rho: float, eta: float, gamma: float) -> np.ndarray:
    """Gatheral-Jacquier risk-neutral density proxy g(k); g(k) >= 0 <=> no butterfly arbitrage.

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2,  where w', w'' are dw/dk, d2w/dk2.
    (Gatheral-Jacquier 2014; the second-term coefficient is w'^2/4, NOT (w'/4)^2.)
    """
    k = np.asarray(k, dtype=float)
    ph = float(phi(theta, eta, gamma))
    root = np.sqrt((ph * k + rho) ** 2 + (1.0 - rho * rho))
    # w and derivatives in k
    w = 0.5 * theta * (1.0 + rho * ph * k + root)
    dw = 0.5 * theta * (rho * ph + ph * (ph * k + rho) / root)
    d2w = 0.5 * theta * (ph * ph * (1.0 - rho * rho)) / (root ** 3)
    g = (1.0 - k * dw / (2.0 * w)) ** 2 - (dw * dw / 4.0) * (1.0 / w + 0.25) + 0.5 * d2w
    return g
