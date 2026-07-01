"""SSVI model + Gatheral-Jacquier no-arbitrage quantities."""
import numpy as np
from essvi.ssvi import model


def test_w_positive_and_atm():
    k = np.linspace(-1, 1, 41)
    theta, rho, eta, gamma = 0.04, -0.3, 0.7, 0.45
    w = model.w_of_k(k, theta, rho, eta, gamma)
    assert np.all(w > 0)
    # at k=0, w = theta (ATM total variance by construction)
    w0 = model.w_of_k(np.array([0.0]), theta, rho, eta, gamma)
    assert np.isclose(w0[0], theta, atol=1e-12)


def test_gj_butterfly_satisfied_when_eta_bound_holds():
    # eta*(1+|rho|) <= 2 should imply both butterfly conditions at every theta
    thetas = np.array([0.005, 0.02, 0.08, 0.2])
    rho, gamma = -0.3, 0.45
    eta = 2.0 / (1.0 + abs(rho)) - 1e-6      # right at the sufficient bound
    a, b = model.gj_butterfly_values(thetas, rho, eta, gamma)
    assert np.all(a < 4.0), a
    assert np.all(b <= 4.0 + 1e-12), b


def test_density_nonnegative_for_calm_smile():
    # a well-behaved arbitrage-free slice should have non-negative density across the core
    k = np.linspace(-0.5, 0.5, 101)
    g = model.implied_density(k, theta=0.04, rho=-0.3, eta=0.7, gamma=0.45)
    assert np.all(g > 0), g.min()


def test_theta_phi_monotone():
    th = np.linspace(0.001, 0.5, 50)
    tp = model.theta_phi(th, eta=0.7, gamma=0.45)
    assert np.all(np.diff(tp) > 0)            # monotone increasing
    # analytic derivative matches a tight central difference at sample points
    h = 1e-6
    pts = np.array([0.01, 0.05, 0.1, 0.3])
    d = model.d_theta_phi(pts, eta=0.7, gamma=0.45)
    fd = (model.theta_phi(pts + h, 0.7, 0.45) - model.theta_phi(pts - h, 0.7, 0.45)) / (2 * h)
    assert np.allclose(d, fd, rtol=1e-5), (d, fd)


def test_implied_density_integrates_to_one():
    # The Gatheral-Jacquier g(k) defines the risk-neutral density of log-strike:
    #   p(k) = g(k)/sqrt(2*pi*w(k)) * exp(-d2(k)^2/2),  d2 = -k/sqrt(w) - sqrt(w)/2
    # which integrates to 1 in the forward measure. The correct second-term coefficient
    # (w'^2/4) gives 1.000; the old (w'/4)^2 gave ~1.039 (a 4x-too-small butterfly penalty).
    trap = getattr(np, "trapezoid", None) or np.trapz
    for theta, rho, eta, gamma in [(0.04, -0.3, 0.7, 0.45), (0.02, -0.5, 0.9, 0.40),
                                   (0.10, -0.1, 0.5, 0.45), (0.25, -0.2, 0.6, 0.45)]:
        k = np.linspace(-6.0, 6.0, 300001)
        w = model.w_of_k(k, theta, rho, eta, gamma)
        g = model.implied_density(k, theta, rho, eta, gamma)
        d2 = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
        p = g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d2 * d2)
        integral = float(trap(p, k))
        assert abs(integral - 1.0) < 2e-3, (theta, rho, eta, gamma, integral)
