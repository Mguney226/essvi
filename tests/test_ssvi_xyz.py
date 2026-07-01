"""Golden test vs the reference §10 worked example, plus a synthetic fit round-trip."""
import numpy as np

from essvi.ssvi import model
from essvi.ssvi.fit import fit_joint
from essvi.ssvi.objective import Slice
from essvi.repro import xyz_reference as xyz


def test_reference_phi_simple_matches():
    p1 = model.phi_simple(xyz.THETA_1, xyz.ETA, xyz.GAMMA)
    p2 = model.phi_simple(xyz.THETA_2, xyz.ETA, xyz.GAMMA)
    assert abs(p1 - xyz.PHI_1) < 1e-3, (p1, xyz.PHI_1)
    assert abs(p2 - xyz.PHI_2) < 1e-3, (p2, xyz.PHI_2)


def test_reference_gj_butterfly_values():
    s = 1.0 + abs(xyz.RHO)
    a1 = xyz.THETA_1 * xyz.PHI_1 * s
    b1 = xyz.THETA_1 * xyz.PHI_1**2 * s
    a2 = xyz.THETA_2 * xyz.PHI_2 * s
    b2 = xyz.THETA_2 * xyz.PHI_2**2 * s
    assert abs(a1 - xyz.GJ_1A) < 1e-3 and abs(b1 - xyz.GJ_1B) < 1e-3
    assert abs(a2 - xyz.GJ_2A) < 1e-3 and abs(b2 - xyz.GJ_2B) < 1e-3
    # all well inside the no-arb thresholds (<4, <=4)
    assert a1 < 4 and a2 < 4 and b1 <= 4 and b2 <= 4


def test_full_form_phi_within_half_percent_of_simple():
    p_full = model.phi(xyz.THETA_1, xyz.ETA, xyz.GAMMA)
    assert abs(p_full - xyz.PHI_1) / xyz.PHI_1 < 0.005


def _make_synthetic_slices(thetas, rho, eta, gamma):
    slices = []
    for i, th in enumerate(thetas):
        k = np.linspace(-0.35, 0.35, 21)
        w = model.w_of_k(k, th, rho, eta, gamma)
        slices.append(
            Slice(
                expiry=f"E{i}",
                t=0.05 * (i + 1),
                k=k,
                w_obs=w,
                omega=np.ones_like(k),
                forward=5000.0,
                discount=0.99,
                theta_atm=float(model.w_of_k(np.array([0.0]), th, rho, eta, gamma)[0]),
                median_spread=0.01,
            )
        )
    return slices


def test_synthetic_fit_recovers_parameters():
    thetas_true = np.array([0.006, 0.012, 0.02, 0.03, 0.045])
    rho_t, eta_t, gamma_t = -0.30, 1.0, 0.45    # eta*(1+|rho|)=1.3 < 2 (feasible)
    slices = _make_synthetic_slices(thetas_true, rho_t, eta_t, gamma_t)
    fit = fit_joint(slices)
    assert fit is not None and fit.success
    assert abs(fit.rho - rho_t) < 0.02, fit.rho
    assert abs(fit.eta - eta_t) < 0.05, fit.eta
    assert abs(fit.gamma - gamma_t) < 0.05, fit.gamma
    assert np.allclose(fit.thetas, thetas_true, atol=5e-4), fit.thetas
    assert fit.loss < 1e-8, fit.loss
