"""arb_ok no-arbitrage verifier: per-slice shape + independent corroboration of the butterfly check."""
import numpy as np

from essvi.diagnostics import verify


def test_verify_surface_is_per_slice():
    # arb_ok is reported PER SLICE (one verdict per expiry), not a single global flag.
    thetas = np.array([0.01, 0.04, 0.09])
    rhos = np.array([-0.3, -0.3, -0.3])
    arb, bfly_min, cal_min = verify.verify_surface(thetas, rhos, 0.7, 0.45)
    assert arb.shape == (3,) and bfly_min.shape == (3,) and cal_min.shape == (3,)
    assert arb.dtype == bool
    assert np.all(arb)                                  # this calm surface is arb-free on every slice


def test_fd_butterfly_agrees_with_analytic_on_arbfree():
    # On well-behaved arb-free slices, both the analytic g(k) and the structurally-independent
    # finite-difference Breeden-Litzenberger density are non-negative.
    for theta, rho, eta, gamma in [(0.04, -0.3, 0.7, 0.45), (0.02, -0.5, 0.9, 0.40),
                                   (0.10, -0.1, 0.5, 0.45), (0.25, -0.2, 0.6, 0.45)]:
        ok_a, gmin = verify.butterfly_ok(theta, rho, eta, gamma)
        ok_fd, cmin = verify.butterfly_ok_fd(theta, rho, eta, gamma)
        assert ok_a and ok_fd, (theta, rho, eta, gamma, gmin, cmin)


def test_fd_butterfly_independently_catches_violations():
    # The FD-BL check uses NO analytic g(k); it must independently flag genuine butterfly violations,
    # proving the analytic check is corroborated by an independent method (not merely self-confirming).
    fd_on_violations = []
    for theta in (0.3, 0.5, 0.8, 1.0):
        for eta in (4.0, 5.0, 6.0):
            for rho in (-0.6, -0.3):
                ok_a, _ = verify.butterfly_ok(theta, rho, eta, 0.45)
                if not ok_a:                            # analytic says this surface violates butterfly
                    fd_on_violations.append(verify.butterfly_ok_fd(theta, rho, eta, 0.45)[0])
    assert len(fd_on_violations) >= 10, "no genuine violations found to corroborate against"
    agreed = sum(1 for ok_fd in fd_on_violations if not ok_fd)
    # FD has a coarser noise floor, so it may miss only hairline-marginal cases; require strong agreement
    assert agreed >= 0.85 * len(fd_on_violations), f"FD corroborated {agreed}/{len(fd_on_violations)}"
