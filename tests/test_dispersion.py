"""Tests for the P5 implied/realized correlation solvers (derived/dispersion.py) vs synthetic ground truth."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "derived"))
import dispersion as D  # noqa: E402


def test_implied_correlation_recovers_known_rho():
    rng = np.random.default_rng(0)
    n = 20
    w = rng.uniform(0.5, 2.0, n); w /= w.sum()
    sig = rng.uniform(0.15, 0.6, n)
    iv = sig ** 2
    own = float(np.sum(w ** 2 * sig ** 2)); cross = float(np.sum(w * sig) ** 2 - own)
    for rho_true in (0.2, 0.5, 0.8):
        iv_index = own + rho_true * cross                          # index variance from the identity
        rho, diag = D.implied_correlation(iv_index, iv, w)
        assert diag["status"] == "ok" and abs(rho - rho_true) < 1e-9


def test_implied_correlation_clamps_above_one():
    w = np.ones(5) / 5; sig = np.full(5, 0.3); iv = sig ** 2
    iv_index = 0.135                                                # > the rho=1 max (0.09) -> rho>1 -> clamp
    rho, diag = D.implied_correlation(iv_index, iv, w)
    assert diag["status"] == "clamped" and rho == 1.0


def test_implied_correlation_drops_nan_members():
    w = np.array([0.4, 0.3, 0.3]); iv = np.array([0.09, np.nan, 0.04])
    rho, diag = D.implied_correlation(0.06, iv, w)
    assert diag["n"] == 2                                           # the NaN member removed from the basket


def test_implied_correlation_insufficient():
    rho, diag = D.implied_correlation(np.nan, np.array([0.09, 0.04]), np.array([0.5, 0.5]))
    assert np.isnan(rho) and diag["status"] == "insufficient"


def test_dispersion_metrics_full_set():
    rng = np.random.default_rng(2); n = 30
    w = rng.uniform(0.5, 2.0, n); sig = rng.uniform(0.15, 0.6, n); iv = sig ** 2
    wn = w / w.sum()
    own = float(np.sum(wn ** 2 * sig ** 2)); cross = float(np.sum(wn * sig) ** 2 - own)
    rho_true = 0.45
    iv_index = own + rho_true * cross
    m = D.dispersion_metrics(iv_index, iv, w)
    assert m["status"] == "ok"
    assert abs(m["rho_clean"] - rho_true) < 1e-9             # clean = CBOE COR algebra (machine precision)
    # CORRECTED ordering (the spec's GATE-2 "dirty<=clean" is backwards): the naive dirty proxy OVER-states
    # correlation because it attributes the own-variance floor to correlation -> rho_dirty >= rho_clean.
    assert m["rho_dirty"] >= m["rho_clean"] - 1e-12
    assert m["dispersion_var"] > 0 and m["dspx"] > 0         # diversified basket -> positive dispersion
    assert abs(m["missing_mass"]) < 1e-9                     # all members computable


def test_dispersion_metrics_perfect_correlation_limit():
    # GATE 2: rho_clean -> 1 AND rho_dirty -> 1 as all pairwise rho -> 1 (index var = squared weighted vol). NOTE
    # DSPX does NOT vanish here -- the dispersion Sum w sig^2 - (Sum w sig)^2 is the Jensen gap (cross-sectional
    # vol variance), so DSPX and the clean rho have DIFFERENT "perfect-correlation" points -- the multi-number
    # discipline: the three estimators are shipped side by side precisely because they differ.
    rng = np.random.default_rng(3); n = 20
    w = rng.uniform(0.5, 2.0, n); sig = rng.uniform(0.2, 0.5, n); iv = sig ** 2
    wn = w / w.sum()
    iv_index = float(np.sum(wn * sig) ** 2)                  # rho_ij = 1 (clean sense)
    m = D.dispersion_metrics(iv_index, iv, w)
    assert abs(m["rho_clean"] - 1.0) < 1e-9 and abs(m["rho_dirty"] - 1.0) < 1e-9


def test_dispersion_metrics_missing_mass():
    w = np.array([0.4, 0.3, 0.2, 0.1]); iv = np.array([0.09, 0.04, np.nan, 0.01])
    m = D.dispersion_metrics(0.05, iv, w)
    assert abs(m["missing_mass"] - 0.2) < 1e-9 and m["n"] == 3   # the 0.2-weight member dropped


def test_realized_correlation_pairwise_recovers():
    rng = np.random.default_rng(4)
    n, T, rho_t = 15, 500, 0.5
    f = rng.standard_normal(T)
    R = np.sqrt(rho_t) * f[:, None] + np.sqrt(1 - rho_t) * rng.standard_normal((T, n))
    rho, nu = D.realized_correlation_pairwise(R, np.ones(n) / n)
    assert nu == n and abs(rho - rho_t) < 0.1


def test_realized_correlation_recovers_factor_rho():
    rng = np.random.default_rng(1)
    n, T, rho_t = 12, 600, 0.4
    f = rng.standard_normal(T)
    R = (np.sqrt(rho_t) * f[:, None] + np.sqrt(1 - rho_t) * rng.standard_normal((T, n))) * 0.02
    rho, nu = D.realized_correlation(R, np.ones(n) / n)
    assert nu == n and abs(rho - rho_t) < 0.1


def test_implied_correlation_weight_scale_invariant():
    # the clean estimator is a ratio of weight-quadratics -> w and c*w MUST give the same rho (renormalization)
    rng = np.random.default_rng(7); n = 18
    w = rng.uniform(0.5, 2.0, n); sig = rng.uniform(0.15, 0.6, n); iv = sig ** 2
    wn = w / w.sum(); own = float(np.sum(wn ** 2 * sig ** 2)); cross = float(np.sum(wn * sig) ** 2 - own)
    iv_index = own + 0.45 * cross
    r1, _ = D.implied_correlation(iv_index, iv, w)
    r2, _ = D.implied_correlation(iv_index, iv, 100.0 * w)          # raw-weight footgun would diverge
    assert abs(r1 - r2) < 1e-12 and abs(r1 - 0.45) < 1e-9


def test_dispersion_metrics_reconstruction_identity():
    # the gate-G2 reconstruction: rho_clean == (iv_I - own_var)/(wsig_sq - own_var) from the returned aggregates
    rng = np.random.default_rng(8); n = 22
    w = rng.uniform(0.5, 2.0, n); sig = rng.uniform(0.15, 0.6, n); iv = sig ** 2
    wn = w / w.sum(); own = float(np.sum(wn ** 2 * sig ** 2)); cross = float(np.sum(wn * sig) ** 2 - own)
    iv_index = own + 0.37 * cross
    m = D.dispersion_metrics(iv_index, iv, w)
    recon = (iv_index - m["own_var"]) / (m["wsig_sq"] - m["own_var"])
    assert abs(recon - m["rho_clean"]) < 1e-12


def test_variance_contributions_sum_to_index_var():
    rng = np.random.default_rng(9); n = 25
    w = rng.uniform(0.5, 2.0, n); sig = rng.uniform(0.15, 0.6, n); iv = sig ** 2
    wn = w / w.sum(); own = float(np.sum(wn ** 2 * sig ** 2)); cross = float(np.sum(wn * sig) ** 2 - own)
    rho_true = 0.55; iv_index = own + rho_true * cross
    m = D.dispersion_metrics(iv_index, iv, w)
    idx, contrib, share = D.variance_contributions(iv_index, iv, w, m["rho_clean"])
    assert idx.size == n and abs(float(contrib.sum()) - iv_index) < 1e-9   # decomposition closes to the index var
    assert abs(float(share.sum()) - 1.0) < 1e-9


def test_realized_pairwise_complete_beats_zerofill_on_missing():
    # one column has 40% NaN; pairwise-complete should track the truth better than the old NaN->0 fill
    rng = np.random.default_rng(10); n, T, rho_t = 10, 400, 0.6
    f = rng.standard_normal(T)
    R = np.sqrt(rho_t) * f[:, None] + np.sqrt(1 - rho_t) * rng.standard_normal((T, n))
    miss = rng.random(T) < 0.4
    Rm = R.copy(); Rm[miss, 0] = np.nan
    rho_pc, _ = D.realized_correlation_pairwise(Rm, np.ones(n) / n)
    C_zero = np.corrcoef(np.where(np.isfinite(Rm), Rm, 0.0).T)       # the OLD biased path
    ww = np.outer(np.ones(n) / n, np.ones(n) / n); np.fill_diagonal(ww, 0.0)
    rho_zero = float(np.sum(ww * C_zero) / np.sum(ww))
    assert abs(rho_pc - rho_t) <= abs(rho_zero - rho_t) + 1e-9       # pairwise-complete no worse, and less biased
