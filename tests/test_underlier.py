"""Realized-variance leg (P4) -- core estimators tested against synthetic ground truth.

The realized leg runs on the de-Am parity-implied SPOT series (no traded underlier). Every estimator here is
a pure function of a 1-min log-price array; correctness is checked against a synthetic path whose integrated
variance, jump, and microstructure-noise level are KNOWN. Sources: Andersen-Dobrev-Schaumburg 2012 (MedRV);
Barndorff-Nielsen et al 2008 (realized kernel); Jacod et al 2009 (pre-averaging/MRC).
"""
import numpy as np
import pytest

from essvi.io import underlier


def _synth_logprice(n=390, iv_true=0.04, *, seed, omega=0.0, jump=None):
    """Deterministic synthetic intraday log-price path over a unit day.
    Constant spot vol sigma=sqrt(iv_true) -> integrated variance == iv_true. Optional i.i.d. noise (omega)
    and one additive jump (idx, size). Seeded Generator -> reproducible (no Math.random / wall-clock)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / n
    incr = np.sqrt(iv_true) * np.sqrt(dt) * rng.standard_normal(n)
    logp = np.concatenate([[0.0], np.cumsum(incr)])
    if jump is not None:
        idx, size = jump
        logp[idx:] += size
    if omega:
        logp = logp + omega * rng.standard_normal(logp.size)
    return logp


def test_realized_variance_recovers_integrated_variance():
    # average RV over many independent clean paths -> the integrated variance, to tight MC error
    iv = 0.04
    rvs = [underlier.realized_variance(_synth_logprice(390, iv, seed=s)) for s in range(400)]
    assert abs(np.mean(rvs) - iv) < 0.04 * 0.02            # <2% bias over 400 paths


def test_medrv_is_jump_robust_while_rv_is_not():
    # a single 5% jump must inflate RV by ~jump^2 but leave MedRV (continuous part) ~unchanged
    iv, jump = 0.04, 0.05
    clean = _synth_logprice(390, iv, seed=7)
    jumped = _synth_logprice(390, iv, seed=7, jump=(200, jump))
    assert underlier.realized_variance(jumped) - underlier.realized_variance(clean) > 0.5 * jump ** 2
    assert abs(underlier.med_rv(jumped) - underlier.med_rv(clean)) < 0.3 * iv


def test_medrv_unbiased_for_integrated_variance_no_jump():
    iv = 0.04
    medrvs = [underlier.med_rv(_synth_logprice(390, iv, seed=s)) for s in range(400)]
    assert abs(np.mean(medrvs) - iv) < 0.04 * 0.05        # MedRV slightly noisier than RV but ~unbiased


def test_signature_plot_flags_microstructure_noise():
    # i.i.d. noise biases finest-grid RV upward; the signature plot RV(Delta) must DECREASE as Delta coarsens
    iv, omega = 0.04, 0.004
    noisy = _synth_logprice(390, iv, seed=3, omega=omega)
    sig = underlier.volatility_signature(noisy, steps=(1, 2, 5, 10))
    assert sig[1] > sig[10]                                # finest grid most noise-inflated
    assert sig[1] > iv                                     # raw RV biased up by 2*n*omega^2


def test_bns_jump_z_calibrated_under_no_jumps():
    # the BNS z-statistic must be ~N(0,1) on continuous (no-jump) paths so a threshold (~|z|>3) is a real test
    zs = [underlier.bns_jump_z(_synth_logprice(390, 0.04, seed=s)) for s in range(500)]
    assert abs(np.mean(zs)) < 0.25 and 0.75 < np.std(zs) < 1.35


def test_bns_z_and_jump_variation_detect_a_jump():
    # realistic daily regime (iv=4e-4 ~ 2%/day vol); a 5% jump is then a true ~2.5-sigma discontinuity that
    # must (a) push z well past the 3-sigma threshold and (b) make JV ~ jump^2 while leaving the clean day clear
    iv = 4e-4
    clean = _synth_logprice(390, iv, seed=11)
    jumped = _synth_logprice(390, iv, seed=11, jump=(200, 0.05))
    assert underlier.bns_jump_z(jumped) > 3.0
    assert underlier.bns_jump_z(clean) < 3.0
    assert abs(underlier.jump_variation(jumped) - 0.05 ** 2) < 0.05 ** 2 * 0.45
    assert underlier.jump_variation(clean) < iv * 0.5     # ~no jump variation on a clean path


# --- P4 noise-robust + jump-timing estimators (parallel-built) -------------------------------------

def _synth_logprice_ar1_noise(n=390, iv_true=4e-4, *, seed, omega=0.006, phi=0.7):
    """Clean BM path PLUS AR(1)-DEPENDENT additive noise u_t = phi*u_{t-1} + e_t.
    AR(1) noise makes the noise-return autocovariance spill past lag 1, so q_hat > 1.
    Seeded Generator only (no Math.random / wall-clock)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / n
    incr = np.sqrt(iv_true) * np.sqrt(dt) * rng.standard_normal(n)
    logp = np.concatenate([[0.0], np.cumsum(incr)])
    m = logp.size
    e = omega * np.sqrt(1.0 - phi * phi) * rng.standard_normal(m)  # stationary innovation scale
    u = np.empty(m)
    u[0] = omega * rng.standard_normal()
    for t in range(1, m):
        u[t] = phi * u[t - 1] + e[t]
    return logp + u


def test_mrc_beats_raw_rv_under_iid_noise():
    # MODERATE noise (omega=0.002, ratio ~8) -- MRC's design regime, where the discrete-psi estimator recovers iv
    # to ~1%. At the EXTREME ratio 70 no n=390 estimator is unbiased; the selector routes those to the kernel and
    # test_mrc_never_negative_at_high_noise covers the floor there. raw RV is still badly noise-inflated here.
    iv, omega = 4e-4, 0.002
    n = 390
    mrcs, rvs = [], []
    for s in range(300):
        lp = _synth_logprice(n + 1, iv, seed=s, omega=omega)  # n returns
        mrcs.append(underlier.mrc_preavg(lp))
        rvs.append(underlier.realized_variance(lp))
    mrc_mean, rv_mean = np.mean(mrcs), np.mean(rvs)
    rv_noise_bias = 2.0 * n * omega ** 2          # ~3.1e-3, ~8x the signal
    # raw RV is dominated by noise
    assert rv_mean > iv + 0.5 * rv_noise_bias
    # MRC recovers iv far better than raw RV
    assert abs(mrc_mean - iv) < abs(rv_mean - iv)
    # and within an honest tolerance of the truth
    assert abs(mrc_mean - iv) < 0.20 * iv


def test_mrc_recovers_iv_on_clean_path():
    iv = 4e-4
    n = 390
    mrcs = [underlier.mrc_preavg(_synth_logprice(n + 1, iv, seed=s)) for s in range(300)]
    assert abs(np.mean(mrcs) - iv) < 0.20 * iv

def test_realized_kernel_recovers_iv_on_clean_path():
    # clean (noise-free) path: the Parzen RK ~ integrated variance to MC error, averaged over many paths
    iv = 4e-4
    rks = [underlier.realized_kernel(_synth_logprice(390, iv, seed=s)) for s in range(300)]
    assert abs(np.mean(rks) - iv) < iv * 0.05            # ~0.9% measured bias on clean paths


def test_realized_kernel_noise_robust_while_rv_is_biased():
    # i.i.d. noise (omega=0.006) inflates raw RV ~70x; the realized kernel is noise-robust and lands near truth.
    # With the spec-mandated noise estimator omega2_hat = -gamma_1, the non-flat-top Parzen RK carries a stable
    # finite-sample upward bias of ~20% at this extreme noise ratio (2*n*omega^2/iv ~ 70) -- a property of the
    # estimator, not MC error (SE over 300 paths ~3%). We assert <=22%; RK is still ~50x closer to truth than RV.
    iv, omega = 4e-4, 0.006
    paths = [_synth_logprice(390, iv, seed=s, omega=omega) for s in range(300)]
    rks = [underlier.realized_kernel(p) for p in paths]
    rvs = [underlier.realized_variance(p) for p in paths]
    mean_rk = float(np.mean(rks))
    mean_rv = float(np.mean(rvs))
    assert abs(mean_rk - iv) < iv * 0.22                 # RK noise-robust, near truth
    assert mean_rv > 5.0 * iv                            # raw RV badly biased up by ~2 n omega^2
    assert abs(mean_rk - iv) < 0.1 * abs(mean_rv - iv)   # RK dramatically closer to truth than RV


def test_realized_kernel_explicit_bandwidth_used():
    # passing bandwidth bypasses the auto rule; H=1 gives RK = gamma0 + 2*k(0)*gamma1 = RV + 2*gamma1
    p = _synth_logprice(390, 4e-4, seed=1, omega=0.006)
    r = np.diff(p)
    g0 = float(np.dot(r, r))
    g1 = float(np.dot(r[1:], r[:-1]))
    assert abs(underlier.realized_kernel(p, bandwidth=1) - (g0 + 2.0 * g1)) < 1e-12

def test_lee_mykland_detects_single_injected_jump():
    # one large jump (5% at idx 200) in a realistic daily regime (iv=4e-4 ~ 2%/day) must be timed to within
    # +/-1 of its true index on essentially every seeded path (Lee-Mykland 2008 Gumbel jump-timing test)
    iv, jidx, jsize = 4e-4, 200, 0.05
    hits = 0
    for s in range(200):
        lp = _synth_logprice(390, iv, seed=s, jump=(jidx, jsize))
        flags = underlier.lee_mykland_jumps(lp)
        if any(abs(f - jidx) <= 1 for f in flags):
            hits += 1
    assert hits >= 190, hits                                   # measured 200/200


def test_lee_mykland_few_false_flags_on_clean_paths():
    # on continuous (no-jump) paths the Gumbel threshold must control the family-wise error: the average
    # number of false jump flags across >=200 seeded clean paths is small (~alpha-level, here ~0)
    iv = 4e-4
    counts = [len(underlier.lee_mykland_jumps(_synth_logprice(390, iv, seed=1000 + s))) for s in range(200)]
    assert np.mean(counts) < 0.1, np.mean(counts)             # measured 0.0


def test_lee_mykland_returns_sorted_ints():
    lp = _synth_logprice(390, 4e-4, seed=5, jump=(200, 0.05))
    flags = underlier.lee_mykland_jumps(lp)
    assert flags == sorted(flags)
    assert all(isinstance(f, int) for f in flags)

def test_noise_features_omega2_recovers_iid_noise_variance_and_qhat_is_one():
    # i.i.d. noise omega=0.006 -> omega2 ~ omega^2 within ~25% (avg over >=200 paths); q_hat == 1
    omega = 0.006
    feats = [underlier.noise_features(_synth_logprice(390, 4e-4, seed=s, omega=omega))
             for s in range(200)]
    omega2_hat = np.mean([f["omega2"] for f in feats])
    assert abs(omega2_hat - omega ** 2) < 0.25 * omega ** 2
    # i.i.d. noise -> pure MA(1) returns: the order-of-dependence estimate is 1 (median over paths)
    qhats = np.array([f["q_hat"] for f in feats])
    assert np.median(qhats) == 1


def test_noise_features_naive_is_biased_above_clean_omega2():
    # naive (1/2n) sum r^2 carries IV/(2n) on top of omega^2, so it sits above the clean omega2 estimator
    omega = 0.006
    feats = [underlier.noise_features(_synth_logprice(390, 4e-4, seed=s, omega=omega))
             for s in range(200)]
    naive = np.mean([f["omega2_naive"] for f in feats])
    clean = np.mean([f["omega2"] for f in feats])
    assert naive > clean


def test_noise_features_signature_slope_negative_under_noise():
    # i.i.d. noise inflates the finest-grid RV most -> RV(step) decreases in step -> negative OLS slope
    omega = 0.006
    slopes = [underlier.noise_features(_synth_logprice(390, 4e-4, seed=s, omega=omega))["signature_slope"]
              for s in range(200)]
    assert np.mean(slopes) < 0.0


def test_noise_features_qhat_discriminates_dependent_noise():
    # q_hat>1 is the DEPENDENT-noise SIGNAL that gates estimator selection. Under the multiplicity-corrected
    # (Bonferroni) threshold it flags i.i.d. noise as q>1 only ~3% of the time (was ~45% with the naive 1.96
    # band -- the audit's "mis-routing" bug), but flags AR(1)-dependent noise several times more often. A perfect
    # per-day classifier of WEAK AR(1) is not achievable (its lag-2 return-autocorrelation is genuinely faint);
    # the discriminating power is what matters, and routing is second-order (both MRC and the kernel de-noise).
    fi = np.mean([underlier.noise_features(_synth_logprice(390, 4e-4, seed=s, omega=0.006))["q_hat"] > 1
                  for s in range(300)])
    fa = np.mean([underlier.noise_features(_synth_logprice_ar1_noise(390, 4e-4, seed=s, omega=0.006, phi=0.7))["q_hat"] > 1
                  for s in range(300)])
    assert fi < 0.08              # i.i.d. rarely flagged dependent (Bonferroni fix)
    assert fa > 4.0 * fi          # dependent noise flagged several times more often -- real discriminating power


# --- cross-day realized-vol pieces (5-min floor, whole-day combiner, estimator selection) ----------

def test_rv_5min_floor_more_robust_than_raw_rv_under_noise():
    # the averaged 5-min subgrid RV is the CONSERVATIVE floor: it cuts the residual noise bias ~5x (n/5 returns)
    # vs raw 1-min RV but does NOT eliminate it (that is MRC/kernel's job). At EXTREME noise it is dramatically
    # closer to truth than raw RV; at MODERATE noise it lands near truth.
    iv = 4e-4
    fe = np.mean([underlier.realized_variance_5min(_synth_logprice(390, iv, seed=s, omega=0.006)) for s in range(200)])
    re = np.mean([underlier.realized_variance(_synth_logprice(390, iv, seed=s, omega=0.006)) for s in range(200)])
    assert abs(fe - iv) < 0.3 * abs(re - iv)            # floor's residual noise bias ~5x smaller than raw RV's
    fm = np.mean([underlier.realized_variance_5min(_synth_logprice(390, iv, seed=s, omega=0.0015)) for s in range(200)])
    assert abs(fm - iv) < 2.0 * iv                      # near truth at a moderate noise level


def test_whole_day_rv_adds_floored_overnight():
    assert underlier.whole_day_rv(3e-4, 1e-4) == pytest.approx(4e-4)
    assert underlier.whole_day_rv(3e-4, -0.01) == pytest.approx(3e-4)   # negative overnight floored to 0


def test_select_estimator_follows_noise_diagnostics():
    # the chosen method follows the diagnostics: dependent (q_hat>1) OR high-magnitude noise (nr>20) -> kernel;
    # moderate i.i.d. noise -> MRC; else the 5-min floor. We replicate that exact decision and assert consistency
    # across moderate-noise, high-noise, and (ambiguous) weak-AR(1) days.
    for lp in (_synth_logprice(390, 4e-4, seed=2, omega=0.002),                          # moderate iid -> mrc
               _synth_logprice(390, 4e-4, seed=2, omega=0.006),                          # high iid -> kernel
               _synth_logprice_ar1_noise(390, 4e-4, seed=2, omega=0.006, phi=0.8)):      # ambiguous AR(1)
        sel = underlier.select_rv_estimator(lp)
        f = underlier.noise_features(lp); rk = underlier.realized_kernel(lp)
        nr = 2.0 * (lp.size - 1) * f["omega2"] / rk if rk > 0 else np.inf
        if f["q_hat"] > 1 or nr > 20.0:
            assert sel["method"].startswith("kernel")
        else:
            assert sel["method"].split("->")[0] in ("mrc", "rv5min")


# --- HARQ forecast (the expected-future-RV leg of the VRP) -----------------------------------------

def _synth_rv_series(n=400, *, seed, persistence=0.9, base=4e-4, vol=0.3):
    """Synthetic daily realized-variance series with HAR-like persistence: AR(1) in log-RV around a mean.
    Seeded Generator only. Returns an (n,) array of positive variances."""
    rng = np.random.default_rng(seed)
    mu = np.log(base)
    lrv = np.empty(n)
    lrv[0] = mu
    for t in range(1, n):
        lrv[t] = mu + persistence * (lrv[t - 1] - mu) + vol * rng.standard_normal()
    return np.exp(lrv)


def test_harq_forecast_is_strictly_pit_causal():
    # the forecast at t must use ONLY data up to t: recomputing on the truncated series reproduces it exactly
    rv = _synth_rv_series(320, seed=1)
    full = underlier.harq_forecast_series(rv)
    for t in (90, 160, 260):
        trunc = underlier.harq_forecast_series(rv[: t + 1])
        a, b = full[t], trunc[t]
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9   # no look-ahead


def test_harq_forecast_beats_random_walk_oos():
    # on a persistent, mean-reverting RV series HARQ's 1-step OOS MSE must beat the random walk (last value)
    rv = _synth_rv_series(450, seed=2)
    fc = underlier.harq_forecast_series(rv)
    valid = [t for t in range(60, len(rv) - 1) if not np.isnan(fc[t])]
    harq_mse = np.mean([(fc[t] - rv[t + 1]) ** 2 for t in valid])
    rw_mse = np.mean([(rv[t] - rv[t + 1]) ** 2 for t in valid])
    assert harq_mse < rw_mse


def test_harq_forecast_positive_and_finite():
    rv = _synth_rv_series(200, seed=3)
    fc = underlier.harq_forecast_series(rv)
    good = fc[~np.isnan(fc)]
    assert good.size > 0 and np.all(good > 0) and np.all(np.isfinite(good))


def test_harq_forecast_pit_at_multi_day_horizon():
    # at horizon>1 the forecast at t must NOT use training targets whose window extends past t (a subtle
    # look-ahead: y_s = mean RV over s+1..s+H is only OBSERVABLE once s+H <= t). Recomputing on the truncated
    # series rv[:t+1] must reproduce the full-series forecast exactly -- the load-bearing VRP leakage guard.
    rv = _synth_rv_series(360, seed=9)
    H = 21
    full = underlier.harq_forecast_series(rv, horizon=H)
    for t in (120, 200, 300):
        trunc = underlier.harq_forecast_series(rv[:t + 1], horizon=H)
        a, b = full[t], trunc[t]
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, (t, a, b)


# --- VRP netting core (implied minus annualized realized forecast) + cross-sectional z ---------------

def test_vrp_series_recovers_premium_and_is_pit():
    # synthetic: implied = annualized expected-forward realized * 1.25 (a 25% variance premium, by construction).
    # the netted vrp must be positive and ~ the premium fraction, and strictly causal (no look-ahead).
    rv = _synth_rv_series(500, seed=4)          # daily realized-variance series
    H, ann = 21, 252.0
    n = len(rv); iv = np.full(n, np.nan)
    for t in range(n - H):
        iv[t] = ann * rv[t + 1:t + 1 + H].mean() * 1.25     # implied richer than the TRUE forward realized
    vrp = underlier.vrp_series(rv, iv, horizon=H, ann=ann)
    valid = [t for t in range(60, n - H) if not np.isnan(vrp[t]) and not np.isnan(iv[t])]
    mv = float(np.mean([vrp[t] for t in valid])); miv = float(np.mean([iv[t] for t in valid]))
    assert mv > 0                                            # implied > expected realized
    assert 0.10 < mv / miv < 0.35                            # ~ the 0.25/1.25 = 0.20 premium fraction
    for t in (120, 250, 400):                                # strict PIT
        tr = underlier.vrp_series(rv[:t + 1], iv[:t + 1], horizon=H, ann=ann)
        a, b = vrp[t], tr[t]
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9


def test_cross_sectional_z_standardizes_and_precision_weights():
    v = np.array([1., 2., 3., 4., 5.])
    z = underlier.cross_sectional_z(v)
    assert abs(np.mean(z)) < 1e-9 and abs(np.std(z) - 1.0) < 1e-9    # zero-mean unit-sd
    # a NaN member is skipped, not propagated
    z2 = underlier.cross_sectional_z(np.array([1., np.nan, 3.]))
    assert np.isnan(z2[1]) and not np.isnan(z2[0]) and not np.isnan(z2[2])
    # precision weighting pulls the center toward the high-precision members
    vals = np.array([0., 10.]); prec = np.array([100.0, 1.0])       # first member far more precise
    zp = underlier.cross_sectional_z(vals, precisions=prec)
    assert abs(zp[0]) < abs(zp[1])                                  # near-zero deviation for the precise one


# --- adversarial-audit fixes: robustness + correct constants (16 confirmed bugs) --------------------

def test_mrc_never_negative_at_high_noise():
    # CRITICAL bug: noise-bias over-subtraction drove MRC negative ~30% of days at n=390 -> -inf VRP. Floored.
    iv = 4e-4; omega = np.sqrt(70 * iv / (2 * 390))          # the "extreme" noise ratio 70 the kernel docstring cites
    vals = [underlier.mrc_preavg(_synth_logprice(390, iv, seed=s, omega=omega)) for s in range(800)]
    assert min(vals) >= 0.0                                  # never a negative variance


def test_select_rv_estimator_clamps_nonneg_and_handles_degenerate():
    iv = 4e-4; omega = np.sqrt(70 * iv / (2 * 390))
    for s in range(400):
        sel = underlier.select_rv_estimator(_synth_logprice(390, iv, seed=s, omega=omega))
        assert sel["rv"] >= 0.0                              # selector never ships negative RV
    deg = underlier.select_rv_estimator(np.array([4.6, np.nan, 4.6, 4.7] + [4.6] * 37))  # a NaN price day
    assert np.isnan(deg["rv"]) and deg["method"] == "degenerate"


def test_harq_robust_to_zero_and_nan_days():
    # a single RV=0 (flat/stale spot) or NaN day must NOT crash or inject inf into the whole forecast
    rv = _synth_rv_series(400, seed=5)
    rv0 = rv.copy(); rv0[100] = 0.0
    f0 = underlier.harq_forecast_series(rv0, horizon=21)
    assert np.all(np.isfinite(f0[~np.isnan(f0)])) and not np.any(np.isinf(f0))
    rvn = rv.copy(); rvn[123] = np.nan
    fn = underlier.harq_forecast_series(rvn, horizon=21)     # must not raise LinAlgError
    assert np.all(np.isfinite(fn[~np.isnan(fn)]))


def test_harq_no_spurious_premium_at_multi_day_horizon():
    # the horizon>1 target must be log(MEAN of RV levels), not mean(log RV); else a ~13% geometric-vs-arithmetic
    # low bias inflates the VRP. On a flat-synthetic where implied == true forward realized, the SYSTEMATIC vrp
    # (seed-averaged -- a single 600-day path carries ~8% sampling noise) must be ~0.
    H, ann = 21, 252.0
    rels = []
    for sd in range(8):
        rv = _synth_rv_series(600, seed=sd); n = len(rv); iv = np.full(n, np.nan)
        for t in range(n - H):
            iv[t] = ann * rv[t + 1:t + 1 + H].mean()        # implied == TRUE forward realized (zero true premium)
        vrp = underlier.vrp_series(rv, iv, horizon=H, ann=ann)
        valid = [t for t in range(80, n - H) if np.isfinite(vrp[t])]
        rels.append(np.mean([vrp[t] for t in valid]) / np.mean([iv[t] for t in valid]))
    assert abs(np.mean(rels)) < 0.05                         # systematic premium ~0.027 (was ~+0.13 pre-fix)


def test_bns_jump_z_tightly_calibrated_at_production_resolution():
    # _BNS_THETA must be the MedRV constant (~0.96), not the bipower 0.609 -> std(z)~1, not ~1.25
    zs = [underlier.bns_jump_z(_synth_logprice(390, 0.04, seed=s)) for s in range(2000)]
    assert 0.88 < np.std(zs) < 1.12                         # tight band the old 0.609 fails


def test_cross_sectional_z_untrusted_precision_not_scored():
    # a member with NaN/zero precision must NOT receive a (spurious off-scale) z -> NaN, not a number
    z = underlier.cross_sectional_z(np.array([0., 10., 5.]), precisions=np.array([1.0, 1.0, np.nan]))
    assert np.isnan(z[2]) and np.isfinite(z[0]) and np.isfinite(z[1])


def test_lee_mykland_fwer_near_alpha_on_clean_paths():
    # without the spurious 1/c the family-wise false-flag rate ~ alpha (was ~0.0005, far too conservative -> misses jumps)
    flagged = sum(1 for s in range(2000) if underlier.lee_mykland_jumps(_synth_logprice(390, 4e-4, seed=s), alpha=0.01))
    assert 0.003 < flagged / 2000 < 0.03


def test_degenerate_inputs_return_nan_not_false_clean():
    deg = np.array([4.6, 4.6])                      # < 3 returns
    short_const = np.array([4.6, 4.6, 4.6])         # too short to subsample at step 5
    assert np.isnan(underlier.bns_jump_z(deg))
    assert np.isnan(underlier.bns_jump_z(np.array([np.nan] * 10)))
    assert np.isnan(underlier.bns_jump_z(np.array([4.6] * 10)))   # constant -> zero variance -> unknown, not "no jump"
    assert np.isnan(underlier.jump_variation(deg))
    # 5-min floor falls back to the plain RV (0.0) on a too-short series instead of NaN (consistency with realized_variance)
    assert underlier.realized_variance_5min(short_const) == underlier.realized_variance(short_const) == 0.0


# --- re-audit fixes: noise-magnitude routing + whole-day/empty guards -------------------------------

def test_select_rv_estimator_routes_high_noise_to_kernel():
    # at the EXTREME noise ratio 70 the selector must route to the always-PSD kernel (NOT floored MRC, which
    # ships ~5.7x-biased there). Shipped rv then ~within 30% of iv. Re-audit material fix.
    iv = 4e-4; omega = np.sqrt(70 * iv / (2 * 390))
    sels = [underlier.select_rv_estimator(_synth_logprice(390, iv, seed=s, omega=omega)) for s in range(300)]
    assert np.mean([s["rv"] for s in sels]) < 1.5 * iv                      # not the old 5.65x
    assert np.mean([s["method"].startswith("kernel") for s in sels]) > 0.8  # mostly routed to the kernel


def test_select_rv_returns_nr_quality_signal():
    # select_rv exposes nr = 2*n*omega^2/kernel (the run_derived NAME-level suitability signal): finite+small on
    # a clean path, larger on a noisy one, +inf on a degenerate day.
    clean = _synth_logprice(390, 4e-4, seed=1)
    sel = underlier.select_rv_estimator(clean)
    assert "nr" in sel and np.isfinite(sel["nr"]) and sel["nr"] >= 0
    noisy = underlier.select_rv_estimator(_synth_logprice(390, 4e-4, seed=1, omega=0.006))
    assert noisy["nr"] > sel["nr"]
    assert underlier.select_rv_estimator(np.array([1.0, np.nan, 1.0, 1.0]))["nr"] == float("inf")


def test_harq_masks_nan_days_not_floored():
    # untrusted (NaN) days must be DROPPED from the regression, not floored to ~0. A near-constant RV series with
    # ~8% NaN days must still forecast ~the same level as the clean series (flooring would drag it well down).
    rng = np.random.default_rng(9)
    rv = 4e-4 * np.exp(rng.normal(0.0, 0.05, 400))            # ~constant RV
    rvn = rv.copy(); rvn[::13] = np.nan                       # ~8% of days untrusted
    f_full = underlier.harq_forecast_series(rv, horizon=21)
    f_nan = underlier.harq_forecast_series(rvn, horizon=21)
    ok = np.isfinite(f_full) & np.isfinite(f_nan)
    assert ok.sum() > 50
    ratio = float(np.median(f_nan[ok] / f_full[ok]))
    assert 0.9 < ratio < 1.1                                  # masked ~ full (flooring would push << 0.9)


def test_harq_nan_current_day_yields_nan_forecast():
    # if day t's own RV is untrusted (NaN), there is no daily regressor -> NO forecast (NaN), not a guess
    rv = _synth_rv_series(400, seed=4)
    rv[300] = np.nan
    f = underlier.harq_forecast_series(rv, horizon=21)
    assert np.isnan(f[300])


def test_whole_day_rv_guards_both_legs():
    assert np.isnan(underlier.whole_day_rv(np.nan, 1e-4))      # degenerate open-to-close -> unknown
    assert underlier.whole_day_rv(3e-4, np.nan) == pytest.approx(3e-4)   # NaN overnight floored to 0
    assert underlier.whole_day_rv(3e-4, -0.01) == pytest.approx(3e-4)


def test_harq_empty_series_returns_empty():
    assert underlier.harq_forecast_series(np.array([])).size == 0


# NOTE: the realized leg's overnight is supplied by run_derived as the ROBUST observable overnight gap of the
# de-Am implied-spot open/close (validated clean: SPY overnight 7.1% / 33% of total variance, glitches/splits
# winsorized). The model-free implied-overnight (forward-variance difference of consecutive surfaces) is a
# RISK-NEUTRAL quantity -- using it as the realized overnight would zero the overnight VRP -- so it is NOT part
# of P4; it is reserved for a future overnight-VRP decomposition (realized gap vs implied forward variance).
