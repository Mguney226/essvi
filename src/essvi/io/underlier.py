"""underlier.py -- realized-variance leg (P4) over the de-Am parity-implied SPOT series.

The OPRA archive has NO traded underlier and NO overnight tape, so the physical leg is built from the 1-min
de-Americanized parity-implied spot persisted in `underlier_state` (disclosed everywhere). The parity
regression produces DEPENDENT (price-correlated), NOT i.i.d., microstructure noise -- so estimator choice is
driven by the volatility SIGNATURE PLOT, and i.i.d.-noise (TSRV-style) corrections are not assumed.

This module is a library of PURE FUNCTIONS of a 1-min log-price array; the cross-date netting + forecast
orchestration lives in `derived/run_derived.py`. Built incrementally with TDD against synthetic ground truth.

Estimators (sources): realized variance; MedRV (Andersen-Dobrev-Schaumburg 2012, jump-robust); volatility
signature (Andersen-Bollerslev-Diebold-Labys); [next] pre-averaging/MRC (Jacod-Li-Mykland-Podolskij-Vetter
2009), realized kernel (Barndorff-Nielsen-Hansen-Lunde-Shephard 2008), BNS jump test, HARQ forecast
(Bollerslev-Patton-Quaedvlieg 2016), Hansen-Lunde whole-day with model-free overnight.
"""
from __future__ import annotations

import numpy as np

# MedRV normalization: 1/E[median(|Z1|,|Z2|,|Z3|)^2] for standard normals = pi/(6-4*sqrt(3)+pi) ~ 1.4193,
# so c * median(|r|)^2 is unbiased for the local variance of one return (Andersen-Dobrev-Schaumburg 2012).
_MEDRV_C = float(np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi))


def _log_returns(log_prices) -> np.ndarray:
    return np.diff(np.asarray(log_prices, dtype=float))


def realized_variance(log_prices) -> float:
    """Sum of squared 1-min log returns over the window. Unbiased for integrated variance in the ABSENCE of
    noise and jumps; in their presence it is inflated by 2*n*omega^2 (noise) and by the squared jumps."""
    r = _log_returns(log_prices)
    return float(np.dot(r, r))


def med_rv(log_prices) -> float:
    """MedRV (Andersen-Dobrev-Schaumburg 2012): a jump-robust estimator of the CONTINUOUS integrated
    variance. Uses the median of each consecutive triple of |returns|, which is insensitive to a lone jump
    return (the jump sits in at most one of the three and the median discards it)."""
    r = np.abs(_log_returns(log_prices))
    m = r.size
    if m < 3:
        return float("nan")
    trip_med = np.median(np.stack([r[:-2], r[1:-1], r[2:]]), axis=0)   # m-2 overlapping triples
    return float(_MEDRV_C * (m / (m - 2.0)) * np.dot(trip_med, trip_med))


# MedRQ normalization (Andersen-Dobrev-Schaumburg 2012): makes c * n * sum(median|r|^4) unbiased for the
# integrated quarticity, the denominator of the BNS jump test.
_MEDRQ_C = float(3.0 * np.pi / (9.0 * np.pi + 72.0 - 52.0 * np.sqrt(3.0)))
# variance constant of the MedRV ratio jump statistic. The textbook pi^2/4+pi-5=0.609 is for BIPOWER variation;
# our continuous estimator is MedRV, whose ratio-test variance is ~0.96 (measured n*Var(RJ)=0.978 at n=390, the
# 1-min production resolution; ->0.93 asymptotically). Using the bipower 0.609 with MedRV inflated std(z) to ~1.25
# (double the false-positive rate) -- caught by the adversarial audit.
_BNS_THETA = 0.96


def med_rq(log_prices) -> float:
    """MedRQ: jump-robust integrated QUARTICITY (Andersen-Dobrev-Schaumburg 2012), the variance scale for the
    BNS jump test. Same median-of-triples construction as MedRV but on the 4th power."""
    r = np.abs(_log_returns(log_prices))
    m = r.size
    if m < 3:
        return float("nan")
    trip_med = np.median(np.stack([r[:-2], r[1:-1], r[2:]]), axis=0)
    return float(_MEDRQ_C * m * (m / (m - 2.0)) * np.dot(trip_med ** 2, trip_med ** 2))


def jump_variation(log_prices) -> float:
    """Jump variation JV = max(RV - MedRV, 0): the discontinuous part of quadratic variation (BNS split).
    RV captures total QV (continuous + jumps); MedRV captures only the continuous part. NB MedRV is NOT robust
    to two jumps inside ONE median triple (adjacent-minute jumps) -- it can then exceed RV and zero JV; the
    Lee-Mykland timing test is the cross-check for that rare case. NaN on degenerate (<3-return) input."""
    rv, mrv = realized_variance(log_prices), med_rv(log_prices)
    if not np.isfinite(rv) or not np.isfinite(mrv):
        return float("nan")
    return float(max(rv - mrv, 0.0))


def bns_jump_z(log_prices) -> float:
    """Barndorff-Nielsen-Shephard ratio jump test statistic (~N(0,1) under the null of no jumps). A large
    positive z (>~3) flags a statistically significant jump day. Uses the MedRV/MedRQ jump-robust pair so the
    test is itself insensitive to the jump it is detecting."""
    rv, mrv, mrq = realized_variance(log_prices), med_rv(log_prices), med_rq(log_prices)
    m = _log_returns(log_prices).size
    if not (rv > 0) or not (mrv > 0) or not np.isfinite(mrq):
        return float("nan")        # degenerate/unusable input is UNKNOWN, not a clean "no jump" (was 0.0 ->
        #                            laundered bad data as a jump-free day). Audit fix.
    rj = (rv - mrv) / rv                                       # relative jump measure
    denom = np.sqrt(_BNS_THETA * (1.0 / m) * max(1.0, mrq / mrv ** 2))
    return float(rj / denom) if denom > 0 else 0.0


def volatility_signature(log_prices, steps=(1, 2, 5, 10)) -> dict[int, float]:
    """Realized variance recomputed at coarsening sampling intervals (every `step`-th observation). On a
    DEPENDENT-noise series the finest grid is the most noise-inflated, so a downward-sloping RV(step) plot
    is the diagnostic that licenses (or rejects) the noise-robust correction. Returns {step: RV}."""
    lp = np.asarray(log_prices, dtype=float)
    out: dict[int, float] = {}
    for s in steps:
        r = np.diff(lp[::s])
        out[int(s)] = float(np.dot(r, r))
    return out


# --- noise-robust + jump-timing estimators (P4, parallel-built + integration-verified) ----------

def mrc_preavg(log_prices, theta: float = 1.0 / 3.0) -> float:
    """MRC / pre-averaging noise-robust integrated-variance estimator (Jacod-Li-Mykland-Podolskij-Vetter
    2009; Christensen-Kinnebrock-Podolskij 2010). Pre-averages the 1-min returns over a window
    k_n = floor(theta*sqrt(n)) with the weight g(x)=min(x,1-x) (so psi1=int g'^2=1, psi2=int g^2=1/12),
    which annihilates i.i.d. microstructure noise faster than the signal accumulates, then removes the
    residual noise bias. Far less inflated than raw RV (which carries a 2*n*omega^2 noise bias).

        MRC = (n/(n-k_n+2)) * (1/(psi2*k_n)) * sum_i Ybar_i^2  -  (psi1/(theta_eff^2*psi2)) * omega2_hat

    with Ybar_i = sum_{j=1}^{k_n-1} g(j/k_n)*r_{i+j}, theta_eff = k_n/sqrt(n), and omega2_hat = -gamma_1
    (the negative lag-1 return autocovariance), the i.i.d.-noise variance estimate that is more robust at
    1-min sampling than (1/2n)*sum r^2."""
    r = _log_returns(log_prices)
    n = r.size
    if n < 4:
        return float("nan")

    k_n = int(np.floor(theta * np.sqrt(n)))
    if k_n < 2:
        k_n = 2
    theta_eff = k_n / np.sqrt(n)

    # g(j/k_n) on the full grid j=0..k_n (g(0)=g(1)=0). Use the DISCRETE finite-sample normalizers (CKP 2010):
    # psi2_kn=(1/k_n) sum_{j=1}^{k_n-1} g^2 (=0.0857 at k_n=6 vs the continuous 1/12=0.0833) and psi1_kn=
    # k_n sum (g_j-g_{j-1})^2 (=1 for this piecewise-linear g). The continuous 1/12 over-subtracted noise (more
    # negative days) AND added a ~6% clean-path inflation at the production k_n=6 (adversarial audit).
    jj = np.arange(0, k_n + 1) / k_n
    gj = np.minimum(jj, 1.0 - jj)
    psi2 = float(np.sum(gj[1:k_n] ** 2) / k_n)
    psi1 = float(k_n * np.sum(np.diff(gj) ** 2))
    g = gj[1:k_n]                                     # pre-averaging weights g(j/k_n), j=1..k_n-1

    # Ybar_i = sum_{j=1}^{k_n-1} g(j/k_n) * r_{i+j}, i = 0 .. n-k_n
    # sliding weighted sum: cross-correlate r with the weight vector g (in index order j=1..k_n-1)
    ybar = np.correlate(r, g, mode="valid")
    rv_pa = (n / (n - k_n + 2.0)) * (1.0 / (psi2 * k_n)) * np.dot(ybar, ybar)

    # omega2_hat = -gamma_1 (negative lag-1 autocovariance of returns); under i.i.d. additive noise
    # r_i = (efficient) + (eps_i - eps_{i-1}) so E[gamma_1] = -omega^2.
    gamma1 = float(np.dot(r[:-1], r[1:]) / n)
    omega2_hat = max(-gamma1, 0.0)

    bias = (psi1 / (theta_eff ** 2 * psi2)) * omega2_hat
    # FLOOR at 0: at the production k_n=6 the noise-bias subtraction over-shoots into NEGATIVE "variance" on ~30%
    # of high-noise days (adversarial audit), which would poison the downstream log-HARQ forecast (-> +/-inf VRP).
    # A realized variance is non-negative by definition.
    return float(max(rv_pa - bias, 0.0))

def _parzen_weight(x: np.ndarray) -> np.ndarray:
    """Parzen kernel weight k(x): 1 - 6x^2 + 6x^3 on [0, 1/2], 2(1-x)^3 on (1/2, 1], 0 for x > 1
    (Barndorff-Nielsen-Hansen-Lunde-Shephard 2008)."""
    x = np.asarray(x, dtype=float)
    w = np.zeros_like(x)
    lo = (x >= 0.0) & (x <= 0.5)
    hi = (x > 0.5) & (x <= 1.0)
    w[lo] = 1.0 - 6.0 * x[lo] ** 2 + 6.0 * x[lo] ** 3
    w[hi] = 2.0 * (1.0 - x[hi]) ** 3
    return w


def realized_kernel(log_prices, bandwidth: int | None = None) -> float:
    """Parzen realized kernel (Barndorff-Nielsen-Hansen-Lunde-Shephard 2008, Econometrica 76(6):1481-1536):
    a noise-robust estimator of integrated variance that is consistent under i.i.d. microstructure noise.
    RK = gamma_0 + sum_{h=1}^{H} k((h-1)/H) * (gamma_h + gamma_{-h}), where gamma_h = sum_t r_t r_{t-h} is the
    h-th realized autocovariance (gamma_{-h} = gamma_h for a single series, so the term is 2*gamma_h) and k is
    the Parzen kernel. The autocovariance terms cancel the bias that i.i.d. noise injects into raw RV.

    The (non-flat-top) Parzen bandwidth follows BNHLS 2009 (J. Econometrics): H = ceil(c* * xi^(4/5) * n^(3/5))
    with c* = 3.5134, xi^2 = omega2_hat / sqrt(IQ_hat), omega2_hat = -gamma_1 (the i.i.d. noise-variance
    estimator) and IQ_hat the integrated quarticity (med_rq). H is guarded to [1, n-1]. If `bandwidth` is given
    it is used directly (still guarded).

    NB: with omega2_hat = -gamma_1, RK is essentially unbiased on a clean path and noise-robust under noise, but
    carries a stable finite-sample upward bias (~20% at an extreme noise ratio 2*n*omega^2/iv ~ 70) versus the
    ~70x bias of raw RV at the same ratio."""
    r = _log_returns(log_prices)
    n = r.size
    if n < 2:
        return float("nan")

    def gamma(h: int) -> float:
        if h == 0:
            return float(np.dot(r, r))
        return float(np.dot(r[h:], r[:-h]))

    g0 = gamma(0)

    if bandwidth is None:
        c_star = 3.5134
        omega2_hat = max(-gamma(1), 0.0)            # i.i.d. noise variance: -gamma_1
        iq_hat = med_rq(log_prices)                 # integrated quarticity (jump-robust)
        if omega2_hat <= 0.0 or not np.isfinite(iq_hat) or iq_hat <= 0.0:
            H = 1
        else:
            xi2 = omega2_hat / np.sqrt(iq_hat)
            H = int(np.ceil(c_star * (xi2 ** (2.0 / 5.0)) * (n ** (3.0 / 5.0))))
    else:
        H = int(bandwidth)

    H = max(1, min(H, n - 1))

    rk = g0
    for h in range(1, H + 1):
        w = float(_parzen_weight(np.array([(h - 1) / H]))[0])
        if w == 0.0:
            continue
        rk += w * 2.0 * gamma(h)                     # gamma_h + gamma_{-h} = 2*gamma_h for one series
    return float(rk)

def lee_mykland_jumps(log_prices, window: int | None = None, alpha: float = 0.01) -> list[int]:
    """Lee-Mykland (2008) nonparametric intraday jump-TIMING test (Rev. Financial Studies 21:2535-2563).

    Standardizes each 1-min return by a LOCAL spot-vol estimate built from a trailing window of bipower
    variation, L_i = r_i / sigma_hat_i, with
        sigma_hat_i^2 = (1/(K-2)) * sum_{j=i-K+2}^{i-1} |r_{j-1}| |r_j| / mu1^2,   mu1 = sqrt(2/pi).
    Under the null of no jump the maximum of |L_i| obeys a Gumbel law; return i is flagged as a jump when
        |L_i| > G = beta_n * (-log(-log(1-alpha))) + C_n,
    with c = mu1, C_n = (2 log n)^.5 / c - (log pi + log log n)/(2 c (2 log n)^.5), beta_n = 1/(c (2 log n)^.5).
    The bipower construction makes sigma_hat itself jump-robust, so a jump return does not contaminate its own
    standardizer; the Gumbel threshold gives family-wise control at level alpha. Returns the sorted list of
    flagged return indices (0-based into the return series, i.e. into _log_returns).

    Args:
        log_prices: 1-min log-price array (length n+1 for n returns).
        window: trailing bipower window K. Default min(floor(sqrt(252*n)), n//4) -- the Lee-Mykland Table 1
            sqrt(252*n) local-window scale, capped to o(n) so a single-day series keeps a testable interior
            (the raw sqrt(252*n) exceeds n for n ~ 390 and would leave no return testable).
        alpha: family-wise significance level of the Gumbel test.

    Returns:
        Sorted list of return indices i (into _log_returns) flagged as jumps.
    """
    r = _log_returns(log_prices)
    n = r.size
    if n < 5:
        return []
    mu1 = np.sqrt(2.0 / np.pi)
    if window is not None:
        K = int(window)
    else:
        # Lee-Mykland Table 1 local window ~ sqrt(252*n); their asymptotics require K = o(n) so the trailing
        # bipower stays local. On a single-day series (n ~ 390) sqrt(252*n) exceeds n and leaves no testable
        # interior, so cap K to keep the bulk of returns testable (K = o(n), here ~ n/4).
        K = min(int(np.floor(np.sqrt(252.0 * n))), max(3, n // 4))
    K = max(K, 3)
    K = min(K, n)                                              # cannot look back further than we have

    absr = np.abs(r)
    bp = absr[:-1] * absr[1:]                                  # |r_{j-1}||r_j|, stored at index j-1 in 0..n-2

    c = mu1
    sqrt2logn = np.sqrt(2.0 * np.log(n))
    # sigma_hat is ALREADY mu1-normalized, so L_i ~ N(0,1) and the standard max-|N(0,1)| Gumbel constants apply
    # WITHOUT Lee-Mykland's 1/c factor -- including it double-counts the normalization and made the threshold too
    # high (measured FWER 0.0005 << alpha, missing jumps; removing it gives FWER ~0.0125 ~ alpha). Audit fix.
    _ = c  # (c = mu1 retained for the bipower local-vol estimate above; not used in the Gumbel constants)
    C_n = sqrt2logn - (np.log(np.pi) + np.log(np.log(n))) / (2.0 * sqrt2logn)
    beta_n = 1.0 / sqrt2logn
    G = beta_n * (-np.log(-np.log(1.0 - alpha))) + C_n

    flags: list[int] = []
    for i in range(K, n):
        # trailing bipower terms |r_{j-1}||r_j| for j = i-K+2 .. i-1  ->  bp index (j-1) = i-K+1 .. i-2
        seg = bp[i - K + 1:i - 1]
        if seg.size == 0:
            continue
        var_hat = seg.sum() / ((K - 2) * mu1 * mu1)
        sigma_hat = np.sqrt(var_hat)
        if sigma_hat <= 0.0:
            continue
        L = r[i] / sigma_hat
        if abs(L) > G:
            flags.append(int(i))
    return sorted(flags)

def noise_features(log_prices) -> dict:
    """Microstructure-noise DIAGNOSTICS over a 1-min log-price array that drive estimator selection.

    The parity-implied spot carries microstructure noise; whether it is i.i.d. or DEPENDENT
    (parity/strike-structured) decides whether i.i.d.-noise (TSRV-style) corrections are valid.
    Returns four diagnostics:

      omega2          : noise-variance estimate omega^2 = max(-gamma_1, 0), where gamma_1 is the
                        lag-1 sample autocovariance of returns. For i.i.d. additive noise the
                        observed returns are MA(1) with gamma_1 = -omega^2, so -gamma_1 is a clean,
                        signal-free estimate of the noise variance (Ait-Sahalia-Mykland-Zhang 2005;
                        Hansen-Lunde 2006).
      omega2_naive    : the high-frequency estimator (1/(2n)) * sum r^2. Equals omega^2 + IV/(2n),
                        i.e. biased upward by IV/(2n); shipped alongside omega2 so the signal
                        contamination of the naive estimator is visible.
      q_hat           : the largest lag h in 1..hmax at which the sample return autocorrelation rho_h
                        is significant under a white-noise null, using the Bartlett / Box-Jenkins
                        cumulative standard error sd(rho_h) ~ sqrt((1 + 2*sum_{j<h} rho_j^2) / n)
                        (Box-Jenkins 1976). This band widens once lower lags carry autocorrelation -- as
                        they do for the MA(1) returns induced by i.i.d. noise -- so q_hat == 1 under
                        i.i.d. noise (pure MA(1)); q_hat > 1 signals DEPENDENT (parity-structured)
                        noise whose autocovariance persists past lag 1, where i.i.d. corrections are
                        invalid.
      signature_slope : OLS slope of RV(step) versus step over steps 1..10 (the volatility
                        signature plot, Andersen-Bollerslev-Diebold-Labys 2000). Negative under
                        noise (the finest grid is the most noise-inflated).

    Parameters
    ----------
    log_prices : array_like
        1-min log-price path.

    Returns
    -------
    dict
        {"omega2": float, "omega2_naive": float, "q_hat": int, "signature_slope": float}.
    """
    r = _log_returns(log_prices)
    n = r.size
    if n < 3:
        return {"omega2": float("nan"), "omega2_naive": float("nan"),
                "q_hat": 0, "signature_slope": float("nan")}

    rc = r - r.mean()
    gamma0 = float(np.dot(rc, rc) / n)

    def gamma(h: int) -> float:
        return float(np.dot(rc[:-h], rc[h:]) / n)

    gamma1 = gamma(1)
    omega2 = max(-gamma1, 0.0)
    omega2_naive = float(np.dot(r, r) / (2.0 * n))

    # q_hat = largest lag h (<= hmax) whose sample return autocorrelation rho_h is significant under a
    # white-noise null. The significance band uses the Bartlett / Box-Jenkins cumulative standard error
    # sd(rho_h) ~ sqrt((1 + 2*sum_{j<h} rho_j^2) / n), which widens once lower lags carry autocorrelation
    # (as they do for the MA(1) returns induced by i.i.d. noise). This suppresses spurious high-lag
    # crossings so q_hat == 1 under i.i.d. noise, while q_hat > 1 flags genuine DEPENDENT
    # (parity-structured) noise whose autocovariance persists past lag 1.
    hmax = min(20, n - 1)
    # BONFERRONI threshold across the hmax lags (audit fix): the naive 1.96 band let a lone high-lag false
    # positive set q_hat=20 on ~42% of i.i.d. days. A multiplicity-corrected z = Phi^{-1}(1 - 0.05/(2*hmax))
    # suppresses those spurious crossings while still catching genuine multi-lag (AR(1)) dependence. (Contiguous-
    # from-lag-1 was tried and UNDER-detected AR(1); the corrected max-significant-lag threads both.)
    from scipy.stats import norm as _norm
    zthr = float(_norm.ppf(1.0 - 0.05 / (2.0 * hmax)))
    q_hat = 0
    if gamma0 > 0:
        cum = 0.0
        for h in range(1, hmax + 1):
            rho_h = gamma(h) / gamma0
            se = np.sqrt((1.0 + 2.0 * cum) / n)
            if abs(rho_h) > zthr * se:
                q_hat = h
            cum += rho_h * rho_h

    steps = np.arange(1, 11)
    rvs = np.array([volatility_signature(log_prices, steps=(int(s),))[int(s)] for s in steps])
    x = steps.astype(float)
    signature_slope = float(np.polyfit(x, rvs, 1)[0])

    return {"omega2": float(omega2), "omega2_naive": omega2_naive,
            "q_hat": int(q_hat), "signature_slope": signature_slope}


# --- cross-day pieces: conservative floor, whole-day combiner, signature-plot estimator selection -----

def realized_variance_5min(log_prices, step: int = 5) -> float:
    """Conservative 5-min realized-variance FLOOR: average RV over the `step` offset subgrids (subsample to
    1-in-`step`, average the subgrids -- ZMA-style averaging cuts variance). Far less noise-inflated than 1-min
    RV; the robust fallback when the high-frequency noise estimate is untrustworthy (Liu-Patton-Sheppard 2015:
    5-min RV is hard to beat)."""
    lp = np.asarray(log_prices, dtype=float)
    rvs = []
    for off in range(step):
        sub = lp[off::step]
        if sub.size >= 2:
            r = np.diff(sub)
            rvs.append(float(np.dot(r, r)))
    # fall back to the plain RV when the series is too short to subsample (keeps it consistent with
    # realized_variance, which returns 0.0 on a constant/short series rather than NaN). Audit fix.
    return float(np.mean(rvs)) if rvs else realized_variance(lp)


def whole_day_rv(rv_open_to_close: float, overnight_variance: float) -> float:
    """Hansen-Lunde (2005) whole-day realized variance = noise-robust open-to-close RV + (floored) overnight
    variance. run_derived supplies the overnight piece as the ROBUST squared overnight gap of the de-Am
    implied-spot open/close (close_{t-1} -> open_t), winsorized for single-minute fit glitches and with splits
    excluded; this puts the realized leg on the same whole-day calendar clock as the implied iv_var. A
    degenerate open-to-close (NaN/inf) makes the whole day UNKNOWN -> NaN (not silently passed through); a
    NaN/negative overnight is floored to 0 so a bad gap never poisons or reduces the day."""
    oc = float(rv_open_to_close)
    if not np.isfinite(oc):
        return float("nan")
    ov = float(overnight_variance)
    return oc + (max(ov, 0.0) if np.isfinite(ov) else 0.0)


def select_rv_estimator(log_prices) -> dict:
    """Signature-plot estimator SELECTION: the noise diagnostics decide which open-to-close RV estimator to
    trust on this name-day. DEPENDENT (parity-structured) noise (q_hat>1) -> the realized kernel (handles
    serially-correlated noise); i.i.d. noise with a downward-sloping signature -> MRC pre-averaging; otherwise
    the conservative 5-min floor. Returns {rv, method, q_hat, nr, signature_slope, rv_floor}, where `nr` =
    2*n*omega^2/kernel is the day's noise-to-signal ratio -- the orchestrator's NAME-level suitability signal."""
    lp = np.asarray(log_prices, dtype=float)
    # degenerate guard: a NaN/short price day has no trustworthy RV -> flag it explicitly, never ship a bare NaN
    # into the downstream whole-day/HARQ (audit: silent NaN propagation).
    if lp.size < 3 or not np.all(np.isfinite(lp)):
        return {"rv": float("nan"), "method": "degenerate", "q_hat": 0, "nr": float("inf"),
                "signature_slope": float("nan"), "rv_floor": float("nan")}
    f = noise_features(lp)
    floor = realized_variance_5min(lp)
    rk = realized_kernel(lp)                      # ~unbiased signal estimate -> the noise denominator AND the
    q = int(f["q_hat"]) if f["q_hat"] is not None else 0   # high-noise choice (always PSD, no flooring bias)
    n_ret = lp.size - 1
    # noise-to-signal ratio ~= 2*n*omega^2 / IV (dividing by the ~unbiased kernel, NOT the noise-inflated 5-min
    # floor which would deflate it). The floored MRC develops a large UPWARD bias once noise dominates (re-audit:
    # +38% at ratio 70), so route by noise MAGNITUDE, not just q_hat: dependent (q>1) OR high noise -> kernel;
    # moderate i.i.d. noise -> MRC (cheaper + slightly better in its regime); otherwise the 5-min floor.
    nr = (2.0 * n_ret * f["omega2"] / rk) if (np.isfinite(rk) and rk > 0) else np.inf
    # NOTE: there is NO reliable per-DAY "untrusted" gate. A name whose de-Am SPOT is pervasively noise-dominated
    # (an illiquid EM/sector ETF) is statistically indistinguishable PER DAY from recoverable i.i.d. noise (same
    # nr, same q_hat) -- yet the kernel inflates its RV because the true signal is tiny relative to the noise.
    # That is caught at the NAME level in run_derived via the median nr (a VRP-INDEPENDENT spot-quality signal);
    # `nr` is returned for it. Per day we still route by noise magnitude (q>1 OR high nr -> the always-PSD kernel).
    if q > 1 or nr > 20.0:
        method, rv = "kernel", rk
    elif f["signature_slope"] < 0 and q <= 1:
        method, rv = "mrc", mrc_preavg(lp)        # moderate i.i.d. noise -> pre-averaging
    else:
        method, rv = "rv5min", floor              # otherwise the conservative 5-min floor
    # a non-positive / non-finite chosen estimate falls back to the conservative 5-min floor, then clamps >= 0
    # (a realized variance cannot be negative; the MRC floor + this guard kill the -inf-VRP path).
    if not np.isfinite(rv) or rv <= 0.0:
        rv, method = (floor if np.isfinite(floor) else 0.0), method + "->floor"
    rv = max(float(rv), 0.0)
    return {"rv": rv, "method": method, "q_hat": q, "nr": float(nr),
            "signature_slope": float(f["signature_slope"]), "rv_floor": float(floor)}


# --- HARQ forecast: expected future realized variance (the leg the VRP nets against) -----------------

def harq_forecast_series(rv_series, rq_series=None, *, horizon: int = 1, train: int = 250,
                         min_train: int = 50, dates=None) -> np.ndarray:
    """HARQ forecast series (Bollerslev-Patton-Quaedvlieg 2016), STRICTLY point-in-time. For each date t,
    forecasts E_t[mean RV over the next `horizon` days] from a LOG-HAR regression (daily/weekly/monthly lags)
    whose DAILY coefficient is scaled by sqrt(RQ_t) -- down-weighting the daily lag when it is noisily measured.
    The regression is OLS on the trailing window of (regressors at s, realized target over s+1..s+horizon) pairs
    with s < t ONLY -- no look-ahead, the load-bearing leakage guard for the VRP. Reduces to log-HAR when
    `rq_series` is omitted (the Q column is then zero, harmlessly rank-deficient). Returns an (n,) array;
    entries with insufficient trailing history are NaN.
    """
    rv = np.asarray(rv_series, dtype=float)
    n = rv.size
    if n == 0:
        return np.full(0, np.nan)                           # empty series -> empty forecast (no ValueError)
    # ISLAND mask (optional `dates`): the history has ~6-month gaps; a HAR weekly/monthly lag or a forward target
    # that is ARRAY-adjacent across a gap mixes two disjoint regimes. blk[s] = start index of s's contiguous
    # island, blk_end[s] = its last index, so lags clamp to >= blk[s] and the target stays <= blk_end[s].
    blk = np.zeros(n, dtype=int); blk_end = np.full(n, n - 1, dtype=int)
    if dates is not None and len(dates) == n:
        import datetime as _dt
        def _d(x): return x.date() if isinstance(x, _dt.datetime) else x
        for i in range(1, n):
            blk[i] = i if (_d(dates[i]) - _d(dates[i - 1])).days > 7 else blk[i - 1]
        for i in range(n - 2, -1, -1):
            blk_end[i] = i if (_d(dates[i + 1]) - _d(dates[i])).days > 7 else blk_end[i + 1]
    rq = None if rq_series is None else np.asarray(rq_series, dtype=float)
    valid = np.isfinite(rv)                                 # untrusted-noise / missing (NaN/inf) days -> MASKED
    # A genuine flat/stale day (FINITE, RV<=0) is real ~0 variance -> floored so log() is finite. A MASKED day
    # (NaN) is MISSING, NOT low-variance -> kept NaN and DROPPED from the regression, never floored to ~0 (which
    # would bias a name's forecast down by ~ its untrusted-day fraction). floor + the exp() overflow cap come
    # ONLY from the initial window rv[:min_train] (always past for any forecast date) -> no look-ahead.
    init_fin = rv[:max(min_train, 1)]
    init_fin = init_fin[np.isfinite(init_fin) & (init_fin > 0)]
    floor = max(float(np.median(init_fin) * 1e-4) if init_fin.size else 1e-300, 1e-300)
    cap = float(np.log(1e6 * float(np.max(init_fin)))) if init_fin.size else 700.0
    rvc = np.where(valid & (rv > 0), rv, np.where(valid, floor, np.nan))   # masked -> NaN; flat -> floor
    lrv = np.log(rvc)                                       # NaN exactly on masked days

    def _wmean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float("nan")

    def regressors(s: int) -> np.ndarray:
        d = lrv[s]
        w = _wmean(lrv[max(blk[s], s - 4):s + 1])        # weekly (5-day) lag, clamped to s's island, masked ignored
        m = _wmean(lrv[max(blk[s], s - 21):s + 1])       # monthly (22-day) lag, clamped to s's island
        q = np.sqrt(max(rq[s], 0.0)) if rq is not None else 0.0
        return np.array([1.0, d, w, m, q * d])           # HARQ: q*d = the RQ-scaled daily term

    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - train)
        X, y = [], []
        # STRICT PIT: a training pair (X_s, y_s) is usable at t only once its target window is fully realized,
        # i.e. s + horizon <= t (the mean RV over s+1..s+horizon is observed by t). Using s up to t-1 would
        # let the most-recent targets peek past t -- the multi-day VRP look-ahead this guard exists to kill.
        for s in range(lo, t - horizon + 1):
            tw = rvc[s + 1:min(s + 1 + horizon, blk_end[s] + 1)]   # target clamped to s's island (no cross-gap days)
            if not np.any(np.isfinite(tw)):
                continue                                     # all-masked forward window -> unusable pair
            xs = regressors(s)
            if not np.all(np.isfinite(xs)):
                continue                                     # base day (or all its lags) masked -> skip pair
            X.append(xs)
            # target = log of the ARITHMETIC MEAN of OBSERVED RV LEVELS over the window (what the VRP nets
            # against), NOT the mean of log-RV -- the latter is a geometric mean, ~13% low, spuriously inflating
            # the VRP. nanmean averages only the trustworthy days in the window.
            y.append(float(np.log(np.nanmean(tw))))
        if len(y) < min_train:
            continue
        xt = regressors(t)
        if not np.all(np.isfinite(xt)):
            continue                                         # day t itself untrusted/missing -> no forecast (NaN)
        A = np.asarray(X); b = np.asarray(y)
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        # log-spec -> LEVEL forecast needs the Jensen / log-normal correction: E[RV]=exp(E[logRV]+sigma^2/2),
        # else the level forecast is biased low by ~20-25% and would systematically INFLATE the VRP. sigma^2
        # = the in-sample log-residual variance (estimated only on the trailing window, so still strictly PIT).
        resid = b - A @ beta
        resid_var = float(resid @ resid / max(len(b) - A.shape[1], 1))
        out[t] = float(np.exp(min(xt @ beta + 0.5 * resid_var, cap)))
    return out


# --- VRP netting + cross-sectional standardization (the cross-date orchestrator's core) --------------

def vrp_series(rv_daily, iv_var_annualized, *, horizon: int = 21, rq_daily=None, ann: float = 252.0) -> np.ndarray:
    """Point-in-time variance-risk-premium series: implied variance minus the HARQ forecast of realized
    variance, in IDENTICAL annualized units (the convention-match invariant -- GATE 5). `rv_daily` = per-day
    realized-variance estimates; harq_forecast_series gives E_t[mean daily RV over the next `horizon` days],
    annualized by `ann` (=252 trading days) and subtracted from the annualized implied variance. vrp>0 is the
    usual positive variance risk premium (implied richer than expected realized). Strictly causal."""
    iv = np.asarray(iv_var_annualized, dtype=float)
    fc = harq_forecast_series(rv_daily, rq_daily, horizon=horizon)    # E_t[mean daily RV over next horizon]
    return iv - ann * fc


def cross_sectional_z(values, precisions=None) -> np.ndarray:
    """Cross-sectional z-score of one date's panel, NaN-safe. With `precisions` (inverse-variance weights) the
    center and scale are precision-weighted so noisily-estimated members pull the moments less. NaN members
    return NaN (excluded from the moments)."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    w = (np.where(ok, 1.0, 0.0) if precisions is None
         else np.where(ok & np.isfinite(np.asarray(precisions, dtype=float)), np.asarray(precisions, dtype=float), 0.0))
    sw = float(w.sum())
    if sw <= 0:
        return np.full(v.shape, np.nan)
    mu = float(np.sum(w * np.where(ok, v, 0.0)) / sw)
    sd = float(np.sqrt(np.sum(w * np.where(ok, (v - mu) ** 2, 0.0)) / sw))
    z = np.full(v.shape, np.nan)
    scored = ok & (w > 0)            # only members with a positive (trusted) weight get a z; NaN/zero-precision
    if sd > 0:                       # members are NOT scored (audit: they got spurious off-scale z-scores)
        z[scored] = (v[scored] - mu) / sd
    return z
