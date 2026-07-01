"""Pure-function helpers + builders for the derived-analytics layer (forward-vol, event/earnings-vol,
VRP). No I/O; builders return list[dict] of bare metric rows (mirrors diagnostics/iv_history.py).

Geometry helpers, factored once and reused by every derived schema:
  - cm_node:        constant-maturity (theta, rho, psi) by the iv_history interpolation rule
  - otm_black:      undiscounted OTM Black price per unit forward (engine d1=(-k+w/2)/sqrt(w) convention)
  - recertify_pair: calendar no-arb re-certification of a CM-interpolated tenor pair

Reuses the FIXED ssvi.model.implied_density (canonical Gatheral-Jacquier w'^2/4); never re-derives g(k).
The shared bkm_cumulants helper uses the FORWARD-centered (rate-invariant) mu = -V/2-W/6-X/24; it is
reused by earnings_vol and vrp.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import ndtr

from ..constants import CM_TENORS, DAYCOUNT, EPS_ARB, GRID_K_MAX, KFIT_MAX
from ..diagnostics.verify import EPS_BFLY_FD, calendar_ok_pair_rhos, dense_k
from ..ssvi.model import phi, theta_phi, w_of_k

_trap = getattr(np, "trapezoid", None) or np.trapz


def cm_node(slice_ts: np.ndarray, thetas: np.ndarray, rho_arr: np.ndarray,
            eta: float, gamma: float, t_cm: float) -> tuple[float, float, float]:
    """Constant-maturity (theta_cm, rho_cm, psi_cm) at t_cm by the iv_history rule: theta linear in t,
    rho interpolated in (rho*psi)/psi space (preserves the Hendriks-Martini coupling), clipped to
    +-0.999. slice_ts/thetas/rho_arr must be ordered by increasing t. Mirrors iv_history:60-63."""
    psi = theta_phi(thetas, eta, gamma)
    rho_psi = rho_arr * psi
    theta_cm = float(np.interp(t_cm, slice_ts, thetas))
    psi_cm = float(theta_phi(np.array([theta_cm]), eta, gamma)[0])
    rho_cm = float(np.clip(np.interp(t_cm, slice_ts, rho_psi) / max(psi_cm, 1e-12), -0.999, 0.999))
    return theta_cm, rho_cm, psi_cm


def otm_black(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Undiscounted OTM Black price per unit forward at log-moneyness k, total variance w, using the
    engine convention d1=(-k+w/2)/sqrt(w), d2=d1-sqrt(w). OTM = call for k>=0, put for k<0. Self-
    contained (matches verify.butterfly_ok_fd's call price); tests assert it agrees with iv.black."""
    k = np.asarray(k, float)
    w = np.asarray(w, float)
    sw = np.sqrt(np.maximum(w, 1e-300))
    d1 = (-k + 0.5 * w) / sw
    d2 = d1 - sw
    call = ndtr(d1) - np.exp(k) * ndtr(d2)            # OTM call,  k >= 0
    put = np.exp(k) * ndtr(-d2) - ndtr(-d1)           # OTM put,   k < 0
    return np.where(k >= 0.0, call, put)


def recertify_pair(theta1: float, rho1: float, theta2: float, rho2: float,
                   eta: float, gamma: float) -> tuple[bool, float]:
    """Calendar no-arb re-certification of a CM-interpolated tenor pair (t1<t2): w(k,t2)-w(k,t1) >= -eps
    on the dense grid. CM-interpolated slices are NOT the fitted slices verify_surface ran on, so any
    derived metric that differences/integrates across tenors must re-certify before trusting the pair."""
    return calendar_ok_pair_rhos(theta1, rho1, theta2, rho2, eta, gamma)


def fwd_rnd_check(k: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Risk-neutral density check on an ARBITRARY total-variance curve w(k) (e.g. the forward smile
    w_fwd, which is NOT a single eSSVI slice). Builds the undiscounted Black call price per unit forward
    and takes the numerical Breeden-Litzenberger 2nd strike-derivative -> the RND. Returns
    (density_min, integral): butterfly-arb-free <=> density_min >= -eps; integral -> 1 on full support
    (shortfall = tail mass beyond the grid). Independent of the analytic g(k). Mirrors verify.butterfly_ok_fd."""
    k = np.asarray(k, float)
    w = np.asarray(w, float)
    sw = np.sqrt(np.maximum(w, 1e-300))
    d1 = (-k + 0.5 * w) / sw
    C = ndtr(d1) - np.exp(k) * ndtr(d1 - sw)             # undiscounted call / forward
    kf = np.exp(k)                                       # K / F
    slope = np.diff(C) / np.diff(kf)
    curv = np.diff(slope) / (0.5 * (kf[2:] - kf[:-2]))   # d2C/dK2 ~ RND, on interior k
    return float(np.nanmin(curv)), float(_trap(curv, kf[1:-1]))


def _solve_k_fwd_delta(k: np.ndarray, w_fwd: np.ndarray, target: float):
    """k where the forward call delta N(d1)=target on the forward smile w_fwd(k), d1=(-k+w/2)/sqrt(w).
    Returns None if N(d1) is non-monotone over the bracket (the forward-smile delta guard)."""
    sw = np.sqrt(np.maximum(w_fwd, 1e-300))
    dlt = ndtr((-k + 0.5 * w_fwd) / sw)                  # forward call delta, should decrease in k
    if np.any(np.diff(dlt) > 1e-12):                     # non-monotone -> 25-delta strike ill-defined
        return None
    f = dlt - target
    idx = np.where(np.diff(np.sign(f)) != 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x0, x1, y0, y1 = k[i], k[i + 1], f[i], f[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def log_contract_var(k: np.ndarray, w: np.ndarray, t: float) -> tuple[float, float]:
    """Model-free log-contract implied variance K_log^2(t) = (2/t) integral e^{-k} OTM_Black(k,w) dk over
    the dense grid; returns (K_log^2, tail_frac) where tail_frac is the |k|>KFIT_MAX share of the
    integral. Used by the forward var-swap anchor and the VRP implied leg (Lee-bounded closed-form wing
    extension is added with VRP)."""
    o = otm_black(k, w)
    integrand = np.exp(-k) * o
    val = (2.0 / t) * float(_trap(integrand, k))
    tail = (2.0 / t) * float(_trap(np.where(np.abs(k) <= KFIT_MAX, 0.0, integrand), k))
    return val, (abs(tail) / abs(val) if val != 0 else 0.0)


def bkm_cumulants(k: np.ndarray, w: np.ndarray) -> dict:
    """mu-centered Bakshi-Kapadia-Madan (2003) risk-neutral cumulants of the log-return ln(S_T/F) on a
    total-variance curve w(k), k=ln(K/F). Integrating the OTM Black strip otm_black(k,w) (per unit
    forward) with the e^{-k}=(dK/K^2)*F jacobian yields the V/W/X contracts directly. The drift
    mu=E^Q[ln(S_T/F)] is FORWARD-centered, hence RATE-INVARIANT: mu=-V/2-W/6-X/24 with NO (e^{rT}-1)
    term (that term belongs to a SPOT-centered strip k=ln(K/S)). So this is purely a function of (k,w)
    -- standardized rn_skew/rn_kurt are r-invariant by construction. Returns the contracts, mu, raw
    cumulants k2/k3/k4, standardized rn_skew/rn_kurt, tail_frac. Shared by earnings_vol + vrp."""
    o = otm_black(k, w)
    e_k = np.exp(-k)
    V = float(_trap(e_k * 2.0 * (1.0 - k) * o, k))                  # quadratic contract (per forward)
    W = float(_trap(e_k * (6.0 * k - 3.0 * k * k) * o, k))          # cubic   (put leg's sign flips via k<0)
    X = float(_trap(e_k * (12.0 * k * k - 4.0 * k ** 3) * o, k))    # quartic
    mu = -0.5 * V - W / 6.0 - X / 24.0                             # E^Q[ln(S_T/F)], forward-centered, r-free
    k2 = V - mu * mu                                               # variance (2nd cumulant)
    k3 = W - 3.0 * mu * V + 2.0 * mu ** 3                          # 3rd central moment = 3rd cumulant
    m4 = X - 4.0 * mu * W + 6.0 * mu * mu * V - 3.0 * mu ** 4      # 4th central moment
    k4 = m4 - 3.0 * k2 * k2                                        # 4th cumulant
    ok = k2 > 1e-12
    o_in = e_k * o
    denom = max(abs(float(_trap(o_in, k))), 1e-300)
    tail = abs(float(_trap(np.where(np.abs(k) <= KFIT_MAX, 0.0, o_in), k))) / denom
    return {"V": V, "W": W, "X": X, "mu": mu, "k2": k2, "k3": k3, "k4": k4,
            "rn_skew": (k3 / k2 ** 1.5 if ok else None),
            "rn_kurt": (m4 / (k2 * k2) if ok else None), "tail_frac": tail}


_MACRO_VAR = {"FOMC", "CPI", "NFP", "PPI"}     # non-earnings variance-driving macro events (contaminate DJ)


def build_earnings_vol(ts, underlying, slice_ts, thetas, rho_arr, eta, gamma,
                       expiry_dates, reprice, underlier_state, event, macro_cal=None):
    """Earnings-vol layer (single-name Dubinsky-Johannes). Ex-ante TERM-STRUCTURE strip of the one
    deterministic earnings jump off the arb-free eSSVI w-curve. PRIMARY estimator (form A): with a
    NON-spanning near expiry (settles before tau, ex-event) and a spanning far expiry (settles after),
      event_var = J^2 = (theta_span - theta_near) - vbar_diff*(t_span - t_near)
    vbar_diff = the diffusive forward-variance rate from event-free brackets (calendar-certified, floored
    at 0), no-arb-CAPPED at (theta_span-theta_near)/dt so the shipped `event_var` is >=0. The signed,
    pre-cap DJ estimate is shipped separately as `event_var_raw`. Two CO-SPANNING expiries identify only
    the diffusion (the lump cancels in theta2-theta1=nu*dt) -> both_span=True, unidentified. Ships
    `event_var` (>=0) + `event_var_raw` (signed) + SE/band, two move numbers (RMS sigma_J, MAD E|J| under
    an EXPLICIT Gaussian-jump assumption), spanning-slice raw BKM rn_skew/rn_kurt + RND integrity stamp,
    and honesty/leakage flags. SINGLE-EVENT ASSUMPTION: all spanning excess variance is attributed to the
    one earnings jump; a co-located MACRO variance event (FOMC/CPI/NFP/PPI) inside (T_near,T_span] folds its
    own variance in, so the single jump is NOT identifiable -> when `macro_cal` is supplied such a row is
    flagged `macro_in_window=True` / `derive_status='macro_contaminated'` with `event_var` nulled (the sibling
    event_vol layer deconvolves the multi-event case). `macro_cal=None` -> flag undeterminable (stays null).
    DEFERRED (need the event-vol NNLS deconvolution): drift, kappa, event skew/kurt, RND quantiles, the
    non-Gaussian MAD. `event` = ONE leakage-gated bracketing dict or None. Returns [row] | []."""
    if event is None:
        return []
    slice_ts = np.asarray(slice_ts, float)
    thetas = np.asarray(thetas, float)
    rho_arr = np.asarray(rho_arr, float)
    edate = event["earnings_date"]
    from_8k = bool(event.get("from_8k", False))
    tau = (edate - ts.date()).days / DAYCOUNT
    base = {
        "ts": ts, "underlying": underlying, "earnings_date": edate, "from_8k": from_8k,
        "session": "unknown", "event_tau": tau, "date_conf_weight": (1.0 if from_8k else 0.5),
        "resid_known": (None if underlier_state is None else underlier_state.get("resid_known")),
        "macro_in_window": None,                            # reserved: needs the P2.5 macro calendar
    }

    def _row(status, **kw):
        r = dict(base); r["derive_status"] = status; r.update(kw); return r

    if edate < ts.date():
        return [_row("event_in_past")]
    if slice_ts.size < 2:
        return [_row("thin", n_brackets=int(slice_ts.size))]

    # span classification: the calendar carries NO intraday 8-K timestamp or session, so BMO/AMC is
    # genuinely unknown. An expiry SPANS only if it settles STRICTLY AFTER the earnings date; an expiry
    # settling ON the date is session-ambiguous (AMC -> ex-event, BMO -> holds the jump) and is excluded
    # from BOTH the near and span pools rather than mis-assigned.
    ev_days = (edate - ts.date()).days
    exp_days = np.array([(d - ts.date()).days for d in expiry_dates])
    spans = exp_days > ev_days
    sameday = exp_days == ev_days
    nonspan = np.where(~spans & ~sameday)[0]
    spanidx = np.where(spans)[0]
    n_br = int(slice_ts.size)
    if spanidx.size == 0:
        return [_row("no_bracket", n_brackets=n_br)]
    if nonspan.size == 0:
        # every (non-ambiguous) fitted expiry spans -> the jump cancels in theta differences
        return [_row("both_span_unidentified", both_span=True, n_brackets=n_br)]

    near = int(nonspan[np.argmax(slice_ts[nonspan])])      # latest ex-event expiry (closest below tau)
    span = int(spanidx[np.argmin(slice_ts[spanidx])])      # earliest spanning expiry (closest above tau)
    t_near, t_span = float(slice_ts[near]), float(slice_ts[span])
    th_near, th_span = float(thetas[near]), float(thetas[span])
    rh_span = float(rho_arr[span])
    dts = t_span - t_near
    arb_raw_ok = bool(th_span - th_near >= -EPS_ARB)
    # macro co-location (A2): once macro_cal is supplied, populate the previously-reserved flag. A non-earnings
    # VARIANCE macro event whose date lands strictly inside the (near_expiry, span_expiry] jump bracket folds
    # its own variance into theta_span-theta_near, so the single earnings jump is unidentifiable.
    near_date, span_date = expiry_dates[near], expiry_dates[span]
    if macro_cal is None:
        n_macro = 0                                          # flag undeterminable -> base stays None
    else:
        n_macro = sum(1 for m in macro_cal
                      if str(m.get("event_type", "")).upper() in _MACRO_VAR and near_date < m["date"] <= span_date)
        base["macro_in_window"] = bool(n_macro)
    if t_span < 3.0 / DAYCOUNT:                              # 1/(T-t) blow-up guard
        return [_row("near_expiry_guard", expiry_near=expiry_dates[near], expiry_span=expiry_dates[span],
                     t_near=t_near, t_span=t_span, both_span=False, n_brackets=n_br)]
    if dts <= 1e-9:                                          # duplicate / non-monotone bracket geometry
        return [_row("degenerate_brackets", expiry_near=expiry_dates[near], expiry_span=expiry_dates[span],
                     t_near=t_near, t_span=t_span, both_span=False, n_brackets=n_br)]
    if n_macro:                                             # co-located macro variance -> jump not pure-earnings
        return [_row("macro_contaminated", expiry_near=near_date, expiry_span=span_date,
                     t_near=t_near, t_span=t_span, both_span=False, n_brackets=n_br)]

    # diffusive baseline rate vbar_diff from event-free (non-spanning) brackets, calendar-certified
    a = None
    if nonspan.size >= 2:
        a = int(nonspan[np.argsort(slice_ts[nonspan])[-2]])   # 2nd-latest ex-event expiry
        da = t_near - float(slice_ts[a])
    if a is not None and da > 1e-9:
        vbar = (th_near - float(thetas[a])) / da
        baseline_arb_ok = recertify_pair(float(thetas[a]), float(rho_arr[a]), th_near,
                                         float(rho_arr[near]), eta, gamma)[0]
        if not baseline_arb_ok:
            vbar = max(vbar, 0.0)                              # a violated baseline cannot inflate J^2
        n_diff, tier = 2, "pre_only"
    else:
        a = None
        vbar = th_near / t_near                                # form (C): flat-diffusion from the near slice
        baseline_arb_ok, n_diff, tier = True, 1, "near_level"
    cap = (th_span - th_near) / dts
    capped = vbar > cap
    vbar_base = vbar
    vbar = min(vbar, cap)                                    # no-arb cap: keep shipped event_var >= 0
    event_var = (th_span - th_near) - vbar * dts            # capped, >= 0
    event_var_raw = (th_span - th_near) - vbar_base * dts   # signed DJ estimate (can be < 0)

    # SE: reprice-RMSE-propagated theta-noise proxy (no posterior theta covariance is persisted); zeroed
    # when the cap binds (event_var is then the no-arb bound, not the noisy free estimate)
    def _stheta(i):
        rm = reprice[i] if (reprice is not None and i < len(reprice) and reprice[i] is not None) else None
        return None if rm is None else 2.0 * np.sqrt(max(thetas[i], 0.0) * slice_ts[i]) * (rm / 100.0)
    s_sp, s_nr = _stheta(span), _stheta(near)
    s_vb = (None if (s_nr is None) else (s_nr / t_near if a is None
            else np.hypot(s_nr, _stheta(a) or 0.0) / (t_near - float(slice_ts[a]))))
    ev_se = (0.0 if capped else
             (None if (s_sp is None or s_nr is None) else
              float(np.sqrt(s_sp ** 2 + s_nr ** 2 + (dts * (s_vb or 0.0)) ** 2))))

    # spanning-slice risk-neutral moments (raw BKM, well-conditioned) + de-eventized RND integrity stamp.
    # NOTE: the jump-DISTRIBUTION moments (drift, kappa, event skew/kurt, RND quantiles) need to isolate
    # the jump from the diffusion via the event-vol NNLS deconvolution and are intentionally DEFERRED to
    # P2.5 -- naive de-eventized cumulant differencing conflates the smile-shape change with the jump and
    # is ill-conditioned for small jumps. v1 ships only the well-conditioned legs.
    k = dense_k()
    cu = bkm_cumulants(k, w_of_k(k, th_span, rh_span, eta, gamma))
    tf = cu["tail_frac"]
    tail = tf if np.isfinite(tf) else None
    clean = tail is not None and tail <= 0.15
    th_dev = max(th_span - event_var, 1e-9)                  # diffusive-only total variance to T_span
    _dmin, integ = fwd_rnd_check(k, w_of_k(k, th_dev, rh_span, eta, gamma))
    # two move numbers: RMS from the strip (headline); MAD under an EXPLICIT Gaussian-jump assumption
    # (straddle-recoverable, = sqrt(2/pi)*RMS). The non-Gaussian MAD/kappa land with the P2.5 deconvolution.
    sig_rms = float(np.sqrt(max(event_var, 0.0)))
    mad = float(np.sqrt(2.0 / np.pi) * sig_rms)

    status = "ok"
    if not baseline_arb_ok:
        status = "baseline_arb_fail"
    elif base["resid_known"] is not None and not bool(base["resid_known"]):
        status = "div_contam_unknown"
    elif event_var_raw < -1e-9 and arb_raw_ok:
        status = "neg_var_clipped"                           # the signed estimate went negative; capped to 0

    # +-1 session date-confidence band for report-date-fallback (from_8k=False) events
    ev_lo = ev_hi = None
    if not from_8k:
        lo = (th_span - th_near) - vbar * (t_span - (t_near + 1.0 / DAYCOUNT))
        hi = (th_span - th_near) - vbar * (t_span - (t_near - 1.0 / DAYCOUNT))
        ev_lo, ev_hi = min(lo, hi), max(lo, hi)

    return [_row(status, expiry_near=expiry_dates[near], expiry_span=expiry_dates[span],
                 t_near=t_near, t_span=t_span, theta_near=th_near, theta_span=th_span,
                 vbar_diff=vbar, vbar_diff_capped=bool(capped), baseline_arb_ok=bool(baseline_arb_ok),
                 baseline_tier=tier, n_diff_brackets=n_diff,
                 event_var=event_var, event_var_raw=event_var_raw, event_var_se=ev_se,
                 event_var_lo=ev_lo, event_var_hi=ev_hi, both_span=False, arb_raw_ok=arb_raw_ok,
                 sigma_J_rms=sig_rms, e_abs_j_mad=mad,
                 rn_skew=(cu["rn_skew"] if clean else None), rn_kurt=(cu["rn_kurt"] if clean else None),
                 tail_frac=tail, rnd_integral=(integ if np.isfinite(integ) else None),
                 event_share=(event_var / th_span if th_span > 1e-12 else None),
                 iv_bump_spanning=float(np.sqrt(th_span / t_span)
                                        - np.sqrt(max(th_span - event_var, 0.0) / t_span)),
                 contam_ok=None, n_brackets=n_br)]


def build_vrp_implied(ts, underlying, slice_ts, thetas, rho_arr, eta, gamma):
    """VRP implied-leg layer (per snapshot). THREE labeled model-free variance strikes per CM tenor off
    the arb-free eSSVI smile (integrate the MODEL, not the quotes), all per-unit-forward and rate-
    invariant in k=ln(K/F):
      iv_var   = K_log^2 = (2/t) integral e^{-k} OTM dk   (Carr-Madan log-contract / VIX-equivalent QV)
      svix2    = SVIX^2  = (2/t) integral e^{+k} OTM dk   (Martin jump-robust simple-return variance)
      k_var    = Var(ln S_T/F)/t = bkm k2 / t             (squared-log-return variance)
    skew_corr = k_var - iv_var (Du-Kapadia jump/skew surplus; -> 0 for a lognormal, signed under jumps).
    Re-certifies the CM smile's FD-BL density before integrating (CM nodes are NOT fitted slices) -> null
    + vrp_status='arb_violation' on failure. Ships rn_skew/rn_kurt (forward-centered BKM), iv_var_core
    (|k|<=KFIT_MAX quoted span) vs iv_var (full grid) so the wing tail leverage is auditable via tail_frac.
    Reconciles to published VIX on SPX. Lee-bounded closed-form extension beyond k=GRID_K_MAX is a later
    upgrade; tail_frac is the v1 quality flag. Returns list[dict] (one row per CM tenor in the span)."""
    slice_ts = np.asarray(slice_ts, float)
    thetas = np.asarray(thetas, float)
    rho_arr = np.asarray(rho_arr, float)
    order = np.argsort(slice_ts)                          # self-enforce the increasing-t precondition
    slice_ts, thetas, rho_arr = slice_ts[order], thetas[order], rho_arr[order]
    if slice_ts.size < 2:
        return []
    tmin, tmax = float(slice_ts.min()), float(slice_ts.max())
    k = dense_k()
    core = np.abs(k) <= KFIT_MAX
    ek_m, ek_p = np.exp(-k), np.exp(k)
    rows = []
    for d in CM_TENORS:
        t = d / DAYCOUNT
        if not (tmin <= t <= tmax):
            continue
        th, rh, _ = cm_node(slice_ts, thetas, rho_arr, eta, gamma, t)
        w = w_of_k(k, th, rh, eta, gamma)
        dmin, integ = fwd_rnd_check(k, w)                    # re-certify the CM-interpolated smile
        ok = dmin >= -EPS_BFLY_FD
        o = otm_black(k, w)
        klog2 = (2.0 / t) * float(_trap(ek_m * o, k))        # exact log-contract var swap = VIX^2-equiv
        klog2_core = (2.0 / t) * float(_trap(np.where(core, ek_m * o, 0.0), k))
        svix2 = (2.0 / t) * float(_trap(ek_p * o, k))        # Martin simple-return variance
        cu = bkm_cumulants(k, w)
        kvar = cu["k2"] / t                                  # variance of the log return, annualized
        tf = ((1.0 - klog2_core / klog2) if (ok and klog2 > 1e-12) else None)
        # flag (don't null) excessive wing-tail leverage, matching the 0.15 bar the sibling layers use
        status = ("arb_violation" if not ok
                  else ("tail_unreliable" if (tf is not None and tf >= 0.15) else "ok"))
        rows.append({
            "ts": ts, "underlying": underlying, "tenor": int(d),
            "theta_cm": th, "rho_cm": rh, "atm_var": th / t,
            "iv_var": (klog2 if ok else None), "iv_var_core": (klog2_core if ok else None),
            "svix2": (svix2 if ok else None), "k_var": (kvar if ok else None),
            "skew_corr": ((kvar - klog2) if ok else None),
            "rn_skew": (cu["rn_skew"] if ok else None), "rn_kurt": (cu["rn_kurt"] if ok else None),
            "tail_frac": tf,
            "density_min": (dmin if np.isfinite(dmin) else None),
            "rnd_integral": (integ if np.isfinite(integ) else None),
            "vrp_status": status,
        })
    return rows


def _folded_normal_mean(mu: float, sigma: float) -> float:
    """E|X| for X~N(mu,sigma^2): the straddle-recoverable MAD move; = sigma*sqrt(2/pi) at mu=0."""
    if sigma <= 1e-12:
        return abs(float(mu))
    a = mu / sigma
    return float(sigma * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * a * a) + mu * (2.0 * ndtr(a) - 1.0))


def _event_moves(x: float):
    """Gaussian martingale jump moves from x = E^Q[J^2] (>=0): solve the Gaussian variance s^2 from
    E[J^2]=Var+mu^2=s^2+s^4/4=x with the martingale drift mu=-s^2/2; return (drift, RMS, MAD, kappa_kurt).
    kappa_kurt = E|J|/sqrt(E[J^2]) -> sqrt(2/pi)=0.7979 as x->0 (zero-drift limit); it drifts strictly above
    that for finite jumps purely because the martingale drift mu=-s^2/2 makes the non-centered |J| asymmetric
    (e.g. ~0.79789 at x=0.04, ~0.79810 at x=0.25). Under the forced-Gaussian assumption it is a deterministic
    function of x (=mad/rms), so it is NOT an independent Gaussianity test."""
    x = max(float(x), 0.0)
    s2 = -2.0 + 2.0 * np.sqrt(1.0 + x)
    s = float(np.sqrt(max(s2, 0.0)))
    mu = -0.5 * s2
    rms = float(np.sqrt(x))
    mad = _folded_normal_mean(mu, s)
    return mu, rms, mad, (mad / rms if rms > 1e-12 else None)


_VAR_EVENTS = {"EARNINGS", "FOMC", "CPI", "NFP", "PPI"}
_FLOW_EVENTS = {"OPEX", "QUAD_WITCH"}


def build_event_vol(ts, underlying, slice_ts, thetas, rho_arr, eta, gamma, expiry_dates, reprice,
                    underlier_state, macro_cal=None, earnings_event=None, scope="index"):
    """Event-vol layer: the GENERAL multi-event Dubinsky-Johannes variance strip on the arb-free eSSVI
    ATM total-variance ladder (engine identity w(0;T)=theta). theta_i = sum_m G[i,m] nu_m + sum_k A[i,k]
    x_k with x_k=(sigma_J,k)^2>=0, A[i,k]=1{tau_k<=t_i} (staircase incidence), G[i,m]=event-free segment
    length below t_i. ONE bracketed event -> the exact earnings form-A closed form; >=2 co-located events
    -> regularized non-negative least squares (Tikhonov-smoothed baseline, deterministic Lawson-Hanson/
    BVLS). OPEX/quad-witch are SIGNED dealer-gamma FLOW (never coerced >=0, never an implied move). Ships
    signed event_var_raw + no-arb-capped event_var, Gaussian-jump moves (RMS/MAD/drift/kappa_kurt), and
    identifiability flags (co-located events with no expiry between them -> bracket_sum_only). DEFERRED to
    P3: the non-Gaussian jump distribution (RND quantiles, event skew/kurt), the cross-asset macro PRIOR,
    and the contamination certificate. `macro_cal`=list of {date,event_type,session,source}|None;
    `earnings_event`=single-name bracket dict|None. Returns list[dict] | []."""
    if macro_cal is None and earnings_event is None:
        return []
    slice_ts = np.asarray(slice_ts, float)
    thetas = np.asarray(thetas, float)
    order = np.argsort(slice_ts)
    slice_ts, thetas = slice_ts[order], thetas[order]
    if slice_ts.size < 2:
        return []
    tmax = float(slice_ts.max())
    base = {"ts": ts, "underlying": underlying}

    def _row(status, **kw):
        r = dict(base); r["scope"] = kw.pop("scope", scope); r["derive_status"] = status; r.update(kw); return r

    raw = []
    if earnings_event is not None:
        raw.append((earnings_event["earnings_date"], "EARNINGS", "unknown",
                    "earnings:from_8k" if earnings_event.get("from_8k") else "earnings:report_date"))
    for m in (macro_cal or []):
        raw.append((m["date"], str(m["event_type"]).upper(), str(m.get("session", "unknown")),
                    str(m.get("source", "macro"))))

    rows, parsed = [], []
    for ed, typ, sess, src in raw:
        tau = (ed - ts.date()).days / DAYCOUNT
        if tau <= 0.0:                                       # past OR same-day -> no forward bracket
            if typ == "EARNINGS":
                rows.append(_row("event_in_past", event_date=ed, event_type=typ))
            continue
        if tau <= tmax:                                      # only events bracketed by the fitted ladder
            parsed.append({"date": ed, "tau": tau, "type": typ, "session": sess, "source": src})
    if not parsed:
        return rows
    arb_ok = bool(np.all(np.diff(thetas) >= -EPS_ARB))       # calendar pre-condition (fwd var >= 0)
    var_ev = [e for e in parsed if e["type"] in _VAR_EVENTS]
    flow_ev = [e for e in parsed if e["type"] in _FLOW_EVENTS]
    K = len(var_ev)
    t = slice_ts

    def _nu_eventfree():
        """piecewise diffusive rate per inter-expiry segment (held forward); NaN-safe on 0-length segs."""
        seg_lo = np.concatenate([[0.0], t[:-1]])
        seg_len = t - seg_lo
        safe = seg_len > 1e-12
        return seg_lo, np.where(safe, (thetas - np.concatenate([[0.0], thetas[:-1]])) / np.where(safe, seg_len, 1.0), 0.0)

    def _emit_var(e, ev, ev_raw, alloc, nu_at, ident_rel):
        mu, rms, mad, kk = _event_moves(ev)
        rows.append(_row("ok", event_date=e["date"], event_type=e["type"], event_session=e["session"],
                         event_source=e["source"], tau=e["tau"], nu_c=nu_at, n_eventfree_segments=int(t.size),
                         event_var=ev, event_var_raw=ev_raw, ident_rel=ident_rel, sigma_J_rms=rms,
                         e_abs_j_mad=mad, drift=mu, kappa_kurt=kk, kappa_alloc_method=alloc,
                         n_events_in_window=K, arb_raw_ok=True, contam_ok=None))

    if K and not arb_ok:
        for e in var_ev:
            rows.append(_row("baseline_arb_fail", event_date=e["date"], event_type=e["type"],
                             event_session=e["session"], event_source=e["source"], tau=e["tau"],
                             n_events_in_window=K, arb_raw_ok=False))
    elif K == 1:                                             # exact earnings form-A bracket (exactly identified)
        e = var_ev[0]
        ns = np.where(t < e["tau"])[0]
        if ns.size == 0:                                     # no ex-event anchor -> the jump cancels
            rows.append(_row("both_span_unidentified", event_date=e["date"], event_type=e["type"],
                             tau=e["tau"], both_span=True, n_events_in_window=1))
        else:
            near, span = int(ns[-1]), int(np.where(t >= e["tau"])[0][0])
            dt = t[span] - t[near]
            if dt <= 1e-9:                                   # duplicate / non-monotone bracket -> no NaN
                rows.append(_row("degenerate_brackets", event_date=e["date"], event_type=e["type"],
                                 tau=e["tau"], n_events_in_window=1))
            else:
                _seg, rate = _nu_eventfree()
                vbar = max(float(rate[near]), 0.0)           # diffusive rate from the ex-event near segment
                ev_raw = (thetas[span] - thetas[near]) - vbar * dt
                _emit_var(e, max(ev_raw, 0.0), ev_raw, "bracket", vbar, 0.0)   # ident_rel=0 (no regularization)
    elif K >= 2:                                             # regularized NNLS staircase
        N = t.size
        seg_lo = np.concatenate([[0.0], t[:-1]])
        seg_len = t - seg_lo
        G = np.tril(np.tile(seg_len, (N, 1)))                # G[i,m]=seg_len[m] for m<=i (cumulative diff)
        Aev = np.array([[1.0 if var_ev[kk]["tau"] <= t[i] + 1e-12 else 0.0 for kk in range(K)]
                        for i in range(N)])
        alias = [any(j != kk and np.array_equal(Aev[:, kk], Aev[:, j]) for j in range(K)) for kk in range(K)]
        lam = 1e-6
        D = np.zeros((max(N - 2, 0), N + K))
        for i in range(N - 2):
            D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
        Aaug = np.vstack([np.hstack([G, Aev]), np.sqrt(lam) * D])
        baug = np.concatenate([thetas, np.zeros(max(N - 2, 0))])
        lb = np.zeros(N + K)
        try:
            u = lsq_linear(Aaug, baug, bounds=(lb, np.inf), method="bvls", max_iter=1000).x
            x_raw = np.linalg.lstsq(Aaug, baug, rcond=None)[0][N:]
        except (np.linalg.LinAlgError, ValueError):
            # a degenerate/ill-conditioned staircase can make the SVD inside bvls/lstsq fail to converge.
            # Degrade gracefully so the whole name-day SURVIVES (regression: AMD 2024-12-17 silent drop, a
            # 679k-quote day lost to an uncaught LinAlgError): emit each var event unsolved + flagged. The
            # flow-event loop below still runs.
            for e in var_ev:
                rows.append(_row("nnls_unconverged", event_date=e["date"], event_type=e["type"],
                                 event_session=e["session"], event_source=e["source"], tau=e["tau"],
                                 n_events_in_window=K, arb_raw_ok=arb_ok))
            u = None
        if u is not None:
            nu_seg, x_hat = u[:N], u[N:]
            resid = float(np.linalg.norm((np.hstack([G, Aev])) @ u - thetas))
            for kk, e in enumerate(var_ev):
                if alias[kk]:
                    rows.append(_row("bracket_sum_only", event_date=e["date"], event_type=e["type"],
                                     event_session=e["session"], event_source=e["source"], tau=e["tau"],
                                     kappa_alloc_method="bracket_sum_only", n_events_in_window=K, arb_raw_ok=True))
                    continue
                seg = int(np.searchsorted(t, e["tau"], side="left"))
                xv, xr = float(x_hat[kk]), float(x_raw[kk])
                # the staircase is structurally rank-deficient by K (a jump is colinear with a 1-segment nu
                # spike), so x_k is identified only by the Tikhonov prior. ident_rel = |capped - signed| / x
                # surfaces how much the regularization (not theta) drove this value -> flag the un-identified.
                ident = abs(xv - xr) / max(xv, 1e-9)
                mu, rms, mad, kkk = _event_moves(xv)
                rows.append(_row("weakly_identified" if ident > 0.5 else "ok",
                                 event_date=e["date"], event_type=e["type"], event_session=e["session"],
                                 event_source=e["source"], tau=e["tau"], nu_c=float(nu_seg[min(seg, N - 1)]),
                                 n_eventfree_segments=N, event_var=xv, event_var_raw=xr, ident_rel=ident,
                                 event_var_se=resid, sigma_J_rms=rms, e_abs_j_mad=mad, drift=mu, kappa_kurt=kkk,
                                 kappa_alloc_method="nnls", n_events_in_window=K, arb_raw_ok=True, contam_ok=None))

    for e in flow_ev:                                        # OPEX/quad-witch: signed forward-var anomaly
        lo, hi = np.where(t <= e["tau"])[0], np.where(t > e["tau"])[0]
        flow = None
        if lo.size and hi.size:
            i0, i1 = int(lo[-1]), int(hi[0])
            dt = t[i1] - t[i0]
            flow = float((thetas[i1] - thetas[i0]) / dt - thetas[i0] / t[i0]) if dt > 1e-9 else None
        rows.append(_row("ok", scope="flow", event_date=e["date"], event_type=e["type"],
                         event_session=e["session"], event_source=e["source"], tau=e["tau"],
                         opex_signed_flow=flow, n_events_in_window=K, arb_raw_ok=arb_ok))
    return rows


def build_forward_vol(ts, underlying, slice_ts, thetas, rho_arr, eta, gamma, is_eod):
    """Forward-vol layer (research-upgraded). Per adjacent CM-tenor pair T1<T2 in the fitted span: the
    deterministic price-additive forward smile w_fwd=w(.;T2)-w(.;T1), CLOSED-FORM forward skew/curvature
    from the eSSVI slice params, forward 25/10-delta RR/BF on the reconstructed forward smile, a
    model-free forward var-swap anchor, and a DOUBLE arbitrage gate (HM parent calendar on the CM pair
    AND FD-Breeden-Litzenberger butterfly on w_fwd). Returns (rows, grid_rows); grid only when is_eod."""
    slice_ts = np.asarray(slice_ts, float)
    thetas = np.asarray(thetas, float)
    rho_arr = np.asarray(rho_arr, float)
    tmin, tmax = float(slice_ts.min()), float(slice_ts.max())
    tenors = [d for d in CM_TENORS if tmin <= d / DAYCOUNT <= tmax]
    k = dense_k()
    rows, grid_rows = [], []
    for a in range(len(tenors) - 1):
        T1, T2 = tenors[a], tenors[a + 1]
        t1, t2 = T1 / DAYCOUNT, T2 / DAYCOUNT
        dt = t2 - t1
        if dt <= 0:
            continue
        th1, rh1, _ = cm_node(slice_ts, thetas, rho_arr, eta, gamma, t1)
        th2, rh2, _ = cm_node(slice_ts, thetas, rho_arr, eta, gamma, t2)
        ph1 = float(phi(np.array([th1]), eta, gamma)[0])
        ph2 = float(phi(np.array([th2]), eta, gamma)[0])
        w1 = w_of_k(k, th1, rh1, eta, gamma)
        w2 = w_of_k(k, th2, rh2, eta, gamma)
        w_fwd = w2 - w1
        var_min = float(np.min(w_fwd)) / dt
        sig_fwd = np.sqrt(np.maximum(w_fwd, 0.0) / dt)
        fwd_var = (th2 - th1) / dt
        fwd_atm = float(np.sqrt(max(fwd_var, 0.0)))
        # CLOSED-FORM forward skew/curvature from slice params (no finite differencing)
        wp = th2 * rh2 * ph2 - th1 * rh1 * ph1                                    # w_fwd'(0)
        wpp = 0.5 * (th2 * ph2 * ph2 * (1 - rh2 * rh2) - th1 * ph1 * ph1 * (1 - rh1 * rh1))  # w_fwd''(0)
        # skew/curv are meaningless (and overflow ~1e29) as fwd_atm->0; that regime is already
        # arb_ok=False (calendar-degenerate), so NULL them rather than ship a finite lie that can
        # overflow a downstream float32 L2/variance aggregation that forgot to filter on arb_ok.
        if fwd_atm < 1e-6:
            fwd_skew = fwd_curv = None
        else:
            s_atm = fwd_atm
            fwd_skew = wp / (2.0 * dt * s_atm)
            fwd_curv = wpp / (2.0 * dt * s_atm) - wp * wp / (4.0 * dt * dt * s_atm ** 3)
        # DOUBLE arb gate: HM parent calendar on the CM pair AND FD-BL butterfly on the forward smile
        cal_ok, _cal = recertify_pair(th1, rh1, th2, rh2, eta, gamma)
        dmin, integ = fwd_rnd_check(k, w_fwd)
        bfly_ok = dmin >= -EPS_BFLY_FD                  # same FD-BL tolerance as verify.butterfly_ok_fd
        arb_ok = bool(cal_ok and bfly_ok)

        def _rrbf(call_lo, call_hi):
            kP = _solve_k_fwd_delta(k, w_fwd, call_hi)       # put side  (high call delta, low k)
            kC = _solve_k_fwd_delta(k, w_fwd, call_lo)       # call side (low call delta, high k)
            if kP is None or kC is None:
                return None, None, False
            sP = float(np.sqrt(max(np.interp(kP, k, w_fwd), 0.0) / dt))
            sC = float(np.sqrt(max(np.interp(kC, k, w_fwd), 0.0) / dt))
            return sC - sP, 0.5 * (sC + sP) - fwd_atm, True
        rr25, bf25, ok25 = _rrbf(0.25, 0.75)
        rr10, bf10, ok10 = _rrbf(0.10, 0.90)
        # model-free forward var-swap anchor: K_fwd-var^2 = [t2 K_log^2(t2) - t1 K_log^2(t1)] / dt
        kv1, tf1 = log_contract_var(k, w1, t1)
        kv2, tf2 = log_contract_var(k, w2, t2)
        fwd_vswap = (t2 * kv2 - t1 * kv1) / dt
        tail_frac = max(tf1, tf2)
        clean_vs = abs(tail_frac) < 0.15
        rows.append({
            "ts": ts, "underlying": underlying, "t1": int(T1), "t2": int(T2), "dt": dt,
            "fwd_atm_vol": fwd_atm, "fwd_var": fwd_var, "fwd_skew": fwd_skew, "fwd_curv": fwd_curv,
            "fwd_rr_25d": rr25, "fwd_bf_25d": bf25, "fwd_rr_10d": rr10, "fwd_bf_10d": bf10,
            "fwd_var_swap": (fwd_vswap if clean_vs else None),
            "fwd_convexity_adj": ((fwd_vswap - fwd_var) if clean_vs else None),   # varswap - atm, >=0 eq
            "near_atm_vol": float(np.sqrt(max(th1, 0.0) / t1)),
            "far_atm_vol": float(np.sqrt(max(th2, 0.0) / t2)),
            "fwd_var_min": var_min, "fwd_density_min": dmin, "fwd_density_integral": integ,
            "tail_frac": tail_frac, "eta": float(eta), "gamma": float(gamma),
            "arb_ok": arb_ok, "delta_ok": bool(ok25 and ok10),
            "derive_status": ("ok" if arb_ok else ("cal_fail" if not cal_ok else "bfly_fail")),
        })
        if is_eod:
            for j in range(len(k)):
                grid_rows.append({
                    "ts": ts, "underlying": underlying, "t1": int(T1), "t2": int(T2),
                    "k": float(k[j]), "iv_fwd": float(sig_fwd[j]), "w_fwd": float(w_fwd[j]),
                    "flag": ("ok" if w_fwd[j] >= -EPS_ARB else "clipped_neg"),
                })
    return rows, grid_rows
