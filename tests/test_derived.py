"""Shared geometry helpers for the derived-analytics layer (diagnostics/derived.py)."""
import datetime as _dt

import numpy as np
from scipy.special import ndtr

from essvi.diagnostics import derived
from essvi.io.writer import build_table
from essvi.ssvi.model import theta_phi, w_of_k

_TS = _dt.datetime(2026, 1, 22, 20, 45, tzinfo=_dt.timezone.utc)
DAYCOUNT = 365.0


def _sigma_fwd_ref(slice_ts, thetas, rhos, eta, gamma, t1, t2):
    """Independent machine-exact (sig0, skew, curv) at k=0 of sigma_fwd(k)=sqrt((w(.;t2)-w(.;t1))/dt),
    reconstructed from the CM nodes: skew by complex step (no subtraction error), curv by repeated
    Richardson (Romberg) on that exact first derivative. Never uses the builder's closed form."""
    th1, rh1, _ = derived.cm_node(slice_ts, thetas, rhos, eta, gamma, t1)
    th2, rh2, _ = derived.cm_node(slice_ts, thetas, rhos, eta, gamma, t2)
    dt = t2 - t1

    def _w(kc, theta, rho):                               # complex-safe eSSVI w; explicit u*u, never **2
        ph = eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))
        u = ph * kc + rho
        return 0.5 * theta * (1.0 + rho * ph * kc + np.sqrt(u * u + (1.0 - rho * rho)))

    def sig(kc):
        return np.sqrt((_w(kc, th2, rh2) - _w(kc, th1, rh1)) / dt)

    def dsig(x):
        return float(np.imag(sig(x + 1e-20j)) / 1e-20)    # exact 1st deriv

    R = [(dsig(4e-3 / 2 ** i) - dsig(-4e-3 / 2 ** i)) / (2 * 4e-3 / 2 ** i) for i in range(5)]
    for kk in range(1, len(R)):
        f = 4 ** kk
        R = [(f * R[i + 1] - R[i]) / (f - 1.0) for i in range(len(R) - 1)]
    return float(np.real(sig(0.0 + 0j))), dsig(0.0), R[0], th1, th2


def test_cm_node_matches_iv_history_rule():
    # cm_node must reproduce the exact CM (theta, rho) interpolation iv_history uses internally
    slice_ts = np.array([0.05, 0.2, 0.5, 1.0])
    thetas = np.array([0.004, 0.016, 0.04, 0.08])
    rhos = np.array([-0.40, -0.35, -0.30, -0.25])
    eta, gamma = 0.7, 0.45
    psi = theta_phi(thetas, eta, gamma)
    rho_psi = rhos * psi
    for t_cm in (0.1, 0.3, 0.75):
        theta_ref = float(np.interp(t_cm, slice_ts, thetas))
        psi_cm = float(theta_phi(np.array([theta_ref]), eta, gamma)[0])
        rho_ref = float(np.clip(np.interp(t_cm, slice_ts, rho_psi) / psi_cm, -0.999, 0.999))
        th, rh, ps = derived.cm_node(slice_ts, thetas, rhos, eta, gamma, t_cm)
        assert abs(th - theta_ref) < 1e-12 and abs(rh - rho_ref) < 1e-12 and abs(ps - psi_cm) < 1e-12


def test_otm_black_matches_explicit_and_parity():
    # otm_black equals the explicit undiscounted Black formula (per unit forward), OTM side
    k = np.linspace(-0.6, 0.6, 61)
    w = np.full_like(k, 0.04)
    sw = np.sqrt(w)
    d1 = (-k + 0.5 * w) / sw
    d2 = d1 - sw
    ref = np.where(k >= 0, ndtr(d1) - np.exp(k) * ndtr(d2), np.exp(k) * ndtr(-d2) - ndtr(-d1))
    got = derived.otm_black(k, w)
    assert np.allclose(got, ref, atol=1e-12), float(np.max(np.abs(got - ref)))
    # put-call parity sanity at k=0: call-put = 1 - e^k = 0, so OTM call(0)==OTM put(0)
    c0 = float(derived.otm_black(np.array([1e-9]), np.array([0.04]))[0])
    p0 = float(derived.otm_black(np.array([-1e-9]), np.array([0.04]))[0])
    assert abs(c0 - p0) < 1e-6


def test_recertify_pair_certifies_and_flags():
    # a monotone-in-theta pair certifies; an inverted pair (shorter tenor has MORE variance) flags
    ok, dmin = derived.recertify_pair(0.02, -0.3, 0.05, -0.3, 0.7, 0.45)
    assert ok and dmin >= -1e-6
    bad, dmin2 = derived.recertify_pair(0.05, -0.3, 0.02, -0.3, 0.7, 0.45)
    assert not bad and dmin2 < 0.0


# --- forward-vol (P2.3) ---------------------------------------------------------------------------
# arb-free, calendar-monotone synthetic surface spanning the full CM-tenor grid (7..365d)
_TS_F = np.array([0.02, 0.08, 0.25, 0.5, 1.0])
_TH_F = np.array([0.0016, 0.0060, 0.0170, 0.0320, 0.0620])     # monotone increasing theta (ATM var)
_RH_F = np.array([-0.62, -0.55, -0.46, -0.38, -0.30])          # smile flattens / de-skews with tenor
_ETA_F, _GAMMA_F = 0.85, 0.42


def test_forward_vol_closed_form_skew_curv_machine_exact():
    # the emitted CLOSED-FORM fwd_skew/fwd_curv must equal an independent complex-step + Romberg
    # reference for sigma_fwd(k) at k=0 - to machine precision (skew) and < 1e-6 (curv), every pair
    rows, _ = derived.build_forward_vol(_TS, "TEST", _TS_F, _TH_F, _RH_F, _ETA_F, _GAMMA_F, is_eod=False)
    assert rows, "no forward_vol pairs emitted on the synthetic span"
    for r in rows:
        if r["fwd_skew"] is None:                                  # nulled on degenerate fwd_atm
            continue
        t1, t2 = r["t1"] / DAYCOUNT, r["t2"] / DAYCOUNT
        _s0, skew_ref, curv_ref, _a, _b = _sigma_fwd_ref(_TS_F, _TH_F, _RH_F, _ETA_F, _GAMMA_F, t1, t2)
        assert abs(skew_ref - r["fwd_skew"]) < 1e-9, (r["t1"], r["t2"], skew_ref, r["fwd_skew"])
        assert abs(curv_ref - r["fwd_curv"]) < 1e-6, (r["t1"], r["t2"], curv_ref, r["fwd_curv"])


def test_forward_vol_additivity_and_arb_gate():
    # additivity w_fwd(0)=theta2-theta1 (=> fwd_var*dt=theta2-theta1); arb-free synthetic certifies
    # with delta_ok and zero UNFLAGGED negative forward variance; rows round-trip the schema
    rows, grid = derived.build_forward_vol(_TS, "TEST", _TS_F, _TH_F, _RH_F, _ETA_F, _GAMMA_F, is_eod=True)
    for r in rows:
        dt = r["t2"] / DAYCOUNT - r["t1"] / DAYCOUNT
        _s0, _sk, _cv, th1, th2 = _sigma_fwd_ref(_TS_F, _TH_F, _RH_F, _ETA_F, _GAMMA_F,
                                                 r["t1"] / DAYCOUNT, r["t2"] / DAYCOUNT)
        assert abs(r["fwd_var"] * dt - (th2 - th1)) < 1e-12
        assert r["derive_status"] == "ok" and r["arb_ok"] and r["delta_ok"]
        assert not (r["fwd_var_min"] < -1e-9 and r["arb_ok"])     # no unflagged negative fwd variance
        assert r["fwd_rr_25d"] is not None and r["fwd_bf_25d"] is not None
    # negative-skew equity smile => forward var swap >= ATM forward variance (convexity_adj >= 0)
    clean = [r for r in rows if r["fwd_var_swap"] is not None]
    assert clean and all(r["fwd_convexity_adj"] >= -1e-9 for r in clean)
    assert build_table("forward_vol", rows).num_rows == len(rows)
    assert build_table("forward_vol_grid", grid).num_rows == len(grid)


def test_forward_vol_flags_calendar_violation():
    # a NON-monotone theta term structure (a dip) makes some CM pair invert -> the double arb gate must
    # flag derive_status='cal_fail' / arb_ok=False on the offending pair (calendar positivity is the
    # necessary condition the forward smile fails first)
    th_bad = np.array([0.0016, 0.0060, 0.0500, 0.0150, 0.0620])   # 0.05 -> 0.015 dip at t=0.25->0.5
    rows, _ = derived.build_forward_vol(_TS, "TEST", _TS_F, th_bad, _RH_F, _ETA_F, _GAMMA_F, is_eod=False)
    flagged = [r for r in rows if r["derive_status"] != "ok"]
    assert flagged, "calendar inversion not detected by the arb gate"
    assert all((not r["arb_ok"]) for r in flagged)
    assert any(r["derive_status"] == "cal_fail" for r in rows)


# --- earnings-vol (P2.4): DJ term-structure strip on synthetic slices w/ a KNOWN injected jump --------
_E_NU = 0.36 ** 2            # flat diffusion rate
_E_J2 = 0.075 ** 2          # injected earnings jump variance
_E_TAUD = 30                # event 30 calendar days out
_E_TS = _dt.datetime(2026, 1, 15, 20, 45, tzinfo=_dt.timezone.utc)
_E_US = {"resid_known": True, "resid_max": 0.0}
_E_EVENT = {"earnings_date": _E_TS.date() + _dt.timedelta(days=_E_TAUD), "from_8k": True}


def _earn_slices(days, rho):
    t = np.array(days) / DAYCOUNT
    th = _E_NU * t + _E_J2 * (np.array(days) >= _E_TAUD)
    return t, th, np.full(len(days), rho), [_E_TS.date() + _dt.timedelta(days=int(d)) for d in days]


def test_earnings_vol_recovers_injected_jump_and_roundtrips():
    # DJ strip must recover the injected J^2 and diffusion nu to machine precision; row round-trips schema
    t, th, rho, exp = _earn_slices([12, 23, 58], rho=-0.45)     # 12,23d ex-event + 58d spanning the 30d event
    rows = derived.build_earnings_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp,
                                      None, _E_US, _E_EVENT)
    assert len(rows) == 1
    r = rows[0]
    assert r["derive_status"] == "ok" and not r["both_span"]
    assert abs(r["event_var"] - _E_J2) < 1e-9 and abs(r["vbar_diff"] - _E_NU) < 1e-9
    assert abs(r["sigma_J_rms"] - np.sqrt(_E_J2)) < 1e-9
    assert abs(r["e_abs_j_mad"] - np.sqrt(2 / np.pi) * r["sigma_J_rms"]) < 1e-12   # Gaussian-assumption MAD
    assert abs(r["event_share"] - _E_J2 / r["theta_span"]) < 1e-9
    assert build_table("earnings_vol", rows).num_rows == 1


def test_earnings_vol_identifiability_fork():
    # two CO-SPANNING expiries identify only the diffusion (lump cancels) -> J^2 unidentified, flagged
    t, th, rho, exp = _earn_slices([44, 79], rho=-0.45)        # both span the 30d event
    r = derived.build_earnings_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp,
                                   None, _E_US, _E_EVENT)[0]
    assert r["derive_status"] == "both_span_unidentified" and r["both_span"]
    assert r.get("event_var") is None


def test_earnings_vol_leakage_gate():
    # a PAST earnings event is never used (no look-ahead); None event emits nothing
    t, th, rho, exp = _earn_slices([12, 23, 58], rho=-0.45)
    past = {"earnings_date": _E_TS.date() - _dt.timedelta(days=5), "from_8k": True}
    r = derived.build_earnings_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp,
                                   None, _E_US, past)[0]
    assert r["derive_status"] == "event_in_past" and r.get("event_var") is None
    assert derived.build_earnings_vol(_E_TS, "T", t, th, rho, _ETA_F, _GAMMA_F, exp,
                                      None, _E_US, None) == []


def test_bkm_cumulants_gaussian_on_flat_black():
    # BKM cumulants of a flat (no-skew) Black smile -> Gaussian: rn_skew->0, rn_kurt->3, k2->w
    kw = np.arange(-12, 12 + 1e-9, 0.01)
    cu = derived.bkm_cumulants(kw, np.full_like(kw, 0.04))
    assert abs(cu["rn_skew"]) < 1e-3
    assert abs(cu["rn_kurt"] - 3.0) < 5e-3        # ~2e-3 residual is trapezoid discretization, not formula
    assert abs(cu["k2"] - 0.04) < 1e-4 and abs(cu["mu"] + 0.02) < 1e-4


def test_bkm_cumulants_rate_invariant():
    # bkm_cumulants is a pure function of (k,w): rn_skew/rn_kurt must be identical regardless of any rate
    # (the drift is FORWARD-centered, r-free) -- the regression for the spot-vs-forward mu bug
    k = dense_k() if False else np.arange(-2, 2 + 1e-9, 0.005)
    w = w_of_k(k, 0.05, -0.45, _ETA_F, _GAMMA_F)
    cu = derived.bkm_cumulants(k, w)
    assert "discount" not in derived.bkm_cumulants.__code__.co_varnames   # no rate argument at all
    assert cu["rn_skew"] is not None and cu["rn_skew"] < 0               # equity left-skew


def test_earnings_vol_baseline_arb_fail_floors_vbar():
    # a DECREASING front-end (theta dips) violates the baseline calendar -> vbar floored at 0, flagged,
    # and event_var is NOT inflated by the spurious negative diffusion rate
    days = [12, 23, 58]
    t = np.array(days) / DAYCOUNT
    th = np.array([0.0080, 0.003991, 0.025937])      # 12d theta > 23d theta: baseline calendar violated
    rho = np.full(3, -0.45)
    exp = [_E_TS.date() + _dt.timedelta(days=d) for d in days]
    r = derived.build_earnings_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp, None, _E_US, _E_EVENT)[0]
    assert r["derive_status"] == "baseline_arb_fail" and r["baseline_arb_ok"] is False
    assert r["vbar_diff"] == 0.0                       # floored
    assert abs(r["event_var"] - (r["theta_span"] - r["theta_near"])) < 1e-12   # no inflation


# --- earnings-vol macro contamination (A2 fix): a co-located FOMC/CPI inside the (T_near,T_span] jump
#     bracket means the single-event DJ attribution is NOT pure earnings -> flag it, never claim 'ok' ----
def _macro(days_out, typ="FOMC"):
    return [{"date": _E_TS.date() + _dt.timedelta(days=days_out), "event_type": typ,
             "session": "afternoon_1400", "source": "official:test"}]


def test_earnings_vol_flags_macro_contamination():
    # earnings 30d, near=23d ex-event, span=58d -> a FOMC at 45d sits INSIDE the (23d,58d] bracket and folds
    # its own variance into the jump -> the row must flag macro_in_window and NOT claim a clean 'ok'
    t, th, rho, exp = _earn_slices([12, 23, 58], rho=-0.45)
    r = derived.build_earnings_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp,
                                   None, _E_US, _E_EVENT, macro_cal=_macro(45))[0]
    assert r["macro_in_window"] is True
    assert r["derive_status"] != "ok"            # single-event attribution invalid under macro co-location
    assert r.get("event_var") is None            # pure-earnings jump unidentifiable -> null, not inflated
    assert build_table("earnings_vol", [r]).num_rows == 1


def test_earnings_vol_macro_outside_bracket_stays_clean():
    # SAME setup but the FOMC is at 10d (before the near 23d expiry) -> outside the bracket, no contamination
    # -> the clean DJ recovery is unchanged and macro_in_window is explicitly False (not null)
    t, th, rho, exp = _earn_slices([12, 23, 58], rho=-0.45)
    r = derived.build_earnings_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp,
                                   None, _E_US, _E_EVENT, macro_cal=_macro(10))[0]
    assert r["macro_in_window"] is False
    assert r["derive_status"] == "ok"
    assert abs(r["event_var"] - _E_J2) < 1e-9


def test_earnings_vol_macro_none_leaves_flag_null():
    # backward-compat: when no macro calendar is supplied the flag is undeterminable -> stays None
    t, th, rho, exp = _earn_slices([12, 23, 58], rho=-0.45)
    r = derived.build_earnings_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp,
                                   None, _E_US, _E_EVENT)[0]
    assert r["macro_in_window"] is None and r["derive_status"] == "ok"


# --- event-vol (P2.5): multi-event DJ NNLS strip ---------------------------------------------------
_EV_NU = 0.30 ** 2
_EV_DAYS = [5, 12, 19, 26, 40, 54]
_EV_TAUE, _EV_TAUM = 15, 33
_EV_X = [0.06 ** 2, 0.045 ** 2]            # injected earnings + FOMC variances
_EV_EARN = {"earnings_date": _E_TS.date() + _dt.timedelta(days=_EV_TAUE), "from_8k": True}
_EV_FOMC = [{"date": _E_TS.date() + _dt.timedelta(days=_EV_TAUM), "event_type": "FOMC",
             "session": "afternoon_1400", "source": "official:fomc"}]


def _ev_slices(with_events):
    t = np.array(_EV_DAYS) / DAYCOUNT
    th = _EV_NU * t.copy()
    if with_events:
        th = th + _EV_X[0] * (np.array(_EV_DAYS) >= _EV_TAUE) + _EV_X[1] * (np.array(_EV_DAYS) >= _EV_TAUM)
    return t, th, np.full(len(_EV_DAYS), -0.45), [_E_TS.date() + _dt.timedelta(days=d) for d in _EV_DAYS]


def test_event_vol_single_event_form_a_and_roundtrip():
    # one bracketed event reduces EXACTLY to the earnings form-A jump; row round-trips the schema
    t, th, rho, exp = _ev_slices(True)
    rows = derived.build_event_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp, None,
                                   {"resid_known": True}, macro_cal=None, earnings_event=_EV_EARN)
    ev = next(r for r in rows if r["event_type"] == "EARNINGS" and r["derive_status"] == "ok")
    assert abs(ev["event_var"] - _EV_X[0]) < 1e-9 and ev["kappa_alloc_method"] == "bracket"
    assert build_table("event_vol", rows).num_rows == len(rows)
    # no calendar at all -> empty
    assert derived.build_event_vol(_E_TS, "T", t, th, rho, _ETA_F, _GAMMA_F, exp, None, None) == []


def test_event_vol_multi_event_nnls_separates():
    # two co-located events (earnings + FOMC, distinct expiries between) are separated by the NNLS
    t, th, rho, exp = _ev_slices(True)
    rows = derived.build_event_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp, None,
                                   {"resid_known": True}, macro_cal=_EV_FOMC, earnings_event=_EV_EARN)
    by = {r["event_type"]: r for r in rows if r["derive_status"] == "ok"}
    assert abs(by["EARNINGS"]["event_var"] - _EV_X[0]) < 1e-4
    assert abs(by["FOMC"]["event_var"] - _EV_X[1]) < 1e-4
    assert by["EARNINGS"]["kappa_alloc_method"] == "nnls"


def test_event_vol_nnls_solver_failure_degrades_gracefully(monkeypatch):
    # regression for the AMD 2024-12-17 silent drop: a degenerate staircase made the bvls/lstsq SVD fail to
    # converge (LinAlgError), which crashed a 679k-quote name-day. The whole name-day must SURVIVE -- the var
    # events degrade to 'nnls_unconverged' (event_var null) and flow events still process.
    def _boom(*a, **k):
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
    monkeypatch.setattr(derived, "lsq_linear", _boom)
    t, th, rho, exp = _ev_slices(True)        # earnings 15d + FOMC 33d -> K=2 -> NNLS path
    rows = derived.build_event_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp, None,
                                   {"resid_known": True}, macro_cal=_EV_FOMC, earnings_event=_EV_EARN)
    var_rows = [r for r in rows if r["event_type"] in derived._VAR_EVENTS]
    assert var_rows and all(r["derive_status"] == "nnls_unconverged" for r in var_rows)
    assert all(r.get("event_var") is None for r in var_rows)
    assert build_table("event_vol", rows).num_rows == len(rows)    # still schema-valid, no crash


def test_event_vol_flat_no_jump_and_gaussian_kurt():
    # flat ladder -> ~0 event variance; the Gaussian-jump kurtosis identity holds
    t, th, rho, exp = _ev_slices(False)
    rows = derived.build_event_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp, None,
                                   {"resid_known": True}, earnings_event=_EV_EARN)
    ev = next(r for r in rows if r["event_type"] == "EARNINGS" and r["derive_status"] == "ok")
    assert abs(ev["event_var"]) < 1e-9
    _mu, _rms, _mad, kk = derived._event_moves(_EV_X[0])
    assert abs(kk - np.sqrt(2 / np.pi)) < 2e-3


def test_event_vol_same_date_bracket_sum_only():
    # two events on the SAME date share an incidence column -> only the sum is identified (flagged)
    t, th, rho, exp = _ev_slices(True)
    same = [{"date": _E_TS.date() + _dt.timedelta(days=_EV_TAUE), "event_type": "FOMC",
             "session": "afternoon_1400", "source": "official:fomc"}]   # FOMC ON the earnings date
    rows = derived.build_event_vol(_E_TS, "TEST", t, th, rho, _ETA_F, _GAMMA_F, exp, None,
                                   {"resid_known": True}, macro_cal=same, earnings_event=_EV_EARN)
    flagged = [r for r in rows if r["derive_status"] == "bracket_sum_only"]
    assert len(flagged) == 2 and all(r.get("event_var") is None for r in flagged)


def test_vrp_flat_lognormal_collapse():
    # on a flat Black smile the 3 strikes collapse: K_log^2 ~ K_var ~ sigma^2, skew_corr ~ 0, Gaussian BKM
    k = np.arange(-2, 2 + 1e-9, 0.005)
    sig, t = 0.20, 30 / DAYCOUNT
    w = np.full_like(k, sig * sig * t)
    o = derived.otm_black(k, w)
    trap = getattr(np, "trapezoid", None) or np.trapz
    klog2 = (2 / t) * float(trap(np.exp(-k) * o, k))
    svix2 = (2 / t) * float(trap(np.exp(k) * o, k))
    cu = derived.bkm_cumulants(k, w)
    assert abs(np.sqrt(klog2) - sig) < 1e-3 and abs(np.sqrt(cu["k2"] / t) - sig) < 1e-3
    assert abs(cu["k2"] / t - klog2) < 5e-4          # skew_corr ~ 0 for a lognormal
    assert svix2 >= klog2 - 1e-9                      # symmetric: SVIX^2 >= K_log^2


def test_vrp_implied_left_skew_equity():
    # left-skewed (rho<0) arb-free synthetic span: every CM tenor certifies, K_log^2 >= SVIX^2,
    # skew_corr >= 0 (Du-Kapadia), rn_skew < 0, and the layer round-trips the schema
    rows = derived.build_vrp_implied(_TS, "TEST", _TS_F, _TH_F, _RH_F, _ETA_F, _GAMMA_F)
    assert rows and all(r["vrp_status"] == "ok" for r in rows)
    for r in rows:
        assert r["iv_var"] >= r["svix2"] - 1e-9       # left-skew: log-contract overweights the rich put wing
        assert r["skew_corr"] >= -1e-9                # DK: VIX undervalues negatively-skewed variance
        assert r["rn_skew"] < 0                       # equity left-skew
        assert 0.0 <= r["tail_frac"] < 0.5
    assert build_table("vrp_implied", rows).num_rows == len(rows)


def test_earnings_vol_degenerate_brackets_no_crash():
    # duplicate / non-monotone slice_ts must flag, never raise ZeroDivisionError
    days = [23, 23, 58]                                # duplicate near year-fractions
    t = np.array(days) / DAYCOUNT
    th = _E_NU * t + _E_J2 * (np.array(days) >= _E_TAUD)
    exp = [_E_TS.date() + _dt.timedelta(days=int(d)) for d in days]
    r = derived.build_earnings_vol(_E_TS, "TEST", t, th, np.full(3, -0.45), _ETA_F, _GAMMA_F,
                                   exp, None, _E_US, _E_EVENT)[0]
    assert r["derive_status"] in ("degenerate_brackets", "ok")   # must not crash; flagged if dup is the bracket
