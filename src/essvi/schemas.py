"""The four output-layer Arrow schemas (exact column order/types the docs + methodology promise).

Single source of truth: pipeline writers must produce exactly these.
"""
from __future__ import annotations

import pyarrow as pa

_TS = pa.timestamp("ns", tz="UTC")

VOL_SURFACE_PARAMS = pa.schema(
    [
        ("ts", _TS),
        ("underlying", pa.string()),
        ("root", pa.string()),              # listing series the slice's quotes came from (SPX | SPXW ...)
        ("expiry", pa.date32()),
        ("t", pa.float64()),                # Act/365F years to settlement (disambiguates AM/PM same-date)
        ("theta", pa.float64()),
        ("rho", pa.float64()),
        ("phi", pa.float64()),
        ("eta", pa.float64()),              # global eSSVI params (were post-hoc migrated; now declared)
        ("gamma", pa.float64()),
        ("forward", pa.float64()),
        ("discount", pa.float64()),
        ("reprice_rmse", pa.float64()),     # Black vol points, in-sample
        ("arb_ok", pa.bool_()),
        ("dvol_dborrow", pa.float64()),
        ("dvol_ddiv", pa.float64()),
        ("fit_method", pa.string()),        # ssvi | essvi
        ("fit_init", pa.string()),          # warm | cold
        ("fit_status", pa.string()),        # ok | insufficient_quotes | no_forward | below_min_tenor | failed
        ("median_spread", pa.float64()),    # methodology §04: per-slice median quoted spread (eSSVI trigger input)
        ("n_quotes", pa.int32()),           # methodology §06: count of quotes entering the fit (observed count on failure)
    ]
)

VOL_SURFACE_GRID = pa.schema(
    [
        ("ts", _TS),
        ("underlying", pa.string()),
        ("root", pa.string()),
        ("expiry", pa.date32()),
        ("t", pa.float64()),                # Act/365F years
        ("k", pa.float64()),                # log-forward moneyness
        ("iv", pa.float64()),               # Black vol
        ("w", pa.float64()),                # total variance
        ("flag", pa.string()),              # observed | interpolated | extrapolated
        ("dist_k", pa.float64()),
        ("dist_t", pa.float64()),
    ]
)

IV_HISTORY = pa.schema(
    [
        ("ts", _TS),
        ("underlying", pa.string()),
        ("tenor", pa.int16()),              # constant-maturity days
        ("atm_iv", pa.float64()),
        ("rr_25d", pa.float64()),
        ("bf_25d", pa.float64()),
        ("slope", pa.float64()),
        ("fwd", pa.float64()),
    ]
)

OPTION_QUOTES = pa.schema(
    [
        ("ts", _TS),
        ("underlying", pa.string()),
        ("root", pa.string()),
        ("expiry", pa.date32()),
        ("strike", pa.float64()),
        ("right", pa.string()),             # C | P
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("mid", pa.float64()),
        ("bid_sz", pa.int32()),
        ("ask_sz", pa.int32()),
        ("iv", pa.float64()),               # de-Americanized Black vol (= Black for European index)
        ("delta", pa.float64()),
        ("gamma", pa.float64()),
        ("vega", pa.float64()),
        ("theta", pa.float64()),            # greek
        ("borrow", pa.float64()),            # de-Am NET CARRY q (= borrow rate - dividend yield), not a pure borrow rate
        ("quote_weight", pa.float64()),
        ("flag", pa.string()),
    ]
)

UNDERLIER_STATE = pa.schema([
    ("ts", _TS),
    ("underlying", pa.string()),
    ("spot", pa.float64()),
    ("carry", pa.float64()),            # de-Am net carry q (borrow - dividend yield), implied from the forward curve
    ("spot_status", pa.string()),       # ok | insufficient
    ("n_expiries", pa.int16()),
    ("resid_max", pa.float64()),        # NULL when <=2 expiries (residual unmeasurable)
    ("resid_known", pa.bool_()),        # False => residual unmeasurable, NOT a clean reading
])

FORWARD_VOL = pa.schema([
    ("ts", _TS), ("underlying", pa.string()),
    ("t1", pa.int16()), ("t2", pa.int16()), ("dt", pa.float64()),     # CM tenor pair (days) + (t2-t1)/365
    ("fwd_atm_vol", pa.float64()), ("fwd_var", pa.float64()),
    ("fwd_skew", pa.float64()), ("fwd_curv", pa.float64()),           # CLOSED-FORM from slice params
    ("fwd_rr_25d", pa.float64()), ("fwd_bf_25d", pa.float64()),       # forward-delta space, nullable
    ("fwd_rr_10d", pa.float64()), ("fwd_bf_10d", pa.float64()),
    ("fwd_var_swap", pa.float64()), ("fwd_convexity_adj", pa.float64()),  # anchor + (varswap - atm_var), >=0 eq
    ("near_atm_vol", pa.float64()), ("far_atm_vol", pa.float64()),
    ("fwd_var_min", pa.float64()),            # min_k w_fwd / dt (calendar slack)
    ("fwd_density_min", pa.float64()),        # min FD-Breeden-Litzenberger forward RND (butterfly)
    ("fwd_density_integral", pa.float64()),   # forward RND integral on the grid (-> 1)
    ("tail_frac", pa.float64()),
    ("eta", pa.float64()), ("gamma", pa.float64()),
    ("arb_ok", pa.bool_()),                   # DOUBLE gate: HM parent calendar AND FD-BL butterfly
    ("delta_ok", pa.bool_()),                 # forward-delta monotone (RR/BF valid)
    ("derive_status", pa.string()),           # ok | cal_fail | bfly_fail
])

FORWARD_VOL_GRID = pa.schema([
    ("ts", _TS), ("underlying", pa.string()),
    ("t1", pa.int16()), ("t2", pa.int16()), ("k", pa.float64()),
    ("iv_fwd", pa.float64()), ("w_fwd", pa.float64()), ("flag", pa.string()),
])

EARNINGS_VOL = pa.schema([                            # single-name Dubinsky-Johannes earnings strip
    ("ts", _TS), ("underlying", pa.string()),
    ("earnings_date", pa.date32()), ("from_8k", pa.bool_()),   # 8-K item-2.02 date (full weight) vs fallback
    ("session", pa.string()),                          # 'unknown' (calendar is date-resolution only)
    ("event_tau", pa.float64()), ("date_conf_weight", pa.float64()),
    ("resid_known", pa.bool_()),                       # de-Am div-contamination known (from underlier_state)
    ("expiry_near", pa.date32()), ("expiry_span", pa.date32()),  # ex-event anchor + spanning expiry
    ("t_near", pa.float64()), ("t_span", pa.float64()),
    ("theta_near", pa.float64()), ("theta_span", pa.float64()),  # ATM total variance w(0;T)=theta
    ("vbar_diff", pa.float64()), ("vbar_diff_capped", pa.bool_()),   # diffusive rate, no-arb-capped
    ("baseline_arb_ok", pa.bool_()),                   # event-free baseline bracket calendar-certified
    ("baseline_tier", pa.string()), ("n_diff_brackets", pa.int16()),
    ("event_var", pa.float64()),                       # J^2 = E^Q[J^2], no-arb-CAPPED at >=0 (headline)
    ("event_var_raw", pa.float64()),                   # SIGNED pre-cap DJ estimate (can be < 0)
    ("event_var_se", pa.float64()),                    # reprice-RMSE-propagated SE proxy (0 when capped)
    ("event_var_lo", pa.float64()), ("event_var_hi", pa.float64()),  # +-1 session band (from_8k=False)
    ("both_span", pa.bool_()),                         # identifiability: both expiries span -> J^2 unidentified
    ("arb_raw_ok", pa.bool_()),                        # raw calendar theta_span>=theta_near
    ("sigma_J_rms", pa.float64()),                     # RMS implied move = sqrt(max(J^2,0))
    ("e_abs_j_mad", pa.float64()),                     # MAD move under an EXPLICIT Gaussian-jump assumption
    ("rn_skew", pa.float64()), ("rn_kurt", pa.float64()),   # spanning-slice raw BKM RN moments
    ("tail_frac", pa.float64()), ("rnd_integral", pa.float64()),
    ("event_share", pa.float64()), ("iv_bump_spanning", pa.float64()),
    ("contam_ok", pa.bool_()),                         # contamination certificate (null until P3 re-fits)
    ("macro_in_window", pa.bool_()),                   # reserved: macro co-located in (T_near,T_span); null till P2.5
    ("n_brackets", pa.int16()),
    ("derive_status", pa.string()),                    # ok | no_bracket | both_span_unidentified | thin |
])                                                     # near_expiry_guard | degenerate_brackets | event_in_past |
#                                                        baseline_arb_fail | div_contam_unknown | neg_var_clipped

VRP_IMPLIED = pa.schema([                             # 3-strike model-free variance (per CM tenor)
    ("ts", _TS), ("underlying", pa.string()),
    ("tenor", pa.int16()),                             # constant-maturity days
    ("theta_cm", pa.float64()), ("rho_cm", pa.float64()),
    ("atm_var", pa.float64()),                         # theta_cm / t (ATM variance, annualized)
    ("iv_var", pa.float64()),                          # K_log^2 log-contract var swap (VIX^2-equivalent)
    ("iv_var_core", pa.float64()),                     # K_log^2 over |k|<=KFIT_MAX (quoted span)
    ("svix2", pa.float64()),                           # Martin SVIX^2 (jump-robust simple-return variance)
    ("k_var", pa.float64()),                           # Var(ln S_T/F)/t (BKM squared-log-return variance)
    ("skew_corr", pa.float64()),                       # k_var - iv_var (Du-Kapadia surplus; >0 eq: VIX undervalues)
    ("rn_skew", pa.float64()), ("rn_kurt", pa.float64()),   # forward-centered BKM RN moments
    ("tail_frac", pa.float64()),                       # share of K_log^2 from the |k|>KFIT_MAX wing
    ("density_min", pa.float64()),                     # FD-BL min density on the CM smile (re-certification)
    ("rnd_integral", pa.float64()),
    ("vrp_status", pa.string()),                       # ok | tail_unreliable (tail_frac>=0.15) | arb_violation
])

EVENT_VOL = pa.schema([                               # general multi-event Dubinsky-Johannes variance strip
    ("ts", _TS), ("underlying", pa.string()),
    ("scope", pa.string()),                            # index | single_name | flow (OPEX)
    ("event_date", pa.date32()), ("event_type", pa.string()),   # EARNINGS|FOMC|CPI|NFP|PPI|OPEX|QUAD_WITCH
    ("event_session", pa.string()), ("event_source", pa.string()),
    ("tau", pa.float64()),                             # Act/365 years to the event
    ("nu_c", pa.float64()),                            # local diffusive variance rate at tau
    ("n_eventfree_segments", pa.int16()),
    ("event_var", pa.float64()),                       # x_k = (sigma_J)^2, no-arb-capped >=0 (headline)
    ("event_var_raw", pa.float64()),                   # SIGNED pre-cap estimate
    ("ident_rel", pa.float64()),                       # |capped-signed|/x: regularization sensitivity (0=data-identified)
    ("event_var_se", pa.float64()),                    # NNLS residual ||Ax-b|| (multi-event)
    ("sigma_J_rms", pa.float64()),                     # RMS move = sqrt(event_var)
    ("e_abs_j_mad", pa.float64()),                     # MAD move (Gaussian-jump, straddle-recoverable)
    ("drift", pa.float64()),                           # E^Q[J] = martingale mu = -s^2/2
    ("kappa_kurt", pa.float64()),                      # E|J|/sqrt(E[J^2]); ->sqrt(2/pi)=0.7979 as event_var->0,
    #                                                  #   drifts slightly above for finite jumps (martingale drift
    #                                                  #   mu=-s^2/2); forced-Gaussian => deterministic in event_var,
    #                                                  #   NOT an independent Gaussianity test
    ("opex_signed_flow", pa.float64()),                # COARSE signed gamma-flow proxy (fwd-var anomaly vs the
    #                                                  #   cumulative-AVERAGE rate; contaminated by term-structure
    #                                                  #   slope, non-zero even with no OPEX effect); never a move
    ("kappa_alloc_method", pa.string()),               # bracket | nnls | bracket_sum_only
    ("n_events_in_window", pa.int16()), ("both_span", pa.bool_()), ("arb_raw_ok", pa.bool_()),
    ("contam_ok", pa.bool_()),                         # contamination certificate (null until P3)
    ("derive_status", pa.string()),                    # ok | weakly_identified | event_in_past |
])                                                     # both_span_unidentified | baseline_arb_fail |
#                                                        bracket_sum_only | degenerate_brackets

LAYERS = {
    "vol_surface_params": VOL_SURFACE_PARAMS,
    "vol_surface_grid": VOL_SURFACE_GRID,
    "iv_history": IV_HISTORY,
    "option_quotes": OPTION_QUOTES,
    "underlier_state": UNDERLIER_STATE,
    "forward_vol": FORWARD_VOL,
    "forward_vol_grid": FORWARD_VOL_GRID,
    "earnings_vol": EARNINGS_VOL,
    "vrp_implied": VRP_IMPLIED,
    "event_vol": EVENT_VOL,
}
