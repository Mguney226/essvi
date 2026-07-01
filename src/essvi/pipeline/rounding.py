"""Deterministic pre-write rounding."""
from __future__ import annotations

from ..constants import (
    ROUND_IV,
    ROUND_K,
    ROUND_PARAM,
    ROUND_PRICE,
    ROUND_RMSE,
    ROUND_T,
    ROUND_W,
)

_ROUND = {
    "vol_surface_params": {
        "t": ROUND_T,                                       # the per-expiry tenor: round consistently with
        "theta": ROUND_PARAM, "rho": ROUND_PARAM, "phi": ROUND_PARAM,   # vol_surface_grid.t (was shipped raw
        "forward": 4, "discount": 8, "reprice_rmse": ROUND_RMSE,        # float64 -> cross-layer join mismatch)
        "dvol_dborrow": 4, "dvol_ddiv": 4, "eta": ROUND_PARAM, "gamma": ROUND_PARAM,
        "median_spread": ROUND_PRICE,                       # $ spread, round like prices (n_quotes is int, no round)
    },
    "vol_surface_grid": {"t": ROUND_T, "k": ROUND_K, "iv": ROUND_IV, "w": ROUND_W,
                          "dist_k": ROUND_K, "dist_t": ROUND_T},
    "iv_history": {"atm_iv": ROUND_IV, "rr_25d": ROUND_IV, "bf_25d": ROUND_IV,
                    "slope": ROUND_IV, "fwd": 4},
    "option_quotes": {"strike": 4, "bid": ROUND_PRICE, "ask": ROUND_PRICE, "mid": ROUND_PRICE,
                       "iv": ROUND_IV, "delta": 6, "gamma": 8, "vega": 6, "theta": 6,
                       "borrow": 6, "quote_weight": 6},
    "underlier_state": {"spot": ROUND_PRICE, "carry": ROUND_PARAM, "resid_max": ROUND_PARAM},
    "forward_vol": {"dt": ROUND_T, "fwd_atm_vol": ROUND_IV, "fwd_var": ROUND_W, "fwd_skew": ROUND_IV,
                    "fwd_curv": ROUND_IV, "fwd_rr_25d": ROUND_IV, "fwd_bf_25d": ROUND_IV,
                    "fwd_rr_10d": ROUND_IV, "fwd_bf_10d": ROUND_IV, "fwd_var_swap": ROUND_W,
                    "fwd_convexity_adj": ROUND_W, "near_atm_vol": ROUND_IV, "far_atm_vol": ROUND_IV,
                    "fwd_var_min": ROUND_W, "fwd_density_min": ROUND_PARAM,
                    "fwd_density_integral": ROUND_PARAM, "tail_frac": ROUND_IV,
                    "eta": ROUND_PARAM, "gamma": ROUND_PARAM},
    "forward_vol_grid": {"k": ROUND_K, "iv_fwd": ROUND_IV, "w_fwd": ROUND_W},
    "earnings_vol": {"event_tau": ROUND_T, "date_conf_weight": ROUND_PARAM,
                     "t_near": ROUND_T, "t_span": ROUND_T, "theta_near": ROUND_W, "theta_span": ROUND_W,
                     "vbar_diff": ROUND_W, "event_var": ROUND_W, "event_var_raw": ROUND_W,
                     "event_var_se": ROUND_W, "event_var_lo": ROUND_W, "event_var_hi": ROUND_W,
                     "sigma_J_rms": ROUND_IV, "e_abs_j_mad": ROUND_IV, "iv_bump_spanning": ROUND_IV,
                     "rn_skew": ROUND_PARAM, "rn_kurt": ROUND_PARAM, "tail_frac": ROUND_IV,
                     "rnd_integral": ROUND_PARAM, "event_share": ROUND_PARAM},
    "vrp_implied": {"theta_cm": ROUND_W, "rho_cm": ROUND_PARAM, "atm_var": ROUND_W,
                    "iv_var": ROUND_W, "iv_var_core": ROUND_W, "svix2": ROUND_W, "k_var": ROUND_W,
                    "skew_corr": ROUND_W, "rn_skew": ROUND_PARAM, "rn_kurt": ROUND_PARAM,
                    "tail_frac": ROUND_PARAM, "density_min": ROUND_PARAM, "rnd_integral": ROUND_PARAM},
    "event_vol": {"tau": ROUND_T, "nu_c": ROUND_W, "event_var": ROUND_W, "event_var_raw": ROUND_W,
                  "ident_rel": ROUND_PARAM, "event_var_se": ROUND_W, "sigma_J_rms": ROUND_IV,
                  "e_abs_j_mad": ROUND_IV, "drift": ROUND_W, "kappa_kurt": ROUND_PARAM,
                  "opex_signed_flow": ROUND_W},
}


def round_rows(layer: str, rows: list[dict]) -> list[dict]:
    spec = _ROUND.get(layer, {})
    for r in rows:
        for col, nd in spec.items():
            v = r.get(col)
            if v is not None:
                try:
                    # + 0.0 normalizes -0.0 -> 0.0 (they have different byte reprs; -0.0 broke cross-machine
                    # byte-sha256 + the determinism gate data-sha for numerically identical fits).
                    r[col] = round(float(v), nd) + 0.0
                except (TypeError, ValueError):
                    pass
    return rows
