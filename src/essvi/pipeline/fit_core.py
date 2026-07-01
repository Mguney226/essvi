"""Shared fit-resolution + row-emission for BOTH the index (European) and single-name (de-Am'd American)
pipelines. Extracted verbatim from pipeline/snapshot.py so the warm/cold decision, the eSSVI escalation,
the §06 failure taxonomy, and the four-layer emission live in ONE place.

The only piece that genuinely diverges between the two pipelines is how option_quotes rows are built
(index: parity-Black IV of the observed mids; single-name: the de-Americalized IV) - so that is injected
as `option_quote_fn`. Everything else is identical.

Slice candidates are duck-typed: any object with .root, .expiry (date), .t, .status, .slice (a Slice),
.sig_mid, .quote_w_norm works. The shared SliceCand dataclass below is the canonical container.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from ..constants import MIN_EXPIRIES, WARM_RMSE_FACTOR, WARM_RMSE_MARGIN
from ..diagnostics import derived as der
from ..forward.term_structure import imply_spot_carry
from ..diagnostics import iv_history as ivh
from ..diagnostics import sensitivities as sens
from ..diagnostics.grid import cm_tenor_grid, slice_grid
from ..diagnostics.reprice import reprice_rmse
from ..diagnostics.verify import verify_surface
from ..ssvi.essvi import fit_essvi, fit_essvi_warm
from ..ssvi.fit import fit_joint
from ..ssvi.model import phi
from ..ssvi.objective import Slice


@dataclass
class SliceCand:
    """One (root, expiry) fit candidate, shared by both pipelines."""
    root: str
    expiry: dt.date
    t: float
    status: str = "ok"
    slice: Slice | None = None
    sig_mid: np.ndarray | None = None           # fitted-quote mid IVs (for reprice)
    quote_w_norm: np.ndarray | None = None       # fitted-quote normalized weights (for provenance)
    quotes_df: object = None                     # surviving quotes payload for the option_quotes layer
    n_two_sided: int = 0


@dataclass
class FitResolution:
    """Outcome of the warm/cold fit decision over a snapshot's fit candidates."""
    fitted: bool                                 # warm_used or a cold fit succeeded -> a surface exists
    warm_used: bool
    fit_obj: object = None                       # SSVIFit (cold) | None (warm/unfit); for parity callers
    thetas: np.ndarray | None = None
    rhos: np.ndarray | None = None
    eta: float = 0.0
    gamma: float = 0.0
    fit_method: str | None = None
    fit_init: str = "cold"


def _warm_passes(thetas, rhos, eta, gamma, fit_cands: list, prev_reprice_med) -> bool:
    """Quality gate on a warm fit: arbitrage-free on the dense grid AND reprice did not regress vs the
    prior snapshot. Ported from universe/name_pipeline so BOTH the index and single-name pipelines
    re-gate warm fits identically (the index path previously shipped warm fits with no re-check)."""
    arb_ok, _, _ = verify_surface(thetas, rhos, eta, gamma)
    if not bool(np.all(arb_ok)):
        return False
    if prev_reprice_med is None:
        return True
    rmses = [reprice_rmse(float(thetas[i]), float(rhos[i]), eta, gamma, c.t, c.slice.k, c.sig_mid)
             for i, c in enumerate(fit_cands)]
    med = float(np.median(rmses)) if rmses else float("inf")
    return med <= max(WARM_RMSE_FACTOR * prev_reprice_med, prev_reprice_med + WARM_RMSE_MARGIN)


def resolve_fit(cands: list, fit_cands: list, warm_state: dict | None,
                regate_warm: bool = True) -> FitResolution:
    """The warm-path (coverage>=0.7 -> fit_essvi_warm, re-gated for arb + reprice-regression) + cold
    cascade (fit_joint -> fit_essvi escalation) decision. A warm fit that fails its quality gate falls
    through to a full cold re-anchor. Mutates cand.status for insufficient/failed cases."""
    fit_init = "cold"
    warm_used = False
    e_warm = None
    fit = None
    if len(fit_cands) < MIN_EXPIRIES:
        for c in cands:
            if c.status == "ok":
                c.status = "insufficient_quotes"
        return FitResolution(fitted=False, warm_used=False, fit_obj=None)

    if warm_state and warm_state.get("seeds"):
        seeds_map = warm_state["seeds"]
        keys = [(c.root, c.expiry.isoformat()) for c in fit_cands]
        coverage = sum(1 for k in keys if k in seeds_map) / max(len(keys), 1)
        if coverage >= 0.7:
            seeds = []
            for c, k in zip(fit_cands, keys):
                if k in seeds_map:
                    seeds.append(seeds_map[k])
                else:
                    seeds.append((c.slice.theta_atm, warm_state.get("rho_med", -0.3)))
            e_warm = fit_essvi_warm([c.slice for c in fit_cands], seeds,
                                    warm_state["eta"], warm_state["gamma"])
            if e_warm is not None:
                warm_used = True
                if regate_warm and not _warm_passes(e_warm.thetas, e_warm.rhos, e_warm.eta, e_warm.gamma,
                                                     fit_cands, warm_state.get("prev_reprice_med")):
                    warm_used, e_warm = False, None     # failed quality gate -> fall through to cold re-anchor
    if not warm_used:
        fit = fit_joint([c.slice for c in fit_cands])   # cold path only when warm unavailable/failed

    if warm_used:
        return FitResolution(fitted=True, warm_used=True, fit_obj=None,
                             thetas=e_warm.thetas, rhos=e_warm.rhos, eta=e_warm.eta, gamma=e_warm.gamma,
                             fit_method="essvi", fit_init="warm")
    if fit is None:
        for c in fit_cands:
            if c.status == "ok" and len(fit_cands) >= MIN_EXPIRIES:
                c.status = "failed"
        return FitResolution(fitted=False, warm_used=False, fit_obj=None)

    # --- cold cascade: eSSVI escalation (§04): per-expiry rho when the global fit misses ---
    thetas = fit.thetas
    rhos = np.full(len(fit_cands), fit.rho)
    fit_method = fit.fit_method
    global_rmse = [
        reprice_rmse(float(thetas[i]), fit.rho, fit.eta, fit.gamma, c.t, c.slice.k, c.sig_mid)
        for i, c in enumerate(fit_cands)
    ]
    spread_vp = [50.0 * c.slice.median_spread / (c.slice.forward * np.sqrt(c.t) * 0.4 + 1e-9)
                 for c in fit_cands]  # rough vol-pt equivalent of half the median $ spread
    trigger = fit.rho_at_bound or any(r > max(s, 0.25) for r, s in zip(global_rmse, spread_vp))
    if trigger:
        e = fit_essvi([c.slice for c in fit_cands], fit)
        thetas, rhos, fit_method = e.thetas, e.rhos, "essvi"
        eta_f, gamma_f = e.eta, e.gamma
    else:
        eta_f, gamma_f = fit.eta, fit.gamma
    return FitResolution(fitted=True, warm_used=False, fit_obj=fit,
                         thetas=thetas, rhos=rhos, eta=eta_f, gamma=gamma_f,
                         fit_method=fit_method, fit_init=fit_init)


def _iv_history_rows(ts, underlying, fit_cands, thetas, rhos, eta, gamma):
    slice_ts = np.array([c.t for c in fit_cands])
    thetas = np.array(thetas)
    forwards = np.array([c.slice.forward for c in fit_cands])
    rows = ivh.build_iv_history(slice_ts, thetas, rhos, forwards, eta, gamma)
    return [{
        "ts": ts, "underlying": underlying, "tenor": r["tenor"], "atm_iv": r["atm_iv"],
        "rr_25d": r["rr_25d"], "bf_25d": r["bf_25d"], "slope": r["slope"], "fwd": r["fwd"],
    } for r in rows]


def emit_surface_rows(ts, underlying, cands: list, fit_cands: list, res: FitResolution,
                      option_quote_fn, is_eod: bool = True, event_cal=None,
                      macro_cal=None) -> tuple[dict, dict | None]:
    """Assemble the four layers + §06 failure-taxonomy rows + the next snapshot's warm_state from a
    resolved fit. `option_quote_fn(ts, underlying, cand, theta, fit_obj) -> list[dict]` is the only
    pipeline-specific injection (index parity-Black vs single-name de-Am IV). `event_cal` (default None,
    so the index path stays clean) is the single leakage-gated bracketing earnings event for this name;
    `macro_cal` (default None) is the loaded macro-events calendar for the event-vol strip."""
    params, grid, quotes_rows, iv_hist_rows, underlier_state_rows = [], [], [], [], []
    forward_vol_rows, forward_vol_grid_rows, earnings_vol_rows, reprices = [], [], [], []
    vrp_implied_rows, event_vol_rows = [], []
    fit = res.fit_obj
    if res.fitted:
        thetas, rhos, eta_f, gamma_f = res.thetas, res.rhos, res.eta, res.gamma
        arb_ok, bfly_min, cal_min = verify_surface(thetas, rhos, eta_f, gamma_f)
        for i, c in enumerate(fit_cands):
            th = float(thetas[i])
            rh = float(rhos[i])
            ph = float(phi(th, eta_f, gamma_f))
            rmse = reprice_rmse(th, rh, eta_f, gamma_f, c.t, c.slice.k, c.sig_mid)
            reprices.append(rmse)
            params.append({
                "ts": ts, "underlying": underlying, "root": c.root, "expiry": c.expiry,
                "t": c.t, "theta": th, "rho": rh, "phi": ph,
                "eta": float(eta_f), "gamma": float(gamma_f),   # explicit global eSSVI params (P1.4 spec col)
                "forward": c.slice.forward, "discount": c.slice.discount,
                "reprice_rmse": rmse, "arb_ok": bool(arb_ok[i]),
                "dvol_dborrow": sens.dvol_dborrow(th, rh, eta_f, gamma_f, c.t),
                "dvol_ddiv": sens.dvol_ddiv(th, rh, eta_f, gamma_f, c.t),
                "fit_method": res.fit_method, "fit_init": res.fit_init, "fit_status": "ok",
                "median_spread": float(c.slice.median_spread), "n_quotes": int(len(c.slice.k)),
            })
            gd = slice_grid(th, rh, eta_f, gamma_f, c.t, c.slice.k, c.quote_w_norm, dist_t=0.0)
            for j in range(len(gd["k"])):
                grid.append({
                    "ts": ts, "underlying": underlying, "root": c.root, "expiry": c.expiry,
                    "t": c.t, "k": float(gd["k"][j]), "iv": float(gd["iv"][j]),
                    "w": float(gd["w"][j]), "flag": gd["flag"][j],
                    "dist_k": float(gd["dist_k"][j]), "dist_t": float(gd["dist_t"][j]),
                })
            quotes_rows.extend(option_quote_fn(ts, underlying, c, th, fit))
        iv_hist_rows = _iv_history_rows(ts, underlying, fit_cands, thetas, rhos, eta_f, gamma_f)
        # underlier_state (§02): persist the spot + net carry implied from the forward curve, once per
        # snapshot. Was discarded; the derived layer (VRP realized leg, div-contam gating) needs it.
        sc = imply_spot_carry(np.array([c.t for c in fit_cands]),
                              np.array([c.slice.forward for c in fit_cands]),
                              np.array([c.slice.discount for c in fit_cands]))
        underlier_state_rows = [{
            "ts": ts, "underlying": underlying, "spot": sc.spot, "carry": sc.carry,
            "spot_status": sc.status, "n_expiries": sc.n_expiries,
            "resid_max": sc.resid_max, "resid_known": sc.resid_known,
        }]
        # forward-vol (§3A, research-upgraded): forward smile between adjacent CM tenors, closed-form
        # forward skew/curv, double arb gate. forward_vol_grid is EOD-gated by the orchestrators.
        forward_vol_rows, forward_vol_grid_rows = der.build_forward_vol(
            ts, underlying, np.array([c.t for c in fit_cands]), thetas, rhos, eta_f, gamma_f,
            is_eod=is_eod)
        # VRP implied leg (§3B): 3-strike model-free variance per CM tenor (index + single-name)
        vrp_implied_rows = der.build_vrp_implied(
            ts, underlying, np.array([c.t for c in fit_cands]), thetas, rhos, eta_f, gamma_f)
        # earnings-vol (§3D): single-name DJ earnings strip; emitted ONLY when the caller passes a
        # leakage-gated bracketing event (index path passes event_cal=None -> stays clean).
        if event_cal is not None:
            earnings_vol_rows = der.build_earnings_vol(
                ts, underlying, np.array([c.t for c in fit_cands]), thetas, rhos, eta_f, gamma_f,
                [c.expiry for c in fit_cands], reprices,
                underlier_state_rows[0] if underlier_state_rows else None, event_cal, macro_cal=macro_cal)
        # event-vol (§3C): general multi-event DJ variance strip; emits only when a calendar is supplied
        # (index passes macro_cal, single-name passes both) -> the no-calendar/regress path stays clean.
        if macro_cal is not None or event_cal is not None:
            event_vol_rows = der.build_event_vol(
                ts, underlying, np.array([c.t for c in fit_cands]), thetas, rhos, eta_f, gamma_f,
                [c.expiry for c in fit_cands], reprices,
                underlier_state_rows[0] if underlier_state_rows else None,
                macro_cal=macro_cal, earnings_event=event_cal,
                scope=("single_name" if event_cal is not None else "index"))
        # constant-maturity grid nodes (§05): once per snapshot at the underlier level, distinguishable
        # from listed-expiry nodes by dist_t > 0 and a synthetic constant-maturity expiry date.
        for cm in cm_tenor_grid([c.t for c in fit_cands], thetas, rhos, eta_f, gamma_f,
                                [c.slice.k for c in fit_cands]):
            grid.append({
                "ts": ts, "underlying": underlying, "root": underlying,
                "expiry": ts.date() + dt.timedelta(days=cm["tenor"]), "t": cm["t_cm"],
                "k": cm["k"], "iv": cm["iv"], "w": cm["w"], "flag": cm["flag"],
                "dist_k": cm["dist_k"], "dist_t": cm["dist_t"],
            })

    # failure-taxonomy rows: every non-ok slice gets a params status row (null fit fields)
    for c in cands:
        if c.status != "ok" or fit is None:
            if any(p["expiry"] == c.expiry and p["fit_status"] == "ok" for p in params):
                continue
            params.append({
                "ts": ts, "underlying": underlying, "root": c.root, "expiry": c.expiry,
                "t": c.t, "theta": None, "rho": None, "phi": None, "forward": None, "discount": None,
                "reprice_rmse": None, "arb_ok": None, "dvol_dborrow": None, "dvol_ddiv": None,
                "fit_method": None, "fit_init": None, "fit_status": c.status if c.status != "ok" else "failed",
                "median_spread": None, "n_quotes": (int(c.n_two_sided) or None),   # §06 observed count where known
            })

    new_warm = None
    if res.fitted and fit_cands:
        ok_rmses = [p["reprice_rmse"] for p in params
                    if p["fit_status"] == "ok" and p["reprice_rmse"] is not None]
        new_warm = {
            "eta": float(res.eta), "gamma": float(res.gamma),
            "rho_med": float(np.median(res.rhos)),
            "prev_reprice_med": float(np.median(ok_rmses)) if ok_rmses else None,
            "seeds": {(c.root, c.expiry.isoformat()): (float(res.thetas[i]), float(res.rhos[i]))
                      for i, c in enumerate(fit_cands)},
        }
    return ({"vol_surface_params": params, "vol_surface_grid": grid,
             "iv_history": iv_hist_rows, "option_quotes": quotes_rows,
             "underlier_state": underlier_state_rows,
             "forward_vol": forward_vol_rows, "forward_vol_grid": forward_vol_grid_rows,
             "earnings_vol": earnings_vol_rows, "vrp_implied": vrp_implied_rows,
             "event_vol": event_vol_rows}, new_warm)
