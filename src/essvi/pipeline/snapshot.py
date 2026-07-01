"""Process one snapshot (SPX∪SPXW union) into the four output layers, to methodology spec.

Failure taxonomy (§06) is enforced structurally: every (root, expiry) is seeded with a status and
only upgraded to `ok` on success; a converged-but-arb-failing slice ships AS FITTED with arb_ok=false.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.special import ndtr

from ..constants import KFIT_MAX, MAX_REL_SPREAD, MIN_STRIKES, NOISE_CAP_VP, PRICE_FLOOR
from ..forward.parity import imply_forward_discount
from ..iv import black
from ..quotes.moneyness import parse_expiry, settle_kind, snapshot_datetime, year_fraction
from ..quotes.reduce import reduce_snapshot
from ..ssvi.objective import Slice
from .fit_core import SliceCand, emit_surface_rows, resolve_fit  # shared fit/emit (also single-name)


def _build_slice(root, expiry, t, g: pl.DataFrame) -> SliceCand:
    cand = SliceCand(root=root, expiry=expiry, t=t)
    # pivot to find two-sided strikes (both C and P)
    calls = g.filter(pl.col("option_type") == "C")
    puts = g.filter(pl.col("option_type") == "P")
    cmap = dict(zip(calls["strike"].to_list(), zip(calls["mid_c"].to_list(), calls["spread_c"].to_list())))
    pmap = dict(zip(puts["strike"].to_list(), zip(puts["mid_c"].to_list(), puts["spread_c"].to_list())))
    two_sided = sorted(set(cmap) & set(pmap))
    cand.n_two_sided = len(two_sided)
    if len(two_sided) < MIN_STRIKES:
        cand.status = "insufficient_quotes"
        return cand

    K = np.array(two_sided, dtype=float)
    cmp_ = np.array([cmap[k][0] - pmap[k][0] for k in two_sided])      # C - P
    wts = np.array([1.0 / max(cmap[k][1] + pmap[k][1], 1e-6) ** 2 for k in two_sided])
    fr = imply_forward_discount(K, cmp_, wts, d_seed=0.99)
    if fr.status != "ok":
        cand.status = "no_forward"
        return cand
    F, D = fr.forward, fr.discount

    # OTM-side quotes -> Black IV; keep all surviving two-sided contracts for option_quotes
    rows = g.with_columns(
        np.log(g["strike"].to_numpy() / F).alias("k") if False else (pl.col("strike").log() - np.log(F)).alias("k")
    )
    ks = rows["k"].to_numpy()
    mids = rows["mid_c"].to_numpy()
    sprs = rows["spread_c"].to_numpy()
    bsz = rows["bid_size"].to_numpy()
    asz = rows["ask_size"].to_numpy()
    is_c = (rows["option_type"] == "C").to_numpy()
    is_otm = ((ks >= 0) & is_c) | ((ks < 0) & ~is_c)
    pnorm = mids / (D * F)
    w_obs = black.implied_total_variance(ks, pnorm, is_c)
    # §01 liquidity filter: drop deep-OTM noise (price floor, rel-spread cap, two-sided size, k cap)
    liquid = (
        (mids >= PRICE_FLOOR)
        & (sprs / np.maximum(mids, 1e-9) <= MAX_REL_SPREAD)
        & (bsz > 0) & (asz > 0)
        & (np.abs(ks) <= KFIT_MAX)
    )
    good = is_otm & liquid & np.isfinite(w_obs) & (w_obs > 0)
    # vol-noise cap: half-spread in Black vol points via vega must be <= NOISE_CAP_VP
    vega_price = D * F * black.norm_vega_sw(ks, np.where(good, w_obs, 1.0)) * np.sqrt(t)
    noise_vp = 100.0 * (sprs / 2.0) / np.maximum(vega_price, 1e-9)
    good = good & (noise_vp <= NOISE_CAP_VP)
    if good.sum() < MIN_STRIKES:
        cand.status = "insufficient_quotes"
        cand.quotes_df = rows
        return cand

    kk = ks[good]
    ww = w_obs[good]
    iv = np.sqrt(ww / t)
    qw = rows["quote_weight"].to_numpy()[good]
    spr = rows["spread_c"].to_numpy()[good]
    mid_g = mids[good]
    vega = black.norm_vega_sw(kk, ww)
    omega = np.maximum(vega, 1e-8) * np.minimum(mid_g / np.maximum(spr, 1e-6), 50.0) * qw

    order = np.argsort(kk)
    kk, ww, iv, qw, omega = kk[order], ww[order], iv[order], qw[order], omega[order]
    theta_atm = float(np.interp(0.0, kk, ww))
    median_spread = float(np.median(spr))

    cand.slice = Slice(
        expiry=expiry.isoformat(), t=t, k=kk, w_obs=ww, omega=omega,
        forward=F, discount=D, theta_atm=theta_atm, median_spread=median_spread,
        settle=settle_kind(root),
    )
    cand.sig_mid = iv
    cand.quote_w_norm = qw / max(qw.max(), 1e-9)
    cand.quotes_df = rows
    return cand


def process_snapshot(df: pl.DataFrame, underlying: str, minute_ns: int,
                     warm_state: dict | None = None, is_eod: bool = True,
                     macro_cal=None) -> tuple[dict, dict | None]:
    """Return ({layer: list[dict]}, warm_state for the next snapshot).

    warm_state: {"eta", "gamma", "seeds": {(root, expiry_iso): (theta, rho)}}.
    When present and covering enough of the chain, the warm path seeds every slice from the previous
    snapshot and skips the cold cascade; any failure falls back to cold. fit_init ships per row."""
    snap_dt = snapshot_datetime(minute_ns)
    ts = snap_dt
    df = reduce_snapshot(df)

    cands: list[SliceCand] = []
    for (root, exp_str), g in df.group_by(["root", "expiry_yyMMdd"]):
        expiry = parse_expiry(exp_str)
        t = year_fraction(expiry, root, snap_dt)
        if t <= 1.0 / 365.0 / 24:                       # expired / sub-hour 0DTE
            cands.append(SliceCand(root=root, expiry=expiry, t=max(t, 0.0), status="below_min_tenor"))
            continue
        cands.append(_build_slice(root, expiry, t, g))

    fit_cands = [c for c in cands if c.status == "ok" and c.slice is not None]
    fit_cands.sort(key=lambda c: c.t)

    res = resolve_fit(cands, fit_cands, warm_state)
    return emit_surface_rows(ts, underlying, cands, fit_cands, res, _option_quote_rows, is_eod=is_eod,
                             macro_cal=macro_cal)


def _option_quote_rows(ts, underlying, c: SliceCand, theta, fit):
    """Surviving two-sided quotes for this slice, with iv + Black greeks."""
    df = c.quotes_df
    if df is None:
        return []
    F, D, t = c.slice.forward, c.slice.discount, c.t
    out = []
    ks = df["k"].to_numpy()
    strikes = df["strike"].to_numpy()
    is_c = (df["option_type"] == "C").to_numpy()
    mids = df["mid_c"].to_numpy()
    pnorm = mids / (D * F)
    w = black.implied_total_variance(ks, pnorm, is_c)
    iv = np.sqrt(np.where(np.isfinite(w) & (w > 0), w / t, np.nan))
    sw = np.sqrt(np.maximum(w, 1e-12))
    d1 = (-ks + 0.5 * w) / sw
    d2 = d1 - sw
    delta = np.where(is_c, D * ndtr(d1), D * (ndtr(d1) - 1.0))
    npdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
    vega = D * F * npdf * np.sqrt(t)
    gamma = np.where(iv > 0, D * npdf / (F * iv * np.sqrt(t)), np.nan)
    theta_g = -np.where(iv > 0, D * F * npdf * iv / (2 * np.sqrt(t)), np.nan)
    bid = df["bid_price"].to_numpy(); ask = df["ask_price"].to_numpy()
    bsz = df["bid_size"].to_numpy(); asz = df["ask_size"].to_numpy()
    qw = df["quote_weight"].to_numpy()
    for i in range(len(ks)):
        out.append({
            "ts": ts, "underlying": underlying, "root": c.root, "expiry": c.expiry,
            "strike": float(strikes[i]), "right": "C" if is_c[i] else "P",
            "bid": float(bid[i]), "ask": float(ask[i]), "mid": float(mids[i]),
            "bid_sz": int(bsz[i]), "ask_sz": int(asz[i]),
            "iv": float(iv[i]) if np.isfinite(iv[i]) else None,
            "delta": float(delta[i]) if np.isfinite(delta[i]) else None,
            "gamma": float(gamma[i]) if np.isfinite(gamma[i]) else None,
            "vega": float(vega[i]) if np.isfinite(vega[i]) else None,
            "theta": float(theta_g[i]) if np.isfinite(theta_g[i]) else None,
            "borrow": 0.0, "quote_weight": float(qw[i]),
            "flag": "observed",
        })
    return out
