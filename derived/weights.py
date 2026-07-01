"""weights.py -- self-computed, survivorship-bias-free, point-in-time index weights for P5 dispersion.

w_i(t) = P_i(t) * S_i(t) / sum_k P_k(t) S_k(t)   (float-cap market-cap weights, by the date-t EOD de-Am close --
this is an EOD panel, so date-t's close IS the weighting price; the CBOE intraday prior-close-freeze does not
apply). P_i = the de-Am parity-implied SPOT (underlier_state, NO traded underlier -- disclosed); S_i = SEC common
shares outstanding, POINT-IN-TIME = the most-recent value FILED on or before t (no look-ahead -- a 10-Q's shares
become known only at its filing date, ~6 weeks after period-end). IWF (float factor) ~ 1 is the disclosed
approximation: exact for ~95% of names, but materially OVER-weights names whose controlling block sits OUTSIDE
the counted public class -- WMT ~+45-80% (Walton family ~45% held -> IWF~0.55, so using full shares over-weights
~1/0.55), META ~+15% (Zuckerberg super-voting Class B). NOTE
GOOGL/GOOG are ~unaffected (the founder Class B is NOT in the two public classes counted). BRK.B uses the
Class-B count (~1.4B), which ~approximates the S&P FLOAT -- Berkshire's Class A (full economic B-equivalent =
ClassB + ClassA*1500 ~2.2B) is largely insider-held (Buffett) hence excluded from float; counting it would
OVER-weight BRK. Membership is the PIT
S&P 500 constituent set as-of t (fja05680, no survivorship bias).

All functions are pure (DataFrames in/out); the orchestrator (run_dispersion) supplies the de-Am spots.
"""
from __future__ import annotations

import datetime as _dt

import polars as pl


def members_asof(membership: pl.DataFrame, dates: list) -> pl.DataFrame:
    """Expand the change-dated PIT membership to a (date, ticker) panel for every `dates` entry: the constituent
    set on date t is the membership snapshot of the most-recent change-date <= t (as-of, no look-ahead)."""
    chg = membership.select("date").unique().sort("date")
    dd = pl.DataFrame({"date": sorted(dates)}).with_columns(pl.col("date").cast(chg["date"].dtype))
    asof = dd.join_asof(chg.with_columns(pl.col("date").alias("chg")), on="date", strategy="backward")
    out = (asof.drop_nulls("chg").join(membership.rename({"date": "chg"}), on="chg")
           .select("date", "ticker"))
    return out


def pit_shares(shares: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Attach POINT-IN-TIME shares to a (date, ticker) panel: for each row, the shares whose `filed` date is the
    most-recent <= date (per ticker). `shares` cols: ticker, end, shares, filed. Uses a per-ticker as-of join."""
    # sort by (ticker, filed_d, end): on a same-FILED tie (a filing restating multiple periods), the most-recent
    # `end` must win -- a bare ['ticker','filed_d'] sort leaves the tie to input row-order (a latent stale-share
    # bug: a re-fetch in a different order could pick the prior-year count).
    sh = (shares.with_columns([pl.col("filed").cast(pl.Date, strict=False).alias("filed_d"),
                               pl.col("end").cast(pl.Date, strict=False).alias("_end_d")])
          .drop_nulls("filed_d").sort(["ticker", "filed_d", "_end_d"]))
    p = panel.sort(["ticker", "date"])
    out = p.join_asof(sh.select("ticker", "filed_d", "_end_d", "shares"), left_on="date", right_on="filed_d",
                      by="ticker", strategy="backward")
    return out                                              # adds `shares` + `_end_d` (filing period-end; null if none)


_SPLIT_RATIOS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
_SPLIT_LO, _SPLIT_HI = 0.55, 1.8                            # a filing-to-filing shares ratio outside this = a split


def _snap_split(inv_ratio: float) -> float:
    """Snap a raw share-multiplier (1/spot_ratio, or the SEC shares jump) to the nearest clean split ratio."""
    cands = _SPLIT_RATIOS + [1.0 / x for x in _SPLIT_RATIOS]
    return min(cands, key=lambda c: abs(c - inv_ratio))


def split_adjust(panel: pl.DataFrame) -> pl.DataFrame:
    """Split-adjust as-filed PIT shares to the (split-adjusted) de-Am price basis. The bug: between a split's
    ex-date and its FIRST post-split 10-Q, the de-Am spot is post-split but the as-filed share count is still
    pre-split, so the megacap is under-weighted ~1/f. Detection uses the SEC SHARES JUMP at the filing boundary
    (the clean factor f, robust even when the split ex-date falls inside a ~6-month data gap where the spot shows
    no jump -- the NVDA/AVGO June-2024 case); the in-island ex-date is the de-Am SPOT jump if present, else (gap
    case) the whole pre-split-filing data segment is post-ex (data resumes after the gap). No look-ahead: the
    factor is known only from the post-split filing, and is applied to PRIOR-segment dates that are already in
    the panel (all <= the boundary). `panel` needs date, ticker, shares, _end_d, spot."""
    rows = panel.sort(["ticker", "date"]).to_dicts()
    by_tkr: dict = {}
    for i, r in enumerate(rows):
        by_tkr.setdefault(r["ticker"], []).append(i)
    for tkr, idxs in by_tkr.items():
        # WITHIN-ISLAND price-split ex-dates from the de-Am SPOT (precise; share factor = 1/spot_ratio).
        spot_ex = []
        for k in range(1, len(idxs)):
            p, q = rows[idxs[k]].get("spot"), rows[idxs[k - 1]].get("spot")
            g = (rows[idxs[k]]["date"] - rows[idxs[k - 1]]["date"]).days
            if p and q and q > 0 and g <= 7 and not (_SPLIT_LO <= p / q <= _SPLIT_HI):
                spot_ex.append((rows[idxs[k]]["date"], _snap_split(q / p)))
        # SHARES-jump boundaries (segments by filing period-end _end_d): a split shows here as a clean-ratio jump.
        segs = []; s0 = 0
        for k in range(1, len(idxs)):
            if rows[idxs[k]].get("_end_d") != rows[idxs[k - 1]].get("_end_d"):
                segs.append((s0, k)); s0 = k
        segs.append((s0, len(idxs)))
        splits = []                                        # (ex_date, factor f, pre_count, post_count)
        for si in range(1, len(segs)):
            a0, a1 = segs[si - 1]; b0, b1 = segs[si]
            sa, sb = rows[idxs[a0]]["shares"], rows[idxs[b0]]["shares"]
            if not sa or not sb or sa <= 0:
                continue
            ratio = sb / sa
            if _SPLIT_LO <= ratio <= _SPLIT_HI:
                continue                                   # not a split (normal buyback/issuance)
            f = _snap_split(ratio); jdate = rows[idxs[b0]]["date"]
            # PRICE-split ex-date: prefer the matching adjacent SPOT-ex within ~200d; else find it from the spot
            # LEVEL drop inside the prior segment (the first date where spot has fallen to ~the pre-level / f --
            # handles a split whose ex has no adjacent ratio because the pre-split spot is null/sparse, e.g. TSCO);
            # else (no level drop -> the whole prior segment is already post-split, e.g. gap-hidden NVDA) the ex
            # precedes the prior data.
            e = next((ed for ed, ef in spot_ex if abs((ed - jdate).days) <= 200 and abs(ef - f) / f < 0.30), None)
            if e is None:
                e = rows[idxs[a0]]["date"]
                pri = [rows[idxs[k]].get("spot") for k in range(a0, a1) if rows[idxs[k]].get("spot")]
                hi = max(pri) if pri else None
                if hi and f > 1:
                    for k in range(a0, a1):
                        s = rows[idxs[k]].get("spot")
                        if s and s <= 1.3 * hi / f:          # spot dropped to the post-split basis here
                            e = rows[idxs[k]]["date"]; break
            splits.append((e, f, sa, sb))
        if not splits:
            continue
        # each date's shares must be in its PRICE basis: post (>= ex) or pre (< ex). Fix where the as-filed count
        # is in the WRONG basis: a pre-count on/after the ex (late filing) -> x f; a post-count before the ex
        # (early filing) -> / f. (Compares the ORIGINAL as-filed value to the split's pre/post counts.)
        for k in idxs:
            r = rows[k]; sh = r["shares"]
            if not sh:
                continue
            for e, f, pre, post in splits:
                if r["date"] >= e and abs(sh / pre - 1.0) < 0.20:
                    r["shares"] = sh * f
                elif r["date"] < e and post > 0 and abs(sh / post - 1.0) < 0.20:
                    r["shares"] = sh / f
    return pl.DataFrame(rows, schema=panel.schema)


def compute_weights(panel_with_shares_spots: pl.DataFrame) -> pl.DataFrame:
    """Given a (date, ticker, shares, spot) panel, compute w_i = P_i S_i / sum per date over the rows that have
    BOTH a positive spot and positive shares; rows missing either get weight=null (their mass is the missing
    fraction, reported by the orchestrator). Returns the panel with `mktcap` + `weight` columns."""
    df = panel_with_shares_spots.with_columns(
        pl.when((pl.col("spot") > 0) & (pl.col("shares") > 0))
        .then(pl.col("spot") * pl.col("shares")).otherwise(None).alias("mktcap"))
    tot = df.group_by("date").agg(pl.col("mktcap").sum().alias("mktcap_sum"))
    return (df.join(tot, on="date")
            .with_columns((pl.col("mktcap") / pl.col("mktcap_sum")).alias("weight"))
            .drop("mktcap_sum"))
