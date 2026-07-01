"""Reduce a raw snapshot (many NBBO updates per contract within the minute) to one quote per
contract, with two-sided filtering and the §01 staleness/width down-weight (quote_weight).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def reduce_snapshot(df: pl.DataFrame, *, max_spread_frac: float = 0.20) -> pl.DataFrame:
    """Input columns: root, expiry_yyMMdd, option_type, strike, sip_timestamp, minute_ns,
    bid_price, ask_price, bid_size, ask_size, mid, spread. Output: one row per
    (root, expiry, strike, option_type) = last valid NBBO, with quote_weight in (0,1]."""
    # last update per contract within the minute
    df = (
        df.sort("sip_timestamp")
        .group_by(["root", "expiry_yyMMdd", "option_type", "strike"], maintain_order=True)
        .last()
    )
    # two-sided, sane
    df = df.filter(
        (pl.col("bid_price") > 0)
        & (pl.col("ask_price") > 0)
        & (pl.col("ask_price") >= pl.col("bid_price"))
    )
    df = df.with_columns(
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_c"),
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread_c"),
    )
    df = df.filter(pl.col("mid_c") > 0)
    # quote_weight: width down-weight (mid/spread relative to a max-width threshold), clipped (0,1]
    df = df.with_columns(
        pl.min_horizontal(
            pl.lit(1.0),
            (max_spread_frac * pl.col("mid_c") / pl.col("spread_c")) ** 2,
        ).clip(1e-3, 1.0).alias("quote_weight")
    )
    return df
