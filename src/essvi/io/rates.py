"""FRED rate curve -> continuously-compounded discount factor seed.

Reads the local FRED CSVs (date,value in percent), builds a per-day zero curve by log-linear
interpolation of the Treasury bill/note tenors, and returns D(t) = exp(-r(t) * t).
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import polars as pl

# Local FRED CSV directory (date,value in percent). Override via the VSE_FRED_DIR
# env var; when the files are absent, callers fall back to a flat curve.
FRED_DIR = Path(os.environ.get("VSE_FRED_DIR", "data/fred/interest_rates"))

# (FRED series file, tenor in years)
_CURVE = [
    ("DTB4WK.csv", 4 / 52),
    ("DTB3.csv", 0.25),
    ("DTB6.csv", 0.5),
    ("DGS1.csv", 1.0),
    ("DGS2.csv", 2.0),
    ("DGS5.csv", 5.0),
    ("DGS10.csv", 10.0),
    ("DGS20.csv", 20.0),
    ("DGS30.csv", 30.0),
]


class DiscountCurve:
    def __init__(self, asof: dt.date, tenors: np.ndarray, rates: np.ndarray):
        self.asof = asof
        self.tenors = tenors          # years
        self.rates = rates            # continuously-compounded zero rates (decimal)

    def zero_rate(self, t: float) -> float:
        t = max(t, 1e-6)
        return float(np.interp(t, self.tenors, self.rates))

    def discount(self, t: float) -> float:
        return float(np.exp(-self.zero_rate(t) * max(t, 0.0)))


def _value_asof(path: Path, asof: dt.date) -> float | None:
    if not path.exists():
        return None
    df = pl.read_csv(path, try_parse_dates=True)
    cols = df.columns
    dcol, vcol = cols[0], cols[1]
    df = df.filter(pl.col(vcol).is_not_null())
    # most recent value on/before asof
    sub = df.filter(pl.col(dcol) <= asof)
    if sub.height == 0:
        return None
    row = sub.sort(dcol).tail(1)
    try:
        return float(row[vcol][0])
    except (TypeError, ValueError):
        return None


def load_curve(asof: dt.date, fred_dir: Path = FRED_DIR) -> DiscountCurve:
    tenors, rates = [], []
    for fname, ty in _CURVE:
        v = _value_asof(fred_dir / fname, asof)
        if v is not None and np.isfinite(v):
            tenors.append(ty)
            rates.append(v / 100.0)            # percent -> decimal; treat as ~cont-comp (close enough)
    if not tenors:
        # fallback flat 4%
        tenors, rates = [0.25, 30.0], [0.04, 0.04]
    tenors = np.array(tenors)
    rates = np.array(rates)
    order = np.argsort(tenors)
    return DiscountCurve(asof, tenors[order], rates[order])
