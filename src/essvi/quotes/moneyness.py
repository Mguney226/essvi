"""Time-to-expiry (settlement-aware) and log-forward moneyness."""
from __future__ import annotations

import datetime as dt
import zoneinfo

import numpy as np

from ..constants import DAYCOUNT, SETTLE_AM_ET, SETTLE_PM_ET

_ET = zoneinfo.ZoneInfo("America/New_York")
_UTC = zoneinfo.ZoneInfo("UTC")

# roots that settle AM (on the open, SET) vs PM (close)
_AM_ROOTS = {"SPX", "NDX"}
_PM_ROOTS = {"SPXW", "NDXP", "QQQ"}


def settle_kind(root: str) -> str:
    return "am" if root in _AM_ROOTS else "pm"


def parse_expiry(yyMMdd: str) -> dt.date:
    s = str(yyMMdd)
    return dt.date(2000 + int(s[:2]), int(s[2:4]), int(s[4:6]))


def snapshot_datetime(minute_ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(minute_ns / 1e9, tz=_UTC)


def year_fraction(expiry: dt.date, root: str, snap_dt: dt.datetime) -> float:
    """Act/365F including the intraday fraction, using the expiry's settlement time."""
    h, m = SETTLE_AM_ET if settle_kind(root) == "am" else SETTLE_PM_ET
    settle_dt = dt.datetime(expiry.year, expiry.month, expiry.day, h, m, tzinfo=_ET).astimezone(_UTC)
    secs = (settle_dt - snap_dt).total_seconds()
    return secs / (DAYCOUNT * 86400.0)


def log_moneyness(strike: np.ndarray, forward: float) -> np.ndarray:
    return np.log(np.asarray(strike, dtype=float) / forward)
