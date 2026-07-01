"""Tests for the PIT index-weight builder (derived/weights.py): survivorship-free membership as-of, no-look-ahead
PIT shares, and the float-cap market-cap weight identity."""
import datetime as dt
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "derived"))
import weights as W  # noqa: E402


def test_members_asof_survivorship_free():
    # BBB is replaced by CCC at the 2024-06-01 reconstitution; the as-of set must reflect the date
    mem = pl.DataFrame({"date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1), dt.date(2024, 6, 1), dt.date(2024, 6, 1)],
                        "ticker": ["AAA", "BBB", "AAA", "CCC"]})
    p = W.members_asof(mem, [dt.date(2024, 3, 1), dt.date(2024, 7, 1)])
    assert set(p.filter(pl.col("date") == dt.date(2024, 3, 1))["ticker"]) == {"AAA", "BBB"}
    assert set(p.filter(pl.col("date") == dt.date(2024, 7, 1))["ticker"]) == {"AAA", "CCC"}


def test_pit_shares_no_lookahead():
    sh = pl.DataFrame({"ticker": ["AAA", "AAA"], "end": ["2024-03-31", "2024-06-30"],
                       "shares": [100.0, 110.0], "filed": ["2024-05-01", "2024-08-01"]})
    panel = pl.DataFrame({"date": [dt.date(2024, 6, 1), dt.date(2024, 9, 1)], "ticker": ["AAA", "AAA"]})
    out = W.pit_shares(sh, panel)
    g = {r["date"]: r["shares"] for r in out.iter_rows(named=True)}
    assert g[dt.date(2024, 6, 1)] == 100.0                  # the 110 (filed 8/1) is future -> not used
    assert g[dt.date(2024, 9, 1)] == 110.0


def test_compute_weights_mktcap_identity():
    p = pl.DataFrame({"date": [dt.date(2024, 1, 1)] * 3, "ticker": ["A", "B", "C"],
                      "shares": [100.0, 200.0, 300.0], "spot": [10.0, 5.0, 2.0]})  # mktcap 1000,1000,600
    w = {r["ticker"]: r["weight"] for r in W.compute_weights(p).iter_rows(named=True)}
    assert abs(w["A"] - 1000 / 2600) < 1e-9 and abs(w["C"] - 600 / 2600) < 1e-9


def test_split_adjust_gap_hidden_split():
    # NVDA-like 10:1 split whose ex-date falls in a ~6-month data gap: Q1 filing (end 2024-04-28) shares 2.46e9
    # PRE-split; Q2 filing (end 2024-07-28) shares 24.5e9 POST-split; the only data dates (Aug) use Q1 shares but
    # a post-split spot. split_adjust must x10 the Q1-segment shares (detected from the SEC shares jump, no spot jump).
    panel = pl.DataFrame({"date": [dt.date(2024, 8, 15), dt.date(2024, 8, 29)], "ticker": ["X", "X"],
                          "shares": [2.46e9, 24.5e9], "_end_d": [dt.date(2024, 4, 28), dt.date(2024, 7, 28)],
                          "spot": [110.0, 112.0]})
    out = W.split_adjust(panel).sort("date")
    assert abs(out["shares"][0] - 24.6e9) < 1e6        # Aug-15 split-adjusted x10 (continuous cap)
    assert out["shares"][1] == 24.5e9                  # Aug-29 (post-split filing) unchanged


def test_split_adjust_noop_on_buyback():
    # a normal ~3% buyback between filings must NOT be treated as a split
    panel = pl.DataFrame({"date": [dt.date(2024, 8, 15), dt.date(2024, 11, 15)], "ticker": ["Y", "Y"],
                          "shares": [1.00e9, 0.97e9], "_end_d": [dt.date(2024, 6, 30), dt.date(2024, 9, 30)],
                          "spot": [50.0, 52.0]})
    out = W.split_adjust(panel).sort("date")
    assert out["shares"][0] == 1.00e9 and out["shares"][1] == 0.97e9   # untouched


def test_compute_weights_drops_missing():
    p = pl.DataFrame({"date": [dt.date(2024, 1, 1)] * 2, "ticker": ["A", "B"],
                      "shares": [100.0, None], "spot": [10.0, 5.0]})
    w = W.compute_weights(p)
    assert w.filter(pl.col("ticker") == "B")["weight"][0] is None     # missing shares -> null weight
