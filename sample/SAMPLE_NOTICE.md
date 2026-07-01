# Sample data notice

This directory is a **small illustrative extract** of the engine's fitted output, included so the
repository is runnable end-to-end (`python sample/load_example.py`). It is **not** the full dataset.

Contents:
- `spx_2026-01-22/` - one SPX trading day, all output layers, so you can see the full schema and
  reconstruct any strike's implied vol from the fitted SSVI parameters.
- `vix_series.parquet` - the engine's 30-day model-free variance (`k_var`) across ~284 trading days,
  so the VIX reproduction is verifiable over a real history (not one cherry-picked day).
- `vix_cboe.csv` - Cboe's publicly published VIX history (their free daily CSV), used only as the
  independent benchmark for the reproduction check.

The underlying option quotes are derived from licensed OPRA market data; only this minimal extract is
included, for demonstration.
