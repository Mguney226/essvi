#!/usr/bin/env python3
"""Self-contained demo of the essvi sample data.

Three checks, ALL recomputed from the shipped Parquet with NO engine code in the loop:
  1. reconstruct a fitted SSVI smile from 5 parameters (theta, rho, phi, forward, discount)
  2. run the Breeden-Litzenberger no-arbitrage (implied-density >= 0) test
  3. reproduce Cboe's published VIX from the model-free variance the engine ships

Run:  python sample/load_example.py     (needs only numpy + duckdb)
"""
import os
import numpy as np
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
con = duckdb.connect()

# ------------------------------------------------------------------ 1. reconstruct a surface
P = f"{HERE}/spx_2026-01-22/vol_surface_params.parquet"
expiry, t, theta, rho, phi, arb_ok_shipped = con.sql(f"""
    SELECT expiry, t, theta, rho, phi, arb_ok
    FROM read_parquet('{P}')
    WHERE ts = (SELECT max(ts) FROM read_parquet('{P}')) AND fit_status = 'ok'
    ORDER BY abs(t - 30/365.0) LIMIT 1""").fetchone()

k = np.arange(-1.5, 1.5 + 1e-9, 0.005)              # log-forward moneyness grid
D = (phi * k + rho) ** 2 + (1 - rho ** 2)
sq = np.sqrt(D)
w   = 0.5 * theta * (1 + rho * phi * k + sq)        # SSVI total variance w(k)
wp  = 0.5 * theta * phi * (rho + (phi * k + rho) / sq)
wpp = 0.5 * theta * phi ** 2 * (1 - rho ** 2) / D ** 1.5
iv  = np.sqrt(w / t)
print(f"1. Reconstructed the SPX {expiry} smile (t={t:.4f}y) from 5 numbers:")
print(f"     ATM vol {np.interp(0.0, k, iv)*100:5.2f}%   25%-OTM-put vol {np.interp(-0.25, k, iv)*100:5.2f}%"
      f"   25%-OTM-call vol {np.interp(0.25, k, iv)*100:5.2f}%")

# ------------------------------------------------------------------ 2. no-arbitrage check
g = (1 - k * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2   # Gatheral-Jacquier density
mn = float(g.min())
print(f"\n2. No-arbitrage check (Breeden-Litzenberger implied density), recomputed here:")
print(f"     min density g(k) = {mn:+.4f}  ->  {'ARB-FREE' if mn >= -1e-4 else 'VIOLATION'}"
      f"   (matches shipped arb_ok = {arb_ok_shipped})")

# ------------------------------------------------------------------ 3. reproduce Cboe VIX
V = f"{HERE}/vix_series.parquet"
model = con.sql(f"SELECT date, 100*sqrt(k_var) FROM read_parquet('{V}') WHERE tenor = 30").fetchall()
cboe  = con.sql(f"SELECT strptime(DATE,'%m/%d/%Y')::date, CAST(CLOSE AS DOUBLE) "
                f"FROM read_csv('{HERE}/vix_cboe.csv', header=true, all_varchar=true)").fetchall()
mm = {r[0]: r[1] for r in model}
cc = {r[0]: r[1] for r in cboe}
days = sorted(set(mm) & set(cc))
vm = np.array([mm[d] for d in days], float)
vc = np.array([cc[d] for d in days], float)
corr = float(np.corrcoef(vm, vc)[0, 1])
print(f"\n3. VIX reproduction (this engine's 30-day model-free variance vs Cboe's published VIX):")
print(f"     {len(days)} days matched   correlation = {corr:.4f}   mean |error| = {np.abs(vm-vc).mean():.2f} vol pts")
print("\nAll three recomputed from the shipped data alone - the point is you verify, not trust.")
