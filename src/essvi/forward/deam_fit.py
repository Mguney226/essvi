"""§02/§03 joint fixed point: one American single-name chain -> de-Americanized European IVs.

Turns an American name's option chain (multiple expiries, one snapshot) into per-contract
European-equivalent Black implied vols, ready for the eSSVI fit, with no external spot/dividend
feed. The forward curve pins spot + carry (term_structure.imply_spot_carry); the de-Am engine
(deam.deam_iv) strips early-exercise premium; we iterate forward <-> de-Am to a fixed point.

Pricer routing (validated empirically, see notes):
  - OTM CALL with q <= 0 (non-positive net carry): American == European EXACTLY (Merton) -> direct Black.
  - OTM PUT, or CALL with q > 0 (dividend-driven), or long-dated T>=1y: real early-exercise
    premium -> the binomial tree (deam.deam_iv). Black is wrong here and so is BS2002.

Single-name de-Am uses the parity-implied CONTINUOUS net carry q only (no discrete ex-date schedule);
the discrete-dividend escrow path in deam.py is validated but dormant in production. The tree inverter's
true accuracy ceiling at the production depth DEAM_N is ~1 bp (validate_deam.py [R], independent reference).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from ..iv import black
from ..iv import deam
from .term_structure import imply_spot_carry

# Convergence is geometric (ratio ~0.37/iter under DAMP). The fixed point's own floor is the tree
# pricing error (~0.01 bp in IV -> ~1 bp in F), so a sub-0.1-bp forward tolerance is meaningless;
# 1e-5 relative (0.1 bp) pins the forward well below any real quote noise.
TOL_F = 1e-5
TOL_CARRY = 1e-4
MAX_ITERS = int(os.environ.get("DEAM_MAXITERS", "5"))   # forward converges in ~4 (geometric, ratio
                                                        # ~0.37); 10 was overkill, reprice unchanged
DAMP = 0.7
TREE_TMIN_LONG = 1.0
DEAM_N = int(os.environ.get("DEAM_N", "191"))   # tree steps; control variate keeps low n accurate
# VSE_GPU_DEAM=1 routes the (no-dividend) de-Am batch to the Metal GPU kernel (fp32, ~4x the 28-core
# CPU at the de-Am). Fresh-producer path only - fp32 yields one of the equally-valid arb-free surfaces.
_GPU_DEAM = os.environ.get("VSE_GPU_DEAM", "").lower() in ("1", "true", "on")
if _GPU_DEAM:
    from ..iv.deam_gpu import deam_iv_batch_gpu as _deam_batch
else:
    _deam_batch = deam.deam_iv_batch
PARITY_BAND = 0.10     # |K/F-1| window of strikes de-Am'd inside the fixed-point loop (parity only)
PARITY_NSTRIKE = 8     # cap parity to the nearest N liquid pairs - densely-struck names (SPY/SPX)
                       # have ~80 strikes in the band, but the forward only needs a handful
KFIT_EMIT = 1.00       # |k| cap for emitted de-Am'd IVs; wider wings are extrapolated by the fit


@dataclass
class ExpiryQuotes:
    """Two-sided quotes for one expiry: arrays aligned by strike."""
    t: float
    strike: np.ndarray          # strikes with BOTH a call and a put
    call_mid: np.ndarray
    put_mid: np.ndarray
    call_spread: np.ndarray
    put_spread: np.ndarray
    d_seed: float = 0.99


@dataclass
class ExpirySurface:
    t: float
    forward: float
    discount: float
    # de-Am'd European IV per OTM contract (the fit observations)
    k: np.ndarray = field(default_factory=lambda: np.array([]))      # log-moneyness
    iv: np.ndarray = field(default_factory=lambda: np.array([]))     # de-Am'd Black IV (sigma*)
    is_call: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    strike: np.ndarray = field(default_factory=lambda: np.array([]))


def _imply_forward(strike: np.ndarray, c_minus_p: np.ndarray, weight: np.ndarray, D: float,
                   band_frac: float = 0.06, min_pairs: int = 3) -> float:
    """Forward from put-call parity with a KNOWN discount D (from FRED). Per-strike parity gives
    F = K + (C-P)/D; we average over near-ATM strikes (tightest, most reliable two-sided quotes).

    D is NEVER implied here - for an American name the near-ATM parity slope is corrupted by the
    put/call early-exercise-premium asymmetry, which pushes a free-D estimate past 1. Borrow and
    dividends enter only through the forward (the carry), so the risk-free D is taken as given.
    """
    order = np.argsort(strike)
    K, y, w = strike[order], c_minus_p[order], np.maximum(weight[order], 1e-12)
    f_per = K + y / D                                   # per-strike forward estimate
    sign = np.sign(y)
    cross = np.where(np.diff(sign) != 0)[0]
    if cross.size:
        i = cross[0]
        x0, x1, y0, y1 = K[i], K[i + 1], y[i], y[i + 1]
        f_rough = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else 0.5 * (x0 + x1)
    else:
        f_rough = K[int(np.argmin(np.abs(y)))]
    band = band_frac * f_rough
    sel = np.abs(K - f_rough) <= band
    if sel.sum() < min_pairs:
        sel = np.abs(K - f_rough) <= 2 * band
    if sel.sum() < min_pairs:
        return float("nan")
    return float(np.average(f_per[sel], weights=w[sel]))


def _european_cp(expiries, idx_list, F, D, S, q, sig_cache, divs):
    """For every expiry's near-ATM parity strikes, return European (C - P) per strike. De-Ams every
    tree-path contract across ALL expiries in ONE parallel kernel call (the q<=0 call leg is American
    == European, taken from the mid directly). Updates sig_cache with each contract's sigma*."""
    nE = len(expiries)
    CE = [np.full(idx_list[i].size, np.nan) for i in range(nE)]
    PE = [np.full(idx_list[i].size, np.nan) for i in range(nE)]
    tp, tK, tT, tr, tisc, tk, tpn, tmap = [], [], [], [], [], [], [], []
    for i, e in enumerate(expiries):
        idx = idx_list[i]
        if idx.size == 0:
            continue
        r_i = -math.log(D[i]) / e.t
        invDF = 1.0 / (D[i] * F[i])
        for m in range(idx.size):
            j = int(idx[m]); K = float(e.strike[j]); kk = math.log(K / F[i])
            for is_call, mid_arr, store in ((True, e.call_mid, CE), (False, e.put_mid, PE)):
                if not _need_tree(is_call, q, r_i, e.t):
                    store[i][m] = mid_arr[j]               # American == European exactly
                else:
                    tp.append(float(mid_arr[j])); tK.append(K); tT.append(e.t); tr.append(r_i)
                    tisc.append(is_call); tk.append(kk); tpn.append(mid_arr[j] * invDF)
                    tmap.append((i, m, is_call))
    if tp:
        tp = np.array(tp); tK = np.array(tK); tT = np.array(tT); tr = np.array(tr)
        tisc = np.array(tisc, dtype=np.bool_)
        seeds = np.full(tp.size, np.nan)
        if not _GPU_DEAM:                                  # GPU de-Am uses bracketed bisection (no seed) ->
            for u, (i, m, sc) in enumerate(tmap):          # the Black-IV seed (64-iter Newton) is pure waste
                cv = sig_cache.get((i, int(idx_list[i][m]), sc))
                if cv is not None and math.isfinite(cv):
                    seeds[u] = cv
            miss = ~np.isfinite(seeds)
            if miss.any():                                 # batched Black seed (total variance / t)
                w = black.implied_total_variance(np.array(tk)[miss], np.array(tpn)[miss], tisc[miss])
                seeds[miss] = np.sqrt(np.where(np.isfinite(w) & (w > 0), w / tT[miss], np.nan))
        if divs:
            res = np.array([deam.deam_iv(tp[u], S, tK[u], tT[u], tr[u], q, bool(tisc[u]),
                                         divs=divs, n=DEAM_N, seed=seeds[u]) for u in range(tp.size)])
        else:
            res = _deam_batch(tp, S, tK, tT, tr, np.full(tp.size, q), tisc, seeds, DEAM_N)
        for u, (i, m, sc) in enumerate(tmap):
            sig = res[u]
            if math.isfinite(sig) and sig > 0:
                (CE if sc else PE)[i][m] = deam.black76(F[i], tK[u], tT[u], sig, D[i], sc)
                sig_cache[(i, int(idx_list[i][m]), sc)] = float(sig)
    return [CE[i] - PE[i] for i in range(nE)]


def _need_tree(is_call: bool, q: float, r: float, t: float) -> bool:
    """True => early-exercise premium is real, must use the tree; False => direct Black is exact."""
    if t >= TREE_TMIN_LONG:
        return True
    if not is_call:                # American puts always carry rate-driven EEP
        return True
    return q > 0                   # dividend-driven call EEP whenever net carry q is positive (Merton)


def _black_iv(k: np.ndarray, pnorm: np.ndarray, is_call: bool | np.ndarray, t: float) -> np.ndarray:
    """Black IV (sqrt(w/t)) for an ARRAY of contracts in ONE call. implied_total_variance runs a
    fixed 64-iteration vectorized solve, so its per-call cost amortizes over the whole array - the
    de-Am must batch this (one call per expiry), never call it per contract."""
    isc = np.full(k.shape, is_call) if np.isscalar(is_call) else np.asarray(is_call, dtype=bool)
    w = black.implied_total_variance(np.asarray(k, dtype=float), np.asarray(pnorm, dtype=float), isc)
    out = np.full(k.shape, np.nan)
    good = np.isfinite(w) & (w > 0)
    out[good] = np.sqrt(w[good] / t)
    return out


def _deam_one(mid: float, S: float, K: float, t: float, r: float, q: float, is_call: bool, divs,
              sig_black: float, seed: float | None = None) -> float:
    """De-Am one contract given its PRECOMPUTED Black IV `sig_black` (the European inversion of the
    American mid, batched by the caller). Returns sig_black directly where American==European
    (q<=0 calls); else seeds the tree solver from `seed` (the previous iteration's cached sigma*,
    near-exact) or sig_black, converging in ~1-3 tree evals."""
    if not _need_tree(is_call, q, r, t):
        return float(sig_black) if (sig_black is not None and math.isfinite(sig_black) and sig_black > 0) else float("nan")
    use_seed = seed if (seed is not None and math.isfinite(seed) and seed > 0) else sig_black
    return deam.deam_iv(mid, S, K, t, r, q, is_call, divs=divs, n=DEAM_N, seed=use_seed)


def deam_european_chain(expiries: list[ExpiryQuotes], divs=(), kfit_max: float = KFIT_EMIT,
                        atm_band: float = PARITY_BAND,
                        liq: list[tuple] | None = None) -> tuple[list[ExpirySurface], dict]:
    """Run the §02 fixed point over one name's chain. Returns (per-expiry surfaces with de-Am'd
    European IVs, diagnostics{spot, carry, n_iters, converged, max_dF}).

    Compute is bounded so this is affordable across a 6,000-name universe - every American de-Am is
    a tree solve, so we only pay for the ones that matter:
      - the fixed-point loop only de-Ams strikes within `atm_band` of the forward (parity needs
        nothing else),
      - emission only de-Ams |k| <= `kfit_max` (wider wings are extrapolated by the fit), and
      - `liq[i] = (call_ok, put_ok)` boolean masks (aligned to expiries[i].strike) drop illiquid
        contracts BEFORE inversion - these would be filtered out of the fit anyway, so inverting
        them is pure waste. Liquidity (price/spread/size) is forward-independent, so the caller can
        precompute these masks. None => every contract is eligible.
    """
    n = len(expiries)
    t_arr = np.array([e.t for e in expiries])
    # discount is KNOWN (FRED), fixed throughout; only the forward is implied/iterated.
    D = np.array([e.d_seed for e in expiries], dtype=float)
    spr = [np.maximum(e.call_spread + e.put_spread, 1e-6) for e in expiries]
    if liq is None:
        liq = [(np.ones(e.strike.size, dtype=bool), np.ones(e.strike.size, dtype=bool))
               for e in expiries]
    liq_both = [c & p for (c, p) in liq]        # parity needs both legs liquid

    # --- init: forward-only parity on RAW American prices -> seed F_T ---
    F = np.full(n, np.nan)
    for i, e in enumerate(expiries):
        F[i] = _imply_forward(e.strike, e.call_mid - e.put_mid, 1.0 / spr[i] ** 2, D[i])

    sc = imply_spot_carry(t_arr, F, D)
    S, q = sc.spot, sc.carry
    converged = False
    it = 0
    max_dF = float("nan")
    sig_cache: dict = {}            # (expiry_idx, strike_idx, is_call) -> last sigma*, the next seed
    for it in range(1, MAX_ITERS + 1):
        F_prev = F.copy()
        q_prev = q
        # --- de-Am the near-ATM parity strikes for ALL expiries in one parallel sweep, re-imply F ---
        idx_list = []
        for i, e in enumerate(expiries):
            dist = np.abs(e.strike / F[i] - 1.0)
            cand = np.where(liq_both[i] & (dist <= atm_band))[0]   # liquid pairs near the forward
            if cand.size < 3:                                       # too few -> widen, ignore band
                cand = np.where(liq_both[i])[0]
                if cand.size == 0:
                    cand = np.arange(e.strike.size)
            idx_list.append(np.sort(cand[np.argsort(dist[cand])[:PARITY_NSTRIKE]]))
        cps = _european_cp(expiries, idx_list, F, D, S, q, sig_cache, divs)
        for i, e in enumerate(expiries):
            idx = idx_list[i]
            cp = cps[i]
            ok = np.isfinite(cp)
            if ok.sum() >= 3:
                f_new = _imply_forward(e.strike[idx][ok], cp[ok], (1.0 / spr[i] ** 2)[idx][ok], D[i])
                if np.isfinite(f_new):
                    F[i] = F[i] + DAMP * (f_new - F[i])
        sc = imply_spot_carry(t_arr, F, D)
        S, q = sc.spot, sc.carry
        d = np.abs(F - F_prev) / np.maximum(F_prev, 1e-9)
        max_dF = float(np.nanmax(d)) if np.any(np.isfinite(d)) else float("inf")  # degenerate: all-NaN F
        if max_dF < TOL_F and abs(q - q_prev) < TOL_CARRY:
            converged = True
            break

    # --- final pass: emit OTM-side de-Am'd European IVs (|k|<=kfit_max) per expiry ---
    # Two-phase so the trees run in ONE compiled, parallel kernel call: phase 1 selects the OTM
    # contracts per expiry and fills the Black-exact ones (q<=0 calls); phase 2 de-Ams every
    # tree-path contract across all expiries at once (deam.deam_iv_batch, prange over all cores).
    recs: list[list] = []                  # per expiry: [[k, strike, is_call, iv|None], ...]
    bp, bK, bT, br, bisc, bseed, bmap = [], [], [], [], [], [], []
    for i, e in enumerate(expiries):
        r_i = -math.log(D[i]) / e.t
        c_ok, p_ok = liq[i]
        k_all = np.log(e.strike / F[i])
        is_call_arr = k_all >= 0.0                           # OTM convention
        side_ok = np.where(is_call_arr, c_ok, p_ok)
        jidx = np.where((np.abs(k_all) <= kfit_max) & side_ok)[0]   # liquid OTM, |k| capped
        rec: list = []
        if jidx.size:
            iscj = is_call_arr[jidx]
            midj = np.where(iscj, e.call_mid[jidx], e.put_mid[jidx])
            if _GPU_DEAM:        # only NON-tree contracts need a Black IV (their output); tree-path -> GPU
                need = np.where(iscj, _need_tree(True, q, r_i, e.t), _need_tree(False, q, r_i, e.t))
                sigbj = np.full(jidx.size, np.nan); nt = ~need
                if nt.any():
                    sigbj[nt] = _black_iv(k_all[jidx][nt], (midj / (D[i] * F[i]))[nt], iscj[nt], e.t)
            else:
                sigbj = _black_iv(k_all[jidx], midj / (D[i] * F[i]), iscj, e.t)   # batched, 1 call
            for m, j in enumerate(jidx):
                is_call = bool(iscj[m])
                if not _need_tree(is_call, q, r_i, e.t):     # American == European -> Black is exact
                    iv = float(sigbj[m]) if (np.isfinite(sigbj[m]) and sigbj[m] > 0) else None
                    rec.append([float(k_all[j]), float(e.strike[j]), is_call, iv])
                else:                                        # tree-path -> queue for the batch kernel
                    seed = sig_cache.get((i, j, is_call))
                    if seed is None or not np.isfinite(seed):
                        seed = sigbj[m]
                    bp.append(float(midj[m])); bK.append(float(e.strike[j])); bT.append(e.t)
                    br.append(r_i); bisc.append(is_call); bseed.append(float(seed))
                    bmap.append((i, len(rec)))
                    rec.append([float(k_all[j]), float(e.strike[j]), is_call, None])
        recs.append(rec)

    if bp:
        if divs:                                             # explicit divs -> per-contract Python path
            for (i, pos), price, K, T, r_i, is_call, seed in zip(
                    bmap, bp, bK, bT, br, bisc, bseed):
                recs[i][pos][3] = deam.deam_iv(price, S, K, T, r_i, q, is_call, divs=divs,
                                               n=DEAM_N, seed=seed)
        else:
            res = _deam_batch(np.array(bp), S, np.array(bK), np.array(bT), np.array(br),
                              np.full(len(bp), q), np.array(bisc, dtype=np.bool_),
                              np.array(bseed), DEAM_N)
            for (i, pos), sig in zip(bmap, res):
                recs[i][pos][3] = float(sig)

    surfaces = []
    for i, e in enumerate(expiries):
        ks, ivs, isc, strikes = [], [], [], []
        for k, K, is_call, iv in recs[i]:
            if iv is not None and np.isfinite(iv) and iv > 0:
                ks.append(k); ivs.append(iv); isc.append(is_call); strikes.append(K)
        surfaces.append(ExpirySurface(
            t=e.t, forward=float(F[i]), discount=float(D[i]),
            k=np.array(ks), iv=np.array(ivs), is_call=np.array(isc, dtype=bool),
            strike=np.array(strikes)))
    return surfaces, {"spot": float(S), "carry": float(q), "n_iters": it,
                      "converged": converged, "max_dF": max_dF}
