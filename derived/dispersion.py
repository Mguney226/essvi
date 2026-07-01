"""dispersion.py -- implied and realized correlation across a name basket (cross-name analytics).

IMPLIED CORRELATION (the dispersion-trade measure): the single average pairwise correlation rho that makes a
weighted basket of constituent variances reproduce the INDEX variance,

    sigma_I^2 = sum_i w_i^2 sigma_i^2  +  rho * sum_{i!=j} w_i w_j sigma_i sigma_j
            => rho = (sigma_I^2 - sum_i w_i^2 sigma_i^2) / ( (sum_i w_i sigma_i)^2 - sum_i w_i^2 sigma_i^2 )

(CBOE legacy COR / Bossu 2005). The numerator is the index variance MINUS the idiosyncratic (own-variance)
floor; the denominator is the total cross-variance (= 2*sum_{i<j} w_i w_j sigma_i sigma_j). rho in [-1/(N-1), 1];
a clean implied correlation requires sigma_I, sigma_i in the SAME units (annualized variance) and the SAME tenor.

All legs come from the fitted vrp_implied.iv_var (model-free implied variance) per CM tenor: the index's and each
PIT-member constituent's. Weights are self-computed (P_i * Shares_i * IWF_i / sum); see weights.py. This module
is the pure math + the realized-correlation leg; the orchestrator (run_dispersion) joins the panel.
"""
from __future__ import annotations

import numpy as np


def implied_correlation(iv_index_var, iv_member_vars, weights):
    """Average implied correlation from the index + member ANNUALIZED implied variances (iv_var) + weights.
    `iv_index_var` scalar; `iv_member_vars`, `weights` length-N arrays. Weights are RENORMALIZED over the
    admissible (finite iv + positive weight) members so the result is weight-SCALE-invariant and identical to
    `dispersion_metrics.rho_clean` -- the clean implied correlation is a ratio of weight-quadratics, so raw vs
    normalized weights give different rho; we always normalize. Returns (rho, diag) with numerator/denominator/
    idio_frac + a `status`. NaN-safe: members with a non-finite iv or weight are dropped from the basket."""
    w = np.asarray(weights, dtype=float)
    iv = np.asarray(iv_member_vars, dtype=float)
    sig = np.sqrt(np.where(iv > 0, iv, np.nan))                      # member vols
    ok = np.isfinite(w) & (w > 0) & np.isfinite(sig)
    if ok.sum() < 2 or not np.isfinite(iv_index_var) or iv_index_var <= 0:
        return float("nan"), {"status": "insufficient", "n": int(ok.sum())}
    w = w[ok] / np.sum(w[ok]); sig = sig[ok]                         # renormalize over the admissible set
    own = float(np.sum(w ** 2 * sig ** 2))                          # idiosyncratic own-variance floor
    cross = float(np.sum(w * sig) ** 2 - own)                       # total cross-variance (2 sum_{i<j} ...)
    if cross <= 0:
        return float("nan"), {"status": "degenerate_cross", "n": int(ok.sum())}
    num = float(iv_index_var) - own
    rho = num / cross
    status = "ok"
    if rho > 1.0 or rho < -1.0 / (ok.sum() - 1):                    # outside the admissible band -> clamp + flag
        status = "clamped"
        rho = float(np.clip(rho, -1.0 / (ok.sum() - 1), 1.0))
    return float(rho), {"status": status, "n": int(ok.sum()), "own_var": own, "cross_var": cross,
                        "idio_frac": own / float(iv_index_var), "num": num}


def dispersion_metrics(iv_index_var, iv_member_vars, weights):
    """Full spec'd dispersion + implied-correlation set (multi-number discipline) from the index + member
    ANNUALIZED model-free implied variances (iv_var = K_log^2, the FULL-smile VIX strip, NOT ATM-only -- so it
    is correlation-robust where ATM COR is biased low, Linders-Schoutens 2014) + PIT float weights:
      rho_clean = (sig_I^2 - sum w^2 sig^2) / ((sum w sig)^2 - sum w^2 sig^2)   [Jacquier-Slaoui = CBOE COR algebra]
      rho_dirty = sig_I^2 / (sum w sig)^2                                       [naive proxy; rho_dirty >= rho_clean]
      D (dispersion, variance) = sum w sig^2 - sig_I^2                          [Gerchik-Ruffo-Schoenleber eq.1]
      DSPX = 100*sqrt(max(D,0))                                                 [S&P DJI 2023, annualized vol pts]
      rho_link = 1 - D / (sum w sig^2 - sum w^2 sig^2)                          [DSPX<->correlation link]
    Weights are renormalized over the COMPUTABLE members (finite iv + weight); the missing-mass fraction (weight
    of un-computable constituents) is reported + must be gated (the index leg covers the FULL basket). rho_clean
    is clamped to [-1/(N-1), 1] + flagged. NaN-safe."""
    w = np.asarray(weights, dtype=float)
    iv = np.asarray(iv_member_vars, dtype=float)
    sig = np.sqrt(np.where(iv > 0, iv, np.nan))
    wfin = np.isfinite(w) & (w > 0)
    ok = wfin & np.isfinite(sig)
    total_w = float(np.sum(w[wfin])) if wfin.any() else 0.0
    if ok.sum() < 2 or not np.isfinite(iv_index_var) or iv_index_var <= 0 or total_w <= 0:
        return {"status": "insufficient", "n": int(ok.sum())}
    wc = w[ok]; sc = sig[ok]; vc = iv[ok]
    miss = 1.0 - float(np.sum(wc)) / total_w
    wn = wc / np.sum(wc)                                            # renormalize over computable members
    own = float(np.sum(wn ** 2 * vc))                              # idiosyncratic own-variance
    wsig = float(np.sum(wn * sc)); wvar = float(np.sum(wn * vc))
    cross = wsig ** 2 - own
    nm1 = int(ok.sum()) - 1
    rc = (float(iv_index_var) - own) / cross if cross > 0 else float("nan")
    status = "ok"
    if np.isfinite(rc) and (rc > 1.0 or rc < -1.0 / nm1):
        status = "clamped"; rc = float(np.clip(rc, -1.0 / nm1, 1.0))
    rd = float(iv_index_var) / wsig ** 2 if wsig > 0 else float("nan")
    D = wvar - float(iv_index_var)
    cross_var = wvar - own
    rl = (1.0 - D / cross_var) if cross_var > 0 else float("nan")
    return {"status": status, "n": int(ok.sum()), "missing_mass": miss, "sum_w_computable": float(np.sum(wc)),
            "rho_clean": rc, "rho_dirty": rd, "rho_link": (float(rl) if np.isfinite(rl) else float("nan")),
            "dispersion_var": D, "dspx": float(100.0 * np.sqrt(max(D, 0.0))),
            "idio_frac": own / float(iv_index_var),
            "own_var": own, "wsig_sq": wsig ** 2, "wvar": wvar}      # raw aggregates: gates reconstruct rho + scale


def _pairwise_complete_corr(R):
    """N x N correlation matrix from a T x N return matrix that may contain NaN, using PAIRWISE-COMPLETE
    observations (each pair's corr uses only rows where BOTH names are finite, with the means/vars taken over
    that joint set) -- NOT zero-filling (which biases a partially-missing name's correlation toward 0). Fully
    vectorized via masked matrix products; entries with < 3 joint observations are NaN."""
    M = np.isfinite(R).astype(float)                                 # T x N finite mask
    Rz = np.where(np.isfinite(R), R, 0.0)
    n = M.T @ M                                                      # joint counts n_ij
    with np.errstate(invalid="ignore", divide="ignore"):
        A = Rz.T @ M                                                 # A[i,j] = sum x_i over joint(i,j)
        Sxx = (Rz * Rz).T @ M                                        # sum x_i^2 over joint(i,j)
        Sxy = Rz.T @ Rz                                              # sum x_i x_j over joint(i,j)
        mx = A / n; my = A.T / n                                     # joint means of x_i, x_j
        cov = Sxy / n - mx * my
        vx = Sxx / n - mx * mx; vy = Sxx.T / n - my * my
        C = cov / np.sqrt(vx * vy)
    C[n < 3] = np.nan
    np.fill_diagonal(C, 1.0)
    return C


def realized_correlation_pairwise(member_returns, weights):
    """Realized average correlation rho_realized = sum_{i!=j} w_i w_j sig_i sig_j rho_ij / sum_{i!=j} w_i w_j sig_i
    sig_j -- the VOL-weighted mean of the actual pairwise realized correlations over the return window (sig_i =
    realized vol). This MATCHES the implied rho_clean's CBOE-COR operator (also w_iw_j sig_i sig_j weighted, with
    IMPLIED sig), so CRP = rho_clean - rho_realized is apples-to-apples; a bare w_iw_j weighting is a different
    operator that biases CRP ~2-3pt (Driessen-Maenhout-Vilkov 2009). Uses PAIRWISE-COMPLETE correlations (no
    zero-fill bias) and averages only over pairs with a computable corr. Returns (rho, n_used)."""
    R = np.asarray(member_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if R.ndim != 2 or R.shape[1] != w.size:
        return float("nan"), 0
    colok = np.isfinite(w) & (w > 0) & (np.sum(np.isfinite(R), axis=0) > R.shape[0] // 2) & (np.nanstd(R, axis=0) > 0)
    if colok.sum() < 2:
        return float("nan"), int(colok.sum())
    R = R[:, colok]; w = w[colok]
    C = _pairwise_complete_corr(R)
    sig_r = np.nanstd(R, axis=0)                                    # realized per-member vol over the window
    ws = w * sig_r                                                  # w_i sig_i -> matches the implied COR operator
    ww = np.outer(ws, ws)
    valid = np.isfinite(C); np.fill_diagonal(valid, False)          # exclude diagonal + uncomputable pairs
    ww = ww * valid
    den = float(np.sum(ww))
    num = float(np.sum(ww * np.where(valid, C, 0.0)))
    return (float(np.clip(num / den, -1.0, 1.0)) if den > 0 else float("nan")), int(colok.sum())


def variance_contributions(iv_index_var, iv_member_vars, weights, rho):
    """Per-name decomposition of the index implied variance under the single-correlation model (a spec headline
    differentiator -- which names drive index variance). With S = sum_j w_j sig_j (renormalized weights) and the
    fitted average correlation `rho`,
        contrib_i = w_i sig_i * [ (1-rho) w_i sig_i + rho S ]   so that  sum_i contrib_i = sigma_I^2 (model).
    Returns (idx_arr, contrib, share) where idx_arr indexes the admissible members within the input arrays,
    contrib_i is the variance contribution, and share_i = contrib_i / sigma_I^2. NaN-safe."""
    w = np.asarray(weights, dtype=float); iv = np.asarray(iv_member_vars, dtype=float)
    sig = np.sqrt(np.where(iv > 0, iv, np.nan))
    ok = np.isfinite(w) & (w > 0) & np.isfinite(sig)
    if ok.sum() < 2 or not np.isfinite(iv_index_var) or iv_index_var <= 0 or not np.isfinite(rho):
        return np.array([], dtype=int), np.array([]), np.array([])
    idx = np.where(ok)[0]
    wn = w[ok] / np.sum(w[ok]); sc = sig[ok]
    S = float(np.sum(wn * sc))
    contrib = wn * sc * ((1.0 - rho) * wn * sc + rho * S)
    share = contrib / float(iv_index_var)
    return idx, contrib, share


def realized_correlation(member_log_returns, weights):
    """Average realized correlation over a window from member daily LOG-RETURN columns (T x N) + weights, via the
    same index-vs-basket identity on REALIZED variances: rho = (RV_basket-implied_index - sum w_i^2 RV_i) / cross.
    Here the 'index' realized variance is the variance of the weighted-return series sum_i w_i r_i (the basket),
    so rho is the weighted-average realized pairwise correlation. Returns (rho, n_used)."""
    R = np.asarray(member_log_returns, dtype=float)                 # T x N
    w = np.asarray(weights, dtype=float)
    if R.ndim != 2 or R.shape[1] != w.size:
        return float("nan"), 0
    colok = np.isfinite(w) & (w > 0) & np.all(np.isfinite(R), axis=0)
    if colok.sum() < 2:
        return float("nan"), int(colok.sum())
    R = R[:, colok]; w = w[colok]
    rv = np.var(R, axis=0)                                          # per-member realized variance
    own = float(np.sum(w ** 2 * rv))
    basket = float(np.var(R @ w))                                   # weighted-basket realized variance
    cross = float((np.sum(w * np.sqrt(rv))) ** 2 - own)
    if cross <= 0:
        return float("nan"), int(colok.sum())
    rho = (basket - own) / cross
    return float(np.clip(rho, -1.0, 1.0)), int(colok.sum())
