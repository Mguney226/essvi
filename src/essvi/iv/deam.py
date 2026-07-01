"""De-Americanization.

Single-name and ETF options are American; the early-exercise premium contaminates a naive
Black inversion. We price every American contract on an American engine - escrowed discrete
dividends, the day's borrow and discount - and root-find the single volatility sigma* that
reprices the American NBBO mid. That sigma* IS the European-equivalent Black implied vol that
ships in `iv`, comparable to index IV across the whole universe. The European-equivalent price
Black(F, K, t, sigma*, D) is what feeds the §02 parity construction.

Two independent American engines are provided and cross-checked in the test harness:
  - `crr_american`  : Cox-Ross-Rubinstein binomial - the obvious-correct reference.
  - `lr_american`   : Leisen-Reimer binomial (Peizer-Pratt) - O(1/n^2), fast, the production path.
Both use the escrowed-dividend model: the diffusive process is S* = S - PV(future divs), and a
node's exercise value uses the actual spot S*_node + PV(divs still to come at that node's time).

Determinism: fixed step counts, no wall-clock early stop, scipy brentq with fixed tol.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numba import njit, prange
from scipy.optimize import brentq
from scipy.special import ndtr

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)

# ---------------------------------------------------------------- European (Black-76) leg
def black76(F: float, K: float, t: float, sigma: float, D: float, is_call: bool) -> float:
    """Black-76 price of a European option on a forward F, discount factor D."""
    if t <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return D * intrinsic
    sw = sigma * math.sqrt(t)
    d1 = (math.log(F / K) + 0.5 * sw * sw) / sw
    d2 = d1 - sw
    if is_call:
        return D * (F * ndtr(d1) - K * ndtr(d2))
    return D * (K * ndtr(-d2) - F * ndtr(-d1))


# ----------------------------------------------------------------- dividend bookkeeping
def _pv_future_divs(divs: Sequence[tuple[float, float]], r: float, t_now: float, T: float) -> float:
    """PV at time t_now of dividends with ex-time in (t_now, T]."""
    return sum(amt * math.exp(-r * (td - t_now)) for td, amt in divs if t_now < td <= T + 1e-12)


# ----------------------------------------------------------------- CRR American (reference)
def crr_american(S: float, K: float, T: float, r: float, q: float, sigma: float,
                 is_call: bool, n: int = 1600, divs: Sequence[tuple[float, float]] = ()) -> float:
    """American option price by a CRR binomial tree, escrowed discrete dividends + continuous
    carry q (borrow). The obvious-correct reference engine."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    Sstar0 = S - _pv_future_divs(divs, r, 0.0, T)  # diffusive spot at t=0
    j = np.arange(n + 1)
    # terminal: all divs paid, actual spot == diffusive node value
    ST = Sstar0 * u ** j * d ** (n - j)
    val = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
    for i in range(n - 1, -1, -1):
        val = disc * (p * val[1:i + 2] + (1.0 - p) * val[0:i + 1])
        ti = i * dt
        jj = np.arange(i + 1)
        Snode = Sstar0 * u ** jj * d ** (i - jj) + _pv_future_divs(divs, r, ti, T)
        ex = np.maximum(Snode - K, 0.0) if is_call else np.maximum(K - Snode, 0.0)
        val = np.maximum(val, ex)
    return float(val[0])


# ----------------------------------------------------------------- Leisen-Reimer American
def _pp_inv(z: float, n: int) -> float:
    """Peizer-Pratt inversion (method 2) of the normal CDF used by the Leisen-Reimer tree.
    Clamped strictly inside (0, 1): at extreme |z| (tiny sigma, or deep ITM/OTM strikes) the raw
    value saturates to exactly 0.0 or 1.0, which would divide-by-zero in the u/d construction
    below. The clamp degrades gracefully to a drift-only (no-diffusion) tree, the correct limit."""
    c = 1.0 if z >= 0 else -1.0
    a = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
    p = 0.5 + c * 0.5 * math.sqrt(1.0 - math.exp(-a))
    return min(max(p, 1e-12), 1.0 - 1e-12)


@njit(cache=True)
def _pp_inv_nb(z: float, n: int) -> float:
    c = 1.0 if z >= 0 else -1.0
    a = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
    p = 0.5 + c * 0.5 * math.sqrt(1.0 - math.exp(-a))
    return min(max(p, 1e-12), 1.0 - 1e-12)


@njit(cache=True)
def _lr_am_eu_nb(S, K, T, r, q, sigma, is_call, n):
    """Compiled twin of _lr_am_eu for the NO-DIVIDEND path (Sstar0 == S). Numerically identical to
    the numpy version - same Peizer-Pratt p/u/d, same in-place backward induction - but ~50x faster,
    which is what makes the de-Am affordable across the 6,000-name universe. Dividend chains keep the
    numpy path (american_price branches on `divs`)."""
    if T <= 0.0 or sigma <= 0.0:
        intr = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intr, intr
    if n % 2 == 0:
        n += 1
    dt = T / n
    sw = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sw
    d2 = d1 - sw
    p = _pp_inv_nb(d2, n)
    pbar = _pp_inv_nb(d1, n)
    u = math.exp((r - q) * dt) * pbar / p
    d = (math.exp((r - q) * dt) - p * u) / (1.0 - p)
    disc = math.exp(-r * dt)
    am = np.empty(n + 1)
    eu = np.empty(n + 1)
    for k in range(n + 1):
        ST = S * u ** k * d ** (n - k)
        pay = max(ST - K, 0.0) if is_call else max(K - ST, 0.0)
        am[k] = pay
        eu[k] = pay
    for i in range(n - 1, -1, -1):
        for k in range(i + 1):                 # increasing k: am[k+1] still holds the i+1 level value
            am[k] = disc * (p * am[k + 1] + (1.0 - p) * am[k])
            eu[k] = disc * (p * eu[k + 1] + (1.0 - p) * eu[k])
            Snode = S * u ** k * d ** (i - k)
            ex = max(Snode - K, 0.0) if is_call else max(K - Snode, 0.0)
            if ex > am[k]:
                am[k] = ex
    return am[0], eu[0]


def _lr_am_eu(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool,
              n: int = 257, divs: Sequence[tuple[float, float]] = ()) -> tuple[float, float]:
    """One Leisen-Reimer lattice, returning BOTH the American (early-exercise) and the European
    (no early-exercise) value. Their shared discretization error is what makes the control variate
    in american_price() accurate. Probabilities are clamped (see _pp_inv) so deep ITM/OTM or
    tiny-sigma nodes degrade to a drift-only tree instead of dividing by zero."""
    if T <= 0 or sigma <= 0:
        intr = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intr, intr
    if n % 2 == 0:
        n += 1
    dt = T / n
    Sstar0 = S - _pv_future_divs(divs, r, 0.0, T)
    sw = sigma * math.sqrt(T)
    d1 = (math.log(Sstar0 / K) + (r - q + 0.5 * sigma * sigma) * T) / sw
    d2 = d1 - sw
    p = _pp_inv(d2, n)
    pbar = _pp_inv(d1, n)
    u = math.exp((r - q) * dt) * pbar / p
    d = (math.exp((r - q) * dt) - p * u) / (1.0 - p)
    disc = math.exp(-r * dt)
    j = np.arange(n + 1)
    ST = Sstar0 * u ** j * d ** (n - j)
    payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
    am = payoff
    eu = payoff.copy()
    for i in range(n - 1, -1, -1):
        am = disc * (p * am[1:i + 2] + (1.0 - p) * am[0:i + 1])
        eu = disc * (p * eu[1:i + 2] + (1.0 - p) * eu[0:i + 1])
        ti = i * dt
        jj = np.arange(i + 1)
        Snode = Sstar0 * u ** jj * d ** (i - jj) + _pv_future_divs(divs, r, ti, T)
        ex = np.maximum(Snode - K, 0.0) if is_call else np.maximum(K - Snode, 0.0)
        am = np.maximum(am, ex)            # early exercise applies only to the American leg
    return float(am[0]), float(eu[0])


def lr_american(S: float, K: float, T: float, r: float, q: float, sigma: float,
                is_call: bool, n: int = 101, divs: Sequence[tuple[float, float]] = ()) -> float:
    """Raw Leisen-Reimer American tree value (no control variate). american_price() is the
    accurate production entry; this raw form is kept for cross-checking against crr_american()."""
    return _lr_am_eu(S, K, T, r, q, sigma, is_call, n=n, divs=divs)[0]


def american_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
                   is_call: bool, divs: Sequence[tuple[float, float]] = (), n: int = 257) -> float:
    """Production American price via a CONTROL VARIATE: the exact Black-76 European value plus the
    tree-estimated early-exercise premium (american_tree - european_tree from one lattice). The
    tree's discretization error largely cancels in the premium, so even deep-OTM / short-dated
    contracts - where a raw tree is noisy on a tiny price - invert to vol accurately, and modest n
    suffices. (A raw LR is only ~O(1/n) on the American free boundary; the control variate sidesteps
    that by putting the exact value in the European leg and leaving the tree only the small premium.)"""
    if divs:
        am, eu = _lr_am_eu(S, K, T, r, q, sigma, is_call, n=n, divs=divs)
    else:
        am, eu = _lr_am_eu_nb(float(S), float(K), float(T), float(r), float(q),
                              float(sigma), bool(is_call), int(n))
    Sstar0 = S - _pv_future_divs(divs, r, 0.0, T)
    F = Sstar0 * math.exp((r - q) * T)
    eu_exact = black76(F, K, T, sigma, math.exp(-r * T), is_call)
    return eu_exact + (am - eu)


# --------------------------------------------------- compiled parallel batch de-Am (no-div path)
@njit(cache=True)
def _black76_nb(F, K, t, sigma, D, is_call):
    if sigma <= 0.0 or t <= 0.0:
        intr = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return D * intr
    sw = sigma * math.sqrt(t)
    d1 = (math.log(F / K) + 0.5 * sw * sw) / sw
    d2 = d1 - sw
    nd1 = 0.5 * math.erfc(-d1 / _SQRT2)
    nd2 = 0.5 * math.erfc(-d2 / _SQRT2)
    if is_call:
        return D * (F * nd1 - K * nd2)
    return D * (K * (1.0 - nd2) - F * (1.0 - nd1))


@njit(cache=True)
def _american_price_nb(S, K, T, r, q, sigma, is_call, n):
    """Control-variate American price (no divs): exact Black + tree-estimated early-exercise premium."""
    am, eu = _lr_am_eu_nb(S, K, T, r, q, sigma, is_call, n)
    F = S * math.exp((r - q) * T)
    D = math.exp(-r * T)
    return _black76_nb(F, K, T, sigma, D, is_call) + (am - eu)


@njit(cache=True)
def _deam_iv_scalar_nb(price, S, K, T, r, q, is_call, n, seed, lo, hi):
    """Compiled twin of deam_iv: seeded quasi-Newton (Black vega slope) + bisection backstop. Returns
    NaN on sub-intrinsic price or no root in [lo,hi] - identical contract to the Python solver."""
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if not (price == price) or price < intrinsic - 1e-9:
        return np.nan
    F = S * math.exp((r - q) * T)
    D = math.exp(-r * T)
    sqrtT = math.sqrt(T)
    sig = seed
    if not (sig == sig) or sig <= lo or sig >= hi:
        sig = 0.3
    for _ in range(24):                                    # seeded Newton
        am = _american_price_nb(S, K, T, r, q, sig, is_call, n)
        sw = sig * sqrtT
        d1 = (math.log(F / K) + 0.5 * sw * sw) / sw
        vega = F * D * math.exp(-0.5 * d1 * d1) * _INV_SQRT2PI * sqrtT
        if vega < 1e-10:
            break
        sig_new = sig - (am - price) / vega
        if sig_new <= lo or sig_new >= hi:
            break
        if abs(sig_new - sig) < 1e-7:
            return sig_new
        sig = sig_new
    a = lo                                                  # bisection backstop
    b = hi
    fa = _american_price_nb(S, K, T, r, q, a, is_call, n) - price
    fb = _american_price_nb(S, K, T, r, q, b, is_call, n) - price
    if not (fa == fa) or not (fb == fb) or fa > 0.0 or fb < 0.0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (a + b)
        fm = _american_price_nb(S, K, T, r, q, mid, is_call, n) - price
        if fm > 0.0:
            b = mid
        else:
            a = mid
        if b - a < 1e-8:
            break
    return 0.5 * (a + b)


@njit(parallel=True, cache=True)
def deam_iv_batch(prices, S, Ks, Ts, rs, qs, is_calls, seeds, n):
    """De-Am an ARRAY of one name's contracts across all cores at once. Each contract is an
    independent root-find, so prange fans them over the machine with no Python orchestration between
    tree solves. This is what makes a whole name's de-Am sub-second on a many-core box."""
    m = prices.shape[0]
    out = np.empty(m)
    for i in prange(m):
        out[i] = _deam_iv_scalar_nb(prices[i], S, Ks[i], Ts[i], rs[i], qs[i],
                                    is_calls[i], n, seeds[i], 0.01, 5.0)
    return out


# ----------------------------------------------------------------- de-Americanization
def deam_iv(price: float, S: float, K: float, T: float, r: float, q: float, is_call: bool,
            divs: Sequence[tuple[float, float]] = (), n: int = 257,
            lo: float = 0.01, hi: float = 5.0, tol: float = 1e-8, seed: float | None = None) -> float:
    """The single volatility sigma* that makes the American engine reprice `price` (the American
    NBBO mid). Returns NaN ONLY when the price is below intrinsic or genuinely outside
    [engine(lo), engine(hi)] (no root in the bracket).

    `seed` (e.g. the Black IV of the American mid) enables a quasi-Newton fast path: american_price
    is strictly monotone in sigma, so a few Newton steps using the European Black vega as the slope
    converge in ~2-3 tree solves instead of ~50 for a blind brentq bracket. The bracketed brentq is
    kept as a backstop for any case where Newton leaves the domain or stalls (deep ITM/OTM, vega->0),
    so the result is identical to the pure-brentq solver - just far fewer tree evaluations."""
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if not math.isfinite(price) or price < intrinsic - 1e-9:
        return float("nan")
    if seed is not None and math.isfinite(seed) and lo < seed < hi:
        F = (S - _pv_future_divs(divs, r, 0.0, T)) * math.exp((r - q) * T)
        sqrtT = math.sqrt(T)
        sig = float(seed)
        for _ in range(16):
            am = american_price(S, K, T, r, q, sig, is_call, divs=divs, n=n)
            sw = sig * sqrtT
            d1 = (math.log(F / K) + 0.5 * sw * sw) / sw
            vega = (S - _pv_future_divs(divs, r, 0.0, T)) * math.exp((r - q) * T) \
                * math.exp(-r * T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * sqrtT
            if vega < 1e-10:
                break                                   # flat -> let brentq handle it
            sig_new = sig - (am - price) / vega
            if not (lo < sig_new < hi):
                break                                   # left the domain -> brentq backstop
            if abs(sig_new - sig) < 1e-7:
                return float(sig_new)                   # converged in sigma (sub-0.1bp)
            sig = sig_new
    f = lambda s: american_price(S, K, T, r, q, s, is_call, divs=divs, n=n) - price
    flo, fhi = f(lo), f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)) or flo > 0.0 or fhi < 0.0:
        return float("nan")  # price outside [engine(lo), engine(hi)] -> no root in the bracket
    try:
        return float(brentq(f, lo, hi, xtol=tol, maxiter=200))
    except ValueError:
        return float("nan")


def deam_european_iv(price: float, S: float, K: float, T: float, r: float, q: float, F: float,
                     D: float, is_call: bool, divs: Sequence[tuple[float, float]] = (),
                     n: int = 201) -> tuple[float, float, float]:
    """Full §03 step for one contract: returns (sigma*, european_equivalent_price, eep).
    sigma* is the shipped Black IV; the European price = Black(F,K,T,sigma*,D) feeds §02 parity;
    eep = American mid - European price is the stripped early-exercise premium."""
    s = deam_iv(price, S, K, T, r, q, is_call, divs=divs, n=n)
    if not math.isfinite(s):
        return float("nan"), float("nan"), float("nan")
    euro = black76(F, K, T, s, D, is_call)
    return s, euro, price - euro
