"""Black price -> implied-variance roundtrip, including deep wings."""
import numpy as np
from essvi.iv import black


def test_roundtrip_recovers_total_variance():
    rng = np.linspace(-1.5, 1.5, 61)          # log-moneyness incl. wings
    w_true = 0.04 + 0.02 * rng**2             # a smile in total variance
    is_call = black.otm_is_call(rng)
    p = black.norm_price(rng, w_true, is_call)
    w_rec = black.implied_total_variance(rng, p, is_call)
    assert np.all(np.isfinite(w_rec)), "all OTM prices should invert"
    assert np.allclose(w_rec, w_true, atol=1e-7), np.max(np.abs(w_rec - w_true))


def test_atm_price_and_vega():
    # ATM (k=0): normalized call price ~ 2*N(sqrt(w)/2) - 1 ; vega = n(0) at small w
    w = np.array([0.04])
    p = black.norm_price(np.array([0.0]), w, np.array([True]))
    from scipy.special import ndtr
    expected = 2 * ndtr(np.sqrt(w) / 2) - 1
    assert np.allclose(p, expected, atol=1e-12)
    v = black.norm_vega_sw(np.array([0.0]), w)
    assert v[0] > 0


def test_below_intrinsic_returns_nan():
    # a price below intrinsic must not invert
    k = np.array([0.5])           # OTM call, intrinsic 0; feed a negative price
    p = np.array([-0.01])
    w = black.implied_total_variance(k, p, np.array([True]))
    assert np.isnan(w[0])
