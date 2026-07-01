"""Fixed constants from the methodology spec (§04, §05). Pinned here, never tuned per name."""

# --- Grid (§05) ---
GRID_K_MIN = -2.00
GRID_K_MAX = 2.00
GRID_K_STEP = 0.025            # 161 points per slice
DENSE_K_STEP = 0.005           # 5x-denser independent verification grid
EPS_ARB = 1e-6                 # total-variance tolerance for arb_ok checks

# Constant-maturity tenors for iv_history (calendar days)
CM_TENORS = (7, 14, 30, 60, 90, 120, 180, 270, 365)

# --- Fit (§04) ---
HUBER_DELTA = 1.345            # 95% Gaussian-efficiency constant
MIN_STRIKES = 5               # two-sided strikes per expiry to attempt a full fit
MIN_EXPIRIES = 3
GAMMA_DEFAULT = 0.40           # frozen gamma for reduced fits
WARM_RMSE_FACTOR = 1.5         # warm reprice may rise to 1.5x the prior snapshot (or +WARM_RMSE_MARGIN vp)
WARM_RMSE_MARGIN = 1.0         # before the warm fit is rejected and the snapshot cold-re-anchors

# SLSQP
SLSQP_FTOL = 1e-9
SLSQP_MAXITER = 400

# Power-law phi(theta) parameter box
ETA_MIN = 1e-6
GAMMA_MIN = 1e-6
GAMMA_MAX = 0.5 - 1e-6
RHO_ABS_MAX = 0.999
RHO_SMOOTH = 1e-8              # |rho| ~ sqrt(rho^2 + RHO_SMOOTH^2) for differentiability

# Deterministic 16-point cold-start grid (rho x eta x gamma)
COLD_START_RHO = (-0.8, -0.4, 0.0, 0.4)
COLD_START_ETA = (0.5, 1.5)
COLD_START_GAMMA = (0.2, 0.4)

# --- Day count / time ---
DAYCOUNT = 365.0               # Act/365F
# Index settlement times (ET) -> used for the intraday year-fraction of the expiry day
SETTLE_AM_ET = (9, 30)         # SPX/NDX AM-settled (SET on the open)
SETTLE_PM_ET = (16, 0)        # SPXW/NDXP PM-settled (close)

# --- Output rounding (deterministic, pre-write) ---
ROUND_IV = 8
ROUND_W = 10
ROUND_K = 6
ROUND_T = 8
ROUND_PARAM = 10
ROUND_RMSE = 6
ROUND_PRICE = 6

# --- Provenance (§05) ---
OBSERVED_WEIGHT_MIN = 0.25     # a surviving quote with fit weight >= this within |dk|<=0.025 -> observed
OBSERVED_DK = 0.025

# --- §01 liquidity filter for fit quotes (trim deep-OTM noise) ---
PRICE_FLOOR = 0.10            # min mid price to enter the fit
MAX_REL_SPREAD = 0.50         # max (ask-bid)/mid to enter the fit
KFIT_MAX = 1.00               # moneyness sanity cap for fit quotes (wings beyond ship extrapolated)
NOISE_CAP_VP = 0.50           # max half-spread expressed in Black vol points (via vega):
                              # 100*(spread/2)/(D*F*n(d1)*sqrt(t)) <= cap. Bounds achievable rmse.
EXTRAP_DK_WARN = 0.5           # usage rec: don't price extrapolated points beyond this dist_k
