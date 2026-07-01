"""essvi: arbitrage-free fitted implied-volatility surfaces for US equity options."""
import os as _os

# Single-thread the BLAS/OpenMP backends for bit-reproducible fits, set at package import so
# determinism does not depend on the caller pinning them. setdefault() respects an explicit override.
# For full effect these must be set before numpy/BLAS first loads, so the orchestrators (run_m2.py,
# run_names.py) ALSO pin them at their entry point before importing numpy/polars/duckdb.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__version__ = "0.4.0"
