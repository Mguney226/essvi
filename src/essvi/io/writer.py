"""Build schema-exact Arrow tables and write snappy Parquet, deterministically sorted."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..pipeline.rounding import round_rows
from ..schemas import LAYERS

# deterministic sort keys per layer (stable row order for reproducible output)
_SORT = {
    "vol_surface_params": ["expiry"],
    "vol_surface_grid": ["expiry", "k"],
    "iv_history": ["tenor"],
    "option_quotes": ["expiry", "strike", "right"],
    "underlier_state": ["ts"],
    "forward_vol": ["t1", "t2"],
    "forward_vol_grid": ["t1", "t2", "k"],
    "earnings_vol": ["ts", "earnings_date"],
    "vrp_implied": ["tenor"],
    "event_vol": ["event_date", "event_type"],
}


def build_table(layer: str, rows: list[dict]) -> pa.Table:
    schema = LAYERS[layer]
    rows = round_rows(layer, rows)
    keys = _SORT.get(layer, [])
    if keys:
        rows = sorted(rows, key=lambda r: tuple((r.get(k) is None, r.get(k)) for k in keys))
    # ensure every schema column present in every row (None where absent)
    cols = [f.name for f in schema]
    norm = [{c: r.get(c) for c in cols} for r in rows]
    return pa.Table.from_pylist(norm, schema=schema)


def write_layer(layer: str, rows: list[dict], out_path: Path) -> int:
    table = build_table(layer, rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy", write_statistics=True)
    return table.num_rows
