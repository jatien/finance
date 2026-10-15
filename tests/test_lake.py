from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from finance.storage import Lake, SchemaError, connect, get_spec


def _prices(close: float, symbols=("A", "B")) -> pd.DataFrame:
    return pd.DataFrame({"symbol": list(symbols), "date": ["2025-12-31", "2026-01-02"][: len(symbols)],
                         "close": close, "adj_close": close, "volume": [100.0, None][: len(symbols)]})


def test_empty_dataset_has_typed_view(tmp_path):
    con = connect(Lake(tmp_path))
    desc = dict(con.execute("DESCRIBE prices_daily").fetchall()[i][:2] for i in range(3))
    assert desc == {"symbol": "VARCHAR", "date": "DATE", "open": "DOUBLE"}
    assert con.execute("SELECT count(*) FROM prices_daily").fetchone()[0] == 0


def test_write_partitions_and_latest_wins(tmp_path):
    lake = Lake(tmp_path)
    spec = get_spec("prices_daily")
    res = lake.write(spec, _prices(1.0), source="t", run_id="r1")
    assert res.rows == 2 and len(res.files) == 2  # dues particions d'any
    assert {f.parent.name for f in res.files} == {"year=2025", "year=2026"}
    lake.write(spec, _prices(2.0, symbols=("A",)), source="t", run_id="r2")
    df = connect(lake).execute("SELECT symbol, close, run_id, year FROM prices_daily ORDER BY symbol").df()
    assert df.to_dict("records") == [
        {"symbol": "A", "close": 2.0, "run_id": "r2", "year": 2025},
        {"symbol": "B", "close": 1.0, "run_id": "r1", "year": 2026},
    ]


def test_schema_validation(tmp_path):
    lake = Lake(tmp_path)
    spec = get_spec("prices_daily")
    with pytest.raises(SchemaError, match="obligatòries"):
        lake.write(spec, pd.DataFrame({"symbol": ["A"], "close": [1.0]}), source="t", run_id="r")
    with pytest.raises(SchemaError, match="nuls"):
        lake.write(spec, pd.DataFrame({"symbol": [None], "date": ["2026-01-02"]}), source="t", run_id="r")
    with pytest.raises(SchemaError, match="source"):
        lake.write(spec, _prices(1.0))


def test_compaction_keeps_latest_and_merges_files(tmp_path):
    lake = Lake(tmp_path)
    spec = get_spec("prices_daily")
    for i in range(4):
        lake.write(spec, _prices(float(i)), source="t", run_id=f"r{i}")
    assert len(lake.files(spec)) == 8
    res = lake.compact(spec, min_files=2)
    assert res.files_after == 2 and res.rows_before == 8 and res.rows_after == 2
    assert len(lake.files(spec)) == 2
    df = connect(lake).execute("SELECT close FROM prices_daily_raw").df()
    assert (df["close"] == 3.0).all()


def test_far_future_dates_and_schema_evolution(tmp_path):
    lake = Lake(tmp_path)
    spec = get_spec("macro_observations")
    lake.write(spec, pd.DataFrame({
        "series_id": ["X"], "date": [date(2020, 1, 1)], "value": [1.0],
        "realtime_start": [date(2020, 2, 1)], "realtime_end": [date(9999, 12, 31)],
        "vintage_mode": ["alfred"],
    }), source="t", run_id="r")
    # Fitxer "antic" sense la columna `vintage_mode`: el catàleg l'ha de llegir com a NULL
    old = pa.table({"series_id": ["Y"], "date": [date(2020, 1, 1)], "value": [2.0],
                    "realtime_start": [date(2020, 1, 1)], "realtime_end": [date(9999, 12, 31)],
                    "source": ["t"], "ingested_at": pa.array([pd.Timestamp.now(tz="UTC")], pa.timestamp("us", tz="UTC")),
                    "run_id": ["old"]})
    pq.write_table(old, lake.dataset_dir(spec) / "old.parquet")
    df = connect(lake).execute("SELECT series_id, realtime_end, vintage_mode FROM macro_observations ORDER BY 1").df()
    assert str(df["realtime_end"].iloc[0])[:10] == "9999-12-31"
    assert df["vintage_mode"].iloc[0] == "alfred" and pd.isna(df["vintage_mode"].iloc[1])
