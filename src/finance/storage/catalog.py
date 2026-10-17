"""Catàleg DuckDB: vistes deduplicades sobre el lake Parquet + vistes/macros de conveniència.

Dos modes:
- `connect()`: connexió en memòria amb totes les vistes (jobs, notebooks, models).
- `build_catalog_file()`: fitxer `catalog.duckdb` persistent amb les mateixes vistes, per
  obrir-lo des de VSCode (SQLTools), DBeaver o la CLI de duckdb. Es regenera després de
  cada job i es substitueix de forma atòmica (els lectors oberts no es veuen afectats).

Per a cada dataset X hi ha:
- `X_raw`: totes les files de tots els fitxers (inclou versions antigues).
- `X`: una fila per clau primària, la més recent segons `dedup_order`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb

from finance.storage.lake import Lake
from finance.storage.schemas import REGISTRY, DatasetSpec, duckdb_type

log = logging.getLogger(__name__)


def _empty_select(spec: DatasetSpec) -> str:
    cols = ", ".join(f'CAST(NULL AS {duckdb_type(f.type)}) AS "{f.name}"' for f in spec.schema)
    return f"SELECT {cols} WHERE false"


def raw_view_sql(lake: Lake, spec: DatasetSpec) -> str:
    if not lake.files(spec):
        return _empty_select(spec)
    opts = ["union_by_name=true"]
    if spec.partition:
        pfield = spec.schema.field(spec.partition.column)
        opts += ["hive_partitioning=true",
                 f"hive_types={{'{pfield.name}': {duckdb_type(pfield.type)}}}"]
    else:
        opts.append("hive_partitioning=false")
    src = f"read_parquet('{lake.glob(spec)}', {', '.join(opts)})"
    # Projecció explícita amb l'esquema actual: columnes noves apareixen com a NULL
    cols = ", ".join(f'"{n}"' for n in spec.schema.names)
    return f"SELECT {cols} FROM {src}"


def dedup_view_sql(spec: DatasetSpec) -> str:
    pk = ", ".join(f'"{c}"' for c in spec.primary_key)
    order = ", ".join(f'"{c}" DESC' for c in spec.dedup_order)
    src = f"{spec.name}_raw"
    if spec.snapshot_by:
        # Files de l'última ingesta de cada grup. Amb una unió contra el màxim per grup (que
        # només llegeix dues columnes) en lloc d'una finestra sobre totes les columnes: els
        # filtres de la consulta (p. ex. per data) arriben a la lectura del Parquet.
        # Equivalent a deduplicar primer per clau: la fila més recent de cada clau del grup
        # sempre pertany a l'última ingesta.
        grp = [f'"{c}"' for c in spec.snapshot_by]
        on = " AND ".join(f"r.{g} IS NOT DISTINCT FROM m.{g}" for g in grp)
        src = (f"(SELECT r.* FROM {spec.name}_raw r JOIN (SELECT {', '.join(grp)}, "
               f"max(ingested_at) AS _max_ing FROM {spec.name}_raw GROUP BY ALL) m "
               f"ON {on} AND r.ingested_at = m._max_ing)")
        if spec.pk_unique_per_ingest:
            sql = f"SELECT * FROM {src}"
            if spec.view_filter:
                sql = f"SELECT * FROM ({sql}) WHERE {spec.view_filter}"
            return sql
    sql = (f"SELECT * FROM {src} "
           f"QUALIFY row_number() OVER (PARTITION BY {pk} ORDER BY {order}) = 1")
    if spec.view_filter:
        sql = f"SELECT * FROM ({sql}) WHERE {spec.view_filter}"
    return sql


# Vistes derivades (depenen de les vistes base)
DERIVED_VIEWS: dict[str, str] = {
    # Valor vigent avui de cada observació macro
    "macro_latest": """
        SELECT series_id, date, value, realtime_start, vintage_mode
        FROM macro_observations WHERE realtime_end = DATE '9999-12-31'
    """,
    # Darrera foto de cada univers dinàmic
    "universe_current": """
        SELECT * FROM universe_membership
        QUALIFY as_of_date = max(as_of_date) OVER (PARTITION BY universe)
    """,
    # Metadades més recents per instrument
    "instruments_current": """
        SELECT * FROM instruments
        QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY as_of_date DESC) = 1
    """,
    # Darrera foto de cadena d'opcions per subjacent
    "options_latest": """
        SELECT * FROM option_snapshots
        QUALIFY snapshot_ts = max(snapshot_ts) OVER (PARTITION BY underlying)
    """,
    # Estat de l'última execució de cada job
    "job_status": """
        SELECT job, status, started_at, finished_at, duration_s, items_ok, items_failed,
               rows_written, message
        FROM ingest_runs
        QUALIFY row_number() OVER (PARTITION BY job ORDER BY started_at DESC) = 1
    """,
    # Cobertura de preus per símbol
    "prices_coverage": """
        SELECT symbol, min(date) AS first_date, max(date) AS last_date, count(*) AS n_rows,
               max(ingested_at) AS last_ingested_at
        FROM prices_daily GROUP BY symbol
    """,
}

# Macros: consultes point-in-time sense look-ahead
MACROS: dict[str, str] = {
    # Observacions macro tal com es coneixien a la data `at_date`
    "macro_asof": """
        (at_date) AS TABLE
        SELECT series_id, date, value, realtime_start
        FROM macro_observations
        WHERE realtime_start <= at_date AND realtime_end >= at_date
    """,
    # Composició d'un univers vigent a `at_date`.
    # Font principal: intervals de pertinença (historial des de 1996 + fotos pròpies).
    # Sector/nom: de la foto pròpia més propera (anterior si n'hi ha; si no, la primera
    # posterior — classificació GICS actual, aproximació per a dates antigues).
    # Si no hi ha intervals per a l'univers, es fa servir la foto anterior més propera.
    "universe_asof": """
        (uni, at_date) AS TABLE
        WITH m AS (
            SELECT symbol, start_date, start_source FROM universe_intervals
            WHERE universe = uni AND start_date <= at_date
              AND (end_date IS NULL OR end_date > at_date)
        ), s AS (
            SELECT symbol, name, sector, sub_industry FROM universe_membership
            WHERE universe = uni
            QUALIFY row_number() OVER (
                PARTITION BY symbol
                ORDER BY (as_of_date <= at_date) DESC,
                         abs(date_diff('day', as_of_date, CAST(at_date AS DATE)))) = 1
        ), snap AS (
            SELECT * FROM universe_membership
            WHERE universe = uni
              AND as_of_date = (SELECT max(as_of_date) FROM universe_membership
                                WHERE universe = uni AND as_of_date <= at_date)
        )
        SELECT uni AS universe, CAST(at_date AS DATE) AS as_of_date, m.symbol, s.name,
               s.sector, s.sub_industry, m.start_date AS member_since,
               m.start_source AS history_source
        FROM m LEFT JOIN s USING (symbol)
        UNION ALL
        SELECT universe, CAST(at_date AS DATE), symbol, name, sector, sub_industry,
               date_added, 'snapshot'
        FROM snap
        WHERE NOT EXISTS (SELECT 1 FROM universe_intervals WHERE universe = uni)
    """,
}


def create_views(con: duckdb.DuckDBPyConnection, lake: Lake) -> None:
    con.execute("SET TimeZone='UTC'")
    for spec in REGISTRY.values():
        con.execute(f"CREATE OR REPLACE VIEW {spec.name}_raw AS {raw_view_sql(lake, spec)}")
        con.execute(f"CREATE OR REPLACE VIEW {spec.name} AS {dedup_view_sql(spec)}")
    for name, sql in DERIVED_VIEWS.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS {sql}")
    for name, sql in MACROS.items():
        con.execute(f"CREATE OR REPLACE MACRO {name}{sql}")


def connect(lake: Lake, database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database)
    create_views(con, lake)
    return con


def build_catalog_file(lake: Lake, path: Path) -> Path:
    """Regenera el fitxer de catàleg i el substitueix de forma atòmica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".duckdb.building")
    for p in (tmp, Path(str(tmp) + ".wal")):
        p.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        create_views(con, lake)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    os.replace(tmp, path)
    log.debug("catàleg regenerat a %s", path)
    return path
