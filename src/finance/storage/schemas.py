"""Registre de datasets del lake: esquema tipat, clau primària, particionat i ordenació.

Principis:
- Tota fila porta columnes d'auditoria (`source`, `ingested_at`, `run_id`).
- Les escriptures són append-only; la deduplicació per clau primària (es queda la fila
  amb `ingested_at` més recent) la fan les vistes de DuckDB i la compactació.
- Les columnes de partició viuen al camí (estil Hive) i no dins el fitxer.
- Evolució d'esquema: afegir columnes és segur (`union_by_name`); canviar-ne el tipus
  requereix pujar `version` i una migració.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa

UTC_TS = pa.timestamp("us", tz="UTC")

AUDIT_FIELDS: list[pa.Field] = [
    pa.field("source", pa.string(), nullable=False),
    pa.field("ingested_at", UTC_TS, nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
]


@dataclass(frozen=True)
class Partition:
    column: str
    # Deriva la columna de partició a partir del DataFrame (None = ja hi és)
    derive: Callable[[pd.DataFrame], pd.Series] | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str  # relatiu a l'arrel del lake
    fields: list[pa.Field]
    primary_key: tuple[str, ...]
    partition: Partition | None = None
    sort_by: tuple[str, ...] = ()
    description: str = ""
    version: int = 1
    # Ordre per triar la "fila vigent" quan hi ha duplicats de clau
    dedup_order: tuple[str, ...] = ("ingested_at",)
    audit: bool = True
    tags: dict[str, str] = field(default_factory=dict)
    # Dataset que es regenera sencer a cada execució per aquestes claus: la vista només
    # conserva les files de l'última ingesta de cada grup (evita "fantasmes" si la font
    # corregeix l'històric).
    snapshot_by: tuple[str, ...] | None = None
    # Filtre SQL addicional de la vista (pot referenciar `<name>_raw`)
    view_filter: str | None = None

    @property
    def schema(self) -> pa.Schema:
        fs = list(self.fields) + (AUDIT_FIELDS if self.audit else [])
        meta = {b"dataset": self.name.encode(), b"schema_version": str(self.version).encode()}
        return pa.schema(fs, metadata=meta)

    @property
    def file_schema(self) -> pa.Schema:
        """Esquema dins dels fitxers Parquet (sense la columna de partició)."""
        s = self.schema
        if self.partition:
            s = s.remove(s.get_field_index(self.partition.column))
        return s

    @property
    def columns(self) -> list[str]:
        return self.schema.names


def _year_from(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(df[col]).dt.year.astype("int16")
    return f


# ---------------------------------------------------------------------------------------
# Definicions
# ---------------------------------------------------------------------------------------

PRICES_DAILY = DatasetSpec(
    name="prices_daily",
    path="prices_daily",
    description=(
        "Barres diàries OHLCV. `close` és ajustat només per splits (convenció Yahoo); "
        "`adj_close` també per dividends (es re-escriu tot l'històric quan hi ha un nou dividend)."
    ),
    fields=[
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("adj_close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("dividend", pa.float64()),
        pa.field("split_ratio", pa.float64()),
        pa.field("year", pa.int16(), nullable=False),
    ],
    primary_key=("symbol", "date"),
    partition=Partition("year", derive=_year_from("date")),
    sort_by=("symbol", "date"),
)

CORPORATE_ACTIONS = DatasetSpec(
    name="corporate_actions",
    path="corporate_actions",
    description="Dividends, splits i distribucions de guanys de capital per data ex.",
    fields=[
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("action", pa.string(), nullable=False),  # dividend | split | capital_gain
        pa.field("value", pa.float64(), nullable=False),
    ],
    primary_key=("symbol", "date", "action"),
    sort_by=("symbol", "date"),
)

OPTION_SNAPSHOTS = DatasetSpec(
    name="option_snapshots",
    path="option_snapshots",
    description=(
        "Fotos de cadenes d'opcions senceres. Una foto = totes les expiracions i strikes d'un "
        "subjacent a `snapshot_ts`. Base per a superfícies de volatilitat, skew i posicionament."
    ),
    fields=[
        pa.field("snapshot_ts", UTC_TS, nullable=False),  # instant lògic comú de la foto
        pa.field("snapshot_date", pa.date32(), nullable=False),
        pa.field("fetched_at", UTC_TS, nullable=False),  # instant real de la petició
        pa.field("underlying", pa.string(), nullable=False),
        pa.field("underlying_price", pa.float64()),
        pa.field("underlying_bid", pa.float64()),
        pa.field("underlying_ask", pa.float64()),
        pa.field("underlying_quote_ts", UTC_TS),
        pa.field("market_state", pa.string()),
        pa.field("contract_symbol", pa.string(), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),  # C | P
        pa.field("expiry", pa.date32(), nullable=False),
        pa.field("dte", pa.int32(), nullable=False),
        pa.field("strike", pa.float64(), nullable=False),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("mid", pa.float64()),
        pa.field("last_price", pa.float64()),
        pa.field("last_trade_ts", UTC_TS),
        pa.field("volume", pa.int64()),
        pa.field("open_interest", pa.int64()),
        pa.field("implied_vol", pa.float64()),  # IV del proveïdor (es recalcularà internament)
        pa.field("in_the_money", pa.bool_()),
        pa.field("contract_size", pa.string()),
        pa.field("currency", pa.string()),
    ],
    primary_key=("contract_symbol", "snapshot_ts"),
    partition=Partition("snapshot_date"),
    sort_by=("underlying", "snapshot_ts", "expiry", "option_type", "strike"),
)

MACRO_OBSERVATIONS = DatasetSpec(
    name="macro_observations",
    path="macro_observations",
    description=(
        "Observacions de sèries FRED amb vintages (ALFRED). Una observació és vàlida "
        "entre `realtime_start` i `realtime_end`: permet consultes point-in-time sense look-ahead."
    ),
    fields=[
        pa.field("series_id", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("value", pa.float64()),
        pa.field("realtime_start", pa.date32(), nullable=False),
        pa.field("realtime_end", pa.date32(), nullable=False),
        # alfred = vintages reals; observation_date = sèrie no revisable;
        # latest_only = sèrie revisable sense clau API (no és point-in-time!)
        pa.field("vintage_mode", pa.string(), nullable=False),
    ],
    primary_key=("series_id", "date", "realtime_start"),
    sort_by=("series_id", "date", "realtime_start"),
    # Si una sèrie ja té vintages reals, s'ignoren les files carregades sense clau API
    view_filter=(
        "coalesce(vintage_mode, '') = 'alfred' OR series_id NOT IN "
        "(SELECT DISTINCT series_id FROM macro_observations_raw WHERE vintage_mode = 'alfred')"
    ),
)

UNIVERSE_MEMBERSHIP = DatasetSpec(
    name="universe_membership",
    path="universe_membership",
    description="Fotos periòdiques dels membres d'univers dinàmics (p.ex. S&P 500) amb GICS.",
    fields=[
        pa.field("universe", pa.string(), nullable=False),
        pa.field("as_of_date", pa.date32(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("name", pa.string()),
        pa.field("sector", pa.string()),
        pa.field("sub_industry", pa.string()),
        pa.field("headquarters", pa.string()),
        pa.field("date_added", pa.date32()),
        pa.field("cik", pa.string()),
    ],
    primary_key=("universe", "as_of_date", "symbol"),
    sort_by=("universe", "as_of_date", "symbol"),
)

INDEX_CHANGES = DatasetSpec(
    name="index_changes",
    path="index_changes",
    description=(
        "Altes i baixes d'un índex derivades de l'historial de composició "
        "(una fila per símbol que entra o surt; l'altre costat és \"\")."
    ),
    fields=[
        pa.field("universe", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("added_symbol", pa.string(), nullable=False),  # "" si no n'hi ha
        pa.field("added_name", pa.string()),
        pa.field("removed_symbol", pa.string(), nullable=False),  # "" si no n'hi ha
        pa.field("removed_name", pa.string()),
        pa.field("reason", pa.string()),
    ],
    primary_key=("universe", "date", "added_symbol", "removed_symbol"),
    sort_by=("universe", "date"),
    snapshot_by=("universe",),
)

UNIVERSE_INTERVALS = DatasetSpec(
    name="universe_intervals",
    path="universe_intervals",
    description=(
        "Intervals de pertinença a un índex [start_date, end_date). Combina l'historial "
        "extern (des de 1996) amb les fotos pròpies posteriors. Base de `universe_asof` i "
        "de la correcció del biaix de supervivència."
    ),
    fields=[
        pa.field("universe", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("start_date", pa.date32(), nullable=False),
        pa.field("end_date", pa.date32()),  # exclusiu; NULL = encara és membre
        # True = ja era membre abans del començament de l'historial disponible
        pa.field("start_censored", pa.bool_(), nullable=False),
        # Dies d'incertesa: el canvi es va produir dins dels N dies anteriors (0 = data exacta).
        # La data guardada és la d'observació (conservadora: mai abans de saber-ho).
        pa.field("start_window_days", pa.int32(), nullable=False),
        pa.field("end_window_days", pa.int32()),
        pa.field("start_source", pa.string(), nullable=False),
        pa.field("end_source", pa.string()),
    ],
    primary_key=("universe", "symbol", "start_date"),
    sort_by=("universe", "symbol", "start_date"),
    snapshot_by=("universe",),
)

INSTRUMENTS = DatasetSpec(
    name="instruments",
    path="instruments",
    description="Metadades d'instruments (sector/indústria, categoria de fons, mida...) per data.",
    fields=[
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("as_of_date", pa.date32(), nullable=False),
        pa.field("name", pa.string()),
        pa.field("quote_type", pa.string()),
        pa.field("exchange", pa.string()),
        pa.field("currency", pa.string()),
        pa.field("country", pa.string()),
        pa.field("sector", pa.string()),
        pa.field("industry", pa.string()),
        pa.field("category", pa.string()),
        pa.field("fund_family", pa.string()),
        pa.field("market_cap", pa.float64()),
        pa.field("shares_outstanding", pa.float64()),
        pa.field("expense_ratio", pa.float64()),
        pa.field("total_assets", pa.float64()),
        pa.field("beta", pa.float64()),
        pa.field("first_trade_date", pa.date32()),
    ],
    primary_key=("symbol", "as_of_date"),
    sort_by=("symbol", "as_of_date"),
)

INGEST_RUNS = DatasetSpec(
    name="ingest_runs",
    path="_meta/ingest_runs",
    description="Registre de cada execució de job: paràmetres, resultat, errors, versió del codi.",
    fields=[
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("job", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),  # success | partial | failed | skipped
        pa.field("started_at", UTC_TS, nullable=False),
        pa.field("finished_at", UTC_TS, nullable=False),
        pa.field("duration_s", pa.float64()),
        pa.field("params", pa.string()),  # JSON
        pa.field("items_ok", pa.int64()),
        pa.field("items_failed", pa.int64()),
        pa.field("rows_written", pa.int64()),
        pa.field("errors", pa.string()),  # JSON {item: missatge}
        pa.field("message", pa.string()),
        pa.field("host", pa.string()),
        pa.field("code_version", pa.string()),
    ],
    primary_key=("run_id",),
    sort_by=("started_at",),
    dedup_order=("finished_at",),
    audit=False,
)

DQ_RESULTS = DatasetSpec(
    name="dq_results",
    path="_meta/dq_results",
    description="Resultats dels controls de qualitat de dades.",
    fields=[
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("checked_at", UTC_TS, nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("check_name", pa.string(), nullable=False),
        pa.field("key", pa.string(), nullable=False),  # símbol, sèrie o "*"
        pa.field("severity", pa.string(), nullable=False),  # info | warn | error
        pa.field("n_violations", pa.int64(), nullable=False),
        pa.field("details", pa.string()),
    ],
    primary_key=("run_id", "dataset", "check_name", "key"),
    sort_by=("checked_at",),
    dedup_order=("checked_at",),
    audit=False,
)

# ---------------------------------------------------------------------------------------
# Capa 2: features (derivades; es regeneren senceres a cada execució)
# ---------------------------------------------------------------------------------------

FEATURES_DAILY = DatasetSpec(
    name="features_daily",
    path="features/market_daily",
    description=(
        "Features d'estat de mercat (tendència, volatilitat, amplitud, crèdit, tipus, "
        "correlació) en format llarg. El valor de la data t només fa servir informació "
        "disponible al tancament de t. `pct_rank` és el percentil en finestra expansiva "
        "(sense look-ahead)."
    ),
    fields=[
        pa.field("feature_set", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("feature", pa.string(), nullable=False),
        pa.field("family", pa.string(), nullable=False),
        pa.field("value", pa.float64()),
        pa.field("pct_rank", pa.float64()),
        pa.field("year", pa.int16(), nullable=False),
    ],
    primary_key=("feature_set", "feature", "date"),
    partition=Partition("year", derive=_year_from("date")),
    sort_by=("feature_set", "feature", "date"),
    snapshot_by=("feature_set",),
)

SECTOR_FEATURES_DAILY = DatasetSpec(
    name="sector_features_daily",
    path="features/sector_daily",
    description=(
        "Panell de features per ETF de sector/indústria (momentum, volatilitat, força "
        "relativa, beta...). `xs_rank` és el percentil transversal entre els símbols "
        "disponibles aquell dia (base de la rotació sectorial)."
    ),
    fields=[
        pa.field("feature_set", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("group_name", pa.string(), nullable=False),
        pa.field("feature", pa.string(), nullable=False),
        pa.field("value", pa.float64()),
        pa.field("xs_rank", pa.float64()),
        pa.field("year", pa.int16(), nullable=False),
    ],
    primary_key=("feature_set", "symbol", "feature", "date"),
    partition=Partition("year", derive=_year_from("date")),
    sort_by=("feature_set", "symbol", "feature", "date"),
    snapshot_by=("feature_set",),
)

REGISTRY: dict[str, DatasetSpec] = {
    s.name: s
    for s in (
        PRICES_DAILY, CORPORATE_ACTIONS, OPTION_SNAPSHOTS, MACRO_OBSERVATIONS,
        UNIVERSE_MEMBERSHIP, INDEX_CHANGES, UNIVERSE_INTERVALS, INSTRUMENTS, INGEST_RUNS,
        DQ_RESULTS, FEATURES_DAILY, SECTOR_FEATURES_DAILY,
    )
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"dataset desconegut: {name}. Disponibles: {sorted(REGISTRY)}") from None


# ---------------------------------------------------------------------------------------
# Correspondència de tipus Arrow -> DuckDB
# ---------------------------------------------------------------------------------------

def duckdb_type(t: pa.DataType) -> str:
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "VARCHAR"
    if pa.types.is_boolean(t):
        return "BOOLEAN"
    if pa.types.is_int8(t) or pa.types.is_int16(t):
        return "SMALLINT"
    if pa.types.is_int32(t):
        return "INTEGER"
    if pa.types.is_int64(t):
        return "BIGINT"
    if pa.types.is_float32(t):
        return "FLOAT"
    if pa.types.is_float64(t):
        return "DOUBLE"
    if pa.types.is_date32(t):
        return "DATE"
    if pa.types.is_timestamp(t):
        return "TIMESTAMPTZ" if t.tz else "TIMESTAMP"
    raise TypeError(f"tipus Arrow no suportat: {t}")
