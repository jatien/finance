"""Data lake en Parquet: escriptures atòmiques append-only, locks per dataset i compactació.

Disseny (estil LSM):
- Cada escriptura crea fitxers nous i immutables. Mai es modifica un fitxer existent.
- Un fitxer s'escriu com a `*.parquet.tmp` i es fa visible amb `os.replace` (atòmic en POSIX).
- Els lectors deduplicuen per clau primària (vistes DuckDB), així que és segur llegir
  mentre s'escriu o es compacta.
- La compactació fusiona els fitxers d'una partició en un de sol (dedup + ordenació).
  Agafa un lock exclusiu; les escriptures n'agafen un de compartit.
"""

from __future__ import annotations

import fcntl
import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finance.storage.schemas import DatasetSpec, duckdb_type

log = logging.getLogger(__name__)

PARQUET_OPTS = dict(compression="zstd", compression_level=3, row_group_size=256_000,
                    write_statistics=True, use_dictionary=True)


@dataclass
class WriteResult:
    dataset: str
    rows: int = 0
    files: list[Path] = field(default_factory=list)


@dataclass
class CompactResult:
    dataset: str
    partitions: int = 0
    files_before: int = 0
    files_after: int = 0
    rows_before: int = 0
    rows_after: int = 0


class SchemaError(ValueError):
    pass


def _partition_value_str(v: object) -> str:
    if isinstance(v, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(v).date().isoformat()
    return str(v)


def _to_date(x: object) -> date | None:
    if x is None or x is pd.NaT or (not isinstance(x, (str, date)) and pd.isna(x)):
        return None
    if isinstance(x, datetime):  # inclou pd.Timestamp
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, np.datetime64):
        return pd.Timestamp(x).date()
    txt = str(x).strip()
    return date.fromisoformat(txt[:10]) if txt else None


def _coerce_column(s: pd.Series, t: pa.DataType) -> pd.Series:
    """Converteix una columna pandas al tipus Arrow de destí de forma robusta."""
    if pa.types.is_date32(t):
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.dt.date.astype("object").where(s.notna(), None)
        # Element a element: suporta dates fora de rang de datetime64 (p.ex. 9999-12-31)
        return s.map(_to_date).astype("object")
    if pa.types.is_timestamp(t):
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        return ts
    if pa.types.is_integer(t):
        return pd.to_numeric(s, errors="coerce").round().astype("Int64")
    if pa.types.is_floating(t):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    if pa.types.is_boolean(t):
        return s.astype("boolean")
    if pa.types.is_string(t):
        return s.astype("string")
    return s


class Lake:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "_locks").mkdir(exist_ok=True)

    # ---------------------------------------------------------------------------------
    # Rutes
    # ---------------------------------------------------------------------------------
    def dataset_dir(self, spec: DatasetSpec) -> Path:
        return self.root / spec.path

    def glob(self, spec: DatasetSpec) -> str:
        base = self.dataset_dir(spec)
        return str(base / "**" / "*.parquet") if spec.partition else str(base / "*.parquet")

    def files(self, spec: DatasetSpec) -> list[Path]:
        base = self.dataset_dir(spec)
        if not base.exists():
            return []
        pattern = "*/*.parquet" if spec.partition else "*.parquet"
        return sorted(base.glob(pattern))

    def partitions(self, spec: DatasetSpec) -> dict[Path, list[Path]]:
        out: dict[Path, list[Path]] = {}
        for f in self.files(spec):
            out.setdefault(f.parent, []).append(f)
        return out

    # ---------------------------------------------------------------------------------
    # Locks
    # ---------------------------------------------------------------------------------
    @contextmanager
    def lock(self, spec: DatasetSpec, exclusive: bool) -> Iterator[None]:
        path = self.root / "_locks" / f"{spec.name}.lock"
        with open(path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # ---------------------------------------------------------------------------------
    # Validació d'esquema
    # ---------------------------------------------------------------------------------
    def conform(
        self,
        spec: DatasetSpec,
        df: pd.DataFrame,
        *,
        source: str | None = None,
        run_id: str | None = None,
        ingested_at: datetime | None = None,
    ) -> pd.DataFrame:
        """Valida i normalitza un DataFrame a l'esquema del dataset (incloent auditoria)."""
        df = df.copy()
        if spec.audit:
            if source is None or run_id is None:
                raise SchemaError(f"{spec.name}: cal `source` i `run_id`")
            df["source"] = source
            df["run_id"] = run_id
            df["ingested_at"] = pd.Timestamp(ingested_at or datetime.now(UTC))

        schema = spec.schema
        required = [f.name for f in schema if not f.nullable]
        derived = spec.partition.column if spec.partition and spec.partition.derive else None
        missing_required = [c for c in required if c not in df.columns and c != derived]
        if missing_required:
            raise SchemaError(f"{spec.name}: falten columnes obligatòries {missing_required}")
        if derived and derived not in df:
            df[derived] = spec.partition.derive(df)

        for f in schema:
            if f.name not in df.columns:
                df[f.name] = pd.Series([None] * len(df), dtype="object")
            df[f.name] = _coerce_column(df[f.name], f.type)

        nulls = {c: int(df[c].isna().sum()) for c in required if df[c].isna().any()}
        if nulls:
            raise SchemaError(f"{spec.name}: nuls a columnes obligatòries {nulls}")

        df = df[schema.names]
        before = len(df)
        df = df.drop_duplicates(subset=list(spec.primary_key), keep="last")
        if len(df) < before:
            log.debug("%s: %d duplicats dins el lot eliminats", spec.name, before - len(df))
        return df.reset_index(drop=True)

    def to_arrow(self, spec: DatasetSpec, df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
        try:
            return pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            raise SchemaError(f"{spec.name}: conversió a Arrow fallida: {e}") from e

    # ---------------------------------------------------------------------------------
    # Escriptura
    # ---------------------------------------------------------------------------------
    def write(
        self,
        spec: DatasetSpec,
        df: pd.DataFrame,
        *,
        source: str | None = None,
        run_id: str | None = None,
        ingested_at: datetime | None = None,
    ) -> WriteResult:
        result = WriteResult(dataset=spec.name)
        if df is None or df.empty:
            return result
        df = self.conform(spec, df, source=source, run_id=run_id, ingested_at=ingested_at)
        if spec.sort_by:
            df = df.sort_values(list(spec.sort_by), kind="stable")

        base = self.dataset_dir(spec)
        tag = (run_id or "norun").replace("/", "_")[:12]
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")

        if spec.partition:
            groups = df.groupby(spec.partition.column, sort=True, dropna=False)
        else:
            groups = [(None, df)]

        with self.lock(spec, exclusive=False):
            for pval, part in groups:
                if spec.partition:
                    pval = pval[0] if isinstance(pval, tuple) else pval
                    target_dir = base / f"{spec.partition.column}={_partition_value_str(pval)}"
                else:
                    target_dir = base
                target_dir.mkdir(parents=True, exist_ok=True)
                table = self.to_arrow(spec, part, spec.file_schema)
                final = target_dir / f"{stamp}-{tag}-{uuid.uuid4().hex[:8]}.parquet"
                self._atomic_write(table, final)
                result.rows += table.num_rows
                result.files.append(final)
        log.debug("%s: %d files, %d fitxers", spec.name, result.rows, len(result.files))
        return result

    @staticmethod
    def _atomic_write(table: pa.Table, final: Path) -> None:
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, **PARQUET_OPTS)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, final)

    # ---------------------------------------------------------------------------------
    # Lectura directa (per a jobs; l'anàlisi ha de passar pel catàleg DuckDB)
    # ---------------------------------------------------------------------------------
    def _select_list(self, spec: DatasetSpec, con: duckdb.DuckDBPyConnection, src: str,
                     schema: pa.Schema) -> str:
        present = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
        cols = []
        for f in schema:
            t = duckdb_type(f.type)
            if f.name in present:
                cols.append(f'CAST("{f.name}" AS {t}) AS "{f.name}"')
            else:
                cols.append(f'CAST(NULL AS {t}) AS "{f.name}"')
        return ", ".join(cols)

    # ---------------------------------------------------------------------------------
    # Compactació
    # ---------------------------------------------------------------------------------
    def compact(self, spec: DatasetSpec, min_files: int = 2) -> CompactResult:
        res = CompactResult(dataset=spec.name)
        with self.lock(spec, exclusive=True):
            for part_dir, files in self.partitions(spec).items():
                if len(files) < min_files:
                    continue
                self._compact_partition(spec, part_dir, files, res)
        if res.partitions:
            log.info("compactat %s: %d particions, %d→%d fitxers, %d→%d files",
                     spec.name, res.partitions, res.files_before, res.files_after,
                     res.rows_before, res.rows_after)
        return res

    def _compact_partition(self, spec: DatasetSpec, part_dir: Path, files: list[Path],
                           res: CompactResult) -> None:
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        try:
            file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
            src = f"read_parquet({file_list}, union_by_name=true, hive_partitioning=false)"
            select = self._select_list(spec, con, src, spec.file_schema)
            pk = ", ".join(f'"{c}"' for c in spec.primary_key)
            order = ", ".join(f'"{c}" DESC' for c in spec.dedup_order)
            sort = ", ".join(f'"{c}"' for c in spec.sort_by if c in spec.file_schema.names)
            n_before = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
            sql = (
                f"SELECT {select} FROM {src} "
                f"QUALIFY row_number() OVER (PARTITION BY {pk} ORDER BY {order}) = 1"
                + (f" ORDER BY {sort}" if sort else "")
            )
            table = con.execute(sql).to_arrow_table()
        finally:
            con.close()
        table = table.cast(spec.file_schema)
        final = part_dir / f"compact-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}.parquet"
        self._atomic_write(table, final)
        # Primer el fitxer nou, després esborrar els vells: mai hi ha finestra sense dades
        for f in files:
            f.unlink(missing_ok=True)
        res.partitions += 1
        res.files_before += len(files)
        res.files_after += 1
        res.rows_before += n_before
        res.rows_after += table.num_rows

    # ---------------------------------------------------------------------------------
    # Estadístiques
    # ---------------------------------------------------------------------------------
    def stats(self, spec: DatasetSpec) -> dict[str, object]:
        files = self.files(spec)
        size = sum(f.stat().st_size for f in files)
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        return {"dataset": spec.name, "files": len(files), "partitions": len(self.partitions(spec)),
                "rows_raw": rows, "size_mb": round(size / 1e6, 2)}
