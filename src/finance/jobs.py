"""Infraestructura comuna dels jobs (ingesta, qualitat, models): context, registre d'execucions, concurrència."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

import duckdb
import pandas as pd

from finance import __version__
from finance.providers import ProviderError, Providers, TransientProviderError
from finance.settings import REPO_ROOT, Settings
from finance.storage import Lake, build_catalog_file, connect
from finance.storage.schemas import INGEST_RUNS, DatasetSpec
from finance.universe import Universe

log = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")


def code_version() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{__version__}+{sha}{'.dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return __version__


@dataclass
class Context:
    settings: Settings
    lake: Lake
    providers: Providers
    clock: Callable[[], pd.Timestamp] = field(default=lambda: pd.Timestamp.now(tz="UTC"))
    _universe: Universe | None = None

    @classmethod
    def from_settings(cls, settings: Settings, providers: Providers | None = None) -> Context:
        from finance.providers import build_providers
        settings.ensure_dirs()
        return cls(settings=settings, lake=Lake(settings.lake_dir),
                   providers=providers or build_providers(settings))

    def now(self) -> pd.Timestamp:
        return self.clock()

    def con(self) -> duckdb.DuckDBPyConnection:
        """Connexió nova en memòria amb les vistes sobre l'estat actual del lake."""
        return connect(self.lake)

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        con = self.con()
        try:
            return con.execute(sql, params or []).df()
        finally:
            con.close()

    @property
    def universe(self) -> Universe:
        if self._universe is None:
            self._universe = Universe.load(self.settings.universe_file, resolver=self._resolve_dynamic)
        return self._universe

    def _resolve_dynamic(self, name: str) -> list[dict[str, Any]]:
        df = self.query("SELECT symbol, name, sector FROM universe_current WHERE universe = ?", [name])
        if df.empty:
            raise RuntimeError(f"univers dinàmic '{name}' sense dades: executa `fin ingest universe`")
        return df.to_dict("records")

    def refresh_catalog(self) -> None:
        build_catalog_file(self.lake, self.settings.catalog_path)


@dataclass
class Run:
    """Estat d'una execució: s'acumula durant el job i es persisteix a `ingest_runs`."""
    job: str
    run_id: str
    ingested_at: pd.Timestamp
    lake: Lake
    params: dict[str, Any]
    items_ok: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    rows_written: int = 0
    message: str = ""
    skipped: bool = False

    def write(self, spec: DatasetSpec, df: pd.DataFrame, source: str) -> int:
        res = self.lake.write(spec, df, source=source, run_id=self.run_id,
                              ingested_at=self.ingested_at.to_pydatetime())
        self.rows_written += res.rows
        return res.rows

    def ok(self, n: int = 1) -> None:
        self.items_ok += n

    def fail(self, item: str, err: str | BaseException) -> None:
        self.errors[item] = str(err)[:500]

    def skip(self, reason: str) -> None:
        self.skipped = True
        self.message = reason

    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        if self.errors and self.items_ok == 0:
            return "failed"
        if self.errors:
            return "partial"
        return "success"


@dataclass
class RunReport:
    job: str
    run_id: str
    status: str
    items_ok: int
    items_failed: int
    rows_written: int
    duration_s: float
    message: str
    errors: dict[str, str]

    def summary(self) -> str:
        s = (f"[{self.job}] {self.status}: {self.items_ok} ok, {self.items_failed} errors, "
             f"{self.rows_written} files en {self.duration_s:.1f}s")
        return s + (f" — {self.message}" if self.message else "")


class Job(ABC):
    name: str = "job"

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.log = logging.getLogger(f"finance.ingest.{self.name}")

    @abstractmethod
    def execute(self, run: Run, **params: Any) -> None: ...

    def run(self, **params: Any) -> RunReport:
        started = self.ctx.now()
        run_id = f"{self.name}-{started.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run = Run(job=self.name, run_id=run_id, ingested_at=started, lake=self.ctx.lake,
                  params={k: v for k, v in params.items() if v is not None})
        t0 = time.monotonic()
        self.log.info("inici %s %s", run_id, json.dumps(run.params, default=str))
        fatal: BaseException | None = None
        try:
            self.execute(run, **params)
        except Exception as e:  # noqa: BLE001
            fatal = e
            run.fail("__job__", f"{type(e).__name__}: {e}")
            self.log.exception("job %s ha fallat", self.name)
        duration = time.monotonic() - t0
        status = "failed" if fatal else run.status
        report = RunReport(job=self.name, run_id=run_id, status=status, items_ok=run.items_ok,
                           items_failed=len(run.errors), rows_written=run.rows_written,
                           duration_s=duration, message=run.message, errors=dict(run.errors))
        self._record(run, report, started)
        log_fn = self.log.info if status in ("success", "skipped") else self.log.warning
        log_fn(report.summary())
        for item, err in list(run.errors.items())[:20]:
            self.log.warning("  %s: %s", item, err)
        return report

    def _record(self, run: Run, report: RunReport, started: pd.Timestamp) -> None:
        row = pd.DataFrame([{
            "run_id": run.run_id, "job": self.name, "status": report.status,
            "started_at": started, "finished_at": pd.Timestamp.now(tz="UTC"),
            "duration_s": report.duration_s, "params": json.dumps(run.params, default=str),
            "items_ok": report.items_ok, "items_failed": report.items_failed,
            "rows_written": report.rows_written, "errors": json.dumps(report.errors),
            "message": report.message, "host": socket.gethostname(),
            "code_version": code_version(),
        }])
        try:
            self.ctx.lake.write(INGEST_RUNS, row)
        except Exception:  # noqa: BLE001
            self.log.exception("no s'ha pogut registrar l'execució")


def run_concurrently(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    key: Callable[[T], str] = str,
) -> Iterable[tuple[T, R | None, str | None]]:
    """Executa `fn` en paral·lel i retorna (item, resultat, error) a mesura que acaben.

    Els errors de proveïdor es capturen per ítem: un símbol dolent no atura el job.
    """
    items = list(items)
    if not items:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, it): it for it in items}
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                yield it, fut.result(), None
            except (ProviderError, TransientProviderError) as e:
                yield it, None, str(e)
            except Exception as e:  # noqa: BLE001
                log.exception("error inesperat a %s", key(it))
                yield it, None, f"{type(e).__name__}: {e}"


def utc_now() -> datetime:
    return datetime.now(UTC)
