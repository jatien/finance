"""Línia de comandes `fin`.

Exemples:
    fin init                         # crea directoris i catàleg
    fin ingest universe              # membres actuals del S&P 500 (Wikipedia)
    fin ingest universe-history      # historial de composició des de 1996
    fin ingest prices                # preus incrementals de tot l'univers
    fin ingest prices -s NVDA -s AMD # símbols concrets (també fora de l'univers)
    fin ingest options --force       # foto de cadenes encara que el mercat estigui tancat
    fin run daily                    # pipeline de tancament (el que executa systemd)
    fin status                       # estat del lake i dels últims jobs
    fin sql "SELECT * FROM job_status"
"""

from __future__ import annotations

import sys
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from finance.log import setup_logging
from finance.settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Sistema quant: dades i models")
ingest_app = typer.Typer(no_args_is_help=True, help="Jobs d'ingesta individuals")
run_app = typer.Typer(no_args_is_help=True, help="Pipelines compostos (per a systemd)")
app.add_typer(ingest_app, name="ingest")
app.add_typer(run_app, name="run")
console = Console()

Symbols = Annotated[list[str] | None, typer.Option("--symbol", "-s", help="Símbol (repetible)")]


def _ctx():
    from finance.jobs import Context
    s = get_settings()
    s.ensure_dirs()
    setup_logging(s.log_level, s.logs_dir, s.log_json)
    return Context.from_settings(s)


def _exit(reports) -> None:
    reports = reports if isinstance(reports, list) else [reports]
    for r in reports:
        style = {"success": "green", "skipped": "yellow", "partial": "yellow"}.get(r.status, "red")
        console.print(f"[{style}]{escape(r.summary())}[/{style}]")
    if any(r.status == "failed" for r in reports):
        raise typer.Exit(1)


# ---------------------------------------------------------------------------------------
@app.command()
def init() -> None:
    """Crea l'estructura del lake i el catàleg DuckDB."""
    ctx = _ctx()
    ctx.refresh_catalog()
    console.print(f"Lake: {ctx.settings.lake_dir}\nCatàleg: {ctx.settings.catalog_path}")


@ingest_app.command("prices")
def ingest_prices(
    symbols: Symbols = None,
    universe: str = typer.Option("prices", help="Univers definit a universe.yaml"),
    full: bool = typer.Option(False, help="Ignora watermarks i rebaixa tot l'històric"),
) -> None:
    """Preus diaris + accions corporatives (incremental)."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "prices", symbols=symbols, universe=universe, full=full))


@ingest_app.command("options")
def ingest_options(
    symbols: Symbols = None,
    force: bool = typer.Option(False, help="Fes la foto encara que el mercat estigui tancat"),
) -> None:
    """Foto de les cadenes d'opcions dels subjacents configurats."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "options", underlyings=symbols, force=force))


@ingest_app.command("macro")
def ingest_macro(
    series: Annotated[list[str] | None, typer.Option("--series", help="Sèrie FRED (repetible)")] = None,
    full: bool = typer.Option(False, help="Rebaixa tots els vintages"),
) -> None:
    """Sèries macro de FRED (amb vintages si hi ha clau API)."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "macro", series=series, full=full))


@ingest_app.command("universe")
def ingest_universe() -> None:
    """Membres d'univers dinàmics (S&P 500) i historial d'altes/baixes."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "universe"))


@ingest_app.command("universe-history")
def ingest_universe_history() -> None:
    """Historial de composició (S&P 500 des de 1996) combinat amb les fotos pròpies."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "universe_history"))


@ingest_app.command("instruments")
def ingest_instruments(symbols: Symbols = None) -> None:
    """Metadades d'instruments (sector, indústria, categoria de fons...)."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "instruments", symbols=symbols))


# ---------------------------------------------------------------------------------------
@run_app.command("daily")
def run_daily(full: bool = typer.Option(False)) -> None:
    """Tancament: univers (si cal) → preus → macro → qualitat → compactació."""
    from finance.ingest import daily_pipeline
    _exit(daily_pipeline(_ctx(), full=full))


@run_app.command("weekly")
def run_weekly() -> None:
    """Manteniment: univers, metadades, compactació completa."""
    from finance.ingest import weekly_pipeline
    _exit(weekly_pipeline(_ctx()))


@app.command()
def dq(
    dataset: Annotated[list[str] | None, typer.Option("--dataset", "-d")] = None,
    lookback_days: int = typer.Option(400, help="Finestra de preus a revisar (0 = tot l'històric)"),
) -> None:
    """Executa els controls de qualitat i mostra les violacions."""
    from finance.quality.checks import QualityJob
    ctx = _ctx()
    job = QualityJob(ctx)
    report = job.run(datasets=dataset, lookback_days=lookback_days)
    ctx.refresh_catalog()
    res = getattr(job, "results", pd.DataFrame())
    if not res.empty:
        _print_df(res.sort_values(["severity", "dataset", "check_name", "key"]).head(80), "Qualitat de dades")
    _exit(report)


@app.command()
def compact(
    dataset: Annotated[list[str] | None, typer.Option("--dataset", "-d")] = None,
    min_files: int = typer.Option(2, help="Compacta particions amb almenys N fitxers"),
) -> None:
    """Fusiona fitxers petits del lake (dedup per clau primària)."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "compact", datasets=dataset, min_files=min_files))


@app.command()
def status() -> None:
    """Estat del lake, darreres execucions i cobertura."""
    from finance.storage.schemas import REGISTRY
    ctx = _ctx()
    _print_df(pd.DataFrame([ctx.lake.stats(s) for s in REGISTRY.values()]), "Datasets")
    _print_df(ctx.query("SELECT * FROM job_status ORDER BY job"), "Darrera execució per job")
    _print_df(ctx.query("""
        SELECT count(*) AS symbols, min(first_date) AS first, max(last_date) AS last,
               sum(n_rows) AS rows FROM prices_coverage"""), "Preus")
    _print_df(ctx.query("""
        SELECT underlying, max(snapshot_ts) AS last_snapshot, count(DISTINCT snapshot_ts) AS snapshots
        FROM option_snapshots GROUP BY underlying ORDER BY underlying"""), "Opcions")
    _print_df(ctx.query("""
        SELECT dataset, severity, count(*) AS n FROM dq_results
        WHERE run_id = (SELECT max(run_id) FROM dq_results) GROUP BY ALL ORDER BY ALL"""),
              "Darrer control de qualitat")


@app.command()
def sql(query: str, limit: int = typer.Option(50)) -> None:
    """Consulta SQL sobre el catàleg (vistes deduplicades)."""
    ctx = _ctx()
    try:
        _print_df(ctx.query(query).head(limit), None)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def catalog() -> None:
    """Regenera catalog.duckdb (per a VSCode SQLTools / DBeaver / duckdb CLI)."""
    ctx = _ctx()
    ctx.refresh_catalog()
    console.print(str(ctx.settings.catalog_path))


def _print_df(df: pd.DataFrame, title: str | None) -> None:
    if df is None or df.empty:
        if title:
            console.print(f"[dim]{title}: (buit)[/dim]")
        return
    t = Table(title=title, show_lines=False)
    for c in df.columns:
        t.add_column(str(c), overflow="fold")
    for row in df.itertuples(index=False):
        t.add_row(*["" if pd.isna(v) else str(v) for v in row])
    console.print(t)


def main() -> None:  # pragma: no cover
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
