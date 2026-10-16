"""Línia de comandes `fin`.

Exemples:
    fin init                         # crea directoris i catàleg
    fin ingest universe              # membres actuals del S&P 500 (Wikipedia)
    fin ingest universe-history      # historial de composició des de 1996
    fin ingest prices                # preus incrementals de tot l'univers
    fin ingest prices -s NVDA -s AMD # símbols concrets (també fora de l'univers)
    fin ingest options --force       # foto de cadenes encara que el mercat estigui tancat
    fin run daily                    # pipeline de tancament (el que executa systemd)
    fin features build               # recalcula les features (capa 2) i les escriu al lake
    fin features show                # estat actual del mercat segons les features
    fin features report              # informe HTML: correlació de Spearman, cobertura, sèries
    fin regimes build                # classifica el règim de mercat de cada sessió (capa 3)
    fin regimes show                 # règim actual, exposició recomanada i últims canvis
    fin regimes report               # informe HTML de validació (estadístiques, backtest, sensibilitat)
    fin portfolio build              # cartera objectiu i registre de decisions (capes 4-5)
    fin portfolio show               # cartera objectiu actual i últims canvis suggerits
    fin portfolio report             # informe HTML: walk-forward contra referències i ablacions
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
features_app = typer.Typer(no_args_is_help=True, help="Capa 2: features de mercat i sectorials")
app.add_typer(run_app, name="run")
app.add_typer(features_app, name="features")
regimes_app = typer.Typer(no_args_is_help=True, help="Capa 3: règims de mercat")
app.add_typer(regimes_app, name="regimes")
portfolio_app = typer.Typer(no_args_is_help=True, help="Capes 4-5: rotació sectorial i cartera objectiu")
app.add_typer(portfolio_app, name="portfolio")
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


@features_app.command("build")
def features_build(
    start: str | None = typer.Option(None, help="Data inicial dels preus (per defecte, tot)"),
    no_sp500: bool = typer.Option(False, "--no-sp500", help="Omet l'amplitud del S&P 500"),
) -> None:
    """Calcula features de mercat i panell sectorial i les escriu al lake."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "features", start=start, include_sp500=not no_sp500))


@features_app.command("show")
def features_show(
    feature_set: str = typer.Option("core_v1", "--set"),
    group: str = typer.Option("sectors", help="Grup del panell: sectors | industries"),
) -> None:
    """Últim valor i percentil històric de cada feature + rànquing sectorial."""
    from finance.features.market import CATALOG
    ctx = _ctx()
    df = ctx.query("""
        SELECT feature, family, date, value, pct_rank FROM features_daily
        WHERE feature_set = ?
        QUALIFY date = max(date) OVER (PARTITION BY feature)""", [feature_set])
    if df.empty:
        console.print("[yellow]sense features: executa `fin features build`[/yellow]")
        raise typer.Exit(1)
    order = {k: i for i, k in enumerate(CATALOG)}
    df = df.sort_values("feature", key=lambda s: s.map(order))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = df["value"].round(4)
    df["pct_rank"] = (100 * df["pct_rank"]).round(0)
    _print_df(df.rename(columns={"pct_rank": "percentil"}), f"Estat de mercat ({feature_set})")
    rank = ctx.query("""
        SELECT symbol, feature, value, xs_rank FROM sector_features_daily
        WHERE feature_set = ? AND group_name = ?
          AND feature IN ('risk_adj_mom', 'mom_12_1', 'rel_strength_3m', 'dist_sma200')
        QUALIFY date = max(date) OVER ()""", [feature_set, group])
    if not rank.empty:
        wide = rank.pivot(index="symbol", columns="feature", values="value").round(3)
        wide = wide.sort_values("risk_adj_mom", ascending=False).reset_index()
        _print_df(wide, f"Panell {group} (ordenat per momentum ajustat per risc)")


@features_app.command("report")
def features_report(
    feature_set: str = typer.Option("core_v1", "--set"),
    out: str | None = typer.Option(None, help="Fitxer HTML de sortida"),
) -> None:
    """Informe HTML: redundàncies (Spearman), cobertura i sèries clau."""
    from pathlib import Path

    from finance.features.diagnostics import build_report
    from finance.features.job import read_market_features
    ctx = _ctx()
    feats = read_market_features(ctx, feature_set, "value")
    if feats.empty:
        console.print("[yellow]sense features: executa `fin features build`[/yellow]")
        raise typer.Exit(1)
    pct = read_market_features(ctx, feature_set, "pct_rank")
    msg = ctx.query("""SELECT message FROM ingest_runs WHERE job = 'features'
                      ORDER BY started_at DESC LIMIT 1""")
    notes = [] if msg.empty or not msg["message"].iloc[0] else msg["message"].iloc[0].split("; ")[1:]
    path = Path(out) if out else ctx.settings.data_dir / "reports" / f"features_{feature_set}.html"
    build_report(feats, pct, path, feature_set=feature_set, notes=notes)
    console.print(str(path))


@regimes_app.command("build")
def regimes_build() -> None:
    """Classifica cada sessió (tendència × volatilitat + crisi) i ho escriu al lake."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "regimes"))


@regimes_app.command("show")
def regimes_show(regime_set: str = typer.Option("rules_v1", "--set")) -> None:
    """Règim actual, exposició recomanada i últims canvis de règim."""
    from finance.regimes.evaluation import episodes
    from finance.regimes.job import read_regimes
    ctx = _ctx()
    reg = read_regimes(ctx, regime_set)
    if reg.empty:
        console.print("[yellow]sense règims: executa `fin regimes build`[/yellow]")
        raise typer.Exit(1)
    last = reg.iloc[-1]
    console.print(f"[bold]{reg.index[-1]:%Y-%m-%d}[/bold]  règim [bold]{last['regime']}[/bold] "
                  f"(tendència {last['trend']}, volatilitat {last['vol']}, crisi {bool(last['crisis'])}) · "
                  f"{int(last['days_in_regime'])} sessions · exposició recomanada "
                  f"[bold]{last['equity_exposure']:.0%}[/bold] · acció: {last['action']}")
    ep = episodes(reg["regime"]).tail(10).iloc[::-1]
    ep["start"] = ep["start"].dt.date
    ep["end"] = ep["end"].dt.date
    ep["exposure"] = ep["regime"].map(reg.groupby("regime")["equity_exposure"].first())
    _print_df(ep, "Últims episodis")


@regimes_app.command("report")
def regimes_report(
    out: str | None = typer.Option(None, help="Fitxer HTML de sortida"),
    cost_bps: float = typer.Option(5.0, help="Cost per unitat de rotació (punts bàsics)"),
) -> None:
    """Informe de validació: estadístiques per règim, backtest, ablacions i sensibilitat."""
    from pathlib import Path

    from finance.regimes.config import RegimeConfig
    from finance.regimes.job import load_eval_inputs
    from finance.regimes.report import build_report
    ctx = _ctx()
    cfg = RegimeConfig()
    path = Path(out) if out else ctx.settings.data_dir / "reports" / f"regimes_{cfg.name}.html"
    build_report(load_eval_inputs(ctx, cfg), cfg, path, cost_bps)
    console.print(str(path))


@portfolio_app.command("build")
def portfolio_build(
    strategy_set: Annotated[list[str] | None, typer.Option("--set", help="Estratègia (repetible); per defecte, totes les actives")] = None,
) -> None:
    """Calcula la cartera objectiu de les estratègies actives i escriu el registre de decisions."""
    from finance.ingest import run_job
    _exit(run_job(_ctx(), "portfolio", strategies=strategy_set))


@portfolio_app.command("show")
def portfolio_show(strategy_set: str | None = typer.Option(None, "--set", help="Per defecte, la de referència"),
                   n: int = typer.Option(3, help="Revisions amb canvis a mostrar")) -> None:
    """Cartera objectiu actual i últims canvis suggerits."""
    from finance.portfolio.job import read_targets
    ctx = _ctx()
    df = read_targets(ctx, strategy_set)
    if df.empty:
        console.print("[yellow]sense cartera: executa `fin portfolio build`[/yellow]")
        raise typer.Exit(1)
    last = df[df["date"] == df["date"].max()]
    head = last.iloc[0]
    console.print(f"[bold]{head['strategy_set']}[/bold] · [bold]{head['date']:%Y-%m-%d}[/bold]  règim [bold]{head['regime']}[/bold] · "
                  f"renda variable {head['equity_exposure']:.0%}"
                  + (f" · cobertura {head['hedge']}" if head["equity_exposure"] < 1 else ""))
    cur = last[last["target_weight"] > 0][["symbol", "sleeve", "target_weight", "action"]].copy()
    cur["target_weight"] = cur["target_weight"].map("{:.1%}".format)
    _print_df(cur, "Cartera objectiu")
    ch = df[df["action"] != "hold"]
    dates = sorted(ch["date"].unique())[-n:][::-1]
    ch = ch[ch["date"].isin(dates)].sort_values(["date", "change"], ascending=[False, False]).copy()
    ch["date"] = ch["date"].dt.date
    for c in ("prev_weight", "target_weight", "change"):
        ch[c] = ch[c].map("{:+.1%}".format if c == "change" else "{:.1%}".format)
    _print_df(ch[["date", "regime", "symbol", "action", "prev_weight", "target_weight", "change"]],
              "Últims canvis suggerits")


@portfolio_app.command("report")
def portfolio_report(
    out: str | None = typer.Option(None, help="Fitxer HTML de sortida"),
    no_sensitivity: bool = typer.Option(False, "--no-sensitivity", help="Omet la graella (més ràpid)"),
) -> None:
    """Informe walk-forward: referències, ablacions, episodis, assignació i sensibilitat."""
    from pathlib import Path

    from finance.portfolio.config import ROT_V1, ROT_V1B, ROT_V2, ROT_V2B
    from finance.portfolio.job import load_inputs
    from finance.portfolio.report import build_report
    ctx = _ctx()
    cfg = ROT_V1
    path = Path(out) if out else ctx.settings.data_dir / "reports" / f"portfolio_{cfg.name}.html"
    # Les entrades de rot_v2 són un superconjunt (correlació accions-bons ràpida)
    build_report(load_inputs(ctx, ROT_V2), cfg, path, with_sensitivity=not no_sensitivity, cfg_v2=ROT_V2,
                 band_pairs=[(ROT_V1, ROT_V1B), (ROT_V2, ROT_V2B)])
    console.print(str(path))


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
