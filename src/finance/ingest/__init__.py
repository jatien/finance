"""Jobs d'ingesta i pipelines compostos."""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from finance.features.job import FeaturesJob
from finance.ingest.macro import MacroJob
from finance.ingest.maintenance import CompactJob
from finance.ingest.options import OptionsJob
from finance.ingest.prices import PricesJob
from finance.ingest.reference import InstrumentsJob, UniverseJob
from finance.ingest.universe_history import UniverseHistoryJob
from finance.jobs import Context, Job, RunReport
from finance.paper.job import PaperJob
from finance.paper.report import ReportJob
from finance.portfolio.job import PortfolioJob
from finance.quality.checks import QualityJob
from finance.regimes.job import RegimesJob

log = logging.getLogger(__name__)

JOBS: dict[str, type[Job]] = {
    j.name: j for j in (PricesJob, OptionsJob, MacroJob, UniverseJob, UniverseHistoryJob,
                        InstrumentsJob, QualityJob, CompactJob, FeaturesJob,
                        RegimesJob, PortfolioJob, PaperJob, ReportJob)
}

UNIVERSE_MAX_AGE = timedelta(days=7)


def run_job(ctx: Context, name: str, *, refresh_catalog: bool = True, **params) -> RunReport:
    report = JOBS[name](ctx).run(**params)
    if refresh_catalog and name != "compact":
        ctx.refresh_catalog()
    return report


def universe_is_stale(ctx: Context) -> bool:
    df = ctx.query("SELECT max(as_of_date) AS d FROM universe_membership")
    last = df["d"].iloc[0]
    return pd.isna(last) or (ctx.now().date() - pd.Timestamp(last).date()) > UNIVERSE_MAX_AGE


def daily_pipeline(ctx: Context, *, full: bool = False) -> list[RunReport]:
    """Pipeline de tancament: univers (si cal) → preus → macro → qualitat → features → règims → cartera → paper → compactació → informe."""
    reports = []
    if universe_is_stale(ctx):
        reports.append(run_job(ctx, "universe", refresh_catalog=False))
        reports.append(run_job(ctx, "universe_history", refresh_catalog=False))
    reports.append(run_job(ctx, "prices", refresh_catalog=False, full=full))
    reports.append(run_job(ctx, "macro", refresh_catalog=False, full=full))
    reports.append(run_job(ctx, "dq", refresh_catalog=False))
    # Les features llegeixen el lake a través del catàleg en memòria (sempre al dia)
    reports.append(run_job(ctx, "features", refresh_catalog=False))
    reports.append(run_job(ctx, "regimes", refresh_catalog=False))
    reports.append(run_job(ctx, "portfolio", refresh_catalog=False))
    # Paper trading: només avança fins a l'última sessió amb preus de tots els actius
    reports.append(run_job(ctx, "paper", refresh_catalog=False))
    reports.append(run_job(ctx, "compact", min_files=8))
    reports.append(run_job(ctx, "report", refresh_catalog=False))
    return reports


def weekly_pipeline(ctx: Context) -> list[RunReport]:
    """Manteniment setmanal: univers, historial de composició, metadades, compactació completa."""
    return [
        run_job(ctx, "universe", refresh_catalog=False),
        run_job(ctx, "universe_history", refresh_catalog=False),
        run_job(ctx, "instruments", refresh_catalog=False),
        run_job(ctx, "compact", min_files=2),
    ]


__all__ = ["JOBS", "Context", "RunReport", "daily_pipeline", "run_job", "weekly_pipeline"]
