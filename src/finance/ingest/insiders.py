"""Ingesta d'operacions d'insiders (SEC Form 4/5).

1. Datasets trimestrals: es carreguen els trimestres disponibles des de
   `insider_start_year` que encara no són al lake (`source_file`). Tots els emissors.
2. Cua diària: per als dies posteriors a l'últim trimestre publicat es llegeix l'índex
   diari d'EDGAR i es baixen només les presentacions d'emissors de l'univers (CIK del
   S&P 500), per limitar les peticions. Quan surt el trimestre, el TSV substitueix
   aquestes presentacions a la vista (`snapshot_by` per accession).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from finance.jobs import Job, Run, run_concurrently
from finance.providers.sec import TRANSACTION_FORMS, DisabledInsiders
from finance.storage.schemas import INSIDER_TRANSACTIONS


def quarter_end(year: int, quarter: int) -> date:
    first_next = date(year + (quarter == 4), 1 if quarter == 4 else 3 * quarter + 1, 1)
    return first_next - timedelta(days=1)


class InsidersJob(Job):
    name = "insiders"

    def execute(self, run: Run, full: bool = False, quarters: list[str] | None = None,
                daily: bool = True, max_days: int | None = None, all_issuers: bool = False,
                **_: Any) -> None:
        ctx = self.ctx
        provider = ctx.providers.insiders
        if provider is None or isinstance(provider, DisabledInsiders):
            run.skip(getattr(provider, "reason", "sense proveïdor d'insiders"))
            return

        loaded = set(ctx.query("SELECT DISTINCT source_file FROM insider_transactions")["source_file"])
        available = provider.list_quarters()
        start_year = ctx.settings.insider_start_year
        if quarters:
            wanted = {q.lower() for q in quarters}
            targets = [q for q in available if q.key in wanted]
        else:
            targets = [q for q in available if q.year >= start_year and (full or q.key not in loaded)]

        for q in targets:
            try:
                df = provider.fetch_quarter(q)
            except Exception as e:  # noqa: BLE001
                run.fail(q.key, f"{type(e).__name__}: {e}")
                continue
            n = run.write(INSIDER_TRANSACTIONS, df, provider.name)
            self.log.info("insiders %s: %d operacions", q.key, n)
            run.ok()

        if not daily or not available:
            return
        last = available[-1]
        first_day = quarter_end(last.year, last.quarter) + timedelta(days=1)
        # L'índex d'un dia només és complet l'endemà: fins ahir (dia natural a Nova York)
        today = ctx.now().tz_convert("America/New_York").date()
        limit = max_days if max_days is not None else ctx.settings.insider_max_daily_days
        days = [d.date() for d in pd.date_range(first_day, today - timedelta(days=1), freq="B")][-limit:]
        if not days:
            return
        self._daily(run, provider, days, all_issuers)

    def _issuer_ciks(self) -> set[str]:
        df = self.ctx.query("""
            SELECT DISTINCT cik FROM universe_membership WHERE cik IS NOT NULL AND cik <> ''
        """)
        return {str(c).strip().zfill(10) for c in df["cik"]}

    def _daily(self, run: Run, provider, days: list[date], all_issuers: bool) -> None:
        ciks = None if all_issuers else self._issuer_ciks()
        if ciks is not None and not ciks:
            run.message = "sense CIK de l'univers: executa `fin ingest universe` (cua diària omesa)"
            return
        # Dies amb índex publicat (festius fora); si no es pot saber, es proven tots
        published: set[date] | None = set()
        for y, q in sorted({(d.year, (d.month - 1) // 3 + 1) for d in days}):
            try:
                got = provider.index_days(y, q)
            except Exception as e:  # noqa: BLE001
                self.log.warning("llista de l'índex %sQ%s no disponible: %s", y, q, e)
                got = None
            if got is None:
                published = None
                break
            published |= got
        if published is not None:
            days = [d for d in days if d in published]
        known = set(self.ctx.query(
            "SELECT DISTINCT accession_number FROM insider_transactions WHERE filing_date >= ?",
            [days[0]])["accession_number"])
        for day in days:
            try:
                idx = provider.daily_index(day)
            except Exception as e:  # noqa: BLE001
                # 404 = festiu o dia sense índex: no és un error
                if "no existeix" not in str(e):
                    run.fail(f"index {day}", e)
                continue
            idx = idx[idx["form_type"].isin(TRANSACTION_FORMS)]
            if ciks is not None:
                idx = idx[idx["cik"].isin(ciks)]
            idx = idx.drop_duplicates("accession_number")
            idx = idx[~idx["accession_number"].isin(known)]
            if idx.empty:
                continue
            frames = []
            items = list(idx.itertuples(index=False))
            for it, df, err in run_concurrently(
                    lambda r, d=day: provider.fetch_filing(r.path, r.accession_number, d),
                    items, workers=3, key=lambda r: r.accession_number):
                if err is not None:
                    run.fail(it.accession_number, err)
                    continue
                if df is not None and not df.empty:
                    frames.append(df)
                known.add(it.accession_number)
            if frames:
                run.write(INSIDER_TRANSACTIONS, pd.concat(frames, ignore_index=True), provider.name)
            run.ok()
            self.log.info("insiders %s: %d presentacions noves", day, len(items))
