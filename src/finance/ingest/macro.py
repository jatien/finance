"""Ingesta de sèries macro (FRED/ALFRED) amb semàntica point-in-time.

Sèries revisables amb clau API: es guarden tots els vintages. L'actualització incremental
demana els vintages vigents des de `W = darrera ingesta - marge`. FRED retalla el
`realtime_start` dels vintages que ja eren vigents abans de W i els retorna amb
`realtime_start = W`; aquí es reassignen al `realtime_start` original guardat, de manera
que la fila existent s'actualitza (p.ex. es tanca el seu `realtime_end` si s'ha revisat)
en lloc de crear un interval solapat.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from finance.jobs import Job, Run, run_concurrently
from finance.storage.schemas import MACRO_OBSERVATIONS
from finance.universe import MacroSeries

VINTAGE_MARGIN_DAYS = 7
NON_REVISABLE_TAIL_DAYS = 45


class MacroJob(Job):
    name = "macro"

    def execute(self, run: Run, series: list[str] | None = None, full: bool = False, **_: Any) -> None:
        ctx = self.ctx
        all_series = {s.series_id: s for s in ctx.universe.macro}
        targets = [all_series.get(s.upper()) or MacroSeries(series_id=s.upper(), name=s.upper())
                   for s in series] if series else list(all_series.values())
        provider = ctx.providers.macro
        if not provider.supports_vintages and any(s.revisable for s in targets):
            run.message = "sense clau FRED: sèries revisables NO point-in-time (vintage_mode=latest_only)"
            self.log.warning("FIN_FRED_API_KEY no definida: les sèries revisables no tindran vintages")

        state = ctx.query("""
            SELECT series_id, max(date) AS last_date,
                   max(ingested_at) FILTER (WHERE vintage_mode = 'alfred') AS last_vintage_ingest
            FROM macro_observations GROUP BY series_id
        """)
        last_date = dict(zip(state["series_id"], pd.to_datetime(state["last_date"]).dt.date, strict=True))
        # Només compta com a "ja carregada amb vintages" si hi ha files ALFRED: una sèrie
        # carregada abans sense clau API es torna a baixar sencera amb vintages.
        vint_state = state.dropna(subset=["last_vintage_ingest"])
        last_ing = dict(zip(vint_state["series_id"],
                            pd.to_datetime(vint_state["last_vintage_ingest"]).dt.date, strict=True))

        def work(s: MacroSeries) -> pd.DataFrame:
            vint = s.revisable and provider.supports_vintages
            if vint:
                w = None if full or s.series_id not in last_ing else last_ing[s.series_id] - timedelta(days=VINTAGE_MARGIN_DAYS)
                df = provider.fetch_observations(s.series_id, vintages=True, realtime_start=w)
                if w is not None:
                    df = self._reanchor(s.series_id, df, w)
                return df
            df = provider.fetch_observations(s.series_id, vintages=s.revisable)
            ld = last_date.get(s.series_id)
            if ld is not None and not full and not s.revisable:
                df = df[pd.to_datetime(df["date"]).dt.date >= ld - timedelta(days=NON_REVISABLE_TAIL_DAYS)]
            return df

        for s, df, err in run_concurrently(work, targets, workers=2, key=lambda s: s.series_id):
            if err is not None:
                run.fail(s.series_id, err)
                continue
            assert df is not None
            run.write(MACRO_OBSERVATIONS, df, provider.name)
            run.ok()

    def _reanchor(self, series_id: str, df: pd.DataFrame, w: date) -> pd.DataFrame:
        """Reassigna els vintages retallats a W al seu realtime_start original."""
        clipped = df["realtime_start"] == w
        if not clipped.any():
            return df
        stored = self.ctx.query(
            """SELECT date, realtime_start FROM macro_observations
               WHERE series_id = ? AND realtime_start <= ? AND realtime_end >= ?""",
            [series_id, w, w],
        )
        if stored.empty:
            return df
        anchor = dict(zip(pd.to_datetime(stored["date"]).dt.date,
                          pd.to_datetime(stored["realtime_start"]).dt.date, strict=True))
        df = df.copy()
        df.loc[clipped, "realtime_start"] = [anchor.get(d, w) for d in df.loc[clipped, "date"]]
        return df
