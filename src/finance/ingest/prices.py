"""Ingesta incremental de preus diaris i accions corporatives.

Lògica per símbol:
1. Watermark = última data guardada. Si no n'hi ha (o `full`), es baixa tot l'històric.
2. Si n'hi ha, es baixa des de `watermark - overlap` per cobrir correccions del proveïdor.
3. Si dins el tram nou hi ha un dividend/split, o el factor d'ajust (`adj_close/close`) del
   tram solapat no coincideix amb el guardat, l'històric ajustat ha canviat: es torna a
   baixar sencer (re-ajust).
4. Les barres solapades es reescriuen sempre: Yahoo publica una barra provisional just
   després del tancament i la corregeix hores després; l'endemà queda la definitiva.
5. Mai es guarda la barra de la sessió en curs (seria parcial): es talla a l'última
   sessió tancada segons el calendari de la borsa de cada instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from finance.calendar import last_completed_session
from finance.jobs import Job, Run, run_concurrently
from finance.storage.schemas import CORPORATE_ACTIONS, PRICES_DAILY
from finance.universe import Instrument, to_yahoo_symbol

FLUSH_ROWS = 400_000


@dataclass
class _Task:
    inst: Instrument
    start: date | None  # None = històric complet
    end: date
    watermark: date | None


@dataclass
class _Fetched:
    df: pd.DataFrame
    reloaded: bool  # s'ha re-baixat tot l'històric per re-ajust


class PricesJob(Job):
    name = "prices"

    def execute(
        self,
        run: Run,
        universe: str = "prices",
        symbols: list[str] | None = None,
        full: bool = False,
        **_: Any,
    ) -> None:
        ctx = self.ctx
        instruments = self._select(universe, symbols)
        if not instruments:
            run.skip("cap instrument seleccionat")
            return

        cov = ctx.query("SELECT symbol, last_date FROM prices_coverage")
        watermarks: dict[str, date] = dict(zip(cov["symbol"], pd.to_datetime(cov["last_date"]).dt.date,
                                                strict=True))
        history_start = date.fromisoformat(ctx.universe.config.history_start)
        overlap = ctx.settings.price_overlap_days
        now = ctx.now()
        cutoffs: dict[str, date] = {}

        tasks: list[_Task] = []
        up_to_date = 0
        for inst in instruments:
            end = cutoffs.setdefault(inst.calendar, last_completed_session(inst.calendar, now))
            wm = None if full else watermarks.get(inst.symbol)
            if wm is not None and wm >= end:
                up_to_date += 1
                continue
            start = (wm - timedelta(days=overlap)) if wm else history_start
            tasks.append(_Task(inst, start, end, wm))
        run.ok(up_to_date)
        self.log.info("%d instruments: %d al dia, %d a actualitzar (%d nous)", len(instruments),
                      up_to_date, len(tasks), sum(t.watermark is None for t in tasks))
        if not tasks:
            run.message = "tot al dia"
            return

        stored = self._stored_overlap([t for t in tasks if t.watermark is not None])
        buffer: list[pd.DataFrame] = []
        buffered = 0
        reloaded = 0
        provider = ctx.providers.prices

        def work(t: _Task) -> _Fetched:
            df = self._clip(provider.fetch_daily(t.inst.symbol, t.start, t.end), t.end)
            if t.watermark is not None and self._needs_reload(df, t.watermark, stored.get(t.inst.symbol)):
                full_df = provider.fetch_daily(t.inst.symbol, history_start, t.end)
                return _Fetched(self._clip(full_df, t.end), reloaded=True)
            return _Fetched(df, reloaded=False)

        for t, res, err in run_concurrently(work, tasks, workers=ctx.settings.yahoo_max_workers,
                                            key=lambda t: t.inst.symbol):
            if err is not None:
                run.fail(t.inst.symbol, err)
                continue
            assert res is not None
            if res.df.empty:
                run.fail(t.inst.symbol, "cap barra tancada al rang")
                continue
            reloaded += res.reloaded
            buffer.append(res.df)
            buffered += len(res.df)
            run.ok()
            if buffered >= FLUSH_ROWS:
                self._flush(run, buffer)
                buffer, buffered = [], 0
        self._flush(run, buffer)
        run.message = f"{reloaded} re-ajustos d'històric" if reloaded else ""

    # ---------------------------------------------------------------------------------
    def _select(self, universe: str, symbols: list[str] | None) -> list[Instrument]:
        uni = self.ctx.universe
        known = {i.symbol: i for i in uni.instruments(universe)}
        if not symbols:
            return list(known.values())
        out = []
        for s in symbols:
            sym = to_yahoo_symbol(s)
            out.append(known.get(sym) or Instrument(symbol=sym, asset_class="adhoc", group="adhoc",
                                                    calendar=uni.config.default_calendar))
        return out

    @staticmethod
    def _clip(df: pd.DataFrame, end: date) -> pd.DataFrame:
        return df[pd.to_datetime(df["date"]).dt.date <= end].reset_index(drop=True)

    def _stored_overlap(self, tasks: list[_Task]) -> dict[str, pd.Series]:
        if not tasks:
            return {}
        start = min(t.start for t in tasks if t.start is not None)
        syms = [t.inst.symbol for t in tasks]
        df = self.ctx.query(
            """SELECT symbol, date, adj_close / nullif(close, 0) AS factor FROM prices_daily
               WHERE date >= ? AND list_contains(?, symbol)""",
            [start, syms],
        )
        out: dict[str, pd.Series] = {}
        for sym, g in df.groupby("symbol"):
            out[str(sym)] = pd.Series(g["factor"].to_numpy(), index=pd.to_datetime(g["date"]).dt.date)
        return out

    def _needs_reload(self, df: pd.DataFrame, watermark: date, stored: pd.Series | None) -> bool:
        dates = pd.to_datetime(df["date"]).dt.date
        new = df[dates > watermark]
        if ((new["dividend"] > 0) | (new["split_ratio"] > 0) | (new["capital_gain"] > 0)).any():
            return True
        if stored is None or stored.empty:
            return False
        # Es compara el factor d'ajust (adj_close / close), no el preu: una barra provisional
        # corregida per Yahoo canvia close i adj_close alhora (factor intacte) i no requereix
        # re-ajust; un dividend nou canvia el factor de tot l'històric anterior.
        close = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
        fetched = pd.Series((df["adj_close"] / close).to_numpy(), index=dates)
        common = fetched.index.intersection(stored.index)
        if len(common) == 0:
            return False
        a = fetched.loc[common].astype(float).to_numpy()
        b = stored.loc[common].astype(float).to_numpy()
        mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
        if not mask.any():
            return False
        rel = np.abs(a[mask] / b[mask] - 1.0)
        return bool((rel > self.ctx.settings.adj_close_tolerance).any())

    def _flush(self, run: Run, frames: list[pd.DataFrame]) -> None:
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        source = self.ctx.providers.prices.name
        run.write(PRICES_DAILY, df.drop(columns=["capital_gain"]), source)
        actions = self._actions(df)
        if not actions.empty:
            run.write(CORPORATE_ACTIONS, actions, source)

    @staticmethod
    def _actions(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for col, name in (("dividend", "dividend"), ("split_ratio", "split"), ("capital_gain", "capital_gain")):
            sel = df[df[col] > 0]
            if not sel.empty:
                parts.append(pd.DataFrame({"symbol": sel["symbol"], "date": sel["date"],
                                           "action": name, "value": sel[col]}))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
