"""Fotos de cadenes d'opcions senceres.

Cada execució defineix un `snapshot_ts` lògic comú (arrodonit al minut) perquè totes
les cadenes d'una mateixa foto siguin comparables transversalment. A més es guarda
`fetched_at` (instant real de cada petició) i la cotització del subjacent en aquell moment.

Aquestes dades no es poden recuperar a posteriori amb fonts gratuïtes: cada dia que no
es fa la foto és històric perdut. Per això és un job independent i robust a errors
parcials (una expiració dolenta no invalida la resta).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from finance.calendar import get_calendar, is_open
from finance.jobs import Job, Run, run_concurrently
from finance.providers import ChainSlice, ProviderError
from finance.storage.schemas import OPTION_SNAPSHOTS
from finance.universe import to_yahoo_symbol


class OptionsJob(Job):
    name = "options"
    calendar = "XNYS"

    def execute(
        self,
        run: Run,
        underlyings: list[str] | None = None,
        force: bool = False,
        **_: Any,
    ) -> None:
        ctx = self.ctx
        cfg = ctx.universe.options
        now = ctx.now()
        if not force and not is_open(self.calendar, now):
            run.skip("mercat tancat (usa --force per fer la foto igualment)")
            return

        snapshot_ts = now.floor("min")
        tz = get_calendar(self.calendar).tz
        snapshot_date = snapshot_ts.tz_convert(tz).date()
        targets = [to_yahoo_symbol(u) for u in (underlyings or cfg.underlyings)]
        provider = ctx.providers.options

        def work(underlying: str) -> tuple[pd.DataFrame, dict[str, str]]:
            expiries = provider.list_expiries(underlying)
            sel = [e for e in expiries if cfg.min_dte_days <= (e - snapshot_date).days <= cfg.max_dte_days]
            frames, errs = [], {}
            for exp in sel:
                try:
                    frames.append(self._frame(provider.fetch_chain(underlying, exp), underlying,
                                              snapshot_ts, snapshot_date))
                except Exception as e:  # noqa: BLE001
                    errs[f"{underlying} {exp}"] = str(e)
            if not frames:
                raise ProviderError(f"{underlying}: cap expiració descarregada ({len(sel)} intentades)")
            df = pd.concat(frames, ignore_index=True)
            return self._filter_moneyness(df, cfg.moneyness_min, cfg.moneyness_max), errs

        n_contracts = 0
        for und, res, err in run_concurrently(work, targets, workers=max(1, ctx.settings.yahoo_max_workers // 2)):
            if err is not None:
                run.fail(und, err)
                continue
            assert res is not None
            df, partial = res
            for k, v in partial.items():
                run.fail(k, v)
            n_contracts += run.write(OPTION_SNAPSHOTS, df, provider.name)
            run.ok()
        run.message = f"foto {snapshot_ts.isoformat()}: {n_contracts} contractes"

    @staticmethod
    def _frame(sl: ChainSlice, underlying: str, snapshot_ts: pd.Timestamp,
               snapshot_date: date) -> pd.DataFrame:
        df = sl.contracts.copy()
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        valid = (bid > 0) & (ask > 0) & (ask >= bid)
        df["mid"] = np.where(valid, (bid + ask) / 2.0, np.nan)
        df["underlying"] = underlying
        df["snapshot_ts"] = snapshot_ts
        df["snapshot_date"] = snapshot_date
        df["fetched_at"] = sl.fetched_at
        df["underlying_price"] = sl.quote.price
        df["underlying_bid"] = sl.quote.bid
        df["underlying_ask"] = sl.quote.ask
        df["underlying_quote_ts"] = sl.quote.quote_ts
        df["market_state"] = sl.quote.market_state
        df["dte"] = (pd.to_datetime(df["expiry"]) - pd.Timestamp(snapshot_date)).dt.days
        return df

    @staticmethod
    def _filter_moneyness(df: pd.DataFrame, lo: float | None, hi: float | None) -> pd.DataFrame:
        if lo is None and hi is None:
            return df
        m = df["strike"] / df["underlying_price"]
        keep = pd.Series(True, index=df.index)
        if lo is not None:
            keep &= m >= lo
        if hi is not None:
            keep &= m <= hi
        return df[keep | df["underlying_price"].isna()]
