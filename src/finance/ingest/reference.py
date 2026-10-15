"""Dades de referència: composició d'índexs (S&P 500) i metadades d'instruments."""

from __future__ import annotations

from typing import Any

import pandas as pd

from finance.jobs import Job, Run, run_concurrently
from finance.providers import ProviderError
from finance.storage.schemas import INDEX_CHANGES, INSTRUMENTS, UNIVERSE_MEMBERSHIP
from finance.universe import to_yahoo_symbol


class UniverseJob(Job):
    """Foto dels membres actuals + historial d'altes/baixes (contra el biaix de supervivència)."""

    name = "universe"

    def execute(self, run: Run, universes: list[str] | None = None, **_: Any) -> None:
        dynamic = sorted({g.dynamic for g in self.ctx.universe.config.groups.values() if g.dynamic})
        targets = universes or dynamic
        provider = self.ctx.providers.universe
        today = self.ctx.now().date()
        for uni in targets:
            try:
                members = provider.constituents(uni)
                members["symbol"] = members["symbol"].map(to_yahoo_symbol)
                members["universe"] = uni
                members["as_of_date"] = today
                run.write(UNIVERSE_MEMBERSHIP, members, provider.name)
                run.ok()
            except Exception as e:  # noqa: BLE001
                run.fail(f"{uni} membres", e)
                continue
            try:
                ch = provider.changes(uni)
                for c in ("added_symbol", "removed_symbol"):
                    ch[c] = ch[c].map(lambda s: to_yahoo_symbol(s) if s else "")
                ch["universe"] = uni
                run.write(INDEX_CHANGES, ch, provider.name)
            except ProviderError as e:
                # No crític: les fotos periòdiques de membres ja construeixen l'historial
                run.message = f"{uni}: historial de canvis no disponible ({e})"
                self.log.warning(run.message)
            except Exception as e:  # noqa: BLE001
                run.fail(f"{uni} canvis", e)
        # El univers dinàmic ha canviat: invalidar la memòria cau del context
        self.ctx._universe = None


class InstrumentsJob(Job):
    name = "instruments"

    def execute(self, run: Run, universe: str = "instruments", symbols: list[str] | None = None,
                **_: Any) -> None:
        ctx = self.ctx
        syms = [to_yahoo_symbol(s) for s in symbols] if symbols else ctx.universe.symbols(universe)
        today = ctx.now().date()
        provider = ctx.providers.instruments
        rows: list[dict[str, Any]] = []
        for sym, info, err in run_concurrently(provider.fetch_info, syms,
                                               workers=ctx.settings.yahoo_max_workers):
            if err is not None:
                run.fail(sym, err)
                continue
            rows.append({"symbol": sym, "as_of_date": today, **(info or {})})
            run.ok()
            if len(rows) >= 200:
                run.write(INSTRUMENTS, pd.DataFrame(rows), provider.name)
                rows = []
        if rows:
            run.write(INSTRUMENTS, pd.DataFrame(rows), provider.name)
