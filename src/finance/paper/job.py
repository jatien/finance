"""Job `paper`: avança el paper trading de cada estratègia activa i escriu les sessions noves.

Només s'escriuen sessions posteriors a l'última registrada: tornar a executar el job el mateix
dia no canvia res, i el passat no es recalcula mai amb dades noves.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from finance.jobs import Job, Run
from finance.paper.book import BookState, BookUpdate, advance, start, state_from_rows
from finance.portfolio.config import ACTIVE, CASH, PortfolioConfig
from finance.portfolio.construct import build_targets
from finance.portfolio.job import load_inputs
from finance.storage.schemas import PAPER_BOOK, PAPER_NAV


def read_paper_nav(ctx, strategy_set: str | None = None) -> pd.DataFrame:
    where, params = ("WHERE strategy_set = ?", [strategy_set]) if strategy_set else ("", [])
    try:
        df = ctx.query(f"SELECT * EXCLUDE (source, ingested_at, run_id, year) FROM paper_nav {where} "
                       "ORDER BY strategy_set, date", params)
    except Exception:  # noqa: BLE001 — dataset encara buit / vista inexistent
        return pd.DataFrame()
    for c in ("date", "inception", "pending_from"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def read_paper_book(ctx, strategy_set: str | None = None, date=None) -> pd.DataFrame:
    where, params = [], []
    if strategy_set:
        where.append("strategy_set = ?")
        params.append(strategy_set)
    if date is not None:
        where.append("date = ?")
        params.append(pd.Timestamp(date).date())
    sql = ("SELECT * EXCLUDE (source, ingested_at, run_id, year) FROM paper_book "
           + (f"WHERE {' AND '.join(where)} " if where else "") + "ORDER BY strategy_set, date, weight DESC")
    try:
        df = ctx.query(sql, params)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def read_state(ctx, strategy_set: str) -> BookState | None:
    nav = read_paper_nav(ctx, strategy_set)
    if nav.empty:
        return None
    last = nav["date"].max()
    return state_from_rows(nav[nav["date"] == last], read_paper_book(ctx, strategy_set, last))


def complete_until(ctx, symbols: list[str]) -> pd.Timestamp | None:
    """Última sessió amb preu real (no omplert) de tots els `symbols` que en tenen.

    `load_inputs` omple forats de fins a 3 sessions; el registre congelat no ha de decidir mai
    amb un preu omplert, així que no s'avança més enllà d'aquesta data.
    """
    syms = [s for s in symbols if s != CASH]
    df = ctx.query("SELECT symbol, max(date) AS d FROM prices_daily WHERE symbol IN (SELECT unnest(?)) "
                   "GROUP BY symbol", [syms])
    if df.empty:
        return None
    d = pd.to_datetime(df["d"])
    # Un actiu que ja no cotitza (o sense dades recents) no bloqueja la resta
    live = d[d >= d.max() - pd.Timedelta(days=10)]
    return pd.Timestamp(live.min())


def update_strategy(ctx, cfg: PortfolioConfig) -> tuple[BookUpdate, list[str]]:
    inp = load_inputs(ctx, cfg)
    targets = build_targets(inp, cfg)
    notes = list(inp.notes)
    end = complete_until(ctx, list(targets.weights.columns))
    if end is not None and end < inp.index[-1]:
        notes.append(f"{cfg.name}: preus incomplets després del {end:%Y-%m-%d}; s'avança fins aquí")
        inp = inp.truncate(end)
        targets = build_targets(inp, cfg)
    state = read_state(ctx, cfg.name)
    upd = start(inp, targets, cfg) if state is None else advance(state, inp, targets, cfg)
    return upd, notes


class PaperJob(Job):
    name = "paper"

    def execute(self, run: Run, strategies: list[str] | None = None, **_: Any) -> None:
        cfgs = [c for c in ACTIVE if strategies is None or c.name in strategies]
        if not cfgs:
            raise ValueError(f"cap estratègia activa coincideix amb {strategies}")
        run.params["strategies"] = [c.name for c in cfgs]
        msgs: list[str] = []
        for cfg in cfgs:
            try:
                upd, notes = update_strategy(self.ctx, cfg)
            except Exception as e:  # noqa: BLE001 — una estratègia no atura les altres
                run.fail(cfg.name, f"{type(e).__name__}: {e}")
                continue
            if upd.nav.empty:
                msgs.append(f"{cfg.name}: al dia ({upd.state.date:%Y-%m-%d})")
                run.ok()
                continue
            run.write(PAPER_NAV, upd.nav, f"paper:{cfg.name}")
            run.write(PAPER_BOOK, upd.book, f"paper:{cfg.name}")
            run.ok()
            n = len(upd.nav)
            traded = int(upd.nav["traded"].sum())
            first = upd.nav["date"].min()
            label = "inici" if upd.state.inception == first else f"{n} sessió(ns)"
            msgs.append(f"{cfg.name}: {label} fins a {upd.state.date:%Y-%m-%d}, NAV {upd.state.nav:.4f}"
                        + (f", {traded} amb operacions" if traded else ""))
            msgs.extend(x for x in notes if x not in msgs)
        run.message = "; ".join(msgs)
