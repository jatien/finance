"""Job `insider_signals`: operacions del lake → esdeveniments per acció i amplitud.

Escriu `insider_events` i `insider_breadth` (es regeneren sencers per `signal_set`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from finance.calendar import last_completed_session, sessions_between
from finance.insiders.config import InsiderConfig
from finance.insiders.signals import breadth, build_events, prepare_trades
from finance.jobs import Job, Run
from finance.storage.schemas import INSIDER_BREADTH, INSIDER_EVENTS


@dataclass
class InsiderInputs:
    trades: pd.DataFrame  # operacions P/S preparades (amb data de senyal i elegibilitat)
    sessions: pd.DatetimeIndex
    members: pd.Series  # True si l'emissor era al S&P 500 a la data de senyal
    notes: list[str] = field(default_factory=list)


CIK_MAP_SQL = """
    SELECT lpad(trim(cik), 10, '0') AS cik, symbol, sector FROM universe_membership
    WHERE cik IS NOT NULL AND trim(cik) <> ''
    QUALIFY row_number() OVER (PARTITION BY lpad(trim(cik), 10, '0') ORDER BY as_of_date DESC) = 1
"""


def cik_symbols(ctx) -> pd.DataFrame:
    """CIK → símbol i sector GICS de la foto més recent en què apareix cada CIK."""
    return ctx.query(CIK_MAP_SQL)


def sp500_member_flags(ctx, keys: pd.DataFrame) -> pd.Series:
    """Per a cada (symbol, signal_date): era membre del S&P 500 aquell dia?"""
    if keys.empty:
        return pd.Series(dtype=bool)
    con = ctx.con()
    try:
        k = keys[["symbol", "signal_date"]].copy()
        k["signal_date"] = pd.to_datetime(k["signal_date"]).dt.date
        k["_i"] = range(len(k))
        con.register("k", k)
        df = con.execute("""
            SELECT k._i, count(u.symbol) > 0 AS member FROM k
            LEFT JOIN universe_intervals u
              ON u.universe = 'sp500' AND u.symbol = k.symbol AND u.start_date <= k.signal_date
             AND (u.end_date IS NULL OR u.end_date > k.signal_date)
            GROUP BY k._i
        """).df().set_index("_i")["member"]
    finally:
        con.close()
    return pd.Series(df.reindex(range(len(keys))).fillna(False).values, index=keys.index).astype(bool)


def load_inputs(ctx, cfg: InsiderConfig, end: date | None = None) -> InsiderInputs:
    notes: list[str] = []
    # Historial previ per classificar els rutinaris (necessita `routine_years` anys enrere)
    hist_start = date(cfg.start.year - cfg.routine_years - 1, 1, 1)
    end = end or last_completed_session(now=ctx.now())
    sessions = sessions_between(hist_start, end)
    con = ctx.con()
    try:
        n = con.execute("SELECT count(*) FROM insider_transactions").fetchone()[0]
        if not n:
            raise RuntimeError("sense operacions d'insiders: executa `fin ingest insiders`")
        con.execute(f"CREATE TEMP VIEW _cik_map AS {CIK_MAP_SQL}")
        t = prepare_trades(con, cfg, sessions, cik_map="_cik_map", intervals="universe_intervals",
                           start=hist_start, end=end)
        first = con.execute("SELECT min(filing_date) FROM insider_transactions").fetchone()[0]
    finally:
        con.close()
    t = t[t["signal_date"] <= pd.Timestamp(end)].reset_index(drop=True)
    members = t["member"].astype(bool)
    if not members.any():
        notes.append("sense historial del S&P 500: l'amplitud no es pot calcular "
                     "(`fin ingest universe-history`)")
    if first is not None and pd.Timestamp(first) > pd.Timestamp(hist_start):
        notes.append(f"l'historial comença el {pd.Timestamp(first):%Y-%m-%d}: els primers "
                     f"{cfg.routine_years} anys tenen molts insiders 'nous' (sense classificar)")
    return InsiderInputs(trades=t, sessions=sessions, members=members, notes=notes)


def add_adv(ctx, events: pd.DataFrame, cfg: InsiderConfig) -> pd.DataFrame:
    """Import / volum mitjà diari en USD de les `adv_window` sessions anteriors a la senyal."""
    if events.empty:
        return events.assign(value_adv=pd.Series(dtype=float))
    con = ctx.con()
    try:
        k = events[["symbol", "signal_date"]].drop_duplicates().copy()
        k["signal_date"] = pd.to_datetime(k["signal_date"]).dt.date
        con.register("k", k)
        adv = con.execute(f"""
            WITH p AS (
                SELECT symbol, date,
                       avg(close * volume) OVER (PARTITION BY symbol ORDER BY date
                           ROWS BETWEEN {cfg.adv_window} PRECEDING AND 1 PRECEDING) AS adv
                FROM prices_daily WHERE symbol IN (SELECT DISTINCT symbol FROM k)
            )
            SELECT k.symbol, k.signal_date, p.adv FROM k
            ASOF LEFT JOIN p ON p.symbol = k.symbol AND p.date <= k.signal_date
        """).df()
    finally:
        con.close()
    adv["signal_date"] = pd.to_datetime(adv["signal_date"])
    out = events.merge(adv, on=["symbol", "signal_date"], how="left")
    out["value_adv"] = out["value"] / out["adv"].where(out["adv"] > 0)
    return out.drop(columns="adv")


def compute(ctx, cfg: InsiderConfig, end: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame, InsiderInputs]:
    inp = load_inputs(ctx, cfg, end)
    t = inp.trades
    ev = build_events(t, cfg)
    ev = ev[ev["signal_date"] >= pd.Timestamp(cfg.start)].reset_index(drop=True)
    ev["in_sp500"] = sp500_member_flags(ctx, ev).values if not ev.empty else []
    ev = add_adv(ctx, ev, cfg)
    sess = inp.sessions[inp.sessions >= pd.Timestamp(cfg.start)]
    br = breadth(t, inp.sessions, cfg, inp.members)
    br = br[br["date"] >= sess[0]] if len(sess) else br.iloc[0:0]
    return ev, br, inp


class InsiderSignalsJob(Job):
    name = "insider_signals"

    def execute(self, run: Run, **_: Any) -> None:
        cfg = InsiderConfig()
        run.params["insider_config"] = cfg.to_params()
        n = self.ctx.query("SELECT count(*) AS n FROM insider_transactions")["n"].iloc[0]
        if not n:
            run.skip("sense operacions d'insiders (`fin ingest insiders`, cal FIN_SEC_USER_AGENT)")
            return
        ev, br, inp = compute(self.ctx, cfg)
        if inp.notes:
            run.message = " · ".join(inp.notes)
        if not ev.empty:
            run.write(INSIDER_EVENTS, ev.assign(signal_set=cfg.name), f"insiders:{cfg.name}")
        if not br.empty:
            run.write(INSIDER_BREADTH, br.assign(signal_set=cfg.name), f"insiders:{cfg.name}")
        run.ok()
        if not ev.empty:
            last = ev["signal_date"].max()
            recent = ev[(ev["signal_date"] > last - pd.Timedelta(days=30)) & ev["in_sp500"]]
            counts = recent["event"].value_counts().to_dict()
            run.message = (run.message + " · " if run.message else "") + (
                f"{len(ev)} esdeveniments; últims 30 dies al S&P 500: {counts}")


def read_events(ctx, signal_set: str = "ins_v1", since: date | None = None) -> pd.DataFrame:
    sql = ("SELECT * EXCLUDE (source, ingested_at, run_id, year) FROM insider_events "
           "WHERE signal_set = ?")
    params: list = [signal_set]
    if since is not None:
        sql += " AND signal_date >= ?"
        params.append(since)
    df = ctx.query(sql + " ORDER BY signal_date, symbol", params)
    if not df.empty:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def read_breadth(ctx, signal_set: str = "ins_v1") -> pd.DataFrame:
    df = ctx.query("SELECT scope, date, buyers, sellers, buy_ratio, ratio_pct FROM insider_breadth "
                   "WHERE signal_set = ? ORDER BY scope, date", [signal_set])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
