"""Controls de qualitat de dades sobre el lake.

Cada control retorna files (dataset, check, key, severity, n_violations, details).
Els controls de preus miren per defecte una finestra recent (`lookback_days`): el que importa
operativament és detectar problemes nous; l'històric antic es pot auditar amb lookback 0.
Severitats: `error` = dada probablement incorrecta o pipeline trencat; `warn` = cal mirar-ho;
`info` = característica coneguda de la font.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import duckdb
import pandas as pd

from finance.calendar import last_completed_session, sessions_between
from finance.jobs import Job, Run
from finance.storage.schemas import DQ_RESULTS

Check = Callable[[duckdb.DuckDBPyConnection, dict[str, Any]], pd.DataFrame]
COLS = ["dataset", "check_name", "key", "severity", "n_violations", "details"]


def _rows(df: pd.DataFrame, dataset: str, check: str, severity: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLS)
    out = pd.DataFrame({
        "dataset": dataset, "check_name": check, "key": df["key"].astype(str),
        "severity": severity, "n_violations": df["n"].astype("int64"),
        "details": df["details"].astype(str) if "details" in df else "",
    })
    return out[out["n_violations"] > 0]


# ---------------------------------------------------------------------------------------
# Preus
# ---------------------------------------------------------------------------------------

def prices_bad_close(con, p):
    df = con.execute("""
        SELECT symbol AS key, count(*) AS n, string_agg(CAST(date AS VARCHAR), ',' ORDER BY date DESC) AS details
        FROM prices_daily WHERE close IS NULL OR close <= 0 OR adj_close IS NULL OR adj_close <= 0
        GROUP BY symbol
    """).df()
    df["details"] = df["details"].str.slice(0, 300)
    return _rows(df, "prices_daily", "bad_close", "error")


def prices_ohlc_inconsistent(con, p):
    df = con.execute("""
        SELECT symbol AS key, count(*) AS n, max(date)::VARCHAR AS details
        FROM prices_daily
        WHERE (high < greatest(open, close, low) * (1 - 1e-6)
           OR low  > least(open, close, high)  * (1 + 1e-6))
          AND date >= ?
        GROUP BY symbol
    """, [p["since"]]).df()
    return _rows(df, "prices_daily", "ohlc_inconsistent", "warn")


def prices_extreme_returns(con, p):
    thr = p.get("extreme_return", 0.35)
    df = con.execute(f"""
        WITH r AS (
            SELECT symbol, date,
                   adj_close / lag(adj_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS ret
            FROM prices_daily WHERE symbol NOT LIKE '^%'
        )
        SELECT symbol AS key, count(*) AS n,
               string_agg(date::VARCHAR || ':' || round(ret, 3)::VARCHAR, ',' ORDER BY date DESC) AS details
        FROM r WHERE abs(ret) > {thr} AND date >= ? GROUP BY symbol
    """, [p["since"]]).df()
    df["details"] = df["details"].str.slice(0, 300)
    return _rows(df, "prices_daily", "extreme_return", "warn")


def prices_stale_values(con, p):
    # 5 tancaments seguits idèntics amb volum: preu congelat a la font
    df = con.execute("""
        WITH x AS (
            SELECT symbol, date, close, volume,
                   max(close) OVER w - min(close) OVER w AS rng, count(*) OVER w AS nw
            FROM prices_daily WHERE symbol NOT LIKE '^%'
            WINDOW w AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
        )
        SELECT symbol AS key, count(*) AS n, max(date)::VARCHAR AS details
        FROM x WHERE nw = 5 AND rng = 0 AND volume > 0 AND date >= ? GROUP BY symbol
    """, [p["since"]]).df()
    return _rows(df, "prices_daily", "stale_values", "warn")


def prices_missing_sessions(con, p):
    cov = con.execute("SELECT symbol, first_date, last_date FROM prices_coverage").df()
    if cov.empty:
        return pd.DataFrame(columns=COLS)
    lo = max(pd.to_datetime(cov["first_date"]).min(), pd.Timestamp(p["since"]))
    hi = pd.to_datetime(cov["last_date"]).max()
    sessions = pd.DataFrame({"d": sessions_between(lo.date(), hi.date(), "XNYS").date})
    con.register("_sessions", sessions)
    try:
        df = con.execute("""
            SELECT c.symbol AS key, count(*) AS n,
                   string_agg(s.d::VARCHAR, ',' ORDER BY s.d DESC) AS details
            FROM prices_coverage c
            JOIN _sessions s ON s.d BETWEEN c.first_date AND c.last_date AND s.d >= ?
            ANTI JOIN prices_daily pdl ON pdl.symbol = c.symbol AND pdl.date = s.d
            GROUP BY c.symbol
        """, [p["since"]]).df()
    finally:
        con.unregister("_sessions")
    df["details"] = df["details"].str.slice(0, 300)
    return _rows(df, "prices_daily", "missing_sessions", "warn")


def prices_stale_symbols(con, p):
    last = last_completed_session("XNYS", p.get("now"))
    limit = last - timedelta(days=p.get("stale_days", 5))
    df = con.execute("""
        SELECT symbol AS key, 1 AS n, 'última data ' || last_date::VARCHAR AS details
        FROM prices_coverage WHERE last_date < ?
    """, [limit]).df()
    return _rows(df, "prices_daily", "stale_symbol", "error")


# ---------------------------------------------------------------------------------------
# Opcions (darrera foto de cada subjacent)
# ---------------------------------------------------------------------------------------

def options_quality(con, p):
    df = con.execute("""
        SELECT underlying AS key,
               count(*) AS total,
               count(*) FILTER (WHERE bid > ask AND ask > 0) AS crossed,
               count(*) FILTER (WHERE mid IS NULL) AS no_mid,
               count(*) FILTER (WHERE implied_vol <= 0 OR implied_vol > 5) AS bad_iv,
               count(*) FILTER (WHERE underlying_price IS NULL) AS no_spot,
               max(snapshot_ts)::VARCHAR AS ts
        FROM options_latest GROUP BY underlying
    """).df()
    if df.empty:
        return pd.DataFrame(columns=COLS)
    out = []
    for col, check, sev in (("crossed", "crossed_quotes", "warn"), ("bad_iv", "iv_out_of_range", "info"),
                            ("no_spot", "missing_underlying_price", "error")):
        x = df.assign(n=df[col], details=df["ts"])
        out.append(_rows(x, "option_snapshots", check, sev))
    ratio = df.assign(n=(df["no_mid"] / df["total"] > 0.5).astype(int),
                      details=(df["no_mid"] / df["total"]).round(3).astype(str) + " sense mid @ " + df["ts"])
    out.append(_rows(ratio, "option_snapshots", "mostly_no_quotes", "warn"))
    return pd.concat(out, ignore_index=True)


def options_count_drop(con, p):
    df = con.execute("""
        WITH c AS (
            SELECT underlying, snapshot_ts, count(*) AS n FROM option_snapshots
            WHERE snapshot_date >= current_date - INTERVAL 10 DAY
            GROUP BY ALL
        ), l AS (
            SELECT *, lag(n) OVER (PARTITION BY underlying ORDER BY snapshot_ts) AS prev
            FROM c QUALIFY row_number() OVER (PARTITION BY underlying ORDER BY snapshot_ts DESC) = 1
        )
        SELECT underlying AS key, 1 AS n, n || ' vs ' || prev || ' contractes' AS details
        FROM l WHERE prev IS NOT NULL AND n < 0.6 * prev
    """).df()
    return _rows(df, "option_snapshots", "contract_count_drop", "warn")


# ---------------------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------------------

# Dies màxims des de la data de l'última observació (inclou el retard de publicació:
# una dada mensual de l'1-ago es publica a mitjan setembre o més tard)
MAX_AGE_DAYS = {"D": 10, "W": 21, "M": 100, "Q": 200}


def macro_stale(con, p):
    freq = p.get("macro_freq", {})
    df = con.execute("SELECT series_id, max(date) AS last FROM macro_latest GROUP BY series_id").df()
    if df.empty:
        return pd.DataFrame(columns=COLS)
    today = pd.Timestamp(p.get("now") or pd.Timestamp.now(tz="UTC")).date()
    df["age"] = (pd.Timestamp(today) - pd.to_datetime(df["last"])).dt.days
    df["limit"] = df["series_id"].map(lambda s: MAX_AGE_DAYS.get(freq.get(s, "D"), 10))
    bad = df[df["age"] > df["limit"]]
    x = pd.DataFrame({"key": bad["series_id"], "n": 1,
                      "details": "última observació " + bad["last"].astype(str)})
    return _rows(x, "macro_observations", "stale_series", "warn")


def macro_pit_overlap(con, p):
    # Dos vintages d'una mateixa observació no poden ser vigents alhora
    df = con.execute("""
        WITH x AS (
            SELECT series_id, date, realtime_start,
                   lag(realtime_end) OVER (PARTITION BY series_id, date ORDER BY realtime_start) AS prev_end
            FROM macro_observations
        )
        SELECT series_id AS key, count(*) AS n, max(date)::VARCHAR AS details
        FROM x WHERE prev_end IS NOT NULL AND prev_end >= realtime_start GROUP BY series_id
    """).df()
    return _rows(df, "macro_observations", "pit_overlap", "error")


def macro_not_pit(con, p):
    df = con.execute("""
        SELECT series_id AS key, 1 AS n, 'sèrie revisable sense vintages' AS details
        FROM macro_observations WHERE vintage_mode = 'latest_only' GROUP BY series_id
    """).df()
    return _rows(df, "macro_observations", "not_point_in_time", "info")


CHECKS: dict[str, list[Check]] = {
    "prices_daily": [prices_bad_close, prices_ohlc_inconsistent, prices_extreme_returns,
                     prices_stale_values, prices_missing_sessions, prices_stale_symbols],
    "option_snapshots": [options_quality, options_count_drop],
    "macro_observations": [macro_stale, macro_pit_overlap, macro_not_pit],
}


class QualityJob(Job):
    name = "dq"

    def execute(self, run: Run, datasets: list[str] | None = None, lookback_days: int = 400,
                **_: Any) -> None:
        ctx = self.ctx
        since = (ctx.now() - pd.Timedelta(days=lookback_days)).date() if lookback_days > 0 else date(1900, 1, 1)
        params = {"now": ctx.now(), "since": since,
                  "macro_freq": {s.series_id: s.frequency for s in ctx.universe.macro}}
        con = ctx.con()
        results = []
        try:
            for ds in datasets or list(CHECKS):
                for check in CHECKS[ds]:
                    try:
                        results.append(check(con, params))
                        run.ok()
                    except Exception as e:  # noqa: BLE001
                        run.fail(f"{ds}.{check.__name__}", e)
        finally:
            con.close()
        df = pd.concat([r for r in results if not r.empty], ignore_index=True) if any(
            not r.empty for r in results) else pd.DataFrame(columns=COLS)
        if not df.empty:
            df["run_id"] = run.run_id
            df["checked_at"] = run.ingested_at
            run.write(DQ_RESULTS, df, source="dq")
        counts = df.groupby("severity").size().to_dict() if not df.empty else {}
        run.message = "violacions: " + json.dumps(counts) if counts else "cap violació"
        self.results = df
