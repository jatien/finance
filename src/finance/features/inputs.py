"""Entrades point-in-time de la capa de features.

`MarketData` agrupa tot el que necessiten les features, ja alineat al calendari de
sessions i expressat en "data de disponibilitat": la fila t conté només informació que
existia al tancament de t. A partir d'aquí, el càlcul de features és una funció pura i
causal de `MarketData`, i `MarketData.truncate(t)` permet verificar-ho (tests de truncament).

Fonts:
- Preus: `prices_daily.adj_close` (retorn total). Índexs (^VIX...) no porten dividends.
- Macro: `macro_observations` amb semàntica point-in-time (vegeu `macro_pit_daily`).
- Amplitud del S&P 500: membres vigents a cada data (`universe_intervals`, sense biaix de
  supervivència en la pertinença) amb la seva SMA200 calculada a DuckDB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finance.features.config import FeatureConfig

log = logging.getLogger(__name__)


@dataclass
class MarketData:
    prices: pd.DataFrame  # índex: sessions; columnes: símbols (adj_close)
    macro: pd.DataFrame = field(default_factory=pd.DataFrame)  # columnes: nom de feature
    sp500_breadth: pd.DataFrame = field(default_factory=pd.DataFrame)  # breadth_200, coverage
    groups: dict[str, list[str]] = field(default_factory=dict)  # grup -> símbols del panell
    notes: list[str] = field(default_factory=list)  # avisos (sèries excloses, cobertura...)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.prices.index

    def truncate(self, end: pd.Timestamp | str) -> MarketData:
        """Còpia amb només les files <= end (simula estar a la data `end`)."""
        end = pd.Timestamp(end)

        def cut(df: pd.DataFrame) -> pd.DataFrame:
            return df.loc[df.index <= end].copy() if not df.empty else df.copy()

        return MarketData(prices=cut(self.prices), macro=cut(self.macro),
                          sp500_breadth=cut(self.sp500_breadth),
                          groups={k: list(v) for k, v in self.groups.items()},
                          notes=list(self.notes))


# ---------------------------------------------------------------------------------------
# Macro point-in-time
# ---------------------------------------------------------------------------------------

def macro_pit_daily(obs: pd.DataFrame, index: pd.DatetimeIndex, lag_sessions: int = 1) -> pd.Series:
    """Converteix observacions d'una sèrie FRED en una sèrie diària point-in-time.

    `obs`: files de `macro_observations` d'una sola sèrie (date, value, realtime_start,
    vintage_mode). Data de disponibilitat de cada observació:
    - `alfred` (vintages reals): la seva primera publicació (min realtime_start), amb el
      valor d'aquella primera publicació. Les revisions posteriors d'aquesta observació no
      s'hi apliquen: és l'enfocament "first release", que mai fa servir el futur.
    - `observation_date` (sèrie no revisable): la data de l'observació + `lag_sessions`
      sessions (FRED publica les sèries diàries l'endemà).

    El valor a t és el de l'observació més recent disponible a t (merge asof).
    """
    if obs.empty or len(index) == 0:
        return pd.Series(np.nan, index=index, dtype=float)
    df = obs.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df = df.dropna(subset=["value"])
    if df.empty:
        return pd.Series(np.nan, index=index, dtype=float)

    if (df["vintage_mode"] == "alfred").any():
        df = df[df["vintage_mode"] == "alfred"]
        first = df.sort_values("realtime_start").groupby("date", as_index=False).first()
        first["available"] = first["realtime_start"]
        avail = first[["date", "available", "value"]]
    else:
        avail = df[["date", "value"]].copy()
        # Sessió de disponibilitat: la `lag_sessions`-èsima sessió posterior a la data
        if lag_sessions <= 0:
            pos = index.searchsorted(avail["date"], side="left")
        else:
            pos = index.searchsorted(avail["date"], side="right") + (lag_sessions - 1)
        ok = pos < len(index)
        avail = avail[ok].copy()
        avail["available"] = index[pos[ok]]

    # Si diverses observacions es fan disponibles alhora, guanya la més recent
    avail = avail.sort_values(["available", "date"])
    avail["available"] = pd.to_datetime(avail["available"]).astype("datetime64[ns]")
    left = pd.DataFrame({"t": pd.DatetimeIndex(index).astype("datetime64[ns]")})
    merged = pd.merge_asof(left, avail.rename(columns={"available": "t"}), on="t",
                           direction="backward")
    return pd.Series(merged["value"].to_numpy(dtype=float), index=index)


# ---------------------------------------------------------------------------------------
# Càrrega des del lake
# ---------------------------------------------------------------------------------------

def required_symbols(cfg: FeatureConfig, groups: dict[str, list[str]]) -> list[str]:
    singles = [cfg.market, cfg.equal_weight, cfg.long_bond, cfg.mid_bond, cfg.high_yield,
               cfg.vix, cfg.vix3m, *cfg.defensive, *cfg.cyclical]
    syms = list(dict.fromkeys(singles + [s for g in groups.values() for s in g]))
    return syms


def load_prices(ctx, symbols: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    where = ["symbol IN (SELECT unnest(?))"]
    params: list = [symbols]
    if start:
        where.append("date >= ?")
        params.append(pd.Timestamp(start).date())
    if end:
        where.append("date <= ?")
        params.append(pd.Timestamp(end).date())
    df = ctx.query(f"""
        SELECT date, symbol, coalesce(adj_close, close) AS px FROM prices_daily
        WHERE {' AND '.join(where)} AND coalesce(adj_close, close) > 0
    """, params)
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="date", columns="symbol", values="px")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def load_macro(ctx, cfg: FeatureConfig, index: pd.DatetimeIndex, notes: list[str]) -> pd.DataFrame:
    ids = list(cfg.macro_series.values())
    obs = ctx.query("""
        SELECT series_id, date, value, realtime_start, vintage_mode
        FROM macro_observations WHERE series_id IN (SELECT unnest(?))
    """, [ids])
    out = {}
    for name, sid in cfg.macro_series.items():
        o = obs[obs["series_id"] == sid]
        if o.empty:
            notes.append(f"macro {sid}: sense dades al lake (`fin ingest macro --series {sid}`)")
            continue
        if (o["vintage_mode"] == "latest_only").all():
            # Sèrie revisable carregada sense clau FRED: el valor guardat és el revisat
            # d'avui, que no existia en el passat. Fer-la servir seria look-ahead.
            notes.append(f"macro {sid}: exclosa (revisable sense vintages; cal FIN_FRED_API_KEY)")
            continue
        out[name] = macro_pit_daily(o, index, cfg.macro_lag_sessions)
    return pd.DataFrame(out, index=index)


SP500_BREADTH_SQL = """
WITH members AS (
    SELECT replace(symbol, '.', '-') AS symbol, start_date,
           coalesce(end_date, DATE '9999-12-31') AS end_date
    FROM universe_intervals WHERE universe = 'sp500'
), px AS (
    SELECT p.symbol, p.date, coalesce(p.adj_close, p.close) AS px,
           avg(coalesce(p.adj_close, p.close)) OVER w AS sma,
           count(*) OVER w AS n
    FROM prices_daily p
    WHERE p.symbol IN (SELECT DISTINCT symbol FROM members)
    WINDOW w AS (PARTITION BY p.symbol ORDER BY p.date ROWS BETWEEN {n1} PRECEDING AND CURRENT ROW)
), cal AS (
    SELECT DISTINCT date FROM prices_daily WHERE symbol = ? AND date >= ?
), m AS (
    SELECT c.date, count(*) AS n_members
    FROM cal c JOIN members mb ON mb.start_date <= c.date AND mb.end_date > c.date
    GROUP BY c.date
), b AS (
    SELECT c.date,
           count(*) FILTER (WHERE x.n = {n}) AS n_valid,
           count(*) FILTER (WHERE x.n = {n} AND x.px > x.sma) AS n_above
    FROM cal c
    JOIN members mb ON mb.start_date <= c.date AND mb.end_date > c.date
    JOIN px x ON x.symbol = mb.symbol AND x.date = c.date
    GROUP BY c.date
)
SELECT m.date, m.n_members, coalesce(b.n_valid, 0) AS n_valid, coalesce(b.n_above, 0) AS n_above
FROM m LEFT JOIN b USING (date) ORDER BY m.date
"""


def load_sp500_breadth(ctx, cfg: FeatureConfig, index: pd.DatetimeIndex,
                       notes: list[str]) -> pd.DataFrame:
    """% de membres vigents del S&P 500 per sobre de la seva SMA200.

    La pertinença és point-in-time, però el lake només té preus dels tickers que han estat
    a l'índex en alguna foto pròpia o al grup dinàmic: els que van sortir fa anys sovint no
    tenen preus. `coverage` = membres amb SMA vàlida / membres vigents; per sota de
    `sp500_min_coverage` no es publica el valor (biaix de supervivència massa gran).
    """
    if len(index) == 0:
        return pd.DataFrame()
    n = cfg.sma_long
    sql = SP500_BREADTH_SQL.format(n=n, n1=n - 1)
    try:
        df = ctx.query(sql, [cfg.market, index[0].date()])
    except Exception as e:  # noqa: BLE001
        notes.append(f"amplitud S&P 500 no disponible: {e}")
        return pd.DataFrame()
    if df.empty:
        notes.append("amplitud S&P 500: sense historial de composició (`fin ingest universe-history`)")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").reindex(index)
    coverage = df["n_valid"] / df["n_members"]
    breadth = (df["n_above"] / df["n_valid"]).where(coverage >= cfg.sp500_min_coverage)
    return pd.DataFrame({"breadth_200": breadth, "coverage": coverage}, index=index)


def load_market_data(ctx, cfg: FeatureConfig | None = None, start: str | None = None,
                     end: str | None = None, *, include_sp500: bool = True) -> MarketData:
    cfg = cfg or FeatureConfig()
    notes: list[str] = []
    uni = ctx.universe
    groups = {}
    for g in dict.fromkeys((*cfg.panel_groups, cfg.breadth_group)):
        try:
            groups[g] = [i.symbol for i in uni.group(g)]
        except KeyError:
            notes.append(f"grup '{g}' no definit a universe.yaml")
    symbols = required_symbols(cfg, groups)
    prices = load_prices(ctx, symbols, start, end)
    if prices.empty or cfg.market not in prices:
        raise RuntimeError(f"sense preus de {cfg.market} al lake: executa `fin ingest prices`")
    missing = sorted(set(symbols) - set(prices.columns))
    if missing:
        notes.append(f"sense preus: {', '.join(missing)}")
    # Calendari mestre: sessions amb tancament del mercat de referència
    prices = prices.loc[prices[cfg.market].notna()]
    # Forats puntuals (festius diferents, dades absents un dia): arrossega com a molt 3 sessions
    prices = prices.ffill(limit=3)
    index = prices.index
    macro = load_macro(ctx, cfg, index, notes)
    breadth = load_sp500_breadth(ctx, cfg, index, notes) if include_sp500 else pd.DataFrame()
    for n_ in notes:
        log.warning(n_)
    return MarketData(prices=prices, macro=macro, sp500_breadth=breadth,
                      groups={g: [s for s in v if s in prices.columns] for g, v in groups.items()},
                      notes=notes)
