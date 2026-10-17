"""Dades addicionals per a l'inversor a Espanya: EUR/USD, tipus del BCE, dividends."""

from __future__ import annotations

import pandas as pd


def macro_series(ctx, series_id: str) -> pd.Series:
    """Sèrie macro no revisable (última versió). Retorna una sèrie indexada per data."""
    df = ctx.query("SELECT date, value FROM macro_observations WHERE series_id = ? ORDER BY date", [series_id])
    if df.empty:
        raise RuntimeError(f"sense dades de {series_id}: executa `fin ingest macro --series {series_id}`")
    s = df.set_index(pd.to_datetime(df["date"]))["value"].astype(float)
    return s[~s.index.duplicated(keep="last")]


def dividends(ctx, symbols: list[str]) -> pd.DataFrame:
    df = ctx.query("SELECT date, symbol, dividend FROM prices_daily WHERE symbol IN (SELECT unnest(?)) "
                   "AND dividend > 0", [symbols])
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot_table(index="date", columns="symbol", values="dividend", aggfunc="sum")
