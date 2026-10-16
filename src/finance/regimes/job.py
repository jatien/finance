"""Job `regimes`: classifica cada sessió a partir de les features del lake i ho hi escriu.

També carrega les entrades de l'avaluació (preus del mercat i sectors, tipus del cash) per
a l'informe `fin regimes report`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from finance.features.config import TRADING_DAYS, FeatureConfig
from finance.features.inputs import load_prices, macro_pit_daily
from finance.features.job import read_market_features
from finance.jobs import Job, Run
from finance.regimes.config import RegimeConfig
from finance.regimes.rules import classify
from finance.storage.schemas import REGIMES_DAILY


def load_features(ctx, cfg: RegimeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = read_market_features(ctx, cfg.feature_set, "value")
    if values.empty:
        raise RuntimeError(f"sense features '{cfg.feature_set}': executa `fin features build`")
    pct = read_market_features(ctx, cfg.feature_set, "pct_rank").reindex(values.index)
    return values, pct


class RegimesJob(Job):
    name = "regimes"

    def execute(self, run: Run, **_: Any) -> None:
        cfg = RegimeConfig()
        run.params["regime_config"] = cfg.to_params()
        values, pct = load_features(self.ctx, cfg)
        reg = classify(values, pct, cfg).dropna(subset=["regime"])
        if reg.empty:
            run.skip("cap sessió amb règim definit: falten features (SMA200 del mercat i VIX)")
            return
        df = reg.reset_index(names="date")
        df["regime_set"] = cfg.name
        df["days_in_regime"] = df["days_in_regime"].astype("int32")
        run.write(REGIMES_DAILY, df, f"regimes:{cfg.name}")
        run.ok()
        last = reg.iloc[-1]
        run.message = (f"{cfg.name} {reg.index[-1]:%Y-%m-%d}: {last['regime']} "
                       f"({int(last['days_in_regime'])} sessions), exposició {last['equity_exposure']:.0%}")


def read_regimes(ctx, regime_set: str = "rules_v1") -> pd.DataFrame:
    df = ctx.query("SELECT * EXCLUDE (source, ingested_at, run_id, year) FROM regimes_daily "
                   "WHERE regime_set = ? ORDER BY date", [regime_set])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# ---------------------------------------------------------------------------------------
# Entrades de l'avaluació
# ---------------------------------------------------------------------------------------

@dataclass
class EvalInputs:
    values: pd.DataFrame
    pct: pd.DataFrame
    market: pd.Series  # preus del mercat (adj_close)
    asset_ret: pd.Series  # retorn diari del mercat (t-1 → t)
    cash_ret: pd.Series  # retorn diari del cash (t-1 → t)
    sectors: pd.DataFrame  # preus dels 11 sectors
    notes: list[str] = field(default_factory=list)


def load_eval_inputs(ctx, cfg: RegimeConfig | None = None) -> EvalInputs:
    cfg = cfg or RegimeConfig()
    fcfg = FeatureConfig()
    values, pct = load_features(ctx, cfg)
    idx = values.index
    sector_syms = [i.symbol for i in ctx.universe.group(fcfg.breadth_group)]
    px = load_prices(ctx, [fcfg.market, *sector_syms]).reindex(idx).ffill(limit=3)
    market = px[fcfg.market]
    notes: list[str] = []
    cash = cash_returns(ctx, idx, notes)
    asset_ret = market / market.shift(1) - 1
    return EvalInputs(values=values, pct=pct, market=market, asset_ret=asset_ret,
                      cash_ret=cash, sectors=px[[s for s in sector_syms if s in px]], notes=notes)


def cash_returns(ctx, index: pd.DatetimeIndex, notes: list[str]) -> pd.Series:
    """Retorn diari del cash (t-1 → t): T-bill 3M de FRED (DGS3MO), disponible l'endemà."""
    obs = ctx.query("""SELECT date, value, realtime_start, vintage_mode FROM macro_observations
                       WHERE series_id = 'DGS3MO'""")
    if obs.empty:
        notes.append("sense DGS3MO: el cash rendeix 0 % (`fin ingest macro --series DGS3MO`)")
        return pd.Series(0.0, index=index)
    rate = macro_pit_daily(obs, index, FeatureConfig().macro_lag_sessions).ffill()
    cash = (rate / 100 / TRADING_DAYS).shift(1).fillna(0.0)
    return cash.replace([np.inf, -np.inf], 0.0)
