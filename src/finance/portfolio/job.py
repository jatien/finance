"""Job `portfolio`: cartera objectiu del sistema complet i registre de decisions al lake."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from finance.features.config import FeatureConfig
from finance.features.inputs import load_prices
from finance.features.transforms import rolling_corr
from finance.features.job import read_market_features, read_sector_features
from finance.jobs import Job, Run
from finance.portfolio.config import CASH, PortfolioConfig
from finance.portfolio.construct import PANEL_FEATURES, PortfolioInputs, Targets, build_targets
from finance.regimes.job import cash_returns, read_regimes
from finance.storage.schemas import PORTFOLIO_TARGETS


def load_inputs(ctx, cfg: PortfolioConfig | None = None) -> PortfolioInputs:
    cfg = cfg or PortfolioConfig()
    notes: list[str] = []
    regime = read_regimes(ctx, cfg.regime_set)
    if regime.empty:
        raise RuntimeError(f"sense règims '{cfg.regime_set}': executa `fin regimes build`")
    sectors = [i.symbol for i in ctx.universe.group(cfg.sector_group)]
    long_bond = FeatureConfig().long_bond
    px = load_prices(ctx, list(dict.fromkeys([cfg.market, *sectors, cfg.bond_hedge, cfg.gold, long_bond])))
    if px.empty or cfg.market not in px:
        raise RuntimeError(f"sense preus de {cfg.market}")
    px = px.loc[px[cfg.market].notna()].ffill(limit=3)
    for s in (cfg.bond_hedge, cfg.gold):
        if s not in px:
            notes.append(f"sense preus de {s}: la cobertura farà servir cash")
            px[s] = np.nan
    idx = px.index
    panel = {}
    for f in PANEL_FEATURES:
        w = read_sector_features(ctx, f, cfg.feature_set, "value", group=cfg.sector_group)
        if w.empty:
            raise RuntimeError(f"sense panell sectorial '{f}': executa `fin features build`")
        panel[f] = w.reindex(index=idx, columns=sectors)
    mkt = read_market_features(ctx, cfg.feature_set, "value")
    sb = mkt["stock_bond_corr_3m"].reindex(idx) if "stock_bond_corr_3m" in mkt else pd.Series(np.nan, index=idx)
    fast = {}
    if cfg.hedge_fast_window:
        if long_bond in px and px[long_bond].notna().any():
            fast[cfg.hedge_fast_window] = rolling_corr(px[cfg.market], px[long_bond], cfg.hedge_fast_window)
        else:
            notes.append(f"sense preus de {long_bond}: la cobertura no té confirmació ràpida")
    if long_bond not in (cfg.market, *sectors, cfg.bond_hedge, cfg.gold):
        px = px.drop(columns=long_bond, errors="ignore")
    return PortfolioInputs(prices=px, cash_ret=cash_returns(ctx, idx, notes),
                           regime=regime.reindex(idx), panel=panel, stock_bond_corr=sb,
                           sectors=[s for s in sectors if s in px], notes=notes,
                           stock_bond_corr_fast=fast)


def sleeve_of(symbol: str, cfg: PortfolioConfig) -> str:
    if symbol == CASH:
        return "cash"
    if symbol in (cfg.bond_hedge, cfg.gold):
        return "cobertura"
    if symbol == cfg.market:
        return "nucli"
    return "renda_variable"


def decision_log(t: Targets, cfg: PortfolioConfig) -> pd.DataFrame:
    """Una fila per (revisió, actiu) amb pes objectiu o anterior > 0, i l'acció suggerida.

    L'acció es mesura respecte de l'objectiu de la revisió anterior (no de la cartera real,
    que depèn de l'execució de cadascú): enter, exit, add/reduce (canvi > banda) o hold.
    """
    W = t.weights[t.reviews].fillna(0.0)
    if W.empty:
        return pd.DataFrame(columns=PORTFOLIO_TARGETS.columns)
    prev = W.shift(1).fillna(0.0)
    tw = W.stack().rename("target_weight")
    pw = prev.stack().rename("prev_weight")
    df = pd.concat([tw, pw], axis=1)
    df.index.names = ["date", "symbol"]
    df = df[(df["target_weight"] > 1e-9) | (df["prev_weight"] > 1e-9)].reset_index()
    df["change"] = df["target_weight"] - df["prev_weight"]
    df["action"] = np.select(
        [(df["prev_weight"] <= 1e-9), (df["target_weight"] <= 1e-9),
         df["change"] > cfg.band, df["change"] < -cfg.band],
        ["enter", "exit", "add", "reduce"], default="hold")
    first = df["date"] == df["date"].min()
    df.loc[first, "action"] = "enter"
    df["sleeve"] = df["symbol"].map(lambda s: sleeve_of(s, cfg))
    if not t.log.empty:
        info = t.log[["regime", "equity_exposure", "hedge"]]
        df = df.join(info, on="date")
    df["strategy_set"] = cfg.name
    return df


class PortfolioJob(Job):
    name = "portfolio"

    def execute(self, run: Run, **_: Any) -> None:
        cfg = PortfolioConfig()
        run.params["portfolio_config"] = cfg.to_params()
        inp = load_inputs(self.ctx, cfg)
        t = build_targets(inp, cfg)
        log = decision_log(t, cfg)
        if log.empty:
            run.skip("sense cartera: cal un any de momentum sectorial i règims definits")
            return
        run.write(PORTFOLIO_TARGETS, log, f"portfolio:{cfg.name}")
        run.ok()
        last = log[log["date"] == log["date"].max()]
        alloc = ", ".join(f"{r.symbol} {r.target_weight:.0%}" for r in
                          last[last["target_weight"] > 0].sort_values("target_weight", ascending=False).itertuples())
        run.message = "; ".join([f"{cfg.name} {last['date'].max():%Y-%m-%d}: {alloc}", *inp.notes])


def read_targets(ctx, strategy_set: str = "rot_v1") -> pd.DataFrame:
    df = ctx.query("SELECT * EXCLUDE (source, ingested_at, run_id, year) FROM portfolio_targets "
                   "WHERE strategy_set = ? ORDER BY date, target_weight DESC", [strategy_set])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
