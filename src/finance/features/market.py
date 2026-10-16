"""Features d'estat de mercat (una sèrie per feature, compartida per tot l'univers).

Cada feature declara la seva família i el sentit de l'estrès (`stress_sign`): +1 si un
valor alt indica estrès/risc (VIX, correlació...), -1 si un valor alt indica mercat sa
(tendència, amplitud). La capa de règims ho fa servir per orientar les regles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance.features import transforms as tf
from finance.features.config import FeatureConfig
from finance.features.inputs import MarketData


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    family: str
    stress_sign: int
    description: str


CATALOG: dict[str, FeatureInfo] = {f.name: f for f in [
    # Tendència
    FeatureInfo("mkt_dist_sma200", "trend", -1, "Mercat / SMA200 - 1"),
    FeatureInfo("mkt_sma50_sma200", "trend", -1, "SMA50 / SMA200 - 1 (creu daurada/de la mort)"),
    FeatureInfo("mkt_mom_12_1", "trend", -1, "Retorn 12-1 mesos del mercat"),
    FeatureInfo("mkt_mom_3m", "trend", -1, "Retorn 3 mesos del mercat"),
    FeatureInfo("mkt_drawdown_1y", "trend", -1, "Caiguda des del màxim d'1 any (<= 0)"),
    # Volatilitat
    FeatureInfo("mkt_rvol_1m", "volatility", 1, "Volatilitat realitzada 21 sessions, anualitzada"),
    FeatureInfo("mkt_rvol_ratio", "volatility", 1, "Vol realitzada 1 mes / 1 any"),
    FeatureInfo("vix", "volatility", 1, "Nivell del VIX"),
    FeatureInfo("vix_term", "volatility", 1, "VIX / VIX3M (> 1: corba invertida, estrès agut)"),
    FeatureInfo("vix_rvol_spread", "volatility", -1,
                "VIX - vol realitzada 1 mes (prima de risc de volatilitat; negativa en pànic)"),
    # Amplitud
    FeatureInfo("sector_breadth_200", "breadth", -1, "% de sectors per sobre de la SMA200"),
    FeatureInfo("sector_breadth_50", "breadth", -1, "% de sectors per sobre de la SMA50"),
    FeatureInfo("sp500_breadth_200", "breadth", -1,
                "% de membres vigents del S&P 500 per sobre de la SMA200"),
    FeatureInfo("ew_vs_cw_3m", "breadth", -1,
                "Força relativa 3 mesos del S&P equiponderat (RSP) vs ponderat (SPY)"),
    FeatureInfo("defensive_vs_cyclical_3m", "breadth", 1,
                "Força relativa 3 mesos de sectors defensius vs cíclics"),
    # Crèdit
    FeatureInfo("hy_vs_treasury_3m", "credit", -1, "Força relativa 3 mesos HYG vs IEF"),
    FeatureInfo("baa_spread", "credit", 1, "Spread Baa - Treasury 10Y (punts percentuals)"),
    FeatureInfo("baa_spread_chg_3m", "credit", 1, "Canvi del spread Baa en 3 mesos"),
    FeatureInfo("nfci", "credit", 1, "Chicago Fed NFCI (> 0: condicions més tenses que la mitjana)"),
    # Tipus
    FeatureInfo("curve_10y3m", "rates", -1, "Pendent 10Y - 3M (invertida < 0)"),
    # Correlació
    FeatureInfo("sector_avg_corr_3m", "correlation", 1,
                "Correlació mitjana entre sectors (63 sessions)"),
    FeatureInfo("sector_absorption_1y", "correlation", 1,
                "Absorption ratio de la matriu de correlació sectorial (252 sessions)"),
    FeatureInfo("stock_bond_corr_3m", "correlation", 1,
                "Correlació SPY-TLT (63 sessions): > 0 vol dir que els bons no cobreixen"),
]}


def _get(px: pd.DataFrame, sym: str) -> pd.Series:
    return px[sym] if sym in px else pd.Series(np.nan, index=px.index, dtype=float)


def _basket(px: pd.DataFrame, syms: tuple[str, ...]) -> pd.Series:
    """Índex equiponderat (rebalanceig diari) dels símbols disponibles, en base 1."""
    cols = [s for s in syms if s in px]
    if not cols:
        return pd.Series(np.nan, index=px.index, dtype=float)
    r = tf.log_returns(px[cols]).mean(axis=1, skipna=True)
    level = np.exp(r.fillna(0).cumsum())
    return level.where(px[cols].notna().any(axis=1))


def compute_market_features(data: MarketData, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """DataFrame ample (sessions × features) amb els valors en brut."""
    cfg = cfg or FeatureConfig()
    px = data.prices
    m = _get(px, cfg.market)
    f: dict[str, pd.Series] = {}

    # Tendència -------------------------------------------------------------------------
    f["mkt_dist_sma200"] = tf.dist_to_sma(m, cfg.sma_long)
    f["mkt_sma50_sma200"] = tf.sma(m, cfg.sma_short) / tf.sma(m, cfg.sma_long) - 1
    f["mkt_mom_12_1"] = tf.momentum(m, cfg.mom_long, cfg.mom_skip)
    f["mkt_mom_3m"] = tf.momentum(m, cfg.mom_short)
    f["mkt_drawdown_1y"] = tf.rolling_drawdown(m, cfg.dd_window)

    # Volatilitat -----------------------------------------------------------------------
    rv = tf.realized_vol(m, cfg.vol_short)
    f["mkt_rvol_1m"] = rv
    f["mkt_rvol_ratio"] = rv / tf.realized_vol(m, cfg.vol_long)
    vix = _get(px, cfg.vix)
    f["vix"] = vix
    f["vix_term"] = vix / _get(px, cfg.vix3m)
    f["vix_rvol_spread"] = vix - 100 * rv

    # Amplitud --------------------------------------------------------------------------
    sectors = [s for s in data.groups.get(cfg.breadth_group, []) if s in px]
    if sectors:
        f["sector_breadth_200"] = tf.fraction_above_sma(px[sectors], cfg.sma_long)
        f["sector_breadth_50"] = tf.fraction_above_sma(px[sectors], cfg.sma_short)
    if "breadth_200" in data.sp500_breadth:
        f["sp500_breadth_200"] = data.sp500_breadth["breadth_200"].reindex(px.index)
    f["ew_vs_cw_3m"] = tf.log_ratio_change(_get(px, cfg.equal_weight), m, cfg.mom_short)
    f["defensive_vs_cyclical_3m"] = tf.log_ratio_change(
        _basket(px, cfg.defensive), _basket(px, cfg.cyclical), cfg.mom_short)

    # Crèdit i tipus --------------------------------------------------------------------
    f["hy_vs_treasury_3m"] = tf.log_ratio_change(_get(px, cfg.high_yield), _get(px, cfg.mid_bond),
                                                 cfg.mom_short)
    macro = data.macro.reindex(px.index) if not data.macro.empty else pd.DataFrame(index=px.index)
    if "baa_spread" in macro:
        f["baa_spread"] = macro["baa_spread"]
        f["baa_spread_chg_3m"] = macro["baa_spread"] - macro["baa_spread"].shift(cfg.mom_short)
    if "nfci" in macro:
        f["nfci"] = macro["nfci"]
    if "curve_10y3m" in macro:
        f["curve_10y3m"] = macro["curve_10y3m"]

    # Correlació ------------------------------------------------------------------------
    if sectors:
        short = tf.correlation_structure(px[sectors], cfg.corr_short, cfg.absorption_k_frac)
        long_ = tf.correlation_structure(px[sectors], cfg.corr_long, cfg.absorption_k_frac)
        f["sector_avg_corr_3m"] = short["avg_corr"]
        f["sector_absorption_1y"] = long_["absorption"]
    f["stock_bond_corr_3m"] = tf.rolling_corr(m, _get(px, cfg.long_bond), cfg.corr_short)

    out = pd.DataFrame(f, index=px.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    # Ordre del catàleg (les absents per manca de dades simplement no hi són)
    return out[[c for c in CATALOG if c in out]]


def percentile_ranks(features: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FeatureConfig()
    return tf.expanding_pct_rank(features, cfg.pct_min_periods)
