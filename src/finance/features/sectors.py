"""Panell de features per ETF de sector i d'indústria (base de la rotació sectorial).

Cada feature té també un `xs_rank`: el percentil transversal dins del seu grup
(sectors GICS per una banda, indústries per l'altra) entre els símbols amb dades aquell
dia. Un ETF que encara no cotitzava simplement no hi participa: l'univers creix sense
fer servir tickers del futur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finance.features import transforms as tf
from finance.features.config import FeatureConfig
from finance.features.inputs import MarketData

PANEL_FEATURES: dict[str, str] = {
    "mom_12_1": "Retorn 12-1 mesos",
    "mom_6_1": "Retorn 6-1 mesos",
    "mom_3m": "Retorn 3 mesos",
    "vol_3m": "Volatilitat realitzada 63 sessions, anualitzada",
    "risk_adj_mom": "Momentum 12-1 / volatilitat 1 any",
    "rel_strength_3m": "Força relativa 3 mesos vs mercat (log)",
    "rel_strength_6m": "Força relativa 6 mesos vs mercat (log)",
    "dist_sma200": "Preu / SMA200 - 1",
    "beta_1y": "Beta respecte del mercat (252 sessions)",
}


def compute_group_panel(px: pd.DataFrame, market: pd.Series, cfg: FeatureConfig) -> dict[str, pd.DataFrame]:
    """Per a un grup de símbols: {feature: DataFrame sessions × símbols}."""
    mom12 = tf.momentum(px, cfg.mom_long, cfg.mom_skip)
    log_rel = np.log(px.div(market, axis=0))
    return {
        "mom_12_1": mom12,
        "mom_6_1": tf.momentum(px, cfg.mom_mid, cfg.mom_skip),
        "mom_3m": tf.momentum(px, cfg.mom_short),
        "vol_3m": tf.realized_vol(px, cfg.vol_mid),
        "risk_adj_mom": mom12 / tf.realized_vol(px, cfg.vol_long),
        "rel_strength_3m": log_rel - log_rel.shift(cfg.mom_short),
        "rel_strength_6m": log_rel - log_rel.shift(cfg.mom_mid),
        "dist_sma200": tf.dist_to_sma(px, cfg.sma_long),
        "beta_1y": tf.rolling_beta(px, market, cfg.beta_window),
    }


def compute_sector_panel(data: MarketData, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Format llarg: date, symbol, group_name, feature, value, xs_rank."""
    cfg = cfg or FeatureConfig()
    px = data.prices
    market = px[cfg.market]
    frames = []
    seen: set[str] = set()
    for group in cfg.panel_groups:
        syms = [s for s in data.groups.get(group, []) if s in px and s not in seen]
        seen.update(syms)
        if not syms:
            continue
        panel = compute_group_panel(px[syms], market, cfg)
        for feat, wide in panel.items():
            ranks = tf.cross_sectional_rank(wide)
            long_ = pd.DataFrame({
                "value": wide.stack(future_stack=True),
                "xs_rank": ranks.stack(future_stack=True),
            })
            long_.index.names = ["date", "symbol"]
            long_ = long_.dropna(subset=["value"]).reset_index()
            long_["group_name"] = group
            long_["feature"] = feat
            frames.append(long_)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "group_name", "feature", "value", "xs_rank"])
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "symbol", "group_name", "feature", "value", "xs_rank"]]
