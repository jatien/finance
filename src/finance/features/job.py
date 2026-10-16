"""Job `features`: calcula les features des del lake i les hi escriu.

Es regenera tota la història a cada execució (és barat i garanteix coherència si el
proveïdor reajusta preus). Els datasets tenen `snapshot_by=feature_set`: la vista només
mostra l'última execució de cada conjunt; la compactació elimina les versions antigues.
La configuració completa queda guardada a `ingest_runs.params` (backlog del model).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from finance.features.config import FeatureConfig
from finance.features.inputs import load_market_data
from finance.features.market import CATALOG, compute_market_features, percentile_ranks
from finance.features.sectors import compute_sector_panel
from finance.jobs import Job, Run
from finance.storage.schemas import FEATURES_DAILY, SECTOR_FEATURES_DAILY


def market_long(features: pd.DataFrame, pct: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    v = features.stack(future_stack=True).rename("value")
    p = pct.reindex_like(features).stack(future_stack=True).rename("pct_rank")
    df = pd.concat([v, p], axis=1)
    df.index.names = ["date", "feature"]
    df = df.dropna(subset=["value"]).reset_index()
    df["family"] = df["feature"].map(lambda f: CATALOG[f].family)
    df["feature_set"] = feature_set
    return df


class FeaturesJob(Job):
    name = "features"

    def execute(self, run: Run, start: str | None = None, include_sp500: bool = True,
                **_: Any) -> None:
        cfg = FeatureConfig()
        run.params["feature_config"] = cfg.to_params()
        data = load_market_data(self.ctx, cfg, start=start, include_sp500=include_sp500)

        feats = compute_market_features(data, cfg)
        pct = percentile_ranks(feats, cfg)
        n = run.write(FEATURES_DAILY, market_long(feats, pct, cfg.name), f"features:{cfg.name}")
        self.log.info("%d features de mercat, %d files", feats.shape[1], n)
        run.ok()

        panel = compute_sector_panel(data, cfg)
        panel["feature_set"] = cfg.name
        n = run.write(SECTOR_FEATURES_DAILY, panel, f"features:{cfg.name}")
        self.log.info("panell sectorial: %d símbols, %d files", panel["symbol"].nunique(), n)
        run.ok()

        last = feats.index.max()
        run.message = "; ".join([f"{cfg.name} fins a {last:%Y-%m-%d}", *data.notes])


# ---------------------------------------------------------------------------------------
# Lectura (per a la capa de règims, notebooks i informes)
# ---------------------------------------------------------------------------------------

def read_market_features(ctx, feature_set: str = "core_v1", column: str = "value") -> pd.DataFrame:
    """Features de mercat en format ample (sessions × features). `column`: value | pct_rank."""
    if column not in ("value", "pct_rank"):
        raise ValueError("column ha de ser 'value' o 'pct_rank'")
    df = ctx.query(f"SELECT date, feature, {column} AS v FROM features_daily WHERE feature_set = ?",
                   [feature_set])
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="date", columns="feature", values="v")
    wide.index = pd.to_datetime(wide.index)
    return wide[[c for c in CATALOG if c in wide]].sort_index()


def read_sector_features(ctx, feature: str, feature_set: str = "core_v1",
                         column: str = "value", group: str | None = None) -> pd.DataFrame:
    """Una feature del panell en format ample (sessions × símbols). `column`: value | xs_rank."""
    if column not in ("value", "xs_rank"):
        raise ValueError("column ha de ser 'value' o 'xs_rank'")
    sql = (f"SELECT date, symbol, {column} AS v FROM sector_features_daily "
           "WHERE feature_set = ? AND feature = ?")
    params: list = [feature_set, feature]
    if group:
        sql += " AND group_name = ?"
        params.append(group)
    df = ctx.query(sql, params)
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="date", columns="symbol", values="v")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()
