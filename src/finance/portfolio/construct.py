"""Construcció de la cartera objectiu: règim (quant risc) × rotació (on) + cobertura.

    objectiu_t = E_t · sectors_t + (1 - E_t) · cobertura_t

- E_t: exposició a renda variable del règim (capa 3), decidida al tancament de t.
- sectors_t: top-k sectors per momentum ajustat per risc, amb selecció conscient de la
  correlació i pesos inversos a la volatilitat. En règim baixista, inclinació a beta baixa.
- cobertura_t: bons (si la correlació accions-bons és negativa), or + cash (si no) o cash.

Els blocs es recalculen a les dates de revisió: primera sessió de cada mes i cada canvi de
règim. Saber que avui és la primera sessió del mes només requereix el calendari fins avui,
així que la regla és causal (el test de truncament ho verifica). L'objectiu decidit al
tancament de t s'executa al tancament de t i guanya el retorn de t+1 (motor `backtest`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finance.portfolio.config import CASH, PortfolioConfig

PANEL_FEATURES = ("risk_adj_mom", "rel_strength_6m", "vol_3m", "beta_1y")


@dataclass
class PortfolioInputs:
    prices: pd.DataFrame  # sessions × símbols (adj_close): mercat, sectors, cobertures
    cash_ret: pd.Series  # retorn diari del cash (t-1 → t)
    regime: pd.DataFrame  # columnes: regime, trend, equity_exposure
    panel: dict[str, pd.DataFrame]  # feature → sessions × sectors
    stock_bond_corr: pd.Series
    sectors: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.prices.index

    def truncate(self, end) -> PortfolioInputs:
        end = pd.Timestamp(end)

        def cut(x):
            return x.loc[x.index <= end].copy()

        return PortfolioInputs(prices=cut(self.prices), cash_ret=cut(self.cash_ret),
                               regime=cut(self.regime),
                               panel={k: cut(v) for k, v in self.panel.items()},
                               stock_bond_corr=cut(self.stock_bond_corr),
                               sectors=list(self.sectors), notes=list(self.notes))

    def asset_returns(self) -> pd.DataFrame:
        """Retorns simples diaris (t-1 → t) de tots els actius, inclòs CASH."""
        r = self.prices / self.prices.shift(1) - 1
        r[CASH] = self.cash_ret.reindex(self.index).fillna(0.0)
        return r


# ---------------------------------------------------------------------------------------
# Dates de revisió
# ---------------------------------------------------------------------------------------

def review_dates(index: pd.DatetimeIndex, regime: pd.Series) -> pd.Series:
    """True a la primera sessió de cada mes i a cada canvi de règim (causal)."""
    month = pd.Series(index.to_period("M"), index=index)
    new_month = month.ne(month.shift())
    r = regime.reindex(index)
    changed = r.ne(r.shift()) & r.notna()
    return (new_month | changed) & r.notna()


# ---------------------------------------------------------------------------------------
# Blocs
# ---------------------------------------------------------------------------------------

def sector_scores(panel: dict[str, pd.DataFrame], t: pd.Timestamp, bear: bool,
                  cfg: PortfolioConfig) -> pd.Series:
    ram = panel["risk_adj_mom"].loc[t]
    vol = panel["vol_3m"].loc[t]
    ok = ram.notna() & vol.notna() & (vol > 0)
    if not ok.any():
        return pd.Series(dtype=float)
    score = ram[ok].rank(pct=True)
    rs = panel["rel_strength_6m"].loc[t][ok]
    score = score + cfg.rs_weight * rs.rank(pct=True).fillna(0.5)
    if bear and cfg.defensive_tilt:
        beta = panel["beta_1y"].loc[t][ok]
        score = score + cfg.low_beta_weight * (-beta).rank(pct=True).fillna(0.5)
    return score.sort_values(ascending=False, kind="stable")


def select_sectors(scores: pd.Series, corr: pd.DataFrame | None, cfg: PortfolioConfig) -> tuple[list[str], bool]:
    """Selecció voraç per puntuació evitant parells massa correlacionats.

    Retorna (seleccionats, el_filtre_ha_canviat_la_selecció).
    """
    ranked = list(scores.index)
    naive = ranked[: cfg.top_k]
    if corr is None or cfg.max_pair_corr >= 1:
        return naive, False
    chosen: list[str] = []
    for s in ranked:
        if not _too_correlated(s, chosen, corr, cfg.max_pair_corr):
            chosen.append(s)
        if len(chosen) == cfg.top_k:
            break
    for s in ranked:  # places buides: els millors restants
        if len(chosen) == cfg.top_k:
            break
        if s not in chosen:
            chosen.append(s)
    return chosen, set(chosen) != set(naive)


def _too_correlated(s: str, chosen: list[str], corr: pd.DataFrame, threshold: float) -> bool:
    if s not in corr.index:
        return False  # sense història suficient per estimar-la: no es penalitza
    for x in chosen:
        if x in corr.columns:
            c = corr.at[s, x]
            if pd.notna(c) and c >= threshold:
                return True
    return False


def _max_pair(corr: pd.DataFrame | None, chosen: list[str]) -> float:
    if corr is None or not set(chosen) <= set(corr.index) or len(chosen) < 2:
        return np.nan
    m = corr.loc[chosen, chosen].to_numpy()
    return float(m[np.triu_indices(len(chosen), 1)].max())


def sleeve_weights(chosen: list[str], vol: pd.Series, cfg: PortfolioConfig) -> dict[str, float]:
    if not chosen:
        return {}
    if cfg.inverse_vol:
        iv = 1.0 / vol[chosen]
        w = iv / iv.sum()
    else:
        w = pd.Series(1.0 / len(chosen), index=chosen)
    return {k: float(v) for k, v in w.items()}


def hedge_weights(sb_corr: float, available: set[str], cfg: PortfolioConfig) -> tuple[dict[str, float], str]:
    if not cfg.hedge or pd.isna(sb_corr):
        return {CASH: 1.0}, "cash"
    if sb_corr < cfg.bond_hedge_max_corr and cfg.bond_hedge in available:
        return {cfg.bond_hedge: 1.0}, "bons"
    if cfg.gold in available:
        return {cfg.gold: cfg.gold_share, CASH: 1.0 - cfg.gold_share}, "or+cash"
    return {CASH: 1.0}, "cash"


# ---------------------------------------------------------------------------------------
# Objectius diaris
# ---------------------------------------------------------------------------------------

@dataclass
class Targets:
    weights: pd.DataFrame  # sessions × actius (inclòs CASH); files NaN abans de començar
    reviews: pd.Series  # bool: data de revisió (es pot negociar)
    log: pd.DataFrame  # una fila per revisió: règim, exposició, sectors, cobertura...


def build_targets(inp: PortfolioInputs, cfg: PortfolioConfig | None = None,
                  exposure: pd.Series | None = None) -> Targets:
    """Pesos objectiu diaris. `exposure` permet substituir E_t (p. ex. 1.0 = sense règim)."""
    cfg = cfg or PortfolioConfig()
    idx = inp.index
    reg = inp.regime.reindex(idx)
    expo = (exposure.reindex(idx) if exposure is not None else reg["equity_exposure"]).astype(float)
    regime_label = reg["regime"].where(expo.notna())
    # Sense substitució: revisió a cada canvi de règim; amb substitució, a cada canvi d'E
    rev = review_dates(idx, regime_label if exposure is None else expo)
    sb_corr = inp.stock_bond_corr.reindex(idx)
    log_ret = np.log(inp.prices[inp.sectors]).diff()
    assets = list(dict.fromkeys([*inp.sectors, cfg.bond_hedge, cfg.gold, CASH]))
    W = np.full((len(idx), len(assets)), np.nan)
    col = {a: i for i, a in enumerate(assets)}
    rows = []
    current: np.ndarray | None = None
    started = False
    for i, t in enumerate(idx):
        if rev.iloc[i]:
            bear = reg["trend"].iloc[i] == "bear"
            scores = sector_scores(inp.panel, t, bear, cfg)
            if len(scores) >= cfg.top_k:
                lo = max(0, i - cfg.corr_window + 1)
                window = log_ret.iloc[lo: i + 1]
                corr = window.dropna(axis=1, how="any").corr() if i + 1 >= cfg.corr_window else None
                chosen, bound = select_sectors(scores, corr, cfg)
                sw = sleeve_weights(chosen, inp.panel["vol_3m"].loc[t], cfg)
                avail = {a for a in (cfg.bond_hedge, cfg.gold)
                         if a in inp.prices and pd.notna(inp.prices.at[t, a])}
                hw, hname = hedge_weights(sb_corr.iloc[i], avail, cfg)
                e = float(expo.iloc[i])
                vec = np.zeros(len(assets))
                for a, w in sw.items():
                    vec[col[a]] += e * w
                for a, w in hw.items():
                    vec[col[a]] += (1 - e) * w
                current, started = vec, True
                rows.append({"date": t, "regime": regime_label.iloc[i], "equity_exposure": e,
                             "sectors": ",".join(chosen), "corr_filter_bound": bound,
                             "max_corr_chosen": _max_pair(corr, chosen),
                             "hedge": hname, "n_candidates": len(scores)})
            elif started:
                rows.append({"date": t, "regime": regime_label.iloc[i], "equity_exposure": float(expo.iloc[i]),
                             "sectors": "", "corr_filter_bound": False, "max_corr_chosen": np.nan,
                             "hedge": "sense canvis", "n_candidates": len(scores)})
        if started:
            W[i] = current
    weights = pd.DataFrame(W, index=idx, columns=assets)
    log = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return Targets(weights=weights, reviews=rev & weights.notna().all(axis=1), log=log)


def fixed_targets(index: pd.DatetimeIndex, weights: dict[str, float] | pd.DataFrame,
                  monthly: bool = True, start: pd.Timestamp | None = None) -> Targets:
    """Objectius per a referències (buy & hold, 60/40...). `weights` constant o per data."""
    if isinstance(weights, dict):
        W = pd.DataFrame([weights] * len(index), index=index)
    else:
        W = weights.reindex(index)
    if start is not None:
        W = W.loc[W.index >= start].reindex(index)
    month = pd.Series(index.to_period("M"), index=index)
    rev = month.ne(month.shift()) if monthly else pd.Series(False, index=index)
    valid = W.notna().all(axis=1)
    first = valid.idxmax() if valid.any() else None
    if first is not None:
        rev.loc[first] = True
    return Targets(weights=W, reviews=rev & valid, log=pd.DataFrame())
