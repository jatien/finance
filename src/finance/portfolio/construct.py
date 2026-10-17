"""Construcció de la cartera objectiu: règim (quant risc) × rotació (on) + cobertura.

    objectiu_t = E_t · sectors_t + (1 - E_t) · cobertura_t

- E_t: exposició a renda variable del règim (capa 3), decidida al tancament de t.
- sectors_t: bloc de renda variable segons la regla del règim (`SleeveRule`):
      c · mercat + (1 - c) · top-k sectors
  amb selecció conscient de la correlació i pesos inversos a la volatilitat. A rot_v1, c = 0 i
  la puntuació és momentum ajustat per risc (amb inclinació a beta baixa en tendència
  baixista); a rot_v2 el nucli i la puntuació canvien amb el règim.
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

from finance.portfolio.config import CASH, PortfolioConfig, SleeveRule

PANEL_FEATURES = ("risk_adj_mom", "mom_12_1", "rel_strength_6m", "vol_3m", "beta_1y")


@dataclass
class PortfolioInputs:
    prices: pd.DataFrame  # sessions × símbols (adj_close): mercat, sectors, cobertures
    cash_ret: pd.Series  # retorn diari del cash (t-1 → t)
    regime: pd.DataFrame  # columnes: regime, trend, equity_exposure
    panel: dict[str, pd.DataFrame]  # feature → sessions × sectors
    stock_bond_corr: pd.Series
    sectors: list[str]
    notes: list[str] = field(default_factory=list)
    # Correlació accions-bons ràpida (rot_v2): {finestra: sèrie}. Es calcula a `load_inputs`.
    stock_bond_corr_fast: dict[int, pd.Series] = field(default_factory=dict)

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
                               sectors=list(self.sectors), notes=list(self.notes),
                               stock_bond_corr_fast={k: cut(v) for k, v in self.stock_bond_corr_fast.items()})

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
                  cfg: PortfolioConfig, rule: SleeveRule | None = None) -> pd.Series:
    """Puntuació dels sectors elegibles a t (més alta = millor), ordenada.

    Elegible = té momentum ajustat per risc i volatilitat (és a dir, 12 mesos d'història):
    el mateix conjunt per a totes les puntuacions, perquè la data d'inici no en depengui.
    """
    rule = rule or cfg.rule_for(None, "bear" if bear else "bull")
    ram = panel["risk_adj_mom"].loc[t]
    vol = panel["vol_3m"].loc[t]
    ok = ram.notna() & vol.notna() & (vol > 0)
    if not ok.any():
        return pd.Series(dtype=float)
    if rule.score == "trend":
        score = panel["mom_12_1"].loc[t][ok].rank(pct=True).fillna(0.0)
    else:
        score = ram[ok].rank(pct=True)
    rs = panel["rel_strength_6m"].loc[t][ok]
    score = score + cfg.rs_weight * rs.rank(pct=True).fillna(0.5)
    if rule.score == "risk_adj_low_beta":
        beta = panel["beta_1y"].loc[t][ok]
        score = score + cfg.low_beta_weight * (-beta).rank(pct=True).fillna(0.5)
    return score.sort_values(ascending=False, kind="stable")


def absolute_momentum_ok(scores: pd.Series, panel: dict[str, pd.DataFrame], t: pd.Timestamp) -> pd.Series:
    """Filtre de momentum absolut: només els sectors amb retorn 12-1 > 0 (manté l'ordre)."""
    mom = panel["mom_12_1"].loc[t].reindex(scores.index)
    return scores[mom > 0]


def select_sectors(scores: pd.Series, corr: pd.DataFrame | None, cfg: PortfolioConfig,
                   incumbents: list[str] | None = None) -> tuple[list[str], bool]:
    """Selecció voraç per puntuació evitant parells massa correlacionats.

    Amb `rank_buffer` > 0, els sectors de `incumbents` (la selecció anterior) que encara són
    dins del top-(k + buffer) es mantenen; només les places que queden lliures es tornen a
    triar (amb el filtre de correlació respecte de tots els ja triats).

    Retorna (seleccionats, el_filtre_ha_canviat_la_selecció). "El filtre" inclou la banda de
    rang: True si la selecció difereix del top-k ingenu.
    """
    ranked = list(scores.index)
    naive = ranked[: cfg.top_k]
    kept: list[str] = []
    if cfg.rank_buffer > 0 and incumbents:
        zone = set(ranked[: cfg.top_k + cfg.rank_buffer])
        kept = [s for s in ranked if s in zone and s in incumbents][: cfg.top_k]
    elif corr is None or cfg.max_pair_corr >= 1:
        return naive, False
    chosen: list[str] = list(kept)
    if corr is None or cfg.max_pair_corr >= 1:
        for s in ranked:
            if len(chosen) == cfg.top_k:
                break
            if s not in chosen:
                chosen.append(s)
        chosen = [s for s in ranked if s in chosen]
        return chosen, set(chosen) != set(naive)
    for s in ranked:
        if len(chosen) == cfg.top_k:
            break
        if s not in chosen and not _too_correlated(s, chosen, corr, cfg.max_pair_corr):
            chosen.append(s)
    for s in ranked:  # places buides: els millors restants
        if len(chosen) == cfg.top_k:
            break
        if s not in chosen:
            chosen.append(s)
    if kept:  # ordre per puntuació, com sense banda
        chosen = [s for s in ranked if s in chosen]
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


def equity_weights(chosen: list[str], vol: pd.Series, rule: SleeveRule,
                   cfg: PortfolioConfig) -> dict[str, float]:
    """Pesos dins del bloc de renda variable (sumen 1): nucli + satèl·lit.

    Cada sector elegit ocupa 1/top_k del satèl·lit (repartit per volatilitat inversa entre els
    elegits); les places buides (filtre absolut) passen al nucli.
    """
    sat = 1.0 - rule.core_share
    filled = sat * len(chosen) / cfg.top_k if chosen else 0.0
    out = {s: filled * w for s, w in sleeve_weights(chosen, vol, cfg).items()}
    core = 1.0 - filled
    if core > 1e-12:
        out[cfg.market] = out.get(cfg.market, 0.0) + core
    return out


def sleeve_weights(chosen: list[str], vol: pd.Series, cfg: PortfolioConfig) -> dict[str, float]:
    if not chosen:
        return {}
    if cfg.inverse_vol:
        iv = 1.0 / vol[chosen]
        w = iv / iv.sum()
    else:
        w = pd.Series(1.0 / len(chosen), index=chosen)
    return {k: float(v) for k, v in w.items()}


def hedge_weights(sb_corr: float, available: set[str], cfg: PortfolioConfig,
                  sb_corr_fast: float = np.nan, in_bonds: bool = False) -> tuple[dict[str, float], str]:
    """Cobertura per a la part (1 - E). `sb_corr_fast` només compta si `hedge_fast_window`.

    Histèresi: si ja s'és en bons (`in_bonds`), el llindar puja `hedge_hysteresis`; si no, baixa.
    """
    if not cfg.hedge or pd.isna(sb_corr):
        return {CASH: 1.0}, "cash"
    thr = cfg.bond_hedge_max_corr + (cfg.hedge_hysteresis if in_bonds else -cfg.hedge_hysteresis)
    fast_ok = (cfg.hedge_fast_window is None or pd.isna(sb_corr_fast) or sb_corr_fast < thr)
    if sb_corr < thr and fast_ok and cfg.bond_hedge in available:
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
    sb_fast = (inp.stock_bond_corr_fast.get(cfg.hedge_fast_window, pd.Series(np.nan, index=idx)).reindex(idx)
               if cfg.hedge_fast_window else pd.Series(np.nan, index=idx))
    log_ret = np.log(inp.prices[inp.sectors]).diff()
    assets = list(dict.fromkeys([*inp.sectors, cfg.market, cfg.bond_hedge, cfg.gold, CASH]))
    W = np.full((len(idx), len(assets)), np.nan)
    col = {a: i for i, a in enumerate(assets)}
    rows = []
    current: np.ndarray | None = None
    started = False
    incumbents: list[str] = []
    in_bonds = False
    for i, t in enumerate(idx):
        if rev.iloc[i]:
            trend = reg["trend"].iloc[i]
            rule = cfg.rule_for(regime_label.iloc[i], trend)
            scores = sector_scores(inp.panel, t, trend == "bear", cfg, rule)
            if len(scores) >= cfg.top_k:
                candidates = absolute_momentum_ok(scores, inp.panel, t) if rule.abs_mom_filter else scores
                lo = max(0, i - cfg.corr_window + 1)
                window = log_ret.iloc[lo: i + 1]
                corr = window.dropna(axis=1, how="any").corr() if i + 1 >= cfg.corr_window else None
                chosen, bound = select_sectors(candidates, corr, cfg, incumbents)
                incumbents = chosen
                ew = equity_weights(chosen, inp.panel["vol_3m"].loc[t], rule, cfg)
                avail = {a for a in (cfg.bond_hedge, cfg.gold)
                         if a in inp.prices and pd.notna(inp.prices.at[t, a])}
                hw, hname = hedge_weights(sb_corr.iloc[i], avail, cfg, sb_fast.iloc[i], in_bonds)
                in_bonds = hname == "bons"
                e = float(expo.iloc[i])
                vec = np.zeros(len(assets))
                for a, w in ew.items():
                    vec[col[a]] += e * w
                for a, w in hw.items():
                    vec[col[a]] += (1 - e) * w
                current, started = vec, True
                rows.append({"date": t, "regime": regime_label.iloc[i], "equity_exposure": e,
                             "sectors": ",".join(chosen), "core_share": ew.get(cfg.market, 0.0),
                             "score": rule.score, "n_filtered_out": len(scores) - len(candidates),
                             "corr_filter_bound": bound, "max_corr_chosen": _max_pair(corr, chosen),
                             "hedge": hname, "n_candidates": len(scores)})
            elif started:
                rows.append({"date": t, "regime": regime_label.iloc[i], "equity_exposure": float(expo.iloc[i]),
                             "sectors": "", "core_share": np.nan, "score": "", "n_filtered_out": 0,
                             "corr_filter_bound": False, "max_corr_chosen": np.nan,
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
