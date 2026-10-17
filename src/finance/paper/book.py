"""Motor del paper trading: avança una cartera sessió a sessió i en deixa constància.

Convencions (les mateixes que el backtest amb un retard d'una sessió, `backtest.lagged`):
- Al tancament de t es decideix l'objectiu amb les dades fins a t (`build_targets`).
- L'informe surt després del tancament; l'objectiu s'executa al tancament de t+1, si el
  disparador de l'estratègia ho demana (revisió nova, o deriva amb `check_daily`).
- La mecànica diària és `backtest.step`: la mateixa funció que fa servir el backtest.

L'estat (`BookState`) és el que cal per continuar l'endemà: pesos en cartera, objectiu vigent
i si és nou. Es guarda al lake (`paper_book` + `paper_nav`) i es reconstrueix d'allà; mai es
recalcula el passat amb dades noves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finance.portfolio.backtest import Trigger, rolling_cov, step
from finance.portfolio.config import CASH, PortfolioConfig
from finance.portfolio.construct import PortfolioInputs, Targets, sector_scores

EPS = 1e-9


@dataclass
class BookState:
    strategy: str
    inception: pd.Timestamp
    date: pd.Timestamp  # última sessió registrada
    weights: pd.Series  # pesos al tancament de `date` (després de negociar)
    pending: pd.Series  # objectiu vigent per a la sessió següent
    pending_new: bool  # l'objectiu s'ha decidit al tancament de `date`
    pending_from: pd.Timestamp  # data de decisió de l'objectiu vigent
    pending_reason: str
    nav: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def force(self) -> bool:
        """El dia d'inici l'objectiu s'executa sencer, sense passar pel disparador."""
        return self.date == self.inception


@dataclass
class BookUpdate:
    state: BookState
    nav: pd.DataFrame  # files de paper_nav
    book: pd.DataFrame  # files de paper_book


def trigger_of(cfg: PortfolioConfig) -> Trigger:
    return Trigger(cfg.rebalance, cfg.band, cfg.te_band)


def sleeve_of(symbol: str, cfg: PortfolioConfig) -> str:
    if symbol == CASH:
        return "cash"
    if symbol in (cfg.bond_hedge, cfg.gold):
        return "cobertura"
    if symbol == cfg.market:
        return "nucli"
    return "renda_variable"


# ---------------------------------------------------------------------------------------
# Context de cada decisió
# ---------------------------------------------------------------------------------------

def _hedge_at(log: pd.DataFrame, t: pd.Timestamp) -> str | None:
    if log.empty:
        return None
    h = log.loc[:t, "hedge"]
    h = h[h != "sense canvis"]
    return h.iloc[-1] if len(h) else None


def decision_reason(log: pd.DataFrame, t: pd.Timestamp) -> str:
    """Per què hi ha un objectiu nou a t: canvi de règim, revisió mensual, canvi de cobertura."""
    if log.empty or t not in log.index:
        return "revisió"
    row = log.loc[t]
    prev = log.loc[log.index < t]
    prev = prev[prev["hedge"] != "sense canvis"]
    parts = []
    if len(prev) and prev["regime"].iloc[-1] != row["regime"]:
        parts.append(f"canvi de règim {prev['regime'].iloc[-1]} → {row['regime']}")
    elif t.to_period("M") != (prev.index[-1].to_period("M") if len(prev) else None):
        parts.append("revisió mensual")
    else:
        parts.append("revisió")
    if len(prev) and row["hedge"] not in ("sense canvis", prev["hedge"].iloc[-1]) and row["equity_exposure"] < 1:
        parts.append(f"cobertura {prev['hedge'].iloc[-1]} → {row['hedge']}")
    return "; ".join(parts)


def sector_ranks(inp: PortfolioInputs, t: pd.Timestamp, cfg: PortfolioConfig) -> dict[str, int]:
    """Rang (1 = millor) de cada sector a t amb la puntuació de la regla vigent."""
    reg = inp.regime.reindex(inp.index)
    trend = reg["trend"].get(t)
    rule = cfg.rule_for(reg["regime"].get(t), trend)
    scores = sector_scores(inp.panel, t, trend == "bear", cfg, rule)
    return {s: i + 1 for i, s in enumerate(scores.index)}


# ---------------------------------------------------------------------------------------
# Inici i avanç
# ---------------------------------------------------------------------------------------

def start(inp: PortfolioInputs, targets: Targets, cfg: PortfolioConfig,
          date: pd.Timestamp | None = None) -> BookUpdate:
    """Obre el llibre a `date` (per defecte, l'última sessió): 100 % cash i, com a objectiu per a
    la sessió següent, la cartera objectiu vigent (encara que avui no sigui dia de revisió)."""
    W = targets.weights.dropna()
    if W.empty:
        raise RuntimeError(f"{cfg.name}: sense cartera objectiu (cal un any de momentum i règims)")
    d0 = pd.Timestamp(date) if date is not None else W.index[-1]
    if d0 not in W.index:
        raise RuntimeError(f"{cfg.name}: sense objectiu a {d0:%Y-%m-%d}")
    assets = list(targets.weights.columns)
    w = pd.Series(0.0, index=assets)
    w[CASH] = 1.0
    state = BookState(strategy=cfg.name, inception=d0, date=d0, weights=w,
                      pending=W.loc[d0].astype(float), pending_new=True, pending_from=d0,
                      pending_reason="inici del paper trading: compra de la cartera objectiu",
                      nav=1.0, ranks=sector_ranks(inp, d0, cfg))
    nav = _nav_row(state, inp, targets, cfg, gross=0.0, cost=0.0, turnover=0.0, traded=False,
                   reason=None, distance=np.nan, kind=None, bench=np.nan, late=False)
    book = _book_rows(state, inp, cfg, pre=w, trade=w * 0.0, traded=False)
    return BookUpdate(state, pd.DataFrame([nav]), book)


def advance(state: BookState, inp: PortfolioInputs, targets: Targets, cfg: PortfolioConfig,
            end: pd.Timestamp | None = None) -> BookUpdate:
    """Avança el llibre per totes les sessions posteriors a `state.date` fins a `end`.

    Les sessions que no són l'última es marquen `recorded_late` (recuperació: s'han escrit
    amb dades posteriors a la sessió següent).
    """
    idx = targets.weights.index
    end = idx[-1] if end is None else pd.Timestamp(end)
    days = idx[(idx > state.date) & (idx <= end)]
    if len(days) == 0:
        return BookUpdate(state, pd.DataFrame(), pd.DataFrame())
    assets = list(targets.weights.columns)
    rets = inp.asset_returns().reindex(index=idx, columns=assets)
    trig = trigger_of(cfg)
    non_cash = np.array([a != CASH for a in assets])
    navs, books = [], []
    for d in days:
        i = idx.get_loc(d)
        w = state.weights.reindex(assets).fillna(0.0).to_numpy()
        pending = state.pending.reindex(assets).fillna(0.0).to_numpy()
        tgt = pending if (state.pending_new or cfg.check_daily) else None
        cov_i = _cov_at(rets, assets, i, cfg) if (tgt is not None and cfg.rebalance == "tracking_error") else None
        dist, kind = trig.distance(pending, _drift(w, rets.iloc[i].to_numpy()), cov_i) if tgt is not None else (np.nan, None)
        res = step(w, rets.iloc[i].to_numpy(dtype=float), tgt, trigger=trig, cost_bps=cfg.cost_bps,
                   non_cash=non_cash, cov_i=cov_i, force=state.force)
        reason = None
        if res.traded:
            if state.force:
                reason = state.pending_reason
            elif state.pending_new:
                reason = f"objectiu del {state.pending_from:%Y-%m-%d} ({state.pending_reason})"
            else:
                thr = cfg.te_band if kind == "te" else cfg.band
                reason = f"deriva: {'TE' if kind == 'te' else 'desviació'} {dist:.2%} > {thr:.0%}"
        pre = pd.Series(res.pre_weights, index=assets)
        new_w = pd.Series(res.weights, index=assets)
        nav_val = state.nav * (1 + res.gross - res.cost)
        # Decisió al tancament de d (per a la sessió següent)
        new_obj = bool(targets.reviews.get(d, False)) and targets.weights.loc[d].notna().all()
        state = BookState(
            strategy=state.strategy, inception=state.inception, date=d, weights=new_w,
            pending=targets.weights.loc[d].astype(float) if new_obj else state.pending,
            pending_new=new_obj, pending_from=d if new_obj else state.pending_from,
            pending_reason=decision_reason(targets.log, d) if new_obj else state.pending_reason,
            nav=nav_val, ranks=sector_ranks(inp, d, cfg) if new_obj else state.ranks)
        bench = rets[cfg.market].iloc[i] if cfg.market in rets else np.nan
        navs.append(_nav_row(state, inp, targets, cfg, gross=res.gross, cost=res.cost, turnover=res.turnover,
                             traded=res.traded, reason=reason, distance=dist, kind=kind,
                             bench=bench, late=d != days[-1]))
        books.append(_book_rows(state, inp, cfg, pre=pre, trade=new_w - pre, traded=res.traded))
    return BookUpdate(state, pd.DataFrame(navs), pd.concat(books, ignore_index=True))


def replay(inp: PortfolioInputs, targets: Targets, cfg: PortfolioConfig,
           inception: pd.Timestamp, end: pd.Timestamp | None = None) -> BookUpdate:
    """Recalcula el llibre des de `inception` amb les dades actuals (per comparar amb el registre)."""
    s = start(inp, targets, cfg, inception)
    a = advance(s.state, inp, targets, cfg, end)
    return BookUpdate(a.state, pd.concat([s.nav, a.nav], ignore_index=True),
                      pd.concat([s.book, a.book], ignore_index=True))


# ---------------------------------------------------------------------------------------
# Auxiliars
# ---------------------------------------------------------------------------------------

def _drift(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    r = np.where(np.isnan(r), 0.0, r)
    g = float(w @ r)
    return w * (1 + r) / (1 + g)


def _cov_at(rets: pd.DataFrame, assets: list[str], i: int, cfg: PortfolioConfig) -> np.ndarray | None:
    if i + 1 < cfg.te_window:
        return None
    window = rets.iloc[i - cfg.te_window + 1: i + 1]
    return rolling_cov(window, assets, cfg.te_window)[-1]


def _nav_row(state: BookState, inp: PortfolioInputs, targets: Targets, cfg: PortfolioConfig, *,
             gross: float, cost: float, turnover: float, traded: bool, reason: str | None,
             distance: float, kind: str | None, bench: float, late: bool) -> dict:
    d = state.date
    reg = inp.regime.reindex(inp.index)
    return {
        "strategy_set": state.strategy, "date": d, "inception": state.inception, "nav": state.nav,
        "ret_gross": gross, "cost": cost, "ret_net": gross - cost, "turnover": turnover,
        "traded": traded, "reason": reason, "distance": distance, "distance_kind": kind,
        "pending_new": state.pending_new, "pending_from": state.pending_from,
        "pending_reason": state.pending_reason, "regime": reg["regime"].get(d),
        "equity_exposure": float(reg["equity_exposure"].get(d, np.nan)),
        "hedge": _hedge_at(targets.log, d), "benchmark_ret": bench, "recorded_late": late,
    }


def _book_rows(state: BookState, inp: PortfolioInputs, cfg: PortfolioConfig, *,
               pre: pd.Series, trade: pd.Series, traded: bool) -> pd.DataFrame:
    w = state.weights
    tgt = state.pending.reindex(w.index).fillna(0.0)
    pre = pre.reindex(w.index).fillna(0.0)
    trade = trade.reindex(w.index).fillna(0.0)
    keep = (w > EPS) | (pre > EPS) | (tgt > EPS)
    rows = []
    for s in w.index[keep]:
        if not traded or abs(trade[s]) < EPS:
            action = "hold"
        elif pre[s] <= EPS:
            action = "enter"
        elif w[s] <= EPS:
            action = "exit"
        else:
            action = "add" if trade[s] > 0 else "reduce"
        px = 1.0 if s == CASH else inp.prices[s].get(state.date) if s in inp.prices else np.nan
        rows.append({"strategy_set": state.strategy, "date": state.date, "symbol": s,
                     "sleeve": sleeve_of(s, cfg), "weight": float(w[s]), "weight_pre": float(pre[s]),
                     "trade": float(trade[s]), "action": action, "target_weight": float(tgt[s]),
                     "price": float(px) if px is not None and pd.notna(px) else np.nan,
                     "score_rank": state.ranks.get(s)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["score_rank"] = df["score_rank"].astype("Int32")
    return df


def state_from_rows(nav: pd.DataFrame, book: pd.DataFrame) -> BookState | None:
    """Reconstrueix l'estat a partir de l'última sessió registrada (files del lake)."""
    if nav.empty:
        return None
    last = nav.sort_values("date").iloc[-1]
    d = pd.Timestamp(last["date"])
    b = book[pd.to_datetime(book["date"]) == d]
    ranks = {r.symbol: int(r.score_rank) for r in b.itertuples() if pd.notna(r.score_rank)}
    return BookState(
        strategy=last["strategy_set"], inception=pd.Timestamp(last["inception"]), date=d,
        weights=b.set_index("symbol")["weight"].astype(float),
        pending=b.set_index("symbol")["target_weight"].astype(float),
        pending_new=bool(last["pending_new"]), pending_from=pd.Timestamp(last["pending_from"]),
        pending_reason=last["pending_reason"] or "", nav=float(last["nav"]), ranks=ranks)
