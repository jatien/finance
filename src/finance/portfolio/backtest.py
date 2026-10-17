"""Motor de simulació walk-forward: pesos que deriven, banda de no-negociació i costos.

Convenció temporal (la mateixa a tot el sistema):
- `rets` a la fila t és el retorn del dia t-1 → t.
- L'objectiu de la fila t es decideix amb dades fins al tancament de t i s'executa al
  tancament de t (cost carregat a t). El primer retorn que guanya és el de t+1.
- Entre revisions, els pesos deriven amb els preus (no hi ha reequilibri implícit diari).
- En una revisió es negocia només si la cartera s'allunya prou de l'objectiu (disparador):
    max_abs         algun pes s'allunya més de `band`
    tracking_error  sqrt(dᵀ Σ_t d · 252) > te_band, d = cartera - objectiu, Σ_t de les
                    `te_window` sessions fins a t (Palomar, 6.1.5). Sense Σ vàlida → max_abs.
  Si es negocia, es va fins a l'objectiu sencer.
- Amb `check_daily`, el disparador també es comprova les sessions sense revisió, contra
  l'últim objectiu: la deriva dels preus pot provocar un reequilibri a mig mes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance.portfolio.config import CASH
from finance.portfolio.construct import Targets


@dataclass
class SimResult:
    returns: pd.Series  # retorn net diari (NaN abans de començar)
    turnover: pd.Series  # nominal negociat (sense cash) a cada sessió
    weights: pd.DataFrame  # pesos en cartera al final de cada sessió (després de negociar)
    trades: pd.Series  # bool: s'ha negociat al tancament

    @property
    def start(self) -> pd.Timestamp:
        return self.returns.first_valid_index()


def rolling_cov(rets: pd.DataFrame, assets: list[str], window: int) -> np.ndarray:
    """Covariància diària mòbil (sessions × actius × actius), causal: la fila t usa fins a t."""
    R = rets.reindex(columns=assets).to_numpy(dtype=float)
    n, k = R.shape
    out = np.full((n, k, k), np.nan)
    for i in range(window - 1, n):
        X = R[i - window + 1: i + 1]
        ok = ~np.isnan(X).any(axis=0)
        if ok.sum() == 0:
            continue
        Xc = X[:, ok] - X[:, ok].mean(axis=0)
        C = Xc.T @ Xc / (window - 1)
        out[i][np.ix_(ok, ok)] = C
    return out


def _tracking_error(d: np.ndarray, C: np.ndarray) -> float:
    """TE anualitzat de la diferència de pesos `d`; NaN si Σ no cobreix els actius implicats."""
    nz = np.abs(d) > 1e-12
    if not nz.any():
        return 0.0
    sub = C[np.ix_(nz, nz)]
    if np.isnan(sub).any():
        return np.nan
    v = float(d[nz] @ sub @ d[nz]) * 252
    return float(np.sqrt(max(v, 0.0)))


@dataclass
class Trigger:
    """Disparador del reequilibri (vegeu el docstring del mòdul)."""
    rebalance: str = "max_abs"
    band: float = 0.02
    te_band: float = 0.01

    def __post_init__(self) -> None:
        if self.rebalance not in ("max_abs", "tracking_error"):
            raise ValueError(f"rebalance desconegut: {self.rebalance}")

    def distance(self, tgt: np.ndarray, cur: np.ndarray, cov_i: np.ndarray | None) -> tuple[float, str]:
        """(distància, tipus): TE anualitzat si es pot calcular; si no, desviació màxima de pes."""
        if self.rebalance == "tracking_error" and cov_i is not None:
            te = _tracking_error(cur - tgt, cov_i)
            if not np.isnan(te):
                return te, "te"
        return float(np.max(np.abs(tgt - cur))), "max_abs"

    def fires(self, tgt: np.ndarray, cur: np.ndarray, cov_i: np.ndarray | None) -> bool:
        d, kind = self.distance(tgt, cur, cov_i)
        return d > (self.te_band if kind == "te" else self.band)


@dataclass
class StepResult:
    weights: np.ndarray  # pesos al final de la sessió (després de negociar)
    pre_weights: np.ndarray  # pesos després de la deriva, abans de negociar
    gross: float  # retorn brut de la sessió (abans de costos)
    cost: float
    turnover: float
    traded: bool


def step(w: np.ndarray, r: np.ndarray, tgt: np.ndarray | None, *, trigger: Trigger, cost_bps: float,
         non_cash: np.ndarray, cov_i: np.ndarray | None = None, force: bool = False) -> StepResult:
    """Una sessió: deriva dels pesos amb els retorns `r` i, si toca, negociació fins a `tgt`.

    És l'única implementació de la mecànica diària: la fan servir el backtest (`simulate`) i el
    paper trading, de manera que tots dos donen exactament el mateix resultat amb les mateixes
    dades. Un actiu amb pes > 0 sense retorn aquell dia (forat de dades) es tracta com a 0.
    """
    r = np.where(np.isnan(r), 0.0, r)
    gross = float(w @ r)
    pre = w * (1 + r) / (1 + gross)
    if tgt is not None and (force or trigger.fires(tgt, pre, cov_i)):
        to = float(np.abs(tgt - pre)[non_cash].sum())
        return StepResult(tgt.copy(), pre, gross, to * cost_bps / 1e4, to, True)
    return StepResult(pre, pre, gross, 0.0, 0.0, False)


def simulate(targets: Targets, rets: pd.DataFrame, cost_bps: float = 5.0, band: float = 0.02, *,
             rebalance: str = "max_abs", te_band: float = 0.01, cov: np.ndarray | None = None,
             check_daily: bool = False) -> SimResult:
    trigger = Trigger(rebalance, band, te_band)
    if rebalance == "tracking_error" and cov is None:
        raise ValueError("rebalance='tracking_error' necessita `cov` (vegeu rolling_cov)")
    idx = targets.weights.index
    assets = list(targets.weights.columns)
    R = rets.reindex(index=idx, columns=assets).to_numpy(dtype=float)
    W = targets.weights.to_numpy(dtype=float)
    rev = targets.reviews.reindex(idx).fillna(False).to_numpy(dtype=bool)
    non_cash = np.array([a != CASH for a in assets])
    n = len(idx)
    out_r = np.full(n, np.nan)
    out_to = np.zeros(n)
    out_tr = np.zeros(n, dtype=bool)
    held = np.full((n, len(assets)), np.nan)
    w: np.ndarray | None = None
    last_tgt: np.ndarray | None = None
    for i in range(n):
        tgt = None
        if rev[i] and not np.isnan(W[i]).any():
            tgt = last_tgt = W[i]
        elif check_daily and last_tgt is not None and w is not None:
            tgt = last_tgt
        cov_i = cov[i] if cov is not None else None
        if w is None:
            if tgt is None:
                continue
            # Entrada inicial: sense cartera prèvia, es compra l'objectiu sencer
            to = float(np.abs(tgt)[non_cash].sum())
            out_r[i], out_to[i], out_tr[i] = -to * cost_bps / 1e4, to, True
            w = tgt.copy()
        else:
            res = step(w, R[i], tgt, trigger=trigger, cost_bps=cost_bps, non_cash=non_cash, cov_i=cov_i)
            w = res.weights
            out_r[i], out_to[i], out_tr[i] = res.gross - res.cost, res.turnover, res.traded
        held[i] = w
    return SimResult(returns=pd.Series(out_r, index=idx), turnover=pd.Series(out_to, index=idx),
                     weights=pd.DataFrame(held, index=idx, columns=assets),
                     trades=pd.Series(out_tr, index=idx))


def simulate_cfg(targets: Targets, rets: pd.DataFrame, cfg, cov_cache: dict | None = None) -> SimResult:
    """`simulate` amb els paràmetres d'execució d'una `PortfolioConfig`.

    `cov_cache` (opcional) evita recalcular la covariància mòbil entre estratègies que
    comparteixen actius i finestra.
    """
    cov = None
    if cfg.rebalance == "tracking_error":
        key = (tuple(targets.weights.columns), cfg.te_window, len(targets.weights))
        if cov_cache is not None and key in cov_cache:
            cov = cov_cache[key]
        else:
            r = rets.reindex(targets.weights.index)
            cov = rolling_cov(r, list(targets.weights.columns), cfg.te_window)
            if cov_cache is not None:
                cov_cache[key] = cov
    return simulate(targets, rets, cfg.cost_bps, cfg.band, rebalance=cfg.rebalance,
                    te_band=cfg.te_band, cov=cov, check_daily=cfg.check_daily)


def lagged(targets: Targets, sessions: int = 1) -> Targets:
    """Objectius executats `sessions` sessions més tard que la decisió (retard d'implementació).

    El backtest suposa que es negocia al tancament del mateix dia en què es decideix, cosa que
    a la pràctica no es pot fer: l'informe surt després del tancament. Amb `sessions=1`,
    l'objectiu decidit al tancament de t s'executa al tancament de t+1 (com el paper trading).
    """
    if sessions <= 0:
        return targets
    return Targets(weights=targets.weights.shift(sessions),
                   reviews=targets.reviews.shift(sessions, fill_value=False).astype(bool),
                   log=targets.log)
