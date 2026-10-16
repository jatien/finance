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


def simulate(targets: Targets, rets: pd.DataFrame, cost_bps: float = 5.0, band: float = 0.02, *,
             rebalance: str = "max_abs", te_band: float = 0.01, cov: np.ndarray | None = None,
             check_daily: bool = False) -> SimResult:
    if rebalance not in ("max_abs", "tracking_error"):
        raise ValueError(f"rebalance desconegut: {rebalance}")
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

    def out_of_band(tgt: np.ndarray, cur: np.ndarray, i: int) -> bool:
        if rebalance == "tracking_error":
            te = _tracking_error(cur - tgt, cov[i])
            if not np.isnan(te):
                return te > te_band
        return bool(np.max(np.abs(tgt - cur)) > band)

    for i in range(n):
        r_i = np.nan
        if w is not None:
            ri = R[i]
            # Un actiu amb pes > 0 sense retorn aquell dia (forat de dades) es tracta com a 0
            ri = np.where(np.isnan(ri), 0.0, ri)
            gross = float(w @ ri)
            w = w * (1 + ri) / (1 + gross)
            r_i = gross
        tgt = None
        if rev[i] and not np.isnan(W[i]).any():
            tgt = last_tgt = W[i]
        elif check_daily and last_tgt is not None and w is not None:
            tgt = last_tgt
        if tgt is not None and (w is None or out_of_band(tgt, w, i)):
            prev = np.zeros(len(assets)) if w is None else w
            to = float(np.abs(tgt - prev)[non_cash].sum())
            cost = to * cost_bps / 1e4
            r_i = (0.0 if np.isnan(r_i) else r_i) - cost
            out_to[i] = to
            out_tr[i] = True
            w = tgt.copy()
        out_r[i] = r_i
        if w is not None:
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
