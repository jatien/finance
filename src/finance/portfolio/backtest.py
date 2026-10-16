"""Motor de simulació walk-forward: pesos que deriven, banda de no-negociació i costos.

Convenció temporal (la mateixa a tot el sistema):
- `rets` a la fila t és el retorn del dia t-1 → t.
- L'objectiu de la fila t es decideix amb dades fins al tancament de t i s'executa al
  tancament de t (cost carregat a t). El primer retorn que guanya és el de t+1.
- Entre revisions, els pesos deriven amb els preus (no hi ha reequilibri implícit diari).
- En una revisió, si cap pes s'allunya de l'objectiu més de `band`, no es negocia.
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


def simulate(targets: Targets, rets: pd.DataFrame, cost_bps: float = 5.0, band: float = 0.02) -> SimResult:
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
    for i in range(n):
        r_i = np.nan
        if w is not None:
            ri = R[i]
            # Un actiu amb pes > 0 sense retorn aquell dia (forat de dades) es tracta com a 0
            ri = np.where(np.isnan(ri), 0.0, ri)
            gross = float(w @ ri)
            w = w * (1 + ri) / (1 + gross)
            r_i = gross
        if rev[i] and not np.isnan(W[i]).any():
            tgt = W[i]
            if w is None or np.max(np.abs(tgt - w)) > band:
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
