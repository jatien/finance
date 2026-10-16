"""Transformacions pures i causals sobre sèries temporals.

Regla d'or: el valor a la data t només pot dependre de les files <= t. Res de `bfill`,
finestres centrades, ni normalitzacions amb estadístics de tota la mostra. Els tests de
truncament (`tests/test_features.py`) ho verifiquen per a totes les features.

Regla de preus: `adj_close` es reajusta cap enrere amb cada dividend (es multiplica tota la
història anterior per un factor). Per això tota feature de preus ha de ser invariant a
l'escala (ràtios, retorns, log-diferències), mai un nivell absolut de preu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finance.features.config import TRADING_DAYS


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices).diff()


def sma(x: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    return x.rolling(n, min_periods=n).mean()


def dist_to_sma(prices: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    """Distància relativa del preu a la seva mitjana mòbil: P / SMA(n) - 1."""
    return prices / sma(prices, n) - 1


def momentum(prices: pd.DataFrame | pd.Series, lookback: int, skip: int = 0) -> pd.DataFrame | pd.Series:
    """Retorn simple de t-lookback a t-skip (p. ex. 12-1: lookback=252, skip=21)."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def realized_vol(prices: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    """Volatilitat realitzada anualitzada (desviació dels log-retorns diaris)."""
    return log_returns(prices).rolling(n, min_periods=n).std() * np.sqrt(TRADING_DAYS)


def rolling_drawdown(prices: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    """Caiguda des del màxim de les últimes n sessions (<= 0)."""
    return prices / prices.rolling(n, min_periods=n).max() - 1


def log_ratio_change(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Canvi en n sessions del log(a/b): força relativa de a respecte de b."""
    r = np.log(a / b)
    return r - r.shift(n)


def rolling_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    ra, rb = log_returns(a), log_returns(b)
    return ra.rolling(n, min_periods=n).corr(rb)


def rolling_beta(asset: pd.DataFrame, market: pd.Series, n: int) -> pd.DataFrame:
    """Beta de cada columna respecte del mercat: cov(r_i, r_m) / var(r_m)."""
    ra = log_returns(asset)
    rm = log_returns(market)
    var_m = rm.rolling(n, min_periods=n).var()
    cov = ra.rolling(n, min_periods=n).cov(rm)
    return cov.div(var_m, axis=0)


def fraction_above_sma(prices: pd.DataFrame, n: int) -> pd.Series:
    """Fracció de columnes amb preu per sobre de la seva SMA(n), entre les que en tenen."""
    s = sma(prices, n)
    valid = s.notna() & prices.notna()
    above = (prices > s) & valid
    count = valid.sum(axis=1)
    return (above.sum(axis=1) / count).where(count > 0)


def correlation_structure(prices: pd.DataFrame, n: int, k_frac: float = 0.2,
                          min_assets: int = 3) -> pd.DataFrame:
    """Correlació mitjana entre parells i absorption ratio en finestra mòbil.

    - `avg_corr`: mitjana de les correlacions fora de la diagonal. Quan tots els sectors es
      mouen junts, el mercat el governa un sol factor (por/risc): senyal d'estrès.
    - `absorption`: fracció de la variància (de la matriu de correlació) explicada pels
      k = ceil(k_frac · N) primers vectors propis (Kritzman et al., 2011). És la versió
      espectral de la mateixa idea; connecta amb l'anàlisi de Marchenko-Pastur.

    A cada data només entren els actius amb la finestra completa: un sector que cotitza
    des de fa menys de n sessions (XLRE 2015, XLC 2018) no hi entra fins que en té prou.
    """
    rets = log_returns(prices).to_numpy()
    idx = prices.index
    out = np.full((len(idx), 2), np.nan)
    for t in range(n, len(idx)):
        w = rets[t - n + 1: t + 1]
        cols = ~np.isnan(w).any(axis=0)
        if cols.sum() < min_assets:
            continue
        w = w[:, cols]
        sd = w.std(axis=0, ddof=1)
        keep = sd > 0
        if keep.sum() < min_assets:
            continue
        c = np.corrcoef(w[:, keep], rowvar=False)
        m = c.shape[0]
        out[t, 0] = (c.sum() - m) / (m * (m - 1))
        eig = np.linalg.eigvalsh(c)[::-1]
        k = max(1, int(np.ceil(k_frac * m)))
        out[t, 1] = eig[:k].sum() / eig.sum()
    return pd.DataFrame(out, index=idx, columns=["avg_corr", "absorption"])


def expanding_pct_rank(x: pd.Series | pd.DataFrame, min_periods: int) -> pd.Series | pd.DataFrame:
    """Percentil (0-1] del valor actual dins de tota la seva història fins a t.

    És la normalització point-in-time: només compara amb el passat. Fa les features
    comparables entre èpoques (un VIX de 20 no significava el mateix el 2005 que el 2021).
    """
    return x.expanding(min_periods=min_periods).rank(pct=True)


def cross_sectional_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Percentil transversal per data (entre les columnes amb dades aquell dia)."""
    return x.rank(axis=1, pct=True)
