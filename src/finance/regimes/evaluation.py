"""Validació dels règims: tenen significat econòmic i serveixen per gestionar el risc?

Aquí sí que es fan servir rendiments futurs (és avaluació, no senyal). Regla d'alineació:
el règim i l'exposició decidits al tancament de t s'apliquen al retorn de t → t+1.

Preguntes que respon:
1. Els règims separen la distribució de rendiments i volatilitat? (`regime_stats`)
2. Són estables o canvien massa sovint? (episodis, durada, canvis/any, matriu de transició)
3. La diversificació s'evapora en estrès? (`conditional_sector_corr`)
4. Una cartera SPY + cash governada pel règim redueix les caigudes, i a quin cost?
   (`overlay_returns`, `compare_strategies`, `episode_table`)
5. El resultat depèn d'un punt màgic de paràmetres? (`sensitivity`)
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from finance.features.config import TRADING_DAYS
from finance.regimes.config import REGIMES, RegimeConfig
from finance.regimes.rules import classify, trend_state

# Episodis de referència (dates de pic i vall del S&P 500) i les recuperacions posteriors
EPISODES: dict[str, tuple[str, str]] = {
    "Punt com (2000-02)": ("2000-03-24", "2002-10-09"),
    "Crisi financera (2007-09)": ("2007-10-09", "2009-03-09"),
    "Recuperació 2009": ("2009-03-09", "2010-03-09"),
    "Deute europeu (2011)": ("2011-04-29", "2011-10-03"),
    "Xina / petroli (2015-16)": ("2015-05-21", "2016-02-11"),
    "Q4 2018": ("2018-09-20", "2018-12-24"),
    "Covid (2020)": ("2020-02-19", "2020-03-23"),
    "Recuperació 2020": ("2020-03-23", "2020-08-31"),
    "Inflació (2022)": ("2022-01-03", "2022-10-12"),
    "Aranzels (2025)": ("2025-02-19", "2025-04-08"),
}


# ---------------------------------------------------------------------------------------
# Estadístiques per règim
# ---------------------------------------------------------------------------------------

def next_returns(prices: pd.Series) -> pd.Series:
    """Retorn simple de t a t+1, indexat a t (el que s'obté amb la decisió presa a t)."""
    return prices.shift(-1) / prices - 1


def forward_return(prices: pd.Series, h: int) -> pd.Series:
    return prices.shift(-h) / prices - 1


def episodes(regime: pd.Series) -> pd.DataFrame:
    r = regime.dropna()
    if r.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "sessions"])
    grp = r.ne(r.shift()).cumsum()
    ep = r.groupby(grp).agg(["first", "size"])
    starts = r.index.to_series().groupby(grp).first()
    ends = r.index.to_series().groupby(grp).last()
    return pd.DataFrame({"regime": ep["first"].to_numpy(), "start": starts.to_numpy(),
                         "end": ends.to_numpy(), "sessions": ep["size"].to_numpy()})


def regime_stats(regime: pd.Series, prices: pd.Series, horizons: tuple[int, ...] = (21, 63)) -> pd.DataFrame:
    """Rendiment, volatilitat i rendiments futurs del mercat condicionats al règim d'avui."""
    r1 = next_returns(prices)
    fwd = {h: forward_return(prices, h) for h in horizons}
    ep = episodes(regime)
    rows = []
    labels = [g for g in REGIMES if (regime == g).any()] + ["tots"]
    defined = regime.notna()
    for g in labels:
        m = defined if g == "tots" else (regime == g)
        x = r1[m].dropna()
        e = ep if g == "tots" else ep[ep["regime"] == g]
        row = {
            "regime": g,
            "temps_%": 100 * m.sum() / defined.sum(),
            "episodis": len(e),
            "durada_mitjana": e["sessions"].mean() if len(e) else np.nan,
            "ret_anual_%": 100 * x.mean() * TRADING_DAYS,
            "vol_anual_%": 100 * x.std() * np.sqrt(TRADING_DAYS),
            "sharpe": x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else np.nan,
            "pitjor_dia_%": 100 * x.min() if len(x) else np.nan,
        }
        for h, f in fwd.items():
            y = f[m].dropna()
            row[f"fwd{h}_mitjana_%"] = 100 * y.mean()
            row[f"fwd{h}_positiu_%"] = 100 * (y > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def transition_matrix(regime: pd.Series) -> pd.DataFrame:
    """Nombre de canvis de règim (files: des de, columnes: cap a)."""
    r = regime.dropna()
    prev, cur = r.shift(), r
    m = (prev != cur) & prev.notna()
    tm = pd.crosstab(prev[m], cur[m])
    present = [g for g in REGIMES if g in set(r)]
    return tm.reindex(index=present, columns=present, fill_value=0)


def switches_per_year(regime: pd.Series) -> float:
    r = regime.dropna()
    if len(r) < 2:
        return np.nan
    n = int((r != r.shift()).iloc[1:].sum())
    return n / (len(r) / TRADING_DAYS)


def conditional_sector_corr(regime: pd.Series, sector_prices: pd.DataFrame,
                            groups: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Correlació de rendiments diaris entre sectors, agrupant les sessions de cada règim.

    Retorna (resum: correlació mitjana fora de la diagonal per règim, {règim: matriu}).
    `groups` permet comparar blocs (p. ex. tots els règims "bull" contra els "bear").
    """
    rets = np.log(sector_prices).diff()
    groups = groups or {g: [g] for g in REGIMES}
    mats, rows = {}, []
    for name, members in groups.items():
        m = regime.isin(members)
        x = rets[m]
        if m.sum() < 60:
            continue
        c = x.corr(min_periods=60)
        mats[name] = c
        vals = c.to_numpy()[~np.eye(len(c), dtype=bool)]
        rows.append({"regime": name, "sessions": int(m.sum()),
                     "corr_mitjana": float(np.nanmean(vals))})
    return pd.DataFrame(rows), mats


# ---------------------------------------------------------------------------------------
# Backtest de la superposició (SPY + cash)
# ---------------------------------------------------------------------------------------

def overlay_returns(exposure: pd.Series, asset_ret: pd.Series, cash_ret: pd.Series,
                    cost_bps: float = 5.0) -> pd.DataFrame:
    """Retorns diaris d'una cartera amb pes `exposure` a l'actiu i la resta en cash.

    `exposure` es decideix al tancament de t i s'aplica al retorn del dia t+1: el pes
    efectiu del dia t és exposure.shift(1). Cost = |canvi de pes| · cost_bps.
    `asset_ret` i `cash_ret` són retorns del dia (de t-1 a t), indexats a t.
    """
    w = exposure.shift(1)
    turnover = w.diff().abs().fillna(0.0)
    cost = turnover * cost_bps / 1e4
    r = w * asset_ret + (1 - w) * cash_ret.fillna(0.0) - cost
    return pd.DataFrame({"ret": r, "weight": w, "turnover": turnover})


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def perf_metrics(r: pd.Series, cash_ret: pd.Series, weight: pd.Series | None = None,
                 turnover: pd.Series | None = None) -> dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {}
    years = len(r) / TRADING_DAYS
    total = float((1 + r).prod())
    cagr = total ** (1 / years) - 1
    ex = r - cash_ret.reindex(r.index).fillna(0.0)
    vol = r.std() * np.sqrt(TRADING_DAYS)
    dd = max_drawdown(r)
    down = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    out = {
        "CAGR_%": 100 * cagr,
        "vol_%": 100 * vol,
        "sharpe": ex.mean() / ex.std() * np.sqrt(TRADING_DAYS) if ex.std() > 0 else np.nan,
        "sortino": ex.mean() * TRADING_DAYS / down if down > 0 else np.nan,
        "max_dd_%": 100 * dd,
        "calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "pitjor_any_%": 100 * float(((1 + r).rolling(TRADING_DAYS).apply(np.prod, raw=True) - 1).min()),
    }
    if weight is not None:
        out["exposicio_mitjana_%"] = 100 * float(weight.reindex(r.index).mean())
    if turnover is not None:
        out["rotacio_anual"] = float(turnover.reindex(r.index).sum() / years)
    return out


def strategy_exposures(values: pd.DataFrame, pct: pd.DataFrame, cfg: RegimeConfig) -> dict[str, pd.Series]:
    """Estratègies a comparar (totes amb el mateix actiu i cash).

    - SPY buy & hold: referència.
    - Filtre SMA200: 100 % si bull, 0 % si bear (mateixa histèresi). Línia de base clàssica.
    - Règims (complet): l'exposició del mapa de règims.
    - Ablacions: sense eix de volatilitat i sense crisi, per saber què aporta cada peça.
    """
    full = classify(values, pct, cfg)
    defined = full["equity_exposure"].notna()
    tr = trend_state(values, cfg)
    out = {
        "SPY buy & hold": pd.Series(1.0, index=values.index).where(defined),
        "Filtre SMA200": (1.0 - tr).where(defined),
        "Règims (complet)": full["equity_exposure"],
        "Règims sense vol": classify(values, pct, cfg.variant(use_vol_axis=False))["equity_exposure"].where(defined),
        "Règims sense crisi": classify(values, pct, cfg.variant(use_crisis=False))["equity_exposure"].where(defined),
    }
    return out


def compare_strategies(exposures: dict[str, pd.Series], asset_ret: pd.Series, cash_ret: pd.Series,
                       cost_bps: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(taula de mètriques, retorns diaris per estratègia)."""
    rets, rows = {}, []
    for name, e in exposures.items():
        o = overlay_returns(e, asset_ret, cash_ret, cost_bps)
        o = o[o["weight"].notna()]
        rets[name] = o["ret"]
        rows.append({"estrategia": name, **perf_metrics(o["ret"], cash_ret, o["weight"], o["turnover"])})
    return pd.DataFrame(rows), pd.DataFrame(rets)


def episode_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Retorn total de cada estratègia en cada episodi de referència (%)."""
    rows = []
    for name, (a, b) in EPISODES.items():
        w = returns.loc[(returns.index > pd.Timestamp(a)) & (returns.index <= pd.Timestamp(b))]
        if len(w) < 5 or w.isna().all().all():
            continue
        row = {"episodi": name, "inici": a, "fi": b}
        for c in returns:
            row[c] = 100 * float((1 + w[c].fillna(0)).prod() - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity(values: pd.DataFrame, pct: pd.DataFrame, asset_ret: pd.Series, cash_ret: pd.Series,
                base: RegimeConfig, cost_bps: float = 5.0,
                grid: dict[str, list] | None = None) -> pd.DataFrame:
    """Mètriques del classificador complet en una graella de paràmetres.

    No és per triar el millor punt: és per comprovar que el resultat no depèn d'un valor
    concret. Si només un punt de la graella funciona, la regla no és robusta.
    """
    grid = grid or {
        "trend_band": [0.0, 0.01, 0.02, 0.03],
        "confirm_days": [1, 3, 5],
        "vol_enter_pct": [0.65, 0.75, 0.85],
    }
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        changes = dict(zip(keys, combo, strict=True))
        if "vol_enter_pct" in changes:
            changes["vol_exit_pct"] = round(changes["vol_enter_pct"] - 0.15, 4)
        cfg = base.variant(**changes)
        reg = classify(values, pct, cfg)
        o = overlay_returns(reg["equity_exposure"], asset_ret, cash_ret, cost_bps)
        o = o[o["weight"].notna()]
        m = perf_metrics(o["ret"], cash_ret, o["weight"], o["turnover"])
        rows.append({**changes, "canvis_any": switches_per_year(reg["regime"]),
                     **{k: m.get(k) for k in ("CAGR_%", "sharpe", "max_dd_%", "calmar", "rotacio_anual")}})
    return pd.DataFrame(rows)
