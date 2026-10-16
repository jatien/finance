"""Comparació walk-forward del sistema complet amb referències i ablacions.

Totes les estratègies passen pel mateix motor (`backtest.simulate`) amb els mateixos costos
i la mateixa banda, i s'avaluen a la mateixa finestra: des del primer dia en què el sistema
complet té cartera (necessita 12 mesos de momentum sectorial → gener de 2000).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finance.features.config import TRADING_DAYS
from finance.portfolio.backtest import SimResult, simulate
from finance.portfolio.config import CASH, PortfolioConfig
from finance.portfolio.construct import PortfolioInputs, Targets, build_targets, fixed_targets
from finance.regimes.evaluation import episode_table, perf_metrics

MAIN = "Sistema complet"


def regime_on_market(inp: PortfolioInputs, cfg: PortfolioConfig) -> Targets:
    """Superposició de la setmana 4: E al mercat, la resta en cash."""
    e = inp.regime["equity_exposure"].reindex(inp.index)
    W = pd.DataFrame({cfg.market: e, CASH: 1 - e})
    t = fixed_targets(inp.index, W, monthly=False)
    changed = e.ne(e.shift()) & e.notna()
    return Targets(weights=W, reviews=(t.reviews | changed) & W.notna().all(axis=1), log=pd.DataFrame())


def sixty_forty(inp: PortfolioInputs, cfg: PortfolioConfig) -> Targets:
    """60/40 SPY/IEF mensual. Abans que IEF cotitzi (juliol de 2002), la part de bons és cash."""
    has_bond = inp.prices[cfg.bond_hedge].notna() if cfg.bond_hedge in inp.prices else pd.Series(False, index=inp.index)
    W = pd.DataFrame({cfg.market: 0.6, cfg.bond_hedge: 0.4 * has_bond, CASH: 0.4 * ~has_bond}, index=inp.index)
    return fixed_targets(inp.index, W.astype(float))


def equal_weight_sectors(inp: PortfolioInputs) -> Targets:
    listed = inp.prices[inp.sectors].notna() & inp.prices[inp.sectors].shift(1).notna()
    W = listed.div(listed.sum(axis=1), axis=0).where(listed.sum(axis=1) > 0)
    return fixed_targets(inp.index, W.astype(float))


def strategy_targets(inp: PortfolioInputs, cfg: PortfolioConfig | None = None) -> dict[str, Targets]:
    cfg = cfg or PortfolioConfig()
    full = build_targets(inp, cfg)
    start = full.weights.dropna().index.min()
    ones = pd.Series(1.0, index=inp.index)
    out = {
        "SPY buy & hold": fixed_targets(inp.index, {cfg.market: 1.0}, monthly=False, start=start),
        "60/40 SPY/IEF": sixty_forty(inp, cfg),
        "Sectors equiponderats": equal_weight_sectors(inp),
        "Momentum sectorial (sense règim)": build_targets(
            inp, cfg.variant(defensive_tilt=False, max_pair_corr=1.01, hedge=False), exposure=ones),
        "Règims sobre SPY": regime_on_market(inp, cfg),
        MAIN: full,
        "Sense filtre de correlació": build_targets(inp, cfg.variant(max_pair_corr=1.01)),
        "Sense cobertura (cash)": build_targets(inp, cfg.variant(hedge=False)),
        "Sense inclinació defensiva": build_targets(inp, cfg.variant(defensive_tilt=False)),
    }
    return out


def run_all(inp: PortfolioInputs, cfg: PortfolioConfig | None = None,
            targets: dict[str, Targets] | None = None) -> tuple[dict[str, SimResult], pd.Timestamp]:
    cfg = cfg or PortfolioConfig()
    targets = targets or strategy_targets(inp, cfg)
    rets = inp.asset_returns()
    sims = {k: simulate(t, rets, cfg.cost_bps, cfg.band) for k, t in targets.items()}
    start = sims[MAIN].start
    return sims, start


def equity_share(sim: SimResult, cfg: PortfolioConfig, sectors: list[str]) -> pd.Series:
    cols = [c for c in sim.weights if c in (*sectors, cfg.market)]
    return sim.weights[cols].sum(axis=1)


def metrics_table(sims: dict[str, SimResult], inp: PortfolioInputs, cfg: PortfolioConfig,
                  start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = []
    for name, s in sims.items():
        r = s.returns.loc[start:end]
        m = perf_metrics(r, inp.cash_ret, equity_share(s, cfg, inp.sectors).shift(1), s.turnover)
        rows.append({"estrategia": name, **m})
    return pd.DataFrame(rows)


def returns_frame(sims: dict[str, SimResult], start: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({k: s.returns for k, s in sims.items()}).loc[start:]


def episodes(sims: dict[str, SimResult], start: pd.Timestamp) -> pd.DataFrame:
    return episode_table(returns_frame(sims, start))


def holdings_summary(sim: SimResult, inp: PortfolioInputs, start: pd.Timestamp) -> pd.DataFrame:
    """% de sessions que cada actiu és a la cartera i pes mitjà quan hi és."""
    w = sim.weights.loc[start:]
    rows = []
    for c in w:
        held = w[c] > 1e-6
        if not held.any():
            continue
        rows.append({"actiu": c, "sessions_%": 100 * held.mean(), "pes_mitja_quan_hi_es_%": 100 * w.loc[held, c].mean()})
    return pd.DataFrame(rows).sort_values("sessions_%", ascending=False)


def review_summary(log: pd.DataFrame, start: pd.Timestamp) -> dict[str, float]:
    lg = log.loc[start:]
    lg = lg[lg["hedge"] != "sense canvis"]
    years = max((lg.index.max() - lg.index.min()).days / 365.25, 1e-9) if len(lg) else np.nan
    return {
        "revisions": len(lg),
        "revisions_per_any": len(lg) / years if len(lg) else np.nan,
        "filtre_correlacio_actiu_%": 100 * lg["corr_filter_bound"].mean() if len(lg) else np.nan,
        "corr_max_mitjana_triats": float(lg["max_corr_chosen"].mean()) if len(lg) else np.nan,
    }


def hedge_time_share(log: pd.DataFrame, index: pd.DatetimeIndex, start: pd.Timestamp) -> pd.Series:
    h = log.loc[log["hedge"] != "sense canvis", "hedge"].reindex(index).ffill().loc[start:]
    return (100 * h.value_counts(normalize=True)).round(1)


def annual_returns(sims: dict[str, SimResult], names: list[str], start: pd.Timestamp) -> pd.DataFrame:
    df = returns_frame(sims, start)[names].fillna(0)
    yr = (1 + df).groupby(df.index.year).prod() - 1
    return 100 * yr


def years_of(r: pd.Series) -> float:
    return len(r.dropna()) / TRADING_DAYS


def sensitivity(inp: PortfolioInputs, cfg: PortfolioConfig, grid: dict[str, list] | None = None,
                bands: tuple[float, ...] = (0.0, 0.02, 0.05)) -> pd.DataFrame:
    """Sistema complet en una graella de paràmetres (robustesa, no selecció del millor punt)."""
    import itertools
    grid = grid or {"top_k": [2, 3, 4, 5], "max_pair_corr": [0.7, 0.8, 1.01]}
    rets = inp.asset_returns()
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        changes = dict(zip(keys, combo, strict=True))
        t = build_targets(inp, cfg.variant(**changes))
        for b in bands:
            s = simulate(t, rets, cfg.cost_bps, b)
            r = s.returns.dropna()
            m = perf_metrics(r, inp.cash_ret, None, s.turnover)
            rows.append({**changes, "band": b,
                         **{k: m.get(k) for k in ("CAGR_%", "sharpe", "max_dd_%", "calmar", "rotacio_anual")}})
    return pd.DataFrame(rows)
