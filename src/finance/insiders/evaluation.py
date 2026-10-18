"""Validació de les senyals d'insiders abans d'usar-les per decidir.

1. **Estudi d'esdeveniments**: rendiment en excés (contra SPY i contra l'ETF del sector)
   des del tancament de la sessió de senyal fins a h sessions. Error estàndard agrupat
   per mes (els esdeveniments d'un mateix mes no són independents).
2. **Cartera en temps de calendari**: compra a parts iguals les accions amb esdeveniment
   recent i les manté `hold_sessions`. És el que es podria executar de debò: alfa i beta
   contra SPY, ràtio d'informació i exposició.
3. **Ablació dels filtres**: compres sense filtrar (plans 10b5-1, rutinaris) contra
   compres filtrades. Si els filtres no milloren res, sobren.
4. **Amplitud agregada**: el ràtio compres/vendes del mercat prediu el SPY a 63 sessions?

Finestres: disseny (≤ `design_end`) i comprovació (posterior). Criteris pre-registrats a
docs/insiders.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from finance.features.config import TRADING_DAYS
from finance.insiders.config import InsiderConfig
from finance.insiders.signals import build_events

MARKET = "SPY"


@dataclass
class EvalData:
    events: pd.DataFrame
    px: pd.DataFrame  # adj_close: sessions × símbols (accions, SPY, ETFs sectorials)
    sector_etf: dict[str, str]  # sector GICS -> ETF


def sector_etfs(ctx) -> dict[str, str]:
    """Sector GICS -> ETF sectorial (grup `sectors` de l'univers; buit si no existeix)."""
    if "sectors" not in ctx.universe.config.groups:
        return {}
    return {i.sector: i.symbol for i in ctx.universe.group("sectors") if i.sector}


def windows(cfg: InsiderConfig, end: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    d = pd.Timestamp(cfg.design_end)
    return {"disseny": (pd.Timestamp(cfg.start), d),
            "comprovació": (d + pd.Timedelta(days=1), end),
            "tot": (pd.Timestamp(cfg.start), end)}


# ---------------------------------------------------------------------------------------
# 1. Estudi d'esdeveniments
# ---------------------------------------------------------------------------------------

def event_returns(events: pd.DataFrame, px: pd.DataFrame, sector_etf: dict[str, str],
                  horizons: tuple[int, ...]) -> pd.DataFrame:
    """Afegeix ret_h, ex_spy_h i ex_sec_h (en tant per u) a cada esdeveniment.

    Entrada al tancament de `signal_date` (primera sessió en què se sabia). Si l'ETF del
    sector no té preu (XLRE abans del 2015, XLC abans del 2018) es compara amb el SPY.
    """
    ev = events.copy()
    if ev.empty:
        return ev
    px = px.ffill(limit=5)
    idx = px.index
    pos = idx.searchsorted(pd.to_datetime(ev["signal_date"]).values)
    ok = (pos < len(idx)) & (idx[np.minimum(pos, len(idx) - 1)] == pd.to_datetime(ev["signal_date"]).values)
    arr = px.to_numpy(dtype=float)
    col = {s: i for i, s in enumerate(px.columns)}
    sym_i = ev["symbol"].map(col)
    spy_i = col.get(MARKET)
    sec_i = ev["sector"].map(lambda s: col.get(sector_etf.get(s)) if isinstance(s, str) else None)
    n = len(idx)

    def fwd(ci: pd.Series, h: int) -> np.ndarray:
        out = np.full(len(ev), np.nan)
        for k, (p, c, good) in enumerate(zip(pos, ci, ok, strict=True)):
            if good and c is not None and not pd.isna(c) and p + h < n:
                a, b = arr[p, int(c)], arr[p + h, int(c)]
                if a > 0 and b > 0:
                    out[k] = b / a - 1
        return out

    spy_col = pd.Series([spy_i] * len(ev), index=ev.index)
    for h in horizons:
        r = fwd(sym_i, h)
        m = fwd(spy_col, h) if spy_i is not None else np.full(len(ev), np.nan)
        s = fwd(sec_i, h)
        s = np.where(np.isnan(s), m, s)
        ev[f"ret_{h}"] = r
        ev[f"ex_spy_{h}"] = r - m
        ev[f"ex_sec_{h}"] = r - s
    return ev


def clustered_t(values: pd.Series, dates: pd.Series) -> tuple[float, int]:
    """t de la mitjana amb agrupació per mes: mitjana de les mitjanes mensuals / error."""
    d = pd.DataFrame({"v": values.values, "m": pd.to_datetime(dates.values).to_period("M")}).dropna()
    if d.empty:
        return np.nan, 0
    mm = d.groupby("m")["v"].mean()
    if len(mm) < 3 or mm.std(ddof=1) == 0:
        return np.nan, len(mm)
    return float(mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))), len(mm)


def event_summary(ev: pd.DataFrame, cfg: InsiderConfig, end: pd.Timestamp,
                  bench: str = "ex_sec") -> pd.DataFrame:
    """Taula per esdeveniment × finestra × horitzó: N, mitjana, mediana, encerts, t mensual."""
    rows = []
    for win, (a, b) in windows(cfg, end).items():
        w = ev[(ev["signal_date"] >= a) & (ev["signal_date"] <= b)]
        for e, g in w.groupby("event", sort=False):
            for h in cfg.horizons:
                v = g[f"{bench}_{h}"].dropna()
                if v.empty:
                    continue
                t, months = clustered_t(v, g.loc[v.index, "signal_date"])
                rows.append({"finestra": win, "esdeveniment": e, "horitzo": h, "N": len(v),
                             "mesos": months, "mitjana_%": 100 * v.mean(),
                             "mediana_%": 100 * v.median(), "encerts_%": 100 * (v > 0).mean(),
                             "t_mensual": t})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# 2. Cartera en temps de calendari
# ---------------------------------------------------------------------------------------

def calendar_portfolio(events: pd.DataFrame, px: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Rendiment diari d'una cartera equiponderada de les accions amb esdeveniment actiu.

    Una acció entra al tancament de `signal_date` (guanya des de la sessió següent) i
    surt al cap de `hold` sessions. Retorna port, market i n (posicions actives).
    """
    rets = px.pct_change(fill_method=None)
    idx = px.index
    syms = [s for s in events["symbol"].unique() if s in px.columns]
    if not syms or MARKET not in px:
        return pd.DataFrame(columns=["port", "market", "n"])
    col = {s: i for i, s in enumerate(syms)}
    start = np.zeros((len(idx) + hold + 2, len(syms)))
    pos = idx.searchsorted(pd.to_datetime(events["signal_date"]).values)
    for p, s in zip(pos, events["symbol"], strict=True):
        if s in col and p < len(idx):
            start[p + 1, col[s]] += 1
            start[p + 1 + hold, col[s]] -= 1
    active = (np.cumsum(start, axis=0)[: len(idx)] > 0)
    r = rets[syms].to_numpy(dtype=float)
    held = np.where(active & ~np.isnan(r), r, np.nan)
    cnt = (~np.isnan(held)).sum(axis=1)
    port = np.where(cnt > 0, np.nansum(held, axis=1) / np.maximum(cnt, 1), np.nan)
    n = (active & ~np.isnan(r)).sum(axis=1)
    return pd.DataFrame({"port": port, "market": rets[MARKET].values, "n": n}, index=idx)


def portfolio_stats(cp: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> dict:
    d = cp[(cp.index >= a) & (cp.index <= b)].dropna(subset=["market"])
    inv = d[d["n"] > 0].dropna(subset=["port"])
    if len(inv) < 60:
        return {"dies": len(inv)}
    ex = inv["port"] - inv["market"]
    x = inv["market"].to_numpy()
    y = inv["port"].to_numpy()
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    resid = y - beta * x
    alpha = resid.mean()
    se = resid.std(ddof=1) / np.sqrt(len(resid))
    return {
        "dies": len(inv), "exposicio_%": 100 * len(inv) / max(len(d), 1),
        "posicions_mitjanes": float(inv["n"].mean()),
        "excés_anual_%": 100 * ex.mean() * TRADING_DAYS,
        "IR": float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ex.std() > 0 else np.nan,
        "alfa_anual_%": 100 * alpha * TRADING_DAYS, "t_alfa": float(alpha / se) if se > 0 else np.nan,
        "beta": float(beta),
    }


def portfolio_table(events: pd.DataFrame, px: pd.DataFrame, cfg: InsiderConfig,
                    end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, curves = [], {}
    for e, g in events.groupby("event", sort=False):
        cp = calendar_portfolio(g, px, cfg.hold_sessions)
        curves[e] = cp
        for win, (a, b) in windows(cfg, end).items():
            rows.append({"esdeveniment": e, "finestra": win, **portfolio_stats(cp, a, b)})
    return pd.DataFrame(rows), curves


# ---------------------------------------------------------------------------------------
# 3. Ablació dels filtres
# ---------------------------------------------------------------------------------------

def unfiltered_buys(trades: pd.DataFrame, cfg: InsiderConfig) -> pd.DataFrame:
    """Esdeveniments 'buy' sense els filtres de pla 10b5-1 ni d'insider rutinari."""
    t = trades.assign(eligible=True)  # candidates: ja filtrades per rol, import i preu
    ev = build_events(t, replace(cfg, cluster_min_insiders=10**6))
    return ev[ev["event"] == "buy"].assign(event="buy_sense_filtres")


def routine_buys(trades: pd.DataFrame, cfg: InsiderConfig) -> pd.DataFrame:
    """Compres només d'insiders rutinaris (haurien de ser poc informatives)."""
    t = trades.assign(eligible=trades["cls"] == "routine")
    ev = build_events(t, replace(cfg, cluster_min_insiders=10**6))
    return ev[ev["event"] == "buy"].assign(event="buy_rutinaris")


# ---------------------------------------------------------------------------------------
# 4. Amplitud agregada → mercat
# ---------------------------------------------------------------------------------------

def breadth_predictive(br: pd.DataFrame, spy: pd.Series, cfg: InsiderConfig, end: pd.Timestamp,
                       h: int = 63, step: int = 21) -> pd.DataFrame:
    """Rendiment del SPY a h sessions per terços del percentil del ràtio de compres.

    Mostreig cada `step` sessions per reduir el solapament de les finestres.
    """
    m = br[br["scope"] == "market"].set_index("date")["ratio_pct"]
    fwd = spy.shift(-h) / spy - 1
    d = pd.DataFrame({"pct": m, "fwd": fwd}).dropna().iloc[::step]
    rows = []
    for win, (a, b) in windows(cfg, end).items():
        w = d[(d.index >= a) & (d.index <= b)]
        if len(w) < 9:
            continue
        terc = pd.cut(w["pct"], [0, 1 / 3, 2 / 3, 1.0], labels=["baix", "mig", "alt"], include_lowest=True)
        g = w.groupby(terc, observed=False)["fwd"]
        rows.append({"finestra": win, "N": len(w),
                     **{f"SPY_{k}_%": 100 * v for k, v in g.mean().items()},
                     "alt_menys_baix_%": 100 * (g.mean().get("alt", np.nan) - g.mean().get("baix", np.nan)),
                     "spearman": float(w["pct"].rank().corr(w["fwd"].rank()))})  # sense scipy
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Criteris pre-registrats
# ---------------------------------------------------------------------------------------

def verdict(summary: pd.DataFrame, ports: pd.DataFrame, ablation: pd.DataFrame,
            breadth_tab: pd.DataFrame, h: int = 63) -> pd.DataFrame:
    def ev_row(e, win):
        r = summary[(summary["esdeveniment"] == e) & (summary["finestra"] == win) & (summary["horitzo"] == h)]
        return r.iloc[0] if len(r) else None

    def port_row(e, win):
        r = ports[(ports["esdeveniment"] == e) & (ports["finestra"] == win)]
        return r.iloc[0] if len(r) and "alfa_anual_%" in r and pd.notna(r.iloc[0].get("alfa_anual_%")) else None

    rows = []
    d, c = ev_row("cluster_buy", "disseny"), ev_row("cluster_buy", "comprovació")
    ok = None if d is None or c is None else bool(d["mitjana_%"] > 0 and d["t_mensual"] >= 2 and c["mitjana_%"] > 0)
    rows.append({"criteri": f"C1 compra en grup: excés vs sector a {h} sessions > 0 amb t ≥ 2 (disseny) i > 0 (comprovació)",
                 "disseny": None if d is None else f"{d['mitjana_%']:.2f} % (t {d['t_mensual']:.1f}, N {d['N']})",
                 "comprovació": None if c is None else f"{c['mitjana_%']:.2f} % (t {c['t_mensual']:.1f}, N {c['N']})",
                 "passa": ok})
    d, c = port_row("cluster_buy", "disseny"), port_row("cluster_buy", "comprovació")
    ok = None if d is None or c is None else bool(d["alfa_anual_%"] > 0 and d["IR"] >= 0.3 and c["alfa_anual_%"] > 0)
    rows.append({"criteri": "C2 cartera de compres en grup: alfa > 0 a les dues finestres i IR ≥ 0,3 (disseny)",
                 "disseny": None if d is None else f"alfa {d['alfa_anual_%']:.1f} %/any, IR {d['IR']:.2f}",
                 "comprovació": None if c is None else f"alfa {c['alfa_anual_%']:.1f} %/any, IR {c['IR']:.2f}",
                 "passa": ok})
    res = {}
    for win in ("disseny", "comprovació"):
        f = ablation[(ablation["esdeveniment"] == "buy") & (ablation["finestra"] == win) & (ablation["horitzo"] == h)]
        u = ablation[(ablation["esdeveniment"] == "buy_sense_filtres") & (ablation["finestra"] == win)
                     & (ablation["horitzo"] == h)]
        res[win] = None if f.empty or u.empty else (f["mitjana_%"].iloc[0], u["mitjana_%"].iloc[0])
    ok = None if None in res.values() else all(a > b for a, b in res.values())
    rows.append({"criteri": f"C3 filtres (10b5-1, rutinaris): compres filtrades > sense filtrar a {h} sessions",
                 **{w: None if v is None else f"{v[0]:.2f} % vs {v[1]:.2f} %" for w, v in res.items()},
                 "passa": ok})
    bt = breadth_tab.set_index("finestra") if not breadth_tab.empty else pd.DataFrame()
    have = all(w in bt.index for w in ("disseny", "comprovació"))
    ok = bool(have and (bt.loc[["disseny", "comprovació"], "alt_menys_baix_%"] > 0).all()) if have else None
    rows.append({"criteri": "C4 amplitud: SPY a 63 sessions més alt amb ràtio de compres alt que baix (dues finestres)",
                 **{w: (f"{bt.loc[w, 'alt_menys_baix_%']:.2f} pp" if w in bt.index else None)
                    for w in ("disseny", "comprovació")},
                 "passa": ok})
    return pd.DataFrame(rows)
