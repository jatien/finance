"""Informe diari: què fer a la sessió següent, com va el paper trading i si les dades són fiables.

Es genera després del pipeline diari (job `report`) a `<FIN_DATA_DIR>/reports/daily/`:
`AAAA-MM-DD.html`, `latest.html` i `latest.md` (resum de text per llegir per SSH).

Ordre de les seccions = ordre de les preguntes de cada matí:
1. Puc fiar-me de les dades d'avui?            (salut de dades: si no, no s'opera)
2. Què he de fer a la sessió següent?          (ordres per estratègia, amb imports)
3. En quin règim som i què ha canviat?
4. Com va el paper trading des de l'inici?     (contra SPY; comprovació contra les dades actuals)
5. Registre de decisions                        (cada operació i per què)
6. Context: backtest recent i rànquing de sectors (simulació, etiquetada com a tal)
"""

from __future__ import annotations

import html as _html
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finance.calendar import last_completed_session, sessions_between
from finance.features.diagnostics import GRID, INK, INK_2, MUTED, SURFACE, _layout
from finance.features.job import read_market_features
from finance.jobs import Job, Run
from finance.paper.book import EPS, BookState, _cov_at, replay, sector_ranks, sleeve_of, trigger_of
from finance.paper.job import read_paper_book, read_paper_nav, read_state
from finance.portfolio.backtest import lagged, simulate, simulate_cfg
from finance.portfolio.config import ACTIVE, CASH, PortfolioConfig
from finance.portfolio.construct import PortfolioInputs, Targets, build_targets, fixed_targets
from finance.portfolio.evaluation import regime_on_market
from finance.portfolio.job import load_inputs
from finance.regimes.job import read_regimes
from finance.regimes.report import REGIME_LABELS

log = logging.getLogger(__name__)

# Paleta categòrica validada (scripts/validate_palette.js): ordre fix, mai cíclica.
# El verd queda per sota de 3:1 de contrast: per això cada gràfic té etiquetes directes i taula.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
WARN_BG, WARN_INK = "#fdf3d7", "#6b4e00"
BAD_BG, BAD_INK = "#fbe3e1", "#8a1c12"
OK_BG, OK_INK = "#e3f4ec", "#0d5c3b"
ACTION_LABELS = {"enter": "comprar (nova)", "add": "comprar", "reduce": "vendre", "exit": "vendre (tot)",
                 "hold": "mantenir"}
KEY_FEATURES = [  # (feature, etiqueta, format)
    ("mkt_dist_sma200", "SPY vs SMA200", "{:+.1%}"),
    ("vix", "VIX", "{:.1f}"),
    ("vix_term", "VIX / VIX3M", "{:.2f}"),
    ("sector_breadth_200", "Sectors sobre SMA200", "{:.0%}"),
    ("stock_bond_corr_3m", "Correlació SPY–TLT (3 m)", "{:+.2f}"),
    ("baa_spread_chg_3m", "Canvi spread Baa (3 m)", "{:+.2f} pp"),
]


# ---------------------------------------------------------------------------------------
# Dades de l'informe
# ---------------------------------------------------------------------------------------

@dataclass
class Order:
    symbol: str
    sleeve: str
    current: float
    target: float
    amount: float  # import (capital × NAV × canvi de pes); > 0 compra
    price: float
    shares: float

    @property
    def change(self) -> float:
        return self.target - self.current

    @property
    def action(self) -> str:
        if self.current <= EPS:
            return "enter"
        if self.target <= EPS:
            return "exit"
        return "add" if self.change > 0 else "reduce"


@dataclass
class StrategyView:
    cfg: PortfolioConfig
    state: BookState | None
    orders: list[Order] = field(default_factory=list)
    order_note: str = ""
    will_trade: bool = False
    distance: float = np.nan
    distance_kind: str | None = None
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)
    nav: pd.DataFrame = field(default_factory=pd.DataFrame)
    replay_gap: float = np.nan  # màx |NAV registrat - NAV recalculat amb dades actuals|
    perf: dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyReport:
    generated_at: pd.Timestamp
    data_date: pd.Timestamp | None
    expected_session: pd.Timestamp
    sessions_behind: int
    issues: list[tuple[str, str]]  # (severitat: error|warn, text)
    regime: dict[str, Any]
    regime_history: pd.DataFrame
    strategies: list[StrategyView]
    log: pd.DataFrame
    backtest: pd.DataFrame
    backtest_curves: pd.DataFrame
    market: list[tuple[str, str, str]]  # (etiqueta, valor, percentil)
    ranking: pd.DataFrame
    next_review: pd.Timestamp | None
    capital: float
    currency: str
    insiders: dict[str, Any] | None = None  # context d'insiders (no modifica les ordres)

    @property
    def tradable(self) -> bool:
        return not any(sev == "error" for sev, _ in self.issues)


def build_daily(ctx, capital: float | None = None, currency: str | None = None,
                strategies: tuple[PortfolioConfig, ...] = ACTIVE) -> DailyReport:
    capital = float(capital if capital is not None else ctx.settings.paper_capital)
    currency = currency or ctx.settings.paper_currency
    now = ctx.now()
    expected = pd.Timestamp(last_completed_session("XNYS", now))
    ref = strategies[0]
    inputs = {c.name: load_inputs(ctx, c) for c in strategies}
    targets = {c.name: build_targets(inputs[c.name], c) for c in strategies}
    inp0 = inputs[ref.name]
    data_date = inp0.index[-1] if len(inp0.index) else None
    behind = 0 if data_date is None else len(sessions_between((data_date + pd.Timedelta(days=1)).date(),
                                                              expected.date()))
    issues = data_issues(ctx, now, data_date, expected, behind)

    views = [strategy_view(ctx, c, inputs[c.name], targets[c.name], capital) for c in strategies]
    for v in views:
        if v.state is None:
            issues.append(("warn", f"{v.cfg.name}: el paper trading encara no ha començat (`fin paper update`)"))
        elif data_date is not None and v.state.date < data_date:
            issues.append(("warn", f"{v.cfg.name}: paper trading registrat fins a {v.state.date:%Y-%m-%d}, "
                                   f"dades fins a {data_date:%Y-%m-%d} (`fin paper update`)"))
        if pd.notna(v.replay_gap) and v.replay_gap > 1e-6:
            issues.append(("warn", f"{v.cfg.name}: recalculat amb les dades d'avui, el NAV del paper trading "
                                   f"difereix fins a {v.replay_gap:.4f}: el proveïdor ha corregit preus "
                                   "(normal amb dividends). El registre conserva el que es va decidir."))

    reg = read_regimes(ctx, ref.regime_set)
    regime = {}
    hist = pd.DataFrame()
    if not reg.empty:
        last = reg.iloc[-1]
        prev = reg["regime"].iloc[:-1]
        changed_at = reg.index[-int(last["days_in_regime"])] if pd.notna(last.get("days_in_regime")) else None
        regime = {"regime": last["regime"], "label": REGIME_LABELS.get(last["regime"], last["regime"]),
                  "exposure": float(last["equity_exposure"]), "days": int(last["days_in_regime"] or 0),
                  "action": last.get("action"), "since": changed_at,
                  "previous": prev[prev != last["regime"]].iloc[-1] if (prev != last["regime"]).any() else None,
                  "changed_today": bool(len(prev) and prev.iloc[-1] != last["regime"])}
        runs = (reg["regime"] != reg["regime"].shift()).cumsum()
        g = reg.groupby(runs).agg(regime=("regime", "first"), start=("regime", lambda s: s.index[0]),
                                  end=("regime", lambda s: s.index[-1]), sessions=("regime", "size"),
                                  exposure=("equity_exposure", "first"))
        hist = g.tail(6).iloc[::-1].reset_index(drop=True)

    bt, curves = backtest_context(inputs, targets, strategies)
    return DailyReport(
        generated_at=now, data_date=data_date, expected_session=expected, sessions_behind=behind,
        issues=issues, regime=regime, regime_history=hist, strategies=views,
        log=decision_log(ctx, strategies), backtest=bt, backtest_curves=curves,
        market=market_snapshot(ctx, ref), ranking=sector_ranking(inp0, ref, views[0]),
        next_review=next_review(data_date), capital=capital, currency=currency,
        insiders=insider_context(ctx, views[0], data_date))


# ---------------------------------------------------------------------------------------
# Peces
# ---------------------------------------------------------------------------------------


def insider_context(ctx, view: StrategyView, data_date, days: int = 14) -> dict[str, Any] | None:
    """Esdeveniments d'insiders recents al S&P 500 i amplitud per sector (només context).

    Les estratègies congelades no fan servir aquesta informació: la senyal està en
    validació (docs/insiders.md). Si no hi ha dades d'insiders, la secció no apareix.
    """
    try:
        from finance.insiders.evaluation import sector_etfs
        from finance.insiders.job import read_breadth, read_events
        ev = read_events(ctx)
        if ev.empty:
            return None
        end = pd.Timestamp(data_date) if data_date is not None else ev["signal_date"].max()
        recent = ev[(ev["signal_date"] > end - pd.Timedelta(days=days)) & ev["in_sp500"]
                    & (ev["event"] != "buy")].sort_values("signal_date", ascending=False)
        br = read_breadth(ctx)
        etf = sector_etfs(ctx)
        held = set(view.holdings["actiu"]) if not view.holdings.empty else set()
        sec = pd.DataFrame()
        if not br.empty:
            last = br[br["date"] <= end].dropna(subset=["buy_ratio"]).groupby("scope").tail(1)
            sec = last.assign(etf=last["scope"].map(etf).fillna(""),
                              held=last["scope"].map(etf).isin(held))
            sec = sec.sort_values("buy_ratio", ascending=False)
        return {"events": recent, "breadth": sec, "days": days}
    except Exception:  # noqa: BLE001
        log.exception("context d'insiders no disponible")
        return None


def data_issues(ctx, now: pd.Timestamp, data_date, expected: pd.Timestamp, behind: int) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    if data_date is None:
        return [("error", "no hi ha preus al lake")]
    if behind > 0:
        issues.append(("error", f"les dades acaben el {data_date:%Y-%m-%d} i l'última sessió tancada és el "
                                f"{expected:%Y-%m-%d} ({behind} sessió(ns) de retard): no operis amb aquest informe"))
    try:
        runs = ctx.query("""
            SELECT job, status, message, started_at FROM (
              SELECT *, row_number() OVER (PARTITION BY job ORDER BY started_at DESC) AS rn
              FROM ingest_runs WHERE started_at >= ?) WHERE rn = 1
        """, [(now - pd.Timedelta(hours=36)).to_pydatetime()])
    except Exception:  # noqa: BLE001
        runs = pd.DataFrame()
    for r in runs.itertuples():
        if r.job in ("report", "options"):
            continue
        if r.status == "failed":
            issues.append(("error", f"el job `{r.job}` ha fallat a l'última execució"))
        elif r.status == "partial":
            issues.append(("warn", f"el job `{r.job}` ha acabat amb errors parcials"))
    try:
        dq = ctx.query("""
            SELECT dataset, check_name, key, details FROM dq_results
            WHERE severity = 'error' AND run_id = (SELECT run_id FROM ingest_runs WHERE job = 'dq'
                                                   ORDER BY started_at DESC LIMIT 1)
        """)
    except Exception:  # noqa: BLE001
        dq = pd.DataFrame()
    if len(dq):
        sample = ", ".join(f"{r.dataset}/{r.check_name}/{r.key}" for r in dq.head(4).itertuples())
        issues.append(("warn", f"{len(dq)} errors de qualitat de dades ({sample}{'…' if len(dq) > 4 else ''})"))
    return issues


def strategy_view(ctx, cfg: PortfolioConfig, inp: PortfolioInputs, targets: Targets, capital: float) -> StrategyView:
    state = read_state(ctx, cfg.name)
    v = StrategyView(cfg=cfg, state=state)
    if state is None:
        return v
    nav = read_paper_nav(ctx, cfg.name)
    v.nav = nav
    assets = list(targets.weights.columns)
    w = state.weights.reindex(assets).fillna(0.0)
    tgt = state.pending.reindex(assets).fillna(0.0)
    # Ordres per a la sessió següent: el disparador s'avalua amb els pesos d'avui (l'endemà
    # derivaran una mica: és una previsió, l'execució real la decideix el job de demà).
    live = state.force or state.pending_new or cfg.check_daily
    if live:
        i = targets.weights.index.get_loc(state.date) if state.date in targets.weights.index else None
        rets = inp.asset_returns().reindex(columns=assets)
        cov_i = _cov_at(rets, assets, i, cfg) if (i is not None and cfg.rebalance == "tracking_error") else None
        trig = trigger_of(cfg)
        v.distance, v.distance_kind = trig.distance(tgt.to_numpy(), w.to_numpy(), cov_i)
        v.will_trade = state.force or trig.fires(tgt.to_numpy(), w.to_numpy(), cov_i)
    thr = cfg.te_band if v.distance_kind == "te" else cfg.band
    dist_txt = (f"{'TE' if v.distance_kind == 'te' else 'desviació màxima'} {v.distance:.2%}"
                if pd.notna(v.distance) else "")
    if v.will_trade:
        why = state.pending_reason if (state.force or state.pending_new) else "deriva"
        v.order_note = (f"Executar al tancament de la sessió següent. Motiu: {why}"
                        + (f" ({dist_txt} > {thr:.0%})" if dist_txt and not state.force else "") + ".")
        value = capital * state.nav
        book = read_paper_book(ctx, cfg.name, state.date).set_index("symbol")
        for s in assets:
            d = float(tgt[s] - w[s])
            if abs(d) < 1e-4 or s == CASH:
                continue
            px = float(book["price"].get(s, np.nan)) if s in book.index else np.nan
            if np.isnan(px) and s in inp.prices:
                px = float(inp.prices[s].iloc[-1])
            amt = value * d
            v.orders.append(Order(s, sleeve_of(s, cfg), float(w[s]), float(tgt[s]), amt, px,
                                  amt / px if px and px > 0 else np.nan))
        v.orders.sort(key=lambda o: o.amount)  # vendes primer: alliberen efectiu
    elif live:
        v.order_note = f"Cap ordre: la cartera és dins de la banda ({dist_txt} ≤ {thr:.0%})."
    else:
        v.order_note = ("Cap ordre: no hi ha objectiu nou i aquesta estratègia només negocia a les "
                        "revisions (inici de mes o canvi de règim).")
    rows = []
    for s in assets:
        if w[s] > EPS or tgt[s] > EPS:
            rows.append({"actiu": s, "bloc": sleeve_of(s, cfg), "pes": w[s], "objectiu": tgt[s],
                         "rang": state.ranks.get(s)})
    v.holdings = pd.DataFrame(rows).sort_values("pes", ascending=False) if rows else pd.DataFrame()
    # Rendiment des de l'inici i comprovació contra les dades actuals
    nv = nav.set_index("date")
    bench = (1 + nv["benchmark_ret"].fillna(0.0)).cumprod()
    bench.iloc[0] = 1.0
    navs = nv["nav"]
    v.perf = {"inception": state.inception, "sessions": len(nv) - 1, "nav": float(navs.iloc[-1]),
              "ret": float(navs.iloc[-1] - 1), "bench_ret": float(bench.iloc[-1] - 1),
              "max_dd": float((navs / navs.cummax() - 1).min()), "trades": int(nv["traded"].sum()),
              "cost": float(nv["cost"].sum()), "value": capital * float(navs.iloc[-1]),
              "bench_curve": bench, "late": int(nv["recorded_late"].sum())}
    try:
        rp = replay(inp, targets, cfg, state.inception, state.date).nav.set_index("date")["nav"]
        v.replay_gap = float((rp.reindex(navs.index) - navs).abs().max())
    except Exception:  # noqa: BLE001 — p. ex. l'inici ja no és a l'índex de dades
        v.replay_gap = np.nan
    return v


def decision_log(ctx, strategies: tuple[PortfolioConfig, ...], n: int = 12) -> pd.DataFrame:
    nav = read_paper_nav(ctx)
    if nav.empty:
        return pd.DataFrame()
    nav = nav[nav["strategy_set"].isin([c.name for c in strategies]) & nav["traded"]]
    if nav.empty:
        return pd.DataFrame()
    nav = nav.sort_values("date", ascending=False).head(n)
    book = read_paper_book(ctx)
    rows = []
    for r in nav.itertuples():
        b = book[(book["strategy_set"] == r.strategy_set) & (book["date"] == r.date) & (book["action"] != "hold")]
        trades = ", ".join(f"{x.symbol} {x.trade:+.1%}" for x in b.sort_values("trade").itertuples()
                           if x.symbol != CASH)
        rows.append({"data": r.date.date(), "estratègia": r.strategy_set, "motiu": r.reason,
                     "operacions": trades, "rotació": r.turnover, "cost_pb": 1e4 * r.cost})
    return pd.DataFrame(rows)


def backtest_context(inputs: dict[str, PortfolioInputs], targets: dict[str, Targets],
                     strategies: tuple[PortfolioConfig, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulació (no paper) amb el mateix retard d'una sessió: 1 m, 3 m, any en curs i 12 m."""
    ref = strategies[0]
    inp = inputs[ref.name]
    rets = inp.asset_returns()
    curves = {}
    for c in strategies:
        s = simulate_cfg(lagged(targets[c.name], 1), inputs[c.name].asset_returns(), c)
        curves[c.name] = s.returns
    end = inp.index[-1]
    start = targets[ref.name].weights.dropna().index.min()
    curves["SPY"] = simulate(fixed_targets(inp.index, {ref.market: 1.0}, monthly=False, start=start),
                             rets, ref.cost_bps, ref.band).returns
    curves["Règims sobre SPY"] = simulate(lagged(regime_on_market(inp, ref), 1), rets, ref.cost_bps, ref.band).returns
    df = pd.DataFrame(curves).fillna(0.0)
    windows = {"1 mes": end - pd.DateOffset(months=1), "3 mesos": end - pd.DateOffset(months=3),
               "any en curs": pd.Timestamp(end.year, 1, 1) - pd.Timedelta(days=1),
               "12 mesos": end - pd.DateOffset(months=12)}
    rows = []
    for name in df:
        r = df[name]
        row = {"estratègia": name}
        for label, a in windows.items():
            row[label] = float((1 + r.loc[r.index > a]).prod() - 1)
        eq = (1 + r.loc[r.index > windows["12 mesos"]]).cumprod()
        row["caiguda màx. 12 m"] = float((eq / eq.cummax() - 1).min())
        rows.append(row)
    last12 = df.loc[df.index > windows["12 mesos"]]
    return pd.DataFrame(rows), (1 + last12).cumprod()


def market_snapshot(ctx, cfg: PortfolioConfig) -> list[tuple[str, str, str]]:
    try:
        val = read_market_features(ctx, cfg.feature_set, "value")
        pct = read_market_features(ctx, cfg.feature_set, "pct_rank")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for f, label, fmt in KEY_FEATURES:
        if f not in val or val[f].dropna().empty:
            continue
        x = val[f].dropna()
        p = pct[f].reindex(x.index).iloc[-1] if f in pct else np.nan
        out.append((label, fmt.format(x.iloc[-1]), f"{p:.0%}" if pd.notna(p) else "—"))
    return out


def sector_ranking(inp: PortfolioInputs, cfg: PortfolioConfig, view: StrategyView) -> pd.DataFrame:
    t = inp.index[-1]
    try:
        ranks = sector_ranks(inp, t, cfg)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    held = set(view.holdings["actiu"]) if not view.holdings.empty else set()
    rows = []
    for s, rk in sorted(ranks.items(), key=lambda kv: kv[1]):
        rows.append({"rang": rk, "sector": s, "momentum 12-1": inp.panel["mom_12_1"].at[t, s],
                     "força relativa 6 m": inp.panel["rel_strength_6m"].at[t, s],
                     "vol 3 m": inp.panel["vol_3m"].at[t, s],
                     "en cartera": "sí" if s in held else ""})
    return pd.DataFrame(rows)


def next_review(data_date) -> pd.Timestamp | None:
    if data_date is None:
        return None
    first = (data_date + pd.offsets.MonthBegin(1)).normalize()
    s = sessions_between(first.date(), (first + pd.Timedelta(days=10)).date())
    return s[0] if len(s) else None


# ---------------------------------------------------------------------------------------
# Sortides
# ---------------------------------------------------------------------------------------

def _e(x: Any) -> str:
    return _html.escape("" if x is None else str(x))


def _money(x: float, cur: str) -> str:
    return f"{x:,.0f} {cur}".replace(",", " ")


def _table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    if df.empty:
        return "<p class='muted'>—</p>"
    fmt = fmt or {}
    head = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                s = "—"
            elif c in fmt:
                s = fmt[c].format(v)
            else:
                s = str(v)
            num = isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
            cells.append(f"<td class='{'n' if num else ''}'>{_e(s)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _dodge(values: list[float], min_gap: float) -> list[float]:
    """Separa verticalment les etiquetes finals perquè no se superposin (manté l'ordre)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if out[hi] - out[lo] < min_gap:
            out[hi] = out[lo] + min_gap
    return out


def _curve_chart(curves: pd.DataFrame, names: list[str], title: str, height: int = 340) -> str:
    """Corbes en base 1 amb etiqueta directa al final (la identitat no depèn només del color)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    ends: list[tuple[str, pd.Timestamp, float]] = []
    lo, hi = np.inf, -np.inf
    for i, n in enumerate(names):
        if n not in curves:
            continue
        y = curves[n].dropna()
        if y.empty:
            continue
        col = SERIES[i] if i < len(SERIES) else MUTED
        fig.add_trace(go.Scatter(x=y.index, y=y.values, name=n, mode="lines",
                                 line={"color": col, "width": 2},
                                 hovertemplate="%{y:.3f}<extra>" + _e(n) + "</extra>"))
        ends.append((n, y.index[-1], float(y.values[-1])))
        lo, hi = min(lo, float(y.min())), max(hi, float(y.max()))
    if ends:
        pos = _dodge([v for _, _, v in ends], 0.07 * max(hi - lo, 1e-6))
        for (n, x, v), yp in zip(ends, pos, strict=True):
            fig.add_annotation(x=x, y=yp, text=f"{n} {v - 1:+.1%}", showarrow=False, xanchor="left",
                               xshift=6, font={"size": 11, "color": INK_2})
        x0 = min(curves[n].dropna().index[0] for n, _, _ in ends)
        x1 = max(x for _, x, _ in ends)
        fig.update_xaxes(range=[x0, x1 + (x1 - x0) * 0.3])  # espai per a les etiquetes
    fig.update_yaxes(gridcolor=GRID, tickformat=".2f")
    fig.update_xaxes(showgrid=False, nticks=6)
    _layout(fig, title, height)
    fig.update_layout(hovermode="x unified", legend={"orientation": "h", "y": -0.15},
                      margin={"r": 16, "l": 44, "t": 50, "b": 40}, autosize=True, width=None)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": False})


def _plotly_js_version() -> str:
    from plotly.offline import get_plotlyjs_version
    return get_plotlyjs_version()


def render_html(rep: DailyReport) -> str:
    cur = rep.currency
    banner_cls, banner_txt = ("ok", "Dades al dia: es pot operar amb aquest informe.") if rep.tradable else \
        ("bad", "No operis amb aquest informe: hi ha problemes de dades.")
    issues = "".join(f"<li class='{sev}'>{_e(t)}</li>" for sev, t in rep.issues)
    ref = rep.strategies[0]
    action_needed = any(v.will_trade for v in rep.strategies[:1])

    # Ordres
    blocks = []
    for k, v in enumerate(rep.strategies):
        tag = "referència" if k == 0 else "en paral·lel"
        if v.state is None:
            blocks.append(f"<div class='card'><h3>{_e(v.cfg.name)} <span class='tag'>{tag}</span></h3>"
                          "<p class='muted'>Encara no hi ha paper trading.</p></div>")
            continue
        if v.orders:
            od = pd.DataFrame([{"acció": ACTION_LABELS[o.action], "actiu": o.symbol, "bloc": o.sleeve,
                                "pes actual": o.current, "pes objectiu": o.target, "canvi": o.change,
                                f"import ({cur})": o.amount, "preu": o.price, "accions ≈": o.shares}
                               for o in v.orders])
            orders = _table(od, {"pes actual": "{:.1%}", "pes objectiu": "{:.1%}", "canvi": "{:+.1%}",
                                 f"import ({cur})": "{:+,.0f}", "preu": "{:,.2f}", "accions ≈": "{:+,.1f}"})
        else:
            orders = ""
        hold = _table(v.holdings, {"pes": "{:.1%}", "objectiu": "{:.1%}", "rang": "{:.0f}"})
        p = v.perf
        blocks.append(f"""<div class='card'><h3>{_e(v.cfg.name)} <span class='tag'>{tag}</span>
<span class='tag {"act" if v.will_trade else ""}'>{"cal operar" if v.will_trade else "sense ordres"}</span></h3>
<p>{_e(v.order_note)}</p>{orders}
<details><summary>Cartera actual i objectiu · valor {_money(p["value"], cur)}</summary>{hold}</details></div>""")

    # Paper trading
    perf_rows = []
    for v in rep.strategies:
        if v.state is None:
            continue
        p = v.perf
        perf_rows.append({"estratègia": v.cfg.name, "inici": f"{p['inception']:%Y-%m-%d}", "sessions": p["sessions"],
                          "rendiment": p["ret"], "SPY": p["bench_ret"], "diferència": p["ret"] - p["bench_ret"],
                          "caiguda màx.": p["max_dd"], "sessions amb operacions": p["trades"],
                          "costos": p["cost"], f"valor ({cur})": p["value"],
                          "recalculat vs registre": v.replay_gap})
    perf = pd.DataFrame(perf_rows)
    perf_html = _table(perf, {"rendiment": "{:+.2%}", "SPY": "{:+.2%}", "diferència": "{:+.2%}",
                              "caiguda màx.": "{:.2%}", "costos": "{:.3%}", f"valor ({cur})": "{:,.0f}",
                              "recalculat vs registre": "{:.1e}"})
    paper_chart = ""
    longest = max((v.perf.get("sessions", 0) for v in rep.strategies if v.state), default=0)
    if longest >= 5:
        curves = pd.DataFrame({v.cfg.name: v.nav.set_index("date")["nav"] for v in rep.strategies if v.state})
        curves["SPY"] = ref.perf["bench_curve"] if ref.state else np.nan
        paper_chart = _curve_chart(curves, list(curves.columns), "Paper trading des de l'inici (base 1)")
    else:
        paper_chart = (f"<p class='muted'>El gràfic apareix a partir de 5 sessions de paper trading "
                       f"(ara: {longest}). Fins llavors, la referència és la simulació de la secció 6.</p>")

    # Règim
    rg = rep.regime
    regime_html = "<p class='muted'>Sense règims.</p>"
    if rg:
        change = (f"<b>Canvi avui</b>: {_e(REGIME_LABELS.get(rg['previous'], rg['previous']))} → {_e(rg['label'])}."
                  if rg["changed_today"] else
                  f"Des del {rg['since']:%Y-%m-%d} ({rg['days']} sessions)." if rg.get("since") is not None else "")
        hist = rep.regime_history.copy()
        if not hist.empty:
            hist["regime"] = hist["regime"].map(lambda g: REGIME_LABELS.get(g, g))
            hist["start"] = hist["start"].dt.date
            hist["end"] = hist["end"].dt.date
            hist = hist.rename(columns={"regime": "règim", "start": "inici", "end": "fi", "sessions": "sessions",
                                        "exposure": "renda variable"})
        regime_html = f"""<div class='kpis'><div class='kpi'><span>Règim</span><b>{_e(rg['label'])}</b></div>
<div class='kpi'><span>Renda variable recomanada</span><b>{rg['exposure']:.0%}</b></div>
<div class='kpi'><span>Propera revisió mensual</span><b>{f"{rep.next_review:%d-%m-%Y}" if rep.next_review is not None else "—"}</b></div></div>
<p>{change} Un canvi de règim provoca una revisió immediata.</p>
<details><summary>Últims règims</summary>{_table(hist, {"renda variable": "{:.0%}"})}</details>"""

    market = "".join(f"<tr><td>{_e(a)}</td><td class='n'>{_e(b)}</td><td class='n'>{_e(c)}</td></tr>"
                     for a, b, c in rep.market)
    bt_fmt = {c: "{:+.1%}" for c in rep.backtest.columns if c != "estratègia"}
    bt_names = [v.cfg.name for v in rep.strategies] + ["SPY"]
    bt_chart = _curve_chart(rep.backtest_curves, bt_names, "Simulació: últims 12 mesos (base 1, execució t+1)", 320)
    rank = _table(rep.ranking, {"momentum 12-1": "{:+.1%}", "força relativa 6 m": "{:+.2f}", "vol 3 m": "{:.1%}"})
    log = _table(rep.log, {"rotació": "{:.1%}", "cost_pb": "{:.1f}"})
    days = ["dilluns", "dimarts", "dimecres", "dijous", "divendres", "dissabte", "diumenge"]
    date_txt = f"{days[rep.data_date.weekday()]} {rep.data_date:%d-%m-%Y}" if rep.data_date is not None else "—"

    return f"""<!doctype html><html lang="ca"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe diari {rep.data_date:%Y-%m-%d}</title>
<script src="https://cdn.plot.ly/plotly-{_plotly_js_version()}.min.js"></script>
<style>
:root{{--bg:{SURFACE};--ink:{INK};--ink2:{INK_2};--muted:{MUTED};--grid:{GRID}}}
body{{background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
max-width:1100px;margin:20px auto;padding:0 16px;line-height:1.45}}
h1{{font-size:22px;margin:0}} h2{{font-size:17px;margin:32px 0 8px}} h3{{font-size:15px;margin:0 0 6px}}
.sub,.muted{{color:var(--ink2)}} .muted{{font-size:13px}}
.banner{{padding:10px 14px;border-radius:8px;margin:14px 0;font-weight:600}}
.banner.ok{{background:{OK_BG};color:{OK_INK}}} .banner.bad{{background:{BAD_BG};color:{BAD_INK}}}
ul.issues{{margin:6px 0 0;padding-left:18px}} li.error{{color:{BAD_INK}}} li.warn{{color:{WARN_INK}}}
.card{{border:1px solid var(--grid);border-radius:8px;padding:12px 14px;margin:10px 0}}
.tag{{font-size:11px;font-weight:600;border:1px solid var(--grid);border-radius:10px;padding:1px 8px;
margin-left:6px;color:var(--ink2);vertical-align:middle}} .tag.act{{background:{WARN_BG};color:{WARN_INK};border-color:transparent}}
.kpis{{display:flex;flex-wrap:wrap;gap:10px}} .kpi{{border:1px solid var(--grid);border-radius:8px;padding:8px 14px}}
.kpi span{{display:block;font-size:12px;color:var(--ink2)}} .kpi b{{font-size:19px}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}}
th{{text-align:left;color:var(--ink2);font-weight:600;border-bottom:1px solid #c3c2b7;padding:6px 8px;white-space:nowrap}}
td{{border-bottom:1px solid var(--grid);padding:5px 8px}} td.n{{text-align:right;white-space:nowrap}}
details{{margin-top:8px}} summary{{cursor:pointer;color:var(--ink2);font-size:13px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} .grid2>div{{min-width:0}} @media (max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Informe diari · {_e(date_txt)}</h1>
<p class="sub">Dades fins al tancament del {rep.data_date:%Y-%m-%d} · generat {rep.generated_at.tz_convert("Europe/Madrid"):%d-%m-%Y %H:%M} (Madrid) ·
capital nocional {_money(rep.capital, cur)}</p>
<div class="banner {banner_cls}">{banner_txt}{" · Avui cal operar (" + _e(ref.cfg.name) + ")." if action_needed and rep.tradable else ""}</div>
{f"<ul class='issues'>{issues}</ul>" if issues else ""}

<h2>1. Què fer a la sessió següent</h2>
<p class="muted">Les ordres s'executen al tancament de la sessió següent (com el paper trading). Imports sobre
el valor actual de cada estratègia; "accions" és orientatiu al preu de tancament d'avui. Vendes primer.</p>
{"".join(blocks)}

<h2>2. Règim de mercat</h2>
{regime_html}

<h2>3. Paper trading</h2>
<p class="muted">Registre congelat: cada sessió s'escriu una vegada amb les dades d'aquell dia i no es
recalcula mai. "Recalculat vs registre" compara amb el que donaria el mateix motor amb les dades d'avui
(0 = cap correcció del proveïdor).</p>
{perf_html}
{paper_chart}

<h2>4. Registre de decisions</h2>
{log}

<h2>5. Context de mercat</h2>
<div class="grid2"><div><div class="wrap"><table><thead><tr><th>Indicador</th><th>Valor</th><th>Percentil històric</th></tr></thead>
<tbody>{market}</tbody></table></div></div>
<div><h3>Rànquing de sectors ({_e(ref.cfg.name)})</h3>{rank}</div></div>

<h2>6. Simulació recent (no és paper trading)</h2>
<p class="muted">Backtest amb les mateixes regles, costos i retard d'una sessió. Serveix de referència mentre
el paper trading acumula història.</p>
{_table(rep.backtest, bt_fmt)}
{bt_chart}
{_insiders_html(rep)}
</body></html>"""



def _insiders_html(rep: DailyReport) -> str:
    ins = rep.insiders
    if not ins:
        return ""
    labels = {"cluster_buy": "compra en grup", "ceo_cfo_buy": "compra CEO/CFO", "cluster_sell": "venda en grup"}
    ev = ins["events"]
    if ev.empty:
        ev_html = f"<p class='muted'>Cap compra en grup, del CEO/CFO ni venda en grup als últims {ins['days']} dies.</p>"
    else:
        t = pd.DataFrame({"data": ev["signal_date"].dt.strftime("%Y-%m-%d"), "acció": ev["symbol"],
                          "sector": ev["sector"], "esdeveniment": ev["event"].map(labels).fillna(ev["event"]),
                          "insiders": ev["n_insiders"], "import (M$)": ev["value"] / 1e6,
                          "import / volum diari": ev["value_adv"]})
        ev_html = _table(t, {"import (M$)": "{:,.2f}", "import / volum diari": "{:.3f}", "insiders": "{:.0f}"})
    br = ins["breadth"]
    br_html = ""
    if not br.empty:
        b = pd.DataFrame({"àmbit": br["scope"].where(br["scope"] != "market", "Mercat (S&P 500)"),
                          "ETF": br["etf"], "en cartera": br["held"].map({True: "sí", False: ""}),
                          "compres": br["buyers"], "vendes": br["sellers"], "ràtio de compres": br["buy_ratio"],
                          "percentil": br["ratio_pct"]})
        br_html = _table(b, {"compres": "{:.0f}", "vendes": "{:.0f}", "ràtio de compres": "{:.0%}",
                             "percentil": "{:.0%}"})
    return f"""<h2>7. Insiders (context, no modifica les ordres)</h2>
<p class="muted">Form 4 de la SEC: compres i vendes en mercat obert de consellers i directius, fora de plans
10b5-1 i d'insiders rutinaris. La senyal està en validació (<code>fin insiders report</code>, docs/insiders.md):
les estratègies congelades no la fan servir.</p>
<div class="grid2"><div><h3>Esdeveniments recents al S&amp;P 500</h3><div class="wrap">{ev_html}</div></div>
<div><h3>Amplitud per sector (63 sessions)</h3><div class="wrap">{br_html}</div></div></div>"""


def render_text(rep: DailyReport) -> str:
    cur = rep.currency
    L = [f"# Informe diari {rep.data_date:%Y-%m-%d}", ""]
    L.append("OK: dades al dia." if rep.tradable else "NO OPERIS: problemes de dades.")
    L += [f"- [{sev}] {t}" for sev, t in rep.issues]
    if rep.regime:
        rg = rep.regime
        L += ["", f"Règim: {rg['label']} · renda variable {rg['exposure']:.0%} · {rg['days']} sessions"
              + (" · CANVI AVUI" if rg["changed_today"] else "")]
    for k, v in enumerate(rep.strategies):
        L += ["", f"## {v.cfg.name}{' (referència)' if k == 0 else ''}"]
        if v.state is None:
            L.append("sense paper trading")
            continue
        L.append(v.order_note)
        for o in v.orders:
            L.append(f"  {ACTION_LABELS[o.action]:<15} {o.symbol:<5} {o.current:6.1%} → {o.target:6.1%}  "
                     f"{o.amount:+10,.0f} {cur}  ≈ {o.shares:+.1f} accions")
        p = v.perf
        L.append(f"Paper des del {p['inception']:%Y-%m-%d}: {p['ret']:+.2%} (SPY {p['bench_ret']:+.2%}), "
                 f"valor {_money(p['value'], cur)}")
    ins = rep.insiders
    if ins and not ins["events"].empty:
        L += ["", f"## Insiders (últims {ins['days']} dies, només context)"]
        for r in ins["events"].itertuples():
            L.append(f"  {r.signal_date:%Y-%m-%d} {r.symbol:<6} {r.event:<13} {r.n_insiders} insiders "
                     f"{r.value / 1e6:,.2f} M$")
    return "\n".join(L) + "\n"


def write_daily(rep: DailyReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = render_html(rep)
    text = render_text(rep)
    dated = out_dir / f"{rep.data_date:%Y-%m-%d}.html"
    paths = {"html": dated, "latest_html": out_dir / "latest.html", "latest_md": out_dir / "latest.md"}
    for p, content in ((dated, html), (paths["latest_html"], html), (paths["latest_md"], text)):
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)
    return paths


class ReportJob(Job):
    """Genera l'informe diari. No escriu al lake (només el registre d'execució)."""

    name = "report"

    def execute(self, run: Run, capital: float | None = None, out_dir: str | None = None, **_: Any) -> None:
        rep = build_daily(self.ctx, capital=capital)
        out = Path(out_dir) if out_dir else self.ctx.settings.data_dir / "reports" / "daily"
        paths = write_daily(rep, out)
        run.ok()
        ref = rep.strategies[0]
        run.params["summary"] = json.dumps({"tradable": rep.tradable, "orders": len(ref.orders)})
        run.message = (f"{paths['html']} · {'operable' if rep.tradable else 'NO operable'}"
                       f" · {ref.cfg.name}: {len(ref.orders)} ordre(s)")
