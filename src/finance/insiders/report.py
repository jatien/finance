"""Informe HTML de validació de les senyals d'insiders (plotly, extra `viz`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from finance.features.diagnostics import GRID, INK, INK_2, SURFACE, _layout
from finance.features.inputs import load_prices
from finance.insiders import evaluation as ev
from finance.insiders.config import EVENT_LABELS, EVENTS, InsiderConfig
from finance.insiders.job import compute

SERIES = {"cluster_buy": "#2a78d6", "ceo_cfo_buy": "#eb6834", "buy": "#1baf7a", "cluster_sell": "#e87ba4"}


@dataclass
class InsiderEval:
    cfg: InsiderConfig
    end: pd.Timestamp
    events: pd.DataFrame  # amb rendiments
    summary: pd.DataFrame
    ports: pd.DataFrame
    curves: dict[str, pd.DataFrame]
    ablation: pd.DataFrame
    breadth: pd.DataFrame
    breadth_tab: pd.DataFrame
    verdict: pd.DataFrame
    coverage: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def run_evaluation(ctx, cfg: InsiderConfig | None = None) -> InsiderEval:
    cfg = cfg or InsiderConfig()
    events, br, inp = compute(ctx, cfg)
    notes = list(inp.notes)
    sp = events[events["in_sp500"]].copy()
    t = inp.trades
    t_sp = t[t["member"].astype(bool)]
    abl_events = pd.concat([sp[sp["event"] == "buy"], ev.unfiltered_buys(t_sp, cfg),
                            ev.routine_buys(t_sp, cfg)], ignore_index=True)
    abl_events = abl_events[abl_events["signal_date"] >= pd.Timestamp(cfg.start)]
    etf = ev.sector_etfs(ctx)
    syms = sorted(set(sp["symbol"]) | set(abl_events["symbol"]) | set(etf.values()) | {ev.MARKET})
    px = load_prices(ctx, syms, start=str(cfg.start - pd.Timedelta(days=10)))
    if px.empty or ev.MARKET not in px:
        raise RuntimeError("sense preus: executa `fin ingest prices`")
    end = px.index[-1]
    sp = ev.event_returns(sp, px, etf, cfg.horizons)
    summary = ev.event_summary(sp, cfg, end)
    ports, curves = ev.portfolio_table(sp, px, cfg, end)
    abl = ev.event_returns(abl_events, px, etf, cfg.horizons)
    ablation = ev.event_summary(abl, cfg, end)
    btab = ev.breadth_predictive(br, px[ev.MARKET], cfg, end)
    ver = ev.verdict(summary, ports, ablation, btab)
    h = max(cfg.horizons)
    cov = (sp.assign(con_preu=sp[f"ret_{cfg.horizons[0]}"].notna() | (sp["signal_date"] > px.index[-h]))
           .groupby("event")["con_preu"].agg(["size", "mean"]).reset_index()
           .rename(columns={"event": "esdeveniment", "size": "N", "mean": "amb_preu"}))
    cov["amb_preu"] = (100 * cov["amb_preu"]).round(1)
    low = cov[cov["amb_preu"] < 85]
    if not low.empty:
        notes.append("cobertura de preus < 85 % en alguns esdeveniments: possible biaix de supervivència")
    return InsiderEval(cfg=cfg, end=end, events=sp, summary=summary, ports=ports, curves=curves,
                       ablation=ablation, breadth=br, breadth_tab=btab, verdict=ver,
                       coverage=cov, notes=notes)


# ---------------------------------------------------------------------------------------

def _table(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_html(index=False, border=0, classes="t", na_rep="—",
                      float_format=lambda v: f"{v:,.{digits}f}")


def _curves_fig(res: InsiderEval):
    import plotly.graph_objects as go
    fig = go.Figure()
    for e in EVENTS:
        cp = res.curves.get(e)
        if cp is None or cp.empty:
            continue
        ex = (cp["port"] - cp["market"]).where(cp["n"] > 0).fillna(0.0)
        cum = np.exp(np.log1p(ex).cumsum())
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=EVENT_LABELS[e],
                                 line={"color": SERIES[e], "width": 1.6}))
    fig.add_vline(x=pd.Timestamp(res.cfg.design_end), line_dash="dot", line_color=INK_2)
    fig.update_yaxes(type="log", gridcolor=GRID, title="valor relatiu al SPY")
    fig.update_xaxes(gridcolor=GRID)
    return _layout(fig, "Carteres en temps de calendari: excés acumulat sobre el SPY", 420)


def _breadth_fig(res: InsiderEval, px_spy: pd.Series):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    m = res.breadth[res.breadth["scope"] == "market"].set_index("date")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=px_spy.index, y=px_spy.values, name="SPY", line={"color": INK, "width": 1.2}),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=m.index, y=m["buy_ratio"], name="Ràtio de compres (63 sessions)",
                             line={"color": "#2a78d6", "width": 1.2}), row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(row=2, col=1, gridcolor=GRID, tickformat=".0%")
    fig.update_xaxes(gridcolor=GRID)
    return _layout(fig, "Amplitud d'insiders del S&P 500 i mercat", 460)


def build_report(ctx, res: InsiderEval, path: Path) -> Path:
    cfg = res.cfg
    verdict = res.verdict.copy()
    verdict["passa"] = verdict["passa"].map({True: "sí", False: "no", None: "—"})
    s = res.summary
    main = s[s["horitzo"] == 63].copy()
    main["esdeveniment"] = main["esdeveniment"].map(EVENT_LABELS).fillna(main["esdeveniment"])
    allh = s[s["finestra"] == "tot"].pivot_table(index="esdeveniment", columns="horitzo",
                                                 values="mitjana_%").reset_index()
    allh.columns = [str(c) if c == "esdeveniment" else f"{c} sessions" for c in allh.columns]
    ports = res.ports.copy()
    ports["esdeveniment"] = ports["esdeveniment"].map(EVENT_LABELS).fillna(ports["esdeveniment"])
    abl = res.ablation[res.ablation["horitzo"] == 63]
    spy = load_prices(ctx, [ev.MARKET], start=str(cfg.start))[ev.MARKET]
    recent = res.events[res.events["signal_date"] > res.end - pd.Timedelta(days=90)]
    recent = recent[recent["event"] != "buy"].sort_values("signal_date", ascending=False)
    rec_cols = ["signal_date", "symbol", "sector", "event", "n_insiders", "value", "value_adv", "names"]
    recent = recent[rec_cols].assign(signal_date=recent["signal_date"].dt.date, value=recent["value"] / 1e6)
    recent = recent.rename(columns={"value": "import_M$", "value_adv": "import/ADV"})
    last_b = res.breadth.dropna(subset=["buy_ratio"]).sort_values("date").groupby("scope").tail(1)
    last_b = last_b.sort_values("buy_ratio", ascending=False)[["scope", "date", "buyers", "sellers", "buy_ratio", "ratio_pct"]]
    last_b["date"] = pd.to_datetime(last_b["date"]).dt.date
    notes = "".join(f"<li>{n}</li>" for n in res.notes)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Insiders — {cfg.name}</title>
<style>
body{{background:{SURFACE};color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.45}}
h1{{font-size:22px;margin-bottom:4px}} h2{{font-size:17px;margin-top:36px}} h3{{font-size:14px}}
p.sub{{color:{INK_2};margin-top:0}} .note{{color:{INK_2};font-size:13px;max-width:900px}}
table.t{{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}}
table.t th{{text-align:left;color:{INK_2};font-weight:600;border-bottom:1px solid #c3c2b7;padding:6px 8px}}
table.t td{{border-bottom:1px solid {GRID};padding:5px 8px}} .wrap{{overflow-x:auto}}
</style></head><body>
<h1>Insiders — {cfg.name}</h1>
<p class="sub">Form 4 de la SEC · esdeveniments del S&P 500 vigent · {cfg.start} → {res.end:%Y-%m-%d} ·
disseny fins al {cfg.design_end}, comprovació després</p>
{f"<ul class='note'>{notes}</ul>" if notes else ""}
<h2>Veredicte (criteris pre-registrats a docs/insiders.md)</h2>
<div class="wrap">{_table(verdict)}</div>
<h2>1. Estudi d'esdeveniments: excés sobre l'ETF del sector a 63 sessions</h2>
<p class="note">Entrada al tancament de la primera sessió posterior a la presentació. t agrupat per mes.</p>
<div class="wrap">{_table(main.drop(columns="horitzo"))}</div>
<h3>Excés mitjà (%) per horitzó, tot el període</h3><div class="wrap">{_table(allh)}</div>
<h2>2. Carteres en temps de calendari (mantenir {cfg.hold_sessions} sessions)</h2>
{_curves_fig(res).to_html(full_html=False, include_plotlyjs="cdn")}
<div class="wrap">{_table(ports)}</div>
<h2>3. Ablació dels filtres (compres, 63 sessions)</h2>
<p class="note">buy = filtrades; buy_sense_filtres = inclou plans 10b5-1 i rutinaris; buy_rutinaris = només rutinaris.</p>
<div class="wrap">{_table(abl.drop(columns="horitzo"))}</div>
<h2>4. Amplitud agregada i mercat</h2>
{_breadth_fig(res, spy).to_html(full_html=False, include_plotlyjs=False)}
<div class="wrap">{_table(res.breadth_tab)}</div>
<h2>Estat actual</h2>
<h3>Amplitud per sector (últimes 63 sessions)</h3><div class="wrap">{_table(last_b)}</div>
<h3>Esdeveniments dels últims 90 dies (sense 'buy' simple)</h3><div class="wrap">{_table(recent)}</div>
<h3>Cobertura de preus</h3><div class="wrap">{_table(res.coverage, 1)}</div>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
