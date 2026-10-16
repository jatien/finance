"""Informe HTML de la rotació sectorial i la cartera objectiu (plotly, extra `viz`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finance.features.diagnostics import GRID, INK, INK_2, SURFACE, _layout
from finance.portfolio import evaluation as ev
from finance.portfolio.config import CASH, PortfolioConfig
from finance.portfolio.construct import PortfolioInputs
from finance.portfolio.job import decision_log
from finance.regimes.report import REGIME_COLORS, REGIME_LABELS

# Paleta categòrica (ordre fix): estratègies del gràfic de capital
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CURVE_STRATEGIES = ["SPY buy & hold", "60/40 SPY/IEF", "Momentum sectorial (sense règim)",
                    "Règims sobre SPY", ev.MAIN]
# Blocs de la cartera per al gràfic d'assignació (5 categories, ordre fix)
SLEEVE_COLORS = {"Sectors cíclics": "#2a78d6", "Sectors defensius": "#1baf7a",
                 "Bons": "#6250d6", "Or": "#eda100", "Cash": "#898781"}
ACTION_LABELS = {"enter": "entrar", "add": "augmentar", "reduce": "reduir", "exit": "sortir", "hold": "mantenir"}


def _table(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_html(index=False, border=0, classes="t", na_rep="—",
                      float_format=lambda v: f"{v:,.{digits}f}")


def _sleeves(w: pd.DataFrame, cfg: PortfolioConfig, sectors: list[str]) -> pd.DataFrame:
    defensive = [s for s in cfg.defensive if s in w]
    cyclical = [s for s in sectors if s in w and s not in defensive]
    return pd.DataFrame({
        "Sectors cíclics": w[cyclical].sum(axis=1),
        "Sectors defensius": w[defensive].sum(axis=1),
        "Bons": w.get(cfg.bond_hedge, 0.0),
        "Or": w.get(cfg.gold, 0.0),
        "Cash": w.get(CASH, 0.0),
    })


def build_report(inp: PortfolioInputs, cfg: PortfolioConfig, path: Path, *, with_sensitivity: bool = True) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    targets = ev.strategy_targets(inp, cfg)
    sims, start = ev.run_all(inp, cfg, targets)
    main_t = targets[ev.MAIN]
    main = sims[ev.MAIN]
    end = main.returns.index[-1]
    full = ev.metrics_table(sims, inp, cfg, start)
    halves = [(a, b, ev.metrics_table(sims, inp, cfg, pd.Timestamp(a), pd.Timestamp(b)))
              for a, b in (("2000-01-01", "2009-12-31"), ("2010-01-01", f"{end:%Y-%m-%d}"))]
    episodes = ev.episodes(sims, start)
    reviews = ev.review_summary(main_t.log, start)
    hedge_share = ev.hedge_time_share(main_t.log, inp.index, start)
    holdings = ev.holdings_summary(main, inp, start)
    annual = ev.annual_returns(sims, CURVE_STRATEGIES, start).round(1).reset_index(names="any")
    sens = ev.sensitivity(inp, cfg) if with_sensitivity else pd.DataFrame()

    # Cartera objectiu actual i últimes decisions --------------------------------------
    log = decision_log(main_t, cfg)
    last_date = log["date"].max()
    current = log[(log["date"] == last_date) & (log["target_weight"] > 0)].sort_values("target_weight", ascending=False)
    cur_show = pd.DataFrame({
        "actiu": current["symbol"], "bloc": current["sleeve"],
        "pes objectiu": current["target_weight"].map("{:.1%}".format),
        "acció": current["action"].map(ACTION_LABELS)})
    changes = log[(log["action"] != "hold")].copy()
    recent_dates = sorted(changes["date"].unique())[-6:][::-1]
    recent = changes[changes["date"].isin(recent_dates)].sort_values(["date", "change"], ascending=[False, False])
    recent_show = pd.DataFrame({
        "data": recent["date"].dt.date, "règim": recent["regime"].map(lambda g: REGIME_LABELS.get(g, g)),
        "actiu": recent["symbol"], "acció": recent["action"].map(ACTION_LABELS),
        "de": recent["prev_weight"].map("{:.0%}".format), "a": recent["target_weight"].map("{:.0%}".format)})
    reg_now = inp.regime["regime"].dropna().iloc[-1]
    expo_now = float(inp.regime["equity_exposure"].dropna().iloc[-1])
    hedge_now = main_t.log["hedge"][main_t.log["hedge"] != "sense canvis"].iloc[-1]

    # 1) Capital i caiguda --------------------------------------------------------------
    f1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.65, 0.35],
                       subplot_titles=["Capital (base 1, escala log)", "Caiguda des del màxim"])
    rets = ev.returns_frame(sims, start)
    for i, name in enumerate(CURVE_STRATEGIES):
        r = rets[name].fillna(0)
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        col = SERIES[i]
        width = 2.4 if name == ev.MAIN else 1.3
        f1.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name, legendgroup=name,
                                line={"color": col, "width": width},
                                hovertemplate="%{y:.2f}<extra>" + name + "</extra>"), 1, 1)
        f1.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name, legendgroup=name, showlegend=False,
                                line={"color": col, "width": 1},
                                hovertemplate="%{y:.1%}<extra>" + name + "</extra>"), 2, 1)
    f1.update_yaxes(type="log", row=1, col=1)
    f1.update_yaxes(tickformat=".0%", row=2, col=1)
    f1.update_yaxes(gridcolor=GRID)
    f1.update_xaxes(showgrid=False)
    f1.update_annotations(font={"size": 12, "color": INK_2}, x=0, xanchor="left")
    _layout(f1, f"Walk-forward {start:%Y-%m} → {end:%Y-%m} (cost {cfg.cost_bps:g} pb, banda {cfg.band:.0%})", 700)
    f1.update_layout(hovermode="x unified", legend={"orientation": "h", "y": -0.08})

    # 2) Assignació del sistema complet per blocs -----------------------------------------
    sl = _sleeves(main.weights.loc[start:], cfg, inp.sectors).resample("W-FRI").last()
    f2 = go.Figure()
    for name, colr in SLEEVE_COLORS.items():
        f2.add_trace(go.Scatter(x=sl.index, y=sl[name], name=name, stackgroup="one", mode="lines",
                                line={"width": 0.5, "color": colr}, fillcolor=colr,
                                hovertemplate="%{y:.0%}<extra>" + name + "</extra>"))
    f2.update_yaxes(tickformat=".0%", range=[0, 1], gridcolor=GRID)
    f2.update_xaxes(showgrid=False)
    _layout(f2, "Assignació del sistema complet (pesos en cartera, setmanal)", 380)
    f2.update_layout(hovermode="x unified", legend={"orientation": "h", "y": -0.15})

    def mt(t: pd.DataFrame) -> str:
        return _table(t[["estrategia", "CAGR_%", "vol_%", "sharpe", "max_dd_%", "calmar"]])

    hedge_tab = hedge_share.rename_axis("cobertura").reset_index(name="% del temps")
    rev_tab = pd.DataFrame([{k.replace("_", " "): v for k, v in reviews.items()}])
    notes = "".join(f"<li>{n}</li>" for n in inp.notes)
    sens_html = ""
    if not sens.empty:
        sens_html = f"""<h2>5. Sensibilitat</h2>
<p class="note">Graella sobre el nombre de sectors, el llindar de correlació (1,01 = sense filtre) i la
banda de no-negociació. Serveix per veure si el resultat depèn d'un punt concret, no per triar-lo.</p>
<div class="wrap">{_table(sens.sort_values(["top_k", "max_pair_corr", "band"]))}</div>"""

    html = f"""<!doctype html><html lang="ca"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cartera {cfg.name}</title>
<style>
body{{background:{SURFACE};color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.45}}
h1{{font-size:22px;margin-bottom:4px}} h2{{font-size:17px;margin-top:36px}}
p.sub{{color:{INK_2};margin-top:0}} .note{{color:{INK_2};font-size:13px;max-width:900px}}
table.t{{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}}
table.t th{{text-align:left;color:{INK_2};font-weight:600;border-bottom:1px solid #c3c2b7;padding:6px 8px}}
table.t td{{border-bottom:1px solid {GRID};padding:5px 8px}}
.wrap{{overflow-x:auto}} .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
@media (max-width:800px){{.grid2{{grid-template-columns:1fr}}}}
.card{{display:inline-block;border:1px solid {GRID};border-radius:8px;padding:12px 16px;margin:4px 12px 4px 0}}
.card b{{font-size:20px;display:block}} .swatch{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
</style></head><body>
<h1>Rotació sectorial i cartera — {cfg.name}</h1>
<p class="sub">Règims {cfg.regime_set} × top-{cfg.top_k} sectors + cobertura · última decisió {last_date:%Y-%m-%d}</p>
<div>
<div class="card">Règim<b><span class="swatch" style="background:{REGIME_COLORS[reg_now]}"></span>{REGIME_LABELS[reg_now]}</b></div>
<div class="card">Renda variable<b>{expo_now:.0%}</b></div>
<div class="card">Cobertura<b>{hedge_now if expo_now < 1 else "— (no cal)"}</b></div>
</div>
{f"<ul>{notes}</ul>" if notes else ""}
<div class="grid2"><div><h2>Cartera objectiu actual</h2><div class="wrap">{_table(cur_show)}</div></div>
<div><h2>Últims canvis suggerits</h2><div class="wrap">{_table(recent_show)}</div></div></div>

<h2>1. Comparació walk-forward</h2>
<p class="note">Mateix motor per a totes: pesos que deriven entre revisions, revisió a l'inici de mes i
a cada canvi de règim, banda de no-negociació de {cfg.band:.0%}, {cfg.cost_bps:g} pb per unitat negociada.
La cartera decidida al tancament de t guanya el retorn de t+1. Cash = T-bills 3M. Abans de juliol de
2002 (IEF) el 60/40 té la part de bons en cash.</p>
<div class="wrap">{_table(full)}</div>
<div class="grid2">{"".join(f"<div><h2>{a[:4]}–{b[:4]}</h2><div class='wrap'>{mt(t)}</div></div>" for a, b, t in halves)}</div>
{f1.to_html(full_html=False, include_plotlyjs="cdn")}

<h2>2. Episodis de referència (retorn total, %)</h2>
<div class="wrap">{_table(episodes, 1)}</div>

<h2>3. Què fa el sistema</h2>
{f2.to_html(full_html=False, include_plotlyjs=False)}
<div class="grid2"><div><h2>Temps en cada actiu</h2><div class="wrap">{_table(holdings, 1)}</div></div>
<div><h2>Revisions i filtre de correlació</h2><div class="wrap">{_table(rev_tab)}</div>
<h2>Tipus de cobertura</h2><div class="wrap">{_table(hedge_tab, 1)}</div></div></div>

<h2>4. Rendiment anual (%)</h2>
<div class="wrap">{_table(annual, 1)}</div>
{sens_html}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
