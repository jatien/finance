"""Informe HTML de validació dels règims (plotly, extra `viz`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finance.features.diagnostics import DIVERGING, GRID, INK, INK_2, SURFACE, _layout
from finance.regimes import evaluation as ev
from finance.regimes.config import REGIMES, RegimeConfig
from finance.regimes.job import EvalInputs
from finance.regimes.rules import classify

# Escala ordinal de risc: blaus (bull) → vermells (bear) → vermell fosc (crisi)
REGIME_COLORS = {
    "bull_calm": "#2a78d6",
    "bull_volatile": "#86b6ef",
    "bear_calm": "#ec8b87",
    "bear_volatile": "#d03b3b",
    "crisis": "#7a1717",
}
REGIME_LABELS = {
    "bull_calm": "Alcista · calma",
    "bull_volatile": "Alcista · volàtil",
    "bear_calm": "Baixista · calma",
    "bear_volatile": "Baixista · volàtil",
    "crisis": "Crisi",
}
# Paleta categòrica per a les estratègies (ordre fix)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def _table(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_html(index=False, border=0, classes="t", na_rep="—",
                      float_format=lambda v: f"{v:,.{digits}f}")


def _regime_bands(fig, regime: pd.Series, row: int | None = None) -> None:
    ep = ev.episodes(regime)
    for e in ep.itertuples():
        kw = {"row": row, "col": 1} if row else {}
        fig.add_vrect(x0=e.start, x1=e.end + pd.Timedelta(days=1), fillcolor=REGIME_COLORS[e.regime],
                      opacity=0.3, line_width=0, layer="below", **kw)


def build_report(E: EvalInputs, cfg: RegimeConfig, path: Path, cost_bps: float = 5.0) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    reg = classify(E.values, E.pct, cfg)
    regime = reg["regime"]
    stats = ev.regime_stats(regime, E.market)
    tm = ev.transition_matrix(regime).rename(index=REGIME_LABELS, columns=REGIME_LABELS)
    tm.columns.name = None
    trans = tm.rename_axis("des de \\ cap a").reset_index()
    corr_summary, mats = ev.conditional_sector_corr(regime, E.sectors)
    exps = ev.strategy_exposures(E.values, E.pct, cfg)
    tab_all, rets = ev.compare_strategies(exps, E.asset_ret, E.cash_ret, cost_bps)
    halves = []
    for a, b in [("1994", "2009"), ("2010", "2100")]:
        t, _ = ev.compare_strategies({k: v.loc[a:b] for k, v in exps.items()},
                                     E.asset_ret.loc[a:b], E.cash_ret.loc[a:b], cost_bps)
        halves.append((a, b, t))
    episodes = ev.episode_table(rets)
    sens = ev.sensitivity(E.values, E.pct, E.asset_ret, E.cash_ret, cfg, cost_bps)

    last = reg.dropna(subset=["regime"]).iloc[-1]
    last_date = reg.dropna(subset=["regime"]).index[-1]
    recent = ev.episodes(regime).tail(8).iloc[::-1]
    recent["regime"] = recent["regime"].map(REGIME_LABELS)
    recent["start"] = recent["start"].dt.date
    recent["end"] = recent["end"].dt.date

    # 1) Preu del mercat amb el règim de fons ------------------------------------------
    px = E.market.loc[regime.dropna().index]
    f1 = go.Figure()
    _regime_bands(f1, regime)
    f1.add_trace(go.Scatter(x=px.index, y=px.values, mode="lines", line={"color": INK, "width": 1.2},
                            name="SPY", hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"))
    for g in REGIMES:  # entrades de llegenda per als colors de fons
        f1.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=REGIME_LABELS[g],
                                marker={"size": 12, "symbol": "square", "color": REGIME_COLORS[g]}))
    f1.update_yaxes(type="log", gridcolor=GRID, title="SPY (retorn total, log)")
    f1.update_xaxes(showgrid=False)
    _layout(f1, "SPY i règim de mercat", 460)
    f1.update_layout(legend={"orientation": "h", "y": -0.12})

    # 2) Corbes de capital, caiguda i exposició ----------------------------------------
    f2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                       row_heights=[0.5, 0.3, 0.2],
                       subplot_titles=["Capital (base 1, log)", "Caiguda des del màxim", "Exposició a SPY"])
    for i, c in enumerate(rets):
        r = rets[c].fillna(0)
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        col = SERIES[i % len(SERIES)]
        width = 2.2 if c == "Règims (complet)" else 1.3
        f2.add_trace(go.Scatter(x=eq.index, y=eq.values, name=c, line={"color": col, "width": width},
                                legendgroup=c, hovertemplate="%{y:.2f}<extra>" + c + "</extra>"), 1, 1)
        f2.add_trace(go.Scatter(x=dd.index, y=dd.values, name=c, line={"color": col, "width": 1},
                                legendgroup=c, showlegend=False,
                                hovertemplate="%{y:.1%}<extra>" + c + "</extra>"), 2, 1)
    w = exps["Règims (complet)"].shift(1).dropna()
    f2.add_trace(go.Scatter(x=w.index, y=w.values, line={"color": SERIES[2], "width": 1, "shape": "hv"},
                            name="exposició", showlegend=False, fill="tozeroy",
                            fillcolor="rgba(27,175,122,0.15)",
                            hovertemplate="%{y:.0%}<extra>Règims</extra>"), 3, 1)
    f2.update_yaxes(type="log", row=1, col=1)
    f2.update_yaxes(tickformat=".0%", row=2, col=1)
    f2.update_yaxes(tickformat=".0%", range=[0, 1.05], row=3, col=1)
    f2.update_yaxes(gridcolor=GRID)
    f2.update_xaxes(showgrid=False)
    f2.update_annotations(font={"size": 12, "color": INK_2}, x=0, xanchor="left")
    _layout(f2, f"Superposició SPY + T-bills (cost {cost_bps:g} pb per unitat de rotació)", 820)
    f2.update_layout(hovermode="x unified", legend={"orientation": "h", "y": -0.06})

    # 3) Correlació sectorial: calma alcista vs crisi -----------------------------------
    pair = [g for g in ("bull_calm", "crisis") if g in mats]
    f3 = make_subplots(rows=1, cols=len(pair), subplot_titles=[
        f"{REGIME_LABELS[g]} (mitjana {corr_summary.set_index('regime').loc[g, 'corr_mitjana']:.2f})"
        for g in pair], horizontal_spacing=0.12)
    for j, g in enumerate(pair, start=1):
        m = mats[g]
        f3.add_trace(go.Heatmap(z=m.to_numpy(), x=list(m.columns), y=list(m.index), zmin=-1, zmax=1,
                                colorscale=DIVERGING, xgap=2, ygap=2, showscale=(j == len(pair)),
                                texttemplate="%{z:.2f}", textfont={"size": 9},
                                hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"), 1, j)
        f3.update_yaxes(autorange="reversed", row=1, col=j)
    f3.update_annotations(font={"size": 12, "color": INK_2})
    _layout(f3, "Correlació entre sectors segons el règim (rendiments diaris)", 480)

    def h_tab(t):
        return _table(t[["estrategia", "CAGR_%", "vol_%", "sharpe", "max_dd_%", "calmar"]])

    stats_show = stats.copy()
    stats_show["regime"] = stats_show["regime"].map(lambda g: REGIME_LABELS.get(g, g))
    corr_show = corr_summary.copy()
    corr_show["regime"] = corr_show["regime"].map(REGIME_LABELS)
    exp_map = pd.DataFrame({"règim": [REGIME_LABELS[g] for g in REGIMES],
                            "exposició a renda variable": [f"{cfg.exposure[g]:.0%}" for g in REGIMES]})
    notes = "".join(f"<li>{n}</li>" for n in E.notes)

    html = f"""<!doctype html><html lang="ca"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Règims {cfg.name}</title>
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
<h1>Capa de règims — {cfg.name}</h1>
<p class="sub">Classificador per regles sobre les features {cfg.feature_set} ·
{regime.dropna().index[0]:%Y-%m-%d} → {last_date:%Y-%m-%d}</p>
<div>
<div class="card">Règim actual<b><span class="swatch" style="background:{REGIME_COLORS[last['regime']]}"></span>{REGIME_LABELS[last['regime']]}</b></div>
<div class="card">Sessions en el règim<b>{int(last['days_in_regime'])}</b></div>
<div class="card">Exposició recomanada<b>{last['equity_exposure']:.0%}</b></div>
<div class="card">Canvis de règim per any<b>{ev.switches_per_year(regime):.1f}</b></div>
</div>
{f"<ul>{notes}</ul>" if notes else ""}
{f1.to_html(full_html=False, include_plotlyjs="cdn")}
<div class="grid2"><div><h2>Mapa d'exposició</h2>{_table(exp_map)}</div>
<div><h2>Últims episodis</h2><div class="wrap">{_table(recent[["regime", "start", "end", "sessions"]])}</div></div></div>

<h2>1. Els règims separen rendiments i risc?</h2>
<p class="note">Rendiment del dia següent i rendiments a 21 i 63 sessions condicionats al règim
d'avui (valors anualitzats en %). Un règim útil per reduir risc ha de tenir rendiments baixos o
volatilitat alta.</p>
<div class="wrap">{_table(stats_show)}</div>
<h2>Transicions entre règims</h2><div class="wrap">{_table(trans, 0)}</div>

<h2>2. La diversificació s'evapora en estrès?</h2>
<div class="wrap">{_table(corr_show, 3)}</div>
{f3.to_html(full_html=False, include_plotlyjs=False)}

<h2>3. Superposició sobre SPY: estratègies i ablacions</h2>
<p class="note">L'exposició decidida al tancament de t s'aplica al retorn de t+1; la resta rendeix
T-bills a 3 mesos (FRED DGS3MO, disponibles l'endemà). Sharpe sobre el cash.</p>
<div class="wrap">{_table(tab_all)}</div>
<div class="grid2">{"".join(f"<div><h2>{a}–{min(int(b), last_date.year)}</h2><div class='wrap'>{h_tab(t)}</div></div>" for a, b, t in halves)}</div>
{f2.to_html(full_html=False, include_plotlyjs=False)}
<h2>Episodis de referència (retorn total, %)</h2>
<p class="note">Inclou les recuperacions: és on es paga el retard de qualsevol filtre de tendència.</p>
<div class="wrap">{_table(episodes, 1)}</div>

<h2>4. Sensibilitat als paràmetres</h2>
<p class="note">No serveix per triar el millor punt. Si el resultat només aguanta en una
combinació concreta, la regla no és robusta.</p>
<div class="wrap">{_table(sens.sort_values("calmar", ascending=False))}</div>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


__all__ = ["REGIME_COLORS", "REGIME_LABELS", "build_report"]
