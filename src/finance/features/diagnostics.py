"""Diagnòstic del conjunt de features: cobertura, redundància i informe HTML.

La matriu de correlació (Spearman) entre features serveix per detectar redundàncies: dues
features que mesuren el mateix (p. ex. VIX i vol realitzada) no han d'entrar totes dues a
les regles de règim, perquè comptarien el mateix senyal dues vegades. Spearman i no
Pearson: les features tenen cues gruixudes i relacions monòtones no lineals.

Advertència: les features són molt persistents (autocorrelació alta), així que la
significació estadística d'aquestes correlacions és molt menor del que sembla pel nombre
de sessions. Serveixen per comparar magnituds, no per fer tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from finance.features.market import CATALOG


def coverage(features: pd.DataFrame, pct: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for c in features:
        s = features[c].dropna()
        info = CATALOG.get(c)
        rows.append({
            "feature": c,
            "family": info.family if info else "",
            "first_date": s.index.min().date() if len(s) else None,
            "last_date": s.index.max().date() if len(s) else None,
            "n_obs": int(len(s)),
            "last_value": float(s.iloc[-1]) if len(s) else np.nan,
            "last_pct": float(pct[c].dropna().iloc[-1]) if pct is not None and pct[c].notna().any() else np.nan,
            "description": info.description if info else "",
        })
    return pd.DataFrame(rows)


def spearman_matrix(features: pd.DataFrame, min_periods: int = 504,
                    sample: str | None = "W-FRI") -> pd.DataFrame:
    """Correlació de Spearman per parells (observacions comunes de cada parell).

    `sample`: remostreig (per defecte setmanal) per reduir el pes de l'autocorrelació
    diària; no canvia gaire l'estimació però fa explícit que hi ha menys informació.
    """
    x = features.resample(sample).last() if sample else features
    mp = min_periods // 5 if sample == "W-FRI" else min_periods
    return x.corr(method="spearman", min_periods=mp)


def redundant_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    cols = list(corr.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "spearman": round(float(r), 3)})
    df = pd.DataFrame(rows, columns=["feature_a", "feature_b", "spearman"])
    return df.reindex(df["spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Informe HTML (plotly, opcional: `uv sync --extra viz`)
# ---------------------------------------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"
# Divergent blau <-> vermell amb punt mig gris (correlació negativa / positiva)
DIVERGING = [[0.0, "#104281"], [0.25, "#6da7ec"], [0.5, "#f0efec"],
             [0.75, "#ec8b87"], [1.0, "#a3201f"]]

# Sèries mostrades en el panell temporal (una per família, les més llegibles)
KEY_SERIES = ["mkt_dist_sma200", "vix", "vix_term", "sector_breadth_200",
              "sp500_breadth_200", "baa_spread", "sector_avg_corr_3m", "stock_bond_corr_3m"]


def _layout(fig, title: str, height: int):
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": INK}, "x": 0, "xanchor": "left"},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK_2, "size": 12},
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        hoverlabel={"bgcolor": "white", "font": {"color": INK}},
    )
    return fig


def build_report(features: pd.DataFrame, pct: pd.DataFrame, path: Path, *,
                 feature_set: str, notes: list[str] | None = None,
                 threshold: float = 0.8) -> Path:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("cal plotly: `uv sync --extra viz`") from e

    corr = spearman_matrix(features)
    pairs = redundant_pairs(corr, threshold)
    cov = coverage(features, pct)

    # Heatmap de correlació (ordre del catàleg = agrupat per família)
    labels = list(corr.columns)
    z = corr.to_numpy()
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(z)))
    heat = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
        text=text, texttemplate="%{text}", textfont={"size": 9},
        hovertemplate="%{y} × %{x}<br>Spearman: %{z:.2f}<extra></extra>",
        colorbar={"title": {"text": "ρ"}, "thickness": 12},
    ))
    heat.update_yaxes(autorange="reversed", tickfont={"size": 10})
    heat.update_xaxes(tickangle=-45, tickfont={"size": 10})
    _layout(heat, "Correlació de Spearman entre features (dades setmanals)", 820)

    # Petits múltiples: sèries clau en brut
    keys = [k for k in KEY_SERIES if k in features]
    ts = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=0.025,
                       subplot_titles=[f"{k} — {CATALOG[k].description}" for k in keys])
    for i, k in enumerate(keys, start=1):
        s = features[k].dropna().resample("W-FRI").last().dropna()  # setmanal: fitxer lleuger
        ts.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=k,
                                line={"color": SERIES_1, "width": 1.5},
                                hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>" + k + "</extra>"),
                     row=i, col=1)
    ts.update_xaxes(showgrid=False, linecolor=GRID)
    ts.update_yaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", tickfont={"size": 10})
    ts.update_annotations(font={"size": 11, "color": INK_2}, x=0, xanchor="left")
    _layout(ts, "Sèries clau (valors en brut)", 170 * len(keys) + 80)
    ts.update_layout(showlegend=False, hovermode="x unified")

    def table(df: pd.DataFrame) -> str:
        return df.to_html(index=False, border=0, classes="t", float_format=lambda v: f"{v:.3f}",
                          na_rep="—")

    notes_html = "".join(f"<li>{n}</li>" for n in (notes or [])) or "<li>Cap</li>"
    html = f"""<!doctype html><html lang="ca"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Features {feature_set}</title>
<style>
body{{background:{SURFACE};color:{INK};font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.45}}
h1{{font-size:22px;margin-bottom:4px}} h2{{font-size:17px;margin-top:32px}}
p.sub{{color:{INK_2};margin-top:0}}
table.t{{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}}
table.t th{{text-align:left;color:{INK_2};font-weight:600;border-bottom:1px solid #c3c2b7;padding:6px 8px}}
table.t td{{border-bottom:1px solid {GRID};padding:5px 8px}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>Capa de features — {feature_set}</h1>
<p class="sub">{len(features.columns)} features de mercat · {features.index.min():%Y-%m-%d} → {features.index.max():%Y-%m-%d}</p>
<h2>Avisos de la càrrega</h2><ul>{notes_html}</ul>
<h2>Parells redundants (|ρ| ≥ {threshold})</h2>
<div class="wrap">{table(pairs) if len(pairs) else "<p>Cap parell per sobre del llindar.</p>"}</div>
{heat.to_html(full_html=False, include_plotlyjs="cdn")}
<h2>Cobertura i últim valor</h2>
<div class="wrap">{table(cov)}</div>
{ts.to_html(full_html=False, include_plotlyjs=False)}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
