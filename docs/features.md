# Capa 2: features

Describe el estado del mercado cada día con un conjunto pequeño de variables
interpretables. **No toma decisiones**: la capa de regímenes (3) y la de señales (4) leen
estas features del lake.

```bash
uv run fin features build      # recalcula toda la historia (≈20 s) y la escribe al lake
uv run fin features show       # último valor + percentil histórico + ranking sectorial
uv run fin features report     # HTML: redundancias (Spearman), cobertura, series clave
```

`fin run daily` ya incluye `features` después de precios, macro y calidad.

## Reglas point-in-time

1. **El valor de la fecha t solo usa información disponible al cierre de t.** Las capas
   superiores operan como pronto en t+1.
2. **Test de truncamiento** (`tests/test_features.py`): calcular con datos hasta t debe dar
   exactamente lo mismo que con toda la historia, para todas las fechas ≤ t. Un control
   comprueba que el test detecta una ventana centrada (look-ahead deliberado).
3. **Invariancia de escala**: `adj_close` se reajusta hacia atrás con cada dividendo, así
   que ninguna feature de precios puede depender del nivel absoluto (solo ratios,
   retornos y log-diferencias). Hay un test que multiplica los precios por 3,7.
4. **Normalización expansiva**: `pct_rank` es el percentil del valor de hoy dentro de toda
   su historia *hasta hoy* (mínimo 252 observaciones). Nunca con estadísticos de la muestra
   completa.
5. **Macro**:
   - Series no revisables (diarias de FRED): disponibles la sesión siguiente a su fecha.
   - Series revisables con vintages (ALFRED): se usa la **primera publicación** de cada
     observación, en su fecha de publicación; las revisiones posteriores nunca se ven.
   - Series revisables cargadas sin clave de FRED (`latest_only`): **se excluyen** (su
     valor es el revisado de hoy). Por eso `nfci` solo aparece con `FIN_FRED_API_KEY`.
6. **Universo creciente**: XLRE (2015) y XLC (2018) entran en amplitud, correlación y
   rankings solo cuando tienen la ventana completa.

## Features de mercat (`features_daily`, conjunto `core_v1`)

`stress_sign` indica hacia dónde apunta el estrés: +1 = valor alto es estrés.

| Familia | Feature | Definición | Estrés | Desde |
|---|---|---|---|---|
| Tendencia | `mkt_dist_sma200` | SPY / SMA200 − 1 | −1 | 1993 |
| | `mkt_sma50_sma200` | SMA50 / SMA200 − 1 | −1 | 1993 |
| | `mkt_mom_12_1` | retorno 12-1 meses | −1 | 1994 |
| | `mkt_mom_3m` | retorno 3 meses | −1 | 1993 |
| | `mkt_drawdown_1y` | caída desde el máximo de 1 año | −1 | 1994 |
| Volatilidad | `mkt_rvol_1m` | vol realizada 21 sesiones (anual) | +1 | 1993 |
| | `mkt_rvol_ratio` | vol 1 mes / vol 1 año | +1 | 1994 |
| | `vix` | VIX | +1 | 1993 |
| | `vix_term` | VIX / VIX3M (> 1: curva invertida) | +1 | 2006 |
| | `vix_rvol_spread` | VIX − vol realizada (prima de volatilidad) | −1 | 1993 |
| Amplitud | `sector_breadth_200` / `_50` | % de sectores sobre su SMA200 / SMA50 | −1 | 1999 |
| | `sp500_breadth_200` | % de miembros vigentes del S&P 500 sobre su SMA200 | −1 | ver nota |
| | `ew_vs_cw_3m` | RSP vs SPY, 3 meses | −1 | 2003 |
| | `defensive_vs_cyclical_3m` | (XLP, XLU, XLV) vs (XLY, XLK, XLI, XLF), 3 meses | +1 | 1999 |
| Crédito | `hy_vs_treasury_3m` | HYG vs IEF, 3 meses | −1 | 2007 |
| | `baa_spread`, `baa_spread_chg_3m` | Moody's Baa − Treasury 10Y y su cambio | +1 | 1993 |
| | `nfci` | Chicago Fed NFCI (solo con vintages) | +1 | — |
| Tipos | `curve_10y3m` | pendiente 10Y − 3M | −1 | 1993 |
| Correlación | `sector_avg_corr_3m` | correlación media entre sectores, 63 sesiones | +1 | 1999 |
| | `sector_absorption_1y` | absorption ratio (Kritzman): varianza de los ⌈N/5⌉ primeros vectores propios, 252 sesiones | +1 | 1999 |
| | `stock_bond_corr_3m` | correlación SPY–TLT, 63 sesiones (> 0: los bonos no cubren) | +1 | 2002 |

**Nota sobre `sp500_breadth_200`**: la pertenencia al índice es point-in-time, pero
Yahoo no tiene precios de muchas empresas que salieron hace años. La columna `coverage`
(miembros con precio / miembros vigentes) baja hacia el pasado (~30 % en 1996, ~50 % en
2009) y el valor solo se publica con cobertura ≥ 50 %. Incluso así, los supervivientes
sesgan la amplitud al alza en los primeros años. Para la capa de regímenes se recomienda
`sector_breadth_200` como serie principal hasta disponer de precios de deslistados.

## Panel sectorial (`sector_features_daily`)

Para los 11 sectores GICS y los ~22 ETFs de industria: `mom_12_1`, `mom_6_1`, `mom_3m`,
`vol_3m`, `risk_adj_mom` (momentum 12-1 / vol 1 año), `rel_strength_3m`/`6m` (vs SPY),
`dist_sma200`, `beta_1y`. `xs_rank` es el percentil transversal dentro del grupo ese día.

## Redundancias (datos reales 1993–2026, semanales)

| Par | Spearman | Decisión para la capa 3 |
|---|---|---|
| `sector_breadth_200` – `sp500_breadth_200` | 0,90 | usar la sectorial (más historia, sin sesgo de supervivencia) |
| `mkt_dist_sma200` – `mkt_sma50_sma200` | 0,87 | una sola medida de tendencia |
| `mkt_rvol_1m` – `vix` | 0,83 | una sola de nivel; `vix_term` aporta información distinta (ρ 0,64 con VIX) |
| `mkt_dist_sma200` – `sector_breadth_200` | 0,83 | la amplitud confirma, no añade un eje nuevo |

Las features son muy persistentes: estas correlaciones sirven para comparar magnitudes,
no como tests estadísticos.

## Cambiar parámetros

`FeatureConfig` (`features/config.py`) usa ventanas convencionales que **no se optimizan**
sobre el backtest. Cualquier cambio exige un nombre nuevo (`core_v2`): cada conjunto
convive en el lake con su historia, y la configuración completa de cada ejecución queda
en `ingest_runs.params`.
