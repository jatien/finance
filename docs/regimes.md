# Capa 3: regímenes de mercado

Clasifica cada sesión en un régimen y traduce el régimen a una **exposición recomendada a
renta variable**. Es la parte "cuánto riesgo tomar" del sistema; la parte "dónde ponerlo"
(rotación sectorial) es la capa 4.

```bash
uv run fin regimes build       # clasifica toda la historia y la escribe en `regimes_daily`
uv run fin regimes show        # régimen actual, sesiones en él, exposición y acción
uv run fin regimes report      # validación: estadísticas, backtest, ablaciones, sensibilidad
```

`fin run daily` ejecuta `regimes` justo después de `features`.

## Reglas (`rules_v1`)

| Eje | Estados | Regla | Histéresis |
|---|---|---|---|
| Tendencia | bull / bear | `mkt_dist_sma200` (SPY / SMA200 − 1) | entra en bear < −2 %, vuelve a bull > +2 %, 3 sesiones seguidas |
| Volatilidad | calm / volatile | percentil expansivo del VIX | entra ≥ 0,75, sale < 0,60, 3 sesiones |
| Crisis | sí / no | VIX/VIX3M ≥ 1 **y** cambio del spread Baa en 3 meses en percentil ≥ 0,85 | entra al momento, sale tras 5 sesiones con VIX/VIX3M < 0,95 |

Régimen = `crisis` si hay crisis; si no, `<tendencia>_<volatilidad>`.

| Régimen | Exposición |
|---|---|
| bull_calm | 100 % |
| bull_volatile | 75 % |
| bear_calm | 50 % |
| bear_volatile | 25 % |
| crisis | 0 % |

`action` compara la exposición con la del día anterior: `add`, `reduce` o `hold`. Es la
base de las sugerencias de entrada/salida. La exposición decidida al cierre de *t* se aplica
desde *t+1*.

**Elección de features**: solo una por cada grupo redundante de la capa 2. Se usa el VIX y no
la vol realizada (ρ = 0,83), y la distancia a la SMA200 y no el cruce SMA50/200 (ρ = 0,87).
La amplitud no entra porque confirma la tendencia (ρ = 0,83) en lugar de aportar un eje nuevo.

**Antes de julio de 2006 no hay crisis**: no existe VIX3M. La primera versión usaba como
sustituto el VIX en el percentil ≥ 0,95. Con solo 3-4 años de historia, ese percentil no
mide estrés agudo: marcaba todo 1997 como crisis con el mercado subiendo un 40 %. El
sustituto está desactivado (`crisis_fallback=False`). Es una decisión por falta de datos, no
por el resultado del backtest.

## Resultados (SPY 1994–2026, coste 5 pb por unidad de rotación, efectivo = T-bills 3M)

### ¿Separan los regímenes los rendimientos?

| Régimen | % tiempo | Rend. anual | Vol anual | Sharpe | Fwd 63d > 0 |
|---|---|---|---|---|---|
| bull_calm | 56 % | 9,9 % | 12,0 % | 0,82 | 74 % |
| bull_volatile | 20 % | **21,8 %** | 18,6 % | **1,17** | 80 % |
| bear_calm | 4 % | **−12,0 %** | 15,0 % | −0,80 | 47 % |
| bear_volatile | 12 % | 11,6 % | 24,5 % | 0,47 | 56 % |
| crisis | 9 % | 14,0 % | **37,7 %** | 0,37 | 66 % |

- **La tendencia es lo que separa los rendimientos.** El único régimen con rendimiento medio
  negativo es bear_calm: caídas lentas y prolongadas.
- **La volatilidad separa el riesgo, no el rendimiento.** bull_volatile es el mejor régimen:
  la volatilidad alta dentro de una tendencia alcista suele preceder a rebotes (prima de
  riesgo de volatilidad). Reducir exposición ahí cuesta rentabilidad.
- **La crisis marca volatilidad extrema (37 %) y peores días de −11 %**, pero sus rendimientos
  futuros son positivos. Las señales de estrés agudo suelen dispararse cerca de los mínimos.
- **La diversificación desaparece en estrés.** La correlación media entre sectores pasa de
  0,47 (bull_calm) a 0,80 (crisis). Esto condiciona la capa 4: en crisis, "los 3 mejores
  sectores" son prácticamente una sola apuesta.

### Superposición SPY + efectivo

| Estrategia | CAGR | Vol | Sharpe | Máx. caída | Calmar | Rotación/año |
|---|---|---|---|---|---|---|
| SPY buy & hold | 10,9 % | 18,7 % | 0,51 | −55,2 % | 0,20 | 0 |
| Filtro SMA200 (0/100 %) | 9,6 % | 12,9 % | 0,58 | **−21,6 %** | **0,44** | 1,0 |
| **Regímenes (completo)** | 9,2 % | **11,2 %** | **0,61** | −25,6 % | 0,36 | 1,6 |
| Regímenes sin vol | 10,3 % | 13,1 % | 0,62 | −30,7 % | 0,34 | 1,2 |
| Regímenes sin crisis | 9,1 % | 12,0 % | 0,58 | −27,1 % | 0,34 | 1,1 |

Por subperiodos:
- **1994–2009**: todos los filtros ganan claramente a buy & hold (Sharpe 0,48–0,52 frente a
  0,28).
- **2010–2026**: mercado alcista casi continuo. Todos cuestan 4-5 puntos de CAGR frente al
  14,1 % de buy & hold, con Sharpe similar o algo inferior. La reducción de la caída máxima
  (de −34 % a −21 %) es lo que se compra con ese coste.

Episodios (retorno total):
- **Crisis rápidas**: el override de crisis es lo que más aporta. En Covid 2020, −8 % frente a
  −34 % (buy & hold) y −19 % (filtro SMA200). En aranceles 2025, −10 % frente a −19 %.
- **Recuperaciones**: es donde se paga el retraso. En la recuperación de 2009, +17 % frente a
  +72 %; en la de 2020, +11 % frente a +57 %.

### Sensibilidad (36 combinaciones)

Ninguna combinación de banda (0–3 %), confirmación (1–5 días) y umbral de vol (0,65–0,85)
rompe el resultado:
- CAGR entre 8,7 % y 10,0 %, Sharpe entre 0,57 y 0,69, máxima caída entre −17 % y −30 %.
- El parámetro que más mueve la caída máxima es el umbral de volatilidad.
- Sin banda ni confirmación, el sistema cambia de régimen unas 10 veces al año, frente a ~3,4
  con la configuración actual.
- La configuración por defecto está en la zona media, no en el mejor punto.

## Lectura honesta

1. La regla **no supera al filtro SMA200 simple en Calmar** (0,36 frente a 0,44). Sí lo
   supera en volatilidad, Sharpe y Sortino. El filtro de tendencia sigue siendo una línea de
   base muy exigente.
2. El eje de volatilidad, **usado para recortar exposición en tendencia alcista, resta
   rentabilidad**. Por lo que sabemos de la volatilidad, encaja mejor como herramienta de
   *dimensionamiento* (vol targeting) o para decidir *cómo* cubrirse (opciones baratas o
   caras) que como señal de salida.
3. Todo lo anterior es dentro de muestra. Cualquier `rules_v2` ajustada mirando estos
   resultados necesita validarse en un tramo que no se haya usado para diseñarla.

### Hipótesis para `rules_v2` (no implementadas, pendientes de decisión)

- bull_volatile al 100 %: la volatilidad solo cuenta en tendencia bajista.
- Sustituir el eje de vol por un escalado continuo de exposición `min(1, vol_objetivo / vol_realizada)`.
- Salida de crisis más rápida o reentrada parcial cuando la curva del VIX se normaliza, para
  pagar menos retraso en las recuperaciones.
- Cobertura con TLT o GLD en lugar de efectivo según `stock_bond_corr_3m` (capa 4/5).
