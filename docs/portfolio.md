# Capas 4-5: rotación sectorial y cartera objetivo

Combina los regímenes (capa 3: *cuánto* riesgo tomar) con una rotación entre los 11 sectores
GICS (*dónde* ponerlo) y una cartera de cobertura para la parte que no va a renta variable.
Cada revisión queda en un registro de decisiones con las acciones sugeridas.

```bash
uv run fin portfolio build       # cartera objetivo + registro de decisiones (`portfolio_targets`)
uv run fin portfolio show        # cartera actual y últimos cambios sugeridos
uv run fin portfolio report      # walk-forward contra referencias, ablaciones y sensibilidad (~1 min)
```

`fin run daily` ejecuta `portfolio` después de `regimes`.

## Cómo se construye (`rot_v1`)

```
objetivo_t = E_t · sectores_t + (1 − E_t) · cobertura_t
```

| Pieza | Regla |
|---|---|
| `E_t` | exposición del régimen (`rules_v1`): 100 / 75 / 50 / 25 / 0 % |
| Puntuación de sectores | rango(momentum 12-1 / vol 1 año) + 0,25 · rango(fuerza relativa 6 m vs SPY) |
| Inclinación defensiva | en tendencia bajista se suma rango(−beta): favorece sectores de beta baja |
| Selección | top-3 de forma voraz; un sector se salta si su correlación (126 sesiones) con uno ya elegido es ≥ 0,80. Las plazas vacías se llenan con los mejores restantes |
| Pesos dentro del bloque | inversos a la volatilidad de 3 meses |
| Cobertura | correlación SPY–TLT (3 m) < 0 → IEF; si no → 50 % GLD + 50 % efectivo; sin datos → efectivo |
| Revisión | primera sesión de cada mes y cada cambio de régimen |
| Ejecución | al cierre de la revisión; rinde desde t+1. Banda de no negociación del 2 %, coste de 5 pb por unidad negociada (el efectivo no cuenta) |

Entre revisiones los pesos derivan con los precios: el motor (`backtest.simulate`) no
reequilibra cada día de forma implícita.

**Registro de decisiones** (`portfolio_targets`): cartera objetivo en cada revisión, con
`action` respecto a la revisión anterior: `enter`, `add`, `reduce` (cambio > banda), `exit`
o `hold`. Se guarda en el lake (append-only, auditado con `ingest_runs`, que conserva la
configuración completa). No se usa un SQLite aparte: el lake ya cumple esa función y se
copia con el resto de datos.

## Resultados (walk-forward 2000-01 → 2026-10)

El periodo empieza en 2000 porque el momentum sectorial necesita 12 meses de historia (los
SPDR sectoriales cotizan desde diciembre de 1998).

| Estrategia | CAGR | Vol | Sharpe | Máx. caída | Calmar | Rotación/año |
|---|---|---|---|---|---|---|
| SPY buy & hold | 8,3 % | 19,2 % | 0,41 | −55,2 % | 0,15 | 0 |
| 60/40 SPY/IEF | 6,8 % | 10,9 % | 0,47 | −32,2 % | 0,21 | 0,1 |
| Sectores equiponderados | 8,7 % | 18,0 % | 0,44 | −53,1 % | 0,16 | 0,1 |
| Momentum sectorial (sin régimen) | 7,5 % | 17,5 % | 0,39 | −45,7 % | 0,16 | 6,0 |
| Regímenes sobre SPY (semana 4) | 7,3 % | 11,1 % | **0,51** | −25,6 % | 0,29 | 1,7 |
| **Sistema completo** | 6,6 % | 11,4 % | 0,44 | **−22,6 %** | 0,29 | 8,6 |
| sin filtro de correlación | 7,1 % | 11,7 % | 0,47 | −22,6 % | 0,31 | 8,0 |
| sin cobertura (efectivo) | 5,8 % | 10,9 % | 0,39 | −18,8 % | 0,31 | 7,0 |
| sin inclinación defensiva | 6,6 % | 11,4 % | 0,44 | −22,1 % | 0,30 | 8,7 |

| Subperiodo | SPY | Regímenes sobre SPY | Sistema completo |
|---|---|---|---|
| 2000–2009 (CAGR / máx. caída) | −0,9 % / −55 % | 3,3 % / −26 % | **5,6 % / −16 %** |
| 2010–2026 | **14,2 % / −34 %** | 9,8 % / −21 % | 7,2 % / −23 % |

Episodios: crisis financiera **+5,7 %** (SPY −55 %), Covid −3,2 % (SPY −34 %), aranceles
2025 −6,7 % (SPY −19 %). Inflación 2022: −21,2 %, peor que sin cobertura (−12,9 %).

## Lectura honesta

1. **El valor viene del régimen y de la cobertura, no de elegir sectores.** Dentro de la
   renta variable, la rotación queda por debajo de tener SPY: el momentum sectorial sin
   régimen (7,5 %) pierde frente a SPY (8,3 %) y frente a los sectores equiponderados
   (8,7 %). Coincide con lo que mostró la versión anterior del sistema (momentum
   sectorial por debajo de buy & hold), ahora con datos point-in-time y costes.
2. **La puntuación ajustada por volatilidad se inclina hacia los defensivos.** XLU y XLP son
   los sectores que más tiempo están en cartera (41 % y 37 % de las sesiones). Funcionó en
   2000–2009 (el sistema completo es la mejor estrategia del periodo) y costó mucho en el
   mercado liderado por tecnología de 2010–2026.
3. **Rotación alta**: 8,6 unidades al año, frente a 1,7 de regímenes sobre SPY. A 5 pb son
   ~0,4 % anual: no explica la diferencia, pero en una cuenta real con impuestos sí pesaría.
   La banda de no negociación casi no cambia nada (0–5 %): la rotación viene de cambios de
   sector enteros, no de pequeños reajustes.
4. **El filtro de correlación** actúa en el 32 % de las revisiones, pero no mejora el
   resultado (−0,5 puntos de CAGR). La correlación media máxima entre los sectores elegidos
   es 0,68.
5. **La cobertura con IEF aporta +0,8 puntos de CAGR frente a efectivo**, pero en 2022 la
   regla llegó tarde: la correlación de 3 meses aún era negativa cuando bonos y acciones
   caían juntos.
6. **Sensibilidad** (top-k 2–5 × filtro 0,7/0,8/sin filtro × banda 0/2/5 %, 36 combinaciones):
   CAGR 6,5–7,9 %, Sharpe 0,44–0,58. `rot_v1` está en la parte baja de la rejilla; ninguna
   combinación supera con claridad a "regímenes sobre SPY". Cuantos más sectores (más cerca
   del mercado), mejor: otra señal de que la selección resta.

### Hipótesis para `rot_v2` (no implementadas: hay que validarlas fuera de esta muestra)

- **Núcleo + satélite**: E · (70 % SPY + 30 % rotación). Mantiene la exposición al mercado y
  limita el coste de equivocarse de sectores.
- Momentum **sin dividir por volatilidad** (o con la volatilidad como restricción de peso y no
  en la puntuación), para quitar el sesgo estructural hacia defensivos.
- Rotación entre **industrias** (SMH, IGV...) en lugar de sectores: la dispersión es mayor y el
  momentum suele ser más persistente.
- Cobertura con una señal de correlación más rápida (21 sesiones) o con el nivel de la
  inflación, para evitar el caso 2022.
