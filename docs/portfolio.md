# Capas 4-5: rotación sectorial y cartera objetivo

Combina los regímenes (capa 3: *cuánto* riesgo tomar) con una rotación entre los 11 sectores
GICS (*dónde* ponerlo) y una cartera de cobertura para la parte que no va a renta variable.
Cada revisión queda en un registro de decisiones con las acciones sugeridas.

```bash
uv run fin portfolio build       # estrategias activas (rot_v1, rot_v2) → registro de decisiones (`portfolio_targets`)
uv run fin portfolio show        # cartera actual y últimos cambios (por defecto rot_v1; --set rot_v2)
uv run fin portfolio report      # walk-forward contra referencias, ablaciones y sensibilidad (~1 min)
```

`fin run daily` ejecuta `portfolio` después de `regimes`. Escribe todas las estrategias de
`ACTIVE` (`portfolio/config.py`) con su `strategy_set`: es el paper trading en paralelo. Los
bloques de `sleeve` son `renda_variable`, `nucli` (SPY en `rot_v2`), `cobertura` y `cash`.

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

## `rot_v2`: construcción condicionada al régimen (semana 6)

### Pre-registro

Escrito y confirmado en git **antes** de ejecutar ningún backtest de `rot_v2`. Las reglas
salen de las hipótesis anteriores (que sí se formularon mirando 2000–2026 con `rot_v1`), así
que ninguna ventana histórica es una validación limpia; la única fuera de muestra real es el
paper trading desde el 16-oct-2026. Lo que este protocolo sí evita es ajustar `rot_v2` a sus
propios resultados.

En `rot_v1`, el régimen solo decide *cuánto* riesgo (E). En `rot_v2` también decide *cómo* se
construye el bloque de renta variable:

```
objetivo_t = E_t · [c_r · SPY + (1 − c_r) · satélite_r] + (1 − E_t) · cobertura_t
```

| Tendencia | Núcleo SPY `c` | Satélite (top-3, pesos inversos a la vol) | Motivo |
|---|---|---|---|
| alcista | 70 % | momentum 12-1 **sin dividir por vol** + 0,25 · fuerza relativa 6 m; solo sectores con momentum 12-1 > 0 (las plazas vacías van a SPY) | La selección de `rot_v1` restaba frente a SPY y se inclinaba a defensivos en mercados alcistas |
| bajista | 50 % | puntuación defensiva de `rot_v1` (riesgo + beta baja) | La inclinación defensiva fue lo que funcionó en 2000–2009 |
| crisis | — | E = 0: todo en cobertura | Sin cambios |

- **E** sigue siendo `rules_v1` (100/75/50/25/0 %). Cambiar la exposición es trabajo de la
  capa 3 (`rules_v2`); aquí se aísla el efecto de la construcción.
- **Cobertura**: IEF solo si la correlación SPY–TLT es negativa a 3 meses **y** a 1 mes (21
  sesiones); si no, 50 % oro + 50 % efectivo. Motivo: el caso de 2022.
- Sin cambios: revisión mensual y en cada cambio de régimen, filtro de correlación 0,80,
  banda del 2 %, 5 pb.

**Protocolo de evaluación** (fijado de antemano):

1. Una sola ejecución con la configuración `ROT_V2` de `portfolio/config.py`. Cualquier
   cambio posterior es `rot_v3`.
2. Ventanas: completa (2000-01 → 2026-10), *diseño* 2000–2015 y *comprobación* 2016–2026.
3. Ablaciones que aíslan cada pieza: sin núcleo, sin filtro de momentum absoluto,
   puntuación de `rot_v1` en alcista, cobertura de `rot_v1` y **sin condicionar** (la regla
   alcista en todos los regímenes). La última responde a la pregunta de la semana: ¿aporta
   algo que la construcción dependa del régimen?
4. `rot_v2` se considera mejor que `rot_v1` solo si mejora el Sharpe **y** no empeora la caída
   máxima en más de 3 puntos, en las dos ventanas. Si no, se registra como resultado
   negativo y las dos estrategias siguen en paper trading.
5. Las dos (`rot_v1` y `rot_v2`) se escriben cada día en `portfolio_targets` desde ya.

### Resultados (una sola ejecución, 16-oct-2026)

| Estrategia | Completa: CAGR / Sharpe / máx. caída | 2000–2015 | 2016–2026 | Rotación/año |
|---|---|---|---|---|
| SPY buy & hold | 8,3 % / 0,41 / −55,2 % | 4,1 % / 0,21 / −55,2 % | 15,0 % / 0,75 / −33,7 % | 0 |
| Regímenes sobre SPY | 7,3 % / 0,51 / −25,6 % | 4,7 % / 0,32 / −25,6 % | 11,4 % / 0,79 / −21,4 % | 1,7 |
| `rot_v1` | 6,6 % / 0,44 / **−22,6 %** | 6,1 % / 0,44 / **−15,7 %** | 7,3 % / 0,45 / −22,6 % | 8,6 |
| **`rot_v2`** | **8,2 % / 0,57** / −26,0 % | 6,3 % / 0,45 / −21,4 % | 10,9 % / 0,75 / −26,0 % | **5,2** |
| v2 sin núcleo | 7,8 % / 0,52 / −19,8 % | 7,1 % / 0,50 / −15,7 % | 8,9 % / 0,56 / −19,8 % | 9,1 |
| v2 sin filtro absoluto | 8,0 % / 0,56 / −26,0 % | 6,2 % / 0,44 / −21,4 % | 10,7 % / 0,73 / −26,0 % | 5,1 |
| v2 con puntuación v1 en alcista | 8,0 % / 0,56 / −26,8 % | 6,2 % / 0,44 / −21,4 % | 10,7 % / 0,73 / −26,8 % | 5,0 |
| v2 con cobertura v1 | 8,2 % / 0,57 / −26,0 % | 6,3 % / 0,45 / −21,4 % | 10,9 % / 0,75 / −26,0 % | 5,1 |
| **v2 sin condicionar** | 8,1 % / 0,57 / −26,2 % | 6,3 % / 0,44 / −22,9 % | 10,9 % / 0,74 / −26,2 % | 5,1 |

Episodios: punto com −20,3 % (`rot_v1` −13,6 %), crisis financiera +1,2 % (+5,7 %), recuperación
2020 +12,8 % (+7,5 %), inflación 2022 −25,4 % (−21,2 %).

**Veredicto del criterio pre-registrado: no pasa.** El Sharpe mejora en las dos ventanas
(0,44 → 0,45 y 0,45 → 0,75), pero la caída máxima empeora 5,7 puntos en 2000–2015 y 3,4 en
2016–2026. `rot_v1` sigue como referencia; las dos se registran cada día.

### Lectura honesta

1. **Condicionar la construcción al régimen no aporta nada.** "Sin condicionar" (la regla
   alcista también en tendencia bajista) da prácticamente lo mismo que `rot_v2` (Sharpe 0,57
   en ambas). Era la pregunta de la semana y la respuesta es no: el régimen sirve para decidir
   *cuánto* riesgo (E), no *cómo* repartirlo. Encaja con que en estrés las correlaciones
   sectoriales suben a 0,80: en bajista y crisis hay poco que elegir.
2. **La mejora de `rot_v2` viene casi entera del núcleo de SPY**, es decir, de hacer *menos*
   selección de sectores. Es la tercera vez que aparece lo mismo (v. lectura de `rot_v1`).
   El coste es la caída: con un 70 % en SPY, el sistema hereda más del mercado en las caídas
   rápidas que arrancan en tendencia alcista (2000, 2022).
3. **El filtro de momentum absoluto y el cambio de puntuación aportan poco** (+0,1–0,2 puntos
   de CAGR cada uno), aunque la puntuación sin dividir por la volatilidad sí elimina el sesgo
   defensivo que tenía `rot_v1` en 2010–2026.
4. **La confirmación rápida de la cobertura no cambia nada, y el diagnóstico explica por qué.**
   Entre enero y marzo de 2022 la correlación SPY–TLT era negativa a 1 y a 3 meses (−0,05 a
   −0,33) mientras los bonos caían con fuerza: el 23-feb el sistema entra en crisis con un
   100 % en IEF. El problema no era la velocidad de la correlación sino la señal: el signo
   de la correlación no dice si los bonos van a subir. Hace falta una condición sobre el
   propio activo de cobertura.
5. **"Sin núcleo" es la variante con mejor Calmar y menor caída** (−19,8 %). Se ha visto
   mirando los resultados, así que solo es una hipótesis.

### Hipótesis para `rot_v3` (no implementadas)

- Cobertura con **filtro de tendencia del propio bono** (IEF sobre su SMA200 o momentum de 6 m
  > 0) en lugar de, o además de, la correlación.
- Núcleo **dependiente de la volatilidad** en lugar del régimen: más satélite cuando la
  dispersión entre sectores es alta, más SPY cuando es baja.
- Probar "sin núcleo" con la puntuación de tendencia en una muestra que no se haya usado:
  industrias (SMH, IGV...) en lugar de sectores, o el tramo de paper trading.

## Bandas de no negociación (semana 7)

### Diagnóstico previo (sobre las estrategias existentes, sin reglas nuevas)

| | Rotación/año | Altas/bajas de activos | Reajustes de peso | En cambios de régimen | Operaciones/año |
|---|---|---|---|---|---|
| `rot_v1` | 8,6 | 7,1 (83 %) | 1,5 | 3,6 | 11,1 |
| `rot_v2` | 5,2 | 3,8 (74 %) | 1,3 | 3,3 | 9,8 |

En la mitad de las revisiones mensuales entra al menos un sector nuevo (sale otro). La banda
del 2 % solo actúa sobre los reajustes de peso, la parte pequeña: por eso no cambiaba nada.

### Pre-registro

Escrito y confirmado en git **antes** de ejecutar ningún backtest de las versiones con
bandas. Una banda por cada fuente de rotación (`BANDS_V1` en `portfolio/config.py`):

| Banda | Regla | Valor | Ataca |
|---|---|---|---|
| De rango | un sector en cartera se queda mientras esté en el top-(k + 2): "compra el top-3, vende cuando sale del top-5" | `rank_buffer = 2` | altas/bajas de sectores |
| De tracking error | se reequilibra solo si `sqrt((w − w*)ᵀ Σ (w − w*) · 252) > 1 %`, con Σ de 63 sesiones (Palomar, sec. 6.1.5). Mide la distancia en riesgo, no en peso | `te_band = 1 %` | reajustes pequeños y cambios entre activos casi equivalentes |
| Comprobación diaria | el disparador anterior se evalúa cada sesión contra el último objetivo, no solo en las revisiones | `check_daily = True` | deriva de precios entre revisiones |
| Histéresis de cobertura | se entra en bonos con correlación < −0,10 y se sale con > +0,10 | `hedge_hysteresis = 0,10` | cambios bonos ↔ oro+efectivo |

Se aplican a las dos estrategias activas: `rot_v1b = rot_v1 + bandas` y
`rot_v2b = rot_v2 + bandas`. La exposición E y la puntuación no cambian.

Valores convencionales, no optimizados: top-5 para mantener un top-3 es la regla habitual de
los "hold buffers" de momentum; 1 % de TE es del orden de lo que mueve un reajuste del 5 % de
un sector y muy por debajo de lo que mueve un cambio de sector o de exposición (~5 %).

**Criterio** (para cada pareja `rot_vX → rot_vXb`):

1. La rotación anual (muestra completa 2000-01 → 2026-10) baja al menos un **25 %**, y
2. en las ventanas de diseño (2000–2015) y comprobación (2016–2026), el Sharpe no baja más de
   **0,03** y la caída máxima no empeora más de **2 puntos**.

Si pasa, la versión con bandas sustituye a la original en el registro diario de decisiones
(la original queda en el código como referencia). Si no, se registra el resultado y no cambia
nada. Ablaciones: cada banda por separado sobre `rot_v1` y `rot_v2`. Sensibilidad (robustez,
no selección): `rank_buffer` 0–3 × `te_band` 0,5/1/2 %.
