# Arquitectura

## Visión general

El sistema se construye por capas. Cada capa solo depende de las de abajo y se comunica
con ellas a través del lake (datos versionados) y del registro de ejecuciones.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 6. Ejecución / alertas     órdenes sugeridas: entrar, añadir, reducir, cubrir │  (futuro)
│ 5. Cartera                 asignación por régimen, límites de riesgo, opciones │  (futuro)
│ 4. Señales                 rotación sectorial, momentum, timing de entrada     │  (futuro)
│ 3. Regímenes               bull/bear, volatilidad, estrés de crédito           │  (futuro)
│ 2. Features                retornos, vol realizada/implícita, skew, breadth... │  (futuro)
├────────────────────────────────────────────────────────────────────────────┤
│ 1. Datos  ← ESTA FASE                                                        │
│    proveedores → jobs de ingesta → lake Parquet → catálogo DuckDB            │
│    + calidad de datos + registro de ejecuciones (backlog auditable)          │
└────────────────────────────────────────────────────────────────────────────┘
```

## Capa de datos

```
src/finance/
├── settings.py        configuración (variables FIN_*, .env, /etc/finance/finance.env)
├── universe.py        universo de instrumentos (config/universe.yaml), grupos estáticos y dinámicos
├── calendar.py        calendarios de bolsa: sesiones cerradas, mercado abierto, festivos
├── jobs.py            marco común de jobs: contexto, registro de ejecuciones, concurrencia
├── providers/         fuentes de datos detrás de interfaces abstractas
│   ├── base.py        PriceProvider, OptionChainProvider, MacroProvider, UniverseProvider...
│   ├── yahoo.py       precios, acciones corporativas, cadenas de opciones, metadatos
│   ├── fred.py        series macro con vintages (ALFRED) o sin clave (fredgraph)
│   ├── wikipedia.py   composición actual del S&P 500 con GICS
│   ├── sp500_history.py  historial de composición desde 1996 (fja05680/sp500)
│   └── ratelimit.py   token bucket compartido + reintentos con backoff exponencial
├── storage/
│   ├── schemas.py     registro de datasets: esquema Arrow, clave primaria, partición
│   ├── lake.py        escrituras atómicas append-only, locks, compactación
│   └── catalog.py     vistas DuckDB deduplicadas + vistas/macros point-in-time
├── ingest/            jobs: prices, options, macro, universe, instruments, compact
├── quality/checks.py  controles de calidad → dataset dq_results
└── cli.py             comando `fin`
```

### Decisiones de diseño

**Lake append-only + deduplicación en lectura (estilo LSM).** Cada escritura crea ficheros
Parquet nuevos e inmutables (escritos como `.tmp` y publicados con `rename`, atómico en
POSIX). Nunca se modifica un fichero. Las vistas de DuckDB eligen, para cada clave
primaria, la fila con `ingested_at` más reciente. Consecuencias:

- Un fallo a mitad de un job nunca deja el lake corrupto.
- Se puede leer mientras se escribe o compacta.
- Re-ingerir es seguro (idempotente): la versión nueva gana.
- La compactación (diaria/semanal) fusiona ficheros pequeños y elimina versiones antiguas.

**Esquemas explícitos y versionados.** Cada dataset declara tipos Arrow, columnas
obligatorias, clave primaria y partición. Los datos se validan y convierten antes de
escribirse; un cambio de formato del proveedor falla de forma ruidosa en lugar de
contaminar el histórico. Añadir columnas es compatible (`union_by_name`).

**Auditoría en cada fila.** `source`, `ingested_at` y `run_id` permiten saber de dónde y
cuándo viene cada dato y enlazarlo con su ejecución en `ingest_runs` (que guarda
parámetros, errores por símbolo y versión del código — commit de git).

**Point-in-time desde el principio.** Los backtests con información que no existía en la
fecha simulada son la causa más común de resultados falsos. Por eso:
- Las series macro revisables guardan todos sus vintages (`realtime_start`/`realtime_end`)
  y `macro_asof(fecha)` devuelve lo que se sabía en esa fecha.
- Composición histórica del S&P 500 (sesgo de supervivencia): `universe_intervals` combina
  el historial externo desde 1996 con las fotos semanales propias posteriores a su última
  fecha. `universe_asof('sp500', fecha)` devuelve los miembros vigentes. Los cambios vistos
  solo entre dos fotos se fechan el día en que se observaron (conservador) y guardan la
  anchura de la ventana de incertidumbre (`start_window_days`/`end_window_days`). El
  sector viene de la foto propia más cercana: para fechas antiguas es la clasificación GICS
  actual (aproximación; p. ej. GOOGL figura en Communication Services en 2008).
- Nunca se guarda la barra de la sesión en curso (sería parcial).

**Ajuste por dividendos.** Yahoo da `close` (ajustado solo por splits) y `adj_close`
(splits + dividendos), y reescala todo el histórico de `adj_close` con cada dividendo.
El job de precios lo detecta (dividendo nuevo o `adj_close` solapado distinto del
guardado) y vuelve a descargar el histórico completo del símbolo. Las acciones
corporativas se guardan aparte para poder reconstruir retornos totales propios.

**Opciones: fotos completas.** La cadena entera de cada subyacente (todas las
expiraciones hasta ~2 años y todos los strikes) con un `snapshot_ts` común. Es la base para
superficies de volatilidad, skew, term structure y posicionamiento (open interest). Yahoo
no ofrece histórico de opciones: el valor de este dataset crece con cada día capturado.

**Datasets regenerados (`snapshot_by`).** Los derivados de una fuente que puede corregir
su pasado (historial del índice) se reescriben enteros en cada ejecución; la vista solo
muestra la última ingesta, así una fecha corregida no deja intervalos fantasma.

**Paso a vintages sin migraciones.** Si una serie macro se cargó sin clave de FRED y luego
con ella, la vista descarta automáticamente las filas sin vintages de esa serie.

**Proveedores intercambiables.** Los jobs solo conocen las interfaces de `providers/base.py`.
Pasar a una fuente profesional (Polygon, Tiingo, IBKR, Databento, ORATS para opciones) es
escribir una clase nueva; el almacenamiento, el catálogo y los modelos no cambian.

**Límites de peticiones y fallos parciales.** Un token bucket compartido limita las
peticiones a Yahoo. Los errores transitorios (429, red, 5xx) se reintentan con backoff;
los definitivos (símbolo inexistente) no. Un símbolo que falla no detiene el job: la
ejecución queda como `partial` con el error registrado.

## Datasets

| Dataset | Clave | Partición | Contenido |
|---|---|---|---|
| `prices_daily` | symbol, date | year | OHLCV diario, `adj_close`, dividendo, split |
| `corporate_actions` | symbol, date, action | — | dividendos, splits, ganancias de capital |
| `option_snapshots` | contract_symbol, snapshot_ts | snapshot_date | cadenas completas + cotización del subyacente |
| `macro_observations` | series_id, date, realtime_start | — | series FRED con vintages |
| `universe_membership` | universe, as_of_date, symbol | — | fotos de composición (S&P 500 + GICS) |
| `universe_intervals` | universe, symbol, start_date | — | pertenencia [inicio, fin) desde 1996; se regenera entera cada semana |
| `index_changes` | universe, date, added, removed | — | altas/bajas derivadas del historial |
| `instruments` | symbol, as_of_date | — | sector, industria, categoría de fondo, tamaño, beta |
| `ingest_runs` | run_id | — | registro de cada ejecución |
| `dq_results` | run_id, dataset, check_name, key | — | violaciones de calidad |

## Universo

`config/universe.yaml` define grupos con significado económico, pensados para las capas
siguientes:

- **Sectores GICS (11 SPDR) + ~22 ETFs de industria** → detección de sectores/temas en auge
  y rotación sectorial.
- **Factores** (momentum, quality, value, min-vol, high beta) → régimen de factores.
- **Estructura de volatilidad** (VIX 9D/30D/3M/6M, VVIX, SKEW, MOVE) → régimen de volatilidad.
- **Tipos y crédito** (curva, spreads HY/IG, condiciones financieras) → régimen macro/estrés.
- **S&P 500 completo** (dinámico) → acciones individuales y amplitud de mercado (breadth).
- **Opciones** sobre índices y sectores → skew, term structure, put/call.

## Cómo extender

- **Nuevo dataset**: declarar un `DatasetSpec` en `storage/schemas.py` y añadirlo a
  `REGISTRY`. Las vistas se crean solas.
- **Nuevo proveedor**: implementar la interfaz en `providers/` y conectarlo en
  `build_providers`.
- **Nuevo job**: subclase de `Job` (`jobs.py`) con `execute(run, ...)`; usar
  `run.write(spec, df, source)`. El registro en `ingest_runs` es automático.
- **Nuevo control de calidad**: función en `quality/checks.py` añadida a `CHECKS`.
