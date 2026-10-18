# finance

Sistema cuantitativo de cartera dinámica: detección de régimen de mercado (volatilidad,
bull/bear), sectores en auge, y sugerencias de entrada, ampliación y reducción de posiciones
en fondos, acciones y opciones.

El proyecto se construye por capas (ver [docs/architecture.md](docs/architecture.md)).
**Fase actual: alertas activas y ejecución real (semana 11).**

| Capa | Estado |
|---|---|
| 1. Datos: ingesta, data lake Parquet/DuckDB, opciones, macro, calidad, registro de ejecuciones | ✅ |
| 2. Features: tendencia, volatilidad, amplitud, crédito, correlación, panel sectorial ([docs](docs/features.md)) | ✅ |
| 3. Regímenes: tendencia × volatilidad + crisis, exposición recomendada y validación ([docs](docs/regimes.md)) | ✅ |
| 4. Señales: rotación sectorial por régimen, cobertura, acciones entrar/añadir/reducir/salir ([docs](docs/portfolio.md)) | ✅ |
| 5. Cartera: asignación por régimen y walk-forward ✅; construcción condicionada al régimen (`rot_v2`, [resultado negativo](docs/portfolio.md#rot_v2-construcción-condicionada-al-régimen-semana-6)) ✅; bandas de no negociación (`rot_v1b`, [referencia actual](docs/portfolio.md#bandas-de-no-negociación-semana-7)) ✅; límites de riesgo y opciones | parcial |
| 6. Paper trading congelado e informe diario ([docs](docs/paper.md)) ✅; alertas activas y ejecución real con IBKR o manual ([docs](docs/ejecucion.md)) ✅ | ✅ |
| Insiders: Form 4 de la SEC, compras en grupo/CEO, amplitud por sector y validación pre-registrada ([docs](docs/insiders.md), [resultado negativo](docs/insiders.md#resultados-17-oct-2026-resultado-negativo)) | ✅ (contexto, no decide) |

## Qué datos recoge

- **Precios diarios** desde 1990 de ~580 instrumentos: S&P 500 completo, 11 sectores GICS,
  ~22 ETFs de industria (semis, software, biotech, bancos regionales...), factores, bonos,
  crédito, materias primas, internacionales e índices de volatilidad (VIX 9D/30D/3M/6M,
  VVIX, SKEW, MOVE). Incremental, con detección de re-ajustes por dividendos.
- **Cadenas de opciones completas** de 21 subyacentes (índices + sectores), 3 fotos por
  sesión. Yahoo no da histórico de opciones: cada foto acumula un activo que no se puede
  recuperar después.
- **Macro (FRED/ALFRED)**: curva de tipos, spreads de crédito, condiciones financieras,
  empleo, inflación... Las series revisables guardan todas sus versiones (vintages): p. ej.
  el empleo de marzo de 2020 se publicó como −701k y hoy figura como −1,40M; un backtest
  ve la cifra que existía en cada fecha.
- **Composición histórica del S&P 500 desde 1996** (altas y bajas, ~1.200 tickers) a partir
  de [fja05680/sp500](https://github.com/fja05680/sp500), enlazada con fotos semanales
  propias de Wikipedia (con sector/subindustria GICS) para el tramo reciente.
- **Metadatos** de instrumentos (sector, industria, categoría de fondo, tamaño, beta).
- **Operaciones de insiders (SEC Form 4/5) desde 2006**: compras y ventas de consejeros y
  directivos de todos los emisores (datasets trimestrales de la SEC) y, para el S&P 500, al
  día siguiente vía EDGAR. Con marca de plan 10b5-1 y fecha de presentación (point-in-time).

Todo con esquema validado, auditoría por fila (`source`, `ingested_at`, `run_id`) y un
registro de cada ejecución (`ingest_runs`) con parámetros, errores y versión del código.

## Inicio rápido (desarrollo)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # gestor de entorno (si no lo tienes)
git clone https://github.com/jatien/finance.git && cd finance
uv sync --extra dev                                # crea .venv con Python 3.13
cp .env.example .env                               # ajusta FIN_DATA_DIR y FIN_FRED_API_KEY

uv run fin init
uv run fin ingest universe                         # S&P 500 actual (Wikipedia)
uv run fin ingest universe-history                 # historial de composición desde 1996
uv run fin run daily                               # precios + macro + calidad (~5 min la primera vez)
uv run fin ingest options --force                  # foto de opciones (fuera de horario: --force)
uv run fin features build                          # capa 2: features de mercado y sectoriales
uv run fin features show                           # estado actual del mercado
uv run fin features report                         # informe HTML (uv sync --extra viz)
uv run fin regimes build                           # capa 3: régimen de cada sesión
uv run fin regimes show                            # régimen actual y exposición recomendada
uv run fin regimes report                          # validación: backtest, ablaciones, sensibilidad
uv run fin portfolio build                         # capas 4-5: rot_v1b y rot_v2 al registro de decisiones
uv run fin portfolio show [--set rot_v2]           # cartera actual y cambios sugeridos
uv run fin portfolio report                        # walk-forward contra referencias
uv run fin paper update                            # semana 8: paper trading congelado
uv run fin report daily                            # informe diario: qué hacer en la sesión siguiente
uv run fin ingest insiders                         # Form 4 de la SEC (requiere FIN_SEC_USER_AGENT)
uv run fin insiders build                          # eventos (compras en grupo, CEO/CFO) y amplitud
uv run fin insiders show                           # eventos recientes y amplitud por sector
uv run fin insiders report                         # validación: event study y criterios
uv run fin alerts test                             # semana 11: prueba de alertas (ntfy/Telegram/correo)
uv run fin exec status                             # ejecución real: planes, órdenes, cuenta
uv run fin status
uv run fin sql "SELECT * FROM prices_coverage ORDER BY last_date LIMIT 10"
uv run pytest
```

Desde Python (notebooks, modelos):

```python
from finance.jobs import Context
from finance.settings import get_settings

ctx = Context.from_settings(get_settings())
px = ctx.query("""
    SELECT date, symbol, adj_close FROM prices_daily
    WHERE symbol IN ('XLK','XLE','XLF') AND date >= '2020-01-01'
""").pivot(index="date", columns="symbol", values="adj_close")

# Lo que se sabía de las nóminas el 10-abr-2020 (sin look-ahead)
ctx.query("SELECT * FROM macro_asof(DATE '2020-04-10') WHERE series_id = 'PAYEMS'")

# Miembros del S&P 500 el día de la quiebra de Lehman (incluye LEHMQ, AIG, BSC ya fuera...)
ctx.query("SELECT * FROM universe_asof('sp500', DATE '2008-09-15')")
```

## Servidor

Instalación en Debian, SSH, timers de systemd, copias de seguridad y conexión desde VSCode:
[docs/server.md](docs/server.md).

```bash
sudo FIN_USER=$USER bash deploy/bootstrap.sh
```

## Estructura

```
config/universe.yaml   universo de instrumentos, opciones y series macro
src/finance/           código (comentarios en catalán)
  providers/           Yahoo, FRED, Wikipedia, SEC EDGAR (intercambiables)
  storage/             lake Parquet + catálogo DuckDB
  ingest/              jobs de ingesta
  quality/             controles de calidad
  features/            capa 2: features point-in-time (mercado + panel sectorial)
  regimes/             capa 3: regímenes por reglas, exposición y validación
  portfolio/           capas 4-5: rotación sectorial, cobertura, motor walk-forward
  paper/               semana 8: paper trading congelado e informe diario
  insiders/            semana 10: señales de insiders (Form 4) y su validación
  execution/           semana 11: plan, salvaguardas, envío a IBKR, conciliación
  alerts/              semana 11: reglas y envío de alertas con deduplicación
tests/                 pytest con proveedores simulados (sin red)
deploy/                bootstrap del servidor, unidades systemd, backup, IB Gateway
docs/                  arquitectura y operación
```
