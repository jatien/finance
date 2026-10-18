# Insiders (semana 10)

Seguimiento de las compras y ventas de consejeros y directivos (Form 4 de la SEC) como
fuente de señales por acción, por sector y para el mercado.

## Por qué no Kestrel Terminal

[Kestrel Terminal](https://www.kestrelterminal.com/) es un rastreador web de Form 4 (detección
de *cluster buys*, alertas, filtros por cargo). Su fuente es EDGAR y **no ofrece API**
(octubre de 2026), así que no se puede conectar al sistema. Lo que hace se reproduce aquí
desde la fuente original, gratuita, con historial desde 2006 y con control point-in-time,
que una herramienta web no da.

## Datos

```bash
# .env o /etc/finance/finance.env — la SEC exige identificarse
FIN_SEC_USER_AGENT="Javi tu@correo.com"

uv run fin ingest insiders            # trimestres pendientes + cola diaria (~20 min la 1.ª vez)
uv run fin insiders build             # eventos y amplitud
uv run fin insiders show              # eventos recientes en el S&P 500 y amplitud por sector
uv run fin insiders report            # validación (HTML)
```

| Fuente | Cobertura | Uso |
|---|---|---|
| [Insider Transactions Data Sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets) (DERA) | Todos los emisores, trimestral, desde 2003 | Histórico. Se publica días después de cerrar el trimestre |
| Índice diario de EDGAR + XML de cada Form 4 | Emisores del S&P 500 (CIK), diario | Días posteriores al último trimestre publicado |

Cuando sale el trimestre, sus filas **sustituyen** a las del XML provisional de cada
presentación (`snapshot_by` por `accession_number`): no hay duplicados. Se comprobó con
presentaciones reales que ambas vías dan las mismas operaciones (el TSV redondea precios
a 2 decimales).

Dataset `insider_transactions`: una fila por línea no derivada de cada Form 4/5, con rol
del titular, código de operación, importe, acciones tras la operación y la marca de plan
10b5-1. La fecha que cuenta es `filing_date`, **no** `trans_date`: una operación no se
conoce hasta que se presenta (hasta 2 días hábiles después; los Form 5, meses).

## Señales (`ins_v1`)

Operación **informativa** (literatura: Lakonishok y Lee 2001; Cohen, Malloy y Pomorski 2012):

- Compra (`P`) o venta (`S`) **en mercado abierto**. Concesiones (`A`), ejercicios (`M`),
  retenciones fiscales (`F`) y regalos (`G`) no dicen nada sobre las expectativas.
- De un **consejero o directivo** (los accionistas >10 % que no lo son, normalmente fondos,
  quedan fuera).
- **Fuera de un plan 10b5-1** (operaciones preprogramadas). Se detecta por la casilla
  AFF10B5ONE (desde 2023) y por las notas al pie que mencionan el plan (antes).
- De un insider **no rutinario**: rutinario = operó en el mismo mes natural en cada uno de
  los 3 años anteriores (CMP 2012). Con menos de 3 años de historial queda como "nuevo" y
  se acepta.
- Importe ≥ 10.000 $ y precio > 0.

Eventos por acción (fecha de señal = primera sesión **posterior** a la presentación):

| Evento | Definición |
|---|---|
| `buy` | Al menos una compra informativa |
| `ceo_cfo_buy` | Compra informativa del CEO o del CFO |
| `cluster_buy` | ≥ 3 insiders distintos comprando en 30 días naturales (una vez cada 90 días) |
| `cluster_sell` | Ídem con ventas |

Amplitud (`insider_breadth`): pares insider-empresa-mes con compras y ventas informativas en
las últimas 63 sesiones, entre los miembros vigentes del S&P 500, para el mercado y por
sector GICS. `buy_ratio` = compras / (compras + ventas) y su percentil expansivo.

## Validación: criterios pre-registrados

Escritos **antes** de ver ningún resultado. Ventana de diseño 2006–2018, de comprobación
2019–hoy. Entrada al cierre de la sesión de señal (se puede ejecutar de verdad).

| Criterio | Pasa si |
|---|---|
| C1 | `cluster_buy`: exceso medio frente al ETF del sector a 63 sesiones > 0 con t ≥ 2 (agrupado por mes) en diseño, y > 0 en comprobación |
| C2 | Cartera equiponderada de `cluster_buy` (mantener 63 sesiones): alfa frente al SPY > 0 en ambas ventanas e IR ≥ 0,3 en diseño |
| C3 | Los filtros (10b5-1, rutinarios) aportan: compras filtradas > compras sin filtrar a 63 sesiones en ambas ventanas |
| C4 | Amplitud agregada: el SPY a 63 sesiones rinde más en el tercio alto del percentil de compras que en el bajo, en ambas ventanas |

Uso según el resultado:

- **C1 y C2 pasan** → el evento entra como señal candidata para la capa de acciones
  individuales (aún por construir) y se destaca en el informe diario. **No** modifica las
  estrategias congeladas del paper trading (`rot_v1b`).
- **No pasan** → resultado negativo documentado; la información queda como contexto en el
  informe diario, sin influir en ninguna decisión.
- **C4 pasa** → candidata a entrada adicional de la capa de regímenes (solo como prueba
  pre-registrada en una semana futura, nunca cambiando `rules_v1` a posteriori).

## Sesgos y limitaciones conocidos

- **Supervivencia en precios**: Yahoo no tiene precios de muchas empresas deslistadas. Se
  intentan descargar los miembros históricos del S&P 500, pero los eventos sin precio se
  pierden; el informe muestra la cobertura. Algunos tickers deslistados se han
  reutilizado por otra empresa: riesgo residual de precios equivocados.
- **10b5-1 antes de 2023**: solo por notas al pie; la casilla obligatoria llegó en 2023 y
  el porcentaje de operaciones marcadas sube. La amplitud usa percentiles expansivos y
  puede tener un salto estructural ese año.
- **Sector**: clasificación GICS actual (como en el resto del sistema).
- **Grandes empresas**: la literatura encuentra el efecto sobre todo en pequeñas y
  medianas; el S&P 500 es el universo más difícil para esta señal.
- **Enmiendas** (`4/A`) excluidas de las señales para no contar dos veces.

## Resultados (17-oct-2026): resultado negativo

Datos: 81 trimestres de la SEC (2006 → 2026T3, 7,35 M de líneas) + cola diaria de
octubre. Eventos del S&P 500 vigente; precios de los miembros actuales y de 179 de los 465
históricos sin precio en el lake (el resto no está en Yahoo). Cobertura de precios de los
eventos: 73–82 %.

**Veredicto: no pasa** (C1, C2 y C4 fallan; C3 pasa por un margen irrelevante). La señal
no entra en ninguna decisión; queda como contexto en el informe diario.

| Criterio | Diseño 2006–2018 | Comprobación 2019–2026 | Pasa |
|---|---|---|---|
| C1 `cluster_buy`, exceso vs sector a 63 sesiones | +2,50 % (t 1,2; N 293) | +0,26 % (t 0,1; N 175) | no |
| C2 cartera `cluster_buy` | alfa +3,0 %/año, IR 0,29 | alfa −6,2 %/año, IR −0,21 | no |
| C3 filtradas vs sin filtrar (63 s) | 1,82 % vs 1,81 % | 0,00 % vs −0,22 % | sí (irrelevante) |
| C4 amplitud: SPY tercio alto − bajo (63 s) | −4,4 pp | +0,4 pp | no |

Exceso medio frente al ETF del sector a 63 sesiones:

| Evento | Diseño | Comprobación |
|---|---|---|
| `buy` (compra informativa) | **+1,82 % (t 3,5; N 4.186)** | +0,00 % (t 0,4; N 2.526) |
| `cluster_buy` | +2,50 % (t 1,2) — mediana −0,26 % | +0,26 % (t 0,1) |
| `ceo_cfo_buy` | +1,59 % (t 1,4) | −0,19 % (t 0,2) |
| `cluster_sell` | +0,27 % (t 1,9) | −0,62 % (t −2,4) |
| compras de rutinarios (ablación) | −0,20 % (N 56) | −1,87 % (N 51) |

Lectura:

- **El efecto existía y ha desaparecido.** Las compras de directivos predecían ~1,8 % de
  exceso trimestral en 2006–2018 (significativo), y nada desde 2019: coherente con la
  literatura de decaimiento de anomalías publicadas y con la popularidad de las
  herramientas de seguimiento (OpenInsider, Kestrel...). En el S&P 500 la señal ya llega
  descontada.
- **Las compras en grupo no son mejores** que una compra individual: pocas (≈25 al año),
  media inflada por unos pocos casos extremos (mediana negativa) y alfa negativa desde 2019.
- **Los filtros apenas cambian nada** en el S&P 500 (pocas compras están en planes 10b5-1),
  aunque las compras de insiders rutinarios sí son peores, como predice CMP 2012.
- **La amplitud agregada no anticipa el mercado**: en diseño, incluso al revés (los
  insiders compran en las caídas, que siguieron cayendo en 2008).
- `cluster_sell` en comprobación (−0,6 %, t −2,4) es lo único significativo desde 2019,
  pero no estaba pre-registrado como criterio y en diseño tenía el signo contrario:
  **hipótesis para una prueba futura**, no una señal.
- Sesgo de supervivencia: ~20 % de los eventos sin precio (empresas deslistadas); si
  algo, infla los resultados — refuerza la conclusión negativa.

Siguiente prueba razonable (pre-registrar antes de mirar): el mismo estudio fuera del
S&P 500 (pequeñas y medianas, donde la literatura sitúa el efecto). Requiere precios de
ese universo.
