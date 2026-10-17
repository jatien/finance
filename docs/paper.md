# Semana 8: paper trading e informe diario

Desde aquí el sistema deja de ser solo un backtest: cada noche decide qué haría con las
reglas actuales, lo apunta en un registro que no se puede reescribir y genera un informe con
las órdenes para la sesión siguiente.

```bash
uv run fin paper update          # avanza el paper trading hasta la última sesión con datos
uv run fin paper show [--set X]  # cartera, objetivo vigente y rendimiento desde el inicio
uv run fin paper log             # registro de decisiones: cada operación y su motivo
uv run fin report daily          # informe diario (HTML + resumen de texto)
```

`fin run daily` ejecuta `paper` después de `portfolio` y el informe al final. Estrategias
registradas: las de `ACTIVE` en `portfolio/config.py` (hoy `rot_v1b`, la referencia, y
`rot_v2`).

## Cómo funciona

| Paso | Cuándo | Qué |
|---|---|---|
| Decisión | cierre de *t* (el pipeline corre a las 03:15 de Madrid) | objetivo con datos hasta *t* (`build_targets`) |
| Informe | mañana de *t+1* | órdenes previstas para la sesión siguiente, con importes |
| Ejecución | cierre de *t+1* | si el disparador de la estrategia lo pide (revisión nueva, o deriva con TE > 1 % en `rot_v1b`) |
| Registro | noche de *t+1* | `paper_book` (posiciones) y `paper_nav` (valor, operaciones, motivo) |

- **Ejecución a t+1, no a t.** El backtest supone que se opera al cierre del mismo día en que
  se decide, que en la práctica es imposible. El paper trading opera en la sesión siguiente.
  Medido en 2000–2026, el retraso de una sesión no empeora el resultado:

  | Estrategia | Retraso 0 (Sharpe / caída) | Retraso 1 | Retraso 2 |
  |---|---|---|---|
  | `rot_v1b` | 0,61 / −16,4 % | 0,63 / −18,0 % | 0,63 / −19,9 % |
  | `rot_v2` | 0,57 / −26,0 % | 0,59 / −22,7 % | 0,59 / −22,6 % |
  | Regímenes sobre SPY | 0,51 / −25,6 % | 0,51 / −25,4 % | 0,52 / −27,3 % |

  Las señales son mensuales y lentas. Las diferencias de caída máxima (±2–4 puntos) indican
  que ese número depende bastante del camino concreto: no hay que leerlo con más precisión.
- **Mismo motor que el backtest.** La mecánica de cada sesión (deriva, disparador, costes) es
  una única función, `backtest.step`. Los tests comprueban que el paper trading es
  exactamente el backtest con retraso de una sesión, y que avanzar sesión a sesión (con solo
  los datos de cada día) da lo mismo que recalcularlo todo de golpe.
- **Inicio.** El primer día se registra 100 % en efectivo con la cartera objetivo vigente como
  orden para la sesión siguiente, aunque no sea día de revisión. Así no hay que esperar al
  mes siguiente.

## Registro congelado

`paper_book` y `paper_nav` son append-only de verdad: cada sesión se escribe una sola vez,
con los datos disponibles ese día, y nunca se recalcula. Así:

- Si Yahoo corrige un precio más tarde (dividendos, barras provisionales), el registro
  conserva lo que se decidió. El informe recalcula el mismo periodo con los datos actuales y
  avisa si difiere ("recalculado vs registro"): es la medida de cuánto se mueven los datos.
- Volver a ejecutar el job el mismo día no escribe nada.
- Si el servidor estuvo apagado, el job recupera las sesiones que faltan y las marca
  `recorded_late` (se escribieron con datos posteriores a la sesión siguiente).
- No se avanza más allá de la última sesión con precio real de **todos** los activos: el
  registro nunca decide con un precio rellenado.
- Una estrategia nueva (p. ej. `rot_v3`) abre su propio libro. Para cambiar de estrategia no
  se reescribe nada: se añade a `ACTIVE`.

`portfolio_targets` sigue existiendo, pero es otra cosa: el objetivo en cada revisión según el
backtest completo, regenerado cada día. El registro operativo es `paper_*`.

## Informe diario

En `$FIN_DATA_DIR/reports/daily/`: `AAAA-MM-DD.html`, `latest.html` y `latest.md`. El orden
de las secciones sigue las preguntas de cada mañana:

1. **¿Me puedo fiar de los datos?** Banda verde o roja. Rojo (no operar) si los precios van
   con retraso respecto a la última sesión cerrada o si algún job del pipeline falló. Avisos
   (amarillo) por errores de calidad de datos, paper trading atrasado o correcciones del
   proveedor.
2. **Qué hacer en la sesión siguiente.** Por estrategia: si hay que operar y por qué
   (revisión mensual, cambio de régimen, deriva), órdenes con peso actual → objetivo, importe
   sobre el capital nocional (`FIN_PAPER_CAPITAL`, 10 000 por defecto) y número aproximado de
   participaciones. Las ventas van primero. Si no hay que operar, dice por qué (p. ej. "TE 0,4 %
   ≤ 1 %").
3. **Régimen**: estado, desde cuándo, exposición recomendada y próxima revisión mensual.
4. **Paper trading**: rendimiento desde el inicio contra SPY, caída máxima, costes y la
   comprobación contra los datos actuales. Gráfico a partir de 5 sesiones.
5. **Registro de decisiones**: cada sesión con operaciones, el motivo y los cambios de peso.
6. **Contexto**: indicadores clave con su percentil histórico y ranking de sectores.
7. **Simulación reciente** (claramente separada del paper trading): 1 m, 3 m, año en curso y
   12 meses del backtest con retraso de una sesión, mientras el paper trading acumula historia.

## Ejemplo: primer informe (cierre del 16-oct-2026)

Generado al desarrollar esta semana. El paper trading del servidor empieza el primer día que
corra `fin run daily` allí, con su propia fecha de inicio.

- Régimen alcista y en calma desde el 13-abr (131 sesiones), 100 % renta variable.
- `rot_v1b`: comprar XLV 40,6 %, XLE 32,5 %, XLK 26,9 %. `rot_v2`: SPY 70 % + los mismos
  tres sectores en el satélite.
- Simulación a 12 meses: `rot_v1b` +10,8 %, `rot_v2` +15,8 %, SPY +16,8 %; caída máxima
  −6,8 %, −8,3 % y −8,9 %. En un año alcista sin sobresaltos es lo esperable: el sistema
  cobra en las caídas, no en las subidas.
- La correlación SPY–TLT a 3 meses está en +0,39 (percentil 98): si el régimen cambiara, la
  cobertura iría a oro + efectivo, no a bonos.

## Qué falta (no incluido a propósito)

- **Ejecuciones reales**: el registro supone que se opera al cierre al precio de cierre. Si
  se empieza a operar de verdad, conviene apuntar las ejecuciones reales para medir el
  *slippage* frente al paper.
- **Avisos activos** (correo, ntfy, Telegram): hoy el informe se consulta. Es lo siguiente
  natural para la capa 6.
- **Decidir con el paper trading**: con 1–2 operaciones al mes, hacen falta muchos meses
  para que la diferencia entre `rot_v1b` y `rot_v2` sea significativa. Antes de eso, el valor
  del paper trading es comprobar que la operativa funciona (datos, órdenes, costes), no
  comparar estrategias.
