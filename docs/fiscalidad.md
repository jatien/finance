# Semana 9: impuestos en España y comisiones de IBKR

Hasta ahora todos los resultados eran antes de impuestos, con un coste fijo de 5 pb por
unidad negociada y con ETFs de EE. UU. Para un inversor residente en España con Interactive
Brokers eso no es realista:

1. **No puede comprar SPY, XLK…**: los ETFs domiciliados en EE. UU. no tienen el KID que exige
   PRIIPs a un minorista de la UE. Opera con los equivalentes **UCITS** irlandeses, que
   replican los mismos índices: los *SPDR S&P U.S. Select Sector UCITS* en Xetra (ZPDT, ZPDE,
   ZPDH…) e iShares Core S&P 500 (SXR8). Los de acumulación no reparten dividendos, así que no
   tributan hasta la venta. A cambio cuestan algo más: TER más alto y retención del 15 % sobre
   los dividendos de EE. UU. dentro del fondo.
2. **Cada venta con ganancia tributa** en la base del ahorro (19–30 %). Los ETFs no tienen
   traspasos sin coste fiscal como los fondos. FIFO obligatorio. Regla de los dos meses.
3. **Comisiones por orden con mínimo** (IBKR Fixed en Xetra: 0,05 %, mínimo 3 €, máximo 29 €).
4. **Todo se mide en euros**: el tipo de cambio EUR/USD también genera ganancias y pérdidas
   fiscales.

Código en `src/finance/fiscal/`. Parámetros y fuentes en `fiscal/config.py`.

## Modelo

| Pieza | Cómo se modeliza |
|---|---|
| Precios | Serie de retorno total del ETF de EE. UU. ÷ EUR/USD (FRED `DEXUSEU`, desde 1999) × coste UCITS diario: (TER UCITS − TER EE. UU.) + 15 % × rentabilidad por dividendo de los 12 meses previos (solo renta variable) |
| Efectivo | Euros, remunerado al tipo de depósito del BCE − 0,5 % (mínimo 0); los intereses son rendimientos del capital mobiliario |
| Comisiones | 0,05 % del importe, mínimo 3 €, máximo 29 € por orden. No se envían órdenes de menos de 250 € |
| Lotes | FIFO por activo; el coste incluye la comisión de compra |
| Regla de los dos meses | Una pérdida no se integra si se recompra el mismo activo en los dos meses anteriores o posteriores; se suma al coste de los lotes recomprados (se integra al venderlos) |
| IRPF | Escala del ahorro 2026 aplicada cada año (también a años antiguos: "como si las reglas de hoy hubieran existido siempre"); pérdidas contra ganancias, contra intereses hasta el 25 %, resto arrastrado 4 años |
| Pago | El impuesto del año *Y* sale de fuera de la cartera en julio de *Y+1* (no obliga a vender) |
| Final | Se vende todo el último día y se paga el impuesto pendiente |
| Métrica | **TIR de los flujos del inversor**: −80.000 € al inicio, −300 €/mes, −impuestos, +valor final neto |

Las decisiones (objetivos, disparador, rankings) son las de cada estrategia con una sesión de
retraso, como el paper trading. Comprobación: con comisiones proporcionales y sin impuestos ni
divisa, el simulador reproduce el motor de pesos (rot_v1b ×9,94 frente a ×9,93 en 2000–2026).

## Pre-registro

Escrito y confirmado en git **antes** de ejecutar ninguna simulación con impuestos.

**Inversor**: 80.000 € iniciales, 300 €/mes, IBKR Fixed, escala progresiva real.

**Política "solo añadir" (`acum_v1`)**, sobre los objetivos de `rot_v1b`:

- **Compras**: las aportaciones y el efectivo que sobra respecto del objetivo van a los
  activos infraponderados, en proporción a lo que les falta.
- **No se vende para reequilibrar.** Lo que sube se queda.
- **Solo se vende en dos casos**, en una revisión (inicio de mes o cambio de régimen):
  1. El régimen baja la renta variable y la cartera tiene más de 10 puntos por encima del
     objetivo: se vende hasta el objetivo. Primero lo que ya no está en el objetivo; dentro de
     cada grupo, lo que menos impuesto genera por euro vendido (pérdidas primero). Cuando el
     régimen vuelve a subir la renta variable, se vende la cobertura sobrante.
  2. Un sector en cartera cae por debajo del puesto 8 de 11 del ranking (top-3 + 5) y no está
     en el objetivo: se vende entero.
- No se compra un activo vendido con pérdida en los dos meses anteriores (la pérdida se
  perdería).

**Estrategias comparadas** (todas con el mismo inversor, impuestos y comisiones):
SPY comprar y mantener (UCITS), regímenes sobre SPY, `rot_v1b`, `rot_v2`, `acum_v1` y tres
ablaciones de `acum_v1`: sin ventas por régimen, sin venta por degradación y sobre los
objetivos de `rot_v2`.

**Fechas de inicio**: 1-feb-2000, 2-ene-2008 y 4-ene-2016, todas hasta oct-2026.

**Criterio**: `acum_v1` pasa a ser la referencia operativa (sustituye a `rot_v1b` en el paper
trading) si su TIR neta supera la de `rot_v1b` en al menos 2 de los 3 inicios **y** su caída
máxima (TWR antes de impuestos) no es más de 5 puntos peor en ninguno. Si no, se registra el
resultado y `rot_v1b` sigue de referencia. La comparación con SPY y con regímenes sobre SPY se
informa tal cual, sin regla de decisión.
