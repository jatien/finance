# Semana 11: alertas activas y ejecución real

Último paso de la capa 6. El sistema deja de esperar a que consultes el informe: te avisa
cuando hay algo que hacer y, si lo activas, lleva las órdenes al bróker y mide lo que pasa
de verdad (ejecuciones, *slippage*, comisiones, real frente a paper).

```bash
uv run fin alerts test                  # prueba de los canales configurados
uv run fin exec status                  # modo, parada, últimos planes, órdenes y cuenta
uv run fin exec plan                    # plan para la sesión siguiente (el pipeline lo hace solo)
uv run fin exec approve rot_v1b-20261019
uv run fin exec submit                  # envía las órdenes de hoy (timer 16:45)
uv run fin exec sync                    # ejecuciones y foto de la cuenta (timer 17:50 y nocturno)
uv run fin exec fill ZPDT BUY 20 120.41 --fee 3    # modo manual: apuntar una ejecución
uv run fin exec halt "mercado raro"     # interruptor: no se envía nada y se cancela lo vivo
uv run fin exec resume
```

## Alertas

| Nivel | Cuándo |
|---|---|
| crítica | datos en rojo (no operar), job fallido, plan bloqueado, órdenes no enviadas u omitidas, ejecución parcial o no ejecutada, parada activada, caída > 20 % |
| importante | órdenes para la sesión siguiente, plan pendiente de aprobación, órdenes enviadas, cambio de régimen, caída > 10 % y > 15 % |
| info | resultado de una ejecución completa, resumen semanal (viernes, se envía siempre) |

Por defecto se envían importante y crítica (`FIN_ALERT_MIN_LEVEL`). Cada alerta tiene una clave
(fecha, plan, episodio): un mismo hecho no se avisa dos veces aunque el pipeline se repita, y
si ningún canal la entrega se reintenta en la ejecución siguiente. Todo queda en `alerts_sent`.

**Canales** (se pueden combinar):

- **ntfy** (recomendado). Instala la app ntfy en el móvil, suscríbete a un tema con nombre
  difícil de adivinar y pon `FIN_ALERT_NTFY_URL=https://ntfy.sh/ese-tema`. Sin cuenta. El tema
  es público para quien conozca el nombre: para algo privado, servidor propio con token.
- **Telegram**: crea un bot con @BotFather, escríbele una vez y usa tu `chat_id`.
- **Correo**: SMTP con STARTTLS (en Gmail, contraseña de aplicación).

**Vigilancia externa.** Si el Debian se apaga, nadie avisa. Con `FIN_ALERT_HEALTHCHECK_URL`
(p. ej. healthchecks.io, gratuito) el pipeline hace un *ping* al terminar (o `/fail` si algo
falló) y el servicio te avisa si un día no llega.

## Ejecución: modos

| `FIN_EXEC_MODE` | Qué hace |
|---|---|
| `off` (defecto) | Nada. El informe y las alertas siguen funcionando. |
| `manual` | Cada noche calcula el plan contra tu cartera real (la que apuntas tú) y te lo manda. Ejecutas en la app de IBKR y apuntas cada ejecución con `fin exec fill`. |
| `ibkr` + cuenta paper de IBKR | Envía las órdenes solo a la cuenta paper (DU…) a las 16:45. Sin aprobación. |
| `ibkr` + cuenta real | Exige `FIN_EXEC_LIVE=true`, `FIN_EXEC_ACCOUNT` igual a la cuenta conectada y **aprobar cada plan** (`fin exec approve`). |

Recorrido recomendado: `manual` unas semanas (sin riesgo operativo, mides *slippage* real) →
`ibkr` con cuenta paper de IBKR (comprueba la automatización de punta a punta) → cuenta real
con aprobación. Desactivar la aprobación con cuenta real requiere `FIN_EXEC_LIVE=true` y
`FIN_EXEC_REQUIRE_APPROVAL=false` explícitos.

## Ciclo diario

| Hora (Madrid) | Paso | Qué |
|---|---|---|
| 03:15 | `fin run daily` | … paper → informe → `exec sync` (ayer) → `exec plan` (hoy) → alertas |
| mañana | tú | Alerta con el plan; si es cuenta real, `fin exec approve <plan>` por SSH |
| 16:45 | `finance-exec-submit` | Ventas, espera hasta 10 min a que se ejecuten, compras con el efectivo real |
| 17:30 | Xetra cierra | Las órdenes no ejecutadas caducan (válidas solo el día) |
| 17:50 | `finance-exec-sync` | Ejecuciones, estado final, *slippage*, foto de la cuenta; alerta si algo quedó a medias |

**Qué se opera.** El plan toma la cartera objetivo vigente del paper trading de la estrategia
de referencia (`rot_v1b`, o `FIN_EXEC_STRATEGY`) y la aplica a la cuenta real con los
**UCITS de Xetra en EUR** del mapeo fiscal (`fiscal/config.py`, ampliable en
`config/execution.yaml`): XLK → ZPDT, XLE → ZPDE, XLV → ZPDH, SPY → SXR8, GLD → 4GLD…
Un minorista de la UE no puede comprar los ETF de EE. UU.

**Cuándo hay órdenes.** Cuando el paper trading opera en esa sesión (revisión mensual, cambio de
régimen, deriva > banda) **o** cuando la cuenta real se ha separado del objetivo más de
`FIN_EXEC_BAND` (5 %): órdenes que no se ejecutaron, aportaciones nuevas… Lo pendiente se
replanifica sola cada noche contra el objetivo, sin arrastrar órdenes viejas.

**Cómo se dimensiona.** Valor gestionado = efectivo + posiciones de los instrumentos del mapa
(lo que tengas fuera del sistema no se toca ni cuenta). Acciones enteras, redondeadas a la baja;
ventas primero; las compras no superan el efectivo tras vender menos un colchón del 1 % (nunca
margen ni cortos). Órdenes < 250 € no se envían (la comisión mínima de 3 € pesaría > 1,2 %),
salvo para cerrar una posición entera.

**Órdenes.** Límite a referencia ± 30 pb (punto medio bid/ask), válidas el día, horario
regular. Cada una lleva `orderRef = fin:<plan>:<símbolo>`: si el job se interrumpe y se repite,
no se duplica nada.

## Salvaguardas

No se envía ninguna orden si:

- hay **parada manual** (`fin exec halt`; también cancela las órdenes vivas);
- el informe está en **rojo**: precios atrasados respecto a la última sesión cerrada o algún
  job del pipeline fallido;
- la cuenta conectada es **real** sin `FIN_EXEC_LIVE=true`, no coincide con `FIN_EXEC_ACCOUNT`,
  o no es la cuenta para la que se hizo el plan;
- el plan es de **otra sesión** o Xetra está **cerrado**;
- un símbolo con peso objetivo **no tiene instrumento operable** (ticker + ISIN verificado):
  el plan entero queda bloqueado, no se compra una cartera a medias;
- una orden supera el **80 %** del valor gestionado o el plan rota más del **105 %** (topes
  contra errores de precio o cantidad, no contra la estrategia).

Y se omite una orden concreta si el precio se ha movido > 3 % desde el plan o no hay cotización.
Todo bloqueo genera alerta crítica y queda registrado en `exec_plans.blockers`.

**ISIN pendientes.** Hoy están verificados SPY, XLK, XLE, XLV y GLD. XLF, XLI, XLP, XLU, XLY y
XLC tienen ticker pero no ISIN; XLB, XLRE e IEF no tienen instrumento. Antes de operar,
busca cada uno en IBKR y añádelo a `config/execution.yaml`. Si el objetivo pide uno que falta,
recibirás una alerta de plan bloqueado: no es un error, es la salvaguarda funcionando.

## Registro (backlog para mejorar el modelo)

| Dataset | Qué |
|---|---|
| `exec_plans` | Un plan por sesión: estado, motivo, bloqueos, valor, desviación, rotación, aprobación. Cada cambio de estado es una fila nueva (historial completo). |
| `exec_orders` | Órdenes de cada plan: referencia, límite, ejecutado, precio medio, comisión, *slippage* en pb (positivo = en contra). |
| `exec_fills` | Cada ejecución (del bróker o apuntada a mano) y los movimientos de efectivo. |
| `exec_account` | Foto diaria de la cuenta: valor, efectivo, posiciones, aportaciones. |
| `alerts_sent` | Cada alerta, canales que la entregaron y errores. |

El informe diario añade la sección **3b. Ejecución real**: último plan y sus órdenes,
*slippage* medio ponderado, comisiones, y rentabilidad real (EUR, descontando aportaciones)
frente al paper en el mismo periodo (USD). La diferencia mezcla tres cosas que conviene
separar más adelante: el tipo EUR/USD, el coste de los UCITS (ver fiscalidad) y el retraso de
ejecutar en Xetra a las 16:45 en lugar de al cierre de Nueva York.

## Instalar IB Gateway en el Debian

1. **Usuario de API.** En IBKR (Settings → Users & Access Rights) crea un usuario secundario
   solo para la API, sin permiso de retirar fondos. Activa la cuenta paper (Settings → Paper
   Trading Account).
2. **Gateway en Docker** (imagen `gnzsnz/ib-gateway`, con IBC para el inicio de sesión
   automático):

   ```bash
   sudo apt install docker.io docker-compose-plugin
   sudo install -m 600 deploy/ibgateway/ibgateway.env.example /etc/finance/ibgateway.env
   sudoedit /etc/finance/ibgateway.env       # usuario, contraseña, TRADING_MODE=paper
   cd deploy/ibgateway && sudo docker compose --env-file /etc/finance/ibgateway.env up -d
   ```

   Los puertos solo escuchan en `127.0.0.1` (la API de IB no cifra ni autentica): 4002 paper,
   4001 real. La cuenta real pide confirmar el 2FA en IBKR Mobile en cada inicio de sesión
   completo (aprox. semanal); el reinicio diario automático (`AUTO_RESTART_TIME`) lo evita
   entre medias.
3. **Configuración** en `/etc/finance/finance.env`: `FIN_EXEC_MODE=ibkr`, `FIN_IBKR_PORT=4002`
   y los canales de alerta. Vuelve a ejecutar `deploy/bootstrap.sh` para instalar el extra
   `broker` y activar los timers `finance-exec-*`.
4. **Prueba**: `fin exec plan --force` y `fin exec status`. Si el Gateway no responde, el plan
   queda bloqueado con el motivo y te llega la alerta.

Para pasar a la cuenta real: `TRADING_MODE=live` y `FIN_IBKR_PORT=4001`, más
`FIN_EXEC_LIVE=true` y `FIN_EXEC_ACCOUNT=U…`. Sin esas dos líneas, el sistema se niega.

## Límites conocidos

- **Horario.** El paper ejecuta al cierre de Nueva York (22:00 Madrid); la cuenta real, en
  Xetra hacia las 16:45. Con señales mensuales el efecto es pequeño, pero el *slippage* medido
  es frente al precio de Xetra al planificar, no frente al paper.
- **Festivos distintos.** Si Xetra abre y Nueva York no (o al revés), el plan va a la siguiente
  sesión de Xetra tras la decisión.
- **Ejecuciones de días anteriores.** La API de IBKR solo devuelve las ejecuciones del día: si el
  servidor estuvo apagado en el `sync`, apunta lo que falte con `fin exec fill`.
- **Impuestos.** El plan no tiene en cuenta la regla de los dos meses ni el FIFO: la
  simulación fiscal (`docs/fiscalidad.md`) mide su coste, pero el plan aún no lo evita.
- **Opciones.** La ejecución cubre ETFs (UCITS). Las opciones siguen fuera del alcance.
