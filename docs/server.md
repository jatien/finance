# Servidor Debian: instalación, SSH y operación

El ordenador Debian actúa como servidor de datos: ejecuta las ingestas programadas, guarda el
data lake y el registro de ejecuciones. Tú trabajas desde tu portátil por SSH (terminal o
VSCode Remote-SSH).

## 1. Acceso SSH por clave

En el **portátil**:

```bash
ssh-keygen -t ed25519 -C "javi@portatil"          # si aún no tienes clave
ssh-copy-id javi@IP_DEL_DEBIAN                     # copia la clave pública
```

Añade un alias en `~/.ssh/config` del portátil:

```
Host finance
    HostName IP_DEL_DEBIAN        # o el nombre Tailscale (ver más abajo)
    User javi
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
```

Ahora `ssh finance` entra sin contraseña.

### Endurecer SSH (solo después de comprobar que la clave funciona)

En el **Debian**, edita `/etc/ssh/sshd_config.d/10-finance.conf`:

```
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
AllowUsers javi
```

```bash
sudo sshd -t && sudo systemctl reload ssh    # -t valida la sintaxis antes de recargar
```

Mantén una sesión SSH abierta mientras lo pruebas desde otra terminal, para no quedarte fuera.

### Acceso desde fuera de casa: Tailscale (recomendado)

En lugar de abrir el puerto 22 en el router, instala [Tailscale](https://tailscale.com/download/linux)
en el Debian y en el portátil. El Debian queda accesible en una red privada cifrada
(`ssh javi@debian-finance`) sin exponer nada a internet.

Cortafuegos básico:

```bash
sudo apt install ufw
sudo ufw default deny incoming && sudo ufw allow OpenSSH && sudo ufw enable
```

## 2. Instalación

```bash
ssh finance
git clone https://github.com/jatien/finance.git /tmp/finance
sudo FIN_USER=$USER bash /tmp/finance/deploy/bootstrap.sh
```

El script es idempotente: deja el código en `/srv/finance/app`, los datos en
`/srv/finance/data`, la configuración en `/etc/finance/finance.env` y las unidades systemd
activadas. Para **actualizar** el código basta con volver a ejecutarlo
(`sudo FIN_USER=$USER bash /srv/finance/app/deploy/bootstrap.sh`).

Después:

1. Pon tu clave de FRED en `/etc/finance/finance.env` → `FIN_FRED_API_KEY=...`
   (`sudo nano /etc/finance/finance.env`; el fichero es `640 root:<usuario>` y nunca va al
   repositorio). Sin ella, las series revisables (empleo, IPC...) no tienen vintages y los
   backtests macro tendrían look-ahead.
2. Identifícate ante la SEC para la ingesta de insiders (Form 4) en el mismo fichero →
   `FIN_SEC_USER_AGENT="Tu Nombre tu@correo.com"`. Sin ella el job `insiders` se omite. La
   primera carga del histórico (2006 → hoy, ~80 trimestres, ~1 GB de descarga) tarda unos
   20 minutos: `cd /srv/finance/app && sudo -u <usuario> .venv/bin/fin ingest insiders`.
3. Carga inicial: `sudo systemctl start finance-weekly.service` (universo + metadatos) y
   luego `sudo systemctl start finance-daily.service` (≈580 instrumentos desde 1990, unos
   minutos).

## 3. Calendario de jobs

| Timer | Cuándo | Qué hace |
|---|---|---|
| `finance-options` | L–V 10:00, 12:45, 15:30 (hora NY) | Foto de cadenas de opciones completas |
| `finance-daily` | L–V 21:15 (hora NY; 03:15 en Madrid) | Universo si tiene >7 días → precios → macro → calidad → features → regímenes → cartera → paper trading → compactación → informe diario |
| `finance-weekly` | Sábado 10:00 (Madrid) | Universo S&P 500, historial de composición, metadatos, compactación completa |
| `finance-backup` | Diario 05:30 (Madrid), tras el pipeline diario | Copia del lake (si hay destino configurado) |

Los horarios usan la zona horaria de NY, así que los cambios de horario de verano de EE. UU.
y Europa (que no coinciden) no descuadran nada. Los festivos de NYSE se detectan con el
calendario de mercado: el job sale como `skipped`.

Las fotos de opciones **no se pueden recuperar a posteriori** con fuentes gratuitas: cada
día sin foto es historia perdida. Por eso conviene que el Debian esté encendido en horario
de mercado (15:30–22:00 en Madrid).

## 4. Operación diaria

```bash
systemctl list-timers 'finance-*'               # próximas ejecuciones
journalctl -u finance-daily -n 100              # log de la última ejecución
journalctl -u 'finance-*' --since today -f      # seguir en directo

cd /srv/finance/app && set -a && . /etc/finance/finance.env && set +a
.venv/bin/fin status                            # estado del lake y últimos jobs
.venv/bin/fin dq                                # controles de calidad
.venv/bin/fin sql "SELECT * FROM job_status"
```

### Informe diario (paper trading)

Cada mañana hay un informe nuevo en `$FIN_DATA_DIR/reports/daily/`:

```bash
cat $FIN_DATA_DIR/reports/daily/latest.md       # resumen: ¿se puede operar? órdenes de hoy
.venv/bin/fin report daily --capital 25000      # regenerarlo con otro capital nocional
.venv/bin/fin paper show                        # cartera del paper trading
.venv/bin/fin paper log                         # registro de decisiones
```

`latest.html` se abre desde VSCode (Remote-SSH → clic derecho → *Open with Live Server* o
*Reveal in Explorer* y abrirlo en el navegador local). Ver [paper.md](paper.md).

Logs estructurados (JSON Lines) en `/srv/finance/data/logs/fin.jsonl` (rotación 10 × 20 MB).

## 5. VSCode Remote-SSH y SQL

1. Extensión **Remote - SSH** → *Connect to Host* → `finance` → abre `/srv/finance/app`.
2. Para consultar los datos con **SQLTools**, instala el driver de DuckDB para SQLTools y crea
   una conexión al fichero `/srv/finance/data/catalog.duckdb` en **modo solo lectura**.
   El catálogo se regenera tras cada job (sustitución atómica), así que si una vista no
   muestra datos recientes, reconecta.
3. También vale la CLI: `duckdb -readonly /srv/finance/data/catalog.duckdb`.

Vistas útiles: `prices_daily`, `option_snapshots`, `options_latest`, `macro_latest`,
`macro_asof(DATE '2020-03-15')`, `universe_current`, `universe_asof('sp500', DATE '...')`,
`instruments_current`, `prices_coverage`, `job_status`, `ingest_runs`, `dq_results`.

## 6. Copias de seguridad

En `/etc/finance/finance.env`:

```bash
# Opción A: restic (cifrado, deduplicado; disco externo, SFTP, S3/B2...)
FIN_BACKUP_RESTIC_REPO=/mnt/backup/finance-restic
RESTIC_PASSWORD_FILE=/etc/finance/restic.pass

# Opción B: rsync con snapshots por enlaces duros (directorio local o montado)
FIN_BACKUP_TARGET=/mnt/backup/finance
FIN_BACKUP_KEEP=14
```

Vuelve a ejecutar `bootstrap.sh` para activar `finance-backup.timer`. Como los Parquet son
inmutables, cada snapshot solo ocupa lo nuevo. Para restaurar: copia el snapshot a
`FIN_DATA_DIR` y ejecuta `fin catalog`.

## 7. Ampliar hardware

El lake está particionado y es solo-anexar; DuckDB lee Parquet en paralelo y por columnas.
Escala bien a decenas de GB en un solo equipo. Si en el futuro se entrena en GPU u otra
máquina, basta con montar el lake (NFS/SSHFS) o sincronizarlo con `rsync`: el formato no
depende de este servidor.

## Alertas y ejecución real (semana 11)

Con `FIN_EXEC_MODE=manual|ibkr` en `/etc/finance/finance.env`, `bootstrap.sh` activa dos
timers más: `finance-exec-submit` (16:45 Madrid, envía las órdenes) y `finance-exec-sync`
(17:50, concilia). Las alertas salen al final de `finance-daily` sin timer propio. IB Gateway
corre en Docker y solo escucha en `127.0.0.1`. Todo en [ejecucion.md](ejecucion.md).

```bash
systemctl list-timers 'finance-exec-*'
journalctl -u finance-exec-submit -n 50
```
