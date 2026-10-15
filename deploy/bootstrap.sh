#!/usr/bin/env bash
# Instal·la (o actualitza) el sistema al servidor Debian. Idempotent: es pot tornar a executar.
#
#   sudo FIN_USER=javi bash deploy/bootstrap.sh
#
# Variables opcionals:
#   FIN_USER   usuari que executa els jobs (per defecte, qui ha fet sudo)
#   FIN_HOME   arrel de la instal·lació           (per defecte /srv/finance)
#   REPO_URL   repositori                           (per defecte https://github.com/jatien/finance.git)
#   BRANCH     branca a desplegar                   (per defecte main)
#   NO_TIMERS  =1 per no activar els timers de systemd
set -euo pipefail

FIN_USER="${FIN_USER:-${SUDO_USER:-}}"
FIN_HOME="${FIN_HOME:-/srv/finance}"
REPO_URL="${REPO_URL:-https://github.com/jatien/finance.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="$FIN_HOME/app"
DATA_DIR="$FIN_HOME/data"
ENV_DIR="/etc/finance"
ENV_FILE="$ENV_DIR/finance.env"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "executa-ho amb sudo"
[[ -n "$FIN_USER" ]] || die "defineix FIN_USER (usuari no root que executarà els jobs)"
id "$FIN_USER" &>/dev/null || die "l'usuari $FIN_USER no existeix"
grep -qi debian /etc/os-release || log "avís: no sembla Debian; continuo igualment"
SYSTEMD_VER=$(systemctl --version | awk 'NR==1{print $2}')
(( SYSTEMD_VER >= 235 )) || die "cal systemd >= 235 (zones horàries a OnCalendar)"

log "Paquets del sistema"
apt-get update -qq
apt-get install -y -qq git curl ca-certificates rsync tzdata >/dev/null

log "Directoris a $FIN_HOME"
install -d -o "$FIN_USER" -g "$FIN_USER" -m 750 "$FIN_HOME" "$DATA_DIR"
install -d -o root -g "$FIN_USER" -m 750 "$ENV_DIR"

as_user() { sudo -u "$FIN_USER" -H bash -lc "$*"; }

log "Codi ($REPO_URL @ $BRANCH)"
if [[ -d "$APP_DIR/.git" ]]; then
    as_user "git -C '$APP_DIR' fetch --quiet origin '$BRANCH' && git -C '$APP_DIR' checkout --quiet '$BRANCH' && git -C '$APP_DIR' pull --ff-only --quiet"
else
    as_user "git clone --quiet --branch '$BRANCH' '$REPO_URL' '$APP_DIR'"
fi

log "uv i entorn Python"
if ! as_user "command -v uv" &>/dev/null; then
    as_user "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
UV="$(as_user 'command -v uv || echo ~/.local/bin/uv')"
# uv descarrega la versió de Python de .python-version (no depèn del Python de Debian)
as_user "cd '$APP_DIR' && '$UV' sync --locked --no-dev"

log "Configuració ($ENV_FILE)"
if [[ ! -f "$ENV_FILE" ]]; then
    sed "s|^FIN_DATA_DIR=.*|FIN_DATA_DIR=$DATA_DIR|" "$APP_DIR/.env.example" > "$ENV_FILE"
    log "creat $ENV_FILE — afegeix-hi FIN_FRED_API_KEY"
fi
chown root:"$FIN_USER" "$ENV_FILE"
chmod 640 "$ENV_FILE"

log "Unitats systemd"
for unit in "$APP_DIR"/deploy/systemd/*.{service,timer}; do
    name="$(basename "$unit")"
    sed -e "s|@FIN_USER@|$FIN_USER|g" -e "s|@APP_DIR@|$APP_DIR|g" -e "s|@ENV_FILE@|$ENV_FILE|g" \
        "$unit" > "/etc/systemd/system/$name"
done
chmod +x "$APP_DIR/deploy/backup.sh"
systemctl daemon-reload

log "Inicialització del lake"
as_user "set -a; source '$ENV_FILE'; set +a; cd '$APP_DIR' && .venv/bin/fin init"

if [[ "${NO_TIMERS:-0}" != "1" ]]; then
    log "Activant timers"
    systemctl enable --now finance-daily.timer finance-options.timer finance-weekly.timer
    if grep -qE '^FIN_BACKUP_(TARGET|RESTIC_REPO)=.+' "$ENV_FILE"; then
        systemctl enable --now finance-backup.timer
    else
        log "còpies desactivades: defineix FIN_BACKUP_TARGET a $ENV_FILE i torna a executar"
    fi
fi

cat <<EOF

Fet. Pròxims passos:
  1. Clau FRED (gratuïta) a $ENV_FILE: FIN_FRED_API_KEY=...
  2. Càrrega inicial (uns minuts):  sudo systemctl start finance-weekly.service && sudo systemctl start finance-daily.service
  3. Seguiment:                      journalctl -u 'finance-*' -f
  4. Estat:                          systemctl list-timers 'finance-*'
                                     sudo -u $FIN_USER bash -c 'set -a; . $ENV_FILE; $APP_DIR/.venv/bin/fin status'
EOF
