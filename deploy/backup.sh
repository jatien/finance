#!/usr/bin/env bash
# Còpia de seguretat del data lake.
#
# Opció A (recomanada): restic -> FIN_BACKUP_RESTIC_REPO (+ RESTIC_PASSWORD_FILE)
#   Snapshots xifrats i deduplicats; pot apuntar a un disc extern, SFTP o S3/B2.
# Opció B: rsync amb snapshots per enllaços durs -> FIN_BACKUP_TARGET (directori local o muntat)
#   Com que els fitxers Parquet són immutables, cada snapshot només ocupa el que ha canviat.
#
# El catàleg DuckDB no es copia: es regenera amb `fin catalog`.
set -euo pipefail

DATA_DIR="${FIN_DATA_DIR:?FIN_DATA_DIR no definida}"
DATA_DIR="${DATA_DIR/#\~/$HOME}"
KEEP="${FIN_BACKUP_KEEP:-14}"

if [[ -n "${FIN_BACKUP_RESTIC_REPO:-}" ]]; then
    export RESTIC_REPOSITORY="$FIN_BACKUP_RESTIC_REPO"
    restic snapshots &>/dev/null || restic init
    restic backup --tag finance --exclude '*.tmp' --exclude 'catalog.duckdb*' --exclude '_locks' "$DATA_DIR"
    restic forget --tag finance --keep-daily "$KEEP" --keep-weekly 8 --keep-monthly 12 --prune
    exit 0
fi

TARGET="${FIN_BACKUP_TARGET:?defineix FIN_BACKUP_TARGET o FIN_BACKUP_RESTIC_REPO}"
mkdir -p "$TARGET"
STAMP="$(date +%Y-%m-%dT%H%M)"
LATEST="$TARGET/latest"
LINK=()
[[ -d "$LATEST" ]] && LINK=(--link-dest "$(readlink -f "$LATEST")")

rsync -a --delete "${LINK[@]}" \
    --exclude '*.tmp' --exclude 'catalog.duckdb*' --exclude '_locks/' \
    "$DATA_DIR/" "$TARGET/$STAMP/"
ln -sfn "$TARGET/$STAMP" "$LATEST"

# Retenció: conserva els últims $KEEP snapshots
ls -1d "$TARGET"/20* 2>/dev/null | sort | head -n -"$KEEP" | xargs -r rm -rf
echo "còpia feta a $TARGET/$STAMP"
