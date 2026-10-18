"""Enviament d'alertes amb deduplicació.

Cada alerta té una clau que identifica l'esdeveniment (p. ex. `orders:rot_v1b:2026-10-16`).
Si ja s'ha lliurat bé per algun canal (`alerts_sent.ok`), no es torna a enviar: el pipeline es
pot executar diverses vegades el mateix dia sense repetir avisos. Si no s'ha pogut lliurar per
cap canal, es torna a provar a la propera execució.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from finance.providers.base import ProviderError
from finance.providers.notify import level_rank
from finance.storage.schemas import ALERTS_SENT

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    key: str
    level: str  # info | important | critical
    title: str
    body: str


def delivered_keys(ctx, keys: list[str]) -> set[str]:
    if not keys:
        return set()
    try:
        df = ctx.query("SELECT alert_key FROM alerts_sent WHERE ok AND alert_key IN (SELECT unnest(?))", [keys])
    except Exception:  # noqa: BLE001 — dataset encara buit
        return set()
    return set(df["alert_key"])


def dispatch(ctx, run, alerts: list[Alert], *, min_level: str | None = None) -> list[Alert]:
    """Envia les alertes noves de nivell ≥ min_level per tots els canals. Retorna les enviades."""
    min_level = min_level or ctx.settings.alert_min_level
    alerts = [a for a in alerts if level_rank(a.level) >= level_rank(min_level)]
    # Dins d'una mateixa crida, una clau només un cop
    alerts = list({a.key: a for a in alerts}.values())
    done = delivered_keys(ctx, [a.key for a in alerts])
    notifiers = ctx.providers.notifiers
    sent, rows = [], []
    for a in alerts:
        if a.key in done:
            continue
        if not notifiers:
            log.info("alerta sense canals configurats: %s — %s", a.title, a.body.splitlines()[0] if a.body else "")
            continue
        ok_ch, bad = [], []
        for n in notifiers:
            try:
                n.send(a.title, a.body, a.level)
                ok_ch.append(n.name)
            except (ProviderError, ValueError) as e:
                bad.append(f"{n.name}: {e}")
        rows.append({"alert_key": a.key, "level": a.level, "title": a.title, "body": a.body,
                     "delivered": ",".join(ok_ch), "failed": "; ".join(bad)[:500], "ok": bool(ok_ch),
                     "sent_at": pd.Timestamp.now(tz="UTC")})
        if ok_ch:
            sent.append(a)
        else:
            run.fail(f"alerta:{a.key}", "; ".join(bad))
    if rows:
        run.write(ALERTS_SENT, pd.DataFrame(rows), "alerts")
    return sent
