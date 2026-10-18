"""Salvaguardes abans de planificar o enviar ordres reals.

Cada funció retorna una llista de bloquejos (text); buida = es pot continuar. La filosofia és
la de l'informe diari: davant del dubte, no s'opera i s'avisa.
"""

from __future__ import annotations

import pandas as pd

from finance.calendar import is_open, last_completed_session, sessions_between
from finance.providers.base import AccountSnapshot
from finance.settings import Settings

EXEC_CAL = "XETR"  # els UCITS s'operen a Xetra


def halted(settings: Settings) -> list[str]:
    f = settings.halt_file
    if f.exists():
        why = f.read_text(encoding="utf-8").strip() or "sense motiu"
        return [f"aturada manual activa ({why}): `fin exec resume` per reprendre"]
    return []


def data_health(ctx, data_date: pd.Timestamp | None) -> list[str]:
    """Els mateixos errors que posen l'informe diari en vermell (dades endarrerides, jobs fallits)."""
    from finance.paper.report import data_issues
    now = ctx.now()
    expected = pd.Timestamp(last_completed_session("XNYS", now))
    behind = 0 if data_date is None else len(sessions_between((data_date + pd.Timedelta(days=1)).date(),
                                                              expected.date()))
    return [t for sev, t in data_issues(ctx, now, data_date, expected, behind) if sev == "error"]


def account_safety(settings: Settings, snap: AccountSnapshot) -> list[str]:
    out = []
    if settings.exec_mode == "ibkr" and not snap.is_paper:
        if not settings.exec_live:
            out.append(f"compte real connectat ({snap.account}) sense FIN_EXEC_LIVE=true: no s'opera")
        elif settings.exec_account != snap.account:
            out.append(f"compte real {snap.account} ≠ FIN_EXEC_ACCOUNT ({settings.exec_account})")
    if snap.currency not in (settings.exec_currency, "BASE"):
        out.append(f"moneda base del compte {snap.currency} ≠ {settings.exec_currency}")
    if snap.net_liquidation <= 0:
        out.append("valor del compte ≤ 0")
    return out


def requires_approval(settings: Settings, snap: AccountSnapshot) -> bool:
    if settings.exec_mode != "ibkr":
        return False  # mode manual: ja l'executes tu
    if settings.exec_require_approval is not None:
        # Amb compte real l'aprovació no es pot desactivar sense FIN_EXEC_LIVE explícit
        return settings.exec_require_approval or (not snap.is_paper and not settings.exec_live)
    return not snap.is_paper


def next_exec_session(decision_date: pd.Timestamp) -> pd.Timestamp:
    """Primera sessió de Xetra posterior al tancament de decisió (data de NY)."""
    from finance.calendar import get_calendar
    cal = get_calendar(EXEC_CAL)
    return pd.Timestamp(cal.date_to_session(pd.Timestamp(decision_date) + pd.Timedelta(days=1), direction="next"))


def exec_today(now: pd.Timestamp) -> pd.Timestamp:
    from finance.calendar import get_calendar
    tz = get_calendar(EXEC_CAL).tz
    ts = pd.Timestamp(now)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return ts.tz_convert(tz).normalize().tz_localize(None)


def market_window(now: pd.Timestamp, session: pd.Timestamp) -> list[str]:
    today = exec_today(now)
    if today != pd.Timestamp(session):
        return [f"el pla és per a la sessió del {pd.Timestamp(session):%Y-%m-%d} i avui és {today:%Y-%m-%d}"]
    if not is_open(EXEC_CAL, now):
        return ["Xetra és tancat ara mateix"]
    return []
