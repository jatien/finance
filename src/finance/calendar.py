"""Calendaris de borsa: sessions completades, mercat obert, sessions esperades."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=16)
def get_calendar(code: str = "XNYS") -> xcals.ExchangeCalendar:
    return xcals.get_calendar(code)


def _utc(now: datetime | pd.Timestamp | None) -> pd.Timestamp:
    ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def last_completed_session(code: str = "XNYS", now: datetime | None = None) -> date:
    """Última sessió el tancament de la qual ja ha passat (evita barres diàries parcials)."""
    cal = get_calendar(code)
    ts = _utc(now)
    local_day = ts.tz_convert(cal.tz).normalize().tz_localize(None)
    local_day = min(local_day, cal.last_session)
    if cal.is_session(local_day):
        if ts >= cal.session_close(local_day):
            return local_day.date()
        return cal.previous_session(local_day).date()
    return cal.date_to_session(local_day, direction="previous").date()


def is_open(code: str = "XNYS", now: datetime | None = None) -> bool:
    return bool(get_calendar(code).is_open_on_minute(_utc(now).floor("min")))


def is_session(day: date, code: str = "XNYS") -> bool:
    return bool(get_calendar(code).is_session(pd.Timestamp(day)))


def sessions_between(start: date, end: date, code: str = "XNYS") -> pd.DatetimeIndex:
    cal = get_calendar(code)
    first = max(pd.Timestamp(start), cal.first_session)
    last = min(pd.Timestamp(end), cal.last_session)
    if first > last:
        return pd.DatetimeIndex([])
    return cal.sessions_in_range(first, last)
