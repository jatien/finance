from datetime import date

import pytest

from finance.calendar import is_open, last_completed_session


@pytest.mark.parametrize(("now", "expected"), [
    ("2026-10-15 21:00", date(2026, 10, 15)),  # dijous després del tancament (17:00 NY)
    ("2026-10-15 15:00", date(2026, 10, 14)),  # dijous en sessió: la barra d'avui és parcial
    ("2026-10-15 12:00", date(2026, 10, 14)),  # abans de l'obertura
    ("2026-10-17 12:00", date(2026, 10, 16)),  # dissabte
    ("2026-11-26 22:00", date(2026, 11, 25)),  # Thanksgiving (festiu)
    ("2026-10-16 01:00", date(2026, 10, 15)),  # matinada UTC = vespre NY
])
def test_last_completed_session(now, expected):
    assert last_completed_session("XNYS", now) == expected


def test_is_open():
    assert is_open("XNYS", "2026-10-15 15:00")
    assert not is_open("XNYS", "2026-10-15 21:00")
    assert not is_open("XNYS", "2026-10-17 15:00")
