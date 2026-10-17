"""Insiders (setmana 10): operacions declarades a la SEC (Form 4) com a senyal.

    config.py       paràmetres (fixats abans de la validació)
    signals.py      operacions informatives, rutinaris, esdeveniments i amplitud
    job.py          job `insider_signals` (escriu `insider_events` i `insider_breadth`)
    evaluation.py   estudi d'esdeveniments, carteres de calendari, ablacions, criteris
    report.py       informe HTML
"""

from finance.insiders.config import EVENT_LABELS, EVENTS, InsiderConfig

__all__ = ["EVENTS", "EVENT_LABELS", "InsiderConfig"]
