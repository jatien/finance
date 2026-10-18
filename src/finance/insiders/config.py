"""Paràmetres de les senyals d'insiders (fixats abans de mirar els resultats de 2019+)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

# Esdeveniments per acció, de més a menys específic
EVENTS = ("cluster_buy", "ceo_cfo_buy", "buy", "cluster_sell")
EVENT_LABELS = {
    "cluster_buy": "Compra en grup (≥3 insiders en 30 dies)",
    "ceo_cfo_buy": "Compra del CEO/CFO",
    "buy": "Compra oportunista (qualsevol directiu/conseller)",
    "cluster_sell": "Venda en grup (≥3 insiders en 30 dies)",
}

EVENT_SHORT = {"cluster_buy": "compra grup", "ceo_cfo_buy": "compra CEO/CFO", "buy": "compra",
               "cluster_sell": "venda grup"}


@dataclass(frozen=True)
class InsiderConfig:
    name: str = "ins_v1"
    start: date = date(2006, 1, 1)

    # Operació "informativa": compra (P) o venda (S) en mercat obert d'un conseller o
    # directiu, fora d'un pla 10b5-1, d'un insider no rutinari i d'import mínim.
    min_value: float = 10_000.0
    # Rutinari (Cohen, Malloy i Pomorski, 2012): ha operat el mateix mes natural de
    # cadascun dels N anys anteriors. Amb menys de N anys d'historial: "nou".
    routine_years: int = 3

    # Compra/venda en grup: ≥ min insiders diferents en una finestra de dies naturals
    cluster_window_days: int = 30
    cluster_min_insiders: int = 3
    # Un cop emès, no es torna a emetre per la mateixa acció fins passats N dies
    cooldown_days: int = 90

    # Amplitud (mercat i sectors): insiders comprant/venent a la finestra mòbil
    breadth_window: int = 63  # sessions
    breadth_min_history: int = 252

    # Avaluació
    horizons: tuple[int, ...] = (5, 21, 63, 126)
    hold_sessions: int = 63  # cartera en temps de calendari
    design_end: date = date(2018, 12, 31)  # disseny ≤ aquesta data; comprovació després
    adv_window: int = 60

    def to_params(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in d.items()}
