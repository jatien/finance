"""Capa 3: règims de mercat per regles (tendència × volatilitat + crisi) i la seva validació.

    config.py       paràmetres i mapa d'exposició
    rules.py        màquines d'estats amb histèresi i classificador
    evaluation.py   estadístiques per règim, backtest de superposició, ablacions, sensibilitat
    report.py       informe HTML
    job.py          job `regimes` (escriu al lake) i càrrega d'entrades de l'avaluació
"""

from finance.regimes.config import REGIMES, RegimeConfig
from finance.regimes.rules import classify, hysteresis

__all__ = ["REGIMES", "RegimeConfig", "classify", "hysteresis"]
