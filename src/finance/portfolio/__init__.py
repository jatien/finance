"""Capes 4-5: rotació sectorial per règim, cobertura i cartera objectiu.

    config.py       paràmetres (rot_v1)
    construct.py    puntuació i selecció de sectors, cobertura, objectius diaris
    backtest.py     motor walk-forward: deriva de pesos, banda de no-negociació, costos
    evaluation.py   referències, ablacions, mètriques, episodis
    report.py       informe HTML
    job.py          job `portfolio` (registre de decisions al lake) i càrrega d'entrades
"""

from finance.portfolio.config import CASH, PortfolioConfig
from finance.portfolio.construct import PortfolioInputs, Targets, build_targets

__all__ = ["CASH", "PortfolioConfig", "PortfolioInputs", "Targets", "build_targets"]
