"""Paràmetres del conjunt de features.

Tots els paràmetres són convencionals (200 sessions, 12-1 mesos, 63 sessions ≈ 3 mesos...)
i NO s'optimitzen sobre el backtest: amb només 4-5 mercats baixistes des de 1993, ajustar
finestres seria sobreajust. La sensibilitat es mesura després variant-los en bloc.

Si es canvia qualsevol paràmetre, cal canviar `name` (p. ex. `core_v2`): el dataset
guarda el nom i cada versió conviu al lake amb la seva pròpia història.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Sessions per any de borsa (anualització de volatilitats)
TRADING_DAYS = 252


@dataclass(frozen=True)
class FeatureConfig:
    name: str = "core_v1"

    # Instruments ------------------------------------------------------------------
    market: str = "SPY"  # proxy del mercat (retorn total, des de 1993)
    equal_weight: str = "RSP"  # S&P 500 equiponderat (amplitud: RSP vs SPY)
    long_bond: str = "TLT"
    mid_bond: str = "IEF"
    high_yield: str = "HYG"
    vix: str = "^VIX"
    vix3m: str = "^VIX3M"
    defensive: tuple[str, ...] = ("XLP", "XLU", "XLV")
    cyclical: tuple[str, ...] = ("XLY", "XLK", "XLI", "XLF")
    # Grups de universe.yaml que formen el panell de rotació
    panel_groups: tuple[str, ...] = ("sectors", "industries")
    # Grup que defineix l'amplitud sectorial i la matriu de correlació sectorial
    breadth_group: str = "sectors"
    # Sèries FRED (via macro_observations, point-in-time)
    macro_series: dict[str, str] = field(default_factory=lambda: {
        "baa_spread": "BAA10Y",  # Moody's Baa - Treasury 10Y, des de 1986
        "curve_10y3m": "T10Y3M",  # pendent de la corba, des de 1982
        "nfci": "NFCI",  # condicions financeres (revisable: només amb vintages reals)
    })
    # Retard de publicació de les sèries FRED sense vintages: el valor del dia d es
    # publica l'endemà (FRED actualitza les sèries diàries al matí del dia hàbil següent).
    macro_lag_sessions: int = 1

    # Finestres (sessions) ---------------------------------------------------------
    sma_long: int = 200
    sma_short: int = 50
    mom_long: int = 252  # 12 mesos
    mom_skip: int = 21  # salta l'últim mes (reversió a curt termini)
    mom_mid: int = 126  # 6 mesos
    mom_short: int = 63  # 3 mesos
    vol_short: int = 21
    vol_mid: int = 63
    vol_long: int = 252
    dd_window: int = 252
    corr_short: int = 63
    corr_long: int = 252  # absorption ratio
    absorption_k_frac: float = 0.2  # fracció de vectors propis (Kritzman: 1/5)
    beta_window: int = 252

    # Normalització ------------------------------------------------------------------
    pct_min_periods: int = 252  # percentil expansiu: mínim un any d'història de la feature

    # Amplitud del S&P 500: mínim de membres amb dades per publicar el valor
    sp500_min_coverage: float = 0.5

    def to_params(self) -> dict[str, Any]:
        d = asdict(self)
        d["defensive"] = list(self.defensive)
        d["cyclical"] = list(self.cyclical)
        d["panel_groups"] = list(self.panel_groups)
        return d
