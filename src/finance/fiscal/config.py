"""Paràmetres de l'inversor resident a Espanya amb Interactive Brokers.

Fonts (consultades el 17-oct-2026; cal revisar-les cada any):
- Escala de la base de l'estalvi 2026 (Llei 7/2024): 19 % fins a 6.000 €, 21 % fins a
  50.000 €, 23 % fins a 200.000 €, 27 % fins a 300.000 €, 30 % a partir d'aquí.
- Pèrdues patrimonials: es compensen amb guanys; amb rendiments del capital mobiliari
  (interessos, dividends) només fins al 25 % d'aquests; la resta s'arrossega 4 anys.
- Regla dels dos mesos (art. 33.5.f LIRPF): la pèrdua per la venda de valors cotitzats no
  s'integra si s'han adquirit valors homogenis en els dos mesos anteriors o posteriors;
  s'integra quan es venen aquests valors.
- FIFO obligatori per a valors homogenis (art. 37.2 LIRPF).
- ETFs: sense règim de traspassos (a diferència dels fons), cada venda tributa.
- Un minorista de la UE no pot comprar ETFs domiciliats als EUA (PRIIPs/KID): s'opera amb
  els equivalents UCITS (Irlanda) a Xetra en EUR. Els d'acumulació no reparteixen dividends.
- IBKR Fixed, Alemanya (Xetra): 0,05 % del valor, mínim 3 €, màxim 29 € per ordre
  (aproximat: confirmar a la taula oficial d'IBKR).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SAVINGS_SCALE_2026: tuple[tuple[float, float], ...] = (
    (6_000.0, 0.19), (50_000.0, 0.21), (200_000.0, 0.23), (300_000.0, 0.27), (float("inf"), 0.30))


@dataclass(frozen=True)
class SpainTax:
    scale: tuple[tuple[float, float], ...] = SAVINGS_SCALE_2026
    loss_carry_years: int = 4
    rcm_offset_cap: float = 0.25  # pèrdues patrimonials contra interessos/dividends
    two_month_rule: bool = True
    # L'impost de l'any Y es paga amb la declaració (juny de Y+1). Es modelitza com una
    # sortida de diners de l'inversor la primera sessió de juliol (no ven posicions).
    settle_month: int = 7

    def tax_on(self, base: float) -> float:
        if base <= 0:
            return 0.0
        tax, lo = 0.0, 0.0
        for hi, rate in self.scale:
            if base <= lo:
                break
            tax += (min(base, hi) - lo) * rate
            lo = hi
        return tax


@dataclass(frozen=True)
class Broker:
    """Comissió per ordre: clip(rate · import, min_fee, max_fee). Ordres < min_order no s'envien."""
    name: str = "IBKR Fixed Xetra"
    rate: float = 0.0005
    min_fee: float = 3.0
    max_fee: float = 29.0
    min_order: float = 250.0  # per sota, el mínim de 3 € és > 1,2 %: no val la pena

    def fee(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return float(min(max(self.rate * amount, self.min_fee), self.max_fee))


@dataclass(frozen=True)
class Ucits:
    """Equivalent operable des d'Espanya d'un ETF dels EUA (mateix índex)."""
    name: str
    xetra: str | None  # ticker a Xetra (EUR)
    isin: str | None  # None = no verificat: confirmar a IBKR abans d'operar
    ter: float  # TER de l'UCITS
    us_ter: float  # TER de l'ETF dels EUA que fem servir com a sèrie de preus
    equity: bool = True  # la retenció del 15 % sobre dividends dels EUA només afecta accions
    wht: float = 0.15

    def annual_drag(self, div_yield: float) -> float:
        """Cost anual addicional de l'UCITS respecte de la sèrie de retorn total de l'ETF dels EUA."""
        return (self.ter - self.us_ter) + (self.wht * max(div_yield, 0.0) if self.equity else 0.0)


# Sèrie dels EUA → UCITS. Sectors: SPDR S&P U.S. Select Sector UCITS ETF (Acc), mateixos índexs.
UCITS_MAP: dict[str, Ucits] = {
    "SPY": Ucits("iShares Core S&P 500 UCITS (Acc)", "SXR8", "IE00B5BMR087", 0.0007, 0.0945),
    "XLK": Ucits("SPDR S&P U.S. Technology Select Sector UCITS (Acc)", "ZPDT", "IE00BWBXM948", 0.0015, 0.0008),
    "XLE": Ucits("SPDR S&P U.S. Energy Select Sector UCITS (Acc)", "ZPDE", "IE00BWBXM492", 0.0015, 0.0008),
    "XLV": Ucits("SPDR S&P U.S. Health Care Select Sector UCITS (Acc)", "ZPDH", "IE00BWBXM617", 0.0015, 0.0008),
    "XLF": Ucits("SPDR S&P U.S. Financials Select Sector UCITS (Acc)", "ZPDF", None, 0.0015, 0.0008),
    "XLI": Ucits("SPDR S&P U.S. Industrials Select Sector UCITS (Acc)", "ZPDI", None, 0.0015, 0.0008),
    "XLP": Ucits("SPDR S&P U.S. Consumer Staples Select Sector UCITS (Acc)", "ZPDS", None, 0.0015, 0.0008),
    "XLU": Ucits("SPDR S&P U.S. Utilities Select Sector UCITS (Acc)", "ZPDU", None, 0.0015, 0.0008),
    "XLY": Ucits("SPDR S&P U.S. Consumer Discretionary Select Sector UCITS (Acc)", "ZPDD", None, 0.0015, 0.0008),
    "XLC": Ucits("SPDR S&P U.S. Communication Services Select Sector UCITS (Acc)", "ZPDK", None, 0.0015, 0.0008),
    "XLB": Ucits("SPDR S&P U.S. Materials Select Sector UCITS (Acc)", None, None, 0.0015, 0.0008),
    "XLRE": Ucits("UCITS de Real Estate dels EUA (per triar)", None, None, 0.0040, 0.0008),
    "IEF": Ucits("iShares $ Treasury Bond 7-10yr UCITS", None, None, 0.0007, 0.0015, equity=False),
    "GLD": Ucits("Xetra-Gold (ETC)", "4GLD", "DE000A0S9GB0", 0.0000, 0.0040, equity=False),
}


@dataclass(frozen=True)
class Investor:
    initial: float = 80_000.0
    monthly: float = 300.0  # aportació la primera sessió de cada mes
    cash_spread: float = 0.005  # IBKR remunera l'efectiu en EUR a referència - 0,5 %
    tax: SpainTax = field(default_factory=SpainTax)
    broker: Broker = field(default_factory=Broker)

    def to_params(self) -> dict[str, Any]:
        return asdict(self)
