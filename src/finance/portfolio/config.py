"""Paràmetres de la rotació sectorial i de la construcció de cartera.

Com a les capes anteriors: valors convencionals fixats abans del backtest, sense optimitzar.
Qualsevol canvi exigeix un nom nou (`rot_v2`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CASH = "CASH"  # actiu sintètic: rendeix T-bills 3M (FRED DGS3MO)


@dataclass(frozen=True)
class PortfolioConfig:
    name: str = "rot_v1"
    regime_set: str = "rules_v1"
    feature_set: str = "core_v1"

    # Univers ---------------------------------------------------------------------------
    sector_group: str = "sectors"
    market: str = "SPY"
    defensive: tuple[str, ...] = ("XLP", "XLU", "XLV")

    # Selecció de sectors (cada inici de mes i cada canvi de règim) ----------------------
    top_k: int = 3
    # Puntuació = rang(momentum 12-1 / vol 1 any) + rs_weight · rang(força relativa 6m)
    rs_weight: float = 0.25
    # En règim baixista s'afegeix low_beta_weight · rang(-beta): inclinació defensiva
    defensive_tilt: bool = True
    low_beta_weight: float = 1.0
    # Selecció conscient de la correlació: un sector no entra si la seva correlació (126
    # sessions) amb algun ja triat és >= max_pair_corr. Si no n'hi ha prou, s'omplen les
    # places amb els millors restants (el filtre és una preferència, no redueix el nombre).
    corr_window: int = 126
    max_pair_corr: float = 0.80
    # Pesos dins del bloc de sectors: inversos a la volatilitat de 3 mesos
    inverse_vol: bool = True

    # Cobertura (la part 1 - exposició) ---------------------------------------------------
    # Si la correlació SPY-TLT de 3 mesos < bond_hedge_max_corr, els bons cobreixen →
    # bond_hedge. Si no (p. ex. 2022), or + cash a parts iguals. Sense dades → cash.
    hedge: bool = True
    bond_hedge: str = "IEF"  # 7-10 anys: cobertura amb menys risc de durada que TLT
    gold: str = "GLD"
    bond_hedge_max_corr: float = 0.0
    gold_share: float = 0.5

    # Execució -----------------------------------------------------------------------------
    cost_bps: float = 5.0  # per unitat de nominal negociat (sense comptar el cash)
    # Banda de no-negociació: en una data de revisió només es reequilibra si algun pes
    # s'allunya més d'aquesta quantitat de l'objectiu (estabilitat i menys costos)
    band: float = 0.02

    extra: dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        d = asdict(self)
        d["defensive"] = list(self.defensive)
        return d

    def variant(self, **changes: Any) -> PortfolioConfig:
        d = asdict(self)
        d.update(changes)
        d["defensive"] = tuple(d["defensive"])
        return PortfolioConfig(**d)
