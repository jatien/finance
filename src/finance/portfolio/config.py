"""Paràmetres de la rotació sectorial i de la construcció de cartera.

Com a les capes anteriors: valors convencionals fixats abans del backtest, sense optimitzar.
Qualsevol canvi exigeix un nom nou (`rot_v3`...). `ROT_V2` es va fixar abans d'executar-ne
cap backtest (pre-registre a docs/portfolio.md); no s'ha d'ajustar mirant-ne els resultats.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CASH = "CASH"  # actiu sintètic: rendeix T-bills 3M (FRED DGS3MO)

SCORES = ("risk_adj", "risk_adj_low_beta", "trend")


@dataclass(frozen=True)
class SleeveRule:
    """Com es construeix el bloc de renda variable en un règim (o tendència).

    bloc_RV = core_share · mercat + (1 - core_share) · satèl·lit

    - score: puntuació del satèl·lit.
        risk_adj           rang(momentum 12-1 / vol 1 any) + rs_weight · rang(força relativa 6m)
        risk_adj_low_beta  l'anterior + low_beta_weight · rang(-beta)   (inclinació defensiva)
        trend              rang(momentum 12-1) + rs_weight · rang(força relativa 6m): sense dividir
                           per la volatilitat, per no inclinar-se estructuralment cap als defensius
    - abs_mom_filter: només són candidats els sectors amb momentum 12-1 > 0 (momentum absolut).
      Les places que queden buides van al mercat (nucli), no a cash: la quantitat de risc la
      decideix el règim; el filtre només decideix on es posa.
    """

    core_share: float = 0.0
    score: str = "risk_adj"
    abs_mom_filter: bool = False

    def __post_init__(self) -> None:
        if self.score not in SCORES:
            raise ValueError(f"score desconegut: {self.score} (vàlids: {SCORES})")
        if not 0.0 <= self.core_share <= 1.0:
            raise ValueError("core_share ha d'estar entre 0 i 1")


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
    # Confirmació ràpida (rot_v2): els bons només cobreixen si TAMBÉ la correlació SPY-TLT de
    # `hedge_fast_window` sessions és < bond_hedge_max_corr. None = només la de 3 mesos (rot_v1).
    # Motiu: el 2022 la correlació de 3 mesos encara era negativa quan bons i accions ja queien junts.
    hedge_fast_window: int | None = None

    # Construcció per règim ----------------------------------------------------------------
    # Parelles (clau, SleeveRule). La clau és un règim ("bull_volatile") o una tendència
    # ("bull"/"bear"); el règim té prioritat. None = comportament de rot_v1: sense nucli,
    # puntuació risk_adj i, si defensive_tilt, risk_adj_low_beta en tendència baixista.
    playbook: tuple[tuple[str, SleeveRule], ...] | None = None

    # Execució -----------------------------------------------------------------------------
    cost_bps: float = 5.0  # per unitat de nominal negociat (sense comptar el cash)
    # Banda de no-negociació: en una data de revisió només es reequilibra si algun pes
    # s'allunya més d'aquesta quantitat de l'objectiu (estabilitat i menys costos)
    band: float = 0.02

    extra: dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        d = asdict(self)
        d["defensive"] = list(self.defensive)
        d["playbook"] = None if self.playbook is None else {k: asdict(r) for k, r in self.playbook}
        return d

    def variant(self, **changes: Any) -> PortfolioConfig:
        d = {f: getattr(self, f) for f in self.__dataclass_fields__}
        d.update(changes)
        d["defensive"] = tuple(d["defensive"])
        pb = d["playbook"]
        if isinstance(pb, dict):
            d["playbook"] = tuple(pb.items())
        return PortfolioConfig(**d)

    def rule_for(self, regime: str | None, trend: str | None) -> SleeveRule:
        """Regla de construcció del bloc de renda variable per a un règim i una tendència."""
        if self.playbook is None:
            bear = trend == "bear" and self.defensive_tilt
            return SleeveRule(score="risk_adj_low_beta" if bear else "risk_adj")
        rules = dict(self.playbook)
        if regime in rules:
            return rules[regime]
        if trend in rules:
            return rules[trend]
        raise KeyError(f"{self.name}: cap regla per al règim {regime!r} ni la tendència {trend!r}")


# rot_v2 — construcció condicionada al règim (setmana 6). Pre-registrada: vegeu docs/portfolio.md.
#   Tendència alcista: 70 % mercat (SPY) + 30 % satèl·lit de momentum pur amb filtre absolut.
#   Tendència baixista: 50 % mercat + 50 % satèl·lit defensiu (rot_v1 en baixista).
#   Crisi: l'exposició és 0 (rules_v1), tot en cobertura; la regla no s'arriba a fer servir.
#   Cobertura: bons només si la correlació accions-bons és negativa a 3 mesos I a 1 mes.
ROT_V2 = PortfolioConfig(
    name="rot_v2",
    playbook=(
        ("bull", SleeveRule(core_share=0.70, score="trend", abs_mom_filter=True)),
        ("bear", SleeveRule(core_share=0.50, score="risk_adj_low_beta")),
    ),
    hedge_fast_window=21,
)

ROT_V1 = PortfolioConfig()

# Estratègies que el job `portfolio` escriu cada dia al registre de decisions (paper trading).
# La primera és la de referència per a `fin portfolio show` i l'informe.
ACTIVE: tuple[PortfolioConfig, ...] = (ROT_V2, ROT_V1)
STRATEGIES: dict[str, PortfolioConfig] = {c.name: c for c in ACTIVE}
