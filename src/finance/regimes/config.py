"""Paràmetres del classificador de règims per regles.

Valors convencionals, triats abans de mirar el backtest i NO optimitzats: des de 1993 hi ha
només 4-5 mercats baixistes, i ajustar llindars sobre tan pocs episodis és sobreajust. La
robustesa es comprova amb una graella de sensibilitat (`evaluation.sensitivity`): la regla
ha de funcionar raonablement en tot un rang, no en un punt màgic.

Qualsevol canvi de paràmetres exigeix un nom nou (`rules_v2`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Règims ordenats de menys a més risc
REGIMES: tuple[str, ...] = ("bull_calm", "bull_volatile", "bear_calm", "bear_volatile", "crisis")


@dataclass(frozen=True)
class RegimeConfig:
    name: str = "rules_v1"
    feature_set: str = "core_v1"

    # Eix de tendència: mercat vs SMA200 amb banda d'histèresi -------------------------
    # Entra en "bear" quan la distància a la SMA200 és < -trend_band durant
    # `confirm_days` sessions seguides; torna a "bull" quan és > +trend_band.
    # La banda evita el serrat (whipsaw) quan el preu oscil·la al voltant de la mitjana.
    trend_band: float = 0.02
    confirm_days: int = 3

    # Eix de volatilitat: percentil expansiu del VIX ----------------------------------
    # VIX i vol realitzada tenen rho = 0.83 (redundants): es fa servir només el VIX,
    # que és prospectiu. Histèresi en percentils.
    vol_enter_pct: float = 0.75
    vol_exit_pct: float = 0.60

    # Crisi (estrès agut): sobreescriu els altres eixos -------------------------------
    # Corba del VIX invertida (VIX/VIX3M >= 1) I deteriorament del crèdit (canvi del
    # spread Baa en 3 mesos al percentil >= crisis_credit_pct).
    # Abans del juliol de 2006 no hi ha VIX3M. La primera versió feia servir com a
    # substitut el VIX al percentil >= 0,95, però amb només 3-4 anys d'història el percentil
    # expansiu del VIX no mesura estrès agut: marcava tot el 1997 com a crisi (amb el
    # mercat +40 %). Per això el substitut està desactivat per defecte: sense corba del VIX
    # no es declara crisi (decisió per manca de dades, no per resultat del backtest).
    crisis_fallback: bool = False
    crisis_term_enter: float = 1.0
    crisis_term_exit: float = 0.95
    crisis_credit_pct: float = 0.85
    crisis_vix_pct_fallback: float = 0.95
    crisis_vix_pct_exit_fallback: float = 0.85
    crisis_confirm_exit: int = 5  # entrar és immediat; sortir exigeix confirmació

    # Exposició a renda variable per règim (la resta va a la cartera de cobertura) -----
    exposure: dict[str, float] = field(default_factory=lambda: {
        "bull_calm": 1.0,
        "bull_volatile": 0.75,
        "bear_calm": 0.5,
        "bear_volatile": 0.25,
        "crisis": 0.0,
    })

    # Interruptors per a les ablacions ------------------------------------------------
    use_vol_axis: bool = True
    use_crisis: bool = True

    def to_params(self) -> dict[str, Any]:
        return asdict(self)

    def variant(self, **changes: Any) -> RegimeConfig:
        """Còpia amb canvis (per a ablacions i sensibilitat)."""
        d = asdict(self)
        d.update(changes)
        return RegimeConfig(**d)
