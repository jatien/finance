"""Pla d'execució: de la cartera objectiu (pesos) a ordres en accions senceres sobre el compte real.

Funcions pures (sense lake ni broker): el job els passa la foto del compte i les cotitzacions.

Regles:
- Valor gestionat = efectiu + posicions dels instruments del mapa. Les posicions d'altres
  instruments (comprades fora del sistema) no es toquen ni compten.
- S'opera si el paper trading opera aquesta sessió (`force`) o si el compte real s'ha
  desviat de l'objectiu més que la banda (ordres no executades, aportacions...).
- Vendes primer. Les compres es limiten a l'efectiu disponible després de vendre (menys un
  coixí): mai marge ni curts. Accions senceres, arrodonides a la baixa.
- Ordres per sota de l'import mínim no s'envien (la comissió mínima pesaria massa), excepte
  les sortides completes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from finance.portfolio.config import CASH
from finance.providers.base import AccountSnapshot, BrokerQuote, TradableInstrument

EPS = 1e-9


@dataclass(frozen=True)
class PlanLimits:
    band: float = 0.05
    min_order: float = 250.0
    max_order_pct: float = 0.8
    max_turnover: float = 1.05
    cash_buffer: float = 0.01


@dataclass
class PlannedOrder:
    symbol: str
    instrument: TradableInstrument
    side: str  # BUY | SELL
    quantity: int
    ref_price: float
    current_weight: float
    target_weight: float
    ref: str

    @property
    def amount(self) -> float:
        return self.quantity * self.ref_price


@dataclass
class Plan:
    plan_id: str
    orders: list[PlannedOrder] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    net_liquidation: float = 0.0
    managed_value: float = 0.0
    cash: float = 0.0
    max_deviation: float = float("nan")
    needs_trade: bool = False
    current: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    target: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def turnover(self) -> float:
        return sum(o.amount for o in self.orders) / self.managed_value if self.managed_value > 0 else 0.0

    @property
    def sells(self) -> list[PlannedOrder]:
        return [o for o in self.orders if o.side == "SELL"]

    @property
    def buys(self) -> list[PlannedOrder]:
        return [o for o in self.orders if o.side == "BUY"]


def order_ref(plan_id: str, symbol: str) -> str:
    return f"fin:{plan_id}:{symbol}"


def build_plan(plan_id: str, target: pd.Series, snap: AccountSnapshot, quotes: dict[str, BrokerQuote],
               instruments: dict[str, TradableInstrument], missing: dict[str, str] | None = None, *,
               force: bool, limits: PlanLimits | None = None) -> Plan:
    """`target`: pesos per sèrie del senyal (inclou CASH, suma 1). `instruments`: només els operables."""
    missing = missing or {}
    limits = limits or PlanLimits()
    plan = Plan(plan_id=plan_id, net_liquidation=snap.net_liquidation, cash=snap.cash)
    target = target.fillna(0.0).astype(float)
    target = target[target.abs() > EPS]
    by_ticker = {inst.ticker: s for s, inst in instruments.items()}

    held: dict[str, float] = {}
    for p in snap.positions:
        s = by_ticker.get(p.ticker)
        if s is None:
            plan.notes.append(f"posició fora del sistema: {p.ticker} ({p.quantity:g}); no es toca")
            continue
        held[s] = held.get(s, 0.0) + p.quantity
    if any(q < -EPS for q in held.values()):
        plan.blockers.append("hi ha posicions curtes al compte: revisa'l abans d'operar")

    symbols = sorted((set(target.index) - {CASH}) | set(held))
    prices: dict[str, float] = {}
    for s in symbols:
        inst = instruments.get(s)
        if inst is None:
            plan.blockers.append(f"{s}: no operable ({missing.get(s, 'sense instrument al mapa')})")
            continue
        px = quotes.get(inst.ticker).ref if inst.ticker in quotes else None
        if px is None:
            plan.blockers.append(f"{s} ({inst.ticker}): sense cotització")
            continue
        prices[s] = px
    if plan.blockers:
        return plan

    value = snap.cash + sum(held.get(s, 0.0) * prices[s] for s in held)
    plan.managed_value = value
    if value <= 0:
        plan.blockers.append("valor gestionat ≤ 0: no hi ha capital per operar")
        return plan
    cur = pd.Series({s: held.get(s, 0.0) * prices[s] / value for s in symbols}, dtype=float)
    cur[CASH] = snap.cash / value
    tgt = target.reindex(cur.index.union(target.index)).fillna(0.0)
    cur = cur.reindex(tgt.index).fillna(0.0)
    plan.current, plan.target = cur, tgt
    plan.max_deviation = float((cur - tgt).abs().max())
    plan.needs_trade = force or plan.max_deviation > limits.band
    if not plan.needs_trade:
        return plan

    sells, buys = [], []
    for s in symbols:
        px, q0 = prices[s], held.get(s, 0.0)
        inst = instruments[s]
        if tgt.get(s, 0.0) <= EPS:
            if q0 > EPS:  # sortida completa: sempre, encara que sigui petita
                sells.append(PlannedOrder(s, inst, "SELL", int(round(q0)), px, cur[s], 0.0, order_ref(plan_id, s)))
            continue
        delta = tgt[s] * value / px - q0
        q = math.floor(abs(delta) + 1e-9)
        if q == 0 or q * px < limits.min_order:
            continue
        o = PlannedOrder(s, inst, "SELL" if delta < 0 else "BUY", q, px, cur[s], float(tgt[s]), order_ref(plan_id, s))
        (sells if delta < 0 else buys).append(o)

    # Compres finançades amb l'efectiu després de vendre, menys el coixí
    budget = snap.cash + sum(o.amount for o in sells) - limits.cash_buffer * value
    need = sum(o.amount for o in buys)
    if buys and need > budget:
        k = max(budget, 0.0) / need
        plan.notes.append(f"compres reduïdes al {k:.0%}: efectiu insuficient després de vendre")
        for o in buys:
            o.quantity = math.floor(o.quantity * k + 1e-9)
        buys = [o for o in buys if o.quantity > 0 and o.amount >= limits.min_order]
    plan.orders = sorted(sells, key=lambda o: -o.amount) + sorted(buys, key=lambda o: -o.amount)

    # Topalls contra errors (preu o quantitat absurds), no contra l'estratègia
    for o in plan.orders:
        if o.amount > limits.max_order_pct * value + EPS:
            plan.blockers.append(f"{o.symbol}: ordre de {o.amount:,.0f} > {limits.max_order_pct:.0%} del valor gestionat")
    if plan.turnover > limits.max_turnover + EPS:
        plan.blockers.append(f"rotació del pla {plan.turnover:.0%} > {limits.max_turnover:.0%}")
    return plan
