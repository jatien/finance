"""Comptabilitat fiscal d'un inversor a Espanya: lots FIFO, regla dels dos mesos i IRPF anual.

Unitats: participacions i euros. El cost d'un lot inclou la comissió de compra; l'import de
venda descompta la comissió de venda (com a la declaració).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from finance.fiscal.config import SpainTax

TWO_MONTHS = pd.DateOffset(months=2)


@dataclass
class Lot:
    shares: float
    cost: float  # cost total en EUR (inclou comissió i pèrdues diferides per la regla dels 2 mesos)
    date: pd.Timestamp

    @property
    def unit_cost(self) -> float:
        return self.cost / self.shares if self.shares > 0 else 0.0


@dataclass
class PendingLoss:
    """Pèrdua realitzada que encara es pot anul·lar si es recompra dins dels dos mesos següents."""
    symbol: str
    date: pd.Timestamp
    year: int
    loss: float  # < 0, part encara integrada
    shares: float  # participacions venudes amb pèrdua encara no "cobertes" per recompres


@dataclass
class YearResult:
    year: int
    gains: float  # guanys i pèrdues patrimonials nets de l'any (abans d'arrossegaments)
    interest: float  # rendiments del capital mobiliari (interessos de l'efectiu)
    base: float  # base liquidable de l'estalvi
    tax: float
    carry_used: float
    carry_left: float


@dataclass
class Ledger:
    tax: SpainTax
    lots: dict[str, deque[Lot]] = field(default_factory=dict)
    gains: dict[int, float] = field(default_factory=dict)
    interest: dict[int, float] = field(default_factory=dict)
    pending: list[PendingLoss] = field(default_factory=list)
    carry: list[tuple[int, float]] = field(default_factory=list)  # (any d'origen, pèrdua > 0)
    deferred_total: float = 0.0  # pèrdues diferides per la regla dels 2 mesos (informatiu)
    results: list[YearResult] = field(default_factory=list)

    # -- posicions -----------------------------------------------------------------------
    def shares(self, symbol: str) -> float:
        return float(sum(lot.shares for lot in self.lots.get(symbol, ())))

    def cost_basis(self, symbol: str) -> float:
        return float(sum(lot.cost for lot in self.lots.get(symbol, ())))

    def unrealized(self, symbol: str, price: float) -> float:
        return self.shares(symbol) * price - self.cost_basis(symbol)

    def buy(self, symbol: str, shares: float, cost: float, date: pd.Timestamp) -> None:
        if shares <= 0:
            return
        lot = Lot(shares, cost, date)
        self.lots.setdefault(symbol, deque()).append(lot)
        if self.tax.two_month_rule:
            self._absorb_pending(symbol, lot, date)

    def sell(self, symbol: str, shares: float, proceeds: float, date: pd.Timestamp) -> float:
        """Ven `shares` (FIFO) per `proceeds` nets. Retorna el guany/pèrdua integrat aquest any."""
        q = self.lots.get(symbol)
        held = self.shares(symbol)
        if not q or shares <= 0:
            return 0.0
        shares = min(shares, held)
        unit_px = proceeds / shares
        left, cost = shares, 0.0
        while left > 1e-12 and q:
            lot = q[0]
            take = min(lot.shares, left)
            c = lot.unit_cost * take
            cost += c
            lot.cost -= c
            lot.shares -= take
            left -= take
            if lot.shares <= 1e-12:
                q.popleft()
        if not q:
            self.lots.pop(symbol, None)
        result = unit_px * shares - cost
        year = date.year
        if result < 0 and self.tax.two_month_rule:
            result = self._two_month_before(symbol, result, shares, date)
        self.gains[year] = self.gains.get(year, 0.0) + result
        return result

    def add_interest(self, amount: float, date: pd.Timestamp) -> None:
        self.interest[date.year] = self.interest.get(date.year, 0.0) + amount

    # -- regla dels dos mesos ------------------------------------------------------------
    def _two_month_before(self, symbol: str, loss: float, sold: float, date: pd.Timestamp) -> float:
        """Recompres dels 2 mesos anteriors encara en cartera: la pèrdua proporcional es difereix."""
        recent = [lot for lot in self.lots.get(symbol, ()) if lot.date > date - TWO_MONTHS]
        bought = sum(lot.shares for lot in recent)
        frac = min(1.0, bought / sold) if sold > 0 else 0.0
        deferred = loss * frac
        if frac > 0:
            for lot in recent:  # la pèrdua diferida s'afegeix al cost: s'integra quan es venguin
                lot.cost += -deferred * lot.shares / bought
            self.deferred_total += -deferred
        rest = loss - deferred
        if rest < 0:
            self.pending.append(PendingLoss(symbol, date, date.year, rest, sold * (1 - frac)))
        return rest

    def _absorb_pending(self, symbol: str, lot: Lot, date: pd.Timestamp) -> None:
        """Una compra dins dels 2 mesos posteriors a una venda amb pèrdua l'anul·la (proporcionalment)."""
        self.pending = [p for p in self.pending if date - TWO_MONTHS < p.date]
        avail = lot.shares
        for p in self.pending:
            if p.symbol != symbol or avail <= 1e-12 or p.shares <= 1e-12:
                continue
            cover = min(avail, p.shares)
            moved = p.loss * cover / p.shares  # < 0
            self.gains[p.year] = self.gains.get(p.year, 0.0) - moved
            lot.cost += -moved
            self.deferred_total += -moved
            p.loss -= moved
            p.shares -= cover
            avail -= cover

    # -- liquidació anual ----------------------------------------------------------------
    def settle(self, year: int) -> YearResult:
        """IRPF de l'any `year` (art. 49 LIRPF): compensació, arrossegament i escala."""
        g = g0 = self.gains.pop(year, 0.0)
        rcm = rcm0 = self.interest.pop(year, 0.0)
        self.carry = [(y, v) for y, v in self.carry if year - y <= self.tax.loss_carry_years and v > 1e-9]
        cap = self.tax.rcm_offset_cap * max(rcm, 0.0)
        used = 0.0
        if g < 0:
            off = min(-g, cap)
            rcm -= off
            cap -= off
            if -g - off > 1e-9:
                self.carry.append((year, -g - off))
            g = 0.0
        new_carry = []
        for y, v in self.carry:
            if y == year:
                new_carry.append((y, v))
                continue
            use = min(v, g) if g > 0 else 0.0  # primer contra guanys
            g -= use
            v -= use
            used += use
            use2 = min(v, cap)  # després contra el 25 % dels interessos
            rcm -= use2
            cap -= use2
            v -= use2
            used += use2
            if v > 1e-9:
                new_carry.append((y, v))
        self.carry = new_carry
        base = max(g, 0.0) + max(rcm, 0.0)
        res = YearResult(year, g0, max(rcm0, 0.0), base, self.tax.tax_on(base),
                         used, sum(v for _, v in self.carry))
        self.results.append(res)
        return res
