"""Simulació després d'impostos i comissions per a un inversor a Espanya.

Diferències amb el motor de pesos (`portfolio.backtest`):
- Comptabilitat en participacions i euros: preus dels EUA convertits amb EUR/USD (l'IRPF es
  calcula en euros, així que el canvi de divisa també genera guanys o pèrdues).
- Operable des d'Espanya: cada ETF dels EUA es substitueix pel seu UCITS (mateix índex), amb
  el seu cost addicional (TER i retenció del 15 % sobre dividends dins del fons).
- Comissions per ordre (mínim i màxim), ordres massa petites que no s'envien.
- Lots FIFO, regla dels dos mesos i IRPF anual amb l'escala de l'estalvi (`ledger`).
- Aportacions periòdiques. L'impost es paga cada juliol des de fora de la cartera; al final
  es liquida tot i es paga l'impost pendent. La mètrica principal és la TIR d'aquests fluxos.

Les decisions (objectius, disparador, rànquings) són les de l'estratègia amb un retard d'una
sessió, igual que el paper trading. Dues polítiques d'execució:
- "target": seguir l'objectiu com el backtest (vendre per reequilibrar quan el disparador ho diu).
- "accumulate" (acum): no vendre per reequilibrar. Les aportacions i l'efectiu sobrant van als
  actius infraponderats; només es ven quan el règim redueix la renda variable o quan un sector
  cau molt avall del rànquing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from finance.fiscal.config import UCITS_MAP, Investor
from finance.fiscal.ledger import TWO_MONTHS, Ledger
from finance.paper.book import sector_ranks
from finance.portfolio.backtest import Trigger, lagged, rolling_cov
from finance.portfolio.config import CASH, PortfolioConfig
from finance.portfolio.construct import PortfolioInputs, Targets


@dataclass(frozen=True)
class Policy:
    kind: str = "target"  # target | accumulate
    # accumulate: es ven renda variable si la cartera en té més de `expo_band` per sobre de
    # l'objectiu (règim que baixa l'exposició) i se'n compra si en té `expo_band` per sota
    expo_band: float = 0.10
    # accumulate: un sector en cartera es ven sencer si el seu rang > top_k + demote_buffer
    demote_buffer: int | None = 5
    # accumulate: en reduir, vendre primer el que menys impost genera per euro venut
    tax_aware_sells: bool = True
    # no comprar un actiu venut amb pèrdua fa menys de 2 mesos (si no, la pèrdua no compta)
    avoid_wash: bool = True
    regime_sells: bool = True  # accumulate: ablació sense vendes per règim

    def __post_init__(self) -> None:
        if self.kind not in ("target", "accumulate"):
            raise ValueError(f"política desconeguda: {self.kind}")


@dataclass
class MarketData:
    """Preus en EUR dels UCITS (sèrie de l'ETF dels EUA / EURUSD × cost UCITS) i efectiu."""
    px: pd.DataFrame  # sessions × actius (EUR)
    cash_rate: pd.Series  # tipus anual de l'efectiu (fracció)
    fx: pd.Series  # USD per EUR


def market_data(inp: PortfolioInputs, fx_usd_per_eur: pd.Series, ecb_rate_pct: pd.Series,
                dividends: pd.DataFrame | None, investor: Investor, *, ucits: bool = True) -> MarketData:
    idx = inp.index
    fx = fx_usd_per_eur.reindex(idx).ffill()
    px = inp.prices.div(fx, axis=0)
    if ucits:
        for s in px.columns:
            u = UCITS_MAP.get(s)
            if u is None:
                continue
            dy = _div_yield(dividends, inp.prices, s, idx)
            drag = pd.Series(u.ter - u.us_ter, index=idx) + (u.wht * dy if u.equity else 0.0)
            px[s] = px[s] * (1 - drag.fillna(u.ter - u.us_ter) / 252).cumprod()
    rate = (ecb_rate_pct.reindex(idx).ffill() / 100 - investor.cash_spread).clip(lower=0.0)
    return MarketData(px=px, cash_rate=rate.fillna(0.0), fx=fx)


def _div_yield(dividends: pd.DataFrame | None, prices: pd.DataFrame, s: str, idx) -> pd.Series:
    """Rendiment per dividend dels últims 12 mesos (point-in-time): Σ dividends 252 sessions / preu."""
    if dividends is None or s not in dividends or s not in prices:
        return pd.Series(0.0, index=idx)
    d = dividends[s].reindex(idx).fillna(0.0).rolling(252, min_periods=1).sum()
    close = prices[s].reindex(idx)
    return (d / close).fillna(0.0).clip(0, 0.15)


@dataclass
class TaxSimResult:
    name: str
    value: pd.Series  # valor de mercat de la cartera (EUR, abans de l'impost latent)
    twr: pd.Series  # retorn diari ponderat en el temps (net de comissions, abans d'impostos)
    flows: pd.Series  # fluxos de l'inversor (negatiu = posa diners; positiu = rep)
    taxes: pd.DataFrame  # una fila per any liquidat
    orders: pd.DataFrame  # data, actiu, import (+compra/-venda), comissió, guany realitzat
    final_net: float  # el que queda després de vendre-ho tot i pagar l'impost
    ledger: Ledger
    info: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        contrib = self.info.get("contributed", np.nan)
        tw = self.twr.dropna()
        eq = (1 + tw).cumprod()
        years = len(tw) / 252
        alt = self.flows.copy()
        if "value_before_liquidation" in self.info:  # sense vendre al final (impost latent no pagat)
            alt.iloc[-1] += self.info["value_before_liquidation"] - self.final_net
        return {
            "estrategia": self.name, "TIR_neta_%": 100 * irr(self.flows),
            "TIR_sense_liquidar_%": 100 * irr(alt),
            "valor_final_net": self.final_net, "aportat": contrib,
            "impostos": float(self.taxes["tax"].sum()), "comissions": float(self.orders["fee"].sum()),
            "ordres_any": len(self.orders) / max(years, 1e-9),
            "CAGR_abans_impostos_%": 100 * (eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
            "max_dd_%": 100 * float((eq / eq.cummax() - 1).min()),
            "perdues_diferides_2m": self.ledger.deferred_total,
        }


def irr(flows: pd.Series) -> float:
    f = flows[flows != 0]
    if f.empty or (f > 0).sum() == 0 or (f < 0).sum() == 0:
        return np.nan
    t = (f.index - f.index[0]).days.to_numpy() / 365.25
    v = f.to_numpy()

    def npv(r: float) -> float:
        return float(np.sum(v / (1 + r) ** t))

    lo, hi = -0.95, 1.0  # bisecció (sense dependre de scipy)
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9 or hi - lo < 1e-10:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def simulate_tax(name: str, inp: PortfolioInputs, targets: Targets, cfg: PortfolioConfig, md: MarketData,
                 investor: Investor, policy: Policy, start: pd.Timestamp, end: pd.Timestamp | None = None,
                 *, with_tax: bool = True, cash_ret: pd.Series | None = None) -> TaxSimResult:
    """Simula de `start` a `end`. `cash_ret` (retorn diari) substitueix el tipus del BCE (tests)."""
    T = lagged(targets, 1)
    idx = md.px.index
    end = idx[-1] if end is None else pd.Timestamp(end)
    days = idx[(idx >= pd.Timestamp(start)) & (idx <= end)]
    assets = [a for a in T.weights.columns if a != CASH]
    W = T.weights.reindex(columns=[*assets, CASH])
    if CASH not in T.weights.columns:  # objectius sense efectiu explícit (p. ex. 100 % SPY)
        W[CASH] = W[assets].notna().all(axis=1).map({True: 0.0, False: np.nan})
    trig = Trigger(cfg.rebalance, cfg.band, cfg.te_band)
    rets_usd = inp.asset_returns().reindex(columns=[*assets, CASH])
    equity = set(inp.sectors) | {cfg.market}
    sectors = set(inp.sectors)
    br = investor.broker
    led = Ledger(investor.tax)

    cash = 0.0
    funded = False
    contributed = 0.0
    flows: dict[pd.Timestamp, float] = {}
    values, twr, orders, taxes = [], [], [], []
    last_tgt: np.ndarray | None = None
    prev_value: float | None = None
    started = False
    # Objectiu vigent a l'inici: l'última revisió anterior (si no, s'esperaria a la següent)
    prev_rev = T.reviews.loc[:days[0]].iloc[:-1]
    prev_rev = prev_rev[prev_rev]
    for t_rev in prev_rev.index[::-1]:
        row0 = W.loc[t_rev].to_numpy(dtype=float)
        if not np.isnan(row0).any():
            last_tgt = row0
            break
    rank_cache: dict[pd.Timestamp, dict[str, int]] = {}
    last_month = None
    settled_upto = days[0].year - 1

    def add_flow(d, x):
        flows[d] = flows.get(d, 0.0) + x

    def holdings_value(i_px) -> np.ndarray:
        # Un actiu sense posició val 0 encara que no tingui preu (p. ex. abans de cotitzar)
        out = np.zeros(len(assets))
        for k, a in enumerate(assets):
            sh = led.shares(a)
            if sh > 0:
                out[k] = sh * i_px[a]
        return out

    def blocked(a: str, d: pd.Timestamp) -> bool:
        return policy.avoid_wash and any(p.symbol == a and p.date > d - TWO_MONTHS and p.shares > 1e-9
                                         for p in led.pending)

    def do_sell(a: str, amount: float, d, p: float) -> None:
        nonlocal cash
        sh = min(amount / p, led.shares(a))
        if sh <= 1e-12:
            return
        gross = sh * p
        if gross < br.min_order and sh < led.shares(a) - 1e-9:
            return  # venda parcial massa petita
        fee = br.fee(gross)
        g = led.sell(a, sh, gross - fee, d)
        cash += gross - fee
        orders.append({"date": d, "symbol": a, "amount": -gross, "fee": fee, "realized": g})

    def do_buy(a: str, amount: float, d, p: float) -> None:
        nonlocal cash
        amount = min(amount, cash)
        if amount < br.min_order:
            return
        fee = br.fee(amount)
        net = amount - fee
        if net <= 0:
            return
        led.buy(a, net / p, amount, d)
        cash -= amount
        orders.append({"date": d, "symbol": a, "amount": amount, "fee": fee, "realized": 0.0})

    def deploy(tgt: np.ndarray, d, pxd: pd.Series, V: float) -> None:
        """Inverteix l'efectiu que sobra respecte de l'objectiu als actius infraponderats."""
        excess = cash - tgt[-1] * V
        if excess < br.min_order:
            return
        vals = holdings_value(pxd)
        want = np.array([0.0 if (blocked(a, d) or np.isnan(pxd[a])) else max(tgt[k] * V - vals[k], 0.0)
                         for k, a in enumerate(assets)])
        if want.sum() <= 0:
            want = np.array([0.0 if (blocked(a, d) or np.isnan(pxd[a])) else tgt[k] for k, a in enumerate(assets)])
            if want.sum() <= 0:
                return
        alloc = np.minimum(want, excess * want / want.sum()) if want.sum() > excess else want.copy()
        rest = excess - alloc.sum()
        if rest > br.min_order:  # tot l'infrapès cobert: la resta, segons l'objectiu
            w2 = np.array([0.0 if (blocked(a, d) or np.isnan(pxd[a])) else tgt[k] for k, a in enumerate(assets)])
            if w2.sum() > 0:
                alloc += rest * w2 / w2.sum()
        small = (alloc > 0) & (alloc < br.min_order)
        if small.any() and alloc[~small].sum() > 0:  # concentra les ordres petites
            alloc[~small] += alloc[small].sum() * alloc[~small] / alloc[~small].sum()
            alloc[small] = 0.0
        for k in np.argsort(-alloc):
            if alloc[k] > 0:
                do_buy(assets[k], alloc[k], d, pxd[assets[k]])

    def rebalance_to(tgt: np.ndarray, d, pxd: pd.Series, V: float) -> None:
        vals = holdings_value(pxd)
        delta = tgt[:-1] * V - vals
        for k in np.argsort(delta):  # vendes primer
            if delta[k] < 0:
                do_sell(assets[k], -delta[k], d, pxd[assets[k]])
        V2 = cash + holdings_value(pxd).sum()
        deploy(tgt, d, pxd, V2)

    def tax_cost_rank(cands: list[str], pxd: pd.Series) -> list[str]:
        """Ordre de venda: primer el que genera menys guany per euro venut (pèrdues primer)."""
        def g_per_eur(a):
            v = led.shares(a) * pxd[a]
            return led.unrealized(a, pxd[a]) / v if v > 0 else 0.0
        return sorted(cands, key=g_per_eur)

    for i, d in enumerate(days):
        pxd = md.px.loc[d]
        gi = idx.get_loc(d)
        # 1) interessos de l'efectiu (RCM) i aportació mensual
        if funded and cash > 0:
            if cash_ret is not None:
                r_c = float(cash_ret.get(d, 0.0))
            else:
                r_c = float(md.cash_rate.iloc[gi]) * (d - days[i - 1]).days / 365.0
            interest = cash * r_c
            cash += interest
            led.add_interest(interest, d)
        contrib = 0.0
        month = d.to_period("M")
        if not funded:
            funded = True
            cash += investor.initial
            add_flow(d, -investor.initial)
            contrib = investor.initial
        elif month != last_month and investor.monthly > 0:
            cash += investor.monthly
            add_flow(d, -investor.monthly)
            contrib = investor.monthly
        contributed += contrib
        last_month = month
        # 2) impost de l'any anterior (juliol)
        if with_tax and d.month >= investor.tax.settle_month and settled_upto < d.year - 1:
            for y in range(settled_upto + 1, d.year):
                res = led.settle(y)
                taxes.append(res.__dict__ | {"paid": d})
                add_flow(d, -res.tax)
            settled_upto = d.year - 1
        # 3) decisions
        row = W.iloc[gi].to_numpy(dtype=float)
        new_obj = bool(T.reviews.iloc[gi]) and not np.isnan(row).any()
        if new_obj:
            last_tgt = row
        if last_tgt is None:
            values.append(np.nan)
            twr.append(np.nan)
            continue
        V = cash + holdings_value(pxd).sum()
        if not started:
            started = True
            rebalance_to(last_tgt, d, pxd, V)
        elif policy.kind == "target":
            if new_obj or cfg.check_daily:
                cur = np.append(holdings_value(pxd), cash) / V
                cov_i = None
                if cfg.rebalance == "tracking_error" and gi + 1 >= cfg.te_window:
                    cov_i = rolling_cov(rets_usd.iloc[gi - cfg.te_window + 1: gi + 1],
                                        [*assets, CASH], cfg.te_window)[-1]
                if trig.fires(last_tgt, cur, cov_i):
                    rebalance_to(last_tgt, d, pxd, V)
            if contrib > 0:
                deploy(last_tgt, d, pxd, cash + holdings_value(pxd).sum())
        else:
            if new_obj:
                _accumulate_review(policy, cfg, inp, T, gi, last_tgt, assets, equity, sectors, pxd, d, V,
                                   rank_cache, holdings_value, do_sell, tax_cost_rank)
                deploy(last_tgt, d, pxd, cash + holdings_value(pxd).sum())
            elif contrib > 0:
                deploy(last_tgt, d, pxd, cash + holdings_value(pxd).sum())
        Vd = cash + holdings_value(pxd).sum()
        values.append(Vd)
        if prev_value is None:
            twr.append(np.nan)
        else:
            twr.append((Vd - contrib) / prev_value - 1)
        prev_value = Vd

    # 4) liquidació final
    dl = days[-1]
    pxd = md.px.loc[dl]
    value_before = cash + holdings_value(pxd).sum()
    latent = sum(led.unrealized(a, pxd[a]) for a in led.lots)
    for a in list(led.lots):
        do_sell(a, led.shares(a) * pxd[a] + 1.0, dl, pxd[a])
    final_tax = 0.0
    if with_tax:
        for y in range(settled_upto + 1, dl.year + 1):
            res = led.settle(y)
            taxes.append(res.__dict__ | {"paid": dl})
            final_tax += res.tax
    final_net = cash - final_tax
    add_flow(dl, final_net)
    fl = pd.Series(flows).sort_index()
    tx = pd.DataFrame(taxes) if taxes else pd.DataFrame(columns=["year", "tax"])
    return TaxSimResult(name=name, value=pd.Series(values, index=days), twr=pd.Series(twr, index=days),
                        flows=fl, taxes=tx, orders=pd.DataFrame(orders, columns=["date", "symbol", "amount", "fee", "realized"]),
                        final_net=final_net, ledger=led,
                        info={"policy": policy.__dict__, "start": days[0], "contributed": contributed,
                              "value_before_liquidation": value_before, "latent_gain": latent})


def _accumulate_review(policy: Policy, cfg: PortfolioConfig, inp: PortfolioInputs, T: Targets, gi: int,
                       tgt: np.ndarray, assets: list[str], equity: set[str], sectors: set[str],
                       pxd: pd.Series, d: pd.Timestamp, V: float, rank_cache: dict, holdings_value,
                       do_sell, tax_cost_rank) -> None:
    """Vendes permeses a la política "acumular" en una revisió (objectiu nou)."""
    vals = dict(zip(assets, holdings_value(pxd), strict=True))
    tw = dict(zip(assets, tgt[:-1], strict=True))
    # a) sector molt degradat: rang a la data de decisió (sessió anterior)
    if policy.demote_buffer is not None and gi > 0:
        t_dec = inp.index[gi - 1]
        if t_dec not in rank_cache:
            rank_cache[t_dec] = sector_ranks(inp, t_dec, cfg)
        ranks = rank_cache[t_dec]
        for a in assets:
            if a in sectors and vals[a] > 0 and ranks.get(a, 99) > cfg.top_k + policy.demote_buffer and tw[a] <= 0:
                do_sell(a, vals[a] + 1.0, d, pxd[a])
        vals = dict(zip(assets, holdings_value(pxd), strict=True))
    if not policy.regime_sells:
        return
    # b) el règim redueix la renda variable: vendre fins a l'objectiu de renda variable
    eq_now = sum(v for a, v in vals.items() if a in equity) / V
    eq_tgt = sum(w for a, w in tw.items() if a in equity)
    if eq_now > eq_tgt + policy.expo_band:
        need = (eq_now - eq_tgt) * V
        held = [a for a in assets if a in equity and vals[a] > 0]
        out = [a for a in held if tw[a] <= 0]
        rest = [a for a in held if tw[a] > 0]
        order = (tax_cost_rank(out, pxd) + tax_cost_rank(rest, pxd)) if policy.tax_aware_sells else out + rest
        for a in order:
            if need <= 0:
                break
            amt = min(vals[a], need)
            do_sell(a, amt if amt < vals[a] - 1e-6 else vals[a] + 1.0, d, pxd[a])
            need -= amt
    # c) el règim torna a pujar la renda variable: vendre la cobertura que sobra
    elif eq_now < eq_tgt - policy.expo_band:
        for a in assets:
            if a not in equity and vals[a] > tw[a] * V + 1e-6:
                do_sell(a, vals[a] - tw[a] * V if tw[a] > 0 else vals[a] + 1.0, d, pxd[a])
