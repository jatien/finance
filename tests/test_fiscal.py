"""Tests de la fiscalitat espanyola i del simulador després d'impostos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_portfolio import inp  # noqa: F401 — fixture compartida

from finance.fiscal.config import Broker, Investor, SpainTax, Ucits
from finance.fiscal.ledger import Ledger
from finance.fiscal.sim import MarketData, Policy, irr, simulate_tax
from finance.portfolio import build_targets
from finance.portfolio.backtest import lagged, simulate_cfg
from finance.portfolio.config import ROT_V1B, ROT_V2
from finance.portfolio.construct import Targets

D = pd.Timestamp


# ---------------------------------------------------------------------------------------
# IRPF i broker
# ---------------------------------------------------------------------------------------

def test_savings_scale():
    t = SpainTax()
    assert t.tax_on(0) == 0 and t.tax_on(-100) == 0
    assert t.tax_on(6_000) == pytest.approx(1_140)
    assert t.tax_on(10_000) == pytest.approx(1_140 + 4_000 * 0.21)
    assert t.tax_on(350_000) == pytest.approx(1_140 + 44_000 * .21 + 150_000 * .23 + 100_000 * .27 + 50_000 * .30)


def test_broker_fee_min_max():
    b = Broker()
    assert b.fee(1_000) == 3.0  # 0,05 % = 0,50 € → mínim
    assert b.fee(20_000) == pytest.approx(10.0)
    assert b.fee(1_000_000) == 29.0
    assert b.fee(0) == 0.0


def test_ucits_drag():
    u = Ucits("x", None, None, ter=0.0015, us_ter=0.0008)
    assert u.annual_drag(0.02) == pytest.approx(0.0007 + 0.003)
    bond = Ucits("b", None, None, ter=0.0007, us_ter=0.0015, equity=False)
    assert bond.annual_drag(0.04) == pytest.approx(-0.0008)


# ---------------------------------------------------------------------------------------
# Lots, regla dels dos mesos i compensació
# ---------------------------------------------------------------------------------------

def test_fifo_gain():
    led = Ledger(SpainTax())
    led.buy("A", 10, 1_000, D("2020-01-02"))  # 100 €/u
    led.buy("A", 10, 2_000, D("2021-01-04"))  # 200 €/u
    g = led.sell("A", 15, 15 * 250, D("2022-03-01"))
    assert g == pytest.approx(15 * 250 - (1_000 + 5 * 200))  # FIFO: primer els barats
    assert led.shares("A") == pytest.approx(5)
    assert led.cost_basis("A") == pytest.approx(1_000)


def test_two_month_rule_after_sale_defers_loss():
    led = Ledger(SpainTax())
    led.buy("A", 10, 1_000, D("2022-01-03"))
    loss = led.sell("A", 10, 600, D("2022-05-02"))
    assert loss == pytest.approx(-400)
    led.buy("A", 10, 650, D("2022-06-15"))  # recompra dins dels 2 mesos
    assert led.gains[2022] == pytest.approx(0.0)  # la pèrdua no s'integra
    assert led.cost_basis("A") == pytest.approx(1_050)  # va al cost dels nous
    assert led.deferred_total == pytest.approx(400)
    # Es ven després: la pèrdua diferida s'integra llavors
    g = led.sell("A", 10, 650, D("2022-12-01"))
    assert g == pytest.approx(-400)


def test_two_month_rule_before_sale_and_outside_window():
    led = Ledger(SpainTax())
    led.buy("A", 10, 1_000, D("2022-01-03"))
    led.buy("A", 10, 800, D("2022-04-20"))  # compra recent
    loss = led.sell("A", 10, 600, D("2022-05-02"))  # FIFO ven el lot de gener amb -400
    assert loss == pytest.approx(0.0)
    assert led.cost_basis("A") == pytest.approx(1_200)
    led2 = Ledger(SpainTax())
    led2.buy("A", 10, 1_000, D("2022-01-03"))
    led2.sell("A", 10, 600, D("2022-05-02"))
    led2.buy("A", 10, 650, D("2022-07-05"))  # fora dels 2 mesos: la pèrdua compta
    assert led2.gains[2022] == pytest.approx(-400)
    # Rebutjat si la regla està desactivada
    led3 = Ledger(SpainTax(two_month_rule=False))
    led3.buy("A", 10, 1_000, D("2022-01-03"))
    led3.sell("A", 10, 600, D("2022-05-02"))
    led3.buy("A", 10, 650, D("2022-05-10"))
    assert led3.gains[2022] == pytest.approx(-400)


def test_two_month_rule_partial_repurchase():
    led = Ledger(SpainTax())
    led.buy("A", 10, 1_000, D("2022-01-03"))
    led.sell("A", 10, 600, D("2022-05-02"))  # -400 per 10 participacions
    led.buy("A", 4, 260, D("2022-05-20"))  # només 4 de 10
    assert led.gains[2022] == pytest.approx(-240)
    assert led.cost_basis("A") == pytest.approx(260 + 160)


def test_settle_carryforward_and_rcm_cap():
    led = Ledger(SpainTax())
    led.gains[2020] = -10_000
    led.interest[2020] = 1_000
    r = led.settle(2020)
    assert r.tax == pytest.approx(750 * 0.19) and r.carry_left == pytest.approx(10_000 - 250)  # 25 % dels interessos
    led.gains[2021] = 4_000
    r = led.settle(2021)
    assert r.tax == 0 and r.carry_left == pytest.approx(5_750)
    led.gains[2022] = 8_000
    r = led.settle(2022)
    assert r.base == pytest.approx(2_250) and r.tax == pytest.approx(2_250 * 0.19)
    # Caducitat als 4 anys
    led.gains[2023] = -1_000
    led.settle(2023)
    for y in (2024, 2025, 2026, 2027):
        led.settle(y)
    led.gains[2028] = 1_000
    assert led.settle(2028).tax == pytest.approx(190)


def test_irr():
    f = pd.Series([-100.0, 110.0], index=[D("2020-01-01"), D("2021-01-01")])
    assert irr(f) == pytest.approx(0.0997, abs=1e-3)  # 366 dies
    assert np.isnan(irr(pd.Series([-1.0], index=[D("2020-01-01")])))


# ---------------------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------------------

def _ideal(inp):  # noqa: F811
    md = MarketData(px=inp.prices.copy(), cash_rate=pd.Series(0.0, index=inp.index),
                    fx=pd.Series(1.0, index=inp.index))
    inv = Investor(initial=1e6, monthly=0,
                   broker=Broker(rate=ROT_V1B.cost_bps / 1e4, min_fee=0, max_fee=1e12, min_order=0))
    return md, inv


def test_target_policy_reproduces_weight_engine(inp):  # noqa: F811
    """Sense impostos, divisa ni mínims, el simulador en participacions és el motor de pesos."""
    cfg = ROT_V1B
    T = build_targets(inp, cfg)
    md, inv = _ideal(inp)
    start = inp.index[300]
    r = simulate_tax("x", inp, T, cfg, md, inv, Policy("target"), start, with_tax=False, cash_ret=inp.cash_ret)
    TL = lagged(T, 1)
    W = TL.weights.copy()
    rv = TL.reviews.copy()
    W.loc[W.index < start] = np.nan
    rv.loc[rv.index < start] = False
    first = r.twr.first_valid_index()
    prev = r.value.first_valid_index()
    last_rev = TL.reviews.loc[:prev]
    W.loc[prev] = TL.weights.loc[last_rev[last_rev].index[-1]]
    rv.loc[prev] = True
    bt = simulate_cfg(Targets(W, rv, T.log), inp.asset_returns(), cfg)
    a = (1 + r.twr.loc[first:]).prod()
    b = (1 + bt.returns.loc[first:]).prod()
    assert a == pytest.approx(b, rel=0.01)


def test_tax_run_flows_and_accumulate(inp):  # noqa: F811
    idx = inp.index
    md = MarketData(px=inp.prices.copy(), cash_rate=pd.Series(0.02, index=idx), fx=pd.Series(1.0, index=idx))
    inv = Investor()
    T = build_targets(inp, ROT_V2)
    start = idx[300]
    tgt = simulate_tax("t", inp, T, ROT_V2, md, inv, Policy("target"), start)
    acc = simulate_tax("a", inp, T, ROT_V2, md, inv, Policy("accumulate"), start)
    for r in (tgt, acc):
        f = r.flows
        assert f.iloc[0] == -inv.initial and f.iloc[-1] == pytest.approx(r.final_net)
        months = pd.Series(r.value.index.to_period("M")).nunique()
        assert (f == -inv.monthly).sum() + (f.iloc[1:-1] < 0).sum() >= months - 2
        assert r.final_net > 0 and not np.isnan(irr(f))
        assert (r.orders.loc[r.orders["amount"] > 0, "amount"] >= inv.broker.min_order - 1e-9).all()
        assert (r.orders["fee"] >= 0).all()
        assert not r.ledger.lots  # tot liquidat
        s = r.summary()
        assert s["impostos"] >= 0 and s["comissions"] > 0
    # "Acumular" ven menys que seguir l'objectiu
    assert (acc.orders["amount"] < 0).sum() < (tgt.orders["amount"] < 0).sum()


def test_accumulate_sells_on_crisis(inp):  # noqa: F811
    """A la crisi (E = 0) la política acumular ven la renda variable."""
    idx = inp.index
    md = MarketData(px=inp.prices.copy(), cash_rate=pd.Series(0.0, index=idx), fx=pd.Series(1.0, index=idx))
    T = build_targets(inp, ROT_V1B)
    r = simulate_tax("a", inp, T, ROT_V1B, md, Investor(monthly=0), Policy("accumulate"), idx[300], with_tax=False)
    sells = r.orders[(r.orders["amount"] < 0) & (r.orders["date"] > idx[700]) & (r.orders["date"] <= idx[705])]
    assert not sells.empty
    eq = [s for s in ("XLK", "XLF", "XLE", "XLP", "XLU", "XLV", "SPY")]
    held_eq = sum(r.ledger.shares(s) for s in eq)
    assert held_eq == 0  # liquidat al final


def test_buy_and_hold_without_cash_column_invests(inp):  # noqa: F811
    from finance.portfolio.construct import fixed_targets
    idx = inp.index
    md = MarketData(px=inp.prices.copy(), cash_rate=pd.Series(0.0, index=idx), fx=pd.Series(1.0, index=idx))
    T = fixed_targets(idx, {"SPY": 1.0}, monthly=False, start=idx[250])
    inv = Investor()
    r = simulate_tax("spy", inp, T, ROT_V1B, md, inv, Policy("target"), idx[300])
    months = pd.Series(r.value.index.to_period("M")).nunique()
    assert r.info["contributed"] == pytest.approx(inv.initial + inv.monthly * (months - 1))
    buys = r.orders[r.orders["amount"] > 0]
    assert set(buys["symbol"]) == {"SPY"} and len(buys) >= months - 2
    assert (r.orders["amount"] < 0).sum() == 1  # només la liquidació final
    assert r.summary()["aportat"] == pytest.approx(r.info["contributed"])


def test_ucits_ter_units_guard():
    with pytest.raises(ValueError):
        Ucits("x", None, None, ter=0.0015, us_ter=0.0945)
