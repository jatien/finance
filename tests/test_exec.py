"""Tests de la setmana 11: pla d'execució, salvaguardes, broker simulat, mode manual i alertes."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest
from test_paper import _ctx, _daily

from finance.alerts.dispatch import Alert, dispatch
from finance.alerts.rules import report_alerts
from finance.execution.guards import account_safety, next_exec_session, requires_approval
from finance.execution.plan import PlanLimits, build_plan
from finance.portfolio.config import CASH
from finance.providers.base import (
    AccountSnapshot,
    BrokerClient,
    BrokerFill,
    BrokerPosition,
    BrokerQuote,
    Notifier,
    OrderState,
    ProviderError,
    TradableInstrument,
)

INSTRUMENTS_YAML = """
currency: EUR
instruments:
  SPY: {ticker: SXR8, isin: IE00B5BMR087}
  XLK: {ticker: ZPDT, isin: IE00BWBXM948}
  XLF: {ticker: ZPDF, isin: IE00TESTXLF0}
  XLV: {ticker: ZPDH, isin: IE00BWBXM617}
  XLP: {ticker: ZPDS, isin: IE00TESTXLP0}
  IEF: {ticker: TEST7, isin: IE00TESTIEF0}
  GLD: {ticker: 4GLD, isin: DE000A0S9GB0}
"""
PRICES = {"SXR8": 600.0, "ZPDT": 120.0, "ZPDF": 50.0, "ZPDH": 55.0, "ZPDS": 45.0, "TEST7": 5.0, "4GLD": 110.0}


def inst(sym: str, ticker: str) -> TradableInstrument:
    return TradableInstrument(sym, ticker, f"ISIN{ticker}")


# ---------------------------------------------------------------------------------------
# Broker i notificador simulats
# ---------------------------------------------------------------------------------------

@dataclass
class FakeBroker(BrokerClient):
    name: str = "fake"
    acct: str = "DU1234567"
    cash: float = 10_000.0
    prices: dict[str, float] = field(default_factory=lambda: dict(PRICES))
    positions: dict[str, float] = field(default_factory=dict)
    fill_mode: str = "all"  # all | none | half
    placed: list = field(default_factory=list)
    states: dict[str, OrderState] = field(default_factory=dict)
    _fills: list[BrokerFill] = field(default_factory=list)
    now: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2026-10-16 14:50", tz="UTC"))

    def account(self):
        nav = self.cash + sum(q * self.prices[t] for t, q in self.positions.items())
        return AccountSnapshot(self.acct, self.acct.startswith("D"), "EUR", nav, self.cash, self.cash,
                               [BrokerPosition(t, q) for t, q in self.positions.items()])

    def quotes(self, instruments):
        return {i.ticker: BrokerQuote(i.ticker, bid=self.prices[i.ticker] * 0.999, ask=self.prices[i.ticker] * 1.001)
                for i in instruments if i.ticker in self.prices}

    def place(self, req):
        if req.ref in self.states:
            return self.states[req.ref]
        self.placed.append(req)
        t, px = req.instrument.ticker, self.prices[req.instrument.ticker]
        q = {"all": req.quantity, "none": 0, "half": req.quantity // 2}[self.fill_mode]
        if q:
            sign = 1 if req.side == "BUY" else -1
            self.positions[t] = self.positions.get(t, 0) + sign * q
            self.cash -= sign * q * px + 3.0
            self._fills.append(BrokerFill(f"e{len(self._fills)}", req.ref, t, req.side, q, px, 3.0, "EUR", self.now))
        st = OrderState(req.ref, "filled" if q == req.quantity else ("partial" if q else "submitted"),
                        str(len(self.placed)), float(q), px if q else None)
        self.states[req.ref] = st
        return st

    def orders(self):
        return dict(self.states)

    def cancel(self, ref):
        self.states[ref].status = "cancelled"

    def fills(self, since=None):
        return list(self._fills)

    def wait(self, seconds):
        pass


@dataclass
class FakeNotifier(Notifier):
    name: str = "fake"
    sent: list = field(default_factory=list)
    broken: bool = False

    def send(self, title, body, level="info"):
        if self.broken:
            raise ProviderError("caigut")
        self.sent.append((level, title, body))


# ---------------------------------------------------------------------------------------
# Pla (funcions pures)
# ---------------------------------------------------------------------------------------

def _snap(cash=10_000.0, pos=None, acct="DU1"):
    pos = pos or {}
    nav = cash + sum(q * PRICES[t] for t, q in pos.items())
    return AccountSnapshot(acct, True, "EUR", nav, cash, cash, [BrokerPosition(t, q) for t, q in pos.items()])


def _quotes(*tickers):
    return {t: BrokerQuote(t, close=PRICES[t]) for t in tickers}


INSTS = {"XLK": inst("XLK", "ZPDT"), "XLV": inst("XLV", "ZPDH"), "XLF": inst("XLF", "ZPDF")}


def test_plan_from_cash_buys_whole_shares_within_budget():
    tgt = pd.Series({"XLK": 0.5, "XLV": 0.5, CASH: 0.0})
    p = build_plan("p", tgt, _snap(), _quotes("ZPDT", "ZPDH"), INSTS, force=True)
    assert not p.blockers and [o.side for o in p.orders] == ["BUY", "BUY"]
    q = {o.symbol: o.quantity for o in p.orders}
    # 1 % de coixí: les compres no superen 9.900 €
    assert sum(o.amount for o in p.orders) <= 9_900 + 1e-9
    assert q["XLK"] * 120 <= 5_000 and q["XLV"] * 55 <= 5_000
    assert all(o.ref.startswith("fin:p:") for o in p.orders)


def test_plan_sells_first_exits_fully_and_skips_small():
    snap = _snap(cash=100.0, pos={"ZPDT": 40, "ZPDF": 3})  # XLK 4.800, XLF 150 (petita)
    tgt = pd.Series({"XLV": 0.6, "XLK": 0.4})
    p = build_plan("p", tgt, snap, _quotes("ZPDT", "ZPDH", "ZPDF"), INSTS, force=True)
    sides = [(o.side, o.symbol) for o in p.orders]
    assert sides[0][0] == "SELL" and ("SELL", "XLF") in sides  # sortida completa encara que < mínim
    assert next(o for o in p.orders if o.symbol == "XLF").quantity == 3
    # XLK: objectiu 40 % de 5.050 ≈ 2.020 → vendre ~23
    assert next(o for o in p.orders if o.symbol == "XLK").side == "SELL"
    assert next(o for o in p.orders if o.symbol == "XLV").side == "BUY"


def test_plan_no_trade_inside_band_unless_forced():
    snap = _snap(cash=0.0, pos={"ZPDT": 50, "ZPDH": 100})  # 6.000 / 5.500
    tgt = pd.Series({"XLK": 0.5, "XLV": 0.5})
    p = build_plan("p", tgt, snap, _quotes("ZPDT", "ZPDH"), INSTS, force=False)
    assert not p.needs_trade and not p.orders and p.max_deviation < 0.05
    p2 = build_plan("p", tgt, snap, _quotes("ZPDT", "ZPDH"), INSTS, force=True)
    assert p2.needs_trade  # forçat, però les ordres resultants són < mínim → cap
    assert all(o.amount >= 250 for o in p2.orders)


def test_plan_blocks_untradable_or_unquoted_and_ignores_foreign_positions():
    tgt = pd.Series({"XLK": 0.5, "XLB": 0.5})
    p = build_plan("p", tgt, _snap(), _quotes("ZPDT"), INSTS, {"XLB": "sense ticker operable"}, force=True)
    assert any("XLB" in b for b in p.blockers) and not p.orders
    p = build_plan("p", pd.Series({"XLK": 1.0}), _snap(), {}, INSTS, force=True)
    assert any("sense cotització" in b for b in p.blockers)
    snap = AccountSnapshot("DU1", True, "EUR", 20_000, 10_000, 10_000, [BrokerPosition("AAPL", 50)])
    p = build_plan("p", pd.Series({"XLK": 1.0}), snap, _quotes("ZPDT"), INSTS, force=True)
    assert p.managed_value == 10_000 and any("AAPL" in n for n in p.notes)


def test_plan_fat_finger_caps():
    tgt = pd.Series({"XLK": 1.0})
    p = build_plan("p", tgt, _snap(), _quotes("ZPDT"), INSTS, force=True, limits=PlanLimits(max_order_pct=0.5))
    assert any("> 50%" in b for b in p.blockers)


# ---------------------------------------------------------------------------------------
# Salvaguardes
# ---------------------------------------------------------------------------------------

def test_real_account_needs_explicit_live_and_approval():
    from finance.settings import Settings
    s = Settings(_env_file=None, exec_mode="ibkr")
    real = _snap(acct="U7654321")
    real.is_paper = False
    assert any("FIN_EXEC_LIVE" in b for b in account_safety(s, real))
    assert requires_approval(s, real) and not requires_approval(s, _snap())
    s2 = Settings(_env_file=None, exec_mode="ibkr", exec_live=True, exec_account="U0000000")
    assert any("≠ FIN_EXEC_ACCOUNT" in b for b in account_safety(s2, real))
    s3 = Settings(_env_file=None, exec_mode="ibkr", exec_live=True, exec_account="U7654321")
    assert account_safety(s3, real) == [] and requires_approval(s3, real)
    # Desactivar l'aprovació en real només amb FIN_EXEC_LIVE
    s4 = Settings(_env_file=None, exec_mode="ibkr", exec_require_approval=False)
    assert requires_approval(s4, real)


def test_next_exec_session_uses_xetra_calendar():
    assert next_exec_session(pd.Timestamp("2026-10-15")) == pd.Timestamp("2026-10-16")
    assert next_exec_session(pd.Timestamp("2026-10-16")) == pd.Timestamp("2026-10-19")  # cap de setmana
    assert next_exec_session(pd.Timestamp("2026-12-23")) == pd.Timestamp("2026-12-28")  # Nadal a Xetra


# ---------------------------------------------------------------------------------------
# Integració amb el lake: IBKR simulat
# ---------------------------------------------------------------------------------------

def _exec_ctx(tmp_path, providers, clock, **settings):
    ctx = _ctx(tmp_path, providers, clock)
    f = tmp_path / "execution.yaml"
    f.write_text(INSTRUMENTS_YAML)
    ctx.settings.exec_instruments_file = f
    ctx.settings.exec_mode = "ibkr"
    for k, v in settings.items():
        setattr(ctx.settings, k, v)
    _daily(ctx)  # dijous 15-oct: inici del paper (objectiu = cartera per a divendres)
    return ctx


def _run(ctx, action, **kw):
    from finance.ingest import run_job
    return run_job(ctx, "exec", action=action, **kw)


def test_ibkr_paper_account_full_cycle(tmp_path, providers, clock):
    from finance.execution.job import read_orders, read_plans
    from finance.execution.ledger import read_fills
    broker, note = FakeBroker(), FakeNotifier()
    providers.broker = lambda: broker
    providers.notifiers = [note]
    ctx = _exec_ctx(tmp_path, providers, clock)

    r = _run(ctx, "plan")
    assert r.status == "success", r.errors
    plans = read_plans(ctx)
    p = plans.iloc[-1]
    assert p["plan_id"] == "rot_v1b-20261016" and p["status"] == "approved"  # compte paper: sense aprovació
    orders = read_orders(ctx, p["plan_id"])
    assert len(orders) >= 1 and set(orders["side"]) == {"BUY"} and (orders["status"] == "planned").all()
    assert len(note.sent) == 1 and "Órdenes de mañana" in note.sent[0][1]
    # Tornar a planificar no duplica ni torna a avisar
    _run(ctx, "plan")
    assert len(read_plans(ctx)) == 1 and len(note.sent) == 1

    # La nit abans: el pla és per a demà, no s'envia res
    assert _run(ctx, "submit").status == "skipped"
    # Matí de la sessió abans de l'obertura de Xetra: es bloqueja i avisa
    clock.set("2026-10-16 05:00")
    r = _run(ctx, "submit")
    assert r.status == "failed" and not broker.placed
    assert "NO enviadas" in note.sent[-1][1]

    clock.set("2026-10-16 14:45")  # divendres 16:45 a Madrid, Xetra obert
    r = _run(ctx, "submit")
    assert r.status == "success", r.errors
    assert len(broker.placed) == len(orders)
    lim = {o.ref: o.limit_price for o in broker.placed}
    for o in orders.itertuples():  # límit = referència × (1 + 30 pb)
        assert lim[o.ref] == pytest.approx(PRICES[o.ticker] * 1.003, rel=1e-6)
    assert read_plans(ctx).iloc[-1]["status"] == "submitted"
    # Reenviar és idempotent
    _run(ctx, "submit")
    assert len(broker.placed) == len(orders)

    clock.set("2026-10-16 16:00")  # després del tancament de Xetra
    r = _run(ctx, "sync")
    assert r.status == "success", r.errors
    o2 = read_orders(ctx, p["plan_id"])
    assert (o2["status"] == "filled").all() and o2["slippage_bps"].notna().all()
    assert read_plans(ctx).iloc[-1]["status"] == "done"
    assert len(read_fills(ctx)) == len(orders)
    assert all(lv != "critical" for lv, _, _ in note.sent[-1:])  # resultat OK = info (filtrat)

    # L'informe diari mostra l'execució real
    from finance.paper.report import build_daily, render_html, render_text
    rep = build_daily(ctx)
    ex = rep.execution
    assert ex["last_plan"]["plan_id"] == p["plan_id"] and ex["account"]["value"] > 0
    assert abs(ex["slippage_bps"]) < 20 and ex["fill_rate"] == pytest.approx(1.0)
    assert "3b. Execució real" in render_html(rep) and "Execució real (ibkr)" in render_text(rep)


def test_unfilled_orders_expire_and_get_replanned(tmp_path, providers, clock):
    from finance.execution.job import read_orders, read_plans
    broker = FakeBroker(fill_mode="none")
    providers.broker = lambda: broker
    providers.notifiers = [FakeNotifier()]
    ctx = _exec_ctx(tmp_path, providers, clock)
    _run(ctx, "plan")
    clock.set("2026-10-16 14:45")
    _run(ctx, "submit")
    clock.set("2026-10-16 16:00")
    _run(ctx, "sync")
    pid = "rot_v1b-20261016"
    assert read_plans(ctx).set_index("plan_id").loc[pid, "status"] == "expired"
    assert (read_orders(ctx, pid)["status"] == "expired").all()


def test_real_account_is_blocked_without_live_flag(tmp_path, providers, clock):
    from finance.execution.job import read_plans
    note = FakeNotifier()
    providers.broker = lambda: FakeBroker(acct="U7654321")
    providers.notifiers = [note]
    ctx = _exec_ctx(tmp_path, providers, clock)
    _run(ctx, "plan")
    p = read_plans(ctx).iloc[-1]
    assert p["status"] == "blocked" and "FIN_EXEC_LIVE" in p["blockers"]
    assert note.sent[-1][0] == "critical"


def test_real_account_requires_approval(tmp_path, providers, clock):
    from finance.execution.job import read_plans
    broker, note = FakeBroker(acct="U7654321"), FakeNotifier()
    providers.broker = lambda: broker
    providers.notifiers = [note]
    ctx = _exec_ctx(tmp_path, providers, clock, exec_live=True, exec_account="U7654321")
    _run(ctx, "plan")
    pid = read_plans(ctx).iloc[-1]["plan_id"]
    assert read_plans(ctx).iloc[-1]["status"] == "awaiting_approval"
    assert f"fin exec approve {pid}" in note.sent[-1][2]
    clock.set("2026-10-16 14:45")
    r = _run(ctx, "submit")
    assert r.status == "skipped" and not broker.placed
    assert _run(ctx, "approve", plan_id=pid).status == "success"
    assert _run(ctx, "submit").status == "success" and broker.placed


def test_halt_blocks_submission_and_resume(tmp_path, providers, clock):
    broker = FakeBroker()
    providers.broker = lambda: broker
    providers.notifiers = [FakeNotifier()]
    ctx = _exec_ctx(tmp_path, providers, clock)
    _run(ctx, "plan")
    _run(ctx, "halt", reason="prova")
    assert ctx.settings.halt_file.exists()
    clock.set("2026-10-16 14:45")
    r = _run(ctx, "submit")
    assert r.status == "failed" and not broker.placed and "aturada" in str(r.errors)
    _run(ctx, "resume")
    assert _run(ctx, "submit").status == "success" and broker.placed


# ---------------------------------------------------------------------------------------
# Mode manual
# ---------------------------------------------------------------------------------------

def test_manual_mode_plan_fill_and_close(tmp_path, providers, clock):
    from conftest import make_history

    from finance.execution.job import read_account, read_orders, read_plans
    for t, px in PRICES.items():
        h = make_history(f"{t}.DE", start="2026-09-01")
        h[["close", "adj_close"]] = px
        providers.prices.histories[f"{t}.DE"] = h
    note = FakeNotifier()
    providers.notifiers = [note]
    ctx = _exec_ctx(tmp_path, providers, clock, exec_mode="manual", exec_capital=10_000.0)
    r = _run(ctx, "plan")
    assert r.status == "success", r.errors
    p = read_plans(ctx).iloc[-1]
    assert p["status"] == "planned" and p["mode"] == "manual"
    assert "fin exec fill" in note.sent[-1][2]
    orders = read_orders(ctx, p["plan_id"])
    o = orders.iloc[0]
    # S'executa a mà la primera ordre (una mica més cara)
    r = _run(ctx, "fill", ident=o["ticker"], side="BUY", quantity=o["quantity"], price=o["ref_price"] * 1.001, fee=3.0)
    assert r.status == "success" and p["plan_id"] in r.message
    o2 = read_orders(ctx, p["plan_id"]).set_index("symbol").loc[o["symbol"]]
    assert o2["status"] == "filled" and o2["slippage_bps"] == pytest.approx(10.0, abs=0.01)
    # Final de la sessió: la resta caduca, el pla queda parcial
    clock.set("2026-10-16 16:00")
    _run(ctx, "sync")
    st = read_plans(ctx).iloc[-1]["status"]
    assert st == ("done" if len(orders) == 1 else "partial")
    acc = read_account(ctx)
    assert acc["net_liquidation"].iloc[-1] == pytest.approx(10_000 - 3.0 - o["quantity"] * o["ref_price"] * 0.001, abs=0.5)


# ---------------------------------------------------------------------------------------
# Alertes
# ---------------------------------------------------------------------------------------

FACTS = {
    "data_date": "2026-10-16", "expected_session": "2026-10-16", "tradable": True, "issues": [],
    "currency": "USD", "next_review": "2026-11-02",
    "regime": {"regime": "bear_calm", "label": "Baixista", "exposure": 0.5, "days": 1, "changed_today": True,
               "previous_label": "Alcista"},
    "strategies": [{"name": "rot_v1b", "reference": True, "started": True, "decision_date": "2026-10-16",
                    "will_trade": True, "order_note": "canvi de règim", "ret": -0.12, "bench_ret": -0.1,
                    "dd_now": -0.16, "peak_date": "2026-09-01",
                    "orders": [{"symbol": "XLK", "action": "reduce", "current": 0.4, "target": 0.2, "amount": -2000,
                                "shares": -10}]}],
}


def test_report_alerts_rules():
    a = {x.key: x for x in report_alerts(FACTS)}
    assert "regime:2026-10-16:bear_calm" in a
    assert a["orders:rot_v1b:2026-10-16"].level == "important"
    assert {"dd:rot_v1b:0.10:2026-09-01", "dd:rot_v1b:0.15:2026-09-01"} <= set(a)
    assert "dd:rot_v1b:0.20:2026-09-01" not in a
    assert "weekly:2026-10-16" in a  # divendres
    # Amb execució real activa, les ordres de l'estratègia executada les avisa el pla real
    assert not any(k.startswith("orders:rot_v1b") for k in (x.key for x in report_alerts(FACTS, exec_strategy="rot_v1b")))
    red = {**FACTS, "tradable": False, "issues": [["error", "dades endarrerides"]]}
    assert report_alerts(red, weekly=False)[0].level == "critical"


def test_dispatch_dedups_and_retries_failed(tmp_path, providers, clock):
    from finance.jobs import Context, Run
    from finance.settings import Settings
    from finance.storage import Lake
    s = Settings(data_dir=tmp_path / "d", _env_file=None)
    s.ensure_dirs()
    bad = FakeNotifier(broken=True)
    providers.notifiers = [bad]
    ctx = Context(settings=s, lake=Lake(s.lake_dir), providers=providers, clock=clock)

    def run():
        return Run("alerts", "r", pd.Timestamp.now(tz="UTC"), ctx.lake, {})

    al = [Alert("k1", "critical", "t", "b"), Alert("k2", "info", "t", "b")]
    r = run()
    assert dispatch(ctx, r, al) == [] and "alerta:k1" in r.errors  # info filtrada; k1 fallida
    bad.broken = False
    assert [a.key for a in dispatch(ctx, run(), al)] == ["k1"]  # es reintenta
    assert dispatch(ctx, run(), al) == []  # ja lliurada: no es repeteix
    assert len(bad.sent) == 1


def test_ntfy_url_parsing_and_build():
    from finance.providers.notify import NtfyNotifier, build_notifiers
    from finance.settings import Settings
    n = NtfyNotifier("https://ntfy.sh/finance-abc123/")
    assert (n.server, n.topic) == ("https://ntfy.sh", "finance-abc123")
    with pytest.raises(ValueError):
        NtfyNotifier("https://ntfy.sh/")
    s = Settings(_env_file=None, alert_ntfy_url="https://ntfy.example.org/t", alert_telegram_token="x",
                 alert_telegram_chat_id="1")
    assert [x.name for x in build_notifiers(s)] == ["ntfy", "telegram"]


def test_pipeline_runs_exec_then_alerts():
    import inspect

    from finance.ingest import daily_pipeline
    src = inspect.getsource(daily_pipeline)
    assert src.index('"report"') < src.index('action="sync"') < src.index('action="plan"') < src.index('"alerts"')
