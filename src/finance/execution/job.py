"""Job `exec`: execució real de l'estratègia de referència (setmana 11).

Accions (`fin exec <acció>`; totes queden a `ingest_runs` i als datasets `exec_*`):

    plan      nit (després de l'informe): pla per a la sessió següent de Xetra
    approve   aprova un pla (obligatori amb compte real)       reject: el descarta
    submit    sessió (16:45 Madrid): envia les ordres a IBKR — vendes, espera, compres
    sync      després del tancament: execucions, estat de les ordres, foto del compte
    fill      mode manual: apunta una execució feta a mà        deposit/withdraw: efectiu
    halt      atura l'enviament d'ordres (i cancel·la les vives) resume: el reprèn

Cicle d'un pla: planned/awaiting_approval → approved → submitted → done | partial | expired.
També: no_trade (no cal operar), blocked (una salvaguarda ho impedeix), rejected.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from finance.alerts.dispatch import Alert, dispatch
from finance.calendar import get_calendar
from finance.execution.guards import (
    EXEC_CAL,
    account_safety,
    data_health,
    exec_today,
    halted,
    market_window,
    next_exec_session,
    requires_approval,
)
from finance.execution.instruments import InstrumentMap, load_instruments
from finance.execution.ledger import LedgerBroker, read_fills
from finance.execution.plan import Plan, PlanLimits, build_plan
from finance.jobs import Job, Run
from finance.portfolio.config import ACTIVE, CASH, STRATEGIES, PortfolioConfig
from finance.providers.base import (
    AccountSnapshot,
    BrokerClient,
    OrderRequest,
    ProviderError,
    TradableInstrument,
    TransientProviderError,
)
from finance.storage.schemas import EXEC_ACCOUNT, EXEC_FILLS, EXEC_ORDERS, EXEC_PLANS

LOCKED = {"approved", "submitted", "done", "partial", "expired", "rejected"}
OPEN = {"planned", "awaiting_approval", "approved", "submitted"}
FINAL_ORDER = {"filled", "cancelled", "rejected", "expired", "skipped"}

_last_stamp = pd.Timestamp(0, tz="UTC")


def stamp() -> pd.Timestamp:
    """`updated_at` estrictament creixent (la vista es queda amb l'última versió de cada fila)."""
    global _last_stamp
    now = pd.Timestamp.now(tz="UTC")
    _last_stamp = max(now, _last_stamp + pd.Timedelta(microseconds=1))
    return _last_stamp


# ---------------------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------------------

def _q(ctx, sql: str, params=None) -> pd.DataFrame:
    try:
        return ctx.query(sql, params or [])
    except Exception:  # noqa: BLE001 — dataset encara buit
        return pd.DataFrame()


def read_plans(ctx, strategy_set: str | None = None) -> pd.DataFrame:
    where, params = ("WHERE strategy_set = ?", [strategy_set]) if strategy_set else ("", [])
    df = _q(ctx, f"SELECT * EXCLUDE (source, ingested_at, run_id) FROM exec_plans {where} ORDER BY created_at", params)
    for c in ("decision_date", "session"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def get_plan(ctx, plan_id: str) -> dict | None:
    df = _q(ctx, "SELECT * EXCLUDE (source, ingested_at, run_id) FROM exec_plans WHERE plan_id = ?", [plan_id])
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    for c in ("decision_date", "session"):
        row[c] = pd.Timestamp(row[c])
    return row


def read_orders(ctx, plan_id: str | None = None) -> pd.DataFrame:
    where, params = ("WHERE plan_id = ?", [plan_id]) if plan_id else ("", [])
    return _q(ctx, f"SELECT * EXCLUDE (source, ingested_at, run_id) FROM exec_orders {where} "
                   "ORDER BY plan_id, side DESC, symbol", params)


def read_account(ctx) -> pd.DataFrame:
    df = _q(ctx, "SELECT * EXCLUDE (source, ingested_at, run_id) FROM exec_account ORDER BY account, date")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def exec_strategy(settings, name: str | None = None) -> PortfolioConfig:
    name = name or settings.exec_strategy
    if name is None:
        return ACTIVE[0]
    if name not in STRATEGIES:
        raise ValueError(f"estratègia desconeguda: {name}")
    return STRATEGIES[name]


def _limits(s) -> PlanLimits:
    return PlanLimits(band=s.exec_band, min_order=s.exec_min_order, max_order_pct=s.exec_max_order_pct,
                      max_turnover=s.exec_max_turnover, cash_buffer=s.exec_cash_buffer)


def _fmt_orders(orders: pd.DataFrame | list, currency: str) -> str:
    lines = []
    rows = orders.to_dict("records") if isinstance(orders, pd.DataFrame) else orders
    for o in rows:
        side = "VENDER" if o["side"] == "SELL" else "COMPRAR"
        px = o.get("limit_price") if pd.notna(o.get("limit_price")) else o.get("ref_price")
        lines.append(f"• {side} {o['quantity']:g} {o['ticker']} ({o['symbol']}) ~{px:,.2f} {currency} "
                     f"= {o['quantity'] * px:,.0f} {currency} · peso {o['current_weight']:.1%} → {o['target_weight']:.1%}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------------------

class ExecJob(Job):
    name = "exec"

    def __init__(self, ctx):
        super().__init__(ctx)
        self._broker_obj: BrokerClient | None = None
        self._imap: InstrumentMap | None = None
        self.alerts: list[Alert] = []

    # -- infra ---------------------------------------------------------------------------
    @property
    def s(self):
        return self.ctx.settings

    def broker(self) -> BrokerClient:
        if self._broker_obj is None:
            if self.s.exec_mode == "manual":
                self._broker_obj = LedgerBroker(self.ctx, self.imap.yahoo_suffix)
            elif self.ctx.providers.broker is None:
                raise ProviderError("FIN_EXEC_MODE=ibkr però no hi ha broker configurat")
            else:
                self._broker_obj = self.ctx.providers.broker()
        return self._broker_obj

    @property
    def imap(self) -> InstrumentMap:
        if self._imap is None:
            self._imap = load_instruments(self.s.exec_instruments_file, self.s.exec_currency)
        return self._imap

    def alert(self, key: str, level: str, title: str, body: str) -> None:
        self.alerts.append(Alert(key, level, title, body))

    def execute(self, run: Run, action: str = "plan", **kw: Any) -> None:
        if action not in ("approve", "reject", "halt", "resume", "fill", "deposit", "withdraw") \
                and self.s.exec_mode == "off":
            run.skip("execució desactivada (FIN_EXEC_MODE=off)")
            return
        run.params["mode"] = self.s.exec_mode
        try:
            getattr(self, f"_{action}")(run, **kw)
        finally:
            if self._broker_obj is not None:
                self._broker_obj.close()
            if self.alerts:
                dispatch(self.ctx, run, self.alerts)

    def _write_plan(self, run: Run, row: dict) -> None:
        row = {**row, "updated_at": stamp()}
        run.write(EXEC_PLANS, pd.DataFrame([row]), f"exec:{self.s.exec_mode}")

    def _write_orders(self, run: Run, rows: list[dict]) -> None:
        if rows:
            ts = stamp()
            run.write(EXEC_ORDERS, pd.DataFrame([{**r, "updated_at": ts} for r in rows]), f"exec:{self.s.exec_mode}")

    def _snapshot(self, run: Run, snap: AccountSnapshot) -> None:
        today = exec_today(self.ctx.now())
        fills = read_fills(self.ctx)
        flows = 0.0
        if not fills.empty:
            f = fills[fills["exec_time"].dt.tz_convert(get_calendar(EXEC_CAL).tz).dt.normalize().dt.tz_localize(None) == today]
            flows = float(f.loc[f["side"] == "DEPOSIT", "quantity"].sum() - f.loc[f["side"] == "WITHDRAWAL", "quantity"].sum())
        run.write(EXEC_ACCOUNT, pd.DataFrame([{
            "account": snap.account, "date": today, "currency": snap.currency,
            "net_liquidation": snap.net_liquidation, "cash": snap.cash, "net_flows": flows,
            "positions": json.dumps({p.ticker: p.quantity for p in snap.positions}),
            "snapshot_at": pd.Timestamp.now(tz="UTC")}]), f"exec:{self.s.exec_mode}")

    # -- plan ----------------------------------------------------------------------------
    def _plan(self, run: Run, strategy: str | None = None, force: bool = False, **_: Any) -> None:
        from finance.paper.job import read_state
        from finance.paper.report import strategy_view
        from finance.portfolio.construct import build_targets
        from finance.portfolio.job import load_inputs

        cfg = exec_strategy(self.s, strategy)
        run.params["strategy"] = cfg.name
        state = read_state(self.ctx, cfg.name)
        if state is None:
            run.skip(f"{cfg.name}: el paper trading encara no ha començat")
            return
        inp = load_inputs(self.ctx, cfg)
        targets = build_targets(inp, cfg)
        view = strategy_view(self.ctx, cfg, inp, targets, capital=1.0)
        session = next_exec_session(state.date)
        plan_id = f"{cfg.name}-{session:%Y%m%d}"
        prev = get_plan(self.ctx, plan_id)
        if prev is not None and prev["status"] in LOCKED:
            run.message = f"{plan_id}: ja és {prev['status']}, no es replanifica"
            run.ok()
            return

        blockers = halted(self.s) + data_health(self.ctx, inp.index[-1] if len(inp.index) else None)
        now = self.ctx.now()
        base = {"plan_id": plan_id, "strategy_set": cfg.name, "decision_date": state.date, "session": session,
                "mode": self.s.exec_mode, "currency": self.s.exec_currency, "created_at": now,
                "requires_approval": False, "approved_at": None, "notes": None}
        try:
            broker = self.broker()
            snap = broker.account()
        except (ProviderError, TransientProviderError) as e:
            self._write_plan(run, {**base, "status": "blocked", "blockers": f"broker: {e}",
                                   "reason": view.order_note})
            run.fail("broker", e)
            if view.will_trade:
                self.alert(f"exec-blocked:{plan_id}", "critical", f"Plan {plan_id} bloqueado",
                           f"No se puede consultar el broker: {e}\nEl paper trading opera en la sesión del "
                           f"{session:%d-%m}; las órdenes reales no se prepararán hasta que se resuelva.")
            return
        blockers += account_safety(self.s, snap)
        need = {s for s, w in state.pending.items() if w > 1e-9 and s != CASH}
        held = {self.imap.symbol_of(p.ticker) for p in snap.positions} - {None}
        insts = [self.imap.tradable[s] for s in sorted(need | held) if s in self.imap.tradable]
        quotes = broker.quotes(insts) if insts else {}
        plan = build_plan(plan_id, state.pending, snap, quotes, self.imap.tradable, self.imap.missing,
                          force=view.will_trade or force, limits=_limits(self.s))
        blockers += plan.blockers
        wants = view.will_trade or force or plan.needs_trade
        early = plan.managed_value == 0 and bool(plan.blockers)
        approval = requires_approval(self.s, snap)
        if blockers and (wants or early):
            status = "blocked"
        elif not wants or not plan.orders:
            status = "no_trade"
        elif self.s.exec_mode == "manual":
            status = "planned"
        else:
            status = "awaiting_approval" if approval else "approved"
        reason = (view.order_note if view.will_trade else
                  f"deriva del compte real: {plan.max_deviation:.1%} > {self.s.exec_band:.0%}" if plan.needs_trade else
                  f"sense ordres (desviació {plan.max_deviation:.1%})" if np.isfinite(plan.max_deviation) else "")
        self._write_plan(run, {**base, "account": snap.account, "is_paper_account": snap.is_paper, "status": status,
                               "reason": reason, "blockers": "; ".join(blockers) or None,
                               "notes": "; ".join(plan.notes) or None, "net_liquidation": snap.net_liquidation,
                               "managed_value": plan.managed_value, "cash": snap.cash,
                               "max_deviation": plan.max_deviation, "turnover": plan.turnover,
                               "requires_approval": approval and status == "awaiting_approval"})
        rows = [self._order_row(plan_id, o, "planned") for o in plan.orders]
        if prev is not None:  # replanificació: les ordres que ja no hi són es marquen com a omeses
            keep = {o.symbol for o in plan.orders}
            old = read_orders(self.ctx, plan_id)
            for r in old[~old["symbol"].isin(keep)].to_dict("records"):
                rows.append({**r, "status": "skipped", "message": "ja no és al pla replanificat"})
        self._write_orders(run, rows)
        self._snapshot(run, snap)
        run.ok()
        run.message = f"{plan_id}: {status}" + (f", {len(plan.orders)} ordres" if plan.orders else "") + \
            (f" — {'; '.join(blockers)}" if blockers else "")
        self._plan_alert(plan_id, status, session, snap, plan, blockers, reason)

    @staticmethod
    def _order_row(plan_id: str, o, status: str) -> dict:
        return {"plan_id": plan_id, "symbol": o.symbol, "ref": o.ref, "ticker": o.instrument.ticker,
                "isin": o.instrument.isin, "side": o.side, "quantity": float(o.quantity),
                "current_weight": o.current_weight, "target_weight": o.target_weight, "ref_price": o.ref_price,
                "limit_price": None, "status": status, "broker_order_id": None, "filled_qty": 0.0,
                "avg_price": None, "commission": None, "slippage_bps": None, "message": None}

    def _plan_alert(self, plan_id, status, session, snap, plan: Plan, blockers, reason) -> None:
        cur = snap.currency
        orders = [self._order_row(plan_id, o, "planned") for o in plan.orders]
        txt = _fmt_orders(orders, cur)
        head = (f"Sesión {session:%a %d-%m} (Xetra) · cuenta {snap.account}"
                f"{' (paper IBKR)' if snap.is_paper and self.s.exec_mode == 'ibkr' else ''} · "
                f"valor {plan.managed_value:,.0f} {cur}\nMotivo: {reason}")
        if status == "blocked":
            self.alert(f"exec-blocked:{plan_id}", "critical", f"Plan {plan_id} BLOQUEADO",
                       f"{head}\n\nNo se opera:\n" + "\n".join(f"• {b}" for b in blockers)
                       + (f"\n\nÓrdenes que haría:\n{txt}" if txt else ""))
        elif status == "planned":
            self.alert(f"exec-plan:{plan_id}", "important", f"Órdenes para ejecutar a mano ({len(orders)})",
                       f"{head}\n\n{txt}\n\nEjecuta en IBKR (límite ±{self.s.exec_limit_bps:.0f} pb, solo hoy) y apunta "
                       f"cada ejecución:\nfin exec fill TICKER BUY|SELL CANTIDAD PRECIO --fee COMISIÓN")
        elif status == "awaiting_approval":
            self.alert(f"exec-approve:{plan_id}", "important", f"Aprobar plan {plan_id} ({len(orders)} órdenes)",
                       f"{head}\n\n{txt}\n\nCuenta REAL: no se envía nada sin aprobación.\n"
                       f"Aprobar: fin exec approve {plan_id}\nDescartar: fin exec reject {plan_id}\n"
                       f"Envío automático a las 16:45 (Madrid) si está aprobado.")
        elif status == "approved":
            self.alert(f"exec-plan:{plan_id}", "important", f"Órdenes de mañana: {len(orders)}",
                       f"{head}\n\n{txt}\n\nSe enviarán a las 16:45 (Madrid). Para pararlo: fin exec halt \"motivo\"")

    # -- aprovació -----------------------------------------------------------------------
    def _approve(self, run: Run, plan_id: str, **_: Any) -> None:
        p = get_plan(self.ctx, plan_id)
        if p is None:
            raise ValueError(f"pla desconegut: {plan_id}")
        if p["status"] not in ("awaiting_approval", "planned"):
            raise ValueError(f"{plan_id} és {p['status']}: només es pot aprovar un pla pendent")
        if exec_today(self.ctx.now()) > p["session"]:
            raise ValueError(f"{plan_id} era per al {p['session']:%Y-%m-%d}: ja ha passat")
        self._write_plan(run, {**p, "status": "approved", "approved_at": self.ctx.now()})
        run.ok()
        run.message = f"{plan_id} aprovat"

    def _reject(self, run: Run, plan_id: str, **_: Any) -> None:
        p = get_plan(self.ctx, plan_id)
        if p is None:
            raise ValueError(f"pla desconegut: {plan_id}")
        if p["status"] in ("submitted", "done", "partial"):
            raise ValueError(f"{plan_id} ja s'ha enviat ({p['status']}): fes servir `fin exec halt`")
        self._write_plan(run, {**p, "status": "rejected"})
        rows = [{**r, "status": "skipped", "message": "pla descartat"} for r in read_orders(self.ctx, plan_id).to_dict("records")
                if r["status"] == "planned"]
        self._write_orders(run, rows)
        run.ok()
        run.message = f"{plan_id} descartat"

    # -- enviament -----------------------------------------------------------------------
    def _today_plan(self) -> dict | None:
        plans = read_plans(self.ctx)
        if plans.empty:
            return None
        today = exec_today(self.ctx.now())
        p = plans[(plans["session"] == today) & plans["status"].isin(OPEN)]
        return None if p.empty else get_plan(self.ctx, p.iloc[-1]["plan_id"])

    def _latest_price_date(self) -> pd.Timestamp | None:
        cfg = exec_strategy(self.s)
        df = _q(self.ctx, "SELECT max(date) AS d FROM prices_daily WHERE symbol = ?", [cfg.market])
        return None if df.empty or pd.isna(df["d"].iloc[0]) else pd.Timestamp(df["d"].iloc[0])

    def _submit(self, run: Run, **_: Any) -> None:
        if self.s.exec_mode != "ibkr":
            run.skip("mode manual: les ordres s'executen a mà")
            return
        p = self._today_plan()
        if p is None:
            run.skip("cap pla per a la sessió d'avui")
            return
        pid = p["plan_id"]
        run.params["plan_id"] = pid
        if p["status"] in ("planned", "awaiting_approval"):
            run.skip(f"{pid}: pendent d'aprovació (`fin exec approve {pid}`)")
            self.alert(f"exec-unapproved:{pid}", "important", f"{pid} no se ha enviado: falta aprobación",
                       f"Aprobar antes del cierre de Xetra (17:30): fin exec approve {pid} && fin exec submit")
            return
        blockers = halted(self.s) + market_window(self.ctx.now(), p["session"]) + \
            data_health(self.ctx, self._latest_price_date())
        broker = self.broker()
        snap = broker.account()
        blockers += account_safety(self.s, snap)
        if snap.account != p["account"]:
            blockers.append(f"el pla és del compte {p['account']} i hi ha connectat {snap.account}")
        if blockers:
            run.fail("guards", "; ".join(blockers))
            self.alert(f"exec-guard:{pid}:{exec_today(self.ctx.now()):%Y%m%d}", "critical",
                       f"{pid}: órdenes NO enviadas", "\n".join(f"• {b}" for b in blockers))
            return
        orders = read_orders(self.ctx, pid)
        todo = orders[~orders["status"].isin(FINAL_ORDER)]
        insts = {r.symbol: self.imap.tradable[r.symbol] for r in todo.itertuples() if r.symbol in self.imap.tradable}
        quotes = broker.quotes(list(insts.values()))
        existing = broker.orders()
        held = {p_.ticker: p_.quantity for p_ in snap.positions}

        def send(rows: list[dict]) -> list[dict]:
            out = []
            for r in rows:
                if r["ref"] in existing:
                    st = existing[r["ref"]]
                    out.append({**r, "status": st.status, "broker_order_id": st.broker_order_id,
                                "filled_qty": st.filled_qty, "avg_price": st.avg_price})
                    continue
                inst = insts.get(r["symbol"])
                q = quotes.get(inst.ticker).ref if inst and inst.ticker in quotes else None
                if q is None:
                    out.append({**r, "status": "skipped", "message": "sense cotització en enviar"})
                    continue
                move = q / r["ref_price"] - 1
                if abs(move) > self.s.exec_max_price_move:
                    out.append({**r, "status": "skipped",
                                "message": f"el preu s'ha mogut {move:+.1%} des del pla (> {self.s.exec_max_price_move:.0%})"})
                    continue
                qty = int(r["quantity"])
                if r["side"] == "SELL":
                    qty = int(min(qty, max(held.get(inst.ticker, 0.0), 0.0)))  # mai curts
                if qty <= 0:
                    out.append({**r, "status": "skipped", "message": "res a vendre al compte"})
                    continue
                k = 1 + self.s.exec_limit_bps / 1e4 if r["side"] == "BUY" else 1 - self.s.exec_limit_bps / 1e4
                try:
                    st = broker.place(OrderRequest(ref=r["ref"], instrument=inst, side=r["side"], quantity=qty,
                                                   limit_price=q * k))
                except (ProviderError, TransientProviderError, ValueError) as e:
                    run.fail(r["symbol"], e)
                    out.append({**r, "status": "rejected", "message": str(e)[:300]})
                    continue
                out.append({**r, "quantity": float(qty), "limit_price": q * k, "status": st.status,
                            "broker_order_id": st.broker_order_id, "filled_qty": st.filled_qty,
                            "avg_price": st.avg_price, "message": st.message or None})
            return out

        sells = send([r for r in todo.to_dict("records") if r["side"] == "SELL"])
        self._write_orders(run, sells)
        # Les compres esperen les vendes (l'efectiu ha d'existir: mai marge)
        waited = 0
        while any(r["status"] in ("submitted", "partial") for r in sells) and waited < self.s.exec_wait_fills_s:
            broker.wait(15)
            waited += 15
            st = broker.orders()
            sells = [{**r, **({"status": st[r["ref"]].status, "filled_qty": st[r["ref"]].filled_qty,
                               "avg_price": st[r["ref"]].avg_price} if r["ref"] in st else {})} for r in sells]
        buys = [r for r in todo.to_dict("records") if r["side"] == "BUY"]
        if buys:
            snap2 = broker.account()
            budget = snap2.available - self.s.exec_cash_buffer * float(p["managed_value"] or snap2.net_liquidation)
            new = [r for r in buys if r["ref"] not in existing]
            need = sum(r["quantity"] * r["ref_price"] for r in new)
            if new and need > budget:
                k = max(budget, 0.0) / need
                for r in new:
                    r["quantity"] = float(np.floor(r["quantity"] * k + 1e-9))
                    if r["quantity"] <= 0 or r["quantity"] * r["ref_price"] < self.s.exec_min_order:
                        r["status"] = "skipped"
                        r["message"] = f"efectiu insuficient (disponible {snap2.available:,.0f})"
            done_skip = [r for r in buys if r.get("status") == "skipped"]
            buys = send([r for r in buys if r.get("status") != "skipped"]) + done_skip
        self._write_orders(run, sells + buys)
        self._write_plan(run, {**p, "status": "submitted"})
        sent = [r for r in sells + buys if r["status"] in ("submitted", "partial", "filled")]
        skipped = [r for r in sells + buys if r["status"] in ("skipped", "rejected")]
        run.ok(len(sent))
        run.message = f"{pid}: {len(sent)} ordres enviades" + (f", {len(skipped)} omeses" if skipped else "")
        body = _fmt_orders([r for r in sent], self.s.exec_currency) or "(cap)"
        if skipped:
            body += "\n\nOmitidas:\n" + "\n".join(f"• {r['ticker']}: {r['message']}" for r in skipped)
        self.alert(f"exec-submitted:{pid}", "critical" if skipped else "important",
                   f"{pid}: {len(sent)} órdenes enviadas" + (f", {len(skipped)} omitidas" if skipped else ""), body)

    # -- sincronització --------------------------------------------------------------------
    def _sync(self, run: Run, **_: Any) -> None:
        now = self.ctx.now()
        today = exec_today(now)
        plans = read_plans(self.ctx)
        open_p = plans[plans["status"].isin(OPEN) & (plans["session"] <= today)] if not plans.empty else plans
        broker = self.broker()
        snap = broker.account()
        states, new_fills = {}, []
        if broker.can_submit and not open_p.empty:
            states = broker.orders()
            since = pd.Timestamp(open_p["session"].min()).tz_localize(get_calendar(EXEC_CAL).tz)
            known = set(read_fills(self.ctx).get("exec_id", pd.Series(dtype=str)))
            for f in broker.fills(since):
                if f.exec_id in known:
                    continue
                pid, sym = _parse_ref(f.ref)
                new_fills.append({"exec_id": f.exec_id, "exec_time": f.time, "plan_id": pid, "ref": f.ref,
                                  "symbol": sym, "ticker": f.ticker, "isin": f.isin, "side": f.side,
                                  "quantity": f.quantity, "price": f.price, "commission": f.commission,
                                  "currency": f.currency, "origin": "broker"})
            if new_fills:
                run.write(EXEC_FILLS, pd.DataFrame(new_fills), "exec:ibkr")
        fills = read_fills(self.ctx)
        closed_now = not get_calendar(EXEC_CAL).is_open_on_minute(pd.Timestamp(now).floor("min"))
        summary = []
        for p in open_p.to_dict("records"):
            over = p["session"] < today or (p["session"] == today and closed_now and
                                             pd.Timestamp(now) > get_calendar(EXEC_CAL).session_close(p["session"]))
            rows, status = self._reconcile(read_orders(self.ctx, p["plan_id"]), fills, states, over)
            self._write_orders(run, rows)
            status = status or p["status"]
            if status != p["status"]:
                self._write_plan(run, {**p, "status": status})
            summary.append(f"{p['plan_id']}: {status}")
            if status in ("done", "partial", "expired"):
                self._fills_alert(p, rows, status)
        self._snapshot(run, snap)
        run.ok()
        run.message = "; ".join(summary) or "cap pla obert"
        if new_fills:
            run.message += f"; {len(new_fills)} execucions noves"

    def _reconcile(self, orders: pd.DataFrame, fills: pd.DataFrame, states: dict,
                   over: bool) -> tuple[list[dict], str | None]:
        """Actualitza les ordres amb les execucions. Retorna (files canviades, estat del pla o None
        si no canvia). `over` = la sessió ja ha tancat: el que no s'ha executat ha caducat (DAY)."""
        changed, live = [], []
        for r in orders.to_dict("records"):
            if r["status"] == "skipped":
                continue
            f = fills[fills["ref"] == r["ref"]] if not fills.empty else fills
            qty = float(f["quantity"].sum()) if len(f) else 0.0
            new = dict(r)
            if qty > 0:
                vwap = float((f["quantity"] * f["price"]).sum() / qty)
                new.update(filled_qty=qty, avg_price=vwap, commission=float(f["commission"].fillna(0).sum()),
                           slippage_bps=_slippage(r["side"], vwap, r["ref_price"]))
            st = states.get(r["ref"])
            if qty >= r["quantity"] - 1e-9:
                new["status"] = "filled"
            elif st is not None and st.status in ("cancelled", "rejected"):
                new["status"] = "partial" if qty > 0 else st.status
                new["message"] = st.message or new.get("message")
            elif over:
                new["status"] = "partial" if qty > 0 else "expired"
            elif qty > 0:
                new["status"] = "partial"
            live.append(new)
            if any(_differs(new.get(k), r.get(k)) for k in ("status", "filled_qty", "avg_price", "commission")):
                changed.append(new)
        sts = {x["status"] for x in live}
        if not live:
            return changed, "expired"
        if sts == {"filled"}:
            return changed, "done"
        if not over:
            return changed, ("submitted" if sts & {"submitted", "partial"} else None)
        return changed, ("partial" if sts & {"filled", "partial"} else "expired")

    def _fills_alert(self, p: dict, rows: list[dict], status: str) -> None:
        orders = read_orders(self.ctx, p["plan_id"])
        upd = {r["symbol"]: r for r in rows}
        allr = [upd.get(r["symbol"], r) for r in orders.to_dict("records")]
        lines = []
        for r in allr:
            if r["status"] == "skipped":
                continue
            fq = r.get("filled_qty") or 0
            s = f"• {r['side']} {r['ticker']}: {fq:g}/{r['quantity']:g}"
            if fq:
                s += f" a {r['avg_price']:,.3f}"
                if pd.notna(r.get("slippage_bps")):
                    s += f" (slippage {r['slippage_bps']:+.0f} pb)"
            lines.append(s)
        level = "info" if status == "done" else "critical"
        label = {"done": "ejecutado", "partial": "ejecución PARCIAL", "expired": "NO ejecutado"}[status]
        self.alert(f"exec-result:{p['plan_id']}", level, f"{p['plan_id']}: {label}",
                   "\n".join(lines) + ("" if status == "done" else
                                       "\n\nLo pendiente se replanifica esta noche contra la cartera objetivo."))

    # -- mode manual i efectiu ------------------------------------------------------------
    def _resolve(self, ident: str) -> TradableInstrument:
        ident = ident.upper()
        if ident in self.imap.tradable:
            return self.imap.tradable[ident]
        s = self.imap.symbol_of(ident)
        if s is None:
            raise ValueError(f"{ident}: no és cap instrument operable (config/execution.yaml)")
        return self.imap.tradable[s]

    def _fill(self, run: Run, ident: str, side: str, quantity: float, price: float, fee: float = 0.0,
              when: str | None = None, **_: Any) -> None:
        side = side.upper()
        if side not in ("BUY", "SELL") or quantity <= 0 or price <= 0:
            raise ValueError("cal BUY|SELL, quantitat > 0 i preu > 0")
        inst = self._resolve(ident)
        ts = pd.Timestamp(when, tz="Europe/Madrid").tz_convert("UTC") if when else self.ctx.now()
        # Enllaça amb l'ordre oberta més recent d'aquest símbol i sentit
        orders = read_orders(self.ctx)
        plans = read_plans(self.ctx)
        ref = pid = None
        if not orders.empty:
            open_ids = set(plans.loc[plans["status"].isin(OPEN | {"partial"}), "plan_id"]) if not plans.empty else set()
            cand = orders[(orders["symbol"] == inst.symbol) & (orders["side"] == side) &
                          orders["plan_id"].isin(open_ids) & ~orders["status"].isin({"filled", "skipped"})]
            if not cand.empty:
                o = cand.iloc[-1]
                ref, pid = o["ref"], o["plan_id"]
        n = len(read_fills(self.ctx))
        row = {"exec_id": f"manual-{ts:%Y%m%dT%H%M%S}-{inst.ticker}-{n}", "exec_time": ts, "plan_id": pid,
               "ref": ref, "symbol": inst.symbol, "ticker": inst.ticker, "isin": inst.isin, "side": side,
               "quantity": float(quantity), "price": float(price), "commission": float(fee),
               "currency": inst.currency, "origin": "manual"}
        run.write(EXEC_FILLS, pd.DataFrame([row]), "exec:manual")
        run.ok()
        run.message = f"{side} {quantity:g} {inst.ticker} a {price:,.3f}" + (f" (pla {pid})" if pid else " (sense pla)")
        if pid:
            p = get_plan(self.ctx, pid)
            rows, status = self._reconcile(read_orders(self.ctx, pid), read_fills(self.ctx), {}, over=False)
            self._write_orders(run, rows)
            if p and status == "done" and p["status"] != "done":
                self._write_plan(run, {**p, "status": "done"})

    def _cash(self, run: Run, amount: float, side: str, **_: Any) -> None:
        if amount <= 0:
            raise ValueError("l'import ha de ser > 0")
        ts = self.ctx.now()
        n = len(read_fills(self.ctx))
        run.write(EXEC_FILLS, pd.DataFrame([{
            "exec_id": f"cash-{ts:%Y%m%dT%H%M%S}-{n}", "exec_time": ts, "plan_id": None, "ref": None,
            "symbol": CASH, "ticker": CASH, "isin": None, "side": side, "quantity": float(amount), "price": 1.0,
            "commission": 0.0, "currency": self.s.exec_currency, "origin": "manual"}]), "exec:manual")
        run.ok()
        run.message = f"{side} {amount:,.2f} {self.s.exec_currency}"

    def _deposit(self, run: Run, amount: float, **_: Any) -> None:
        self._cash(run, amount, "DEPOSIT")

    def _withdraw(self, run: Run, amount: float, **_: Any) -> None:
        self._cash(run, amount, "WITHDRAWAL")

    # -- aturada ---------------------------------------------------------------------------
    def _halt(self, run: Run, reason: str = "", **_: Any) -> None:
        self.s.exec_dir.mkdir(parents=True, exist_ok=True)
        self.s.halt_file.write_text(f"{reason or 'sense motiu'} · {self.ctx.now():%Y-%m-%d %H:%M} UTC\n",
                                    encoding="utf-8")
        cancelled = 0
        if self.s.exec_mode == "ibkr":
            try:
                b = self.broker()
                for ref, st in b.orders().items():
                    if st.active:
                        b.cancel(ref)
                        cancelled += 1
            except (ProviderError, TransientProviderError) as e:
                run.fail("broker", f"no s'han pogut cancel·lar les ordres vives: {e}")
        run.ok()
        run.message = f"aturada activada; {cancelled} ordres cancel·lades"
        self.alert(f"exec-halt:{self.ctx.now():%Y%m%dT%H%M}", "critical", "Ejecución DETENIDA",
                   f"Motivo: {reason or '—'}\nÓrdenes vivas canceladas: {cancelled}\nReanudar: fin exec resume")

    def _resume(self, run: Run, **_: Any) -> None:
        existed = self.s.halt_file.exists()
        self.s.halt_file.unlink(missing_ok=True)
        run.ok()
        run.message = "aturada desactivada" if existed else "no hi havia cap aturada"


def _differs(a, b) -> bool:
    if (a is None or (isinstance(a, float) and np.isnan(a))) and (b is None or (isinstance(b, float) and np.isnan(b))):
        return False
    return a != b


def _parse_ref(ref: str | None) -> tuple[str | None, str | None]:
    if not ref or not ref.startswith("fin:"):
        return None, None
    parts = ref.split(":")
    return (parts[1], parts[2]) if len(parts) >= 3 else (None, None)


def _slippage(side: str, price: float, ref: float) -> float | None:
    if not ref or not np.isfinite(ref) or ref <= 0:
        return None
    s = (price - ref) / ref * 1e4
    return float(s if side == "BUY" else -s)
