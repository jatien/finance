"""Broker Interactive Brokers a través d'IB Gateway (API TWS) amb `ib_async`.

Instal·lació: `uv sync --extra broker`. IB Gateway corre al servidor Debian (docs/ejecucion.md):
port 4002 = compte paper, 4001 = compte real. L'API no xifra ni autentica: el port només ha
d'escoltar a 127.0.0.1 (o arribar-hi per túnel SSH).

Salvaguardes pròpies d'aquesta capa (la resta són a `execution/guards.py`):
- Els comptes paper d'IBKR comencen per "D" (DU1234567). Un compte que no hi comença és real.
- Totes les ordres porten `orderRef = fin:<pla>:<símbol>`; abans d'enviar-ne una es comprova
  que el broker no en tingui ja una amb el mateix `ref` (idempotència si el job es reinicia).
- Només ordres límit, vàlides el dia (DAY), dins de l'horari regular.
- Els contractes es resolen per ISIN i es comprova la moneda: mai per ticker sol.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC

import pandas as pd

from finance.providers.base import (
    AccountSnapshot,
    BrokerClient,
    BrokerFill,
    BrokerPosition,
    BrokerQuote,
    OrderRequest,
    OrderState,
    ProviderError,
    TradableInstrument,
    TransientProviderError,
)
from finance.settings import Settings

log = logging.getLogger(__name__)
REF_PREFIX = "fin:"

# orderStatus.status d'IBKR → estat propi
_STATUS = {
    "PendingSubmit": "submitted", "ApiPending": "submitted", "PreSubmitted": "submitted",
    "Submitted": "submitted", "PendingCancel": "submitted", "Filled": "filled",
    "Cancelled": "cancelled", "ApiCancelled": "cancelled", "Inactive": "rejected",
}


def is_paper_account(account: str) -> bool:
    return account.upper().startswith("D")


class IBKRBroker(BrokerClient):
    name = "ibkr"

    def __init__(self, settings: Settings):
        self.s = settings
        self._ib = None
        self._acct: str | None = None
        self._contracts: dict[str, object] = {}  # ticker → Contract qualificat
        self._min_tick: dict[str, float] = {}

    # -- connexió ------------------------------------------------------------------------
    def _conn(self):
        if self._ib is not None and self._ib.isConnected():
            return self._ib
        try:
            from ib_async import IB
        except ImportError as e:  # pragma: no cover
            raise ProviderError("falta ib_async: `uv sync --extra broker`") from e
        ib = IB()
        try:
            ib.connect(self.s.ibkr_host, self.s.ibkr_port, clientId=self.s.ibkr_client_id, timeout=20)
        except Exception as e:  # noqa: BLE001 — ConnectionRefused, TimeoutError...
            raise TransientProviderError(
                f"IB Gateway no respon a {self.s.ibkr_host}:{self.s.ibkr_port} ({type(e).__name__}): "
                "està engegat i amb la sessió iniciada?") from e
        accounts = ib.managedAccounts()
        if self.s.exec_account:
            if self.s.exec_account not in accounts:
                ib.disconnect()
                raise ProviderError(f"el compte {self.s.exec_account} no és dels connectats ({accounts})")
            self._acct = self.s.exec_account
        elif len(accounts) == 1:
            self._acct = accounts[0]
        else:
            ib.disconnect()
            raise ProviderError(f"diversos comptes {accounts}: defineix FIN_EXEC_ACCOUNT")
        ib.reqMarketDataType(self.s.ibkr_market_data_type)
        self._ib = ib
        return ib

    def close(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None

    def wait(self, seconds: float) -> None:
        self._conn().sleep(seconds)

    # -- contractes ----------------------------------------------------------------------
    def _contract(self, inst: TradableInstrument):
        if inst.ticker in self._contracts:
            return self._contracts[inst.ticker]
        from ib_async import Contract
        ib = self._conn()
        probe = Contract(secType="STK", secIdType="ISIN", secId=inst.isin, exchange="SMART",
                         currency=inst.currency)
        details = ib.reqContractDetails(probe)
        if not details:
            raise ProviderError(f"{inst.ticker}: cap contracte per a l'ISIN {inst.isin} en {inst.currency}")
        # Preferència: mateix ticker; si n'hi ha diversos, Xetra
        same = [d for d in details if d.contract.symbol == inst.ticker] or details
        xetra = [d for d in same if (d.contract.primaryExchange or "").startswith("IBIS")]
        d = (xetra or same)[0]
        c = d.contract
        if c.currency != inst.currency:
            raise ProviderError(f"{inst.ticker}: el contracte trobat cotitza en {c.currency}, no en {inst.currency}")
        c.exchange = self.s.exec_exchange
        self._contracts[inst.ticker] = c
        self._min_tick[inst.ticker] = float(d.minTick or 0.001)
        return c

    def round_price(self, ticker: str, price: float, side: str) -> float:
        tick = self._min_tick.get(ticker, 0.001)
        n = price / tick
        n = math.floor(n + 1e-9) if side == "BUY" else math.ceil(n - 1e-9)  # mai més enllà del límit
        return round(n * tick, 6)

    # -- compte --------------------------------------------------------------------------
    def account(self) -> AccountSnapshot:
        ib = self._conn()
        vals = ib.accountSummary(self._acct)
        base = self.s.exec_currency

        def tag(name: str) -> float:
            rows = [v for v in vals if v.tag == name]
            pick = [v for v in rows if v.currency in (base, "BASE")] or rows
            if not pick:
                raise ProviderError(f"IBKR: falta {name} al resum del compte")
            return float(pick[0].value)

        positions = []
        by_conid = {c.conId: t for t, c in self._contracts.items()}
        for p in ib.positions(self._acct):
            if abs(p.position) < 1e-9:
                continue
            c = p.contract
            positions.append(BrokerPosition(ticker=by_conid.get(c.conId, c.symbol), quantity=float(p.position),
                                            avg_cost=float(p.avgCost), currency=c.currency))
        currencies = {v.currency for v in vals if v.tag == "NetLiquidation"}
        cur = base if base in currencies or "BASE" in currencies or not currencies else sorted(currencies)[0]
        return AccountSnapshot(account=self._acct, is_paper=is_paper_account(self._acct), currency=cur,
                               net_liquidation=tag("NetLiquidation"), cash=tag("TotalCashValue"),
                               available=tag("AvailableFunds"), positions=positions)

    def quotes(self, instruments: list[TradableInstrument]) -> dict[str, BrokerQuote]:
        ib = self._conn()
        contracts = {inst.ticker: self._contract(inst) for inst in instruments}
        tickers = ib.reqTickers(*contracts.values())
        by_conid = {c.conId: t for t, c in contracts.items()}
        out = {}
        for tk in tickers:
            t = by_conid.get(tk.contract.conId)
            if t is None:
                continue
            out[t] = BrokerQuote(ticker=t, bid=_num(tk.bid), ask=_num(tk.ask), last=_num(tk.last),
                                 close=_num(tk.close), ts=pd.Timestamp.now(tz="UTC"))
        return out

    # -- ordres --------------------------------------------------------------------------
    def _own_trades(self):
        return [t for t in self._conn().trades() if (t.order.orderRef or "").startswith(REF_PREFIX)]

    def orders(self) -> dict[str, OrderState]:
        out = {}
        for t in self._own_trades():
            st = t.orderStatus
            status = _STATUS.get(st.status, "submitted")
            if status == "submitted" and st.filled > 0:
                status = "partial"
            msg = "; ".join(e.message for e in t.log if e.message)[-300:]
            out[t.order.orderRef] = OrderState(ref=t.order.orderRef, status=status,
                                               broker_order_id=str(t.order.permId or t.order.orderId),
                                               filled_qty=float(st.filled), avg_price=_num(st.avgFillPrice),
                                               message=msg)
        return out

    def place(self, req: OrderRequest) -> OrderState:
        if not req.ref.startswith(REF_PREFIX):
            raise ValueError(f"ref sense prefix {REF_PREFIX}: {req.ref}")
        existing = self.orders()
        if req.ref in existing:  # ja enviada (p. ex. el job es va interrompre): no es duplica
            return existing[req.ref]
        from ib_async import LimitOrder
        ib = self._conn()
        c = self._contract(req.instrument)
        px = self.round_price(req.instrument.ticker, req.limit_price, req.side)
        order = LimitOrder(req.side, int(req.quantity), px, tif=req.tif, orderRef=req.ref,
                           account=self._acct, outsideRth=False, transmit=True)
        trade = ib.placeOrder(c, order)
        ib.sleep(2)
        st = self.orders().get(req.ref)
        if st is None:
            return OrderState(ref=req.ref, status=_STATUS.get(trade.orderStatus.status, "submitted"),
                              broker_order_id=str(trade.order.orderId))
        return st

    def cancel(self, ref: str) -> None:
        ib = self._conn()
        for t in self._own_trades():
            if t.order.orderRef == ref and not t.isDone():
                ib.cancelOrder(t.order)

    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]:
        from ib_async import ExecutionFilter
        ib = self._conn()
        f = ExecutionFilter(acctCode=self._acct)
        if since is not None:
            f.time = pd.Timestamp(since).tz_convert("UTC").strftime("%Y%m%d-%H:%M:%S")
        raw = ib.reqExecutions(f)
        ib.sleep(1)  # els informes de comissió arriben just després
        by_conid = {c.conId: t for t, c in self._contracts.items()}
        out = []
        for fl in raw:
            ex = fl.execution
            if not (ex.orderRef or "").startswith(REF_PREFIX):
                continue
            cr = fl.commissionReport
            ts = pd.Timestamp(fl.time)
            ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
            out.append(BrokerFill(exec_id=ex.execId, ref=ex.orderRef,
                                  ticker=by_conid.get(fl.contract.conId, fl.contract.symbol),
                                  side="BUY" if ex.side == "BOT" else "SELL", quantity=float(ex.shares),
                                  price=float(ex.price),
                                  commission=float(cr.commission) if cr and cr.commission and cr.commission < 1e9 else None,
                                  currency=(cr.currency if cr else None) or fl.contract.currency, time=ts))
        return out


def _num(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None
