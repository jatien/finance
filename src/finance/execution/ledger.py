"""Mode manual: el "broker" és el registre propi d'execucions (`exec_fills`).

Les posicions i l'efectiu surten de les execucions apuntades amb `fin exec fill` i dels
moviments `fin exec deposit|withdraw` (més `FIN_EXEC_CAPITAL` com a efectiu inicial). La
cotització de referència és l'últim tancament de Yahoo del ticker de Xetra (p. ex. ZPDT.DE),
a través del proveïdor de preus. No envia ordres: el pla arriba per alerta i s'executa a mà.
"""

from __future__ import annotations

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


def read_fills(ctx) -> pd.DataFrame:
    try:
        df = ctx.query("SELECT * EXCLUDE (source, ingested_at, run_id, year) FROM exec_fills ORDER BY exec_time")
    except Exception:  # noqa: BLE001 — dataset encara buit
        return pd.DataFrame()
    if not df.empty:
        df["exec_time"] = pd.to_datetime(df["exec_time"], utc=True)
    return df


def ledger_state(fills: pd.DataFrame, initial_cash: float = 0.0) -> tuple[float, dict[str, float], float]:
    """(efectiu, {ticker: quantitat}, fluxos nets d'aportacions)."""
    cash, flows, pos = float(initial_cash), 0.0, {}
    for r in fills.itertuples(index=False):
        fee = float(r.commission) if pd.notna(r.commission) else 0.0
        if r.side == "DEPOSIT":
            cash += r.quantity
            flows += r.quantity
        elif r.side == "WITHDRAWAL":
            cash -= r.quantity
            flows -= r.quantity
        elif r.side == "BUY":
            cash -= r.quantity * r.price + fee
            pos[r.ticker] = pos.get(r.ticker, 0.0) + r.quantity
        elif r.side == "SELL":
            cash += r.quantity * r.price - fee
            pos[r.ticker] = pos.get(r.ticker, 0.0) - r.quantity
    return cash, {t: q for t, q in pos.items() if abs(q) > 1e-9}, flows


class LedgerBroker(BrokerClient):
    name = "manual"
    can_submit = False

    def __init__(self, ctx, suffix: str = ".DE"):
        self.ctx = ctx
        self.suffix = suffix
        self._quotes: dict[str, BrokerQuote] = {}

    def account(self) -> AccountSnapshot:
        cash, pos, _ = ledger_state(read_fills(self.ctx), self.ctx.settings.exec_capital)
        value = cash
        positions = []
        for t, q in pos.items():
            px = self._last_close(t)
            value += q * (px or 0.0)
            positions.append(BrokerPosition(ticker=t, quantity=q))
        return AccountSnapshot(account="manual", is_paper=False, currency=self.ctx.settings.exec_currency,
                               net_liquidation=value, cash=cash, available=cash, positions=positions)

    def _last_close(self, ticker: str) -> float | None:
        if ticker in self._quotes:
            return self._quotes[ticker].ref
        today = self.ctx.now().date()
        try:
            df = self.ctx.providers.prices.fetch_daily(f"{ticker}{self.suffix}", today - pd.Timedelta(days=14), today)
        except (ProviderError, TransientProviderError):
            return None
        df = df.dropna(subset=["close"]) if not df.empty else df
        if df.empty:
            return None
        last = df.sort_values("date").iloc[-1]
        q = BrokerQuote(ticker=ticker, close=float(last["close"]), ts=pd.Timestamp(last["date"]))
        self._quotes[ticker] = q
        return q.ref

    def quotes(self, instruments: list[TradableInstrument]) -> dict[str, BrokerQuote]:
        out = {}
        for inst in instruments:
            if self._last_close(inst.ticker) is not None:
                out[inst.ticker] = self._quotes[inst.ticker]
        return out

    def place(self, req: OrderRequest) -> OrderState:
        raise ProviderError("mode manual: les ordres s'executen a mà i s'apunten amb `fin exec fill`")

    def orders(self) -> dict[str, OrderState]:
        return {}

    def cancel(self, ref: str) -> None:
        return None

    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]:
        return []
