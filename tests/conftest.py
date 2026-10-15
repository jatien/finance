"""Fixtures: context amb proveïdors falsos (sense xarxa), lake temporal i rellotge controlable."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance.jobs import Context
from finance.providers import (
    ChainSlice,
    InstrumentProvider,
    MacroProvider,
    OptionChainProvider,
    PriceProvider,
    ProviderError,
    Providers,
    UnderlyingQuote,
    UniverseProvider,
)
from finance.providers.base import PRICE_COLUMNS
from finance.settings import Settings
from finance.storage import Lake

UNIVERSE_YAML = """
defaults: {calendar: XNYS, history_start: "2024-01-02"}
groups:
  etfs:
    asset_class: etf
    symbols: {SPY: {}, XLK: {sector: "Information Technology"}}
  stocks:
    asset_class: stock
    dynamic: sp500
universes:
  prices: [etfs, stocks]
  instruments: [etfs]
options:
  underlyings: [SPY]
  max_dte_days: 60
macro:
  DGS10: {name: "10Y", frequency: D, revisable: false}
  PAYEMS: {name: "Payrolls", frequency: M, revisable: true}
"""


def make_history(symbol: str, start: str = "2024-01-02", end: str = "2026-12-31", seed: int = 0) -> pd.DataFrame:
    from finance.calendar import sessions_between
    days = sessions_between(date.fromisoformat(start), date.fromisoformat(end))
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    return pd.DataFrame({
        "symbol": symbol, "date": days.date, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "adj_close": close, "volume": 1_000_000,
        "dividend": 0.0, "split_ratio": 0.0, "capital_gain": 0.0,
    })


@dataclass
class FakePrices(PriceProvider):
    name: str = "fake"
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    calls: list[tuple[str, date | None, date | None]] = field(default_factory=list)

    def fetch_daily(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if symbol not in self.histories:
            raise ProviderError(f"{symbol}: no existeix")
        df = self.histories[symbol]
        d = pd.to_datetime(df["date"]).dt.date
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= d >= start
        if end is not None:
            mask &= d <= end
        out = df[mask]
        if out.empty:
            raise ProviderError(f"{symbol}: sense dades")
        return out[PRICE_COLUMNS].reset_index(drop=True)

    def add_dividend(self, symbol: str, on: date, amount: float) -> None:
        """Simula un dividend: Yahoo re-escala tot l'adj_close anterior."""
        df = self.histories[symbol].copy()
        d = pd.to_datetime(df["date"]).dt.date
        idx = df.index[d == on][0]
        df.loc[idx, "dividend"] = amount
        prev_close = df.loc[idx - 1, "close"]
        factor = 1 - amount / prev_close
        df.loc[d < on, "adj_close"] = df.loc[d < on, "close"] * factor
        self.histories[symbol] = df


@dataclass
class FakeOptions(OptionChainProvider):
    name: str = "fake"
    expiries: list[date] = field(default_factory=list)
    fail: set[date] = field(default_factory=set)

    def list_expiries(self, underlying):
        return list(self.expiries)

    def fetch_chain(self, underlying, expiry):
        if expiry in self.fail:
            raise ProviderError(f"{underlying} {expiry}: error simulat")
        strikes = [90.0, 100.0, 110.0]
        rows = []
        for otype in ("C", "P"):
            for k in strikes:
                rows.append({
                    "contract_symbol": f"{underlying}{expiry:%y%m%d}{otype}{int(k * 1000):08d}",
                    "option_type": otype, "expiry": expiry, "strike": k,
                    "bid": 1.0 if k != 110 else 0.0, "ask": 1.2, "last_price": 1.1,
                    "last_trade_ts": pd.Timestamp("2026-10-14 15:00", tz="UTC"),
                    "volume": 10.0, "open_interest": 100.0, "implied_vol": 0.2,
                    "in_the_money": False, "contract_size": "REGULAR", "currency": "USD",
                })
        return ChainSlice(contracts=pd.DataFrame(rows),
                          quote=UnderlyingQuote(100.0, 99.9, 100.1, pd.Timestamp.now(tz="UTC"), "REGULAR"),
                          fetched_at=pd.Timestamp.now(tz="UTC"))


@dataclass
class FakeMacro(MacroProvider):
    name: str = "fake"
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    vintages: bool = True
    calls: list[tuple[str, bool, date | None]] = field(default_factory=list)

    @property
    def supports_vintages(self):
        return self.vintages

    def fetch_observations(self, series_id, *, vintages, realtime_start=None):
        self.calls.append((series_id, vintages, realtime_start))
        return self.frames[series_id].copy()


class FakeUniverse(UniverseProvider):
    name = "fake"

    def constituents(self, universe):
        return pd.DataFrame({"symbol": ["AAPL", "BRK.B"], "name": ["Apple", "Berkshire"],
                             "sector": ["Information Technology", "Financials"],
                             "sub_industry": [None, None], "headquarters": [None, None],
                             "date_added": [date(1982, 11, 30), date(2010, 2, 16)],
                             "cik": ["0000320193", "0001067983"]})

    def changes(self, universe):
        raise ProviderError("sense taula de canvis")


class FakeInstruments(InstrumentProvider):
    name = "fake"

    def fetch_info(self, symbol):
        return {"name": symbol, "quote_type": "ETF", "currency": "USD"}


class Clock:
    def __init__(self, ts: str):
        self.ts = pd.Timestamp(ts, tz="UTC")

    def __call__(self) -> pd.Timestamp:
        return self.ts

    def set(self, ts: str) -> None:
        self.ts = pd.Timestamp(ts, tz="UTC")


@pytest.fixture
def clock() -> Clock:
    # Dijous 15-oct-2026 21:00 UTC = 17:00 NY: sessió del dia ja tancada
    return Clock("2026-10-15 21:00")


@pytest.fixture
def providers() -> Providers:
    prices = FakePrices(histories={s: make_history(s, seed=i) for i, s in
                                   enumerate(["SPY", "XLK", "AAPL", "BRK-B"])})
    return Providers(prices=prices, options=FakeOptions(), instruments=FakeInstruments(),
                     macro=FakeMacro(), universe=FakeUniverse())


@pytest.fixture
def ctx(tmp_path: Path, providers: Providers, clock: Clock) -> Context:
    uni = tmp_path / "universe.yaml"
    uni.write_text(UNIVERSE_YAML)
    settings = Settings(data_dir=tmp_path / "data", universe_file=uni, yahoo_max_workers=2,
                        _env_file=None)
    settings.ensure_dirs()
    return Context(settings=settings, lake=Lake(settings.lake_dir), providers=providers, clock=clock)
