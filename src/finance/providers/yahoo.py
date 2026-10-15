"""Proveïdor Yahoo Finance (via yfinance): preus diaris, accions corporatives, opcions, metadades.

Notes de Yahoo:
- `Close` ve ajustat per splits però NO per dividends; `Adj Close` sí per tots dos.
- Les dades d'opcions tenen un retard de ~15 min durant la sessió i els bid/ask poden
  sortir a 0 fora d'hores. Per això les fotos es programen dins la sessió.
- No és una font institucional: la capa de proveïdors està pensada per substituir-la.
"""

from __future__ import annotations

import contextlib
import logging
import re
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
import yfinance.exceptions as yfe

from finance.providers.base import (
    PRICE_COLUMNS,
    ChainSlice,
    InstrumentProvider,
    OptionChainProvider,
    PriceProvider,
    ProviderError,
    TransientProviderError,
    UnderlyingQuote,
)
from finance.providers.ratelimit import RateLimiter, with_retries

log = logging.getLogger(__name__)

# Fer que yfinance llanci excepcions en lloc d'imprimir-les i retornar buit
with contextlib.suppress(AttributeError):  # versions antigues no tenen yf.config
    yf.config.debug.hide_exceptions = False  # type: ignore[attr-defined]

_DEFINITIVE = (yfe.YFPricesMissingError, yfe.YFTickerMissingError, yfe.YFTzMissingError,
               yfe.YFInvalidPeriodError, yfe.YFNotImplementedError)


def _classify(exc: Exception, what: str) -> Exception:
    if isinstance(exc, (ProviderError, TransientProviderError)):
        return exc
    if isinstance(exc, yfe.YFRateLimitError):
        return TransientProviderError(f"{what}: límit de peticions de Yahoo")
    if isinstance(exc, _DEFINITIVE):
        return ProviderError(f"{what}: {exc}")
    m = re.search(r"HTTP Error (\d{3})", str(exc))
    if m:
        code = int(m.group(1))
        # 401 = crumb/cookie caducat (yfinance el renova); 404 = símbol inexistent
        if 400 <= code < 500 and code not in (401, 408, 429):
            return ProviderError(f"{what}: HTTP {code}")
    name = type(exc).__name__.lower()
    if isinstance(exc, (ConnectionError, TimeoutError)) or "timeout" in name or "connection" in name:
        return TransientProviderError(f"{what}: {type(exc).__name__}: {exc}")
    if isinstance(exc, (KeyError, ValueError, TypeError, IndexError)):
        return ProviderError(f"{what}: {type(exc).__name__}: {exc}")
    # HTTP 5xx, errors de curl, etc.: tractar com a transitoris (amb reintents limitats)
    return TransientProviderError(f"{what}: {type(exc).__name__}: {exc}")


def _f(x: Any) -> float | None:
    try:
        v = float(x)
        return v if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


class _YahooBase:
    name = "yahoo"

    def __init__(self, limiter: RateLimiter | None = None):
        self.limiter = limiter or RateLimiter(2.0, burst=4)

    def _ticker(self, symbol: str) -> yf.Ticker:
        return yf.Ticker(symbol)


class YahooPrices(_YahooBase, PriceProvider):
    def fetch_daily(self, symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        return with_retries(self._fetch)(symbol, start, end)

    def _fetch(self, symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        self.limiter.acquire()
        kwargs: dict[str, Any] = dict(interval="1d", auto_adjust=False, actions=True, repair=False)
        if start is None:
            kwargs["period"] = "max"
        else:
            kwargs["start"] = start.isoformat()
        if end is not None:
            kwargs["end"] = (end + timedelta(days=1)).isoformat()  # Yahoo: end exclusiu
        try:
            raw = self._ticker(symbol).history(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise _classify(e, symbol) from e
        if raw is None or raw.empty:
            raise ProviderError(f"{symbol}: sense dades en el rang")
        return self.normalize(symbol, raw)

    @staticmethod
    def normalize(symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)  # data local de la borsa
        df = pd.DataFrame({
            "symbol": symbol,
            "date": idx.normalize().date,
            "open": raw.get("Open"),
            "high": raw.get("High"),
            "low": raw.get("Low"),
            "close": raw.get("Close"),
            "adj_close": raw.get("Adj Close", raw.get("Close")),
            "volume": raw.get("Volume"),
            "dividend": raw.get("Dividends", 0.0),
            "split_ratio": raw.get("Stock Splits", 0.0),
            "capital_gain": raw.get("Capital Gains", 0.0),
        }).reset_index(drop=True)
        for c in ("dividend", "split_ratio", "capital_gain"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # Files sense cap preu (Yahoo n'hi posa en dies de dividend fora de sessió)
        df = df.dropna(subset=["open", "high", "low", "close"], how="all")
        return df[PRICE_COLUMNS].drop_duplicates(subset=["date"], keep="last")


class YahooOptions(_YahooBase, OptionChainProvider):
    def list_expiries(self, underlying: str) -> list[date]:
        return with_retries(self._expiries)(underlying)

    def _expiries(self, underlying: str) -> list[date]:
        self.limiter.acquire()
        try:
            exps = self._ticker(underlying).options
        except Exception as e:  # noqa: BLE001
            raise _classify(e, f"{underlying} expiries") from e
        if not exps:
            raise ProviderError(f"{underlying}: sense opcions llistades")
        return sorted(date.fromisoformat(x) for x in exps)

    def fetch_chain(self, underlying: str, expiry: date) -> ChainSlice:
        return with_retries(self._chain)(underlying, expiry)

    def _chain(self, underlying: str, expiry: date) -> ChainSlice:
        self.limiter.acquire()
        fetched_at = pd.Timestamp.now(tz="UTC")
        try:
            oc = self._ticker(underlying).option_chain(expiry.isoformat())
        except Exception as e:  # noqa: BLE001
            raise _classify(e, f"{underlying} {expiry}") from e
        parts = []
        for frame, otype in ((oc.calls, "C"), (oc.puts, "P")):
            if frame is not None and not frame.empty:
                parts.append(frame.assign(option_type=otype))
        if not parts:
            raise ProviderError(f"{underlying} {expiry}: cadena buida")
        raw = pd.concat(parts, ignore_index=True)
        contracts = pd.DataFrame({
            "contract_symbol": raw["contractSymbol"],
            "option_type": raw["option_type"],
            "expiry": expiry,
            "strike": raw["strike"],
            "bid": raw.get("bid"),
            "ask": raw.get("ask"),
            "last_price": raw.get("lastPrice"),
            "last_trade_ts": raw.get("lastTradeDate"),
            "volume": raw.get("volume"),
            "open_interest": raw.get("openInterest"),
            "implied_vol": raw.get("impliedVolatility"),
            "in_the_money": raw.get("inTheMoney"),
            "contract_size": raw.get("contractSize"),
            "currency": raw.get("currency"),
        })
        u = getattr(oc, "underlying", None) or {}
        qts = u.get("regularMarketTime")
        quote = UnderlyingQuote(
            price=_f(u.get("regularMarketPrice")),
            bid=_f(u.get("bid")),
            ask=_f(u.get("ask")),
            quote_ts=pd.Timestamp(qts, unit="s", tz="UTC") if qts else None,
            market_state=u.get("marketState"),
        )
        return ChainSlice(contracts=contracts, quote=quote, fetched_at=fetched_at)


class YahooInstruments(_YahooBase, InstrumentProvider):
    def fetch_info(self, symbol: str) -> dict[str, Any]:
        return with_retries(self._info)(symbol)

    def _info(self, symbol: str) -> dict[str, Any]:
        self.limiter.acquire()
        try:
            info = self._ticker(symbol).info or {}
        except Exception as e:  # noqa: BLE001
            raise _classify(e, symbol) from e
        if not info or info.get("quoteType") is None:
            raise ProviderError(f"{symbol}: sense metadades")
        ftd = info.get("firstTradeDateMilliseconds")
        if ftd is None and info.get("firstTradeDateEpochUtc") is not None:
            ftd = info["firstTradeDateEpochUtc"] * 1000
        return {
            "name": info.get("longName") or info.get("shortName"),
            "quote_type": info.get("quoteType"),
            "exchange": info.get("exchange"),
            "currency": info.get("currency"),
            "country": info.get("country"),
            "sector": info.get("sector") or info.get("sectorDisp"),
            "industry": info.get("industry") or info.get("industryDisp"),
            "category": info.get("category"),
            "fund_family": info.get("fundFamily"),
            "market_cap": _f(info.get("marketCap")),
            "shares_outstanding": _f(info.get("sharesOutstanding")),
            "expense_ratio": _f(info.get("netExpenseRatio") or info.get("annualReportExpenseRatio")),
            "total_assets": _f(info.get("totalAssets") or info.get("netAssets")),
            "beta": _f(info.get("beta") or info.get("beta3Year")),
            "first_trade_date": pd.Timestamp(ftd, unit="ms").date() if ftd else None,
        }
