"""Interfícies dels proveïdors de dades.

Cada proveïdor retorna DataFrames ja normalitzats als noms de columna dels datasets, de
manera que canviar Yahoo per Polygon, Tiingo, IBKR o Databento només requereix una nova
implementació d'aquestes classes, sense tocar jobs ni emmagatzematge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


class ProviderError(Exception):
    """Error definitiu (no s'ha de reintentar): símbol inexistent, sense dades..."""


class TransientProviderError(Exception):
    """Error transitori (límit de peticions, xarxa): es pot reintentar."""


@dataclass
class FetchResult:
    data: pd.DataFrame
    errors: dict[str, str] = field(default_factory=dict)


# Columnes que ha de retornar un PriceProvider
PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume",
                 "dividend", "split_ratio", "capital_gain"]

# Columnes que ha de retornar un OptionChainProvider (per contracte)
OPTION_COLUMNS = ["contract_symbol", "option_type", "expiry", "strike", "bid", "ask",
                  "last_price", "last_trade_ts", "volume", "open_interest", "implied_vol",
                  "in_the_money", "contract_size", "currency"]


class PriceProvider(ABC):
    name: str

    @abstractmethod
    def fetch_daily(self, symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        """Barres diàries [start, end] (ambdós inclosos) amb les columnes PRICE_COLUMNS.

        `start=None` vol dir tot l'històric disponible. Llança ProviderError o
        TransientProviderError.
        """


@dataclass
class UnderlyingQuote:
    price: float | None
    bid: float | None
    ask: float | None
    quote_ts: pd.Timestamp | None
    market_state: str | None


@dataclass
class ChainSlice:
    """Cadena d'una expiració: contractes + cotització del subjacent en aquell moment."""
    contracts: pd.DataFrame
    quote: UnderlyingQuote
    fetched_at: pd.Timestamp


class OptionChainProvider(ABC):
    name: str

    @abstractmethod
    def list_expiries(self, underlying: str) -> list[date]: ...

    @abstractmethod
    def fetch_chain(self, underlying: str, expiry: date) -> ChainSlice: ...


class MacroProvider(ABC):
    name: str

    @abstractmethod
    def fetch_observations(
        self,
        series_id: str,
        *,
        vintages: bool,
        realtime_start: date | None = None,
    ) -> pd.DataFrame:
        """Columnes: series_id, date, value, realtime_start, realtime_end, vintage_mode."""

    @property
    @abstractmethod
    def supports_vintages(self) -> bool: ...


class UniverseProvider(ABC):
    name: str

    @abstractmethod
    def constituents(self, universe: str) -> pd.DataFrame:
        """Columnes: symbol, name, sector, sub_industry, headquarters, date_added, cik."""



class IndexHistoryProvider(ABC):
    """Historial de composició d'un índex: el conjunt de membres vigent a cada data de canvi."""

    name: str

    @abstractmethod
    def membership_history(self, universe: str) -> list[tuple[date, frozenset[str]]]:
        """Llista ordenada de (data, membres) amb símbols en convenció Yahoo.

        Cada conjunt és vigent des de la seva data fins a la següent (exclosa).
        """


class InstrumentProvider(ABC):
    name: str

    @abstractmethod
    def fetch_info(self, symbol: str) -> dict[str, Any]:
        """Diccionari amb les columnes del dataset `instruments` (sense symbol/as_of_date)."""


class InsiderProvider(ABC):
    """Operacions d'insiders (formularis 4/5 de la SEC) normalitzades a `INSIDER_COLUMNS`."""

    name: str

    @abstractmethod
    def list_quarters(self) -> list[Any]:
        """Datasets trimestrals disponibles (objectes amb `year`, `quarter`, `key`)."""

    @abstractmethod
    def fetch_quarter(self, q: Any) -> pd.DataFrame:
        """Totes les operacions no derivades d'un trimestre."""

    def index_days(self, year: int, quarter: int) -> set[date] | None:
        """Dies amb índex diari publicat en un trimestre (None = desconegut: prova'ls tots)."""
        return None

    @abstractmethod
    def daily_index(self, day: date) -> pd.DataFrame:
        """Presentacions d'un dia: form_type, name, cik, date_filed, path, accession_number."""

    @abstractmethod
    def fetch_filing(self, path: str, accession: str, filing_date: date) -> pd.DataFrame:
        """Operacions d'una presentació concreta (via XML)."""


# ---------------------------------------------------------------------------------------
# Setmana 11: alertes i execució real
# ---------------------------------------------------------------------------------------

class Notifier(ABC):
    """Canal d'alertes (ntfy, Telegram, correu...). Llança ProviderError si no es lliura."""

    name: str

    @abstractmethod
    def send(self, title: str, body: str, level: str = "info") -> None: ...


@dataclass(frozen=True)
class TradableInstrument:
    """Instrument que s'opera de veritat per a una sèrie del senyal (p. ex. XLK → ZPDT a Xetra)."""
    symbol: str  # sèrie del senyal (ETF dels EUA)
    ticker: str  # ticker a la plaça d'execució
    isin: str
    currency: str = "EUR"
    name: str = ""


@dataclass
class BrokerPosition:
    ticker: str
    quantity: float
    avg_cost: float | None = None
    isin: str | None = None
    currency: str | None = None


@dataclass
class AccountSnapshot:
    account: str
    is_paper: bool
    currency: str
    net_liquidation: float
    cash: float
    available: float  # efectiu disponible per comprar (sense marge)
    positions: list[BrokerPosition] = field(default_factory=list)


@dataclass
class BrokerQuote:
    ticker: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    close: float | None = None
    ts: pd.Timestamp | None = None

    @property
    def ref(self) -> float | None:
        """Preu de referència: punt mig si hi ha bid/ask vàlids, si no l'últim, si no el tancament."""
        def ok(x):
            return x is not None and x == x and x > 0
        if ok(self.bid) and ok(self.ask) and self.ask >= self.bid:
            return (self.bid + self.ask) / 2
        for x in (self.last, self.close):
            if ok(x):
                return float(x)
        return None


@dataclass(frozen=True)
class OrderRequest:
    ref: str  # identificador propi (orderRef): idempotència entre execucions
    instrument: TradableInstrument
    side: str  # BUY | SELL
    quantity: int
    limit_price: float
    tif: str = "DAY"


@dataclass
class OrderState:
    ref: str
    status: str  # submitted | filled | partial | cancelled | rejected | expired
    broker_order_id: str | None = None
    filled_qty: float = 0.0
    avg_price: float | None = None
    message: str = ""

    @property
    def active(self) -> bool:
        return self.status in ("submitted", "partial")


@dataclass
class BrokerFill:
    exec_id: str
    ref: str | None
    ticker: str
    side: str  # BUY | SELL
    quantity: float
    price: float
    commission: float | None
    currency: str | None
    time: pd.Timestamp
    isin: str | None = None


class BrokerClient(ABC):
    """Interfície del broker. La implementació real és IBKR (`providers/ibkr.py`); els tests
    fan servir un broker simulat. El codi d'execució només coneix aquesta interfície."""

    name: str
    can_submit: bool = True  # False = mode manual: el pla s'executa a mà

    @abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abstractmethod
    def quotes(self, instruments: list[TradableInstrument]) -> dict[str, BrokerQuote]:
        """Cotització per ticker (pot faltar si no n'hi ha)."""

    @abstractmethod
    def place(self, req: OrderRequest) -> OrderState: ...

    @abstractmethod
    def orders(self) -> dict[str, OrderState]:
        """Estat de les ordres d'avui conegudes pel broker, per `ref` (només les pròpies)."""

    @abstractmethod
    def cancel(self, ref: str) -> None: ...

    @abstractmethod
    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]: ...

    def wait(self, seconds: float) -> None:
        import time
        time.sleep(seconds)

    def close(self) -> None:  # noqa: B027 — opcional
        pass
