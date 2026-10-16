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
