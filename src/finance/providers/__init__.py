"""Proveïdors de dades. Construcció centralitzada a partir de la configuració."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from finance.providers.base import (
    AccountSnapshot,
    BrokerClient,
    BrokerFill,
    BrokerPosition,
    BrokerQuote,
    ChainSlice,
    FetchResult,
    IndexHistoryProvider,
    InsiderProvider,
    InstrumentProvider,
    MacroProvider,
    Notifier,
    OptionChainProvider,
    OrderRequest,
    OrderState,
    PriceProvider,
    ProviderError,
    TradableInstrument,
    TransientProviderError,
    UnderlyingQuote,
    UniverseProvider,
)
from finance.providers.ratelimit import RateLimiter
from finance.settings import Settings


@dataclass
class Providers:
    prices: PriceProvider
    options: OptionChainProvider
    instruments: InstrumentProvider
    macro: MacroProvider
    universe: UniverseProvider
    index_history: IndexHistoryProvider
    insiders: InsiderProvider | None = None
    # Setmana 11: canals d'alertes i broker (connexió mandrosa: només quan cal operar)
    notifiers: list[Notifier] = field(default_factory=list)
    broker: Callable[[], BrokerClient] | None = None


def build_providers(settings: Settings) -> Providers:
    from finance.providers.fred import make_fred
    from finance.providers.notify import build_notifiers
    from finance.providers.sec import DisabledInsiders, SecEdgarInsiders
    from finance.providers.sp500_history import Fja05680History
    from finance.providers.wikipedia import WikipediaIndex
    from finance.providers.yahoo import YahooInstruments, YahooOptions, YahooPrices

    # Un sol limitador per a Yahoo: totes les peticions comparteixen el mateix pressupost
    yahoo_limiter = RateLimiter(settings.yahoo_rate_per_sec, burst=max(2, int(settings.yahoo_rate_per_sec * 2)))
    return Providers(
        prices=YahooPrices(yahoo_limiter),
        options=YahooOptions(yahoo_limiter),
        instruments=YahooInstruments(yahoo_limiter),
        macro=make_fred(settings.fred_api_key, settings.fred_rate_per_sec),
        universe=WikipediaIndex(),
        index_history=Fja05680History(),
        insiders=(SecEdgarInsiders(settings.sec_user_agent, settings.sec_rate_per_sec)
                  if settings.sec_user_agent and "@" in settings.sec_user_agent
                  else DisabledInsiders("FIN_SEC_USER_AGENT no definit (la SEC exigeix 'Nom correu')")),
        notifiers=build_notifiers(settings),
        broker=_broker_factory(settings, yahoo_limiter),
    )


def _broker_factory(settings: Settings, limiter: RateLimiter) -> Callable[[], BrokerClient] | None:
    if settings.exec_mode == "ibkr":
        def make() -> BrokerClient:
            from finance.providers.ibkr import IBKRBroker
            return IBKRBroker(settings)
        return make
    return None  # mode manual: el broker és el registre propi (execution/ledger.py)


__all__ = [
    "AccountSnapshot", "BrokerClient", "BrokerFill", "BrokerPosition", "BrokerQuote", "Notifier",
    "OrderRequest", "OrderState", "TradableInstrument",
    "ChainSlice", "FetchResult", "IndexHistoryProvider", "InsiderProvider", "InstrumentProvider", "MacroProvider", "OptionChainProvider",
    "PriceProvider", "ProviderError", "Providers", "TransientProviderError", "UnderlyingQuote",
    "UniverseProvider", "build_providers",
]
