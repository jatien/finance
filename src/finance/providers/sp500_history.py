"""Historial de composició del S&P 500 des de 1996 (repositori públic fja05680/sp500).

Origen: llista d'Andreas Clenow (*Trading Evolved*, 1996–2019) ampliada pel mantenidor amb
els canvis posteriors, verificats contra Wikipedia i notes de premsa. Format: una fila per
data de canvi amb tots els membres vigents a partir d'aquella data.

Limitacions conegudes (documentades per l'autor):
- 1996–2000 té entre 487 i 494 membres: probablement hi falten alguns símbols.
- És a nivell de ticker: un canvi de ticker apareix com una baixa i una alta.
- S'actualitza cada pocs mesos; el tram fins avui el cobreixen les fotes pròpies.
"""

from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests

from finance.providers.base import IndexHistoryProvider, ProviderError, TransientProviderError
from finance.providers.ratelimit import with_retries
from finance.universe import to_yahoo_symbol

log = logging.getLogger(__name__)

URLS = {
    "sp500": (
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
        "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
    ),
}
MIN_ROWS = 1000
MIN_MEMBERS, MAX_MEMBERS = 450, 520


class Fja05680History(IndexHistoryProvider):
    name = "fja05680"

    def __init__(self, urls: dict[str, str] | None = None):
        self.urls = urls or URLS
        self.session = requests.Session()

    def membership_history(self, universe: str) -> list[tuple[date, frozenset[str]]]:
        if universe not in self.urls:
            raise ProviderError(f"sense historial per a l'univers {universe}")
        text = with_retries(self._download, attempts=4, max_wait=30)(self.urls[universe])
        return self.parse(text)

    def _download(self, url: str) -> str:
        try:
            r = self.session.get(url, timeout=(10, 90))
        except (requests.ConnectionError, requests.Timeout) as e:
            raise TransientProviderError(str(e)) from e
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientProviderError(f"HTTP {r.status_code}")
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code}: el fitxer s'ha mogut?")
        return r.text

    @staticmethod
    def parse(text: str) -> list[tuple[date, frozenset[str]]]:
        df = pd.read_csv(io.StringIO(text), dtype=str)
        if list(df.columns[:2]) != ["date", "tickers"]:
            raise ProviderError(f"format inesperat: columnes {list(df.columns)}")
        df = df.dropna(subset=["date", "tickers"])
        out: list[tuple[date, frozenset[str]]] = []
        for d, tickers in zip(df["date"], df["tickers"], strict=True):
            members = frozenset(to_yahoo_symbol(t) for t in tickers.split(",") if t.strip())
            out.append((date.fromisoformat(d.strip()[:10]), members))
        out.sort(key=lambda x: x[0])
        # Validacions: un canvi de format o un fitxer truncat ha de fallar, no contaminar
        if len(out) < MIN_ROWS:
            raise ProviderError(f"només {len(out)} files: fitxer truncat?")
        if len({d for d, _ in out}) != len(out):
            raise ProviderError("dates duplicades")
        sizes = [len(m) for _, m in out]
        if min(sizes) < MIN_MEMBERS or max(sizes) > MAX_MEMBERS:
            raise ProviderError(f"mida de l'índex fora de rang: {min(sizes)}–{max(sizes)}")
        return out
