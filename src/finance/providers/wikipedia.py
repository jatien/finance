"""Composició d'índexs des de Wikipedia (membres actuals + historial d'altes i baixes)."""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from finance.providers.base import ProviderError, TransientProviderError, UniverseProvider
from finance.providers.ratelimit import with_retries

log = logging.getLogger(__name__)

URLS = {"sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"}
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) finance-ingest/0.1"}


def _flatten(cols: pd.Index) -> list[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(p).strip() for p in c if p and not str(p).startswith("Unnamed")]
            # ('Added', 'Ticker') -> 'added ticker'; ('Reason', 'Reason') -> 'reason'
            dedup = list(dict.fromkeys(parts))
            out.append(" ".join(dedup).lower())
        else:
            out.append(str(c).strip().lower())
    return out


def _pick(cols: list[str], key: str, prefix: bool = False) -> str | None:
    return next((c for c in cols if (c.startswith(key) if prefix else key in c)), None)


def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


class WikipediaIndex(UniverseProvider):
    name = "wikipedia"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._cache: dict[str, list[pd.DataFrame]] = {}

    def _tables(self, universe: str) -> list[pd.DataFrame]:
        if universe not in URLS:
            raise ProviderError(f"univers no suportat: {universe}")
        if universe not in self._cache:
            self._cache[universe] = with_retries(self._download)(URLS[universe])
        return self._cache[universe]

    def _download(self, url: str) -> list[pd.DataFrame]:
        try:
            r = self.session.get(url, timeout=60)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise TransientProviderError(str(e)) from e
        if r.status_code >= 500 or r.status_code == 429:
            raise TransientProviderError(f"HTTP {r.status_code}")
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code}")
        return pd.read_html(io.StringIO(r.text))

    def constituents(self, universe: str) -> pd.DataFrame:
        tables = self._tables(universe)
        for t in tables:
            cols = _flatten(t.columns)
            if "symbol" in cols and any("gics sector" in c for c in cols):
                t = t.copy()
                t.columns = cols
                def pick(key: str, cols: list[str] = cols) -> str | None:
                    return _pick(cols, key)
                out = pd.DataFrame({
                    "symbol": _clean_str(t["symbol"]),
                    "name": _clean_str(t[pick("security")]) if pick("security") else pd.NA,
                    "sector": _clean_str(t[pick("gics sector")]),
                    "sub_industry": _clean_str(t[pick("sub-industry")]) if pick("sub-industry") else pd.NA,
                    "headquarters": _clean_str(t[pick("headquarters")]) if pick("headquarters") else pd.NA,
                    "date_added": pd.to_datetime(t[pick("date added")], errors="coerce").dt.date
                    if pick("date added") else None,
                    "cik": t[pick("cik")].astype("string").str.zfill(10) if pick("cik") else pd.NA,
                })
                out = out.dropna(subset=["symbol"])
                if len(out) < 400:
                    raise ProviderError(f"{universe}: només {len(out)} membres; format canviat?")
                return out.reset_index(drop=True)
        raise ProviderError(f"{universe}: no s'ha trobat la taula de membres")
