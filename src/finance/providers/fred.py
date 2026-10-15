"""Proveïdor FRED/ALFRED (St. Louis Fed) per a sèries macro, amb vintages point-in-time.

- Amb clau API: endpoint oficial, suporta vintages (`realtime_start`/`realtime_end`).
- Sense clau: CSV públic `fredgraph.csv` (només el valor vigent avui).
"""

from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests

from finance.providers.base import MacroProvider, ProviderError, TransientProviderError
from finance.providers.ratelimit import RateLimiter, with_retries

log = logging.getLogger(__name__)

API_URL = "https://api.stlouisfed.org/fred/series/observations"
GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FAR_FUTURE = date(9999, 12, 31)
EARLIEST_RT = "1776-07-04"  # valor que FRED interpreta com "des de sempre"
PAGE = 100_000


def _get(session: requests.Session, url: str, params: dict, what: str) -> requests.Response:
    try:
        r = session.get(url, params=params, timeout=(10, 45))
    except (requests.ConnectionError, requests.Timeout) as e:
        raise TransientProviderError(f"{what}: {e}") from e
    if r.status_code == 429 or r.status_code >= 500:
        raise TransientProviderError(f"{what}: HTTP {r.status_code}")
    if r.status_code >= 400:
        raise ProviderError(f"{what}: HTTP {r.status_code}: {r.text[:200]}")
    return r


def _to_date(s: pd.Series) -> pd.Series:
    # 9999-12-31 desborda datetime64[ns]: parsejar com a date de Python
    return s.map(lambda x: date.fromisoformat(str(x)[:10]))


class FredApi(MacroProvider):
    name = "fred_api"

    def __init__(self, api_key: str, limiter: RateLimiter | None = None):
        self.api_key = api_key
        self.limiter = limiter or RateLimiter(1.5, burst=2)
        self.session = requests.Session()

    @property
    def supports_vintages(self) -> bool:
        return True

    def fetch_observations(self, series_id: str, *, vintages: bool,
                           realtime_start: date | None = None) -> pd.DataFrame:
        return with_retries(self._fetch, attempts=4, max_wait=20)(series_id, vintages, realtime_start)

    def _fetch(self, series_id: str, vintages: bool, realtime_start: date | None) -> pd.DataFrame:
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json",
                  "limit": PAGE, "sort_order": "asc"}
        if vintages:
            params["realtime_start"] = realtime_start.isoformat() if realtime_start else EARLIEST_RT
            params["realtime_end"] = FAR_FUTURE.isoformat()
        rows: list[dict] = []
        offset = 0
        while True:
            self.limiter.acquire()
            r = _get(self.session, API_URL, {**params, "offset": offset}, series_id)
            payload = r.json()
            obs = payload.get("observations", [])
            rows.extend(obs)
            total = int(payload.get("count", len(rows)))
            offset += len(obs)
            if not obs or offset >= total:
                break
        if not rows:
            raise ProviderError(f"{series_id}: sense observacions")
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
        df["date"] = _to_date(df["date"])
        out = pd.DataFrame({"series_id": series_id, "date": df["date"], "value": df["value"]})
        if vintages:
            out["realtime_start"] = _to_date(df["realtime_start"])
            out["realtime_end"] = _to_date(df["realtime_end"])
            out["vintage_mode"] = "alfred"
        else:
            out["realtime_start"] = out["date"]
            out["realtime_end"] = FAR_FUTURE
            out["vintage_mode"] = "observation_date"
        return out


class FredGraph(MacroProvider):
    """Sense clau. Només valors vigents; per a sèries revisables NO és point-in-time."""

    name = "fred_graph"

    def __init__(self, limiter: RateLimiter | None = None):
        self.limiter = limiter or RateLimiter(1.0, burst=1)
        # Ull: el CDN de FRED rebutja alguns User-Agent personalitzats; es deixa el per defecte
        self.session = requests.Session()

    @property
    def supports_vintages(self) -> bool:
        return False

    def fetch_observations(self, series_id: str, *, vintages: bool,
                           realtime_start: date | None = None) -> pd.DataFrame:
        return with_retries(self._fetch, attempts=4, max_wait=20)(series_id, vintages)

    def _fetch(self, series_id: str, vintages: bool) -> pd.DataFrame:
        self.limiter.acquire()
        r = _get(self.session, GRAPH_URL, {"id": series_id}, series_id)
        raw = pd.read_csv(io.StringIO(r.text))
        if raw.shape[1] < 2 or raw.empty:
            raise ProviderError(f"{series_id}: CSV inesperat")
        dates = _to_date(raw.iloc[:, 0])
        values = pd.to_numeric(raw.iloc[:, 1].replace(".", None), errors="coerce")
        return pd.DataFrame({
            "series_id": series_id,
            "date": dates,
            "value": values,
            "realtime_start": dates,
            "realtime_end": FAR_FUTURE,
            "vintage_mode": "latest_only" if vintages else "observation_date",
        })


def make_fred(api_key: str | None, rate: float = 1.5) -> MacroProvider:
    if api_key:
        return FredApi(api_key, RateLimiter(rate, burst=2))
    log.debug("FIN_FRED_API_KEY no definida: s'usa fredgraph.csv (sense vintages)")
    return FredGraph()
