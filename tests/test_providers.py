from datetime import date

import pandas as pd
import pytest

from finance.providers.base import ProviderError, TransientProviderError
from finance.providers.fred import FredApi
from finance.providers.ratelimit import RateLimiter, with_retries
from finance.providers.yahoo import YahooPrices, _classify
from finance.universe import to_yahoo_symbol


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.text = payload, status, ""

    def json(self):
        return self._p


def test_fred_api_paging_and_vintages(monkeypatch):
    pages = [
        {"count": 3, "observations": [
            {"date": "2026-09-01", "value": "100", "realtime_start": "2026-10-02", "realtime_end": "2026-11-05"},
            {"date": "2026-09-01", "value": "105", "realtime_start": "2026-11-06", "realtime_end": "9999-12-31"}]},
        {"count": 3, "observations": [
            {"date": "2026-10-01", "value": ".", "realtime_start": "2026-11-06", "realtime_end": "9999-12-31"}]},
    ]
    seen = []

    def fake_get(session, url, params, what):
        seen.append(params["offset"])
        return _Resp(pages[len(seen) - 1])

    monkeypatch.setattr("finance.providers.fred._get", fake_get)
    df = FredApi("k", RateLimiter(1000, burst=1000)).fetch_observations("PAYEMS", vintages=True)
    assert seen == [0, 2]
    assert df["realtime_end"].tolist()[-1] == date(9999, 12, 31)
    assert pd.isna(df["value"].iloc[2])  # "." = valor absent
    assert set(df["vintage_mode"]) == {"alfred"}


def test_retry_only_transient():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientProviderError("429")
        return "ok"

    assert with_retries(flaky, max_wait=0.01)() == "ok" and calls["n"] == 3

    def bad():
        calls["n"] += 1
        raise ProviderError("404")

    calls["n"] = 0
    with pytest.raises(ProviderError):
        with_retries(bad, max_wait=0.01)()
    assert calls["n"] == 1


def test_yahoo_error_classification():
    assert isinstance(_classify(Exception("HTTP Error 404: Not Found"), "X"), ProviderError)
    assert isinstance(_classify(Exception("HTTP Error 401: Unauthorized"), "X"), TransientProviderError)
    assert isinstance(_classify(TimeoutError("t"), "X"), TransientProviderError)


def test_yahoo_normalize():
    idx = pd.DatetimeIndex(["2026-10-14", "2026-10-15"]).tz_localize("America/New_York")
    raw = pd.DataFrame({"Open": [1.0, 2.0], "High": [1, 2], "Low": [1, 2], "Close": [1, 2],
                        "Adj Close": [0.9, 2], "Volume": [10, 20], "Dividends": [0.1, 0],
                        "Stock Splits": [0, 0]}, index=idx)
    df = YahooPrices.normalize("SPY", raw)
    assert df["date"].tolist() == [date(2026, 10, 14), date(2026, 10, 15)]
    assert df["capital_gain"].tolist() == [0.0, 0.0]


def test_symbol_normalization():
    assert to_yahoo_symbol("brk.b") == "BRK-B"
    assert to_yahoo_symbol("^VIX") == "^VIX"
