from datetime import date

import pytest

from finance.ingest import run_job
from finance.providers import ProviderError
from finance.providers.sp500_history import Fja05680History


def _asof(ctx, d):
    return sorted(ctx.query("SELECT symbol FROM universe_asof('sp500', ?)", [d])["symbol"])


def test_history_combined_with_snapshots(ctx, clock):
    run_job(ctx, "universe")  # foto 15-oct: AAPL, BRK-B (GONE ja no hi és)
    r = run_job(ctx, "universe_history")
    assert r.status == "success"

    assert _asof(ctx, date(2000, 1, 3)) == ["AAPL", "OLD"]
    assert _asof(ctx, date(2010, 2, 16)) == ["AAPL", "BRK-B", "OLD"]  # entra el mateix dia
    assert _asof(ctx, date(2020, 6, 22)) == ["AAPL", "BRK-B"]  # end_date exclusiu
    assert _asof(ctx, date(2026, 9, 1)) == ["AAPL", "BRK-B", "GONE"]
    assert _asof(ctx, date(2026, 10, 15)) == ["AAPL", "BRK-B"]

    iv = ctx.query("SELECT * FROM universe_intervals WHERE symbol = 'GONE'").iloc[0]
    # Baixa observada entre l'última data externa i la foto: data conservadora + finestra
    assert iv["end_date"].date() == date(2026, 10, 15)
    assert iv["end_window_days"] == (date(2026, 10, 15) - date(2026, 8, 18)).days
    assert iv["end_source"] == "snapshot" and iv["start_window_days"] == 0
    aapl = ctx.query("SELECT * FROM universe_intervals WHERE symbol = 'AAPL'").iloc[0]
    assert aapl["start_censored"] and aapl["end_date"] is None or str(aapl["end_date"]) == "NaT"

    sectors = ctx.query("SELECT symbol, sector FROM universe_asof('sp500', DATE '2015-01-02')")
    assert dict(zip(sectors["symbol"], sectors["sector"], strict=True))["BRK-B"] == "Financials"

    ch = ctx.query("SELECT date, added_symbol, removed_symbol FROM index_changes ORDER BY date, added_symbol")
    assert len(ch) == 4  # BRK-B entra, OLD surt, GONE entra, GONE surt


def test_rerun_replaces_corrected_history(ctx, providers):
    run_job(ctx, "universe_history")
    # La font corregeix la data d'entrada de BRK-B: no ha de quedar l'interval antic
    h = providers.index_history.history
    h[1] = (date(2010, 3, 1), h[1][1])
    run_job(ctx, "universe_history")
    starts = ctx.query("SELECT start_date FROM universe_intervals WHERE symbol = 'BRK-B'")
    assert [d.date() for d in starts["start_date"]] == [date(2010, 3, 1)]


def test_fallback_to_snapshots_without_history(ctx):
    run_job(ctx, "universe")
    assert _asof(ctx, date(2026, 10, 20)) == ["AAPL", "BRK-B"]


def test_parser_validates():
    rows = "\n".join(f"{1996 + i // 300}-01-{(i % 28) + 1:02d},\"{','.join(f'S{j}' for j in range(500))}\""
                     for i in range(10))
    with pytest.raises(ProviderError, match="truncat"):
        Fja05680History.parse("date,tickers\n" + rows)
    with pytest.raises(ProviderError, match="format"):
        Fja05680History.parse("day,members\n2020-01-01,A")
