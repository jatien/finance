from datetime import date

from finance.ingest import run_job


def _cov(ctx):
    return ctx.query("SELECT symbol, last_date, n_rows FROM prices_coverage ORDER BY symbol").set_index("symbol")


def test_requires_universe_for_dynamic_group_but_static_still_runs(ctx):
    r = run_job(ctx, "prices")
    assert r.status == "success"
    assert set(_cov(ctx).index) == {"SPY", "XLK"}  # sp500 encara no resolt


def test_full_then_incremental(ctx, providers, clock):
    run_job(ctx, "universe")
    r = run_job(ctx, "prices")
    assert r.status == "success" and r.items_ok == 4
    cov = _cov(ctx)
    assert set(cov.index) == {"AAPL", "BRK-B", "SPY", "XLK"}
    assert (cov["last_date"].dt.date == date(2026, 10, 15)).all()

    # Mateix dia: res a fer
    providers.prices.calls.clear()
    r = run_job(ctx, "prices")
    assert providers.prices.calls == [] and r.message == "tot al dia"

    # Endemà en plena sessió: només fins a la sessió tancada (16-oct no, és parcial)
    clock.set("2026-10-16 15:00")
    assert run_job(ctx, "prices").rows_written == 0
    # Endemà després del tancament: incremental amb solapament
    clock.set("2026-10-16 22:00")
    providers.prices.calls.clear()
    run_job(ctx, "prices")
    starts = {c[1] for c in providers.prices.calls}
    assert starts == {date(2026, 10, 5)}  # 15-oct menys 10 dies
    assert (_cov(ctx)["last_date"].dt.date == date(2026, 10, 16)).all()


def test_new_dividend_triggers_full_reload(ctx, providers, clock):
    run_job(ctx, "prices", symbols=["SPY"])
    providers.prices.add_dividend("SPY", date(2026, 10, 16), 2.0)
    clock.set("2026-10-16 22:00")
    providers.prices.calls.clear()
    r = run_job(ctx, "prices", symbols=["SPY"])
    assert "re-ajust" in r.message
    assert [c[1] for c in providers.prices.calls] == [date(2026, 10, 5), date(2024, 1, 2)]
    # L'històric guardat reflecteix el nou ajust
    old = ctx.query("SELECT adj_close, close FROM prices_daily WHERE symbol='SPY' AND date='2025-01-02'")
    assert old["adj_close"].iloc[0] < old["close"].iloc[0]
    acts = ctx.query("SELECT * FROM corporate_actions WHERE symbol='SPY'")
    assert acts[["action", "value"]].values.tolist() == [["dividend", 2.0]]


def test_bad_symbol_gives_partial(ctx):
    r = run_job(ctx, "prices", symbols=["SPY", "NOPE"])
    assert r.status == "partial" and "NOPE" in r.errors
    runs = ctx.query("SELECT status, items_failed FROM ingest_runs")
    assert runs.to_dict("records") == [{"status": "partial", "items_failed": 1}]


def test_provisional_bar_correction_does_not_reload(ctx, providers, clock):
    run_job(ctx, "prices", symbols=["SPY"])
    h = providers.prices.histories["SPY"]
    d = __import__("pandas").to_datetime(h["date"]).dt.date
    h.loc[d == date(2026, 10, 15), ["close", "adj_close"]] *= 1.003  # correcció de Yahoo
    clock.set("2026-10-16 22:00")
    providers.prices.calls.clear()
    r = run_job(ctx, "prices", symbols=["SPY"])
    assert len(providers.prices.calls) == 1 and "re-ajust" not in r.message
    final = ctx.query("SELECT close FROM prices_daily WHERE symbol='SPY' AND date='2026-10-15'")
    assert final["close"].iloc[0] == h.loc[d == date(2026, 10, 15), "close"].iloc[0]
