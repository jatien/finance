from datetime import date

import pandas as pd

from finance.ingest import run_job

FAR = date(9999, 12, 31)


# --------------------------------------------------------------------------- opcions
def test_options_skip_when_closed(ctx):
    r = run_job(ctx, "options")
    assert r.status == "skipped"


def test_options_snapshot(ctx, providers, clock):
    clock.set("2026-10-15 15:30")  # sessió oberta
    providers.options.expiries = [date(2026, 10, 16), date(2026, 11, 20), date(2027, 6, 18)]
    providers.options.fail = {date(2026, 11, 20)}
    r = run_job(ctx, "options")
    assert r.status == "partial" and r.items_ok == 1  # una expiració falla, el subjacent compta
    df = ctx.query("SELECT * FROM option_snapshots")
    assert set(df["expiry"].dt.date) == {date(2026, 10, 16)}  # 2027 fora de max_dte (60)
    assert set(df["dte"]) == {1}
    assert df.loc[df["strike"] == 110, "mid"].isna().all()  # bid=0 -> sense mid
    assert (df.loc[df["strike"] == 100, "mid"] == 1.1).all()
    assert df["snapshot_ts"].nunique() == 1


# --------------------------------------------------------------------------- macro
def _vintage_frame(rows):
    return pd.DataFrame(rows, columns=["date", "value", "realtime_start", "realtime_end"]).assign(
        series_id="PAYEMS", vintage_mode="alfred")


def test_macro_vintage_reanchoring(ctx, providers, clock):
    m = providers.macro
    m.frames["DGS10"] = pd.DataFrame({
        "series_id": "DGS10", "date": [date(2026, 10, 14), date(2026, 10, 15)], "value": [4.1, 4.2],
        "realtime_start": [date(2026, 10, 14), date(2026, 10, 15)], "realtime_end": FAR,
        "vintage_mode": "observation_date"})
    # Càrrega inicial: setembre publicat el 2-oct
    m.frames["PAYEMS"] = _vintage_frame([(date(2026, 9, 1), 100.0, date(2026, 10, 2), FAR)])
    assert run_job(ctx, "macro").status == "success"

    # Un mes després: revisió de setembre (6-nov). FRED retorna el vintage antic retallat a W.
    clock.set("2026-11-10 21:00")
    w = date(2026, 10, 8)  # darrera ingesta (15-oct) - 7 dies
    m.frames["PAYEMS"] = _vintage_frame([
        (date(2026, 9, 1), 100.0, w, date(2026, 11, 5)),
        (date(2026, 9, 1), 105.0, date(2026, 11, 6), FAR),
        (date(2026, 10, 1), 200.0, date(2026, 11, 6), FAR),
    ])
    run_job(ctx, "macro", series=["PAYEMS"])
    assert m.calls[-1] == ("PAYEMS", True, w)
    obs = ctx.query("SELECT date, value, realtime_start, realtime_end FROM macro_observations "
                    "WHERE series_id='PAYEMS' ORDER BY date, realtime_start")
    assert len(obs) == 3  # cap interval duplicat
    first = obs.iloc[0]
    assert first["realtime_start"].date() == date(2026, 10, 2)  # reancorat a l'original
    assert first["realtime_end"].date() == date(2026, 11, 5)  # tancat per la revisió

    # Point-in-time: el 20-oct es coneixia 100, el 10-nov es coneix 105
    pit = lambda d: ctx.query(  # noqa: E731
        "SELECT value FROM macro_asof(?) WHERE series_id='PAYEMS' AND date='2026-09-01'", [d])["value"].tolist()
    assert pit(date(2026, 10, 20)) == [100.0]
    assert pit(date(2026, 11, 10)) == [105.0]

    run_job(ctx, "dq", datasets=["macro_observations"])
    assert ctx.query("SELECT * FROM dq_results WHERE check_name='pit_overlap'").empty


# --------------------------------------------------------------------------- qualitat
def test_dq_detects_problems(ctx, providers):
    h = providers.prices.histories["SPY"]
    h.loc[100, "high"] = h.loc[100, "low"] * 0.5  # OHLC impossible
    h.loc[200, ["close", "adj_close"]] = h.loc[199, "adj_close"] * 2  # salt de +100%
    providers.prices.histories["SPY"] = h.drop(index=[300])  # sessió que falta
    run_job(ctx, "prices", symbols=["SPY"])
    r = run_job(ctx, "dq", datasets=["prices_daily"], lookback_days=0)
    found = set(ctx.query("SELECT check_name FROM dq_results WHERE key='SPY'")["check_name"])
    assert {"ohlc_inconsistent", "extreme_return", "missing_sessions"} <= found
    assert r.status == "success"


def test_universe_and_instruments(ctx):
    run_job(ctx, "universe")
    uni = ctx.query("SELECT symbol FROM universe_current ORDER BY 1")["symbol"].tolist()
    assert uni == ["AAPL", "BRK-B"]
    r = run_job(ctx, "instruments")
    assert r.items_ok == 2
    assert len(ctx.query("SELECT * FROM instruments_current")) == 2
