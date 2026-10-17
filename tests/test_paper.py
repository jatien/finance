"""Tests del paper trading i de l'informe diari: equivalència, congelació, salut de dades."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_portfolio import UNIVERSE_PORTFOLIO, inp  # noqa: F401 — fixture compartida

from finance.paper.book import advance, replay, start, state_from_rows
from finance.portfolio import CASH, build_targets
from finance.portfolio.backtest import lagged, simulate_cfg
from finance.portfolio.config import ROT_V1B, ROT_V2

# ---------------------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [ROT_V1B, ROT_V2], ids=lambda c: c.name)
def test_incremental_equals_replay(inp, cfg):  # noqa: F811
    """Avançar sessió a sessió, amb només les dades de cada dia i l'estat reconstruït de les
    files guardades, dona exactament el mateix que recalcular-ho tot de cop."""
    idx = inp.index
    inception = idx[1020]
    full = replay(inp, build_targets(inp, cfg), cfg, inception, idx[1080])
    part = inp.truncate(inception)
    s = start(part, build_targets(part, cfg), cfg, inception)
    state = state_from_rows(s.nav, s.book)
    navs = [s.nav]
    for d in idx[1021:1081]:
        ti = inp.truncate(d)
        u = advance(state, ti, build_targets(ti, cfg), cfg)
        assert len(u.nav) == 1 and not u.nav["recorded_late"].iloc[0]
        state = state_from_rows(u.nav, u.book)
        navs.append(u.nav)
    inc = pd.concat(navs, ignore_index=True)
    pd.testing.assert_series_equal(inc["nav"], full.nav["nav"], check_names=False, rtol=0, atol=1e-12)
    assert list(inc["traded"]) == list(full.nav["traded"])


@pytest.mark.parametrize("cfg", [ROT_V1B, ROT_V2], ids=lambda c: c.name)
def test_paper_matches_backtest_with_one_session_lag(inp, cfg):  # noqa: F811
    """Des de la segona sessió després de l'inici, el paper trading és el backtest amb
    execució a t+1 (`lagged`), mateix motor: mateixos retorns i mateixes operacions."""
    idx = inp.index
    inception = idx[1000]
    T = build_targets(inp, cfg)
    paper = replay(inp, T, cfg, inception).nav.set_index("date")
    # Backtest que comença a l'inici: objectiu vigent forçat com a primera revisió
    W = T.weights.copy()
    W.loc[W.index < inception] = np.nan
    rev = T.reviews.copy()
    rev.loc[rev.index < inception] = False
    rev.loc[inception] = True
    bt = simulate_cfg(lagged(type(T)(W, rev, T.log), 1), inp.asset_returns(), cfg)
    later = paper.index[2:]
    np.testing.assert_allclose(paper.loc[later, "ret_net"], bt.returns.loc[later], atol=1e-12)
    assert list(paper.loc[later, "traded"]) == list(bt.trades.loc[later])


def test_inception_buys_target_next_session(inp):  # noqa: F811
    cfg = ROT_V2  # només negocia a les revisions: l'inici ha de comprar igualment
    T = build_targets(inp, cfg)
    d0 = inp.index[1003]
    assert not T.reviews.loc[d0]
    s = start(inp, T, cfg, d0)
    assert s.nav["nav"].iloc[0] == 1.0 and not s.nav["traded"].iloc[0]
    assert s.book.set_index("symbol").loc[CASH, "weight"] == 1.0
    u = advance(s.state, inp, T, cfg, inp.index[1004])
    assert u.nav["traded"].iloc[0] and u.nav["reason"].iloc[0].startswith("inici")
    held = u.book.set_index("symbol")["weight"]
    target = T.weights.loc[d0]
    assert held.reindex(target.index).fillna(0).to_numpy() == pytest.approx(target.to_numpy())
    # Sense revisió ni deriva comprovada cada dia, rot_v2 no torna a negociar fins a la revisió
    u2 = advance(u.state, inp, T, cfg, inp.index[1010])
    assert not u2.nav["traded"].any() or T.reviews.loc[inp.index[1004]:inp.index[1009]].any()


def test_catch_up_marks_late_rows(inp):  # noqa: F811
    T = build_targets(inp, ROT_V1B)
    s = start(inp, T, ROT_V1B, inp.index[1000])
    u = advance(s.state, inp, T, ROT_V1B, inp.index[1005])
    assert list(u.nav["recorded_late"]) == [True, True, True, True, False]


def test_book_actions_and_weights(inp):  # noqa: F811
    T = build_targets(inp, ROT_V1B)
    u = replay(inp, T, ROT_V1B, inp.index[1000], inp.index[1100])
    for _, g in u.book.groupby("date"):
        assert g["weight"].sum() == pytest.approx(1.0)
        assert g["target_weight"].sum() == pytest.approx(1.0)
        assert np.allclose(g["trade"], g["weight"] - g["weight_pre"])
    traded_days = set(u.nav.loc[u.nav["traded"], "date"])
    acted = u.book[u.book["action"] != "hold"]
    assert set(acted["date"]) <= traded_days
    assert set(acted["action"]) <= {"enter", "add", "reduce", "exit"}


# ---------------------------------------------------------------------------------------
# Integració: lake congelat i informe diari
# ---------------------------------------------------------------------------------------

def _ctx(tmp_path, providers, clock):
    from conftest import make_history

    from finance.jobs import Context
    from finance.settings import Settings
    from finance.storage import Lake
    for i, s in enumerate(["IEF", "GLD", "^VIX", "XLF", "XLV", "XLP"]):
        providers.prices.histories[s] = make_history(s, start="2023-01-03", seed=20 + i)
    for s in ("SPY", "XLK"):
        providers.prices.histories[s] = make_history(s, start="2023-01-03", seed=hash(s) % 100)
    uni = tmp_path / "universe.yaml"
    uni.write_text(UNIVERSE_PORTFOLIO)
    settings = Settings(data_dir=tmp_path / "data", universe_file=uni, yahoo_max_workers=2, _env_file=None)
    settings.ensure_dirs()
    return Context(settings=settings, lake=Lake(settings.lake_dir), providers=providers, clock=clock)


def _daily(ctx):
    from finance.ingest import run_job
    for job in ("prices", "features", "regimes", "portfolio", "paper"):
        r = run_job(ctx, job)
        assert r.status == "success", (job, r.errors, r.message)
    return r


def test_paper_job_is_frozen_and_report_renders(tmp_path, providers, clock):
    from finance.ingest import run_job
    from finance.paper.job import read_paper_book, read_paper_nav
    from finance.paper.report import build_daily, write_daily

    ctx = _ctx(tmp_path, providers, clock)
    _daily(ctx)  # dijous 15-oct: inici
    nav0 = read_paper_nav(ctx)
    assert set(nav0["strategy_set"]) == {"rot_v1b", "rot_v2"}
    assert (nav0["nav"] == 1.0).all() and nav0["date"].nunique() == 1
    # Tornar-lo a executar el mateix dia no escriu res
    r = run_job(ctx, "paper")
    assert r.rows_written == 0 and "al dia" in r.message
    # Divendres: una sessió nova; s'executa la compra inicial
    clock.set("2026-10-16 21:00")
    _daily(ctx)
    nav1 = read_paper_nav(ctx)
    assert nav1.groupby("strategy_set").size().tolist() == [2, 2]
    assert nav1.loc[nav1["date"] == nav1["date"].max(), "traded"].all()
    clock.set("2026-10-19 21:00")  # dilluns
    _daily(ctx)
    nav2 = read_paper_nav(ctx)
    # El proveïdor corregeix el tancament de divendres de tots els actius (+5 %): el retorn de
    # dilluns ja registrat canviaria. El registre no es toca; l'informe ho detecta.
    for h in providers.prices.histories.values():
        m = pd.to_datetime(h["date"]) == pd.Timestamp("2026-10-16")
        h.loc[m, ["close", "adj_close"]] *= 1.05
    run_job(ctx, "prices", full=True)
    r = run_job(ctx, "paper")
    assert r.rows_written == 0
    nav3 = read_paper_nav(ctx)
    pd.testing.assert_frame_equal(nav3[nav2.columns].reset_index(drop=True), nav2.reset_index(drop=True))
    old = nav3[nav3["date"] <= "2026-10-16"].reset_index(drop=True)
    pd.testing.assert_frame_equal(old[nav1.columns], nav1.reset_index(drop=True))
    assert not read_paper_book(ctx, "rot_v1b").empty

    rep = build_daily(ctx, capital=20_000)
    assert rep.tradable and rep.sessions_behind == 0
    ref = rep.strategies[0]
    assert ref.cfg.name == "rot_v1b" and ref.perf["sessions"] == 2
    assert ref.perf["value"] == pytest.approx(20_000 * ref.perf["nav"])
    assert any("ha corregit preus" in t for _, t in rep.issues)  # correcció detectada
    paths = write_daily(rep, tmp_path / "out")
    html = paths["html"].read_text(encoding="utf-8")
    for section in ("Què fer a la sessió següent", "Règim de mercat", "Paper trading",
                    "Registre de decisions", "Simulació recent"):
        assert section in html
    assert "rot_v1b" in paths["latest_md"].read_text(encoding="utf-8")


def test_report_blocks_trading_on_stale_data(tmp_path, providers, clock):
    from finance.paper.report import build_daily, render_text
    ctx = _ctx(tmp_path, providers, clock)
    _daily(ctx)
    # Han passat dues sessions i no hi ha preus nous
    clock.set("2026-10-19 21:00")
    rep = build_daily(ctx)
    assert not rep.tradable and rep.sessions_behind == 2
    assert render_text(rep).splitlines()[2].startswith("NO OPERIS")


def test_daily_pipeline_includes_paper_and_report():
    import inspect

    from finance.ingest import daily_pipeline
    src = inspect.getsource(daily_pipeline)
    assert src.index('"portfolio"') < src.index('"paper"') < src.index('"report"')
