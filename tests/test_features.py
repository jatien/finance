"""Tests de la capa de features: sense look-ahead, point-in-time i integració amb el lake."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from finance.features import (
    FeatureConfig,
    MarketData,
    compute_market_features,
    compute_sector_panel,
    percentile_ranks,
)
from finance.features import transforms as tf
from finance.features.inputs import load_macro, load_sp500_breadth, macro_pit_daily
from finance.features.job import read_market_features, read_sector_features
from finance.ingest import run_job
from finance.storage.schemas import MACRO_OBSERVATIONS, UNIVERSE_INTERVALS

SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
INDUSTRIES = ["SMH", "KRE", "XBI"]
# Sectors que comencen a cotitzar a mitja història (com XLRE el 2015 i XLC el 2018)
LATE_START = {"XLRE": 800, "XLC": 1100, "XBI": 500}


@pytest.fixture(scope="module")
def synth() -> MarketData:
    """Mercat sintètic amb un factor comú, un episodi d'estrès i sectors que neixen tard."""
    rng = np.random.default_rng(42)
    n = 1600
    idx = pd.bdate_range("2010-01-04", periods=n)
    vol = np.full(n, 0.01)
    vol[900:1000] = 0.035  # episodi d'estrès
    drift = np.where((np.arange(n) >= 900) & (np.arange(n) < 1000), -0.003, 0.0004)
    mkt = drift + vol * rng.standard_normal(n)
    cols = {}
    names = ["SPY", "RSP", "TLT", "IEF", "HYG", *SECTORS, *INDUSTRIES]
    for i, s in enumerate(names):
        beta = 1.0 if s not in ("TLT", "IEF") else -0.3
        idio = 0.006 * rng.standard_normal(n)
        r = beta * mkt + idio + 0.0001 * (i % 5)
        p = 50 * np.exp(np.cumsum(r))
        if s in LATE_START:
            p[: LATE_START[s]] = np.nan
        cols[s] = p
    rv = pd.Series(mkt, index=idx).rolling(21, min_periods=1).std().to_numpy() * np.sqrt(252)
    cols["^VIX"] = 100 * rv + 3 + rng.normal(0, 0.5, n)
    cols["^VIX3M"] = 100 * pd.Series(rv).rolling(63, min_periods=1).mean().to_numpy() + 4
    prices = pd.DataFrame(cols, index=idx)
    macro = pd.DataFrame({
        "baa_spread": 2 + np.cumsum(rng.normal(0, 0.02, n)),
        "curve_10y3m": 1 + np.cumsum(rng.normal(0, 0.02, n)),
    }, index=idx)
    breadth = pd.DataFrame({"breadth_200": rng.uniform(0, 1, n), "coverage": 0.9}, index=idx)
    return MarketData(prices=prices, macro=macro, sp500_breadth=breadth,
                      groups={"sectors": SECTORS, "industries": INDUSTRIES})


CUTS = [450, 899, 950, 1250, 1599]


# ---------------------------------------------------------------------------------------
# Sense look-ahead: calcular amb dades fins a t ha de donar el mateix que amb tota la
# història, per a totes les dates <= t. És el test més important de la capa.
# ---------------------------------------------------------------------------------------

def test_market_features_truncation_invariant(synth):
    cfg = FeatureConfig()
    full = compute_market_features(synth, cfg)
    full_pct = percentile_ranks(full, cfg)
    assert full.shape[1] >= 18
    for cut in CUTS:
        t = synth.index[cut]
        part = compute_market_features(synth.truncate(t), cfg)
        pd.testing.assert_frame_equal(part, full.loc[:t], check_exact=False, rtol=1e-10, atol=1e-12)
        pd.testing.assert_frame_equal(percentile_ranks(part, cfg), full_pct.loc[:t],
                                      check_exact=False, rtol=1e-10)


def test_sector_panel_truncation_invariant(synth):
    full = compute_sector_panel(synth).set_index(["date", "symbol", "feature"]).sort_index()
    for cut in CUTS:
        t = synth.index[cut]
        part = compute_sector_panel(synth.truncate(t)).set_index(["date", "symbol", "feature"]).sort_index()
        expected = full[full.index.get_level_values("date") <= t]
        pd.testing.assert_frame_equal(part, expected, check_exact=False, rtol=1e-10, atol=1e-12)


def test_truncation_test_catches_leakage(synth):
    """Control: una transformació amb look-ahead (finestra centrada) ha de fallar el test."""
    s = synth.prices["SPY"]

    def leaky(x: pd.Series) -> pd.Series:
        return x.rolling(21, center=True).mean()

    t = synth.index[1000]
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(leaky(s.loc[:t]), leaky(s).loc[:t])


def test_features_scale_invariant(synth):
    """adj_close es re-escala cap enrere amb cada dividend: cap feature pot dependre del nivell."""
    scaled = synth.truncate(synth.index[-1])
    vol_idx = ["^VIX", "^VIX3M"]
    other = [c for c in scaled.prices if c not in vol_idx]
    scaled.prices[other] = scaled.prices[other] * 3.7
    a = compute_market_features(synth)
    b = compute_market_features(scaled)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-8, atol=1e-10)
    pa_ = compute_sector_panel(synth)
    pb_ = compute_sector_panel(scaled)
    pd.testing.assert_frame_equal(pa_, pb_, check_exact=False, rtol=1e-8, atol=1e-10)


def test_late_listed_sector_does_not_leak(synth):
    panel = compute_sector_panel(synth)
    xlc = panel[panel["symbol"] == "XLC"]
    assert xlc["date"].min() > synth.index[LATE_START["XLC"]]
    # Abans que XLC existeixi, el rang transversal es calcula sobre 10 sectors
    t = synth.index[LATE_START["XLC"] - 1]
    day = panel[(panel["date"] == t) & (panel["group_name"] == "sectors") & (panel["feature"] == "dist_sma200")]
    assert set(day["symbol"]) == set(SECTORS) - {"XLC"}
    assert day["xs_rank"].max() == pytest.approx(1.0)


def test_feature_values_make_sense(synth):
    f = compute_market_features(synth)
    calm, stress = synth.index[850], synth.index[990]
    assert f.loc[stress, "mkt_rvol_1m"] > 2 * f.loc[calm, "mkt_rvol_1m"]
    assert f.loc[stress, "sector_avg_corr_3m"] > f.loc[calm, "sector_avg_corr_3m"]
    assert f.loc[stress, "mkt_drawdown_1y"] < -0.15
    assert f.loc[stress, "mkt_dist_sma200"] < 0
    assert f["mkt_drawdown_1y"].max() <= 0
    assert f[["sector_breadth_200", "sector_breadth_50"]].stack().dropna().between(0, 1).all()
    assert f["sector_absorption_1y"].dropna().between(0, 1).all()
    assert f["stock_bond_corr_3m"].dropna().between(-1, 1).all()


# ---------------------------------------------------------------------------------------
# Transformacions
# ---------------------------------------------------------------------------------------

def test_correlation_structure_known_values():
    rng = np.random.default_rng(0)
    n = 300
    common = rng.standard_normal(n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    # Correlació teòrica entre parells: 1 / (1 + 1) = 0.5
    px = pd.DataFrame({f"S{i}": 100 * np.exp(np.cumsum(0.01 * (common + rng.standard_normal(n))))
                       for i in range(6)}, index=idx)
    cs = tf.correlation_structure(px, 250)
    assert cs["avg_corr"].iloc[-1] == pytest.approx(0.5, abs=0.08)
    # Un factor amb rho=0.5 i N=6: lambda_1 = 1 + 5·0.5 = 3.5; k=ceil(1.2)=2 vectors
    assert 3.5 / 6 < cs["absorption"].iloc[-1] < 0.85


def test_momentum_skip_and_drawdown():
    px = pd.Series(np.arange(1, 301, dtype=float), index=pd.bdate_range("2020-01-01", periods=300))
    m = tf.momentum(px, 252, 21)
    assert m.iloc[-1] == pytest.approx(px.iloc[-22] / px.iloc[-253] - 1)
    assert tf.rolling_drawdown(px, 20).dropna().eq(0).all()


# ---------------------------------------------------------------------------------------
# Macro point-in-time
# ---------------------------------------------------------------------------------------

def _obs(rows, mode):
    return pd.DataFrame([{"date": d, "value": v, "realtime_start": rs, "vintage_mode": mode}
                         for d, v, rs in rows])


def test_macro_first_release_never_uses_revision():
    idx = pd.bdate_range("2020-03-02", "2020-06-30")
    obs = _obs([
        (date(2020, 3, 1), -701.0, date(2020, 4, 3)),  # primera publicació
        (date(2020, 3, 1), -1373.0, date(2020, 5, 8)),  # revisió (no s'ha de veure mai)
        (date(2020, 4, 1), -20500.0, date(2020, 5, 8)),
    ], "alfred")
    s = macro_pit_daily(obs, idx)
    assert s.loc[: "2020-04-02"].isna().all()
    assert s.loc["2020-04-03"] == -701.0
    assert s.loc["2020-05-07"] == -701.0
    assert s.loc["2020-05-08"] == -20500.0
    assert (s != -1373.0).all()


def test_macro_observation_date_published_next_session():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    obs = _obs([(d.date(), float(i), d.date()) for i, d in enumerate(idx)], "observation_date")
    s = macro_pit_daily(obs, idx, lag_sessions=1)
    assert np.isnan(s.iloc[0])
    assert (s.iloc[1:].to_numpy() == np.arange(len(idx) - 1)).all()
    s0 = macro_pit_daily(obs, idx, lag_sessions=0)
    assert (s0.to_numpy() == np.arange(len(idx))).all()


def test_revisable_series_without_vintages_is_excluded(ctx):
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    ts = pd.Timestamp("2026-10-15", tz="UTC").to_pydatetime()
    rows = pd.concat([
        _obs([(d.date(), 2.0, d.date()) for d in idx], "observation_date").assign(series_id="BAA10Y"),
        _obs([(d.date(), -0.5, d.date()) for d in idx[::5]], "latest_only").assign(series_id="NFCI"),
    ]).assign(realtime_end=date(9999, 12, 31))
    ctx.lake.write(MACRO_OBSERVATIONS, rows, source="test", run_id="t", ingested_at=ts)
    notes: list[str] = []
    m = load_macro(ctx, FeatureConfig(), idx, notes)
    assert "baa_spread" in m and "nfci" not in m
    assert any("NFCI" in n and "exclosa" in n for n in notes)


# ---------------------------------------------------------------------------------------
# Integració amb el lake
# ---------------------------------------------------------------------------------------

def test_sp500_breadth_uses_point_in_time_membership(ctx):
    run_job(ctx, "universe")
    run_job(ctx, "prices")
    ts = pd.Timestamp("2026-10-15", tz="UTC").to_pydatetime()
    exit_day = date(2025, 6, 2)
    iv = pd.DataFrame([
        {"universe": "sp500", "symbol": "AAPL", "start_date": date(2000, 1, 3), "end_date": None},
        {"universe": "sp500", "symbol": "BRK.B", "start_date": date(2010, 2, 16), "end_date": exit_day},
    ]).assign(start_censored=False, start_window_days=0, end_window_days=0,
              start_source="test", end_source="test")
    ctx.lake.write(UNIVERSE_INTERVALS, iv, source="test", run_id="t", ingested_at=ts)
    px = ctx.query("SELECT date, symbol, adj_close FROM prices_daily").pivot(
        index="date", columns="symbol", values="adj_close")
    px.index = pd.to_datetime(px.index)
    notes: list[str] = []
    b = load_sp500_breadth(ctx, FeatureConfig(), px.index, notes)
    assert b["breadth_200"].iloc[:199].isna().all()  # cap membre amb SMA200 completa
    after = px.index[px.index >= pd.Timestamp(exit_day)]
    aapl_above = (px["AAPL"] > px["AAPL"].rolling(200).mean()).astype(float)
    pd.testing.assert_series_equal(b.loc[after, "breadth_200"], aapl_above.loc[after],
                                   check_names=False)
    assert (b.loc[after, "coverage"] == 1.0).all()


def test_features_job_end_to_end(ctx):
    run_job(ctx, "prices")
    r1 = run_job(ctx, "features")
    assert r1.status == "success", r1.errors
    f = read_market_features(ctx)
    assert {"mkt_dist_sma200", "mkt_rvol_1m", "mkt_drawdown_1y"} <= set(f.columns)
    p = read_market_features(ctx, column="pct_rank")
    assert p.stack().dropna().between(0, 1).all()
    n1 = ctx.query("SELECT count(*) AS n FROM features_daily")["n"].iloc[0]
    # Re-executar no duplica files (la vista només mostra l'última foto del conjunt)
    r2 = run_job(ctx, "features")
    assert r2.status == "success"
    assert ctx.query("SELECT count(*) AS n FROM features_daily")["n"].iloc[0] == n1
    runs = ctx.query("SELECT params FROM ingest_runs WHERE job = 'features'")
    assert runs["params"].str.contains("feature_config").all()
    # El grup de sectors no existeix a l'univers de test: el panell queda buit però no falla
    assert read_sector_features(ctx, "mom_12_1").empty
