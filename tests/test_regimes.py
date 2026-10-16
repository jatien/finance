"""Tests de la capa de règims: histèresi, causalitat, alineació temporal del backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance.features import FeatureConfig, MarketData, compute_market_features, percentile_ranks
from finance.ingest import run_job
from finance.regimes import RegimeConfig, classify, hysteresis
from finance.regimes import evaluation as ev
from finance.regimes.job import read_regimes
from finance.regimes.rules import crisis_state, trend_state


def _s(values, start="2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


@pytest.fixture(scope="module")
def feats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features sintètiques amb tendència, volatilitat i crèdit que canvien de règim."""
    rng = np.random.default_rng(7)
    n = 2500
    idx = pd.bdate_range("2004-01-02", periods=n)
    t = np.arange(n)
    dist = 0.08 * np.sin(t / 90) + 0.01 * rng.standard_normal(n)
    vix = 18 + 8 * np.sin(t / 60 + 1) + rng.normal(0, 1.5, n)
    term = 0.9 + 0.12 * np.sin(t / 45) + rng.normal(0, 0.02, n)
    term[:600] = np.nan  # VIX3M no existeix al principi
    credit = np.cumsum(rng.normal(0, 0.05, n))
    values = pd.DataFrame({"mkt_dist_sma200": dist, "vix": vix, "vix_term": term,
                           "baa_spread_chg_3m": pd.Series(credit).diff(63).to_numpy()}, index=idx)
    values.iloc[:200, 0] = np.nan  # SMA200 en escalfament
    pct = percentile_ranks(values, FeatureConfig())
    return values, pct


# ---------------------------------------------------------------------------------------
# Histèresi
# ---------------------------------------------------------------------------------------

def test_hysteresis_band_prevents_whipsaw():
    # Distància a la SMA200 oscil·lant ±1,5 % dins la banda de ±2 %: cap canvi d'estat
    d = _s(0.015 * np.sin(np.arange(300) / 3))
    st = trend_state(pd.DataFrame({"mkt_dist_sma200": d}), RegimeConfig())
    assert st.nunique() == 1
    # Sense banda ni confirmació, la mateixa sèrie canvia desenes de vegades
    raw = trend_state(pd.DataFrame({"mkt_dist_sma200": d}), RegimeConfig(trend_band=0.0, confirm_days=1))
    assert (raw.diff().abs() > 0).sum() > 20


def test_hysteresis_confirmation_days():
    enter = _s([0, 1, 1, 0, 1, 1, 1, 0, 0, 0]).astype(bool)
    exit_ = ~enter
    valid = pd.Series(True, index=enter.index)
    st = hysteresis(enter, exit_, valid, pd.Series(False, index=enter.index), 3, 2)
    # entra a la 3a sessió seguida (índex 6) i surt a la 2a sessió seguida de sortida (índex 8)
    assert st.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]


def test_hysteresis_holds_state_without_data():
    enter = _s([1, 1, 0, 0, 0]).astype(bool)
    valid = _s([1, 1, 0, 0, 1]).astype(bool)
    st = hysteresis(enter, ~enter, valid, pd.Series(False, index=enter.index), 1, 1)
    assert st.tolist() == [0, 1, 1, 1, 0]


def test_crisis_needs_vix_term_structure_and_credit():
    n = 30
    vals = pd.DataFrame({"vix_term": [np.nan] * 10 + [1.1] * 5 + [0.9] * 15,
                         "vix": 40.0}, index=pd.bdate_range("2020-01-01", periods=n))
    pct = pd.DataFrame({"vix": 0.99, "baa_spread_chg_3m": 0.95}, index=vals.index)
    cr = crisis_state(vals, pct, RegimeConfig())
    assert (cr.iloc[:10] == 0).all()  # sense VIX3M no hi ha crisi (substitut desactivat)
    assert (cr.iloc[10:15] == 1).all()  # entrada immediata
    assert (cr.iloc[15:19] == 1).all() and cr.iloc[19] == 0  # sortida després de 5 sessions
    # Sense deteriorament del crèdit, la corba invertida sola no és crisi
    calm_credit = pct.assign(baa_spread_chg_3m=0.5)
    assert (crisis_state(vals, calm_credit, RegimeConfig()) == 0).all()
    # El substitut antic sí que declara crisi abans del VIX3M
    assert crisis_state(vals, pct, RegimeConfig(crisis_fallback=True)).iloc[1] == 1


# ---------------------------------------------------------------------------------------
# Classificador
# ---------------------------------------------------------------------------------------

def test_classify_is_truncation_invariant(feats):
    values, pct = feats
    full = classify(values, pct)
    for cut in (350, 700, 1500, 2499):
        t = values.index[cut]
        part = classify(values.loc[:t], pct.loc[:t])
        pd.testing.assert_frame_equal(part, full.loc[:t])


def test_classify_outputs(feats):
    values, pct = feats
    cfg = RegimeConfig()
    reg = classify(values, pct, cfg)
    assert reg["regime"].iloc[:200].isna().all()  # sense tendència no hi ha règim
    defined = reg.dropna(subset=["regime"])
    assert set(defined["regime"]) <= set(cfg.exposure)
    assert len(set(defined["regime"])) >= 4
    assert (defined["equity_exposure"] == defined["regime"].map(cfg.exposure)).all()
    # Acció coherent amb el canvi d'exposició
    d = defined["equity_exposure"].diff()
    assert (defined.loc[d > 0, "action"] == "add").all()
    assert (defined.loc[d < 0, "action"] == "reduce").all()
    # Comptador de sessions: reinicia a cada canvi de règim
    starts = defined["regime"] != defined["regime"].shift()
    assert (defined.loc[starts, "days_in_regime"] == 1).all()
    # Ablació: sense eix de volatilitat només hi ha calma (o crisi)
    nv = classify(values, pct, cfg.variant(use_vol_axis=False)).dropna(subset=["regime"])
    assert not nv["regime"].str.endswith("volatile").any()


# ---------------------------------------------------------------------------------------
# Avaluació
# ---------------------------------------------------------------------------------------

def test_overlay_applies_exposure_next_day():
    """Una exposició decidida al tancament de t no pot capturar el retorn de t."""
    ret = _s([0.0, 0.0, 0.10, 0.0, -0.10, 0.0])
    exp = _s([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])  # s'entra el dia del +10 % (massa tard)
    o = ev.overlay_returns(exp, ret, _s([0.0] * 6), cost_bps=0)
    assert o["ret"].iloc[2] == 0.0  # el +10 % del dia 2 no es captura
    assert o["ret"].iloc[4] == pytest.approx(-0.10)  # encara invertit el dia de la caiguda
    o2 = ev.overlay_returns(exp, ret, _s([0.0] * 6), cost_bps=10)
    assert o2["turnover"].sum() == pytest.approx(2.0)
    assert o2["ret"].sum() == pytest.approx(o["ret"].sum() - 2 * 0.001)


def test_regime_stats_use_next_day_returns():
    px = _s([100, 100, 110, 110, 99])
    regime = pd.Series(["bear_calm", "bull_calm", "bear_calm", "crisis", "bear_calm"], index=px.index)
    st = ev.regime_stats(regime, px, horizons=(1,)).set_index("regime")
    # El règim del dia t s'associa al retorn t → t+1 (no al del mateix dia)
    assert st.loc["bull_calm", "fwd1_mitjana_%"] == pytest.approx(10.0)
    assert st.loc["crisis", "fwd1_mitjana_%"] == pytest.approx(-10.0)


def test_episodes_and_switches():
    reg = pd.Series(["a", "a", "b", "b", "b", "a"], index=pd.bdate_range("2020-01-01", periods=6))
    ep = ev.episodes(reg)
    assert ep["sessions"].tolist() == [2, 3, 1]
    assert ev.max_drawdown(_s([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_sensitivity_grid_shape(feats):
    values, pct = feats
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.0003, 0.01, len(values)), index=values.index)
    s = ev.sensitivity(values, pct, ret, ret * 0, RegimeConfig(),
                       grid={"trend_band": [0.0, 0.02], "confirm_days": [1, 3]})
    assert len(s) == 4 and s["calmar"].notna().all()
    # Més confirmació → menys canvis de règim
    b0 = s[s["trend_band"] == 0.0].set_index("confirm_days")["canvis_any"]
    assert b0.loc[3] < b0.loc[1]


# ---------------------------------------------------------------------------------------
# Integració
# ---------------------------------------------------------------------------------------

def test_regimes_job_end_to_end(ctx):
    from conftest import make_history
    ctx.providers.prices.histories["^VIX"] = make_history("^VIX", seed=9)
    run_job(ctx, "prices", symbols=["SPY", "XLK", "^VIX"])
    run_job(ctx, "features")
    r = run_job(ctx, "regimes")
    assert r.status == "success", r.errors
    reg = read_regimes(ctx)
    assert not reg.empty and reg["regime"].notna().all()
    assert reg["equity_exposure"].between(0, 1).all()
    n1 = len(reg)
    run_job(ctx, "regimes")
    assert len(read_regimes(ctx)) == n1


def test_regimes_from_synthetic_market_detect_stress():
    """Amb un mercat sintètic amb un crac, el règim ha de reduir l'exposició durant la caiguda."""
    rng = np.random.default_rng(3)
    n = 1500
    idx = pd.bdate_range("2008-01-02", periods=n)
    vol = np.full(n, 0.008)
    vol[1000:1100] = 0.03
    drift = np.where((np.arange(n) >= 1000) & (np.arange(n) < 1100), -0.004, 0.0005)
    r = drift + vol * rng.standard_normal(n)
    spy = 100 * np.exp(np.cumsum(r))
    rv = pd.Series(r).rolling(21, min_periods=1).std().to_numpy() * np.sqrt(252)
    prices = pd.DataFrame({"SPY": spy, "^VIX": 100 * rv + 4, "^VIX3M": 100 * pd.Series(rv).rolling(
        63, min_periods=1).mean().to_numpy() + 5}, index=idx)
    data = MarketData(prices=prices, macro=pd.DataFrame(
        {"baa_spread": 2 + np.r_[np.zeros(1000), np.linspace(0, 2, 100), np.full(n - 1100, 2)]}, index=idx))
    f = compute_market_features(data)
    reg = classify(f, percentile_ranks(f))
    before = reg["equity_exposure"].iloc[700:950].mean()
    during = reg["equity_exposure"].iloc[1040:1100].mean()
    # (en calma, el percentil d'un VIX gairebé constant és soroll: només es compara)
    assert during < 0.3 and before > during + 0.3
