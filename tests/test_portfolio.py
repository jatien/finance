"""Tests de la rotació sectorial i la cartera: causalitat, motor, cobertura, registre."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance.features import FeatureConfig
from finance.features.sectors import compute_group_panel
from finance.portfolio import CASH, PortfolioConfig, PortfolioInputs, Targets, build_targets
from finance.portfolio import evaluation as ev
from finance.portfolio.backtest import simulate
from finance.portfolio.config import ACTIVE, ROT_V1, ROT_V2, SleeveRule
from finance.portfolio.construct import (
    PANEL_FEATURES,
    absolute_momentum_ok,
    equity_weights,
    fixed_targets,
    hedge_weights,
    review_dates,
    sector_scores,
    select_sectors,
)
from finance.portfolio.job import decision_log
from finance.regimes.config import RegimeConfig

SECTORS = ["XLK", "XLF", "XLE", "XLP", "XLU", "XLV"]
N = 1300
GLD_START = 700
XLE_START = 400  # sector que neix tard


@pytest.fixture(scope="module")
def inp() -> PortfolioInputs:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2010-01-04", periods=N)
    mkt = 0.0004 + 0.01 * rng.standard_normal(N)
    betas = {"XLK": 1.3, "XLF": 1.2, "XLE": 1.0, "XLP": 0.5, "XLU": 0.4, "XLV": 0.7}
    drift = {"XLK": 0.0006, "XLF": 0.0, "XLE": 0.0002, "XLP": 0.0001, "XLU": 0.0, "XLV": 0.0003}
    cols = {"SPY": 100 * np.exp(np.cumsum(mkt))}
    for s in SECTORS:
        r = betas[s] * mkt + drift[s] + 0.006 * rng.standard_normal(N)
        p = 50 * np.exp(np.cumsum(r))
        if s == "XLE":
            p[:XLE_START] = np.nan
        cols[s] = p
    cols["IEF"] = 100 * np.exp(np.cumsum(0.0001 - 0.2 * mkt + 0.003 * rng.standard_normal(N)))
    g = 100 * np.exp(np.cumsum(0.0002 + 0.009 * rng.standard_normal(N)))
    g[:GLD_START] = np.nan
    cols["GLD"] = g
    prices = pd.DataFrame(cols, index=idx)
    panel = compute_group_panel(prices[SECTORS], prices["SPY"], FeatureConfig())
    # Règims: bull_calm, després un tram baixista i una crisi, i tornada a bull
    reg = np.array(["bull_calm"] * N, dtype=object)
    reg[600:700] = "bear_volatile"
    reg[700:760] = "crisis"
    reg[900:950] = "bull_volatile"
    reg[:250] = None
    exposure = RegimeConfig().exposure
    regime = pd.DataFrame({"regime": reg}, index=idx)
    regime["trend"] = np.where(pd.Series(reg).isin(["bear_calm", "bear_volatile"]), "bear", "bull")
    regime["equity_exposure"] = regime["regime"].map(exposure)
    regime.loc[regime["regime"].isna(), "trend"] = None
    sb = pd.Series(np.where(np.arange(N) < 1000, -0.3, 0.3), index=idx, dtype=float)
    sb.iloc[:300] = np.nan
    fast =pd.Series(np.where(np.arange(N) < 980, -0.2, 0.4), index=idx, dtype=float)
    fast.iloc[:280] = np.nan
    return PortfolioInputs(prices=prices, cash_ret=pd.Series(0.0001, index=idx), regime=regime,
                           panel={k: panel[k] for k in PANEL_FEATURES},
                           stock_bond_corr=sb, sectors=SECTORS, stock_bond_corr_fast={21: fast})


# ---------------------------------------------------------------------------------------
# Causalitat
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [ROT_V1, ROT_V2], ids=lambda c: c.name)
def test_targets_truncation_invariant(inp, cfg):
    full = build_targets(inp, cfg)
    for cut in (420, 640, 701, 985, 1001, N - 1):
        t = inp.index[cut]
        part = build_targets(inp.truncate(t), cfg)
        pd.testing.assert_frame_equal(part.weights, full.weights.loc[:t])
        pd.testing.assert_series_equal(part.reviews, full.reviews.loc[:t])
        pd.testing.assert_frame_equal(part.log, full.log.loc[:t])


def test_simulation_truncation_invariant(inp):
    t_full = build_targets(inp)
    rets = inp.asset_returns()
    full = simulate(t_full, rets)
    cut = inp.index[900]
    part = simulate(build_targets(inp.truncate(cut)), inp.truncate(cut).asset_returns())
    pd.testing.assert_series_equal(part.returns, full.returns.loc[:cut])


def test_target_applies_from_next_session():
    idx = pd.bdate_range("2020-01-01", periods=6)
    rets = pd.DataFrame({"A": [0, 0, 0.10, 0, -0.10, 0], CASH: 0.0}, index=idx, dtype=float)
    W = pd.DataFrame({"A": [0, 0, 1, 1, 1, 1], CASH: [1, 1, 0, 0, 0, 0]}, index=idx, dtype=float)
    rev = pd.Series([True, False, True, False, False, False], index=idx)
    s = simulate(Targets(W, rev, pd.DataFrame()), rets, cost_bps=0, band=0)
    assert s.returns.iloc[2] == 0.0  # decidit al tancament del dia del +10 %: no es captura
    assert s.returns.iloc[4] == pytest.approx(-0.10)


# ---------------------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------------------

def test_costs_exclude_cash_leg_and_band_blocks_small_changes():
    idx = pd.bdate_range("2020-01-01", periods=4)
    rets = pd.DataFrame({"A": 0.0, "B": 0.0, CASH: 0.0}, index=idx)
    W = pd.DataFrame({"A": [1.0, 0.5, 0.49, 0.0], "B": 0.0, CASH: [0.0, 0.5, 0.51, 1.0]}, index=idx)
    rev = pd.Series(True, index=idx)
    s = simulate(Targets(W, rev, pd.DataFrame()), rets, cost_bps=10, band=0.02)
    # A: 0 → 1 (1 unitat), 1 → 0,5 (0,5), canvi de 0,01 bloquejat per la banda, 0,5 → 0 (0,5)
    assert s.turnover.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.5])
    assert s.trades.tolist() == [True, True, False, True]
    assert s.returns.iloc[0] == pytest.approx(-0.001)


def test_weights_drift_between_reviews():
    idx = pd.bdate_range("2020-01-01", periods=3)
    rets = pd.DataFrame({"A": [0, 0.10, 0], "B": [0, 0, 0]}, index=idx, dtype=float)
    W = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    rev = pd.Series([True, False, False], index=idx)
    s = simulate(Targets(W, rev, pd.DataFrame()), rets, cost_bps=0)
    assert s.weights.iloc[1]["A"] == pytest.approx(0.55 / 1.05)
    assert s.returns.iloc[1] == pytest.approx(0.05)


# ---------------------------------------------------------------------------------------
# Construcció
# ---------------------------------------------------------------------------------------

def test_targets_respect_exposure_and_listing(inp):
    cfg = PortfolioConfig()
    t = build_targets(inp, cfg)
    w = t.weights.dropna()
    assert np.allclose(w.sum(axis=1), 1.0)
    e = inp.regime["equity_exposure"].reindex(w.index)
    assert np.allclose(w[SECTORS].sum(axis=1), e)
    # Mai pes en un actiu abans que cotitzi
    assert (w.loc[w.index < inp.index[GLD_START], "GLD"] == 0).all()
    assert (w.loc[w.index < inp.index[XLE_START + 252], "XLE"] == 0).all()
    # Exactament top_k sectors quan hi ha renda variable
    n_held = (w[SECTORS] > 0).sum(axis=1)
    assert set(n_held[e > 0]) == {cfg.top_k}
    # Crisi: tot en cobertura; correlació accions-bons negativa → bons
    crisis = w.loc[inp.index[710]]
    assert crisis[SECTORS].sum() == 0 and crisis["IEF"] == pytest.approx(1.0)
    # Revisió a cada canvi de règim
    assert t.reviews.loc[inp.index[600]] and t.reviews.loc[inp.index[700]]


def test_review_dates_monthly_and_on_regime_change():
    idx = pd.bdate_range("2021-01-25", "2021-03-10")
    reg = pd.Series("bull_calm", index=idx)
    reg.loc["2021-02-10":] = "bear_calm"
    rev = review_dates(idx, reg)
    assert list(rev[rev].index.strftime("%Y-%m-%d")) == ["2021-01-25", "2021-02-01", "2021-02-10", "2021-03-01"]


def test_correlation_filter_skips_near_duplicate():
    cfg = PortfolioConfig(top_k=2)
    scores = pd.Series({"A": 3.0, "B": 2.9, "C": 1.0, "D": 0.5})
    m = np.full((4, 4), 0.3)
    np.fill_diagonal(m, 1.0)
    m[0, 1] = m[1, 0] = 0.95
    corr = pd.DataFrame(m, index=list("ABCD"), columns=list("ABCD"))
    chosen, bound = select_sectors(scores, corr, cfg)
    assert chosen == ["A", "C"] and bound
    assert select_sectors(scores, corr, cfg.variant(max_pair_corr=1.01)) == (["A", "B"], False)
    # Si tot és massa correlacionat, s'omple igualment amb els millors restants
    hi = pd.DataFrame(0.99, index=list("ABCD"), columns=list("ABCD"))
    assert select_sectors(scores, hi, cfg)[0] == ["A", "B"]


def test_defensive_tilt_only_in_bear(inp):
    cfg = PortfolioConfig()
    t = inp.index[650]
    bull = sector_scores(inp.panel, t, bear=False, cfg=cfg)
    bear = sector_scores(inp.panel, t, bear=True, cfg=cfg)
    beta = inp.panel["beta_1y"].loc[t]
    low_beta = beta.idxmin()
    assert bear.rank(ascending=False)[low_beta] <= bull.rank(ascending=False)[low_beta]
    assert sector_scores(inp.panel, t, bear=True, cfg=cfg.variant(defensive_tilt=False)).equals(bull)


def test_hedge_rules():
    cfg = PortfolioConfig()
    assert hedge_weights(-0.2, {"IEF", "GLD"}, cfg) == ({"IEF": 1.0}, "bons")
    assert hedge_weights(0.3, {"IEF", "GLD"}, cfg) == ({"GLD": 0.5, CASH: 0.5}, "or+cash")
    assert hedge_weights(0.3, {"IEF"}, cfg) == ({CASH: 1.0}, "cash")
    assert hedge_weights(np.nan, {"IEF", "GLD"}, cfg) == ({CASH: 1.0}, "cash")
    assert hedge_weights(-0.2, {"IEF"}, cfg.variant(hedge=False)) == ({CASH: 1.0}, "cash")


def test_fixed_targets_and_benchmarks(inp):
    t = fixed_targets(inp.index, {"SPY": 1.0}, monthly=False, start=inp.index[300])
    assert t.reviews.sum() == 1 and t.weights.iloc[:300].isna().all().all()
    sims, start = ev.run_all(inp, PortfolioConfig())
    assert set(sims) >= {"SPY buy & hold", "60/40 SPY/IEF", ev.MAIN}
    table = ev.metrics_table(sims, inp, PortfolioConfig(), start)
    assert table["CAGR_%"].notna().all()
    assert sims["SPY buy & hold"].returns.first_valid_index() == start


# ---------------------------------------------------------------------------------------
# Registre de decisions
# ---------------------------------------------------------------------------------------

def test_decision_log_actions():
    idx = pd.bdate_range("2020-01-01", periods=4)
    W = pd.DataFrame({"A": [0.5, 0.5, 0.6, 0.0], "B": [0.5, 0.49, 0.4, 0.0], CASH: [0, 0.01, 0, 1.0]}, index=idx)
    t = Targets(W, pd.Series(True, index=idx), pd.DataFrame())
    log = decision_log(t, PortfolioConfig())
    a = log.set_index(["date", "symbol"])["action"]
    assert a[(idx[0], "A")] == "enter"
    assert a[(idx[1], "B")] == "hold"  # -1 pp, dins la banda
    assert a[(idx[1], CASH)] == "enter"
    assert a[(idx[2], "A")] == "add" and a[(idx[2], "B")] == "reduce" and a[(idx[2], CASH)] == "exit"
    assert a[(idx[3], "A")] == "exit"
    assert set(log["sleeve"]) == {"renda_variable", "cash"}


# ---------------------------------------------------------------------------------------
# Integració amb el lake
# ---------------------------------------------------------------------------------------

UNIVERSE_PORTFOLIO = """
defaults: {calendar: XNYS, history_start: "2024-01-02"}
groups:
  etfs:
    asset_class: etf
    symbols: {SPY: {}, IEF: {}, GLD: {}, ^VIX: {}}
  sectors:
    asset_class: etf_sector
    symbols: {XLK: {}, XLF: {}, XLV: {}, XLP: {}}
universes:
  prices: [etfs, sectors]
  instruments: [etfs]
options:
  underlyings: [SPY]
  max_dte_days: 60
macro:
  DGS10: {name: "10Y", frequency: D, revisable: false}
"""


def test_portfolio_job_end_to_end(tmp_path, providers, clock):
    from conftest import make_history

    from finance.ingest import run_job
    from finance.jobs import Context
    from finance.portfolio.job import read_targets
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
    ctx = Context(settings=settings, lake=Lake(settings.lake_dir), providers=providers, clock=clock)
    for job in ("prices", "features", "regimes", "portfolio"):
        r = run_job(ctx, job)
        assert r.status == "success", (job, r.errors, r.message)
    df = read_targets(ctx)
    assert not df.empty and set(df["strategy_set"]) == {"rot_v1"}
    v2 = read_targets(ctx, "rot_v2")
    assert not v2.empty and (v2.loc[v2["symbol"] == "SPY", "sleeve"] == "nucli").all()
    last = df[df["date"] == df["date"].max()]
    assert last["target_weight"].sum() == pytest.approx(1.0)
    assert set(df["action"]) <= {"enter", "add", "reduce", "exit", "hold"}


# ---------------------------------------------------------------------------------------
# rot_v2: construcció condicionada al règim
# ---------------------------------------------------------------------------------------

def test_rule_for_playbook_and_v1_fallback():
    assert ROT_V1.rule_for("bear_calm", "bear").score == "risk_adj_low_beta"
    assert ROT_V1.rule_for("bull_calm", "bull") == SleeveRule()
    assert ROT_V1.variant(defensive_tilt=False).rule_for("bear_calm", "bear").score == "risk_adj"
    bull = ROT_V2.rule_for("bull_volatile", "bull")
    assert (bull.core_share, bull.score, bull.abs_mom_filter) == (0.70, "trend", True)
    assert ROT_V2.rule_for("bear_volatile", "bear").core_share == 0.50
    # Un règim concret té prioritat sobre la tendència
    special = SleeveRule(core_share=1.0)
    cfg = ROT_V2.variant(playbook={**dict(ROT_V2.playbook), "bull_volatile": special})
    assert cfg.rule_for("bull_volatile", "bull") is special
    assert cfg.rule_for("bull_calm", "bull") == bull
    with pytest.raises(KeyError):
        ROT_V2.variant(playbook=(("bull", bull),)).rule_for("bear_calm", "bear")
    with pytest.raises(ValueError):
        SleeveRule(score="xyz")
    assert ROT_V2.to_params()["playbook"]["bull"]["core_share"] == 0.70
    assert [c.name for c in ACTIVE] == ["rot_v1", "rot_v2"]


def test_equity_weights_core_and_empty_slots():
    vol = pd.Series({"A": 0.1, "B": 0.2, "C": 0.2})
    rule = SleeveRule(core_share=0.7)
    w = equity_weights(["A", "B", "C"], vol, rule, ROT_V2)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["SPY"] == pytest.approx(0.7)
    assert w["A"] == pytest.approx(0.3 * 0.5)  # vol inversa: 10 / (10 + 5 + 5)
    # Una plaça buida (filtre absolut): 1/3 del satèl·lit passa al nucli
    w2 = equity_weights(["A", "B"], vol, rule, ROT_V2)
    assert w2["SPY"] == pytest.approx(0.8) and sum(w2.values()) == pytest.approx(1.0)
    assert equity_weights([], vol, rule, ROT_V2) == {"SPY": 1.0}
    assert "SPY" not in equity_weights(["A", "B", "C"], vol, SleeveRule(), ROT_V1)


def test_trend_score_ignores_volatility(inp):
    t = inp.index[1100]
    trend = sector_scores(inp.panel, t, False, ROT_V2, SleeveRule(score="trend"))
    rs = inp.panel["rel_strength_6m"].loc[t][trend.index].rank(pct=True)
    expected = inp.panel["mom_12_1"].loc[t][trend.index].rank(pct=True) + ROT_V2.rs_weight * rs
    pd.testing.assert_series_equal(trend, expected.sort_values(ascending=False, kind="stable"), check_names=False)


def test_absolute_momentum_filter_sends_slots_to_core(inp):
    t = inp.index[1100]
    scores = sector_scores(inp.panel, t, False, ROT_V2, ROT_V2.rule_for("bull_calm", "bull"))
    mom = inp.panel["mom_12_1"].loc[t]
    kept = absolute_momentum_ok(scores, inp.panel, t)
    assert list(kept.index) == [s for s in scores.index if mom[s] > 0]
    # Tots els sectors amb momentum negatiu → tot el bloc de renda variable al mercat
    neg = {k: v.copy() for k, v in inp.panel.items()}
    neg["mom_12_1"].loc[:, :] = -0.1
    tg = build_targets(PortfolioInputs(**{**inp.__dict__, "panel": neg}), ROT_V2)
    w = tg.weights.loc[inp.index[1100]]
    e = inp.regime["equity_exposure"].iloc[1100]
    assert w["SPY"] == pytest.approx(e) and w[SECTORS].sum() == pytest.approx(0.0)


def test_v2_targets_by_regime(inp):
    tg = build_targets(inp, ROT_V2)
    w = tg.weights.dropna()
    assert np.allclose(w.sum(axis=1), 1.0)
    e = inp.regime["equity_exposure"].reindex(w.index)
    assert np.allclose(w[[*SECTORS, "SPY"]].sum(axis=1), e)
    lg = tg.log
    bull = lg[lg["regime"].isin(["bull_calm", "bull_volatile"])]
    bear = lg[lg["regime"] == "bear_volatile"]
    assert (bull["core_share"] >= 0.70 - 1e-12).all() and (bull["score"] == "trend").all()
    assert np.allclose(bear["core_share"], 0.50) and (bear["score"] == "risk_adj_low_beta").all()
    assert w.loc[inp.index[650], "SPY"] == pytest.approx(0.25 * 0.5)  # bear_volatile: E = 25 %
    assert w.loc[inp.index[720], [*SECTORS, "SPY"]].sum() == 0  # crisi


def test_fast_hedge_confirmation():
    cfg = ROT_V2
    # 3 mesos negativa però 1 mes positiva → no hi ha bons (cas 2022)
    assert hedge_weights(-0.2, {"IEF", "GLD"}, cfg, 0.3) == ({"GLD": 0.5, CASH: 0.5}, "or+cash")
    assert hedge_weights(-0.2, {"IEF", "GLD"}, cfg, -0.1) == ({"IEF": 1.0}, "bons")
    assert hedge_weights(-0.2, {"IEF", "GLD"}, cfg, np.nan) == ({"IEF": 1.0}, "bons")
    # rot_v1 ignora la correlació ràpida
    assert hedge_weights(-0.2, {"IEF", "GLD"}, ROT_V1, 0.3) == ({"IEF": 1.0}, "bons")


def test_v2_comparison_and_ablations(inp):
    res = ev.v2_comparison(inp, ROT_V1, ROT_V2)
    assert {ev.V1, ev.V2, "v2 sense condicionar"} <= set(res["sims"])
    full = res["tables"]["Completa"].set_index("estrategia")
    assert full["CAGR_%"].notna().all()
    v = res["verdict"].set_index("finestra")["passa"]
    assert list(v.index) == ["Disseny 2000–2015", "Comprovació 2016–"]
    assert v["Disseny 2000–2015"] in (True, False) and v["Comprovació 2016–"] is None  # dades sintètiques fins al 2014
    # "Sense condicionar" fa servir la regla alcista també en tendència baixista
    lg = res["targets"]["v2 sense condicionar"].log
    assert (lg.loc[lg["regime"] == "bear_volatile", "score"] == "trend").all()
