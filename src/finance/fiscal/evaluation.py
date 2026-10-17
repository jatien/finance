"""Comparació pre-registrada de la setmana 9 (vegeu docs/fiscalidad.md).

Inversor: 80.000 € inicials + 300 €/mes, IBKR Fixed (Xetra), UCITS, IRPF amb l'escala de
l'estalvi 2026, liquidació total al final. Tres dates d'inici, totes fins a l'última sessió.
"""

from __future__ import annotations

import pandas as pd

from finance.fiscal.config import Investor
from finance.fiscal.data import dividends, macro_series
from finance.fiscal.sim import MarketData, Policy, TaxSimResult, market_data, simulate_tax
from finance.portfolio.config import ROT_V1B, ROT_V2, PortfolioConfig
from finance.portfolio.construct import PortfolioInputs, Targets, build_targets, fixed_targets
from finance.portfolio.evaluation import regime_on_market
from finance.portfolio.job import load_inputs

STARTS = ("2000-02-01", "2008-01-02", "2016-01-04")
ACUM_V1 = Policy(kind="accumulate", expo_band=0.10, demote_buffer=5, tax_aware_sells=True, avoid_wash=True)
REF = "rot_v1b"
CANDIDATE = "acum_v1"
MAX_DD_SLACK = 5.0  # punts


def strategies(inp: PortfolioInputs) -> dict[str, tuple[Targets, PortfolioConfig, Policy]]:
    t1 = build_targets(inp, ROT_V1B)
    t2 = build_targets(inp, ROT_V2)
    spy = fixed_targets(inp.index, {ROT_V1B.market: 1.0}, monthly=False, start=t1.weights.dropna().index.min())
    return {
        "SPY comprar i mantenir": (spy, ROT_V1B, Policy("target")),
        "Règims sobre SPY": (regime_on_market(inp, ROT_V1B), ROT_V1B, Policy("target")),
        REF: (t1, ROT_V1B, Policy("target")),
        "rot_v2": (t2, ROT_V2, Policy("target")),
        CANDIDATE: (t1, ROT_V1B, ACUM_V1),
        "acum sense vendes per règim": (t1, ROT_V1B, Policy("accumulate", regime_sells=False)),
        "acum sense degradació": (t1, ROT_V1B, Policy("accumulate", demote_buffer=None)),
        "acum sobre rot_v2": (t2, ROT_V2, ACUM_V1),
    }


def load_market(ctx, inp: PortfolioInputs, investor: Investor) -> MarketData:
    return market_data(inp, macro_series(ctx, "DEXUSEU"), macro_series(ctx, "ECBDFR"),
                       dividends(ctx, list(inp.prices.columns)), investor)


def run(ctx, investor: Investor | None = None, starts: tuple[str, ...] = STARTS,
        names: list[str] | None = None) -> dict[str, dict[str, TaxSimResult]]:
    investor = investor or Investor()
    inp = load_inputs(ctx, ROT_V2)  # superconjunt d'entrades
    md = load_market(ctx, inp, investor)
    strat = strategies(inp)
    out: dict[str, dict[str, TaxSimResult]] = {}
    for s in starts:
        out[s] = {}
        for name, (t, cfg, pol) in strat.items():
            if names and name not in names:
                continue
            out[s][name] = simulate_tax(name, inp, t, cfg, md, investor, pol, pd.Timestamp(s))
    return out


def summary_table(results: dict[str, dict[str, TaxSimResult]]) -> pd.DataFrame:
    rows = []
    for s, res in results.items():
        for r in res.values():
            rows.append({"inici": s, **r.summary()})
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame) -> pd.DataFrame:
    """acum_v1 passa a ser la referència si guanya rot_v1b en TIR neta en >= 2 de 3 inicis i la
    seva caiguda màxima no és més de 5 punts pitjor en cap inici."""
    rows = []
    for s, g in table.groupby("inici"):
        g = g.set_index("estrategia")
        if REF not in g.index or CANDIDATE not in g.index:
            continue
        rows.append({"inici": s, "TIR_ref_%": g.at[REF, "TIR_neta_%"], "TIR_acum_%": g.at[CANDIDATE, "TIR_neta_%"],
                     "dd_ref_%": g.at[REF, "max_dd_%"], "dd_acum_%": g.at[CANDIDATE, "max_dd_%"],
                     "guanya_TIR": bool(g.at[CANDIDATE, "TIR_neta_%"] > g.at[REF, "TIR_neta_%"]),
                     "caiguda_ok": bool(g.at[CANDIDATE, "max_dd_%"] >= g.at[REF, "max_dd_%"] - MAX_DD_SLACK)})
    v = pd.DataFrame(rows)
    if not v.empty:
        v.attrs["passa"] = bool(v["guanya_TIR"].sum() >= 2 and v["caiguda_ok"].all())
    return v
