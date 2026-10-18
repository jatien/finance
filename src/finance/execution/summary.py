"""Resum de l'execució real per a l'informe diari: plans recents, ordres, slippage i real vs paper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def exec_summary(ctx, paper_nav: pd.DataFrame | None = None) -> dict[str, Any]:
    from finance.execution.job import exec_strategy, read_account, read_orders, read_plans
    s = ctx.settings
    out: dict[str, Any] = {"mode": s.exec_mode, "strategy": exec_strategy(s).name,
                           "halted": s.halt_file.read_text(encoding="utf-8").strip() if s.halt_file.exists() else None}
    plans = read_plans(ctx)
    out["plans"] = plans.tail(5).iloc[::-1].reset_index(drop=True) if not plans.empty else pd.DataFrame()
    active = plans[plans["status"] != "no_trade"] if not plans.empty else plans
    out["last_plan"] = active.iloc[-1].to_dict() if not active.empty else None
    out["orders"] = read_orders(ctx, out["last_plan"]["plan_id"]) if out["last_plan"] else pd.DataFrame()
    allo = read_orders(ctx)
    filled = allo[allo["filled_qty"].fillna(0) > 0] if not allo.empty else allo
    if not filled.empty:
        w = filled["filled_qty"] * filled["avg_price"]
        sl = filled["slippage_bps"]
        ok = sl.notna()
        out["slippage_bps"] = float((sl[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        out["commission"] = float(filled["commission"].fillna(0).sum())
        out["traded_value"] = float(w.sum())
        out["fill_rate"] = float(filled["filled_qty"].sum() / allo.loc[allo["status"] != "skipped", "quantity"].sum())
    acc = read_account(ctx)
    if not acc.empty:
        acc = acc[acc["account"] == acc["account"].iloc[-1]]
        first, last = acc.iloc[0], acc.iloc[-1]
        flows = float(acc["net_flows"].iloc[1:].fillna(0).sum())
        out["account"] = {"account": last["account"], "date": last["date"], "value": float(last["net_liquidation"]),
                          "cash": float(last["cash"]), "currency": last["currency"], "since": first["date"],
                          "ret": (float(last["net_liquidation"]) - flows) / float(first["net_liquidation"]) - 1
                          if first["net_liquidation"] > 0 else np.nan}
        if paper_nav is not None and not paper_nav.empty:
            nv = paper_nav.set_index("date")["nav"]
            a, b = nv.asof(first["date"]), nv.asof(last["date"])
            out["account"]["paper_ret"] = float(b / a - 1) if pd.notna(a) and pd.notna(b) and a > 0 else np.nan
    return out
