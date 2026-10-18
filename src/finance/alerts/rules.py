"""Regles d'alerta a partir dels fets de l'informe diari (`latest.json`) i del registre de jobs.

Nivells: critical (no operar / cal actuar ja), important (hi ha ordres o canvis), info (resum).
Per defecte només s'envien important i critical (`FIN_ALERT_MIN_LEVEL`). La clau de cada
alerta inclou la data o l'episodi: un mateix fet no s'avisa dues vegades.
"""

from __future__ import annotations

from typing import Any

from finance.alerts.dispatch import Alert

ACTIONS = {"enter": "comprar (nueva)", "add": "comprar", "reduce": "vender", "exit": "vender (todo)"}
DD_LEVELS = (0.10, 0.15, 0.20)


def report_alerts(f: dict[str, Any], *, exec_strategy: str | None = None, weekly: bool = True) -> list[Alert]:
    out: list[Alert] = []
    day = f.get("data_date") or "sin-datos"
    cur = f.get("currency", "USD")
    if not f.get("tradable", False):
        errs = [t for sev, t in f.get("issues", []) if sev == "error"]
        out.append(Alert(f"data:{day}:{f.get('expected_session')}", "critical", "NO OPERAR: problema de datos",
                         "\n".join(f"• {t}" for t in errs) or "informe en rojo"))
    rg = f.get("regime") or {}
    if rg.get("changed_today"):
        out.append(Alert(f"regime:{day}:{rg.get('regime')}", "important",
                         f"Cambio de régimen: {rg.get('label')}",
                         f"{rg.get('previous_label') or '—'} → {rg.get('label')}\n"
                         f"Exposición a renta variable recomendada: {rg.get('exposure', 0):.0%}\n"
                         "Las estrategias revisan la cartera en la sesión siguiente."))
    for s in f.get("strategies", []):
        if not s.get("started"):
            continue
        name = s["name"]
        if s.get("will_trade") and s.get("orders") and name != exec_strategy:
            lines = [f"• {ACTIONS.get(o['action'], o['action'])} {o['symbol']}: {o['current']:.1%} → {o['target']:.1%} "
                     f"({o['amount']:+,.0f} {cur})" for o in s["orders"]]
            out.append(Alert(f"orders:{name}:{s.get('decision_date')}", "important" if s.get("reference") else "info",
                             f"{name}: órdenes para la sesión siguiente ({len(lines)})",
                             f"{s.get('order_note', '')}\n\n" + "\n".join(lines)))
        dd = s.get("dd_now")
        if dd is not None:
            for thr in DD_LEVELS:
                if dd <= -thr:
                    out.append(Alert(f"dd:{name}:{thr:.2f}:{s.get('peak_date')}", "important" if thr < 0.2 else "critical",
                                     f"{name}: caída del {dd:.1%} desde máximos",
                                     f"Máximo el {s.get('peak_date')}. Umbral superado: {thr:.0%}.\n"
                                     f"Rentabilidad desde el inicio {s.get('ret', 0):+.2%} (SPY {s.get('bench_ret', 0):+.2%})."))
    if weekly and f.get("data_date"):
        import pandas as pd
        if pd.Timestamp(f["data_date"]).weekday() == 4:
            out.append(weekly_summary(f))
    return out


def weekly_summary(f: dict[str, Any]) -> Alert:
    rg = f.get("regime") or {}
    L = [("Datos OK" if f.get("tradable") else "Datos con problemas"),
         f"Régimen: {rg.get('label', '—')} · renta variable {rg.get('exposure', 0):.0%} · {rg.get('days', 0)} sesiones"]
    for s in f.get("strategies", []):
        if s.get("started"):
            L.append(f"{s['name']}: {s.get('ret', 0):+.2%} desde el inicio (SPY {s.get('bench_ret', 0):+.2%}), "
                     f"caída actual {s.get('dd_now') or 0:.1%}")
    if f.get("next_review"):
        L.append(f"Próxima revisión mensual: {f['next_review']}")
    return Alert(f"weekly:{f['data_date']}", "info", f"Resumen semanal {f['data_date']}", "\n".join(L))


def failed_job_alerts(runs) -> list[Alert]:
    """`runs`: últimes execucions (DataFrame amb run_id, job, status, errors, message)."""
    out = []
    for r in runs.itertuples():
        # Les fallades controlades d'`exec` (salvaguardes) ja envien la seva pròpia alerta
        if r.job == "exec" and "__job__" not in (r.errors or ""):
            continue
        if r.status == "failed":
            out.append(Alert(f"jobfail:{r.run_id}", "critical", f"Job `{r.job}` ha fallado",
                             f"{(r.message or '')[:300]}\n{(r.errors or '')[:600]}\n\nLog: fin status / journalctl -u finance-daily"))
    return out
