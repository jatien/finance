"""Job `alerts`: avalua les regles després de l'informe diari i envia les alertes noves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from finance.alerts.dispatch import Alert, dispatch
from finance.alerts.rules import failed_job_alerts, report_alerts
from finance.jobs import Job, Run


class AlertsJob(Job):
    name = "alerts"

    def execute(self, run: Run, facts_file: str | None = None, test: bool = False, **_: Any) -> None:
        s = self.ctx.settings
        if test:
            sent = dispatch(self.ctx, run, [Alert(f"test:{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}", "critical",
                                                  "Prueba de alertas", "Si lees esto, el canal funciona.")],
                            min_level="info")
            run.ok(len(sent))
            run.message = f"prova enviada per {', '.join(n.name for n in self.ctx.providers.notifiers) or 'cap canal'}"
            return
        alerts: list[Alert] = []
        since = (self.ctx.now() - pd.Timedelta(hours=12)).to_pydatetime()
        try:
            runs = self.ctx.query("""SELECT run_id, job, status, errors, message FROM ingest_runs
                                     WHERE started_at >= ? AND job NOT IN ('alerts', 'options')""", [since])
        except Exception:  # noqa: BLE001
            runs = pd.DataFrame()
        alerts += failed_job_alerts(runs)
        path = Path(facts_file) if facts_file else s.data_dir / "reports" / "daily" / "latest.json"
        if path.exists():
            facts = json.loads(path.read_text(encoding="utf-8"))
            from finance.execution.job import exec_strategy
            ex = exec_strategy(s).name if s.exec_mode != "off" else None
            alerts += report_alerts(facts, exec_strategy=ex, weekly=s.alert_weekly_summary)
        else:
            alerts.append(Alert(f"nofacts:{self.ctx.now():%Y%m%d}", "critical", "Sin informe diario",
                                f"No existe {path}: el job `report` no ha llegado a ejecutarse."))
        weekly = [a for a in alerts if a.key.startswith("weekly:")]
        rest = [a for a in alerts if not a.key.startswith("weekly:")]
        sent = dispatch(self.ctx, run, rest) + dispatch(self.ctx, run, weekly, min_level="info")
        run.ok()
        if not self.ctx.providers.notifiers:
            run.message = f"{len(alerts)} alertes avaluades; cap canal configurat (FIN_ALERT_*)"
        else:
            run.message = f"{len(sent)} alertes enviades de {len(alerts)} avaluades"
