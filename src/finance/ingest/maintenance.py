"""Manteniment del lake: compactació i regeneració del catàleg."""

from __future__ import annotations

from typing import Any

from finance.jobs import Job, Run
from finance.storage.schemas import REGISTRY


class CompactJob(Job):
    name = "compact"

    def execute(self, run: Run, datasets: list[str] | None = None, min_files: int = 4, **_: Any) -> None:
        targets = datasets or list(REGISTRY)
        before = after = 0
        for name in targets:
            try:
                res = self.ctx.lake.compact(REGISTRY[name], min_files=min_files)
                before += res.files_before
                after += res.files_after
                run.ok()
            except Exception as e:  # noqa: BLE001
                run.fail(name, e)
        run.message = f"{before} fitxers fusionats en {after}"
        self.ctx.refresh_catalog()
