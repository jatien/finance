"""Historial de composició d'índexs: intervals de pertinença i altes/baixes.

Combina dues fonts en una única seqüència de (data, membres):
1. Historial extern (des de 1996) fins a la seva última data.
2. Fotos pròpies (`universe_membership`, de Wikipedia) posteriors a aquesta data.

D'aquesta seqüència es deriven:
- `universe_intervals`: [start_date, end_date) per símbol (base de `universe_asof`).
- `index_changes`: una fila per alta o baixa.

Tots dos datasets es regeneren sencers a cada execució (`snapshot_by`): si la font externa
corregeix una data antiga, la versió nova substitueix l'anterior sense deixar restes.

Els canvis detectats entre dues fotos tenen data incerta: es guarda la data d'observació
(conservadora, sense look-ahead) i l'amplada de la finestra en dies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from finance.jobs import Job, Run
from finance.storage.schemas import INDEX_CHANGES, UNIVERSE_INTERVALS


@dataclass(frozen=True)
class Observation:
    day: date
    members: frozenset[str]
    source: str
    exact: bool  # True = la data és la data efectiva del canvi


def build_sequence(external: list[tuple[date, frozenset[str]]], external_source: str,
                   snapshots: list[tuple[date, frozenset[str]]]) -> list[Observation]:
    seq = [Observation(d, m, external_source, True) for d, m in external]
    last = seq[-1].day if seq else date.min
    seq += [Observation(d, m, "snapshot", False) for d, m in sorted(snapshots) if d > last]
    return seq


def derive(universe: str, seq: list[Observation],
           names: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (intervals, canvis) a partir d'una seqüència ordenada d'observacions."""
    names = names or {}
    intervals: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    open_: dict[str, dict[str, Any]] = {}
    prev: Observation | None = None
    for obs in seq:
        if prev is None:
            for s in obs.members:
                open_[s] = dict(universe=universe, symbol=s, start_date=obs.day, start_censored=True,
                                start_window_days=0, start_source=obs.source)
            prev = obs
            continue
        window = 0 if obs.exact else (obs.day - prev.day).days
        note = (f"derived:{obs.source}" if obs.exact
                else f"derived:{obs.source}; observat entre {prev.day} i {obs.day}")
        for s in sorted(obs.members - prev.members):
            open_[s] = dict(universe=universe, symbol=s, start_date=obs.day, start_censored=False,
                            start_window_days=window, start_source=obs.source)
            changes.append(dict(universe=universe, date=obs.day, added_symbol=s,
                                added_name=names.get(s), removed_symbol="", removed_name=None,
                                reason=note))
        for s in sorted(prev.members - obs.members):
            iv = open_.pop(s)
            intervals.append({**iv, "end_date": obs.day, "end_window_days": window,
                              "end_source": obs.source})
            changes.append(dict(universe=universe, date=obs.day, added_symbol="", added_name=None,
                                removed_symbol=s, removed_name=names.get(s), reason=note))
        prev = obs
    for iv in open_.values():
        intervals.append({**iv, "end_date": None, "end_window_days": None, "end_source": None})
    return pd.DataFrame(intervals), pd.DataFrame(changes)


class UniverseHistoryJob(Job):
    name = "universe_history"

    def execute(self, run: Run, universes: list[str] | None = None, **_: Any) -> None:
        ctx = self.ctx
        dynamic = sorted({g.dynamic for g in ctx.universe.config.groups.values() if g.dynamic})
        provider = ctx.providers.index_history
        for uni in universes or dynamic:
            try:
                external = provider.membership_history(uni)
            except Exception as e:  # noqa: BLE001
                run.fail(uni, e)
                continue
            snaps_df = ctx.query(
                "SELECT as_of_date, symbol, name FROM universe_membership WHERE universe = ?", [uni])
            snapshots = [
                (pd.Timestamp(d).date(), frozenset(g["symbol"]))
                for d, g in snaps_df.groupby("as_of_date")
            ]
            names = dict(zip(snaps_df["symbol"], snaps_df["name"], strict=True))
            seq = build_sequence(external, provider.name, snapshots)
            intervals, changes = derive(uni, seq, names)
            source = f"{provider.name}+snapshot" if len(seq) > len(external) else provider.name
            run.write(UNIVERSE_INTERVALS, intervals, source)
            if not changes.empty:
                run.write(INDEX_CHANGES, changes, source)
            run.ok()
            n_snap = len(seq) - len(external)
            run.message = (f"{uni}: {len(external)} dates externes ({external[0][0]}→{external[-1][0]})"
                           f" + {n_snap} fotos pròpies; {len(intervals)} intervals, {len(changes)} canvis")
