"""Univers d'instruments: càrrega del YAML, resolució de grups (estàtics i dinàmics)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class Instrument(BaseModel):
    symbol: str
    asset_class: str
    group: str
    name: str | None = None
    sector: str | None = None
    calendar: str = "XNYS"


class GroupConfig(BaseModel):
    asset_class: str = "unknown"
    calendar: str | None = None
    symbols: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
    dynamic: str | None = None  # p.ex. "sp500": es resol des del lake

    @model_validator(mode="after")
    def _check(self) -> GroupConfig:
        if not self.symbols and not self.dynamic:
            raise ValueError("un grup necessita `symbols` o `dynamic`")
        return self


class OptionsConfig(BaseModel):
    underlyings: list[str]
    max_dte_days: int = 760
    min_dte_days: int = 0
    moneyness_min: float | None = None
    moneyness_max: float | None = None


class MacroSeries(BaseModel):
    series_id: str
    name: str
    frequency: str = "D"  # D, W, M, Q
    revisable: bool = False


class UniverseConfig(BaseModel):
    defaults: dict[str, Any] = Field(default_factory=dict)
    groups: dict[str, GroupConfig]
    universes: dict[str, list[str]]
    options: OptionsConfig
    macro: dict[str, MacroSeries]

    @model_validator(mode="before")
    @classmethod
    def _inject_series_ids(cls, data: dict[str, Any]) -> dict[str, Any]:
        macro = data.get("macro") or {}
        data["macro"] = {sid: {"series_id": sid, **(cfg or {})} for sid, cfg in macro.items()}
        return data

    @model_validator(mode="after")
    def _check_refs(self) -> UniverseConfig:
        for uname, groups in self.universes.items():
            missing = [g for g in groups if g not in self.groups]
            if missing:
                raise ValueError(f"univers '{uname}' referencia grups inexistents: {missing}")
        return self

    @property
    def default_calendar(self) -> str:
        return self.defaults.get("calendar", "XNYS")

    @property
    def history_start(self) -> str:
        return str(self.defaults.get("history_start", "1990-01-01"))


# Funció que, donat el nom d'un univers dinàmic (p.ex. "sp500"), en retorna els membres
# com a llista de diccionaris amb almenys `symbol` (i opcionalment `name`, `sector`).
DynamicResolver = Callable[[str], list[dict[str, Any]]]


def to_yahoo_symbol(symbol: str) -> str:
    """Normalitza tickers a la convenció de Yahoo (BRK.B -> BRK-B)."""
    s = symbol.strip().upper()
    if s.startswith("^"):
        return s
    return s.replace(".", "-")


class Universe:
    def __init__(self, config: UniverseConfig, resolver: DynamicResolver | None = None):
        self.config = config
        self._resolver = resolver

    @classmethod
    def load(cls, path: Path, resolver: DynamicResolver | None = None) -> Universe:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(UniverseConfig.model_validate(raw), resolver)

    def group(self, name: str) -> list[Instrument]:
        g = self.config.groups[name]
        cal = g.calendar or self.config.default_calendar
        if g.dynamic:
            if self._resolver is None:
                raise RuntimeError(f"el grup '{name}' és dinàmic però no hi ha resolver")
            members = self._resolver(g.dynamic)
            return [
                Instrument(
                    symbol=to_yahoo_symbol(m["symbol"]), asset_class=g.asset_class, group=name,
                    name=m.get("name"), sector=m.get("sector"), calendar=cal,
                )
                for m in members
            ]
        out = []
        for sym, attrs in g.symbols.items():
            attrs = dict(attrs or {})
            out.append(
                Instrument(
                    symbol=to_yahoo_symbol(sym),
                    asset_class=attrs.pop("asset_class", g.asset_class),
                    group=name,
                    calendar=attrs.pop("calendar", cal),
                    **attrs,
                )
            )
        return out

    def instruments(self, universe: str, *, skip_unresolved: bool = True) -> list[Instrument]:
        """Instruments d'un univers, sense duplicats (el primer grup guanya)."""
        seen: dict[str, Instrument] = {}
        for gname in self.config.universes[universe]:
            try:
                members = self.group(gname)
            except RuntimeError:
                if skip_unresolved:
                    continue
                raise
            for inst in members:
                seen.setdefault(inst.symbol, inst)
        return list(seen.values())

    def symbols(self, universe: str) -> list[str]:
        return [i.symbol for i in self.instruments(universe)]

    @property
    def options(self) -> OptionsConfig:
        return self.config.options

    @property
    def macro(self) -> list[MacroSeries]:
        return list(self.config.macro.values())
