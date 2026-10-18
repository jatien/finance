"""Sèrie del senyal (ETF dels EUA) → instrument operable (UCITS a Xetra).

Font per defecte: `fiscal.config.UCITS_MAP`. `config/execution.yaml` hi afegeix ISIN verificats
o substitueix instruments. Sense ticker i ISIN, el símbol no és operable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from finance.fiscal.config import UCITS_MAP
from finance.providers.base import TradableInstrument
from finance.settings import REPO_ROOT

DEFAULT_FILE = REPO_ROOT / "config" / "execution.yaml"


@dataclass
class InstrumentMap:
    currency: str
    yahoo_suffix: str
    tradable: dict[str, TradableInstrument]
    missing: dict[str, str]  # símbol → per què no es pot operar

    def get(self, symbol: str) -> TradableInstrument | None:
        return self.tradable.get(symbol)

    def symbol_of(self, ticker: str) -> str | None:
        for s, inst in self.tradable.items():
            if inst.ticker == ticker:
                return s
        return None


def load_instruments(path: Path | None = None, currency: str = "EUR") -> InstrumentMap:
    path = path or DEFAULT_FILE
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = cfg or {}
    cur = cfg.get("currency", currency)
    entries: dict[str, dict] = {s: {"ticker": u.xetra, "isin": u.isin, "name": u.name}
                                for s, u in UCITS_MAP.items()}
    for s, over in (cfg.get("instruments") or {}).items():
        entries[s] = {**entries.get(s, {}), **(over or {})}
    tradable, missing = {}, {}
    for s, e in entries.items():
        if not e.get("ticker"):
            missing[s] = "sense ticker operable"
        elif not e.get("isin"):
            missing[s] = f"{e['ticker']}: ISIN no verificat (config/execution.yaml)"
        else:
            tradable[s] = TradableInstrument(symbol=s, ticker=str(e["ticker"]), isin=str(e["isin"]),
                                             currency=e.get("currency", cur), name=e.get("name", ""))
    return InstrumentMap(currency=cur, yahoo_suffix=cfg.get("yahoo_suffix", ".DE"),
                         tradable=tradable, missing=missing)
