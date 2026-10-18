"""Configuració global, llegida de variables d'entorn (prefix FIN_) i de `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FIN_",
        env_file=(".env", "/etc/finance/finance.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("~/finance-data"))
    universe_file: Path = Field(default=REPO_ROOT / "config" / "universe.yaml")
    fred_api_key: str | None = None
    log_level: str = "INFO"
    log_json: bool = False

    # Proveïdors: límits de peticions (peticions/segon) i paral·lelisme
    yahoo_rate_per_sec: float = 2.0
    yahoo_max_workers: int = 4
    yahoo_batch_size: int = 50
    fred_rate_per_sec: float = 1.5
    # SEC EDGAR (insiders): User-Agent obligatori "Nom correu"; màxim oficial 10 peticions/s
    sec_user_agent: str | None = None
    sec_rate_per_sec: float = 5.0
    # Primer any de l'històric d'insiders (datasets trimestrals des del 2003)
    insider_start_year: int = 2006
    # Dies màxims que el job diari recupera via XML després de l'últim trimestre publicat
    insider_max_daily_days: int = 120

    # Ingesta incremental: dies que es tornen a baixar per cobrir correccions del proveïdor
    price_overlap_days: int = 10
    # Tolerància relativa per detectar que l'adj_close històric ha canviat (nou dividend/split)
    adj_close_tolerance: float = 1e-5

    # Paper trading: capital nocional per convertir pesos en imports i accions a l'informe diari
    paper_capital: float = 10_000.0
    paper_currency: str = "USD"

    # Setmana 11: execució real ------------------------------------------------
    # off: no fa res · manual: pla + alertes, les execucions s'apunten a mà (`fin exec fill`)
    # ibkr: envia les ordres a IB Gateway (compte paper per defecte; real només amb exec_live)
    exec_mode: Literal["off", "manual", "ibkr"] = "off"
    exec_instruments_file: Path = Field(default=REPO_ROOT / "config" / "execution.yaml")
    exec_strategy: str | None = None  # per defecte, la de referència (ACTIVE[0])
    exec_currency: str = "EUR"
    exec_capital: float = 0.0  # mode manual: efectiu inicial (o `fin exec deposit`)
    exec_band: float = 0.05  # opera si màx |pes real - objectiu| > banda (a més de quan opera el paper)
    exec_min_order: float = 250.0  # per sota, la comissió mínima d'IBKR (3 €) pesa > 1,2 %
    exec_limit_bps: float = 30.0  # límit = referència ± 30 pb
    exec_max_price_move: float = 0.03  # no envia si el preu s'ha mogut > 3 % des del pla
    exec_max_order_pct: float = 0.8  # cap ordre > 80 % del valor gestionat (topall contra errors)
    exec_max_turnover: float = 1.05  # import total del pla ≤ 105 % del valor gestionat
    exec_cash_buffer: float = 0.01  # efectiu que no es fa servir per comprar
    exec_wait_fills_s: int = 600  # espera les vendes abans d'enviar les compres
    exec_require_approval: bool | None = None  # None = obligatòria amb compte real
    exec_live: bool = False  # cal True per operar en un compte real
    exec_account: str | None = None  # compte real esperat (ha de coincidir)
    exec_exchange: str = "SMART"  # o IBIS (Xetra) per forçar la plaça
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002  # IB Gateway: 4002 paper, 4001 real
    ibkr_client_id: int = 17
    ibkr_market_data_type: int = 3  # 1 temps real; 3 = temps real si hi ha subscripció, si no diferit

    # Setmana 11: alertes ------------------------------------------------------
    alert_ntfy_url: str | None = None  # p. ex. https://ntfy.sh/un-tema-dificil-d-endevinar
    alert_ntfy_token: str | None = None
    alert_telegram_token: str | None = None
    alert_telegram_chat_id: str | None = None
    alert_smtp_host: str | None = None
    alert_smtp_port: int = 587
    alert_smtp_user: str | None = None
    alert_smtp_password: str | None = None
    alert_smtp_from: str | None = None
    alert_smtp_to: str | None = None
    alert_min_level: Literal["info", "important", "critical"] = "important"
    alert_weekly_summary: bool = True  # resum de divendres encara que no passi res
    alert_healthcheck_url: str | None = None  # ping en acabar el pipeline (healthchecks.io...)

    @field_validator("data_dir", "universe_file", "exec_instruments_file", mode="after")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    # Rutes derivades ----------------------------------------------------------
    @property
    def lake_dir(self) -> Path:
        return self.data_dir / "lake"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.duckdb"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def exec_dir(self) -> Path:
        return self.data_dir / "exec"

    @property
    def halt_file(self) -> Path:
        """Interruptor d'aturada: si existeix, no s'envia cap ordre."""
        return self.exec_dir / "HALT"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.lake_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
