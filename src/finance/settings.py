"""Configuració global, llegida de variables d'entorn (prefix FIN_) i de `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # Ingesta incremental: dies que es tornen a baixar per cobrir correccions del proveïdor
    price_overlap_days: int = 10
    # Tolerància relativa per detectar que l'adj_close històric ha canviat (nou dividend/split)
    adj_close_tolerance: float = 1e-5

    @field_validator("data_dir", "universe_file", mode="after")
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

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.lake_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
