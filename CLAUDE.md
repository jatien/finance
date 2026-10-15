# Convencions del projecte

- Comentaris i docstrings del codi en català; documentació (`README.md`, `docs/`) en castellà.
- Entorn amb `uv` (Python de `.python-version`). Tests: `uv run pytest` (sense xarxa; els
  proveïdors falsos són a `tests/conftest.py`). Lint: `uv run ruff check src tests`.
- Dades fora del repositori (`FIN_DATA_DIR`). Mai fer commit de Parquet, DuckDB ni `.env`.
- El lake és append-only: no modificar fitxers existents; la deduplicació la fan les vistes
  (`storage/catalog.py`) i la compactació (`storage/lake.py`).
- Tot dataset nou es declara a `storage/schemas.py` (esquema Arrow, clau primària, partició).
- Tot accés a fonts externes passa per una interfície de `providers/base.py`.
- Point-in-time: cap feature ni model pot fer servir dades no disponibles a la data simulada
  (`macro_asof`, `universe_asof`, `ingested_at`).
- Noms de columna: evitar paraules reservades de DuckDB (`check`, `asof`...).
