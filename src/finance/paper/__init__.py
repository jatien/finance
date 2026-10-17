"""Setmana 8: paper trading congelat i informe diari.

    book.py     motor: estat del llibre, inici, avanç sessió a sessió (mateix pas que el backtest)
    job.py      job `paper`: escriu les sessions noves a `paper_book` i `paper_nav`
    report.py   informe diari (HTML + resum de text) i job `report`
"""

from finance.paper.book import BookState, advance, replay, start

__all__ = ["BookState", "advance", "replay", "start"]
