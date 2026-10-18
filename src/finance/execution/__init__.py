"""Setmana 11: execució real de l'estratègia de referència.

    instruments.py  sèrie del senyal (ETF dels EUA) → UCITS operable (ticker, ISIN)
    plan.py         objectiu en pesos → ordres en accions senceres (funcions pures)
    guards.py       salvaguardes: aturada, dades, compte real, finestra de mercat, aprovació
    ledger.py       mode manual: posicions i efectiu a partir de les execucions apuntades
    job.py          job `exec`: plan, approve, submit, sync, fill, halt...
"""
