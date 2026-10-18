"""Setmana 11: alertes actives.

    dispatch.py  enviament amb deduplicació (`alerts_sent`)
    rules.py     regles: dades en vermell, canvi de règim, ordres, caigudes, jobs fallits
    job.py       job `alerts` (al final del pipeline diari)
Els canals (ntfy, Telegram, correu) són a `providers/notify.py`.
"""
