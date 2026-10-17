"""Senyals d'insiders: preparació a DuckDB i esdeveniments/amplitud amb numpy/pandas.

Principis (literatura: Lakonishok i Lee 2001; Cohen, Malloy i Pomorski 2012):
- Les **compres** en mercat obert són informatives; les vendes molt menys (liquiditat,
  diversificació, impostos). Les atorgaments (A), exercicis (M) i retencions fiscals (F)
  no aporten informació i s'ignoren.
- Les operacions dins d'un **pla 10b5-1** són preprogramades: fora.
- Els insiders **rutinaris** (operen sempre el mateix mes) no prediuen res; els
  oportunistes sí. Classificació point-in-time (només mira anys anteriors).
- Els accionistes >10 % que no són directius ni consellers (fons) queden fora.
- **Point-in-time**: una operació es coneix a partir de la presentació (`filing_date`),
  no de quan es va fer. La senyal és vàlida a la primera sessió posterior a la
  presentació (moltes es presenten després del tancament).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finance.insiders.config import InsiderConfig

CEO_CFO_RE = (r"\b(?:CEO|CFO|CO-CEO|CHIEF EXECUTIVE|CHIEF FINANCIAL|PRINCIPAL EXECUTIVE|"
              r"PRINCIPAL FINANCIAL)\b")

# Columnes de les operacions preparades (una fila per operació P/S candidata)
PREPARED_COLUMNS = ["symbol", "issuer_cik", "sector", "owner_cik", "owner_name", "is_officer",
                    "is_director", "side", "cls", "plan_10b5_1", "ceo_cfo", "filing_date",
                    "signal_date", "value", "member", "eligible"]


def _fetch(con, sql: str, params: list | None = None) -> pd.DataFrame:
    """Resultat de DuckDB com a DataFrame amb cadenes `object` (les agregacions per grup
    sobre cadenes Arrow són molt lentes) i dates com a datetime64[ns]."""
    cols = con.execute(sql, params or []).fetchnumpy()
    out = {}
    for k, a in cols.items():
        if np.ma.isMaskedArray(a):
            if a.dtype.kind in "fiu":
                a = a.astype("float64").filled(np.nan) if a.dtype.kind == "f" or a.mask.any() else a.filled(0)
            elif a.dtype.kind == "b":
                a = a.filled(False)
            elif a.dtype.kind == "M":
                a = a.filled(np.datetime64("NaT"))
            else:
                a = a.astype(object).filled(None)
        if getattr(a, "dtype", None) is not None and a.dtype.kind == "M":
            a = a.astype("datetime64[ns]")
        out[k] = a
    return pd.DataFrame(out)


def prepare_trades(con, cfg: InsiderConfig, sessions: pd.DatetimeIndex, *,
                   source: str = "insider_transactions", cik_map: str | None = None,
                   intervals: str | None = None, start=None, end=None) -> pd.DataFrame:
    """Operacions P/S candidates, anotades, calculades a DuckDB (escala a 20 anys de Form 4).

    - `cls`: 'routine' | 'opportunistic' | 'new' (CMP 2012). Rutinari = el mateix titular
      ha operat (P o S) a la mateixa empresa el mateix mes natural de cadascun dels
      `routine_years` anys anteriors; 'new' = encara no té prou historial. Point-in-time:
      només mira anys anteriors.
    - `signal_date`: primera sessió estrictament posterior a `filing_date`.
    - `symbol`/`sector`: via CIK (`cik_map`: cik, symbol, sector) quan existeix; el CIK és
      estable quan l'empresa canvia de ticker (FB → META). Si no, el símbol declarat.
    - `member`: l'emissor era al S&P 500 a la data de senyal (`intervals`).
    - Candidates: consellers o directius, import ≥ `min_value`, preu > 0, amb símbol i
      data de senyal. `eligible` = candidata, fora de pla 10b5-1 i no rutinària.
    `source`: taula o vista amb les columnes de `insider_transactions` (també en tests).
    """
    con.register("_sess", pd.DataFrame({"d": pd.DatetimeIndex(sessions).date}))
    y = int(cfg.routine_years)
    where = ["trans_code IN ('P', 'S')", "form_type IN ('4', '5')"]
    params: list = []
    if start is not None:
        where.append("filing_date >= ?")
        params.append(start)
    if end is not None:
        where.append("filing_date <= ?")
        params.append(end)
    cmap = (f"SELECT lpad(trim(cik), 10, '0') AS cik, any_value(symbol) AS symbol, "
            f"any_value(sector) AS sector FROM {cik_map} GROUP BY 1") if cik_map else \
        "SELECT NULL::VARCHAR AS cik, NULL::VARCHAR AS symbol, NULL::VARCHAR AS sector WHERE false"
    member = (f"""coalesce((SELECT true FROM {intervals} u WHERE u.universe = 'sp500'
                 AND u.symbol = c.symbol AND u.start_date <= c.signal_date
                 AND (u.end_date IS NULL OR u.end_date > c.signal_date) LIMIT 1), false)"""
              if intervals else "false")
    sql = f"""
    WITH t AS (
        SELECT issuer_cik, symbol AS rep_symbol,
               coalesce(TRY_CAST(owner_cik AS BIGINT), hash(owner_name)::BIGINT) AS owner_cik,
               owner_name, coalesce(is_officer, false) AS is_officer,
               coalesce(is_director, false) AS is_director, officer_title,
               CASE WHEN trans_code = 'P' THEN 'buy' ELSE 'sell' END AS side,
               coalesce(plan_10b5_1, false) AS plan_10b5_1, filing_date,
               abs(value) AS value, price,
               year(coalesce(trans_date, filing_date)) * 12 + month(coalesce(trans_date, filing_date)) AS ym
        FROM {source} WHERE {' AND '.join(where)}
    ), k AS (SELECT DISTINCT issuer_cik, owner_cik, ym FROM t),
    f AS (SELECT issuer_cik, owner_cik, min(ym) AS first_ym FROM k GROUP BY ALL),
    r AS (
        SELECT k.issuer_cik, k.owner_cik, k.ym FROM k
        WHERE {' AND '.join(f'EXISTS (SELECT 1 FROM k k{i} WHERE k{i}.issuer_cik = k.issuer_cik AND k{i}.owner_cik = k.owner_cik AND k{i}.ym = k.ym - {12 * i})' for i in range(1, y + 1))}
    ), cm AS ({cmap}),
    c AS (
        SELECT coalesce(cm.symbol, t.rep_symbol) AS symbol, t.issuer_cik, cm.sector,
               t.owner_cik, t.owner_name, t.is_officer, t.is_director, t.side,
               CASE WHEN t.ym - f.first_ym < {12 * y} THEN 'new'
                    WHEN r.ym IS NOT NULL THEN 'routine' ELSE 'opportunistic' END AS cls,
               t.plan_10b5_1,
               coalesce(regexp_matches(upper(t.officer_title), '{CEO_CFO_RE}'), false) AS ceo_cfo,
               t.filing_date, s.d AS signal_date, t.value
        FROM t
        JOIN f USING (issuer_cik, owner_cik)
        LEFT JOIN r USING (issuer_cik, owner_cik, ym)
        LEFT JOIN cm ON cm.cik = t.issuer_cik
        ASOF LEFT JOIN _sess s ON t.filing_date < s.d
        WHERE (t.is_officer OR t.is_director) AND t.value >= {float(cfg.min_value)} AND t.price > 0
    )
    SELECT symbol, issuer_cik, sector, owner_cik, owner_name, is_officer, is_director, side, cls,
           plan_10b5_1, ceo_cfo, filing_date, signal_date, value, {member} AS member,
           (NOT plan_10b5_1 AND cls <> 'routine') AS eligible
    FROM c WHERE signal_date IS NOT NULL AND symbol IS NOT NULL
    """
    df = _fetch(con, sql, params)
    con.unregister("_sess")
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in PREPARED_COLUMNS})
    return df[PREPARED_COLUMNS]


def _clusters(s: pd.DataFrame, cfg: InsiderConfig) -> list[dict]:
    """Dies en què una acció creua el llindar d'insiders diferents a la finestra mòbil.

    `s`: operacions elegibles d'un costat (compres o vendes). Tot amb numpy sobre arrays
    ordenats per (símbol, data): un bucle de pandas per acció és massa lent amb 20 anys.
    """
    s = s.sort_values(["symbol", "signal_date"], kind="stable")
    sym = s["symbol"].to_numpy(dtype=object)
    d = s["signal_date"].to_numpy(dtype="datetime64[ns]")
    owners = s["owner_cik"].to_numpy(dtype=object)
    officer = s["is_officer"].fillna(False).to_numpy(dtype=bool)
    ceo = s["ceo_cfo"].to_numpy(dtype=bool)
    val = s["value"].abs().to_numpy(dtype=float)
    names = s["owner_name"].to_numpy(dtype=object)
    filing = s["filing_date"].to_numpy(dtype="datetime64[ns]")
    cik = s["issuer_cik"].to_numpy(dtype=object)
    sector = s["sector"].to_numpy(dtype=object)
    win = np.timedelta64(cfg.cluster_window_days, "D")
    cool = np.timedelta64(cfg.cooldown_days, "D")
    bounds = np.flatnonzero(sym[1:] != sym[:-1]) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(sym)]])
    out: list[dict] = []
    for i0, i1 in zip(starts, ends, strict=True):
        if len(set(owners[i0:i1])) < cfg.cluster_min_insiders:
            continue
        dd = d[i0:i1]
        days = np.unique(dd)
        lo = i0 + np.searchsorted(dd, days - win, side="right")
        hi = i0 + np.searchsorted(dd, days, side="right")
        last_emit = None
        for day, a, b in zip(days, lo, hi, strict=True):
            if last_emit is not None and day - last_emit < cool:
                continue
            uniq = set(owners[a:b])
            if len(uniq) < cfg.cluster_min_insiders:
                continue
            nm = sorted({n for n in names[a:b] if isinstance(n, str) and n})[:6]
            out.append({"symbol": sym[i0], "issuer_cik": cik[i0], "sector": sector[i0],
                        "signal_date": pd.Timestamp(day), "n_insiders": len(uniq),
                        "n_officers": len(set(owners[a:b][officer[a:b]])),
                        "value": float(val[a:b].sum()), "ceo_cfo": bool(ceo[a:b].any()),
                        "first_filing": pd.Timestamp(filing[a:b].min()),
                        "last_filing": pd.Timestamp(filing[a:b].max()), "names": "; ".join(nm)})
            last_emit = day
    return out


def build_events(t: pd.DataFrame, cfg: InsiderConfig) -> pd.DataFrame:
    """Esdeveniments per (símbol, data de senyal, tipus) a partir d'operacions preparades."""
    cols = ["symbol", "signal_date", "event", "issuer_cik", "sector", "n_insiders", "n_officers",
            "value", "ceo_cfo", "first_filing", "last_filing", "names"]
    e = t[t["eligible"]]
    if e.empty:
        return pd.DataFrame(columns=cols)
    frames = []
    buys = e[e["side"] == "buy"]
    if not buys.empty:
        g = buys.groupby(["symbol", "signal_date"], sort=False)
        daily = g.agg(issuer_cik=("issuer_cik", "first"), sector=("sector", "first"),
                      n_insiders=("owner_cik", "nunique"), value=("value", "sum"),
                      ceo_cfo=("ceo_cfo", "any"), first_filing=("filing_date", "min"),
                      last_filing=("filing_date", "max")).reset_index()
        # Noms (fins a 6 per dia): sense lambdes per grup, que són lentes amb molts grups
        nm = (buys[["symbol", "signal_date", "owner_name"]].dropna().drop_duplicates()
              .sort_values(["symbol", "signal_date", "owner_name"]))
        nm = nm.groupby(["symbol", "signal_date"], sort=False).head(6)
        names = nm.groupby(["symbol", "signal_date"], sort=False)["owner_name"].agg("; ".join).rename("names")
        daily = daily.merge(names, on=["symbol", "signal_date"], how="left")
        offi = (buys[buys["is_officer"].fillna(False)].groupby(["symbol", "signal_date"])["owner_cik"]
                .nunique().rename("n_officers"))
        daily = daily.merge(offi, on=["symbol", "signal_date"], how="left")
        daily["n_officers"] = daily["n_officers"].fillna(0).astype(int)
        frames.append(daily.assign(event="buy"))
        ceo = buys[buys["ceo_cfo"]]
        if not ceo.empty:
            frames.append(daily[daily.set_index(["symbol", "signal_date"]).index.isin(
                ceo.set_index(["symbol", "signal_date"]).index)].assign(event="ceo_cfo_buy"))
    for side, ev in (("buy", "cluster_buy"), ("sell", "cluster_sell")):
        s = e[e["side"] == side]
        # Accions amb prou insiders en algun moment (evita bucles inútils)
        many = s.groupby("symbol")["owner_cik"].nunique()
        s = s[s["symbol"].isin(many[many >= cfg.cluster_min_insiders].index)]
        rows = _clusters(s, cfg) if not s.empty else []
        if rows:
            frames.append(pd.DataFrame(rows).assign(event=ev))
    out = pd.concat(frames, ignore_index=True)
    out["value"] = out["value"].abs()
    return out[cols].sort_values(["signal_date", "symbol", "event"]).reset_index(drop=True)


def breadth(t: pd.DataFrame, sessions: pd.DatetimeIndex, cfg: InsiderConfig,
            members: pd.Series | None = None) -> pd.DataFrame:
    """Amplitud d'insiders per sessió: mercat ('market') i cada sector.

    Compta parelles (insider, empresa, mes) amb compres/vendes elegibles conegudes a la
    finestra mòbil de `breadth_window` sessions (un insider que compra en diversos trams
    compta una vegada). `members`: booleà alineat amb `t`, True si l'emissor era al
    S&P 500 a la data de senyal (només es compten aquests).
    """
    e = t[t["eligible"]].copy()
    if members is not None:
        e = e[members.reindex(e.index).fillna(False).astype(bool)]
    cols = ["scope", "date", "buyers", "sellers", "buy_ratio", "ratio_pct"]
    if e.empty or len(sessions) == 0:
        return pd.DataFrame(columns=cols)
    e = e[["signal_date", "side", "owner_cik", "issuer_cik", "sector"]].copy()
    e["ym"] = (e["signal_date"].dt.year * 12 + e["signal_date"].dt.month).astype("int32")
    key = e["owner_cik"].astype(str) + "|" + e["issuer_cik"].astype(str) + "|" + e["side"]
    e = e.assign(_k=key).sort_values("signal_date").drop_duplicates(["_k", "ym"])
    out = []
    scopes = [("market", e)] + [(sec, g) for sec, g in e.groupby("sector") if isinstance(sec, str)]
    for scope, g in scopes:
        daily = (g.groupby(["signal_date", "side"]).size().unstack("side")
                 .reindex(columns=["buy", "sell"]).reindex(sessions).fillna(0.0))
        roll = daily.rolling(cfg.breadth_window, min_periods=1).sum()
        tot = roll["buy"] + roll["sell"]
        ratio = (roll["buy"] / tot).where(tot >= 5)
        pct = ratio.expanding(min_periods=cfg.breadth_min_history).rank(pct=True)
        out.append(pd.DataFrame({"scope": scope, "date": sessions, "buyers": roll["buy"].values,
                                 "sellers": roll["sell"].values, "buy_ratio": ratio.values,
                                 "ratio_pct": pct.values}))
    return pd.concat(out, ignore_index=True)[cols]
