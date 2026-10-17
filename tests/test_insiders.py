"""Tests de la setmana 10: parseig del Form 4, ingesta, rutinaris, esdeveniments i avaluació."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import pytest

from finance.ingest import run_job
from finance.ingest.insiders import quarter_end
from finance.insiders import InsiderConfig
from finance.insiders import evaluation as ev
from finance.insiders.signals import breadth, build_events, prepare_trades
from finance.providers.base import InsiderProvider, ProviderError
from finance.providers.sec import (
    INSIDER_COLUMNS,
    QuarterFile,
    extract_ownership_xml,
    normalize_form345,
    normalize_symbol,
    parse_daily_index,
    parse_ownership_xml,
)

XML = """<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <issuer><issuerCik>0000320193</issuerCik><issuerName>Apple Inc.</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000001</rptOwnerCik><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>0</isDirector><isOfficer>1</isOfficer>
      <officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000002</rptOwnerCik><rptOwnerName>DOE TRUST</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isTenPercentOwner>1</isTenPercentOwner></reportingOwnerRelationship>
  </reportingOwner>
  <aff10b5One>0</aff10b5One>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-10-14</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.5</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction><value>5000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-10-14</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value>151</value><footnoteId id="F1"/></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction><value>4800</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>I</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes><footnote id="F1">Sold pursuant to a Rule 10b5-1 trading plan adopted May 1.</footnote></footnotes>
</ownershipDocument>"""

INDEX = """Description:           Daily Index of EDGAR Dissemination Feed by Form Type
Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------
4                APPLE INC                                                     320193      20261015    edgar/data/320193/0000320193-26-000001.txt
4                DOE JANE                                                      1           20261015    edgar/data/1/0000320193-26-000001.txt
SC 13G/A         SOME FUND                                                     999         20261015    edgar/data/999/0000999999-26-000002.txt
4                OTHER CORP                                                    777         20261015    edgar/data/777/0000777777-26-000003.txt
"""


def test_parse_xml_joint_owners_and_plan():
    parts = parse_ownership_xml(XML, "0000320193-26-000001", date(2026, 10, 15))
    df = normalize_form345(parts["sub"], parts["owners"], parts["trans"], parts["footnotes"], "xml")
    assert list(df.columns) == INSIDER_COLUMNS
    assert len(df) == 2  # una fila per operació, no per titular
    buy, sell = df.iloc[0], df.iloc[1]
    assert buy["n_owners"] == 2 and buy["is_officer"] and buy["is_ten_pct"]
    assert buy["officer_title"] == "Chief Executive Officer"
    assert buy["trans_code"] == "P" and buy["value"] == pytest.approx(150_500)
    assert not buy["plan_10b5_1"] and sell["plan_10b5_1"]  # la nota F1 només és a la venda
    assert buy["direct"] and not sell["direct"]
    assert buy["filing_date"] == date(2026, 10, 15) and buy["symbol"] == "AAPL"
    assert extract_ownership_xml(f"<SEC-DOCUMENT><XML>\n{XML[22:]}\n</XML>") is not None


def test_tsv_format_matches_xml():
    """El format TSV (dates '15-OCT-2026', notes per columna _FN) dona el mateix resultat."""
    sub = pd.DataFrame([{"ACCESSION_NUMBER": "A1", "FILING_DATE": "15-OCT-2026", "DOCUMENT_TYPE": "4",
                         "ISSUERCIK": "320193", "ISSUERNAME": "Apple", "ISSUERTRADINGSYMBOL": "aapl",
                         "REMARKS": None, "AFF10B5ONE": "true"}])
    owners = pd.DataFrame([{"ACCESSION_NUMBER": "A1", "RPTOWNERCIK": "1", "RPTOWNERNAME": "X",
                            "RPTOWNER_RELATIONSHIP": "Director,Officer", "RPTOWNER_TITLE": None}])
    trans = pd.DataFrame([{"ACCESSION_NUMBER": "A1", "NONDERIV_TRANS_SK": "9", "SECURITY_TITLE": "CS",
                           "TRANS_DATE": "14-OCT-2026", "TRANS_CODE": "s", "TRANS_SHARES": "10",
                           "TRANS_PRICEPERSHARE": "2", "TRANS_PRICEPERSHARE_FN": None,
                           "TRANS_ACQUIRED_DISP_CD": "D", "SHRS_OWND_FOLWNG_TRANS": "0",
                           "DIRECT_INDIRECT_OWNERSHIP": "D"}])
    df = normalize_form345(sub, owners, trans, pd.DataFrame(columns=["ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"]), "2026q4")
    r = df.iloc[0]
    assert r["issuer_cik"] == "0000320193" and r["symbol"] == "AAPL" and r["trans_code"] == "S"
    assert r["is_director"] and r["is_officer"] and r["plan_10b5_1"] and r["plan_flag"]
    assert r["trans_date"] == date(2026, 10, 14)


def test_symbols_and_index():
    s = normalize_symbol(pd.Series(["brk.b", "NYSE: KO", "NONE", " msft ", None]))
    assert s.tolist()[:2] == ["BRK-B", "KO"] and pd.isna(s.iloc[2]) and s.iloc[3] == "MSFT"
    idx = parse_daily_index(INDEX)
    assert len(idx) == 4 and idx["form_type"].tolist()[2] == "SC 13G/A"
    assert idx["cik"].iloc[0] == "0000320193"
    assert idx["accession_number"].iloc[0] == "0000320193-26-000001"
    assert quarter_end(2026, 3) == date(2026, 9, 30) and quarter_end(2025, 4) == date(2025, 12, 31)


# ---------------------------------------------------------------------------------------
# Ingesta amb proveïdor fals
# ---------------------------------------------------------------------------------------

def _row(acc, line, filing, *, code="P", owner="1", cik="0000320193", sym="AAPL", value=50_000.0,
         officer=True, title="CFO", plan=False, source="2026q3", trans=None):
    return {"accession_number": acc, "line_no": line, "filing_date": filing, "form_type": "4",
            "issuer_cik": cik, "issuer_name": sym, "symbol": sym, "owner_cik": owner,
            "owner_name": f"OWNER {owner}", "n_owners": 1, "is_director": not officer,
            "is_officer": officer, "is_ten_pct": False, "is_other": False, "officer_title": title,
            "security_title": "Common", "trans_date": trans or filing, "trans_code": code,
            "acq_disp": "A" if code == "P" else "D", "shares": value / 100, "price": 100.0,
            "value": value, "shares_after": 1.0, "direct": True, "plan_10b5_1": plan,
            "plan_flag": pd.NA, "source_file": source}


@dataclass
class FakeInsiders(InsiderProvider):
    name: str = "fake_sec"
    quarters: dict[str, pd.DataFrame] = field(default_factory=dict)
    days: dict[date, pd.DataFrame] = field(default_factory=dict)
    filings: dict[str, pd.DataFrame] = field(default_factory=dict)
    fetched: list[str] = field(default_factory=list)

    def list_quarters(self):
        return [QuarterFile(int(k[:4]), int(k[-1]), k) for k in sorted(self.quarters)]

    def fetch_quarter(self, q):
        return pd.DataFrame(self.quarters[q.key])

    def daily_index(self, day):
        if day not in self.days:
            raise ProviderError("idx: no existeix")
        return self.days[day]

    def fetch_filing(self, path, accession, filing_date):
        self.fetched.append(accession)
        return pd.DataFrame(self.filings[accession])


def test_insiders_job_quarter_supersedes_daily(ctx, providers, clock):
    run_job(ctx, "universe")  # CIK del S&P 500 (fals: AAPL i BRK.B)
    q3 = [_row("Q-1", 0, date(2026, 8, 3))]
    fake = FakeInsiders(quarters={"2026q3": q3})
    idx = pd.DataFrame([
        {"form_type": "4", "name": "APPLE", "cik": "0000320193", "date_filed": "20261013",
         "path": "edgar/data/320193/X-1.txt", "accession_number": "X-1"},
        {"form_type": "4", "name": "OTHER", "cik": "0000000777", "date_filed": "20261013",
         "path": "edgar/data/777/X-2.txt", "accession_number": "X-2"},
    ])
    fake.days = {date(2026, 10, 13): idx}
    fake.filings = {"X-1": [_row("X-1", 0, date(2026, 10, 13), source="xml"),
                            _row("X-1", 1, date(2026, 10, 13), code="S", source="xml")]}
    providers.insiders = fake
    rep = run_job(ctx, "insiders")
    assert rep.status == "success", rep.errors
    assert fake.fetched == ["X-1"]  # només emissors de l'univers
    df = ctx.query("SELECT * FROM insider_transactions ORDER BY accession_number, line_no")
    assert df["accession_number"].tolist() == ["Q-1", "X-1", "X-1"]
    # Segona execució: no torna a baixar presentacions conegudes
    run_job(ctx, "insiders")
    assert fake.fetched == ["X-1"]
    # Surt el trimestre amb la mateixa presentació (una sola línia): substitueix l'XML sencer
    fake.quarters["2026q4"] = [_row("X-1", 0, date(2026, 10, 13), source="2026q4")]
    run_job(ctx, "insiders")
    df = ctx.query("SELECT * FROM insider_transactions WHERE accession_number = 'X-1'")
    assert len(df) == 1 and df["source_file"].iloc[0] == "2026q4"


def test_insiders_job_skips_without_user_agent(ctx):
    rep = run_job(ctx, "insiders")  # Providers per defecte del test: sense insiders
    assert rep.status == "skipped"


# ---------------------------------------------------------------------------------------
# Senyals
# ---------------------------------------------------------------------------------------

def _prep(rows, cfg, sessions, cik=None, intervals=None):
    """Executa la preparació (SQL) sobre operacions en memòria, com fa el job sobre el lake."""
    import duckdb
    cols = ["issuer_cik", "symbol", "owner_cik", "owner_name", "is_officer", "is_director",
            "officer_title", "trans_code", "form_type", "plan_10b5_1", "filing_date", "trans_date",
            "value", "price"]
    con = duckdb.connect()
    con.register("trades", pd.DataFrame(rows)[cols])
    if cik is not None:
        con.register("cikmap", cik)
    if intervals is not None:
        con.register("ivals", intervals)
    try:
        return prepare_trades(con, cfg, sessions, source="trades",
                              cik_map="cikmap" if cik is not None else None,
                              intervals="ivals" if intervals is not None else None)
    finally:
        con.close()


def _sessions(start="2018-01-02", end="2026-12-31"):
    return pd.bdate_range(start, end)


def test_routine_classification():
    cfg = InsiderConfig()
    rows = []
    for y in (2019, 2020, 2021, 2022):  # el titular 1 ven cada abril: el 2022 és rutinari
        rows.append(_row(f"A{y}", 0, date(y, 4, 10), owner="1", code="S"))
    rows.append(_row("B", 0, date(2022, 6, 10), owner="1", code="P"))  # juny: oportunista
    rows.append(_row("C", 0, date(2022, 4, 10), owner="2", code="P"))  # sense historial: nou
    t = _prep(rows, cfg, _sessions()).sort_values(["filing_date", "owner_cik"])
    assert t["cls"].tolist() == ["new", "new", "new", "routine", "new", "opportunistic"]
    assert not t.loc[t["cls"] == "routine", "eligible"].item()


def test_signal_date_is_next_session():
    cfg = InsiderConfig()
    rows = [_row(f"x{i}", 0, d, owner=str(i)) for i, d in
            enumerate([date(2026, 10, 15), date(2026, 10, 16), date(2026, 10, 17)])]
    t = _prep(rows, cfg, pd.DatetimeIndex(pd.to_datetime(["2026-10-15", "2026-10-16", "2026-10-19"])))
    # Presentat dijous -> divendres; divendres o dissabte -> dilluns
    got = t.sort_values("filing_date")["signal_date"].dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2026-10-16", "2026-10-19", "2026-10-19"]


def test_cluster_buy_and_filters():
    cfg = InsiderConfig()
    rows = [
        _row("a", 0, date(2026, 3, 2), owner="1", title="Chief Executive Officer"),
        _row("b", 0, date(2026, 3, 10), owner="2", officer=False, title=None),
        _row("c", 0, date(2026, 3, 20), owner="3", title="VP"),
        _row("d", 0, date(2026, 3, 25), owner="4", title="VP"),  # dins del període de calma
        _row("e", 0, date(2026, 3, 21), owner="5", plan=True),  # pla 10b5-1: fora
        _row("f", 0, date(2026, 3, 21), owner="6", value=500.0),  # massa petita: fora
        _row("g", 0, date(2026, 5, 4), owner="7", code="S", title="VP"),
    ]
    t = _prep(rows, cfg, _sessions())
    assert len(t) == 6  # la petita ni tan sols és candidata
    assert not t.loc[t["owner_cik"] == 5, "eligible"].item()
    assert t.loc[t["owner_cik"] == 1, "ceo_cfo"].item() and not t.loc[t["owner_cik"] == 3, "ceo_cfo"].item()
    ev_ = build_events(t, cfg)
    cl = ev_[ev_["event"] == "cluster_buy"]
    assert len(cl) == 1  # una sola emissió malgrat el 4t comprador (calma de 90 dies)
    assert cl["signal_date"].iloc[0] == pd.Timestamp("2026-03-23")  # sessió després del 20-mar
    assert cl["n_insiders"].iloc[0] == 3 and cl["ceo_cfo"].iloc[0]
    assert set(ev_.loc[ev_["event"] == "ceo_cfo_buy", "signal_date"]) == {pd.Timestamp("2026-03-03")}
    assert (ev_["event"] == "buy").sum() == 4 and "cluster_sell" not in set(ev_["event"])
    # Ablació: sense filtres, la compra del pla (dissabte -> dilluns 23) s'afegeix a aquell dia
    unf = ev.unfiltered_buys(t, cfg).set_index("signal_date")
    assert unf.loc[pd.Timestamp("2026-03-23"), "n_insiders"] == 2


def test_cik_maps_to_current_symbol_and_membership():
    cfg = InsiderConfig()
    cik = pd.DataFrame({"cik": ["0001326801"], "symbol": ["META"], "sector": ["Communication Services"]})
    iv = pd.DataFrame({"universe": ["sp500"], "symbol": ["META"], "start_date": [date(2013, 12, 23)],
                       "end_date": [None]})
    rows = [_row("a", 0, date(2026, 3, 2), sym="FB", cik="0001326801"),
            _row("b", 0, date(2026, 3, 2), sym="ZZZ", cik="0000000009", owner="2")]
    t = _prep(rows, cfg, _sessions(), cik=cik, intervals=iv).sort_values("symbol")
    assert t["symbol"].tolist() == ["META", "ZZZ"]
    assert t["sector"].tolist()[0] == "Communication Services"
    assert t["member"].tolist() == [True, False]


def test_breadth_counts_window():
    cfg = InsiderConfig(breadth_window=5, breadth_min_history=1)
    rows = [_row(f"b{i}", 0, date(2026, 3, 2), owner=str(i)) for i in range(4)]
    rows += [_row("s1", 0, date(2026, 3, 2), owner="9", code="S")]
    t = _prep(rows, cfg, _sessions())
    t["sector"] = "Information Technology"
    b = breadth(t, _sessions("2026-03-01", "2026-03-31"), cfg)
    m = b[b["scope"] == "market"].set_index("date")
    assert m.loc["2026-03-03", "buyers"] == 4 and m.loc["2026-03-03", "sellers"] == 1
    assert m.loc["2026-03-03", "buy_ratio"] == pytest.approx(0.8)
    assert m.loc["2026-03-10", "buyers"] == 0  # fora de la finestra de 5 sessions
    assert set(b["scope"]) == {"market", "Information Technology"}


# ---------------------------------------------------------------------------------------
# Avaluació
# ---------------------------------------------------------------------------------------

def test_event_returns_and_calendar_portfolio():
    idx = pd.bdate_range("2026-01-01", periods=40)
    px = pd.DataFrame({"SPY": 100 * 1.001 ** np.arange(40), "AAA": 100 * 1.003 ** np.arange(40),
                       "XLK": 100 * 1.002 ** np.arange(40)}, index=idx)
    events = pd.DataFrame({"symbol": ["AAA"], "signal_date": [idx[5]], "event": ["cluster_buy"],
                           "sector": ["Information Technology"]})
    r = ev.event_returns(events, px, {"Information Technology": "XLK"}, (5,))
    assert r["ret_5"].item() == pytest.approx(1.003 ** 5 - 1)
    assert r["ex_spy_5"].item() == pytest.approx(1.003 ** 5 - 1.001 ** 5)
    assert r["ex_sec_5"].item() == pytest.approx(1.003 ** 5 - 1.002 ** 5)
    cp = ev.calendar_portfolio(events, px, hold=3)
    # Entra al tancament de la sessió 5: guanya les sessions 6, 7 i 8, i res més
    assert cp["n"].tolist()[5:10] == [0, 1, 1, 1, 0]
    assert cp["port"].iloc[6] == pytest.approx(0.003)
    t, months = ev.clustered_t(pd.Series([0.01, 0.02, 0.03, 0.01]),
                               pd.Series(pd.to_datetime(["2020-01-05", "2020-02-05", "2020-03-05", "2020-04-05"])))
    assert months == 4 and t > 0


def test_daily_report_insider_context(ctx, monkeypatch):
    """La secció d'insiders de l'informe diari: esdeveniments recents i amplitud amb l'ETF."""
    from types import SimpleNamespace

    from finance.paper.report import _insiders_html, insider_context
    from finance.storage.schemas import INSIDER_BREADTH, INSIDER_EVENTS

    monkeypatch.setattr(ev, "sector_etfs", lambda _ctx: {"Information Technology": "XLK"})
    view = SimpleNamespace(holdings=pd.DataFrame({"actiu": ["XLK"]}))
    assert insider_context(ctx, view, pd.Timestamp("2026-10-15")) is None  # sense dades: sense secció
    ev_ = pd.DataFrame([{"signal_set": "ins_v1", "symbol": "AAPL", "signal_date": date(2026, 10, 14),
                         "event": "cluster_buy", "issuer_cik": "0000320193", "sector": "Information Technology",
                         "in_sp500": True, "n_insiders": 3, "n_officers": 2, "value": 2e6, "value_adv": 0.01,
                         "ceo_cfo": True, "names": "A; B; C"}])
    ctx.lake.write(INSIDER_EVENTS, ev_, source="t", run_id="t")
    br = pd.DataFrame([{"signal_set": "ins_v1", "scope": s, "date": date(2026, 10, 15), "buyers": 5.0,
                        "sellers": 20.0, "buy_ratio": 0.2, "ratio_pct": 0.5}
                       for s in ("market", "Information Technology")])
    ctx.lake.write(INSIDER_BREADTH, br, source="t", run_id="t")
    ins = insider_context(ctx, view, pd.Timestamp("2026-10-15"))
    assert ins is not None and ins["events"]["symbol"].tolist() == ["AAPL"]
    it = ins["breadth"].set_index("scope").loc["Information Technology"]
    assert it["etf"] == "XLK" and bool(it["held"])
    html = _insiders_html(SimpleNamespace(insiders=ins))
    assert "compra en grup" in html and "AAPL" in html
