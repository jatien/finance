"""SEC EDGAR: operacions d'insiders (formularis 3/4/5 de la secció 16).

Dues vies que acaben al mateix format normalitzat (`normalize_form345`):

- **Datasets trimestrals** ("Insider Transactions Data Sets", DERA): un zip per trimestre
  amb TSVs (SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, FOOTNOTES...). Es publiquen pocs
  dies després de tancar el trimestre i cobreixen des del 2003. Base de l'històric.
- **Índex diari + XML**: per als dies posteriors a l'últim trimestre publicat. L'índex
  diari (`form.YYYYMMDD.idx`) llista les presentacions; de cada Form 4 es baixa el fitxer
  de la presentació i se n'extreu l'`ownershipDocument`.

La SEC exigeix un User-Agent identificable ("Nom correu") i un màxim de 10 peticions/s.
Font gratuïta i pública: és la mateixa que fan servir serveis com Kestrel Terminal,
OpenInsider o els APIs comercials d'insiders.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

from finance.providers.base import InsiderProvider, ProviderError, TransientProviderError
from finance.providers.ratelimit import RateLimiter, with_retries

log = logging.getLogger(__name__)

SEC_BASE = "https://www.sec.gov"
DATASETS_PAGE = f"{SEC_BASE}/data-research/sec-markets-data/insider-transactions-data-sets"
DAILY_INDEX = SEC_BASE + "/Archives/edgar/daily-index/{y}/QTR{q}/form.{d:%Y%m%d}.idx"
ARCHIVES = SEC_BASE + "/Archives/"

# Formularis amb operacions (el 3 és la declaració inicial de tinença: no en té)
TRANSACTION_FORMS = ("4", "4/A", "5", "5/A")
_PLAN_RE = re.compile(r"10b5[\s\-‐–]*1", re.IGNORECASE)
_QUARTER_RE = re.compile(r"(\d{4})q([1-4])_form345\.zip", re.IGNORECASE)

# Columnes normalitzades (sense les d'auditoria ni la partició)
INSIDER_COLUMNS = [
    "accession_number", "line_no", "filing_date", "form_type", "issuer_cik", "issuer_name",
    "symbol", "owner_cik", "owner_name", "n_owners", "is_director", "is_officer",
    "is_ten_pct", "is_other", "officer_title", "security_title", "trans_date", "trans_code",
    "acq_disp", "shares", "price", "value", "shares_after", "direct", "plan_10b5_1",
    "plan_flag", "source_file",
]


# ---------------------------------------------------------------------------------------
# Normalització comuna (TSV trimestral i XML diari)
# ---------------------------------------------------------------------------------------

def _dates(s: pd.Series) -> pd.Series:
    """Accepta '28-JUL-2026' (TSV) i '2026-07-28' (XML). Valors invàlids -> NaT."""
    s = s.astype("string").str.strip()
    a = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    b = pd.to_datetime(s.str[:10], format="%Y-%m-%d", errors="coerce")
    return a.fillna(b).dt.date


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _flag(s: pd.Series) -> pd.Series:
    """'1'/'true'/'0'/'false'/buit -> booleà amb nuls (pd.NA si no s'informa)."""
    m = s.astype("string").str.strip().str.lower().map(
        {"1": True, "true": True, "0": False, "false": False})
    return m.astype("boolean")


def normalize_symbol(s: pd.Series) -> pd.Series:
    """Símbol declarat a la presentació -> convenció Yahoo (BRK.B -> BRK-B).

    Elimina prefixos de borsa ('NYSE: X'), espais i valors buits ('NONE', 'N/A').
    """
    s = s.astype("string").str.upper().str.strip()
    s = s.str.replace(r"^.*[:]\s*", "", regex=True).str.split().str[-1]
    s = s.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    return s.where(~s.isin(["NONE", "N-A", "NA", "", "-"]))


def normalize_form345(sub: pd.DataFrame, owners: pd.DataFrame, trans: pd.DataFrame,
                      footnotes: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Uneix les taules del Form 3/4/5 en una fila per operació no derivada.

    - Presentacions conjuntes (diversos titulars): una sola fila amb el primer titular,
      `n_owners` i els rols combinats (OR). Evita comptar la mateixa operació N vegades.
    - `plan_10b5_1`: operació dins d'un pla 10b5-1 (preprogramada, poc informativa). Es
      marca si alguna nota al peu de la línia o de la presentació l'esmenta; `plan_flag` és
      la casella AFF10B5ONE (només existeix des del 2023 i indica que el pla l'empara).
    - `line_no`: ordre dins la presentació. No és estable entre el TSV i l'XML; per això
      la vista del lake substitueix la presentació sencera (`snapshot_by` per accession).
    """
    if trans.empty or sub.empty:
        return pd.DataFrame(columns=INSIDER_COLUMNS)
    sub = sub.copy()
    sub["DOCUMENT_TYPE"] = sub["DOCUMENT_TYPE"].astype("string").str.strip()
    sub = sub[sub["DOCUMENT_TYPE"].isin(TRANSACTION_FORMS)]

    # Titulars: rols combinats per presentació
    o = owners.copy()
    rel = o["RPTOWNER_RELATIONSHIP"].astype("string").fillna("")
    for col, key in (("is_director", "Director"), ("is_officer", "Officer"),
                     ("is_ten_pct", "TenPercentOwner"), ("is_other", "Other")):
        o[col] = rel.str.contains(key, regex=False)
    o["RPTOWNER_TITLE"] = o["RPTOWNER_TITLE"].astype("string").str.strip()
    og = o.groupby("ACCESSION_NUMBER", sort=False).agg(
        owner_cik=("RPTOWNERCIK", "first"), owner_name=("RPTOWNERNAME", "first"),
        n_owners=("RPTOWNERCIK", "size"), is_director=("is_director", "any"),
        is_officer=("is_officer", "any"), is_ten_pct=("is_ten_pct", "any"),
        is_other=("is_other", "any"),
        officer_title=("RPTOWNER_TITLE", "first"),  # "first" salta els nuls
    ).reset_index()

    # Notes al peu que esmenten un pla 10b5-1 (per presentació i identificador)
    plan_notes: set[tuple[str, str]] = set()
    if not footnotes.empty:
        fn = footnotes[footnotes["FOOTNOTE_TXT"].astype("string").str.contains(_PLAN_RE, na=False)]
        plan_notes = set(zip(fn["ACCESSION_NUMBER"], fn["FOOTNOTE_ID"].astype("string").str.strip(), strict=True))
    remarks_plan = set(sub.loc[sub.get("REMARKS", pd.Series(dtype="string")).astype("string")
                               .str.contains(_PLAN_RE, na=False), "ACCESSION_NUMBER"])

    t = trans[trans["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"])].copy()
    if t.empty:
        return pd.DataFrame(columns=INSIDER_COLUMNS)
    if "_FNS" not in t:
        fn_cols = [c for c in t.columns if c.endswith("_FN")]
        fns = pd.Series("", index=t.index, dtype="string")
        for c in fn_cols:  # concatenació vectoritzada (un apply per fila és ~100x més lent)
            fns = fns + "," + t[c].astype("string").fillna("")
        t["_FNS"] = fns
    ids = t["_FNS"].astype("string").fillna("").str.findall(r"F\d+").explode().dropna()
    note_hit = pd.Series([(a, i) in plan_notes for a, i in
                          zip(t.loc[ids.index, "ACCESSION_NUMBER"], ids, strict=True)],
                         index=ids.index, dtype=bool)
    hit = note_hit.groupby(level=0).any().reindex(t.index, fill_value=False)
    t["plan_10b5_1"] = hit.astype(bool) | t["ACCESSION_NUMBER"].isin(remarks_plan)
    if "NONDERIV_TRANS_SK" in t:
        t["_ord"] = pd.to_numeric(t["NONDERIV_TRANS_SK"], errors="coerce")
    else:
        t["_ord"] = np.arange(len(t))
    t = t.sort_values(["ACCESSION_NUMBER", "_ord"], kind="stable")
    t["line_no"] = t.groupby("ACCESSION_NUMBER").cumcount().astype("int32")

    df = t.merge(sub, on="ACCESSION_NUMBER", how="inner").merge(og, on="ACCESSION_NUMBER", how="left")
    shares = _num(df["TRANS_SHARES"])
    price = _num(df["TRANS_PRICEPERSHARE"])
    flag = _flag(df["AFF10B5ONE"]) if "AFF10B5ONE" in df else pd.Series(pd.NA, index=df.index, dtype="boolean")
    out = pd.DataFrame({
        "accession_number": df["ACCESSION_NUMBER"].astype("string").str.strip(),
        "line_no": df["line_no"],
        "filing_date": _dates(df["FILING_DATE"]),
        "form_type": df["DOCUMENT_TYPE"],
        "issuer_cik": df["ISSUERCIK"].astype("string").str.strip().str.zfill(10),
        "issuer_name": df["ISSUERNAME"].astype("string").str.strip(),
        "symbol": normalize_symbol(df["ISSUERTRADINGSYMBOL"]),
        "owner_cik": df["owner_cik"].astype("string").str.strip().str.zfill(10),
        "owner_name": df["owner_name"].astype("string").str.strip(),
        "n_owners": df["n_owners"].fillna(1).astype("int32"),
        "is_director": df["is_director"].fillna(False).astype(bool),
        "is_officer": df["is_officer"].fillna(False).astype(bool),
        "is_ten_pct": df["is_ten_pct"].fillna(False).astype(bool),
        "is_other": df["is_other"].fillna(False).astype(bool),
        "officer_title": df["officer_title"].astype("string"),
        "security_title": df["SECURITY_TITLE"].astype("string").str.strip(),
        "trans_date": _dates(df["TRANS_DATE"]),
        "trans_code": df["TRANS_CODE"].astype("string").str.strip().str.upper(),
        "acq_disp": df["TRANS_ACQUIRED_DISP_CD"].astype("string").str.strip().str.upper(),
        "shares": shares,
        "price": price,
        "value": shares * price,
        "shares_after": _num(df["SHRS_OWND_FOLWNG_TRANS"]),
        "direct": df["DIRECT_INDIRECT_OWNERSHIP"].astype("string").str.strip().str.upper().eq("D"),
        "plan_10b5_1": df["plan_10b5_1"].astype(bool) | flag.fillna(False).astype(bool),
        "plan_flag": flag,
        "source_file": source_file,
    })
    return out.dropna(subset=["filing_date", "accession_number"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# XML (ownershipDocument)
# ---------------------------------------------------------------------------------------

def _txt(el: ET.Element | None, path: str) -> str | None:
    if el is None:
        return None
    node = el.find(path)
    if node is None:
        return None
    v = node.find("value")
    s = (v.text if v is not None else node.text) or ""
    s = s.strip()
    return s or None


def parse_ownership_xml(xml: str, accession: str, filing_date: date) -> dict[str, pd.DataFrame]:
    """Converteix un `ownershipDocument` a les quatre taules amb els noms de columna del TSV."""
    root = ET.fromstring(xml)
    issuer = root.find("issuer")
    sub = pd.DataFrame([{
        "ACCESSION_NUMBER": accession, "FILING_DATE": filing_date.isoformat(),
        "DOCUMENT_TYPE": _txt(root, "documentType"),
        "ISSUERCIK": _txt(issuer, "issuerCik"), "ISSUERNAME": _txt(issuer, "issuerName"),
        "ISSUERTRADINGSYMBOL": _txt(issuer, "issuerTradingSymbol"),
        "REMARKS": _txt(root, "remarks"), "AFF10B5ONE": _txt(root, "aff10b5One"),
    }])
    owners = []
    for ro in root.findall("reportingOwner"):
        rel = ro.find("reportingOwnerRelationship")
        roles = [name for tag, name in (("isDirector", "Director"), ("isOfficer", "Officer"),
                                        ("isTenPercentOwner", "TenPercentOwner"), ("isOther", "Other"))
                 if (_txt(rel, tag) or "").lower() in ("1", "true")]
        owners.append({
            "ACCESSION_NUMBER": accession,
            "RPTOWNERCIK": _txt(ro, "reportingOwnerId/rptOwnerCik"),
            "RPTOWNERNAME": _txt(ro, "reportingOwnerId/rptOwnerName"),
            "RPTOWNER_RELATIONSHIP": ",".join(roles),
            "RPTOWNER_TITLE": _txt(rel, "officerTitle"),
        })
    trans = []
    for tr in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        fns = ",".join(sorted({f.get("id", "") for f in tr.iter("footnoteId")}))
        trans.append({
            "ACCESSION_NUMBER": accession,
            "SECURITY_TITLE": _txt(tr, "securityTitle"),
            "TRANS_DATE": _txt(tr, "transactionDate"),
            "TRANS_CODE": _txt(tr, "transactionCoding/transactionCode"),
            "TRANS_SHARES": _txt(tr, "transactionAmounts/transactionShares"),
            "TRANS_PRICEPERSHARE": _txt(tr, "transactionAmounts/transactionPricePerShare"),
            "TRANS_ACQUIRED_DISP_CD": _txt(tr, "transactionAmounts/transactionAcquiredDisposedCode"),
            "SHRS_OWND_FOLWNG_TRANS": _txt(tr, "postTransactionAmounts/sharesOwnedFollowingTransaction"),
            "DIRECT_INDIRECT_OWNERSHIP": _txt(tr, "ownershipNature/directOrIndirectOwnership"),
            "_FNS": fns,
        })
    notes = [{"ACCESSION_NUMBER": accession, "FOOTNOTE_ID": f.get("id"),
              "FOOTNOTE_TXT": "".join(f.itertext()).strip()}
             for f in root.findall("footnotes/footnote")]
    return {
        "sub": sub,
        "owners": pd.DataFrame(owners, columns=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                                "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]),
        "trans": pd.DataFrame(trans),
        "footnotes": pd.DataFrame(notes, columns=["ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"]),
    }


def extract_ownership_xml(submission_text: str) -> str | None:
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", submission_text, re.DOTALL)
    return m.group(0) if m else None


def parse_daily_index(text: str) -> pd.DataFrame:
    """Índex diari per tipus de formulari (amplada fixa) -> form_type, name, cik, date, path."""
    rows = []
    started = False
    for line in text.splitlines():
        if line.startswith("-----"):
            started = True
            continue
        if not started or not line.strip():
            continue
        parts = line.rstrip().rsplit(None, 3)
        if len(parts) < 4:
            continue
        head, cik, filed, path = parts
        form = head[:17].strip() or head.split()[0]
        rows.append({"form_type": form, "name": head[17:].strip(), "cik": cik.zfill(10),
                     "date_filed": filed, "path": path})
    df = pd.DataFrame(rows, columns=["form_type", "name", "cik", "date_filed", "path"])
    df["accession_number"] = df["path"].str.extract(r"(\d{10}-\d{2}-\d{6})")[0]
    return df


# ---------------------------------------------------------------------------------------
# Proveïdor
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QuarterFile:
    year: int
    quarter: int
    url: str

    @property
    def key(self) -> str:
        return f"{self.year}q{self.quarter}"


class SecEdgarInsiders(InsiderProvider):
    name = "sec_edgar"

    def __init__(self, user_agent: str, rate_per_sec: float = 5.0, session: requests.Session | None = None):
        if not user_agent or "@" not in user_agent:
            raise ProviderError("la SEC exigeix un User-Agent amb correu: defineix FIN_SEC_USER_AGENT")
        self.limiter = RateLimiter(min(rate_per_sec, 9.0), burst=2)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    def _get(self, url: str, timeout: float = 60) -> requests.Response:
        def once() -> requests.Response:
            self.limiter.acquire()
            try:
                r = self.session.get(url, timeout=timeout)
            except requests.RequestException as e:
                raise TransientProviderError(f"{url}: {e}") from e
            if r.status_code in (429, 500, 502, 503, 504):
                raise TransientProviderError(f"{url}: HTTP {r.status_code}")
            if r.status_code == 404:
                raise ProviderError(f"{url}: no existeix")
            if r.status_code == 403:
                raise ProviderError(f"{url}: 403 (User-Agent rebutjat o límit de la SEC)")
            r.raise_for_status()
            return r
        return with_retries(once, attempts=5, max_wait=30)()

    def list_quarters(self) -> list[QuarterFile]:
        html = self._get(DATASETS_PAGE).text
        found: dict[str, QuarterFile] = {}
        for href in re.findall(r'href="([^"]+_form345\.zip)"', html, flags=re.IGNORECASE):
            m = _QUARTER_RE.search(href)
            if not m:
                continue
            url = href if href.startswith("http") else SEC_BASE + href
            q = QuarterFile(int(m.group(1)), int(m.group(2)), url)
            found[q.key] = q
        if not found:
            raise ProviderError("no s'ha trobat cap dataset trimestral a la pàgina de la SEC")
        return sorted(found.values(), key=lambda q: (q.year, q.quarter))

    def fetch_quarter(self, q: QuarterFile) -> pd.DataFrame:
        content = self._get(q.url, timeout=300).content
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = {n.split("/")[-1].upper(): n for n in zf.namelist()}

            def tsv(name: str, cols: list[str] | None = None) -> pd.DataFrame:
                key = f"{name}.TSV"
                if key not in names:
                    return pd.DataFrame(columns=cols or [])
                with zf.open(names[key]) as fh:
                    return pd.read_csv(fh, sep="\t", dtype=str, quoting=3, usecols=cols,
                                       on_bad_lines="skip", encoding_errors="replace",
                                       keep_default_na=False, na_values=[""])

            sub = tsv("SUBMISSION")
            owners = tsv("REPORTINGOWNER", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                            "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
            trans = tsv("NONDERIV_TRANS")
            notes = tsv("FOOTNOTES", ["ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"])
        return normalize_form345(sub, owners, trans, notes, source_file=q.key)

    def index_days(self, year: int, quarter: int) -> set[date]:
        """Dies amb índex diari publicat (la SEC respon 403, no 404, als dies sense fitxer)."""
        js = self._get(f"{SEC_BASE}/Archives/edgar/daily-index/{year}/QTR{quarter}/index.json").json()
        out = set()
        for it in js.get("directory", {}).get("item", []):
            m = re.match(r"form\.(\d{8})\.idx", it.get("name", ""))
            if m:
                out.add(pd.Timestamp(m.group(1)).date())
        return out

    def daily_index(self, day: date) -> pd.DataFrame:
        q = (day.month - 1) // 3 + 1
        text = self._get(DAILY_INDEX.format(y=day.year, q=q, d=day)).text
        return parse_daily_index(text)

    def fetch_filing(self, path: str, accession: str, filing_date: date) -> pd.DataFrame:
        text = self._get(ARCHIVES + path.lstrip("/")).text
        xml = extract_ownership_xml(text)
        if xml is None:
            raise ProviderError(f"{accession}: sense ownershipDocument")
        try:
            parts = parse_ownership_xml(xml, accession, filing_date)
        except ET.ParseError as e:
            raise ProviderError(f"{accession}: XML invàlid ({e})") from e
        return normalize_form345(parts["sub"], parts["owners"], parts["trans"], parts["footnotes"],
                                 source_file="xml")


class DisabledInsiders(InsiderProvider):
    """Sense User-Agent configurat: el job ho registra com a omès en lloc de fallar."""

    name = "disabled"

    def __init__(self, reason: str):
        self.reason = reason

    def list_quarters(self):
        raise ProviderError(self.reason)

    def fetch_quarter(self, q):
        raise ProviderError(self.reason)

    def index_days(self, year, quarter):
        raise ProviderError(self.reason)

    def daily_index(self, day):
        raise ProviderError(self.reason)

    def fetch_filing(self, path, accession, filing_date):
        raise ProviderError(self.reason)
