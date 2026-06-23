"""NIST WebBook lookup utilities."""
import re
import requests

from ivette.util import http
from ivette.util.patterns import THERMO_REGEX

NIST_BASE = "https://webbook.nist.gov/cgi/cbook.cgi"

NIST_KEYWORDS = [
    "Thermochemistry", "Thermodynamic", "Enthalpy", "Entropy",
    "Gibbs", "Heat of", "Heat Capacity", "Heat of Formation",
]

NIST_SEARCH_LINK_RE = re.compile(r'href="(/cgi/cbook.cgi\?ID=[^"&]+&Units=SI)"', re.IGNORECASE)
TITLE_RE = re.compile(r"<title>([^<]+)</title>", re.IGNORECASE)


def parse_html_title(html: str) -> str:
    m = TITLE_RE.search(html)
    return m.group(1).strip() if m else ""


def is_nist_search_results(html: str) -> bool:
    return "Search Results" in html and "/cgi/cbook.cgi?ID=" in html


def extract_nist_search_link(html: str) -> str | None:
    m = NIST_SEARCH_LINK_RE.search(html)
    return m.group(1) if m else None


def nist_page_keywords(html: str) -> bool:
    text = html.lower()
    return any(kw.lower() in text for kw in NIST_KEYWORDS)


def nist_page_snippet(html: str) -> str:
    match = THERMO_REGEX.search(html)
    return match.group(1).strip()[:250] if match else ""


def query_nist(params: dict, timeout: int = http.DEFAULT_TIMEOUT) -> requests.Response:
    r = http.get(NIST_BASE, params=params, timeout=timeout)
    r.raise_for_status()
    return r


def follow_nist_search(name_or_id: str, query_type: str,
                       timeout: int = http.DEFAULT_TIMEOUT) -> requests.Response:
    params = {query_type: name_or_id, "Units": "SI"}
    r = query_nist(params, timeout=timeout)
    if is_nist_search_results(r.text):
        link = extract_nist_search_link(r.text)
        if link:
            r = http.get(f"https://webbook.nist.gov{link}", timeout=timeout)
            r.raise_for_status()
    return r


def check_nist_entry(
    inchikey: str,
    name: str = None,
    cas: str = None,
    inchi: str = None,
) -> dict:
    record = {
        "NIST_Found": False,
        "NIST_URL": "",
        "NIST_Query_Method": "",
        "NIST_Title": "",
        "NIST_Notes": "",
        "NIST_Snippet": "",
        "NIST_HTML": "",
    }
    query_order = []
    if inchikey:
        query_order.append(("Type", inchikey, "InChIKey"))
    if cas:
        query_order.append(("Name", cas, "CAS"))
    if name:
        query_order.append(("Name", name, "Name"))
    if inchi:
        query_order.append(("InChI", inchi, "InChI"))

    for query_type, query_value, method in query_order:
        if not query_value:
            continue
        try:
            r = follow_nist_search(query_value, query_type)
        except Exception as exc:
            record["NIST_Notes"] = f"Lookup failed: {exc}"
            continue
        html = r.text
        title = parse_html_title(html)
        record.update(NIST_Title=title, NIST_URL=r.url, NIST_Query_Method=method)

        if "Registry Number Not Found" in title or "Internal Error" in title:
            record["NIST_Notes"] = "No NIST entry found"
            continue
        if is_nist_search_results(html):
            record["NIST_Notes"] = "Search results page returned, no direct compound page"
            continue
        record["NIST_HTML"] = html
        if nist_page_keywords(html):
            record["NIST_Found"] = True
            record["NIST_Snippet"] = nist_page_snippet(html)
            record["NIST_Notes"] = "Found thermochemistry-related content"
            return record
        record["NIST_Notes"] = "NIST compound page found, no explicit thermo keywords"
        return record

    return record