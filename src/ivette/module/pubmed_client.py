"""PubMed ESearch/EFetch for thermochemistry abstracts."""
import re
import time
import xml.etree.ElementTree as ET

from ivette.util import http
from ivette.util.patterns import THERMO_REGEX

ENTREZ_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ENTREZ_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

PUBMED_THERMO_TERMS = [
    "enthalpy", "entropy", "gibbs free energy", "free energy",
    "heat of formation", "heat of combustion",
]


def build_pubmed_query(name: str) -> str:
    safe_name = name.replace('"', "")
    terms = " OR ".join([f'"{t}"[TIAB]' for t in PUBMED_THERMO_TERMS])
    return f'"{safe_name}"[TIAB] AND ({terms})'


def pubmed_search(name: str, max_results: int = 10, api_key: str = "") -> tuple[int, list[str]]:
    params = {"db": "pubmed", "retmode": "json", "retmax": str(max_results), "term": build_pubmed_query(name)}
    if api_key:
        params["api_key"] = api_key
    data = http.get_json(ENTREZ_ESEARCH, params=params).get("esearchresult", {})
    return int(data.get("count", "0")), data.get("idlist", [])


def pubmed_fetch_abstracts(pmids: list[str], api_key: str = "") -> list[dict]:
    if not pmids:
        return []
    params = {"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)}
    if api_key:
        params["api_key"] = api_key
    r = http.get(ENTREZ_EFETCH, params=params)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    articles = []
    for article in root.findall("PubmedArticle"):
        pmid_el = article.find("MedlineCitation/PMID")
        title_el = article.find("MedlineCitation/Article/ArticleTitle")
        abstract_parts = [
            el.text for el in article.findall("MedlineCitation/Article/Abstract/AbstractText") if el.text
        ]
        articles.append({
            "pmid": pmid_el.text if pmid_el is not None else "",
            "title": title_el.text if title_el is not None else "",
            "abstract": " ".join(abstract_parts),
        })
    return articles


def extract_thermo_matches(text: str) -> list[str]:
    matches = []
    for match in THERMO_REGEX.finditer(text):
        snippet = match.group(1).strip()
        if snippet not in matches:
            matches.append(snippet)
            if len(matches) >= 5:
                break
    return matches


def pubmed_links(pmids: list[str]) -> str:
    return " ".join(f"https://pubmed.ncbi.nlm.nih.gov/{p}/" for p in pmids if p)


def empty_pubmed_result() -> dict:
    """The PubMed columns with default/empty values (used when PubMed is off)."""
    return {
        "PubMed_Thermo_Count": 0,
        "PubMed_Top_PMIDs": "",
        "PubMed_PubMed_Links": "",
        "PubMed_Abstract_Match_Count": 0,
        "PubMed_Example_Matches": "",
        "PubMed_Supplementary_Count": 0,
    }


def analyze_pubmed(name: str, max_results: int = 10, api_key: str = "") -> dict:
    result = empty_pubmed_result()
    try:
        total, ids = pubmed_search(name, max_results=max_results, api_key=api_key)
        result["PubMed_Thermo_Count"] = total
        result["PubMed_Top_PMIDs"] = ",".join(ids)
        result["PubMed_PubMed_Links"] = pubmed_links(ids)
        if not ids:
            return result
        time.sleep(0.2)
        articles = pubmed_fetch_abstracts(ids, api_key=api_key)
        matches, supplementary = [], 0
        for art in articles:
            text = " ".join([art["title"], art["abstract"]]).strip()
            if not text:
                continue
            if re.search(r"supplementary|supporting information|supplemental", text, re.I):
                supplementary += 1
            found = extract_thermo_matches(text)
            if found:
                result["PubMed_Abstract_Match_Count"] += 1
                for s in found:
                    if s not in matches:
                        matches.append(s)
                        if len(matches) >= 5:
                            break
        result["PubMed_Example_Matches"] = " | ".join(matches[:5])
        result["PubMed_Supplementary_Count"] = supplementary
    except Exception as exc:
        result["PubMed_Example_Matches"] = f"Error: {exc}"
    return result