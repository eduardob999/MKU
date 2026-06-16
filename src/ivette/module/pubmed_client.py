"""PubMed ESearch/EFetch for thermochemistry abstracts."""
import re
import time
import xml.etree.ElementTree as ET
import requests

ENTREZ_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ENTREZ_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

PUBMED_THERMO_TERMS = [
    "enthalpy", "entropy", "gibbs free energy", "free energy",
    "heat of formation", "heat of combustion",
]

THERMO_REGEX = re.compile(
    r"((?:ΔH|Delta H|enthalpy|ΔS|Delta S|entropy|ΔG|Delta G|Gibbs free energy|free energy"
    r"|heat of formation|heat of combustion)[^\.\n]{0,120}?"
    r"[-+]?\d+(?:\.\d+)?\s*(?:kJ/mol|kJ mol-1|J/mol|J mol-1|kcal/mol|kcal mol-1"
    r"|cal/mol|cal mol-1|kcal per mol|kJ per mol))",
    re.IGNORECASE,
)


def build_pubmed_query(name: str) -> str:
    safe_name = name.replace('"', "")
    terms = " OR ".join([f'"{t}"[TIAB]' for t in PUBMED_THERMO_TERMS])
    return f'"{safe_name}"[TIAB] AND ({terms})'


def pubmed_search(name: str, max_results: int = 10) -> tuple[int, list[str]]:
    params = {"db": "pubmed", "retmode": "json", "retmax": str(max_results), "term": build_pubmed_query(name)}
    r = requests.get(ENTREZ_ESEARCH, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("esearchresult", {})
    return int(data.get("count", "0")), data.get("idlist", [])


def pubmed_fetch_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    r = requests.get(ENTREZ_EFETCH, params={"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)}, timeout=30)
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


def analyze_pubmed(name: str, max_results: int = 10) -> dict:
    result = {
        "PubMed_Thermo_Count": 0,
        "PubMed_Top_PMIDs": "",
        "PubMed_PubMed_Links": "",
        "PubMed_Abstract_Match_Count": 0,
        "PubMed_Example_Matches": "",
        "PubMed_Supplementary_Count": 0,
    }
    try:
        total, ids = pubmed_search(name, max_results=max_results)
        result["PubMed_Thermo_Count"] = total
        result["PubMed_Top_PMIDs"] = ",".join(ids)
        result["PubMed_PubMed_Links"] = pubmed_links(ids)
        if not ids:
            return result
        time.sleep(0.2)
        articles = pubmed_fetch_abstracts(ids)
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