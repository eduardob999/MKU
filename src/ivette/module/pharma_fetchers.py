"""Pharmacology data fetchers: PubChem BioAssay, ChEMBL, BindingDB.

Provides:
- fetch_pubchem_bioassays(cid)
- fetch_chembl_activities_by_inchikey(inchikey)
- fetch_bindingdb_activities_by_inchikey(inchikey)

Returns list of dicts with standardized keys:
{ 'Source','AssayID','Target','ActivityType','Value','Unit','Relation','pChemblValue','Reference','AssayDescription','URL' }
"""

import json
import logging
import os
import re
import time
from typing import Dict, List
from urllib.parse import quote_plus


import requests
from bs4 import BeautifulSoup

PUBCHEM_PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CACHE_FILENAME = os.path.join(os.path.dirname(__file__), "chembl_target_name_cache.json")

# Default values; can be overridden from CLI/menu.
DEFAULT_PUBCHEM_MAX_AIDS = 20
DEFAULT_activity_limit = 500
DEFAULT_CHEMBL_MAX_PAGES = 3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target name cache
# ---------------------------------------------------------------------------

def _load_target_name_cache(path: str | None = None) -> Dict[str, str]:
    cache_path = path or CACHE_FILENAME
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
    except Exception:
        logger.debug("Failed to load target name cache %s", cache_path)
    return {}


def _save_target_name_cache(cache: Dict[str, str], path: str | None = None) -> None:
    cache_path = path or CACHE_FILENAME
    try:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.debug("Failed to save target name cache %s: %s", cache_path, exc)


_TARGET_NAME_CACHE: Dict[str, str] = _load_target_name_cache()


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _unit_to_molar_multiplier(unit: str) -> float | None:
    if not unit:
        return None
    u = str(unit).strip().lower()
    u = u.replace("\u00b5", "u").replace("µ", "u").replace("μ", "u")
    mapping = {
        "pm": 1e-12, "pmol": 1e-12, "pmol/l": 1e-12,
        "nm": 1e-9,  "nmol": 1e-9,  "nmol/l": 1e-9,
        "um": 1e-6,  "umol": 1e-6,  "umol/l": 1e-6, "micromolar": 1e-6,
        "mm": 1e-3,  "mmol": 1e-3,  "mmol/l": 1e-3,
        "m":  1.0,   "mol":  1.0,   "mol/l":  1.0,
    }
    if u in mapping:
        return mapping[u]
    if u.endswith("m") and len(u) <= 3:
        prefixes = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3}
        return prefixes.get(u[0])
    return None


# ---------------------------------------------------------------------------
# PubChem BioAssay
# ---------------------------------------------------------------------------

def fetch_pubchem_bioassays(
    cid: str,
    max_aids: int = DEFAULT_PUBCHEM_MAX_AIDS,
) -> List[Dict]:
    """Fetch PubChem BioAssay activity records for a given CID.

    Limits to PUBCHEM_MAX_AIDS assays and makes a single CSV request per AID
    instead of 3 separate calls (description + summary + CSV).
    """
    out = []

    # Step 1: get AIDs
    try:
        r = requests.get(
            f"{PUBCHEM_PUG_BASE}/compound/cid/{cid}/aids/JSON", timeout=30
        )
        r.raise_for_status()
        data = r.json()
        aids = []
        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "AID" in item:
                        aids.append(str(item["AID"]))
            elif isinstance(v, dict) and "AID" in v:
                aids.append(str(v["AID"]))
        aids = list(dict.fromkeys(aids))[:max_aids]
    except Exception as exc:
        logger.debug("PubChem AID lookup failed for %s: %s", cid, exc)
        return out

    if not aids:
        return out

    # Step 2: single CSV fetch per AID (replaces 3-call pattern)
    for aid in aids:
        try:
            r = requests.get(
                f"{PUBCHEM_PUG_BASE}/assay/aid/{aid}/CSV", timeout=30
            )
            if r.status_code != 200 or not r.text.strip():
                out.append({
                    "Source": "PubChem", "AssayID": aid, "AssayTitle": "",
                    "AssayDescription": "", "ActivityType": None,
                    "Value": None, "Unit": None, "Outcome": None,
                    "URL": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
                })
                continue

            lines = [ln for ln in r.text.splitlines() if ln.strip()]
            if not lines:
                continue
            header = [h.strip().strip('"') for h in lines[0].split(",")]

            def col(name):
                try:
                    return header.index(name)
                except ValueError:
                    return None

            outcome_idx = col("PUBCHEM_ACTIVITY_OUTCOME")
            conc_idx = col("PUBCHEM_ACTIVITY_CONCENTRATION")
            unit_idx = col("PUBCHEM_ACTIVITY_CONCENTRATION_UNIT")

            for ln in lines[1:]:
                parts = [p.strip().strip('"') for p in ln.split(",")]
                outcome = parts[outcome_idx] if outcome_idx is not None and outcome_idx < len(parts) else ""
                conc = parts[conc_idx] if conc_idx is not None and conc_idx < len(parts) else ""
                unit = parts[unit_idx] if unit_idx is not None and unit_idx < len(parts) else ""
                try:
                    val = float(conc) if conc else None
                except Exception:
                    val = None
                out.append({
                    "Source": "PubChem", "AssayID": aid, "AssayTitle": "",
                    "AssayDescription": "", "ActivityType": "Concentration",
                    "Value": val, "Unit": unit, "Outcome": outcome,
                    "URL": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
                })
            time.sleep(0.1)
        except Exception as exc:
            logger.debug("PubChem assay parse failed for AID %s: %s", aid, exc)
    return out


# ---------------------------------------------------------------------------
# ChEMBL
# ---------------------------------------------------------------------------

def _batch_resolve_target_names(
    target_ids: List[str],
    cache: Dict[str, str],
    cache_path: str | None,
) -> bool:
    """Resolve a list of ChEMBL target IDs to names, updating cache in-place.

    Skips IDs already in cache. Returns True if cache was modified.
    """
    missing = [t for t in target_ids if t and t not in cache]
    if not missing:
        return False

    dirty = False
    for target_id in missing:
        try:
            r = requests.get(
                f"{CHEMBL_BASE}/target/{target_id}.json", timeout=15
            )
            r.raise_for_status()
            td = r.json()
            name = td.get("pref_name") or td.get("target_name") or ""
            if not name:
                for comp in td.get("target_components", []):
                    if isinstance(comp, dict):
                        name = comp.get("component_name") or comp.get("accession") or ""
                        if name:
                            break
            cache[target_id] = name or ""
            dirty = True
            time.sleep(0.05)
        except Exception:
            cache[target_id] = ""
            dirty = True

    if dirty:
        _save_target_name_cache(cache, cache_path)
    return dirty


def fetch_chembl_activities_by_inchikey(
    inchikey: str,
    cache_path: str | None = None,
    activity_limit: int = DEFAULT_activity_limit,
    max_pages: int = DEFAULT_CHEMBL_MAX_PAGES,
) -> List[Dict]:

    out = []

    cache = (
        _TARGET_NAME_CACHE
        if cache_path is None
        else _load_target_name_cache(cache_path)
    )

    try:
        r = requests.get(
            f"{CHEMBL_BASE}/molecule"
            f"?molecule_structures__standard_inchi_key={inchikey}"
            f"&format=json",
            timeout=30,
        )

        r.raise_for_status()

        data = r.json()

        results = (
            data.get("molecules")
            or data.get("molecule")
            or []
        )

    except Exception as exc:
        logger.debug(
            "ChEMBL molecule lookup failed for %s: %s",
            inchikey,
            exc,
        )
        return out

    if not results:
        return out

    chembl_id = (
        results[0].get("molecule_chembl_id")
        or results[0].get("chembl_id")
    )

    if not chembl_id:
        return out

    raw_activities = []

    offset = 0

    for _ in range(max_pages):

        try:

            r = requests.get(
                f"{CHEMBL_BASE}/activity"
                f"?molecule_chembl_id={chembl_id}"
                f"&limit={activity_limit}"
                f"&offset={offset}"
                f"&format=json",
                timeout=60,
            )

            r.raise_for_status()

            data = r.json()

            page = (
                data.get("activities")
                or data.get("activity")
                or []
            )

            raw_activities.extend(page)

            if len(page) < activity_limit:
                break

            offset += activity_limit

            time.sleep(0.1)

        except Exception as exc:

            logger.debug(
                "ChEMBL activities fetch failed for %s: %s",
                chembl_id,
                exc,
            )

            break

    if not raw_activities:
        return out

    unique_targets = list({
        a.get("target_chembl_id")
        for a in raw_activities
        if a.get("target_chembl_id")
    })

    _batch_resolve_target_names(
        unique_targets,
        cache,
        cache_path,
    )

    for a in raw_activities:

        target_id = (
            a.get("target_chembl_id")
            or ""
        )

        try:
            pchembl = (
                float(a["pchembl_value"])
                if a.get("pchembl_value")
                not in ("", None)
                else None
            )
        except Exception:
            pchembl = None

        try:
            value = (
                float(a["standard_value"])
                if a.get("standard_value")
                not in ("", None)
                else None
            )
        except Exception:
            value = None

        out.append({

            "Source": "ChEMBL",

            "AssayID":
                a.get("assay_chembl_id")
                or a.get("assay_id"),

            "Target":
                target_id,

            "TargetName":
                cache.get(target_id, ""),

            "ActivityType":
                a.get("standard_type"),

            "Value":
                value,

            "Unit":
                a.get("standard_units"),

            "Relation":
                a.get("standard_relation"),

            "pChemblValue":
                pchembl,

            "Reference":
                a.get("document_chembl_id")
                or a.get("src_id"),

            "AssayDescription":
                a.get("assay_description"),

            "URL": (
                f"https://www.ebi.ac.uk/chembl/assay/inspect/"
                f"{a['assay_chembl_id']}"
                if a.get("assay_chembl_id")
                else ""
            ),
        })

    return out


# ---------------------------------------------------------------------------
# BindingDB
# ---------------------------------------------------------------------------

def fetch_bindingdb_activities_by_inchikey(inchikey: str) -> List[Dict]:
    """Fetch BindingDB activities by InChIKey via HTML scraping."""
    out = []
    pattern = re.compile(
        r"(IC50|Ki|Kd|EC50)[\s:=]*([0-9,.]+)\s*(nM|uM|µM|mM|M|pM)?", re.I
    )
    candidates = [
        f"https://www.bindingdb.org/bind/chemsearch/marvin/SimpleSearch.jsp?search={quote_plus(inchikey)}",
        f"https://www.bindingdb.org/unbreak/search?search={quote_plus(inchikey)}",
        f"https://www.bindingdb.org/bind/chemsearch/marvin/results.jsp?search={quote_plus(inchikey)}",
    ]
    try:
        page_text = None
        for url in candidates:
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and inchikey.lower() in r.text.lower():
                    page_text = r.text
                    break
            except Exception:
                continue

        if not page_text:
            return out

        soup = BeautifulSoup(page_text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "bindingdb" in href or href.startswith("/bind/") or "BindingDB" in a.text:
                if href.startswith("/"):
                    href = "https://www.bindingdb.org" + href
                links.append(href)
        detail_pages = list(dict.fromkeys(links))[:20] or [candidates[0]]

        for url in detail_pages:
            try:
                r = requests.get(url, timeout=30)
                if r.status_code != 200:
                    continue
                for m in pattern.finditer(r.text):
                    atype = m.group(1).upper()
                    try:
                        val = float(m.group(2).replace(",", ""))
                    except Exception:
                        continue
                    unit = (m.group(3) or "").strip()
                    multiplier = _unit_to_molar_multiplier(unit)
                    value_m = val * multiplier if multiplier is not None else None
                    out.append({
                        "Source": "BindingDB",
                        "AssayID": url.split("/")[-1],
                        "AssayTitle": "", "AssayDescription": "",
                        "ActivityType": atype,
                        "Value": val, "Unit": unit, "Value_M": value_m,
                        "Relation": None, "Reference": url, "URL": url,
                    })
                time.sleep(0.1)
            except Exception:
                continue
    except Exception as exc:
        logger.debug("BindingDB fetch failed for %s: %s", inchikey, exc)
    return out