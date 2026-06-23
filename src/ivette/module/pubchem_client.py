"""PubChem REST and PUG-View data fetching."""
import html
import re
from html.parser import HTMLParser

from ivette.util import http

PUBCHEM_BASE = http.PUBCHEM_PUG
PUBCHEM_PUG_VIEW_BASE = http.PUBCHEM_PUG_VIEW


def get_pubchem_details(cid: str) -> dict:
    result = {"name": "", "cas": "", "inchi": "", "synonyms": []}
    try:
        info = http.get_json(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
        ).get("InformationList", {}).get("Information", [])
        if info and "Synonym" in info[0]:
            result["synonyms"] = info[0]["Synonym"]
            for s in result["synonyms"]:
                if re.match(r"^\d{1,2}-\d{2,3}-\d{1,4}$", s):
                    result["cas"] = s
                if not result["name"] and any(c.isalpha() for c in s):
                    if "CAS" not in s and not re.match(r"^\d+-\d+-\d+$", s):
                        result["name"] = s
        if not result["name"] and result["synonyms"]:
            result["name"] = result["synonyms"][0]
    except Exception:
        pass
    try:
        info = http.get_json(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/xrefs/RN/JSON"
        ).get("InformationList", {}).get("Information", [])
        if info and "RN" in info[0] and info[0]["RN"]:
            result["cas"] = result["cas"] or info[0]["RN"][0]
    except Exception:
        pass
    try:
        info = http.get_json(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/InChI/JSON"
        ).get("PropertyTable", {}).get("Properties", [])
        if info and "InChI" in info[0]:
            result["inchi"] = info[0]["InChI"]
    except Exception:
        pass
    return result


def normalize_pug_view_value(value_obj) -> tuple[str, str]:
    if isinstance(value_obj, dict):
        if "StringWithMarkup" in value_obj:
            texts = [
                item.get("String", "") if isinstance(item, dict) else str(item)
                for item in value_obj.get("StringWithMarkup", [])
            ]
            text = " ".join(t.strip() for t in texts if t).strip()
            unit = value_obj.get("Unit") or ""
            return (f"{text} {unit}".strip() if text else unit), unit
        if "Number" in value_obj:
            nums = ", ".join(str(n) for n in value_obj.get("Number", []))
            unit = value_obj.get("Unit") or ""
            return (f"{nums} {unit}".strip() if nums else unit), unit
        if "String" in value_obj:
            return str(value_obj.get("String", "")).strip(), ""
        if "Text" in value_obj:
            return str(value_obj.get("Text", "")).strip(), ""
    return str(value_obj).strip(), ""


def find_pug_sections(obj, headings: set) -> list:
    matched = []
    if isinstance(obj, dict):
        if obj.get("TOCHeading") in headings:
            matched.append(obj)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                matched.extend(find_pug_sections(value, headings))
    elif isinstance(obj, list):
        for item in obj:
            matched.extend(find_pug_sections(item, headings))
    return matched


def extract_pubchem_property_rows_from_pug(data: dict) -> list[dict]:
    record = data.get("Record", {})
    headings = {"Computed Properties", "Chemical and Physical Properties", "Experimental Properties", "Chemical Classes"}
    sections = find_pug_sections(record, headings)
    rows = []
    for section in sections:
        for item in section.get("Section", []):
            name = item.get("TOCHeading", "").strip()
            for info in item.get("Information", []):
                value, unit = normalize_pug_view_value(info.get("Value", {}))
                reference = info.get("Reference", [])
                if isinstance(reference, list):
                    reference = " | ".join(str(r).strip() for r in reference if r)
                elif reference is None:
                    reference = ""
                rows.append({"PropertyName": name, "PropertyValue": value, "PropertyUnit": unit, "Reference": reference})
    return rows


class PubChemPropertyHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.div_stack = []
        self.in_row = False
        self.row_div_depth = 0
        self.current_row = []
        self.current_cell = False
        self.cell_div_depth = 0
        self.cell_text = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag != "div":
            self.div_stack.append((tag, None))
            return
        attrs = dict(attrs)
        cls = attrs.get("class", "")
        if "sm:table-row" in cls and not self.in_row:
            self.in_row = True
            self.row_div_depth = len(self.div_stack)
            self.current_row = []
        elif self.in_row and "sm:table-cell" in cls and not self.current_cell:
            self.current_cell = True
            self.cell_div_depth = len(self.div_stack)
            self.cell_text = []
        self.div_stack.append((tag, cls))

    def handle_endtag(self, tag):
        if not self.div_stack:
            return
        self.div_stack.pop()
        if self.current_cell and len(self.div_stack) < self.cell_div_depth:
            text = html.unescape(re.sub(r"\s+", " ", "".join(self.cell_text).strip()))
            self.current_row.append(text)
            self.current_cell = False
        if self.in_row and len(self.div_stack) < self.row_div_depth:
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_row = False

    def handle_data(self, data):
        if self.current_cell:
            self.cell_text.append(data)


def extract_pubchem_property_rows_from_html(html_text: str) -> list[dict]:
    parser = PubChemPropertyHtmlParser()
    parser.feed(html_text)
    parser.close()
    rows = []
    for row in parser.rows:
        if len(row) < 3:
            continue
        name, value, reference = row[0], row[1], row[2]
        if name.strip().lower() in {"property name", "property value", "reference"}:
            continue
        rows.append({"PropertyName": name, "PropertyValue": value, "PropertyUnit": "", "Reference": reference})
    return rows


def fetch_pubchem_property_rows(cid: str) -> list[dict]:
    try:
        rows = extract_pubchem_property_rows_from_pug(
            http.get_json(f"{PUBCHEM_PUG_VIEW_BASE}/{cid}/JSON")
        )
        if rows:
            return rows
    except Exception:
        pass
    try:
        text = http.get_text(
            f"{http.PUBCHEM_COMPOUND}/{cid}", headers=http.BROWSER_HEADERS
        )
        return extract_pubchem_property_rows_from_html(text)
    except Exception:
        return []