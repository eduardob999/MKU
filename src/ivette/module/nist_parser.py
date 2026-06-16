"""Parse thermochemical property tables from NIST WebBook HTML."""
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup


def normalize_nist_headers(headers: list[str]) -> list[str]:
    normalized = []
    for header in headers:
        text = re.sub(r"\s+", " ", header.strip())
        key = text.lower()
        if key.startswith("quantity") or key in {"quantity", "property", "property name"}:
            normalized.append("PropertyName")
        elif key.startswith("value") or key == "value":
            normalized.append("PropertyValue")
        elif key.startswith("unit") or key == "units":
            normalized.append("PropertyUnit")
        elif "method" in key:
            normalized.append("Method")
        elif "reference" in key:
            normalized.append("Reference")
        elif "comment" in key or "note" in key or "remark" in key:
            normalized.append("Comment")
        elif "temperature" in key or key in {"t", "t (k)"}:
            normalized.append("Temperature")
        elif "pressure" in key or key in {"p", "p (kpa)", "p (atm)"}:
            normalized.append("Pressure")
        elif "phase" in key or "state" in key:
            normalized.append("Phase")
        elif "reaction" in key and "equation" not in key:
            normalized.append("ReactionEquation")
        else:
            normalized.append(text)
    return normalized


def split_header_unit(header: str) -> tuple[str, str]:
    header = header.strip()
    match = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", header)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return header, ""


def split_value_unit(value: str) -> tuple[str, str]:
    value = str(value).strip()
    if not value:
        return "", ""
    match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", value)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.match(r"^([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s+(.+))?$", value)
    if match:
        return match.group(1).strip(), (match.group(2) or "").strip()
    return value, ""


def is_condition_label(header_text: str) -> bool:
    low = header_text.lower()
    return any(k in low for k in ["temperature", "pressure", "phase", "state", "condition", "conditions"])


def format_condition(fields: dict) -> str:
    return " | ".join(f"{k}={v}" for k, v in fields.items() if v)


def extract_reaction_equation(table) -> str:
    if table is None:
        return ""
    for sibling in table.find_previous_siblings():
        if sibling.name in {"h2", "h3", "p"}:
            text = sibling.get_text(" ", strip=True)
            if "=" in text:
                return text
    prev = table.find_previous(text=lambda t: isinstance(t, str) and "=" in t)
    return prev.strip() if prev else ""


def find_repeated_header_groups(normalized_headers: list[str]) -> list[list[str]] | None:
    start_indices = [i for i, h in enumerate(normalized_headers) if h == "PropertyName"]
    if len(start_indices) < 2:
        return None
    group_size = start_indices[1] - start_indices[0]
    if group_size <= 0 or len(normalized_headers) % group_size != 0:
        return None
    groups = [normalized_headers[i: i + group_size] for i in range(0, len(normalized_headers), group_size)]
    return groups if all(g == groups[0] for g in groups) else None


def build_nist_rows_from_data(
    row_values: list[str],
    headers: list[str],
    normalized_headers: list[str],
    metadata: dict,
    table_heading: str,
    group_offset: int = 0,
    group_size: int = None,
) -> list[dict]:
    source_url = metadata.get("SourceURL", "")
    subsection_text = metadata.get("Subsection", "")
    section_title = metadata.get("Section", "")
    if section_title.lower().endswith(" data"):
        section_title = section_title[:-5].strip()

    base_record = {
        "CID": metadata.get("CID", ""),
        "InChIKey": metadata.get("InChIKey", ""),
        "PubChemName": metadata.get("PubChemName", ""),
        "PubChem_URL": metadata.get("PubChem_URL", ""),
        "Source": "NIST",
        "Section": section_title,
        "Subsection": subsection_text,
        "PropertyName": "",
        "PropertyValue": "",
        "PropertyUnit": "",
        "Reference": "",
        "Method": "",
        "Comment": "",
        "Condition": "",
        "ReactionEquation": "",
        "SourceURL": source_url,
    }

    if group_size:
        headers = headers[group_offset: group_offset + group_size]
        normalized_headers = normalized_headers[group_offset: group_offset + group_size]
        row_values = row_values[group_offset: group_offset + group_size]

    row_map = {}
    for idx, header in enumerate(headers):
        value = row_values[idx] if idx < len(row_values) else ""
        row_map[normalized_headers[idx]] = value.strip()
        row_map[header] = value.strip()

    if "PropertyName" in normalized_headers and "PropertyValue" in normalized_headers:
        if row_map.get("PropertyName") or row_map.get("PropertyValue"):
            record = dict(base_record)
            record["PropertyName"] = row_map.get("PropertyName", "")
            record["PropertyValue"] = row_map.get("PropertyValue", "")
            unit = row_map.get("PropertyUnit", "")
            if not unit:
                _, unit = split_value_unit(record["PropertyValue"])
            record["PropertyUnit"] = unit
            record["Reference"] = row_map.get("Reference", "")
            record["Method"] = row_map.get("Method", "")
            record["Comment"] = row_map.get("Comment", "")
            conditions = {k: row_map.get(k, "") for k in ["Temperature", "Pressure", "Phase"]}
            record["Condition"] = format_condition({k: v for k, v in conditions.items() if v})
            record["ReactionEquation"] = extract_reaction_equation(metadata.get("table"))
            return [record]

    measurement_columns = []
    condition_fields = {}
    for idx, header in enumerate(headers):
        normalized_header = normalized_headers[idx]
        cell_value = row_values[idx] if idx < len(row_values) else ""
        if normalized_header in {"Reference", "Method", "Comment"}:
            base_record[normalized_header] = cell_value.strip()
        elif normalized_header in {"Temperature", "Pressure", "Phase"}:
            condition_fields[header] = cell_value.strip()
        elif normalized_header == "PropertyUnit":
            base_record["PropertyUnit"] = cell_value.strip()
        elif normalized_header == "PropertyName":
            base_record["PropertyName"] = cell_value.strip()
        elif normalized_header == "PropertyValue":
            base_record["PropertyValue"] = cell_value.strip()
        else:
            if is_condition_label(header):
                condition_fields[header] = cell_value.strip()
            else:
                measurement_columns.append((idx, header, normalized_header, cell_value.strip()))

    rows = []
    if measurement_columns:
        for idx, header, normalized_header, cell_value in measurement_columns:
            record = dict(base_record)
            name, unit = split_header_unit(header)
            record["PropertyName"] = name
            if cell_value:
                value, value_unit = split_value_unit(cell_value)
                record["PropertyValue"] = value
                record["PropertyUnit"] = value_unit or record["PropertyUnit"] or unit
            else:
                record["PropertyValue"] = ""
                record["PropertyUnit"] = record["PropertyUnit"] or unit
            record["Condition"] = format_condition({k: v for k, v in condition_fields.items() if v})
            record["ReactionEquation"] = extract_reaction_equation(metadata.get("table"))
            rows.append(record)
        return rows

    record = dict(base_record)
    record["PropertyName"] = table_heading or ""
    if record["PropertyValue"]:
        value, value_unit = split_value_unit(record["PropertyValue"])
        record["PropertyValue"] = value
        record["PropertyUnit"] = value_unit or record["PropertyUnit"]
    record["Condition"] = format_condition({k: v for k, v in condition_fields.items() if v})
    record["ReactionEquation"] = extract_reaction_equation(metadata.get("table"))
    return [record]


def extract_nist_section_links(html: str, base_url: str = "https://webbook.nist.gov/cgi/cbook.cgi") -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    sections, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "Mask=" not in href:
            continue
        section_text = a.get_text(" ", strip=True)
        if not section_text:
            continue
        low = section_text.lower()
        if not any(k in low for k in ["thermochemistry", "phase", "reaction", "ion energetics", "henry", "solubility"]):
            continue
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        sections.append({"section": section_text, "url": url})
    return sections


def extract_nist_property_rows(html: str, metadata: dict) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in soup.find_all("table", class_="data"):
        heading_el = table.find_previous(["h2", "h3"])
        table_heading = heading_el.get_text(" ", strip=True) if heading_el else ""
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if not headers:
            continue
        normalized_headers = normalize_nist_headers(headers)
        repeated_groups = find_repeated_header_groups(normalized_headers)
        for tr in table.find_all("tr"):
            values = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if not values or all(not v for v in values) or values == headers:
                continue
            if repeated_groups:
                group_size = len(repeated_groups[0])
                for group_idx in range(len(repeated_groups)):
                    group_offset = group_idx * group_size
                    for row in build_nist_rows_from_data(
                        values, headers, normalized_headers,
                        {**metadata, "table": table}, table_heading,
                        group_offset=group_offset, group_size=group_size,
                    ):
                        if row["PropertyName"] or row["PropertyValue"]:
                            rows.append(row)
            else:
                for row in build_nist_rows_from_data(
                    values, headers, normalized_headers,
                    {**metadata, "table": table}, table_heading,
                ):
                    if row["PropertyName"] or row["PropertyValue"]:
                        rows.append(row)
    return rows