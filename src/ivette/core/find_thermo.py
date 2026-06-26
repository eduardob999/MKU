#!/usr/bin/env python3
"""find_thermo.py — orchestrator. All logic lives in sibling modules."""
import csv
import os
import subprocess
import sys
import time

from ivette.util import http
from ivette.util import applog
from ivette.util.csvio import write_csv
from ivette.util.timing import TimingLog
from ivette.module.cli import build_arg_parser, interactive_parameter_menu
from ivette.module.nist_client import check_nist_entry
from ivette.module.nist_parser import extract_nist_section_links, extract_nist_property_rows
from ivette.module.pubchem_client import get_pubchem_details, fetch_pubchem_property_rows
from ivette.module.pubmed_client import analyze_pubmed
from ivette.module.wide_output import build_wide_output, merge_pharma_into_wide, strip_property_value_unit, write_values_only_wide
from ivette.module.pharma_fetchers import fetch_pubchem_bioassays, fetch_chembl_activities_by_inchikey, fetch_bindingdb_activities_by_inchikey

REPORT_FIELDNAMES = [
    "CID", "InChIKey", "PubChemName", "PubChem_CAS", "PubChem_InChI", "PubChem_URL",
    "NIST_Found", "NIST_URL", "NIST_Query_Method", "NIST_Title", "NIST_Notes", "NIST_Snippet",
    "PubMed_Thermo_Count", "PubMed_Top_PMIDs", "PubMed_PubMed_Links",
    "PubMed_Abstract_Match_Count", "PubMed_Example_Matches", "PubMed_Supplementary_Count",
]

PARSED_FIELDNAMES = [
    "CID", "InChIKey", "PubChemName", "PubChem_URL", "Source", "Section", "Subsection",
    "PropertyName", "PropertyValue", "PropertyUnit", "Reference", "Method",
    "Comment", "Condition", "ReactionEquation", "SourceURL",
]


def validate_parsed_csv(path: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"Error: parsed CSV '{path}' does not exist.")

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        first_row = next(reader, None)

    required = {"CID", "PropertyName", "PropertyValue"}
    missing = required - set(header)
    if missing:
        raise SystemExit(
            f"Error: '{path}' is missing required columns: {missing}\n"
            f"Columns present: {sorted(header)}\n"
            f"Expected thermo_parsed.csv — did you pass the wrong file?"
        )

    pharma_cols = {"assay_id", "target_name", "activity_value", "AID", "ChEMBL_ID"}
    report_cols = {"NIST_Found", "NIST_URL", "PubMed_Thermo_Count"}
    wide_cols   = {"MW_Median", "Tb_Median", "Tm_Median", "LogP_Median"}
    for label, clues in [("pharmacology", pharma_cols), ("report", report_cols), ("wide/ML", wide_cols)]:
        found = set(header) & clues
        if found:
            raise SystemExit(
                f"Error: '{path}' looks like a {label} CSV, not a parsed thermo file.\n"
                f"Clue columns found: {found}"
            )

    if first_row is None:
        raise SystemExit(f"Error: '{path}' has no data rows.")


def run_cleaning_pipeline(parsed_input, cleaned_output, summary_output, ml_output, rare_output, report_output):
    script_path = os.path.join(os.path.dirname(__file__), "clean_thermo.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"clean_thermo.py not found at {script_path}")
    subprocess.run([
        sys.executable, script_path, parsed_input,
        "--output", cleaned_output,
        "--summary-output", summary_output,
        "--ml-output", ml_output,
        "--rare-output", rare_output,
        "--report-output", report_output,
    ], check=True)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.menu:
        args = interactive_parameter_menu(args)

    applog.configure()
    log = applog.get_logger("dataset")
    timing_log_path = getattr(args, "timing_log", "timing_log.txt")
    log.info("dataset run starting | input=%s max=%s pharma=%s",
             args.input, args.max, getattr(args, "fetch_pharma", None))
    tlog = TimingLog(timing_log_path)

    if not os.path.exists(args.input):
        raise SystemExit(f"Error: input file '{args.input}' does not exist.")

    with open(args.input, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [row for i, row in enumerate(reader) if not args.max or i < args.max]

    report, available, parsed_rows, pharma_rows = [], [], [], []

    for r in rows:
        cid = r.get("CID") or r.get("cid") or ""
        inchikey = r.get("InChIKey") or r.get("inchikey") or ""
        pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else r.get("PubChem_URL", "")
        print(f"\nProcessing CID={cid} InChIKey={inchikey}")

        t0 = time.perf_counter()
        pubchem = get_pubchem_details(cid) if cid else {"name": "", "cas": "", "inchi": ""}
        tlog.record(f"CID={cid}  PubChem details", time.perf_counter() - t0)

        name = pubchem.get("name") or r.get("PubChemName") or r.get("Name") or ""
        cas, inchi = pubchem.get("cas", ""), pubchem.get("inchi", "")

        t0 = time.perf_counter()
        nist = check_nist_entry(inchikey, name=name, cas=cas, inchi=inchi)
        tlog.record(f"CID={cid}  NIST lookup", time.perf_counter() - t0)

        t0 = time.perf_counter()
        pubmed = analyze_pubmed(name or inchikey, max_results=args.pubmed_max)
        tlog.record(f"CID={cid}  PubMed analysis", time.perf_counter() - t0)

        entry = {
            "CID": cid, "InChIKey": inchikey, "PubChemName": name,
            "PubChem_CAS": cas, "PubChem_InChI": inchi, "PubChem_URL": pubchem_url,
            **{k: nist[k] for k in ["NIST_Found", "NIST_URL", "NIST_Query_Method", "NIST_Title", "NIST_Notes", "NIST_Snippet"]},
            **pubmed,
        }
        report.append(entry)
        if entry["NIST_Found"] or entry["PubMed_Abstract_Match_Count"] > 0:
            available.append(entry)

        if args.parsed_output and cid:
            t0 = time.perf_counter()
            for prop in fetch_pubchem_property_rows(cid):
                unit = prop.get("PropertyUnit", "")
                parsed_rows.append({
                    "CID": cid, "InChIKey": inchikey, "PubChemName": name,
                    "PubChem_URL": pubchem_url, "Source": "PubChem",
                    "Section": "", "Subsection": "",
                    "PropertyName": prop["PropertyName"],
                    "PropertyValue": strip_property_value_unit(prop["PropertyValue"], unit),
                    "PropertyUnit": unit, "Reference": prop["Reference"],
                    "Method": "", "Comment": "", "Condition": "",
                    "ReactionEquation": "", "SourceURL": "",
                })
            tlog.record(f"CID={cid}  PubChem property rows", time.perf_counter() - t0)

            if nist.get("NIST_HTML"):
                t0 = time.perf_counter()
                for section in extract_nist_section_links(nist["NIST_HTML"], nist["NIST_URL"]):
                    try:
                        section_html = http.get_text(section["url"])
                        sec_name = section["section"]
                        if sec_name.lower().endswith(" data"):
                            sec_name = sec_name[:-5].strip()
                        parsed_rows.extend(extract_nist_property_rows(section_html, {
                            "CID": cid, "InChIKey": inchikey, "PubChemName": name,
                            "PubChem_URL": pubchem_url, "SourceURL": section["url"],
                            "Section": sec_name, "Subsection": section["section"],
                        }))
                    except Exception as exc:
                        print(f"Warning: NIST section parse failed {section['url']}: {exc}", file=sys.stderr)
                tlog.record(f"CID={cid}  NIST section parsing", time.perf_counter() - t0)

        time.sleep(0.2)

    t0 = time.perf_counter()
    write_csv(args.output, REPORT_FIELDNAMES, report)
    write_csv(args.available_output, REPORT_FIELDNAMES, available)
    tlog.record("Writing report CSVs", time.perf_counter() - t0)
    print(f"Wrote report to {args.output}")
    print(f"Wrote available-data report to {args.available_output}")

    if args.parsed_output:
        t0 = time.perf_counter()
        write_csv(args.parsed_output, PARSED_FIELDNAMES, parsed_rows)
        tlog.record("Writing parsed properties CSV", time.perf_counter() - t0)
        print(f"Wrote parsed properties to {args.parsed_output}")

    if args.fetch_pharma:

        total = len(rows)
        start_ts = time.time()

        for idx, r in enumerate(rows, start=1):

            cid = r.get("CID") or r.get("cid") or ""
            inchikey = r.get("InChIKey") or r.get("inchikey") or ""
            compound_t0 = time.perf_counter()

            sources = [
                (fetch_pubchem_bioassays,              cid,      f"PubChem CID={cid}",           {"max_aids": args.pubchem_max_aids}),
                (fetch_chembl_activities_by_inchikey,  inchikey, f"ChEMBL InChIKey={inchikey}",  {"cache_path": args.chembl_target_cache, "activity_limit": args.chembl_activity_limit, "max_pages": args.chembl_max_pages}),
                (fetch_bindingdb_activities_by_inchikey, inchikey, f"BindingDB InChIKey={inchikey}", {}),
            ]

            for fetch_fn, key, label, kwargs in sources:
                if not key:
                    continue
                t0 = time.perf_counter()
                try:
                    for a in fetch_fn(key, **kwargs):
                        a.update({"CID": cid, "InChIKey": inchikey})
                        pharma_rows.append(a)
                except Exception as exc:
                    print(f"\nWarning: {label} fetch failed: {exc}", file=sys.stderr)
                tlog.record(f"CID={cid}  pharma {label}", time.perf_counter() - t0)

            tlog.record(f"CID={cid}  pharma total", time.perf_counter() - compound_t0)
            time.sleep(0.15)

            elapsed = time.time() - start_ts
            pct = (idx / total) * 100 if total else 100.0
            eta = (elapsed / idx) * (total - idx) if idx < total else 0.0
            print(f"Progress: {idx}/{total} ({pct:5.1f}%) elapsed {elapsed:5.1f}s ETA {eta:5.1f}s")

        if pharma_rows:
            t0 = time.perf_counter()
            pharma_keys = sorted({k for d in pharma_rows for k in d.keys()})
            write_csv(args.pharma_output, pharma_keys, pharma_rows)
            tlog.record("Writing pharmacology CSV", time.perf_counter() - t0)
            print(f"Wrote pharmacology rows to {args.pharma_output}")

    if args.merge_pharma and pharma_rows:
        t0 = time.perf_counter()
        pharma_keys = sorted({k for d in pharma_rows for k in d.keys()})
        write_csv(args.merged_pharma_output, pharma_keys, pharma_rows)
        tlog.record("Writing merged pharmacology CSV", time.perf_counter() - t0)
        print(f"Wrote raw pharmacology rows to {args.merged_pharma_output}")

    if args.wide_output and args.wide_from_clean:
        if not args.parsed_output:
            raise SystemExit("Cannot build wide output from clean thermo without a parsed output CSV")
        validate_parsed_csv(args.parsed_output)  # <-- before subprocess
        t0 = time.perf_counter()
        run_cleaning_pipeline(
            args.parsed_output, args.cleaned_output, args.summary_output,
            args.wide_output, args.rare_output, args.cleaning_report,
        )
        tlog.record("Cleaning pipeline (clean_thermo.py)", time.perf_counter() - t0)
        print(f"Cleaned: {args.cleaned_output}, Summary: {args.summary_output}, ML: {args.wide_output}")
    elif args.wide_output:
        t0 = time.perf_counter()
        build_wide_output(parsed_rows, args.wide_output)
        tlog.record("Building wide output", time.perf_counter() - t0)
        print(f"Wrote wide output to {args.wide_output}")

    if args.wide_output and args.merge_pharma and os.path.exists(args.merged_pharma_output):
        t0 = time.perf_counter()
        try:
            merge_pharma_into_wide(args.merged_pharma_output, args.wide_output)
            tlog.record("Merging pharmacology into wide output", time.perf_counter() - t0)
            print(f"Merged pharmacology into {args.wide_output}")
        except Exception as exc:
            tlog.record("Merging pharmacology into wide output (FAILED)", time.perf_counter() - t0)
            print(f"Warning: pharma merge failed: {exc}", file=sys.stderr)

    if args.wide_output and os.path.exists(args.wide_output):
        t0 = time.perf_counter()
        values_only_path = args.wide_output.replace(".csv", "_values_only.csv")
        write_values_only_wide(args.wide_output, values_only_path)
        tlog.record("Writing values-only wide CSV", time.perf_counter() - t0)
        print(f"Wrote values-only wide output to {values_only_path}")

    tlog.finalize(compound_count=len(rows))
    print(f"Timing log written to {timing_log_path}")


if __name__ == "__main__":
    raise SystemExit(main())