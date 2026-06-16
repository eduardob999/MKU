#!/usr/bin/env python3
"""
download_physchem.py

Download physicochemical properties from PubChem for compounds matching
a given substructure (SMILES).

Usage examples:
  python scripts/download_physchem.py --smiles "c1ccccc1" --max 200 --output benzene_matches.csv

Notes:
  - This tool queries PubChem PUG-REST. It returns CIDs for the substructure
    search and then fetches the requested properties in batches.
  - It is intentionally lightweight and depends only on `requests` and `pandas`.
"""
import argparse
import time
import requests
import sys
from urllib.parse import quote
import csv
from rdkit import Chem

NITRO_ZWITTER_SMARTS = Chem.MolFromSmarts("[c,C][N+](=O)[O-]")  # nitro on any carbon

def is_nitro_zwitterion(smiles: str) -> bool:
    """True if molecule contains at least one nitro group and has zero net charge."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if "." in smiles:               # exclude salts and mixtures
        return False
    net_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if net_charge != 0:             # exclude anything with residual charge
        return False
    return mol.HasSubstructMatch(NITRO_ZWITTER_SMARTS)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

DEFAULT_PROPERTIES = [
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "Complexity",
    "SMILES",
    "InChIKey",
]


def interleave_rows(all_rows_by_sub: list[list[dict]]) -> list[dict]:
    """Round-robin interleave rows from each substructure.
    
    Substructures with fewer matches exhaust first; the remaining ones
    continue cycling until all rows are placed.
    """
    from itertools import zip_longest
    result = []
    sentinel = object()
    for group in zip_longest(*all_rows_by_sub, fillvalue=sentinel):
        for item in group:
            if item is not sentinel:
                result.append(item)
    return result


def get_cids_for_substructure(smiles, max_records=1000, max_retries=5, use_smarts=False):
    input_type = "smarts" if use_smarts else "smiles"
    submit_url = f"{PUBCHEM_BASE}/compound/substructure/{input_type}/JSON"
    # Convert SMILES nitro-imidazole cores to SMARTS that match both NH and N-substituted
    # e.g. c1c[nH]cn1 → c1c[nH,n]cn1 so metronidazole-like compounds are included
    query = smiles.replace("[nH]", "[nH,n]") if not use_smarts else smiles

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                submit_url,
                data={"smarts" if use_smarts else "smiles": query},
                params={"MaxRecords": str(max_records)},
                timeout=30,
            )
            break
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = min(4 * (attempt + 1), 30)
            print(f"  POST connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}",
                  file=sys.stderr)
            time.sleep(wait)
    else:
        raise requests.exceptions.ConnectionError(
            f"Failed to submit substructure search after {max_retries} attempts"
        )

    if resp.status_code == 202:
        data = resp.json()
        listkey = data.get("Waiting", {}).get("ListKey")
        if not listkey:
            raise requests.HTTPError(f"No ListKey in response: {data}")
        poll_url = f"{PUBCHEM_BASE}/compound/listkey/{quote(listkey, safe='')}/cids/JSON"
        for attempt in range(30):
            try:
                r = requests.get(poll_url, params={"MaxRecords": str(max_records)}, timeout=30)
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                print(f"  Poll connection error (attempt {attempt + 1}/30), retrying: {e}",
                      file=sys.stderr)
                time.sleep(2)
                continue
            if r.status_code == 200:
                cids = r.json().get("IdentifierList", {}).get("CID", [])
                return cids[:max_records]
            if r.status_code not in (202, 503):
                r.raise_for_status()
            time.sleep(0.5 + attempt * 0.2)
        raise requests.HTTPError("Timed out waiting for PubChem substructure search results")
    else:
        resp.raise_for_status()
        cids = resp.json().get("IdentifierList", {}).get("CID", [])
        return cids[:max_records]


def fetch_properties_for_cids(cids, properties, max_retries=5):
    """Fetch properties for a list of CIDs. Returns list of dicts.
    Retries on connection errors and 5xx responses with exponential backoff.
    Splits and retries smaller batches on 400 errors.
    """
    if not cids:
        return []

    prop_str = ",".join(properties)
    cid_str = ",".join(str(int(x)) for x in cids)
    url = f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/{prop_str}/JSON"

    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json().get("PropertyTable", {}).get("Properties", [])

        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            # Network-level failure (RemoteDisconnected, connection aborted, etc.)
            wait = 2 ** attempt
            print(f"  Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}",
                  file=sys.stderr)
            time.sleep(wait)

        except requests.HTTPError:
            if resp is not None and resp.status_code == 400 and len(cids) > 1:
                # Bad request — split batch and retry halves
                mid = len(cids) // 2
                return (fetch_properties_for_cids(cids[:mid], properties, max_retries) +
                        fetch_properties_for_cids(cids[mid:], properties, max_retries))
            elif resp is not None and resp.status_code in (429, 500, 502, 503, 504):
                # Rate limited or server error — back off and retry
                wait = 2 ** attempt
                print(f"  HTTP {resp.status_code} (attempt {attempt + 1}/{max_retries}), retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                raise

    raise requests.exceptions.ConnectionError(
        f"Failed to fetch properties after {max_retries} attempts for {len(cids)} CIDs"
    )


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def main(argv=None):
    p = argparse.ArgumentParser(description="Download physchem properties from PubChem by substructure (SMILES)")
    p.add_argument("--smiles", help="Substructure as a SMILES string (required unless --input-file used)")
    p.add_argument("--input-file", default="heterocycles.csv", help="File with one SMILES per line, or a CSV with SMILES in the first column")
    p.add_argument("--prepend", nargs="+", default=[], metavar="SMILES",
                   help="Exact SMILES to look up and prepend to the output (e.g. --prepend 'c1ccccc1' 'CCO')")
    p.add_argument("--max", type=int, default=None, help="Max records to retrieve per substructure (default: 500)")
    p.add_argument("--properties", nargs="+", default=None, help="List of properties to fetch (space separated)")
    p.add_argument("--batch-size", type=int, default=None, help="Number of CIDs per properties request (default: 100)")
    p.add_argument("--sleep", type=float, default=None, help="Seconds to sleep between requests (default: 0.2)")
    p.add_argument("--output", default="smiles.csv", help="Output CSV filename")
    p.add_argument("--limit-substructures", type=int, default=None, help="Only process first N substructures from input file")
    p.add_argument("--no-menu", action="store_true", help="Skip interactive menu and use defaults/CLI args as-is")
    args = p.parse_args(argv)

    # ------------------------------------------------------------------ #
    #  Interactive menu — skipped if --no-menu or if stdin is not a tty   #
    # ------------------------------------------------------------------ #
    if not args.no_menu and sys.stdin.isatty():
        print("\n=== PubChem Physchem Downloader ===\n")
        print("Press Enter to accept the value shown in [brackets].\n")

        def ask(prompt, default, cast=str):
            while True:
                raw = input(f"  {prompt} [{default}]: ").strip()
                if raw == "":
                    return cast(default)
                try:
                    return cast(raw)
                except ValueError:
                    print(f"    ! Expected {cast.__name__}, got: {raw!r}")

        def ask_yn(prompt, default=True):
            hint = "Y/n" if default else "y/N"
            raw = input(f"  {prompt} [{hint}]: ").strip().lower()
            if raw == "":
                return default
            return raw in ("y", "yes")

        # --- input source ---
        if not args.smiles and not args.input_file:
            use_file = ask_yn("Load SMILES from a file? (No = enter a single SMILES string)", default=True)
            if use_file:
                args.input_file = ask("Path to input file (plain .txt or .csv)", "input_smiles.txt")
            else:
                args.smiles = ask("SMILES string", "c1ccccc1")
        else:
            src = args.input_file or args.smiles
            print(f"  Input source : {src}")

        # --- prepend compounds ---
        if not args.prepend:
            add_prepend = ask_yn("Prepend specific compounds to the top of the output?", default=False)
            if add_prepend:
                print("  Enter SMILES separated by spaces (quote each one if they contain spaces).")
                raw_prepend = input("  Prepend SMILES: ").strip()
                args.prepend = raw_prepend.split() if raw_prepend else []

        # --- output file ---
        if args.output is None:
            args.output = ask("Output CSV filename", "pubchem_physchem.csv")

        # --- max records ---
        if args.max is None:
            args.max = ask("Max records per substructure", 500, int)

        # --- limit substructures ---
        if args.input_file and args.limit_substructures is None:
            limit_raw = input("  Process only first N substructures? (Enter to process all): ").strip()
            args.limit_substructures = int(limit_raw) if limit_raw else None

        # --- batch size ---
        if args.batch_size is None:
            args.batch_size = ask("CIDs per property-fetch batch", 100, int)

        # --- sleep ---
        if args.sleep is None:
            args.sleep = ask("Sleep between requests (seconds)", 0.2, float)

        # --- properties ---
        if args.properties is None:
            print(f"\n  Default properties: {', '.join(DEFAULT_PROPERTIES)}")
            keep_defaults = ask_yn("Use default properties?", default=True)
            if keep_defaults:
                args.properties = DEFAULT_PROPERTIES
            else:
                print("  Enter property names separated by spaces.")
                print("  Valid names: MolecularWeight, XLogP, TPSA, HBondDonorCount,")
                print("               HBondAcceptorCount, RotatableBondCount, Complexity,")
                print("               SMILES, InChIKey, MolecularFormula, Charge, ... (any PubChem property)")
                raw_props = input("  Properties: ").strip()
                args.properties = raw_props.split() if raw_props else DEFAULT_PROPERTIES

        # --- confirmation ---
        print("\n--- Run parameters ---")
        print(f"  Input        : {args.input_file or args.smiles}")
        print(f"  Prepend      : {args.prepend or 'none'}")
        print(f"  Output       : {args.output}")
        print(f"  Max records  : {args.max}")
        print(f"  Limit subs   : {args.limit_substructures or 'all'}")
        print(f"  Batch size   : {args.batch_size}")
        print(f"  Sleep        : {args.sleep}s")
        print(f"  Properties   : {', '.join(args.properties)}")
        print()
        if not ask_yn("Proceed?", default=True):
            print("Aborted.")
            return 0
        print()

    # ------------------------------------------------------------------ #
    #  Apply defaults for anything still unset (--no-menu / non-tty)      #
    # ------------------------------------------------------------------ #
    if args.max is None:        args.max = 500
    if args.batch_size is None: args.batch_size = 100
    if args.sleep is None:      args.sleep = 0.2
    if args.output is None:     args.output = "pubchem_physchem.csv"
    if args.properties is None: args.properties = DEFAULT_PROPERTIES

    # ------------------------------------------------------------------ #
    #  Fetch prepended compounds via exact-structure lookup                #
    # ------------------------------------------------------------------ #
    prepend_rows = []
    for smiles in args.prepend:
        print(f"Looking up prepend compound: {smiles}")
        url = f"{PUBCHEM_BASE}/compound/smiles/{quote(smiles, safe='')}/property/{','.join(args.properties)}/JSON"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            rows = resp.json().get("PropertyTable", {}).get("Properties", [])
            for r in rows:
                r.setdefault("QuerySubstructure", smiles)
            prepend_rows.extend(rows)
            print(f"  Found {len(rows)} record(s)")
        except requests.HTTPError as e:
            print(f"  Warning: could not fetch prepend compound {smiles!r}: {e}", file=sys.stderr)
        time.sleep(args.sleep)

    # ------------------------------------------------------------------ #
    #  Validate input source                                               #
    # ------------------------------------------------------------------ #
    subs = []
    if args.input_file:
        with open(args.input_file, "r") as fh:
            for line in fh:
                s = line.strip().split(",")[0]
                if s and s != "SMILES":
                    subs.append(s)
        if not subs:
            p.error(f"No SMILES found in '{args.input_file}' — is it the right file?")
        if args.limit_substructures:
            subs = subs[:args.limit_substructures]
    elif args.smiles:
        subs = [args.smiles]
    else:
        p.error("either --smiles or --input-file is required")

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    rows_by_sub = []                        # list of lists, one per substructure
    for sub in subs:
        print(f"Searching PubChem for substructure: {sub}")
        try:
            cids = get_cids_for_substructure(sub, max_records=args.max)
        except (requests.HTTPError, requests.exceptions.ConnectionError) as e:
            print(f"  Skipping {sub}: {e}", file=sys.stderr)
            continue
        print(f"Found {len(cids)} CIDs (lim {args.max})")
        sub_rows = []
        for batch in chunked(cids, args.batch_size):
            try:
                props = fetch_properties_for_cids(batch, args.properties)
            except (requests.HTTPError, requests.exceptions.ConnectionError) as e:
                print(f"Property fetch error for batch of {len(batch)}: {e}", file=sys.stderr)
                time.sleep(args.sleep)
                continue
            for r in props:
                smi = r.get("SMILES", "")
                if not is_nitro_zwitterion(smi):
                    continue
                r.setdefault("QuerySubstructure", sub)
                sub_rows.append(r)
            time.sleep(args.sleep)

        # Sort by molecular weight ascending after properties are known
        sub_rows.sort(key=lambda r: float(r.get("MolecularWeight", 0) or 0))

        if sub_rows:
            rows_by_sub.append(sub_rows)

    if not rows_by_sub and not prepend_rows:
        print("No properties retrieved. Exiting.")
        return 1

    all_rows = interleave_rows(rows_by_sub)

    # ------------------------------------------------------------------ #
    #  Merge: prepend rows first, then deduplicate by CID                 #
    # ------------------------------------------------------------------ #
    combined = prepend_rows + all_rows

    fieldnames = []
    seen = set()
    for r in combined:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    if "CID" in fieldnames:
        seen_cids = set()
        unique_rows = []
        for r in combined:
            cid = r.get("CID")
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            unique_rows.append(r)
        combined = unique_rows

    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in combined:
            writer.writerow(r)

    print(f"Saved {len(combined)} rows to {args.output} ({len(prepend_rows)} prepended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
