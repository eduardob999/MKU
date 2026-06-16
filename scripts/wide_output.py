"""Build wide-format CSV outputs and merge pharmacology data."""
import csv
import os
import re
import statistics


NUMERIC_VALUE_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def extract_numeric_value(value) -> float | None:
    text = str(value).strip() if value is not None else ""
    match = NUMERIC_VALUE_RE.search(text)
    try:
        return float(match.group(0)) if match else None
    except ValueError:
        return None


def compute_iqr(values: list[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n < 2:
        return 0.0
    lower = sorted_vals[: n // 2]
    upper = sorted_vals[(n + 1) // 2:]
    return statistics.median(upper) - statistics.median(lower)


def strip_property_value_unit(value: str, unit: str) -> str:
    if not value or not unit:
        return value
    suffix = f" {unit}"
    return value[: -len(suffix)].strip() if value.endswith(suffix) else value


def build_wide_output(parsed_rows: list[dict], wide_csv_path: str) -> None:
    wide_map = {}
    value_groups = {}
    seen_props = []
    identity = {}  # CID -> {InChIKey, PubChemName, PubChem_URL}

    for row in parsed_rows:
        cid = row["CID"]
        prop = row["PropertyName"]
        unit = row.get("PropertyUnit", "")
        column_base = f"{prop} ({unit})" if unit else prop
        value = strip_property_value_unit(row["PropertyValue"], unit)

        # ✅ Accumulate identity BEFORE skipping non-numeric rows
        if cid not in identity:
            identity[cid] = {"InChIKey": "", "PubChemName": "", "PubChem_URL": ""}
        for field in ("InChIKey", "PubChemName", "PubChem_URL"):
            if not identity[cid][field] and row.get(field):
                identity[cid][field] = row[field]

        numeric_value = extract_numeric_value(value)
        if numeric_value is None:
            continue  # safe to skip now — identity already captured

        value_groups.setdefault((cid, column_base), []).append(numeric_value)
        if column_base not in seen_props:
            seen_props.append(column_base)

    property_names = []
    for column_base in seen_props:
        property_names.extend([
            f"{column_base}_Median",
            f"{column_base}_Count",
            f"{column_base}_IQR",
        ])

    for (cid, column_base), values in value_groups.items():
        meta = identity.get(cid, {"InChIKey": "", "PubChemName": "", "PubChem_URL": ""})
        wide_map.setdefault(cid, {
            "CID": cid,
            "InChIKey": meta["InChIKey"],
            "PubChemName": meta["PubChemName"],
            "PubChem_URL": meta["PubChem_URL"],
        })
        wide_map[cid].update({
            f"{column_base}_Median": statistics.median(values),
            f"{column_base}_Count": len(values),
            f"{column_base}_IQR": compute_iqr(values),
        })

    fieldnames = ["CID", "InChIKey", "PubChemName", "PubChem_URL"] + property_names
    with open(wide_csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in wide_map.values():
            writer.writerow(row)


def merge_pharma_into_wide(
    merged_pharma_csv: str,
    wide_csv: str,
    qc_review_csv: str = "pharma_qc_review.csv",
    relative_iqr_threshold: float = 2.0,
) -> None:

    if not os.path.exists(merged_pharma_csv):
        raise FileNotFoundError(merged_pharma_csv)

    if not os.path.exists(wide_csv):
        raise FileNotFoundError(wide_csv)

    def sanitize_label(text: str) -> str:
        return " ".join(str(text or "").split())

    # =========================================================
    # Stage 1
    # Raw measurements grouped by assay
    # =========================================================

    assay_groups = {}

    with open(merged_pharma_csv, newline="") as fh:

        reader = csv.DictReader(fh)

        for r in reader:

            cid = (r.get("CID") or "").strip()

            if not cid:
                continue

            relation = (r.get("Relation") or "").strip()

            # skip censored measurements
            if relation in {">", "<", ">=", "<="}:
                continue

            assay_id = (
                r.get("AssayID")
                or "NO_ASSAY"
            ).strip()

            source = (
                r.get("Source")
                or ""
            ).strip()

            target_name = (
                r.get("TargetName")
                or ""
            ).strip()

            target = (
                r.get("Target")
                or ""
            ).strip()

            activity_type = (
                r.get("ActivityType")
                or ""
            ).strip()

            target_label = (
                target_name
                or target
                or "Unchecked"
            )

            pchembl = r.get("pChemblValue")

            try:
                pchembl = (
                    float(pchembl)
                    if pchembl not in ("", None)
                    else None
                )
            except Exception:
                pchembl = None

            if pchembl is not None:

                value = pchembl

                feature_name = (
                    f"{source}:pActivity "
                    f"[{sanitize_label(target_label)}]"
                )

            else:

                raw_value = r.get("Value")

                try:
                    value = float(raw_value)
                except Exception:
                    continue

                if value <= 0:
                    continue

                feature_name = (
                    f"{source}:{activity_type} "
                    f"[{sanitize_label(target_label)}]"
                )

            key = (
                cid,
                feature_name,
                assay_id,
            )

            assay_groups.setdefault(
                key,
                []
            ).append(value)

    # =========================================================
    # Stage 2
    # Collapse replicates within assay
    # =========================================================

    target_groups = {}
    qc_rows = []

    for (
        cid,
        feature_name,
        assay_id,
    ), values in assay_groups.items():

        values = sorted(values)

        assay_median = statistics.median(values)
        assay_iqr = compute_iqr(values)

        assay_relative_iqr = (
            assay_iqr / abs(assay_median)
            if assay_median != 0
            else 0
        )

        # QC at assay level
        if (
            len(values) >= 3
            and assay_relative_iqr > relative_iqr_threshold
        ):

            qc_rows.append({
                "QCLevel": "ASSAY",
                "CID": cid,
                "FeatureName": feature_name,
                "WideColumn": f"{feature_name}_Median",
                "AssayID": assay_id,
                "Count": len(values),
                "Median": assay_median,
                "IQR": assay_iqr,
                "RelativeIQR": assay_relative_iqr,
                "Min": min(values),
                "Max": max(values),
                "FoldRange": (
                    max(values) / min(values)
                    if min(values) > 0 else ""
                ),
                "Values": ";".join(
                    str(v) for v in values
                ),
            })

            continue

        target_groups.setdefault(
            (cid, feature_name),
            []
        ).append(assay_median)

    # =========================================================
    # Stage 3
    # QC between assays
    # =========================================================

    pharma_map = {}
    activity_columns = []

    for (
        cid,
        feature_name,
    ), assay_medians in target_groups.items():

        assay_medians = sorted(assay_medians)

        median_val = statistics.median(
            assay_medians
        )

        iqr_val = compute_iqr(
            assay_medians
        )

        relative_iqr = (
            iqr_val / abs(median_val)
            if median_val != 0
            else 0
        )

        # NEW:
        # Reject heterogeneous targets even if
        # individual assays looked clean.
        if (
            len(assay_medians) >= 3
            and relative_iqr > relative_iqr_threshold
        ):

            qc_rows.append({
                "QCLevel": "TARGET",
                "CID": cid,
                "FeatureName": feature_name,
                "WideColumn": f"{feature_name}_Median",
                "AssayID": "",
                "Count": len(assay_medians),
                "Median": median_val,
                "IQR": iqr_val,
                "RelativeIQR": relative_iqr,
                "Min": min(assay_medians),
                "Max": max(assay_medians),
                "FoldRange": (
                    max(assay_medians) /
                    min(assay_medians)
                    if min(assay_medians) > 0
                    else ""
                ),
                "Values": ";".join(
                    str(v)
                    for v in assay_medians
                ),
            })

            continue

        min_val = min(assay_medians)
        max_val = max(assay_medians)

        cols = {

            f"{feature_name}_Median":
                median_val,

            f"{feature_name}_Count":
                len(assay_medians),

            f"{feature_name}_IQR":
                iqr_val,

            f"{feature_name}_RelativeIQR":
                relative_iqr,

            f"{feature_name}_Min":
                min_val,

            f"{feature_name}_Max":
                max_val,

            f"{feature_name}_FoldRange":
                (
                    max_val / min_val
                    if min_val > 0
                    else ""
                ),

            f"{feature_name}_AssayCount":
                len(assay_medians),
        }

        for col in cols:

            if col not in activity_columns:
                activity_columns.append(col)

        pharma_map.setdefault(
            cid,
            {}
        ).update(cols)

    # =========================================================
    # Write QC report
    # =========================================================

    if qc_rows:

        qc_fields = [
            "QCLevel",
            "CID",
            "FeatureName",
            "WideColumn",
            "AssayID",
            "Count",
            "Median",
            "IQR",
            "RelativeIQR",
            "Min",
            "Max",
            "FoldRange",
            "Values",
        ]

        with open(
            qc_review_csv,
            "w",
            newline=""
        ) as fh:

            writer = csv.DictWriter(
                fh,
                fieldnames=qc_fields,
            )

            writer.writeheader()

            for row in qc_rows:
                writer.writerow(row)

    # =========================================================
    # Merge into wide file
    # =========================================================

    with open(wide_csv, newline="") as fh:

        reader = csv.DictReader(fh)

        wide_fieldnames = list(
            reader.fieldnames or []
        )

        rows = list(reader)

    for col in activity_columns:

        if col not in wide_fieldnames:
            wide_fieldnames.append(col)

    for row in rows:

        cid = (
            row.get("CID")
            or row.get("cid")
            or ""
        )

        pdata = pharma_map.get(
            cid,
            {}
        )

        for col in activity_columns:
            row[col] = pdata.get(
                col,
                ""
            )

    with open(
        wide_csv,
        "w",
        newline=""
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=wide_fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        f"[QC] {len(qc_rows)} rejected assay/target groups "
        f"written to {qc_review_csv}"
    )


def write_values_only_wide(wide_csv: str, values_only_csv: str) -> None:
    """Write a copy of the wide CSV with only _Median columns, renamed to drop the suffix."""
    with open(wide_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    id_cols = ["CID", "InChIKey", "PubChemName", "PubChem_URL"]
    value_cols = [f for f in fieldnames if f.endswith("_Median")]
    renamed = {f: f[: -len("_Median")] for f in value_cols}

    out_fieldnames = id_cols + [renamed[f] for f in value_cols]

    with open(values_only_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {col: row.get(col, "") for col in id_cols}
            for col in value_cols:
                out_row[renamed[col]] = row.get(col, "")
            writer.writerow(out_row)