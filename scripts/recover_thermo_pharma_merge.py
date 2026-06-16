#!/usr/bin/env python3
"""
Recover the original thermo/pharma merged dataset.

Equivalent to the original find_thermo.py pipeline:

    thermo_parsed.csv
        |
        v
    build_wide_output()
        |
        v
    wide thermo CSV
        |
        v
    merge_pharma_into_wide()
        |
        v
    final thermo + pharma CSV

Inputs:
    thermo_parsed.csv
    thermo_pharma_merged.csv

Outputs:
    thermo_ml_recovered.csv
    thermo_ml_recovered_values_only.csv
    pharma_qc_review.csv
"""

import csv
import os
import re
import statistics


NUMERIC_VALUE_RE = re.compile(
    r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
)


# ============================================================
# Utilities
# ============================================================

def extract_numeric_value(value):

    text = (
        str(value).strip()
        if value is not None
        else ""
    )

    match = NUMERIC_VALUE_RE.search(text)

    try:
        return float(match.group(0)) if match else None
    except ValueError:
        return None



def compute_iqr(values):

    values = sorted(values)

    if len(values) < 2:
        return 0.0

    lower = values[:len(values)//2]

    upper = values[(len(values)+1)//2:]

    return (
        statistics.median(upper)
        -
        statistics.median(lower)
    )



def strip_property_value_unit(value, unit):

    if not value or not unit:
        return value

    suffix = f" {unit}"

    if value.endswith(suffix):

        return value[:-len(suffix)].strip()

    return value



# ============================================================
# Original build_wide_output()
# ============================================================

def build_wide_output(
        parsed_csv,
        output_csv
):

    wide_map = {}

    value_groups = {}

    seen_props = []

    identity = {}


    with open(parsed_csv, newline="") as fh:

        reader = csv.DictReader(fh)


        for row in reader:

            cid = row["CID"]

            prop = row["PropertyName"]

            unit = row.get(
                "PropertyUnit",
                ""
            )


            column_base = (
                f"{prop} ({unit})"
                if unit
                else prop
            )


            value = strip_property_value_unit(
                row.get("PropertyValue", ""),
                unit
            )


            # preserve identity before numeric filtering

            if cid not in identity:

                identity[cid] = {

                    "InChIKey": "",
                    "PubChemName": "",
                    "PubChem_URL": "",

                }


            for field in (
                "InChIKey",
                "PubChemName",
                "PubChem_URL",
            ):

                if (
                    not identity[cid][field]
                    and row.get(field)
                ):

                    identity[cid][field] = row[field]



            numeric = extract_numeric_value(
                value
            )


            if numeric is None:
                continue



            value_groups.setdefault(
                (cid, column_base),
                []
            ).append(
                numeric
            )


            if column_base not in seen_props:

                seen_props.append(
                    column_base
                )



    property_names = []


    for prop in seen_props:

        property_names.extend([

            f"{prop}_Median",
            f"{prop}_Count",
            f"{prop}_IQR",

        ])



    for (
        cid,
        prop
    ), values in value_groups.items():


        meta = identity.get(
            cid,
            {}
        )


        wide_map.setdefault(
            cid,
            {

                "CID": cid,

                "InChIKey":
                    meta.get("InChIKey",""),

                "PubChemName":
                    meta.get("PubChemName",""),

                "PubChem_URL":
                    meta.get("PubChem_URL",""),

            }
        )


        wide_map[cid].update({

            f"{prop}_Median":
                statistics.median(values),

            f"{prop}_Count":
                len(values),

            f"{prop}_IQR":
                compute_iqr(values),

        })



    fieldnames = [

        "CID",
        "InChIKey",
        "PubChemName",
        "PubChem_URL",

    ] + property_names



    with open(
        output_csv,
        "w",
        newline=""
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames
        )

        writer.writeheader()


        for row in wide_map.values():

            writer.writerow(row)



    print(
        f"Built thermo wide file: {output_csv}"
    )



# ============================================================
# Original merge_pharma_into_wide()
# ============================================================

def merge_pharma_into_wide(
        pharma_csv,
        wide_csv,
        qc_csv="pharma_qc_review.csv",
        relative_iqr_threshold=2.0
):


    def sanitize_label(text):

        return " ".join(
            str(text or "").split()
        )



    assay_groups = {}


    with open(pharma_csv,newline="") as fh:

        reader = csv.DictReader(fh)


        for r in reader:


            cid = (
                r.get("CID")
                or ""
            ).strip()


            if not cid:
                continue



            relation = (
                r.get("Relation")
                or ""
            ).strip()


            if relation in {
                ">",
                "<",
                ">=",
                "<="
            }:

                continue



            assay_id = (
                r.get("AssayID")
                or "NO_ASSAY"
            )



            source = (
                r.get("Source")
                or ""
            )



            target = (

                r.get("TargetName")
                or r.get("Target")
                or "Unchecked"

            )



            pchembl = r.get(
                "pChemblValue"
            )



            try:

                if pchembl not in ("",None):

                    value = float(
                        pchembl
                    )

                    feature = (

                        f"{source}:pActivity "
                        f"[{sanitize_label(target)}]"

                    )


                else:

                    value = float(
                        r.get("Value")
                    )


                    if value <= 0:

                        continue


                    feature = (

                        f"{source}:"
                        f"{r.get('ActivityType','')} "
                        f"[{sanitize_label(target)}]"

                    )


            except Exception:

                continue



            assay_groups.setdefault(
                (
                    cid,
                    feature,
                    assay_id
                ),
                []
            ).append(value)




    target_groups = {}

    qc_rows = []



    for (
        cid,
        feature,
        assay
    ), values in assay_groups.items():


        median = statistics.median(values)

        iqr = compute_iqr(values)


        relative = (

            iqr / abs(median)

            if median != 0

            else 0

        )



        if (
            len(values) >= 3
            and relative > relative_iqr_threshold
        ):


            qc_rows.append({

                "QCLevel": "ASSAY",
                "CID": cid,
                "FeatureName": feature,
                "AssayID": assay,
                "Count": len(values),
                "Median": median,
                "IQR": iqr,
                "RelativeIQR": relative,

            })


            continue



        target_groups.setdefault(
            (
                cid,
                feature
            ),
            []
        ).append(median)




    pharma_map = {}

    activity_columns = []



    for (
        cid,
        feature
    ), values in target_groups.items():


        median = statistics.median(values)

        iqr = compute_iqr(values)

        relative = (

            iqr / abs(median)

            if median != 0

            else 0

        )



        if (
            len(values) >= 3
            and relative > relative_iqr_threshold
        ):


            qc_rows.append({

                "QCLevel": "TARGET",
                "CID": cid,
                "FeatureName": feature,
                "Count": len(values),
                "Median": median,
                "IQR": iqr,
                "RelativeIQR": relative,

            })


            continue



        cols = {

            f"{feature}_Median":
                median,

            f"{feature}_Count":
                len(values),

            f"{feature}_IQR":
                iqr,

            f"{feature}_RelativeIQR":
                relative,

            f"{feature}_Min":
                min(values),

            f"{feature}_Max":
                max(values),

            f"{feature}_FoldRange":
                (
                    max(values)/min(values)
                    if min(values)>0
                    else ""
                ),

            f"{feature}_AssayCount":
                len(values),

        }


        for c in cols:

            if c not in activity_columns:

                activity_columns.append(c)



        pharma_map.setdefault(
            cid,
            {}
        ).update(cols)




    if qc_rows:

        with open(
            qc_csv,
            "w",
            newline=""
        ) as fh:


            writer = csv.DictWriter(
                fh,
                fieldnames=qc_rows[0].keys()
            )

            writer.writeheader()

            writer.writerows(qc_rows)



    with open(wide_csv,newline="") as fh:

        reader = csv.DictReader(fh)

        rows = list(reader)

        fields = list(
            reader.fieldnames
            or []
        )



    for row in rows:

        cid = row.get("CID","")

        pdata = pharma_map.get(
            cid,
            {}
        )


        for col,value in pdata.items():

            row[col] = value

            if col not in fields:

                fields.append(col)



    with open(
        wide_csv,
        "w",
        newline=""
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)



    print(
        "Merged pharmacology data"
    )



# ============================================================
# Original values-only writer
# ============================================================

def write_values_only_wide(
        wide_csv,
        output_csv
):

    with open(wide_csv,newline="") as fh:

        reader = csv.DictReader(fh)

        rows = list(reader)

        fields = list(
            reader.fieldnames
            or []
        )


    ids = [

        "CID",
        "InChIKey",
        "PubChemName",
        "PubChem_URL",

    ]


    median_cols = [

        f for f in fields

        if f.endswith("_Median")

    ]


    output_fields = (

        ids

        +

        [
            f[:-7]
            for f in median_cols
        ]

    )



    with open(
        output_csv,
        "w",
        newline=""
    ) as fh:


        writer = csv.DictWriter(
            fh,
            fieldnames=output_fields
        )

        writer.writeheader()



        for row in rows:

            out = {

                c:
                    row.get(c,"")

                for c in ids

            }


            for col in median_cols:

                out[
                    col[:-7]
                ] = row.get(
                    col,
                    ""
                )


            writer.writerow(out)



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":


    wide_file = (
        "thermo_ml_recovered.csv"
    )


    build_wide_output(
        "thermo_parsed.csv",
        wide_file
    )


    merge_pharma_into_wide(
        "thermo_pharma_merged.csv",
        wide_file
    )


    write_values_only_wide(
        wide_file,
        "thermo_ml_recovered_values_only.csv"
    )


    print()
    print("DONE")
    print(wide_file)
    print("thermo_ml_recovered_values_only.csv")