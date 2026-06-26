"""Interactive menu and argument parsing for the thermo finder."""
import sys
import argparse

from ivette.util.paths import export_path, log_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Find NIST WebBook and PubMed thermo information")

    # Output defaults live under data/exports/ (never the repo root); the
    # interactive app overrides these with per-run paths.
    p.add_argument("--input", default=export_path("smiles.csv"))
    p.add_argument("--output", default=export_path("thermo_report.csv"))
    p.add_argument("--available-output", default=export_path("thermo_available.csv"))
    p.add_argument("--parsed-output", default=export_path("thermo_parsed.csv"))

    p.add_argument("--fetch-pharma", action="store_true", default=True)
    p.add_argument("--pharma-output", default=export_path("pharma_parsed.csv"))

    p.add_argument("--merge-pharma", action="store_true", default=True)
    p.add_argument("--merged-pharma-output", default=export_path("thermo_pharma_merged.csv"))

    p.add_argument("--chembl-target-cache", default="scripts/chembl_target_name_cache.json")

    p.add_argument("--cleaned-output", default=export_path("thermo_cleaned.csv"))
    p.add_argument("--summary-output", default=export_path("thermo_summary.csv"))
    p.add_argument("--ml-output", default=export_path("thermo_ml.csv"))
    p.add_argument("--rare-output", default=export_path("rare_properties.csv"))
    p.add_argument("--cleaning-report", default=export_path("cleaning_report.txt"))
    p.add_argument("--timing-log", default=log_path("timing_log.txt"),
                   help="Per-run timing log (default: data/logs/timing_log.txt)")

    p.add_argument("--wide-output", default=export_path("thermo_wide.csv"))
    p.add_argument("--wide-from-clean", action="store_true", default=True)

    p.add_argument("--max", type=int, default=40)
    p.add_argument("--pubmed-max", type=int, default=10)

    # Pharmacology tuning parameters
    p.add_argument("--pubchem-max-aids", type=int, default=20)
    p.add_argument("--chembl-activity-limit", type=int, default=500)
    p.add_argument("--chembl-max-pages", type=int, default=3)

    p.add_argument("--menu", action="store_true")

    return p


def interactive_parameter_menu(args: argparse.Namespace) -> argparse.Namespace:

    def prompt_string(prompt_text, current_value):
        new_value = input(f"{prompt_text} [{current_value}]: ").strip()
        return new_value if new_value else current_value

    def prompt_int(prompt_text, current_value):
        while True:
            new_value = input(f"{prompt_text} [{current_value}]: ").strip()

            if not new_value:
                return current_value

            try:
                return int(new_value)
            except ValueError:
                print("Please enter a valid integer or leave blank.")

    while True:

        print("\nInteractive parameter menu")
        print(" 1) Input CSV file:           ", args.input)
        print(" 2) Output CSV report:        ", args.output)
        print(" 3) Available-data CSV:       ", args.available_output)
        print(" 4) Parsed output CSV:        ", args.parsed_output)
        print(" 5) Cleaned output CSV:       ", args.cleaned_output)
        print(" 6) Wide output CSV:          ", args.wide_output)
        print(" 7) Wide from cleaned data:   ", args.wide_from_clean)
        print(" 8) Max rows to process:      ", args.max)
        print(" 9) PubMed max results:       ", args.pubmed_max)
        print("10) Fetch pharmacology:       ", args.fetch_pharma)
        print("11) Pharmacology output CSV:  ", args.pharma_output)
        print("12) Merge pharmacology:       ", args.merge_pharma)
        print("13) Merged pharma output CSV: ", args.merged_pharma_output)

        print("\n--- Pharmacology settings ---")
        print("14) PubChem max AIDs:         ", args.pubchem_max_aids)
        print("15) ChEMBL page size:         ", args.chembl_activity_limit)
        print("16) ChEMBL max pages:         ", args.chembl_max_pages)

        print("\n17) Run with current values")
        print("18) Quit")

        choice = input("Choose an option [17]: ").strip() or "17"

        actions = {

            "1": lambda: setattr(
                args,
                "input",
                prompt_string("Input CSV file", args.input)
            ),

            "2": lambda: setattr(
                args,
                "output",
                prompt_string("Output CSV report", args.output)
            ),

            "3": lambda: setattr(
                args,
                "available_output",
                prompt_string("Available-data CSV", args.available_output)
            ),

            "4": lambda: setattr(
                args,
                "parsed_output",
                prompt_string("Parsed output CSV", args.parsed_output)
            ),

            "5": lambda: setattr(
                args,
                "cleaned_output",
                prompt_string("Cleaned output CSV", args.cleaned_output)
            ),

            "6": lambda: setattr(
                args,
                "wide_output",
                prompt_string("Wide output CSV", args.wide_output)
            ),

            "7": lambda: (
                setattr(args, "wide_from_clean", not args.wide_from_clean),
                print(f"Wide from clean: {args.wide_from_clean}")
            ),

            "8": lambda: setattr(
                args,
                "max",
                prompt_int("Max rows to process", args.max)
            ),

            "9": lambda: setattr(
                args,
                "pubmed_max",
                prompt_int("PubMed max results", args.pubmed_max)
            ),

            "10": lambda: (
                setattr(args, "fetch_pharma", not args.fetch_pharma),
                print(f"Fetch pharmacology: {args.fetch_pharma}")
            ),

            "11": lambda: setattr(
                args,
                "pharma_output",
                prompt_string("Pharmacology output CSV", args.pharma_output)
            ),

            "12": lambda: (
                setattr(args, "merge_pharma", not args.merge_pharma),
                print(f"Merge pharmacology: {args.merge_pharma}")
            ),

            "13": lambda: setattr(
                args,
                "merged_pharma_output",
                prompt_string(
                    "Merged pharmacology output CSV",
                    args.merged_pharma_output
                )
            ),

            "14": lambda: setattr(
                args,
                "pubchem_max_aids",
                prompt_int(
                    "PubChem max AIDs",
                    args.pubchem_max_aids
                )
            ),

            "15": lambda: setattr(
                args,
                "chembl_activity_limit",
                prompt_int(
                    "ChEMBL page size",
                    args.chembl_activity_limit
                )
            ),

            "16": lambda: setattr(
                args,
                "chembl_max_pages",
                prompt_int(
                    "ChEMBL max pages",
                    args.chembl_max_pages
                )
            ),

            "17": lambda: None,

            "18": lambda: sys.exit(0),
        }

        if choice in actions:

            actions[choice]()

            if choice == "17":
                print("Running with current configuration.\n")
                return args

        else:
            print("Invalid choice.")