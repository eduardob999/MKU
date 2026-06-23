"""CSV writing helpers."""

import csv


def write_csv(path, fieldnames, rows):
    """Write ``rows`` (dicts) to ``path`` with a header of ``fieldnames``."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
