"""String, numeric, and iterable parsing helpers shared across modules."""

import re

_NUMERIC_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def chunked(iterable, size):
    """Yield successive ``size``-length slices of ``iterable``."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def extract_numeric(value, *, coerce=False):
    """Return the first numeric token in ``value`` as a float, else ``None``.

    With ``coerce=False`` (default) non-string input yields ``None`` — matching
    the long-format cleaning path. With ``coerce=True`` the value is
    ``str``-coerced first — matching the wide-output path.
    """
    if coerce:
        text = str(value).strip() if value is not None else ""
    elif isinstance(value, str) and value.strip():
        text = value
    else:
        return None
    match = _NUMERIC_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def slugify(text):
    """Filesystem-safe name for a target/column label."""
    return (
        str(text)
        .replace("/", "_")
        .replace(":", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "_")
    )


def collapse_ws(text):
    """Collapse all runs of whitespace in ``text`` to single spaces, trimmed."""
    return re.sub(r"\s+", " ", str(text).strip())
