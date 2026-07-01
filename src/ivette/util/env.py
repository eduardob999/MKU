"""Minimal ``.env`` loader (no third-party dependency).

Reads ``KEY=VALUE`` lines from the project's ``.env`` (gitignored) into the
process environment so credentials/config can live in one local file instead of
being passed on every command. Existing environment variables win unless
``override=True``, so a shell/CLI value can still take precedence.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]   # src/ivette/util/env.py → repo root


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict:
    """Load ``.env`` (repo root by default) into ``os.environ``; return what it set.

    Lines are ``KEY=VALUE``; blanks and ``#`` comments are ignored; surrounding
    single/double quotes on the value are stripped.
    """
    p = Path(path) if path else _repo_root() / ".env"
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
    return loaded
