"""Journal of in-flight long-running runs, powering the main menu's Resume entry.

A run records itself here *before* its heavy work and clears itself *only after*
clean completion. So if the process is interrupted (Ctrl-C) or the machine crashes
(WSL out-of-disk, power loss), the entry is left behind — that leftover IS the
signal that a run needs resuming. Nothing here executes work; it just remembers
enough to re-enter the run, which then resumes via its own checkpoint (completed
compounds are skipped).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ivette.util.jsonstore import read_json, write_json
from ivette.util.paths import RESUME_FILE


def _load() -> dict:
    return read_json(RESUME_FILE, default={"runs": {}}) or {"runs": {}}


def _save(data: dict) -> None:
    Path(RESUME_FILE).parent.mkdir(parents=True, exist_ok=True)
    write_json(RESUME_FILE, data)


def record_run(key: str, record: dict) -> None:
    """Mark a run as in-flight under ``key`` (upsert). Call before the heavy work."""
    data = _load()
    record = dict(record)
    record.setdefault("started", datetime.now().isoformat(timespec="seconds"))
    data.setdefault("runs", {})[key] = record
    _save(data)


def clear_run(key: str) -> None:
    """Remove a run's journal entry. Call only after it completes cleanly."""
    data = _load()
    if key in data.get("runs", {}):
        del data["runs"][key]
        _save(data)


def active_runs() -> list[tuple[str, dict]]:
    """In-flight runs, oldest first — pruning any whose working dir is gone.

    Pruning is the "clean after itself" step for entries that can never resume
    (e.g. the geometry set was deleted).
    """
    data = _load()
    runs = data.get("runs", {})
    live, changed = {}, False
    for key, rec in runs.items():
        gd = rec.get("geometry_dir")
        if gd and not Path(gd).exists():
            changed = True
            continue
        live[key] = rec
    if changed:
        data["runs"] = live
        _save(data)
    return sorted(live.items(), key=lambda kv: kv[1].get("started", ""))
