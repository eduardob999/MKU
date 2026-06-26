"""Named per-stage parameter presets, persisted as JSON under ``data/presets/``.

One file per stage (the keys of :data:`ivette.core.params.STAGES`), mapping a
preset name to a serialised parameter dict. UI-agnostic — the editor and any
future web layer share this store.
"""

from __future__ import annotations

import json

from ivette.util.paths import PRESET_DIR


def _file(stage: str):
    return PRESET_DIR / f"{stage}.json"


def _read(stage: str) -> dict:
    path = _file(stage)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(stage: str, data: dict) -> None:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    _file(stage).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def list_presets(stage: str) -> "list[str]":
    """Preset names saved for ``stage``, alphabetically."""
    return sorted(_read(stage).keys())


def load_preset(stage: str, name: str):
    """The saved parameter dict for ``name``, or ``None`` if absent."""
    return _read(stage).get(name)


def save_preset(stage: str, name: str, params_dict: dict) -> None:
    """Create or overwrite preset ``name`` for ``stage``."""
    data = _read(stage)
    data[name] = params_dict
    _write(stage, data)


def delete_preset(stage: str, name: str) -> bool:
    """Remove preset ``name``; returns True if it existed."""
    data = _read(stage)
    if name not in data:
        return False
    del data[name]
    _write(stage, data)
    return True
