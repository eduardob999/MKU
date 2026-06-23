"""JSON persistence helpers and a generic metadata store.

Replaces the five hand-written ``load_*_metadata`` / ``save_*_metadata`` /
``next_*_id`` / ``register_*`` triplets that previously lived in ``__main__``.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_json(path, default=None):
    """Return parsed JSON from ``path``, or ``default`` if it does not exist."""
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, data, *, indent=4):
    """Write ``data`` to ``path`` as indented JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)


def next_id(existing_keys, prefix, *, width=6):
    """Return the next ``{prefix}_{N:0{width}d}`` id given existing keys.

    Ids are expected to look like ``set_000007``; the numeric suffix after the
    first underscore is incremented past the current maximum.
    """
    numbers = [int(k.split("_")[1]) for k in existing_keys if "_" in k]
    nxt = max(numbers) + 1 if numbers else 1
    return f"{prefix}_{nxt:0{width}d}"


class MetadataStore:
    """A JSON file holding records in a dict under a single top-level key.

    Example::

        STRUCTURES = MetadataStore(STRUCTURE_METADATA_FILE, "sets", "set")
    """

    def __init__(self, path, root_key, id_prefix):
        self.path = Path(path)
        self.root_key = root_key
        self.id_prefix = id_prefix

    def ensure(self):
        """Create the parent directory and an empty metadata file if missing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            write_json(self.path, {self.root_key: {}})

    def load(self):
        return read_json(self.path, default={self.root_key: {}})

    def save(self, metadata):
        write_json(self.path, metadata)

    def records(self, metadata=None):
        md = metadata if metadata is not None else self.load()
        return md.get(self.root_key, {})

    def items(self):
        return self.records().items()

    def get(self, record_id):
        return self.records().get(record_id)

    def next_id(self, metadata=None):
        md = metadata if metadata is not None else self.load()
        return next_id(md.get(self.root_key, {}).keys(), self.id_prefix)

    def register(self, record, record_id=None):
        """Insert ``record`` under a fresh id (or ``record_id``); return the id."""
        md = self.load()
        bucket = md.setdefault(self.root_key, {})
        rid = record_id or next_id(bucket.keys(), self.id_prefix)
        bucket[rid] = record
        self.save(md)
        return rid

    def update(self, record_id, **changes):
        md = self.load()
        md[self.root_key][record_id].update(changes)
        self.save(md)

    def delete(self, record_id):
        md = self.load()
        existed = record_id in md.get(self.root_key, {})
        if existed:
            del md[self.root_key][record_id]
            self.save(md)
        return existed
