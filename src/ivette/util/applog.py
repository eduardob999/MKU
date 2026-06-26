"""Central application logging and per-run timing for Ivette.

Goals:
  * One rotating log (``data/logs/ivette.log``) captures the *important* events
    and their timings; a second ``errors.log`` keeps only failures for quick
    triage. Every record is ISO-8601 timestamped — time is metadata on every line.
  * The console handler is deliberately quiet (ERROR+) so the Rich/questionary
    TUI is never spammed; routine progress and warnings live in the file only,
    while modules keep their own ``print``/``ui`` calls for live feedback.
  * Avoid overlogging: callers log milestones (run start/finish, per-compound
    success/failure, benchmark winners) at INFO, and chatty per-item detail at
    DEBUG, which the INFO file handler drops unless you raise the level.
  * Long operations also drop a machine-readable ``timing.json`` into their own
    output directory via :class:`RunTiming`, so per-run timing travels with the
    run's data instead of a global file.

Use the ``ivette.*`` logger tree exclusively (via :func:`get_logger`) so this
module's handlers — and only these — handle Ivette's records.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from contextlib import contextmanager
from pathlib import Path

from ivette.util.paths import LOG_DIR

_CONFIGURED = False
_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_ROOT_NAME = "ivette"


def configure(*, level: int = logging.INFO,
              console_level: int = logging.ERROR,
              force: bool = False) -> logging.Logger:
    """Install the file/error/console handlers on the ``ivette`` logger once.

    Idempotent: repeated calls are no-ops unless ``force`` is given. Returns the
    configured ``ivette`` root logger.
    """
    global _CONFIGURED
    root = logging.getLogger(_ROOT_NAME)
    if _CONFIGURED and not force:
        return root

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root.setLevel(logging.DEBUG)      # handlers decide what is actually emitted
    root.propagate = False            # don't leak into the stdlib root logger
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "ivette.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "errors.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)
    root.addHandler(error_handler)

    console = logging.StreamHandler()   # stderr; kept quiet so the TUI stays clean
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    _CONFIGURED = True
    root.info("logging configured | file=%s level=%s console=%s",
              LOG_DIR / "ivette.log",
              logging.getLevelName(level), logging.getLevelName(console_level))
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``ivette`` logger, e.g. ``get_logger("gaussian")``."""
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def _fields(fields: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)


@contextmanager
def log_step(logger, label: str, **fields):
    """Time a step and log exactly one line for it.

    Emits an INFO record on success (carrying ``duration_s`` and any extra
    ``fields``) or an ERROR record with traceback on failure, then re-raises.
    ``logger`` may be a :class:`logging.Logger` or a short name string.
    """
    if isinstance(logger, str):
        logger = get_logger(logger)
    extra = _fields(fields)
    suffix = f" {extra}" if extra else ""
    start = time.perf_counter()
    logger.debug("start: %s%s", label, suffix)
    try:
        yield
    except Exception:
        dur = time.perf_counter() - start
        logger.exception("FAILED: %s duration_s=%.3f%s", label, dur, suffix)
        raise
    dur = time.perf_counter() - start
    logger.info("%s duration_s=%.3f%s", label, dur, suffix)


class RunTiming:
    """Accumulate per-step timings for a single run and persist them as JSON.

    Writes ``<run_dir>/timing.json`` and updates it after every step so partial
    progress survives an interrupt. Per-step lines are also mirrored to the
    central log at DEBUG (kept out of the INFO file to avoid overlogging); the
    final :meth:`summary` is logged at INFO.
    """

    def __init__(self, run_dir, *, run_label: str = "", logger=None):
        self.path = Path(run_dir) / "timing.json"
        self.run_label = run_label
        self.logger = logger or get_logger("timing")
        self._start = time.time()
        self.steps: list[dict] = []
        self._flush()

    def _flush(self) -> None:
        data = {
            "run_label": self.run_label,
            "started": time.strftime(_DATEFMT, time.localtime(self._start)),
            "elapsed_s": round(time.time() - self._start, 3),
            "steps": self.steps,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass   # timing is best-effort; never break a run over it

    def record(self, label: str, seconds: float, **fields) -> None:
        self.steps.append({
            "label": label,
            "seconds": round(seconds, 3),
            "at": time.strftime(_DATEFMT, time.localtime()),
            **fields,
        })
        self._flush()
        prefix = f"{self.run_label}: " if self.run_label else ""
        self.logger.debug("%s%s duration_s=%.3f %s", prefix, label, seconds, _fields(fields))

    @contextmanager
    def step(self, label: str, **fields):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(label, time.perf_counter() - start, **fields)

    def summary(self, **fields) -> None:
        total = time.time() - self._start
        self._flush()
        self.logger.info("%s complete duration_s=%.3f steps=%d %s",
                         self.run_label or "run", total, len(self.steps), _fields(fields))
