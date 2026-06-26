"""Timing utilities: a console context-manager Timer and a file-backed log.

Both mirror into the central logger (:mod:`ivette.util.applog`): chatty per-step
detail goes to DEBUG (kept out of the INFO log to avoid overlogging) and the
run summary to INFO, so timings show up in ``data/logs/ivette.log`` without any
extra wiring at the call sites.
"""

import os
import time

from ivette.util.applog import get_logger

_log = get_logger("timing")


class Timer:
    """Context manager that prints elapsed time for a named step."""

    def __init__(self, label):
        self.label = label
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        print(f"[START] {self.label} ...")
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        print(f"[DONE]  {self.label} — {elapsed:.1f}s")
        _log.debug("%s duration_s=%.3f", self.label, elapsed)


class TimingLog:
    """Accumulate partial timing entries and append them to a log file."""

    def __init__(self, path):
        self.path = path
        self.entries: list[tuple[str, float]] = []
        self._run_start = time.time()
        # Route under the requested directory; create it so per-run logs never
        # fail just because the run dir hasn't been made yet.
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"\n# Run started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            fh.write(f"{'label':<60}  {'seconds':>10}\n")
            fh.write("-" * 74 + "\n")
        _log.info("timing log started | file=%s", self.path)

    def record(self, label, elapsed):
        # Append immediately so partial progress survives an interrupted run.
        self.entries.append((label, elapsed))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"{label:<60}  {elapsed:>10.3f}\n")
        _log.debug("%s duration_s=%.3f", label, elapsed)

    def finalize(self, compound_count=0):
        total = time.time() - self._run_start
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write("-" * 74 + "\n")
            fh.write(f"{'TOTAL':<60}  {total:>10.3f}\n")
            if compound_count:
                rate = compound_count / total if total > 0 else 0.0
                fh.write(f"{'COMPOUNDS PROCESSED':<60}  {compound_count:>10}\n")
                fh.write(f"{'COMPOUNDS/SEC':<60}  {rate:>10.3f}\n")
        rate = compound_count / total if (compound_count and total > 0) else 0.0
        _log.info("dataset run complete duration_s=%.3f compounds=%d rate_per_s=%.3f",
                  total, compound_count, rate)
