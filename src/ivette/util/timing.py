"""Timing utilities: a console context-manager Timer and a file-backed log."""

import time


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


class TimingLog:
    """Accumulate partial timing entries and append them to a log file."""

    def __init__(self, path):
        self.path = path
        self.entries: list[tuple[str, float]] = []
        self._run_start = time.time()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"\n# Run started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            fh.write(f"{'label':<60}  {'seconds':>10}\n")
            fh.write("-" * 74 + "\n")

    def record(self, label, elapsed):
        # Append immediately so partial progress survives an interrupted run.
        self.entries.append((label, elapsed))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"{label:<60}  {elapsed:>10.3f}\n")

    def finalize(self, compound_count=0):
        total = time.time() - self._run_start
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write("-" * 74 + "\n")
            fh.write(f"{'TOTAL':<60}  {total:>10.3f}\n")
            if compound_count:
                rate = compound_count / total if total > 0 else 0.0
                fh.write(f"{'COMPOUNDS PROCESSED':<60}  {compound_count:>10}\n")
                fh.write(f"{'COMPOUNDS/SEC':<60}  {rate:>10.3f}\n")
