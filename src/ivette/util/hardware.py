"""Hardware detection and resource planning for Gaussian batch runs.

Gaussian's ``%nprocshared`` parallelism scales sub-linearly — past roughly
8 cores a single job wastes CPU. For a set of many molecules, total wall time
is minimized by running several jobs in parallel, each using a moderate core
count, with ``%mem`` sized so the parallel jobs don't oversubscribe RAM and swap.

:func:`recommend_gaussian_resources` turns the detected hardware (usable physical
cores + available memory, respecting CPU affinity and cgroup limits) into a
``(jobs, nproc, mem)`` plan. Detection is stdlib-only with graceful fallbacks;
all knobs are injectable so the planner can be unit-tested without real hardware.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from ivette.util.jsonstore import read_json, write_json
from ivette.util.paths import GAUSSIAN_BENCHMARK_FILE

# Gaussian per-job efficiency sweet spot and a hard cap on cores per job.
DEFAULT_SWEET_SPOT = 8
DEFAULT_MAX_NPROC_PER_JOB = 16
# A freq/opt job needs headroom; never propose less than this per job.
DEFAULT_MIN_MEM_PER_JOB_MB = 2048
# Fraction of available RAM to leave for the OS / scratch I/O.
DEFAULT_MEM_HEADROOM = 0.15
DEFAULT_BENCHMARK_THREADS = (4, 6, 7)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _read_int(path) -> int | None:
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _cgroup_cpu_limit() -> int | None:
    """CPU cores allowed by the cgroup quota (v2 then v1), or None if unlimited."""
    # cgroup v2: "<quota> <period>" or "max <period>"
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, math.floor(int(quota) / int(period)))
        return None
    except (OSError, ValueError):
        pass
    # cgroup v1
    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and quota > 0 and period and period > 0:
        return max(1, math.floor(quota / period))
    return None


def usable_cpu_count() -> int:
    """Logical CPUs this process may actually use (affinity ∩ cgroup quota)."""
    try:
        logical = len(os.sched_getaffinity(0))  # Linux: respects taskset/cgroup
    except AttributeError:
        logical = os.cpu_count() or 1
    limit = _cgroup_cpu_limit()
    if limit:
        logical = min(logical, limit)
    return max(1, logical)


def physical_core_count() -> int:
    """Best-effort physical core count, capped by what this process may use.

    Parses ``/proc/cpuinfo`` for unique (physical id, core id) pairs so that
    SMT/Hyper-Threading siblings aren't double-counted; falls back to the usable
    logical count when that information isn't available (non-Linux, WSL, etc.).
    """
    usable = usable_cpu_count()
    cores: set[tuple[str, str]] = set()
    try:
        phys = core = None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key == "physical id":
                phys = val
            elif key == "core id":
                core = val
            elif line.strip() == "" and phys is not None and core is not None:
                cores.add((phys, core))
                phys = core = None
        if phys is not None and core is not None:
            cores.add((phys, core))
    except OSError:
        pass
    physical = len(cores) if cores else usable
    return max(1, min(physical, usable))


def _meminfo_mb(key: str) -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1]) // 1024  # value is in kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def total_memory_mb() -> int | None:
    mb = _meminfo_mb("MemTotal:")
    if mb is not None:
        return mb
    try:
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) // (1024 * 1024)
    except (ValueError, OSError, AttributeError):
        return None


def available_memory_mb() -> int | None:
    """Currently-available RAM (cgroup-aware), falling back to total."""
    avail = _meminfo_mb("MemAvailable:")
    total = total_memory_mb()
    # Respect a cgroup memory ceiling (v2 then v1) if smaller than the host total.
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        limit = _read_int(path)
        if limit and 0 < limit < (1 << 62):  # ignore "max"/unlimited sentinels
            limit_mb = limit // (1024 * 1024)
            total = min(total, limit_mb) if total else limit_mb
            if avail is not None:
                avail = min(avail, limit_mb)
    return avail if avail is not None else total


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class GaussianResources:
    jobs: int           # parallel Gaussian jobs
    nproc: int          # %nprocshared per job
    mem: str            # %mem per job, e.g. "12GB"
    mem_mb: int         # mem per job in MB
    cores: int          # usable physical cores detected
    total_mem_mb: int | None
    available_mem_mb: int | None

    def summary(self) -> str:
        mem_total = f"{self.total_mem_mb // 1024}GB" if self.total_mem_mb else "unknown"
        return (
            "Hardware optimization:\n"
            f"  Detected cores      : {self.cores} usable physical\n"
            f"  Detected memory     : {mem_total} total\n"
            f"  -> Parallel jobs    : {self.jobs}\n"
            f"  -> Cores per job    : {self.nproc} (%nprocshared)\n"
            f"  -> Memory per job   : {self.mem} (%mem)\n"
            f"  -> Total core usage : {self.jobs * self.nproc}/{self.cores}"
        )


def benchmark_key(
    *,
    stage: str,
    cores: int,
    available_mem_mb: int | None,
    job_label: str,
    operation: str | None = None,
    fixed_threads: int | None = None,
    preopt: str | None = None,
) -> str:
    mem_part = "unknown" if available_mem_mb is None else str(available_mem_mb)
    parts = [f"stage={stage}", f"cores={cores}", f"mem={mem_part}", f"job={job_label}"]
    if operation is not None:
        parts.append(f"op={operation}")
    if fixed_threads is not None:
        parts.append(f"fixed_threads={fixed_threads}")
    if preopt is not None:
        parts.append(f"preopt={preopt}")
    return ";".join(parts)


def preopt_benchmark_key(*, cores: int, available_mem_mb: int | None, fixed_threads: int, job_label: str = "nitrobenzene") -> str:
    return benchmark_key(
        stage="preopt",
        cores=cores,
        available_mem_mb=available_mem_mb,
        fixed_threads=fixed_threads,
        job_label=job_label,
    )


def thread_benchmark_key(*, cores: int, available_mem_mb: int | None, preopt: str, job_label: str = "nitrobenzene") -> str:
    return benchmark_key(
        stage="threads",
        cores=cores,
        available_mem_mb=available_mem_mb,
        preopt=preopt,
        job_label=job_label,
    )


def load_benchmark_cache(path: str | Path = GAUSSIAN_BENCHMARK_FILE) -> dict:
    return read_json(path, default={"benchmarks": {}}) or {"benchmarks": {}}


def save_benchmark_cache(cache: dict, path: str | Path = GAUSSIAN_BENCHMARK_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(path, cache)


def get_cached_best_threads(key: str, *, path: str | Path = GAUSSIAN_BENCHMARK_FILE) -> int | None:
    cache = load_benchmark_cache(path)
    entry = cache.get("benchmarks", {}).get(key)
    if not entry:
        return None
    best = entry.get("best_threads", entry.get("best_value"))
    return int(best) if best is not None else None


def get_cached_benchmark_value(key: str, *, path: str | Path = GAUSSIAN_BENCHMARK_FILE):
    cache = load_benchmark_cache(path)
    entry = cache.get("benchmarks", {}).get(key)
    if not entry:
        return None
    return entry.get("best_value", entry.get("best_threads"))


def get_cached_benchmark_rows(key: str, *, path: str | Path = GAUSSIAN_BENCHMARK_FILE) -> list[dict] | None:
    cache = load_benchmark_cache(path)
    entry = cache.get("benchmarks", {}).get(key)
    if not entry:
        return None
    rows = entry.get("runs")
    return rows if isinstance(rows, list) else None


def store_benchmark_result(
    key: str,
    rows: list[dict],
    best_value,
    *,
    path: str | Path = GAUSSIAN_BENCHMARK_FILE,
) -> None:
    cache = load_benchmark_cache(path)
    cache.setdefault("benchmarks", {})[key] = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "best_value": best_value,
        "runs": rows,
    }
    if isinstance(best_value, int):
        cache["benchmarks"][key]["best_threads"] = best_value
    save_benchmark_cache(cache, path)


def benchmark_thread_plan(
    detected_cores: int,
    *,
    available_mem_mb: int | None = None,
    operation: str = "opt",
    job_label: str = "nitrobenzene",
) -> list[int]:
    key = benchmark_key(
        cores=detected_cores,
        available_mem_mb=available_mem_mb,
        operation=operation,
        job_label=job_label,
    )
    cached = get_cached_best_threads(key)
    if cached and 1 <= cached <= detected_cores:
        return [cached]

    candidates = [t for t in DEFAULT_BENCHMARK_THREADS if 1 <= t <= detected_cores]
    if detected_cores not in candidates:
        candidates.append(detected_cores)
    return sorted(dict.fromkeys(candidates))


def default_threads_after_benchmark(
    detected_cores: int,
    *,
    available_mem_mb: int | None = None,
    operation: str = "opt",
    job_label: str = "nitrobenzene",
) -> int:
    key = benchmark_key(
        cores=detected_cores,
        available_mem_mb=available_mem_mb,
        operation=operation,
        job_label=job_label,
    )
    cached = get_cached_best_threads(key)
    if cached and 1 <= cached <= detected_cores:
        return cached
    return max(1, round(detected_cores / DEFAULT_SWEET_SPOT))


def _format_mem(mb: int) -> str:
    gb = mb // 1024
    return f"{gb}GB" if gb >= 1 else f"{max(256, mb)}MB"


def recommend_gaussian_resources(
    n_tasks: int = 1,
    *,
    cores: int | None = None,
    available_mem_mb: int | None = None,
    sweet_spot: int = DEFAULT_SWEET_SPOT,
    max_nproc_per_job: int = DEFAULT_MAX_NPROC_PER_JOB,
    min_mem_per_job_mb: int = DEFAULT_MIN_MEM_PER_JOB_MB,
    mem_headroom: float = DEFAULT_MEM_HEADROOM,
) -> GaussianResources:
    """Plan ``(jobs, nproc, mem)`` for running ``n_tasks`` Gaussian jobs.

    For a single task one job gets all usable cores. For a batch we run
    ``jobs ≈ cores / sweet_spot`` parallel jobs (bounded by task count and RAM),
    redistribute leftover cores into ``nproc``, and split available memory across
    the jobs with headroom. ``cores`` / ``available_mem_mb`` may be supplied to
    override detection (used in tests).
    """
    detected_cores = physical_core_count() if cores is None else max(1, cores)
    total_mem = total_memory_mb()
    avail_mem = available_memory_mb() if available_mem_mb is None else available_mem_mb

    n_tasks = max(1, n_tasks)
    usable_mem = int(avail_mem * (1 - mem_headroom)) if avail_mem else None

    if n_tasks == 1:
        # A single molecule can't be parallelized across jobs — give it the box.
        jobs = 1
        nproc = detected_cores
    else:
        # Throughput rises as each job approaches the efficient core count, so
        # aim for jobs ≈ cores / sweet_spot (e.g. 14 cores -> 2 jobs of 7, which
        # beats 1 job of 14). Bound by the task count and available memory, then
        # spread leftover cores back into nproc.
        jobs = max(1, round(detected_cores / sweet_spot))
        jobs = min(jobs, n_tasks)
        if usable_mem:
            jobs = min(jobs, max(1, usable_mem // min_mem_per_job_mb))
        nproc = max(1, min(max_nproc_per_job, detected_cores // jobs))

    if usable_mem:
        mem_mb = max(min_mem_per_job_mb, usable_mem // jobs)
    else:
        mem_mb = min_mem_per_job_mb

    return GaussianResources(
        jobs=jobs,
        nproc=nproc,
        mem=_format_mem(mem_mb),
        mem_mb=mem_mb,
        cores=detected_cores,
        total_mem_mb=total_mem,
        available_mem_mb=avail_mem,
    )
