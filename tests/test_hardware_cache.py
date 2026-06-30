"""Gaussian hardware-benchmark cache: persistence + key stability.

Regression guard for the bug where the benchmark re-ran every time because the
cache key included currently-available RAM (which fluctuates). The fix keys on
total memory, so the stored result is reused.
"""

from ivette.util import hardware


def test_benchmark_cache_round_trips(tmp_path):
    p = tmp_path / "bench.json"
    key = hardware.preopt_benchmark_key(cores=8, available_mem_mb=32000, fixed_threads=4)

    assert hardware.get_cached_benchmark_rows(key, path=p) is None   # cold
    rows = [{"preopt_mode": "pm7", "total_seconds": 12.0, "success": True}]
    hardware.store_benchmark_result(key, rows, "pm7", path=p)
    assert hardware.get_cached_benchmark_rows(key, path=p) == rows    # reused

    ckey = hardware.thread_benchmark_key(cores=8, available_mem_mb=32000, preopt="winner")
    hardware.store_benchmark_result(ckey, [{"threads": 7, "success": True}], 7, path=p)
    assert hardware.get_cached_best_threads(ckey, path=p) == 7


def test_store_self_heals_stale_memory_variants(tmp_path):
    p = tmp_path / "bench.json"
    # Same signature, different memory token (the old free-RAM bug).
    k_old = "stage=preopt;cores=14;mem=24717;job=nitrobenzene;fixed_threads=7"
    k_new = "stage=preopt;cores=14;mem=28061;job=nitrobenzene;fixed_threads=7"
    hardware.store_benchmark_result(k_old, [{"a": 1}], "pm7", path=p)
    hardware.store_benchmark_result(k_new, [{"a": 2}], "pm7", path=p)
    cache = hardware.load_benchmark_cache(p)["benchmarks"]
    assert set(cache) == {k_new}            # stale free-RAM variant dropped


def test_prune_collapses_duplicates_keeping_newest(tmp_path):
    p = tmp_path / "bench.json"
    import json
    json.dump({"benchmarks": {
        "stage=preopt;cores=14;mem=24717;job=nitrobenzene;fixed_threads=7":
            {"updated": "2026-06-30T14:26:13", "runs": []},
        "stage=preopt;cores=14;mem=28061;job=nitrobenzene;fixed_threads=7":
            {"updated": "2026-06-30T14:50:47", "runs": []},
        "stage=threads;cores=14;mem=28061;job=nitrobenzene;preopt=winner":
            {"updated": "2026-06-30T14:53:32", "runs": []},
    }}, open(p, "w"))
    removed = hardware.prune_benchmark_cache(path=p)
    assert removed == 1
    kept = set(hardware.load_benchmark_cache(p)["benchmarks"])
    assert kept == {
        "stage=preopt;cores=14;mem=28061;job=nitrobenzene;fixed_threads=7",
        "stage=threads;cores=14;mem=28061;job=nitrobenzene;preopt=winner",
    }


def test_benchmark_key_stable_with_total_mem_but_varies_with_free_mem():
    # Same inputs → identical key (so a cached run is found next time).
    k = dict(cores=8, available_mem_mb=32000, fixed_threads=4)
    assert hardware.preopt_benchmark_key(**k) == hardware.preopt_benchmark_key(**k)
    # Different memory → different key. This is exactly why passing *available*
    # (fluctuating) RAM defeated the cache; the call site now passes total RAM.
    assert (hardware.preopt_benchmark_key(**k)
            != hardware.preopt_benchmark_key(cores=8, available_mem_mb=31994, fixed_threads=4))
