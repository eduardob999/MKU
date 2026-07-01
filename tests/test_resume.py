"""Resume journal: record while in-flight, clear on completion, prune stale."""

from ivette.util import resume


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "RESUME_FILE", tmp_path / "resume.json")


def test_record_then_clear_round_trip(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    gd = tmp_path / "geom"
    gd.mkdir()

    assert resume.active_runs() == []
    resume.record_run("gaussian:g1:opt:1", {
        "kind": "gaussian", "model_id": "model_1", "geometry_dir": str(gd),
        "operation": "opt then freq", "cosmo": True, "label": "opt+freq COSMO"})
    active = resume.active_runs()
    assert [k for k, _ in active] == ["gaussian:g1:opt:1"]
    assert active[0][1]["started"]                 # timestamp auto-added

    resume.clear_run("gaussian:g1:opt:1")
    assert resume.active_runs() == []              # cleared → nothing to resume


def test_active_runs_prunes_entries_whose_dir_is_gone(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    gone = tmp_path / "deleted_geom"
    resume.record_run("marcus:g9", {"kind": "marcus", "geometry_dir": str(gone),
                                    "target": "IC50"})
    # dir never created → stale entry is pruned (and removed from the file).
    assert resume.active_runs() == []
    assert resume.active_runs() == []              # stays gone after prune


def test_runs_sorted_oldest_first(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    resume.record_run("k1", {"geometry_dir": str(tmp_path / "a"), "started": "2026-06-30T10:00:00"})
    resume.record_run("k2", {"geometry_dir": str(tmp_path / "b"), "started": "2026-06-30T09:00:00"})
    assert [k for k, _ in resume.active_runs()] == ["k2", "k1"]
