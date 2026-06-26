"""Tests for the named per-stage preset store (ivette.util.presets)."""

from ivette.util import presets
from ivette.core import params as P


def test_save_list_load_delete_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "PRESET_DIR", tmp_path)

    assert presets.list_presets("training") == []
    assert presets.load_preset("training", "fast") is None

    fast = P.to_dict(P.TrainingParams(n_estimators=50, max_depth=3))
    presets.save_preset("training", "fast", fast)
    accurate = P.to_dict(P.TrainingParams(n_estimators=2000))
    presets.save_preset("training", "accurate", accurate)

    assert presets.list_presets("training") == ["accurate", "fast"]
    loaded = presets.load_preset("training", "fast")
    assert P.from_dict(P.TrainingParams, loaded).n_estimators == 50

    # round-trips through the dataclass cleanly
    assert P.from_dict(P.TrainingParams, accurate).n_estimators == 2000

    assert presets.delete_preset("training", "fast") is True
    assert presets.delete_preset("training", "fast") is False
    assert presets.list_presets("training") == ["accurate"]


def test_presets_are_namespaced_per_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "PRESET_DIR", tmp_path)
    presets.save_preset("training", "x", {"n_estimators": 10})
    presets.save_preset("gaussian", "x", {"method": "PBE0"})
    assert presets.list_presets("training") == ["x"]
    assert presets.list_presets("gaussian") == ["x"]
    assert presets.load_preset("training", "x") != presets.load_preset("gaussian", "x")


def test_corrupt_preset_file_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "PRESET_DIR", tmp_path)
    (tmp_path / "training.json").write_text("{ not valid json")
    assert presets.list_presets("training") == []   # degrades gracefully
