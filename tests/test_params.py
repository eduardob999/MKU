"""Tests for the central parameter config (ivette.core.params)."""

from ivette.core import params as P


def test_stage_registry_maps_to_dataclasses():
    assert set(P.STAGES) == {"structures", "download", "dataset", "gaussian", "training"}
    title, cls = P.STAGES["training"]
    assert title and cls is P.TrainingParams


def test_to_dict_and_round_trip():
    tp = P.TrainingParams()
    d = P.to_dict(tp)
    assert d["n_estimators"] == 500 and d["radius"] == 2
    assert P.from_dict(P.TrainingParams, d) == tp


def test_from_dict_ignores_unknown_and_fills_missing():
    # Partial + stale keys: known ones applied, unknown ignored, rest defaulted.
    tp = P.from_dict(P.TrainingParams, {"max_depth": 9, "obsolete_flag": 123})
    assert tp.max_depth == 9
    assert tp.n_estimators == 500          # default preserved
    assert not hasattr(tp, "obsolete_flag")


def test_describe_reports_kinds_and_choices():
    info = {f.name: f for f in P.describe(P.GaussianParams())}
    assert info["timeout"].kind == "int"
    assert info["method"].kind == "str"
    assert info["preopt_mode"].choices == ["auto", "none", "pm7", "gaussian631g"]

    dinfo = {f.name: f for f in P.describe(P.DatasetParams())}
    assert dinfo["fetch_pharma"].kind == "bool"

    sinfo = {f.name: f for f in P.describe(P.StructureParams())}
    assert sinfo["ring_sizes"].kind == "ints"   # list-of-ints hint

    tinfo = {f.name: f for f in P.describe(P.TrainingParams())}
    assert tinfo["learning_rate"].kind == "float"
    assert tinfo["learning_rate"].help          # every field documents itself


def test_every_field_has_help_text():
    for _stage, (_title, cls) in P.STAGES.items():
        for fi in P.describe(cls()):
            assert fi.help, f"{cls.__name__}.{fi.name} is missing help text"
