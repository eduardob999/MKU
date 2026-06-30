"""Calculation-set storage: one-per-operation upsert + parsed-result linkage."""

from ivette.util import storage
from ivette.util.jsonstore import MetadataStore


def _isolate(tmp_path, monkeypatch):
    """Point the calc + descriptor stores at a throwaway location."""
    monkeypatch.setattr(storage, "CALCULATIONS",
                        MetadataStore(tmp_path / "calc.json", "calculations", "calc"))
    monkeypatch.setattr(storage, "DFT_DESCRIPTORS",
                        MetadataStore(tmp_path / "dft.json", "dft_descriptors", "dft"))
    monkeypatch.setattr(storage, "DFT_DESCRIPTOR_DIR", tmp_path / "dft")


def test_calculation_set_is_one_per_operation(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    kw = dict(model_id="model_000001", target="IC50", geometry_id="geometry_000001")
    cid1 = storage.register_or_update_calculation_set(
        name="opt then freq", kind="opt_freq", operation="opt then freq",
        cosmo=False, charge_states=None, output_dir="/g/opt_then_freq", **kw)

    # Re-running the SAME operation updates the same set rather than duplicating.
    cid1b = storage.register_or_update_calculation_set(
        name="opt then freq", kind="opt_freq", operation="opt then freq",
        cosmo=False, charge_states=None, output_dir="/g/opt_then_freq", **kw)
    assert cid1b == cid1

    # A different operation (its own output dir) is a distinct set.
    cid2 = storage.register_or_update_calculation_set(
        name="opt+freq COSMO", kind="cosmo", operation="opt then freq", cosmo=True,
        charge_states=[["neutral", 0, 1], ["anion", -1, 2]],
        output_dir="/g/opt_then_freq_COSMO", **kw)
    assert cid2 != cid1

    sets = dict(storage.calculation_sets_for_geometry("geometry_000001"))
    assert set(sets) == {cid1, cid2}
    assert sets[cid2]["cosmo"] is True
    assert storage.find_calculation_set("geometry_000001", "/g/opt_then_freq_COSMO") == cid2


def test_parsed_result_set_links_to_its_calc_set(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    calc_id = storage.register_or_update_calculation_set(
        model_id="model_000001", target="IC50", geometry_id="geometry_000001",
        name="opt then freq", kind="opt_freq", operation="opt then freq",
        cosmo=False, charge_states=None, output_dir="/g/opt_then_freq")

    dft_id, _ = storage.save_dft_descriptor_set(
        [{"CID": "1", "gibbs_G": -1.0}], "IC50 DFT", "model_000001", "IC50",
        "geometry_000001", calc_id=calc_id)

    linked = storage.dft_descriptor_sets_for_calc(calc_id)
    assert [d for d, _ in linked] == [dft_id]
    # An unrelated calc id has no parsed results.
    assert storage.dft_descriptor_sets_for_calc("calc_999999") == []


def test_dft_comparison_history_round_trip(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    dft_id, _ = storage.save_dft_descriptor_set(
        [{"CID": "1", "gibbs_G": -1.0}], "IC50 DFT", "model_000001", "IC50",
        "geometry_000001")

    cid1, entry1 = storage.add_dft_comparison(dft_id, {"delta_cv_r2": 0.12, "target": "IC50"})
    cid2, _ = storage.add_dft_comparison(dft_id, {"delta_cv_r2": -0.03, "target": "IC50"})
    assert (cid1, cid2) == ("cmp_001", "cmp_002")        # ids increment
    assert entry1["delta_cv_r2"] == 0.12 and "created" in entry1

    saved = storage.DFT_DESCRIPTORS.get(dft_id)["comparisons"]
    assert [c["id"] for c in saved] == ["cmp_001", "cmp_002"]
