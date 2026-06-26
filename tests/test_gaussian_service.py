"""Tests for the headless Gaussian batch service.

``batch_run`` is replaced with a fake that records its calls, so these run
instantly (no real Gaussian) and pin down the resume/restart/skip decisions,
per-charge-state directories, settings forwarding, and the callback contract.
"""

import json

from ivette.services import gaussian as svc


class _FakeResult:
    cid = "x"

    def __init__(self, success):
        self.success = success


def _fake_batch(calls, n_success=2, n_fail=0):
    def fake_batch_run(*, sdf_dir, work_dir, operation, resume, checkpoint,
                       cosmo, charge, multiplicity, **kw):
        calls.append({"work_dir": work_dir, "resume": resume, "charge": charge,
                      "multiplicity": multiplicity, "cosmo": cosmo, **kw})
        return [_FakeResult(True)] * n_success + [_FakeResult(False)] * n_fail
    return fake_batch_run


def _settings(**over):
    base = dict(jobs=1, nproc=1, mem="1GB", preopt_mode="none", preopt_basis_set="6-31G*")
    base.update(over)
    return base


def test_runs_each_charge_state_with_own_dir_and_settings(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "batch_run", _fake_batch(calls, n_success=3))
    geom = tmp_path / "geom"; geom.mkdir()
    root = tmp_path / "gaussian"

    runs = svc.run_charge_state_batches(
        geom, root, operation="opt then freq", cosmo=True,
        charge_states=[("neutral", 0, 1), ("anion", -1, 2)],
        batch_settings=_settings(jobs=2, nproc=4, preopt_mode="pm7"),
    )

    assert len(calls) == 2 and len(runs) == 2
    assert runs[0].state_name == "neutral" and runs[0].n_success == 3 and runs[0].n_failed == 0
    assert runs[1].charge == -1 and runs[1].multiplicity == 2
    # benchmark-derived settings forwarded straight through
    assert calls[0]["jobs"] == 2 and calls[0]["nproc"] == 4 and calls[0]["preopt_mode"] == "pm7"
    assert calls[0]["cosmo"] is True
    # each state gets its own subdirectory
    assert calls[0]["work_dir"].endswith("neutral")
    assert calls[1]["work_dir"].endswith("anion")


def test_skip_decision_does_not_run_batch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "batch_run", _fake_batch(calls))
    geom = tmp_path / "geom"; geom.mkdir()
    root = tmp_path / "gaussian"; root.mkdir()
    (root / "checkpoint.json").write_text(json.dumps({"111": {"success": True}}))

    runs = svc.run_charge_state_batches(
        geom, root, operation="opt", cosmo=False, charge_states=[("", 0, 1)],
        batch_settings=_settings(), decide_existing=lambda name, n: "skip",
    )
    assert calls == []                  # batch never ran
    assert runs[0].skipped is True
    assert runs[0].n_existing == 1


def test_restart_clears_dir_and_disables_resume(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "batch_run", _fake_batch(calls))
    geom = tmp_path / "geom"; geom.mkdir()
    root = tmp_path / "gaussian"; root.mkdir()
    (root / "checkpoint.json").write_text(json.dumps({"111": {"success": True}}))
    stale = root / "stale.txt"; stale.write_text("old")

    runs = svc.run_charge_state_batches(
        geom, root, operation="opt", cosmo=False, charge_states=[("", 0, 1)],
        batch_settings=_settings(), decide_existing=lambda name, n: "restart",
    )
    assert not stale.exists()           # directory was wiped
    assert calls[0]["resume"] is False
    assert runs[0].skipped is False


def test_resume_keeps_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "batch_run", _fake_batch(calls))
    geom = tmp_path / "geom"; geom.mkdir()
    root = tmp_path / "gaussian"; root.mkdir()
    (root / "checkpoint.json").write_text(json.dumps({"111": {"success": True}}))
    keep = root / "keep.txt"; keep.write_text("data")

    svc.run_charge_state_batches(
        geom, root, operation="opt", cosmo=False, charge_states=[("", 0, 1)],
        batch_settings=_settings(), decide_existing=lambda name, n: "resume",
    )
    assert keep.exists()
    assert calls[0]["resume"] is True


def test_progress_callbacks_fire_in_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "batch_run", _fake_batch(calls))
    geom = tmp_path / "geom"; geom.mkdir()
    root = tmp_path / "gaussian"
    starts, dones = [], []

    svc.run_charge_state_batches(
        geom, root, operation="opt", cosmo=False,
        charge_states=[("neutral", 0, 1), ("anion", -1, 2)],
        batch_settings=_settings(),
        on_state_start=lambda s: starts.append(s.state_name),
        on_state_done=lambda s: dones.append(s.state_name),
    )
    assert starts == ["neutral", "anion"]
    assert dones == ["neutral", "anion"]
