"""PBS cluster helpers: queue choice, Gaussian input, array script, qstat, submit."""

from ivette.services import pbs


def test_pick_queue_auto_small_vs_apc():
    assert pbs.pick_queue(8, 12) == "SMALL"            # fits SMALL limits
    assert pbs.pick_queue(13, 12) == "APC"             # too many cores
    assert pbs.pick_queue(8, 64) == "APC"              # too much memory
    assert pbs.pick_queue(8, 12, walltime_h=24) == "APC"   # over 12 h
    assert pbs.pick_queue(56, 900, requested="APC") == "APC"
    assert pbs.pick_queue(8, 12, requested="SDF") == "SDF"  # explicit wins


def test_build_opt_freq_input_uses_link1_and_cosmo():
    gjf = pbs.build_gaussian_input(
        "  C 0 0 0", method="PBE0", basis_set="6-311G", charge=-1, multiplicity=2,
        nproc=8, mem="12GB", cosmo=True, operation="opt then freq", chk="x.chk")
    assert "#p PBE0/6-311G opt scrf=(cpcm,solvent=water)" in gjf
    assert "--Link1--" in gjf
    assert "freq geom=allcheck guess=read scrf=(cpcm,solvent=water)" in gjf
    assert "-1 2" in gjf            # charge / multiplicity line


def test_build_single_point_input_no_link1():
    gjf = pbs.build_gaussian_input(
        "  C 0 0 0", method="B3LYP", basis_set="6-31G*", charge=0, multiplicity=1,
        nproc=4, mem="8GB", cosmo=False, operation="sp")
    assert "--Link1--" not in gjf
    assert "#p B3LYP/6-31G* sp" in gjf


def test_build_pbs_array_script_matches_site_template():
    s = pbs.build_pbs_array_script(5, queue="SMALL", ncpus=8, mem_gb=12,
                                   module="g16/c01", jobname="g16_job", walltime_hours=12)
    assert s.startswith("#!/bin/csh")
    assert "#PBS -q SMALL" in s
    assert "#PBS -l select=1:ncpus=8:mem=12gb" in s
    assert "#PBS -l walltime=12:00:00" in s
    assert "#PBS -J 1-5" in s
    assert "module load g16/c01" in s
    assert "rung16" in s and "GAUSS_SCRDIR" in s
    assert 'cd "$PBS_O_WORKDIR/$wd"' in s   # workdir relative to submit dir


def test_array_finished_detects_live_vs_done():
    running = ("Job id            Name  User  Time  S  Queue\n"
               "12345[].fe3-adm   g16   me    0     R  APC\n")
    done = "12345[].fe3-adm   g16   me   10:00   F  APC\n"
    assert pbs.array_finished(running, "12345[].fe3-adm") is False
    assert pbs.array_finished(done, "12345[].fe3-adm") is True
    assert pbs.array_finished("", "12345[].fe3-adm") is True   # gone = finished


class _FakeTransport:
    """Records calls; returns a job id from qsub and a scripted qstat sequence."""
    def __init__(self, qstat_seq):
        self.calls = []
        self._qstat = list(qstat_seq)

    def run(self, command):
        self.calls.append(("run", command))
        if "qsub" in command:
            return 0, "999[].fe3-adm\n", ""
        if command.startswith("qstat"):
            return 0, (self._qstat.pop(0) if self._qstat else ""), ""
        return 0, "", ""

    def push(self, local, remote):
        self.calls.append(("push", local, remote))
        return 0

    def pull(self, remote, local):
        self.calls.append(("pull", remote, local))
        return 0


def test_submit_batch_stages_polls_and_pulls(tmp_path):
    local_root = tmp_path / "geom" / "opt_then_freq_COSMO"
    local_root.mkdir(parents=True)
    running = "999[].fe3-adm g16 me 0 R APC\n"
    done = "999[].fe3-adm g16 me 1:00 F APC\n"
    ft = _FakeTransport([running, done])

    res = pbs.submit_batch(
        ft, local_root=str(local_root), remote_root="ivette_runs/geom",
        manifest_lines=["water 100.gjf 100_freq.log"],
        script_text="#!/bin/csh\n", poll_seconds=0, sleep=lambda s: None)

    assert res.job_id == "999[].fe3-adm"
    assert res.polls == 2                       # polled until 'F'
    assert (local_root / "manifest.txt").exists() and (local_root / "job.qsub").exists()
    kinds = [c[0] for c in ft.calls]
    assert "push" in kinds and "pull" in kinds  # staged up and fetched back
    assert any("qsub" in c[1] for c in ft.calls if c[0] == "run")
    # results are mirrored back INTO local_root (not its parent) — round-trips in place
    assert ("pull", "ivette_runs/geom", str(local_root)) in ft.calls


def test_submit_batch_refuses_oversized_array(tmp_path):
    import pytest
    local = tmp_path / "g"
    local.mkdir()
    ft = _FakeTransport(["done"])
    with pytest.raises(ValueError, match="max_jobs"):
        pbs.submit_batch(ft, local_root=str(local), remote_root="ivette_runs",
                         manifest_lines=["a", "b", "c"], script_text="x",
                         max_jobs=2, sleep=lambda s: None)
    # nothing was submitted
    assert not any("qsub" in c[-1] for c in ft.calls if c[0] == "run")


def test_submit_batch_strips_leading_tilde(tmp_path):
    # "~/x" must become home-relative "x" — a quoted "~" won't expand remotely.
    local = tmp_path / "g"
    local.mkdir()
    ft = _FakeTransport(["999 g me 1 F APC\n"])
    pbs.submit_batch(ft, local_root=str(local), remote_root="~/ivette_runs/geom",
                     manifest_lines=["w a b"], script_text="x", poll_seconds=0,
                     sleep=lambda s: None)
    assert not any("~" in c[-1] for c in ft.calls)              # no tilde anywhere
    assert ("push", str(local), "ivette_runs/geom") in ft.calls
