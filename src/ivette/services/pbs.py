"""PBS (Altair) job generation + submission for Gaussian on the cluster.

The pure helpers here (queue choice, Gaussian input, PBS array script, qstat
parsing) are deterministic and unit-tested. :func:`submit_batch` ties them to a
``transport`` (see :mod:`ivette.util.remote`) that performs the actual SSH/rsync,
so the orchestration can be tested with a fake transport and no live cluster.

Design notes
------------
* One **PBS job array** runs many molecules at once — the cluster's scheduler is
  the parallelism, far beyond a local process pool.
* opt-then-freq is emitted as a **single Gaussian input** using ``--Link1--`` +
  ``geom=allcheck guess=read`` so each molecule is one sub-job whose output is a
  combined ``*_freq.log`` — byte-compatible with the local pipeline's parsers
  (so cluster and local results are interchangeable).
* Gaussian is launched via the site wrapper ``rung16 <input> <output>`` after
  ``module load <gaussian_module>`` in a **csh** script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# SMALL queue limits (high priority, fast start); larger/longer jobs go to APC.
SMALL_MAX_CORES = 12
SMALL_MAX_MEM_GB = 48
SMALL_MAX_WALLTIME_H = 12


def pick_queue(ncpus: int, mem_gb: int, walltime_h: int = 0, requested: str = "auto") -> str:
    """Choose a queue. ``auto`` → SMALL when the job fits its limits, else APC."""
    if requested and requested != "auto":
        return requested
    fits_small = (
        ncpus <= SMALL_MAX_CORES
        and mem_gb <= SMALL_MAX_MEM_GB
        and (walltime_h == 0 or walltime_h <= SMALL_MAX_WALLTIME_H)
    )
    return "SMALL" if fits_small else "APC"


def build_gaussian_input(coord_block: str, *, method: str, basis_set: str,
                         charge: int, multiplicity: int, nproc: int, mem: str,
                         cosmo: bool, operation: str, extra_keywords: str = "",
                         chk: str = "mol.chk", title: str = "Ivette job",
                         max_disk: str = "") -> str:
    """Gaussian input (.gjf/.com). opt+freq becomes a two-section ``--Link1--`` job.

    The freq section uses ``geom=allcheck guess=read`` to continue from the
    optimized geometry in the checkpoint, so one job yields a combined log whose
    last SCF + thermochemistry + Standard-orientation block match what the local
    split opt/freq pipeline produces.
    """
    scrf = " scrf=(cpcm,solvent=water)" if cosmo else ""
    extra = f" {extra_keywords.strip()}" if extra_keywords.strip() else ""
    mdisk = f" MaxDisk={max_disk.strip()}" if max_disk.strip() else ""
    link0 = f"%chk={chk}\n%nprocshared={nproc}\n%mem={mem}\n"
    op = operation.lower()

    if "opt" in op and "freq" in op:
        return (
            f"{link0}#p {method}/{basis_set} opt{scrf}{mdisk}{extra}\n\n"
            f"{title} (opt)\n\n{charge} {multiplicity}\n{coord_block}\n\n"
            f"--Link1--\n{link0}"
            f"#p {method}/{basis_set} freq geom=allcheck guess=read{scrf}{mdisk}{extra}\n\n"
        )
    route = operation.strip() or "sp"
    return (
        f"{link0}#p {method}/{basis_set} {route}{scrf}{mdisk}{extra}\n\n"
        f"{title}\n\n{charge} {multiplicity}\n{coord_block}\n\n"
    )


def build_pbs_array_script(count: int, *, queue: str, ncpus: int, mem_gb: int,
                           module: str, jobname: str = "ivette_g16",
                           manifest_name: str = "manifest.txt",
                           walltime_hours: int = 0) -> str:
    """A csh PBS **job-array** script: sub-job *i* runs the *i*-th manifest line.

    Each manifest line is ``<workdir> <input> <output>`` where ``workdir`` is
    **relative to the submit directory** (``$PBS_O_WORKDIR``); the sub-job cd's
    into that per-compound directory and runs ``rung16``. Scratch is the per-job
    ``$TMPDIR`` (fast, auto-cleaned) per the cluster's guidance.
    """
    walltime = f"#PBS -l walltime={walltime_hours}:00:00\n" if walltime_hours else ""
    return (
        "#!/bin/csh\n"
        f"#PBS -q {queue}\n"
        f"#PBS -N {jobname}\n"
        f"#PBS -l select=1:ncpus={ncpus}:mem={mem_gb}gb\n"
        f"{walltime}"
        f"#PBS -J 1-{count}\n"
        "#PBS -j oe\n"
        "\n"
        "source /etc/profile.d/modules.csh\n"
        f"module load {module}\n"
        "\n"
        f'set line = `sed -n "${{PBS_ARRAY_INDEX}}p" "$PBS_O_WORKDIR/{manifest_name}"`\n'
        "set wd  = `echo \"$line\" | awk '{print $1}'`\n"
        "set inp = `echo \"$line\" | awk '{print $2}'`\n"
        "set out = `echo \"$line\" | awk '{print $3}'`\n"
        'cd "$PBS_O_WORKDIR/$wd"\n'   # manifest workdir is relative to the submit dir
        'setenv GAUSS_SCRDIR "$TMPDIR"\n'
        'rung16 "$inp" "$out"\n'
    )


_QSTAT_ROW = re.compile(r"^(\S+)\s+\S+\s+\S+\s+\S+\s+([A-Z])\s+\S+", re.MULTILINE)
# A job (id may carry an array suffix like 123[].fe3-adm) is "live" in these states.
_LIVE_STATES = {"Q", "R", "B", "E", "H", "T", "W", "S", "U"}


def parse_qstat_states(text: str) -> dict[str, str]:
    """``{job_id: state}`` parsed from ``qstat`` / ``qstat -x`` table output."""
    return {m.group(1): m.group(2) for m in _QSTAT_ROW.finditer(text)
            if m.group(1).lower() != "job"}


def array_finished(text: str, job_id: str) -> bool:
    """True when ``job_id`` is no longer running (gone, or in a finished state).

    With ``qstat -x`` a finished array shows state ``F``/``X``; without ``-x`` it
    simply disappears. Either way it is not in a live state.
    """
    base = job_id.split(".")[0].split("[")[0]
    for jid, state in parse_qstat_states(text).items():
        if jid.split(".")[0].split("[")[0] == base and state in _LIVE_STATES:
            return False
    return True


@dataclass
class SubmitResult:
    job_id: str
    polls: int


def submit_batch(transport, *, local_root, remote_root, manifest_lines,
                 script_text, manifest_name="manifest.txt", script_name="job.qsub",
                 poll_seconds=30, max_polls=None, sleep=None, progress=None) -> SubmitResult:
    """Stage inputs up, ``qsub`` the array, poll to completion, pull results back.

    ``transport`` provides ``run(cmd)->(rc,out,err)``, ``push(local,remote)`` and
    ``pull(remote,local)`` (see :mod:`ivette.util.remote`); injecting a fake makes
    this testable without SSH. Inputs must already be written under ``local_root``;
    results are pulled back into the same tree, so the existing parsers consume
    them unchanged.
    """
    import time
    from pathlib import Path

    sleep = sleep if sleep is not None else time.sleep
    # A quoted "~" doesn't expand in a remote shell; treat "~/x" as home-relative "x".
    if remote_root.startswith("~/"):
        remote_root = remote_root[2:]
    lr = Path(local_root)
    (lr / manifest_name).write_text("\n".join(manifest_lines) + "\n")
    (lr / script_name).write_text(script_text)

    transport.run(f'mkdir -p "{remote_root}"')
    transport.push(str(lr), remote_root)

    rc, out, err = transport.run(f'cd "{remote_root}" && qsub "{script_name}"')
    if rc != 0:
        raise RuntimeError(f"qsub failed: {(err or out).strip()}")
    job_id = out.strip().splitlines()[-1].strip()

    polls = 0
    while True:
        polls += 1
        _rc, qout, _qerr = transport.run(f'qstat -x "{job_id}"')
        if array_finished(qout, job_id):
            break
        if max_polls and polls >= max_polls:
            raise TimeoutError(f"job {job_id} still not finished after {polls} polls")
        if progress:
            progress(job_id, polls)
        sleep(poll_seconds)

    transport.pull(remote_root, str(lr))   # mirror results back into local_root (in place)
    return SubmitResult(job_id=job_id, polls=polls)
