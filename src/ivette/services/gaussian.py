"""Headless Gaussian batch orchestration.

The production half of the Gaussian pipeline — running the batch once per charge
state, each in its own directory with independent resume/restart protection —
expressed without any UI. The caller supplies callbacks for the one human
decision (what to do about pre-existing results) and for progress reporting, so
the terminal menu and a future web/job server can both drive it.

The expensive hardware sizing + benchmarking step stays in the caller: its
results are passed in via ``batch_settings``, so this service is pure
"run the batches" logic.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ivette.module.gaussian16_pipeline import batch_run
from ivette.util import applog, jsonstore

_log = applog.get_logger("gaussian.service")

# What to do when a charge state already has results on disk.
EXISTING_RESUME = "resume"
EXISTING_RESTART = "restart"
EXISTING_SKIP = "skip"


@dataclass
class ChargeStateRun:
    """Outcome of one charge state's batch (or a skip)."""
    label: str
    state_name: str
    charge: int
    multiplicity: int
    work_dir: Path
    n_existing: int = 0          # completed molecules found before this run
    skipped: bool = False
    results: list = field(default_factory=list)
    n_success: int = 0
    n_failed: int = 0


# decide_existing(state_name, n_existing) -> "resume" | "restart" | "skip"
DecideExisting = Callable[[str, int], str]
StateHook = Callable[["ChargeStateRun"], None]


def run_charge_state_batches(
    geometry_dir,
    gaussian_root,
    *,
    operation: str,
    cosmo: bool,
    charge_states,
    batch_settings: dict,
    decide_existing: Optional[DecideExisting] = None,
    on_state_start: Optional[StateHook] = None,
    on_state_done: Optional[StateHook] = None,
) -> "list[ChargeStateRun]":
    """Run the Gaussian batch for each ``(label, charge, multiplicity)`` state.

    ``batch_settings`` carries the benchmark-derived knobs (``jobs``, ``nproc``,
    ``mem``, ``preopt_mode``, ``preopt_basis_set``). When a state already has
    results, ``decide_existing`` is consulted (default: resume); ``restart``
    wipes the state's directory first, ``skip`` leaves it untouched. ``on_state_start``
    / ``on_state_done`` are optional progress hooks. Returns one
    :class:`ChargeStateRun` per state.
    """
    geometry_dir = Path(geometry_dir)
    gaussian_root = Path(gaussian_root)
    runs: list[ChargeStateRun] = []

    for label, charge, multiplicity in charge_states:
        work_dir = (gaussian_root / label) if label else gaussian_root
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = work_dir / "checkpoint.json"

        done = jsonstore.read_json(checkpoint, default={}) if checkpoint.exists() else {}
        n_existing = sum(1 for v in done.values() if isinstance(v, dict) and v.get("success"))

        state = ChargeStateRun(
            label=label, state_name=label or "neutral",
            charge=charge, multiplicity=multiplicity,
            work_dir=work_dir, n_existing=n_existing,
        )
        if on_state_start:
            on_state_start(state)

        resume = True
        if done or any(work_dir.glob("*/*.log")):
            decision = decide_existing(state.state_name, n_existing) if decide_existing else EXISTING_RESUME
            if decision == EXISTING_SKIP:
                state.skipped = True
                _log.info("charge state skipped | state=%s dir=%s", state.state_name, work_dir.name)
                runs.append(state)
                if on_state_done:
                    on_state_done(state)
                continue
            if decision == EXISTING_RESTART:
                shutil.rmtree(work_dir, ignore_errors=True)
                work_dir.mkdir(parents=True, exist_ok=True)
                resume = False

        results = batch_run(
            sdf_dir=str(geometry_dir),
            work_dir=str(work_dir),
            operation=operation,
            resume=resume,
            checkpoint=str(checkpoint),
            cosmo=cosmo,
            charge=charge,
            multiplicity=multiplicity,
            **batch_settings,
        )
        state.results = results
        state.n_success = sum(r.success for r in results)
        state.n_failed = len(results) - state.n_success
        runs.append(state)
        if on_state_done:
            on_state_done(state)

    return runs
