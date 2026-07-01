#!/usr/bin/env python3
"""Stage 3 — Submit ONE trivial (non-Gaussian) PBS job through the real path.

Medium-low risk: exercises the full submit -> qstat poll -> retrieve loop using
``pbs.submit_batch`` (the actual production code), but the job just runs
``hostname``/``date`` — no Gaussian, tiny resources, 5-minute cap. This isolates
"does our scheduler round-trip work" from "does Gaussian work" (Stage 4).

    python scripts/hpc/test_3_trivial_job.py --user YOURNAME
"""

import tempfile
import time
from pathlib import Path

import _common as C

C.add_src_to_path()
from ivette.services import pbs


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--queue", default="SMALL")
    p.add_argument("--poll", type=int, default=10, help="seconds between qstat polls")
    args = p.parse_args()
    hp = C.load_hpc(args)
    C.require_user(hp)
    transport = C.make_transport(hp)
    remote = C.selftest_root(hp)

    # A trivial csh PBS job — no manifest use, no Gaussian.
    script = (
        "#!/bin/csh\n"
        f"#PBS -q {args.queue}\n"
        "#PBS -N ivette_selftest\n"
        "#PBS -l select=1:ncpus=1:mem=1gb\n"
        "#PBS -l walltime=00:05:00\n"
        "#PBS -j oe\n\n"
        "cd $PBS_O_WORKDIR\n"
        'echo "ran on: `hostname`" > selftest_out.txt\n'
        'echo "date: `date`" >> selftest_out.txt\n'
        'echo "TRIVIAL_OK" >> selftest_out.txt\n'
    )

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "trivial"
        local.mkdir()
        transport.run(f'rm -rf "{remote}"; mkdir -p "{remote}"')

        print(f"Submitting trivial job to {args.queue} ...")
        t0 = time.time()
        try:
            res = pbs.submit_batch(
                transport, local_root=str(local), remote_root=remote,
                manifest_lines=["unused unused unused"], script_text=script,
                poll_seconds=args.poll, max_polls=120,   # up to ~20 min
                progress=lambda jid, n: print(f"  poll {n}: {jid} still queued/running..."))
        except Exception as exc:
            transport.run(f'rm -rf "{remote}"')
            C.fail(f"submit/poll failed: {exc}")

        elapsed = time.time() - t0
        print(f"\nJob {res.job_id} finished after {res.polls} poll(s), ~{elapsed:.0f}s.")
        out_file = local / "selftest_out.txt"
        content = out_file.read_text() if out_file.exists() else "(missing)"
        print(f"---- selftest_out.txt ----\n{content}\n--------------------------")

        transport.run(f'rm -rf "{remote}"')
        if "TRIVIAL_OK" in content:
            C.ok("Full submit -> poll -> retrieve loop works. Proceed to Stage 4 (one Gaussian).")
        else:
            C.fail("Job ran but its output didn't come back correctly. "
                   "Check the queue name and that qstat -x is available.")


if __name__ == "__main__":
    main()
