#!/usr/bin/env python3
"""Stage 4 — ONE real Gaussian job (water opt+freq) via rung16 on the cluster.

Medium risk: the first actual DFT on the cluster, but deliberately tiny — a single
water molecule, one PBS sub-job, small resources, 15-minute cap. It verifies the
whole real chain end to end: module load g16/c01 -> rung16 -> --Link1-- opt+freq
-> MaxDisk/$TMPDIR scratch -> a valid ``*_freq.log`` pulled back with
"Normal termination". Only after this is green should you run real batches.

    python scripts/hpc/test_4_one_gaussian.py --user YOURNAME
    python scripts/hpc/test_4_one_gaussian.py --user YOURNAME --cosmo   # also test CPCM
"""

import tempfile
import time
from pathlib import Path

import _common as C

C.add_src_to_path()
from ivette.services import pbs

WATER = ("  O    0.000000  0.000000  0.117300\n"
         "  H    0.000000  0.757200 -0.469300\n"
         "  H    0.000000 -0.757200 -0.469300")


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--queue", default="SMALL")
    p.add_argument("--method", default="PBE0")
    p.add_argument("--basis", default="6-311G")
    p.add_argument("--cosmo", action="store_true", help="add CPCM(water) solvation")
    p.add_argument("--poll", type=int, default=15)
    args = p.parse_args()
    hp = C.load_hpc(args)
    C.require_user(hp)
    transport = C.make_transport(hp)
    remote = C.selftest_root(hp)

    ncpus, mem_gb = 4, 8
    gjf = pbs.build_gaussian_input(
        WATER, method=args.method, basis_set=args.basis, charge=0, multiplicity=1,
        nproc=ncpus, mem=f"{mem_gb}GB", cosmo=args.cosmo, operation="opt then freq",
        chk="water.chk", title="water self-test", max_disk="5GB")
    script = pbs.build_pbs_array_script(
        1, queue=args.queue, ncpus=ncpus, mem_gb=mem_gb, module=hp.gaussian_module,
        jobname="ivette_g16_selftest", walltime_hours=1)

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "gauss"
        (local / "water").mkdir(parents=True)
        (local / "water" / "water.gjf").write_text(gjf)
        transport.run(f'rm -rf "{remote}"; mkdir -p "{remote}"')

        print(f"Submitting water opt+freq ({args.method}/{args.basis}"
              f"{' +COSMO' if args.cosmo else ''}) to {args.queue} ...")
        t0 = time.time()
        try:
            res = pbs.submit_batch(
                transport, local_root=str(local), remote_root=remote,
                manifest_lines=["water water.gjf water_freq.log"], script_text=script,
                poll_seconds=args.poll, max_polls=120,   # ~30 min ceiling
                progress=lambda jid, n: print(f"  poll {n}: {jid} queued/running..."))
        except Exception as exc:
            transport.run(f'rm -rf "{remote}"')
            C.fail(f"submit/poll failed: {exc}")

        print(f"\nJob {res.job_id} done after {res.polls} poll(s), ~{time.time()-t0:.0f}s.")
        log = local / "water" / "water_freq.log"
        if not log.exists():
            # keep the remote dir this time so the user can inspect it
            C.fail(f"No log came back at {log}. Inspect the remote dir: "
                   f"ssh {hp.user}@{hp.host} 'ls -la {remote}/water'")
        text = log.read_text(errors="replace")
        normal = "Normal termination" in text
        has_freq = "Sum of electronic and thermal Free Energies=" in text
        tail = "\n".join(text.strip().splitlines()[-3:])
        print(f"log: {log}\nNormal termination: {normal} | has thermochem: {has_freq}")
        print(f"---- log tail ----\n{tail}\n------------------")

        if normal and has_freq:
            transport.run(f'rm -rf "{remote}"')
            C.ok("Real Gaussian opt+freq completed on the cluster and parsed cleanly. "
                 "The cluster path is validated — you can wire Stage 2 of the integration.")
        else:
            C.fail("Gaussian ran but the log is incomplete. Left the remote dir in place "
                   f"for inspection: ssh {hp.user}@{hp.host} 'cat {remote}/water/water_freq.log'")


if __name__ == "__main__":
    main()
