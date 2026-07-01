#!/usr/bin/env python3
"""Stage 0 — OFFLINE. Generate and print the exact files we'd submit.

Zero risk: no SSH, no cluster contact. It builds a real Gaussian input (water,
opt+freq via --Link1--), the PBS job-array script, the queue choice, and a
manifest, writes them under scripts/hpc/_out/, and prints them for you to eyeball
against your cluster's expectations BEFORE any network step.

    python scripts/hpc/test_0_generate.py
"""

from pathlib import Path

import _common as C

C.add_src_to_path()
from ivette.services import pbs
from ivette.core.params import HpcParams

# A trivial 3-atom molecule — the point is to inspect the generated text, not chemistry.
WATER = ("  O    0.000000  0.000000  0.117300\n"
         "  H    0.000000  0.757200 -0.469300\n"
         "  H    0.000000 -0.757200 -0.469300")


def main():
    hp = HpcParams()
    out = Path(__file__).resolve().parent / "_out"
    out.mkdir(exist_ok=True)

    ncpus, mem_gb = 4, 8
    queue = pbs.pick_queue(ncpus, mem_gb, walltime_h=1)
    print(f"pick_queue({ncpus} cores, {mem_gb} GB, 1 h)  ->  {queue}")

    gjf = pbs.build_gaussian_input(
        WATER, method="PBE0", basis_set="6-311G", charge=0, multiplicity=1,
        nproc=ncpus, mem=f"{mem_gb}GB", cosmo=True, operation="opt then freq",
        chk="water.chk", title="water self-test", max_disk="5GB")
    script = pbs.build_pbs_array_script(
        1, queue=queue, ncpus=ncpus, mem_gb=mem_gb, module=hp.gaussian_module,
        jobname="ivette_selftest", walltime_hours=1)
    manifest = "water water.gjf water_freq.log"

    (out / "water.gjf").write_text(gjf)
    (out / "job.qsub").write_text(script)
    (out / "manifest.txt").write_text(manifest + "\n")

    print("\n================ water.gjf (opt then freq, COSMO) ================\n")
    print(gjf)
    print("================ job.qsub (PBS array) ================\n")
    print(script)
    print(f"================ manifest.txt ================\n{manifest}\n")

    C.ok(f"Generated files written to {out}. Review them against your cluster's "
         "conventions (module name, rung16 usage, queue/resource limits) before Stage 1.")


if __name__ == "__main__":
    main()
