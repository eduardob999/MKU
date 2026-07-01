#!/usr/bin/env python3
"""Stage 1 — SSH connectivity + environment probe (no files, no jobs).

Low risk: just logs in and inspects the environment. Verifies you can reach the
login node (VPN up, SSH key working) and that Gaussian is loadable there.

    python scripts/hpc/test_1_connect.py --user YOURNAME
"""

import _common as C


def main():
    args = C.base_parser(__doc__).parse_args()
    hp = C.load_hpc(args)
    C.require_user(hp)
    transport = C.make_transport(hp, timeout=30)

    print(f"Connecting to {hp.user}@{hp.host} (module: {hp.gaussian_module}) ...")
    ok, report = transport.test_connection(module=hp.gaussian_module)
    print("\n---- remote diagnostics ----")
    print(report)
    print("----------------------------")

    # Extra raw probes so we learn the environment (shell, scheduler tools).
    for probe in ("echo login-shell=$0", "which qsub qstat rung16"):
        rc, out, err = transport.run(probe)
        print(f"\n$ {probe}\n{(out + err).strip()}  [rc={rc}]")

    if ok:
        C.ok("Cluster reachable, Gaussian module loads, rung16 found. Proceed to Stage 2.")
    else:
        C.fail("Connection or environment check failed — see diagnostics above. "
               "Common causes: VPN down, SSH key not set up, wrong --user, or a "
               "different module name (pass --module).")


if __name__ == "__main__":
    main()
