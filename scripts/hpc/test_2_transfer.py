#!/usr/bin/env python3
"""Stage 2 — File staging round-trip (rsync up, list, pull back, verify, clean).

Low risk: transfers a couple of tiny text files to a dedicated selftest dir, reads
them back, checks they match, then removes the remote dir. No compute, no jobs.

    python scripts/hpc/test_2_transfer.py --user YOURNAME
"""

import tempfile
from pathlib import Path

import _common as C


def main():
    args = C.base_parser(__doc__).parse_args()
    hp = C.load_hpc(args)
    C.require_user(hp)
    transport = C.make_transport(hp)
    remote = C.selftest_root(hp)

    with tempfile.TemporaryDirectory() as tmp:
        up = Path(tmp) / "up"
        up.mkdir()
        token = "ivette-transfer-token-12345"
        (up / "hello.txt").write_text(token + "\n")
        (up / "sub").mkdir()
        (up / "sub" / "nested.txt").write_text("nested-ok\n")

        print(f"Pushing {up} -> {hp.user}@{hp.host}:{remote}")
        transport.run(f'mkdir -p "{remote}"')
        if transport.push(str(up), remote) != 0:
            C.fail("rsync push failed (check ssh/rsync and the remote path).")

        rc, out, err = transport.run(f'ls -la "{remote}" && cat "{remote}/hello.txt"')
        print(f"\n$ ls + cat on remote:\n{(out + err).strip()}  [rc={rc}]")
        if token not in out:
            C.fail("Uploaded file not found / unreadable on the remote side.")

        down = Path(tmp) / "down"
        down.mkdir()
        if transport.pull(remote, str(down)) != 0:
            C.fail("rsync pull failed.")
        got = (down / "hello.txt").read_text().strip()
        nested = (down / "sub" / "nested.txt").exists()
        print(f"\nPulled back: hello.txt={got!r}  nested present={nested}")

        # Clean up the remote selftest dir no matter what.
        transport.run(f'rm -rf "{remote}"')

        if got == token and nested:
            C.ok("Round-trip transfer verified and remote cleaned up. Proceed to Stage 3.")
        else:
            C.fail("Round-trip mismatch — files did not come back intact.")


if __name__ == "__main__":
    main()
