"""Shared helpers for the progressive HPC (PBS/SSH) self-test scripts.

These scripts let you validate the cluster integration one low-risk step at a
time, in a controlled sandbox, before ever submitting real Gaussian batches.
They reuse the actual project code (``ivette.services.pbs`` /
``ivette.util.remote``), so a green run means the real path works.

Connection settings come from the project's ``.env`` file (``IVETTE_HPC_*`` —
copy ``.env.example`` to ``.env`` and fill it in), with optional CLI overrides
(``--user`` / ``--host`` / ``--module`` / ``--remote-root``). Password auth uses
``IVETTE_HPC_PASSWORD`` via ``sshpass`` if set; otherwise an SSH key is used.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]   # scripts/hpc/_common.py → repo root


def add_src_to_path() -> None:
    src = _project_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--user", help="SSH username (overrides IVETTE_HPC_USER in .env)")
    p.add_argument("--host", help="Login host (overrides .env)")
    p.add_argument("--module", help="Gaussian module (overrides .env)")
    p.add_argument("--remote-root", dest="remote_root", help="Remote staging dir")
    return p


def load_hpc(args):
    """HpcParams from .env (IVETTE_HPC_*), then CLI overrides."""
    add_src_to_path()
    from ivette.util.env import load_dotenv
    from ivette.core.params import hpc_from_env
    load_dotenv()
    hp = hpc_from_env()
    over = {}
    if args.user:
        over["user"] = args.user
    if args.host:
        over["host"] = args.host
    if args.module:
        over["gaussian_module"] = args.module
    if getattr(args, "remote_root", None):
        over["remote_root"] = args.remote_root
    return replace(hp, **over)


def require_user(hp) -> None:
    if not hp.user:
        sys.exit("ERROR: no SSH username. Set IVETTE_HPC_USER in .env "
                 "(copy .env.example) or pass --user <name>.")


def make_transport(hp, *, timeout: int = 60):
    add_src_to_path()
    from ivette.util.remote import RemoteTransport
    return RemoteTransport(host=hp.host, user=hp.user, ssh_options=hp.ssh_options,
                           password=os.environ.get("IVETTE_HPC_PASSWORD", ""),
                           timeout=timeout)


def selftest_root(hp) -> str:
    """A dedicated remote dir, kept well away from any real run data.

    Home-relative: a quoted "~" won't expand in a remote shell, so we strip a
    leading "~/" and let the path resolve against the login home.
    """
    root = hp.remote_root
    if root.startswith("~/"):
        root = root[2:]
    return root.rstrip("/") + "/selftest"


def ok(msg: str) -> None:
    print(f"\n✓ PASS: {msg}")


def fail(msg: str):
    print(f"\n✗ FAIL: {msg}")
    sys.exit(1)
