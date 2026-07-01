"""Thin SSH/rsync transport for driving the PBS cluster from the local machine.

Shells out to the system ``ssh`` and ``rsync``. Two auth modes:

* **SSH key (preferred, default):** set up a key (``ssh-copy-id``) and leave the
  password empty; uses ``BatchMode`` non-interactive auth.
* **Password:** if a password is given (from ``IVETTE_HPC_PASSWORD`` in ``.env``),
  it's fed via ``sshpass -e`` (password passed through the environment, never on
  the command line). Requires ``sshpass`` installed locally.

Off-campus, bring up the KUINS VPN first. The class exposes exactly the three
operations :func:`ivette.services.pbs.submit_batch` needs — ``run`` / ``push`` /
``pull`` — so it can be swapped for a fake in tests.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class RemoteTransport:
    host: str
    user: str
    ssh_options: str = "-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    password: str = ""     # optional; uses sshpass -e when set (SSH key preferred)
    timeout: int = 0       # seconds for run(); 0 = no limit

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    # ── auth plumbing ────────────────────────────────────────────────────────
    def _env(self) -> dict:
        env = os.environ.copy()
        if self.password:
            env["SSHPASS"] = self.password
        return env

    def _opts(self) -> list[str]:
        # With a password, BatchMode would block password auth; accept new host
        # keys so a first connect doesn't hang on the yes/no prompt.
        if self.password:
            return ["-o", "StrictHostKeyChecking=accept-new"]
        return shlex.split(self.ssh_options)

    def _require_sshpass(self) -> None:
        if self.password and not shutil.which("sshpass"):
            raise RuntimeError(
                "A password is set (IVETTE_HPC_PASSWORD) but 'sshpass' is not installed. "
                "Install it (e.g. 'sudo apt-get install sshpass') or use an SSH key and "
                "leave the password unset.")

    def _ssh_argv(self) -> list[str]:
        prefix = ["sshpass", "-e"] if self.password else []
        return [*prefix, "ssh", *self._opts(), self.target]

    def _ssh_e(self) -> str:
        base = "ssh " + " ".join(self._opts())
        return f"sshpass -e {base}" if self.password else base

    # ── operations ───────────────────────────────────────────────────────────
    def run(self, command: str) -> tuple[int, str, str]:
        """Run a remote shell ``command``; returns ``(returncode, stdout, stderr)``."""
        self._require_sshpass()
        proc = subprocess.run(
            [*self._ssh_argv(), command],
            capture_output=True, text=True,
            timeout=(self.timeout or None), env=self._env(),
        )
        return proc.returncode, proc.stdout, proc.stderr

    def push(self, local_path: str, remote_path: str) -> int:
        """rsync a local directory/file up to ``remote_path`` (recursive, archived)."""
        self._require_sshpass()
        cmd = ["rsync", "-az", "-e", self._ssh_e(),
               f"{local_path.rstrip('/')}/", f"{self.target}:{remote_path}/"]
        return subprocess.run(cmd, capture_output=True, text=True, env=self._env()).returncode

    def pull(self, remote_path: str, local_path: str) -> int:
        """rsync a remote directory/file down into ``local_path``."""
        self._require_sshpass()
        cmd = ["rsync", "-az", "-e", self._ssh_e(),
               f"{self.target}:{remote_path.rstrip('/')}/", f"{local_path.rstrip('/')}/"]
        return subprocess.run(cmd, capture_output=True, text=True, env=self._env()).returncode

    def test_connection(self, module: str = "") -> tuple[bool, str]:
        """Verify SSH works and (optionally) Gaussian is loadable + ``rung16`` found.

        Returns ``(ok, report)`` — ``report`` is the remote diagnostics text so the
        UI can show exactly what the cluster said.
        """
        check = "echo OK_SSH; hostname"
        if module:
            check += (f"; source /etc/profile.d/modules.sh 2>/dev/null; "
                      f"module load {module} 2>&1; "
                      f"which rung16 || echo 'rung16 NOT FOUND'")
        try:
            rc, out, err = self.run(check)
        except FileNotFoundError:
            return False, "ssh not found on this machine."
        except RuntimeError as exc:            # sshpass missing
            return False, str(exc)
        except subprocess.TimeoutExpired:
            return False, f"Timed out connecting to {self.target} (VPN up? host reachable?)."
        ok = rc == 0 and "OK_SSH" in out and "NOT FOUND" not in out
        return ok, (out + err).strip() or f"ssh exited {rc}"
