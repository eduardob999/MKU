"""Thin SSH/rsync transport for driving the PBS cluster from the local machine.

Shells out to the system ``ssh`` and ``rsync`` (key-based, non-interactive auth
assumed — set up an SSH key and, off-campus, bring up the KUINS VPN first). The
class exposes exactly the three operations :func:`ivette.services.pbs.submit_batch`
needs — ``run`` / ``push`` / ``pull`` — so it can be swapped for a fake in tests.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass


@dataclass
class RemoteTransport:
    host: str
    user: str
    ssh_options: str = "-o BatchMode=yes"
    timeout: int = 0   # seconds for run(); 0 = no limit

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def _ssh_cmd(self) -> list[str]:
        return ["ssh", *shlex.split(self.ssh_options), self.target]

    def run(self, command: str) -> tuple[int, str, str]:
        """Run a remote shell ``command``; returns ``(returncode, stdout, stderr)``."""
        proc = subprocess.run(
            [*self._ssh_cmd(), command],
            capture_output=True, text=True,
            timeout=(self.timeout or None),
        )
        return proc.returncode, proc.stdout, proc.stderr

    def push(self, local_path: str, remote_path: str) -> int:
        """rsync a local directory/file up to ``remote_path`` (recursive, archived)."""
        cmd = ["rsync", "-az", "-e", " ".join(["ssh", self.ssh_options]),
               f"{local_path.rstrip('/')}/", f"{self.target}:{remote_path}/"]
        return subprocess.run(cmd, capture_output=True, text=True).returncode

    def pull(self, remote_path: str, local_path: str) -> int:
        """rsync a remote directory/file down into ``local_path``."""
        cmd = ["rsync", "-az", "-e", " ".join(["ssh", self.ssh_options]),
               f"{self.target}:{remote_path.rstrip('/')}/", f"{local_path.rstrip('/')}/"]
        return subprocess.run(cmd, capture_output=True, text=True).returncode

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
        except subprocess.TimeoutExpired:
            return False, f"Timed out connecting to {self.target} (VPN up? host reachable?)."
        ok = rc == 0 and "OK_SSH" in out and "NOT FOUND" not in out
        return ok, (out + err).strip() or f"ssh exited {rc}"
