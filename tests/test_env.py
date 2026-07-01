"""`.env` loading, HpcParams-from-env, and SSH/sshpass transport wiring."""

import os

from ivette.util.env import load_dotenv
from ivette.core import params as P
from ivette.util.remote import RemoteTransport


def test_load_dotenv_parses_and_sets(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    envf.write_text('IVETTE_HPC_USER=alice\nIVETTE_HPC_HOST="h.example"\n# a comment\n\nJUNK\n')
    monkeypatch.delenv("IVETTE_HPC_USER", raising=False)
    monkeypatch.delenv("IVETTE_HPC_HOST", raising=False)
    got = load_dotenv(envf)
    assert got["IVETTE_HPC_USER"] == "alice"
    assert os.environ["IVETTE_HPC_HOST"] == "h.example"   # quotes stripped, comment/blank ignored


def test_load_dotenv_respects_existing_env_unless_override(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    envf.write_text("FOO_XYZ=fromfile\n")
    monkeypatch.setenv("FOO_XYZ", "fromenv")
    load_dotenv(envf)
    assert os.environ["FOO_XYZ"] == "fromenv"             # existing wins
    load_dotenv(envf, override=True)
    assert os.environ["FOO_XYZ"] == "fromfile"            # override forces file value


def test_hpc_from_env_overlays_defaults(monkeypatch):
    for k in ("IVETTE_HPC_USER", "IVETTE_HPC_QUEUE", "IVETTE_HPC_WALLTIME_HOURS",
              "IVETTE_HPC_HOST", "IVETTE_HPC_MODULE", "IVETTE_HPC_REMOTE_ROOT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("IVETTE_HPC_USER", "bob")
    monkeypatch.setenv("IVETTE_HPC_QUEUE", "APC")
    monkeypatch.setenv("IVETTE_HPC_WALLTIME_HOURS", "24")
    hp = P.hpc_from_env()
    assert hp.user == "bob" and hp.queue == "APC" and hp.walltime_hours == 24
    assert hp.host == "fe1.scl.kyoto-u.ac.jp"             # untouched default kept


def test_transport_key_auth_by_default():
    t = RemoteTransport(host="h", user="u")
    assert t._ssh_argv()[0] == "ssh"
    assert t._ssh_e().startswith("ssh ")
    assert "SSHPASS" not in t._env()


def test_transport_password_auth_uses_sshpass_via_env():
    t = RemoteTransport(host="h", user="u", password="secret")
    assert t._ssh_argv()[:3] == ["sshpass", "-e", "ssh"]
    assert t._ssh_e().startswith("sshpass -e ssh")
    assert t._env()["SSHPASS"] == "secret"                # password via env, not argv
