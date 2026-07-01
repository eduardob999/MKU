#!/usr/bin/env bash
# One-time: set up SSH *key* auth to the cluster using the password currently in
# .env, verify it, then comment the password out so future connections use the
# key. Key auth is more secure AND avoids the repeated-password-login pattern that
# trips intrusion detection on shared clusters. Safe to re-run.
#
#   bash scripts/hpc/setup_ssh_key.sh
set -u
cd "$(dirname "$0")/../.." || exit 1        # repo root
ENV=.env
[ -f "$ENV" ] || { echo "No .env found (copy .env.example first)."; exit 1; }

val() { grep -E "^$1=" "$ENV" | head -1 | cut -d= -f2- \
        | sed -e 's/^[[:space:]]*//' -e 's/^"//' -e 's/"$//'; }

U="$(val IVETTE_HPC_USER)"
H="$(val IVETTE_HPC_HOST)"
export SSHPASS="$(val IVETTE_HPC_PASSWORD)"

[ -n "$U" ] && [ -n "$H" ] || { echo "IVETTE_HPC_USER / IVETTE_HPC_HOST missing in .env"; exit 2; }
if [ -z "${SSHPASS}" ]; then echo "No IVETTE_HPC_PASSWORD in .env — already key-based? Nothing to do."; exit 0; fi
command -v sshpass >/dev/null || { echo "sshpass not installed."; exit 4; }

echo "target: ${U}@${H}"
KEY="$HOME/.ssh/id_ed25519"
if [ ! -f "$KEY" ]; then echo "generating $KEY ..."; ssh-keygen -t ed25519 -N "" -f "$KEY" -q; fi

echo "installing public key (password used this once) ..."
sshpass -e ssh-copy-id -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 \
        -i "${KEY}.pub" "${U}@${H}"

echo "verifying KEY-ONLY auth (password explicitly disabled) ..."
if ssh -o BatchMode=yes -o PasswordAuthentication=no -o ConnectTimeout=25 \
       "${U}@${H}" 'echo KEY_OK; hostname'; then
    sed -i 's/^IVETTE_HPC_PASSWORD=/#IVETTE_HPC_PASSWORD=/' "$ENV"
    echo "OK: key auth confirmed; IVETTE_HPC_PASSWORD commented out in .env."
else
    echo "WARN: key auth not confirmed; left the password in .env. Check VPN / password."
    exit 5
fi
