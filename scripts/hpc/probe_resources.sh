#!/usr/bin/env bash
# Report the cluster resources available to you: your live usable snapshot
# (qstatmyjobs), per-queue hard limits, Gaussian modules, disk quota, and node
# hardware. Read-only. Uses .env (IVETTE_HPC_USER/HOST) + your SSH key.
#   bash scripts/hpc/probe_resources.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
ENV=.env
val(){ grep -E "^$1=" "$ENV" | head -1 | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/^"//' -e 's/"$//'; }
U="$(val IVETTE_HPC_USER)"; H="$(val IVETTE_HPC_HOST)"
[ -n "$U" ] && [ -n "$H" ] || { echo "Set IVETTE_HPC_USER / IVETTE_HPC_HOST in .env"; exit 2; }

ssh -o BatchMode=yes -o ConnectTimeout=25 "${U}@${H}" 'bash -s' <<'REMOTE'
source /etc/profile.d/modules.sh 2>/dev/null
echo "===== host ====="; hostname
echo; echo "===== YOUR LIVE USABLE SNAPSHOT (qstatmyjobs) ====="
if command -v qstatmyjobs >/dev/null 2>&1; then qstatmyjobs; else echo "(qstatmyjobs not on PATH)"; fi
echo; echo "===== PER-QUEUE HARD LIMITS ====="
for q in SMALL APC APG SDF; do
  echo "--- $q ---"
  qstat -Qf "$q" 2>/dev/null | grep -E "resources_max|resources_default|max_run_res|enabled =|started ="
done
echo; echo "===== GAUSSIAN MODULES ====="
( module -t avail 2>&1 | grep -iE "g16|gaussian" ) || module avail g16 2>&1 | grep -i g16
echo; echo "===== DISK QUOTA ====="
quota -v 2>/dev/null || quota 2>/dev/null || echo "(quota n/a)"
echo; echo "===== COMPUTE-NODE HARDWARE (node count by cores / mem) ====="
pbsnodes -av 2>/dev/null | awk '/resources_available.ncpus/{c=$3} /resources_available.mem/{print c" cores, "$3}' | sort | uniq -c | sort -rn | head -20
REMOTE
