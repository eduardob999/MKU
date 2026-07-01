# HPC (PBS/SSH) progressive self-tests

Validate the Kyoto-U cluster integration **one low-risk step at a time**, in a
sandbox, before ever submitting real Gaussian batches. Each stage is riskier than
the last — **only move on when the current one passes.** Every stage uses a
dedicated remote dir (`<remote_root>/selftest`) and cleans up after itself; none
of them touch your real run data.

## Prerequisites
- **VPN up** (off-campus) so `fe1.scl.kyoto-u.ac.jp` is reachable.
- **Key-based SSH** working: `ssh youruser@fe1.scl.kyoto-u.ac.jp` should log in
  without a password prompt (`ssh-keygen` then `ssh-copy-id`). The scripts use
  non-interactive SSH — password prompts won't work.
- Pass your username with `--user YOURNAME` (or save an `hpc` preset in the app's
  Configuration menu and the scripts pick it up).

Run from the repo root, e.g.:
```
python scripts/hpc/test_1_connect.py --user YOURNAME
```

## The ladder

| Stage | Script | Risk | What it proves |
|------|--------|------|----------------|
| 0 | `test_0_generate.py` | none (offline) | The generated Gaussian input, PBS array script, queue choice and manifest look right. No network. |
| 1 | `test_1_connect.py`  | low | SSH login works and `module load g16/c01` + `rung16` are available. No files, no jobs. |
| 2 | `test_2_transfer.py` | low | rsync up/down round-trips intact; remote dir is created and cleaned. No compute. |
| 3 | `test_3_trivial_job.py` | medium-low | The full **submit → qstat poll → retrieve** loop works, using a trivial `hostname` job (no Gaussian). |
| 4 | `test_4_one_gaussian.py` | medium | **One** real water opt+freq via `rung16` completes and returns a valid `*_freq.log`. First real DFT. Add `--cosmo` to also test CPCM. |

Stage 4 green = the cluster path is trustworthy end-to-end; only then wire the
full batch integration (Stage 2 of the plan) and run real compounds.

## Notes
- Common flags on every script: `--user`, `--host`, `--module`, `--remote-root`,
  `--preset`. Stages 3–4 also take `--queue` and `--poll` (seconds between polls).
- Everything is tiny and capped (1–4 cores, ≤8 GB, ≤1 h walltime, hard poll ceiling),
  so nothing can run away.
- If a stage fails, the script prints the raw remote output so you can see exactly
  what the cluster said — paste that back and we adjust (shell quirks, module names,
  queue limits, `qstat` format all surface here rather than mid-batch).
- Stage 4 deliberately **leaves the remote dir in place if the log is bad**, so you
  can inspect it; on success it cleans up.
