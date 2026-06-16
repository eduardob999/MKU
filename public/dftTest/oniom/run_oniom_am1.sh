#!/bin/bash
cd "$(dirname "$0")"
# run ONIOM with AM1 low-level
stdbuf -oL -eL g16 < oniom_methanol_am1.com 2>&1 | tee oniom_methanol_am1.log
