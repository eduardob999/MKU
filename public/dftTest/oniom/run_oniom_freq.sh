#!/bin/bash
cd "$(dirname "$0")"
# run ONIOM frequency calculation
stdbuf -oL -eL g16 < oniom_methanol_freq.com 2>&1 | tee oniom_methanol_freq.log
