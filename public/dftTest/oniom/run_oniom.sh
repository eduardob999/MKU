#!/bin/bash
cd "$(dirname "$0")"
# run standard ONIOM input
stdbuf -oL -eL g16 < oniom_methanol.com 2>&1 | tee oniom_methanol.log
