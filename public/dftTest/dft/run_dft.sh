#!/bin/bash
cd "$(dirname "$0")"
# run with all processors (uses %nprocshared in input)
stdbuf -oL -eL g16 < water_opt.com 2>&1 | tee water_opt.log
