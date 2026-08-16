#!/usr/bin/env bash
# Local lint/elaboration check for eagle4_drafter RTL using Verilator.
# Does NOT replicate Vivado synthesis (URAM/BRAM mapping, timing) — this
# only verifies syntax, port hookup, and elaboration of your SystemVerilog
# changes. xpm_memory_sdpram/sprom are stubbed by xpm_memory_stubs.sv,
# which is simulation-only and must never be added to the packaged
# component's src fileset.
set -euo pipefail
cd "$(dirname "$0")"

TOP="${1:-eagle4_drafter}"

verilator --lint-only --timing -Wall --Wno-fatal \
    -Wno-DECLFILENAME \
    +incdir+../src \
    ../src/*.sv \
    xpm_memory_stubs.sv \
    --top-module "$TOP"
