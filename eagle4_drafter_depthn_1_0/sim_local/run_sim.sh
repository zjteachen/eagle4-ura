#!/usr/bin/env bash
# =====================================================================
# run_sim.sh — build + run the Tier A smoke-test testbench
# (sim_local/tb/tb_eagle4_drafter.sv) with Verilator.
#
# Usage:
#   ./run_sim.sh g1           # upload STATE G1; HALT over AXI-Lite; check
#                              #   isolated RMSNorm execution (no weights)
#   ./run_sim.sh regs         # ISA WRSR -> special_regs bank / AXI readback
#                              #   test (no weight/activation loads)
#   ./run_sim.sh              # full depth: real WEIGHT/SCALE bank sizes
#                              #   (~14.7M direct-drive pokes -- slow, but
#                              #   the true load, matching real hardware)
#   ./run_sim.sh fast         # tiny WEIGHT/SCALE depth (8 words/lane) --
#                              #   for iterating on the harness itself, NOT
#                              #   a meaningful functional result (most of
#                              #   each bank stays at its simulation-only
#                              #   zero-init, see weight_bank_uram.sv:110)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

SRC_DIR="$(cd ../src && pwd)"
BUILD_DIR="obj_dir_sim"

DEPTH_ARGS=()
SIM_ARGS=()
if [[ "${1:-}" == "fast" ]]; then
    echo "== fast mode: TEST_WEIGHT_DEPTH=8 TEST_SCALE_DEPTH=8 (harness sanity only) =="
    DEPTH_ARGS=(-GTEST_WEIGHT_DEPTH=8 -GTEST_SCALE_DEPTH=8)
elif [[ "${1:-}" == "g1" ]]; then
    SIM_ARGS=(+G1_ONLY)
elif [[ "${1:-}" == "regs" ]]; then
    SIM_ARGS=(+REGS_ONLY)
fi

verilator --binary --timing -Wno-fatal \
    +incdir+"$SRC_DIR" \
    "$SRC_DIR"/*.sv \
    xpm_memory_stubs.sv \
    tb/eagle4_regmap_pkg.sv \
    tb/axi4_mem_slave.sv \
    tb/tb_eagle4_drafter.sv \
    --top-module tb_eagle4_drafter \
    -GLMDOWN_DIR=\""$SRC_DIR"\" \
    "${DEPTH_ARGS[@]}" \
    -Mdir "$BUILD_DIR" \
    -o sim_tb

"$BUILD_DIR/sim_tb" "${SIM_ARGS[@]}"
