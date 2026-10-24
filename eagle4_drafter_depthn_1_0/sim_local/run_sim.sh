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
#   ./run_sim.sh gemv         # bounded real shared-GEMV cutover: HFC/EMB ->
#                              #   QKV/Q/K destinations, runtime groups 4/8,
#                              #   nonzero bases, invalid-config/restart
#   ./run_sim.sh gemv-demo    # editable 35-byte PMEM program from
#                              #   tb/gemv_demo_program.hex; prints PMEM
#                              #   words, special-register AXI readbacks,
#                              #   status, and checks all 128 outputs.
#                              #   Extra args go to the sim binary, e.g.
#                              #   +GEMV_DEMO_PROG=/tmp/edited.hex. For file
#                              #   edits without a rebuild, run the built
#                              #   binary directly:
#                              #     obj_dir_sim/sim_tb +GEMV_DEMO
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
MODE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    fast)
        echo "== fast mode: TEST_WEIGHT_DEPTH=8 TEST_SCALE_DEPTH=8 (harness sanity only) =="
        DEPTH_ARGS=(-GTEST_WEIGHT_DEPTH=8 -GTEST_SCALE_DEPTH=8)
        ;;
    g1)        SIM_ARGS=(+G1_ONLY) ;;
    regs)      SIM_ARGS=(+REGS_ONLY) ;;
    gemv)      SIM_ARGS=(+GEMV_ONLY) ;;
    gemv-demo) SIM_ARGS=(+GEMV_DEMO) ;;
    "")        ;;
    *)
        echo "error: unknown sim mode '$MODE' (use fast, g1, regs, gemv, or gemv-demo)" >&2
        exit 1
        ;;
esac

verilator --binary --timing -Wno-fatal \
    +incdir+"$SRC_DIR" \
    "$SRC_DIR"/*.sv \
    xpm_memory_stubs.sv \
    tb/eagle4_regmap_pkg.sv \
    tb/axi4_mem_slave.sv \
    tb/tb_eagle4_drafter.sv \
    --top-module tb_eagle4_drafter \
    -GLMDOWN_DIR=\""$SRC_DIR"\" \
    -GGEMV_DEMO_PROG_PATH=\""$PWD/tb/gemv_demo_program.hex\"" \
    "${DEPTH_ARGS[@]}" \
    -Mdir "$BUILD_DIR" \
    -o sim_tb

"$BUILD_DIR/sim_tb" "${SIM_ARGS[@]}" "$@"
