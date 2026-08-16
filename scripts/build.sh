#!/usr/bin/env bash
# =====================================================================
# build.sh - single entry point for local checks on eagle4_drafter RTL.
#
#   ./scripts/build.sh lint  [TOP]                 - Verilator lint/elab
#                                                     (works in this env,
#                                                     no Vivado needed)
#   ./scripts/build.sh synth [PART] [TOP] [PERIOD] - Vivado OOC synth
#                                                     check (needs Vivado
#                                                     + license on PATH)
#   ./scripts/build.sh clean                        - remove build/ output
# =====================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IP_DIR="$REPO_ROOT/eagle4_drafter_depthn_1_0"

cmd="${1:-}"
shift || true

case "$cmd" in
    lint)
        "$IP_DIR/sim_local/run_lint.sh" "$@"
        ;;
    synth)
        if ! command -v vivado >/dev/null 2>&1; then
            echo "error: 'vivado' not found on PATH." >&2
            echo "  Synthesis needs a Vivado install + license (not available in this dev container)." >&2
            echo "  Run this on a machine/module with Vivado, e.g.:" >&2
            echo "    module load vivado   # or: source /tools/Xilinx/Vivado/<ver>/settings64.sh" >&2
            echo "    ./scripts/build.sh synth" >&2
            exit 1
        fi
        vivado -mode batch -source "$REPO_ROOT/scripts/synth/run_synth.tcl" -tclargs "$@"
        ;;
    clean)
        rm -rf "$REPO_ROOT/build" "$IP_DIR/sim_local/obj_dir"
        ;;
    *)
        echo "usage: $0 {lint|synth|clean} [args...]" >&2
        exit 1
        ;;
esac
