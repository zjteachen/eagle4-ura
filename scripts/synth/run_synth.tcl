# =====================================================================
# run_synth.tcl - non-project, out-of-context Vivado synthesis for the
# eagle4_drafter IP. Sanity-checks that the RTL actually synthesizes on
# the target part (LUT/BRAM/URAM/DSP budget, Synth-8-xxxx elaboration
# errors) and reports a timing estimate at the packaged clock.
#
# This does NOT need a board or a Vivado project checked out on disk -
# it reads src/*.sv directly and runs synth_design -mode out_of_context.
# Requires Vivado on PATH with a valid license (not available in this
# dev container - run this on a machine/module that has Vivado).
#
# Usage:
#   vivado -mode batch -source scripts/synth/run_synth.tcl \
#       -tclargs [PART] [TOP] [PERIOD_NS]
#
#   PART       default: xcv80-lsva4737-2MHP-e-S (Alveo V80). VERIFY this
#              against your installed board/part files before trusting
#              results - `vivado -mode tcl -eval {puts [get_parts *v80*]}`
#   TOP        default: eagle4_drafter
#   PERIOD_NS  default: 10.0 (100 MHz, matches header comment in
#              eagle4_drafter.sv: "packaged at FREQ_HZ=100 MHz")
# =====================================================================

set argv_list $argv
set PART      [expr {[llength $argv_list] >= 1 ? [lindex $argv_list 0] : "xcv80-lsva4737-2MHP-e-S"}]
set TOP       [expr {[llength $argv_list] >= 2 ? [lindex $argv_list 1] : "eagle4_drafter"}]
set PERIOD_NS [expr {[llength $argv_list] >= 3 ? [lindex $argv_list 2] : "10.0"}]

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set REPO_ROOT  [file normalize "$SCRIPT_DIR/../.."]
set IP_DIR     "$REPO_ROOT/eagle4_drafter_depthn_1_0"
set SRC_DIR    "$IP_DIR/src"
set OUT_DIR    "$REPO_ROOT/build/synth_$TOP"

file mkdir $OUT_DIR

puts "===================================================================="
puts " eagle4_drafter OOC synth check"
puts "   PART      = $PART"
puts "   TOP       = $TOP"
puts "   PERIOD_NS = $PERIOD_NS"
puts "   SRC_DIR   = $SRC_DIR"
puts "   OUT_DIR   = $OUT_DIR"
puts "===================================================================="

# All *.sv together in one read_verilog call: Vivado's elaborator does not
# require single-pass file ordering for cross-file package refs (unlike
# Verilator's -y auto-discovery), but keep it explicit and complete anyway.
set sv_files [glob -directory $SRC_DIR *.sv]
read_verilog -sv $sv_files

set_property include_dirs $SRC_DIR [current_fileset]

synth_design -mode out_of_context -top $TOP -part $PART

create_clock -name ap_clk -period $PERIOD_NS [get_ports ap_clk]

report_utilization  -file "$OUT_DIR/utilization.rpt"
report_timing_summary -file "$OUT_DIR/timing_summary.rpt"
report_design_analysis -congestion -file "$OUT_DIR/design_analysis.rpt"

write_checkpoint -force "$OUT_DIR/${TOP}_synth.dcp"

puts "===================================================================="
puts " Synthesis complete. Reports in: $OUT_DIR"
puts "===================================================================="
