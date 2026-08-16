# Build / verification instructions

## Quick reference

```
make lint                 # Verilator lint + elaborate all RTL (this env, no Vivado needed)
make lint TOP=kv_head_wide_win   # lint/elaborate a single submodule as top

make synth                # Vivado OOC synth check (needs Vivado + license, NOT in this dev env)
make synth PART=xcv80-lsva4737-2MHP-e-S TOP=eagle4_drafter PERIOD=10.0

make clean                # remove build/ output and verilator obj_dir
```

Both targets are thin wrappers around `scripts/build.sh`; call that directly
if you don't have `make`.

## 1. Local RTL lint (Verilator) — works in this dev container

Verilator (`sudo dnf install verilator`, v5.046) lints and elaborates the
full `eagle4_drafter_depthn_1_0/src/*.sv` tree: catches syntax errors, port
mismatches, width issues, and inferred-latch bugs without needing Vivado or
the Alveo board.

It does **not** replicate Xilinx synthesis (URAM/BRAM inference, timing
closure, `Synth-8-xxxx` LUT/RAM sizing errors) — that only shows up under
`make synth`. Treat a clean lint as "my SystemVerilog is well-formed and
elaborates," not "this will synthesize/meet timing."

Two Xilinx `XPM` primitives (`xpm_memory_sdpram`, `xpm_memory_sprom`) are
instantiated directly in `kv_head_wide_win.sv` / `lmdown_gemv_banksplit.sv`.
Their real definitions live inside a Vivado install and aren't in this repo,
so `eagle4_drafter_depthn_1_0/sim_local/xpm_memory_stubs.sv` provides
minimal behavioral stand-ins (same read-latency / read-first semantics,
covers only the port subset this design actually drives). That file is
**simulation-only** — it must never be added to `component.xml`'s packaged
fileset or any synthesis file list, since it would silently shadow the real
Xilinx macros.

Run directly:
```
eagle4_drafter_depthn_1_0/sim_local/run_lint.sh [top_module]
```

## 2. Vivado out-of-context synthesis check — needs Vivado, not available here

`scripts/synth/run_synth.tcl` reads `src/*.sv`, runs
`synth_design -mode out_of_context`, constrains `ap_clk` at the packaged
100 MHz (10 ns, per the header note in `eagle4_drafter.sv`), and writes:

```
build/synth_<TOP>/utilization.rpt
build/synth_<TOP>/timing_summary.rpt
build/synth_<TOP>/design_analysis.rpt
build/synth_<TOP>/<TOP>_synth.dcp
```

Default part is `xcv80-lsva4737-2MHP-e-S` (Alveo V80) — **verify this
against your installed board/part files** before trusting results:
```
vivado -mode tcl -eval 'puts [get_parts *v80*]'
```

Run on a machine that actually has Vivado + a license:
```
module load vivado    # or: source /tools/Xilinx/Vivado/<ver>/settings64.sh
./scripts/build.sh synth [PART] [TOP] [PERIOD_NS]
```

`build/` is gitignored — reports and checkpoints are regenerated, not
committed.

## Recommended flow while iterating on the ISA/datapath

1. Edit RTL.
2. `make lint` — fast, catches most mistakes, runs anywhere.
3. When you have Vivado access: `make synth` — confirms it actually maps to
   LUTs/BRAM/URAM/DSP on the V80 part and get a timing estimate at 100 MHz.
4. Full board bring-up / bitstream is out of scope for this repo's local
   tooling — that still needs the real Vivado IP-packager flow + Alveo V80
   hardware.
