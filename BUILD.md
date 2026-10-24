# Build & Verification

All targets wrap `scripts/build.sh`.

## Quick Reference

```bash
make lint                 # Verilator lint & elaborate eagle4_drafter
make lint TOP=<module>    # Lint specific submodule
make sim MODE=g1          # Isolated ISA test: STATE G1; HALT (no weight load)
make sim MODE=fast        # Fast smoke test (~30s, depth=8, sanity only)
make sim                  # Full smoke test (hardware depth, tens of minutes)
make synth                # Vivado OOC synthesis (default: V80, 100 MHz; requires Vivado)
make synth PART=<part> TOP=<top> PERIOD=<ns>
make clean                # Remove build/ and sim_local/obj_dir*
```

## 1. RTL Lint (Verilator)

Lints and elaborates SystemVerilog RTL (`eagle4_drafter_depthn_1_0/src/*.sv`) locally without Vivado. Catches syntax errors, port mismatches, width mismatches, and inferred latches. Does not check Vivado primitive inference, mapping, or timing.

- **Direct script**: `eagle4_drafter_depthn_1_0/sim_local/run_lint.sh [TOP]`
- **XPM stubs**: `sim_local/xpm_memory_stubs.sv` stubs `xpm_memory_sdpram` and `xpm_memory_sprom` (`kv_head_wide_win.sv`, `lmdown_gemv_banksplit.sv`). Simulation-only; never include in synthesis filesets or `component.xml`.

## 2. End-to-End Simulation (Verilator)

Tier A smoke test (`eagle4_drafter_depthn_1_0/sim_local/tb/tb_eagle4_drafter.sv`) compiles with `verilator --binary`. Validates I/O plumbing, load paths, control flow, and `OUT_ABI_V24` frontier formatting. Does not evaluate numerical accuracy.

- **Direct script**: `eagle4_drafter_depthn_1_0/sim_local/run_sim.sh [fast|g1]`
  - `fast`: sets `TEST_WEIGHT_DEPTH=8` and `TEST_SCALE_DEPTH=8` (~30s).
  - Default: full hardware-scale depth (~14.7M pokes, tens of minutes).
  - `g1`: uploads `STATE G1; HALT` through AXI-Lite, loads deterministic
    full-size hidden/embedding vectors and their two gamma vectors, and skips
    WEIGHT/SCALE loading. Checks one G1 launch, 192 ordered output chunks per
    norm, completion before HALT, PC=1, idle/read-to-clear status, and no later
    stage starts or external DMA. Failures and timeouts exit nonzero. This is
    an ISA execution test, not a numerical-reference comparison; see
    `ISA_TESTING_PLAN.md` for the G1 arithmetic/reference proposal.
- **Configuration**: Overrides `G8_SPLICE=0`, `DEPTH_N=0`, `EMBG_EN=0` to single-step execution; exercises write-DMA (`m_axi_out`) without HBM round-trips.
- **Load paths**: Random weights/scales loaded via direct-drive pins (`boot_we`, `tb_sc_we`); EMB/HFC/GAMMA loaded via AXI4-Lite registers (`sim_local/tb/eagle4_regmap_pkg.sv`). Verified via `dbg_residency.sv`.
- **Known simulation caveats**:
  - Current source defaults set `CONNECT_SEAMS=0`; the testbench's external
    `a_chunk` is zero, so G4 does not numerically exercise the G1→G4 connection.
  - The fast smoke's immediate PMEM pointer read can report 3 rather than 4
    after four writes, before the final registered commit. The G1-only path
    polls for commit. The general smoke prints failed checks but uses `$finish`,
    so inspect its final verdict rather than trusting exit status alone.
  - `SCALE` residency reads 0: `dbg_residency.sv:7686` monitors `dma_sc_we`, missing `tb_sc_we`. Functional data still loads into `scale_bram`.
  - RoPE table uninitialized: `rope_layer.sv:284, 328` hardcodes missing path `/home/y344shi/eagle4_artifacts/p4/rope_cs.hex`. `cos_t=sin_t=0` zeroes Q/K outputs in `rope_pair_rotate.sv`, rendering downstream output weight-independent.
  - Performance: ~100–130 µs sim time/s. RTL contains snapshot logic (`SIM_WSNAP`, `+WSNAP_DUMP`, `+WSNAP_LOAD` in `weight_bank_uram.sv`), but it is not yet wired into `run_sim.sh`.

## 3. Out-of-Context Synthesis (Vivado)

Non-project OOC synthesis via `scripts/synth/run_synth.tcl`. Requires Vivado on `PATH` with valid license (run outside dev container).

- **Execution**: `./scripts/build.sh synth [PART] [TOP] [PERIOD_NS]`
- **Defaults**:
  - `PART`: `xcv80-lsva4737-2MHP-e-S` (Alveo V80; verify with `vivado -mode tcl -eval 'puts [get_parts *v80*]'`)
  - `TOP`: `eagle4_drafter`
  - `PERIOD_NS`: `10.0` (100 MHz `ap_clk`)
- **Outputs**: `build/synth_<TOP>/` containing `utilization.rpt`, `timing_summary.rpt`, `design_analysis.rpt`, and `<TOP>_synth.dcp`.

## Iteration Workflow

1. Edit RTL.
2. `make lint` — check syntax and elaboration.
3. `make sim MODE=fast` — verify I/O, control flow, and output framing.
4. `make sim` — run full hardware-depth smoke test before significant checkpoints.
5. `make synth` — run on Vivado host to verify utilization and 100 MHz timing.
