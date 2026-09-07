# eagle4_drafter Tasks

Task tracker for EAGLE-3 draft-head FPGA offload (`eagle4_drafter_depthn_1_0/`, Alveo V80).

## Active: ISA v0 (FSM → Instruction Fetcher / PMEM)

**Goal:** Make stage sequencing reprogrammable without RTL edits by replacing `top_fsm.sv` with an instruction fetcher and program memory (PMEM). v0 targets a drop-in replacement bit-identical to current `top_fsm` execution.

- [ ] Map transition graph in `top_fsm.sv:165-290` (`state_t`, 20 states `S_IDLE`..`S_LVL_EMBG`).
  - Key branches: `S_G5_ROPE -> S_DONE | S_G6` (`kvonly_act`), `S_G8 -> S_LVL_FEED | S_DONE` (`use_depth_loop`), `S_IDLE -> S_LVL_ARM | S_G1`, `S_LVL_EMBG -> S_G1`, and depth-loop ring `S_LVL_ARM/FEED/SEL/OUT`.
- [ ] Define v0 instruction format: `{stage_sel, wait_signal_sel, predicate_sel, taken_target, fallthrough_target}`. Stage-done handshakes stay in hardware; ISA controls sequencing and branching.
- [ ] Design PMEM: 20-entry memory (LUTRAM or `xpm_memory_sprom` pattern from `lmdown_gemv_banksplit.sv`).
- [ ] Implement `instr_fetch.sv` (PC, fetch, decode, predicate mux, branch resolution).
- [ ] Integrate into `eagle4_drafter.sv`: replace `top_fsm` instance with `instr_fetch` + `pmem` loaded with v0 image.
- [ ] Verify bit-identical execution vs `top_fsm` using per-stage bisect goldens (PDI#4 isolation goldens, `CONNECT_SEAMS`).
- [ ] Pass `make lint` on modified top and submodules.

## Next (Post-v0)

- [ ] Build PMEM assembler/image generator for model stage graph updates without RTL rebuilds.
- [ ] Exploit sequencing flexibility for datapath optimizations (stage reordering, inter-stage pipelining).
- [ ] Determine predicate policy: host-programmable registers vs static config straps (`eagle4_config.svh`).

## Completed Tooling

- [x] Verilator 5.046 setup for local linting and elaboration.
- [x] Simulation memory stubs: `sim_local/xpm_memory_stubs.sv` (`xpm_memory_sdpram`, `xpm_memory_sprom`).
- [x] Build entry points: `Makefile`, `scripts/build.sh`, `scripts/synth/run_synth.tcl`.
- [x] Repository setup and `.gitignore`.
- [x] Tier A end-to-end smoke test in `sim_local/tb/` (`make sim [MODE=fast]`, `OUT_ABI_V24` validation).
  - *Future Tier B extension:* Restore `G8_SPLICE`/`DEPTH_N`/`EMBG_EN` defaults and hook `axi4_mem_slave.sv` onto `m_axi_g8`/`m_axi_g8b`/`m_axi_emb`/`m_axi_embb`.

## Known RTL Gaps

- **SCALE residency counter blind to direct drive:** `dbg_residency.sv:7686` monitors `dma_sc_we`, missing `tb_sc_we`. BRAM loads succeed (verified via `scale_bram.mem[]`), but counter reports 0. Fix: tap `eff_sc_we`/`eff_sc_wr_data`.
- **RoPE cos/sin LUT unmapped:** `rope_layer.sv:284, 328` hardcodes uncommitted `/home/y344shi/eagle4_artifacts/p4/rope_cs.hex`. Read fails, `cos_t=sin_t=0`, zeroing Q/K outputs in `rope_pair_rotate.sv`; downstream output is weight-independent. Fix: generate `rope_cs.hex` (`{cos_t[15:0], sin_t[15:0]}`, 64 pairs, base theta 500000) or parameterize file path.
- **`eagle4_config_intent.txt` drift:** Snapshot claims `DUAL_PORT=1`; actual RTL default is `DUAL_PORT_V=0` (`eagle4_config.svh:134`). Treat `eagle4_config.svh` as source of truth.

## Open Items

- Confirm Alveo V80 part `xcv80-lsva4737-2MHP-e-S` against installed Vivado board files.
- Local container lacks Vivado; run `make synth` on licensed host.
- Full-depth `make sim` runtime: wire snapshot restore (`SIM_WSNAP`, `+WSNAP_DUMP`/`+WSNAP_LOAD` in `weight_bank_uram.sv`) into `run_sim.sh`.
