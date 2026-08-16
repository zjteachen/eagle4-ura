# eagle4_drafter — personal task tracker

Personal working notes for the EAGLE-3 draft-head FPGA offload project
(Alveo V80 IP in `eagle4_drafter_depthn_1_0/`). Not a design spec — just
what's in flight and what's next.

## Now: ISA v0 — FSM -> instruction fetcher / PMEM

**Goal:** make the drafter's stage sequencing reprogrammable (new models,
datapath reorderings) without touching RTL, by replacing the hardcoded
`top_fsm.sv` state machine with an instruction fetcher reading a program
memory (PMEM). v0 target is a drop-in replacement: PMEM is programmed with
instructions that reproduce today's `top_fsm` transition graph exactly, so
behavior stays bit-identical before any real reordering/model-porting work
starts.

`top_fsm.sv` is pipeline-style: a linear chain of stage-select ->
wait-for-`*_done` -> advance, with a few conditional branches. That maps
fairly directly onto a simple sequencer ISA:

- [ ] Enumerate the exact transition graph to preserve (from
      `top_fsm.sv:165-290`, `state_t`, 20 states `S_IDLE`..`S_LVL_EMBG`).
      Note the non-linear edges specifically — these are the cases the ISA
      needs a real branch/predicate for, not just "next":
      - `S_G5_ROPE -> S_DONE | S_G6` on `kvonly_act`
      - `S_G8 -> S_LVL_FEED | S_DONE` on `use_depth_loop`
      - `S_IDLE -> S_LVL_ARM | S_G1` (depth-loop entry vs. legacy single-pass)
      - `S_LVL_EMBG -> S_G1` (RUNG C child-embedding gather detour)
      - the `S_LVL_ARM/FEED/SEL/OUT` depth-loop ring itself
- [ ] Decide instruction encoding for v0. Leaning toward: `{stage_sel,
      wait_signal_sel, predicate_sel, taken_target, fallthrough_target}` —
      i.e. keep "wait for this stage's done pulse" as a hardware handshake
      field (not something the ISA re-implements), and let the ISA only own
      *sequencing and branching*. Each stage's internal microarchitecture is
      untouched.
- [ ] Design PMEM: width = instruction width above, depth = 20 for v0
      (one instruction per current FSM state). Where does it live —
      distributed LUTRAM (tiny, fine) vs. reuse an existing BRAM primitive
      idiom from this codebase (see `xpm_memory_sprom` usage in
      `lmdown_gemv_banksplit.sv` for the pattern this IP already uses for
      ROM-style init).
- [ ] Write the instruction fetcher module (PC register, fetch, decode,
      predicate mux, branch resolution). Should end up smaller than
      `top_fsm.sv`'s case statement, not bigger — if it isn't, the encoding
      is wrong.
- [ ] Bring-up plan: swap `top_fsm` instance for
      `instr_fetch` + `pmem` in `eagle4_drafter.sv`, with the v0 PMEM image
      hand-assembled to match `state_t`'s graph 1:1.
- [ ] Verify bit-identical against current `top_fsm` behavior. Reuse the
      existing per-stage bisect golden methodology referenced in
      `eagle4_drafter.sv` (PDI#4 isolation goldens, `CONNECT_SEAMS` modes)
      rather than inventing a new verification path.
- [ ] `make lint` clean on the new modules + full top before calling v0 done.

## Next (after v0 lands and is verified bit-identical)

- [ ] Design PMEM image format / assembler so a *different* model's stage
      graph (different stage count, different order, skip stages) is just a
      new PMEM image, no RTL change. This is the actual point of the ISA —
      v0 only proves the mechanism is transparent.
- [ ] Look for datapath optimizations the old hardcoded FSM made awkward to
      express (stage reordering, overlap/pipelining across what used to be
      strictly sequential states) now that sequencing is data, not gates.
- [ ] Decide whether conditionals like `kvonly_act` / `use_depth_loop`
      should become PMEM-visible predicates (host-programmable) or stay as
      RTL-level config straps (`eagle4_config.svh`) that select which PMEM
      image gets loaded.

## Tooling (done)

- [x] `sudo dnf install verilator` (5.046) — local lint/elaborate without
      Vivado or the Alveo board.
- [x] `sim_local/xpm_memory_stubs.sv` — behavioral stand-ins for the two
      XPM primitives this design instantiates directly
      (`xpm_memory_sdpram` in `kv_head_wide_win.sv`, `xpm_memory_sprom` in
      `lmdown_gemv_banksplit.sv`), sim-only, kept out of the packaged
      fileset.
- [x] `make lint` / `make synth` / `scripts/build.sh` — see `BUILD.md`.
- [x] Git repo initialized, `.gitignore` for Vivado/Verilator junk.

## Open questions / parking lot

- Exact Alveo V80 part string used in `scripts/synth/run_synth.tcl`
  (`xcv80-lsva4737-2MHP-e-S`) is unverified against this project's actual
  board files — confirm before trusting `make synth` output.
- No Vivado available in this dev container — `make synth` needs to run
  elsewhere. Until then, `make lint` is the only local signal on RTL
  changes.
