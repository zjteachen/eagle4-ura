# ISA Migration & Testing Plan

Long-term plan for replacing the hardwired `top_fsm` sequencer with
instruction-level control of the EAGLE-4 drafter, and for proving the
migration is correct along the way.

Current status (ISA v1, 2026-09):

- `instruction_fetcher.sv` executes byte instructions out of the 32-word
  PMEM: STATE (5-bit sequencer state ID), JMP (word-aligned PMEM target),
  HALT, reserved-as-HALT. Encoding and state IDs: see the
  *ISA State Encodings & Instruction Set* section in `COMPONENTS.md`.
- `eagle4_drafter.sv` decodes the fetcher's state ID into the same
  stage-start pulses, `stage_sel`, and busy/done handshake the FSM used to
  drive, so every existing stage sees identical control timing.
- The depth-N tree loop is **not** expressible in the current ISA (it needs
  data-dependent conditional sequencing on `branch_j`/`level_i`/
  `level_done`/`tree_all_done`). Depth-N is disabled; `top_fsm.sv` is kept
  in the tree, uninstantiated, as the reference for re-implementation.
  DEPTH_N builds fail at elaboration (`$fatal` guard in `eagle4_drafter.sv`)
  rather than silently producing no output, since the loop signals are tied
  off. Remove the guard when conditional jumps land.
- KV-only and G8-optional behavior is expressed by program content (a
  shorter program), not by runtime conditionals.

## Phase 1 — Microprogram state replication

Goal: prove the ISA reproduces the FSM's control of every state, one state
at a time.

For each sequencer state S (G1, G4, G5, G6, G7G, G7U, G7_SILU, G9, G11,
GLM, G12, TREE, G8):

1. Write a microprogram `STATE S; HALT` into PMEM.
2. Run it on a prepared DUT and capture the stage's observable outputs
   (`dbg_stage_capture` taps, `dbg_residency`, per-stage probe RAM).
3. Repeat with the FSM-era build (keep a branch with `u_fsm` instantiated)
   and diff the captures.

The FSM build and the ISA build drive identical start pulses with identical
timing (one cycle after the previous stage's done), so captures should be
bit-identical. A microprogram can prefix the target state with the earlier
states it depends on (e.g. `G1..G9; G11; HALT`) to test states that need
upstream data.

## Phase 2 — Microprogram instruction testing

Goal: exercise the ISA machinery itself.

- JMP: programs that jump over dummy bytes; assert `pc_out` (readable via
  the 0x16C status word) lands on the word-aligned target, and that
  skipped bytes never assert a stage start.
- HALT and reserved opcodes: assert `kernel_busy` drops, `kernel_done`
  pulses exactly once, and PC freezes at the HALT byte.
- Relaunch: run to HALT, write `ap_start` again, assert PC resets to 0 and
  the program reruns.
- Malformed programs (STATE with an unimplemented ID, G8 without a
  preceding TREE) must stall observably (PC frozen, readable at 0x16C)
  rather than corrupt state.

A future **debug instruction** (dedicated opcode with a 5-bit debug operand)
will let a microprogram single-step or dump state mid-program: e.g. halt
with the debug tap index in the operand so the host reads one probe RAM
window per debug instruction. Microprograms combined with the debug
instruction become the unit-test harness for individual instructions:
each instruction gets a program that uses only it plus debug dumps, with
expected PC/instr/state-transition results asserted from the host side.

## Phase 3 — State decomposition into fine-grained instructions

Once every FSM state is replicable, states decompose: e.g. the G7G/G7U
states become explicit LOAD-weight / STREAM / DRAIN instructions; the
single STATE byte stops being a monolithic "run stage" and the ISA gains
real datapath control (memory operands, loop counts, condition codes).
Migrate one state at a time, keeping the microprogram replication harness
green after each step.

## Phase 4 — Numerical reference testing

Once all states are expressible as non-state instructions, bring up a
numpy reference of the EAGLE-3 draft head (the same quantized weights and
fixed-point geometry the RTL loads) and cross-check:

- Per-instruction numerical tests: each datapath instruction's captured
  output chunk compared against the reference at the same point.
- End-to-end: a full program run's OUT_ABI_V24 frontier (ids, scores,
  parents) compared against the reference model's tree for the same input
  embedding/HFC and sampled weights.

## Deferred work

- Depth-N under ISA control: requires conditional jump opcodes plus the
  `branch_j`/`level_i` loop counters (re-implement from `top_fsm.sv`).
- Runtime kvonly / G8 splicing conditionals (currently program-content
  only).
- Larger PMEM or wider instructions if 128 bytes / 8 opcodes become
  limiting.
