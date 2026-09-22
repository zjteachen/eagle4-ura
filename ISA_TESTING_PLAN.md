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

### Implemented: isolated G1 program

Run `make sim MODE=g1` (or `sim_local/run_sim.sh g1` from the IP directory).
The existing `tb_eagle4_drafter.sv` selects `run_g1_only` with `+G1_ONLY`:

- Uploads two 3072-element signed vectors and two Q4.12 unity-gamma vectors
  through the AXI-Lite trio. No WEIGHT/SCALE or downstream gamma loads.
  `h_fc_buf` is one 3072-element buffer, not three raw target-layer vectors;
  this test deliberately does not use the legacy smoke test's `HFC_WORDS`.
- Writes PMEM word 0 as `0x40404001`: bytes `01 40 40 40`, meaning
  `STATE G1; HALT` followed by HALT padding. Polls the PMEM write pointer to
  confirm commit before checking the initial instruction, because the AXI
  write response precedes the registered PMEM commit.
- Launches through `AP_CTRL`, requires exactly one G1 start and completion
  of both norms, and checks 192 ordered 16-element output chunks per norm.
  Checks PC does not advance before G1 completes, one kernel-done pulse,
  HALT at PC=1, and idle/ready with read-to-clear `ap_done`.
- Rejects downstream stage starts and external-memory requests, including
  during a post-HALT observation window. Upload/execution timeouts use
  `$fatal`, so failures exit nonzero.

This proves isolated execution, not numerical equality or FSM-era equivalence.
The fixture gives both input vectors the same sum of squares; differing norm
latencies are not covered. The ISA currently advances on hidden `g1_done`
alone, while the scale engine can skip division on its saturation path.

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

### G1 NumPy reference proposal

G1 can be modeled now, before decomposing the remaining states. With
`DEPTH_N=0`, `u_rmsnorm_g1` consumes `h_fc_buf`/`gamma_emb_buf` and
`u_rmsnorm_g1_e` consumes `input_emb_buf`/`gamma_e_buf`. They normalize
independently; do not normalize their concatenation. Each block reads 192
chunks of 16 signed INT16 values to accumulate a 64-bit sum of squares,
computes one scale, then rereads x with gamma to apply that scale. There is
no mean subtraction, bias, hidden-feature projection, or QKV GEMV in G1.
The full-width R1 seam presents `[embedding_norm, hidden_norm]` to G4.

For either stream, the current G1 parameters give this integer sequence:

```text
S = sum(int64(x[i]) * int64(x[i]))
T = (S * 43691) >> 31
D = isqrt(((T >> 4) + 1) << 12)
s = min(32767, (1 << 24) // D)
a[i] = rtl_sat_scale(x[i] * s)         # shift 10, signed 18-bit intermediate
y[i] = clip((a[i] * gamma[i]) >> 12, -32768, 32767)
```

`SUMSQ_MULT=43691` selects the multiply/shift path; `SUMSQ_RSHIFT=12` is
not an additional shift. Use `math.isqrt` and integer division, not floating
sqrt/reciprocal, for exact floor behavior. Cast INT16 arrays to NumPy INT64
before multiplication; gamma is Q4.12 (`4096` represents one). The scale's
effective fractional width is 10 here despite its signal name `scale_q12`.
Signed shifts truncate toward negative infinity, not toward zero or nearest.
Do not fuse the two multiplies: intermediate truncation/saturation matters.

Keep two reference layers:

1. A float64 mathematical model, `gamma * x / sqrt(mean(x*x) + eps)`, to
   check model intent and quantify quantization error. For Q4.12 x, the
   current integer epsilon bucket corresponds approximately to `eps=2^-16`,
   with additional error from bucketing and the reciprocal approximation.
2. An integer RTL oracle following the sequence above, including exact
   width/overflow semantics, to compare each output element. Start with the
   deterministic test fixture, then zero/tiny vectors (scale rail), signed
   and sparse vectors, varying gamma, and intermediate/final overflow.
   No trained checkpoint is needed to test this primitive.

**Existing saturation caveat:** `rmsnorm_apply.sv::sat_scale` checks product
bits `[31:28]` for sign extension but returns `[27:10]`; it does not include
the result sign bit 27 in the overflow check. Consequently `np.clip(v >> 10,
-131072, 131071)` is the intended saturation, not a bit-exact model over all
inputs. A standalone RTL probe with x=8192, scale=16384, gamma=512 produced
stage-A=-131072 and output=-16384; proper saturation would give output=16383.
The oracle must explicitly reproduce this behavior or a separate arithmetic
fix must establish the intended contract. The G1 execution test changes no RTL.

For comparison, capture all chunks at each block's `out_chunk_strobe` and
`out_chunk_idx_o`, as the G1 test's stream monitor does. Raw `out_valid` is
delayed relative to its data. The existing stage-capture readback only keeps
a prefix (G1 supplies eight 32-bit words on its first strobe), not both full
vectors. Compare raw norm outputs first; a seam-level comparison must also
model `r1_h_align` (`(hidden + 1) >> 1` when `PRECISION_SAFE=1`).

A hardware-oriented decomposition could expose vector sum-of-squares,
fixed-point reciprocal-square-root scale generation, and the two shifted
multiplies plus stores. Specify buffer/length operands, shift/width semantics,
and completion dependencies rather than G1-specific state IDs. In particular,
join both independent norm completions before consuming the R1 seam.

## Deferred work

- Depth-N under ISA control: requires conditional jump opcodes plus the
  `branch_j`/`level_i` loop counters (re-implement from `top_fsm.sv`).
- Runtime kvonly / G8 splicing conditionals (currently program-content
  only).
- Larger PMEM or wider instructions if 128 bytes / 8 opcodes become
  limiting.
