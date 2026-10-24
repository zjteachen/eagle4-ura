# Instruction set design for EAGLE4
This document describes the current ISA implemented by
`eagle4_drafter_depthn_1_0/src/instruction_fetcher.sv` and
`gemv_shared_ctrl.sv`. Iteration 1 is an 8-bit instruction stream over a
32-word (128-byte) PMEM.

- 8-bit instruction width.
- `instr[7:5]` is the opcode; `instr[4:0]` is the operand.
- The PC is byte-granular (0..127) and wraps modulo 128 on a sequential
  advance. JMP targets are 32-bit word aligned.

| Opcode | Mnemonic | Encoding | Behavior |
| :--- | :--- | :--- | :--- |
| `000` | STATE | `0x00 \| state_id` | Pulse the stage start for an implemented `state_id`; hold PC until that stage's done advances it. Retired/unassigned IDs decode as HALT. |
| `001` | JMP | `0x20 \| word_addr` | `PC <- {word_addr, 2'b00}`; one cycle |
| `010` | HALT | `0x40` | Stop; one-cycle `halted`; PC freezes on the HALT byte |
| `011` | WRSR | `0x60 \| reg`, imm low, imm high | One special-register write; `PC += 3` |
| `100` | GEMV | `0x80` only | Latch the actual bank configuration and run the shared GEMV engine; hold PC until operation DONE or a validated configuration ERROR advances it |
| `100` other operands, `101`-`111` | reserved | — | Decode as HALT (fail-safe) |

## WRSR

WRSR is a three-byte instruction:

```text
byte 0: {3'b011, reg[4:0]}    (0x60 | reg)
byte 1: immediate[7:0]
byte 2: immediate[15:8]
```

The immediate is little-endian. Both immediate bytes are fetched
combinationally, so they may cross a 32-bit PMEM word boundary. Byte
addressing wraps modulo 128: at PC 126 the low immediate is byte 127, the
high immediate is byte 0, and the next instruction starts at PC 1; at
PC 127 the low immediate is byte 0, the high immediate is byte 1, and the
next instruction starts at PC 2. Exactly one bank write strobe is produced
per WRSR; a simultaneous `start_en` wins and suppresses it. Reserved
register addresses still strobe the bank but change no storage and read as
zero.

## Special-register bank

The bank is 11 x 16-bit (`special_regs.sv`), addresses 0..10:

| Addr | Field |
| ---: | :--- |
| 0 | `activation_base` (chunk units) |
| 1 | `weight_base[15:0]` (flat 32-bit word units) |
| 2 | `tile_count` |
| 3 | `chunks_per_tile` |
| 4 | `chunks_per_scale_group` |
| 5 | `output_base` (tiles) |
| 6 | `weight_base[16]` |
| 7 | `scale_base[15:0]` (flat scale-entry units) |
| 8 | `scale_base[16]` |
| 9 | `{2'b0, activation_source[2:0], output_destination[3:0], qzero_raw[1:0], scale_frac[4:0]}` |
| 10 | `output_conversion` |

The GEMV controller latches these outputs at GEMV START (and at TREE
session start). They are the sole source of shared-engine bases, bounds,
activation/output routing and numeric mode; no stage-derived fallback is
used.

### Activation sources (register 9 bits 13:11)

| Code | Source | Capacity (16-element chunks) |
| ---: | :--- | ---: |
| 0 | external `a_chunk` stream (base must be 0) | stream |
| 1 | `h_fc_buf` | 192 |
| 2 | `input_emb_buf` | 192 |
| 3 | R1 (post-G1 hidden+embedding norm) | 384 |
| 4 | pre-MLP / R2 | 192 |
| 5 | R5 | 512 |
| 6 | low-rank (GLM output) | 16 |
| 7 | attention (G5 -> G6) | 192 |

Source address = `activation_base + chunk` inside that buffer. The array's
exposed chunk index keeps its SLR lookahead, so the existing one-cycle
activation/weight/scale alignment is preserved.

### Output destinations (register 9 bits 10:7)

| Code | Destination | Capacity (tiles) |
| ---: | :--- | ---: |
| 0 / 1 / 2 | Q / K / V single buffer | 24 each at 128 lanes |
| 3 | R2 (`g6out_buf` + seam) | 24 |
| 4 / 5 | R4 gate / up | 64 |
| 6 | R6 / residual down (`g9down_buf`, or G9 high half with FORREASON) | 24 (48 with FORREASON) |
| 7 | TREE client stream (client mode only) | 1002 |
| 9 | QKV composite (Q then K then V placement tiles) | 72 |

Placement tile = `output_base + engine tile`. Destination-specific
serialization/conversion is retained: Q/K/V store 128 INT32 words; R2 and
R6 store eight 16-bit subchunks; R4 stores eight 18-bit q6.12 subchunks.

## Validation and fail-safe behavior

At START the controller validates, before any lane start:

- `tile_count > 0`, `chunks_per_tile in 1..K_MAX/16`.
- `chunks_per_scale_group ∈ {4, 8}` and `chunks_per_tile % cpg == 0`.
- `qzero_raw == 1`, `scale_frac == 13`, `output_conversion == 0`.
- Weight span `weight_base + tile_count*chunks_per_tile` fits the per-lane
  weight bank; scale span `scale_base + tile_count*(cpt/cpg)` fits the scale
  bank (all in 32-bit wide arithmetic).
- Activation span fits the selected source capacity (external source must
  use base 0).
- Destination span `output_base + tile_count` fits the destination capacity;
  TREE destination 7 is client-only; reserved destinations are rejected.

An invalid configuration produces no array start, sets sticky ERROR, and
pulses the ISA error termination so the PC advances to the next instruction
(typically HALT). A new GEMV START clears sticky DONE and ERROR.

## Status and completion

- `0x170` R: bit0 BUSY, bit1 sticky DONE, bit2 sticky ERROR.
- `0x174` R: active/last engine tile index.
- DONE is asserted only after the destination adapter reports the final
  store/serialize complete. A TREE session keeps BUSY high across the whole
  walk and streaming drain, and latches sticky DONE on the explicit
  `tree_done` session-completion handshake.
- `0x178` W/R: bank read selector; `0x17C` R: `{16'h5A5A, read_data}`.

## State IDs

STATE operands reuse the `top_fsm` state enum numerically. Implemented:
G1=`1`, G5=`3`, G7_SILU=`7`, G11=`9`, TREE=`11`, GLM=`13`, G8=`14`.
The former shared-stage operands G4=`2`, G6=`4`, G7G=`5`, G7U=`6`,
G9=`8`, and G12=`10` are retired: the fetcher decodes them as HALT, and
their engine work is now expressed by GEMV instructions. G12 is no longer a
standalone program step: TREE is the sole G12 client and drives the same
configured engine tile-by-tile with its own stream backpressure. The TREE
scan length is the actual ISA `tile_count`, launch-sampled by the tiler and
validated to 1..1002 (physical TREE capacity) plus the weight/scale spans.
The legacy AXI `0x14C` scan register is removed and the address is reserved
(writes ignored, reads 0); it has no execution effect.

`COMPONENTS.md` "ISA State Encodings & Instruction Set" has the full
historical state table.

## Compiled-mode scope

The production shared path is `GEMV_SCALED=1`, `GEMV_KPAR64=0` (K_PAR=16,
128 lanes by default). The ISA latch/validation, explicit weight/scale bases,
runtime tile bounds and GEMV START/status apply to every shared-array build
variant, but the fixed numeric contract above (packed 2-bit weights, INT16
activations, qzero=1, Q3.13 scales, scaled 48-bit accumulator) is implemented
only by the scaled arm. The dedicated unscaled K_PAR=16 and K_PAR=64 arms
receive the same explicit controller base/bounds through their API adaptation
and remain outside the new scaled numeric path; they do not get a legacy
stage-derived fallback. lmdown/GLM, G8 rescore and the fp16 tail remain
separate engines. `DEPTH_N=1` is still rejected by the ISA build guard.
TREE scan-width variants (`TREE_P`, `TREE_PP`) all consume the same shared
engine client interface.
