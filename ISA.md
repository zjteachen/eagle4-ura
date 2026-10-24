# Instruction set design for EAGLE4
This document describes the current ISA implemented by
`eagle4_drafter_depthn_1_0/src/instruction_fetcher.sv`. Iteration 1 is an
8-bit instruction stream over a 32-word (128-byte) PMEM.

- 8-bit instruction width.
- `instr[7:5]` is the opcode; `instr[4:0]` is the operand.
- The PC is byte-granular (0..127) and wraps modulo 128 on a sequential
  advance. JMP targets are 32-bit word aligned.

| Opcode | Mnemonic | Encoding | Behavior |
| :--- | :--- | :--- | :--- |
| `000` | STATE | `0x00 \| state_id` | Pulse the stage start for `state_id`; hold PC until that stage's done advances it |
| `001` | JMP | `0x20 \| word_addr` | `PC <- {word_addr, 2'b00}`; one cycle |
| `010` | HALT | `0x40` | Stop; one-cycle `halted`; PC freezes on the HALT byte |
| `011` | WRSR | `0x60 \| reg`, imm low, imm high | One special-register write; `PC += 3` |
| `100`-`111` | reserved | — | Decode as HALT (fail-safe) |

## WRSR

WRSR is a three-byte instruction:

```text
byte 0: {3'b011, reg[3:0]}    (0x60 | reg)
byte 1: immediate[7:0]
byte 2: immediate[15:0] high byte
```

The immediate is little-endian. Both immediate bytes are fetched
combinationally, so they may cross a 32-bit PMEM word boundary. Byte
addressing wraps modulo 128: a WRSR at PC 126 or 127 reads its high
immediate byte from PMEM byte 0 (and byte 1 respectively), and the following
instruction starts at PC 1 or 2. Exactly one bank write strobe is produced
per WRSR; a simultaneous `start_en` wins and suppresses it. Reserved
register addresses still strobe the bank but change no storage and read as
zero.

## Special-register bank

The bank is 11 x 16-bit (`special_regs.sv`), addresses 0..10:

| Addr | Field |
| ---: | :--- |
| 0 | `activation_base` |
| 1 | `weight_base[15:0]` |
| 2 | `tile_count` |
| 3 | `chunks_per_tile` |
| 4 | `chunks_per_scale_group` |
| 5 | `output_base` (tiles) |
| 6 | `weight_base[16]` |
| 7 | `scale_base[15:0]` |
| 8 | `scale_base[16]` |
| 9 | `{2'b0, activation_source[2:0], output_destination[3:0], qzero_raw[1:0], scale_frac[4:0]}` |
| 10 | `output_conversion` |

The bank decodes these into configuration outputs for the GEMV operation
controller; the controller integration (including numeric-field validation)
is a follow-on task. AXI readback: write the bank address to `0x178`, read
`0x17C` for `{16'h5A5A, read_data}`.

## State IDs

STATE operands reuse the `top_fsm` state enum numerically. See
`COMPONENTS.md` "ISA State Encodings & Instruction Set" for the full table.
HALT and reserved opcodes remain fail-safe stops; JMP remains word-aligned.
