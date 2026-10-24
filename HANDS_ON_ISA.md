# Hands-on task: the `special_regs` bank (completed and expanded)

**Status:** complete. The original six-entry exercise bank was verified and
then expanded to the 11-entry ISA execution-configuration bank and wired
into the drafter (`eagle4_drafter.sv`) for WRSR writes and AXI readback.
This file now documents the implemented contract; it is no longer a
standalone-only exercise.

## Original exercise (historical)

The starting file was an unfinished draft with incomplete declarations and
an `always_ff` that ignored `rst_en`; it caused the normal Verilator build to
fail even though it was uninstantiated. The first task cleaned it up to a
six-entry bank with synchronous active-low reset, registered reads, and
reset > write > read precedence. That behavior is preserved for addresses
0..5.

## Implemented bank

`eagle4_drafter_depthn_1_0/src/special_regs.sv` now implements:

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

Addresses 11..31 are reserved. The module decodes the raw entries into
configuration outputs for the GEMV operation controller (the controller and
numeric validation are the follow-on integration task).

## Interface

Inputs:

- `clk`, `rst_en` — active-low synchronous reset (1 = normal).
- `write_en`, `write_select[4:0]`, `write_data[15:0]` — ISA WRSR write port.
- `read_en`, `read_select[4:0]` — AXI debug read port.

Outputs:

- `read_data[15:0]` — registered read data.
- `read_ack` — read committed this cycle. A simultaneous write suppresses
  the bank read (write priority), so `read_ack` stays low and the requester
  retries instead of sampling stale `read_data`.
- Decoded config fields: `activation_base`, `weight_base[16:0]`,
  `tile_count`, `chunks_per_tile`, `chunks_per_scale_group`,
  `output_base`, `scale_base[16:0]`, `activation_source[2:0]`,
  `output_destination[3:0]`, `qzero_raw[1:0]`, `scale_frac[4:0]`,
  `output_conversion`.

## Preserved semantics

- Reset > write > read.
- Reserved writes are ignored (no storage change); reserved reads return
  `16'h0000`.
- Any asserted write suppresses the read, even for a reserved address;
  `read_data` holds.
- `read_en` with no write updates `read_data` on the next rising edge;
  `read_data` holds whenever no read commits.

## Verification

- Standalone focused simulation exercises reset, all 11 registers,
  reserved addresses, write-suppressed reads/`read_ack`, and decoded config
  fields.
- The real top smoke (`sim_local/run_sim.sh regs`) uploads WRSR programs over
  AXI, executes them on the actual instruction fetcher/bank, reads the bank
  back through `0x178`/`0x17C`, and covers word crossings, unaligned WRSR
  starts, end-of-PMEM wrap, reserved addresses, restart, JMP/HALT, reserved
  opcodes, and a read/WRSR arbitration loop.

```bash
cd eagle4_drafter_depthn_1_0/sim_local
./run_sim.sh regs
```
