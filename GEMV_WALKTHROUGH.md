# GEMV program walkthrough

This is a simulation-only, editable microprogram demo. It runs the real
`eagle4_drafter` RTL, the real `instruction_fetcher`, the real
`special_regs` bank and the real shared GEMV engine. It is not an FPGA
bitstream/boot flow and uses no trained checkpoint.

## Run

From the repository root:

```sh
make sim MODE=gemv-demo
```

That builds the Verilator model, runs the default program, and reads:

```text
eagle4_drafter_depthn_1_0/sim_local/tb/gemv_demo_program.hex
```

The run prints the program byte/PMEM-word count, the 11 special-register AXI
readbacks, `0x170` status, and the first actual/expected output lanes. It
then checks all 128 destination lanes against the case-A integer oracle.
The successful final line is:

```text
PASS GEMV DEMO: 35 bytes/9 PMEM words, HALT at PC=34, status DONE, no ERROR, all 128 lanes checked.
```

## Edit and rerun without rebuilding

`make sim MODE=gemv-demo` always rebuilds the simulator. To iterate on the
program file only, use the already-built binary:

```sh
cd eagle4_drafter_depthn_1_0/sim_local
$EDITOR tb/gemv_demo_program.hex
obj_dir_sim/sim_tb +GEMV_DEMO
```

`run_sim.sh` embeds the absolute default path at build time, so the same
binary can also be launched from the repository root. To run an alternate
file without touching the tracked default:

```sh
eagle4_drafter_depthn_1_0/sim_local/obj_dir_sim/sim_tb \
  +GEMV_DEMO +GEMV_DEMO_PROG=/tmp/my_gemv_program.hex
```

The `+GEMV_DEMO_PROG=` argument is read at run time; no RTL rebuild is
needed.

### Simulator vs FPGA boundary

- **Real RTL in simulation:** the top-level drafter, instruction fetcher,
  special-register bank, GEMV controller, lane array, destination buffers and
  serializers. The program is executed, not interpreted by the testbench.
- **Real control path:** the PMEM bytes are uploaded through the actual
  AXI-Lite PMEM mailbox (`0x168`/`0x16C`), the HFC activation vector through
  the actual AXI-Lite trio, and the run is started through `0x00`.
- **Simulation-only loading:** the deterministic per-lane WEIGHT and SCALE
  banks are written through the testbench direct-drive pins (`boot_we`,
  `tb_sc_we`). This is not an FPGA boot/DMA path. The testbench also reads
  `q_full_buf` directly for observation.
- **On FPGA:** a host must place the same bytes in PMEM and load the real
  weights, scales and activation through the design's boot/PCIe/DMA
  infrastructure. This demo does not claim that board bring-up flow.

## Default program (35 bytes)

Byte order is little-endian when four bytes are packed into one 32-bit PMEM
word. Byte 0 occupies bits `7:0`, byte 1 bits `15:8`, and so on.

```text
PC  bytes       field
 0  60 00 00    WRSR r0  activation_base = 0 (HFC source, chunk 0)
 3  61 00 00    WRSR r1  weight_base[15:0] = 0
 6  62 01 00    WRSR r2  tile_count = 1 (one 128-row tile)
 9  63 08 00    WRSR r3  chunks_per_tile = 8 (8 x 16 = 128 inputs)
12  64 08 00    WRSR r4  chunks_per_scale_group = 8 (one scale group)
15  65 00 00    WRSR r5  output_base = 0 (first destination tile)
18  66 00 00    WRSR r6  weight_base[16] = 0
21  67 00 00    WRSR r7  scale_base[15:0] = 0
24  68 00 00    WRSR r8  scale_base[16] = 0
27  69 AD 0C    WRSR r9  = 0x0CAD: src=1 HFC, dest=9 QKV, qzero=1, frac=13
30  6A 00 00    WRSR r10 output_conversion = 0
33  80          GEMV
34  40          HALT
```

`r9` is `{2'b0, activation_source[2:0], output_destination[3:0],
qzero_raw[1:0], scale_frac[4:0]}`. This fixed case-A fixture/oracle validates
only the default QKV destination 9 and the safe Q destination 0 edit:
`gv_check_case_a_outputs` always reads `q_full_buf[0]`, and the source, bases,
and dimensions stay fixed for the numeric demo. Other valid destination edits
may execute on the real engine but will fail these fixture expectations; that
is a demo-fixture mismatch, not an engine failure. The controller still
validates the numeric fields and selected source/destination spans before
launching any lane.

The 35 bytes become 9 PMEM words; the unused fourth byte of the last word is
padded with `0x40` (`HALT`). For example, word 0 is `0x6100_0060`
(`60 00 00 61`) and word 8 is `0x4040_8000` (`00 80 40` plus the pad).

## AXI and PMEM registers used

| Address | Use |
| ---: | :--- |
| `0x168` | PMEM write pointer, in 32-bit words. Write 0 to start. |
| `0x16C` | PMEM write data; each write auto-increments `0x168`. A read returns `{16'h504D, 1'b0, PC[6:0], instr[7:0]}`. Poll `0x168` until it reaches 9 before launching/checking instructions (the registered commit follows the AXI response). |
| `0x00` | `AP_CTRL`; write bit 0 = 1 to start. Read bits: 1 done, 2 idle, 3 ready. |
| `0x170` | GEMV status: bit 0 BUSY, bit 1 sticky DONE, bit 2 sticky ERROR. |
| `0x174` | Active/last GEMV tile index. |
| `0x178` | Special-register read selector (0..10). |
| `0x17C` | Special-register read data: `{16'h5A5A, data[15:0]}`. |

The actual run prints all 11 readbacks and checks them against the immediates
parsed from the hex file.

## Safe edit: QKV destination 9 -> Q destination 0

The default `r9` immediate is `0x0CAD` (bytes `AD 0C`). Edit PC 27..29 from:

```text
69 AD 0C
```

to:

```text
69 2D 08
```

which sets `r9 = 0x082D`, destination 0 (Q) instead of 9 (QKV composite).
For this one-tile case both destinations place the first tile's result in
`q_full_buf[0]`, so the observed values are unchanged. Rerun the built
binary (`obj_dir_sim/sim_tb +GEMV_DEMO`) and expect the same lanes, for
example lane 0 actual=58 expected=58, lane 1 actual=-44 expected=-44,
lane 2 actual=59 expected=59, lane 3 actual=-45 expected=-45, followed by
the all-128 PASS. Then restore `69 AD 0C`.

## Optional invalid scale-group edit

To see the validation path rather than a generic PASS, change PC 12..14
from `64 08 00` to `64 02 00` (`chunks_per_scale_group=2`, unsupported).
Rerun the same binary. It prints a `0x170` status with ERROR set, prints
`DEMO RESULT: ERROR`, and exits nonzero with
`configuration ERROR; ... no PASS claimed`. No array lane starts. Restore
`64 08 00` afterward.

## Where the fixture comes from

The tests are in
`eagle4_drafter_depthn_1_0/sim_local/tb/tb_eagle4_drafter.sv`:

- `gemv_x(0,k)` generates the deterministic HFC activation
  `((7*k+3) % 17) - 8`; it is uploaded through the real HFC AXI-Lite path.
- `gemv_wg(0,lane,0,k)` generates packed 2-bit weight codes; the decoded
  weight is `raw - 2`. `gv_load_weights` writes the real per-lane banks.
- `gemv_scale0(lane)` generates a distinct Q3.13 per-lane scale;
  `gv_load_scale` writes the real scale banks.
- `gv_load_case_a_fixture` is the shared case-A provisioning path.
  `gv_check_case_a_outputs` compares every lane against
  `gemv_oracle(0, lane, 0, 0, 8, 8, gemv_scale0(lane), 0)`.

The regression `make sim MODE=gemv` uses the same fixture/oracle helper for
its case A. The demo adds no RTL and no general assembler: the file must
parse as exactly 11 WRSR instructions targeting `r0..r10`, then GEMV
(`0x80`) and HALT (`0x40`). Missing, malformed, structurally wrong, empty,
and over-128-byte files exit nonzero before any upload or execution.
