# Hands-on task: a minimal 6 x 16-bit `special_regs` bank

**Scope:** one small, real project task. Turn the unfinished draft
`eagle4_drafter_depthn_1_0/src/special_regs.sv` into a usable six-entry bank.
Edit only that file: do not instantiate it, add ISA opcodes, or change GEMV or
control logic.

## Why this is a real blocker

The draft has six unused `gemv_*` fields (`:17-22`), an `initial` block with
four empty right-hand sides (`:24-30`), and an `always_ff` using an undeclared
`regs[]` while ignoring `rst_en` (`:32-38`). `sim_local/run_sim.sh:35` passes
every `src/*.sv` file to Verilator, so this un-instantiated module still stops
the normal build. Observed:

```text
$ make sim MODE=g1
%Error: .../src/special_regs.sv:26:23: syntax error, unexpected ';', ...
%Error: Exiting due to 4 error(s)  (same at lines 27-29)
make: *** [Makefile:7: sim] Error 1
```

## Provisional address map

Addresses 0..5 reuse the draft names: 0 `gemv_v_addr` (activation/vector
base), 1 `gemv_m_addr` (weight base), 2 `gemv_tile_count`, 3
`gemv_chunks_per_tile`, 4 `gemv_chunks_per_scale`, 5 `gemv_output_addr`
(destination base). This is **not** the full `ISA_GEMV_PLAN.md` contract; the bank is provisionally six entries (6..31 unsupported).

## Required contract

Keep the existing module name and ports exactly as they are: `clk`, `rst_en`,
`write_en`, `read_en`, `reg_select[4:0]`, `write_data[15:0]`,
`read_data[15:0]`.

- **Storage:** six 16-bit registers at addresses 0 through 5.
- **Reset:** synchronous, active-low (`rst_en == 0` asserts; 1 = normal); no
  `initial` block. At a rising edge it zeroes all six registers and `read_data`.
- **Write:** with `rst_en == 1`, `write_en == 1`, and `reg_select < 6`, store
  `write_data` at that address on the rising edge.
- **Read:** with `rst_en == 1`, `write_en == 0`, and `read_en == 1`, `read_data`
  gets the selected register next edge; invalid selects read `16'h0000`.
- **Invalid addresses:** writes to 6..31 are ignored; reads from 6..31 return
  zero. Create no registers beyond address 5.
- **Hold and precedence:** reset > `write_en` > `read_en`. Any asserted
  `write_en` suppresses the read, even for an invalid address; an invalid
  write changes no register and `read_data` holds. With `write_en == 0` and
  `read_en == 0`, `read_data` holds.

`rst_en` follows the same convention as `instruction_fetcher.sv:27,88`:
active-low synchronous reset, held at 1 during normal operation.

## Steps

1. Delete the six unused `gemv_*` declarations and the incomplete `initial`
   block. They drive no ports and cause the parse failure. Do not keep aliases.
2. Declare one `logic [15:0] regs [0:5];` array.
3. Write one `always_ff @(posedge clk)` block ordered as `if (!rst_en)`, then
   `else if (write_en)`, then `else if (read_en)`. Use nonblocking assignments,
   one driver for `read_data`, and a 5-bit range literal such as `5'd6`.
4. Let `read_data` hold when a write suppresses read or `read_en` is low; do not assign it combinationally.

## Check the result

From the IP directory, lint only this module (the rest of the RTL is excluded):

```bash
cd eagle4_drafter_depthn_1_0
verilator --lint-only --timing -Wall --Wno-fatal -Wno-DECLFILENAME \
  --top-module special_regs src/special_regs.sv
echo "EXIT=$?"
```

**Expected:** no `%Error`, exit status 0. Lint checks syntax and elaboration
only; it does not simulate the behavior below. Treat the checklist as a manual
trace for now, not as lint-proven results. Warnings are tolerated.

## Success checklist

- `rst_en=0` for one edge, then with `rst_en=1` read address 0: value is zero.
- Write `16'hBEEF` to address 2, then read it: next-cycle value is `16'hBEEF`.
- Write address 6 with `write_en=1`: no register changes and the read is
  suppressed, so `read_data` holds; a later read of address 6 with
  `write_en=0` returns `16'h0000`.
- Any `write_en` plus `read_en` in one cycle: read is ignored, `read_data`
  holds.
- `read_en=0`: `read_data` keeps its last value rather than becoming X.

Stop at a standalone, lint-clean bank. Integration into `eagle4_drafter.sv` and
the final GEMV register map are follow-on work.
