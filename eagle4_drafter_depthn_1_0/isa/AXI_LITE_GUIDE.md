# AXI4-Lite Local PMEM Slave (Eagle4 ISA)

`isa/axi_lite.sv` is a small AXI4-Lite slave exposing a real local
`logic [31:0] pmem[0:PMEM_SIZE-1]` array. It preserves the public 9-bit
address port and `PMEM_SIZE` parameter used by the ISA demo. This is not the
instruction fetcher's PMEM; connecting it to a fetcher/top level is a separate
integration.

## Address map and parameter

- `PMEM_SIZE` is in 32-bit words, legal range 1..128. An elaboration-time
  `$error` rejects values outside that range.
- Base `0x000`, window `PMEM_SIZE * 4` bytes. The word index is
  `(byte_address - 0x000) >> 2`, so only aligned byte addresses are valid.
- `PMEM_SIZE=128` (default): `0x000..0x1FC`. This fills the 9-bit address
  space, so every aligned 9-bit address is in range and only misaligned
  addresses are errors.
- `PMEM_SIZE=16`: `0x000..0x03C`; `0x040` and above are out of range.
- In-window aligned access: OKAY (`2'b00`). Misaligned or outside the window:
  DECERR (`2'b11`), read data `0`, no write, regardless of `WSTRB`.
- `WSTRB[0]` writes bits `[7:0]`, ..., `WSTRB[3]` writes bits `[31:24]`. An
  all-zero strobe at a valid address is an OKAY no-op.

Example: with `PMEM_SIZE=64`, PMEM occupies `0x000..0x0FF`. To add a second
non-overlapping range, for example new memory at `0x100..0x1FF`:

1. In the write decode, compare `aw_addr` against `0x100` and the new size,
   select `index = (aw_addr - 0x100) >> 2`, and extend the commit mux with a
   mutually exclusive condition.
2. In the read decode, add the same matching compare and index for `araddr`.
3. Add the new backend in parallel (its own WSTRB merge and read-response
   mux). Keep ranges disjoint so exactly one backend responds.
4. If the new range is a real BRAM, add its read latency/response wait before
   asserting RVALID instead of snapshotting on the AR accept edge.

No extra nonfunctional memory or register range is declared in the module.

## Reset and latency

Reset is active-low and synchronous. On a rising edge with `rst_n == 0` the
whole PMEM array is cleared to zero, along with the AW/W buffers, B response,
and R response. This full-array reset is intentionally simple for teaching and
is not BRAM-optimized; a BRAM-backed version should reset only the protocol
state, not every memory word. If this PMEM is later backed by an external
memory/backend, keep the register/capture protocol and move the actual commit
and read-data path onto that backend's handshake.

The only addressable memory is `pmem`; the write-path `always_ff` is its sole driver.
Writes commit once after both AW and W are captured. Reads snapshot the
selected word on AR acceptance into RDATA/RRESP, so an AR accepted on the same
edge as a write commit returns the old word. One write and one read may be
outstanding independently.

## Handshake behavior

- AW and W are buffered separately and accepted only on a `VALID && READY`
  rising edge; either order or the same cycle is allowed.
- READY is combinational from the registered buffer/response state and reset,
  never from VALID. AW/W READY is low while its buffer is occupied or B is
  pending.
- BVALID/BRESP are asserted after the single commit and held until BREADY.
  No new AW/W is accepted while B is pending.
- RVALID/RDATA/RRESP are held until RREADY; ARREADY is low while a read
  response is outstanding.

## Simulation

The focused self-checking testbench is `isa/tb/tb_axi_lite.sv`. It
instantiates both the default 128-word DUT and a 16-word DUT, drives the
channels at negedge, checks after posedge, and covers ordering, strobes,
response backpressure, latched payloads, reset, and address bounds. From the
repository root:

```bash
make build/tb_axi_lite
./build/tb_axi_lite
```

The Makefile pattern rule compiles only this testbench plus the design
modules it instantiates (Verilator searches `isa/` and `src/` for them).
The executable is `build/tb_axi_lite`; generated build files go into
`build/obj_tb_axi_lite`. For another testbench, add `isa/tb/tb_<name>.sv`
with top module `tb_<name>`, then run `make build/tb_<name>` followed by
`./build/tb_<name>`.

Expected pass line: `TB PASS: <n> checks` (Verilator prints its own simulation
summary after it). The test exits nonzero on failure.
