# Project Memory: EAGLE-4 Drafter ISA Development

## Context & Objectives
- **Target System:** EAGLE-3/4 speculative draft head FPGA IP on AMD Xilinx Alveo V80 (`eagle4_drafter_depthn_1_0/`).
- **Core Assignment:** Replace the hardcoded state machine (`top_fsm.sv`) with an instruction fetcher and program memory (PMEM), enabling programmable stage sequencing across different model architectures without RTL rebuilds.
- **Milestone v0 Target:** Implement a sequencer ISA that reproduces the 20-state transition graph of `top_fsm.sv` bit-identically.
- **Milestone v1+ Target:** Programmable branch predicates, assembler/compiler for arbitrary stage orderings, and inter-stage pipelining.

## Development Baseline & Verification
- **IP RTL Status:** Untouched (`git diff 706a98c -- src/` is empty).
- **Local Verification:**
  - Lint: `make lint` (Verilator 5.046 with `xpm_memory_stubs.sv`).
  - Simulation: `make sim MODE=fast` (~6.3s Tier A smoke test, verifies I/O, control flow, and `OUT_ABI_V24` frontier formatting).
- **Core Documentation:**
  - `BUILD.md`: Commands and simulation caveats (SCALE residency tap, RoPE LUT).
  - `TASKS.md`: Active ISA v0 work items, architectural edge cases, and known RTL gaps.
  - `COMPONENTS.md`: Encyclopedia of all 84 RTL components with full ASCII datapath wiring diagrams.

## Workflow Protocol
1. User poses FPGA/architecture/implementation questions.
2. Inquiries are delegated to specialized subagents (`task` tool) for targeted analysis.
3. Summary findings, design decisions, and timeline progression are recorded concisely into this document.

## Timeline & Milestones
- **2026-08-16:** Baseline commit `706a98c` (packaged IP, initial lint tooling, build scripts).
- **2026-09-07:**
  - Local Tier A simulation testbench added (`sim_local/tb/`, `run_sim.sh`, passing clean).
  - Streamlined `BUILD.md` and `TASKS.md`.
  - Authored comprehensive `COMPONENTS.md` encyclopedia and ASCII architecture diagrams.
  - Initialized ISA v0 architecture memory.
  - **AXI4-Lite & PMEM Loading Architecture:** Analyzed top-level `s_axi_control` and `control_regs.sv`. Selected dedicated auto-incrementing write register pattern at offsets `0x168` (`pmem_waddr`) / `0x16C` (`pmem_wdata`) / `0x170` (`pmem_status`), gated by `wr_fire`.
  - **Physical Memory Primitive Selection:** Evaluated FPGA storage tiers for PMEM v0 (20 words x 32b = 640 bits). Rejected URAM (wastes 99.8% of primitive, needed for weights) and BRAM (wastes 98%, introduces mandatory 1-cycle read latency). Selected **Distributed RAM (LUTRAM / SLICEM)** via `(* ram_style = "distributed" *)` to provide 0-cycle combinational read (`assign fetch_instr = mem[pc]`) matching `top_fsm` zero-latency state transitions.
  - **SystemVerilog Type Rules & PMEM Instantiation Workflow:** Clarified `logic` vs `wire`/`reg` (use `logic` everywhere; `wire` only for multi-driver/tri-state/`inout`; `reg` is legacy Verilog). Established the 3-step PMEM creation workflow: (1) instantiate `pmem_lutram` in `eagle4_drafter.sv` as local storage, (2) drive write port from `control_regs.sv` (`pmem_prog_we`, `pmem_waddr`, `pmem_wdata`), (3) connect read port (`fetch_pc`, `fetch_instr`) to `instruction_fetcher.sv`.
  - **FPGA Fundamentals Codified:** Documented (1) Top-level port-to-FPGA pin mapping via IP-XACT `component.xml` and Vivado Block Design SmartConnect/QDMA shell, (2) SystemVerilog array syntax (packed bit-vector width `[DATA_W-1:0]` before symbol vs unpacked word depth `[0:DEPTH-1]` after symbol), and (3) AXI 5-channel decoupled architecture and cycle-by-cycle `VALID`/`READY` handshakes.
  - **Developer Conceptual Model Validated:** User confirmed core mental model: (1) Static RTL declaration & synthesis primitive inference vs dynamic runtime allocation, (2) AXI addresses as decoded register offsets routed via HDL multiplexers/decoders rather than physical pointers, and (3) PMEM implementation pattern pairing a dual-port storage module with an AXI register decode interface.
  - **AXI Addressing vs Internal Memory Sizing:** Decoupled AXI bus address width (`awaddr[8:0]`, 9-bit byte addressing where `[1:0]` are byte-offset lanes) from internal memory array address width (`ADDR_W = $clog2(DEPTH)` word index). Confirmed PMEM internal addressing requires only $6$ bits (`[5:0]`) for 64 words, decoupled via the `control_regs.sv` `pmem_waddr` pointer.
  - **Host Programming Model & Data Width Matching:** Confirmed 1:1 mapping between 32-bit AXI data bus (`s_axi_control_wdata[31:0]`) and 32-bit PMEM instruction word, eliminating word-packing/deserialization logic. Established host-side auto-increment mailbox pattern: host writes start word once to `0x168` (`ADDR_PMEM_ADDR`), then streams $N$ instructions to the identical address `0x16C` (`ADDR_PMEM_DATA`).
  - **Mailbox Responsibility & Placement:** Established modular boundary: `pmem_lutram.sv` is pure storage (no AXI knowledge); `control_regs.sv` owns mailbox registers (`0x168`/`0x16C`), `wr_fire` single-cycle write strobe, and auto-increment pointer logic; `eagle4_drafter.sv` interconnects them.
  - **Prototype instruction_fetcher.sv Implemented:** Implemented minimal sequential fetcher without speculative branching. 32-word x 32-bit PMEM (128 bytes total). 8-bit instruction output (`instr[7:0]`). Byte-level PC (`PC[6:0]`) with wrap-around at 127 (`PC == 7'd127 ? 0 : PC + 1`). 0-cycle combinational fetch: `PC[6:2]` selects 32-bit word, `PC[1:0]` selects byte slice. Verified clean compilation via `make lint`.
  - **Code Review: control_regs.sv PMEM Additions:** Analyzed user modifications. Identified compiler syntax errors (missing comma after `batch_sib_hi`, semicolons in port list, `output wire` driven procedurally inside `always_ff`), address collision (offset `0x164` conflicts with `obs_rd_data`), comparison bug (`8'b1` toggling between 0 and 1 instead of 32 words), and missing `pmem_we` write strobe required to drive `instruction_fetcher.write_en`.
  - **AXI Mailbox Architecture Clarification:** Clarified single-register vs two-register PMEM loading. Single-register (`0x16C` only) is valid for always loading from word 0 with auto-increment. The second register (`0x168`) exists solely for "seek/random access" (re-uploading without reset or patching individual words). Single-register model accepted for simplicity if PMEM is always loaded linearly from 0.
  - **Single-Register Mailbox Implemented in control_regs.sv:** Integrated single-register AXI mailbox at `0x16C`. Added `output reg [31:0] pmem_write_data`, `output reg [7:0] pmem_write_addr`, and `output reg pmem_we`. Reset initializes `pmem_write_addr <= 0`. On commit of `0x16C`, `pmem_we` pulses for 1 cycle and `pmem_write_addr` auto-increments with wrap at 32 words (`8'd31 -> 8'd0`). Verified `make lint control_regs` clean.
- **2026-09-09:**
  - **Instruction Fetcher & PMEM Fully Integrated & Verified:**
    - In `control_regs.sv`: Added `0x168` write-address seek/readback and `0x16C` status readback `{16'h504D, 1'b0, fetch_pc, fetch_instr}`.
    - In `instruction_fetcher.sv`: Added `pc_out` output port for observability.
    - In `eagle4_drafter.sv`: Instantiated `instruction_fetcher u_fetcher`, wired `pmem_write_data_w`, `pmem_write_addr_w`, `pmem_we_w` from `control_regs`, wired `adv_en` to `start_pulse`, and connected `fetch_instr` / `fetch_pc` back to `control_regs` for live status readback.
    - In `sim_local/tb/tb_eagle4_drafter.sv` & `eagle4_regmap_pkg.sv`: Added automated self-check testing PMEM write-pointer reset via `0x168`, auto-incrementing word loading via `0x16C`, 0-cycle combinational instruction fetch at PC=0, and PC advancement + next instruction fetch upon kernel launch (`ap_start`).
    - Verified: `make lint` passes cleanly; `make sim MODE=fast` passes end-to-end with 0 errors.
- **2026-10-24: WRSR + 11-entry `special_regs` bank integrated.**
  - `instruction_fetcher.sv`: opcode `011` (WRSR) is a three-byte instruction
    (`0x60|reg`, imm low, imm high). Executes in one cycle, produces exactly
    one bank write strobe, advances PC by 3. Immediates are fetched
    combinationally and may cross PMEM word boundaries; byte addressing
    wraps modulo 128 (PC 126/127 read wrapped bytes). Reserved opcodes
    `100..111` remain HALT; JMP/STATE/HALT semantics unchanged.
  - `special_regs.sv`: expanded to 11 x 16-bit entries (0..10) while
    preserving reset > write > read precedence, reserved-write ignore /
    reserved-read zero, and read hold. Added a separate AXI read selector,
    `read_ack`, and decoded configuration outputs for the GEMV controller.
    Addresses: activation_base, weight_base low/high, tile_count,
    chunks_per_tile, chunks_per_scale_group, output_base, scale_base
    low/high, packed routing/numeric word, output_conversion.
  - `control_regs.sv`: `0x178` write/read selector, `0x17C` read
    `{16'h5A5A, read_data}` with a WAIT/CAPTURE state machine. A WRSR write
    suppresses the bank read; the AXI read retries on `read_ack` instead of
    returning stale data.
  - Verification: standalone `special_regs` and `instruction_fetcher`
    directed simulations; new full-top `+REGS_ONLY` mode
    (`make sim MODE=regs`) covering all 11 registers, crossings, unaligned
    WRSR, reserved addresses, restart, JMP/HALT, reserved opcode, end-of-PMEM
    wrap, and a 22-write/23-cycle arbitration loop; existing `+G1_ONLY`
    re-passed to preserve STATE semantics.
