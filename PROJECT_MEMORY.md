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
  - **Code Review: instruction_fetcher.sv Initial Draft:** Identified syntax errors (trailing semicolon in port list, assignment to input ports `adv_en`/`write_en`), polarity inversion (`!adv_en`/`!write_en`), array indexing bug (`PMEM[PC >> 3][PC & 3'b111]` bit-slice indexing vs word indexing), PC mask truncation (`& 8'hF` cutting off 20 states at 16), and 1-cycle registered read latency. Provided refactored clean implementation matching ISA v0 specification.
