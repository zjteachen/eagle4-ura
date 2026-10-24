/*
* Instruction fetcher with a byte-granular PC over a 32-word PMEM and a
* 3-bit opcode / 5-bit operand instruction encoding:
*
*   instr[7:5] = opcode   instr[4:0] = operand
*
*   000 STATE  operand = state ID (0..31, see COMPONENTS.md "ISA State
*               Encodings"). Wait for the corresponding stage-done pulse on
*               adv_en before advancing.
*   001 JMP    operand = PMEM word address (0..31). PC <- {operand, 2'b00},
*               so jump targets are 4-byte (word) aligned; a program must
*               pad to a word boundary before any jump target. Executes
*               autonomously in one cycle (no adv_en needed).
*   010 HALT   Stop execution, pulse `halted` for one cycle. PC remains
*               pointing at the HALT byte for post-mortem readback.
*   011 WRSR   Write the special-register bank. The instruction is three
*               bytes: header = {3'b011, reg[3:0]} (0x60|reg), then the
*               16-bit immediate as two bytes, low byte first. Execute
*               autonomously in one cycle: assert sr_write_en for exactly
*               one cycle and PC += 3. The two immediate bytes are fetched
*               combinationally and may cross a PMEM word boundary; byte
*               addressing is modulo the 128-byte PMEM (e.g. a WRSR at PC
*               127 reads its high immediate byte from PC 0 and leaves the
*               next instruction at PC 2).
*   100..111   Reserved. Decodes as HALT (fail-safe stop) so an
*               under-programmed PMEM can never free-run the pipeline.
*
* Execution runs from PC=0 after each `start_en` pulse and ends at HALT.
* A STATE instruction is presented to the decoder for exactly one cycle
* via `state_pulse`; `is_state`/`state_id` stay valid for the whole wait
* so downstream muxes (e.g. stage_sel) can decode them combinationally.
*/

module instruction_fetcher (
    input  logic        clk,
    input  logic        rst_en,     // Active-low synchronous reset
    input  logic        start_en,   // 1-cycle launch pulse: PC <- 0, begin execution
    input  logic        adv_en,     // Active-high advance strobe (current STATE instr done)
    input  logic        write_en,   // Active-high PMEM write strobe
    input  logic [7:0]  write_addr, // Target word address in PMEM
    input  logic [31:0] write_data, // 32-bit word written to PMEM
    output logic [7:0]  instr,      // 8-bit (1 byte) instruction output
    output logic [6:0]  pc_out,     // Current program counter (0..127)
    output logic        state_pulse,// 1-cycle pulse when a STATE instruction begins
    output logic [4:0]  state_id,   // State ID of the current STATE instruction
    output logic        is_state,   // Level: current instruction is STATE and running
    output logic        running,    // Execution active (between start_en and HALT)
    output logic        halted,     // 1-cycle pulse when HALT/reserved opcode executes
    output logic        sr_write_en,   // 1-cycle WRSR bank write strobe
    output logic [4:0]  sr_write_sel,  // WRSR register address
    output logic [15:0] sr_write_data  // WRSR immediate (little-endian bytes)
);
    localparam logic [2:0] OP_STATE = 3'b000;
    localparam logic [2:0] OP_JMP   = 3'b001;
    localparam logic [2:0] OP_WRSR  = 3'b011;

    // 32 words of 32-bit storage (inferred as distributed RAM)
    // Total capacity = 32 words * 4 bytes/word = 128 bytes of instruction stream
    (* ram_style = "distributed" *)
    logic [31:0] PMEM [0:31];
    // Fail-safe: unwritten PMEM reads as HALT. Without this, simulation
    // fetches X past the programmed bytes while hardware reads 0 — both
    // trap, but X-propagates into the decode. A HALT fill makes a
    // missing-HALT program trap identically in sim and on FPGA.
    initial begin
        for (int i = 0; i < 32; i++) PMEM[i] = 32'h4040_4040;
    end

    // Byte-level PC tracking 128 bytes (7 bits: 0..127)
    logic [6:0] PC;
    // Execution state: armed by start_en, cleared by HALT.
    logic       run_q;
    // High for exactly the first cycle of the current instruction, so
    // state_pulse/halted are single-cycle even though the instruction byte
    // stays at PC for the whole (multi-cycle) stage wait.
    logic       new_instr;

    // Combinational 0-cycle fetch:
    // PC[6:2] (PC >> 2) selects the 32-bit word (0..31)
    // PC[1:0] selects the byte (0..3) within that word.
    // The helper is reused for WRSR immediates so a byte fetch is identical
    // regardless of whether the address is PC, PC+1 or PC+2.
    function automatic logic [7:0] fetch_byte(input logic [6:0] byte_addr);
        logic [31:0] word;
        word = PMEM[byte_addr[6:2]];
        case (byte_addr[1:0])
            2'd0:    fetch_byte = word[7:0];
            2'd1:    fetch_byte = word[15:8];
            2'd2:    fetch_byte = word[23:16];
            default: fetch_byte = word[31:24];
        endcase
    endfunction

    // 7-bit additions wrap modulo 128, matching the existing end-of-PMEM
    // policy. A WRSR at byte 127 therefore reads its high immediate from
    // byte 0 and leaves the next instruction at byte 2.
    wire [6:0] wrsr_imm_lo_addr = PC + 7'd1;
    wire [6:0] wrsr_imm_hi_addr = PC + 7'd2;

    assign instr = fetch_byte(PC);
    assign pc_out = PC;

    wire [2:0] op = instr[7:5];
    // JMP and WRSR execute autonomously (no done to wait for); HALT and
    // reserved opcodes stop execution.
    wire jmp_exec  = run_q && (op == OP_JMP);
    wire wrsr_exec = run_q && (op == OP_WRSR);
    wire halt_exec = run_q && (op != OP_STATE) && (op != OP_JMP) &&
                     (op != OP_WRSR);

    assign state_id    = instr[4:0];
    assign is_state    = run_q && (op == OP_STATE);
    assign state_pulse = new_instr && run_q && (op == OP_STATE);
    assign running     = run_q;
    assign halted      = new_instr && halt_exec;

    // WRSR bank port: exactly one write strobe per WRSR instruction. The
    // immediate is fetched little-endian (low byte first) and may straddle
    // any PMEM word boundary. A simultaneous start pulse wins (the write is
    // dropped along with the old PC), matching the fetcher's start priority.
    assign sr_write_sel  = instr[4:0];
    assign sr_write_data = {fetch_byte(wrsr_imm_hi_addr),
                            fetch_byte(wrsr_imm_lo_addr)};
    assign sr_write_en   = new_instr && wrsr_exec && !start_en;

    always_ff @(posedge clk) begin
        if (!rst_en) begin
            PC        <= '0;
            run_q     <= 1'b0;
            new_instr <= 1'b0;
        end else begin
            // Default: the current instruction is no longer new.
            new_instr <= 1'b0;
            if (start_en) begin
                // New launch: run the program from the top.
                PC        <= '0;
                run_q     <= 1'b1;
                new_instr <= 1'b1;
            end else if (run_q) begin
                if (halt_exec) begin
                    run_q <= 1'b0;
                end else if (jmp_exec) begin
                    // Word-aligned jump: target selects one of the 32 PMEM words.
                    PC        <= {instr[4:0], 2'b00};
                    new_instr <= 1'b1;
                end else if (wrsr_exec) begin
                    // Three-byte WRSR: skip the header and both immediate
                    // bytes. 7-bit arithmetic wraps at the end of PMEM.
                    PC        <= PC + 7'd3;
                    new_instr <= 1'b1;
                end else if (adv_en) begin
                    // Completed a STATE instruction: fetch the next byte.
                    // Wraps automatically at end of PMEM (128 bytes -> 0).
                    PC        <= (PC == 7'd127) ? 7'd0 : (PC + 7'd1);
                    new_instr <= 1'b1;
                end
            end
        end
    end
    // PMEM program port. Writes are ignored while the fetcher is running:
    // a mid-run write to the current instruction word would combinationaly
    // flip the decoded opcode (turning a pending STATE wait into a bogus
    // JMP/HALT). Hosts must program PMEM between launches; a write issued
    // during a run is dropped (the control_regs write pointer still
    // advances — treat a mid-run program as a host protocol error).
    always_ff @(posedge clk) begin
        if (write_en && !run_q && write_addr < 8'd32) begin
            PMEM[write_addr[4:0]] <= write_data;
        end
    end
endmodule
