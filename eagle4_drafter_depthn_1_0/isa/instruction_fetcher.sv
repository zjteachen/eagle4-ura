/*
 * Minimal 16-bit word ISA instruction fetcher / CPU.
 *
 * Program memory (PMEM) is 32 x 16-bit words. The PC is a 5-bit word
 * address (0..31) and wraps naturally from 31 back to 0. Fetch is
 * combinational from PMEM[PC]; one instruction executes per clock while
 * running. There are 16 architectural registers of 16 bits, all cleared
 * by reset.
 *
 * Instruction encoding:
 *     [15:12] opcode
 *     [11:8]  dest
 *     [7:4]   srcA
 *     [3:0]   srcB
 *
 *     opcode 0 (NOOP): no register write, PC advances
 *     opcode 1 (HALT): stop, freeze PC, pulse halted for one cycle
 *     opcode 2 (ADD):  regs[dest] <- regs[srcA] + regs[srcB], PC advances
 *     opcode 3 (SUB):  regs[dest] <- regs[srcA] - regs[srcB], PC advances
 *     opcode 4..15:    fail-safe HALT (freeze PC, pulse halted once)
 *
 * Arithmetic is modulo 2^16. Reading a register in the same cycle that it
 * is written returns the old value.
 *
 * Control:
 *   - rst_en is an active-low synchronous reset. It clears PC, run state,
 *     and all 16 registers; PMEM contents are preserved.
 *   - start_en is a one-cycle launch pulse: PC <- 0 and running is set on
 *     the next clock. Register contents are preserved across a start.
 *     start_en has priority over instruction execution and PMEM writes.
 *   - running is high from launch until HALT (or an unknown opcode).
 *   - halted is a one-cycle pulse when HALT/unknown executes; PC freezes on
 *     that instruction until the next start_en.
 *   - The host programs PMEM through the write port while !running. Writes
 *     issued while running, or in the same cycle as start_en, are dropped.
 *
 * PMEM is initialized to HALT (16'h1000) so an unwritten program traps
 * instead of executing X/zero-filled garbage.
 */

module instruction_fetcher (
    input  logic        clk,
    input  logic        rst_en,      // Active-low synchronous reset
    input  logic        start_en,    // 1-cycle launch pulse: PC <- 0
    input  logic        write_en,    // Active-high PMEM write strobe
    input  logic [4:0]  write_addr,  // PMEM word address (0..31)
    input  logic [15:0] write_data,  // PMEM word data
    output logic [15:0] instr,       // Instruction word at PC
    output logic [4:0]  pc_out,      // Current word PC (0..31)
    output logic        running,     // Execution active (start until HALT)
    output logic        halted       // 1-cycle pulse on HALT/unknown opcode
);

    localparam int         PMEM_WORDS = 128;
    localparam int         NUM_REGS   = 16;
    localparam logic [3:0] OP_NOOP    = 4'd0;
    localparam logic [3:0] OP_HALT    = 4'd1;
    localparam logic [3:0] OP_ADD     = 4'd2;
    localparam logic [3:0] OP_SUB     = 4'd3;
    localparam logic [15:0] HALT_WORD = {OP_HALT, 12'h000};

    // 32 x 16-bit program memory. Initialized to HALT so an unwritten
    // program stops at PC 0 rather than executing undefined contents.
    logic [15:0] PMEM [0:PMEM_WORDS-1];

    initial begin
        for (int i = 0; i < PMEM_WORDS; i++) PMEM[i] = HALT_WORD;
    end

    // 16 x 16-bit architectural register file.
    logic [15:0] regs [0:NUM_REGS-1];

    logic [4:0] PC;
    logic       run_q;

    wire [3:0]  opcode = instr[15:12];
    wire [3:0]  dest   = instr[11:8];
    wire [3:0]  src_a  = instr[7:4];
    wire [3:0]  src_b  = instr[3:0];

    // HALT and every unknown opcode freeze execution. start_en suppresses
    // the pulse because a launch always wins.
    wire halt_decode = (opcode == OP_HALT) || (opcode > OP_SUB);

    assign instr   = PMEM[PC];
    assign pc_out  = PC;
    assign running = run_q;
    assign halted  = run_q && !start_en && halt_decode;

    // Sequential execution. PC+1 is 5 bits, so it wraps 31 -> 0.
    always_ff @(posedge clk) begin
        if (!rst_en) begin
            PC    <= '0;
            run_q <= 1'b0;
            for (int i = 0; i < NUM_REGS; i++) regs[i] <= '0;
        end else if (start_en) begin
            PC    <= '0;
            run_q <= 1'b1;
        end else if (run_q) begin
            // HANDLE INSTRUCTIONS HERE
            case (opcode)
                OP_NOOP: begin
                    PC <= PC + 5'd1;
                end
                OP_HALT: begin
                    run_q <= 1'b0;
                end
                OP_ADD: begin
                    regs[dest] <= regs[src_a] + regs[src_b];
                    PC         <= PC + 5'd1;
                end
                OP_SUB: begin
                    regs[dest] <= regs[src_a] - regs[src_b];
                    PC         <= PC + 5'd1;
                end
                default: begin
                    // Unknown opcode: fail safe by halting in place.
                    run_q <= 1'b0;
                end
            endcase
        end
    end

    // Host programming port. PMEM is writable only while idle; a mid-run
    // write could change the currently decoded instruction. A simultaneous
    // start_en also wins and drops the write.
    always_ff @(posedge clk) begin
        if (write_en && !run_q && !start_en) begin
            PMEM[write_addr] <= write_data;
        end
    end

endmodule
