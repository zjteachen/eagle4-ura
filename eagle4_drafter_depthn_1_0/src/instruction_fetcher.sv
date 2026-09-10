/*
* Prototype instruction fetcher.
*/

module instruction_fetcher (
    input  logic        clk,
    input  logic        rst_en,     // Active-low synchronous reset
    input  logic        adv_en,     // Active-high advance strobe
    input  logic        write_en,   // Active-high PMEM write strobe
    input  logic [7:0]  write_addr, // Target word address in PMEM
    input  logic [31:0] write_data, // 32-bit word written to PMEM
    output logic [7:0]  instr       // 8-bit (1 byte) instruction output
);
    // 32 words of 32-bit storage (inferred as distributed RAM)
    // Total capacity = 32 words * 4 bytes/word = 128 bytes of instruction stream
    (* ram_style = "distributed" *)
    logic [31:0] PMEM [0:31];

    // Byte-level PC tracking 128 bytes (7 bits: 0..127)
    logic [6:0] PC;
    always_ff @(posedge clk) begin
        if (!rst_en) begin
            PC <= '0;
        end else begin
            if (adv_en) begin
                // Increment by 1 byte; wraps automatically at end of PMEM (128 bytes -> 0)
                PC <= (PC == 7'd127) ? 7'd0 : (PC + 7'd1);
            end
        end

        if (write_en && write_addr < 8'd32) begin
            PMEM[write_addr[4:0]] <= write_data;
        end
    end

    // Combinational 0-cycle fetch:
    // PC[6:2] (PC >> 2) selects the 32-bit word (0..31)
    // PC[1:0] selects the byte (0..3) within that word
    wire [31:0] curr_word = PMEM[PC[6:2]];
    assign instr = (PC[1:0] == 2'b00) ? curr_word[7:0]   :
                   (PC[1:0] == 2'b01) ? curr_word[15:8]  :
                   (PC[1:0] == 2'b10) ? curr_word[23:16] :
                                        curr_word[31:24];
endmodule
