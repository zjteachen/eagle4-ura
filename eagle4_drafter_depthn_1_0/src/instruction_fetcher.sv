/*
* Prototype instruction fetcher.
*/

module instruction_fetcher (
// declare inputs
/*
* What does instruction fetcher interface need to be?
* Externally, the master may want to reset the PC value
* the instruction fetcher and PMEM are coupled,
* thus needs a pmem write/write_en signal.
* Both wires as these are 1 cycle pulses that can be driven
* from both ends.
* Also an input is needed for the address.
*/
input wire clk;
input wire rst_en;
input wire adv_en;
input wire write_en;
input logic [7:0] write_addr;
input logic [31:0] write_data;

// declare outputs
/*
* The top level component will use the instruction fetch module to decide on
* what to do next. So the output of this component will likely change often as
* we work on the ISA.
* For now we can set it to which pipeline stage we want to proceed with (as
* the current FSM state machine does)
* So output can just be a 4-bit int.
*/
output logic [3:0] instr;
);
    // store PC
    logic [7:0] PC;
    logic [31:0] PMEM [31:0];

    always_ff @(posedge clk) begin
        if (!rst_en) begin
            PC <= '0;
        end else begin
            if (!adv_en) begin
                adv_en <= '1;
                PC <= (PC + 1) & 8'hf;
            end else begin
                PC <= PC;
            end
        end

        if (!write_en) begin
            write_en <= '1;
            PMEM[write_addr] <= write_data;
        end
        instr <= PMEM[PC >> 3][PC & 3'b111];
    end
endmodule
