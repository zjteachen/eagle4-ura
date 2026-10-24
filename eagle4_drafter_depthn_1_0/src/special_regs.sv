/*
* Special register module that can be set by ISA.
* Used to tune behaviour of hardware modules.
* Short term goal: to allow ISA control over tensor shapes in GEMV engine.
* Write interface faces ISA, read interface faces AXI for debugging.
* Register mapping:
* 0:
* 1:
* 2:
* 3:
* 4:
* 5:
*/

module special_regs (
  input logic clk,
  input logic rst_en,
  input logic write_en,
  input logic read_en,
  input logic [4:0] reg_select,
  input logic [15:0] write_data,
  output logic [15:0] read_data
);
  logic [15:0] regs[0:5];

  always_ff @(posedge clk) begin
    if (!rst_en) begin
      for (int i=0; i<6; i++) regs[i] <= '0;
      read_data <= '0;
    end else begin
      if (reg_select < 5'd6) begin
        if (write_en) begin 
          regs[reg_select[2:0]] <= write_data;
        end else if (read_en) begin
          read_data <= regs[reg_select[2:0]];
        end
      end else begin
        if (read_en && !write_en) begin
          read_data <= '0;
        end
      end
    end
  end

endmodule
