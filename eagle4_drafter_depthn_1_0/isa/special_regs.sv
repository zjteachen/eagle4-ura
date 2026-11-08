/*
*/

module special_regs (
  input  logic        clk,
  input  logic        rst_en,       // Active-low synchronous reset
  input  logic        write_en,     // ISA WRSR write strobe
  input  logic [4:0]  write_select, // ISA WRSR register address
  input  logic [15:0] write_data,   // ISA WRSR immediate
  input  logic        read_en,      // AXI debug read request
  input  logic [4:0]  read_select,  // AXI debug register address
  output logic [15:0] read_data,    // Registered read data (holds when idle)
  output logic        read_ack,     // Read committed this cycle (may sample next)

  output logic [15:0] registers[15:0],
);
  localparam logic [4:0] NUM_REGS = 5'd11;

  logic [15:0] regs [0:10];

  always_ff @(posedge clk) begin
    if (!rst_en) begin
      for (int i = 0; i < 11; i++) regs[i] <= '0;
      read_data <= '0;
    end else begin
      if (write_en) begin
        // Any write (including a reserved one) suppresses the read so
        // read_data cannot be updated from storage in the same cycle.
        if (write_select < NUM_REGS) begin
          regs[write_select[3:0]] <= write_data;
        end
      end else if (read_en) begin
        read_data <= (read_select < NUM_REGS) ? regs[read_select[3:0]] : 16'd0;
      end
    end
  end

  // A read commits only when it is actually serviced. A simultaneous WRSR
  // write wins and leaves read_data untouched; the requester must retry.
  assign read_ack = rst_en && read_en && !write_en;

  assign activation_base        = regs[0];
  assign weight_base            = {regs[6][0], regs[1]};
  assign tile_count             = regs[2];
  assign chunks_per_tile        = regs[3];
  assign chunks_per_scale_group = regs[4];
  assign output_base            = regs[5];
  assign scale_base             = {regs[8][0], regs[7]};
  assign activation_source      = regs[9][13:11];
  assign output_destination     = regs[9][10:7];
  assign qzero_raw              = regs[9][6:5];
  assign scale_frac             = regs[9][4:0];
  assign output_conversion      = regs[10];

endmodule
