/*
* Special register module that can be set by ISA.
* Used to tune behaviour of hardware modules.
* Short term goal: to allow ISA control over tensor shapes in GEMV engine.
* Write interface faces ISA, read interface faces AXI for debugging.
*
* The bank is 11 x 16-bit (addresses 0..10). Addresses 11..31 are reserved:
* writes to them are ignored and reads from them return 16'h0000.
*
* Register mapping (execution configuration is latched by the GEMV engine
* when it lands; this bank only stores / decodes the fields):
* 0  activation_base          starting activation chunk
* 1  weight_base[15:0]        packed 32-bit word base in each lane's bank
* 2  tile_count               output tiles for the operation
* 3  chunks_per_tile          activation/weight chunks processed per tile
* 4  chunks_per_scale_group   chunks sharing one scale entry
* 5  output_base              first output tile inside the destination buffer
* 6  weight_base[16]          bit 16 of the weight base
* 7  scale_base[15:0]         scale-entry base in each lane's scale bank
* 8  scale_base[16]           bit 16 of the scale base
* 9  {2'b0, activation_source[2:0], output_destination[3:0],
*      qzero_raw[1:0], scale_frac[4:0]}
* 10 output_conversion        downstream store conversion selector
*
* Precedence is preserved from the original six-entry exercise bank:
* reset > write > read. Any asserted write_en suppresses the read, even for a
* reserved address; read_data holds in that case. read_ack is the
* combinational commit indicator for the read port: it is low when a
* simultaneous write suppressed the read, letting the AXI readback retry
* instead of silently returning stale read_data.
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

  // Execution configuration fields decoded from the raw registers.
  // The GEMV engine latches these at START in the follow-on integration task.
  output logic [15:0] activation_base,
  output logic [16:0] weight_base,
  output logic [15:0] tile_count,
  output logic [15:0] chunks_per_tile,
  output logic [15:0] chunks_per_scale_group,
  output logic [15:0] output_base,
  output logic [16:0] scale_base,
  output logic [2:0]  activation_source,
  output logic [3:0]  output_destination,
  output logic [1:0]  qzero_raw,
  output logic [4:0]  scale_frac,
  output logic [15:0] output_conversion
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
