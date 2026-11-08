/*
* Small AXI4-Lite slave with a PMEM_SIZE-word local memory.
*
* This is the local PMEM window used by the ISA demo, not the instruction
* fetcher's PMEM. A later integration can connect this memory to the fetcher,
* or replace the storage with an external memory/backend handshake while
* keeping the AXI protocol and payload-capture logic here.
*/

module axi_lite #(
  parameter int PMEM_SIZE = 128 // local PMEM size in 32-bit words, 1..128
) (
  input wire clk,
  input wire rst_n,

  input  wire [8:0]  s_axi_control_awaddr,   // byte address, local window base 0x000
  input  wire        s_axi_control_awvalid,
  output reg         s_axi_control_awready,

  input  wire [31:0] s_axi_control_wdata,
  input  wire [3:0]  s_axi_control_wstrb,
  input  wire        s_axi_control_wvalid,
  output reg         s_axi_control_wready,

  output reg  [1:0]  s_axi_control_bresp,
  output reg         s_axi_control_bvalid,
  input  wire        s_axi_control_bready,

  input  wire [8:0]  s_axi_control_araddr,   // byte address, local window base 0x000
  input  wire        s_axi_control_arvalid,
  output reg         s_axi_control_arready,

  output reg  [31:0] s_axi_control_rdata,
  output reg  [1:0]  s_axi_control_rresp,
  output reg         s_axi_control_rvalid,
  input  wire        s_axi_control_rready
);

  localparam logic [1:0] RESP_OKAY   = 2'b00;
  localparam logic [1:0] RESP_DECERR = 2'b11;
  localparam logic [9:0] PMEM_BYTES  = 10'(PMEM_SIZE * 4);
  localparam int         INDEX_W     = (PMEM_SIZE > 1) ? $clog2(PMEM_SIZE) : 1;

  // Elaboration check: one word uses four byte addresses, and the 9-bit
  // address port can never reach past 128 words.
  generate
    if (PMEM_SIZE < 1 || PMEM_SIZE > 128) begin : gen_bad_pmem_size
      $error("axi_lite: PMEM_SIZE must be in 1..128 words");
    end
  endgenerate

  // Actual local memory. Only the write-path always_ff drives this array;
  // the read path only reads it. Reset clears every word, which is simple
  // for teaching but not BRAM-optimized (see the guide).
  logic [31:0] pmem [0:PMEM_SIZE-1];

  // AW and W are buffered separately so either channel may arrive first.
  logic        aw_pending;
  logic [8:0]  aw_addr;
  logic        w_pending;
  logic [31:0] w_data;
  logic [3:0]  w_strb;

  logic [INDEX_W-1:0] w_index; // (aw_addr - 0x000) >> 2
  logic [INDEX_W-1:0] r_index; // (araddr  - 0x000) >> 2
  logic               w_hit;
  logic               r_hit;

  // Byte-lane merge for WSTRB: selected bytes come from the bus, the rest
  // keep the current PMEM word. An all-zero strobe therefore changes nothing.
  function automatic logic [31:0] merge_wstrb(
    input logic [31:0] old_word,
    input logic [31:0] new_word,
    input logic [3:0]  strb
  );
    logic [31:0] merged;
    merged = old_word;
    for (int byte_lane = 0; byte_lane < 4; byte_lane++) begin
      if (strb[byte_lane]) begin
        merged[byte_lane*8 +: 8] = new_word[byte_lane*8 +: 8];
      end
    end
    return merged;
  endfunction

  // Write address decode: PMEM occupies 0x000..PMEM_SIZE*4-1. A word is
  // selected by the aligned byte address, so addr[1:0] must be zero and the
  // whole address must be below PMEM_BYTES. To add a second non-overlapping
  // range later (for PMEM_SIZE=64: PMEM 0x000..0x0FF, new memory 0x100..0x1FF),
  // compare aw_addr against that new base/size here, select its own index
  // ((aw_addr - base) >> 2), and extend the commit mux below with the same
  // mutually exclusive condition. Mirror the range in the read decode.
  always_comb begin
    w_hit   = (aw_addr[1:0] == 2'b00) && ({1'b0, aw_addr} < PMEM_BYTES);
    w_index = aw_addr[2 +: INDEX_W];
  end

  // Read address decode: same byte-address rules as the write side, using
  // the AR payload while it is being accepted. To add another range (for
  // PMEM_SIZE=64: new memory at 0x100..0x1FF), add the matching base/size
  // compare here, select (araddr - base) >> 2, and extend the read snapshot
  // mux below. If the new range is a real BRAM, add its read-latency/response
  // wait instead of returning pmem data on the accept edge.
  always_comb begin
    r_hit   = (s_axi_control_araddr[1:0] == 2'b00) &&
              ({1'b0, s_axi_control_araddr} < PMEM_BYTES);
    r_index = s_axi_control_araddr[2 +: INDEX_W];
  end

  // Response/ready: READY depends only on registered buffers, B/R state, and
  // reset. It never depends on VALID, so a master may assert VALID whenever it
  // has payload. AW/W are blocked while a write response is outstanding.
  always_comb begin
    s_axi_control_awready = rst_n && !aw_pending && !s_axi_control_bvalid;
    s_axi_control_wready  = rst_n && !w_pending  && !s_axi_control_bvalid;
    s_axi_control_arready = rst_n && !s_axi_control_rvalid;
  end

  // Write-path sequencing: AW/W accept, one-shot commit storage, and B
  // response. Buffers are captured only on VALID && READY edges, so payload
  // changes after acceptance do not affect the committed write.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      // Reset behavior: clear both write buffers, B response state, and the
      // whole local PMEM array. Full-array synchronous reset is deliberately
      // simple and is not how a BRAM-backed implementation should reset.
      aw_pending            <= 1'b0;
      aw_addr               <= 9'b0;
      w_pending             <= 1'b0;
      w_data                <= 32'b0;
      w_strb                <= 4'b0;
      s_axi_control_bvalid  <= 1'b0;
      s_axi_control_bresp   <= RESP_OKAY;
      for (int i = 0; i < PMEM_SIZE; i++) begin
        pmem[i] <= 32'b0;
      end
    end else begin
      // Write accept: each channel is captured independently, once, when its
      // own VALID && READY edge occurs. AW and W may arrive in either order or
      // together. READY was low while its buffer was occupied.
      if (s_axi_control_awvalid && s_axi_control_awready) begin
        aw_pending <= 1'b1;
        aw_addr    <= s_axi_control_awaddr;
      end

      if (s_axi_control_wvalid && s_axi_control_wready) begin
        w_pending <= 1'b1;
        w_data    <= s_axi_control_wdata;
        w_strb    <= s_axi_control_wstrb;
      end

      // Commit storage: exactly one write when both captured payloads exist.
      // The pending flags are cleared here, so a high WVALID cannot commit the
      // same pair twice. The read path snapshots the pre-edge word if an AR
      // handshake shares this edge.
      if (aw_pending && w_pending) begin
        aw_pending <= 1'b0;
        w_pending  <= 1'b0;

        s_axi_control_bvalid <= 1'b1;
        if (w_hit) begin
          pmem[w_index]       <= merge_wstrb(pmem[w_index], w_data, w_strb);
          s_axi_control_bresp <= RESP_OKAY;
        end else begin
          // Misaligned or outside the local window: no storage change,
          // regardless of WSTRB, and DECERR on B.
          s_axi_control_bresp <= RESP_DECERR;
        end
      end

      // Response/ready: hold BVALID/BRESP after commit until BREADY. No new
      // AW or W is accepted during this window because the READY equations
      // include BVALID.
      if (s_axi_control_bvalid && s_axi_control_bready) begin
        s_axi_control_bvalid <= 1'b0;
      end
    end
  end

  // Read-path sequencing: AR snapshot and R response hold. One read may be
  // outstanding, independent of the write path.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      // Reset behavior: clear the single read-response slot.
      s_axi_control_rvalid <= 1'b0;
      s_axi_control_rdata  <= 32'b0;
      s_axi_control_rresp  <= RESP_OKAY;
    end else begin
      // Read snapshot: on AR acceptance, copy the selected word into the R
      // registers. Because the write commit uses nonblocking assignment, an
      // AR accepted on the same edge as a commit returns the old word.
      if (s_axi_control_arvalid && s_axi_control_arready) begin
        s_axi_control_rvalid <= 1'b1;
        if (r_hit) begin
          s_axi_control_rdata <= pmem[r_index];
          s_axi_control_rresp <= RESP_OKAY;
        end else begin
          // Invalid aligned range or misaligned: zero data and DECERR.
          s_axi_control_rdata <= 32'b0;
          s_axi_control_rresp <= RESP_DECERR;
        end
      end

      // Response/ready: hold RDATA/RRESP while RVALID is asserted, and clear
      // RVALID only on the RREADY edge. ARREADY is low until that clears.
      if (s_axi_control_rvalid && s_axi_control_rready) begin
        s_axi_control_rvalid <= 1'b0;
      end
    end
  end

endmodule
