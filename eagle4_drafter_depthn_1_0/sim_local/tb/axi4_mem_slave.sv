// =====================================================================
// axi4_mem_slave.sv — TESTBENCH ONLY. A generic AXI4 (full, "aximm")
// SLAVE: it stands in for the piece of external memory (HBM, on the real
// board) that sits on the other end of one of the DUT's AXI4 master ports.
//
// This is a "Bus Functional Model" (BFM): it doesn't do anything useful on
// its own, it just correctly speaks the AXI4 protocol (the valid/ready
// handshake on each of the 5 channels — AW/W/B for writes, AR/R for reads)
// so the real RTL master can talk to *something* during simulation.
//
// Scope, matching what this design's masters actually do (see
// output_dma_writer.sv header comment: "Fixed INCR burst ... single
// outstanding burst"):
//   - INCR bursts only (address advances by one word per beat — the
//     universal case; WRAP/FIXED bursts are not implemented).
//   - Always-ready on every channel (no backpressure testing — not the
//     point of a smoke test).
//   - One transaction in flight per channel at a time.
//
// Backing store is a plain word array sized by DEPTH_WORDS, addressed as
// (axi_byte_addr - BASE_ADDR) / (DATA_W/8). Reads outside that window
// return zero instead of erroring, since a smoke test cares about "did the
// DUT drive a sane transaction," not memory-protection semantics.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

module axi4_mem_slave #(
    parameter int ID_W         = 1,
    parameter int ADDR_W       = 64,
    parameter int DATA_W       = 32,
    parameter longint BASE_ADDR   = 0,
    parameter int DEPTH_WORDS  = 1024,   // backing store size, in DATA_W-wide words
    parameter int RD_LATENCY   = 2       // cycles from ARVALID&ARREADY to first RVALID (>=1)
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // ---- write address/data/response ----
    input  wire [ID_W-1:0]       awid,
    input  wire [ADDR_W-1:0]     awaddr,
    input  wire [7:0]            awlen,
    input  wire [2:0]            awsize,
    input  wire [1:0]            awburst,
    input  wire                  awvalid,
    output reg                   awready,

    input  wire [DATA_W-1:0]     wdata,
    input  wire [DATA_W/8-1:0]   wstrb,
    input  wire                  wlast,
    input  wire                  wvalid,
    output reg                   wready,

    output reg  [ID_W-1:0]       bid,
    output reg  [1:0]            bresp,
    output reg                   bvalid,
    input  wire                  bready,

    // ---- read address/data ----
    input  wire [ID_W-1:0]       arid,
    input  wire [ADDR_W-1:0]     araddr,
    input  wire [7:0]            arlen,
    input  wire [2:0]            arsize,
    input  wire [1:0]            arburst,
    input  wire                  arvalid,
    output reg                   arready,

    output reg  [ID_W-1:0]       rid,
    output reg  [DATA_W-1:0]     rdata,
    output reg  [1:0]            rresp,
    output reg                   rlast,
    output reg                   rvalid,
    input  wire                  rready
);
    localparam int BYTES_PER_WORD = DATA_W/8;

    reg [DATA_W-1:0] mem [0:DEPTH_WORDS-1];

    function automatic int word_idx(input longint byte_addr);
        longint off;
        off = byte_addr - BASE_ADDR;
        if (off < 0 || (off / BYTES_PER_WORD) >= DEPTH_WORDS) return -1;
        return int'(off / BYTES_PER_WORD);
    endfunction

    // ---------------- WRITE SIDE (AW/W/B) ----------------
    // Simplification: accept AWVALID and WVALID independently (not
    // required to arrive the same cycle -- matches how output_dma_writer's
    // "single outstanding burst" driver behaves) and just count WLAST to
    // know when to fire BVALID.
    reg [ADDR_W-1:0] w_addr_cur;
    reg [ID_W-1:0]   w_id_cur;
    reg              w_busy;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            awready <= 1'b0;
            wready  <= 1'b0;
            bvalid  <= 1'b0;
            bid     <= '0;
            bresp   <= 2'b00;
            w_busy  <= 1'b0;
            w_addr_cur <= '0;
            w_id_cur   <= '0;
        end else begin
            // accept a new write address whenever idle
            awready <= !w_busy;
            if (awvalid && awready) begin
                w_addr_cur <= awaddr;
                w_id_cur   <= awid;
                w_busy     <= 1'b1;
            end

            wready <= w_busy;
            if (w_busy && wvalid && wready) begin
                automatic int idx = word_idx(w_addr_cur);
                if (idx >= 0) begin
                    for (int b = 0; b < BYTES_PER_WORD; b++)
                        if (wstrb[b]) mem[idx][b*8 +: 8] <= wdata[b*8 +: 8];
                end
                w_addr_cur <= w_addr_cur + BYTES_PER_WORD;
                if (wlast) begin
                    w_busy <= 1'b0;
                    bvalid <= 1'b1;
                    bid    <= w_id_cur;
                    bresp  <= 2'b00; // OKAY
                end
            end

            if (bvalid && bready) bvalid <= 1'b0;
        end
    end

    // ---------------- READ SIDE (AR/R) ----------------
    // Not exercised by Tier A (m_axi_out is write-only; m_axi_boot stays
    // silent since we use the TB direct-drive pins instead of the boot-DMA
    // path; m_axi_g8/g8b/m_axi_emb/embb are unused with
    // G8_SPLICE=0/DEPTH_N=0/EMBG_EN=0). Implemented now so Tier B (the real
    // default config, which does exercise these) is "wire up 4 more
    // instances of this same module," not "write a new BFM."
    reg [ADDR_W-1:0] r_addr_cur;
    reg [ID_W-1:0]   r_id_cur;
    reg [8:0]        r_beats_left; // arlen+1, up to 256
    reg [8:0]        r_delay_left;
    reg              r_busy;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            arready      <= 1'b0;
            rvalid       <= 1'b0;
            rlast        <= 1'b0;
            rid          <= '0;
            rdata        <= '0;
            rresp        <= 2'b00;
            r_busy       <= 1'b0;
            r_addr_cur   <= '0;
            r_id_cur     <= '0;
            r_beats_left <= '0;
            r_delay_left <= '0;
        end else begin
            arready <= !r_busy;
            if (arvalid && arready) begin
                r_addr_cur   <= araddr;
                r_id_cur     <= arid;
                r_beats_left <= {1'b0, arlen} + 9'd1;
                r_delay_left <= RD_LATENCY[8:0];
                r_busy       <= 1'b1;
                rvalid       <= 1'b0;
            end

            if (r_busy && !rvalid) begin
                if (r_delay_left > 0) begin
                    r_delay_left <= r_delay_left - 9'd1;
                end else begin
                    automatic int idx = word_idx(r_addr_cur);
                    rdata  <= (idx >= 0) ? mem[idx] : '0;
                    rid    <= r_id_cur;
                    rresp  <= 2'b00;
                    rlast  <= (r_beats_left == 9'd1);
                    rvalid <= 1'b1;
                end
            end else if (rvalid && rready) begin
                rvalid <= 1'b0;
                if (r_beats_left > 1) begin
                    r_addr_cur   <= r_addr_cur + BYTES_PER_WORD;
                    r_beats_left <= r_beats_left - 9'd1;
                    r_delay_left <= 9'd0; // back-to-back beats after the first
                end else begin
                    r_busy <= 1'b0;
                end
            end
        end
    end
endmodule
`default_nettype wire
