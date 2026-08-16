// =====================================================================
// stage_io_buffer_sel.sv — ONE-PARAM SELECTOR wrapper around the inter-stage
//   activation buffer. SEAM_FF picks the storage primitive WITHOUT touching
//   any of the 11 call sites in eagle4_drafter.sv beyond the module name.
//
//     SEAM_FF=0 (default) : instantiate stage_io_buffer    (BRAM ram_style="block")
//                           -> BYTE-IDENTICAL to the pdi18 baseline.
//     SEAM_FF=1           : instantiate stage_io_buffer_ff (FF/REGFILE, NO BRAM)
//                           -> the STRUCTURAL FIX (FF staging, no SLR-fragile BRAM net).
//
//   The port list + parameter list are IDENTICAL to stage_io_buffer (same names,
//   widths, order, incl. the DBG_TAP wired-localizer + the A4 CO-TRAVEL outputs).
//   RD_LAT / AUTO_SEQ / FRAC_BITS / DBG_TAP wiring pass straight through, so every
//   seam keeps its exact read-latency contract in BOTH arms. The FF arm uses the
//   REGFILE storage impl (addressed FF read mux) which is a true drop-in for any
//   access pattern (random / re-read / out-of-order) — bit-exact to the BRAM ref.
//
//   The banked R4 seam (stage_io_buffer_banked, only active when R4_BANKSPLIT=1)
//   is left on its own selector; in the correctness build R4_BANKSPLIT=0 so the
//   active r4g/r4u instances are plain stage_io_buffer and ARE wrapped here.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

module stage_io_buffer_sel #(
    parameter bit SEAM_FF  = 1'b0,          // 0: BRAM stage_io_buffer ; 1: FF stage_io_buffer_ff
    parameter int H        = 3072,
    parameter int K_PAR    = 16,
    parameter int A_WIDTH  = 16,
    parameter bit AUTO_SEQ = 1'b0,
    parameter int FRAC_BITS = 12,
    parameter int RD_LAT   = 0,
    parameter bit DBG_TAP  = 1'b0,
    // derived (do not override) — mirror stage_io_buffer exactly
    localparam int NCHUNKS = H / K_PAR,
    localparam int CW      = (NCHUNKS > 1) ? $clog2(NCHUNKS) : 1
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // ---- producer-side capture --------------------------------------
    input  wire                              wr_valid,
    input  wire [CW-1:0]                     wr_chunk_idx,
    input  wire signed [K_PAR*A_WIDTH-1:0]   wr_chunk,
    input  wire                              wr_clear,
    output wire [CW-1:0]                     wr_count,
    output wire                              full,

    // ---- consumer-side present (next stage's a_chunk) ---------------
    input  wire                              rd_valid,
    input  wire [CW-1:0]                     rd_chunk_idx,
    output wire signed [K_PAR*A_WIDTH-1:0]   rd_chunk,

    // Stage-B' DUAL_PORT: additive 2nd (ODD-chunk) read port (FF/REGFILE arm
    // only; BRAM arm ties rd_chunk_b=0). Inert when unconnected.
    input  wire [CW-1:0]                     rd_chunk_idx_b,
    output wire signed [K_PAR*A_WIDTH-1:0]   rd_chunk_b,

    // ---- DBG_TAP wired localizer (READ-ONLY) ------------------------
    output wire [15:0]                       dbg_rd_chunk_q0,
    output wire                              dbg_rd_guard_q,

    // ---- CO-TRAVEL outputs (A4 slip-invariant fix) ------------------
    output wire [CW-1:0]                     rd_chunk_idx_out,
    output wire                              rd_valid_out
);

    generate
    if (SEAM_FF) begin : g_ff
        // STRUCTURAL FIX: FF / REGFILE staging — NO ram_style="block".
        stage_io_buffer_ff #(
            .H(H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(AUTO_SEQ),
            .FRAC_BITS(FRAC_BITS), .RD_LAT(RD_LAT), .DBG_TAP(DBG_TAP),
            .IMPL("REGFILE")
        ) u_buf (
            .clk(clk), .rst_n(rst_n),
            .wr_valid(wr_valid), .wr_chunk_idx(wr_chunk_idx), .wr_chunk(wr_chunk),
            .wr_clear(wr_clear), .wr_count(wr_count), .full(full),
            .rd_valid(rd_valid), .rd_chunk_idx(rd_chunk_idx), .rd_chunk(rd_chunk),
            .rd_chunk_idx_b(rd_chunk_idx_b), .rd_chunk_b(rd_chunk_b),
            .dbg_rd_chunk_q0(dbg_rd_chunk_q0), .dbg_rd_guard_q(dbg_rd_guard_q),
            .rd_chunk_idx_out(rd_chunk_idx_out), .rd_valid_out(rd_valid_out)
        );
    end else begin : g_bram
        // BASELINE: BRAM stage_io_buffer — byte-identical to pdi18.
        stage_io_buffer #(
            .H(H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(AUTO_SEQ),
            .FRAC_BITS(FRAC_BITS), .RD_LAT(RD_LAT), .DBG_TAP(DBG_TAP)
        ) u_buf (
            .clk(clk), .rst_n(rst_n),
            .wr_valid(wr_valid), .wr_chunk_idx(wr_chunk_idx), .wr_chunk(wr_chunk),
            .wr_clear(wr_clear), .wr_count(wr_count), .full(full),
            .rd_valid(rd_valid), .rd_chunk_idx(rd_chunk_idx), .rd_chunk(rd_chunk),
            .dbg_rd_chunk_q0(dbg_rd_chunk_q0), .dbg_rd_guard_q(dbg_rd_guard_q),
            .rd_chunk_idx_out(rd_chunk_idx_out), .rd_valid_out(rd_valid_out)
        );
        assign rd_chunk_b = '0;   // BRAM arm: 2nd read port unsupported (prod uses SEAM_FF=1)
    end
    endgenerate

endmodule

`default_nettype wire
