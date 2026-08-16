// =====================================================================
// stage_io_buffer_banked.sv — BANK-SPLIT variant of stage_io_buffer for the
// WIDE R4 seam (K_PAR*A_WIDTH = 16*18 = 288-bit). The baseline stage_io_buffer
// stores the whole 288-bit chunk as ONE wide word -> synth packs it into 4
// CASCADED RAMB36E5 (READ_WIDTH_A=72 each). The funcsim/gate-level model flags
// these as cascade-untied (RAMB36E5_INT-126/127/130: "sim behavior may not
// match hardware") and the wide read net is the lane-aliasing class seen on the
// M7 +128 load path.
//
// FIX (mirrors the M7 256b->8x32b bank-split): store the chunk as NBANKS
// INDEPENDENT narrow banks, each a self-contained single-RAMB sync-read buffer
// addressed by the SAME rd_chunk_idx, with the read register intrinsic to each
// bank's RAMB (no separable fabric FF, no 288-bit cascade). Lane l lives in bank
// (l / LANES_PER_BANK) at lane-offset (l % LANES_PER_BANK) -> a fixed, bijective
// lane->bank map (no wide-net shuffle). Bit-identical to stage_io_buffer at the
// chunk level (the read just reassembles the banks in lane order).
//
// Default BANKS so each bank carries 36 bits (= 2 lanes of 18b) -> 8 banks of
// 36b for K_PAR=16,A_WIDTH=18. Each 36b x NCHUNKS(512) bank fits a single
// RAMB18/RAMB36 with NO cascade. Per-bank registered guard mirrors RD_LAT=1.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

module stage_io_buffer_banked #(
    parameter int H              = 8192,
    parameter int K_PAR          = 16,
    parameter int A_WIDTH        = 18,
    parameter int LANES_PER_BANK = 2,            // 2 lanes/bank -> 36b/bank, 8 banks
    parameter int RD_LAT         = 1,            // 1 = registered sync read (silicon BRAM)
    localparam int NCHUNKS       = H / K_PAR,
    localparam int CW            = (NCHUNKS > 1) ? $clog2(NCHUNKS) : 1,
    localparam int NBANKS        = K_PAR / LANES_PER_BANK,
    localparam int BANK_W        = LANES_PER_BANK * A_WIDTH    // 36
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              wr_valid,
    input  wire [CW-1:0]                     wr_chunk_idx,
    input  wire signed [K_PAR*A_WIDTH-1:0]   wr_chunk,
    input  wire                              wr_clear,
    output reg  [CW-1:0]                     wr_count,
    output reg                               full,
    input  wire                              rd_valid,
    input  wire [CW-1:0]                     rd_chunk_idx,
    output wire signed [K_PAR*A_WIDTH-1:0]   rd_chunk
);
    // write bookkeeping (identical semantics to stage_io_buffer)
    always_ff @(posedge clk) begin
        if (!rst_n)        begin wr_count <= '0; full <= 1'b0; end
        else if (wr_clear) begin wr_count <= '0; full <= 1'b0; end
        else if (wr_valid) begin
            if ({1'b0, wr_count} != (CW+1)'(NCHUNKS))   wr_count <= wr_count + 1'b1;
            if ({1'b0, wr_count} == (CW+1)'(NCHUNKS-1)) full <= 1'b1;
        end
    end

    wire rd_guard = rd_valid && ({1'b0, rd_chunk_idx} < (CW+1)'(NCHUNKS));
    reg  rd_guard_q;
    always_ff @(posedge clk) begin
        if (!rst_n) rd_guard_q <= 1'b0;
        else        rd_guard_q <= rd_guard;
    end

    genvar b;
    generate
        for (b = 0; b < NBANKS; b = b + 1) begin : g_bank
            // each bank holds LANES_PER_BANK contiguous lanes -> BANK_W bits
            (* ram_style = "block" *)
            reg [BANK_W-1:0] bank_mem [0:NCHUNKS-1];
            wire [BANK_W-1:0] wslice = wr_chunk[b*BANK_W +: BANK_W];

            always_ff @(posedge clk) begin
                if (wr_valid) bank_mem[wr_chunk_idx] <= wslice;
            end

            if (RD_LAT == 0) begin : g_comb
                wire [BANK_W-1:0] rslice = rd_guard ? bank_mem[rd_chunk_idx] : '0;
                assign rd_chunk[b*BANK_W +: BANK_W] = rslice;
            end else begin : g_reg
                // canonical sync-read BRAM idiom: read reg is the BRAM DOUT reg,
                // single narrow (36b) bank -> a SINGLE non-cascaded RAMB.
                reg [BANK_W-1:0] rbank_q;
                always_ff @(posedge clk) begin
                    rbank_q <= bank_mem[rd_chunk_idx];
                end
                assign rd_chunk[b*BANK_W +: BANK_W] = rd_guard_q ? rbank_q : '0;
            end
        end
    endgenerate

endmodule

`default_nettype wire
