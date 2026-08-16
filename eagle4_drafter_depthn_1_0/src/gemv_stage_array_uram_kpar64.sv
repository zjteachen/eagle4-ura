// =====================================================================
// gemv_stage_array_uram_kpar64.sv   (gemv_kpar64 fork)
// =====================================================================
// URAM-NEUTRAL K_PAR=64 production GEMV stage array.
//
// vs gemv_stage_array_uram (K_PAR=16, 1 bank/lane, gemv_lane_int_stream_wallace):
//   * Per lane the inner lane is gemv_lane_kpar_wallace (the proven
//     K_PAR-parametric lane: N_SG = K_PAR/16 subgroups, each gptq2_dequant16
//     + csa_tree16_int20, registered partial-sum reduce). For K_PAR=64,
//     N_SG=4.
//   * Per lane the weight store is ONE weight_bank_multistage_wide — a
//     SINGLE wide URAM cascade (ROW_W = N_SG*32 = 128 bits) that returns all
//     N_SG words for a chunk in ONE read. This is the URAM-NEUTRAL trick:
//     one dense cascade rounds ONCE (12 URAM/lane => 1,536 @128), unlike the
//     N_SG-independent-bank deinterleave (gemv_stage_array_uram_kpar) which
//     rounds N_SG cascades independently and measured 1,920 URAM/99.7%.
//
// vs gemv_stage_array_uram_kpar (the DISPROVEN K_PAR array):
//   * That array used N_SG independent weight_bank_multistage_kpar banks per
//     lane (sg_we_sel selects which) — NOT URAM-neutral (+25% URAM).
//   * THIS array uses ONE wide bank per lane — URAM-neutral (same total bits,
//     one cascade). The write port takes one PACK-wide row at a time.
//
// BIT-EXACTNESS CONTRACT
//   For a given (stage,column,activation,qweight,qzero) the per-lane acc_out
//   is K_PAR-INVARIANT (only the reduction grouping changes; the INT CSA
//   accumulate is associative-exact). K_PAR=64 MUST equal the K_PAR=16
//   baseline array AND the golden acc_int32, bit-for-bit.
//
// MULTI-DRIVER SAFE: one writer (boot FSM via lane_we_sel) and one reader
//   (its lane) per wide bank. chunk_idx_out from lane 0 only (all lockstep).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module gemv_stage_array_uram_kpar64 #(
    parameter int K        = 64,
    parameter int K_PAR    = 64,
    parameter int LANES    = 4,
    parameter int A_WIDTH  = 16,
    parameter int N_SG      = K_PAR / 16,           // words per row (PACK)
    parameter int ROW_W     = N_SG * 32,            // wide row width
    parameter int N_CHUNKS  = K / K_PAR,
    parameter int CW        = (N_CHUNKS <= 1) ? 1 : $clog2(N_CHUNKS + 2),
    parameter int ROW_DEPTH = 24 * 1024             // per-lane row footprint
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,

    // Activation broadcast (shared across all lanes): K_PAR lanes/chunk
    input  wire signed [K_PAR*A_WIDTH-1:0]   a_chunk,
    input  wire        [1:0]                 qz_raw,

    // Stage selector + tile offset for time-multiplexed N>LANES
    input  wire        [3:0]                 stage_sel,
    input  wire        [15:0]                tile_idx,         // column block index

    // Boot-time weight load. The boot FSM selects the target lane
    // (lane_we_sel) and presents one WIDE row (N_SG words) at wr_row_data,
    // at ROW address wr_row (the boot model assembles PACK consecutive words
    // into wr_row_data — see boot64() in the TB).
    input  wire        [15:0]                lane_we_sel,      // which lane to write
    input  wire                              we,
    input  wire        [16:0]                wr_row,
    input  wire        [ROW_W-1:0]           wr_row_data,

    // Lane status — chunk_idx_out from lane 0 (all lockstep)
    output wire        [CW-1:0]              chunk_idx_out,
    output wire                              chunk_idx_valid,

    // Per-lane accumulator outputs + done aggregation
    output wire signed [32*LANES-1:0]        acc_out_per_lane,
    output wire                              done
);

    wire [LANES-1:0] lane_done;
    wire [CW-1:0]    lane_chunk_idx      [0:LANES-1];
    wire [CW-1:0]    lane_chunk_idx_next [0:LANES-1];
    wire             lane_chunk_valid    [0:LANES-1];

    assign chunk_idx_out   = lane_chunk_idx[0];
    assign chunk_idx_valid = lane_chunk_valid[0];
    assign done = &lane_done;

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : g_lane
            // local column index within this lane's bank (= tile_idx)
            wire [15:0] col_offset = tile_idx[15:0];

            // ---- ONE wide URAM bank per lane (URAM-neutral) ----
            wire [ROW_W-1:0] qw_row;
            wire             lane_we = we && (lane_we_sel == i[15:0]);
            weight_bank_multistage_wide #(
                .K_PAR(K_PAR), .N_SG(N_SG), .ROW_W(ROW_W), .ROW_DEPTH(ROW_DEPTH)
            ) u_bank (
                .clk(clk), .rst_n(rst_n),
                .we(lane_we), .wr_row(wr_row[15:0]), .wr_row_data(wr_row_data),
                .stage_sel(stage_sel),
                .col_offset(col_offset),
                .chunk_idx({{(16-CW){1'b0}}, lane_chunk_idx_next[i]}),
                .rd_row(qw_row)
            );

            // ---- proven K_PAR-wide streaming GEMV lane ----
            // qw_word is the N_SG*32-bit wide row; subgroup s reads bits
            // [s*32 +: 32] — exactly the order weight_bank_multistage_wide
            // packs them (word w of chunk r at row r, slot (w % N_SG)).
            wire signed [31:0] lane_acc;
            gemv_lane_kpar_wallace #(
                .K(K), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH)
            ) u_lane (
                .clk(clk), .rst_n(rst_n), .start(start),
                .a_chunk(a_chunk),
                .qw_word(qw_row),
                .qz_raw(qz_raw),
                .chunk_idx_out(lane_chunk_idx[i]),
                .chunk_idx_valid(lane_chunk_valid[i]),
                .chunk_idx_next_out(lane_chunk_idx_next[i]),
                .acc_out(lane_acc),
                .done(lane_done[i])
            );

            assign acc_out_per_lane[i*32 +: 32] = lane_acc;
        end
    endgenerate

endmodule

`default_nettype wire
