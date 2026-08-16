// =====================================================================
// weight_bank_multistage_wide.sv   (gemv_kpar64 fork)
// =====================================================================
// URAM-NEUTRAL wide-word weight bank for K_PAR>16.
//
// THE URAM-NEUTRALITY MECHANISM (the "16-lane-bank scheduling trick"):
// -------------------------------------------------------------------
//   The K_PAR=16 baseline (weight_bank_multistage) reads ONE 32-bit
//   qweight word per cycle from ONE dense URAM cascade. Its per-lane
//   footprint (97,440 words × 32 bits = 3.118 Mbit) packs to 12 URAM/lane
//   (1,536 @128 lanes) because Vivado 2-col-packs 32b words into 72b
//   URAM288 rows and rounds the cascade depth ONCE.
//
//   K_PAR=64 needs N_SG=4 words per cycle. The DISPROVEN approach
//   (gemv_stage_array_uram_kpar / weight_bank_multistage_kpar) splits the
//   footprint into N_SG INDEPENDENT banks — Vivado then rounds N_SG
//   cascades INDEPENDENTLY, costing +25% URAM (1,920/99.7% @ K_PAR=32,
//   measured in explore/perf_integ128). That breaks URAM-neutrality.
//
//   THIS bank keeps ONE dense cascade per lane (one rounding event) and
//   simply makes the ROW WIDE: PACK = N_SG words = N_SG*32 bits per row.
//   It stores the SAME total bits (3.118 Mbit/lane), addressed by ROW
//   (chunk index 0..K/K_PAR-1) instead of by word. One read returns all
//   N_SG words for the chunk. Per uram_repack/URAM_REPACK_RESULT.md the
//   PACK=2 wide bank (weight_bank_uram_wide2) measured the IDENTICAL URAM
//   count as the K_PAR=16 baseline (8==8, 1.00×) — because the weight bank
//   is bit-count-bound and a single wide cascade rounds once, not N_SG
//   times. PACK=4 (this bank, K_PAR=64) extends the same property: same
//   bits, one cascade => 12 URAM/lane => 1,536 @128 => URAM-NEUTRAL.
//
// LAYOUT (per lane, per stage, per column):
//   The K_PAR=16 baseline stores `cpc16 = K/16` words per column at flat
//   address (stage_base16 + col*cpc16 + word). Here we group those words
//   into rows of PACK = K_PAR/16: row r holds words [r*PACK .. r*PACK+PACK-1]
//   = the PACK 16-lane subgroups consumed in chunk r of the K_PAR lane.
//   Rows per column = cpc16 / PACK = K/K_PAR = N_CHUNKS. So the per-lane
//   ROW footprint = (97,440 words)/PACK rows × (PACK*32) bits/row =
//   97,440*32 bits = SAME 3.118 Mbit/lane. Bit-identical storage.
//
// READ ALIGNMENT: identical to the K_PAR=16 bank — the lane drives
//   chunk_idx_NEXT, URAM data lands at T+1 aligned to the lane's
//   combinational consumption at T+1 (1-cycle registered read).
//
// MULTI-DRIVER SAFE: exactly one writer (boot FSM via lane_we_sel) and one
//   reader (this lane) per bank. The write port takes one PACK-wide row at a
//   time (the boot model assembles PACK words into wr_row_data).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module weight_bank_multistage_wide #(
    parameter int K_PAR  = 64,
    parameter int N_SG   = K_PAR / 16,          // words per row (PACK)
    parameter int ROW_W  = N_SG * 32,           // wide row width in bits
    // ROW depth: per-lane footprint (97,440 words) / N_SG rows, rounded up
    // to a clean URAM cascade. 97,440/4 = 24,360; size to 24*1024 = 24,576.
    parameter int ROW_DEPTH = 24 * 1024,
    parameter int RADDR_W = (ROW_DEPTH <= 1) ? 1 : $clog2(ROW_DEPTH)
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Boot-time write port — one WIDE row (N_SG words) per write.
    input  wire                  we,
    input  wire [RADDR_W-1:0]    wr_row,         // ROW address into THIS bank
    input  wire [ROW_W-1:0]      wr_row_data,    // PACK words assembled

    // Compute-time read port
    input  wire [3:0]            stage_sel,      // 0..5 GEMV stage
    input  wire [15:0]           col_offset,     // output col within stage (local)
    input  wire [15:0]           chunk_idx,      // K-chunk within column (0..K/K_PAR-1)
    output wire [ROW_W-1:0]      rd_row          // N_SG words for this chunk
);

    // -----------------------------------------------------------------
    // Per-stage ROW base offsets and per-column ROW stride.
    // The K_PAR=16 baseline (weight_bank_multistage.sv) uses these WORD
    // bases / strides; here every value is divided by N_SG because rows
    // hold N_SG words. The TOTAL stored bits are identical (rows × ROW_W =
    // words × 32). Bases match the M1 WEIGHT-LAYOUT-FIT table exactly.
    //
    //   K_PAR=16 word table         -> wide-row table (÷ N_SG)
    //   G4_qkv base 0     cpc 384    -> base 0/N         cpc 384/N
    //   G6_o   base 27648 cpc 192    -> base 27648/N     cpc 192/N
    //   G7_g   base 32256 cpc 192    -> ...
    //   G7_u   base 44544 cpc 192
    //   G9     base 56832 cpc 512
    //   G12    base 81408 cpc 16
    // -----------------------------------------------------------------
    function automatic [16:0] stage_base_row(input [3:0] s);
        case (s)
            4'd0: stage_base_row = 17'(17'd0     / N_SG); // G4_qkv
            4'd1: stage_base_row = 17'(17'd27648 / N_SG); // G6_o
            4'd2: stage_base_row = 17'(17'd32256 / N_SG); // G7_g
            4'd3: stage_base_row = 17'(17'd44544 / N_SG); // G7_u
            4'd4: stage_base_row = 17'(17'd56832 / N_SG); // G9_down
            4'd5: stage_base_row = 17'(17'd81408 / N_SG); // G12_lm
            default: stage_base_row = 17'd0;
        endcase
    endfunction

    function automatic [16:0] stage_rows_per_col(input [3:0] s);
        case (s)
            4'd0: stage_rows_per_col = 17'(17'd384 / N_SG); // G4_qkv K=6144
            4'd1: stage_rows_per_col = 17'(17'd192 / N_SG); // G6   K=3072
            4'd2: stage_rows_per_col = 17'(17'd192 / N_SG); // G7_g
            4'd3: stage_rows_per_col = 17'(17'd192 / N_SG); // G7_u
            4'd4: stage_rows_per_col = 17'(17'd512 / N_SG); // G9   K=8192
            4'd5: stage_rows_per_col = 17'(17'd16  / N_SG); // G12  K=256
            default: stage_rows_per_col = 17'd0;
        endcase
    endfunction

    // Registered base+col*stride (same timing idiom as the K_PAR=16 bank:
    // the col multiply only re-evaluates when stage/col change; per-cycle
    // path is just (registered_offset + chunk_idx) — a 17-bit add).
    reg  [16:0] col_offset_acc;
    reg  [3:0]  stage_sel_d;
    reg  [15:0] col_offset_d;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            col_offset_acc <= '0;
            stage_sel_d    <= '0;
            col_offset_d   <= '0;
        end else begin
            if (stage_sel !== stage_sel_d || col_offset !== col_offset_d) begin
                col_offset_acc <= stage_base_row(stage_sel) +
                                  (col_offset * stage_rows_per_col(stage_sel));
                stage_sel_d    <= stage_sel;
                col_offset_d   <= col_offset;
            end
        end
    end

    wire [16:0] rd_addr_full = col_offset_acc + chunk_idx;
    wire [RADDR_W-1:0] rd_addr = rd_addr_full[RADDR_W-1:0];

    // -----------------------------------------------------------------
    // ONE dense wide URAM cascade (one rounding event = URAM-neutral).
    // ROW_W = N_SG*32. weight_bank_uram is the same backing primitive used
    // by the K_PAR=16 path (proven URAM inference at production depth).
    // -----------------------------------------------------------------
    weight_bank_uram #(
        .DEPTH(ROW_DEPTH), .DATA_W(ROW_W)
    ) u_uram (
        .clk(clk), .rst_n(rst_n),
        .we(we), .wr_addr(wr_row), .wr_data(wr_row_data),
        .rd_addr(rd_addr), .rd_data(rd_row)
    );

endmodule

`default_nettype wire
