// =====================================================================
// tree_tile_stream_pkpar.sv   (PERF-INTEG fork)
// =====================================================================
// P-wide full-vocab G12 -> tree intake controller. The perf-path analogue
// of tree_tile_stream (which streams 1 logit/cycle into the serial
// tree_decode). This version streams P logits/cycle into tree_merge_pipe
// (the proven pipelined partial-heap merge tree), cutting the vocab scan
// from V cycles to ceil(V/P) cycles.
//
// CONTRACT (must match the serial tree_tile_stream + tree_decode top-K_F=8):
//   * Indices are streamed MONOTONICALLY INCREASING. Each cycle emits a
//     P-group whose lane g carries global vocab index = base+g, lane 0 the
//     LOWEST index (earlier-index-wins tie rule preserved by tree_merge_pipe).
//   * On the last (partial) group, lanes beyond the final valid vocab index
//     are disabled via lane_en=0 (so the tail is exact for V not a multiple
//     of LANES or P).
//
// GEMV TILE MODEL: the upstream GEMV array still emits LANES logits per tile
// (array_acc_out, array_done). This controller buffers one tile's LANES
// logits, then emits them P at a time into tree_merge_pipe. LANES is assumed
// a multiple of P (128 % 4 == 0, 128 % 8 == 0) so a tile is an integer number
// of P-groups; the final tile's vocab tail is handled by lane_en.
//
// MULTI-DRIVER SAFE: all tree_merge_pipe stream inputs + the GEMV
// start/tile_idx/stage override are driven from this ONE module. The tile
// buffer is written only on array_done; array_acc_out is read-only.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_tile_stream_pkpar #(
    parameter int LANES   = 128,
    parameter int P       = 4,
    parameter int V_MAX   = 128256,
    parameter int IDX_W   = 18,
    parameter int SCORE_W = 32,
    parameter int N_TILES = (V_MAX + LANES - 1) / LANES,
    parameter int LANE_W  = (LANES  <= 1) ? 1 : $clog2(LANES),
    parameter int TILE_W  = (N_TILES <= 1) ? 1 : $clog2(N_TILES),
    parameter int GRP_W   = (LANES/P <= 1) ? 1 : $clog2(LANES/P)
) (
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire                          tree_start,

    // V1 FR-Spec runtime scan length (VARIABLE_VOCAB_RTL_PLAN.md §3.7):
    // 0 / >N_TILES = full vocab. Launch-sampled at tree_start.
    input  wire [15:0]                   n_tiles_cfg,

    // live G12 tile logits (read-only) + tile-valid handshake
    input  wire signed [SCORE_W*LANES-1:0] array_acc_out,
    input  wire                          array_done,

    // G12 GEMV stepping (this controller owns these during the walk)
    output logic                         gemv_tile_start,
    output logic [15:0]                  gemv_tile_idx,
    output logic                         gemv_stage_g12,

    // tree_merge_pipe stream interface (this controller is the SOLE driver)
    output logic                         tree_dec_start,
    output logic                         valid_in,
    output logic signed [P*SCORE_W-1:0]  scores_in,
    output logic [P*IDX_W-1:0]           indices_in,
    output logic [P-1:0]                 lane_en,
    output logic                         last_in,

    output logic                         busy
);

    localparam int GRPS_PER_TILE = LANES / P;
    localparam int LAST_TILE_I        = N_TILES - 1;
    localparam int LAST_VALID_GLOBAL  = V_MAX - 1;
    localparam int LAST_LANE_LASTTILE = V_MAX - 1 - (LAST_TILE_I * LANES);
    initial begin
        if ((LANES % P) != 0)
            $error("tree_tile_stream_pkpar: LANES(%0d) must be a multiple of P(%0d)", LANES, P);
    end

    typedef enum logic [2:0] {
        TT_IDLE   = 3'd0,
        TT_GEMV   = 3'd1,
        TT_WAIT   = 3'd2,
        TT_STREAM = 3'd3,
        TT_DRAIN  = 3'd4
    } tt_state_e;

    tt_state_e            state;
    logic [TILE_W-1:0]    tile;
    logic [GRP_W-1:0]     grp;     // which P-group within the buffered tile

    // one-tile buffer (written on array_done)
    logic signed [SCORE_W-1:0] tile_buf [0:LANES-1];

    localparam logic [TILE_W-1:0] LAST_TILE = TILE_W'(N_TILES - 1);

    // ---- V1 FR-Spec launch-sampled scan length (frozen per walk) ----
    logic [15:0] ntq;
    logic [IDX_W-1:0] lastv_q;
    wire  [15:0] nt_eff = (n_tiles_cfg == 16'd0 || n_tiles_cfg > 16'(N_TILES))
                          ? 16'(N_TILES) : n_tiles_cfg;
    wire  [IDX_W-1:0] lastv_eff = (nt_eff == 16'(N_TILES))
                          ? IDX_W'(V_MAX - 1)
                          : IDX_W'((32'(nt_eff) * LANES) - 1);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state           <= TT_IDLE;
            tile            <= '0;
            grp             <= '0;
            gemv_tile_start <= 1'b0;
            gemv_tile_idx   <= 16'd0;
            gemv_stage_g12  <= 1'b0;
            tree_dec_start  <= 1'b0;
            valid_in        <= 1'b0;
            scores_in       <= '0;
            indices_in      <= '0;
            lane_en         <= '0;
            last_in         <= 1'b0;
            busy            <= 1'b0;
            ntq             <= 16'(N_TILES);
            lastv_q         <= IDX_W'(V_MAX - 1);
        end else begin
            // 1-cycle pulse defaults
            gemv_tile_start <= 1'b0;
            tree_dec_start  <= 1'b0;
            valid_in        <= 1'b0;
            last_in         <= 1'b0;
            lane_en         <= '0;

            // V1: launch-sample the scan length (first read >=2 cycles later)
            if (tree_start) begin
                ntq     <= nt_eff;
                lastv_q <= lastv_eff;
            end

            case (state)
                TT_IDLE: begin
                    gemv_stage_g12 <= 1'b0;
                    busy           <= 1'b0;
                    if (tree_start) begin
                        tree_dec_start <= 1'b1;
                        tile           <= '0;
                        grp            <= '0;
                        gemv_stage_g12 <= 1'b1;
                        busy           <= 1'b1;
                        state          <= TT_GEMV;
                    end
                end
                TT_GEMV: begin
                    gemv_tile_idx   <= 16'(tile);
                    gemv_tile_start <= 1'b1;
                    state           <= TT_WAIT;
                end
                TT_WAIT: begin
                    gemv_tile_idx <= 16'(tile);
                    if (array_done) begin
                        for (int l = 0; l < LANES; l++)
                            tile_buf[l] <= array_acc_out[l*SCORE_W +: SCORE_W];
                        grp   <= '0;
                        state <= TT_STREAM;
                    end
                end
                TT_STREAM: begin
                    // emit P-group `grp` of the buffered tile.
                    valid_in <= 1'b1;
                    for (int g = 0; g < P; g++) begin
                        int unsigned gl;       // lane within tile
                        int unsigned gv;        // global vocab index
                        gl = grp * P + g;
                        gv = tile * LANES + gl;
                        scores_in [g*SCORE_W +: SCORE_W] <= tile_buf[gl];
                        indices_in[g*IDX_W   +: IDX_W]   <= IDX_W'(gv);
                        lane_en[g] <= (gv <= 32'(lastv_q));  // V1 runtime tail
                    end

                    // is this the last group of the last tile (contains V_sub-1)?
                    if ((tile == TILE_W'(ntq - 16'd1)) &&
                        ((tile * LANES + grp * P + (P-1)) >= 32'(lastv_q))) begin
                        last_in <= 1'b1;
                        state   <= TT_DRAIN;
                    end else if (grp == GRP_W'(GRPS_PER_TILE - 1)) begin
                        // finished all groups of this tile -> next tile
                        tile  <= tile + 1'b1;
                        grp   <= '0;
                        state <= TT_GEMV;
                    end else begin
                        grp <= grp + 1'b1;
                    end
                end
                TT_DRAIN: begin
                    // tree_merge_pipe has a +2 pipeline fill; wait for done is
                    // the TB/top's job. Release the GEMV array.
                    gemv_stage_g12 <= 1'b0;
                    busy           <= 1'b0;
                    state          <= TT_IDLE;
                end
                default: state <= TT_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
