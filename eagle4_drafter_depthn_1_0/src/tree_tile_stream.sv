// =====================================================================
// tree_tile_stream.sv   (M6b)
// Full-vocab G12 -> tree_decode tile-stepping controller.
// =====================================================================
//
// THE M6b GAP this closes
// -----------------------
// M6 wired the tree selector as a SINGLE-TILE serial read of the shared
// GEMV array output:
//     tree_score_sel = array_acc_out[(idx % LANES)*32 +: 32]
//     tree_index     = idx                    (0 .. V-1)
// That is only correct if all V logits live in array_acc_out at once. The
// real G12 LM head is K=256 -> V=128256, and the 128-lane GEMV array emits
// only LANES=128 logits per run (one "tile"). V=128256 = 1002 tiles x 128.
//
// This controller steps the G12 GEMV tile-by-tile and streams each tile's
// 128 logits into tree_decode in lockstep, with the FULL 18-bit vocab index
// vocab_idx = tile*LANES + lane. It owns the tree_decode stream inputs and
// the G12 GEMV (re)start / tile_idx during the tree stage. tree_decode is
// reused UNCHANGED (start / valid_in / score_in / index_in / last_in).
//
// HANDSHAKE
// ---------
//   in : tree_start          1-cycle pulse, begin the full-vocab walk
//        array_acc_out        128 x INT32, the live tile's logits (read-only)
//        array_done           1-cycle pulse, the current G12 tile is valid
//   out: tree_dec_start       1-cycle pulse -> tree_decode.start (heap clear)
//        gemv_tile_start      1-cycle pulse -> G12 GEMV array start
//        gemv_tile_idx        column-block index = current tile t
//        gemv_stage_g12       high while this controller owns the GEMV array
//        valid_in/score_in/index_in/last_in  -> tree_decode stream
//        busy                 high for the whole walk (idle->idle)
//
// FSM
// ---
//   TT_IDLE  : wait tree_start. On pulse: assert tree_dec_start (clear heap),
//              latch tile=0, -> TT_GEMV.
//   TT_GEMV  : pulse gemv_tile_start once (for the current tile), -> TT_WAIT.
//   TT_WAIT  : wait array_done (this tile's 128 logits now in array_acc_out).
//              -> TT_STREAM, lane=0.
//   TT_STREAM: drive valid_in=1, score_in=array_acc_out[lane], index_in=
//              tile*LANES+lane for lane=0..LANES-1 (one logit/cycle).
//              last_in pulses ONLY on the final logit of the final tile
//              (tile==N_TILES-1 && lane==LANES-1). After lane LANES-1:
//                - if more tiles: tile++ , -> TT_GEMV
//                - else          : -> TT_DRAIN
//   TT_DRAIN : one cycle for tree_decode.done, -> TT_IDLE.
//
// MULTI-DRIVER SAFE
// -----------------
// All tree_decode stream inputs + the GEMV start/tile_idx/stage override are
// driven from exactly this ONE module (one always_ff + the comb selector for
// score_in). array_acc_out is read-only. tree_decode is unchanged.
//
// SERIAL COST (K_PAR=1 intake): V cycles to stream + (N_TILES * GEMV_lat)
// overlapped behind the stream. At V=128256, ~128256 cycles ~= 641 us @
// 200 MHz. K_PAR-wide intake (compare K_PAR logits/cycle) is the perf
// follow-on; see M6B_RESULT.md.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_tile_stream #(
    parameter int LANES   = 128,
    parameter int V_MAX   = 128256,   // real Llama-3.2-3B vocabulary
    parameter int IDX_W   = 18,       // EAGLE4 token-id lane width
    parameter int SCORE_W = 32,
    // N_TILES = ceil(V_MAX / LANES). For V_MAX=128256, LANES=128 -> 1002 exact.
    parameter int N_TILES = (V_MAX + LANES - 1) / LANES,
    parameter int LANE_W  = (LANES  <= 1) ? 1 : $clog2(LANES),
    parameter int TILE_W  = (N_TILES <= 1) ? 1 : $clog2(N_TILES)
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // start the full-vocab walk (from top_fsm tree_start, 1-cycle pulse)
    input  wire                          tree_start,

    // V1 FR-Spec runtime scan length (VARIABLE_VOCAB_RTL_PLAN.md §3.7):
    // 0 / >N_TILES = full vocab. Launch-sampled at tree_start.
    input  wire [15:0]                   n_tiles_cfg,

    // live G12 tile logits (read-only) + tile-valid handshake
    input  wire signed [SCORE_W*LANES-1:0] array_acc_out,
    input  wire                          array_done,

    // G12 GEMV stepping (this controller owns these during the walk)
    output logic                         gemv_tile_start,   // 1-cyc pulse / tile
    output logic [15:0]                  gemv_tile_idx,     // column-block index
    output logic                         gemv_stage_g12,    // high during walk

    // tree_decode stream interface (this controller is the SOLE driver)
    output logic                         tree_dec_start,    // heap-clear pulse
    output logic                         valid_in,
    output logic signed [SCORE_W-1:0]    score_in,
    output logic [IDX_W-1:0]             index_in,
    output logic                         last_in,

    output logic                         busy
);

    // ----- vocab-tail geometry -----
    // Llama-3.2-3B: V_MAX=128256 = 1002*128 exactly (no tail). But we DO NOT
    // assume an exact multiple — the terminal compare is against the REAL
    // vocab index V_MAX-1, and the last tile streams only the lanes whose
    // global index < V_MAX (LAST_LANE_OF_LAST_TILE below). This guards a
    // future V_MAX that is not a power-of-two or LANES multiple (spec ask).
    localparam int LAST_TILE_I        = N_TILES - 1;
    localparam int LAST_LANE_LASTTILE = V_MAX - 1 - (LAST_TILE_I * LANES);  // 0..LANES-1
    initial begin
        if (LAST_LANE_LASTTILE < 0 || LAST_LANE_LASTTILE >= LANES)
            $error("tree_tile_stream: bad geometry V_MAX=%0d LANES=%0d N_TILES=%0d", V_MAX, LANES, N_TILES);
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
    logic [LANE_W-1:0]    lane;

    // combinational select of the current lane's INT32 logit from the live
    // tile vector. SOLE reader of array_acc_out in this module.
    wire signed [SCORE_W-1:0] lane_logit = array_acc_out[lane*SCORE_W +: SCORE_W];

    // global 18-bit vocab index for the current (tile, lane)
    // = tile*LANES + lane. Computed wide then truncated to IDX_W (fits: 128255).
    wire [IDX_W-1:0] vocab_idx =
        IDX_W'( (tile * LANES) + lane );

    localparam logic [TILE_W-1:0] LAST_TILE = TILE_W'(N_TILES - 1);
    localparam logic [LANE_W-1:0] LAST_LANE = LANE_W'(LANES   - 1);
    // last lane of the LAST tile that is still a valid vocab index (tail).
    localparam logic [LANE_W-1:0] LAST_LANE_LT = LANE_W'(LAST_LANE_LASTTILE);

    // ---- V1 FR-Spec launch-sampled scan length (frozen per walk) ----
    //   last_tile_q : last tile index of THIS walk; last_lane_q : final valid
    //   lane on that tile (= LAST_LANE_LT only for the full scan; a truncated
    //   scan is tile-granular so its last tile is always full -> LAST_LANE).
    logic [TILE_W-1:0] last_tile_q;
    logic [LANE_W-1:0] last_lane_q;
    wire  [15:0] nt_eff = (n_tiles_cfg == 16'd0 || n_tiles_cfg > 16'(N_TILES))
                          ? 16'(N_TILES) : n_tiles_cfg;

    // is the current (tile,lane) the FINAL valid vocab index (== V_sub-1)?
    wire is_final_idx = (tile == last_tile_q) && (lane == last_lane_q);
    // is the current (tile,lane) the last lane to stream for THIS tile?
    // = physical last lane, OR (on the last tile) the vocab-tail lane.
    wire is_tile_end  = (tile == last_tile_q) ? (lane == last_lane_q)
                                              : (lane == LAST_LANE);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state           <= TT_IDLE;
            tile            <= '0;
            lane            <= '0;
            gemv_tile_start <= 1'b0;
            gemv_tile_idx   <= 16'd0;
            gemv_stage_g12  <= 1'b0;
            tree_dec_start  <= 1'b0;
            valid_in        <= 1'b0;
            score_in        <= '0;
            index_in        <= '0;
            last_in         <= 1'b0;
            busy            <= 1'b0;
            last_tile_q     <= LAST_TILE;
            last_lane_q     <= LAST_LANE_LT;
        end else begin
            // 1-cycle pulse defaults
            gemv_tile_start <= 1'b0;
            tree_dec_start  <= 1'b0;
            valid_in        <= 1'b0;
            last_in         <= 1'b0;

            // V1: launch-sample the scan length (first read >=2 cycles later)
            if (tree_start) begin
                last_tile_q <= TILE_W'(nt_eff - 16'd1);
                last_lane_q <= (nt_eff == 16'(N_TILES)) ? LAST_LANE_LT : LAST_LANE;
            end

            case (state)
                // -------------------------------------------------------
                TT_IDLE: begin
                    gemv_stage_g12 <= 1'b0;
                    busy           <= 1'b0;
                    if (tree_start) begin
                        // clear the heap, prime tile 0
                        tree_dec_start <= 1'b1;
                        tile           <= '0;
                        lane           <= '0;
                        gemv_stage_g12 <= 1'b1;
                        busy           <= 1'b1;
                        state          <= TT_GEMV;
                    end
                end
                // -------------------------------------------------------
                TT_GEMV: begin
                    // launch the G12 GEMV for column block = tile.
                    // Zero-extend `tile` (TILE_W bits) to the 16-bit tile_idx
                    // port (bit-select tile[15:0] would read X above TILE_W).
                    gemv_tile_idx   <= 16'(tile);
                    gemv_tile_start <= 1'b1;
                    state           <= TT_WAIT;
                end
                // -------------------------------------------------------
                TT_WAIT: begin
                    // hold tile_idx stable for the whole GEMV
                    gemv_tile_idx <= 16'(tile);
                    if (array_done) begin
                        lane  <= '0;
                        state <= TT_STREAM;
                    end
                end
                // -------------------------------------------------------
                TT_STREAM: begin
                    // stream this tile's logits, one per cycle, up to the
                    // vocab-tail lane on the final tile.
                    valid_in <= 1'b1;
                    score_in <= lane_logit;
                    index_in <= vocab_idx;
                    // last_in pulses ONLY on the global final vocab index.
                    last_in  <= is_final_idx;

                    if (is_tile_end) begin
                        if (tile == last_tile_q) begin
                            state <= TT_DRAIN;          // reached V_sub-1
                        end else begin
                            tile  <= tile + 1'b1;       // next tile
                            lane  <= '0;
                            state <= TT_GEMV;
                        end
                    end else begin
                        lane <= lane + 1'b1;
                    end
                end
                // -------------------------------------------------------
                TT_DRAIN: begin
                    // tree_decode raised done on the last_in cycle; one more
                    // cycle to settle, then release the GEMV array.
                    gemv_stage_g12 <= 1'b0;
                    busy           <= 1'b0;
                    state          <= TT_IDLE;
                end
                // -------------------------------------------------------
                default: state <= TT_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
