// =====================================================================
// tree_tile_stream_pp.sv   (EXP_G12TREE_PINGPONG)
// Serial (1 logit/cycle) full-vocab G12 -> tree_decode controller with a
// 2-entry FF PING-PONG tile buffer that DECOUPLES the G12 GEMV producer
// from the tree_decode stream consumer.
//
// WHAT THE BASELINE tree_tile_stream DOES (the coupling bubble):
//   per tile: TT_GEMV(1) -> TT_WAIT(GEMV latency W) -> TT_STREAM(128).
//   The GEMV array is IDLE for all 128 stream cycles of every tile, and
//   tree_decode is IDLE for all W GEMV cycles: per-tile cost = 1 + W + 128.
//
// WHAT THIS MODULE DOES:
//   PRODUCER FSM: runs the G12 GEMV tile-by-tile back-to-back; on each
//     array_done it captures the 128 logits into ping-pong slot (tile&1)
//     and immediately issues the NEXT tile's GEMV (if the target slot is
//     free). tile_idx+start are registered in the same cycle, preserving
//     the baseline's tile_idx-leads-start-by-1 bank-settle timing (the
//     top's tt_gemv_start_q adds the same +1 as for the baseline).
//   CONSUMER FSM: streams slot (tile&1) into tree_decode 1 logit/cycle,
//     exactly the baseline order (monotone vocab_idx = tile*LANES+lane,
//     last_in on global V_MAX-1), then frees the slot.
//   Steady-state per-tile cost = max(W, 128) + eps  (GEMV runs UNDER the
//   stream). Stream CONTENT and ORDER are bit-identical to the baseline
//   (capture-at-array_done == live-read: the array holds acc stable after
//   done; same equivalence the proven pkpar tile_buf capture relies on).
//
// FF-STAGING LAW (2026-06-26 user mandate): the ping-pong buffer is a
// RE-READ register file -> (* ram_style="registers" *) FF, never BRAM/SRL.
// 2 x 128 x 32b = 8192 FF. Netlist gate must show FDRE only on this seam.
//
// MULTI-DRIVER SAFE: both FSMs live in ONE always_ff; slot_full[pslot] set
// (producer) and slot_full[cslot] cleared (consumer) are provably different
// bits in any cycle where both fire (producer may only capture into a slot
// the consumer is not streaming; assertion below).
//
// Port list IDENTICAL to tree_tile_stream (drop-in for the g_tree_serial
// generate arm; the G8_SPLICE cheap-logit snoop sees the same stream).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_tile_stream_pp #(
    parameter int LANES   = 128,
    parameter int V_MAX   = 128256,
    parameter int IDX_W   = 18,
    parameter int SCORE_W = 32,
    parameter int N_TILES = (V_MAX + LANES - 1) / LANES,
    parameter int LANE_W  = (LANES  <= 1) ? 1 : $clog2(LANES),
    parameter int TILE_W  = (N_TILES <= 1) ? 1 : $clog2(N_TILES)
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

    // tree_decode stream interface (this controller is the SOLE driver)
    output logic                         tree_dec_start,
    output logic                         valid_in,
    output logic signed [SCORE_W-1:0]    score_in,
    output logic [IDX_W-1:0]             index_in,
    output logic                         last_in,

    output logic                         busy
);

    // ----- vocab-tail geometry (identical to baseline tree_tile_stream) -----
    localparam int LAST_TILE_I        = N_TILES - 1;
    localparam int LAST_LANE_LASTTILE = V_MAX - 1 - (LAST_TILE_I * LANES);
    initial begin
        if (LAST_LANE_LASTTILE < 0 || LAST_LANE_LASTTILE >= LANES)
            $error("tree_tile_stream_pp: bad geometry V_MAX=%0d LANES=%0d N_TILES=%0d", V_MAX, LANES, N_TILES);
    end
    localparam logic [TILE_W-1:0] LAST_TILE    = TILE_W'(N_TILES - 1);
    localparam logic [LANE_W-1:0] LAST_LANE    = LANE_W'(LANES   - 1);
    localparam logic [LANE_W-1:0] LAST_LANE_LT = LANE_W'(LAST_LANE_LASTTILE);

    // ----- 2-entry FF ping-pong tile buffer (FF-staging law: registers) -----
    (* ram_style = "registers", shreg_extract = "no" *)
    logic signed [SCORE_W-1:0] tile_buf [0:1][0:LANES-1];
    logic [1:0] slot_full;

    // ----- producer (GEMV stepper) -----
    typedef enum logic [1:0] { P_IDLE = 2'd0, P_ISSUE = 2'd1, P_WAIT = 2'd2 } p_state_e;
    p_state_e          pstate;
    logic [TILE_W:0]   ptile;                 // 0..N_TILES (one extra bit)
    wire               pslot = ptile[0];

    // ----- consumer (tree_decode streamer) -----
    typedef enum logic [1:0] { C_IDLE = 2'd0, C_WAITFULL = 2'd1, C_STREAM = 2'd2, C_DRAIN = 2'd3 } c_state_e;
    c_state_e          cstate;
    logic [TILE_W-1:0] ctile;
    logic [LANE_W-1:0] lane;
    wire               cslot = ctile[0];

    // ---- V1 FR-Spec launch-sampled scan length (frozen per walk) ----
    logic [TILE_W-1:0] last_tile_q;
    logic [LANE_W-1:0] last_lane_q;
    logic [TILE_W:0]   last_ptile_q;    // producer-stop compare (TILE_W+1 wide)
    wire  [15:0] nt_eff = (n_tiles_cfg == 16'd0 || n_tiles_cfg > 16'(N_TILES))
                          ? 16'(N_TILES) : n_tiles_cfg;

    wire is_final_idx = (ctile == last_tile_q) && (lane == last_lane_q);
    wire is_tile_end  = (ctile == last_tile_q) ? (lane == last_lane_q)
                                               : (lane == LAST_LANE);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pstate          <= P_IDLE;
            cstate          <= C_IDLE;
            ptile           <= '0;
            ctile           <= '0;
            lane            <= '0;
            slot_full       <= 2'b00;
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
            last_ptile_q    <= (TILE_W+1)'(N_TILES - 1);
        end else begin
            // 1-cycle pulse defaults
            gemv_tile_start <= 1'b0;
            tree_dec_start  <= 1'b0;
            valid_in        <= 1'b0;
            last_in         <= 1'b0;

            // V1: launch-sample the scan length (first read >=2 cycles later)
            if (tree_start) begin
                last_tile_q  <= TILE_W'(nt_eff - 16'd1);
                last_lane_q  <= (nt_eff == 16'(N_TILES)) ? LAST_LANE_LT : LAST_LANE;
                last_ptile_q <= (TILE_W+1)'(nt_eff - 16'd1);
            end

            // ---------------- producer ----------------
            case (pstate)
                P_IDLE: begin
                    if (tree_start) begin
                        ptile  <= '0;
                        pstate <= P_ISSUE;
                    end
                end
                P_ISSUE: begin
                    // issue GEMV for tile `ptile` only when its target slot
                    // is free (the consumer is not still streaming it).
                    if (!slot_full[pslot]) begin
                        gemv_tile_idx   <= 16'(ptile);
                        gemv_tile_start <= 1'b1;
                        pstate          <= P_WAIT;
                    end
                end
                P_WAIT: begin
                    gemv_tile_idx <= 16'(ptile);    // hold stable for the GEMV
                    if (array_done) begin
                        for (int l = 0; l < LANES; l++)
                            tile_buf[pslot][l] <= array_acc_out[l*SCORE_W +: SCORE_W];
                        slot_full[pslot] <= 1'b1;
                        if (ptile == last_ptile_q) begin   // V1 runtime stop
                            pstate <= P_IDLE;       // all tiles produced
                        end else begin
                            ptile  <= ptile + 1'b1;
                            pstate <= P_ISSUE;      // next GEMV overlaps the stream
                        end
                    end
                end
                default: pstate <= P_IDLE;
            endcase

            // ---------------- consumer ----------------
            case (cstate)
                C_IDLE: begin
                    gemv_stage_g12 <= 1'b0;
                    busy           <= 1'b0;
                    if (tree_start) begin
                        tree_dec_start <= 1'b1;     // heap clear (same as baseline)
                        ctile          <= '0;
                        lane           <= '0;
                        gemv_stage_g12 <= 1'b1;
                        busy           <= 1'b1;
                        cstate         <= C_WAITFULL;
                    end
                end
                C_WAITFULL: begin
                    if (slot_full[cslot]) begin
                        lane   <= '0;
                        cstate <= C_STREAM;
                    end
                end
                C_STREAM: begin
                    // stream this tile's logits from the ping-pong slot,
                    // 1/cycle, identical order + tail semantics to baseline.
                    valid_in <= 1'b1;
                    score_in <= tile_buf[cslot][lane];
                    index_in <= IDX_W'((ctile * LANES) + lane);
                    last_in  <= is_final_idx;

                    if (is_tile_end) begin
                        slot_full[cslot] <= 1'b0;   // free the slot for the producer
                        if (ctile == last_tile_q) begin
                            cstate <= C_DRAIN;
                        end else begin
                            ctile  <= ctile + 1'b1;
                            cstate <= C_WAITFULL;
                        end
                    end else begin
                        lane <= lane + 1'b1;
                    end
                end
                C_DRAIN: begin
                    gemv_stage_g12 <= 1'b0;
                    busy           <= 1'b0;
                    cstate         <= C_IDLE;
                end
                default: cstate <= C_IDLE;
            endcase
        end
    end

    // The producer may only capture into a slot the consumer is not streaming:
    // capture fires in P_WAIT on array_done; if the consumer is mid-stream on
    // the SAME slot the producer could not have issued (slot_full held). Guard.
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && pstate == P_WAIT && array_done &&
            cstate == C_STREAM && (pslot == cslot))
            $error("tree_tile_stream_pp: producer capture into slot %0d while consumer streams it (ptile=%0d ctile=%0d)", pslot, ptile, ctile);
    end
`endif

endmodule

`default_nettype wire
