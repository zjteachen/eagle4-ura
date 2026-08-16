// =====================================================================
// tree_feed_seq.sv  -- TREE FSM STAGE 4: the per-branch candidate feed sequencer
// =====================================================================
//
// This is the module stage 3 left as a top-level "wiring recipe" comment: the
// glue that turns each branch body-pass's G8 output into the candidate stream
// tree_level_fsm consumes. It closes the last structural gap so ONE ap_start
// walks max_depth levels IN THE REAL DATAPATH (not just as isolated modules).
//
// ---------------------------------------------------------------------
// WHAT IT DOES  (per branch body pass; K_F=8 branches per level, 1 at level 0)
// ---------------------------------------------------------------------
//   on branch_feed (top_fsm S_LVL_FEED pulse, fires AFTER that branch's g8_done):
//     1. LATCH the branch's G8 rescored frontier (g8_frontier_scores/_ids) into
//        the bridge's frontier_f32/_tok inputs.  (frontier + ids co-travel.)
//     2. PULSE bridge_start.  The tree_softmax_bridge then sweeps the 512
//        RESCORED logits (g8_cand_dump_score == rescored[] @ g8_cand_rdaddr,
//        routed through the top-level arbiter mux while cand_rd_active) and
//        emits the 8 selected children's per-parent softmax probs as fp32.
//     3. on bridge_done, STREAM the 8 (prob,token) into the FSM over 8 cycles
//        while cand_req is asserted.  The FSM accumulates 8*8 = 64 per level
//        (8 at level 0) via its own ord_cnt; cand_req stays high across the
//        whole level (tree_level_fsm.sv T_FEED), so per-branch gaps are fine
//        and branch j's 8 land at ordinals j*8..j*8+7 -> par_slot = j.
//
// TIMING SAFETY.  The G8 rescored[] buffer this reads is final after S_FINAL
// (== g8_done) and is only cleared at the NEXT branch's g8 launch (that
// branch's tree_start), which is a full body pass (G1..G12, ~10^4 cyc) later --
// far longer than this sequencer's ~1043-cycle bridge sweep + 8-cycle feed. So
// the read window is race-free without stalling top_fsm (no new handshake into
// top_fsm; branch_feed is consumed as a pulse).
//
// ---------------------------------------------------------------------
// RESOURCE / ROUTING DISCIPLINE (URAM at 99.95% -- ZERO new URAM; FF only)
// ---------------------------------------------------------------------
//   * prob_buf[8]/tok_buf[8] : 8-entry register files (read one/cycle in
//     order) -> FF, NOT LUTRAM/BRAM/URAM.  8*(32+18)=400 FF.
//   * bridge_frontier_f32/_tok, small FSM state: a few hundred FF.
//   * NO high-fanout broadcast net is created here: branch_feed is a single
//     pulse, cand_valid/cand_prob/cand_token fan only into the FSM's log2 feed.
//   * every reg has a SINGLE driver (this one always_ff); valid + data
//     co-travel (cand_valid asserts the same cycle its prob/token are driven),
//     so there is no valid-before-data latency skew.
//
// xsim/CPU only -- NO synthesis run in this stage.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_feed_seq
    import tree_score_pkg::*;
#(
    parameter int G8_IDX_W = 18                 // g8 frontier vocab-id width (== TOK_W)
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- trigger: this branch's g8 rescore is done, frontier + rescored[] final
    input  wire                       branch_feed,      // 1-cyc pulse per branch (top_fsm)

    // ---- g8 frontier for THIS branch (rescored fp32 top-8 + vocab ids) ----
    input  wire [K_F*32-1:0]          g8_frontier_scores,
    input  wire [K_F*G8_IDX_W-1:0]    g8_frontier_ids,

    // ---- bridge handshake (tree_softmax_bridge = the softmax producer) ----
    output reg                        bridge_start,       // 1-cyc pulse
    output reg  [K_F*32-1:0]          bridge_frontier_f32,
    output reg  [K_F*TOK_W-1:0]       bridge_frontier_tok,
    input  wire                       bridge_done,
    input  wire [K_F*32-1:0]          bridge_prob_f32,
    input  wire [K_F*TOK_W-1:0]       bridge_tok,

    // ---- tree_level_fsm candidate feed ----
    input  wire                       cand_req,
    output reg                        cand_valid,
    output reg  [31:0]                cand_prob_f32,
    output reg  [TOK_W-1:0]           cand_token,

    // ---- status ----
    output reg                        feed_busy           // 1 while a branch is being fed
);

    typedef enum logic [2:0] {
        F_IDLE  = 3'd0,
        F_START = 3'd1,   // pulse bridge_start (frontier latched in F_IDLE->here)
        F_WAIT  = 3'd2,   // wait bridge_done, snapshot the 8 probs
        F_FEED  = 3'd3,   // stream the 8 (prob,token) while cand_req
        F_DONE  = 3'd4
    } fstate_t;
    fstate_t fs;

    // 8-entry FF result buffers (read one/cycle in order -> registers, not RAM)
    // FF-staging law: pin to registers so a future part/floorplan can't downgrade
    // these regfiles to SRL/LUTRAM (match tree_tile_stream_pp.sv:93 verbatim).
    (* ram_style = "registers", shreg_extract = "no" *)
    reg [31:0]      prob_buf [0:K_F-1];
    (* ram_style = "registers", shreg_extract = "no" *)
    reg [TOK_W-1:0] tok_buf  [0:K_F-1];
    reg [3:0]       fk;

    integer i;

`ifdef FEEDSEQ_RED_STALE
    // RED (c): FRONTIER LATCH STALE-BY-1.  Hold the PREVIOUS branch's frontier
    // and hand THAT to the bridge, so every parent is normalized/selected on
    // the wrong child set and the cross-branch tree must diverge from golden.
    // Single extra register, single driver (this always_ff).
    reg [K_F*32-1:0]       stale_f32;
    reg [K_F*G8_IDX_W-1:0] stale_ids;
`endif

    // helper: g8 vocab id (G8_IDX_W) zero-extended to TOK_W for the bridge/FSM
    function automatic logic [TOK_W-1:0] idext (input logic [G8_IDX_W-1:0] v);
        idext = {{(TOK_W-G8_IDX_W){1'b0}}, v};
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            fs <= F_IDLE; bridge_start <= 1'b0; feed_busy <= 1'b0;
            cand_valid <= 1'b0; cand_prob_f32 <= '0; cand_token <= '0;
            bridge_frontier_f32 <= '0; bridge_frontier_tok <= '0; fk <= '0;
            for (i = 0; i < K_F; i = i + 1) begin prob_buf[i] <= '0; tok_buf[i] <= '0; end
`ifdef FEEDSEQ_RED_STALE
            stale_f32 <= '0; stale_ids <= '0;
`endif
        end else begin
            bridge_start <= 1'b0;
            cand_valid   <= 1'b0;

`ifdef FEEDSEQ_RED_STALE
            // shift-register the frontier one branch behind
            if (branch_feed) begin
                stale_f32 <= g8_frontier_scores;
                stale_ids <= g8_frontier_ids;
            end
`endif
            case (fs)
                // ---- accept a branch: LATCH its frontier, kick the bridge ----
                F_IDLE: begin
                    feed_busy <= 1'b0;
                    if (branch_feed) begin
                        feed_busy <= 1'b1;
`ifdef FEEDSEQ_RED_STALE
                        bridge_frontier_f32 <= stale_f32;
                        for (i = 0; i < K_F; i = i + 1)
                            bridge_frontier_tok[i*TOK_W +: TOK_W] <= idext(stale_ids[i*G8_IDX_W +: G8_IDX_W]);
`else
                        bridge_frontier_f32 <= g8_frontier_scores;
                        for (i = 0; i < K_F; i = i + 1)
                            bridge_frontier_tok[i*TOK_W +: TOK_W] <= idext(g8_frontier_ids[i*G8_IDX_W +: G8_IDX_W]);
`endif
                        fs <= F_START;
                    end
                end

                F_START: begin
                    bridge_start <= 1'b1;   // frontier now stable at bridge inputs
                    fs <= F_WAIT;
                end

                // ---- bridge runs (~1043 cyc); snapshot its 8 probs on done ----
                F_WAIT: begin
                    if (bridge_done) begin
                        for (i = 0; i < K_F; i = i + 1) begin
                            prob_buf[i] <= bridge_prob_f32[i*32 +: 32];
                            tok_buf [i] <= bridge_tok    [i*TOK_W +: TOK_W];
                        end
                        fk <= 4'd0;
                        fs <= F_FEED;
                    end
                end

                // ---- stream 8 (prob,token) into the FSM, one per cand_req cyc ----
                F_FEED: begin
                    if (cand_req) begin
                        cand_valid <= 1'b1;
`ifdef FEEDSEQ_RED_OFFBYONE
                        // RED (a): FEED OFF-BY-ONE.  prob is taken one child ahead
                        // of its token (wrap), so each child is scored with a
                        // SIBLING's probability -> the wrong children survive.
                        cand_prob_f32 <= prob_buf[(fk == K_F-1) ? 3'd0 : (fk[2:0] + 3'd1)];
`else
                        cand_prob_f32 <= prob_buf[fk[2:0]];
`endif
                        cand_token    <= tok_buf[fk[2:0]];
                        if (fk == K_F-1) fs <= F_DONE;
                        else             fk <= fk + 1'b1;
                    end
                end

                F_DONE: begin
                    feed_busy <= 1'b0;
                    fs <= F_IDLE;
                end

                default: fs <= F_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
