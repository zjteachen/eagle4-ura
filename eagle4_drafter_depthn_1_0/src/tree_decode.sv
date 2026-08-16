// =====================================================================
// tree_decode.sv
// Streaming top-K_F selection over a logit/score vector.
// =====================================================================
//
// The EAGLE4 candidate tree expands K_F=8 children per frontier (per
// reference/include/eagle4_gemm_abstract.h drafter_tree_decode). This
// module implements the per-frontier hardware primitive: given a stream
// of V logits (one per cycle on `valid_in`), maintain the K_F highest
// scores and their indices. After `valid_in` has been deasserted (or
// after V scores have arrived, signaled by `last_in`), assert `done`
// and present the K_F (score, index) pairs.
//
// Implementation: K_F-deep sorted insertion. On each new score:
//   - If score > current_min: pop the min, insert new score in sorted
//     position via shift-register. O(K_F) compare/move per cycle.
//
// At K_F=8, V=128256, K_PAR=1: 128,256 cycles = 641 µs at 200 MHz.
// Future scaling: K_PAR=4 or K_PAR=8 (compare K_PAR new candidates per
// cycle, fan-in into the same heap) gets this to ~80–160 µs.
//
// Multi-driver-safe: heap_score / heap_index are unpacked arrays driven
// from a single always_ff block; out_packed concatenates combinationally.
//
// External orchestration: the tree-walk (depth iteration, KV fork) is
// done outside this module — this is the per-frontier selection only.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_decode #(
    parameter int K_F     = 8,         // children per frontier (EAGLE4)
    parameter int V_MAX   = 128256,    // vocabulary size
    parameter int IDX_W   = $clog2(V_MAX),
    parameter int SCORE_W = 32
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,

    // streaming input: one (score, index) per cycle when valid_in
    input  wire                              valid_in,
    input  wire signed [SCORE_W-1:0]         score_in,
    input  wire [IDX_W-1:0]                  index_in,
    input  wire                              last_in,        // pulse on the last valid score

    // output: K_F (score, index) pairs after done
    output wire signed [K_F*SCORE_W-1:0]     topk_scores,
    output wire        [K_F*IDX_W-1:0]       topk_indices,
    output logic                             done
);

    // Min-of-heap sorted descending: heap_score[0] = largest, heap_score[K_F-1] = smallest
    logic signed [SCORE_W-1:0] heap_score [0:K_F-1];
    logic        [IDX_W-1:0]   heap_index [0:K_F-1];
    logic                      heap_init;    // 1 after start; cleared by reset/start

    // -----------------------------------------------------------------
    // TREE-INSERT-ORDER-FIX (2026-06-06)
    // The insertion decision (gt_mask / insert_pos / accept_new) and the
    // heap shift are now BOTH evaluated inside the single always_ff below,
    // off the edge-sampled `score_in`. The previous version split the
    // decision into a separate always_comb that read `score_in`; in a
    // zero-delay simulation that always_comb re-settles AFTER the always_ff
    // NBA region, so on the cycle where `score_in` changes the always_ff
    // sampled the PREVIOUS cycle's `insert_pos` (one logit stale). On a
    // monotone-increasing stream every element is a new maximum, so the
    // stale one-position-low insert emitted consecutive ranks pairwise-
    // swapped (e.g. [..,id18,id19,id16,id17,..] instead of the correct
    // [..,id19,id18,id17,id16,..]); the SET and the by-value #1 stayed
    // correct, only the ORDER was wrong. (In the real datapath the upstream
    // tree_tile_stream registers score_in/valid_in/index_in, so the value
    // is stable before the edge and the bug did not surface there; folding
    // the decision into the always_ff makes the module race-immune for ANY
    // driver / testbench without adding a pipeline stage — still 1
    // logit/cycle, no II change.)
    //
    // gt_mask / insert_pos / accept_new are kept as observability nets,
    // mirrored combinationally for waveform/debug only; they are NOT on the
    // functional path (the always_ff recomputes the decision itself).
    // -----------------------------------------------------------------
    logic [K_F-1:0] gt_mask;
    logic [K_F-1:0] insert_pos;  // one-hot: chosen position (debug mirror)
    logic           accept_new;

    always_comb begin
        for (int k = 0; k < K_F; k = k + 1)
            gt_mask[k] = ($signed(score_in) > heap_score[k]);
        insert_pos = '0;
        accept_new = 1'b0;
        for (int k = 0; k < K_F; k = k + 1) begin
            if (gt_mask[k] && !accept_new) begin
                insert_pos[k] = 1'b1;
                accept_new    = 1'b1;
            end
        end
    end

    // -----------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE = 2'd0, S_RUN = 2'd1, S_DONE = 2'd2 } state_e;
    state_e state;

    localparam logic signed [SCORE_W-1:0] NEG_INF = {1'b1, {(SCORE_W-1){1'b0}}};

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            heap_init <= 1'b0;
            for (int k = 0; k < K_F; k = k + 1) begin
                heap_score[k] <= NEG_INF;
                heap_index[k] <= '0;
            end
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        for (int k = 0; k < K_F; k = k + 1) begin
                            heap_score[k] <= NEG_INF;
                            heap_index[k] <= '0;
                        end
                        heap_init <= 1'b1;
                        state     <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (valid_in) begin
                        // TREE-INSERT-ORDER-FIX (2026-06-06)
                        // Decision + shift evaluated ATOMICALLY off the edge-
                        // sampled score_in (no cross-process settling race).
                        //   gtm[k] : score_in > heap_score[k]   (signed)
                        //   p      : FIRST k from the top with gtm[k] set
                        //   acc    : any gtm bit set (heap admits the element)
                        // Next-state of each slot:
                        //   k <  p : unchanged
                        //   k == p : new (score_in,index_in)
                        //   k >  p : OLD heap[k-1]  (slide at/below p down one,
                        //            dropping the old min). RHS read the pre-edge
                        //            heap via NBA, so a new max (p==0) lands at
                        //            slot 0 and slots 1..K_F-1 take old[k-1].
                        //
                        // PREVIOUS BUG: the decision lived in a separate
                        // always_comb reading score_in; in a zero-delay sim that
                        // comb re-settles AFTER the always_ff NBA region, so on a
                        // cycle where score_in changes the always_ff sampled the
                        // PREVIOUS cycle's insert_pos (one logit stale). On a
                        // monotone-increasing stream every element is a new max,
                        // so the stale one-slot-low insert emitted consecutive
                        // ranks PAIRWISE-SWAPPED (e.g. [..,id18,id19,id16,id17,..]
                        // instead of [..,id19,id18,id17,id16,..]); the SET and the
                        // by-value #1 stayed correct, only the ORDER was wrong.
                        // (In the real datapath tree_tile_stream registers
                        // score_in/valid_in/index_in so it was stable pre-edge and
                        // the bug did not surface there; folding the decision into
                        // the always_ff makes the module race-immune for ANY driver
                        // / TB with NO pipeline stage added -> still 1 logit/cycle.)
                        logic [K_F-1:0] gtm;
                        logic [K_F-1:0] ipos;     // one-hot chosen slot p
                        logic           acc;
                        logic           seen;     // prefix-OR: 1 for all k >= p
                        for (int k = 0; k < K_F; k = k + 1)
                            gtm[k] = (score_in > heap_score[k]);
                        ipos = '0; acc = 1'b0;
                        for (int k = 0; k < K_F; k = k + 1) begin
                            if (gtm[k] && !acc) begin
                                ipos[k] = 1'b1;
                                acc     = 1'b1;
                            end
                        end
                        if (acc) begin
                            // seen turns 1 at slot p and stays 1, so
                            // (seen && !ipos[k]) selects exactly the slots k>p
                            // that slide down by one.
                            seen = 1'b0;
                            for (int k = 0; k < K_F; k = k + 1) begin
                                if (ipos[k]) begin
                                    heap_score[k] <= score_in;
                                    heap_index[k] <= index_in;
                                    seen = 1'b1;
                                end else if (seen && k > 0) begin
                                    heap_score[k] <= heap_score[k-1];
                                    heap_index[k] <= heap_index[k-1];
                                end
                                // else: k < p -> unchanged (NBA holds old value)
                            end
                        end
                    end
                    if (last_in) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end
                end
                S_DONE: begin
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // Combinational pack of unpacked arrays into output bus
    genvar gk;
    generate
        for (gk = 0; gk < K_F; gk = gk + 1) begin : g_pack
            assign topk_scores [gk*SCORE_W +: SCORE_W] = heap_score[gk];
            assign topk_indices[gk*IDX_W   +: IDX_W]   = heap_index[gk];
        end
    endgenerate

endmodule

`default_nettype wire
