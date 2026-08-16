// =====================================================================
// tree_merge_pipe.sv
// Pipelined partial-heap merge-tree for P-wide top-K_F intake @ 200 MHz.
// =====================================================================
//
// Replaces the flat combinational P-stage insertion cascade of
// tree_decode_kpar (bit-exact but 31 logic levels @ P=4 → only ~135 MHz)
// with a 2-stage pipeline whose IN-LOOP combinational depth is FIXED and
// P-independent (≈ one 32-bit compare + a (KF+P)-input popcount + a one-hot
// mux ≈ ≤12 logic levels), so P=4 and P=8 both close 200 MHz.
//
// Contract (MUST match the serial tree_decode + the golden tie rule
// "earlier index wins"): the final top-K_F=8 (scores+indices) is IDENTICAL
// to the serial tree_decode and to tree_decode_kpar. The caller streams
// indices monotonically increasing and packs lane g with the LOWER vocab
// index of the P-group (lane 0 = earliest index).
//
// ------------------------------------------------------------------ WHY THE
// NAIVE REGISTER-SPLIT WAS WRONG (the fold-back race): splitting the P-stage
// cascade with a register INSIDE it makes the next group's lane-0 read a
// heap that has only PARTIALLY absorbed the previous group → a fold-back
// race (verified 7/8 mismatch in the K_PAR study). This design AVOIDS that:
//   * Stage 0 (pre-sort) is purely FEED-FORWARD on the P incoming candidates
//     — no dependence on the heap, so it is safe to register (a fixed skid).
//   * Stage 1 (merge) is the ONLY feedback edge: heap_reg → merge → heap_reg,
//     and it consumes EXACTLY ONE fully-sorted group per cycle, in strict
//     order, each merge seeing the previous group's COMMITTED heap. No group
//     is ever partially folded back. Validated cycle-accurately + 900 random
//     adversarial vectors vs the serial golden (see TREE_MERGE_RESULT.md).
//
// ------------------------------------------------------------------ STRUCTURE
//   Stage 0 (registered, feed-forward):
//     sort the P incoming (score,index) candidates DESCENDING, stable on lane
//     order (lane 0 wins ties → earlier vocab index wins). Disabled lanes are
//     forced to NEG_INF so they never enter the heap.
//   Stage 1 (registered, the feedback loop):
//     rank-merge the registered sorted-P with the running heap_score/index.
//     Each of the (KF+P) elements gets a rank = # elements that strictly
//     out-rank it under the total order
//        - same class (heap vs heap, cand vs cand): lower within-class pos wins
//          (both lists are already sorted descending)
//        - heap-elem vs cand-elem: heap wins iff heap.score >= cand.score
//          (i.e. a candidate only beats a heap slot when STRICTLY greater →
//           reproduces the serial strict-'>' insertion / earlier-index-wins)
//     output[r] = the element whose rank == r, for r in 0..KF-1.
//
// Latency: a group presented on cycle t is committed into the heap on cycle
// t+2 (Stage0 reg + Stage1 reg). `last_in` is tracked through the same 2-deep
// shift so `done` asserts the cycle after the final group is committed.
//
// Interface mirrors tree_decode_kpar (P-wide streaming port).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_merge_pipe #(
    parameter int K_F     = 8,
    parameter int P       = 4,         // candidates consumed per cycle
    parameter int V_MAX   = 128256,
    parameter int IDX_W   = $clog2(V_MAX),
    parameter int SCORE_W = 32
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,

    // P-wide streaming input: P (score,index) pairs per cycle when valid_in.
    // Lane 0 holds the lower vocab index of the group (earlier-index-wins).
    input  wire                              valid_in,
    input  wire signed [P*SCORE_W-1:0]       scores_in,
    input  wire        [P*IDX_W-1:0]         indices_in,
    input  wire        [P-1:0]               lane_en,    // 1=lane carries a real candidate
    input  wire                              last_in,    // pulse on the last valid group

    output wire signed [K_F*SCORE_W-1:0]     topk_scores,
    output wire        [K_F*IDX_W-1:0]       topk_indices,
    output logic                             done
);

    localparam logic signed [SCORE_W-1:0] NEG_INF = {1'b1, {(SCORE_W-1){1'b0}}};
    localparam int N_ELEM = K_F + P;                 // merge-network input width
    localparam int RANK_W = (N_ELEM <= 2) ? 1 : $clog2(N_ELEM);

    // ----------------------------------------------------------------
    // Heap state (descending: slot 0 = largest). Single-writer always_ff.
    // ----------------------------------------------------------------
    logic signed [SCORE_W-1:0] heap_score [0:K_F-1];
    logic        [IDX_W-1:0]   heap_index [0:K_F-1];

    // ================================================================
    // STAGE 0 — combinational pre-sort of the P incoming candidates.
    //   Feed-forward only (depends solely on the streaming inputs).
    //   Sort DESCENDING, stable on lane order (lane 0 = earliest index wins
    //   ties). Disabled lanes mapped to NEG_INF so they cannot enter.
    //   For each output position q (0..P-1) we pick the lane with rank == q,
    //   where lane a out-ranks lane b iff:
    //       score[a] > score[b]  OR  (score[a]==score[b] AND a < b)
    //   (with disabled lanes forced to NEG_INF and to lose all ties).
    // ================================================================
    logic signed [SCORE_W-1:0] cand_sc [0:P-1];
    logic        [IDX_W-1:0]   cand_ix [0:P-1];

    // sorted-P combinational result
    logic signed [SCORE_W-1:0] sort_sc [0:P-1];
    logic        [IDX_W-1:0]   sort_ix [0:P-1];

    always_comb begin
        // effective per-lane score/index (disabled → NEG_INF)
        for (int j = 0; j < P; j++) begin
            if (lane_en[j]) begin
                cand_sc[j] = $signed(scores_in [j*SCORE_W +: SCORE_W]);
                cand_ix[j] = indices_in[j*IDX_W   +: IDX_W];
            end else begin
                cand_sc[j] = NEG_INF;
                cand_ix[j] = '0;
            end
        end

        // rank each lane, then scatter to its sorted slot
        for (int q = 0; q < P; q++) begin
            sort_sc[q] = NEG_INF;
            sort_ix[q] = '0;
        end
        for (int a = 0; a < P; a++) begin
            int unsigned rnk;
            rnk = 0;
            for (int b = 0; b < P; b++) begin
                if (b != a) begin
                    logic a_lt_b; // does lane b out-rank lane a ?
                    a_lt_b = ($signed(cand_sc[b]) > $signed(cand_sc[a])) ||
                             (($signed(cand_sc[b]) == $signed(cand_sc[a])) && (b < a));
                    if (a_lt_b) rnk = rnk + 1;
                end
            end
            // lane a lands at sorted slot rnk
            sort_sc[rnk] = cand_sc[a];
            sort_ix[rnk] = cand_ix[a];
        end
    end

    // ---- Stage 0 registers (feed-forward skid) ----
    logic signed [SCORE_W-1:0] s0_sc [0:P-1];
    logic        [IDX_W-1:0]   s0_ix [0:P-1];
    logic                      s0_valid;
    logic                      s0_last;

    // ================================================================
    // STAGE 1 — combinational rank-merge of (heap descending) with the
    //   registered sorted-P (descending), keeping the top K_F. FIXED depth
    //   (one compare + (KF+P)-input popcount + one-hot mux), P-independent
    //   in critical-path depth (P only widens the popcount tree → +log P).
    //
    //   Element layout in merge[]:  e in 0..K_F-1     = heap[e]   (class HEAP)
    //                               e in K_F..K_F+P-1 = sort[e-K_F](class CAND)
    //   rank(e) = # of OTHER elements that out-rank e:
    //     same class: lower within-class position out-ranks (lists pre-sorted)
    //     heap a vs cand b: heap out-ranks iff heap.sc >= cand.sc
    //                       cand out-ranks iff cand.sc >  heap.sc
    // ================================================================
    logic signed [SCORE_W-1:0] m_sc  [0:N_ELEM-1];
    logic        [IDX_W-1:0]   m_ix  [0:N_ELEM-1];
    logic                      m_cls [0:N_ELEM-1];   // 0 = heap, 1 = cand
    // within-class position is implicit: heap → e, cand → e-K_F

    logic [RANK_W-1:0]         m_rank [0:N_ELEM-1];

    logic signed [SCORE_W-1:0] merged_sc [0:K_F-1];
    logic        [IDX_W-1:0]   merged_ix [0:K_F-1];

    always_comb begin
        // assemble merge inputs
        for (int e = 0; e < K_F; e++) begin
            m_sc[e]  = heap_score[e];
            m_ix[e]  = heap_index[e];
            m_cls[e] = 1'b0;             // heap
        end
        for (int e = 0; e < P; e++) begin
            m_sc[K_F+e]  = s0_sc[e];
            m_ix[K_F+e]  = s0_ix[e];
            m_cls[K_F+e] = 1'b1;         // cand
        end

        // rank each element
        for (int a = 0; a < N_ELEM; a++) begin
            int unsigned rnk;
            int a_pos;
            a_pos = m_cls[a] ? (a - K_F) : a;     // within-class position
            rnk = 0;
            for (int b = 0; b < N_ELEM; b++) begin
                if (b != a) begin
                    logic b_outranks_a;
                    int b_pos;
                    b_pos = m_cls[b] ? (b - K_F) : b;
                    if (m_cls[a] == m_cls[b]) begin
                        // same class: lower within-class position out-ranks
                        b_outranks_a = (b_pos < a_pos);
                    end else if (m_cls[b] == 1'b0) begin
                        // b heap, a cand: heap out-ranks a iff heap.sc >= cand.sc
                        b_outranks_a = ($signed(m_sc[b]) >= $signed(m_sc[a]));
                    end else begin
                        // b cand, a heap: cand out-ranks a iff cand.sc > heap.sc
                        b_outranks_a = ($signed(m_sc[b]) >  $signed(m_sc[a]));
                    end
                    if (b_outranks_a) rnk = rnk + 1;
                end
            end
            m_rank[a] = rnk[RANK_W-1:0];
        end

        // scatter to outputs: output[r] = element whose rank == r (r < K_F)
        for (int r = 0; r < K_F; r++) begin
            merged_sc[r] = NEG_INF;
            merged_ix[r] = '0;
        end
        for (int a = 0; a < N_ELEM; a++) begin
            if (m_rank[a] < RANK_W'(K_F)) begin
                merged_sc[m_rank[a]] = m_sc[a];
                merged_ix[m_rank[a]] = m_ix[a];
            end
        end
    end

    // ================================================================
    // FSM + registers
    // ================================================================
    typedef enum logic [1:0] { S_IDLE = 2'd0, S_RUN = 2'd1, S_DRAIN = 2'd2, S_DONE = 2'd3 } state_e;
    state_e state;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            s0_valid <= 1'b0;
            s0_last  <= 1'b0;
            for (int k = 0; k < K_F; k++) begin
                heap_score[k] <= NEG_INF;
                heap_index[k] <= '0;
            end
            for (int j = 0; j < P; j++) begin
                s0_sc[j] <= NEG_INF;
                s0_ix[j] <= '0;
            end
        end else begin
            done <= 1'b0;

            // ---- Stage 1 commit: merge sorted-P (if valid) into the heap ----
            if (s0_valid) begin
                for (int k = 0; k < K_F; k++) begin
                    heap_score[k] <= merged_sc[k];
                    heap_index[k] <= merged_ix[k];
                end
            end

            // ---- Stage 0 commit: latch the pre-sorted incoming group ----
            // default: no new group this cycle
            s0_valid <= 1'b0;
            s0_last  <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        for (int k = 0; k < K_F; k++) begin
                            heap_score[k] <= NEG_INF;
                            heap_index[k] <= '0;
                        end
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (valid_in) begin
                        for (int j = 0; j < P; j++) begin
                            s0_sc[j] <= sort_sc[j];
                            s0_ix[j] <= sort_ix[j];
                        end
                        s0_valid <= 1'b1;
                        s0_last  <= last_in;
                        if (last_in) state <= S_DRAIN;
                    end
                end
                S_DRAIN: begin
                    // last group is in the Stage0→Stage1 skid; it commits to the
                    // heap on the NEXT clock edge (because s0_valid is high this
                    // cycle from the last S_RUN). Then assert done.
                    if (s0_last) begin
                        // this is the cycle the final group's merge is committing
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

    // ----------------------------------------------------------------
    // Output pack
    // ----------------------------------------------------------------
    genvar gk;
    generate
        for (gk = 0; gk < K_F; gk = gk + 1) begin : g_pack
            assign topk_scores [gk*SCORE_W +: SCORE_W] = heap_score[gk];
            assign topk_indices[gk*IDX_W   +: IDX_W]   = heap_index[gk];
        end
    endgenerate

endmodule

`default_nettype wire
