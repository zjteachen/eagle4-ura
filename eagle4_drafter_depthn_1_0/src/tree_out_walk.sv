// =====================================================================
// tree_out_walk.sv
// EAGLE4 depth-N tree -> SGLang verify ABI, node-major walk.
// =====================================================================
//
// Consumes the accumulated tree that tree_level_fsm holds in tn_mem (via its
// registered tn_rd_* readback port) and emits the tensors the target-side
// verify kernel actually consumes.
//
// THE ABI IS FOUR TENSORS, NOT SIX (reports/SGLANG_TREE_KV_SURFACE.md):
//   draft_token          - the vocab id per tree node, node-major
//   retrive_index        - IDENTITY when nodes are emitted in pool order, so
//                          it is emitted as a CONSTANT and costs no storage
//   retrive_next_token   - first child of each node   (-1 if none)
//   retrive_next_sibling - next sibling of each node  (-1 if none)
// `custom_mask` and `positions` serve the TARGET forward, not the verify
// kernel, and are therefore deliberately NOT built here.
//
// ---------------------------------------------------------------------
// HOW next_token / next_sibling ARE BUILT -- one reverse pass, no sort
// ---------------------------------------------------------------------
// The naive construction is O(N^2) (for each node, scan for its children).
// Instead this walks nodes in DESCENDING pool order maintaining head[p], the
// lowest-indexed child of p seen so far:
//
//     for c = N-1 downto 1:
//         p = parent[c]
//         next_sibling[c] = head[p]     // the child just after c
//         head[p]         = c
//     next_token[p] = head[p]           // p's FIRST child
//
// Descending order is what makes the sibling chain come out ASCENDING, which
// is the order the verify kernel expects. head[] is K_F*MAX_DEPTH entries of
// NODE_W bits -- FF, not BRAM (see the seam-staging mandate: intermediate
// staging is FF; this array is re-read associatively every node).
//
// Node 0..K_F-1 are the level-0 seeds and have NO parent (root sentinel), so
// the loop stops at c >= K_F for the sibling pass.
//
// =====================================================================
// READBACK LATENCY CONTRACT  (2026-07-29 FIX -- reports/results/TREE_PARENT_POINTER_RCA.md)
// ---------------------------------------------------------------------
// The tn_rd_* readback port is TWO cycles deep, not one:
//
//   cycle T   : this FSM registers  tn_rd_idx <= c            (stage 1: address reg)
//   cycle T+1 : tn_rd_idx == c, and tree_level_fsm samples
//               {tn_rd_score,tn_rd_token,tn_rd_parent} <= tn_mem[tn_rd_idx]
//               and tn_rd_idx_q <= tn_rd_idx                  (stage 2: data reg)
//   cycle T+2 : tn_rd_token/parent/score are the words of node c, and
//               tn_rd_depth/tn_rd_pos (derived from tn_rd_idx_q) are ALSO node
//               c's -- all five outputs are co-timed here and ONLY here.
//               (producer: tree_level_fsm.sv:707-713)
//
// tn_rd_idx is a REGISTER of this module (assigned inside the always_ff), so
// the address stage costs a cycle that a "1-cycle read latency" reading of the
// port declaration hides.  The pre-2026-07-29 RTL consumed the data in the
// state immediately after issuing (T+1), i.e. ONE cycle early, so every read
// returned the words of the PREVIOUSLY addressed node:
//
//   * reverse pass, descending c  -> data was node c+1's, so line
//     `head_arr[tn_rd_parent] <= c` executed head[parent[c+1]] <= c and the
//     emitted first-child list came out 7+8s (clamped to 8 at s=0) instead of
//     the SGLang-authoritative 8+8s;
//   * emit pass, ascending c      -> emitted slot c carried node c-1's token /
//     score / pos, so slot 0 was a stale duplicate of slot K_F+1, the true
//     depth-1 frontier sat at pool[1..K_F], and the LAST node's payload
//     (npool-1) was pushed off the end of the beat count and never emitted.
//
// FIX: one dedicated WAIT state per pass (W_REV_WT / W_EMIT_WT) so the index
// (c, held constant across RD->WT->WR because it only advances in the WR
// state) and the data are consumed co-timed at T+2.  Both passes are fixed
// TOGETHER -- fixing one alone relocates the defect instead of removing it.
// Cost: 3 cycles/node per pass instead of 2 (see the OOC/timing note in
// reports/results/TREEWALK_LATENCY_FIX.md); the emit stream is valid-gated so
// the reduced beat rate is transparent to tree_dma_writer.
//
// The pre-fix behaviour is preserved behind `TREEWALK_RED_PREFIX_LATENCY as the
// armed RED arm of the xsim gate -- it MUST reproduce ntok=[8,15,23,...,63].
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_out_walk
    import tree_score_pkg::*;
#(
    parameter int NODE_MAX = N_MAX          // 904 @ MAX_DEPTH_HW=15
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- control ----
    input  wire                    start,        // pulse, from top_fsm S_LVL_OUT
    input  wire [NODE_W-1:0]       n_nodes,      // nodes actually built this tree
    output logic                   done,

    // ---- tree readback (tree_level_fsm tn_rd_*, TWO-cycle read latency from
    //      the tn_rd_idx REGISTER-WRITE cycle -- see the contract above) ----
    output logic [NODE_W-1:0]      tn_rd_idx,
    input  wire  [TOK_W-1:0]       tn_rd_token,
    input  wire  [NODE_W-1:0]      tn_rd_parent,
    input  wire  [DEPTH_W-1:0]     tn_rd_depth,
    input  wire  [POS_W-1:0]       tn_rd_pos,
    input  wire signed [SCORE_W-1:0] tn_rd_score,   // COSTDRAFTTREE-ABI (G1)

    // ---- emit stream to the output writer (one node per beat) ----
    output logic                   emit_valid,
    output logic [NODE_W-1:0]      emit_idx,       // == retrive_index (identity)
    output logic [TOK_W-1:0]       emit_token,     // draft_token
    output logic signed [NODE_W:0] emit_next_tok,  // retrive_next_token   (-1 = none)
    output logic signed [NODE_W:0] emit_next_sib,  // retrive_next_sibling (-1 = none)
    output logic [POS_W-1:0]       emit_pos,
    output logic signed [SCORE_W-1:0] emit_score   // COSTDRAFTTREE-ABI (G1) cumu log2 score
);

    localparam logic signed [NODE_W:0] NONE = -1;

    // ---- link arrays (FF register files; NOT BRAM -- re-read associatively) ----
    (* ram_style = "registers" *) logic signed [NODE_W:0] head_arr [0:NODE_MAX-1];
    (* ram_style = "registers" *) logic signed [NODE_W:0] nsib_arr [0:NODE_MAX-1];

    typedef enum logic [3:0] {
        W_IDLE,
        W_CLR,      // head[]/nsib[] <- NONE
        W_REV_RD,   // reverse pass: register tn_rd_idx <= c          (T)
        W_REV_WT,   // reverse pass: tn_mem[] read is sampled here    (T+1)
        W_REV_WR,   // reverse pass: consume parent[c], link          (T+2)
        W_EMIT_RD,  // forward pass: register tn_rd_idx <= c          (T)
        W_EMIT_WT,  // forward pass: tn_mem[] read is sampled here    (T+1)
        W_EMIT_WR,  // forward pass: emit the beat for node c         (T+2)
        W_DONE
    } wstate_t;
    wstate_t st;

    logic [NODE_W-1:0] c;
    logic [NODE_W-1:0] n_q;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            st <= W_IDLE; c <= '0; n_q <= '0;
            done <= 1'b0; emit_valid <= 1'b0; tn_rd_idx <= '0;
        end else begin
            done       <= 1'b0;
            emit_valid <= 1'b0;

            case (st)
                W_IDLE: if (start) begin
                    n_q <= n_nodes;
                    c   <= '0;
                    st  <= W_CLR;
                end

                // Clear both link arrays. One entry per cycle keeps this a
                // single-write-port register file (a for-loop reset would
                // demand NODE_MAX simultaneous writes).
                W_CLR: begin
                    head_arr[c] <= NONE;
                    nsib_arr[c] <= NONE;
                    if (c == NODE_W'(NODE_MAX-1)) begin
                        // reverse pass starts at the LAST real node
                        c  <= (n_q > NODE_W'(1)) ? (n_q - NODE_W'(1)) : '0;
                        st <= (n_q > NODE_W'(K_F)) ? W_REV_RD : W_EMIT_RD;
                    end else begin
                        c <= c + NODE_W'(1);
                    end
                end

                // ---- reverse pass: build sibling chain + first-child head ----
                W_REV_RD: begin
                    // stage-1 (address register). Data lands TWO cycles later.
                    tn_rd_idx <= c;
`ifdef TREEWALK_RED_PREFIX_LATENCY
                    // RED ARM (the pre-2026-07-29 defect): consume after ONE
                    // cycle. MUST make ntok come out [8,15,23,31,39,47,55,63].
                    st        <= W_REV_WR;
`else
                    st        <= W_REV_WT;
`endif
                end
                W_REV_WT: begin
                    // stage-2: tree_level_fsm samples tn_mem[tn_rd_idx] on THIS
                    // edge. Nothing to do but hold c and tn_rd_idx steady.
                    st <= W_REV_WR;
                end
                W_REV_WR: begin
                    // tn_rd_parent is now genuinely valid for node c (T+2).
                    nsib_arr[c]           <= head_arr[tn_rd_parent];
                    head_arr[tn_rd_parent] <= $signed({1'b0, c});
                    if (c <= NODE_W'(K_F)) begin
                        // nodes 0..K_F-1 are level-0 seeds: no parent to link
                        c  <= '0;
                        st <= W_EMIT_RD;
                    end else begin
                        c  <= c - NODE_W'(1);
                        st <= W_REV_RD;
                    end
                end

                // ---- forward pass: emit node-major ----
                W_EMIT_RD: begin
                    tn_rd_idx <= c;          // stage-1 (address register)
`ifdef TREEWALK_RED_PREFIX_LATENCY
                    // RED ARM: consume after ONE cycle -> slot c carries node
                    // c-1's token/score/pos and node n_q-1 never emits.
                    st        <= W_EMIT_WR;
`else
                    st        <= W_EMIT_WT;
`endif
                end
                W_EMIT_WT: begin
                    st <= W_EMIT_WR;         // stage-2 sample cycle
                end
                W_EMIT_WR: begin
                    // token/score/pos are node c's here (T+2), co-timed with
                    // emit_idx = c, so slot c carries node c -- including the
                    // last node (c = n_q-1), whose payload the pre-fix RTL lost.
                    emit_valid    <= 1'b1;
                    emit_idx      <= c;                 // retrive_index = identity
                    emit_token    <= tn_rd_token;
                    emit_pos      <= tn_rd_pos;
                    emit_score    <= tn_rd_score;       // co-timed with tn_rd_token
                    emit_next_tok <= head_arr[c];       // first child, or -1
                    emit_next_sib <= nsib_arr[c];       // next sibling, or -1
                    if (c + NODE_W'(1) >= n_q) begin
                        st <= W_DONE;
                    end else begin
                        c  <= c + NODE_W'(1);
                        st <= W_EMIT_RD;
                    end
                end

                W_DONE: begin
                    done <= 1'b1;
                    st   <= W_IDLE;
                end

                default: st <= W_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    // The identity claim is load-bearing (retrive_index is emitted as a
    // CONSTANT downstream), so assert it rather than trusting the comment.
    always_ff @(posedge clk)
        if (rst_n && emit_valid)
            assert (emit_idx < n_q)
                else $error("tree_out_walk: emit_idx %0d >= n_nodes %0d", emit_idx, n_q);
`endif

endmodule

`default_nettype wire
