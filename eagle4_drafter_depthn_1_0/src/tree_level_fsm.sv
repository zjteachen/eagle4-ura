// =====================================================================
// tree_level_fsm.sv -- the on-chip TREE LEVEL FSM (depth-N stage 1)
// =====================================================================
//
// expand -> score -> select -> update frontier -> loop while depth < max_depth
//
// SCOPE (stage 1, tree logic only). This module owns the level loop, the
// cumulative score recurrence, the global top-8-of-64 selection, and the
// per-level bookkeeping. It deliberately does NOT own: per-branch KV
// isolation, re-parenting of KV lanes, pruning/retraction, embedding gather,
// or the host output ABI. Those are stage 2+ (see TREE_FSM_STAGE1.md).
//
// ---------------------------------------------------------------------
// SEMANTICS -- pinned to reports/DEPTHN_GOLDEN_REFERENCE.md
// ---------------------------------------------------------------------
//  * every level, ALL K_F=8 frontier nodes expand x 8 children = 64 candidates
//    (§1.2, spec_utils.py:500)
//  * a child's score is the CUMULATIVE PRODUCT of softmax probabilities
//    (§1.3, spec_utils.py:480-482) -- carried here as a SUM in the log2
//    domain, which is exactly order-equivalent. See tree_score_pkg.sv for
//    why linear cannot be used (it underflows fp32 before depth 15).
//  * survivors are the GLOBAL top-8 across all 64 -- NOT per-parent
//    (§1.4, spec_utils.py:483-485). This is the single most-likely-to-be-
//    wrong detail; RED arm (a) exists to prove we did not get it wrong.
//  * pool layout is PARENT-SLOT MAJOR, CHILD-RANK MINOR (§1.2):
//        pool(0,c)   = c
//        pool(L,s,c) = K_F + (L-1)*K_F^2 + s*K_F + c        for L >= 1
//    Candidates are fed in exactly that order and the FSM derives
//    (parent_slot, child_rank) from the arrival ordinal, so a transposed
//    feed disagrees with the ordinal immediately. RED arm (d).
//  * position is LEVEL-UNIFORM: position = ctx_len + depth (§1.5). A level
//    carries one position scalar, not 64 values. RED arm (c).
//
// ---------------------------------------------------------------------
// max_depth (control reg 0x20) IS LIVE HERE -- and cannot go dormant again
// ---------------------------------------------------------------------
// Before this module, 0x20 had ZERO datapath consumers: declared
// (control_regs.sv:81), decoded (:569), read back (:729), wired into
// eagle4_drafter.sv:2579 -- and never read. A testbench asserted only that
// the register READS BACK what was written (tb_ps_h1.sv:203-204 in the
// ps_jewel forks), which is a storage check, not a behaviour check: it would
// pass verbatim with the output port left unconnected.
//
// Here `max_depth_i` is the loop bound itself (T_NEXT), so a wrong value
// changes the number of levels emitted. Dormancy is made structurally hard
// to reintroduce in three independent ways:
//   1. `levels_run_o` reports how many levels actually ran; the gate sweeps
//      max_depth = 1..4 and asserts levels_run_o == max_depth each time. A
//      disconnected max_depth cannot satisfy four different values.
//   2. the T_NEXT comparison is the ONLY exit from the level loop, so
//      trimming max_depth changes the FSM rather than trimming dead logic.
//   3. a simulation assertion fires on any disagreement at `done`.
//
// ---------------------------------------------------------------------
// TIE RULE (normative -- documented because the golden cannot supply one)
// ---------------------------------------------------------------------
// Candidates are streamed in pool order and the selector (tree_decode.sv)
// admits on STRICT greater-than, so an incoming candidate never displaces an
// equal-scoring one already in the heap. Consequence: ON AN EXACT TIE THE
// LOWER POOL INDEX KEEPS THE HIGHER RANK. This is deterministic and
// reproducible; torch.topk's tie order is not, so the gate marks tie-affected
// levels UNVERIFIABLE explicitly (never silently) rather than comparing them.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_level_fsm
    import tree_score_pkg::*;
(
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- control ----
    input  wire                       start,          // pulse: begin a new tree
    input  wire [31:0]                ctx_len_i,      // seq_len of the shared prefix
    input  wire [31:0]                max_depth_i,    // control reg 0x20 -- THE loop bound

    // ---- per-level candidate feed (parent-slot major, child-rank minor) ----
    // level 0 supplies K_F seeds; every level >= 1 supplies N_CAND = 64.
    output logic                      cand_req,       // FSM is accepting candidates
    input  wire                       cand_valid,
    input  wire [31:0]                cand_prob_f32,  // p(child|parent), IEEE-754 binary32
    input  wire [TOK_W-1:0]           cand_token,

    // ---- per-level result ----
    output logic                      level_done,     // pulse: survivors below are valid
    // CO-TIMED WITH level_done (bugfix 2026-07-22): 1 = the level_done being
    // emitted right now is the LAST one of the tree. ADDITIVE -- `done` below
    // keeps its original T_DONE timing so every existing gate/consumer is
    // untouched. See the `last_level` comment for why the depth-N loop cannot
    // use `done` for its exit decision.
    output logic                      level_last,
    output logic [DEPTH_W-1:0]        level_idx_o,    // which level just committed (0-based)
    output logic [POS_W-1:0]          level_pos_o,    // ctx_len + depth (level-uniform)
    output logic [K_F*NODE_W-1:0]     surv_node_o,    // 8 surviving POOL indices, rank order
    output logic [K_F*NODE_W-1:0]     surv_parent_o,  // their parents' pool indices
    // G3 (2026-07-23): the SAME parent, expressed as the parent's SLOT (rank
    // 0..7) in the PREVIOUS level's frontier -- i.e. which of the previous
    // level's 8 body passes produced it. This is the `a`-bank index F4's
    // per-parent for_reasoning re-gather must read (rd_branch). It is NOT
    // derivable from surv_parent_o's low bits beyond level 1: see the
    // cs_pslot declaration for the exact statement of why.
    output logic [K_F*SLOT_W-1:0]     surv_parent_slot_o,
    output logic [K_F*TOK_W-1:0]      surv_token_o,
    output logic [K_F*SCORE_W-1:0]    surv_score_o,   // cumulative log2 score, Q12.28

    // ---- accumulated tree readback (node-major walk for the output stage) ----
    input  wire [NODE_W-1:0]          tn_rd_idx,      // registered, 1-cycle read latency
    output logic [TOK_W-1:0]          tn_rd_token,
    output logic [NODE_W-1:0]         tn_rd_parent,
    output logic [DEPTH_W-1:0]        tn_rd_depth,
    output logic [POS_W-1:0]          tn_rd_pos,
    // COSTDRAFTTREE-ABI (G1): per-node cumulative log2 score, Q12.28 signed, co-timed
    // with tn_rd_token. Emitted downstream as the 5th SoA region so SGLang can prune.
    output logic signed [SCORE_W-1:0] tn_rd_score,

    // ---- status ----
    output logic                      busy,
    output logic                      done,
    output logic [DEPTH_W-1:0]        levels_run_o,   // levels actually emitted (max_depth gate)
    output logic [NODE_W-1:0]         n_nodes_o       // pool nodes written this tree
);

    // =================================================================
    // declarations
    // =================================================================
    typedef enum logic [2:0] {
        T_IDLE   = 3'd0,
        T_LVL    = 3'd1,   // arm the level: reset selector, clear counters
        T_FEED   = 3'd2,   // accept candidates, score them, stream to the selector
        T_DRAIN  = 3'd3,   // flush the log2 pipeline
        T_SELECT = 3'd4,   // wait for the top-8-of-64 verdict
        T_COMMIT = 3'd5,   // publish survivors, update the frontier
        T_NEXT   = 3'd6,   // loop while depth < max_depth  <-- max_depth consumed HERE
        T_DONE   = 3'd7
    } tstate_t;
    tstate_t st;

    logic [CAND_W:0]    ord_cnt;      // candidates accepted this level (0..64)
    logic [CAND_W:0]    scored_cnt;   // candidates that have exited the log2 pipe
    logic [2:0]         drain_cnt;
    logic [DEPTH_W-1:0] level_idx;    // 0-based level being built
    logic [POS_W-1:0]   ctx_len_q;
    logic [DEPTH_W-1:0] max_depth_q;
    logic [NODE_W-1:0]  n_nodes;

    wire lvl0 = (level_idx == 0);

    // =================================================================
    // THE LOOP BOUND -- ONE expression, TWO co-timed consumers
    // =================================================================
    // max_depth_q is read here and nowhere else. T_NEXT uses this wire to
    // decide whether to build another level, and `level_last` is a REGISTERED
    // copy of the SAME wire, so there is still exactly one loop bound.
    //
    // WHY `level_last` EXISTS (bugfix 2026-07-22, depth-N level loop never
    // exited):
    //   `level_done` pulses during T_NEXT, but `done` is registered TWO states
    //   later (T_DONE). top_fsm's depth-N exit edge samples them TOGETHER --
    //       S_LVL_SEL: if (level_done) state_n = tree_all_done ? S_LVL_OUT : ...
    //   -- so tree_all_done was ALWAYS 0 at the sampling instant. Consequence:
    //   the level loop ran ONE LEVEL PAST max_depth and then HUNG in S_LVL_SEL
    //   waiting for a level_done that could never come (this FSM had already
    //   returned to T_IDLE with cand_req low). Measured in
    //   tb_depthn_child_index: 16 gathers at max_depth=2 instead of 8, then no
    //   fsm_done.
    //
    //   The fix is ADDITIVE ON PURPOSE. `done` keeps its exact original
    //   T_DONE timing -- it is the "tree retired, ready for the next start"
    //   handshake that tb_tree_level_fsm/tb_tree_fed_loop rely on with
    //   `@(posedge clk iff done)` followed by a fresh `start`. Re-timing `done`
    //   was tried first and it silently DROPPED one captured step per run
    //   (start landed while st != T_IDLE), turning the stage-1 gate RED with
    //   957/1007 survivor-SET failures -- a one-step golden desync, not a real
    //   survivor bug. Recorded here so nobody re-times `done` again.
    wire last_level = !((level_idx + 1'b1) < max_depth_q);

    // ---- frontier: 8 entries, read associatively every scored candidate ----
    // FF (registers), NOT LUTRAM/BRAM: an 8-entry array read every cycle is a
    // register file, and a BRAM here would inject read latency straight into
    // the score recurrence for no capacity benefit. 8*(40+10) = 400 FF.
    logic signed [SCORE_W-1:0] fr_score [0:K_F-1];
    logic        [NODE_W-1:0]  fr_node  [0:K_F-1];

`ifdef TREE_RED_PER_PARENT
    // RED arm (a) storage only -- not present in the good build.
    logic signed [SCORE_W-1:0] pp_score [0:K_F-1];
    logic        [CAND_W-1:0]  pp_ord   [0:K_F-1];
`endif

    // ---- per-level candidate staging (transient, 64 entries) ----
    // Needed because a survivor is identified by its ordinal only AFTER the
    // selector finishes, so the level's tokens/parents must be held somewhere
    // randomly addressable. 64*(18+10) = 1,792 b -> LUTRAM (async read, 8
    // read ports at commit). This is §4.4's "candidate staging (transient)".
    (* ram_style = "distributed" *)
    logic [TOK_W-1:0]  cs_tok [0:N_CAND-1];
    (* ram_style = "distributed" *)
    logic [NODE_W-1:0] cs_par [0:N_CAND-1];

    // ---- G3: THE POOL->SLOT INVERSION (depth>=3 correctness) --------------
    // cs_par[ord] stores the parent's POOL index (fr_node[par_slot]). F4's
    // banked `a` re-gather needs the OTHER coordinate of the same parent: its
    // SLOT (0..7) in the frontier that was live while this level was fed --
    // because the `a` banks are indexed by BODY PASS, and body pass b of level
    // d-1 is frontier slot b, not pool index b.
    //
    // WHY THE LOW BITS OF THE POOL INDEX ARE NOT THE SLOT (the actual G3 gap):
    //   level 0 pool index == slot          (sink below: fr_node[ord] <= ord)
    //   level L>=1 pool index = K_F + (L-1)*64 + ord, and ord = slot*8 + rank,
    //     so pool[2:0] == ord[2:0] == the parent's own CHILD RANK, which has
    //     nothing to do with its rank among the survivors.
    // So `parent_slot = surv_parent_o[2:0]` is exact ONLY while the parents are
    // level-0 nodes -- i.e. only for the commit of level 1, which is the
    // snapshot the level-2 body passes consume (F4's "exact through depth 2").
    // From the level-2 commit on (= the depth-3 body passes) it reads the wrong
    // bank. RED arm TREE_RED_G3_POOLLOW reinstates exactly that mapping.
    //
    // THE INVERSION IS DONE AT THE SOURCE, NOT BY SEARCHING. A reverse map
    // (latch the grandparent frontier, 8-way match pool index -> slot) is
    // possible but is a SECOND derivation of a quantity the sink already has in
    // hand: the very cycle it writes cs_par[ord] = fr_node[par_slot] it also
    // knows par_slot. Storing that co-derived slot in lockstep makes the pair
    // (cs_par, cs_pslot) an exact inverse by CONSTRUCTION -- fr_node[cs_pslot[o]]
    // == cs_par[o] holds for every ordinal, asserted below -- with no match
    // network, no extra frontier copy, and no way for the two to disagree.
    // Both are written from the SAME `par_slot` wire, so TREE_RED_TRANSPOSE
    // still perturbs them together and cannot be papered over here.
    // Cost: 64*3 = 192 b LUTRAM + 8*3 = 24 FF.
    (* ram_style = "distributed" *)
    logic [SLOT_W-1:0] cs_pslot [0:N_CAND-1];

    // ---- COMMITTED-SURVIVOR SNAPSHOT (bugfix 2026-07-22, child-index bug) ----
    // surv_token_o/surv_parent_o USED to be a combinational view into cs_tok/
    // cs_par indexed by (fr_node[g] - level_base). That view is only valid
    // during the ONE cycle the committed level is still the "current" level:
    //   * T_NEXT bumps level_idx, so level_base jumps to the NEXT level's base
    //     and `fr_node[g] >= level_base` goes FALSE for every g -> the guard
    //     falls back to index '0 and all 8 survivors alias cs_tok[0];
    //   * cs_tok/cs_par are TRANSIENT per-level staging, so the very next
    //     level's feed OVERWRITES the entries the survivors point at.
    // The depth-N consumer (top_fsm S_LVL_EMBG -> emb_gather tok_id) reads
    // these MANY body passes after level_done, i.e. squarely in the invalid
    // window. Latching the survivor token/parent at T_COMMIT makes the view
    // hold for the whole level, which is the contract the port name implies.
    // 8*(18+10) = 224 FF.
    logic [TOK_W-1:0]  sv_tok [0:K_F-1];
    logic [NODE_W-1:0] sv_par [0:K_F-1];
    logic [SLOT_W-1:0] sv_pslot [0:K_F-1];   // G3: the same parent, as a slot

    // ---- accumulated tree store: score + token + parent ----
    // depth and position are DERIVED from the pool index, so they cannot
    // drift out of sync with it -- storing them would create a second source
    // of truth for a quantity the index already implies.
    // COSTDRAFTTREE-ABI (G1): the per-node cumulative log2 score (SCORE_W=40, Q12.28)
    // is stored IN THE SAME WORD as {token,parent}, written from the SAME cum_sat wire
    // in the SAME cycle as the token/parent -- one address, one write-enable, so score
    // can never drift from the node it scores. SGLang's verify prunes the tree by
    // topk(draft_scores, verify_num<=64), impossible without a per-node score.
    // Word = {score[39:0], token[17:0], parent[9:0]} = 68 b x N_MAX(904) = 61,472 b
    //   -> 2 BRAM36 at D=15 (was 1); 136 nodes @ D=3 ship = 9,248 b, still 2 BRAM36
    //   (width 68 forces >=2 BRAM36 regardless of depth). Delta vs baseline: +1 BRAM36.
    //   NEVER URAM (URAM is the binding wall). Single SYNCHRONOUS read port -> BRAM.
    localparam int TN_W = SCORE_W + TOK_W + NODE_W;   // 40+18+10 = 68
    (* ram_style = "block" *)
    logic [TN_W-1:0] tn_mem [0:N_MAX-1];

    // =================================================================
    // derived depth / position from a pool index (single source of truth)
    // =================================================================
    // level 0 occupies pool 0..K_F-1 and is depth 1; level L>=1 occupies
    // K_F + (L-1)*64 .. +63 and is depth L+1. position = ctx_len + depth.
    function automatic [DEPTH_W-1:0] depth_of (input [NODE_W-1:0] p);
        depth_of = (p < K_F) ? DEPTH_W'(1)
                             : DEPTH_W'(2 + ((p - K_F) >> (2*SLOT_W)));  // >>6 == /64
    endfunction

    // level base pool index for the level currently being built
    wire [NODE_W-1:0] level_base   = lvl0 ? NODE_W'(0)
                                          : NODE_W'(K_F + (level_idx - 1) * N_CAND);
    wire [NODE_W-1:0] n_this_level = lvl0 ? NODE_W'(K_F) : NODE_W'(N_CAND);

    // =================================================================
    // log2 conversion pipeline + the score recurrence
    // =================================================================
    logic                     lg_valid_in;
    logic [31:0]              lg_prob;
    logic [CAND_W:0]          lg_ord;
    logic [TOK_W-1:0]         lg_tok;
    wire                      lg_valid_out;
    wire signed [SCORE_W-1:0] lg_out;

    tree_log2_fxp u_log2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(lg_valid_in), .prob_f32(lg_prob),
        .valid_out(lg_valid_out), .log2_q(lg_out)
    );

    // Side-channel delayed to match tree_log2_fxp's 2-cycle latency. The
    // ordinal and the token travel WITH their own datum -- they are not
    // recomputed downstream from a counter, which is exactly the
    // valid-before-data latency-skew class this project has been bitten by.
    // lg_ord/lg_tok are valid alongside lg_valid_in; two more stages line
    // them up with lg_valid_out.
    logic [CAND_W:0]  ord_d1, ord_q;
    logic [TOK_W-1:0] tok_d1, tok_q;
    logic [31:0]      prb_d1, lg_prob_q;   // raw fp32, co-timed (RED_SUM arm only)
    always_ff @(posedge clk) begin
        ord_d1 <= lg_ord;  ord_q     <= ord_d1;
        tok_d1 <= lg_tok;  tok_q     <= tok_d1;
        prb_d1 <= lg_prob; lg_prob_q <= prb_d1;
    end

    // =================================================================
    // RED ARMS -- deliberate defects, compiled in ONLY under a define.
    // Each perturbs exactly ONE thing and the gate is REQUIRED to fail.
    // A gate that cannot detect a wrong tree proves nothing, so these are
    // part of the deliverable, not debug scaffolding.
    //   TREE_RED_PER_PARENT  per-parent selection instead of global top-8
    //   TREE_RED_SUM         scores summed instead of multiplied
    //   TREE_RED_POS         position off by one
    //   TREE_RED_TRANSPOSE   parent-slot/child-rank layout transposed
    // =================================================================

    // parent slot of a candidate = ordinal >> 3 (parent-slot MAJOR)
`ifdef TREE_RED_TRANSPOSE
    // RED (d): read the parent slot from the LOW bits, i.e. treat the feed as
    // child-rank major / parent-slot minor. Every candidate is then scored
    // against the wrong parent's cumulative score.
    wire [SLOT_W-1:0] par_slot = ord_q[SLOT_W-1:0];
`else
    wire [SLOT_W-1:0] par_slot = ord_q[CAND_W-1 -: SLOT_W];
`endif

    // THE RECURRENCE: cumulative product of probabilities, carried as a sum
    // of log2 probabilities. Saturating, so a degenerate path can never wrap
    // around into a high score.
`ifdef TREE_RED_INDEP
    // RED (a): SCORE EACH LEVEL INDEPENDENTLY -- drop the cumulative product by
    // forcing the parent's running score to 1.0 (log2 0). Every child is then
    // ranked on its OWN conditional prob, not the path product, so the global
    // top-8 no longer prefers high-probability PATHS and the tree shape must
    // diverge from the golden. This is the "skip the cumulative product" arm.
    wire signed [SCORE_W-1:0] par_cum  = SCORE_ONE;
`else
    wire signed [SCORE_W-1:0] par_cum  = lvl0 ? SCORE_ONE : fr_score[par_slot];
`endif

`ifdef TREE_RED_SUM
    // RED (b): accumulate a SUM of linear probabilities instead of a product.
    // p is converted fp32 -> linear Q12.28 and added to the parent's running
    // total, so the metric becomes sum(p) along the path rather than prod(p).
    wire [7:0]  red_exp = lg_prob_q[30:23];
    wire [23:0] red_man = {1'b1, lg_prob_q[22:0]};
    wire signed [9:0] red_sh = $signed({2'b00, red_exp}) - $signed(10'sd127) + 10'sd5;
    wire signed [SCORE_W-1:0] red_lin =
            (red_exp == 8'd0)  ? '0 :
            (red_sh >= 0)      ? SCORE_W'({16'd0, red_man} <<   red_sh)
                               : SCORE_W'({16'd0, red_man} >> (-red_sh));
    wire signed [SCORE_W:0] cum_wide = {par_cum[SCORE_W-1], par_cum}
                                     + {red_lin[SCORE_W-1], red_lin};
`else
    wire signed [SCORE_W:0]   cum_wide = {par_cum[SCORE_W-1], par_cum}
                                       + {lg_out [SCORE_W-1], lg_out };
`endif
    localparam logic signed [SCORE_W-1:0] SCORE_MAX = {1'b0, {(SCORE_W-1){1'b1}}};
    wire signed [SCORE_W-1:0] cum_sat  =
            (cum_wide > $signed({1'b0, SCORE_MAX})) ? SCORE_MAX  :
            (cum_wide < $signed({1'b1, SCORE_NINF})) ? SCORE_NINF :
                                                       cum_wide[SCORE_W-1:0];

    // =================================================================
    // global top-8 of 64 -- tree_decode.sv REUSED VERBATIM
    // =================================================================
    // Reuse is exact, not approximate: tree_decode is a streaming K_F-deep
    // sorted-insertion heap with a SIGNED compare, and log2 scores are signed
    // (always <= 0). Re-parameterising V_MAX to N_CAND=64 and SCORE_W to 40
    // gives a global-over-64 selector with no new selection logic and no new
    // tie semantics to reason about.
    //
    // WHY ONE INSTANCE, NOT 8 + A MERGE NETWORK: the 64 candidates are
    // inherently SERIAL -- produced 8-at-a-time by 8 sequential body passes,
    // and the per-parent top-8-of-512 is already done upstream by the
    // existing tree_decode instance in the G12/TREE stage. Eight instances
    // would cost 8x the heap (8*8*40 = 2,560 FF vs 320 FF) plus an 8-way
    // merge network, to save cycles that do not exist: 64 insertion cycles is
    // noise beside a level's 8 full body passes (~10^4 cycles). Worse, a
    // per-parent structure is the exact SEMANTIC TRAP the spec warns about --
    // 8 local top-8s are NOT a global top-8 -- so it would invite the very
    // bug RED arm (a) exists to catch. One instance is smaller, simpler, and
    // structurally incapable of the wrong answer.
    logic                      sel_start;
    logic                      sel_vin;
    logic signed [SCORE_W-1:0] sel_score;
    logic [CAND_W-1:0]         sel_index;
    logic                      sel_last;
    wire  [K_F*SCORE_W-1:0]    sel_scores;
    wire  [K_F*CAND_W-1:0]     sel_indices;
    wire                       sel_done;

    tree_decode #(
        .K_F(K_F), .V_MAX(N_CAND), .IDX_W(CAND_W), .SCORE_W(SCORE_W)
    ) u_sel (
        .clk(clk), .rst_n(rst_n), .start(sel_start),
        .valid_in(sel_vin), .score_in(sel_score), .index_in(sel_index), .last_in(sel_last),
        .topk_scores(sel_scores), .topk_indices(sel_indices), .done(sel_done)
    );

    // =================================================================
    // FSM
    // =================================================================
    integer i;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            st <= T_IDLE; busy <= 1'b0; done <= 1'b0;
            level_done <= 1'b0; level_last <= 1'b0;
            ord_cnt <= '0; scored_cnt <= '0; drain_cnt <= '0;
            level_idx <= '0; n_nodes <= '0; levels_run_o <= '0;
            ctx_len_q <= '0; max_depth_q <= DEPTH_W'(1);
            level_idx_o <= '0; level_pos_o <= '0;
            sel_start <= 1'b0; sel_vin <= 1'b0; sel_last <= 1'b0;
            cand_req <= 1'b0; lg_valid_in <= 1'b0; lg_prob <= '0;
            lg_ord <= '0; lg_tok <= '0;
            for (i = 0; i < K_F; i = i + 1) begin
                fr_score[i] <= SCORE_ONE; fr_node[i] <= '0;
                sv_tok[i]   <= '0;        sv_par[i] <= '0;
                sv_pslot[i] <= '0;
            end
        end else begin
            done <= 1'b0; level_done <= 1'b0; level_last <= 1'b0;
            sel_start <= 1'b0; sel_vin <= 1'b0; sel_last <= 1'b0;
            lg_valid_in <= 1'b0;

            case (st)
                // ---------------------------------------------------------
                T_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy        <= 1'b1;
                        ctx_len_q   <= POS_W'(ctx_len_i);
                        // clamp max_depth into [1, MAX_DEPTH_HW]. 0 degrades to 1
                        // (a single level) so a never-written register behaves
                        // like today's depth-0 pass rather than emitting nothing.
                        max_depth_q <= (max_depth_i == 32'd0)           ? DEPTH_W'(1)
                                     : (max_depth_i > MAX_DEPTH_HW)     ? DEPTH_W'(MAX_DEPTH_HW)
                                                                        : DEPTH_W'(max_depth_i);
                        level_idx    <= '0;
                        n_nodes      <= '0;
                        levels_run_o <= '0;
                        for (i = 0; i < K_F; i = i + 1) fr_score[i] <= SCORE_ONE;
                        st <= T_LVL;
                    end
                end

                // ---------------------------------------------------------
                T_LVL: begin
                    ord_cnt    <= '0;
                    scored_cnt <= '0;
                    drain_cnt  <= '0;
                    sel_start  <= ~lvl0;      // level 0 bypasses the selector
                    cand_req   <= 1'b1;
                    st         <= T_FEED;
                end

                // ---------------------------------------------------------
                // EXPAND + SCORE. One candidate/cycle in, one score/cycle out
                // two cycles later. cand_req deasserts on the last candidate
                // so the producer cannot overrun the level boundary.
                T_FEED: begin
                    cand_req <= 1'b1;
                    if (cand_valid && (ord_cnt < n_this_level)) begin
                        lg_valid_in <= 1'b1;
                        lg_prob     <= cand_prob_f32;
                        lg_ord      <= ord_cnt;          // ordinal travels WITH the datum
                        lg_tok      <= cand_token;
                        ord_cnt     <= ord_cnt + 1'b1;
                        if (ord_cnt + 1'b1 == n_this_level) begin
                            cand_req <= 1'b0;
                            st       <= T_DRAIN;
                        end
                    end
                end

                // ---------------------------------------------------------
                // Flush the log2 pipe. Held until every accepted candidate has
                // been scored -- drain the FULL depth, never a fixed guess.
                T_DRAIN: begin
                    cand_req  <= 1'b0;
                    drain_cnt <= drain_cnt + 1'b1;
                    if (scored_cnt == n_this_level)
                        st <= lvl0 ? T_COMMIT : T_SELECT;
                end

                // ---------------------------------------------------------
                T_SELECT: begin
                    if (sel_done) st <= T_COMMIT;
                end

                // ---------------------------------------------------------
                // SELECT -> UPDATE FRONTIER. Survivors are read out of the
                // selector in rank order; the frontier is replaced wholesale,
                // so a survivor's slot j is its RANK and its parent slot is
                // (ordinal >> 3) -- which is where re-parenting shows up
                // (survivor rank j routinely != parent lane j, §1.4).
                // Level 0 does NOT touch the frontier here: it was already
                // written by the scored-candidate sink below, and all K_F
                // candidates survive in arrival order.
                T_COMMIT: begin
                    if (!lvl0) begin
`ifdef TREE_RED_PER_PARENT
                        // RED (a): PER-PARENT selection -- survivor j is the
                        // best child of parent lane j, so every parent keeps
                        // exactly one descendant. This is the semantic trap
                        // §1.4 warns about: it forbids the concentration and
                        // re-parenting the real rule produces, where several
                        // survivors may share one parent and survivor rank j
                        // routinely != parent lane j.
                        for (i = 0; i < K_F; i = i + 1) begin
                            fr_score[i] <= pp_score[i];
                            fr_node [i] <= level_base + NODE_W'(pp_ord[i]);
                        end
`else
                        for (i = 0; i < K_F; i = i + 1) begin
                            fr_score[i] <= $signed(sel_scores[i*SCORE_W +: SCORE_W]);
                            fr_node [i] <= level_base + NODE_W'(sel_indices[i*CAND_W +: CAND_W]);
                        end
`endif
                    end
                    // ---- survivor snapshot (see sv_tok/sv_par declaration) ----
                    // The ordinal of survivor rank i WITHIN this level's staging:
                    // level 0 keeps all K_F in arrival order (ordinal == rank);
                    // levels >=1 take it from the selector verdict. Read here,
                    // while cs_tok/cs_par still hold THIS level's candidates.
                    for (i = 0; i < K_F; i = i + 1) begin
                        // block-local temp: never storage, just the ordinal
                        automatic logic [CAND_W-1:0] sv_ord;
`ifdef TREE_RED_PER_PARENT
                        sv_ord = lvl0 ? CAND_W'(i) : pp_ord[i];
`else
                        sv_ord = lvl0 ? CAND_W'(i)
                                      : sel_indices[i*CAND_W +: CAND_W];
`endif
                        sv_tok[i] <= cs_tok[sv_ord];
                        sv_par[i] <= cs_par[sv_ord];
                        // G3: the parent's SLOT, snapshot in lockstep with the
                        // parent's POOL index out of the SAME staging ordinal.
                        // Level 0 has no parent -> slot 0, which is also what the
                        // level-1 body passes want (every level-1 child re-gathers
                        // a_0, the single root pass, from bank 0).
                        sv_pslot[i] <= lvl0 ? SLOT_W'(0) : cs_pslot[sv_ord];
`ifndef SYNTHESIS
                        // THE INVERSION IS EXACT, ASSERTED: at T_COMMIT fr_node[]
                        // still holds the frontier that was live while this level
                        // was fed (it is replaced by the NBAs above, in this same
                        // cycle), which is precisely the frontier cs_par was
                        // resolved against. So slot and pool index must agree.
                        if (rst_n && !lvl0 &&
                            fr_node[cs_pslot[sv_ord]] !== cs_par[sv_ord])
                            $error("[TREE_FSM G3] pool<->slot INCONSISTENT rank=%0d ord=%0d: fr_node[%0d]=%0d cs_par=%0d",
                                   i, sv_ord, cs_pslot[sv_ord],
                                   fr_node[cs_pslot[sv_ord]], cs_par[sv_ord]);
`endif
                    end
                    n_nodes      <= level_base + n_this_level;
                    levels_run_o <= level_idx + 1'b1;
                    level_idx_o  <= level_idx;
`ifdef TREE_RED_POS
                    // RED (c): position off by one.
                    level_pos_o  <= ctx_len_q + POS_W'(depth_of(level_base)) + POS_W'(1);
`else
                    level_pos_o  <= ctx_len_q + POS_W'(depth_of(level_base));
`endif
                    level_done   <= 1'b1;
                    // co-timed with level_done -- see `last_level` above
                    level_last   <= last_level;
                    st           <= T_NEXT;
                end

                // ---------------------------------------------------------
                // THE LOOP BOUND. max_depth_q is read here and nowhere else;
                // this comparison is the only exit from the level loop, so the
                // register cannot be trimmed without changing the FSM.
                T_NEXT: begin
                    if (!last_level) begin
                        level_idx <= level_idx + 1'b1;
                        st        <= T_LVL;
                    end else begin
                        st <= T_DONE;
                    end
                end

                // ---------------------------------------------------------
                T_DONE: begin
                    done <= 1'b1; busy <= 1'b0; st <= T_IDLE;
                end

                default: st <= T_IDLE;
            endcase

            // -------------------------------------------------------------
            // scored-candidate sink -- runs concurrently with T_FEED/T_DRAIN.
            // Writes the node store + the level staging arrays and, for levels
            // >= 1, streams into the global selector. Streaming order == pool
            // order, which is what makes the strict-'>' tie rule mean
            // "lower pool index wins".
            // Single always_ff, so these later assignments simply override the
            // defaults above -- no second driver on any of these registers.
            // -------------------------------------------------------------
            if (lg_valid_out) begin
`ifdef TREE_RED_PER_PARENT
                // per-parent running max, indexed by the candidate's own
                // parent lane. child_rank 0 seeds the lane (the feed is
                // parent-major so rank 0 always arrives first). RED arm (a).
                if (ord_q[SLOT_W-1:0] == '0 ||
                    cum_sat > pp_score[ord_q[CAND_W-1 -: SLOT_W]]) begin
                    pp_score[ord_q[CAND_W-1 -: SLOT_W]] <= cum_sat;
                    pp_ord  [ord_q[CAND_W-1 -: SLOT_W]] <= ord_q[CAND_W-1:0];
                end
`endif
                cs_tok[ord_q[CAND_W-1:0]] <= tok_q;
`ifdef TREE_RED_RTL_PARENT_SLOT
                // RED arm (b), RTL-SIDE (distinct define from the TB-side arm on purpose:
                // the TB arm overwrites got_par at the observation point, which
                // would MASK this one entirely if they shared a name).
                // Record the parent as the FRONTIER SLOT rather
                // than the frontier node, i.e. the `topk_cs_index // topk`
                // confusion. Structurally INVISIBLE at level 1 (the level-0
                // survivors ARE 0..7, so slot == node); it only shows from
                // level 2 on, which is why this arm must run >=3 levels.
                cs_par[ord_q[CAND_W-1:0]] <= lvl0 ? NODE_W'(0) : NODE_W'(par_slot);
                tn_mem[level_base + NODE_W'(ord_q)] <=
                        {cum_sat, tok_q, lvl0 ? NODE_W'(0) : NODE_W'(par_slot)};
`else
                cs_par[ord_q[CAND_W-1:0]] <= lvl0 ? NODE_W'(0) : fr_node[par_slot];
                tn_mem[level_base + NODE_W'(ord_q)] <=
                        {cum_sat, tok_q, lvl0 ? NODE_W'(0) : fr_node[par_slot]};
`endif
                // G3: the slot half of the SAME (pool,slot) pair, written from
                // the SAME par_slot wire in the SAME cycle as cs_par above --
                // one derivation, two coordinates, so they cannot drift.
                cs_pslot[ord_q[CAND_W-1:0]] <= lvl0 ? SLOT_W'(0) : par_slot;
                if (lvl0) begin
                    // level 0: all K_F survive in arrival order, so the sink
                    // writes the frontier directly and T_COMMIT leaves it alone.
                    fr_score[ord_q[SLOT_W-1:0]] <= cum_sat;
                    fr_node [ord_q[SLOT_W-1:0]] <= NODE_W'(ord_q);
                end else begin
                    sel_vin   <= 1'b1;
                    sel_score <= cum_sat;
                    sel_index <= CAND_W'(ord_q);
                    sel_last  <= (scored_cnt + 1'b1 == n_this_level);
                end
                scored_cnt <= scored_cnt + 1'b1;
            end
        end
    end

    // =================================================================
    // outputs
    // =================================================================
    // Survivor views. surv_node_o/surv_score_o come straight off the FF
    // frontier. surv_token_o/surv_parent_o come off the T_COMMIT SNAPSHOT
    // (sv_tok/sv_par) -- sampled out of the LUTRAM staging arrays at commit,
    // so they are stable for the WHOLE level rather than for the single
    // cycle the old level_base-relative view was valid (see the sv_tok
    // declaration for the bug that cost). The snapshot still sources the
    // parent from the STAGING array rather than recomputing it, so a
    // transposed feed still cannot be papered over here.
    genvar g;
    generate
        for (g = 0; g < K_F; g = g + 1) begin : g_surv
            assign surv_node_o  [g*NODE_W  +: NODE_W ] = fr_node[g];
            assign surv_score_o [g*SCORE_W +: SCORE_W] = fr_score[g];
`ifdef TREE_RED_SURV_VIEW
            // RED arm (d): the PRE-FIX combinational staging view. Correct only
            // in the T_COMMIT/T_NEXT window; once level_idx advances, level_base
            // outruns fr_node[] and every survivor collapses onto staging index
            // 0 -- the depth-N child-index bug (all 8 children gather the SAME
            // token). This arm exists to prove the gate can still see it.
            assign surv_token_o [g*TOK_W   +: TOK_W  ] =
                       cs_tok[fr_node[g] >= level_base ? (fr_node[g] - level_base) : '0];
            assign surv_parent_o[g*NODE_W  +: NODE_W ] =
                       cs_par[fr_node[g] >= level_base ? (fr_node[g] - level_base) : '0];
`else
            assign surv_token_o [g*TOK_W   +: TOK_W  ] = sv_tok[g];
            assign surv_parent_o[g*NODE_W  +: NODE_W ] = sv_par[g];
`endif
`ifdef TREE_RED_G3_POOLLOW
            // G3 RED ARM: the PRE-INVERSION mapping F4 shipped -- take the
            // parent SLOT from the low bits of the parent POOL index. Exact
            // through the level-1 commit (level-0 pool index == slot), WRONG
            // from the level-2 commit on (= the depth-3 body passes), where the
            // low bits are the parent's own child rank. The G3 gate REQUIRES
            // this arm to fail at level 2 and to AGREE at levels 0/1 -- which
            // is simultaneously the proof that depth<=2 behaviour is unchanged.
            assign surv_parent_slot_o[g*SLOT_W +: SLOT_W] = sv_par[g][SLOT_W-1:0];
`else
            assign surv_parent_slot_o[g*SLOT_W +: SLOT_W] = sv_pslot[g];
`endif
        end
    endgenerate

    // Accumulated-tree readback: synchronous (BRAM) read, with depth/position
    // derived from the registered index so all four outputs are co-timed.
    logic [NODE_W-1:0] tn_rd_idx_q;
    always_ff @(posedge clk) begin
        tn_rd_idx_q <= tn_rd_idx;
        {tn_rd_score, tn_rd_token, tn_rd_parent} <= tn_mem[tn_rd_idx];
    end
    assign tn_rd_depth = depth_of(tn_rd_idx_q);
    assign tn_rd_pos   = ctx_len_q + POS_W'(depth_of(tn_rd_idx_q));

    assign n_nodes_o = n_nodes;

`ifndef SYNTHESIS
    // Simulation guard: if max_depth ever loses its consumer again, the loop
    // bound becomes a constant and this stops agreeing. Deliberately NOT the
    // only protection -- the gate sweeps max_depth = 1..4 and checks
    // levels_run_o, which no amount of register-storage plumbing can fake.
    always_ff @(posedge clk) if (rst_n && done)
        if (levels_run_o !== max_depth_q)
            $error("[TREE_FSM] max_depth DORMANT: ran %0d levels, max_depth=%0d",
                   levels_run_o, max_depth_q);
`endif

endmodule

`default_nettype wire
