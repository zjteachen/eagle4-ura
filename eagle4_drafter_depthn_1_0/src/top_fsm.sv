// =====================================================================
// top_fsm.sv
// EAGLE4 drafter forward — top-level stage orchestrator.
// =====================================================================
//
// Drafter forward = 8 GEMV stages + 2 non-GEMV (RoPE, SiLU) + tree decode.
// Per EAGLE4_DRAFTER_GEMM_FLOW.md the order is:
//
//   G1   — fc_in    (fp16 → 3072, fp16 IP, not 2-bit)
//   G4   — qkv      (3072 → 6144 → split to Q+K+V each 3072)
//   G5   — RoPE on Q/K + KV cache append at ctx_len
//   G6   — o_proj   (3072 → 3072)
//   G7g  — gate     (3072 → 8192)
//   G7u  — up       (3072 → 8192)
//   G7'  — SiLU+mul (silu(gate) * up, elementwise)
//   G9   — down     (8192 → 3072)
//   G11  — fc_out   (3072 → 3072, optional norm + linear)
//   G12  — lm head  (3072 → 128256, vocab)
//   TREE — tree-decode top-K_F selection
//
// All 2-bit GEMVs (G4, G6, G7g, G7u, G9, G12) share the same 128-lane
// PE array via the stage_sel signal; the array's weight bank is reindexed
// per stage.  G1 (G11) are tiny fp16 GEMVs, handled by a dedicated small unit.
//
// This file is the SCAFFOLDING — it defines states, transitions, and the
// per-stage start/done handshake.  Datapath wiring to the compute units
// lands in subsequent ticks.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module top_fsm #(
    // =====================================================================
    // G8 SPLICE (PREP, fork pdi6_s7c_g8 — reports/G8_S7C_SPLICE_PREP_20260528.md)
    //   G8_SPLICE=0 (DEFAULT) : the original 11-stage FSM, BIT-IDENTICAL to
    //     pdi6_s7c. S_TREE -> S_DONE on tree_done; g8_start tied 0; the top's
    //     output writer fires on tree_done with the tree top-K_F=8 frontier.
    //   G8_SPLICE=1 : insert an S_G8 (fused-gather-dot frontier rescore) phase
    //     AFTER the G12-fed tree vocab-walk completes and BEFORE S_DONE. The
    //     tree walk still runs (g8_rescore's internal G8.1 selector snoops the
    //     SAME G12 tile stream as tree_decode to build its top-K_C=512 candidate
    //     set). On tree_done the FSM enters S_G8, pulses g8_start once, and waits
    //     g8_done; the top then drives the output writer from g8's rescored
    //     top-8 (instead of the tree's cheap top-8).
    //   This strap keeps the DEFAULT path byte-for-byte the s7c FSM; the
    //   golden-PDI build flips it to 1 (turnkey).
    // =====================================================================
    parameter bit G8_SPLICE = 1'b0,

    // =====================================================================
    // DEPTH-N LEVEL LOOP (tree FSM stage 2)
    //   DEPTH_N=0 (DEFAULT) : BIT-IDENTICAL to the single-pass FSM above.
    //     `use_depth_loop` folds to a constant 0, every S_LVL_* state is
    //     unreachable and synthesis strips them, and every transition below
    //     reduces to exactly the edge it had before. This is the no-regress
    //     guarantee and it is STRUCTURAL, not a runtime promise.
    //   DEPTH_N=1 : one launch walks max_depth levels. Level 0 is the single
    //     existing body pass (it emits the K_F seed frontier); every level
    //     i>=1 runs K_F body passes, one per frontier branch j, each with its
    //     own KV lane-chain column. The 8 children of each branch are fed to
    //     tree_level_fsm, which selects the global top-8 of 64 per level.
    //
    //   Loop shape (why 1 pass at level 0, K_F passes after):
    //     level 0 : 1 body pass  ->  K_F=8  seeds     (selector bypassed)
    //     level i : K_F passes   ->  K_F*K_F=64 cands (global top-8)
    //   This matches the candidate counts tree_level_fsm expects (stage 1 §3)
    //   and the node budget C = topk + (N-1)*topk^2 it was gated against.
    // =====================================================================
    parameter bit DEPTH_N = 1'b0,
    parameter int K_F     = 8,

    // =====================================================================
    // GAP-4 RUNG C — child-embedding gather-wait (emb_gather top graft).
    //   EMBG_EN=0 (DEFAULT) : BIT-IDENTICAL to the DEPTH_N FSM above. The
    //     `use_embg` guard folds to 0, S_LVL_EMBG is unreachable dead state
    //     synthesis strips, emb_start is tied 0, and every depth-N re-entry
    //     edge (S_LVL_FEED/S_LVL_SEL -> S_G1) reduces to exactly its prior
    //     target. So EMBG_EN=0 (with DEPTH_N=0 OR DEPTH_N=1) is no-regress
    //     BY CONSTRUCTION, not by argument.
    //   EMBG_EN=1 (needs DEPTH_N=1) : before each depth>=1 CHILD body pass
    //     (branch j of level i>=1) the FSM detours through S_LVL_EMBG, pulses
    //     emb_start with that branch's survivor token id, waits emb_done (the
    //     on-chip embed_tokens gather into input_emb_buf), THEN launches G1.
    //     Level-0 (root) and every level-0 re-entry keep the host-loaded
    //     embedding untouched (those edges do NOT pass through S_LVL_EMBG).
    // =====================================================================
    parameter bit EMBG_EN = 1'b0,

    // =====================================================================
    // ACCFIX KVONLY (0x1125, PHASE6_FAST_PRIMING P3). KVONLY_EN=0 (DEFAULT):
    // kvonly_i is ignored, every edge below folds to its prior target —
    // BIT-IDENTICAL BY CONSTRUCTION. KVONLY_EN=1: while the runtime enable
    // kvonly_i (ACCFIX reg 0x15C bit3) is high, a launch runs
    // S_G1 -> S_G4 -> S_G5_ROPE(KV append) -> S_DONE, skipping the depth
    // loop and everything after G5 (G6..TREE/G8/out-walk). The KV row is
    // produced by the SAME datapath that produces it in-line, so a primed
    // row is bit-identical to a full launch's. No tree / OUT-DMA is
    // produced (the host already ignores prime-launch output).
    // =====================================================================
    parameter bit KVONLY_EN = 1'b0
) (
    input  wire  clk,
    input  wire  rst_n,

    // Top-level start (from control_regs.sv via AP_CTRL.start bit)
    input  wire  start,
    output reg   busy,
    output reg   done,

    // ---- DEPTH-N level loop (inert when DEPTH_N=0) ----
    // control reg 0x20. Was DORMANT (declared/decoded/read-back but with ZERO
    // datapath consumers, eagle4_drafter.sv:2338+2579); this port is what makes
    // it live. A register that reads back what was written is a STORAGE check,
    // not a behaviour check.
    input  wire [31:0]           max_depth,
    // ACCFIX KVONLY runtime enable (stable config reg; sampled per launch).
    input  wire                  kvonly_i,
    output reg                   lvl_start,      // pulse ONCE per tree (not per level)
    output reg  [2:0]             branch_j,       // frontier branch this body pass serves (K_F=8)
    output reg  [3:0]            level_i,        // level being built (0-based)
    output reg                   branch_feed,    // pulse: this branch's children are ready
    input  wire                  level_done,     // from tree_level_fsm
    // CO-TIMED with level_done: 1 = that level_done is the LAST of the tree.
    // `tree_all_done` (tree_level_fsm's `done`) is registered TWO states after
    // level_done, so sampling it on the level_done edge ALWAYS read 0 -- the
    // level loop ran one level past max_depth and then hung here. This port is
    // the co-timed flag; tree_all_done is kept and OR'd so the exit is taken on
    // whichever arrives (and so DEPTH_N=0 tie-off behaviour is unchanged).
    // Tied 0 at DEPTH_N=0 (eagle4_drafter g_depth_n else-branch).
    input  wire                  level_last,
    input  wire                  tree_all_done,  // from tree_level_fsm (done)
    output reg                   out_walk_start, // begin the node-major output walk
    input  wire                  out_walk_done,

    // ---- GAP-4 RUNG C: child-embedding gather handshake (inert at EMBG_EN=0) ----
    // emb_start pulses ONCE per depth>=1 child body pass (on entering S_LVL_EMBG);
    // emb_done is emb_gather's live-start-masked sticky done (done_r && !start),
    // so sampling it the same cycle emb_start fires can never see a stale run.
    output reg                   emb_start,
    input  wire                  emb_done,

    // Per-stage start/done handshake (to compute units, wired later)
    output reg   g1_start,   input  wire  g1_done,
    output reg   g4_start,   input  wire  g4_done,
    output reg   g5_start,   input  wire  g5_done,    // RoPE + KV append
    output reg   g6_start,   input  wire  g6_done,
    output reg   g7g_start,  input  wire  g7g_done,
    output reg   g7u_start,  input  wire  g7u_done,
    output reg   g7p_start,  input  wire  g7p_done,   // SiLU + mul
    output reg   g9_start,   input  wire  g9_done,
    output reg   g11_start,  input  wire  g11_done,
    output reg   glm_start,  input  wire  glm_done,   // FIX-G: g11_lmdown dense GEMV
    output reg   g12_start,  input  wire  g12_done,
    output reg   tree_start, input  wire  tree_done,
    // G8 SPLICE handshake (only meaningful when G8_SPLICE=1; g8_start stays 0
    // and S_G8 is unreachable when G8_SPLICE=0).
    output reg   g8_start,   input  wire  g8_done,

    // Stage selector for the shared GEMV array
    output reg [3:0] stage_sel
);

    typedef enum logic [4:0] {
        S_IDLE     = 5'd0,
        S_G1       = 5'd1,
        S_G4       = 5'd2,
        S_G5_ROPE  = 5'd3,
        S_G6       = 5'd4,
        S_G7G      = 5'd5,
        S_G7U      = 5'd6,
        S_G7_SILU  = 5'd7,
        S_G9       = 5'd8,
        S_G11      = 5'd9,
        S_G12      = 5'd10,
        S_TREE     = 5'd11,
        S_DONE     = 5'd12,
        S_GLM      = 5'd13,  // FIX-G: g11_lmdown (3072->256 low_rank) between G11 & G12
        S_G8       = 5'd14,  // G8 SPLICE: fused-gather-dot rescore (after TREE)
        // ---- DEPTH-N level loop (unreachable when DEPTH_N=0) ----
        S_LVL_ARM  = 5'd15,  // pulse lvl_start once, init level_i=0 / branch_j=0
        S_LVL_FEED = 5'd16,  // this branch's children are ready -> hand to tree_level_fsm
        S_LVL_SEL  = 5'd17,  // wait level_done (global top-8 of 64) -> next level or out
        S_LVL_OUT  = 5'd18,  // node-major walk of the accumulated tree to OUT-DMA
        S_LVL_EMBG = 5'd19   // RUNG C: gather this child branch's embedding before S_G1
    } state_t;

    state_t state, state_n;

    // =====================================================================
    // The depth-N guard. With DEPTH_N=0 this is a CONSTANT 0, so every
    // `use_depth_loop ? A : B` below folds to B -- the exact pre-existing
    // edge -- and S_LVL_* becomes unreachable dead state that synthesis
    // removes. That is why depth-0 is bit-identical by construction rather
    // than by argument.
    // =====================================================================
    wire use_depth_loop = DEPTH_N && (max_depth > 32'd1);

    // ACCFIX KVONLY: latched at launch so a mid-launch host write of 0x15C
    // can never truncate/extend an in-flight forward. Folds to constant 0
    // at KVONLY_EN=0 (edges below revert to their prior targets).
    reg kvonly_q;
    always_ff @(posedge clk) begin
        if (!rst_n)               kvonly_q <= 1'b0;
        else if (state == S_IDLE) kvonly_q <= kvonly_i;
    end
    wire kvonly_act = KVONLY_EN && kvonly_q;

    // =====================================================================
    // RUNG C gather-wait guard. EMBG_EN=0 -> constant 0, so every
    // `use_embg ? S_LVL_EMBG : S_G1` below folds to S_G1 (the pre-existing
    // edge) and S_LVL_EMBG is unreachable dead state. Requires the depth
    // loop (a child embedding only exists at level >= 1).
    // =====================================================================
    wire use_embg = EMBG_EN && use_depth_loop;

    // THE depth-N exit condition, sampled on the level_done edge. See the
    // level_last port comment: tree_all_done alone is two cycles too late.
    wire tree_last_w = level_last || tree_all_done;

    // Level 0 emits the K_F seeds from ONE body pass; levels >=1 need one body
    // pass per frontier branch. Derived, never stored -- a second copy of this
    // count would be a dual-source-of-truth for the loop bound.
    localparam int BJ_W  = $clog2(K_F);       // 3
    localparam int BJC_W = BJ_W + 1;          // 4, so K_F=8 is representable
    wire [BJC_W-1:0] n_branch_this_level =
        (level_i == 4'd0) ? BJC_W'(1) : BJC_W'(K_F);
    wire last_branch = ((BJC_W'(branch_j) + BJC_W'(1)) >= n_branch_this_level);

    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= state_n;
    end

    always_comb begin
        state_n = state;
        case (state)
            // DEPTH-N: arm the tree before the first body pass. Folds to the
            // original `S_IDLE -> S_G1` edge when DEPTH_N=0.
            // ACCFIX KVONLY: a KV-only launch never arms the depth loop.
            S_IDLE:    if (start)   state_n = (use_depth_loop && !kvonly_act)
                                              ? S_LVL_ARM : S_G1;
            S_LVL_ARM:              state_n = S_G1;
            S_G1:      if (g1_done)   state_n = S_G4;
            S_G4:      if (g4_done)   state_n = S_G5_ROPE;
            // ACCFIX KVONLY: the KV row is appended by G5; nothing after it
            // is needed on a prime launch.
            S_G5_ROPE: if (g5_done)   state_n = kvonly_act ? S_DONE : S_G6;
            S_G6:      if (g6_done)   state_n = S_G7G;
            S_G7G:     if (g7g_done)  state_n = S_G7U;
            S_G7U:     if (g7u_done)  state_n = S_G7_SILU;
            S_G7_SILU: if (g7p_done)  state_n = S_G9;
            S_G9:      if (g9_done)   state_n = S_G11;
            S_G11:     if (g11_done)  state_n = S_GLM;   // FIX-G: G11 norm -> lmdown
            S_GLM:     if (glm_done)  state_n = S_G12;   // FIX-G: lmdown -> G12
            S_G12:     if (g12_done)  state_n = S_TREE;
            // G8 SPLICE: with G8_SPLICE=1, TREE -> S_G8 -> DONE (rescore before
            // output). With G8_SPLICE=0, TREE -> DONE directly (s7c-identical).
            // DEPTH-N: the body's exit junction. Both edges fold back to their
            // original targets (S_G8 / S_DONE) when use_depth_loop=0.
            S_TREE:    if (tree_done) state_n = G8_SPLICE ? S_G8
                                               : (use_depth_loop ? S_LVL_FEED : S_DONE);
            S_G8:      if (g8_done)   state_n = use_depth_loop ? S_LVL_FEED : S_DONE;

            // ---- DEPTH-N level loop ----
            // This branch's K_F children have been handed over. Level 0 has a
            // single branch; levels >=1 have K_F. Re-entering S_G1 re-pulses
            // g1_start via the (state_n==X)&&(state!=X) idiom below, so a body
            // pass needs no extra re-arm logic.
            // RUNG C: a depth>=1 CHILD body pass detours through S_LVL_EMBG to
            // gather that branch's embedding first (branch_j has ALREADY been
            // incremented in this state's always_ff below). use_embg=0 folds
            // this back to the original `-> S_G1` edge.
            S_LVL_FEED:               state_n = last_branch ? S_LVL_SEL
                                              : (use_embg ? S_LVL_EMBG : S_G1);
            // Gather-wait: emb_done is start-masked, so no stale-done early exit.
            S_LVL_EMBG: if (emb_done)  state_n = S_G1;
            // Global top-8-of-64 for this level. tree_all_done is asserted by
            // tree_level_fsm when level_idx+1 == max_depth -- max_depth is
            // consumed THERE, and this state just follows its verdict, so there
            // is exactly one loop bound in the design.
            // RUNG C: the first branch of the NEXT level (branch_j reset to 0 in
            // the always_ff below) also gathers its embedding before G1.
            S_LVL_SEL: if (level_done) state_n = tree_last_w ? S_LVL_OUT
                                              : (use_embg ? S_LVL_EMBG : S_G1);
            S_LVL_OUT: if (out_walk_done) state_n = S_DONE;

            S_DONE:                   state_n = S_IDLE;
            default:                  state_n = S_IDLE;
        endcase
    end

    // ---- DEPTH-N level/branch counters + one-shot pulses ----
    // Single always_ff owns branch_j/level_i and every depth-N pulse: a pulse
    // register driven from two processes resolves to GND in synthesis (a
    // bug-class this project has already paid for).
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            branch_j <= '0; level_i <= '0;
            lvl_start <= 1'b0; branch_feed <= 1'b0; out_walk_start <= 1'b0;
        end else begin
            // default: all pulses low, so each is exactly 1 cycle
            lvl_start      <= 1'b0;
            branch_feed    <= 1'b0;
            out_walk_start <= 1'b0;

            // F4-LVL (2026-07-28): clear the depth-N counters on EVERY launch
            // edge, not only the S_LVL_ARM one. A max_depth=1 launch takes
            // S_IDLE->S_G1 and previously BYPASSED the only in-band clear, so
            // level_i stayed parked at its last tree value (e.g. 1 after any
            // md>=2 run) and lane_active = DEPTH_N && (level_i != 0) poisoned
            // every subsequent depth-0 forward (KV append to a tree-lane slot
            // + inflated s7a_ctx_len_eff) until a physical cold-AC (rst_n).
            // RED/GREEN: tb/tb_top_fsm_lvl_sticky.sv.
            if (state == S_IDLE && state_n != S_IDLE) begin
                branch_j  <= '0;
                level_i   <= '0;
            end
            if (state == S_IDLE && state_n == S_LVL_ARM) begin
                // ONCE per tree, not per level. tree_level_fsm owns the frontier
                // across levels; re-pulsing start would wipe it. This is the
                // single easiest thing to get wrong here, because every OTHER
                // *_start in this FSM is per-state.
                lvl_start <= 1'b1;
            end

            if (state == S_LVL_FEED) begin
                branch_feed <= 1'b1;
                if (!last_branch) branch_j <= branch_j + 1'b1;
            end

            if (state == S_LVL_SEL && level_done && !tree_last_w) begin
                level_i  <= level_i + 4'd1;
                branch_j <= '0;
            end

            if (state == S_LVL_SEL && level_done && tree_last_w)
                out_walk_start <= 1'b1;
        end
    end

    // Per-stage start pulses (one cycle when entering that state)
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            g1_start <= 0; g4_start <= 0; g5_start <= 0; g6_start <= 0;
            g7g_start <= 0; g7u_start <= 0; g7p_start <= 0; g9_start <= 0;
            g11_start <= 0; glm_start <= 0; g12_start <= 0; tree_start <= 0;
            g8_start  <= 0; emb_start <= 0;
        end else begin
            g1_start   <= (state_n == S_G1)      && (state != S_G1);
            g4_start   <= (state_n == S_G4)      && (state != S_G4);
            g5_start   <= (state_n == S_G5_ROPE) && (state != S_G5_ROPE);
            g6_start   <= (state_n == S_G6)      && (state != S_G6);
            g7g_start  <= (state_n == S_G7G)     && (state != S_G7G);
            g7u_start  <= (state_n == S_G7U)     && (state != S_G7U);
            g7p_start  <= (state_n == S_G7_SILU) && (state != S_G7_SILU);
            g9_start   <= (state_n == S_G9)      && (state != S_G9);
            g11_start  <= (state_n == S_G11)     && (state != S_G11);
            glm_start  <= (state_n == S_GLM)     && (state != S_GLM);
            g12_start  <= (state_n == S_G12)     && (state != S_G12);
            tree_start <= (state_n == S_TREE)    && (state != S_TREE);
            // G8 SPLICE: 1-cycle launch pulse on entering S_G8 (only fires when
            // G8_SPLICE=1, since S_G8 is unreachable otherwise).
            g8_start   <= (state_n == S_G8)      && (state != S_G8);
            // RUNG C: 1-cycle child-embedding gather launch on entering
            // S_LVL_EMBG (unreachable, so tied 0, when EMBG_EN=0).
            emb_start  <= (state_n == S_LVL_EMBG) && (state != S_LVL_EMBG);
        end
    end

    // Stage selector for the shared GEMV array (passed to weight bank to
    // pick the right slice for this stage)
    always_comb begin
        case (state)
            S_G4:  stage_sel = 4'd0;
            S_G6:  stage_sel = 4'd1;
            S_G7G: stage_sel = 4'd2;
            S_G7U: stage_sel = 4'd3;
            S_G9:  stage_sel = 4'd4;
            S_G12: stage_sel = 4'd5;
            default: stage_sel = 4'd0;
        endcase
    end

    // Top-level handshake
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 0; done <= 0;
        end else begin
            busy <= (state != S_IDLE) && (state != S_DONE);
            done <= (state == S_DONE);
        end
    end

endmodule

`default_nettype wire
