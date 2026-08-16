// =====================================================================
// g8_rescore.sv  —  STANDALONE SKELETON for the G8 fused-gather-dot rescore.
//
// Fork: verilog/forks/pdi6_g8  (NEW file; does NOT touch pdi6_s7c).
// Spec: SPEC_G8_RESCORE.md (this fork) + reports FIXH/FIXJ_G8_*_20260527.md.
//
// PURPOSE
//   A concrete, elaboratable starting point that shows the G8 architecture is
//   sound: four sub-blocks wired into one FSM, multi-driver-safe unpacked
//   arrays, a single read-only m_axi gather master, and a streaming dot.  It
//   is NOT the production-optimal datapath — it is the seam where the real
//   DSP58 fp16 pipe / wide-AXI gather drop in.  The TB (tb/tb_g8_rescore.sv)
//   drives it against ref/g8_ref.py's golden frontier.
//
// THE FOUR SUB-BLOCKS (FIX-J §1-4)
//   G8.1  candidate-select : top-K_C over the cheap G12 logits
//                            (reuses the tree_decode sorted-insertion pattern,
//                             widened K_F->K_C; here a K_C-deep heap).
//   G8.2  HBM gather        : ONE read-only m_axi master, bounded sequential
//                            512-row read (here K_C rows of H fp16), ARVALID
//                            asserted ONLY inside the gather FSM window
//                            (hang-safe, modeled on weight_dma_loader.sv).
//   G8.3  dot rescore       : fp16-multiply / fp32-accumulate streaming dot,
//                            one row at a time (stream-and-dot, 0 URAM).
//                            *** SYNTHESIZABLE (2026-05-27) *** the G8.3 dot is
//                            now the hand-built IEEE-754 RTL pipe g8_dot_fp16
//                            (g8_fp16_pkg fp16_mul -> widen -> fp32_add,
//                            FADD_PIPE=1 / II=2, 1 DSP58 / 0 URAM, 125 MHz-clean,
//                            reports/G8_FP16_DOT_PIPE_20260527.md). The previous
//                            SV `real` model is GONE.
//   G8.4  final top-K_F     : top-K_F=8 over the K_C rescored scores, then
//                            resolve through the candidate map -> vocab ids
//                            (reuses tree_decode #(.K_F=8) pattern).
//
// PARAMS default to the SYNTHETIC case (V_SYN=64,H_SYN=16,K_C=16) so the TB is
// fast; set to V=128256,H=3072,K_C=512 for the production instance.
//
// DIMENSION NOTE: the dot is over H (=3072 production), NOT 256.  lm_head_full
// is [V,H] fp16 (788 MB).  See SPEC §0 / ref/g8_ref.py for why "256" in the
// task prompt was a conflation of LM_RANK with H.
//
// MULTI-DRIVER SAFE: heap arrays + score/cand buffers are unpacked arrays each
// driven from exactly ONE always_ff; outputs concatenate combinationally.
//
// SYNTHESIZABILITY (2026-05-27 — the WHOLE G8 skeleton is now synthesizable):
//   1. The G8.3 dot is the synthesizable IEEE-754 pipe g8_dot_fp16 (II=2,
//      fp16-mul/fp32-acc, bit-exact to the old `real` model).
//   2. The G8.1/G8.4 sorted-insertion SELECTORS now compare candidate ordering
//      via the synthesizable g8_fp16_pkg::fp32_gt (native fp32 bit-pattern
//      ordering, the production form) — the SV `real` heaps (heap_real /
//      fheap_real, $bitstoshortreal) are RETIRED.  The heaps store the fp32
//      bit-pattern directly (heap_score / fheap_score); the -inf sentinel is
//      the fp32 bit-pattern 0xFF800000.  fp32_gt is strict-greater, so the
//      stable-insertion tie-break (first-arrived stays ahead) matches the old
//      `real` `>` exactly, and +0/-0 are treated equal (also matching `real`).
//   So the ENTIRE G8 datapath (selectors + gather + dot + packing) is
//   synthesizable RTL — ZERO `real` / `$bitstoshortreal` remain.  The skeleton
//   ELABORATES + simulates (xvlog/xelab/xsim 2025.1, ordered-8/8 PASS on both
//   the synthetic case and the full-dim REAL seed-42 case) and the dot is
//   P&R-ready (1 DSP58 / 0 URAM / +0.051 ns @125 MHz).
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

import g8_fp16_pkg::*;     // synthesizable fp16-mul/fp32-add (for the g8_dot_fp16 pipe)

(* keep_hierarchy = "yes" *)
module g8_rescore #(
    parameter int V_MAX    = 64,                 // vocab (prod 128256)
    parameter int H        = 16,                 // hidden / dot length (prod 3072)
    parameter int K_C      = 16,                 // candidate width (prod 512)
    parameter int K_F      = 8,                  // frontier width (prod 8)
    parameter int IDX_W    = $clog2(V_MAX),      // vocab-id width
    parameter int CIDX_W   = (K_C <= 1) ? 1 : $clog2(K_C), // candidate-pos width
    parameter int FP16_W   = 16,
    parameter int SCORE_W  = 32,                 // fp32 score bus (bit-pattern)
    parameter int AXI_ADDR_W = 64,
    parameter int AXI_DATA_W = 32,               // skeleton: 2 fp16 / 32-bit beat
                                                 // (prod: widen to 256/512-bit)
    parameter int AXI_ID_W   = 1,
    // G8_LANE_FIX (pdi24 G0): narrow-burst active-lane select by element parity
    // (see g8_rescore_p.sv — kept consistent here; this module is dead in prod).
    parameter bit G8_LANE_FIX = 1'b1,
    // ---- bounded sub-burst gather (G8.2 AXI-protocol correctness) ----
    //   A single AXI burst carries at most 256 beats (arlen is 8-bit:
    //   arlen=255 => 256 beats).  For H>256 (prod H=3072) the row MUST be
    //   gathered as ceil(H/MAX_BURST) bounded sub-bursts, NOT one AR with a
    //   truncated arlen.  MAX_BURST=256 is the AXI4 cap (1 beat = 1 fp16 in
    //   the 32-bit skeleton; prod packs more fp16/beat in a wider AXI_DATA_W,
    //   shrinking the beat count, but the sub-burst loop is identical).
    //   The old single-AR path (arlen=H-1, relied on a TB that streamed the
    //   full row regardless of the 8-bit-truncated arlen) is BUG for H>256;
    //   this is the protocol-honoring replacement.  Kept parameterizable so
    //   the synthetic K_C=16/H=16 case (H<=256) still issues exactly one AR.
    parameter int MAX_BURST  = 256,
    // ---- G8 GATHER WATCHDOG (PDI#7 silicon-hardening) ----
    //   A bounded free-running counter, active ONLY while the gather FSM is in
    //   S_GA_REQ / S_GA_DATA / S_GA_WAIT (the states where m_axi can wedge if
    //   lmfull_hbm_base points at an unmapped / mis-perturbed aperture — the
    //   exact failure that permanently wedged the flashed image). It resets on
    //   every accepted AXI handshake (forward progress) and at row finalize. If
    //   it reaches WD_LIMIT with no progress, the gather is ABORTED: ARVALID/
    //   RREADY drop, the in-flight candidate's rescored score is left at its
    //   current value (frontier PASSTHROUGH of the cheap candidate set), the
    //   FSM jumps straight to S_FINAL, and a sticky `g8_err` asserts so the
    //   host sees the abort at ctrl 0x78 bit3 — while `done` STILL asserts so
    //   ap_done is reached (no host-visible hang). DEFAULT 2^22 cycles
    //   (~33 ms @125 MHz) >> a healthy K_C=512 x H=3072 gather, so a healthy
    //   run never trips. Sized down by the TB for the unmapped-base test.
    parameter int unsigned WD_LIMIT = 32'd4_194_304,  // 2^22
    // ---- CHEAP-KEY ORDERING (2026-06-06 second ordering-hazard fix) ----
    //   The cheap-logit stream carries the G12 logit as a SIGNED INT32 (scaled-
    //   GEMV requant), NOT an fp32 bit-pattern. With CHEAP_KEY_INT32=1 (DEFAULT,
    //   the golden fix) the G8.1 candidate selector orders the cheap key with
    //   g8_fp16_pkg::cheap_gt (signed two's-complement >, bit-identical to
    //   tree_decode.sv:71) so the K_C set + ordering match the reference and
    //   tree_decode exactly. CHEAP_KEY_INT32=0 restores the legacy fp32_gt
    //   bit-pattern compare (WRONG for negative logits — kept only to reproduce
    //   the pre-fix behaviour for the regression repro). The G8.4 FINAL rescore
    //   heap is UNAFFECTED: it always uses fp32_gt because it compares the
    //   fp16-dot fp32 rescored scores, which ARE genuine IEEE-754 fp32.
    parameter bit CHEAP_KEY_INT32 = 1'b1
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,         // launch G8 (after G12)

    // ---- G8.1 input: streamed cheap G12 logits ----
    //   score_in carries the cheap logit as an fp32 bit-pattern; the selector
    //   compares them via the synthesizable fp32_gt (= the production native-
    //   fp32-bit-pattern ordering).
    //   `cheap_ready` (OPT1B) backpressures the stream so the rare min-rescan
    //   can run without dropping a logit.  In OPT1C it is tied high (the
    //   combinational heap consumes one/cycle).  A producer MUST hold
    //   (cheap_score,cheap_index,cheap_last) until cheap_valid && cheap_ready.
    input  wire                          cheap_valid,
    output logic                         cheap_ready,   // OPT1B backpressure
    input  wire [SCORE_W-1:0]            cheap_score,   // fp32 bit-pattern
    input  wire [IDX_W-1:0]              cheap_index,   // vocab id
    input  wire                          cheap_last,    // pulse on last logit (V-th)

    // ---- G8.3 second operand: to_logits_norm[H] fp16, preloaded by TB ----
    //   (prod: this comes from the FIX-I-corrected G11 final-norm output.)
    input  wire [FP16_W-1:0]             nrm_rd_data,   // fp16 word at nrm_rd_addr
    output logic [$clog2((H<2)?2:H)-1:0] nrm_rd_addr,   // skeleton reads nrm by addr

    // ---- G8.2 HBM gather m_axi READ master (AR + R only; NO AW/W/B) ----
    output logic                         m_axi_arvalid,
    input  wire                          m_axi_arready,
    output logic [AXI_ADDR_W-1:0]        m_axi_araddr,
    output logic [7:0]                   m_axi_arlen,
    output logic [2:0]                   m_axi_arsize,
    output logic [1:0]                   m_axi_arburst,
    output logic [AXI_ID_W-1:0]          m_axi_arid,
    input  wire                          m_axi_rvalid,
    output logic                         m_axi_rready,
    input  wire [AXI_DATA_W-1:0]         m_axi_rdata,
    input  wire                          m_axi_rlast,
    input  wire [AXI_ID_W-1:0]           m_axi_rid,
    input  wire [1:0]                    m_axi_rresp,   // DEBUG-V2: RRESP observe (read-only)
    input  wire  [AXI_ADDR_W-1:0]        lmfull_hbm_base, // byte base of lm_head_full

    // ---- output: G8-resolved frontier (after done) ----
    output wire  [K_F*IDX_W-1:0]         frontier_ids,    // top-K_F vocab ids (order-strict)
    output wire  [K_F*SCORE_W-1:0]       frontier_scores, // their rescored fp32 scores
    output logic                         done,
    // ---- G8 GATHER WATCHDOG sticky abort flag (PDI#7) ----
    //   Latches 1 if the gather watchdog tripped during this run; cleared on
    //   the next `start`. Routed to ctrl 0x78 bit3 in the top.
    output logic                         g8_err,

    // ---- DEBUG-BLOCK-V2 read-only progress taps (PDI#8, ctrl 0x94/0x98/0x9C) ----
    //   PURELY ADDITIVE: combinational mirrors of internal state, no datapath
    //   feedback. Surfaced to top scope so dbg_observe packs them into the new
    //   debug registers. (memory feedback_no_accuracy_tradeoff: read-only taps.)
    output wire  [3:0]                   dbg_sel_substate,  // {1'b0, state[2:0]} top-FSM phase
    output wire  [9:0]                   dbg_cand_filled,   // # candidate slots filled (0..K_C)
    output wire  [9:0]                   dbg_rows_gathered, // candidate row index reached (0..K_C)
    output wire                          dbg_seen_last,     // sticky cheap_last captured
    output wire                          dbg_in_gather,     // G8 active (not idle/done)
    output wire  [AXI_ADDR_W-1:0]        dbg_last_araddr,   // last AR byte addr issued
    output wire                          dbg_rresp_err,     // sticky RRESP != OKAY

    // ---- PROBE V3 (out3 candidate-set dump) read-only port ----
    //   Combinational read of the G8.1 candidate set for the host out3 dump.
    //   dbg_cand_rdaddr selects a candidate slot 0..K_C-1; the module returns its
    //   vocab id and cheap (pre-rescore) fp32 score bit-pattern. PURELY ADDITIVE.
    input  wire  [CIDX_W-1:0]            dbg_cand_rdaddr,
    output wire  [IDX_W-1:0]             dbg_cand_id,
    output wire  [SCORE_W-1:0]           dbg_cand_cheap_score
);

    // =================================================================
    // shared buffers (each written by exactly ONE always_ff -> driver-safe)
    // =================================================================
    // G8.1 outputs / candidate map:
    logic [IDX_W-1:0]   cand_idx   [0:K_C-1];   // vocab id of each candidate (G8.1 -> G8.4 resolve)
    // G8.3 output: rescored fp32 score per candidate position:
    logic [SCORE_W-1:0] rescored   [0:K_C-1];

    localparam int AR_W = $clog2((H<2)?2:H);     // nrm/row element address width

    // =================================================================
    // G8.1 — candidate-select.  TWO interchangeable implementations:
    //
    //   OPT1B (default, the PRODUCTION fix, SPEC §118): streaming threshold-
    //     select.  A 512-entry candidate set (sel_score/sel_idx, BRAM-mappable)
    //     + a single running-minimum register (the THRESHOLD to beat) + a
    //     sequential min-rescan.  Per accepted logit it does exactly ONE
    //     fp32_gt vs the threshold (NOT K_C compares), and when an entry is
    //     replaced it re-derives the new minimum by a registered one-slot-per-
    //     cycle scan (NO combinational K_C ripple).  Backpressures the input
    //     (cheap_ready) during the rare rescan.  This removes the 176-logic-
    //     level Opt-1c ripple that fails 125 MHz (-36.7 ns, §8) at the cost of
    //     +1 BRAM (the 512-entry set) and a single threshold reg.
    //     SET-EXACT: it keeps the EXACT top-K_C by cheap score (same set the
    //     Opt-1c sorted heap keeps), only the in-array ORDER differs — and G8.1
    //     only needs the SET (G8.3 rescore + G8.4 top-K_F re-order the final 8),
    //     so the resolved frontier is identical (see report §9).
    //
    //   OPT1C (define G8_OPT1C): the original K_C-deep combinational sorted-
    //     insertion heap (tree_decode pattern widened to K_C).  Correctness-
    //     simplest, fmax-worst (the 176-LL ripple).  Kept for A/B reference.
    //
    // Both store the cheap logit as its native fp32 bit-pattern and order via
    // the synthesizable g8_fp16_pkg::fp32_gt (NOT SV `real`).  Strict fp32_gt
    // => stable: an equal score does NOT displace an incumbent (first-arrived
    // stays), matching the retired `real` `>` semantics.
    // =================================================================
    logic               g8_1_done;

    // fp32 bit-pattern for -inf — the empty-set sentinel for the G8.4 FINAL
    // rescore heap (every finite fp32 score beats it via fp32_gt; replaces the
    // old SV `real` NEG_INF). KEEP for the final heap.
    localparam logic [SCORE_W-1:0] FP32_NEG_INF = 32'hFF80_0000;

    // CHEAP-KEY empty-set sentinel: under the INT32 signed-compare path the
    // smallest possible key is INT32_MIN (0x80000000), NOT the fp32 -inf
    // bit-pattern (0xFF800000 = -8388608 as a signed int, which a more-negative
    // real logit would beat -> sentinel slot would never be replaced). Selecting
    // the correct sentinel by CHEAP_KEY_INT32 keeps "every finite key strictly
    // beats an empty slot" true on the integer path AND the legacy fp32 path.
    localparam logic [SCORE_W-1:0] CHEAP_NEG_INF =
        CHEAP_KEY_INT32 ? 32'h8000_0000 : 32'hFF80_0000;

`ifdef G8_OPT1C
    // ---------- OPT1C: combinational K_C-deep sorted-insertion heap ----------
    logic [SCORE_W-1:0] heap_score [0:K_C-1];   // fp32 bit-pattern per heap slot
    logic [IDX_W-1:0]   heap_idx   [0:K_C-1];

    // insertion position (first slot whose stored score < incoming).
    logic [K_C-1:0] cs_gt, cs_pos;
    logic           cs_accept;
    always_comb begin
        for (int k = 0; k < K_C; k = k + 1)
            // cheap key is signed INT32 -> cheap_gt (CHEAP_KEY_INT32=1 golden);
            // fp32_gt only if explicitly reverting to the legacy bit-pattern path.
            cs_gt[k] = CHEAP_KEY_INT32 ? cheap_gt(cheap_score, heap_score[k])
                                       : fp32_gt(cheap_score, heap_score[k]);
        cs_pos    = '0;
        cs_accept = 1'b0;
        for (int k = 0; k < K_C; k = k + 1)
            if (cs_gt[k] && !cs_accept) begin cs_pos[k] = 1'b1; cs_accept = 1'b1; end
    end
`else
    // ---------- OPT1B: streaming threshold-select (production fix) -----------
    // The candidate SET storage (BRAM-mappable: K_C x SCORE_W + K_C x IDX_W).
    logic [SCORE_W-1:0] sel_score [0:K_C-1];   // fp32 bit-pattern per slot
    logic [IDX_W-1:0]   sel_idx   [0:K_C-1];
    // running minimum among the filled slots = the THRESHOLD an incoming logit
    // must strictly beat (fp32_gt) to enter the set once full.
    logic [SCORE_W-1:0] thr_score;             // min score currently in the set
    logic [CIDX_W-1:0]  thr_pos;               // slot holding that minimum
    logic [CIDX_W:0]    fill;                   // # slots filled (0..K_C)
    // one-slot-per-cycle min rescan
    logic [CIDX_W:0]    scan_ptr;               // 0..K_C during a rescan
    logic [SCORE_W-1:0] scan_min;               // running min during the scan
    logic [CIDX_W-1:0]  scan_min_pos;
    // G8.1 sub-FSM
    typedef enum logic [1:0] { SEL_FILL, SEL_SCAN, SEL_THRESH } sel_state_e;
    sel_state_e sel_state;
`endif

    // PDI#7.1 STICKY seen_last (G8_WEDGE_RCA fix b — THE FIX):
    //   On REAL inputs a high-magnitude logit can land in the LAST few hundred
    //   vocab ids and trigger a candidate replacement -> SEL_THRESH enters
    //   SEL_SCAN (a K_C-cycle min-rescan).  The `cheap_last` pulse only fires
    //   for ONE cycle (on the V-th logit); if it coincides with / is dropped
    //   while the selector is mid-SEL_SCAN, the old RTL never reached
    //   S_SEL_DONE -> gather never started -> top FSM parked S_G8 forever.
    //   `seen_last` LATCHES the cheap_last pulse the instant it is observed in
    //   ANY S_SELECT sub-state (FILL/THRESH/SCAN); SEL_SCAN's exit then branches
    //   to S_SEL_DONE when seen_last instead of returning to SEL_THRESH.  This
    //   makes last-logit capture independent of which sub-state cheap_last hits.
    //   Cleared on g8_start re-arm (S_IDLE start) and at reset.
    logic               seen_last;

    // =================================================================
    // top FSM
    // =================================================================
    typedef enum logic [3:0] {
        S_IDLE      = 4'd0,
        S_SELECT    = 4'd1,   // G8.1 stream cheap logits into heap
        S_SEL_DONE  = 4'd2,   // latch candidate map
        S_GA_REQ    = 4'd3,   // G8.2 issue AR for current candidate row
        S_GA_DATA   = 4'd4,   // G8.2 drain R beats, feed the II=2 dot pipe
        S_GA_WAIT   = 4'd9,   // wait for the dot's out_valid (pipe flush)
        S_GA_NEXT   = 4'd5,   // advance candidate / finalize its rescored score
        S_FINAL     = 4'd6,   // G8.4 top-K_F over K_C rescored
        S_RESOLVE   = 4'd7,   // resolve top-K_F positions -> vocab ids
        S_DONE      = 4'd8
    } state_e;
    state_e state;

    // -----------------------------------------------------------------
    // cheap_ready (G8.1 input backpressure)
    //   OPT1C: tied high (the combinational heap consumes one logit/cycle).
    //   OPT1B: high only while the selector can accept this cycle = in S_SELECT
    //          and the sub-FSM is in SEL_FILL or SEL_THRESH (NOT SEL_SCAN, where
    //          the min-rescan holds the stream).  Low in every other top state.
    // -----------------------------------------------------------------
    always_comb begin
`ifdef G8_OPT1C
        cheap_ready = 1'b1;
`else
        cheap_ready = (state == S_SELECT) &&
                      (sel_state == SEL_FILL || sel_state == SEL_THRESH);
`endif
    end

    // gather/dot working state
    logic [CIDX_W:0]   cand_ptr;        // which candidate row (0..K_C)
    logic [AR_W:0]     elem_ptr;        // element within the row (0..H), spans sub-bursts
    logic [AXI_ADDR_W-1:0] row_base;    // HBM byte base of current candidate row
    // bounded sub-burst gather bookkeeping (G8.2 AXI-protocol-correct):
    //   elem_ptr is the GLOBAL element index within the row (0..H), carried
    //   across sub-bursts so the dot accumulates the full H without re-seeding.
    //   burst_beats = beats remaining IN THE CURRENT sub-burst (1..MAX_BURST);
    //   decremented per accepted R beat.  When it hits 0 with elem_ptr<H, the
    //   FSM returns to S_GA_REQ for the next sub-burst (re-AR), else finalizes.
    localparam int BURST_CW = $clog2(MAX_BURST+1);
    logic [BURST_CW-1:0] burst_beats;   // beats left in the current sub-burst

    // ---- G8 GATHER WATCHDOG state (PDI#7 silicon-hardening) ----
    //   wd_cnt counts cycles with NO gather progress while in the gather states.
    //   Any accepted AR (arready) or accepted R beat (rvalid&rready) kicks it to
    //   0. On reaching WD_LIMIT it sets wd_abort for one cycle (the FSM consumes
    //   it to bail to S_FINAL) and latches the sticky g8_err.
    logic [31:0] wd_cnt;
    logic        wd_abort;        // 1-cycle "trip" pulse consumed by the FSM
    // PDI#7.1 WATCHDOG COVERAGE-ALL-STATES (G8_WEDGE_RCA fix c):
    //   The original `in_gather` window covered ONLY the AXI gather states
    //   (S_GA_REQ/S_GA_DATA/S_GA_WAIT). The PDI#7 silicon wedge parked in
    //   S_SELECT/SEL_SCAN (a dropped cheap_last during the late min-rescan) ->
    //   the gather never started, so the gather-only watchdog never armed and
    //   the top FSM held S_G8 forever. Widen the window to "G8 active & not
    //   done" (any state that is NOT S_IDLE and NOT S_DONE) so a stall in
    //   S_SELECT / SEL_SCAN / S_SEL_DONE / S_GA_NEXT / S_FINAL / S_RESOLVE also
    //   trips wd_abort -> g8_err -> 0x78 bit3 and forces `done` (no host hang).
    wire         in_gather = (state != S_IDLE) && (state != S_DONE);
    // any forward progress this cycle (handshake accepted)
    wire         gather_progress = (m_axi_arvalid && m_axi_arready) ||
                                   (m_axi_rvalid  && m_axi_rready);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wd_cnt   <= 32'd0;
            wd_abort <= 1'b0;
            g8_err   <= 1'b0;
        end else begin
            wd_abort <= 1'b0;                 // default: 1-cycle strobe
            if (start) begin
                wd_cnt <= 32'd0;
                g8_err <= 1'b0;               // re-arm each run
            end else if (!in_gather) begin
                wd_cnt <= 32'd0;              // not gathering -> hold cleared
            end else if (gather_progress) begin
                wd_cnt <= 32'd0;              // progress -> kick the dog
            end else if (!g8_err) begin
                if (wd_cnt >= WD_LIMIT[31:0]) begin
                    wd_abort <= 1'b1;         // trip (consumed by the FSM below)
                    g8_err   <= 1'b1;         // sticky abort flag
                end else begin
                    wd_cnt <= wd_cnt + 32'd1;
                end
            end
        end
    end

    // ---------------------------------------------------------------
    // DEBUG-BLOCK-V2 progress taps (PDI#8): sticky last-AR addr + RRESP-error.
    //   ONE always_ff, cleared on `start`. Read-only mirrors below feed the top
    //   dbg_observe; no datapath feedback (memory feedback_no_accuracy_tradeoff).
    // ---------------------------------------------------------------
    logic [AXI_ADDR_W-1:0] dbg_araddr_sticky;
    logic                  dbg_rresp_sticky;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            dbg_araddr_sticky <= '0;
            dbg_rresp_sticky  <= 1'b0;
        end else if (start) begin
            dbg_araddr_sticky <= '0;
            dbg_rresp_sticky  <= 1'b0;        // re-arm each run
        end else begin
            if (m_axi_arvalid && m_axi_arready)
                dbg_araddr_sticky <= m_axi_araddr;            // last AR issued
            if (m_axi_rvalid && m_axi_rready && (m_axi_rresp != 2'b00))
                dbg_rresp_sticky <= 1'b1;                     // sticky RRESP!=OKAY
        end
    end

    // Combinational progress mirrors (multi-driver-safe: read-only assigns).
    assign dbg_sel_substate  = {1'b0, state[2:0]};
    assign dbg_rows_gathered = 10'(cand_ptr);
    assign dbg_seen_last     = seen_last;
    assign dbg_in_gather     = in_gather;
    assign dbg_last_araddr   = dbg_araddr_sticky;
    assign dbg_rresp_err     = dbg_rresp_sticky;
`ifdef G8_OPT1C
    assign dbg_cand_filled   = 10'd0;   // OPT1C heap has no running fill counter
`else
    assign dbg_cand_filled   = 10'(fill);
`endif

    // ---- PROBE V3 out3 candidate-set dump read port ----
    // cand_idx[] (resolved vocab id) is module-scope in both G8.1 arms; rescored[]
    // carries the per-candidate fp32 score (pre-rescore during gather, final after
    // S_FINAL). out3 thus dumps {id, score} for each of the K_C candidates.
    assign dbg_cand_id          = cand_idx[dbg_cand_rdaddr];
    assign dbg_cand_cheap_score = rescored[dbg_cand_rdaddr];

    // ---------------------------------------------------------------
    // G8.3 dot pipe (SYNTHESIZABLE — replaces the old SV `real` accumulate).
    //   IEEE-754 fp16-mul / fp32-acc, FADD_PIPE=1 / II=2 (closes 125 MHz OOC).
    //   The FSM presents one (a,b) every OTHER cycle (II=2) by pacing the AXI
    //   R-channel handshake (m_axi_rready toggles), so a fresh stage-A read of
    //   `acc` always lands the previous element's stage-B write — bit-exact to
    //   the old `real_round_fp16(...)` left-to-right model.
    // ---------------------------------------------------------------
    logic        dot_clear;             // pulse with the FIRST beat of a row
    logic        dot_in_valid;          // present (a,b) this cycle
    logic        dot_in_last;           // pulse with the LAST beat of a row
    logic [15:0] dot_a_fp16;            // gathered lm_head_full row element
    logic [15:0] dot_b_fp16;            // to_logits_norm element
    logic        dot_out_valid;         // pulses when score_fp32 is final
    logic [31:0] dot_score_fp32;        // final fp32 row sum (= rescored[k])

    g8_dot_fp16 #(.FADD_PIPE(1'b1)) u_g8_dot (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear      (dot_clear),
        .in_valid   (dot_in_valid),
        .in_last    (dot_in_last),
        .a_fp16     (dot_a_fp16),
        .b_fp16     (dot_b_fp16),
        .out_valid  (dot_out_valid),
        .score_fp32 (dot_score_fp32)
    );

    // II=2 pacing: accept one AXI R beat, then skip one cycle so the dot's
    // 2-cycle accumulate recurrence settles before the next element arrives.
    logic dot_skip;     // 1 = this cycle is the "skip" half of the II=2 beat

    // G8.4 final top-K_F heap over the K_C rescored scores (SYNTHESIZABLE:
    // fp32 bit-pattern + fp32_gt compare, NOT SV `real`).
    logic [SCORE_W-1:0] fheap_score [0:K_F-1];  // fp32 bit-pattern per slot
    logic [CIDX_W-1:0] fheap_pos  [0:K_F-1];   // candidate POSITION (0..K_C-1)
    logic [K_F-1:0]   f_gt, f_pos; logic f_accept;
    logic [CIDX_W:0]  fin_ptr;                  // scan ptr over K_C
    logic [SCORE_W-1:0] fin_in_score;           // current rescored score (fp32 bits)
    always_comb fin_in_score = rescored[fin_ptr[CIDX_W-1:0]];
    always_comb begin
        for (int k = 0; k < K_F; k = k + 1)
            f_gt[k] = fp32_gt(fin_in_score, fheap_score[k]);
        f_pos = '0; f_accept = 1'b0;
        for (int k = 0; k < K_F; k = k + 1)
            if (f_gt[k] && !f_accept) begin f_pos[k] = 1'b1; f_accept = 1'b1; end
    end

    // resolved frontier output regs
    logic [IDX_W-1:0]   fr_id    [0:K_F-1];
    logic [SCORE_W-1:0] fr_score [0:K_F-1];
    logic [CIDX_W:0]    res_ptr;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            done          <= 1'b0;
            g8_1_done     <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_rready  <= 1'b0;
            m_axi_araddr  <= '0;
            m_axi_arlen   <= '0;
            m_axi_arsize  <= 3'b001;     // 2 bytes (skeleton 16-bit-ish; see note)
            m_axi_arburst <= 2'b01;      // INCR
            m_axi_arid    <= '0;
            nrm_rd_addr   <= '0;
            cand_ptr      <= '0;
            elem_ptr      <= '0;
            row_base      <= '0;
            burst_beats   <= '0;
            dot_clear     <= 1'b0;
            dot_in_valid  <= 1'b0;
            dot_in_last   <= 1'b0;
            dot_a_fp16    <= 16'd0;
            dot_b_fp16    <= 16'd0;
            dot_skip      <= 1'b0;
            fin_ptr       <= '0;
            res_ptr       <= '0;
`ifdef G8_OPT1C
            for (int k = 0; k < K_C; k = k + 1) begin
                heap_score[k] <= CHEAP_NEG_INF; heap_idx[k] <= '0;
                cand_idx[k]   <= '0;            rescored[k] <= 32'h0;
            end
`else
            for (int k = 0; k < K_C; k = k + 1) begin
                sel_score[k] <= CHEAP_NEG_INF; sel_idx[k] <= '0;
                cand_idx[k]  <= '0;            rescored[k] <= 32'h0;
            end
            thr_score    <= CHEAP_NEG_INF; thr_pos <= '0; fill <= '0;
            scan_ptr     <= '0; scan_min <= CHEAP_NEG_INF; scan_min_pos <= '0;
            sel_state    <= SEL_FILL;
`endif
            seen_last    <= 1'b0;     // PDI#7.1 sticky last-logit capture
            for (int k = 0; k < K_F; k = k + 1) begin
                fheap_score[k] <= FP32_NEG_INF; fheap_pos[k] <= '0;
                fr_id[k] <= '0; fr_score[k] <= 32'h0;
            end
        end else begin
            done <= 1'b0;
            // dot pulse defaults (single-cycle strobes; the gather state re-asserts)
            dot_clear    <= 1'b0;
            dot_in_valid <= 1'b0;
            dot_in_last  <= 1'b0;

            // -----------------------------------------------------------
            // G8 GATHER WATCHDOG abort (PDI#7): a wedged HBM gather (bad
            // lmfull_hbm_base -> AR/R never handshakes) tripped wd_abort. Tear
            // down the AXI master and bail to S_FINAL so the run still COMPLETES
            // (done asserts) with a frontier PASSTHROUGH (top-K_F over whatever
            // rescored[] holds so far -> the cheap candidate set when the gather
            // never produced a single score). g8_err (sticky) is already latched
            // above. This branch WINS over the per-state case (placed before it).
            // -----------------------------------------------------------
            if (wd_abort) begin
                m_axi_arvalid <= 1'b0;
                m_axi_rready  <= 1'b0;
                dot_in_valid  <= 1'b0;
                dot_in_last   <= 1'b0;
                dot_clear     <= 1'b0;
                fin_ptr       <= '0;
                for (int k = 0; k < K_F; k = k + 1) begin
                    fheap_score[k] <= FP32_NEG_INF; fheap_pos[k] <= '0;
                end
                state <= S_FINAL;
            end else
            case (state)
            // ---------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    g8_1_done <= 1'b0;
                    seen_last <= 1'b0;        // PDI#7.1 re-arm sticky last capture
`ifdef G8_OPT1C
                    for (int k = 0; k < K_C; k = k + 1) begin
                        heap_score[k] <= CHEAP_NEG_INF; heap_idx[k] <= '0;
                    end
`else
                    for (int k = 0; k < K_C; k = k + 1) begin
                        sel_score[k] <= CHEAP_NEG_INF; sel_idx[k] <= '0;
                    end
                    thr_score <= CHEAP_NEG_INF; thr_pos <= '0; fill <= '0;
                    scan_ptr  <= '0; scan_min <= CHEAP_NEG_INF; scan_min_pos <= '0;
                    sel_state <= SEL_FILL;
`endif
                    state <= S_SELECT;
                end
            end

`ifdef G8_OPT1C
            // --------- OPT1C G8.1: combinational sorted-insertion heap ----
            S_SELECT: begin
                if (cheap_valid && cs_accept) begin
                    for (int k = K_C-1; k >= 0; k = k - 1) begin
                        if (cs_pos[k]) begin
                            heap_score[k] <= cheap_score;
                            heap_idx[k]   <= cheap_index;
                        end else if (k > 0) begin
                            logic any_above; any_above = 1'b0;
                            for (int j = 0; j < k; j = j + 1)
                                if (cs_pos[j]) any_above = 1'b1;
                            if (any_above) begin
                                heap_score[k] <= heap_score[k-1];
                                heap_idx[k]   <= heap_idx[k-1];
                            end
                        end
                    end
                end
                if (cheap_last) state <= S_SEL_DONE;
            end

            // --------- latch candidate map, set up first gather --------
            S_SEL_DONE: begin
                for (int k = 0; k < K_C; k = k + 1)
                    cand_idx[k] <= heap_idx[k];
                cand_ptr <= '0;
                state    <= S_GA_REQ;
            end
`else
            // --------- OPT1B G8.1: streaming threshold-select -------------
            //   sel_state == SEL_FILL  : accept every logit until K_C filled;
            //       when the K_C-th lands, drop into SEL_SCAN to seed the
            //       threshold (= min of the K_C entries).  Input is accepted
            //       (cheap_ready) every cycle in this phase.
            //   sel_state == SEL_SCAN  : registered one-slot/cycle min scan
            //       over sel_score[0..K_C-1] -> thr_score/thr_pos.  Input is
            //       BACKPRESSURED (cheap_ready=0) for the duration; the scan is
            //       K_C cycles (only entered after a fill or a replacement).
            //   sel_state == SEL_THRESH: steady state — ONE fp32_gt vs
            //       thr_score per accepted logit.  On a beat that beats the
            //       threshold, overwrite slot thr_pos and re-enter SEL_SCAN to
            //       re-derive the new minimum.  On a non-beat, just consume.
            //   cheap_last advances to S_SEL_DONE once accepted.
            S_SELECT: begin
                // PDI#7.1 (G8_WEDGE_RCA fix b): latch the last-logit pulse the
                // instant it is observed, in ANY sub-state (FILL/THRESH/SCAN).
                // This is the sticky capture the old per-sub-state `cheap_last`
                // checks missed when the pulse coincided with SEL_SCAN.
                if (cheap_valid && cheap_last) seen_last <= 1'b1;
                // see the always_comb below for cheap_ready; here is the
                // datapath that fires when (cheap_valid && cheap_ready).
                unique case (sel_state)
                SEL_FILL: begin
                    if (cheap_valid) begin   // ready is high in FILL
                        sel_score[fill[CIDX_W-1:0]] <= cheap_score;
                        sel_idx  [fill[CIDX_W-1:0]] <= cheap_index;
                        if (fill == (K_C-1)) begin
                            // K_C-th element just landed -> set is full; scan
                            // for the initial minimum threshold.  (Include this
                            // just-written entry in the seed below.)
                            fill         <= fill + 1'b1;
                            scan_ptr     <= '0;
                            scan_min     <= cheap_score;             // slot fill
                            scan_min_pos <= fill[CIDX_W-1:0];
                            sel_state    <= SEL_SCAN;
                            if (cheap_last) state <= S_SEL_DONE;     // tiny V<=K_C
                        end else begin
                            fill <= fill + 1'b1;
                            if (cheap_last) begin
                                // fewer than K_C logits total (synthetic guard):
                                // unfilled slots stay FP32_NEG_INF, threshold
                                // unused -> go finalize.
                                state <= S_SEL_DONE;
                            end
                        end
                    end
                end
                SEL_SCAN: begin
                    // registered min scan: one slot per cycle, NO ripple.
                    if (scan_ptr == K_C[CIDX_W:0]) begin
                        thr_score <= scan_min;
                        thr_pos   <= scan_min_pos;
                        // PDI#7.1 (G8_WEDGE_RCA fix b): if the last logit was
                        // already seen (latched in ANY sub-state above), the
                        // stream is done — finalize the candidate set instead of
                        // returning to SEL_THRESH (where the dropped cheap_last
                        // would have parked the selector forever). The set is
                        // exact at this point (this scan just re-derived the
                        // threshold; no further logits can enter).
                        if (seen_last) state     <= S_SEL_DONE;
                        else           sel_state <= SEL_THRESH;
                    end else begin
                        if (CHEAP_KEY_INT32
                                ? cheap_gt(scan_min, sel_score[scan_ptr[CIDX_W-1:0]])
                                : fp32_gt (scan_min, sel_score[scan_ptr[CIDX_W-1:0]])) begin
                            // current scan_min is STRICTLY greater than this slot
                            // -> this slot is smaller -> it becomes the new min.
                            // (strict: ties keep the earlier-scanned slot, so an
                            // equal-min replacement targets the LOWEST index — a
                            // benign, set-preserving tie choice.)
                            scan_min     <= sel_score[scan_ptr[CIDX_W-1:0]];
                            scan_min_pos <= scan_ptr[CIDX_W-1:0];
                        end
                        scan_ptr <= scan_ptr + 1'b1;
                    end
                end
                SEL_THRESH: begin
                    if (cheap_valid) begin   // ready is high in THRESH
                        if (CHEAP_KEY_INT32 ? cheap_gt(cheap_score, thr_score)
                                            : fp32_gt(cheap_score, thr_score)) begin
                            // beats the current minimum -> replace it, then
                            // rescan for the new minimum threshold.
                            sel_score[thr_pos] <= cheap_score;
                            sel_idx  [thr_pos] <= cheap_index;
                            scan_ptr     <= '0;
                            scan_min     <= cheap_score;     // seed w/ replacement
                            scan_min_pos <= thr_pos;
                            sel_state    <= SEL_SCAN;
                        end
                        // non-beat: drop the logit (not in the top-K_C).
                        if (cheap_last) state <= S_SEL_DONE;
                    end
                end
                default: sel_state <= SEL_FILL;
                endcase
            end

            // --------- latch candidate map, set up first gather --------
            S_SEL_DONE: begin
                for (int k = 0; k < K_C; k = k + 1)
                    cand_idx[k] <= sel_idx[k];
                cand_ptr <= '0;
                state    <= S_GA_REQ;
            end
`endif

            // ========== G8.2 + G8.3: per-candidate gather + stream-dot =====
            // For candidate row p: row byte base = lmfull_hbm_base + cand_idx[p]*H*2.
            // The row of H fp16 is gathered as ceil(H/MAX_BURST) BOUNDED sub-bursts
            // (AXI4 caps a burst at 256 beats; arlen is 8-bit).  Each pass through
            // S_GA_REQ issues ONE sub-burst AR covering elements [elem_ptr ..
            // elem_ptr+this_burst-1]; the dot keeps accumulating across sub-bursts
            // (acc seeded only at elem_ptr==0, in_last only at elem_ptr==H-1).
            // The ARVALID driver is active ONLY in S_GA_REQ/S_GA_DATA -> hang-safe.
            S_GA_REQ: begin
                if (cand_ptr == K_C[CIDX_W:0]) begin
                    // all candidates rescored
                    fin_ptr <= '0;
                    for (int k = 0; k < K_F; k = k + 1) begin
                        fheap_score[k] <= FP32_NEG_INF; fheap_pos[k] <= '0;
                    end
                    state <= S_FINAL;
                end else if (!m_axi_arvalid) begin
                    // remaining elements in this row = H - elem_ptr; clamp the
                    // sub-burst to MAX_BURST (AXI4 256-beat cap).
                    automatic int unsigned rem  = H - int'(elem_ptr);
                    automatic int unsigned blen = (rem > MAX_BURST) ? MAX_BURST : rem;
                    automatic logic [AXI_ADDR_W-1:0] rb =
                        lmfull_hbm_base +
                        (AXI_ADDR_W'(cand_idx[cand_ptr[CIDX_W-1:0]]) * (H*2));
                    if (elem_ptr == '0) begin
                        // first sub-burst of a NEW row: latch row base, reset the
                        // II=2 pacing.  nrm_rd_addr stays at 0 (set here);
                        // elem_ptr is already 0 (reset at row finalize / start).
                        row_base    <= rb;
                        nrm_rd_addr <= '0;
                        dot_skip    <= 1'b0;     // first beat is a "consume" cycle
                    end
                    // sub-burst AR: address advances by the elements already read.
                    m_axi_araddr  <= rb + (AXI_ADDR_W'(elem_ptr) * 2);
                    m_axi_arlen   <= 8'(blen - 1);    // <= 255 (bounded sub-burst)
                    m_axi_arsize  <= 3'b001;          // 2 bytes/beat (fp16)
                    m_axi_arburst <= 2'b01;           // INCR
                    m_axi_arid    <= '0;
                    m_axi_arvalid <= 1'b1;
                    burst_beats   <= BURST_CW'(blen); // beats to consume this sub-burst
                end else if (m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready  <= 1'b1;
                    state         <= S_GA_DATA;
                end
            end

            // ---- G8.3 stream-dot, II=2 paced (synthesizable g8_dot_fp16) ----
            //   The dot's fp32_add is split across 2 registers (FADD_PIPE=1):
            //   present (a,b) every OTHER cycle so each stage-A read of `acc`
            //   sees the previous element's stage-B write -> bit-exact to the
            //   old `real` left-to-right model.  Pacing = toggle m_axi_rready:
            //     dot_skip=0 : assert rready, accept one beat, feed the dot,
            //                  advance elem/nrm, then set dot_skip=1.
            //     dot_skip=1 : deassert rready (slave holds the next beat),
            //                  clear dot_skip -> next consume cycle.
            S_GA_DATA: begin
`ifdef G8_DBG
                if (cand_ptr == '0)
                    $display("[dbg t=%0t] DATA skip=%b elem=%0d rvalid=%b rready=%b rdata=%h nrm=%h rlast=%b",
                             $time, dot_skip, elem_ptr, m_axi_rvalid, m_axi_rready,
                             m_axi_rdata[15:0], nrm_rd_data, m_axi_rlast);
`endif
                if (!dot_skip) begin
                    m_axi_rready <= 1'b1;
                    if (m_axi_rvalid && m_axi_rready) begin
                        automatic logic row_last   = (elem_ptr == (H-1)); // global last elem
                        automatic logic burst_last = (burst_beats == BURST_CW'(1)); // last beat of sub-burst
                        // feed this element into the dot pipe.  dot_clear seeds acc
                        // on the very first element of the row ONLY (elem_ptr==0),
                        // so it persists across sub-bursts.  dot_in_last fires on
                        // the GLOBAL last element (elem_ptr==H-1), NOT on a per-
                        // sub-burst rlast.
                        dot_in_valid <= 1'b1;
                        dot_clear    <= (elem_ptr == '0);
                        dot_in_last  <= row_last;
                        // G8_LANE_FIX (pdi24 G0, kept CONSISTENT with g8_rescore_p —
                        // this serial module is a dead code path in production):
                        // narrow-burst active lane = addr[1] == elem_ptr[0].
                        dot_a_fp16   <= (G8_LANE_FIX && elem_ptr[0] && (AXI_DATA_W >= 32))
                                        ? m_axi_rdata[((AXI_DATA_W>=32)?31:15) -: 16]
                                        : m_axi_rdata[15:0];        // gathered lm row elem
                        dot_b_fp16   <= nrm_rd_data;                // to_logits_norm[elem]
                        // advance global element + nrm read pointer for the NEXT beat
                        if (!row_last) begin
                            elem_ptr    <= elem_ptr + 1'b1;
                            nrm_rd_addr <= nrm_rd_addr + 1'b1;
                        end
                        burst_beats <= burst_beats - 1'b1;
                        if (row_last) begin
                            // whole row fed; stop accepting, wait for out_valid.
                            // (row_last implies burst_last for the final sub-burst.)
                            m_axi_rready <= 1'b0;
                            dot_skip     <= 1'b0;
                            elem_ptr     <= '0;          // reset for the next row
                            state        <= S_GA_WAIT;
                        end else if (burst_last) begin
                            // sub-burst exhausted but row not done -> re-AR for the
                            // next bounded sub-burst.  Drop rready; S_GA_REQ
                            // (elem_ptr>0 branch) issues the next AR.
                            m_axi_rready <= 1'b0;
                            dot_skip     <= 1'b0;        // first beat of next burst = consume
                            state        <= S_GA_REQ;
                        end else begin
                            // II=2: deassert rready for exactly ONE cycle.  The
                            // slave HOLDS the next beat (backpressure); we re-arm
                            // in the skip cycle so the consume lands 2 cycles
                            // after this one -> a fresh (a,b) every OTHER cycle.
                            m_axi_rready <= 1'b0;
                            dot_skip     <= 1'b1;
                        end
                    end
                end else begin
                    // skip half of the II=2 beat: re-arm rready NOW so the next
                    // cycle (skip=0) actually consumes (rready is registered).
                    m_axi_rready <= 1'b1;
                    dot_skip     <= 1'b0;
                end
            end

            // ---- wait for the dot pipe to flush the final fp32 score ----
            S_GA_WAIT: begin
                m_axi_rready <= 1'b0;
`ifdef G8_DBG
                $display("[dbg t=%0t] WAIT cand=%0d elem=%0d out_valid=%b score=%h",
                         $time, cand_ptr, elem_ptr, dot_out_valid, dot_score_fp32);
`endif
                if (dot_out_valid) begin
                    rescored[cand_ptr[CIDX_W-1:0]] <= dot_score_fp32;
                    state <= S_GA_NEXT;
                end
            end

            S_GA_NEXT: begin
                // advance to the next candidate row (score already latched)
                cand_ptr <= cand_ptr + 1'b1;
                state    <= S_GA_REQ;
            end

            // ========== G8.4: top-K_F over the K_C rescored scores =========
            S_FINAL: begin
                if (fin_ptr != K_C[CIDX_W:0]) begin
                    if (f_accept) begin
                        for (int k = K_F-1; k >= 0; k = k - 1) begin
                            if (f_pos[k]) begin
                                fheap_score[k] <= fin_in_score;
                                fheap_pos[k]   <= fin_ptr[CIDX_W-1:0];
                            end else if (k > 0) begin
                                logic any_above; any_above = 1'b0;
                                for (int j = 0; j < k; j = j + 1)
                                    if (f_pos[j]) any_above = 1'b1;
                                if (any_above) begin
                                    fheap_score[k] <= fheap_score[k-1];
                                    fheap_pos[k]   <= fheap_pos[k-1];
                                end
                            end
                        end
                    end
                    fin_ptr <= fin_ptr + 1'b1;
                end else begin
                    res_ptr <= '0;
                    state   <= S_RESOLVE;
                end
            end

            // ========== resolve top-K_F positions -> vocab ids =============
            S_RESOLVE: begin
                if (res_ptr != K_F[CIDX_W:0]) begin
                    fr_id[res_ptr[CIDX_W-1:0]]    <= cand_idx[fheap_pos[res_ptr[CIDX_W-1:0]]];
                    fr_score[res_ptr[CIDX_W-1:0]] <= rescored[fheap_pos[res_ptr[CIDX_W-1:0]]];
                    res_ptr <= res_ptr + 1'b1;
                end else begin
                    done  <= 1'b1;
                    state <= S_DONE;
                end
            end

            S_DONE: state <= S_IDLE;
            default: state <= S_IDLE;
            endcase
        end
    end

    // combinational pack of unpacked frontier arrays
    genvar gk;
    generate
        for (gk = 0; gk < K_F; gk = gk + 1) begin : g_pack
            assign frontier_ids   [gk*IDX_W   +: IDX_W]   = fr_id[gk];
            assign frontier_scores[gk*SCORE_W +: SCORE_W] = fr_score[gk];
        end
    endgenerate

endmodule

`default_nettype wire
