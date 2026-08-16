// =====================================================================
// tree_score_pkg.sv -- numeric contract for the depth-N tree score path
// =====================================================================
//
// THE DECISION (measured, not asserted): cumulative path scores are held in
// the LOG2 DOMAIN as signed fixed point Q12.28 (40 bits), NOT as a linear
// product of probabilities.
//
// WHY NOT LINEAR (the disqualifying fact):
//   score = prod(p) over up to 15 levels. Measured over all 720 captured
//   levels of ~/eagle4_artifacts/depthn_golden, the per-child conditional
//   probability reaches p_min = 2.13e-06, i.e. log2(p) = -18.84 per level.
//   At depth 15 that path product is 2^-282.6 = 8.49e-86, while the smallest
//   normal fp32 is 2^-126 = 1.18e-38. A linear representation FLUSHES TO ZERO
//   well before depth 15, and once two competing paths are both zero their
//   ORDER -- the only thing that must match -- is irrecoverably lost. No
//   mantissa width fixes this; it is an exponent-range failure.
//
// WHY LOG2 WORKS:
//   * products become sums   -> an ADDER, no multiplier, and the add is EXACT
//     in fixed point (the only error is the one-time log2 conversion).
//   * comparison is order-preserving: log2 is strictly monotonic, so a signed
//     integer compare on the accumulator reproduces the ranking of the
//     products exactly. This is what lets tree_decode.sv be reused verbatim.
//   * range is linear in depth, not exponential: 15 levels x -48 (clamp
//     floor) = -720, which sits inside Q12.28's +/-2048 with 2.8x headroom.
//
// FORMAT SIZING -- both knobs pinned by measurement over the captures:
//
//   integer bits (12, range +/-2048)
//     depth-15 worst observed path   15 x -18.84 = -282.6   (7.2x headroom)
//     depth-15 at the clamp floor    15 x -48    = -720     (2.8x headroom)
//
//   fraction bits (28, ulp = 3.73e-09)
//     the quantity that must not be destroyed is the RELATIVE gap between
//     the 8th and 9th ranked candidate, because that is what decides a
//     survivor. Measured over 432 depth>=1 levels:
//         min non-zero relative gap        2.3688e-04
//         => min gap in log2 units         3.4169e-04   <- the decision floor
//     one ulp is 91,700x finer than that floor.
//
//   log2 LUT (1024 entries + LINEAR INTERPOLATION)
//     the conversion, not the accumulator, is the dominant error term:
//         LUT 1024 + interp   max |err| = 1.75e-07  -> 1949x margin at d=1,
//                                                       65x at depth 15
//         LUT 1024, NO interp max |err| = 1.40e-03  ->  0.2x margin  UNSAFE
//         LUT  256 + interp   max |err| = 2.75e-06  ->  4.1x at depth 15, thin
//         LUT   64 + interp   max |err| = 4.34e-05  ->  0.3x at depth 15 UNSAFE
//
//     HONEST FINDING, recorded rather than buried: the no-interpolation
//     variant PASSES the empirical depth-1..4 gate (720/720 survivor sets)
//     even though its error is 4x LARGER than the decision floor. The
//     captures only reach depth 4, so the gate does not have the depth to
//     reject it -- the analysis does. Interpolation is kept on the strength
//     of the analysis, not the gate. Do not remove it because "the gate
//     still passes".
//
// TIE RULE (normative -- the golden cannot supply one, so we define one):
//   torch.topk's tie order is unspecified, so exact-tie lane assignment is
//   not derivable from the GPU. This design pins: candidates are streamed in
//   POOL ORDER (parent-slot major, child-rank minor) and the selector admits
//   on STRICT greater-than, so on an exact tie the EARLIER-arriving (lower
//   pool index) candidate keeps the higher rank. Deterministic, and it makes
//   the RTL reproducible run-to-run even where the GPU is not.
// =====================================================================

`ifndef TREE_SCORE_PKG_SV
`define TREE_SCORE_PKG_SV

package tree_score_pkg;

    // ---- tree shape (compile-time constant; see DEPTHN_GOLDEN_REFERENCE §1.7:
    //      CostDraftTree cost/ is empty, so width/depth never negotiate) ----
    localparam int K_F        = 8;                 // frontier width / children per node
    localparam int N_CAND     = K_F * K_F;         // 64 candidates scored per level
    localparam int MAX_DEPTH_HW = 15;              // hardware ceiling on max_depth
    localparam int N_MAX      = K_F + K_F*K_F*(MAX_DEPTH_HW-1); // 904 pool nodes @ D=15

    // ---- score format: signed Q12.28 ----
    localparam int SCORE_I    = 12;
    localparam int SCORE_F    = 28;
    localparam int SCORE_W    = SCORE_I + SCORE_F; // 40
    localparam logic signed [SCORE_W-1:0] SCORE_ONE  = '0;              // log2(1.0) = 0
    localparam logic signed [SCORE_W-1:0] SCORE_NINF = {1'b1,{(SCORE_W-1){1'b0}}};
    // clamp floor for log2(p): -48.0 in Q12.28. Observed min is -18.84, so this
    // is 29 log2-units below anything the 512-wide softmax has ever produced.
    localparam logic signed [SCORE_W-1:0] LOG2_FLOOR = -(48 * (1 << SCORE_F));

    // ---- index widths ----
    localparam int CAND_W     = $clog2(N_CAND);    // 6  : 0..63 within a level
    localparam int NODE_W     = $clog2(N_MAX+1);   // 10 : pool index 0..903
    localparam int SLOT_W     = $clog2(K_F);       // 3  : parent slot / child rank
    localparam int TOK_W      = 18;                // vocab id (128256 < 2^18)
    localparam int DEPTH_W    = $clog2(MAX_DEPTH_HW+1); // 4
    localparam int POS_W      = 24;                // position = ctx_len + depth

    // ---- one tree node, as stored (93 b/node budget in §4.4) ----
    localparam int NODE_BITS  = NODE_W + NODE_W + DEPTH_W + POS_W + TOK_W; // 10+10+4+24+18 = 66

endpackage

`endif
