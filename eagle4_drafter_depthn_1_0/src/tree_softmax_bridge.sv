// =====================================================================
// tree_softmax_bridge.sv  -- TREE FSM STAGE 3: the scoring bridge
// =====================================================================
//
// Closes the one structural gap stage 2 left open: `cand_prob_f32` had no
// producer. This module IS the producer. It turns the datapath's per-parent
// G12/G8 rescore LOGITS into the conditional softmax probability p(child|parent)
// that tree_level_fsm consumes, so the tree loop is finally FED.
//
// ---------------------------------------------------------------------
// WHAT IT COMPUTES  (per parent body pass, over the K_C=512 rescore shortlist)
// ---------------------------------------------------------------------
// Mirrors exactly the reference model / SGLang math
// (DEPTHN_GOLDEN_REFERENCE.md §1.3, eagle_worker.py:737, spec_utils.py:480):
//
//     probs = softmax(fp32(G8))                over K_C = 512   (per parent)
//     p_k   = probs[frontier_k]                for the 8 selected children
//
// The per-parent softmax DENOMINATOR is the load-bearing quantity: children of
// DIFFERENT parents are then compared in one GLOBAL top-8, so each parent's
// probabilities must be consistently normalized or the cross-parent ranking
// diverges from the golden. The normalization proof is in TREE_FSM_STAGE3.md §1.
//
// ---------------------------------------------------------------------
// WHY THE LOG2 DOMAIN, AND WHY NOT THE Q4.12 softmax_frontier_q
// ---------------------------------------------------------------------
// stage 1 (tree_score_pkg.sv) pinned a 3.4169e-04 log2 DECISION FLOOR and built
// a 1024-entry+interp log2 (tree_log2_fxp, err 1.75e-07) to stay 1949x below it.
// The parked softmax_frontier_q.sv uses a 32-entry PWL exp + Q4.12 output
// (ULP 2.44e-04). Its A6 micro-gate (depthn_producers) shows it COLLAPSES
// distinct child probs into artificial ties -- e.g. root frontier ranks 2&3
// both round to Q4.12 178 where the fp32 golden is 179 vs 176, ranks 5&6 both
// to 108 where golden is 120 vs 97, max |err| 31 LSB = 0.76%. That is ~2200x
// the decision floor and would silently reshape the tree at the 8/9 boundary.
// So this bridge computes the softmax to the SAME accuracy stage 1 demanded:
//
//     l2_i   = (l_i - max) * log2(e)                          (base-2 logits)
//     Z      = sum_i 2^(l2_i)                     ACCURATE exp2 (1024+interp)
//     log2 p_k = l2_k - log2(Z)                    fixed-point, err ~1e-7
//     prob_k = 2^(log2 p_k)  assembled as IEEE-754 fp32
//
// Feeding an fp32 prob (not a raw log2) keeps tree_level_fsm and ALL its RED
// arms byte-for-byte unchanged: the FSM's own tree_log2_fxp re-takes the log2.
// The prob->fp32->log2 round trip adds only the fp32 mantissa quantization
// (~8.6e-08) on top of two LUTs, ~2.7e-07 total, still 1260x below the floor.
//
// ---------------------------------------------------------------------
// PRIMITIVE REUSE (URAM is at 99.95% -- ZERO new URAM; FF/LUT only)
// ---------------------------------------------------------------------
//   * tree_pow2_lut.svh  : NEW constant LUT, the symmetric partner of the
//     EXISTING tree_log2_lut.svh (same 1024+interp shape, same gen-script
//     pattern, same [Synth 8-3848]-immune localparam-array form).
//   * tree_log2_lut.svh  : REUSED verbatim for log2(Z).
//   * two-pass 512 sweep + fp32->fixed decode: structural reuse of the proven
//     softmax_frontier_q FSM skeleton, but accurate exp2 in place of the
//     32-entry PWL and fp32 (not Q4.12) out.
//
// Latency: MAX pass K_C + SUM pass K_C + log2Z (few) + K_F assemble cycles.
// Combinational LUT reads, single-owner registered accumulators (no
// comb-read stale-by-1 hazard). xsim/CPU only -- NO synthesis run here.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_softmax_bridge
    import tree_score_pkg::*;
#(
    parameter int K_C    = 512,                 // rescore shortlist width
    parameter int CIDX_W = (K_C<=1)?1:$clog2(K_C)
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start,            // 1-cyc pulse per parent

    // ---- BRIDGE_KC (ACCFIX bit2, 2026-07-29) ------------------------------
    // kc_cfg = the runtime K1 shortlist register (0x150; 0 = compile K_C).
    // kc_en  = ACCFIX bit2. When 1: (a) BOTH sweeps and the rescored[] read
    // are bounded at the launch-sampled kcq (kills the stale-slot-ingest
    // hazard: rescored[kcq..511] hold PRIOR-BRANCH scores that corrupted the
    // max and the softmax denominator Z at kc<512 depth-N, PHASE3 §b), and
    // (b) PASS A (S_MAX) is SKIPPED: the max over rescored[0..kcq-1] is
    // exactly the G8 top-1 = frontier_f32[31:0], which feed_seq latches
    // BEFORE pulsing start — same fp32 word through the same decode, so the
    // seeded max_q is bit-identical to the swept one. kc_en=0 => kcq=K_C and
    // the S_MAX pass runs => BYTE-IDENTICAL legacy behaviour.
    input  wire [15:0]                kc_cfg,
    input  wire                       kc_en,

    // ---- 512-candidate read port (drives g8_cand_rdaddr; data comb next cyc) ----
    output reg  [CIDX_W-1:0]          cand_rdaddr,
    output reg                        cand_rd_active,   // 1 while the bridge owns the port
    // SEAM-FF lookahead: the NEXT read address (combinational), one cycle ahead of
    // cand_rdaddr, for g8's FF-staged registered rescored[] read (bridge_score_ff).
    // A 1-cycle sync read fed this address delivers rescored[addr] on the very
    // cycle the max/sum pass consumes it -> the async cross-SLR read (which ties
    // to a constant in the netlist) is removed with ZERO timing/value change.
    output logic [CIDX_W-1:0]         cand_rdaddr_nxt,
    input  wire [31:0]                cand_logit_f32,   // fp32 logit @ cand_rdaddr

    // ---- the K_F selected child logits + their vocab ids (top-8 of the 512,
    //      already picked upstream by the existing tree_decode) ----
    input  wire [K_F*32-1:0]          frontier_f32,     // {f7..f0}, fp32 each
    input  wire [K_F*TOK_W-1:0]       frontier_tok,     // {t7..t0}, TOK_W each

    // ---- result: 8 conditional probs as IEEE-754 fp32, in frontier order ----
    output reg                        done,             // 1-cyc pulse, block valid
    output reg  [K_F*32-1:0]          prob_f32_o,       // {p7..p0} fp32
    output reg  [K_F*TOK_W-1:0]       tok_o             // {t7..t0} co-travelled
);

    `include "tree_log2_lut.svh"
    `include "tree_pow2_lut.svh"

    // log2(e) in Q0.28 : round(1.4426950408889634 * 2^28) = 387270501
    localparam logic [29:0] LOG2E_Q28 = 30'd387270501;
    localparam int          NCLAMP    = 40;     // 2^-40 ~ 9e-13, below any Z term
    localparam logic signed [SCORE_W-1:0] Q_ONE = 40'sd1 <<< SCORE_F; // 1.0 in Q12.28

    // -----------------------------------------------------------------
    // fp32 (IEEE-754) -> signed Q12.28 (SCORE_W = 40).  Round half up on the
    // dropped bits (defensive; logits are ~+/-20 so the int part never nears
    // the +/-2048 range). inf/nan -> +/- saturation.  Mirrors the fp32->Q12
    // decode in softmax_frontier_q.sv generalised to 28 fraction bits.
    // -----------------------------------------------------------------
    function automatic logic signed [SCORE_W-1:0] fp32_to_q1228;
        input logic [31:0] f;
        logic               s; logic [7:0] e; logic [22:0] m; logic [23:0] M;
        integer             ET; logic [63:0] magw, intmag, res;
        logic               rbit; integer sh;
        begin
            s = f[31]; e = f[30:23]; m = f[22:0];
            if (e == 8'd255) begin
                fp32_to_q1228 = s ? -{1'b0,{(SCORE_W-1){1'b1}}} : {1'b0,{(SCORE_W-1){1'b1}}};
            end else begin
                M  = (e==8'd0) ? {1'b0,m} : {1'b1,m};
                ET = (e==8'd0) ? (-149 + SCORE_F) : (e - 127 - 23 + SCORE_F); // *2^ET
                if (M == 24'd0) begin
                    fp32_to_q1228 = '0;
                end else begin
                    magw = {40'd0, M};
                    if (ET >= 0) begin
                        res = (ET >= 40) ? 64'h00FF_FFFF_FFFF_FFFF : (magw << ET);
                    end else begin
                        sh     = -ET;
                        intmag = (sh >= 64) ? 64'd0 : (magw >> sh);
                        rbit   = (sh >= 1 && sh <= 64) ? magw[sh-1] : 1'b0;
                        res    = intmag + (rbit ? 64'd1 : 64'd0);
                    end
                    if (res >= (64'd1 << (SCORE_W-1))) res = (64'd1 << (SCORE_W-1)) - 64'd1;
                    fp32_to_q1228 = s ? -$signed(res[SCORE_W-1:0]) : $signed(res[SCORE_W-1:0]);
                end
            end
        end
    endfunction

    // -----------------------------------------------------------------
    // 2^(-f), f in [0,1) given as Q0.28, returned Q1.28 in (2^27, 2^28].
    //   2^(-f) = 2^(1-f) / 2 = POW2_LUT_interp(1-f) >> 1.
    // 1024-entry linear interpolation, same shape as tree_log2_fxp.
    // -----------------------------------------------------------------
    function automatic logic [29:0] exp2_neg_q28;
        input logic [27:0] f_q28;            // fractional part, [0,1)
        logic [28:0] g;                      // 1 - f in Q0.28, strictly < 2^28 for f>0
        logic [9:0]  idx; logic [17:0] fr;
        logic [29:0] lo, hi; logic [47:0] interp; logic [29:0] val;
        begin
            if (f_q28 == 28'd0) begin
                exp2_neg_q28 = 30'(1) << SCORE_F;         // 2^0 = 1.0 in Q1.28 (=2^28)
            end else begin
                g   = (29'd1 << SCORE_F) - {1'b0, f_q28}; // 1 - f, in (0, 2^28)
                idx = g[27:18];                            // 0..1023 (g < 2^28)
                fr  = g[17:0];
                lo  = POW2_LUT[idx];
                hi  = POW2_LUT[idx + 1];
                interp = ({30'd0, fr} * {30'd0, (hi - lo)}) >> 18;
                val = lo + interp[29:0];                   // 2^(1-f) in Q1.28, [2^28, 2^29)
                exp2_neg_q28 = val >> 1;                   // 2^(-f) in Q1.28, (2^27, 2^28)
            end
        end
    endfunction

    // -----------------------------------------------------------------
    // 2^(+m), m in [0,1) given as Q0.28, returned Q1.28 in [2^28, 2^29).
    // Direct POW2 interpolation -- used to build the fp32 mantissa.
    // -----------------------------------------------------------------
    function automatic logic [29:0] exp2_pos_q28;
        input logic [27:0] m_q28;
        logic [9:0]  idx; logic [17:0] fr;
        logic [29:0] lo, hi; logic [47:0] interp;
        begin
            idx = m_q28[27:18];
            fr  = m_q28[17:0];
            lo  = POW2_LUT[idx];
            hi  = POW2_LUT[idx + 1];
            interp = ({30'd0, fr} * {30'd0, (hi - lo)}) >> 18;
            exp2_pos_q28 = lo + interp[29:0];
        end
    endfunction

    // frontier logit / token slice helpers
    function automatic logic [31:0]      fr_f32; input integer k; fr_f32 = frontier_f32[k*32 +: 32]; endfunction
    function automatic logic [TOK_W-1:0] fr_tok; input integer k; fr_tok = frontier_tok[k*TOK_W +: TOK_W]; endfunction

    // (l - max) * log2(e)  in Q12.28   (delta <= 0)
    function automatic logic signed [SCORE_W-1:0] delta2_of;
        input logic signed [SCORE_W-1:0] l_q, max_q;
        logic signed [SCORE_W:0]   d;         // l - max, <= 0
        logic signed [SCORE_W+30:0] p;        // Q12.(28+28)
        begin
            d = {l_q[SCORE_W-1], l_q} - {max_q[SCORE_W-1], max_q};
            p = d * $signed({1'b0, LOG2E_Q28});
            delta2_of = SCORE_W'(p >>> SCORE_F);
        end
    endfunction

    // 2^(delta2), delta2 <= 0 (Q12.28), returned Q1.28 term for the Z sum.
    function automatic logic [29:0] exp2_term;
        input logic signed [SCORE_W-1:0] delta2;      // <= 0
        logic [SCORE_W-1:0] mag; logic [11:0] n; logic [27:0] f;
        logic [29:0] base;
        begin
            mag = delta2[SCORE_W-1] ? (~delta2 + 1'b1) : delta2;  // |delta2|
            n   = mag[SCORE_W-1:SCORE_F];        // integer part (>=0)
            f   = mag[SCORE_F-1:0];              // fractional Q0.28
            base = exp2_neg_q28(f);              // 2^(-f), Q1.28
            exp2_term = (n >= NCLAMP) ? 30'd0 : (base >> n);
        end
    endfunction

    // -----------------------------------------------------------------
    // log2 of the accumulated Z (Z >= 1.0), returned signed Q12.28 (>= 0).
    // Z_acc is unsigned Q(int).28.  log2(Z) = (msb-28) + log2(1.mant),
    // the second term from LOG2_LUT + interp, same as tree_log2_fxp.
    // -----------------------------------------------------------------
    localparam int ZW = 52;                    // Z accumulator width
    function automatic logic signed [SCORE_W-1:0] fxp_log2_z;
        input logic [ZW-1:0] z;
        integer b, i; logic [ZW-1:0] norm;
        logic [9:0] idx; logic [17:0] fr; logic [28:0] lo, hi, mant; logic [47:0] interp;
        logic signed [SCORE_W-1:0] ip;
        begin
            b = 0;
            for (i = ZW-1; i >= 0; i = i - 1) if (z[i] && b == 0) b = i;  // MSB index
            // integer part = b - 28
            ip = SCORE_W'($signed(b - SCORE_F)) <<< SCORE_F;
            // mantissa fraction: the 28 bits below the MSB, top 10 -> LUT idx
            norm = (b >= 28) ? (z << (ZW-1-b)) : (z << (ZW-1-b));  // MSB now at ZW-1
            // bits [ZW-2 : ZW-11] are the fraction top-10; [ZW-12: ZW-29] interp
            idx = norm[ZW-2 -: 10];
            fr  = norm[ZW-12 -: 18];
            lo  = LOG2_LUT[idx];
            hi  = LOG2_LUT[idx + 1];
            interp = ({30'd0, fr} * {29'd0, (hi - lo)}) >> 18;
            mant = lo + interp[28:0];            // log2(1.mant) in Q0.28
            fxp_log2_z = ip + SCORE_W'({{(SCORE_W-29){1'b0}}, mant});
        end
    endfunction

    // assemble IEEE-754 fp32 from log2p (<= 0, Q12.28), clamped to >= LOG2_FLOOR.
    function automatic logic [31:0] fp32_from_log2;
        input logic signed [SCORE_W-1:0] log2p_in;
        logic signed [SCORE_W-1:0] log2p;
        logic signed [11:0] E; logic [27:0] mfrac; logic [29:0] p2; logic [22:0] mant;
        logic [8:0] ef;
        begin
            log2p = (log2p_in < LOG2_FLOOR) ? LOG2_FLOOR :
                    (log2p_in > 0)          ? '0 : log2p_in;
            E     = log2p[SCORE_W-1:SCORE_F];               // arithmetic floor, <= 0
            mfrac = log2p[SCORE_F-1:0];                     // frac in [0,1), Q0.28
            p2    = exp2_pos_q28(mfrac);                    // 2^mfrac in Q1.28, [2^28,2^29)
            mant  = p2[27:5];                               // drop the leading 1, keep 23b
            ef    = $signed(E) + 9'sd127;                   // biased exponent (>=79, normal)
            fp32_from_log2 = {1'b0, ef[7:0], mant};
        end
    endfunction

    // -----------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------
    typedef enum logic [3:0] { S_IDLE, S_MAX, S_SUM, S_SUMD, S_L2Z, S_L2Z2, S_FIN, S_DONE } st_t;
    st_t state;
    reg  [CIDX_W:0]            ci;
    // BRIDGE_KC: launch-sampled effective shortlist (1..K_C) and its last index.
    reg  [CIDX_W:0]            kcq;
    wire [CIDX_W:0]            kc_lastc = kcq - 1'b1;
    reg  signed [SCORE_W-1:0]  max_q;
    reg  [ZW-1:0]              z_acc;
    reg  [ZW-1:0]              z_l2z_r;
    reg  signed [SCORE_W-1:0]  log2z_q;
    reg  [3:0]                 fk;

    wire signed [SCORE_W-1:0]  cand_q = fp32_to_q1228(cand_logit_f32);

    // -----------------------------------------------------------------
    // Z-ACC TERM PIPELINE (2026-07-24, main-thread). The 0xE4B71119 STEP1
    // killer was THIS cone, not the F4 mux: fp32 decode -> delta2 (41x29
    // multiply by LOG2E) -> exp2 LUT -> variable >>n barrel shift -> ZW-bit
    // accumulate, all in ONE cycle = 22.2 ns / 98 logic levels. Cut it into
    // 3 registered stages; z_acc consumes delayed terms. The SUM over 512
    // candidates is order-and-value IDENTICAL (same adds, 3 cycles later),
    // so numerics are bit-exact vs the flat form; cost is a 3-cycle drain
    // (S_SUMD) per pass — noise against the 512-cycle pass itself.
    //   p1: registered decoded logit    (cuts decode off the mult)
    //   p2: registered delta2           (isolates the LOG2E multiply)
    //   p3: registered exp2 term        (isolates LUT + barrel shift)
    //   acc: z_acc += p3 when p3_v      (wide add alone in its own cycle)
    // -----------------------------------------------------------------
    reg  signed [SCORE_W-1:0]  p1_q;
    reg  signed [SCORE_W-1:0]  p2_delta;
    reg         [29:0]         p3_term;
    reg                        p1_v, p2_v, p3_v;
    reg  [1:0]                 drain_ct;
    // S_FIN per-fk pipeline (same pattern as p1/p2/p3 above)
    reg  signed [SCORE_W-1:0]  f1_q, f2_d;
    reg  [3:0]                 f1_fk, f2_fk;
    reg                        f1_v, f2_v;
    reg                        fin_drain;

`ifdef BRIDGE_RED_NONORM
    // RED (c): NORMALIZATION DROPPED. Feed the UNNORMALIZED numerator log2(2^delta2)
    // = delta2 with NO -log2(Z) subtraction.  Within one parent the ranking is
    // unchanged, but ACROSS parents the per-parent denominator is exactly the
    // information the global top-8 needs, so the tree shape must diverge.
    localparam bit RED_NONORM = 1'b1;
`else
    localparam bit RED_NONORM = 1'b0;
`endif

    integer kk;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; ci <= '0; max_q <= '0; z_acc <= '0; log2z_q <= '0; fk <= '0;
            kcq   <= (CIDX_W+1)'(K_C);
            f1_v <= 1'b0; f2_v <= 1'b0; fin_drain <= 1'b0; f1_q <= '0; f2_d <= '0; f1_fk <= '0; f2_fk <= '0;
            cand_rdaddr <= '0; cand_rd_active <= 1'b0; done <= 1'b0;
            prob_f32_o <= '0; tok_o <= '0;
        end else begin
            done <= 1'b0;
            // term pipeline — runs unconditionally; p1_v gates entry, p3_v gates
            // the accumulate. Placed BEFORE the case so S_IDLE's z_acc <= '0 on
            // start (a later assignment in this block) wins over a stale term.
            p1_q     <= cand_q;
            p1_v     <= (state == S_SUM);
            p2_delta <= delta2_of(p1_q, max_q);
            p2_v     <= p1_v;
            p3_term  <= exp2_term(p2_delta);
            p3_v     <= p2_v;
            if (p3_v) z_acc <= z_acc + {{(ZW-30){1'b0}}, p3_term};
            case (state)
                S_IDLE: begin
                    cand_rd_active <= 1'b0;
                    if (start) begin
                        // BRIDGE_KC: sample the runtime shortlist once per parent.
                        kcq   <= (kc_en && kc_cfg != 16'd0 && kc_cfg < 16'(K_C))
                                 ? (CIDX_W+1)'(kc_cfg) : (CIDX_W+1)'(K_C);
                        // BRIDGE_KC S_MAX-skip: seed max_q from the G8 top-1
                        // (frontier slot 0, latched by feed_seq before start).
                        max_q <= kc_en ? fp32_to_q1228(fr_f32(0))
                                       : {1'b1,{(SCORE_W-1){1'b0}}}; // -inf
                        z_acc <= '0; ci <= '0; cand_rdaddr <= '0;
                        p1_v <= 1'b0; p2_v <= 1'b0; p3_v <= 1'b0;
                        f1_v <= 1'b0; f2_v <= 1'b0; fin_drain <= 1'b0;
                        cand_rd_active <= 1'b1;
                        state <= kc_en ? S_SUM : S_MAX;
                    end
                end
                // PASS A: running max over the 512 logits. cand_rdaddr was set
                // last cycle -> cand_q valid this cycle. (Skipped when kc_en.)
                S_MAX: begin
                    cand_rd_active <= 1'b1;
                    if (cand_q > max_q) max_q <= cand_q;
                    if (ci == kc_lastc) begin
                        ci <= '0; cand_rdaddr <= '0; state <= S_SUM;
                    end else begin
                        ci <= ci + 1'b1; cand_rdaddr <= cand_rdaddr + 1'b1;
                    end
                end
                // PASS B: Z = sum 2^((l-max)*log2e). Address walk only — the
                // decode/delta2/exp2/accumulate now live in the p1/p2/p3
                // pipeline above (bit-exact same sum, 3 cycles later).
                S_SUM: begin
                    cand_rd_active <= 1'b1;
                    if (ci == kc_lastc) begin
                        cand_rd_active <= 1'b0; drain_ct <= 2'd3; state <= S_SUMD;
                    end else begin
                        ci <= ci + 1'b1; cand_rdaddr <= cand_rdaddr + 1'b1;
                    end
                end
                // Drain the 3-stage term pipeline; the last p3_v accumulate
                // lands on the final drain cycle, so z_acc is complete at S_L2Z.
                S_SUMD: begin
                    drain_ct <= drain_ct - 1'b1;
                    if (drain_ct == 2'd1) state <= S_L2Z;
                end
                // log2(Z) once.
                // S_L2Z: fxp_log2_z (priority encoder + LUT) once per parent.
                // Registered z_acc -> log2 in its own cycle so it can never be
                // the hidden next cone behind S_FIN (max_paths=1 only shows one).
                S_L2Z: begin
                    z_l2z_r <= z_acc;
                    state <= S_L2Z2;
                end
                S_L2Z2: begin
                    log2z_q <= fxp_log2_z(z_l2z_r);
                    fk <= '0; state <= S_FIN;
                end
                // FINAL: 8 children -> log2 p_k = delta2_k - log2Z -> fp32.
                // S_FIN (run-3 STEP1 killer, WNS -6.799 after z_acc was fixed):
                // the per-child decode->delta2(LOG2E mult)->-log2z->fp32-encode
                // chain was ONE cycle = 18.97 ns / 84 levels. Same cure as
                // z_acc: a 3-stage per-fk pipeline (f1 decode, f2 delta2-log2z,
                // f3 encode+commit). Values are bit-identical, K_F+2 cycles
                // total instead of K_F — noise once per parent softmax.
                S_FIN: begin
                    // stage f1: fp32 bits -> Q12.28 decode
                    f1_q  <= fp32_to_q1228(fr_f32(fk));
                    f1_fk <= fk;
                    f1_v  <= (fk <= K_F-1) && !fin_drain;
                    // stage f2: delta2 (LOG2E multiply) and Z-normalise
                    f2_d  <= RED_NONORM ? delta2_of(f1_q, max_q)
                                        : (delta2_of(f1_q, max_q) - log2z_q);
                    f2_fk <= f1_fk;
                    f2_v  <= f1_v;
                    // stage f3: fp32 encode + commit
                    if (f2_v) begin
                        prob_f32_o[f2_fk*32 +: 32]  <= fp32_from_log2(f2_d);
                        tok_o[f2_fk*TOK_W +: TOK_W] <= fr_tok(f2_fk);
                        if (f2_fk == 4'(K_F-1)) begin
                            fin_drain <= 1'b0; state <= S_DONE;
                        end
                    end
                    if (fk == 4'(K_F-1)) fin_drain <= 1'b1;
                    else if (!fin_drain)      fk <= fk + 1'b1;
                end
                S_DONE: begin done <= 1'b1; f1_v <= 1'b0; f2_v <= 1'b0; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------
    // SEAM-FF LOOKAHEAD next-read-address. Mirrors EXACTLY the `cand_rdaddr <= ...`
    // RHS in the FSM above, presented combinationally one cycle early. Fed to g8's
    // 1-cycle registered rescored[] read (bridge_score_ff), it delivers
    // rescored[addr] on the very cycle each max/sum step consumes cand_logit_f32.
    // Both passes are commutative reductions over all 512 slots read exactly once,
    // so the emitted probs are BIT-IDENTICAL to the async read; only the physical
    // read moves behind an FDRE (kills the netlist tied-zero collapse). The bridge
    // consume timing is UNCHANGED, so the certified SET signature is preserved by
    // construction. See reports/results/TREE_SCORE_TIEDZERO_FIX.md §Seam FF.
    always_comb begin
        case (state)
            S_IDLE:  cand_rdaddr_nxt = start ? '0 : cand_rdaddr;
            S_MAX:   cand_rdaddr_nxt = (ci == kc_lastc) ? '0
                                                        : (cand_rdaddr + 1'b1);
            S_SUM:   cand_rdaddr_nxt = (ci == kc_lastc) ? cand_rdaddr
                                                        : (cand_rdaddr + 1'b1);
            default: cand_rdaddr_nxt = cand_rdaddr;
        endcase
    end

endmodule

`default_nettype wire
