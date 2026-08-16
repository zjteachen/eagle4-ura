// =====================================================================
// g8_dot_fp16.sv  —  SYNTHESIZABLE fp16-mul / fp32-acc streaming dot product.
//
// Fork: verilog/forks/pdi6_g8  (NEW standalone primitive; does NOT touch
//        pdi6_s7c, does NOT edit g8_rescore.sv — this is the drop-in that
//        replaces the SV `real` G8.3 dot in the skeleton).
//
// PURPOSE
//   The ONE non-synthesizable piece of the G8 rescore skeleton is its G8.3
//   dot, which models the golden's fp16-multiply / fp32-accumulate precision
//   with SystemVerilog `real`.  `real` is NOT synthesizable.  This module is
//   the real-RTL replacement: an IEEE-754 binary16 multiplier
//   (g8_fp16_pkg::fp16_mul) -> fp16->fp32 widen -> IEEE-754 binary32 add,
//   with the running sum REGISTERED so the accumulation order is exactly
//   left-to-right over H — bit-identical to the golden's
//   `np.add.accumulate(prod_fp16.astype(f32), dtype=f32)`.
//
// THE EXACT NUMERIC CONTRACT (from ref/g8_dot_fp16_ref.py + capture_goldens.py)
//   per element h:   p_h(fp16) = round_to_fp16( a_h(fp16) * b_h(fp16) )
//   running sum:     acc(fp32) = fp32_add( acc, fp32(p_h) )   left-to-right
//   final score:     fp32 acc after the last (H-th) element
//
// PIPELINE & II (parameter FADD_PIPE selects the accumulate scheme):
//   The single-cycle (mul->widen->fp32_add->acc) chain is ~20 ns (77 levels)
//   and fails 125 MHz.  The multiply+widen are ALWAYS registered (stage 1) —
//   that does NOT change numerics (product bits identical).  The fp32 ADD is
//   the recurrence (acc feeds itself), so it sets the achievable clock:
//
//   MUL_PIPE (DEFAULT 1): even with the product registered at stage 1, the
//   COMBINATIONAL feed dot_b -> fp16_mul(DSP58 11x11) -> fp16 round -> fp16->fp32
//   widen -> s1_prod32_reg is the OOC critical path of the whole G8 stage
//   (36 logic levels, WNS@125 = -1.807 ns under unplaced routing — see
//   reports/G8_SKELETON_SYNTH_20260527.md §9).  MUL_PIPE=1 registers the fp16
//   product (prod16) BETWEEN the multiply/round and the widen, so neither half
//   is on the same combinational path: stage 0a = DSP mul + fp16 round -> reg;
//   stage 0b = fp16->fp32 widen (subnormal carry chain) -> reg.  This adds ONE
//   cycle of dot INPUT latency (input -> s1_prod32 is now 2 cycles, not 1) but
//   does NOT touch the accumulate recurrence below, and is numerically a pure
//   register delay (identical product/score bits).  The FSM's done-driven
//   S_GA_WAIT (polls out_valid) absorbs the extra cycle with NO dwell edit.
//
//   MREG (DEFAULT 1): MUL_PIPE alone still leaves the path
//   dot_b -> DSP58 11x11 multiply -> fp16 normalize/round LUTs -> s0_prod16_reg
//   (32 logic levels, WNS@125 = -0.409 ns OOC, §10) — the DSP product AND the
//   fp16 round are still in one cycle.  MREG=1 registers the RAW 22-bit DSP58
//   product (the multiplier's own pipeline register, the standard DSP58 MREG)
//   BETWEEN the 11x11 multiply (fp16_mul_a) and the normalize/round
//   (fp16_mul_b), so stage 0_pre = DSP58 multiply -> mreg; stage 0a =
//   normalize/round -> s0_prod16_reg.  This adds ANOTHER cycle of dot INPUT
//   latency (input -> s1_prod32 is now 3 cycles) and is again numerically a
//   pure register delay (the raw product bits are identical, the round happens
//   one cycle later on the same bits).  The accumulate recurrence (FADD_PIPE/
//   II=2) and the done-driven S_GA_WAIT are UNTOUCHED — the +1 cycle is
//   absorbed exactly like MUL_PIPE's.  This is the deep-pipeline lever that
//   removes the placement-dependence of the last ~0.4 ns.
//
//   * FADD_PIPE=0 (II=1): one combinational fp32_add per cycle in the loop.
//       Logic delay ~3.4 ns => closes 125/200 MHz on LOGIC, but the OOC
//       unplaced-route estimate (~7 ns route, non-physical for a tiny module)
//       leaves a negative OOC WNS.  Use when the surrounding design places it
//       (route shrinks) or when II=1 is mandatory.
//   * FADD_PIPE=1 (II=2, DEFAULT): the fp32_add is SPLIT across two registers
//       (fp32_add_a = align+raw-add ; fp32_add_b = normalize+round).  The
//       2-cycle recurrence halves the per-cycle logic and CLOSES 125 MHz OOC
//       with margin.  Producer presents one (a,b) every OTHER cycle (II=2).
//       The dot is gather-bandwidth-bound in G8 (SPEC §5), so II=2 on the dot
//       does not gate the per-forward latency.
//
//   BOTH schemes are BIT-EXACT: the accumulate is strictly sequential
//   left-to-right (one fp32 RNE add per element, acc feeds itself), and
//   fp32_add_b(fp32_add_a(x,y)) == fp32_add(x,y) exactly (same algorithm, cut
//   at the raw-add/normalize boundary).
//
// WHY HAND-BUILT (not a DSP58 FP IP): Versal DSP58 FP mode is fp32 not fp16;
//   a bit-EXACT fp16 product from a DSPFP32 primitive is fragile.  The fp16
//   multiply's 11x11 mantissa multiply maps to ONE DSP58 (INT mode, confirmed
//   in ooc: 1 DSP58, A=11/B=11/P=22).  The rest is LUT logic.
//
// INTERFACE (streaming):
//   - pulse `clear` with the FIRST element of a row (seeds acc from +0.0).
//   - assert `in_valid` with (a_fp16, b_fp16) per element; pulse `in_last` on
//     the H-th element.  FADD_PIPE=1 => present valid every OTHER cycle (II=2).
//   - `out_valid` pulses when `score_fp32` holds the final row sum.
//
// MULTI-DRIVER-SAFE: every reg is written by exactly one always_ff;
//   `score_fp32` is a combinational read of `acc`.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

import g8_fp16_pkg::*;

(* keep_hierarchy = "yes" *)
module g8_dot_fp16 #(
    parameter bit FADD_PIPE = 1'b1,    // 1 = split fp32_add (II=2, closes 125 OOC)
                                       // 0 = single-cycle fp32_add (II=1)
    parameter bit MUL_PIPE  = 1'b1,    // 1 = register fp16_mul OUTPUT (prod16) before
                                       //     the fp16->fp32 widen, breaking the
                                       //     dot_b->fp16_mul(DSP)->fp16_round->widen->reg
                                       //     critical path (the OOC -1.807 ns mul->reg
                                       //     path) into two shallower stages.  Adds ONE
                                       //     cycle of dot input latency; numerically a
                                       //     pure register delay (same product bits).
                                       // 0 = single-stage mul+widen (legacy).
    parameter bit MROUND    = 1'b0,    // 1 = split fp16_mul_b (normalize/round) into
                                       //     b1 (LZD priority-encode + barrel-shift
                                       //     normalize + e_norm, ~16 LL) and b2
                                       //     (guard/round/sticky + 11b round-add carry
                                       //     chain + over/underflow clamp, ~16 LL),
                                       //     registering the normalized shifted/e_norm
                                       //     IR between them.  Cuts the §2 DSP58 ->
                                       //     fp16_mul_b -> s0_prod16_reg 32-LL cone at
                                       //     its natural midpoint.  Adds ONE cycle of
                                       //     dot input latency; pure register delay
                                       //     (fp16_mul_b2(fp16_mul_b1(ir))==fp16_mul_b(ir)
                                       //     by construction).  Sits before the prod16
                                       //     reg; orthogonal to the II=2 accumulate
                                       //     recurrence so II is UNCHANGED.
                                       // 0 = single-stage fp16_mul_b (legacy).
    parameter bit MREG      = 1'b1     // 1 = register the RAW DSP58 22-bit product
                                       //     (the multiplier's MREG) BETWEEN the 11x11
                                       //     multiply (fp16_mul_a) and the fp16
                                       //     normalize/round (fp16_mul_b), splitting the
                                       //     §10 mul->fp16_round->s0_prod16_reg path
                                       //     (-0.409 ns) at the DSP product.  Adds ANOTHER
                                       //     cycle of input latency; pure register delay
                                       //     (identical product/score bits).  Requires
                                       //     MUL_PIPE=1 (sits before the prod16 reg).
                                       // 0 = combinational mul+round (legacy MUL_PIPE path).
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        in_valid,
    input  wire        in_last,
    input  wire [15:0] a_fp16,
    input  wire [15:0] b_fp16,
    output logic       out_valid,
    output logic [31:0] score_fp32
);

    // ---------------- stage 0_pre: DSP58 11x11 multiply (fp16_mul_a) ---------
    //   The heavy DSP58 mantissa multiply + special-case decode.  When MREG=1
    //   we register its RAW product IR (the multiplier's MREG) here, so the
    //   fp16 normalize/round (fp16_mul_b) lands in a SEPARATE cycle -> the §10
    //   path dot_b -> DSP mul -> fp16_round -> s0_prod16_reg (32 levels,
    //   -0.409 ns OOC) is cut at the DSP product boundary.
    logic [49:0] mul_ir;
    always_comb mul_ir = fp16_mul_a(a_fp16, b_fp16);

    // mul_ir / control entering the normalize+round stage (after optional MREG)
    logic [49:0] q_mul_ir;
    logic        q_valid, q_last, q_clear;

    generate
    if (MREG) begin : g_mreg
        // DSP58 MREG: register the raw 22-bit product IR + its control.
        logic [49:0] mreg_ir;
        logic        mreg_valid, mreg_last, mreg_clear;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                mreg_ir <= 50'd0; mreg_valid <= 1'b0; mreg_last <= 1'b0; mreg_clear <= 1'b0;
            end else begin
                mreg_ir    <= mul_ir;
                mreg_valid <= in_valid;
                mreg_last  <= in_valid & in_last;
                mreg_clear <= in_valid & clear;
            end
        end
        assign q_mul_ir = mreg_ir;
        assign q_valid  = mreg_valid;
        assign q_last   = mreg_last;
        assign q_clear  = mreg_clear;
    end else begin : g_nomreg
        // legacy: round the combinational product directly (MUL_PIPE path).
        assign q_mul_ir = mul_ir;
        assign q_valid  = in_valid;
        assign q_last   = in_valid & in_last;
        assign q_clear  = in_valid & clear;
    end
    endgenerate

    // ---------------- stage 0a: fp16 normalize + round (fp16_mul_b) ----------
    //   fp16_mul_b(fp16_mul_a(a,b)) == fp16_mul(a,b) exactly.  When MUL_PIPE=1
    //   we register its OUTPUT (prod16) here, so the fp16->fp32 widen (subnormal
    //   carry chain) lands in a SEPARATE cycle.
    //
    //   When MROUND=1 the normalize/round itself is split: stage 0a1 =
    //   fp16_mul_b1 (LZD priority-encode + barrel-shift normalize + e_norm) ->
    //   reg; stage 0a2 = fp16_mul_b2 (guard/round/sticky + round-add carry chain
    //   + clamp).  This cuts the §2 DSP58 -> fp16_mul_b 32-LL cone at its
    //   midpoint.  Adds ONE register cycle; numerically pure delay
    //   (fp16_mul_b2(fp16_mul_b1(ir)) == fp16_mul_b(ir) by construction).
    logic [15:0] prod16;            // fp16 product entering the prod16 register
    logic [49:0] r_mul_ir;          // round-stage IR feeding fp16_mul_b/_b2
    logic        r_valid, r_last, r_clear;

    generate
    if (MROUND) begin : g_mround
        // stage 0a1: LZD + barrel-shift normalize -> registered b1 IR.
        logic [49:0] b1_ir;
        always_comb b1_ir = fp16_mul_b1(q_mul_ir);

        logic [49:0] s0a_b1;
        logic        s0a_valid, s0a_last, s0a_clear;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                s0a_b1 <= 50'd0; s0a_valid <= 1'b0; s0a_last <= 1'b0; s0a_clear <= 1'b0;
            end else begin
                s0a_b1    <= b1_ir;
                s0a_valid <= q_valid;
                s0a_last  <= q_last;
                s0a_clear <= q_clear;
            end
        end
        assign r_mul_ir = s0a_b1;
        assign r_valid  = s0a_valid;
        assign r_last   = s0a_last;
        assign r_clear  = s0a_clear;
        // stage 0a2: round + pack (consumed by the prod16 reg / widen below).
        always_comb prod16 = fp16_mul_b2(r_mul_ir);
    end else begin : g_nomround
        // single-stage fp16_mul_b (legacy): no extra register, no extra latency.
        assign r_mul_ir = q_mul_ir;
        assign r_valid  = q_valid;
        assign r_last   = q_last;
        assign r_clear  = q_clear;
        always_comb prod16 = fp16_mul_b(r_mul_ir);
    end
    endgenerate

    // pipelined-product domain (after the optional extra stage)
    logic [15:0] p_prod16;             // prod16 entering the widen
    logic        p_valid, p_last, p_clear;

    generate
    if (MUL_PIPE) begin : g_mulpipe
        // stage 0b register: hold the fp16 product + its control one cycle.
        // Control comes from the round-stage domain (r_*) so it stays aligned
        // through the optional MREG AND the optional MROUND register: r_* equals
        // q_* (MROUND=0) or q_* delayed one cycle (MROUND=1), matching prod16.
        logic [15:0] s0_prod16;
        logic        s0_valid, s0_last, s0_clear;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                s0_prod16 <= 16'd0; s0_valid <= 1'b0; s0_last <= 1'b0; s0_clear <= 1'b0;
            end else begin
                s0_prod16 <= prod16;
                s0_valid  <= r_valid;
                s0_last   <= r_last;
                s0_clear  <= r_clear;
            end
        end
        assign p_prod16 = s0_prod16;
        assign p_valid  = s0_valid;
        assign p_last   = s0_last;
        assign p_clear  = s0_clear;
    end else begin : g_nomulpipe
        // no prod16 register: widen the combinational round output directly.
        // Control = the round-stage domain (r_*), aligned through optional MREG
        // and optional MROUND register.
        assign p_prod16 = prod16;
        assign p_valid  = r_valid;
        assign p_last   = r_last;
        assign p_clear  = r_clear;
    end
    endgenerate

    // ---------------- combinational fp16 -> fp32 widen ----------------------
    logic [31:0] prod32;
    always_comb prod32 = fp16_to_fp32(p_prod16);

    // ---------------- stage 1: register widened product + control -----------
    logic [31:0] s1_prod32;
    logic        s1_valid, s1_last, s1_clear;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s1_prod32 <= 32'd0; s1_valid <= 1'b0; s1_last <= 1'b0; s1_clear <= 1'b0;
        end else begin
            s1_prod32 <= prod32;
            s1_valid  <= p_valid;
            s1_last   <= p_last;
            s1_clear  <= p_clear;
        end
    end

    logic [31:0] acc;
    assign score_fp32 = acc;

    generate
    if (!FADD_PIPE) begin : g_ii1
        // ---- II=1: single combinational fp32_add in the loop ----
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                acc <= 32'd0; out_valid <= 1'b0;
            end else begin
                out_valid <= 1'b0;
                if (s1_valid) begin
                    acc <= fp32_add(s1_clear ? 32'd0 : acc, s1_prod32);
                    if (s1_last) out_valid <= 1'b1;
                end
            end
        end
    end else begin : g_ii2
        // ---- II=2: fp32_add split across two registers ----
        //   stage A: ir <= fp32_add_a(seed, s1_prod32)   (align + raw add)
        //   stage B: acc <= fp32_add_b(ir)               (normalize + round)
        // Producer feeds valid every OTHER cycle so each A reads the acc the
        // previous element's B already wrote.
        logic [71:0] ir;
        logic        a_valid, a_last;      // stage-A produced fresh IR this cycle
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                ir <= 72'd0; a_valid <= 1'b0; a_last <= 1'b0;
                acc <= 32'd0; out_valid <= 1'b0;
            end else begin
                out_valid <= 1'b0;
                // stage A: on a valid input beat, latch the aligned raw-add IR.
                a_valid <= s1_valid;
                if (s1_valid) begin
                    ir     <= fp32_add_a(s1_clear ? 32'd0 : acc, s1_prod32);
                    a_last <= s1_last;
                end
                // stage B: only fire when stage A produced fresh IR last cycle.
                if (a_valid) begin
                    acc <= fp32_add_b(ir);
                    if (a_last) out_valid <= 1'b1;
                end
            end
        end
    end
    endgenerate

endmodule

`default_nettype wire
