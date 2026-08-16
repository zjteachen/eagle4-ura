// =====================================================================
// csa_tree16_int28.sv
// Wallace 3:2 CSA tree — WIDENED variant of csa_tree16_int20.
// Reduces 16 x signed INT28 products into 1 x signed INT32 sum.
// =====================================================================
//
// WHY THIS EXISTS (2026-06-20, depth-N attention-score precision fix):
//   The shared csa_tree16_int20 truncates every product to INT20
//   (real +/-0.5 at frac20).  In the ATTENTION SCORE path the
//   per-element q*k products reach +/-41 real (frac20) at depth-1 ->
//   231/256 OVERFLOW the INT20 clamp, sign-wrap, and shrink the true
//   Q.K dot ~5x (RTL 7.79 vs fp32-implied 38.15).  See
//   reports/results/DEPTHN_ATTN_SCORE_FIX.md.
//
//   csa_tree16_int20 is SHARED with the body GEMV path (gemv_lane_*),
//   where the INT20 retention is correct, so it MUST NOT be widened.
//   This is a SEPARATE wider primitive used ONLY by attn_score_dot.
//
//   INT28 input retains real magnitude up to +/-128 at frac20
//   (2^27/2^20 = 128), 3.1x headroom over the measured 41.2 max.
//   16 x INT28 sum needs +4 bits -> INT32 output.
//
//   INT32-WIDEN (2026-06-30, depth-2 v=64 residual root cause):
//   The "41.2 real max" was UNDER-measured. At depth-1 (ctx_len=2) the
//   root-key (K_pos0) dot for heads h11/h17 has element products up to
//   137.9 real (frac20) -> OVERFLOWS the INT28 clamp (2^27/2^20=128),
//   sign-wraps ONE element, and flips pos0's dot from -204 real to +52
//   -> softmax spuriously picks pos0 -> out_vec = V_pos0 (bit-exact
//   golden V0) instead of V_pos1 -> wrong depth-1 child token
//   (8545 vs golden 315). Only h11/h17 have |product|>128, matching the
//   2 observed bad heads exactly. FIX = retain the FULL INT16*INT16
//   product (INT32; the CSA already computes in INT32 internally, and
//   max chunk-sum 298 real / max dot 600 real both fit INT32). This
//   port is now INT32; the sign-extend is a no-op pass-through.
//   Non-overflowing scores are BIT-IDENTICAL (full pf == sext(pf[27:0]))
//   -> depth-0 (ctx_len=1 -> out=V[0], score-agnostic) byte-safe, and
//   all depth-1 heads except h11/h17 unchanged. Only attn_score_dot uses
//   this primitive (GEMV uses csa_tree16_int20, untouched).
//   Evidence: reports/results/DEPTHN_SCORE_INT28_OVERFLOW_FIX.md.
//
// Reduction schedule + pipelining identical to csa_tree16_int20
// (3 stages, latency 3, II=1, bit-exact integer sum).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module csa_tree16_int28 (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          in_valid,
    input  wire signed [31:0]            prods [0:15],        // 16 INT32 products (was INT28; widened 2026-06-30)
    output reg  signed [31:0]            sum_out,             // INT32 sum
    output reg                           out_valid
);

    // INT32-WIDEN (2026-06-30): inputs are now full INT32 products; the
    // former INT28->INT32 sign-extend is a no-op pass-through.
    logic signed [31:0] x [0:15];
    always_comb for (int i = 0; i < 16; i++)
        x[i] = prods[i];

    function automatic [31:0] csa_sum (input [31:0] a, b, c);
        csa_sum = a ^ b ^ c;
    endfunction
    function automatic [31:0] csa_car (input [31:0] a, b, c);
        csa_car = ((a & b) | (a & c) | (b & c)) << 1;
    endfunction

    // STAGE A — combinational 16 -> 6
    logic [31:0] r1 [0:10];
    assign r1[0]  = csa_sum(x[0],  x[1],  x[2]);
    assign r1[1]  = csa_car(x[0],  x[1],  x[2]);
    assign r1[2]  = csa_sum(x[3],  x[4],  x[5]);
    assign r1[3]  = csa_car(x[3],  x[4],  x[5]);
    assign r1[4]  = csa_sum(x[6],  x[7],  x[8]);
    assign r1[5]  = csa_car(x[6],  x[7],  x[8]);
    assign r1[6]  = csa_sum(x[9],  x[10], x[11]);
    assign r1[7]  = csa_car(x[9],  x[10], x[11]);
    assign r1[8]  = csa_sum(x[12], x[13], x[14]);
    assign r1[9]  = csa_car(x[12], x[13], x[14]);
    assign r1[10] = x[15];

    logic [31:0] r2 [0:7];
    assign r2[0] = csa_sum(r1[0], r1[1], r1[2]);
    assign r2[1] = csa_car(r1[0], r1[1], r1[2]);
    assign r2[2] = csa_sum(r1[3], r1[4], r1[5]);
    assign r2[3] = csa_car(r1[3], r1[4], r1[5]);
    assign r2[4] = csa_sum(r1[6], r1[7], r1[8]);
    assign r2[5] = csa_car(r1[6], r1[7], r1[8]);
    assign r2[6] = r1[9];
    assign r2[7] = r1[10];

    logic [31:0] r3 [0:5];
    assign r3[0] = csa_sum(r2[0], r2[1], r2[2]);
    assign r3[1] = csa_car(r2[0], r2[1], r2[2]);
    assign r3[2] = csa_sum(r2[3], r2[4], r2[5]);
    assign r3[3] = csa_car(r2[3], r2[4], r2[5]);
    assign r3[4] = r2[6];
    assign r3[5] = r2[7];

    logic [31:0] sA [0:5];
    logic        vA;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 6; i++) sA[i] <= '0;
            vA <= 1'b0;
        end else begin
            for (int i = 0; i < 6; i++) sA[i] <= r3[i];
            vA <= in_valid;
        end
    end

    // STAGE B — combinational 6 -> 2
    logic [31:0] r4 [0:3];
    assign r4[0] = csa_sum(sA[0], sA[1], sA[2]);
    assign r4[1] = csa_car(sA[0], sA[1], sA[2]);
    assign r4[2] = csa_sum(sA[3], sA[4], sA[5]);
    assign r4[3] = csa_car(sA[3], sA[4], sA[5]);

    logic [31:0] r5 [0:2];
    assign r5[0] = csa_sum(r4[0], r4[1], r4[2]);
    assign r5[1] = csa_car(r4[0], r4[1], r4[2]);
    assign r5[2] = r4[3];

    logic [31:0] r6 [0:1];
    assign r6[0] = csa_sum(r5[0], r5[1], r5[2]);
    assign r6[1] = csa_car(r5[0], r5[1], r5[2]);

    logic [31:0] sB [0:1];
    logic        vB;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sB[0] <= '0; sB[1] <= '0; vB <= 1'b0;
        end else begin
            sB[0] <= r6[0];
            sB[1] <= r6[1];
            vB    <= vA;
        end
    end

    // STAGE C — final CPA (one signed 32-bit add, mapped to DSP58)
    (* USE_DSP = "yes" *) logic signed [31:0] sum_cpa;
    assign sum_cpa = $signed(sB[0]) + $signed(sB[1]);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sum_out   <= '0;
            out_valid <= 1'b0;
        end else begin
            sum_out   <= sum_cpa;
            out_valid <= vB;
        end
    end

endmodule

`default_nettype wire
