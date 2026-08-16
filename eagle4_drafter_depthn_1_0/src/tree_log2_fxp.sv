// =====================================================================
// tree_log2_fxp.sv -- IEEE-754 binary32 probability -> signed Q12.28 log2
// =====================================================================
//
// log2(p) for p in (0,1], returned in the tree_score_pkg score format.
//
// Method (exact-exponent + interpolated-mantissa, the standard split):
//     p   = 2^E * m,  m in [1,2)
//     log2(p) = E + log2(m)
//   E comes straight out of the fp32 exponent field -- EXACT, no error.
//   log2(m) comes from a 1024-entry Q0.28 LUT with LINEAR INTERPOLATION
//   between adjacent entries. Measured max |error| = 1.75e-07 in log2 units,
//   which is 1949x below the measured 3.4169e-04 decision floor (see
//   tree_score_pkg.sv for the full sizing argument and the evidence that
//   dropping the interpolation is UNSAFE despite passing the shallow gate).
//
// The LUT is a constant array in tree_log2_lut.svh, NOT a $readmemh bake --
// deliberately, because a procedurally-loaded array is the exact shape that
// synthesis silently drops with [Synth 8-3848] no-driver (lmdown_gemv.sv:174).
// A localparam array cannot be dropped.
//
// Latency: 2 cycles, fully pipelined, 1 result/cycle. No division, no
// normalisation loop -- fp32 arrives already normalised.
//
// Edge cases, all mapped to LOG2_FLOOR (-48.0) rather than to -inf, so the
// accumulator can never go NaN/saturate-wrap on a degenerate input:
//     p = 0, subnormal, negative, NaN, or +inf  -> LOG2_FLOOR
//     p > 1 (must not happen; a prob)           -> clamped to 0 (log2 1)
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_log2_fxp
    import tree_score_pkg::*;
(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire                          valid_in,
    input  wire [31:0]                   prob_f32,   // IEEE-754 binary32, expected (0,1]

    output logic                         valid_out,
    output logic signed [SCORE_W-1:0]    log2_q      // signed Q12.28
);

    `include "tree_log2_lut.svh"

    // ---------------- stage 0: decompose ----------------
    wire        s_sign =  prob_f32[31];
    wire [7:0]  s_exp  =  prob_f32[30:23];
    wire [22:0] s_man  =  prob_f32[22:0];

    // index = top 10 mantissa bits; fr = remaining 13 bits (weight 2^-13 of a LUT step)
    wire [9:0]  idx0   =  s_man[22:13];
    wire [12:0] fr0    =  s_man[12:0];

    wire        degen0 = s_sign | (s_exp == 8'd0) | (s_exp == 8'hFF);

    logic                        v1, degen1;
    logic signed [9:0]           e1;      // unbiased exponent, -126..127 fits in 10b signed
    logic [12:0]                 fr1;
    logic [28:0]                 lut_lo1, lut_hi1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v1 <= 1'b0; degen1 <= 1'b0; e1 <= '0; fr1 <= '0;
            lut_lo1 <= '0; lut_hi1 <= '0;
        end else begin
            v1      <= valid_in;
            degen1  <= degen0;
            e1      <= $signed({2'b00, s_exp}) - $signed(10'sd127);
            fr1     <= fr0;
            lut_lo1 <= LOG2_LUT[idx0];
            lut_hi1 <= LOG2_LUT[idx0 + 1];   // idx0+1 <= 1024, in range by construction
        end
    end

    // ---------------- stage 1: interpolate + assemble ----------------
    // delta <= 2^28 * log2(1+1/1024) = 378,009  -> 19 bits unsigned
    wire [18:0] delta  = lut_hi1[18:0] - lut_lo1[18:0];
    // delta * fr1 : 19 x 13 = 32 bits, >>13 recovers the Q0.28 increment
    wire [31:0] interp = ({19'd0, fr1} * {13'd0, delta}) >> 13;

    wire [28:0] mant_log = lut_lo1 + interp[28:0];          // log2(m) in Q0.28, [0, 2^28]

    // log2(p) = E*2^28 + log2(m)
    wire signed [SCORE_W-1:0] raw =
            ($signed(e1) <<< SCORE_F) + $signed({{(SCORE_W-29){1'b0}}, mant_log});

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            log2_q    <= LOG2_FLOOR;
        end else begin
            valid_out <= v1;
            if (degen1)                     log2_q <= LOG2_FLOOR;   // 0 / subnormal / NaN / inf / negative
            else if (raw > SCORE_ONE)       log2_q <= SCORE_ONE;    // p>1 cannot happen; clamp anyway
            else if (raw < LOG2_FLOOR)      log2_q <= LOG2_FLOOR;   // deeper than the floor
            else                            log2_q <= raw;
        end
    end

endmodule

`default_nettype wire
