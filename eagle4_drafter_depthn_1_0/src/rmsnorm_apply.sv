// =====================================================================
// rmsnorm_apply.sv
// RMSNorm phase 3: stream-multiply x × scale × gamma in Q4.12.
// =====================================================================
//
// Per cycle on `en`:
//   for i in 0..K_PAR-1:
//     stage A: x_scaled[i] = sat_q12((x[i] * scale_q12) >> 12)
//     stage B: out[i]      = sat_q12((x_scaled[i] * gamma[i]) >> 12)
//
// Caller pre-loads `scale_q12` from rmsnorm_scale_int, then streams
// (x_chunk, gamma_chunk) pairs over H/K_PAR cycles.
//
// 2-stage pipeline: ~3-cycle latency (input → output).
//
// Cost (K_PAR=16):
//   - 16 × DSP58 for stage A multiplies
//   - 16 × DSP58 for stage B multiplies
//   - 2 sat_q12 LUTs per lane × 16 = ~150 LUT
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module rmsnorm_apply #(
    parameter int K_PAR     = 16,
    parameter int A_WIDTH   = 16,
    // rmsnorm_fullh (2026-05-23): the scale-apply right-shift. Legacy = 12
    // (scale_q12 is Q4.12, a dimensionless gain ≤ 8.0). For small-RMS stages
    // (e.g. G1 input-embedding, RMS≈0.05 → gain≈20) the dimensionless gain
    // exceeds the Q4.12 ceiling, so the scale engine emits a Q(4+B).(12-B)
    // scale (B = headroom bits) and this first-stage shift is (12-B). The
    // ×gamma stage stays Q4.12 (gamma is Q4.12) so the OUTPUT remains Q4.12.
    // SCALE_SHR=12 reproduces the legacy datapath bit-for-bit (B=0).
    parameter int SCALE_SHR = 12,
    // G1_SATURATION_FIX (2026-05-26): width of the stage-A x_scaled intermediate.
    // The product (x·scale)>>SCALE_SHR was previously clamped to A_WIDTH=16 bits
    // (Q-format with only ~1 integer bit at the G1 SCALE_SHR=10), so the 4
    // largest-magnitude pre-norm hiddens (|x_scaled| up to 65312 on seed-42 G1)
    // SATURATED at ±32767, collapsing g1_h_norm at exactly those 4 elements.
    // Widening x_scaled to XS_WIDTH bits adds (XS_WIDTH-16) integer bits of
    // headroom so the intermediate no longer clips; the ×gamma stage-B output is
    // still saturated to A_WIDTH, so the OUTPUT Q-format is UNCHANGED (the gamma
    // stage's >>12 + sat_q16 keeps the seam at the same frac). XS_WIDTH=A_WIDTH
    // reproduces the legacy datapath bit-for-bit. 18 covers all observed seeds
    // (max needed = 17 bits @ seed-42/seed-7 G1) with one bit of margin; cost is
    // a 2-bit-wider register + DSP operand (~0 LUT/0 URAM/0 DSP delta).
    parameter int XS_WIDTH = A_WIDTH,
    parameter int SCALE_WIDTH = 16
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              en,
    input  wire signed [SCALE_WIDTH-1:0]     scale_q12,       // from rmsnorm_scale_int
    input  wire signed [K_PAR*A_WIDTH-1:0]   x_chunk,
    input  wire signed [K_PAR*A_WIDTH-1:0]   gamma_chunk,
    output logic signed [K_PAR*A_WIDTH-1:0]  out_chunk,
    output logic                             valid
);

    // Stage-A/Stage-B product widths.
    localparam int PS_WIDTH = A_WIDTH + SCALE_WIDTH;
    localparam int PB_WIDTH = XS_WIDTH + A_WIDTH;

    // Q4.12 saturating round-shift at the legacy fixed 12 (used for ×gamma).
    // Output is A_WIDTH (16-bit) — this is the single saturation point that sets
    // the seam Q-format, so widening XS_WIDTH does NOT change the output frac.
    function automatic signed [A_WIDTH-1:0] sat_q12;
        input signed [PB_WIDTH-1:0] v;
        begin
            // valid output occupies bits [A_WIDTH+12-1 : 12]. Overflow check:
            // bits [PB_WIDTH-1 : A_WIDTH+12-1] (the guard bits INCLUDING the
            // result MSB) must all equal the sign bit, else saturate. For
            // PB_WIDTH=32/A_WIDTH=16 this is v[31:27] — bit-identical to the
            // legacy check.
            if (v[PB_WIDTH-1 : A_WIDTH+12-1] != {(PB_WIDTH-(A_WIDTH+12-1)){v[PB_WIDTH-1]}})
                sat_q12 = v[PB_WIDTH-1] ? 16'sh8000 : 16'sd32767;
            else
                sat_q12 = v[A_WIDTH+12-1 : 12];
        end
    endfunction

    // Parameterized saturating right-shift by SCALE_SHR for the x×scale stage.
    // Result is carried at XS_WIDTH bits (>= A_WIDTH) so the stage-A intermediate
    // has (XS_WIDTH-A_WIDTH) extra integer bits of headroom and no longer clips
    // the few large-magnitude elements (G1_SATURATION_FIX). Guard bits
    // [31 : SCALE_SHR+XS_WIDTH] must all equal the sign bit, else saturate to the
    // XS_WIDTH-bit signed range.
    localparam signed [XS_WIDTH-1:0] XS_MAX =  (1 <<< (XS_WIDTH-1)) - 1;
    localparam signed [XS_WIDTH-1:0] XS_MIN = -(1 <<< (XS_WIDTH-1));
    function automatic signed [XS_WIDTH-1:0] sat_scale;
        input signed [PS_WIDTH-1:0] v;
        begin
            if (v[PS_WIDTH-1:SCALE_SHR+XS_WIDTH] !=
                {(PS_WIDTH-(SCALE_SHR+XS_WIDTH)){v[PS_WIDTH-1]}})
                sat_scale = v[PS_WIDTH-1] ? XS_MIN : XS_MAX;
            else
                sat_scale = v[SCALE_SHR+XS_WIDTH-1 -: XS_WIDTH];
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (SCALE_WIDTH < 16)
            $error("rmsnorm_apply SCALE_WIDTH=%0d must be >=16", SCALE_WIDTH);
        if (XS_WIDTH < A_WIDTH)
            $error("rmsnorm_apply XS_WIDTH=%0d must be >=A_WIDTH=%0d",
                   XS_WIDTH, A_WIDTH);
        if (PS_WIDTH <= SCALE_SHR + XS_WIDTH)
            $error("rmsnorm_apply invalid guard width PS=%0d SHR=%0d XS=%0d",
                   PS_WIDTH, SCALE_SHR, XS_WIDTH);
    end
`endif

    // Internal unpacked array — distinct drivers per lane (multi-driver-safe)
    logic signed [XS_WIDTH-1:0] x_scaled_arr [0:K_PAR-1];
    logic signed [A_WIDTH-1:0] out_arr      [0:K_PAR-1];
    logic                      en_a, en_b;
    logic signed [A_WIDTH-1:0] gamma_d1     [0:K_PAR-1];

    genvar i;
    generate
        for (i = 0; i < K_PAR; i = i + 1) begin : g_lane
            wire signed [A_WIDTH-1:0] x_i     = x_chunk    [i*A_WIDTH +: A_WIDTH];
            wire signed [A_WIDTH-1:0] gamma_i = gamma_chunk[i*A_WIDTH +: A_WIDTH];

            // Stage A
            (* USE_DSP = "yes" *)
            wire signed [PS_WIDTH-1:0] x_scale =
                $signed(x_i) * $signed(scale_q12);

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    x_scaled_arr[i] <= '0;
                    gamma_d1[i]     <= '0;
                end else if (en) begin
                    x_scaled_arr[i] <= sat_scale(x_scale);
                    gamma_d1[i]     <= gamma_i;
                end
            end

            // Stage B
            (* USE_DSP = "yes" *)
            wire signed [PB_WIDTH-1:0] x_gamma = $signed(x_scaled_arr[i]) * $signed(gamma_d1[i]);

            always_ff @(posedge clk) begin
                if (!rst_n) out_arr[i] <= '0;
                else if (en_a) out_arr[i] <= sat_q12(x_gamma);
            end
        end
    endgenerate

    // valid pipeline
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            en_a <= 1'b0; en_b <= 1'b0; valid <= 1'b0;
        end else begin
            en_a  <= en;
            en_b  <= en_a;
            valid <= en_b;
        end
    end

    // Single combinational concat array → packed output
    always_comb begin
        for (int k = 0; k < K_PAR; k++)
            out_chunk[k*A_WIDTH +: A_WIDTH] = out_arr[k];
    end

endmodule

`default_nettype wire
