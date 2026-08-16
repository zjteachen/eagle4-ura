// =====================================================================
// silu_mul.sv
// SiLU(gate) × up = gate × sigmoid(gate) × up   in Q4.12 fixed-point.
// =====================================================================
//
// Per EAGLE4's MLP intermediate stage:
//   silu_out[i] = (gate[i] * sigmoid(gate[i])) * up[i]
//
// Pipeline:
//   stage A (1 cyc): silu_q = sat_q12( (gate * sigmoid(gate)) >> 12 )
//   stage B (1 cyc): out_q  = sat_q12( (silu_q * up) >> 12 )
//
// Total latency: 2 cycles.  Uses sigmoid_pwl_q12 (combinational LUT) plus
// 2 × INT16×INT16 multiplies (DSP58 INT mode).
//
// Cost estimate:
//   - sigmoid_pwl_q12: ~50 LUT
//   - 2 mul + 2 sat: ~80 LUT + 2 DSP58
//   - 2 register stages: ~64 FF
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module silu_mul #(
    // FIX-A (S7C_FRAC_CONFIG_AUDIT §4): the FINAL silu*up product (real range
    // ≈ ±91) saturates at Q4.12 (±8.0). The composite-proven g7p seam is Q8.8
    // (frac=8, golden g7_silu absmax 23370 = 91.3, sat=0). OUT_FRAC selects the
    // OUTPUT binary point of the stage-B product. Default 12 = legacy Q4.12
    // (preserves any isolated silu_mul golden test); the top instantiates with
    // OUT_FRAC=8 so the silu output spans ±128 without clipping.
    // Stage-A (gate*sigmoid) stays Q4.12 — silu(gate) itself is in [-0.28, gate]
    // which the gate_q Q4.12 range already bounds; only the product overflows.
    parameter int OUT_FRAC = 12
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                en,
    input  wire signed [15:0]  gate_q,        // Q4.12 signed
    input  wire signed [15:0]  up_q,          // Q4.12 signed
    output reg  signed [15:0]  out_q,         // Q(16-OUT_FRAC).OUT_FRAC signed
    output reg                 valid
);

    // ---- sigmoid LUT (combinational) ----
    wire [15:0] sigmoid_q;
    sigmoid_pwl_q12 u_sig (.x_q12(gate_q), .sigmoid_q12(sigmoid_q));

    // ---- saturating Q8.24 → Q4.12 (stage A: silu = gate*sigmoid) ----
    function automatic signed [15:0] sat_q12;
        input signed [31:0] v;
        begin
            if (v[31:27] != {5{v[31]}})
                sat_q12 = v[31] ? 16'sh8000 : 16'sd32767;
            else
                sat_q12 = v[27:12];
        end
    endfunction

    // ---- saturating Q8.24 → Q(OUT_FRAC) with round-to-nearest (stage B).
    // Shift the Q8.24 product right by (24-OUT_FRAC), add the half-LSB round
    // bit, then INT16-saturate. At OUT_FRAC=8 the result is Q8.8 (±128 range)
    // so the ±91 silu*up product no longer clips. Mirrors the inline
    // sat16-with-round requant idiom used at the R1/R8 seams.
    localparam int SHR_B = 24 - OUT_FRAC;     // 16 at OUT_FRAC=8
    function automatic signed [15:0] sat_out;
        input signed [31:0] v;
        reg signed [31:0]   r;
        begin
            // round-to-nearest: add 1<<(SHR_B-1) before the arithmetic shift
            r = (SHR_B > 0) ? ((v + (32'sd1 <<< (SHR_B-1))) >>> SHR_B)
                            :  v;
            if (r > 32'sd32767)        sat_out = 16'sd32767;
            else if (r < -32'sd32768)  sat_out = 16'sh8000;
            else                       sat_out = r[15:0];
        end
    endfunction

    // ---- stage A: silu_q = gate * sigmoid(gate) >> 12 ----
    (* USE_DSP = "yes" *)
    wire signed [31:0] silu_q24_comb = $signed(gate_q) * $signed({1'b0, sigmoid_q});

    reg signed [15:0]  silu_q_reg;
    reg signed [15:0]  up_q_pipe;
    reg                en_pipe;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            silu_q_reg <= '0;
            up_q_pipe  <= '0;
            en_pipe    <= 1'b0;
        end else begin
            en_pipe <= en;
            if (en) begin
                silu_q_reg <= sat_q12(silu_q24_comb);
                up_q_pipe  <= up_q;
            end
        end
    end

    // ---- stage B: out_q = silu_q * up_q, requantized to Q(OUT_FRAC) ----
    // silu_q_reg = Q4.12, up_q_pipe = Q4.12 -> product is Q8.24. sat_out shifts
    // it to OUT_FRAC (8 in the top) with rounding + INT16-sat, so the ±91 silu
    // output lands as Q8.8 without the old Q4.12 ±8.0 clip.
    (* USE_DSP = "yes" *)
    wire signed [31:0] out_q24_comb = $signed(silu_q_reg) * $signed(up_q_pipe);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_q <= '0;
            valid <= 1'b0;
        end else begin
            valid <= en_pipe;
            if (en_pipe) begin
                out_q <= sat_out(out_q24_comb);
            end
        end
    end

endmodule

`default_nettype wire
