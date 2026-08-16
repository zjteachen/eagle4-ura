// =====================================================================
// attn_v_combine.sv
// Streaming attn × V accumulator — final step of per-head attention.
// =====================================================================
//
// Computes:
//   out[d] = sum_{p=0..ctx_len-1}  attn[p] * V[p, d]      for each d
//
// Per cycle (on valid_in):
//   attn_q12 — Q4.12 attention weight for position p
//   v_vec    — D_HEAD × Q4.12 elements of V cache at position p
//   For each d: acc[d] += (attn_q12 * v_vec[d]) >> 12   (Q4.12 result)
//
// On clear: all accumulators reset to 0.
//
// Output (combinational from accumulators): out_vec, D_HEAD × INT32.
// Caller reads out_vec when all positions have been streamed in.
//
// Cost: D_HEAD × DSP58 (one MAC each), D_HEAD × 32-bit accumulator.
// For pilot D_HEAD=16: 16 DSP + 16 × 32-bit FF + ~250 LUT.
// Production D_HEAD=128: 128 DSP per head × 24 heads = 3072 DSP for the
// V-combine alone — exceeds V80's 10848 DSP budget at 28%, comfortable.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module attn_v_combine #(
    parameter int D_HEAD  = 16,
    parameter int A_WIDTH = 16
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              clear,
    input  wire                              valid_in,
    // B2' (ACCFIX bit1): round-half instead of floor on the per-position
    // (attn*V)>>>12 product. 0 = legacy floor, BIT-IDENTICAL. ctx=1 is
    // byte-identical even at 1: (4096*V+2048)>>12 == V for every signed V.
    input  wire                              round_en,
    input  wire signed [A_WIDTH-1:0]         attn_q12,            // Q4.12 weight
    input  wire signed [A_WIDTH*D_HEAD-1:0]  v_vec,               // D_HEAD × Q4.12
    output wire signed [32*D_HEAD-1:0]       out_vec              // D_HEAD × INT32 accumulators
);

    // D_HEAD parallel multiply-accumulate units
    logic signed [31:0] acc [0:D_HEAD-1];

    genvar d;
    generate
        for (d = 0; d < D_HEAD; d = d + 1) begin : g_mac
            wire signed [A_WIDTH-1:0] v_d   = v_vec[d*A_WIDTH +: A_WIDTH];
            (* USE_DSP = "yes" *) wire signed [31:0] prod = $signed(attn_q12) * $signed(v_d);
            // Truncate to maintain Q4.12 semantics: shift right 12 with sign-extend.
            // B2' (round_en=1): add the half constant 2048 BEFORE the shift =
            // round-half-up — kills the coherent −0.5 LSB/position DC floor
            // (ANGLE_B_NUMERIC_AT_CTX.md §B2'). |prod| <= 32767*32767 < 2^30,
            // so +2048 can never overflow the 32b product.
            wire signed [31:0] prod_r   = prod + 32'sd2048;
            wire signed [31:0] prod_q12 = round_en ? {{12{prod_r[31]}}, prod_r[31:12]}
                                                   : {{12{prod  [31]}}, prod  [31:12]};

            always_ff @(posedge clk) begin
                if (!rst_n)         acc[d] <= '0;
                else if (clear)     acc[d] <= '0;
                else if (valid_in)  acc[d] <= acc[d] + prod_q12;
            end

            assign out_vec[d*32 +: 32] = acc[d];
        end
    endgenerate

endmodule

`default_nettype wire
