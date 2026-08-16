// =====================================================================
// gptq2_dequant16.sv
// One-cycle (combinational) GPTQ-2bit dequant block — 16 lanes wide.
// =====================================================================
//
// Takes:
//   qw_word  — one int32 of qweight  [K/16, N] holding 16 raw 2-bit weights
//              for ONE output column n, 16 consecutive input rows k.
//              lane i corresponds to k = k_chunk*16 + i.
//   qz_word  — one int32 of qzeros   [K/128, N/16] holding 16 raw 2-bit zeros
//              for ONE group g, 16 consecutive output columns n.
//              The zero for THIS output column is at index n_idx = n % 16.
//   n_idx    — 4-bit selector into qz_word (which of the 16 zeros to use).
//
// Produces:
//   w_s_packed — 16 × 3-bit signed dequantized weights, packed.
//                Lane i = w_s_packed[i*3 +: 3]  ∈ [-4..+2] (3-bit two's-comp).
//                w_signed[i] = w_raw[i] - (z_raw + 1).
//
// Numerical contract: matches reference/src/gptq.cpp gptq2_gemv unpack stage
// (sym=True, group_size=128, pack=16, packed-along-input qweight, packed-along
//  -output qzeros).
//
// This is a pure-combinational lookup + subtract — no clock, no register.
// Wraps into the lane MAC chain just before mac_2bit_* via w_raw + z_raw.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module gptq2_dequant16 (
    input  wire [31:0] qw_word,
    input  wire [31:0] qz_word,
    input  wire [3:0]  n_idx,
    output wire [47:0] w_s_packed       // 16 × 3-bit signed
);

    // ----------------------------------------------------------------
    // Pick the per-output zero from qz_word (1 of 16 raw 2-bit values).
    // ----------------------------------------------------------------
    logic [1:0] raw_z;
    always_comb begin
        raw_z = qz_word[n_idx*2 +: 2];
    end

    // (z_raw + 1) ∈ {1..4}, 3-bit unsigned
    logic [2:0] zp1;
    assign zp1 = {1'b0, raw_z} + 3'd1;

    // ----------------------------------------------------------------
    // 16 parallel lanes: extract raw_w[i] and subtract (z+1).
    // Result is 3-bit signed in [-4..+2].
    // ----------------------------------------------------------------
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : g_lane
            wire [1:0]        raw_w   = qw_word[i*2 +: 2];
            wire signed [3:0] w_diff  = $signed({2'b00, raw_w}) - $signed({1'b0, zp1});
            assign w_s_packed[i*3 +: 3] = w_diff[2:0];   // truncate; range guarantees fit
        end
    endgenerate

endmodule

`default_nettype wire
