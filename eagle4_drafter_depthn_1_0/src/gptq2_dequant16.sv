// Combinational decoding of 16 packed two-bit weights.
// qw_word[2*i +: 2] holds input element i for one output column.
// qz_word holds 16 two-bit zero codes; n_idx selects this column's code.
// Each signed three-bit output is raw_weight - (raw_zero + 1), in [-4, 2].
// There is no scale multiplication or pipeline register in this module.

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
