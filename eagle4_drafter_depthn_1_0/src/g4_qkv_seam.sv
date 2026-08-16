// =====================================================================
// g4_qkv_seam.sv  (S4 — G4 full-width QKV -> full 24-head attention seam)
// =====================================================================
//
// The S2 capture stores the FULL-WIDTH Q/K/V as 24 col-tiles of 128 INT32
// lanes each (q/k/v_full_buf[0:23], packed [32*128-1:0] per tile). The
// captured value per lane is the GEMV accumulator at the G4 OUTPUT Q-format
// = frac=ACT_FRAC = 13 (it is requant_int32(result_q) = round(result_q>>13)
// ~= round(real * 2^13); see scripts/m1b_g4_qkv_golden.py).
//
// The full 24-head attention (attention_full_mha.sv) consumes Q/K/V at the
// g5 Q-format = frac=10 (manifest frac_set.g5 / #190). So the inter-stage
// seam from G4 -> G5 is a fixed-point RIGHT-SHIFT by (G4_FRAC - G5_FRAC) = 3
// with round-half + INT16 saturation, applied per lane.
//
// Mapping: physical output column j = col_tile*128 + lane runs 0..3071, which
// is exactly head (j/128) dim (j%128) — i.e. col-tile T == head T, lane ==
// dim. The attention module's q_in/k_in/v_in are flattened
// [N_HEADS*D_HEAD*A_WIDTH] = [head][dim], element [head*128+dim] at bit
// (head*128+dim)*16. So this seam maps full_buf[T][lane] -> q_in[T*128+lane].
//
// This module is PURELY COMBINATIONAL (a fixed reshape+requant of the three
// 3072x INT32 buffers into three 3072x INT16 vectors). It is the S4 seam
// wiring; the actual attention compute lives in attention_full_mha.
//
// Multi-driver-safe: three independent combinational assigns, one per output;
// no state, no contention.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module g4_qkv_seam #(
    parameter int N_HEADS = 24,
    parameter int D_HEAD  = 128,
    parameter int A_WIDTH = 16,
    parameter int G4_FRAC = 13,            // captured full_buf Q-format
    parameter int G5_FRAC = 10,            // attention input Q-format
    parameter int SHR     = G4_FRAC - G5_FRAC,   // = 3
    parameter int H       = N_HEADS * D_HEAD,    // 3072
    parameter int OUT_W   = H * A_WIDTH          // 3072*16
) (
    // full-width captured buffers (S2 format: per head/col-tile, 128 INT32 lanes)
    input  wire signed [32*D_HEAD-1:0] q_full_buf [0:N_HEADS-1],
    input  wire signed [32*D_HEAD-1:0] k_full_buf [0:N_HEADS-1],
    input  wire signed [32*D_HEAD-1:0] v_full_buf [0:N_HEADS-1],

    // frac=10 INT16 attention inputs ([head][dim], flattened)
    output wire signed [OUT_W-1:0] q_in,
    output wire signed [OUT_W-1:0] k_in,
    output wire signed [OUT_W-1:0] v_in
);

    // round-half-up arithmetic right-shift of a signed INT32 by SHR, then
    // saturate to INT16. (matches m1b_g4_qkv_golden requant style: add half,
    // arithmetic >>, clip.) SHR>=1 expected.
    function automatic logic signed [A_WIDTH-1:0] rsh_sat;
        input signed [31:0] v;
        logic signed [31:0] r;
        begin
            r = (v + (32'sd1 <<< (SHR-1))) >>> SHR;     // round-half, arith shift
            if      (r >  32'sd32767) rsh_sat = 16'sd32767;
            else if (r < -32'sd32768) rsh_sat = 16'sh8000;
            else                       rsh_sat = r[A_WIDTH-1:0];
        end
    endfunction

    genvar hd, dm;
    generate
        for (hd = 0; hd < N_HEADS; hd = hd + 1) begin : g_head
            for (dm = 0; dm < D_HEAD; dm = dm + 1) begin : g_dim
                localparam int E = hd*D_HEAD + dm;   // flat element index
                assign q_in[E*A_WIDTH +: A_WIDTH] = rsh_sat(q_full_buf[hd][dm*32 +: 32]);
                assign k_in[E*A_WIDTH +: A_WIDTH] = rsh_sat(k_full_buf[hd][dm*32 +: 32]);
                assign v_in[E*A_WIDTH +: A_WIDTH] = rsh_sat(v_full_buf[hd][dm*32 +: 32]);
            end
        end
    endgenerate

endmodule

`default_nettype wire
