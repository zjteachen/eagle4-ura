// Combinational Q/K/V requantization from per-head INT32 buffers to packed
// INT16 attention inputs. Element (head, dim) maps to head*D_HEAD + dim.
//
// Add half an output LSB, arithmetic-shift by G4_FRAC-G5_FRAC, then clamp
// to the signed 16-bit range. Requires SHR >= 1 and A_WIDTH = 16.
// The rounding addition is 32-bit; values near INT32_MAX can wrap before
// saturation. This module does not perform attention or rotary embedding.
//
// Defaults use a 13->10 fractional-bit conversion. eagle4_drafter overrides
// G4_FRAC with ACT_FRAC (12, or 11 with PRECISION_SAFE); its selected-head
// requantization path can bypass this full-vector seam.

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
    // One packed D_HEAD-element INT32 vector per head.
    input  wire signed [32*D_HEAD-1:0] q_full_buf [0:N_HEADS-1],
    input  wire signed [32*D_HEAD-1:0] k_full_buf [0:N_HEADS-1],
    input  wire signed [32*D_HEAD-1:0] v_full_buf [0:N_HEADS-1],

    // frac=10 INT16 attention inputs ([head][dim], flattened)
    output wire signed [OUT_W-1:0] q_in,
    output wire signed [OUT_W-1:0] k_in,
    output wire signed [OUT_W-1:0] v_in
);

    // Round ties toward positive infinity, then clamp to INT16.
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
