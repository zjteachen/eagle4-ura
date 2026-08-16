// SILU-Q612-FIX (2026-06-06): ported into pdi6_s7c_g8 from silu_mathcheck.
// =====================================================================
// silu_layer_fix.sv  (FIX-SILU experiment copy of silu_layer.sv)
// K_PAR=16 wide. Gate/up carried at Q6.12 (18-bit per element) — FIX-1.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none
(* keep_hierarchy = "yes" *)
module silu_layer_fix #(
    parameter int K_PAR    = 16,
    parameter int IN_W     = 18,          // Q6.12 per-element input
    parameter int OUT_FRAC = 8
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              en,
    input  wire signed [K_PAR*IN_W-1:0]      gate_chunk,
    input  wire signed [K_PAR*IN_W-1:0]      up_chunk,
    output logic signed [K_PAR*16-1:0]       out_chunk,    // Q8.8 out
    output logic                             valid
);
    logic signed [15:0] out_arr [0:K_PAR-1];
    logic [K_PAR-1:0]   unit_valid;
    genvar i;
    generate
        for (i = 0; i < K_PAR; i = i + 1) begin : g_unit
            wire signed [IN_W-1:0] gate_i = gate_chunk[i*IN_W +: IN_W];
            wire signed [IN_W-1:0] up_i   = up_chunk[i*IN_W +: IN_W];
            wire signed [15:0]     out_i;
            wire                   v_i;
            silu_mul_fix #(.OUT_FRAC(OUT_FRAC), .IN_W(IN_W)) u_silu (
                .clk(clk), .rst_n(rst_n), .en(en),
                .gate_q(gate_i), .up_q(up_i), .out_q(out_i), .valid(v_i)
            );
            always_ff @(posedge clk) begin
                if (!rst_n) out_arr[i] <= '0;
                else if (v_i) out_arr[i] <= out_i;
            end
            assign unit_valid[i] = v_i;
        end
    endgenerate
    always_comb begin
        for (int k = 0; k < K_PAR; k++)
            out_chunk[k*16 +: 16] = out_arr[k];
    end
    always_ff @(posedge clk) begin
        if (!rst_n) valid <= 1'b0;
        else        valid <= unit_valid[0];
    end
endmodule
`default_nettype wire
