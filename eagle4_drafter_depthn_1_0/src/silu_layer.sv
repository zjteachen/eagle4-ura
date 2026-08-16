// =====================================================================
// silu_layer.sv
// K_PAR=16-wide SiLU + element-wise multiply.
// =====================================================================
//
// Per cycle on `en`, computes 16 outputs:
//   out_chunk[i] = silu_mul(gate_chunk[i], up_chunk[i])
//                = (gate[i] * sigmoid(gate[i])) * up[i]
//
// Caller (upstream FSM) time-multiplexes over H_INTER/K_PAR cycles
// to process the full MLP intermediate vector. For H_INTER=8192 and
// K_PAR=16, that's 512 cycles = 1.71 µs at 300 MHz (or 1.90 µs at 270 MHz).
//
// Cost (LANES=K_PAR=16):
//   16 × silu_mul × ~150 LUT  = ~2,400 LUT
//   16 × silu_mul × 2 DSP58   =     32 DSP58
//   16 × sigmoid_pwl_q12      = ~960 LUT  (combinational LUT)
//
// 2-cycle latency (matches silu_mul's 2-stage pipeline).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module silu_layer #(
    parameter int K_PAR    = 16,
    parameter int A_WIDTH  = 16,
    // FIX-A: output binary point of the silu*up product (forwarded to silu_mul).
    // 12 = legacy Q4.12 (clips ±8.0); the top uses 8 = Q8.8 (±128, no clip).
    parameter int OUT_FRAC = 12
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              en,
    input  wire signed [K_PAR*A_WIDTH-1:0]   gate_chunk,
    input  wire signed [K_PAR*A_WIDTH-1:0]   up_chunk,
    output logic signed [K_PAR*A_WIDTH-1:0]  out_chunk,
    output logic                             valid
);

    // Multi-driver-safe pattern: unpacked array → single concat
    logic signed [A_WIDTH-1:0] out_arr [0:K_PAR-1];
    logic [K_PAR-1:0]          unit_valid;

    genvar i;
    generate
        for (i = 0; i < K_PAR; i = i + 1) begin : g_unit
            wire signed [A_WIDTH-1:0] gate_i = gate_chunk[i*A_WIDTH +: A_WIDTH];
            wire signed [A_WIDTH-1:0] up_i   = up_chunk[i*A_WIDTH +: A_WIDTH];
            wire signed [A_WIDTH-1:0] out_i;
            wire                      v_i;

            silu_mul #(.OUT_FRAC(OUT_FRAC)) u_silu (
                .clk(clk), .rst_n(rst_n), .en(en),
                .gate_q(gate_i),
                .up_q(up_i),
                .out_q(out_i),
                .valid(v_i)
            );

            always_ff @(posedge clk) begin
                if (!rst_n) out_arr[i] <= '0;
                else if (v_i) out_arr[i] <= out_i;
            end
            assign unit_valid[i] = v_i;
        end
    endgenerate

    // Single combinational pack array → packed output
    always_comb begin
        for (int k = 0; k < K_PAR; k++)
            out_chunk[k*A_WIDTH +: A_WIDTH] = out_arr[k];
    end

    // All units run lockstep; sample unit 0's valid as the layer's valid
    always_ff @(posedge clk) begin
        if (!rst_n) valid <= 1'b0;
        else        valid <= unit_valid[0];
    end

endmodule

`default_nettype wire
