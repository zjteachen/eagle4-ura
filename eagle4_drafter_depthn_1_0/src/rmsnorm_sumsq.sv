// =====================================================================
// rmsnorm_sumsq.sv
// Streaming K_PAR=16-wide sum-of-squares accumulator for RMSNorm phase 1.
// =====================================================================
//
// Per cycle (on `en`): squares 16 input elements, adds to running sum.
// After H/K_PAR cycles of `en`, `sumsq_out` holds Σ x[i]² for i=0..H-1.
//
// On `clear`: sumsq_out → 0.
//
// Output width 64-bit: max |x[i]|=32768, squared=2^30, summed H=3072 times
// = 2^42 ≈ 4.4 TB. Fits in 48-bit, use 64-bit for safety margin.
//
// Pipeline:
//   stage A (1 cyc): per-lane square (16 × DSP58)
//   stage B (1 cyc): tree-reduce 16 squares to one sum_chunk
//   stage C (1 cyc): accumulate sum_chunk into running sumsq
//
// Total per H=3072: 3 + 192 = 195 cycles + drain = ~200 cycles
//   ≈ 0.74 µs at 270 MHz.
//
// Cost:
//   16 × DSP58 (signed×signed squarers)
//   ~256 LUT for tree-reduce + 64-bit accumulator
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module rmsnorm_sumsq #(
    parameter int K_PAR   = 16,
    parameter int A_WIDTH = 16
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              clear,
    input  wire                              en,
    input  wire signed [K_PAR*A_WIDTH-1:0]   x_chunk,
    output logic [63:0]                      sumsq_out
);

    // Stage A: per-lane squares (combinational mults from x_chunk)
    logic signed [31:0] sq [0:K_PAR-1];
    logic               en_a;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < K_PAR; i++) sq[i] <= '0;
            en_a <= 1'b0;
        end else begin
            for (int i = 0; i < K_PAR; i++) begin
                logic signed [A_WIDTH-1:0] x_i;
                x_i = x_chunk[i*A_WIDTH +: A_WIDTH];
                sq[i] <= $signed(x_i) * $signed(x_i);
            end
            en_a <= en;
        end
    end

    // Stage B: tree-reduce 16 squares to one sum (5 levels of pairs)
    // 16→8→4→2→1, each level combinational
    logic [63:0] sum_chunk;
    logic        en_b;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sum_chunk <= '0;
            en_b      <= 1'b0;
        end else begin
            // Sign-extend each sq[i] to 64-bit and sum them all
            logic signed [63:0] acc;
            acc = 0;
            for (int i = 0; i < K_PAR; i++) acc += $signed({{32{sq[i][31]}}, sq[i]});
            sum_chunk <= acc[63:0];
            en_b      <= en_a;
        end
    end

    // Stage C: accumulate into running sumsq
    always_ff @(posedge clk) begin
        if (!rst_n)         sumsq_out <= '0;
        else if (clear)     sumsq_out <= '0;
        else if (en_b)      sumsq_out <= sumsq_out + sum_chunk;
    end

endmodule

`default_nettype wire
