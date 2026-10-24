// Streaming sum-of-squares accumulator for K_PAR lanes.
// Each en request squares the current chunk, reduces the lanes, and adds the
// result to the 64-bit running sum. clear resets the running sum.
//
// Three registered stages separate lane products, reduction, and accumulation;
// en_a/en_b carry the request through those stages. For A_WIDTH=16 and H=3072,
// the mathematical maximum is H×2^30 < 2^42, within the 64-bit output.

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

    // Stage A: register per-lane signed 32-bit squares.
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

    // Stage B: sum the registered squares into a 64-bit chunk total.
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
