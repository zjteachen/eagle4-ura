// =====================================================================
// score_max_tracker.sv
// Running max accumulator for streamed attention scores.
// =====================================================================
//
// Per-cycle: if `valid`, compare incoming `score` against the running
// max and update if greater.  `clear` resets to the most-negative
// signed value (so first valid score becomes the max).
//
// Used as the first pass of softmax over ctx_len attention scores
// (one head):  pass 1 finds the max, pass 2 computes exp(score - max).
//
// Single 32-bit compare + 32-bit register.  Trivial cost.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module score_max_tracker (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,           // synchronous clear to -INF
    input  wire                  valid,
    input  wire signed [31:0]    score,
    output reg  signed [31:0]    running_max
);

    // Most-negative signed INT32 = 0x80000000
    localparam logic signed [31:0] NEG_INF = 32'sh80000000;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running_max <= NEG_INF;
        end else if (clear) begin
            running_max <= NEG_INF;
        end else if (valid) begin
            if ($signed(score) > running_max) running_max <= score;
        end
    end

endmodule

`default_nettype wire
