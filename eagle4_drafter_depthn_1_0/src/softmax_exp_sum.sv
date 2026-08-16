// =====================================================================
// softmax_exp_sum.sv
// Streaming exp_pwl accumulator (softmax pass 2).
// =====================================================================
//
// Inputs (per cycle, when valid):
//   - score (signed Q4.12, the per-position attention score)
//   - max   (signed Q4.12, the running max from pass 1)
// Output:
//   - exp_val   = exp_pwl(score - max)        (per cycle, Q4.12 unsigned)
//   - sum_so_far = Σ exp_val to date           (signed INT48 accumulator)
//
// On `clear`, sum_so_far resets to 0.
//
// Designed to be paired with score_max_tracker: pass 1 finds max,
// pass 2 (this module) computes exp(score - max) per cycle and
// accumulates sum_so_far.  After ctx_len cycles, sum_so_far holds
// the softmax denominator.  Pass 3 (separate divider, deferred) reads
// per-position exp_vals (buffered or recomputed) and divides by sum.
//
// Reuses exp_pwl_q12 primitive.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module softmax_exp_sum (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,
    input  wire                  valid_in,
    input  wire signed [31:0]    score_in,         // Q-format score
    input  wire signed [31:0]    max_q,            // running max (from pass 1)

    output reg         [31:0]    exp_val,          // per-cycle exp(score - max), Q4.12
    output reg                   exp_valid,        // 1-cycle delayed valid
    output reg  signed [47:0]    sum_so_far        // running sum of exp_val
);

    // delta = score - max  (≤ 0 since max >= all scores)
    wire signed [31:0] delta_comb = score_in - max_q;

    // exp_pwl_q12 (combinational LUT)
    logic [31:0] exp_q_comb;
    exp_pwl_q12 u_exp (.delta_q12(delta_comb), .exp_q12(exp_q_comb));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            exp_val     <= '0;
            exp_valid   <= 1'b0;
            sum_so_far  <= '0;
        end else if (clear) begin
            exp_val     <= '0;
            exp_valid   <= 1'b0;
            sum_so_far  <= '0;
        end else begin
            exp_valid <= valid_in;
            if (valid_in) begin
                exp_val    <= exp_q_comb;
                sum_so_far <= sum_so_far + $signed({16'b0, exp_q_comb});
            end
        end
    end

endmodule

`default_nettype wire
