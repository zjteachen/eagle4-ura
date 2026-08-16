// =====================================================================
// rope_pair_rotate.sv
// One RoPE rotation pair — Q4.12 fixed-point.
// =====================================================================
//
// Computes:
//   x_new = x*cos - y*sin
//   y_new = x*sin + y*cos
//
// All operands are signed Q4.12 (16-bit fixed point, range ±8).  Each
// multiply gives a 32-bit signed Q8.24 intermediate; we shift right 12
// (with saturation/truncation) to return to Q4.12.
//
// Compute path:
//   - 4 × INT16 × INT16 = 32-bit multiplies → Vivado can pack 2 into
//     each DSP58 (INT mode supports two 16×16 multiplies per slice in
//     pre-add/post-add configuration).  Realistic cost: 2-4 DSP58.
//   - 2 × INT32 adds → LUT carry chain
//   - Output shift-and-truncate to 16-bit Q4.12 → 0 logic
//
// 1-cycle latency (registered output).
//
// For a full head rotation, instantiate D_HEAD/2 = 64 of these in
// parallel (or time-multiplex over N cycles for an N-wide pipeline).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module rope_pair_rotate (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                en,
    input  wire signed [15:0]  x_in,
    input  wire signed [15:0]  y_in,
    input  wire signed [15:0]  cos_t,
    input  wire signed [15:0]  sin_t,
    output reg  signed [15:0]  x_out,
    output reg  signed [15:0]  y_out,
    output reg                 valid
);

    // Combinational 16×16 → 32-bit signed multiplies (DSP58 INT mode)
    wire signed [31:0] x_cos = $signed(x_in) * $signed(cos_t);
    wire signed [31:0] y_sin = $signed(y_in) * $signed(sin_t);
    wire signed [31:0] x_sin = $signed(x_in) * $signed(sin_t);
    wire signed [31:0] y_cos = $signed(y_in) * $signed(cos_t);

    // x_new = x*cos - y*sin   (Q8.24 → shift right 12 → Q4.12)
    // y_new = x*sin + y*cos
    wire signed [31:0] x_new_q24 = x_cos - y_sin;
    wire signed [31:0] y_new_q24 = x_sin + y_cos;

    // Saturating round-to-Q4.12: take bits [27:12].
    function automatic signed [15:0] sat_q12;
        input signed [31:0] v;
        begin
            if (v[31:27] != {5{v[31]}})  // overflow into bits >= 28
                sat_q12 = v[31] ? 16'sh8000 : 16'sd32767;
            else
                sat_q12 = v[27:12];
        end
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            x_out <= '0;
            y_out <= '0;
            valid <= 1'b0;
        end else begin
            valid <= en;
            if (en) begin
                x_out <= sat_q12(x_new_q24);
                y_out <= sat_q12(y_new_q24);
            end
        end
    end

endmodule

`default_nettype wire
