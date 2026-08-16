// =====================================================================
// inv_q12.sv
// Iterative reciprocal: inv = 4096*4096 / sum_q12  →  Q4.12 signed.
// =====================================================================
//
// Used in softmax pass-3 to normalize:
//   inv = 1.0 / sum_of_exp           (computed ONCE per softmax)
//   attn[i] = exp_val[i] * inv       (per-position multiply, streamed)
//
// Implementation: 25-cycle iterative integer divide, mirroring the
// ST_DIV phase of TALOS-V2's rms_scale_engine.
//
// Input:  sum_q12 — signed 48-bit accumulator of exp_pwl values
//                   (typical range: positive small integers; saturates to
//                    32767 if sum is too small / inv would overflow)
// Output: inv_q12 — signed Q4.12 reciprocal, saturated to ±32767
//
// FSM:
//   ST_IDLE → ST_DIV → ST_IDLE
//   ~25 cycles per call.  Throughput: one inverse per softmax invocation.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module inv_q12 (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,
    input  wire signed [47:0]    sum_q12,           // softmax denominator
    output logic                 busy,
    output logic                 done,
    output logic signed [15:0]   inv_q12_out        // 1/sum * 4096^2, sat-to-16-bit
);

    typedef enum logic [0:0] { ST_IDLE = 1'd0, ST_DIV = 1'd1 } state_t;
    state_t state;

    logic [31:0] denom_reg;
    logic [32:0] div_rem;
    logic [24:0] div_quot;
    logic [5:0]  div_bit;

    logic [32:0] div_rem_next;
    logic [24:0] div_quot_next;
    logic        div_in_bit;

    function automatic signed [15:0] sat16;
        input signed [63:0] v;
        begin
            if (v > 64'sd32767)         sat16 = 16'sd32767;
            else if (v < -64'sd32768)   sat16 = 16'sh8000;
            else                        sat16 = v[15:0];
        end
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            busy          <= 1'b0;
            done          <= 1'b0;
            inv_q12_out   <= '0;
            denom_reg     <= '0;
            div_rem       <= '0;
            div_quot      <= '0;
            div_bit       <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        // Treat sum_q12 as a positive denominator (softmax sum is ≥ 0).
                        // Clamp to 32-bit range for the divider.
                        if (sum_q12 <= 0 || sum_q12 > 48'sd2147483647) begin
                            inv_q12_out <= 16'sd32767;
                            done <= 1'b1;
                            // stay in ST_IDLE
                        end else begin
                            denom_reg <= sum_q12[31:0];
                            div_rem   <= 33'd0;
                            div_quot  <= 25'd0;
                            div_bit   <= 6'd24;
                            busy      <= 1'b1;
                            state     <= ST_DIV;
                        end
                    end
                end

                ST_DIV: begin
                    div_in_bit    = (div_bit == 6'd24);
                    div_rem_next  = {div_rem[31:0], div_in_bit};
                    div_quot_next = div_quot;

                    if (div_rem_next >= {1'b0, denom_reg}) begin
                        div_rem_next  = div_rem_next - {1'b0, denom_reg};
                        div_quot_next[div_bit] = 1'b1;
                    end

                    div_rem  <= div_rem_next;
                    div_quot <= div_quot_next;

                    if (div_bit == 6'd0) begin
                        inv_q12_out <= sat16($signed({39'd0, div_quot_next}));
                        busy        <= 1'b0;
                        done        <= 1'b1;
                        state       <= ST_IDLE;
                    end else begin
                        div_bit <= div_bit - 6'd1;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
