// Iterative integer inverse-RMS scale engine.
//
// radicand = ((sumsq>>4) + EPS_BUCKET)<<12, stored in 64 bits.
// Compute floor(sqrt(radicand)) in 32 steps, then floor(2^24/root) in
// 25 restoring-division steps. Clamp to the signed SCALE_WIDTH maximum.
// A zero/tiny root bypasses division and returns the maximum immediately.
// The caller scales sumsq to account for vector length and activation format.

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module rmsnorm_scale_int #(
    parameter int SCALE_WIDTH = 16,
    parameter longint unsigned EPS_BUCKET = 1
) (
    input  wire                clk,
    input  wire                resetn,
    input  wire                start,
    input  wire  [63:0]        sumsq,
    output logic               busy,
    output logic               done,
    output logic signed [SCALE_WIDTH-1:0] scale_q12
);

    localparam logic [1:0]
        ST_IDLE = 2'd0,
        ST_SQRT = 2'd1,
        ST_DIV  = 2'd2;

    logic [1:0]  state_reg;
    logic [63:0] radicand_reg;
    logic [65:0] rem_reg;
    logic [32:0] root_reg;
    logic [31:0] denom_reg;
    logic [5:0]  iter_reg;
    logic [32:0] div_rem_reg;
    logic [24:0] div_quot_reg;
    logic [5:0]  div_bit_reg;

    logic [65:0] rem_next;
    logic [32:0] root_shift;
    logic [33:0] cand_next;
    logic [32:0] div_rem_next;
    logic [24:0] div_quot_next;
    logic        div_in_bit;

    localparam logic [24:0] DIV_NUM_U25   = 25'd16777216;
    localparam logic [24:0] SCALE_MAX_U25 =
        (25'd1 << (SCALE_WIDTH-1)) - 25'd1;
    localparam logic signed [SCALE_WIDTH-1:0] SCALE_MAX =
        {1'b0, {(SCALE_WIDTH-1){1'b1}}};
    localparam logic [32:0] RAIL_DENOM =
        33'd1 << (25-SCALE_WIDTH);

    function automatic signed [SCALE_WIDTH-1:0] sat_scale_width;
        input logic [24:0] value;
        begin
            if (value > SCALE_MAX_U25)
                sat_scale_width = SCALE_MAX;
            else
                sat_scale_width = value[SCALE_WIDTH-1:0];
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (SCALE_WIDTH < 16 || SCALE_WIDTH > 25)
            $error("rmsnorm_scale_int SCALE_WIDTH=%0d outside supported 16..25",
                   SCALE_WIDTH);
    end
`endif

    always_ff @(posedge clk) begin
        if (!resetn) begin
            state_reg    <= ST_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            scale_q12    <= '0;
            radicand_reg <= 64'd0;
            rem_reg      <= 66'd0;
            root_reg     <= 33'd0;
            denom_reg    <= 32'd0;
            iter_reg     <= 6'd0;
            div_rem_reg  <= 33'd0;
            div_quot_reg <= 25'd0;
            div_bit_reg  <= 6'd0;
        end else begin
            done <= 1'b0;

            case (state_reg)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        // Bucket the mean-square input and add epsilon before sqrt.
                        radicand_reg <= (((sumsq >> 4) + EPS_BUCKET) << 12);
                        rem_reg <= 66'd0;
                        root_reg <= 33'd0;
                        iter_reg <= 6'd31;
                        state_reg <= ST_SQRT;
                    end
                end

                ST_SQRT: begin
                    rem_next   = (rem_reg << 2) | ((radicand_reg >> (iter_reg * 2)) & 64'd3);
                    root_shift = root_reg << 1;
                    cand_next  = (root_shift << 1) | 34'd1;

                    if (rem_next >= cand_next) begin
                        rem_reg  <= rem_next - cand_next;
                        root_reg <= root_shift + 33'd1;
                    end else begin
                        rem_reg  <= rem_next;
                        root_reg <= root_shift;
                    end

                    if (iter_reg == 6'd0) begin
                        denom_reg <= (rem_next >= cand_next) ? (root_shift + 33'd1) : root_shift;
                        if (((rem_next >= cand_next) ? (root_shift + 33'd1) : root_shift) == 33'd0 ||
                            ((rem_next >= cand_next) ? (root_shift + 33'd1) : root_shift) <= RAIL_DENOM) begin
                            scale_q12 <= SCALE_MAX;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state_reg <= ST_IDLE;
                        end else begin
                            div_rem_reg  <= 33'd0;
                            div_quot_reg <= 25'd0;
                            div_bit_reg  <= 6'd24;
                            state_reg    <= ST_DIV;
                        end
                    end else begin
                        iter_reg <= iter_reg - 6'd1;
                    end
                end

                ST_DIV: begin
                    // Restoring divide; the final quotient is width-saturated below.
                    div_in_bit     = (div_bit_reg == 6'd24);
                    div_rem_next   = {div_rem_reg[31:0], div_in_bit};
                    div_quot_next  = div_quot_reg;

                    if (div_rem_next >= {1'b0, denom_reg}) begin
                        div_rem_next = div_rem_next - {1'b0, denom_reg};
                        div_quot_next[div_bit_reg] = 1'b1;
                    end

                    div_rem_reg  <= div_rem_next;
                    div_quot_reg <= div_quot_next;

                    if (div_bit_reg == 6'd0) begin
                        scale_q12 <= sat_scale_width(div_quot_next);
                        busy <= 1'b0;
                        done <= 1'b1;
                        state_reg <= ST_IDLE;
                    end else begin
                        div_bit_reg <= div_bit_reg - 6'd1;
                    end
                end

                default: begin
                    state_reg <= ST_IDLE;
                    busy <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
