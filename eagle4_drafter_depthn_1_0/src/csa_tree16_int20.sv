// Three-stage Wallace carry-save reduction: 16 signed INT20 inputs to
// one signed INT26 sum, with one input vector accepted per clock.
// Stage A reduces 16->11->8->6 partials; B reduces 6->4->3->2;
// C resolves the final carry-propagate add. Valid follows the same pipeline.

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module csa_tree16_int20 (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          in_valid,
    input  wire signed [19:0]            prods [0:15],        // 16 INT20 products
    output reg  signed [25:0]            sum_out,             // INT26 sum
    output reg                           out_valid
);

    // ----------------------------------------------------------------
    // Sign-extend the 16 INT20 inputs to INT26 so the CSA can run at
    // the final width without losing the sign carry.
    // ----------------------------------------------------------------
    logic signed [25:0] x [0:15];
    always_comb for (int i = 0; i < 16; i++)
        x[i] = $signed({{6{prods[i][19]}}, prods[i]});

    // ----------------------------------------------------------------
    // 3:2 CSA helpers
    //   sum     = a ^ b ^ c                  (1 LUT/bit)
    //   carry   = majority(a,b,c) << 1       (1 LUT/bit + free shift)
    // ----------------------------------------------------------------
    function automatic [25:0] csa_sum (input [25:0] a, b, c);
        csa_sum = a ^ b ^ c;
    endfunction
    function automatic [25:0] csa_car (input [25:0] a, b, c);
        // carry from bit i lives at bit i+1; shift-left-1 captures that
        csa_car = ((a & b) | (a & c) | (b & c)) << 1;
    endfunction

    // ================================================================
    // STAGE A — combinational reduction 16 → 6
    // ================================================================
    // Round 1: 5 CSAs cover x[0..14], x[15] passes through → 11 partials
    logic [25:0] r1 [0:10];
    assign r1[0]  = csa_sum(x[0],  x[1],  x[2]);
    assign r1[1]  = csa_car(x[0],  x[1],  x[2]);
    assign r1[2]  = csa_sum(x[3],  x[4],  x[5]);
    assign r1[3]  = csa_car(x[3],  x[4],  x[5]);
    assign r1[4]  = csa_sum(x[6],  x[7],  x[8]);
    assign r1[5]  = csa_car(x[6],  x[7],  x[8]);
    assign r1[6]  = csa_sum(x[9],  x[10], x[11]);
    assign r1[7]  = csa_car(x[9],  x[10], x[11]);
    assign r1[8]  = csa_sum(x[12], x[13], x[14]);
    assign r1[9]  = csa_car(x[12], x[13], x[14]);
    assign r1[10] = x[15];

    // Round 2: 3 CSAs cover r1[0..8], r1[9..10] pass through → 8 partials
    logic [25:0] r2 [0:7];
    assign r2[0] = csa_sum(r1[0], r1[1], r1[2]);
    assign r2[1] = csa_car(r1[0], r1[1], r1[2]);
    assign r2[2] = csa_sum(r1[3], r1[4], r1[5]);
    assign r2[3] = csa_car(r1[3], r1[4], r1[5]);
    assign r2[4] = csa_sum(r1[6], r1[7], r1[8]);
    assign r2[5] = csa_car(r1[6], r1[7], r1[8]);
    assign r2[6] = r1[9];
    assign r2[7] = r1[10];

    // Round 3: 2 CSAs cover r2[0..5], r2[6..7] pass through → 6 partials
    logic [25:0] r3 [0:5];
    assign r3[0] = csa_sum(r2[0], r2[1], r2[2]);
    assign r3[1] = csa_car(r2[0], r2[1], r2[2]);
    assign r3[2] = csa_sum(r2[3], r2[4], r2[5]);
    assign r3[3] = csa_car(r2[3], r2[4], r2[5]);
    assign r3[4] = r2[6];
    assign r3[5] = r2[7];

    // Pipeline register A→B
    logic [25:0] sA [0:5];
    logic        vA;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 6; i++) sA[i] <= '0;
            vA <= 1'b0;
        end else begin
            for (int i = 0; i < 6; i++) sA[i] <= r3[i];
            vA <= in_valid;
        end
    end

    // ================================================================
    // STAGE B — combinational reduction 6 → 2
    // ================================================================
    // Round 4: 2 CSAs cover sA[0..5] → 4 partials
    logic [25:0] r4 [0:3];
    assign r4[0] = csa_sum(sA[0], sA[1], sA[2]);
    assign r4[1] = csa_car(sA[0], sA[1], sA[2]);
    assign r4[2] = csa_sum(sA[3], sA[4], sA[5]);
    assign r4[3] = csa_car(sA[3], sA[4], sA[5]);

    // Round 5: 1 CSA cover r4[0..2], r4[3] passes through → 3 partials
    logic [25:0] r5 [0:2];
    assign r5[0] = csa_sum(r4[0], r4[1], r4[2]);
    assign r5[1] = csa_car(r4[0], r4[1], r4[2]);
    assign r5[2] = r4[3];

    // Round 6: 1 CSA reduces r5 → 2 partials
    logic [25:0] r6 [0:1];
    assign r6[0] = csa_sum(r5[0], r5[1], r5[2]);
    assign r6[1] = csa_car(r5[0], r5[1], r5[2]);

    // Pipeline register B→C
    logic [25:0] sB [0:1];
    logic        vB;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sB[0] <= '0; sB[1] <= '0; vB <= 1'b0;
        end else begin
            sB[0] <= r6[0];
            sB[1] <= r6[1];
            vB    <= vA;
        end
    end

    // ================================================================
    // STAGE C — final CPA  (one signed 26-bit add)
    // ================================================================
    // Request DSP implementation of the final carry-propagate add.
    (* USE_DSP = "yes" *) logic signed [25:0] sum_cpa;
    assign sum_cpa = $signed(sB[0]) + $signed(sB[1]);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sum_out   <= '0;
            out_valid <= 1'b0;
        end else begin
            sum_out   <= sum_cpa;
            out_valid <= vB;
        end
    end

endmodule

`default_nettype wire
