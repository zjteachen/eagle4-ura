// =====================================================================
// g8_dot_fp16_wide.sv  —  WIDE-CONSUME fp16 dot (P parallel lanes/cycle).
//
// THE GRAND-BUILD G8 CONSUME-PATH WIDENER (widebeat agent abd633).
//   The legacy g8_dot_fp16 consumes ONE fp16-pair per (other) cycle (II=2) and
//   is the 93.4%-of-forward bottleneck (consume-bound, NOT AR-latency-bound —
//   ledger agent a15a31).  This module consumes P fp16-pairs PER CYCLE:
//     P parallel fp16 multipliers (each = the SAME g8_fp16_pkg::fp16_mul that
//     the scalar dot uses, bit-identical product per lane) -> P fp16->fp32
//     widens -> a balanced fp32 ADDER TREE (clog2(P) levels, each level's
//     fp32 add SPLIT into align/raw-add (a) and normalize/round (b) registered
//     sub-stages to close timing) -> ONE fp32 running accumulator at II=2
//     (acc += tree_sum, left-to-right over GROUPS).
//
// NUMERIC CONTRACT (re-association — NOT bit-exact to the scalar dot):
//   scalar g8_dot_fp16 : acc = (((p0+p1)+p2)+p3)+...    (strict left-to-right)
//   this wide dot       : acc = acc + balanced_tree(p[g..g+P-1]) per group g
//   fp32 add is NOT associative -> different fp32 score bits on rows where the
//   rounding diverges.  PRODUCTS p_h are bit-identical (same fp16_mul).  The
//   split fp32_add_b(fp32_add_a(x,y)) == fp32_add(x,y) EXACTLY, so the split is
//   numerically transparent; the ONLY re-association is the tree vs the chain.
//   => per-row score L_inf!=0 vs the scalar dot, top-K_F FRONTIER preserved on
//      seed-42 (339/512 rows differ in fp32 LSBs; top-8 == golden).  The wide
//      golden CONTRACT is FRONTIER-EXACT (re-baselined per ref/g8_dot_wide_ref.py).
//
// TIMING (why the split): the un-split tree had a single combinational fp32_add
//   per level (w_prod32 -> nxt) = 63 logic levels / 15.4 ns -> WNS -5.826 @100M.
//   Splitting each fp32_add into a (align+raw-add) and b (normalize+round)
//   registered stage halves the per-level logic depth.  The acc recurrence is
//   the only feedback; it runs at II=2 (groups presented every OTHER cycle), so
//   effective consume = P/2 fp16/cyc (=8 at P=16), still ~16x the scalar dot.
//
// INTERFACE (group-streaming, II=2):
//   - pulse `clear` with the FIRST group of a row (seeds acc from +0.0).
//   - assert `in_valid` with P (a,b) fp16 pairs per group; pulse `in_last` on
//     the LAST group of the row.  Present groups every OTHER cycle (II=2).
//   - `out_valid` pulses when `score_fp32` holds the final row sum.
//
// MULTI-DRIVER-SAFE: every reg written by exactly one always_ff; score_fp32 is
//   a combinational read of acc.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

import g8_fp16_pkg::*;

(* keep_hierarchy = "yes" *)
module g8_dot_fp16_wide #(
    parameter int  P        = 16,          // lanes consumed per cycle (pow2)
    parameter bit  MREG     = 1'b1         // register raw DSP product (timing)
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                clear,      // first group of a row
    input  wire                in_valid,
    input  wire                in_last,    // last group of a row
    input  wire [P*16-1:0]     a_fp16,     // P fp16 'a' lanes (gathered HBM)
    input  wire [P*16-1:0]     b_fp16,     // P fp16 'b' lanes (nrm activation)
    output logic               out_valid,
    output logic [31:0]        score_fp32
);
    localparam int LV = $clog2(P);         // adder-tree levels

    // ---------------- stage M: P fp16 multiplies (+ optional MREG) -----------
    logic [15:0] prod16 [0:P-1];
    genvar gl;
    generate
        for (gl=0; gl<P; gl++) begin : g_mul
            logic [49:0] ir_c;
            always_comb ir_c = fp16_mul_a(a_fp16[gl*16 +: 16], b_fp16[gl*16 +: 16]);
            if (MREG) begin : g_mreg
                logic [49:0] ir_r;
                always_ff @(posedge clk) ir_r <= ir_c;
                always_comb prod16[gl] = fp16_mul_b(ir_r);
            end else begin : g_nomreg
                always_comb prod16[gl] = fp16_mul_b(ir_c);
            end
        end
    endgenerate

    // control pipelined alongside MREG (one cycle when MREG=1, else zero)
    logic m_valid, m_last, m_clear;
    generate
    if (MREG) begin : g_ctl_mreg
        always_ff @(posedge clk) begin
            if (!rst_n) begin m_valid<=1'b0; m_last<=1'b0; m_clear<=1'b0; end
            else begin
                m_valid <= in_valid;
                m_last  <= in_valid & in_last;
                m_clear <= in_valid & clear;
            end
        end
    end else begin : g_ctl_nomreg
        always_comb begin
            m_valid = in_valid; m_last = in_valid & in_last; m_clear = in_valid & clear;
        end
    end
    endgenerate

    // ---------------- stage W: P fp16->fp32 widen -> REGISTER -----------------
    logic [31:0] w_prod32 [0:P-1];
    logic        w_valid, w_last, w_clear;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            w_valid<=1'b0; w_last<=1'b0; w_clear<=1'b0;
            for (int i=0;i<P;i++) w_prod32[i]<=32'd0;
        end else begin
            for (int i=0;i<P;i++) w_prod32[i] <= fp16_to_fp32(prod16[i]);
            w_valid <= m_valid; w_last <= m_last; w_clear <= m_clear;
        end
    end

    // ---------------- balanced fp32 adder tree, SPLIT a/b per level ----------
    // Each level: NIN -> NOUT=NIN/2.  Two registered sub-stages per level:
    //   sub-a: ir[i] <= fp32_add_a(in[2i], in[2i+1])   (align + raw add)
    //   sub-b: out[i]<= fp32_add_b(ir[i])              (normalize + round)
    // => 2 register stages per level, control rides through both.
    // t_lvl[0] = widened products (registered in stage W).
    logic [31:0] t_lvl [0:LV][0:P-1];
    logic        t_valid [0:LV];
    logic        t_last  [0:LV];
    logic        t_clear [0:LV];
    always_comb begin
        for (int i=0;i<P;i++) t_lvl[0][i] = w_prod32[i];
        t_valid[0]=w_valid; t_last[0]=w_last; t_clear[0]=w_clear;
    end

    genvar lvl;
    generate
        for (lvl=0; lvl<LV; lvl++) begin : g_tree
            localparam int NIN  = P >> lvl;
            localparam int NOUT = NIN >> 1;
            // sub-a registers (raw 72b IR) + control
            logic [71:0] ir_a [0:NOUT-1];
            logic        av, al, ac;
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    av<=1'b0; al<=1'b0; ac<=1'b0;
                    for (int i=0;i<NOUT;i++) ir_a[i]<=72'd0;
                end else begin
                    for (int i=0;i<NOUT;i++)
                        ir_a[i] <= fp32_add_a(t_lvl[lvl][2*i], t_lvl[lvl][2*i+1]);
                    av<=t_valid[lvl]; al<=t_last[lvl]; ac<=t_clear[lvl];
                end
            end
            // sub-b registers (normalized 32b) + control
            logic [31:0] nxt [0:NOUT-1];
            logic        bv, bl, bc;
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    bv<=1'b0; bl<=1'b0; bc<=1'b0;
                    for (int i=0;i<NOUT;i++) nxt[i]<=32'd0;
                end else begin
                    for (int i=0;i<NOUT;i++) nxt[i] <= fp32_add_b(ir_a[i]);
                    bv<=av; bl<=al; bc<=ac;
                end
            end
            always_comb begin
                for (int i=0;i<NOUT;i++) t_lvl[lvl+1][i] = nxt[i];
                for (int i=NOUT;i<P;i++) t_lvl[lvl+1][i] = 32'd0;
                t_valid[lvl+1]=bv; t_last[lvl+1]=bl; t_clear[lvl+1]=bc;
            end
        end
    endgenerate

    wire [31:0] group_sum = t_lvl[LV][0];
    wire        gs_valid  = t_valid[LV];
    wire        gs_last   = t_last[LV];
    wire        gs_clear  = t_clear[LV];

    // ---------------- stage A: acc += group_sum (II=2 recurrence) ------------
    //   split fp32_add across two registers like the scalar dot's FADD_PIPE.
    //   Producer presents groups every OTHER cycle (II=2) so each A reads the
    //   acc the previous group's B already wrote.
    logic [31:0] acc;
    assign score_fp32 = acc;
    logic [71:0] acc_ir;
    logic        acc_a_valid, acc_a_last;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc<=32'd0; out_valid<=1'b0; acc_ir<=72'd0;
            acc_a_valid<=1'b0; acc_a_last<=1'b0;
        end else begin
            out_valid <= 1'b0;
            acc_a_valid <= gs_valid;
            if (gs_valid) begin
                acc_ir     <= fp32_add_a(gs_clear ? 32'd0 : acc, group_sum);
                acc_a_last <= gs_last;
            end
            if (acc_a_valid) begin
                acc <= fp32_add_b(acc_ir);
                if (acc_a_last) out_valid <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
