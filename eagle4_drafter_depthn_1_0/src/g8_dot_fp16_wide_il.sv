// =====================================================================
// g8_dot_fp16_wide_il.sv  —  WIDE-CONSUME fp16 dot, II=1 via M-INTERLEAVED ACC.
//
// LEVER I — break the across-group accumulate recurrence (II=2 -> II=1).
//
//   g8_dot_fp16_wide (the shipped WIDE dot) consumes P fp16-pairs/cycle through
//   a balanced per-group adder tree, but the FINAL accumulate (acc += group_sum,
//   left-to-right over groups) is a split fp32_add feeding ITSELF -> a 2-cycle
//   loop-carried recurrence -> groups must be presented every OTHER cycle (II=2).
//   That II=2 halves the consume rate (P/2 = 8 fp16/cyc at P=16) -> ~15.7x gather
//   instead of the ~31x the wide tree could feed at II=1.
//
//   FIX (classic interleaved-accumulator): M independent partial accumulators
//   acc[0..M-1].  Group g accumulates into acc[g % M].  Because the split
//   fp32_add has latency 2 (stage a -> stage b), any single accumulator is only
//   re-read every M cycles; with M >= ADD_LAT (=2) the recurrence is satisfied
//   when a NEW group is presented EVERY cycle (II=1).  At row end, the M
//   partials are summed by ONE balanced fp32 adder tree (clog2(M) levels) ->
//   the row score.  Doubles consume to P fp16/cyc (=16 at P=16) -> recovers the
//   2x WIDE lost (~15.7x -> ~31x gather).
//
// NUMERIC CONTRACT (re-association — frontier-exact, golden re-baselined):
//   per-element fp16 products p_h are BIT-IDENTICAL to the scalar dot and to the
//   shipped WIDE dot (same fp16_mul lanes).  The ONLY difference vs the scalar
//   chain is fp32 reduction ORDER:
//     scalar chain : (((p0+p1)+p2)+...)               strict left-to-right
//     WIDE (II=2)  : LtoR over balanced-tree group sums
//     this (II=1)  : M-interleaved partials of the SAME group sums, tree-combined
//   fp32 add is non-associative -> per-row LSB differs.  Validated in numpy on
//   the SEED-42/311/701 ref boundaries (reports/GATHER_LEVER_I_DOT_II1.md):
//     M=2/4/8 all preserve the top-8 FRONTIER ordered-8/8 == canonical ref on all
//     three seeds (per-row score reassociates on ~400-434/512 rows, EXPECTED).
//   => same FRONTIER-EXACT contract the shipped WIDE dot already holds; the
//      per-row score golden is re-baselined from this reduction order.
//
// PARAMS:
//   P  = lanes/cycle (pow2), tree levels = clog2(P).
//   M  = interleave depth (pow2, >= 2 to cover the split-add latency).  Final
//        combine tree = clog2(M) split-add levels.
//   MREG = register raw DSP product (timing), same as the shipped WIDE dot.
//
// INTERFACE (group-streaming, II=1):
//   - pulse `clear` with the FIRST group of a row (seeds all M partials from +0.0).
//   - assert `in_valid` with P (a,b) fp16 pairs per group; pulse `in_last` on the
//     LAST group of the row.  Present a group EVERY cycle (II=1).
//   - `out_valid` pulses when `score_fp32` holds the final row sum.
//
// MULTI-DRIVER-SAFE: every reg written by exactly one always_ff; score_fp32 is a
//   combinational read of the final combine output register.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

import g8_fp16_pkg::*;

(* keep_hierarchy = "yes" *)
module g8_dot_fp16_wide_il #(
    parameter int  P    = 16,          // lanes consumed per cycle (pow2)
    parameter int  M    = 4,           // interleave depth (pow2, >=2)
    parameter bit  MREG = 1'b1,        // register raw DSP product (timing)
    // ---- EXP_G8_PINGPONG_II1: ROW_OVERLAP early snapshot ---------------------
    //   =1: at the STAGE-B commit of a row's LAST group, snapshot the M partials
    //   IMMEDIATELY (c_snap[a_lane] takes the just-computed fp32_add_b(a_ir),
    //   the other lanes take their already-final acc[i]) and pulse c_go — no
    //   row_finishing/drain_wait 2-cycle park.  VALUE-IDENTICAL to =0 (the old
    //   path snapshots the same committed values 2 cycles later; between commit
    //   and snapshot nothing writes acc[] in either mode for THIS row), so
    //   score_fp32/out_valid are bit-exact, just 2 cycles earlier.  The payoff:
    //   acc[]/seeded are FREE the very next cycle, so the NEXT row's groups may
    //   stream in back-to-back (clear re-seeds lanes from +0.0) while THIS row's
    //   combine tree drains — the parent may overlap rows (G8_ROWPP) with an
    //   owner FIFO since multiple out_valid flushes can then be in flight
    //   (strictly in row order; the combine tree is a fixed-latency pipe).
    //   =0 (DEFAULT): byte-identical legacy behavior (drain_wait park).
    parameter bit  ROW_OVERLAP = 1'b0
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
    localparam int LV  = $clog2(P);        // per-group adder-tree levels
    localparam int MLV = $clog2(M);        // final combine-tree levels
    localparam int MW  = $clog2(M);        // width to count groups mod M

    // ================= stage M: P fp16 multiplies (+ optional MREG) ==========
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

    // ================= stage W: P fp16->fp32 widen -> REGISTER ===============
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

    // ================= balanced fp32 adder tree, SPLIT a/b per level =========
    //   UNCHANGED vs the shipped WIDE dot: clog2(P) levels, each a/b split.
    //   Produces ONE group_sum + control per cycle (it is fully pipelined and
    //   accepts a fresh group every cycle -> it is NOT the II limiter; the acc
    //   recurrence was).
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

    // ================= stage IL: M-INTERLEAVED partial accumulators ==========
    //   group counter g advances on every valid group; lane = g % M.  Each
    //   acc[m] uses the split fp32_add (a/b) but is only re-read every M cycles,
    //   so it sustains a NEW group EVERY cycle (II=1) when M >= ADD_LAT (=2).
    //
    //   The split add for lane m must read acc[m] from M cycles ago.  Because we
    //   round-robin, the natural pipelined form is: each cycle, the incoming
    //   group_sum is routed to lane (g%M); stage-a computes ir_a_lane = add_a(
    //   acc[g%M], group_sum); stage-b (one cycle later, for the lane chosen the
    //   PREVIOUS cycle) writes acc[prev_lane] = add_b(ir_a_lane_prev).  With M>=2
    //   the lane re-read M cycles later already holds the committed value.
    //
    //   `clear` (first group) zero-seeds ALL M partials (the first M groups then
    //   seed their lane from +0.0 naturally because we gate the read with a
    //   per-lane "seeded" mask cleared on `clear`).
    logic [MW-1:0] gcnt;                 // group index mod M
    logic [31:0]   acc   [0:M-1];        // M partial accumulators
    logic [M-1:0]  seeded;               // lane has received its first group

    // stage-a pipeline regs (computed this cycle, committed next cycle)
    logic [71:0]   a_ir;
    logic [MW-1:0] a_lane;
    logic          a_busy;               // a stage produced fresh ir this cycle
    logic          a_last;

    // row-end orchestration: after the LAST group is committed into its lane,
    // launch the final combine tree over the M partials.
    logic          row_finishing;        // last group seen, draining acc -> combine
    logic [MW:0]   drain_wait;           // cycles to wait for last add_b to commit

    // ---- final combine tree over M partials (split a/b per level) ----
    //   Snapshot the M committed partials into c_lvl[0], then run clog2(M)
    //   split-add levels.  One snapshot per row (row_last).
    logic [31:0] c_lvl [0:MLV][0:M-1];
    logic        c_valid [0:MLV];
    logic [31:0] c_snap [0:M-1];
    logic        c_go;                   // pulse to load snapshot into the tree

    always_comb begin
        for (int i=0;i<M;i++) c_lvl[0][i] = c_snap[i];
        c_valid[0] = c_go;
    end
    genvar cl;
    generate
        if (MLV == 0) begin : g_comb0
            // M==1 degenerate (not used; M>=2) -> pass-through
        end else begin : g_combN
            for (cl=0; cl<MLV; cl++) begin : g_ctree
                localparam int CNIN  = M >> cl;
                localparam int CNOUT = CNIN >> 1;
                logic [71:0] cir [0:CNOUT-1];
                logic        cav;
                always_ff @(posedge clk) begin
                    if (!rst_n) begin cav<=1'b0; for (int i=0;i<CNOUT;i++) cir[i]<=72'd0; end
                    else begin
                        for (int i=0;i<CNOUT;i++)
                            cir[i] <= fp32_add_a(c_lvl[cl][2*i], c_lvl[cl][2*i+1]);
                        cav <= c_valid[cl];
                    end
                end
                logic [31:0] cnx [0:CNOUT-1];
                logic        cbv;
                always_ff @(posedge clk) begin
                    if (!rst_n) begin cbv<=1'b0; for (int i=0;i<CNOUT;i++) cnx[i]<=32'd0; end
                    else begin
                        for (int i=0;i<CNOUT;i++) cnx[i] <= fp32_add_b(cir[i]);
                        cbv <= cav;
                    end
                end
                always_comb begin
                    for (int i=0;i<CNOUT;i++) c_lvl[cl+1][i] = cnx[i];
                    for (int i=CNOUT;i<M;i++) c_lvl[cl+1][i] = 32'd0;
                    c_valid[cl+1] = cbv;
                end
            end
        end
    endgenerate

    wire [31:0] combine_sum   = c_lvl[MLV][0];
    wire        combine_valid = c_valid[MLV];

    assign score_fp32 = combine_sum;

    // ---- main interleaved-accumulate FSM ----
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            gcnt<='0; a_busy<=1'b0; a_lane<='0; a_ir<=72'd0; a_last<=1'b0;
            seeded<='0; row_finishing<=1'b0; drain_wait<='0;
            c_go<=1'b0; out_valid<=1'b0;
            for (int i=0;i<M;i++) begin acc[i]<=32'd0; c_snap[i]<=32'd0; end
        end else begin
            c_go      <= 1'b0;
            out_valid <= 1'b0;

            // ---------- STAGE A: route incoming group_sum to lane gcnt --------
            a_busy <= 1'b0;
            if (gs_valid) begin
                automatic logic [MW-1:0] lane = gcnt;
                automatic logic use_seed = gs_clear | ~seeded[lane];
                a_ir   <= fp32_add_a(use_seed ? 32'd0 : acc[lane], group_sum);
                a_lane <= lane;
                a_busy <= 1'b1;
                a_last <= gs_last;
                // on the first group of a row, clear all seeded flags so each
                // lane self-seeds from +0.0 on its first arrival this row.
                if (gs_clear) seeded <= '0;
                seeded[lane] <= 1'b1;       // this lane is now live
                gcnt <= (gcnt == MW'(M-1)) ? '0 : (gcnt + 1'b1);
            end

            // ---------- STAGE B: commit add_b into the chosen lane ------------
            if (a_busy) begin
                acc[a_lane] <= fp32_add_b(a_ir);
                // if this was the row's last group, kick off the drain/combine.
                if (a_last) begin
                    if (ROW_OVERLAP) begin
                        // EXP_G8_PINGPONG_II1: EARLY snapshot — a_lane's final
                        // value is the expression being committed THIS cycle;
                        // every other lane's acc[] is already its final row total
                        // (its last group committed on an earlier cycle).  Same
                        // values the legacy path snapshots 2 cycles later.
                        for (int i=0;i<M;i++)
                            c_snap[i] <= (MW'(i) == a_lane) ? fp32_add_b(a_ir)
                                                            : acc[i];
                        c_go <= 1'b1;
                    end else begin
                        row_finishing <= 1'b1;
                        drain_wait    <= '0;
                    end
                end
            end

            // ---------- ROW END: wait for the last add_b to settle, snapshot --
            //   After a_last commits in stage B, the chosen lane's acc updates
            //   THIS cycle; snapshot on the NEXT cycle so all M partials are the
            //   committed row totals, then pulse c_go into the combine tree.
            //   (ROW_OVERLAP=1 snapshots at the commit itself, above.)
            if (!ROW_OVERLAP && row_finishing) begin
                if (drain_wait == 0) begin
                    drain_wait <= drain_wait + 1'b1;   // 1-cycle settle for the just-committed lane
                end else begin
                    for (int i=0;i<M;i++) c_snap[i] <= acc[i];
                    c_go          <= 1'b1;
                    row_finishing <= 1'b0;
                end
            end

            // ---------- COMBINE DONE -> score_fp32 valid ----------------------
            if (combine_valid) out_valid <= 1'b1;
        end
    end

endmodule

`default_nettype wire
