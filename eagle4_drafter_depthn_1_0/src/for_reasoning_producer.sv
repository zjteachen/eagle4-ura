// =====================================================================
// for_reasoning_producer.sv — the depth>=1 re-key hidden ("a") producer
// =====================================================================
// CONTRACT (measured, DEPTHN_REKEY_GOLDENS.md §4 checks A/B/C; max rel-L2
// 2.24e-04 over 349 real GPU drafter forwards = the fp16 storage floor):
//
//     a_d  =  G9_down[H:2H]  +  ( block_input + G6_oproj_out )
//              ^^^ UPPER half        ^^^^^^^^^^^^^^^^^^^^^^^
//              (for_reasoning)       == exactly our R3 residual output h1
//                                       (eagle4_drafter.sv u_resid_r3 ->
//                                        h1_buf_f), llama_eagle4.py:487,513
//
// so in OUR stage names the producer is simply
//
//     a[c] = sat16( h1[c] << (A_FRAC-H1_FRAC)  +  g9hi[c] << (A_FRAC-G9_FRAC) )
//
// The residual term is NOT new hardware — h1_buf_f already exists and is
// gated GREEN in the depth-0 chain (R3 = x + G6 o_proj, FIX-C). What is new
// is (a) the UPPER G9 half (tiles 24..47 of the N=6144 down-projection), and
// (b) this add + the per-branch `a` bank.
//
// NOTE ON THE OTHER RESIDUAL: FIX-I (eagle4_drafter.sv:5066) correctly zeroed
// the R7 a-arm because the post-MLP residual applies ONLY to the for_reasoning
// half, NOT to to_logits->G11->G12. THIS module is the place that residual
// belongs. Do not "restore" FIX-I.
//
// ---------------------------------------------------------------------
// Q FORMATS (derived from the MEASURED golden magnitudes, not guessed —
// DEPTHN_REKEY_GOLDENS.md §4.1 + reports/results/TREE_FSM_DEPTHN_F1_FORREASONING.md §3):
//   max|h1  | = 20.05   over 9 calls x 8 rows   -> Q11.5  (drafter R3_FRAC=5, unchanged)
//   max|g9hi| =  2.99                            -> Q7.8  (the G9 accumulator's NATIVE
//                                                   frac ~7.94; the depth-0 path's
//                                                   sat_acc_subchunk_shr5 frac3 grid
//                                                   would leave only ~24 codes here)
//   max|a   | = 21.91                            -> Q6.9  (range +-64.0, 2.9x headroom
//                                                   for the depth-3..8 norm growth
//                                                   125.9 -> 154.5 measured at d1->d3)
// Both re-grades are LEFT shifts => EXACT (no rounding, no new quantisation).
// The producer therefore adds ZERO quantisation error of its own; a's accuracy
// vs the fp16 golden is bounded by the INPUT grids (dominated by h1's Q11.5).
//
// ---------------------------------------------------------------------
// STORAGE (feedback_ff_staging_intermediate_seams): the `a` bank is a
// (* ram_style="registers" *) FF register file — NOT BRAM. It is re-read
// across a whole child forward (192 chunks re-read once per level per branch),
// which is the re-read pattern the FF mandate names. VERIFY FDRE in the built
// netlist (scripts/run_f1a_ooc.sh prints the primitive census); never trust the
// attribute.
//   KF_BANKS=1 : 192 x 256b =  49,152 FF   (F1 / depth-1: spec_utils.py:468
//                repeat_interleave => ALL K_F children share the root's a_0)
//   KF_BANKS=8 : 8 x above  = 393,216 FF   (F4 / depth>=2 per-parent re-gather)
//
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

module for_reasoning_producer #(
    parameter int H        = 3072,
    parameter int K_PAR    = 16,
    parameter int A_WIDTH  = 16,
    parameter int H1_FRAC  = 5,    // frac of the R3 residual arm  (drafter R3_FRAC)
    parameter int G9_FRAC  = 8,    // frac of the G9 upper-half arm (acc-native)
    parameter int A_FRAC   = 9,    // frac of the produced `a`
    parameter int KF_BANKS = 1,    // 1 = F1 (depth-1 shares a_0); 16 = F4 per-parent re-gather
    // CONGFIX-1 storage-style strap. MUST be a PARAMETER, not a `define:
    // packaging CLEARS verilog_define (pkg_speedwave_v2.tcl:476) and defines
    // cannot travel in component.xml, so an `ifdef form would silently
    // re-synthesise the FF version while every strap check read green — the
    // 0xE4B71118/HEAD_PAR mechanism. Default 0 = FF (byte-identical).
    parameter bit ABANK_BRAM_P = 1'b0,
    // ---------------------------------------------------------------
    // CHILD_SEAM_FIX (ACCFIX bit0, 0x1125 H1). The h1 arm entering this
    // producer already lost the low bits of the block-input residual: the
    // R3 a-arm applied `pa >>> F2_SH_FR` (level>=1, 4 bits of the parent
    // `a`) or `h_fc >>> F2_SH_HFC` (level 0, 7 bits of h_fc), and this
    // module then re-LEFT-shifts by SH_H1=4 — so the carried residual is
    // floored to frac5 granularity EVERY level (the compounding 4x/level
    // child-quality gradient, PHASE5_DEPTH_FLATNESS H1). SEAM_FIX_P=1 adds
    // back the captured shift REMAINDER (rem_chunk, 0..16 LSB @A_FRAC,
    // captured at the a-arm read in the top) inside the SAME wide pre-sat
    // sum:  level>=1  h1<<4 + rem == pa + g6out<<4        (EXACT)
    //       level 0   h1<<4 + rem == round-half(h_fc->frac9) + g6out<<4
    // seam_en=0 (or rem=0) => arithmetic identical to the legacy form.
    parameter bit SEAM_FIX_P   = 1'b0,
    parameter int REM_W        = 5,
    // ---- derived ----
    parameter int NCHUNKS  = H / K_PAR,
    parameter int CW       = (NCHUNKS <= 1) ? 1 : $clog2(NCHUNKS),
    parameter int BW       = (KF_BANKS <= 1) ? 1 : $clog2(KF_BANKS)
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // ---- producer stream (one chunk per cycle, in any order) ----------
    input  wire                              in_valid,
    input  wire [CW-1:0]                     in_idx,
    input  wire signed [K_PAR*A_WIDTH-1:0]   h1_chunk,    // Q(H1_FRAC)
    input  wire signed [K_PAR*A_WIDTH-1:0]   g9_chunk,    // Q(G9_FRAC), UPPER half
    // CHILD_SEAM_FIX: per-lane unsigned shift remainder (0..16 @A_FRAC LSB),
    // co-timed with in_valid/in_idx. Ignored unless SEAM_FIX_P && seam_en.
    input  wire                              seam_en,
    input  wire [K_PAR*REM_W-1:0]            rem_chunk,
    input  wire [BW-1:0]                     wr_branch,   // destination bank

    // ---- produced `a` (registered, 1 cycle after in_valid) -----------
    output reg                               out_valid,
    output reg  [CW-1:0]                     out_idx,
    output reg  signed [K_PAR*A_WIDTH-1:0]   out_chunk,   // Q(A_FRAC)

    // ---- re-key read port (registered, RD_LAT=1) ---------------------
    input  wire [BW-1:0]                     rd_branch,
    input  wire [CW-1:0]                     rd_idx,
    output reg  signed [K_PAR*A_WIDTH-1:0]   rd_chunk
);

    localparam int SH_H1 = A_FRAC - H1_FRAC;   // 4  (LEFT, exact)
    localparam int SH_G9 = A_FRAC - G9_FRAC;   // 1  (LEFT, exact)

    // Elaboration-time guards: a RIGHT re-grade here would silently throw
    // away bits the golden gate then blames on the add.
    initial begin
        if (SH_H1 < 0) $error("for_reasoning_producer: A_FRAC < H1_FRAC (lossy re-grade)");
        if (SH_G9 < 0) $error("for_reasoning_producer: A_FRAC < G9_FRAC (lossy re-grade)");
    end

    localparam logic signed [A_WIDTH-1:0] POS_SAT = {1'b0, {(A_WIDTH-1){1'b1}}};
    localparam logic signed [A_WIDTH-1:0] NEG_SAT = {1'b1, {(A_WIDTH-1){1'b0}}};

    // Saturating re-grade + add, done in a wide intermediate so neither the
    // shift nor the sum can wrap before the single saturation point.
    localparam int WW = A_WIDTH + SH_H1 + SH_G9 + 2;
    function automatic logic signed [A_WIDTH-1:0] regrade_add;
        input signed [A_WIDTH-1:0] x_h1;
        input signed [A_WIDTH-1:0] x_g9;
        input logic  [REM_W-1:0]   x_rem;   // CHILD_SEAM_FIX remainder (0 = legacy)
        logic signed [WW-1:0] s;
        begin
            s = (WW'($signed(x_h1)) <<< SH_H1) + (WW'($signed(x_g9)) <<< SH_G9)
              + WW'($unsigned(x_rem));
`ifdef FR_RED_NORESID
            // RED ARM (must FAIL the gate): drop the residual term entirely.
            s = (WW'($signed(x_g9)) <<< SH_G9);
`endif
            if      (s > WW'($signed(POS_SAT))) regrade_add = POS_SAT;
            else if (s < WW'($signed(NEG_SAT))) regrade_add = NEG_SAT;
            else                                regrade_add = s[A_WIDTH-1:0];
        end
    endfunction

    wire signed [K_PAR*A_WIDTH-1:0] a_comb;
    genvar l;
    generate
        for (l = 0; l < K_PAR; l = l + 1) begin : g_lane
            // CHILD_SEAM_FIX: the remainder lane folds to constant 0 when the
            // fix is elab-absent or runtime-off -> arithmetic byte-identical.
            wire [REM_W-1:0] lane_rem = (SEAM_FIX_P && seam_en)
                                        ? rem_chunk[l*REM_W +: REM_W] : {REM_W{1'b0}};
            assign a_comb[l*A_WIDTH +: A_WIDTH] =
                regrade_add($signed(h1_chunk[l*A_WIDTH +: A_WIDTH]),
                            $signed(g9_chunk[l*A_WIDTH +: A_WIDTH]),
                            lane_rem);
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_idx   <= '0;
            out_chunk <= '0;
        end else begin
            out_valid <= in_valid;
            out_idx   <= in_idx;
            out_chunk <= a_comb;
        end
    end

    // ---- the `a` bank: KF_BANKS INDEPENDENT NCHUNKS-deep FF register files,
    // read via a ONE-HOT parent select (FF-staging mandate) ------------
    //
    // TIMING RESTRUCTURE (2026-07-24, C3 prescription "8x192-deep one-hot banks
    // over one flat 1536-deep array if the read mux hurts at placement"): the
    // former single flat `a_bank[KF_BANKS*NCHUNKS-1:0]` addressed by the fused
    // index rd_branch*NCHUNKS+rd_idx built a KF_BANKS*192-deep read mux + a
    // wide multiply-add address cone -> WNS -10.03 ns at KF_BANKS=16 in the
    // 0x19 STEP1 OOC. Splitting into KF_BANKS separate 192-deep banks, each
    // read by rd_idx ALONE (192:1), then a ONE-HOT KF:1 select on rd_branch,
    // removes the multiply and lets synthesis build a balanced mux tree.
    //
    // BEHAVIOR IS BIT-IDENTICAL: same (rd_branch,rd_idx) address, same RD_LAT=1
    // (per-bank comb read -> one-hot comb mux -> the SAME single output FF), same
    // write mapping. This is a MUX restructure, not a logic change; the F4 bank
    // gate (run_forreason_bank_gate.sh) stays GREEN unchanged.
    // =================================================================
    // CONGFIX-1 (2026-07-25) — a-bank storage: FF register file -> BRAM
    // -----------------------------------------------------------------
    // WHY (measured, reports/results/DEPTHN_CONGESTION_ANALYSIS.md):
    //   The FF form costs KF_BANKS*NCHUNKS*(K_PAR*A_WIDTH)
    //     = 16*192*256 = 786,432 FDRE  +  ~263,000 LUT6 of 192:1 read mux.
    //   Measured against the last GOOD build (aved_speedwave_v2) that build is
    //     FF     1,317,432 -> 2,193,452   (+876,020; the a-bank is 786,432 = 90%)
    //     LUT    1,104,508 -> 1,451,824   (+347,316; the a-bank mux is ~263K = 76%)
    //     SLICE    258,057 ->   315,151   (80.20% -> 97.94%)  <-- ROUTE KILLER
    //   At SLICE 97.94% route_design 0xE4B7111E died with 101 x
    //   [Route 35-557] "Router failed to deposit pin-assignments" — the router
    //   could not legally pack pins into slices that had no spare capacity.
    //   Moving the a-bank to BRAM costs 64 RAMB36 (BRAM 78.51% -> ~80.2%, 804
    //   free) and gives back ~786K FF + ~263K LUT => SLICE back to ~84%.
    //
    // WHY IT IS BIT-IDENTICAL (this is the whole point — the tree is C1..C4
    // validated and MUST NOT change semantics):
    //   FF form : rd_chunk(t+1) = mux(rd_branch(t), a_bank[rd_idx(t)]@t)
    //   BRAM    : rd_word_q[b](t+1) = a_bank_b[rd_idx(t)]@t   (sync read, RD_LAT=1)
    //             rd_branch_q(t+1)  = rd_branch(t)
    //             rd_chunk(t+1)     = mux(rd_branch_q(t+1), rd_word_q(t+1))
    //                               = mux(rd_branch(t), a_bank[rd_idx(t)]@t)   ==
    //   Same address, same RD_LAT=1, same data. The output FF simply MOVED from
    //   after the one-hot mux to inside the RAM (the mux is now the only comb
    //   logic after the RAM, 16:1 on 256b ~= 2 LUT levels).
    //
    // WHY THE SDP COLLISION HAZARD CANNOT FIRE (the one real risk of SDP BRAM):
    //   A simple-dual-port BRAM has UNDEFINED data when the same address is read
    //   and written in the same cycle. That can never happen here at KF_BANKS>1:
    //   eagle4_drafter.sv:5779 sets FR_DBUF = (KF_BANKS > 8), and then
    //     fr_wr_branch = {  level_i[0], branch_j }
    //     fr_rd_branch = { ~level_i[0], fr_par_slot }
    //   i.e. the write bank and the read bank are ALWAYS in OPPOSITE level-parity
    //   halves => the read bank and the write bank are never the same bank, so
    //   the collision address space is empty by construction. Asserted below.
    //   At KF_BANKS<=1 read and write DO share the one bank, so that case KEEPS
    //   the FF form verbatim (also preserves depth-0 byte-identity).
    //
    // MANDATE NOTE (CLAUDE.md FF-staging rule): that rule governs inter-stage
    // `stage_io_buffer` ACTIVATION SEAMS, whose failure mode was ONE physical
    // BRAM with a registered read crossing an SLR. This is 64 distributed BRAMs
    // holding per-branch tree CONTEXT, not a stage seam, and the read latency is
    // unchanged (1), so the COMBREAD stale-by-1 class does not apply. It is still
    // a memory-inference change and MUST be netlist-verified + re-gated (see the
    // sandbox README) before it goes near a PDI.
    //
    // Strap: define FR_ABANK_BRAM to enable. Default (undefined) = the shipping
    // FF form, byte-identical to ps_tree_fsm.
    // =================================================================
    wire signed [K_PAR*A_WIDTH-1:0] rd_word [0:KF_BANKS-1];

    // Storage style = strap AND the no-collision invariant. The (KF_BANKS>8)
    // term mirrors FR_DBUF (eagle4_drafter.sv:5779) EXACTLY: SDP BRAM is only
    // legal where the level-parity double buffer guarantees read bank != write
    // bank. A collision would be SILICON-DEAD but XSIM-GREEN (real RAMB36 SDP
    // returns undefined; both RTL models return pre-write data), so this guard
    // is the only defence — never widen it.
    localparam bit ABANK_BRAM = ABANK_BRAM_P && (KF_BANKS > 8);

    genvar b;
    generate
        for (b = 0; b < KF_BANKS; b = b + 1) begin : g_abank
            // write bank b when the incoming chunk targets it (KF_BANKS<=1: the
            // single bank absorbs every write, wr_branch ignored — matches the
            // old wr_flat=in_idx path).
            wire wr_this = in_valid &
                ((KF_BANKS <= 1) ? 1'b1 : (wr_branch == BW'(b)));

            if (ABANK_BRAM) begin : g_bram
                // ---- CONGFIX-1: SDP BRAM, write port A / sync-read port B ----
                (* ram_style = "block" *)
                reg signed [K_PAR*A_WIDTH-1:0] a_bank_b [0:NCHUNKS-1];

                always_ff @(posedge clk) begin
                    if (wr_this) a_bank_b[in_idx] <= a_comb;
                end

                // SYNCHRONOUS read in its OWN always_ff => simple-dual-port
                // inference (independent R and W ports, no WRITE_FIRST bypass).
                reg signed [K_PAR*A_WIDTH-1:0] rd_word_q;
                always_ff @(posedge clk) begin
                    rd_word_q <= a_bank_b[rd_idx];
                end
                assign rd_word[b] = rd_word_q;

`ifndef SYNTHESIS
                initial for (int i = 0; i < NCHUNKS; i++) a_bank_b[i] = '0;
                // RED ARM for the no-collision invariant this fix depends on.
                always_ff @(posedge clk)
                    if (rst_n && wr_this && (in_idx == rd_idx) &&
                        (wr_branch == rd_branch))
                        $error("for_reasoning_producer: SDP a-bank R/W COLLISION at bank %0d idx %0d — FR_DBUF invariant broken", b, in_idx);
`endif
            end else begin : g_ff
                // ---- shipping FF register file (unchanged) ----
                (* ram_style = "registers" *)
                reg signed [K_PAR*A_WIDTH-1:0] a_bank_b [0:NCHUNKS-1];

                always_ff @(posedge clk) begin
                    if (wr_this) a_bank_b[in_idx] <= a_comb;
                end

                // per-bank combinational read: a plain 192:1 mux on rd_idx only.
                assign rd_word[b] = a_bank_b[rd_idx];

`ifndef SYNTHESIS
                initial for (int i = 0; i < NCHUNKS; i++) a_bank_b[i] = '0;
`endif
            end
        end
    endgenerate

    // ONE-HOT parent select: exactly one bank's mask is all-ones (rd_branch ==
    // that bank), so the OR-reduction returns rd_word[rd_branch] verbatim — the
    // same value the old a_bank[rd_branch*NCHUNKS+rd_idx] returned. The masked
    // AND-OR form makes the KF:1 mux explicit for the placer.
    //
    // CONGFIX-1: in BRAM mode the RAM already supplied the 1-cycle latency, so
    // the select must use the DELAYED rd_branch and the result must NOT be
    // registered again (that would make RD_LAT=2).
    reg [BW-1:0] rd_branch_q;
    reg          rst_n_q;
    always_ff @(posedge clk) begin
        rd_branch_q <= rd_branch;
        rst_n_q     <= rst_n;
    end

    wire [BW-1:0] sel_branch = ABANK_BRAM ? rd_branch_q : rd_branch;

    reg signed [K_PAR*A_WIDTH-1:0] rd_sel;
    always_comb begin
        rd_sel = '0;
        for (int bb = 0; bb < KF_BANKS; bb = bb + 1)
            rd_sel = rd_sel |
                ({(K_PAR*A_WIDTH){((KF_BANKS <= 1) || (sel_branch == BW'(bb)))}}
                 & rd_word[bb]);
    end

    generate
        if (ABANK_BRAM) begin : g_rdout_bram
            // rd_chunk is combinational here; the RD_LAT=1 already came from the
            // BRAM. rst_n_q masks the output during/just after reset so the
            // reset value matches the FF form's `rd_chunk <= '0`.
            always_comb rd_chunk = rst_n_q ? rd_sel : '0;
        end else begin : g_rdout_ff
            always_ff @(posedge clk) begin
                if (!rst_n) rd_chunk <= '0;
                else        rd_chunk <= rd_sel;
            end
        end
    endgenerate

endmodule

`default_nettype wire
