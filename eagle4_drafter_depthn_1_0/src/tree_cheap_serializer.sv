// =====================================================================
// tree_cheap_serializer.sv   (TREE_P8-RESTORE fork, 2026-07-09)
// =====================================================================
// P-wide -> serial replay of the G12 cheap-logit stream, so the G8-splice
// candidate snoop (g8_rescore/_p cheap_* intake, a SERIAL 1-logit/cycle
// port) can ride the TREE_P>1 pkpar tree walk UNCHANGED.
//
// WHY THIS EXISTS (the TREE_P constraint this dissolves):
//   G8_SPLICE snoops tt_valid_in/tt_score_in/tt_index_in/tt_last_in — the
//   serial stream that only the TREE_P<=1 g_tree_serial arm drives. The
//   pkpar arm keeps its P-wide stream internal, so the top $error-guarded
//   G8_SPLICE=1 down to TREE_P=1 (eagle4_drafter.sv old ~4678). This module
//   re-materializes the EXACT serial stream from the P-wide one:
//
//   ORDER CONTRACT: tree_tile_stream_pkpar emits monotonically-increasing
//   vocab indices, lane 0 = lowest index of each P-group. Unrolling groups
//   lane 0..P-1 therefore reproduces the serial tiler's 0,1,2,...,V-1 order
//   BIT-IDENTICALLY (same scores, same indices, same order — only the cycle
//   timing differs). G8.1's selection is stream-order-deterministic, so the
//   candidate SET, insertion order, and rescored frontier are unchanged
//   vs TREE_P=1 by construction. `last` rides the FINAL ENABLED lane of the
//   last_in group (= global index V-1), matching the serial tt_last_in
//   contract ("pulses on global idx V-1").
//
//   RATE FEASIBILITY — THE PRE-FILTER (the load-bearing piece):
//   A bare FIFO+1/cycle drain CANNOT keep up: the G12 tile GEMV is only ~28
//   cycles (K-dim = the low-rank GLM output, NOT 3072 — measured 44 cyc/tile
//   at P=8, 156 at P=1, PROVEN_SPEED_LEDGER §1), so intake averages 128
//   logits per ~44-cycle tile ≈ 2.9/cycle >> 1/cycle. (The 2026-06-19 fold
//   attempt measured the same: a serial snoop on the P-wide tap drops
//   502/512 — SPEED_TREEP8_FOLD_READY.md §2.) The fix here is the THRESHOLD
//   PRE-FILTER (the "~1-2k LUT pre-filter" of NOW.md 07-06): the G8.1
//   selector admits post-FILL logits only on STRICT cheap_gt(logit, set-min),
//   and the set-min is MONOTONE NONDECREASING, so dropping any logit <= a
//   LAGGED copy of that min (sel_thresh from g8_rescore_p) is EXACTLY
//   equivalent — the selector would reject it anyway. While sel_thresh_valid
//   =0 (FILL / non-splice) everything passes. CLOSED-LOOP DYNAMICS (fill
//   counts CONSUMED logits, 1/cycle): fill completes ~512 drain-cycles in,
//   by which ~1490 logits arrived unfiltered (backlog ~975); the early
//   threshold (min of the first 512) passes ~512/c of intake as the
//   consumed count c grows, so backlog keeps rising until c ~ 1485 —
//   analytic peak ~1600 on random-order scores (unit-TB-measured: see
//   [SERIALIZER] peak line). survivors total
//   ~ K_C*ln(V/K_C) ~ 3-4K (vs 128,256). A sim-only fatal canary +
//   peak-occupancy report guard the margin.
//   NOTE: an ADVERSARIAL ascending-score walk defeats any finite FIFO — real
//   logit streams are not adversarial; the canary is the honest guard.
//
//   !! [GAP-5 2026-07-22] THE OLD "~1600 / ~2.5x margin" ABOVE IS THE
//   RANDOM-SCORE ANALYTIC MODEL, NOT REAL DATA. Three independent
//   measurements say DEPTH=4096 has almost no real margin:
//     - depth-N full-chain (tb_fullchain_regread_frontier_wide256 +DEPTHN=2,
//       real m1c weights, SPEEDWAVE rate ~28 cyc/tile): survivors 8,212-8,379
//       peak 3,527-3,606 / 4096 EVERY pass (build/rungc_depthn2/xsim.log:96-114)
//       = 1.14x margin, stable (structural), 2.2x the analytic peak.
//     - J5A jewel rate (26.02 cyc/tile, seed-42 golden logits): peak 4,089
//       = SEVEN entries spare (ps_jewel_j5a_asm serializer header, 07-15).
//       Occupancy is RATE-SENSITIVE: ~ -257 entries per +1 cyc/tile.
//     - FR-Spec vocab reorder (front-loads the hot logits): peak 5,116 —
//       OVERFLOWS 4096 outright (FRSPEC_INTEGRATION.md §4.5).
//   THERE IS NO $fatal IN SILICON: overflow silently wraps wr_ptr over
//   un-drained records -> corrupted survivor stream -> plausible-wrong
//   frontier, M7-invisible. Hence SHIP DEPTH = 8192 (2.3x on the depth-N
//   peak, 1.6x on the FR-Spec peak). Any change that makes the G12 walk
//   FASTER, or any vocab reorder, MUST re-measure this peak. RED/GREEN
//   unit gate: tb/tb_gap5_serializer_depth.sv; evidence:
//   reports/results/TREE_FSM_GAP5_SERIALIZER_FIFO.md.
//
//   cheap_ready backpressure: NOT needed — the shipping selector (OPT1C/1F)
//   holds cheap_ready HIGH every cycle (zero drops), the same no-stall
//   contract the serial tiler already imposed (top caveat (c)).
//
// MEMORY STYLE (deliberate, vs the FF-seam mandate): the mandate's target is
// RE-READ activation seams (stage_io_buffer) where a hidden BRAM output
// register makes silicon stale-by-1 vs the xsim comb read. This FIFO is a
// single-pass DRAIN (write-once/read-once, in-order) — forced to
// (* ram_style="distributed" *) LUTRAM (NEVER BRAM: BRAM is the ~100% wall
// AND the hidden-output-reg class) with the async read captured SAME-CYCLE
// into the out_* FDRE below (the safe RDATA pattern: LUTRAM has no hidden
// output register, xsim == silicon by construction). 1024 x 51b distributed
// ~ a few K LUT (vs ~52K FF + ~25K LUT mux as a register-file). The OOC
// census + build netlist gate must show RAM32M/RAM64M + FDRE, ZERO RAMB.
//
// MULTI-DRIVER SAFE: all outputs owned by the single always_ff below.
//
// RED-ARM TOOTH: compile with -d TREE_P8_RED to deliberately DROP lane P-2
// of every group (vocab ids ≡ P-2 mod P — includes golden frontier ids 622
// and 51222 @P=8). The full-chain gate MUST go RED on the frontier under
// this define — proving the gate has teeth.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_cheap_serializer
    import g8_fp16_pkg::*;                             // cheap_gt / fp32_gt (the SAME
                                                       // ordering the G8.1 selector uses)
#(
    parameter int P       = 8,
    parameter int IDX_W   = 18,
    parameter int SCORE_W = 32,
    parameter int DEPTH   = 8192,                      // power-of-2 (see rate note)
                                                       // [GAP-5 2026-07-22] SHIP 4096 -> 8192.
                                                       // Measured peaks vs 4096: depth-N real-data
                                                       // 3,599 (1.14x), jewel-rate golden 4,089
                                                       // (7 entries spare), FR-Spec reorder 5,116
                                                       // (OVERFLOW). See the GAP-5 header note.
    parameter bit CHEAP_KEY_INT32 = 1'b1,              // MUST match the g8 instance
    localparam int PTR_W  = $clog2(DEPTH),
    localparam int CNT_W  = $clog2(DEPTH) + 1,
    localparam int REC_W  = SCORE_W + IDX_W + 1        // {last, index, score}
) (
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           flush,        // tree_start: drop stale state

    // P-wide intake (from tree_tile_stream_pkpar's merge stream, snooped)
    input  wire                           valid_in,
    input  wire signed [P*SCORE_W-1:0]    scores_in,
    input  wire        [P*IDX_W-1:0]      indices_in,
    input  wire        [P-1:0]            lane_en,
    input  wire                           last_in,

    // PRE-FILTER threshold (from g8_rescore_p.sel_thresh*): drop any lane
    // whose score is NOT strictly greater than thresh while thresh_valid.
    // Exactly safe (strict-gt admission + monotone min — header note).
    // The LAST-marked lane is ALWAYS enqueued (even if below threshold) so
    // the g8 selector's cheap_last handshake never goes missing.
    input  wire        [SCORE_W-1:0]      thresh,
    input  wire                           thresh_valid,

    // serial replay (the tt_* contract g8's cheap_* intake expects)
    output logic                          out_valid,
    output logic signed [SCORE_W-1:0]     out_score,
    output logic        [IDX_W-1:0]       out_index,
    output logic                          out_last
);

    initial begin
        if ((DEPTH & (DEPTH-1)) != 0)
            $error("tree_cheap_serializer: DEPTH(%0d) must be a power of 2", DEPTH);
        if (DEPTH < 2*P)
            $error("tree_cheap_serializer: DEPTH(%0d) too small for P(%0d)", DEPTH, P);
    end

    // ---- P-BANKED distributed-LUTRAM FIFO storage ----
    // Up to P records push per cycle: a single memory would need P write
    // ports (unmappable — Vivado 2025.1 segfaults trying). Classic banking:
    // global entry i lives in bank i%P at address i/P; the j-th packed record
    // of a cycle goes to bank (wr_ptr+j)%P — the j are consecutive, so each
    // bank receives AT MOST ONE write/cycle (true 1W1R distributed RAM).
    // The pop reads bank rd_ptr%P (async LUTRAM read captured same-cycle
    // into the out_* FDREs below — no hidden output register; NEVER BRAM).
    localparam int PB_W    = $clog2(P);                // bank-select bits
    localparam int BDEPTH  = DEPTH / P;                // per-bank depth
    localparam int BADDR_W = $clog2(BDEPTH);
    initial begin
        if ((P & (P-1)) != 0)
            $error("tree_cheap_serializer: P(%0d) must be a power of 2 for banking", P);
    end

    // NOTE: the memories live INSIDE g_bank below as per-bank 1-D arrays —
    // a 2-D [bank][addr] array with a variable bank index defeats Vivado's
    // distributed-RAM template (OOC-measured: the whole 4096x51 became
    // 209K FDRE + 98K LUT). Per-bank 1-D + common async read addr + P:1
    // read mux infers true RAM64M/RAM32M LUTRAM.

    logic [PTR_W-1:0] wr_ptr, rd_ptr;
    logic [CNT_W-1:0] count;

    // ---- push-side combinational prep: per-lane enable + packed offsets ----
    //   eff_en[g]  = lane g pushes this cycle
    //   ofs[g]     = # enabled lanes below g (packing offset from wr_ptr)
    //   lane_last[g] = lane g is the global-last logit (final ENABLED lane of
    //                  a last_in group)
    logic [P-1:0]       eff_en;
    logic [P-1:0]       lane_last;
    logic [CNT_W-1:0]   push_n;

    // per-lane strict-greater in the SELECTOR's ordering (g8_fp16_pkg)
    function automatic logic lane_gt(input logic [SCORE_W-1:0] a,
                                     input logic [SCORE_W-1:0] b);
        lane_gt = CHEAP_KEY_INT32 ? cheap_gt(a, b) : fp32_gt(a, b);
    endfunction

    always_comb begin
        int last_en_lane;
        logic [P-1:0] pass;
        eff_en    = '0;
        lane_last = '0;
        push_n    = '0;
        pass      = '0;
        last_en_lane = -1;
        if (valid_in) begin
            // the GLOBAL-last lane = the last ENABLED lane of a last_in group
            // (tiler contract: tail lanes disabled via lane_en). Determined
            // BEFORE filtering so cheap_last can never be filtered away.
            for (int g = 0; g < P; g++)
                if (lane_en[g]) last_en_lane = g;
            for (int g = 0; g < P; g++) begin
`ifdef TREE_P8_RED
                // RED-ARM DEFECT (gate-tooth proof): drop lane P-2, i.e. every
                // vocab id ≡ P-2 (mod P). @P=8 that kills golden frontier ids
                // 622 and 51222 (both ≡6 mod 8) from the G8 candidate set ->
                // the frontier gate MUST go RED (and the cand-set diff too).
                pass[g] = lane_en[g] && (g != (P-2));
`else
                // PRE-FILTER: drop lanes the selector would reject anyway
                pass[g] = lane_en[g] && (!thresh_valid ||
                                         lane_gt(scores_in[g*SCORE_W +: SCORE_W], thresh));
`endif
                // last-marked lane is ALWAYS enqueued (cheap_last delivery)
                eff_en[g] = pass[g] || (last_in && (g == last_en_lane));
            end
            for (int g = 0; g < P; g++)
                if (eff_en[g]) push_n = push_n + 1'b1;
            if (last_in && (last_en_lane >= 0))
                lane_last[last_en_lane] = 1'b1;
        end
    end

    // ---- packing network: pack[j] = the j-th ENABLED lane's record ----
    logic [REC_W-1:0] pack [0:P-1];
    always_comb begin
        int j;
        j = 0;
        for (int g = 0; g < P; g++) pack[g] = '0;
        for (int g = 0; g < P; g++) begin
            if (eff_en[g]) begin
                pack[j] = { lane_last[g],
                            indices_in[g*IDX_W   +: IDX_W],
                            scores_in [g*SCORE_W +: SCORE_W] };
                j = j + 1;
            end
        end
    end

    // ---- per-bank single write port (round-robin banking) ----
    logic [P-1:0]                    bank_we;
    logic [BADDR_W-1:0]              bank_waddr [0:P-1];
    logic [REC_W-1:0]                bank_wdata [0:P-1];
    always_comb begin
        for (int b = 0; b < P; b++) begin
            // j = which packed record (if any) lands in bank b this cycle
            logic [PB_W-1:0]  jj;
            logic [PTR_W-1:0] gaddr;
            jj    = PB_W'(b) - wr_ptr[PB_W-1:0];              // (b - wr_lo) mod P
            gaddr = wr_ptr + PTR_W'(jj);                      // global entry index
            bank_we[b]    = ({1'b0, jj} < push_n[PB_W:0]) && (push_n != '0);
            bank_waddr[b] = gaddr[PTR_W-1:PB_W];
            bank_wdata[b] = pack[jj];
        end
    end
    // per-bank memory + async read at the COMMON read address (rd_ptr/P);
    // the pop below muxes the read bank (rd_ptr%P) into the out_* FDREs.
    logic [REC_W-1:0]   bank_rdata [0:P-1];
    wire  [BADDR_W-1:0] rd_baddr = rd_ptr[PTR_W-1:PB_W];
    generate
        for (genvar gb = 0; gb < P; gb++) begin : g_bank
            (* ram_style = "distributed" *)
            logic [REC_W-1:0] mem [0:BDEPTH-1];
            always_ff @(posedge clk)
                if (bank_we[gb]) mem[bank_waddr[gb]] <= bank_wdata[gb];
            assign bank_rdata[gb] = mem[rd_baddr];
        end
    endgenerate

    // ---- single-owner sequential pointers + registered pop ----
    wire pop = (count != '0);   // registered-output pop, 1/cycle when non-empty

    always_ff @(posedge clk) begin
        if (!rst_n || flush) begin
            wr_ptr    <= '0;
            rd_ptr    <= '0;
            count     <= '0;
            out_valid <= 1'b0;
            out_score <= '0;
            out_index <= '0;
            out_last  <= 1'b0;
        end else begin
            wr_ptr <= wr_ptr + PTR_W'(push_n);

            // pop: serial replay, registered output (bank rd_ptr%P, addr rd_ptr/P)
            if (pop) begin
                {out_last, out_index, out_score} <= bank_rdata[rd_ptr[PB_W-1:0]];
                out_valid <= 1'b1;
                rd_ptr    <= rd_ptr + 1'b1;
            end else begin
                out_valid <= 1'b0;
                out_last  <= 1'b0;
            end

            count <= count + push_n - (pop ? CNT_W'(1) : CNT_W'(0));

            // synthesis translate_off
            if ((count + push_n) > CNT_W'(DEPTH)) begin
`ifdef SER_OVF_WARN_ONLY
                // GAP-5 RED-arm mode: model SILICON (which has no $fatal —
                // wr_ptr silently wraps over un-drained records). Lets the
                // unit gate run PAST the overflow and catch the resulting
                // byte-corruption, proving the severity call. NEVER define
                // this in a correctness gate.
                $error("tree_cheap_serializer: FIFO OVERFLOW (count=%0d push=%0d depth=%0d) — SER_OVF_WARN_ONLY: continuing (silicon model)", count, push_n, DEPTH);
`else
                $fatal(1, "tree_cheap_serializer: FIFO OVERFLOW (count=%0d push=%0d depth=%0d) — DEPTH margin violated", count, push_n, DEPTH);
`endif
            end
            // synthesis translate_on
        end
    end

    // sim-only peak-occupancy + survivor-count report (prints at drain of the
    // last logit) — the margin evidence for DEPTH.
    // synthesis translate_off
    int unsigned peak_occ;
    int unsigned n_pushed;
    always @(posedge clk) begin
        if (!rst_n || flush) begin peak_occ <= 0; n_pushed <= 0; end
        else begin
            if (count > CNT_W'(peak_occ)) peak_occ <= 32'(count);
            n_pushed <= n_pushed + 32'(push_n);
            if (out_valid && out_last)
                $display("[SERIALIZER] survivors=%0d  peak FIFO occ=%0d / %0d  @%0t",
                         n_pushed, peak_occ, DEPTH, $time);
        end
    end
    // synthesis translate_on

endmodule

`default_nettype wire
