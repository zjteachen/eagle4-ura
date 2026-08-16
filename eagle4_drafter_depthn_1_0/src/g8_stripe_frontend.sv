// =====================================================================
// g8_stripe_frontend.sv — J5-β W2 (2026-07-17): N-channel STRIPED HBM
//   gather front-end for g8_rescore_p, FAITHFUL PORT of the GO-proven
//   prototype verilog/forks/exp_g8_stripe/src/g8_stripe_gather.sv
//   (commits 39c0682/baf9792; gate GREEN — byte-identical dumps at
//   N_CH={1,2,4}, 196.27 -> 98.27 cyc/row (2.00x) @N=2, RED teeth).
//
// WHAT CHANGED vs THE PROTOTYPE (integration trim ONLY — the AR engines,
// owner FIFOs, feed lanes and per-channel dots are line-faithful):
//   * candidate table: READ from the parent selector's cand_idx[] via the
//     {cand_addr, cand_id_rd} comb ports (prototype semantics unchanged);
//     the prototype's own cand_idx[] MIRROR WRITE is dropped — the parent
//     g8_rescore_p owns the authoritative table (its S_SEL_DONE copy) and
//     its untouched S_RESOLVE reads it.
//   * score landing: the prototype landed dot results into its own
//     rescored[] array.  rescored[] lives in (and is single-writer-owned
//     by) the parent's main always_ff, so this module EXPORTS the landing
//     events instead: {land_valid[c], land_slot[c], land_score[c]} are
//     comb views of (dot_outv[c], owner-FIFO head, dot_score[c]); the
//     owner-FIFO pop stays in here on the SAME cycle the prototype popped.
//     Channels own DISJOINT slot sets (slot k -> ch k%N_CH) so same-cycle
//     multi-channel landings never collide on a slot.
//   * S_FINAL/S_RESOLVE + the top-K_F heap are NOT ported — the parent's
//     verbatim S_FINAL/S_RESOLVE (g8_rescore_p.sv:2240-2263 lineage) runs
//     on rescored[] after gather_done, exactly as the spec §7.2 prescribes
//     ("selector/S_FINAL/ABI untouched").
//   * multi-invocation (spec §6.7 sticky-done open item): on completion the
//     FSM returns to S_IDLE; gather_done is a STICKY level cleared on the
//     next start — every per-channel register re-arms on every start.
//
// CHANNEL-SPLIT SCHEME (unchanged): GATHER-ORDER ROUND-ROBIN — candidate
//   slot k is owned by channel k % N_CH.  Every channel reads the SAME
//   flat 788 MB fp16 lm-head image at lmfull_hbm_base (row id -> base +
//   id*H*2).  NO physical stripe modulus -> NO host relayout, and the
//   pdi_speed_g8_nchan H1 nactive-vs-layout silent-ABI hazard is
//   structurally deleted.  Balance is perfect (±1 row/channel).
//
// PAST-LESSON COMPLIANCE (verbatim from the prototype header):
//   * lane-true consume: full-width reads ONLY (arsize=log2(AXI_DATA_W/8))
//     — memory project_g8resc_rootcause_axi_narrow_lane.
//   * arlen 8-bit + AXI 4KB rule: per-row sub-bursts capped by MAX_BURST
//     beats AND bytes-to-next-4KB-page (the shipping G8WIDE-NOC F1 split,
//     g8_rescore_p.sv:1571-1604).
//   * OUTST arrival-order (pdi21 8->2 lesson): pairing BY ROW TAG, never
//     by arrival.  One AXI ID per channel; per-channel owner FIFO pushed
//     at the row's first AR, popped at that channel's dot out_valid.
//   * per-channel OUTST discipline: <= OUTST sub-bursts in flight.
//
// NUMERIC CONTRACT: whole rows stay on one channel and enter the
//   IDENTICAL g8_dot_fp16_wide_il (straps mirrored from the parent's
//   g_wide_dot_il instance, g8_rescore_p.sv:1044-1045) — zero new fp32
//   re-association; the 512-slot rescored[] dump must stay BYTE-IDENTICAL
//   to the unmodified G8_NCH=1 arm (gate: md5 equality).
//
// RED TEETH: `STRIPE_RED` corrupts the owner tag of the row carrying the
//   rank-1 golden-frontier winner (vocab id 622) on whichever channel
//   gathers it (tag ^= 1) — 622's score misattributes, 622 drops from the
//   top-8, and the 8-word G8RESC digest MUST diverge from 0x002c953c.
//   (STRENGTHENED 2026-07-19: the prior ch0 row3<->row4 swap hit slots 6/8
//   which are NOT top-8 members -> dead RED arm; see body comment + fix
//   report SPEEDWAVE_V2_XSIM_FIX.md #1.)
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

import g8_fp16_pkg::*;

module g8_stripe_frontend #(
    parameter int N_CH        = 2,      // striped channels (2/4; 1 legal for A/B)
    parameter int H           = 3072,
    parameter int K_C         = 512,
    parameter int IDX_W       = 18,
    parameter int SCORE_W     = 32,
    parameter int AXI_ADDR_W  = 64,
    parameter int AXI_DATA_W  = 256,
    parameter int MAX_BURST   = 256,    // AXI4 8-bit arlen cap
    parameter int OUTST       = 4,      // per-channel outstanding sub-bursts
    parameter int WIDE_P      = 16,
    parameter int WIDE_M      = 4,
    parameter bit ROW_OVERLAP = 1'b1    // == parent G8_ROWPP (ship config)
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start,          // 1-cyc pulse (parent S_SEL_DONE->S_GATHER)
    // K1 x J5-β fold (W2 merge, per the parent's :1483 integration note):
    // runtime shortlist size == the parent's launch-sampled kcq (1..K_C;
    // K_C when kc_cfg=0/unwired).  Sampled HERE at start (launch-coherent).
    input  wire [$clog2(K_C):0]       n_rows,
    input  wire [AXI_ADDR_W-1:0]      lmfull_hbm_base,

    // candidate table read, one comb port per channel (parent cand_idx[])
    output logic [$clog2(K_C)-1:0]    cand_addr  [0:N_CH-1],
    input  wire  [IDX_W-1:0]          cand_id_rd [0:N_CH-1],

    // per-channel nrm activation wide read (comb; FF-seam fanout at the top)
    output logic [$clog2(H)-1:0]      nrm_addr [0:N_CH-1],
    input  wire  [WIDE_P*16-1:0]      nrm_wide [0:N_CH-1],

    // N_CH independent AXI read masters (one NMU / HBM port each)
    output logic [N_CH-1:0]           m_arvalid,
    input  wire  [N_CH-1:0]           m_arready,
    output logic [AXI_ADDR_W-1:0]     m_araddr  [0:N_CH-1],
    output logic [7:0]                m_arlen   [0:N_CH-1],
    output logic [2:0]                m_arsize  [0:N_CH-1],
    output logic [1:0]                m_arburst [0:N_CH-1],
    input  wire  [N_CH-1:0]           m_rvalid,
    output logic [N_CH-1:0]           m_rready,
    input  wire  [AXI_DATA_W-1:0]     m_rdata [0:N_CH-1],
    input  wire  [N_CH-1:0]           m_rlast,

    // score-landing export (parent writes rescored[land_slot] <= land_score)
    output logic [N_CH-1:0]           land_valid,
    output logic [$clog2(K_C)-1:0]    land_slot  [0:N_CH-1],
    output logic [SCORE_W-1:0]        land_score [0:N_CH-1],
    // sticky; cleared on start.  Masked with the LIVE start pulse so the
    // parent (which enters its S_GATHER on the same cycle start is high)
    // can never sample the PREVIOUS run's done — the sticky-done re-arm
    // trap (feedback_sticky_done_rearm_trap), closed at the source.
    output wire                       gather_done,
    // instrumentation
    output logic [31:0]               scored_cnt_dbg
);
    localparam int CIDX_W    = $clog2(K_C);                 // 9
    localparam int FPB       = AXI_DATA_W/16;               // fp16/beat (16)
    localparam int BEATS_ROW = (H + FPB - 1)/FPB;           // 192
    localparam int EW        = $clog2(H)+1;                 // element counter width
    localparam int OWN_D     = 16;                          // owner FIFO depth (>= OUTST rows + dot pipe)
    localparam int OWN_AW    = $clog2(OWN_D);

    typedef enum logic [1:0] { S_IDLE, S_GATHER } state_e;
    state_e state;

    logic [CIDX_W:0]    scored_cnt;
    logic [CIDX_W:0]    n_rows_q;                // launch-sampled shortlist size (K1 fold)
    logic               gdone_r;                 // sticky done (cleared on start)
    assign gather_done = gdone_r && !start;      // live-start mask (re-arm trap kill)

    // ---- per-channel engine state (all written in the ONE always_ff below) --
    logic [CIDX_W:0]  ai_slot  [0:N_CH-1];    // next slot this channel gathers
    logic [EW-1:0]    ai_elem  [0:N_CH-1];    // sub-burst element cursor
    logic             ai_done  [0:N_CH-1];
    logic [$clog2(OUTST+1):0] outst_cnt [0:N_CH-1];  // sub-bursts in flight
    // owner FIFO: slot tags in row-issue order (THE row-tag pairing)
    logic [CIDX_W-1:0] own_mem [0:N_CH-1][0:OWN_D-1];
    logic [OWN_AW:0]   own_wp  [0:N_CH-1], own_rp [0:N_CH-1];
    logic [CIDX_W:0]   rows_pushed [0:N_CH-1];       // per-ch row push count (RED arm)
    // R-consume cursor
    logic [EW-1:0]    rc_elem [0:N_CH-1];

    // registered feed into each channel's dot
    logic              feed_v   [0:N_CH-1];
    logic              feed_cl  [0:N_CH-1];
    logic              feed_ls  [0:N_CH-1];
    logic [WIDE_P*16-1:0] feed_a [0:N_CH-1];
    logic [WIDE_P*16-1:0] feed_b [0:N_CH-1];

    wire              dot_outv  [0:N_CH-1];
    wire [31:0]       dot_score [0:N_CH-1];

    genvar gc;
    generate
        for (gc=0; gc<N_CH; gc++) begin : g_dot
            // EXISTING wide dot, straps identical to the parent's
            // g_wide_dot_il instance (g8_rescore_p.sv:1044-1045)
            g8_dot_fp16_wide_il #(.P(WIDE_P), .M(WIDE_M), .MREG(1'b1),
                                  .ROW_OVERLAP(ROW_OVERLAP)) u_dot (
                .clk(clk), .rst_n(rst_n),
                .clear(feed_cl[gc]), .in_valid(feed_v[gc]), .in_last(feed_ls[gc]),
                .a_fp16(feed_a[gc]), .b_fp16(feed_b[gc]),
                .out_valid(dot_outv[gc]), .score_fp32(dot_score[gc]));
        end
    endgenerate

    // nrm address = current consume cursor (comb read, captured with the beat)
    always_comb for (int c=0;c<N_CH;c++) nrm_addr[c] = rc_elem[c][$clog2(H)-1:0];

    // cand port routing: each channel owns its port for its round-robin slot
    always_comb for (int c=0;c<N_CH;c++) cand_addr[c] = ai_slot[c][CIDX_W-1:0];

    // ---- score-landing export (comb view; pop happens below same-cycle) ----
    always_comb begin
        for (int c=0;c<N_CH;c++) begin
            land_valid[c] = dot_outv[c];
            land_slot [c] = own_mem[c][own_rp[c][OWN_AW-1:0]];
            land_score[c] = dot_score[c];
        end
    end

    assign scored_cnt_dbg = 32'(scored_cnt);

    // =====================================================================
    // MAIN — one always_ff owns every register (multi-driver-safe)
    // =====================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state<=S_IDLE; gdone_r<=1'b0; scored_cnt<='0; n_rows_q<=(CIDX_W+1)'(K_C);
            for (int c=0;c<N_CH;c++) begin
                ai_slot[c]<='0; ai_elem[c]<='0; ai_done[c]<=1'b0;
                outst_cnt[c]<='0; own_wp[c]<='0; own_rp[c]<='0; rows_pushed[c]<='0;
                rc_elem[c]<='0;
                m_arvalid[c]<=1'b0; m_araddr[c]<='0; m_arlen[c]<='0;
                m_arsize[c]<=3'd0; m_arburst[c]<=2'b01; m_rready[c]<=1'b0;
                feed_v[c]<=1'b0; feed_cl[c]<=1'b0; feed_ls[c]<=1'b0;
                feed_a[c]<='0; feed_b[c]<='0;
            end
        end else begin
            // feed pulses are 1-cycle
            for (int c=0;c<N_CH;c++) feed_v[c] <= 1'b0;

            case (state)
            // ---------------------------------------------------------
            S_IDLE: if (start) begin
                scored_cnt<='0; gdone_r<=1'b0;
                n_rows_q <= n_rows;              // K1 fold: launch-coherent sample
                for (int c=0;c<N_CH;c++) begin
                    ai_slot[c] <= (CIDX_W+1)'(c);
                    ai_elem[c]<='0; ai_done[c]<=((CIDX_W+1)'(c) >= n_rows);
                    outst_cnt[c]<='0; own_wp[c]<='0; own_rp[c]<='0; rows_pushed[c]<='0;
                    rc_elem[c]<='0;
                end
                state<=S_GATHER;
            end

            // ---------------------------------------------------------
            S_GATHER: begin
                for (int c=0;c<N_CH;c++) begin
                    // single-writer outst arithmetic (issue and retire can
                    // coincide in one cycle; a plain NBA in each branch would
                    // lose the increment — compute flags, write ONCE below)
                    automatic bit ar_issued, r_retired;
                    ar_issued = 1'b0;
                    r_retired = 1'b0;
                    // ---------------- AR lane (per channel) ----------
                    m_rready[c] <= 1'b1;
                    if (!m_arvalid[c] && !ai_done[c] &&
                        (int'(outst_cnt[c]) < OUTST)) begin
                        if (ai_slot[c] >= n_rows_q) begin   // K1 fold: kcq rows
                            ai_done[c] <= 1'b1;
                        end else begin
                            // sub-burst: MAX_BURST cap + 4KB page cap
                            // (verbatim geometry of g8_rescore_p.sv:1586-1604)
                            automatic int unsigned rem_el = H - int'(ai_elem[c]);
                            automatic int unsigned rem_bt = (rem_el + FPB - 1)/FPB;
                            automatic int unsigned blen_b = (rem_bt > MAX_BURST) ? MAX_BURST : rem_bt;
                            automatic int unsigned span_el;
                            automatic logic [AXI_ADDR_W-1:0] rb =
                                lmfull_hbm_base +
                                (AXI_ADDR_W'(cand_id_rd[c]) * (H*2));
                            automatic logic [AXI_ADDR_W-1:0] ar_start = rb + (AXI_ADDR_W'(ai_elem[c]) * 2);
                            begin
                                automatic int unsigned BPB         = AXI_DATA_W/8;
                                automatic int unsigned bytes_to_pg = 4096 - int'(ar_start[11:0]);
                                automatic int unsigned pg_beats    = bytes_to_pg / BPB;
                                if (blen_b > pg_beats) blen_b = pg_beats;
                                if (blen_b == 0)       blen_b = 1;
                            end
                            span_el = blen_b * FPB;
                            m_araddr [c] <= ar_start;
                            m_arlen  [c] <= 8'(blen_b - 1);
                            m_arsize [c] <= 3'(AXI_DATA_W==256 ? 5 : AXI_DATA_W==512 ? 6 : 5);
                            m_arburst[c] <= 2'b01;
                            m_arvalid[c] <= 1'b1;
                            ar_issued = 1'b1;
                            // first sub-burst of the row: push the OWNER TAG
                            // (parent cand_idx[] is authoritative — no mirror)
                            if (ai_elem[c] == '0) begin
                                automatic logic [CIDX_W-1:0] tag;
                                tag = ai_slot[c][CIDX_W-1:0];
`ifdef STRIPE_RED
                                // RED TEETH (strengthened 2026-07-19,
                                // SPEEDWAVE_V2_XSIM_FIX.md #1): corrupt the
                                // OWNER TAG of the row carrying the rank-1
                                // golden-frontier winner (vocab id 622), on
                                // WHICHEVER channel gathers it. 622's exact
                                // dot score then lands on rescored[tag^1]
                                // instead of its own slot, and 622's real
                                // slot is never written (stays reset 0) -> 622
                                // drops out of the top-8 -> the 8-word G8RESC
                                // digest MUST diverge from 0x002c953c.
                                //   The OLD tooth swapped ch0 push-rows 3<->4
                                // == candidate slots 6/8, which are NOT in the
                                // final top-8, so the perturbation was cosmetic
                                // (feedback_realinput_misorder_cosmetic_for_accept)
                                // -> a genuine DEAD RED arm. This variant is
                                // frontier-affecting BY CONSTRUCTION: it keys
                                // off cand_id_rd[c] (the row's authoritative
                                // vocab id at push time), so it always hits a
                                // top-8-contributing row regardless of its slot
                                // or channel.
                                if (cand_id_rd[c] == IDX_W'(622))
                                    tag = tag ^ CIDX_W'(1);
`endif
                                own_mem[c][own_wp[c][OWN_AW-1:0]] <= tag;
                                own_wp[c] <= own_wp[c] + 1'b1;
                                rows_pushed[c] <= rows_pushed[c] + 1'b1;
                            end
                            if ((int'(ai_elem[c]) + int'(span_el)) >= H) begin
                                ai_elem[c] <= '0;
                                ai_slot[c] <= ai_slot[c] + (CIDX_W+1)'(N_CH);
                            end else begin
                                ai_elem[c] <= ai_elem[c] + EW'(span_el);
                            end
                        end
                    end else if (m_arvalid[c] && m_arready[c]) begin
                        m_arvalid[c] <= 1'b0;
                    end

                    // ---------------- R/feed lane (per channel) -------
                    //   Beats arrive strictly in issue order (single AXI ID,
                    //   in-order slave/NoC).  1 beat = 1 group (GPB=1) ->
                    //   register straight into the channel's II=1 dot.
                    if (m_rvalid[c] && m_rready[c]) begin
                        feed_v [c] <= 1'b1;
                        feed_cl[c] <= (rc_elem[c] == '0);
                        feed_ls[c] <= ((int'(rc_elem[c]) + FPB) >= H);
                        feed_a [c] <= m_rdata[c];
                        feed_b [c] <= nrm_wide[c];
                        if ((int'(rc_elem[c]) + FPB) >= H) rc_elem[c] <= '0;
                        else                               rc_elem[c] <= rc_elem[c] + EW'(FPB);
                        if (m_rlast[c]) r_retired = 1'b1;
                    end

                    // single write: +issue -retire (both flags may be set)
                    outst_cnt[c] <= outst_cnt[c]
                                    + ((ar_issued) ? 1'b1 : 1'b0)
                                    - ((r_retired) ? 1'b1 : 1'b0);
                end

                // ---------------- score landing (all channels) --------
                //   rescored[] write happens in the PARENT (single-writer);
                //   here: pop the owner FIFO + count, same cycle as the
                //   prototype did.
                begin
                    automatic logic [CIDX_W:0] scnt;
                    scnt = scored_cnt;
                    for (int c=0;c<N_CH;c++) begin
                        if (dot_outv[c]) begin
                            own_rp[c] <= own_rp[c] + 1'b1;
                            scnt = scnt + 1'b1;
                        end
                    end
                    scored_cnt <= scnt;
                    if (scnt == n_rows_q) begin              // K1 fold: kcq rows
                        for (int c=0;c<N_CH;c++) m_rready[c] <= 1'b0;
                        gdone_r <= 1'b1;       // sticky until next start
                        state<=S_IDLE;         // multi-invocation re-arm (§6.7)
                    end
                end
            end

            default: state<=S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (AXI_DATA_W != WIDE_P*16)
            $fatal(1, "g8_stripe_frontend: AXI_DATA_W(%0d) != WIDE_P*16(%0d) — GPB must be 1",
                   AXI_DATA_W, WIDE_P*16);
    end
`endif

endmodule

`default_nettype wire
