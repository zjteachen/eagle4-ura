// =====================================================================
// g8_rescore_p.sv — PIPELINED G8 fused-gather-dot (P1) + speculative-
//                   prefetch hooks (P2 ring-cache hit path).
//
// Fork: verilog/forks/pdi6_s7c_g8/.  NEW file; does NOT replace the proven
//       serial g8_rescore.sv (kept as the bit-exact reference DUT).
// Spec: docs/specs/SPEC_G8_PREFETCH.md  (P1 §1, P2 §2, KNOWN edge §4a)
// Coverage policy: reports/EXP_G12_BLOCK_PREFETCH_COVERAGE.md
//                  (running-global-threshold, 100% coverage, budget 5120 rows)
//
// GRAND MERGE (PDI#11): this file is a 3-way merge over the maxprobe base of
//   * ITEM4 : the OPT1G NEIGHBOR-ONLY SYSTOLIC INSERTION SORTER (structurally
//             exact running-min) + the g8_dot MROUND register cut on the SCALAR
//             dot, and
//   * WIDE  : the G8_WIDE 512b-master + WIDE_P-lane fused gather/dot consume
//             path (the consume-bottleneck widener).
//   The WIDE fork's OWN OPT1G (the buffer-then-batch + value-bypass skid using
//   cmin_r_*/s0_*/wr1_*/fill1) is DELIBERATELY DISCARDED — item4's own analysis
//   proved it is NOT bit-exact (7/64 set mismatch on the relocation case).  Only
//   item4's systolic-sorter OPT1G survives.
//
// WHY: serial g8_rescore.sv is AR-round-trip serialized — exactly ONE
//      outstanding AR, idles during every AR latency.  Measured 12.65 ms
//      (2471 cyc/row @100 MHz) >> the bandwidth floor.  This module keeps N
//      (G8_OUTSTANDING) ARs in flight on the SAME ARID (AXI4 same-ID => in-
//      order R, so the consume lane is REORDER-FREE), so the AR latency L is
//      hidden behind the in-flight rows.
//
// WHAT IS UNCHANGED (bit-exact to g8_rescore.sv):
//   * G8.1 streaming threshold-select (OPT1B), G8.4 top-K_F heap, the
//     g8_dot_fp16 pipe, the candidate set, the tie-break semantics.
//   * The PDI#7 gather watchdog (kicks on any AR/R handshake; abort ->
//     frontier passthrough + sticky g8_err + done still asserts).
//   * The 8-bit-arlen sub-burst bound (<=MAX_BURST beats/AR).
//
// WHAT IS NEW (P1):
//   * AR-ISSUE engine: issues the next sub-burst AR while up to
//     G8_OUTSTANDING ARs are unretired (does NOT wait for R data).
//   * Row-descriptor FIFO: one entry per issued sub-burst, telling the
//     R-consume lane which candidate / element-base / row-last each
//     arriving burst belongs to.  Reorder-free (same-ID in-order).
//   * R-CONSUME / dot lane: drains beats in ARRIVAL order, feeds the
//     II=2 dot.  THE FLAGGED EDGE (spec §4a) IS FIXED: the owning cand_ptr
//     is carried as a SIDE-BAND through the dot pipe (latched at the row's
//     last-element feed) so the dot_out_valid flush writes the CORRECT
//     rescored[] slot even though the consume-head descriptor rc_desc has
//     already advanced to a later row.  The shared single-accumulator dot
//     also forbids starting row k+1's `clear` until row k's out_valid has
//     landed (dot_busy gate) — that keeps the fp32 accumulate bit-exact.
//
// WHAT IS NEW (P2, define G8_PREFETCH; generic default 0):
//   * g8_prefetch_thresh.sv snoops the tree top-K insertion port, maintains a
//     running 512th-best estimate, and on admission enqueues a row fetch
//     (budget cap 5120, drop+count when full).  Rows land in a 2-BRAM ring
//     keyed by cand idx.  G8 consumes a HIT directly from the ring (skipping
//     the AR for that row); a MISS goes through the P1 AR pipeline.  The dot
//     math is IDENTICAL on hit vs miss => bit-exact.
//
// STATUS: P1 complete + bit-exact-gated by tb/tb_g8_prefetch.sv; P2 hit-path
//   integrated behind G8_PREFETCH (default OFF, additive).
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

import g8_fp16_pkg::*;

(* keep_hierarchy = "yes" *)
module g8_rescore_p #(
    parameter int V_MAX    = 64,
    parameter int H        = 16,
    parameter int K_C      = 16,
    parameter int K_F      = 8,
    parameter int IDX_W    = $clog2(V_MAX),
    parameter int CIDX_W   = (K_C <= 1) ? 1 : $clog2(K_C),
    parameter int FP16_W   = 16,
    parameter int SCORE_W  = 32,
    parameter int AXI_ADDR_W = 64,
    parameter int AXI_DATA_W = 32,                 // skeleton: 1 fp16/beat
    parameter int AXI_ID_W   = 1,
    parameter int MAX_BURST  = 256,                // AXI4 8-bit arlen cap
    // ---- P1: outstanding-AR pipeline depth ----
    parameter int G8_OUTSTANDING = 8,              // ARs in flight (same ID)
    // ---- P2: speculative prefetch (generic; default DISABLED) ----
    parameter bit G8_PREFETCH        = 1'b0,       // master enable for P2 hooks
    parameter int G8_PREFETCH_DEPTH  = 2,          // ring rows (<=2 BRAM)
    parameter int G8_PREFETCH_BUDGET = 5120,       // cap on speculative fetches
    // ---- PDI#7 watchdog ----
    parameter int unsigned WD_LIMIT = 32'd4_194_304,
    // ---- CHEAP-KEY ORDERING (2026-06-06 second ordering-hazard fix) ----
    //   Cheap-logit stream carries the G12 logit as a SIGNED INT32 (scaled-GEMV
    //   requant), NOT an fp32 bit-pattern. CHEAP_KEY_INT32=1 (DEFAULT, golden)
    //   orders the G8.1 candidate set with g8_fp16_pkg::cheap_gt (signed two's-
    //   complement >, bit-identical to tree_decode.sv:71). =0 restores the
    //   legacy fp32_gt (WRONG for negative logits; repro only). The G8.4 FINAL
    //   rescore heap is UNAFFECTED (always fp32_gt — it compares fp16-dot fp32).
    parameter bit CHEAP_KEY_INT32 = 1'b1,
    // ---- OPT1C: NON-BLOCKING G8.1 selector (2026-06-09 PDI#10 G8_SELSCAN fix) ----
    //   The legacy OPT1B selector drops `cheap_ready` for the entire K_C-cycle
    //   SEL_SCAN min-rescan.  The real path producer (tree_tile_stream.sv) has NO
    //   ready/backpressure and CANNOT stall, so every rescan silently DISCARDS
    //   ~K_C valid logits -> the 512-slot pool freezes on early-vocab ids -> wrong
    //   rescored frontier (G8_RESCORE_INVESTIGATION.md).  OPT1C (=1, DEFAULT)
    //   keeps `cheap_ready` HIGH for ALL of S_SELECT by maintaining the
    //   running-min slot COMBINATIONALLY (a K_C-input min-reduction over the
    //   sel_score[] slots), so no SEL_SCAN phase is needed and ZERO logits are
    //   dropped — bit-exact to the lossless golden, and lossless against the
    //   free-running real tiler.  =0 restores the legacy SEL_SCAN path (repro).
    parameter bit G8_OPT1C = 1'b1,
    // ---- OPT1D: PIPELINED running-min (2026-06-09 PDI#10 G8_OPT1C_TIMING fix) ----
    //   OPT1C's single-cycle K_C=512-wide min-reduction + the K_C-way indexed
    //   write-back is a ~466 ns combinational path (OOC WNS = -456 ns @100 MHz;
    //   2813 logic levels, a LOOKAHEAD8/LUTCY ripple of the 512-deep reduction).
    //   OPT1D (=1, DEFAULT when G8_OPT1C=1) replaces the flat reduction with a
    //   TWO-LEVEL HIERARCHICAL REGISTERED min:
    //     * K_C is split into G8_MIN_NG groups of GSZ=K_C/NG slots.
    //     * grp_min[g]/grp_min_pos[g] : registered per-group min, recomputed
    //       combinationally from that group's GSZ slots (GSZ-wide reduction,
    //       short comb).
    //     * cmin_score/cmin_pos : registered top-level min, recomputed from the
    //       NG group-mins (NG-wide reduction, short comb).
    //   Both levels are REGISTERED, so the longest comb path is max(GSZ,NG)-wide
    //   (=~32 for NG=16), not 512-wide.  `cheap_ready` STILL stays HIGH every
    //   cycle of S_SELECT (no SEL_SCAN, no stall -> zero logits dropped).  The
    //   THRESH replacement uses the registered cmin; the seed-42 freerun golden
    //   gate (tb_g8_freerun_selscan) re-confirms the rescored frontier is
    //   ordered-8/8 == golden.  Set =0 to fall back to OPT1C's flat comb min.
    parameter bit G8_OPT1D   = 1'b0,       // hierarchical-min variant (repro only)
    parameter int G8_MIN_NG  = 16,         // # hierarchical min groups (K_C%NG==0)
    // ---- OPT1F: PIPELINED registered 2-level min (2026-06-09 G8_PIPELINE 100MHz) ----
    //   The shipped OPT1C balanced-flat min (cmin0) is a SINGLE-CYCLE combinational
    //   arc: sel_score[] -> 9-level (K_C=512) comparator tree -> THRESH compare ->
    //   512-way indexed write-decode -> sel_score[*]/CE.  That is the OOC -2.478 ns
    //   @100MHz path (60 logic levels: sel_score_reg[372] -> sel_idx_reg[126]/CE),
    //   ~72-75 MHz post-route.  OPT1F BREAKS this arc with a PIPELINE REGISTER while
    //   staying BIT-EXACT to the flat min and dropping ZERO logits.
    //
    //   STRUCTURE (registered 2-level segmented min, single-writer invariant):
    //     * Slots split into NG=G8_MIN_NG segments of GSZ=K_C/NG.
    //     * seg_min[g]/seg_pos[g] : REGISTERED per-segment min, recomputed each cycle
    //       by a GSZ-wide BALANCED tree over that segment's slots, WITH the in-flight
    //       THRESH/FILL write (wr_en/wr_pos/wr_val) FORWARDED into the tree so the
    //       register reflects the same-cycle write at T+1 (closes the NBA stale-by-1
    //       hole that broke the naive OPT1E).  This GSZ tree is REGISTERED -> OFF the
    //       critical path.
    //     * cmin0F_*/the running global min : a SHORT combinational NG-wide balanced
    //       tree over the REGISTERED seg_min[] (NG=16 -> 4 levels).  This is the only
    //       comb feeding the THRESH compare+decode, so the THRESH arc is
    //       NG-tree(4) + compare + 512-decode, NOT 9-level-tree + compare + decode.
    //
    //   EXACTNESS (proven by tb_g8_opt1d_ab adversarial A/B == flat min, and the
    //   seed-42 freerun golden gate ordered-8/8):  at most ONE slot is written per
    //   cycle (FILL: slot[fill]; THRESH: slot[cmin_pos]).  Its segment's registered
    //   seg_min is recomputed the SAME cycle via the forward, so at T+1 the global
    //   min over the registered seg_min[] is exactly the flat min over sel_score[]
    //   AS-OF-T.  Because the producer is non-stallable and we admit 1 logit/cyc,
    //   the running min the NEXT logit compares against is always the up-to-date
    //   min of all admitted logits (the +1-cycle latency is absorbed: the min the
    //   stream needs at cycle T is the min of slots written <= T-1, which seg_min
    //   reflects).  cheap_ready stays HIGH every cycle -> zero logits dropped.
    //   =0 (with G8_OPT1D=0) falls back to OPT1C's flat comb min (the timing-fail
    //   reference); G8_OPT1F has priority over G8_OPT1D when both set.
    parameter bit G8_OPT1F   = 1'b1,      // PIPELINED registered 2-level min (DEFAULT)
    // NOTE (OPT1E explored + REJECTED): registering the FLAT balanced min and
    //   delaying the THRESH stream by a matched skid does NOT make it exact — the
    //   write->sel_score->cmin0->cmin_r latency is multi-cycle AND has no forward, so
    //   several min-displacing writes are in flight against a stale registered min and
    //   the single-writer invariant breaks (it DROPPED the rank-0 winner).  OPT1F
    //   fixes that by FORWARDING the in-flight write into the (registered) per-segment
    //   recompute, so the registered min is never stale by more than the absorbed
    //   1-cycle stream latency.
    // ---- OPT1G: NEIGHBOR-ONLY SYSTOLIC INSERTION SORTER (the structural 100 MHz fix). -
    //   (PDI#11 Item-4 — Candidate B of reports/pdi9_validation/G8_100MHZ_REDESIGN.md.)
    //   IMPORTANT (PDI#11 Item-4 finding): the originally-specified Candidate B
    //   (buffer-then-batch + 1-DEEP VALUE BYPASS) is NOT bit-exact.  The 1-deep
    //   bypass reconciles the VALUE at the stale min slot, but after a THRESH write
    //   RAISES the min slot the new global min RELOCATES to a different slot that the
    //   registered scalar min has not yet found — a single (pos,val) bypass cannot
    //   recover the relocation.  Measured: 7/64 candidate-set mismatch vs the flat
    //   min on the adversarial A/B stream (tb_g8_opt1g_ab, the relocation case).  A
    //   FLAT registered min cannot be both exact AND off the critical path (the only
    //   exact same-cycle min needs the K_C-wide tree or a full forward, = the bad
    //   arc).  So OPT1G uses a STRUCTURALLY-exact running-min instead: a parallel
    //   neighbor-only INSERTION SORTER (cell_score[]/cell_idx[], constant-depth comb).
    //   =0 keeps OPT1F (the default) active; G8_OPT1G has PRIORITY when set.
    parameter bit G8_OPT1G   = 1'b0,
    // ---- G8_SELPIPE: SEL_PIPE registered-cmin selector retiming (2026-07-09) ----
    //   THE 100MHz IN-CONTEXT fix.  Redteam full-top OOC @10ns measured OPT1F at
    //   WNS -2.935 with EVERY worst arc in this selector (seg_min_reg->seg_pos_reg,
    //   63 logic levels): the arc is serial because the write-forward mux SELECT
    //   (wr_pos_c = cmin0F_pos) is COMBINATIONALLY derived from the NG-wide top
    //   tree, so one cycle chains [top tree 4] + [key_gt cmp] + [decode] + [GSZ
    //   tree 5].  SEL_PIPE breaks the chain by REGISTERING the global min:
    //     * cmin_r_score/cmin_r_pos : REGISTERED global running-min.  The THRESH
    //       compare and the 512-way write-decode and the seg-tree forward-mux
    //       selects ALL come from this register -> no comb tree feeds any select.
    //     * seg_sec/seg_secpos      : REGISTERED per-segment SECOND-min (joint
    //       (min,sec) balanced pair-merge tree, same-cycle write forwarded like
    //       seg_min).  This is what makes the registered min EXACT under
    //       replace-min: when the THRESH write RAISES the min slot, the new
    //       global min RELOCATES — the failure that broke OPT1E and the OPT1G
    //       1-deep-bypass (7/64 mismatch, G8_100MHZ_REDESIGN.md).  With sec:
    //         segnew  = mintb(seg_sec[g_w] if wr_pos==seg_pos[g_w] else
    //                         seg_min[g_w],  wr_val @ wr_pos)
    //         exmin   = NG-tree over REGISTERED seg_min[] with segment g_w masked
    //                   (mask select = REGISTERED wr_pos -> off the tree path)
    //         cmin_r' = mintb(exmin, segnew)   (value tie -> lower pos, = the
    //                   flat lowest-index rule)
    //       All muxes/decodes select on registered values; the longest comb is
    //       max(pair-merge GSZ tree, exmin NG tree + final combine) ≈ half the
    //       OPT1F arc.  CYCLE-IDENTICAL to OPT1F (zero added latency: cmin_r@T
    //       == cmin0F@T == min of writes <= T-1), cheap_ready untouched, the
    //       non-stallable no-drop property preserved.  Bit-exactness gates:
    //       tb_g8_selpipe_ab (K_C=512 freerun A/B vs OPT1F, per-cycle cmin
    //       compare + final set compare + RED drop-one-insertion arm) and the
    //       seed-42 full-chain frontier 8/8.  =0 falls back to unmodified OPT1F.
    parameter bit G8_SELPIPE = 1'b1,
    // ---- G8_WIDE: WIDE-CONSUME gather/dot (grand-build consume-path widener) --
    //   =1 selects the 512-bit m_axi master + P=WIDE_P parallel fp16 dot
    //   (g8_dot_fp16_wide).  Removes the consume bottleneck (P fp16/cyc instead
    //   of the scalar dot's 1 fp16/other-cyc).  AXI_DATA_W MUST be WIDE_P*16 (or
    //   a multiple); descriptors carry beats of AXI_DATA_W/16 fp16 each, so the
    //   beat count per row drops by AXI_DATA_W/16 vs the 32-bit master.  The wide
    //   nrm b-operand arrives via nrm_rd_data_wide (WIDE_P fp16 at nrm_rd_addr..).
    //   NUMERIC: re-associated fp32 accumulate (adder-tree) -> FRONTIER-exact, NOT
    //   per-row-score bit-exact vs the OPT1B scalar dot (re-baselined golden).
    //   =0 (DEFAULT) keeps the legacy 32-bit scalar-dot path fully intact (A/B).
    parameter bit G8_WIDE    = 1'b0,
    parameter int WIDE_P     = 16,         // parallel fp16 lanes/cycle when WIDE
    // ---- LEVER I: WIDE_IL — II=1 interleaved-accumulator wide dot --------------
    //   =1 selects g8_dot_fp16_wide_il (M-interleaved fp32 partials) and the
    //   II=1 group-feed (a group EVERY cycle) with DOUBLE-BUFFERED beats so the
    //   GPB-boundary refetch overlaps feed -> recovers the 2x WIDE conceded at
    //   II=2 (consume 2*HGRP -> HGRP cyc/row).  Only meaningful with G8_WIDE=1.
    //   NUMERIC: rides the SAME WIDE frontier-exact contract (re-associates fp32
    //   one step further; top-8 ordered-8/8 == golden, per-row score re-baselined
    //   from this order — validated standalone, reports/GATHER_LEVER_I_DOT_II1.md).
    //   =0 (DEFAULT) keeps the shipped II=2 g8_dot_fp16_wide path intact (A/B).
    parameter bit WIDE_IL    = 1'b0,
    parameter int WIDE_M     = 4,          // interleave depth (pow2, >=2)
    // ---- MROUND_P: g8_dot scalar-dot multiply-result register cut (item4) ------
    //   Passes through to the SCALAR g8_dot_fp16 .MROUND port (the fp16-mul result
    //   register cut that recovers OOC margin on the scalar dot).  The wide dot
    //   (g8_dot_fp16_wide) does NOT take MROUND.  =1 DEFAULT (item4-validated).
    parameter bit MROUND_P   = 1'b1,
    // ---- SCALAR_II1: scalar-dot II=2 -> II=1 (the no-NoC consume speed lever) ----
    //   reports/results/G8_DOT_II1.md — PROVEN per-row BIT-EXACT (0/512 mismatch,
    //   frontier ordered-8/8 == golden) single-ACCUMULATOR II=1 (FADD_PIPE=0, NOT the
    //   interleaved WIDE_IL). =1 (DEFAULT): scalar g8_dot_fp16 FADD_PIPE=0 AND the AXI
    //   R-consume lane feeds the dot EVERY cycle (drops the II=2 `dot_skip` gap) -> the
    //   dot accepts 1 elem/cyc (2x consume floor). =0: FADD_PIPE=1 + II=2 dot_skip gap
    //   = BYTE-IDENTICAL to the pdi20 shipped scalar path (the safe A/B fallback).
    //   Per-beat a/b pairing (rc_elem/nrm_rd_addr lockstep + alignment canary) is
    //   PRESERVED either way (II=1 only removes self-imposed idle gaps; rvalid still
    //   gates each feed, so no feeding without data).
    parameter bit SCALAR_II1 = 1'b1,
    // ---- G8_CONSUME_HARDEN_P: matched-delay FF-staged nrm b-operand consume ------
    //   (2026-07-03 FIX-CANDIDATE C, reports/results/G8DUMP512_FIXC_CONSUME.md.)
    //   =0 (DEFAULT) is BYTE-IDENTICAL to the shipped scalar consume below (same
    //   products, same G8RESC CRC 0x002c953c / frontier 8/8; the hardened regs are
    //   declared but unused -> synth prunes them).
    //   =1 hardens the scalar gather->dot CONSUME against a SILICON-ONLY nrm
    //   b-operand read-latency slip (the funcsim BRAM/LUTRAM-downgrade class,
    //   memory feedback_funcsim_slice_lutram_downgrade_trap): it OWNS the nrm read
    //   in an EXPLICIT FDRE (nrm_rd_data_q, (* keep *)) — it does NOT rely on the
    //   nrm_buf ram_style="registers" ATTRIBUTE — and feeds the scalar dot from a
    //   matched 1-cycle skid so dot_a (the gathered lm beat) and dot_b (nrm) are
    //   captured for the SAME element k with a deterministic, OWNED 1-cycle
    //   latency.  Adds (a) a synthesizable beat-index consistency check
    //   (nrm_addr_q==h_elem) that raises g8_err on a descriptor/beat handoff drift,
    //   and (b) a sim-only fed-element tap (dbg_fed_elem) that lets the TB assert a
    //   DATA-VALUE canary (dot_b == golden nrm[elem], the teeth the old address-only
    //   canary lacks).  Requires SCALAR_II1=1 (the shipped strap: II=1, no dot_skip).
    //   Math-identical to strap=0 (+1 pipeline latency only).
    parameter bit G8_CONSUME_HARDEN_P = 1'b0,
    // ---- G8_LANE_FIX (pdi24 G0, FIXBUILD_PLAN.md §2.2 — folded into pdi25) ------
    //   ROOT CAUSE of the silicon G8RESC wrong-8 [6621,40756,...]: the SCALAR
    //   gather issues arsize=3'b001 (2-byte NARROW) INCR bursts on the 32-bit
    //   m_axi_g8 bus but consumed m_axi_rdata[15:0] on EVERY beat.  Per AXI4
    //   A3.4.3 narrow-transfer lane semantics the slave returns each beat on the
    //   byte lanes selected by that beat's ADDRESS: element k sits at byte offset
    //   k*2, so ODD elements arrive on lanes [31:16] ([15:0] = inactive-lane fill,
    //   observed ~0 on silicon) -> odd products = 0 -> even-only dot -> the exact
    //   deterministic wrong frontier.  Every legacy TB slave RIGHT-JUSTIFIED the
    //   beat (protocol-wrong) => xsim-green/silicon-wrong.
    //   =1 (DEFAULT): select the active lane by element parity (rc_elem[0] ==
    //   addr[1]; every sub-burst base is even, so parity tracks the lane).
    //   =0: the legacy byte-identical misread (A/B fallback only).
    //   The WIDE path (G8_WIDE=1) is UNAFFECTED: it issues arsize == full bus
    //   width (256b => arsize=5, :1281) -> NO narrow transfers exist there; this
    //   fix covers the CO-PRESENT scalar path (dead at G8_WIDE=1, live fallback
    //   at G8_WIDE=0).
    parameter bit G8_LANE_FIX = 1'b1,
    // ---- G8_ROWPP (EXP_G8_PINGPONG_II1, 2026-07-11) -----------------------------
    //   The WIDE-lane per-row/per-sub-burst BUBBLE killer.  Measured (xsim, real
    //   seed-42 goldens): the shipped WIDE_IL consume idles at (a) every ROW
    //   boundary — the descriptor pop gate `row_first && wdot_busy` parks the feed
    //   for the dot's full flush latency (tree+drain+combine ~19 cyc), and (b)
    //   every SUB-BURST boundary — the beat-ring is reset per descriptor and
    //   rready bounces, re-exposing the AR round-trip.  =1 restructures the WIDE
    //   consume lane into DECOUPLED FETCH/FEED:
    //     * FETCH: its own descriptor cursor (fd_ptr) accepts AXI beats
    //       CONTINUOUSLY across descriptor/row boundaries into a deep beat ring
    //       (RPP_WB_DEPTH=8) — the gather never pauses on a handoff.
    //     * FEED: II=1 group feed from the ring head; a new row's first group NO
    //       LONGER waits for the previous row's out_valid.  Row ownership travels
    //       in a small FIFO (own_mem, pushed at row-first feed, popped at
    //       out_valid — dot results are strictly in row order), replacing the
    //       single wdot_owner register.
    //     * The IL dot gets .ROW_OVERLAP(1): early M-partial snapshot at the last
    //       group's commit (value-identical, 2 cyc earlier) so its accumulators
    //       are free for the next row while the combine tree drains.
    //   BIT-EXACT per row vs WIDE_IL=1/ROWPP=0: same group sums, same lane
    //   phasing (group order unchanged), same combine order — gated by the A/B
    //   rescored[512]+frontier compare + RED owner-corrupt arm (tb_g8pp.sv).
    //   Requires G8_WIDE=1 && WIDE_IL=1.  =0 (DEFAULT): byte-identical legacy.
    parameter bit G8_ROWPP        = 1'b0,
    parameter int ROWPP_OWN_DEPTH = 4,             // rows in flight in the dot (<=2 typ)
    // ---- G8_NCH (J5-β W2, EXP-J5B fold 2026-07-17) --------------------------
    //   N-channel STRIPED HBM gather (gather-order round-robin over the
    //   UNMODIFIED flat lm-head image — NO host relayout; one AXI ID per
    //   channel; per-channel owner FIFO; per-channel untouched
    //   g8_dot_fp16_wide_il).  =1 (DEFAULT): byte-identical legacy — the
    //   stripe generate is OFF, every G8_NCH>1 branch is constant-false, and
    //   the stripe_* ports are tied off.  >=2: S_GATHER delegates to
    //   g8_stripe_frontend; the selector, S_FINAL, S_RESOLVE and the host ABI
    //   are untouched (spec: reports/results/EXP_J5B_G8_STRIPE_SPEC.md §7.2).
    //   Requires G8_WIDE=1 && WIDE_IL=1 && G8_ROWPP=1 (the gate-proven base).
    parameter int G8_NCH          = 1
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,

    // ---- K1 (J5B-W1): runtime rescore-shortlist size (ctrl 0x13C) ----------
    //   # of candidate rows selected + gathered + rescored THIS forward.
    //   0 / >K_C = compile K_C (=512 ship) — the reset default, bit-identical
    //   when never written. Launch-sampled into kcq on `start` (same discipline
    //   as V1 n_tiles_cfg: one coherent value per run, host writes land next
    //   run). Contract: power-of-2 256/128 host values; any K_F<=v<=K_C works.
    //   Finding K1 (ARCH_ONETIME_PACKAGE.md §1): accept flat 512->64 on GPU
    //   A/B (KC256_GPU_ACCEPT_AB.md); gather cost ~ -196 cyc/row removed.
    input  wire [15:0]                   kc_cfg,

    // ---- G8.1 input: streamed cheap G12 logits ----
    input  wire                          cheap_valid,
    output logic                         cheap_ready,
    input  wire [SCORE_W-1:0]            cheap_score,
    input  wire [IDX_W-1:0]              cheap_index,
    input  wire                          cheap_last,

    // ---- G8.3 second operand: to_logits_norm[H] fp16 ----
    input  wire [FP16_W-1:0]             nrm_rd_data,
    // ---- G8_WIDE second operand: WIDE_P consecutive nrm fp16 at nrm_rd_addr ----
    //   (only meaningful when G8_WIDE=1; the top supplies a WIDE_P-wide nrm read
    //   port.  When G8_WIDE=0 leave unconnected / tie 0.)
    input  wire [WIDE_P*FP16_W-1:0]      nrm_rd_data_wide,
    output logic [$clog2((H<2)?2:H)-1:0] nrm_rd_addr,

    // ---- G8.2 HBM gather m_axi READ master (AR + R only) ----
    output logic                         m_axi_arvalid,
    input  wire                          m_axi_arready,
    output logic [AXI_ADDR_W-1:0]        m_axi_araddr,
    output logic [7:0]                   m_axi_arlen,
    output logic [2:0]                   m_axi_arsize,
    output logic [1:0]                   m_axi_arburst,
    output logic [AXI_ID_W-1:0]          m_axi_arid,
    input  wire                          m_axi_rvalid,
    output logic                         m_axi_rready,
    input  wire [AXI_DATA_W-1:0]         m_axi_rdata,
    input  wire                          m_axi_rlast,
    input  wire [AXI_ID_W-1:0]           m_axi_rid,
    input  wire [1:0]                    m_axi_rresp,   // DEBUG-V2: RRESP observe (read-only)
    input  wire  [AXI_ADDR_W-1:0]        lmfull_hbm_base,

    // ---- P2 prefetch hooks (tie off / ignored when G8_PREFETCH==0) --------
    //   prefetch_req   : pulse + cand id, "this candidate just entered top-K_C"
    //   prefetch_evict : pulse + cand id, "this candidate was displaced"
    input  wire                          prefetch_req,
    input  wire [IDX_W-1:0]              prefetch_req_id,
    input  wire                          prefetch_evict,
    input  wire [IDX_W-1:0]              prefetch_evict_id,
    output logic                         prefetch_drop,        // pulse: budget full
    output logic [IDX_W:0]               prefetch_count,       // speculative fetches issued

    // ---- output: G8-resolved frontier ----
    output wire  [K_F*IDX_W-1:0]         frontier_ids,
    output wire  [K_F*SCORE_W-1:0]       frontier_scores,
    output logic                         done,
    output logic                         g8_err,

    // ---- DEBUG-BLOCK-V2 read-only progress taps (PDI#8, ctrl 0x94/0x98/0x9C) ----
    //   PURELY ADDITIVE: combinational mirrors of internal state, no datapath
    //   feedback.  Ported from the serial g8_rescore so the PIPELINED variant has
    //   identical silicon debug visibility (memory feedback_no_accuracy_tradeoff).
    output wire  [3:0]                   dbg_sel_substate,  // {1'b0, state[2:0]} top-FSM phase
    output wire  [9:0]                   dbg_cand_filled,   // # candidate slots filled (0..K_C)
    output wire  [9:0]                   dbg_rows_gathered, // candidate row index reached (0..K_C)
    output wire                          dbg_seen_last,     // sticky cheap_last captured
    output wire                          dbg_in_gather,     // G8 active (not idle/done)
    output wire  [AXI_ADDR_W-1:0]        dbg_last_araddr,   // last AR byte addr issued
    output wire                          dbg_rresp_err,     // sticky RRESP != OKAY

    // ---- PROBE V3 (out3 candidate-set dump) read-only port ----
    //   Combinational read of the G8.1 candidate set for the host out3 dump.
    //   dbg_cand_rdaddr selects a candidate slot 0..K_C-1; returns its vocab id
    //   and cheap (pre-rescore) fp32 score bit-pattern.  PURELY ADDITIVE.
    input  wire  [CIDX_W-1:0]            dbg_cand_rdaddr,
    output wire  [IDX_W-1:0]             dbg_cand_id,
    output wire  [SCORE_W-1:0]           dbg_cand_cheap_score,

    // ---- SEAM-FF: dedicated REGISTERED bridge score-feed sync-read port ----
    //   The tree_softmax_bridge's per-candidate logit MUST NOT arrive through the
    //   async dbg_cand_cheap_score debug tap above: in the BUILT netlist that
    //   unregistered, cross-module, cross-SLR 512-deep combinational read collapses
    //   to a CONSTANT, tying every child softmax to -log2(K_C) (the 0x1F {-9,-18}
    //   silicon SoA). This is a DEDICATED SYNCHRONOUS read of rescored[] with an
    //   FDRE output (bridge_score_ff). The bridge feeds the LOOKAHEAD (next) read
    //   address, so a 1-cycle sync read lands the value on the EXACT cycle the
    //   max/sum pass consumes it (chunk_idx_NEXT idiom) => bit-exact, zero
    //   cycle-count change, ZERO async read on the logit, and the only net that
    //   crosses the SLR is the registered bridge_score_ff. FF-staging mandate,
    //   memory feedback_ff_staging_intermediate_seams. dbg_cand_cheap_score stays
    //   async for the out3/host DEBUG dump ONLY.
    input  wire  [CIDX_W-1:0]            bridge_rdaddr_nxt,   // lookahead read addr
    output logic [SCORE_W-1:0]           bridge_score_ff,     // registered rescored[] out

    // ---- TREE_P8-RESTORE (2026-07-09, ported to SELPIPE+ROWPP base 2026-07-12):
    //   selection-threshold export ----
    //   A CONSERVATIVE (lagged-safe) copy of the G8.1 running admission
    //   threshold, for the upstream tree_cheap_serializer pre-filter.
    //   sel_thresh_valid = in S_SELECT with the K_C set FULL (post-FILL).
    //   Post-fill the set-min is MONOTONE NONDECREASING (a THRESH write
    //   replaces the min with a strictly-greater value), so any consumer
    //   sampling sel_thresh at time T and dropping logits <= it can never
    //   drop a logit the selector would admit at any later time (admission
    //   is strict cheap_gt/key_gt > current-min >= sampled thresh). EXACT.
    //   OPT1G arm: min = cell_score[K_C-1] (descending sorter tail);
    //   OPT1C/D/F arms: min = cmin_score. Under G8_SELPIPE=1 cmin_score is
    //   already the REGISTERED cmin_r (one extra lag = MORE conservative,
    //   still exact — TREE_P8_FOLD_RESULT §6 auditor note). PURELY ADDITIVE
    //   read-only taps.
    output logic [SCORE_W-1:0]           sel_thresh,
    output logic                         sel_thresh_valid,

    // ---- J5-β G8_NCH>=2: striped gather AXI masters (ch c -> its own NMU /
    //   HBM port; ch0 replaces the legacy m_axi_* master at the top mux, ch1+
    //   are NEW m_axi_g8b-class ports).  At G8_NCH=1 (default) these are tied
    //   off (arvalid=0, rready=0) — legacy instantiators leave them
    //   unconnected.  PURELY ADDITIVE.
    output wire  [G8_NCH-1:0]            stripe_arvalid,
    input  wire  [G8_NCH-1:0]            stripe_arready,
    output wire  [AXI_ADDR_W-1:0]        stripe_araddr  [0:G8_NCH-1],
    output wire  [7:0]                   stripe_arlen   [0:G8_NCH-1],
    output wire  [2:0]                   stripe_arsize  [0:G8_NCH-1],
    output wire  [1:0]                   stripe_arburst [0:G8_NCH-1],
    input  wire  [G8_NCH-1:0]            stripe_rvalid,
    output wire  [G8_NCH-1:0]            stripe_rready,
    input  wire  [AXI_DATA_W-1:0]        stripe_rdata   [0:G8_NCH-1],
    input  wire  [G8_NCH-1:0]            stripe_rlast,
    // per-channel WIDE nrm read (comb; the top fans out the FF nrm_buf)
    output wire  [$clog2((H<2)?2:H)-1:0] stripe_nrm_addr [0:G8_NCH-1],
    input  wire  [WIDE_P*FP16_W-1:0]     stripe_nrm_wide [0:G8_NCH-1]
);

    // =================================================================
    // shared buffers (single-writer each)
    // =================================================================
    logic [IDX_W-1:0]   cand_idx   [0:K_C-1];
    logic [SCORE_W-1:0] rescored   [0:K_C-1];

    localparam int AR_W   = $clog2((H<2)?2:H);
    localparam int NSUB   = (H + MAX_BURST - 1) / MAX_BURST;   // sub-bursts / row
    // ---- G8_WIDE beat geometry ----
    //   FPB  = fp16 lanes per AXI beat (= AXI_DATA_W/16).  For WIDE the master
    //   reads FPB fp16/beat; the consume lane drains WIDE_P fp16/cycle, so it
    //   takes (FPB/WIDE_P) cycles to drain a beat (=1 if AXI_DATA_W==WIDE_P*16,
    //   =2 if AXI_DATA_W==512 with WIDE_P=16).  HGRP = groups (of WIDE_P) per row.
    localparam int FPB    = (AXI_DATA_W < 16) ? 1 : (AXI_DATA_W/16);
    localparam int HGRP   = (H + WIDE_P - 1) / WIDE_P;          // groups/row (P-wide)
    // G8_LANE_FIX: MSB of the odd-element active lane on the scalar narrow-burst
    // path ([31:16] on a >=32b bus; degenerate to [15:0] on a 16b skeleton bus).
    localparam int SC_LANE_HI = (AXI_DATA_W >= 32) ? 31 : 15;
    localparam int BEATS_ROW = (H + FPB - 1) / FPB;             // beats/row (wide)
    localparam int WMAXBEATS = (MAX_BURST*1);                   // 8-bit arlen cap
    localparam logic [SCORE_W-1:0] FP32_NEG_INF = 32'hFF80_0000;   // G8.4 final-heap sentinel
    // Cheap-key empty-slot sentinel: INT32_MIN on the signed path (a real logit
    // can be more negative than fp32 -inf's int value 0xFF800000), fp32 -inf on
    // the legacy path. See g8_rescore.sv for the full rationale.
    localparam logic [SCORE_W-1:0] CHEAP_NEG_INF =
        CHEAP_KEY_INT32 ? 32'h8000_0000 : 32'hFF80_0000;
    // K1: cheap-key MAX sentinel (same value every min-reduction in this file
    // already uses as its pad neutral). Slots kc_eff..K_C-1 are PARKED at this
    // value at launch, so every existing min-reduction (flat cmin0 / OPT1D
    // cmin1 / OPT1F seg trees / SELPIPE joint trees) EXCLUDES them without any
    // structural change: a parked slot can never be the running min, so THRESH
    // never evicts into it and the live set stays exactly slots 0..kcq-1.
    // (Degenerate caveat: a real logit numerically == MAXK would tie with the
    // parked slots — impossible for fp32 keys (+inf) and vanishingly unlikely
    // for a saturated INT32 GEMV logit; documented in J5B_W1_IMPL.md.)
    localparam logic [SCORE_W-1:0] CHEAP_KEY_MAX =
        CHEAP_KEY_INT32 ? 32'h7FFF_FFFF : 32'h7F80_0000;
    // K1 launch-sampled shortlist size: kcq in 1..K_C (0/over-range -> K_C).
    // Single driver: the MAIN always_ff (reset + S_IDLE start sample only).
    logic [CIDX_W:0] kcq;
    wire  [CIDX_W:0] kc_eff = (kc_cfg == 16'd0 || kc_cfg > 16'(K_C))
                              ? (CIDX_W+1)'(K_C) : (CIDX_W+1)'(kc_cfg);

    // ===========================================================
    // G8.1 — streaming threshold-select (OPT1B, unchanged from g8_rescore)
    // ===========================================================
    logic [SCORE_W-1:0] sel_score [0:K_C-1];
    logic [IDX_W-1:0]   sel_idx   [0:K_C-1];
    logic [SCORE_W-1:0] thr_score;
    logic [CIDX_W-1:0]  thr_pos;
    logic [CIDX_W:0]    fill;
    logic [CIDX_W:0]    scan_ptr;
    logic [SCORE_W-1:0] scan_min;
    logic [CIDX_W-1:0]  scan_min_pos;
    typedef enum logic [1:0] { SEL_FILL, SEL_SCAN, SEL_THRESH } sel_state_e;
    sel_state_e sel_state;

    // ===========================================================
    // OPT1C — combinational running-min over the K_C sel_score[] slots.
    //   `cmin_pos`/`cmin_score` name the CURRENT minimum-value slot every cycle
    //   (the threshold-select pivot).  Tie-break = LOWEST index wins, matching
    //   the legacy SEL_SCAN walk (it updates only on a STRICT `cheap_gt`/`fp32_gt`
    //   so the first occurrence of the minimum is preserved).  Computed
    //   combinationally so the selector NEVER needs the K_C-cycle SEL_SCAN
    //   phase -> `cheap_ready` stays high every cycle of S_SELECT.  K_C=512
    //   => a ~9-level comparator reduction; functionally exact (the gather
    //   tolerates K_C-boundary tie alternates anyway, see TB serve_zero).
    // ===========================================================
    logic [SCORE_W-1:0] cmin_score;
    logic [CIDX_W-1:0]  cmin_pos;

    // -----------------------------------------------------------------
    // OPT1D — EXACT timing-safe running-min via a TWO-LEVEL hierarchy with
    //   single-group WRITE-FORWARDING (the proven-exact variant; see the A/B and
    //   freerun golden gates).  Replaces OPT1C's flat 512-wide single-cycle comb
    //   reduction (OOC WNS -456 ns @100 MHz, a 2813-level LOOKAHEAD/LUTCY ripple).
    //
    //   Slots split into G8_MIN_NG groups of GSZ=K_C/NG:
    //     grp_min[g]/grp_min_pos[g] : REGISTERED per-group min, recomputed each
    //       cycle by a GSZ-wide comb reduction over that group's slots.
    //     cmin1_* : combinational top-min over the NG group-mins, with the single
    //       just-written group's min FORWARDED so the registered grp_min's 1-cycle
    //       latency is cancelled.  Exactly ONE group changes per cycle, so the
    //       forward is one extra GSZ-wide comb.  Net min == OPT1C's flat min EVERY
    //       cycle (BIT-EXACT, zero latency, verified by tb_g8_opt1d_ab + the
    //       seed-42 freerun golden gate; rescored frontier ordered-8/8 == golden).
    //   The longest comb path is ~max(GSZ, NG) levels, NOT 512.  cheap_ready stays
    //   HIGH (no stall, zero logits dropped).  =0 falls back to OPT1C's flat min.
    // -----------------------------------------------------------------
    localparam int GSZ       = (K_C < G8_MIN_NG) ? 1 : (K_C / G8_MIN_NG);
    localparam int NG_EFF    = (K_C < G8_MIN_NG) ? K_C : G8_MIN_NG;
    logic [SCORE_W-1:0] grp_min      [0:NG_EFF-1];
    logic [CIDX_W-1:0]  grp_min_pos  [0:NG_EFF-1];

    // -----------------------------------------------------------------
    // OPT1F — PIPELINED registered 2-level min (the 100 MHz timing fix).
    //   seg_min[g]/seg_pos[g] : REGISTERED per-segment running-min (GSZ-wide tree,
    //     recomputed each cycle WITH the in-flight write forwarded; see always_ff).
    //   cmin0F_score/cmin0F_pos : SHORT combinational NG-wide balanced reduction over
    //     the registered seg_min[] — the only comb feeding the THRESH compare+decode.
    // -----------------------------------------------------------------
    logic [SCORE_W-1:0] seg_min      [0:NG_EFF-1];
    logic [CIDX_W-1:0]  seg_pos      [0:NG_EFF-1];
    logic [SCORE_W-1:0] cmin0F_score;
    logic [CIDX_W-1:0]  cmin0F_pos;

    // -----------------------------------------------------------------
    // SEL_PIPE (G8_SELPIPE) — registered global min + per-segment second-min.
    //   cmin_r_*   : REGISTERED global running-min (== cmin0F, one full cycle
    //                early in the comb sense: every select hangs off a register).
    //   seg_sec_*  : REGISTERED per-segment second-min (min over the segment
    //                excluding the one slot at seg_pos[g]), maintained by the
    //                joint (min,sec) pair-merge tree in the always_ff below.
    // -----------------------------------------------------------------
    logic [SCORE_W-1:0] cmin_r_score;
    logic [CIDX_W-1:0]  cmin_r_pos;
    logic [SCORE_W-1:0] seg_sec      [0:NG_EFF-1];
    logic [CIDX_W-1:0]  seg_secpos   [0:NG_EFF-1];

    // -----------------------------------------------------------------
    // OPT1G — NEIGHBOR-ONLY SYSTOLIC INSERTION SORTER (the structural 100 MHz fix).
    //
    //   A parallel neighbor-only INSERTION SORTER.  The candidate set is kept in
    //   cell_score[0..K_C-1] DESCENDING (cell[0]=max, cell[K_C-1]=min).  Each cycle a
    //   new logit `e` is inserted: EVERY cell updates in parallel, each looking ONLY
    //   at itself and its immediate upper neighbor (cell[c-1]).  The longest comb arc
    //   is ONE key_gt comparator + a small mux per cell — CONSTANT depth, independent
    //   of K_C (no K_C-wide reduction, no K_C-way write-decode -> the -0.854 ns
    //   sel_score carry-chain violator is eliminated).  The displaced minimum falls
    //   off cell[K_C-1] (discarded).  EXACT by construction: every kept element is in
    //   its true rank; only the true minimum ever drops -> the retained set is the
    //   EXACT global top-K_C by cheap key, bit-identical to OPT1C-flat (no banks, no
    //   hash, no stale-min, no forward invariant).  1 logit/cycle, cheap_ready high
    //   every cycle -> ZERO drop.  Resource: +K_C comparator+register pair (~1.5-2x
    //   selector LUT, +K_C idx/score FF) — affordable (selector is not URAM-binding).
    //   =0 keeps OPT1F (the default) active; G8_OPT1G has PRIORITY when set.
    // -----------------------------------------------------------------
    logic [SCORE_W-1:0] cell_score [0:K_C-1];   // DESCENDING: [0]=max ... [K_C-1]=min
    logic [IDX_W-1:0]   cell_idx   [0:K_C-1];

    function automatic logic key_gt(input logic [SCORE_W-1:0] a,
                                    input logic [SCORE_W-1:0] b);
        key_gt = CHEAP_KEY_INT32 ? cheap_gt(a,b) : fp32_gt(a,b);
    endfunction

    // write-forward side-band: the slot written THIS cycle (set by FILL/THRESH)
    logic               wr_en;
    logic [CIDX_W-1:0]  wr_pos;
    logic [SCORE_W-1:0] wr_val;

    // combinational top-min over the registered group-mins, with the written
    // group's min FORWARDED (OPT1D).  Read by the NEXT cycle's THRESH.
    //   BALANCED tree reductions (both the per-group forward recompute and the
    //   NG-wide top reduction) so the comb is log-depth, not a linear ripple.
    localparam int NGP2 = 1 << $clog2(NG_EFF);   // pow2 >= NG_EFF
    localparam int GSP2 = 1 << $clog2(GSZ);      // pow2 >= GSZ
    logic [SCORE_W-1:0] cmin1_score;
    logic [CIDX_W-1:0]  cmin1_pos;
    always_comb begin
        logic [SCORE_W-1:0] ev [0:NGP2-1];
        logic [CIDX_W-1:0]  ep [0:NGP2-1];
        logic [SCORE_W-1:0] gv [0:GSP2-1];
        logic [CIDX_W-1:0]  gp [0:GSP2-1];
        logic [SCORE_W-1:0] MAXK;
        int n;
        MAXK = CHEAP_KEY_INT32 ? 32'h7FFF_FFFF : 32'h7F80_0000;
        // per-group effective min (registered grp_min, with the one written group
        // recomputed via a BALANCED tree that folds in wr_val@wr_pos).
        for (int g = 0; g < NGP2; g++) begin
            if (g >= NG_EFF) begin ev[g] = MAXK; ep[g] = '0; end
            else if (wr_en && (int'(wr_pos) >= g*GSZ) && (int'(wr_pos) < (g+1)*GSZ)) begin
                for (int j = 0; j < GSP2; j++) begin
                    if (j >= GSZ) begin gv[j] = MAXK; gp[j] = '0; end
                    else begin
                        gv[j] = (int'(wr_pos)==(g*GSZ+j)) ? wr_val : sel_score[g*GSZ+j];
                        gp[j] = CIDX_W'(g*GSZ+j);
                    end
                end
                n = GSP2;
                while (n > 1) begin
                    for (int i=0;i<n/2;i++) begin
                        if (key_gt(gv[2*i], gv[2*i+1])) begin gv[i]=gv[2*i+1]; gp[i]=gp[2*i+1]; end
                        else                            begin gv[i]=gv[2*i];   gp[i]=gp[2*i];   end
                    end
                    n = n/2;
                end
                ev[g] = gv[0]; ep[g] = gp[0];
            end else begin
                ev[g] = grp_min[g]; ep[g] = grp_min_pos[g];
            end
        end
        // BALANCED top reduction over the NG group-mins
        n = NGP2;
        while (n > 1) begin
            for (int i=0;i<n/2;i++) begin
                if (key_gt(ev[2*i], ev[2*i+1])) begin ev[i]=ev[2*i+1]; ep[i]=ep[2*i+1]; end
                else                            begin ev[i]=ev[2*i];   ep[i]=ep[2*i];   end
            end
            n = n/2;
        end
        cmin1_score = ev[0];
        cmin1_pos   = ep[0];
    end

    // OPT1F combinational TOP min: a SHORT NG-wide balanced reduction over the
    //   REGISTERED seg_min[] (NGP2 leaves, pad = MAX key).  LEFT-wins-ties so the
    //   lowest-index segment's min is kept, matching the flat min's lowest-index
    //   tie rule (seg_min[g] itself already holds the lowest-index min within its
    //   segment).  This is the ENTIRE comb cone the THRESH compare+decode hangs off
    //   — NG=16 => 4 levels, vs the flat tree's 9 levels — so the registered min
    //   pipeline cuts the critical arc roughly in half while staying bit-exact.
    always_comb begin
        logic [SCORE_W-1:0] sv [0:NGP2-1];
        logic [CIDX_W-1:0]  sp [0:NGP2-1];
        logic [SCORE_W-1:0] MAXK;
        int n;
        MAXK = CHEAP_KEY_INT32 ? 32'h7FFF_FFFF : 32'h7F80_0000;
        for (int g = 0; g < NGP2; g++) begin
            if (g >= NG_EFF) begin sv[g] = MAXK; sp[g] = '0; end
            else            begin sv[g] = seg_min[g]; sp[g] = seg_pos[g]; end
        end
        n = NGP2;
        while (n > 1) begin
            for (int i=0;i<n/2;i++) begin
                if (key_gt(sv[2*i], sv[2*i+1])) begin sv[i]=sv[2*i+1]; sp[i]=sp[2*i+1]; end
                else                            begin sv[i]=sv[2*i];   sp[i]=sp[2*i];   end
            end
            n = n/2;
        end
        cmin0F_score = sv[0];
        cmin0F_pos   = sp[0];
    end

    // OPT1C flat comb min (kept for G8_OPT1D=0 fallback / repro) — BALANCED TREE
    //   reduction (log2(K_C) levels) instead of a linear ripple, so synthesis maps
    //   it to a shallow comparator tree rather than a 512-deep LOOKAHEAD/LUTCY
    //   chain.  Functionally identical to the linear min (strict ">", lowest-index
    //   tie wins because each node keeps its LEFT operand on equality).
    localparam int KPOW2 = 1 << $clog2(K_C);   // next pow2 >= K_C
    logic [SCORE_W-1:0] cmin0_score;
    logic [CIDX_W-1:0]  cmin0_pos;
    always_comb begin
        logic [SCORE_W-1:0] tv [0:KPOW2-1];
        logic [CIDX_W-1:0]  tp [0:KPOW2-1];
        int n;
        // leaves: real slots, padding = MAX key (never the min): signed INT32_MAX
        // on the cheap path (0x7FFF_FFFF), fp32 +inf (0x7F80_0000) on the fp32 path.
        for (int p = 0; p < KPOW2; p++) begin
            if (p < K_C) begin tv[p] = sel_score[p]; tp[p] = CIDX_W'(p); end
            else begin
                tv[p] = CHEAP_KEY_INT32 ? 32'h7FFF_FFFF : 32'h7F80_0000;
                tp[p] = '0;
            end
        end
        // pairwise balanced reduction; LEFT wins ties (keeps lowest index)
        n = KPOW2;
        while (n > 1) begin
            for (int i = 0; i < n/2; i++) begin
                if (key_gt(tv[2*i], tv[2*i+1])) begin
                    tv[i] = tv[2*i+1]; tp[i] = tp[2*i+1];   // right strictly smaller
                end else begin
                    tv[i] = tv[2*i];   tp[i] = tp[2*i];     // left <= right (tie->left)
                end
            end
            n = n/2;
        end
        cmin0_score = tv[0];
        cmin0_pos   = tp[0];
    end

    // unified running-min handle used by the (OPT1C/F/D) THRESH branch.  DEFAULT =
    // OPT1C with the BALANCED-TREE flat min (cmin0).  OPT1G does NOT use cmin_* (it
    // is a neighbor-only sorter, no running-min reduction); the mux below leaves the
    // OPT1F default selected for the cmin_* handle when OPT1G is active (cmin_* is
    // simply unused on the OPT1G path).
    always_comb begin
        if (G8_OPT1F && G8_SELPIPE)
                           begin cmin_score = cmin_r_score; cmin_pos = cmin_r_pos; end
        else if (G8_OPT1F) begin cmin_score = cmin0F_score; cmin_pos = cmin0F_pos; end
        else if (G8_OPT1D) begin cmin_score = cmin1_score;  cmin_pos = cmin1_pos;  end
        else               begin cmin_score = cmin0_score;  cmin_pos = cmin0_pos;  end
    end

    // PDI#7.1 STICKY seen_last (G8_WEDGE_RCA fix b — ported from the serial
    // g8_rescore into the PIPELINED variant).  The per-sub-state `cheap_last`
    // checks below MISS the last-logit pulse when it coincides with SEL_SCAN
    // (where the selector is backpressured and consumes no stream beat) — the
    // selector then returns to SEL_THRESH and PARKS forever waiting for a
    // cheap_last that already passed.  `seen_last` LATCHES the pulse the instant
    // it is observed in ANY sub-state; the SEL_SCAN exit consults it to advance
    // to S_SEL_DONE instead of SEL_THRESH.  Bit-exact: the candidate set is
    // already final at the scan exit (no further logits can enter), so this only
    // changes the *termination* path, not the selected set.
    logic               seen_last;

    // ===========================================================
    // top FSM (selection phase + gather supervisor + finalize)
    // ===========================================================
    typedef enum logic [3:0] {
        S_IDLE     = 4'd0,
        S_SELECT   = 4'd1,
        S_SEL_DONE = 4'd2,
        S_GATHER   = 4'd3,   // P1: AR-issue + R-consume run concurrently here
        S_FINAL    = 4'd6,
        S_RESOLVE  = 4'd7,
        S_DONE     = 4'd8
    } state_e;
    state_e state;

    // ----- SEL_PIPE RED-arm (xsim-only negative control; gate teeth) -----------
    //   Compiled ONLY with `xvlog -d G8_SELPIPE_RED=<vocab_idx>`: the SELPIPE arm
    //   silently DROPS the THRESH insertion of that one vocab index (both the
    //   sel_score write and the seg/cmin maintenance).  The A/B gate must then
    //   FAIL (candidate-set mismatch >= 1) — proving the gate detects a single
    //   dropped insertion.  Constant 1'b0 (zero logic) in every real build.
`ifdef G8_SELPIPE_RED
    wire red_kill_hit = (cheap_index == IDX_W'(`G8_SELPIPE_RED));
`else
    wire red_kill_hit = 1'b0;
`endif
    logic red_kill;

    // ----- OPT1F same-cycle (COMBINATIONAL) write-intent for the seg_min forward -
    //   PDI#10 100MHz fix-2 (rank-0 drop RCA): the seg_min forward MUST use the
    //   in-flight write DECIDED THIS cycle, NOT the registered wr_en/wr_pos/wr_val
    //   (those are NBA side-band, set at decision-cycle T but only readable at
    //   T+1 — the SAME cycle sel_score[P] already updates).  Forwarding the
    //   registered copy was therefore redundant and left a 1-cycle STALE window:
    //   seg_min[seg(P)]@T+1 still showed P's OLD (low) value, so a candidate
    //   arriving at T+1 saw P as the running min and EVICTED the just-written slot
    //   — which could be the global max (e.g. id 622, the freerun rank-0 drop).
    //   These comb signals mirror the S_SELECT FILL/THRESH write below exactly, so
    //   the recompute at T already folds the T write -> seg_min[seg(P)]@T+1 is
    //   current.  No comb loop: seg_min(reg) -> cmin0F(comb) -> wr_*_c(comb) ->
    //   seg_min next-state; the seg_min register breaks the ring.  (Declared after
    //   `state`/`state_e` so xsim xvlog's strict use-before-decl is satisfied.)
    logic               wr_en_c;
    logic [CIDX_W-1:0]  wr_pos_c;
    logic [SCORE_W-1:0] wr_val_c;
    always_comb begin
        wr_en_c  = 1'b0;
        wr_pos_c = '0;
        wr_val_c = cheap_score;
        red_kill = 1'b0;
        if ((state == S_SELECT) && G8_OPT1C && cheap_valid) begin
            if (fill != kcq) begin                       // K1 runtime shortlist
                wr_en_c  = 1'b1;
                wr_pos_c = fill[CIDX_W-1:0];
                wr_val_c = cheap_score;
            end else if (CHEAP_KEY_INT32 ? cheap_gt(cheap_score, cmin_score)
                                         : fp32_gt (cheap_score, cmin_score)) begin
                // SEL_PIPE RED-arm: drop this one insertion entirely (sim-only)
                red_kill = G8_SELPIPE && red_kill_hit;
                wr_en_c  = !red_kill;
                wr_pos_c = cmin_pos;
                wr_val_c = cheap_score;
            end
        end
    end

    // ----- SEL_PIPE: next global min after this cycle's (single) write ----------
    //   EXACTNESS (the OPT1E/OPT1G-bypass relocation fix): the write either FILLS
    //   slot `fill` or RAISES the current min slot cmin_r_pos.  Only ONE segment
    //   g_w changes; its post-write min is derivable from REGISTERED state:
    //     wr_pos == seg_pos[g_w]  -> the written slot WAS the segment's (lowest-
    //         index) min slot; min over the rest = (seg_sec, seg_secpos) exactly.
    //     wr_pos != seg_pos[g_w]  -> the written slot held value >= seg_min, so
    //         min over the rest = (seg_min, seg_pos) unchanged.
    //   Fold the written value back in with the flat lowest-index tie rule, then
    //   combine with exmin = tree over the UNCHANGED segments (mask select =
    //   registered wr_pos -> no comb tree feeds any select).  Ties use
    //   !key_gt(a,b) && !key_gt(b,a) (compare-equal), then lower position —
    //   identical to the flat balanced tree's left-wins-over-ordered-leaves rule.
    logic [SCORE_W-1:0] cminp_nxt_score;
    logic [CIDX_W-1:0]  cminp_nxt_pos;
    always_comb begin
        logic [SCORE_W-1:0] ev [0:NGP2-1];
        logic [CIDX_W-1:0]  ep [0:NGP2-1];
        logic [SCORE_W-1:0] MAXK;
        logic [SCORE_W-1:0] sn_v, ex_v;
        logic [CIDX_W-1:0]  sn_p, ex_p;
        int n, gw;
        MAXK = CHEAP_KEY_INT32 ? 32'h7FFF_FFFF : 32'h7F80_0000;
        gw   = (GSZ <= 1) ? int'(wr_pos_c) : (int'(wr_pos_c) / GSZ);
        // exmin: balanced top tree over the REGISTERED seg_min[] with the written
        // segment masked out (left-wins ties over ordered leaves = lowest index).
        for (int g = 0; g < NGP2; g++) begin
            if (g >= NG_EFF || g == gw) begin ev[g] = MAXK; ep[g] = '0; end
            else begin ev[g] = seg_min[g]; ep[g] = seg_pos[g]; end
        end
        n = NGP2;
        while (n > 1) begin
            for (int i = 0; i < n/2; i++) begin
                if (key_gt(ev[2*i], ev[2*i+1])) begin ev[i]=ev[2*i+1]; ep[i]=ep[2*i+1]; end
                else                            begin ev[i]=ev[2*i];   ep[i]=ep[2*i];   end
            end
            n = n/2;
        end
        ex_v = ev[0]; ex_p = ep[0];
        // segnew: written segment's min over its NON-written slots (registered)...
        if (wr_pos_c == seg_pos[gw]) begin sn_v = seg_sec[gw]; sn_p = seg_secpos[gw]; end
        else                         begin sn_v = seg_min[gw]; sn_p = seg_pos[gw];    end
        // ...folded with the written value (value tie -> lower position)
        if (key_gt(sn_v, wr_val_c) ||
            (!key_gt(wr_val_c, sn_v) && (wr_pos_c < sn_p))) begin
            sn_v = wr_val_c; sn_p = wr_pos_c;
        end
        // combine exmin with the rebuilt segment (value tie -> lower position)
        if (key_gt(ex_v, sn_v) ||
            (!key_gt(sn_v, ex_v) && (sn_p < ex_p))) begin
            cminp_nxt_score = sn_v; cminp_nxt_pos = sn_p;
        end else begin
            cminp_nxt_score = ex_v; cminp_nxt_pos = ex_p;
        end
    end

    always_comb begin
        if (G8_OPT1C || G8_OPT1G)
            // OPT1C/OPT1G: admit one logit EVERY cycle of S_SELECT — no SEL_SCAN
            // stall, so the free-running, non-stallable tree_tile_stream loses ZERO
            // logits.  OPT1G's neighbor-only sorter inserts it the same cycle; OPT1C
            // writes the running-min slot the same cycle.
            cheap_ready = (state == S_SELECT);
        else
            // legacy OPT1B: ready drops for the entire K_C-cycle SEL_SCAN rescan.
            cheap_ready = (state == S_SELECT) &&
                          (sel_state == SEL_FILL || sel_state == SEL_THRESH);
    end

    // ===========================================================
    // P1 row-descriptor FIFO
    //   One entry PER ISSUED SUB-BURST.  Same-ID in-order => the consume lane
    //   pops in issue order, no reorder buffer needed.  Depth = G8_OUTSTANDING.
    // ===========================================================
    localparam int FIFO_AW = (G8_OUTSTANDING <= 1) ? 1 : $clog2(G8_OUTSTANDING);
    typedef struct packed {
        logic [CIDX_W-1:0] cand_ptr;      // candidate slot this burst serves
        logic [AR_W:0]     elem_base;     // first element index of this sub-burst
        logic [$clog2(MAX_BURST+1)-1:0] beats; // beats in this sub-burst (1..MAX_BURST)
        logic              row_first;     // this sub-burst starts a new row (seed dot)
        logic              row_last;      // this sub-burst ends the row (flush dot)
        logic              from_ring;     // P2: served from prefetch ring, NOT AXI
    } desc_t;
    desc_t                desc_mem [0:(1<<FIFO_AW)-1];
    logic [FIFO_AW:0]     desc_wptr, desc_rptr;          // extra bit for full/empty
    wire                  desc_empty = (desc_wptr == desc_rptr);
    wire                  desc_full  = (desc_wptr[FIFO_AW-1:0] == desc_rptr[FIFO_AW-1:0]) &&
                                       (desc_wptr[FIFO_AW] != desc_rptr[FIFO_AW]);
    wire [FIFO_AW:0]      outstanding = desc_wptr - desc_rptr; // #unretired sub-bursts

    // ===========================================================
    // AR-ISSUE engine state (lane 1)
    // ===========================================================
    logic [CIDX_W:0]      ai_cand;        // candidate being issued (0..K_C)
    logic [AR_W:0]        ai_elem;        // element cursor within ai_cand's row
    logic                 ai_done;        // all sub-bursts of all rows issued

    // ===========================================================
    // R-CONSUME / dot lane state (lane 2)
    // ===========================================================
    logic [$clog2(MAX_BURST+1)-1:0] rc_beats;  // beats left in current sub-burst
    logic                 rc_active;            // a descriptor is being consumed
    desc_t                rc_desc;              // descriptor at the consume head
    logic [AR_W:0]        rc_elem;              // global element cursor for nrm/last

    // dot-ownership side-band (FIX for spec §4a): the cand_ptr that OWNS the
    // in-flight dot accumulation.  Latched when a row's first element is fed,
    // travels with the pipe, used by the dot_out_valid flush.  Decoupled from
    // rc_desc (which advances to later sub-bursts before the dot flushes).
    logic                 dot_busy;             // a row's dot is mid-flight (clear..out_valid)
    logic [CIDX_W-1:0]    dot_owner;            // cand_ptr that the current dot serves

    // ---- G8_CONSUME_HARDEN_P matched-1-stage scalar consume skid + OWNED nrm FF --
    //   nrm_rd_data_q : EXPLICIT FDRE stage of the nrm read (owned, (* keep *)) —
    //     the read latency is architectural (=1), NOT a synth attribute hint.
    //   nrm_addr_q    : the address that produced nrm_rd_data_q (index-consistency).
    //   h_*           : the 1-deep beat skid (lm beat + its element + dot control),
    //     fed to the dot ONE cycle after intake so dot_a/dot_b are matched.
    (* keep = "true" *) logic [15:0] nrm_rd_data_q;   // owned FDRE of nrm[nrm_rd_addr]
    logic [AR_W:0]        nrm_addr_q;           // addr that produced nrm_rd_data_q
    logic                 h_v;                  // skid holds a captured beat -> feed next
    logic [15:0]          h_a;                  // captured lm beat (becomes dot_a)
    logic [AR_W:0]        h_elem;               // element index of h_a
    logic                 h_clear, h_last, h_own_set;
    logic [CIDX_W-1:0]    h_owner;
`ifndef SYNTHESIS
    logic [AR_W:0]        dbg_fed_elem;         // element last fed to dot (TB data canary)
`endif
    // current-cycle scalar AXI beat handshake for the HARDENED consume (comb);
    // referenced by both the intake branch and the standalone feed block.
    wire                  sc_hard_beat = G8_CONSUME_HARDEN_P && !G8_WIDE
                                          && rc_active && !rc_desc.from_ring
                                          && m_axi_rvalid && m_axi_rready;

    // ---- G8_WIDE consume-lane state (parallel to LANE 2 when G8_WIDE) ----
    //   The wide lane drains FPB fp16/beat into WIDE_P-wide groups, one group
    //   (=WIDE_P fp16) fed to the wide dot per cycle.  AXI_DATA_W==WIDE_P*16 =>
    //   one group per beat (1:1).  AXI_DATA_W==512,WIDE_P==16 => 2 groups/beat.
    localparam int GPB = (FPB < WIDE_P) ? 1 : (FPB/WIDE_P);  // groups per beat
    logic [$clog2((BEATS_ROW<2)?2:BEATS_ROW):0] wrc_beats;   // beats left in sub-burst
    logic                 wrc_active;            // a descriptor is being consumed
    desc_t                wrc_desc;              // descriptor at consume head
    logic [AR_W:0]        wrc_grp;               // group cursor (0..HGRP-1) within row
    logic [$clog2((GPB<2)?2:GPB):0] wrc_subg;    // sub-group index within a wide beat
    logic [AXI_DATA_W-1:0] wrc_beatbuf;          // latched current beat (for GPB>1)
    logic                 wdot_busy;             // wide row's dot mid-flight
    logic [CIDX_W-1:0]    wdot_owner;            // cand_ptr the wide dot serves
    logic                 wdot_phase;            // II=2 feed/skip toggle
    logic                 wbeat_have;            // a beat is buffered in wrc_beatbuf

    // ---- LEVER I (WIDE_IL): beat RING for II=1 feed --------------------------
    //   GPB groups drain from the head beat in GPB cycles at II=1 (=2 cyc at
    //   512b/P16).  A single beat buffer would stall 1 cyc per beat at the GPB
    //   boundary while refetching -> ~II=1.5.  A small ring (depth WB_DEPTH=3)
    //   lets beat-fetch keep the tail filled WHILE the head feeds, absorbing the
    //   1-cycle registered-rready latency so the feed never bubbles across a beat
    //   boundary.  rready stays DECOUPLED (registered, gated on post-drain
    //   occupancy + inflight) -> no overrun, no set-and-read deadlock.
    localparam int WB_DEPTH = 3;                   // beat-ring depth (>=2; 3 covers
                                                   // the registered-rready inflight beat)
    localparam int WB_AW    = (WB_DEPTH <= 1) ? 1 : $clog2(WB_DEPTH);
    logic [AXI_DATA_W-1:0] wb_mem [0:WB_DEPTH-1];  // ring of buffered beats
    logic [WB_AW-1:0]      wb_head, wb_tail;       // feed-head / accept-tail
    logic [WB_AW:0]        wb_occ;                 // beats buffered (0..WB_DEPTH)

    // ---- G8_ROWPP: decoupled FETCH/FEED state (inert when G8_ROWPP=0) --------
    //   Deep beat ring (fetch pushes tail, feed drains head), an independent
    //   FETCH descriptor cursor fd_ptr (desc_rptr <= fd_ptr <= desc_wptr), and
    //   the row-owner FIFO that replaces the single wdot_owner register.
    localparam int RPP_WB_DEPTH = 8;               // beat ring depth (pow2)
    localparam int RPP_WB_AW    = (RPP_WB_DEPTH <= 1) ? 1 : $clog2(RPP_WB_DEPTH);
    logic [AXI_DATA_W-1:0] rpp_mem [0:RPP_WB_DEPTH-1];
    logic [RPP_WB_AW-1:0]  rpp_head, rpp_tail;
    logic [RPP_WB_AW:0]    rpp_occ;
    logic [FIFO_AW:0]      fd_ptr;                 // fetch descriptor cursor
    logic [8:0]            fd_beats;               // beats left to FETCH in fd desc
    logic                  fd_active;
    localparam int OWN_AW  = (ROWPP_OWN_DEPTH <= 1) ? 1 : $clog2(ROWPP_OWN_DEPTH);
    logic [CIDX_W-1:0]     own_mem [0:ROWPP_OWN_DEPTH-1];
    logic [OWN_AW:0]       own_wp, own_rp;         // extra bit for occupancy
    wire  [OWN_AW:0]       own_occ = own_wp - own_rp;
`ifdef G8_ROWPP_RED
    // RED-arm (xsim-only gate teeth): corrupt the owner of row `G8_ROWPP_RED
    // (push cand_ptr+1) — the A/B rescored[]/frontier compare MUST then FAIL.
    logic [9:0]            red_rowcnt;
`endif

    // ---- J5-β G8_NCH>=2: striped gather front-end (delegated S_GATHER) ------
    //   Instantiated ONLY at G8_NCH>1; at the default G8_NCH=1 every net below
    //   is tied and every consumer branch is constant-false (bit-identical
    //   legacy).  The front-end reads the authoritative cand_idx[] (comb),
    //   masters the stripe_* AXI ports, and exports score-landing events; the
    //   rescored[] writes stay in THIS module's main always_ff (single-writer).
    logic                stripe_start_r;
    wire  [G8_NCH-1:0]   st_land_v;
    wire  [CIDX_W-1:0]   st_land_slot  [0:G8_NCH-1];
    wire  [SCORE_W-1:0]  st_land_score [0:G8_NCH-1];
    wire                 st_gdone;
    generate if (G8_NCH > 1) begin : g_stripe
        wire [CIDX_W-1:0] st_cand_addr [0:G8_NCH-1];
        logic [IDX_W-1:0] st_cand_id   [0:G8_NCH-1];
        // comb read of the selector's candidate table (stable through gather)
        always_comb for (int c=0;c<G8_NCH;c++)
            st_cand_id[c] = cand_idx[st_cand_addr[c]];
        g8_stripe_frontend #(
            .N_CH(G8_NCH), .H(H), .K_C(K_C), .IDX_W(IDX_W), .SCORE_W(SCORE_W),
            .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W),
            .MAX_BURST(MAX_BURST), .OUTST(G8_OUTSTANDING),
            .WIDE_P(WIDE_P), .WIDE_M(WIDE_M), .ROW_OVERLAP(G8_ROWPP)
        ) u_stripe (
            .clk(clk), .rst_n(rst_n),
            .start(stripe_start_r),
            // K1 x J5-β fold: the frontend walks the RUNTIME shortlist size.
            // kcq was launch-sampled at S_IDLE(start) and is stable long
            // before stripe_start_r fires (S_SEL_DONE exit).
            .n_rows(kcq),
            .lmfull_hbm_base(lmfull_hbm_base),
            .cand_addr(st_cand_addr), .cand_id_rd(st_cand_id),
            .nrm_addr(stripe_nrm_addr), .nrm_wide(stripe_nrm_wide),
            .m_arvalid(stripe_arvalid), .m_arready(stripe_arready),
            .m_araddr(stripe_araddr),   .m_arlen(stripe_arlen),
            .m_arsize(stripe_arsize),   .m_arburst(stripe_arburst),
            .m_rvalid(stripe_rvalid),   .m_rready(stripe_rready),
            .m_rdata(stripe_rdata),     .m_rlast(stripe_rlast),
            .land_valid(st_land_v), .land_slot(st_land_slot),
            .land_score(st_land_score),
            .gather_done(st_gdone), .scored_cnt_dbg());
    end else begin : g_stripe_off
        assign stripe_arvalid = '0;
        assign stripe_rready  = '0;
        for (genvar sc=0; sc<G8_NCH; sc++) begin : g_tie
            assign stripe_araddr [sc] = '0;
            assign stripe_arlen  [sc] = '0;
            assign stripe_arsize [sc] = '0;
            assign stripe_arburst[sc] = 2'b01;
            assign stripe_nrm_addr[sc] = '0;
            assign st_land_slot [sc] = '0;
            assign st_land_score[sc] = '0;
        end
        assign st_land_v = '0;
        assign st_gdone  = 1'b0;
    end endgenerate
    // stripe launch: 1-cyc pulse on the S_SEL_DONE->S_GATHER transition
    // (S_SEL_DONE lasts exactly one cycle).  Own always_ff — the main FSM
    // block is untouched by the launch.
    always_ff @(posedge clk) begin
        if (!rst_n) stripe_start_r <= 1'b0;
        else        stripe_start_r <= (G8_NCH > 1) && (state == S_SEL_DONE);
    end
`ifndef SYNTHESIS
    initial begin
        if (G8_NCH > 1 && !(G8_WIDE && WIDE_IL && G8_ROWPP))
            $fatal(1, "g8_rescore_p: G8_NCH>1 requires G8_WIDE=1 && WIDE_IL=1 && G8_ROWPP=1 (the EXP-J5B gate-proven base)");
    end
`endif

    // ---- watchdog ----
    //   PDI#7.1 WATCHDOG COVERAGE-ALL-STATES (G8_WEDGE_RCA fix c, ported from the
    //   serial g8_rescore): widen the arming window from the AXI gather state only
    //   to "G8 active & not done" (any state that is NOT S_IDLE and NOT S_DONE),
    //   so a stall in S_SELECT / SEL_SCAN / S_SEL_DONE / S_FINAL / S_RESOLVE also
    //   trips wd_abort -> g8_err -> forces `done` (no host hang).  WD_LIMIT=2^22
    //   >> the K_C-cycle legitimate SEL_SCAN stall, so no false trip.
    //   J5-β: stripe-channel handshakes count as progress too (G8_NCH>1 only).
    logic [31:0] wd_cnt;  logic wd_abort;
    wire in_gather = (state != S_IDLE) && (state != S_DONE);
    wire gather_progress = (m_axi_arvalid && m_axi_arready) ||
                           (m_axi_rvalid  && m_axi_rready) ||
                           ((G8_NCH > 1) &&
                            (((stripe_arvalid & stripe_arready) != '0) ||
                             ((stripe_rvalid  & stripe_rready ) != '0)));
    always_ff @(posedge clk) begin
        if (!rst_n) begin wd_cnt<=0; wd_abort<=0; g8_err<=0; end
        else begin
            wd_abort <= 1'b0;
            if (start)              begin wd_cnt<=0; g8_err<=0; end
            else if (!in_gather)    wd_cnt<=0;
            else if (gather_progress) wd_cnt<=0;
            else if (!g8_err) begin
                if (wd_cnt >= WD_LIMIT[31:0]) begin wd_abort<=1'b1; g8_err<=1'b1; end
                else wd_cnt <= wd_cnt + 32'd1;
            end
        end
    end

    // ---------------------------------------------------------------
    // DEBUG-BLOCK-V2 progress taps (PDI#8) — ported from the serial g8_rescore.
    //   Sticky last-AR addr + RRESP-error, cleared on `start`. Read-only mirrors
    //   feed the top dbg_observe; no datapath feedback.
    // ---------------------------------------------------------------
    logic [AXI_ADDR_W-1:0] dbg_araddr_sticky;
    logic                  dbg_rresp_sticky;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            dbg_araddr_sticky <= '0;
            dbg_rresp_sticky  <= 1'b0;
        end else if (start) begin
            dbg_araddr_sticky <= '0;
            dbg_rresp_sticky  <= 1'b0;        // re-arm each run
        end else begin
            if (m_axi_arvalid && m_axi_arready)
                dbg_araddr_sticky <= m_axi_araddr;            // last AR issued
            if (m_axi_rvalid && m_axi_rready && (m_axi_rresp != 2'b00))
                dbg_rresp_sticky <= 1'b1;                     // sticky RRESP!=OKAY
        end
    end

    // Combinational progress mirrors (multi-driver-safe: read-only assigns).
    //   PIPELINED variant: `ai_cand` is the candidate row whose ARs are being
    //   issued (the closest analogue to the serial `cand_ptr` gather cursor).
    assign dbg_sel_substate  = {1'b0, state[2:0]};
    assign dbg_cand_filled   = 10'(fill);
    assign dbg_rows_gathered = 10'(ai_cand);
    assign dbg_seen_last     = seen_last;
    assign dbg_in_gather     = in_gather;
    assign dbg_last_araddr   = dbg_araddr_sticky;
    assign dbg_rresp_err     = dbg_rresp_sticky;

    // ---- PROBE V3 out3 candidate-set dump read port ----
    //   cand_idx[] (resolved vocab id) + rescored[] (per-candidate fp32 score:
    //   pre-rescore during gather, final after S_FINAL).  out3 dumps {id, score}.
    assign dbg_cand_id          = cand_idx[dbg_cand_rdaddr];
    assign dbg_cand_cheap_score = rescored[dbg_cand_rdaddr];

    // ---- SEAM-FF bridge score-feed: REGISTERED (FDRE-output) synchronous read
    //   of rescored[]. Fed the LOOKAHEAD address (bridge_rdaddr_nxt) so the value
    //   lands one cycle later = exactly when the softmax bridge consumes it (see
    //   port note). This is the FF-staging fix for the netlist tied-zero defect
    //   (reports/results/TREE_SCORE_TIEDZERO_FIX.md §Seam FF). Single-writer, one
    //   FDRE bank on the seam; the physical cross-SLR net is now this register's
    //   output, NOT a combinational async array read.
    always_ff @(posedge clk) begin
        if (!rst_n) bridge_score_ff <= '0;
        else        bridge_score_ff <= rescored[bridge_rdaddr_nxt];
    end

    // TREE_P8-RESTORE: REGISTERED conservative threshold export (see port note).
    // Registering keeps this OFF every comb arc of the selector (purely additive)
    // and only makes the threshold MORE conservative (1-cycle-lagged min <=
    // current min -> still exactly safe for the strict-gt admission). Placed
    // after state/fill/cell_score decls (xvlog strict use-before-decl).
    always_ff @(posedge clk) begin
        if (!rst_n || start) begin
            sel_thresh       <= CHEAP_NEG_INF;
            sel_thresh_valid <= 1'b0;
        end else begin
            // K1: OPT1G sorter tail = cell[kcq-1] (kcq-th best); OPT1C/F arms'
            // cmin already excludes the parked slots (they hold CHEAP_KEY_MAX).
            sel_thresh       <= G8_OPT1G ? cell_score[CIDX_W'(kcq - 1'b1)] : cmin_score;
            sel_thresh_valid <= (state == S_SELECT) && (fill == kcq);
        end
    end

    // ---- G8.3 dot pipe (II=2, unchanged) ----
    logic dot_clear, dot_in_valid, dot_in_last, dot_skip;
    logic [15:0] dot_a_fp16, dot_b_fp16;
    logic dot_out_valid;  logic [31:0] dot_score_fp32;
    // wide-dot result + feed (used only when G8_WIDE)
    logic                    wdot_out_valid;
    logic [31:0]             wdot_score_fp32;
    logic                    wdot_clear, wdot_in_valid, wdot_in_last;
    logic [WIDE_P*16-1:0]    wdot_a, wdot_b;

    generate
    if (!G8_WIDE) begin : g_scalar_dot
        // SCALAR dot: item4 MROUND register cut passed via MROUND_P.
        // SCALAR_II1=1 -> FADD_PIPE=0 (single-accumulator II=1, bit-exact, G8_DOT_II1.md).
        g8_dot_fp16 #(.FADD_PIPE(!SCALAR_II1), .MROUND(MROUND_P)) u_g8_dot (
            .clk(clk), .rst_n(rst_n), .clear(dot_clear),
            .in_valid(dot_in_valid), .in_last(dot_in_last),
            .a_fp16(dot_a_fp16), .b_fp16(dot_b_fp16),
            .out_valid(dot_out_valid), .score_fp32(dot_score_fp32));
        assign wdot_out_valid  = 1'b0;
        assign wdot_score_fp32 = 32'd0;
    end else if (WIDE_IL) begin : g_wide_dot_il
        // LEVER I: II=1 interleaved-accumulator wide dot.  Same interface as the
        // II=2 wide dot (extra .M param).  Group-feed below runs at II=1 with
        // double-buffered beats.
        g8_dot_fp16_wide_il #(.P(WIDE_P), .M(WIDE_M), .MREG(1'b1),
                              .ROW_OVERLAP(G8_ROWPP)) u_g8_dot_wide (
            .clk(clk), .rst_n(rst_n), .clear(wdot_clear),
            .in_valid(wdot_in_valid), .in_last(wdot_in_last),
            .a_fp16(wdot_a), .b_fp16(wdot_b),
            .out_valid(wdot_out_valid), .score_fp32(wdot_score_fp32));
        assign dot_out_valid  = 1'b0;
        assign dot_score_fp32 = 32'd0;
    end else begin : g_wide_dot
        // WIDE consume dot: WIDE_P parallel fp16 lanes -> adder tree -> acc (II=2).
        g8_dot_fp16_wide #(.P(WIDE_P), .MREG(1'b1)) u_g8_dot_wide (
            .clk(clk), .rst_n(rst_n), .clear(wdot_clear),
            .in_valid(wdot_in_valid), .in_last(wdot_in_last),
            .a_fp16(wdot_a), .b_fp16(wdot_b),
            .out_valid(wdot_out_valid), .score_fp32(wdot_score_fp32));
        // scalar-dot taps unused in WIDE mode
        assign dot_out_valid  = 1'b0;
        assign dot_score_fp32 = 32'd0;
    end
    endgenerate

`ifndef SYNTHESIS
    initial begin
        if (G8_ROWPP && !(G8_WIDE && WIDE_IL))
            $fatal(1, "g8_rescore_p: G8_ROWPP=1 requires G8_WIDE=1 && WIDE_IL=1");
    end
`endif

    // ===========================================================
    // P2 — speculative prefetch ring (only meaningful when G8_PREFETCH==1)
    //   Shallow ring (G8_PREFETCH_DEPTH rows x H fp16) keyed by cand id.  Rows
    //   are deposited by the prefetch path; G8 consumes HITS directly (no AR).
    //   Budget counter caps speculative fetches; drop+count when full.
    // ===========================================================
    localparam int PF_DEPTH = (G8_PREFETCH_DEPTH < 1) ? 1 : G8_PREFETCH_DEPTH;
    localparam int PF_AW    = (PF_DEPTH <= 1) ? 1 : $clog2(PF_DEPTH);
    // ring storage: one 2-port BRAM region per row slot
    logic [FP16_W-1:0] pf_row   [0:PF_DEPTH-1][0:(H<1?1:H)-1];
    logic [IDX_W-1:0]  pf_key   [0:PF_DEPTH-1];   // cand id stored in this slot
    logic              pf_valid [0:PF_DEPTH-1];
    logic [31:0]       pf_budget;                 // speculative fetches issued

    // ---- G8.4 final top-K_F heap (unchanged) ----
    logic [SCORE_W-1:0] fheap_score [0:K_F-1];
    logic [CIDX_W-1:0]  fheap_pos   [0:K_F-1];
    logic [K_F-1:0] f_gt, f_pos;  logic f_accept;
    logic [CIDX_W:0]  fin_ptr;
    logic [SCORE_W-1:0] fin_in_score;
    always_comb fin_in_score = rescored[fin_ptr[CIDX_W-1:0]];
    always_comb begin
        for (int k=0;k<K_F;k++) f_gt[k] = fp32_gt(fin_in_score, fheap_score[k]);
        f_pos='0; f_accept=1'b0;
        for (int k=0;k<K_F;k++) if (f_gt[k] && !f_accept) begin f_pos[k]=1'b1; f_accept=1'b1; end
    end
    logic [IDX_W-1:0]   fr_id    [0:K_F-1];
    logic [SCORE_W-1:0] fr_score [0:K_F-1];
    logic [CIDX_W:0]    res_ptr;

    assign prefetch_count = pf_budget[IDX_W:0];

    // =================================================================
    // MAIN
    // =================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state<=S_IDLE; done<=1'b0;
            m_axi_arvalid<=1'b0; m_axi_rready<=1'b0;
            m_axi_araddr<='0; m_axi_arlen<='0; m_axi_arsize<=3'b001;
            m_axi_arburst<=2'b01; m_axi_arid<='0; nrm_rd_addr<='0;
            desc_wptr<='0; desc_rptr<='0;
            ai_cand<='0; ai_elem<='0; ai_done<=1'b0;
            rc_active<=1'b0; rc_beats<='0; rc_elem<='0;
            dot_clear<=1'b0; dot_in_valid<=1'b0; dot_in_last<=1'b0;
            dot_a_fp16<=16'd0; dot_b_fp16<=16'd0; dot_skip<=1'b0;
            dot_busy<=1'b0; dot_owner<='0;
            // G8_CONSUME_HARDEN_P skid + owned nrm FF (inert when strap=0)
            nrm_rd_data_q<=16'd0; nrm_addr_q<='0;
            h_v<=1'b0; h_a<=16'd0; h_elem<='0;
            h_clear<=1'b0; h_last<=1'b0; h_own_set<=1'b0; h_owner<='0;
`ifndef SYNTHESIS
            dbg_fed_elem<='0;
`endif
            // G8_WIDE consume regs
            wrc_active<=1'b0; wrc_beats<='0; wrc_grp<='0; wrc_subg<='0;
            wrc_beatbuf<='0; wdot_busy<=1'b0; wdot_owner<='0;
            wdot_clear<=1'b0; wdot_in_valid<=1'b0; wdot_in_last<=1'b0;
            wdot_a<='0; wdot_b<='0; wdot_phase<=1'b0; wbeat_have<=1'b0;
            wb_head<='0; wb_tail<='0; wb_occ<='0;
            // G8_ROWPP fetch/feed + owner FIFO (inert when strap=0)
            rpp_head<='0; rpp_tail<='0; rpp_occ<='0;
            fd_ptr<='0; fd_beats<='0; fd_active<=1'b0;
            own_wp<='0; own_rp<='0;
`ifdef G8_ROWPP_RED
            red_rowcnt<='0;
`endif
            fin_ptr<='0; res_ptr<='0;
            pf_budget<='0; prefetch_drop<=1'b0;
            for (int s=0;s<PF_DEPTH;s++) begin pf_valid[s]<=1'b0; pf_key[s]<='0; end
            for (int k=0;k<K_C;k++) begin
                sel_score[k]<=CHEAP_NEG_INF; sel_idx[k]<='0;
                cand_idx[k]<='0; rescored[k]<=32'h0;
            end
            thr_score<=CHEAP_NEG_INF; thr_pos<='0; fill<='0;
            kcq<=(CIDX_W+1)'(K_C);   // K1: full shortlist until first launch-sample
            scan_ptr<='0; scan_min<=CHEAP_NEG_INF; scan_min_pos<='0; sel_state<=SEL_FILL;
            for (int g=0;g<NG_EFF;g++) begin
                grp_min[g]<=CHEAP_NEG_INF; grp_min_pos[g]<=CIDX_W'(g*GSZ);
                seg_min[g]<=CHEAP_NEG_INF; seg_pos[g]<=CIDX_W'(g*GSZ);
                // SEL_PIPE: sec of an all-NEG_INF segment = NEG_INF at the segment's
                // SECOND position (matches the pair-merge tree's recompute output).
                seg_sec[g]<=CHEAP_NEG_INF;
                seg_secpos[g]<=(GSZ>1) ? CIDX_W'(g*GSZ+1) : CIDX_W'(g*GSZ);
            end
            // SEL_PIPE registered global min == tree(seg_min init) = (-inf, pos 0)
            cmin_r_score<=CHEAP_NEG_INF; cmin_r_pos<='0;
            wr_en<=1'b0; wr_pos<='0; wr_val<=CHEAP_NEG_INF;
            // OPT1G systolic sorter cells (all empty = -inf)
            for (int k=0;k<K_C;k++) begin cell_score[k]<=CHEAP_NEG_INF; cell_idx[k]<='0; end
            seen_last<=1'b0;     // PDI#7.1 sticky last-logit capture (re-armed)
            for (int k=0;k<K_F;k++) begin
                fheap_score[k]<=FP32_NEG_INF; fheap_pos[k]<='0;
                fr_id[k]<='0; fr_score[k]<=32'h0;
            end
        end else begin
            done <= 1'b0;
            dot_clear<=1'b0; dot_in_valid<=1'b0; dot_in_last<=1'b0;
            prefetch_drop<=1'b0;
            wr_en <= 1'b0;           // OPT1D write-forward pulse (default low)

            // -------- OPT1D: registered per-group running-min --------
            //   Recompute each group's min from its GSZ slots EVERY active cycle.
            //   Reads the (pre-NBA) sel_score[] -> grp_min[g] tracks the slot
            //   array with 1-cycle latency.  GSZ-wide comb reduction per group
            //   (short path); the top-min over the NG registered grp_min[] is the
            //   combinational cmin1 above.  This is the timing-safe replacement
            //   for OPT1C's flat K_C-wide comb reduction (OOC WNS -456 ns).
            if (G8_OPT1D) begin
                for (int g=0; g<NG_EFF; g++) begin
                    automatic logic [SCORE_W-1:0] gv [0:GSP2-1];
                    automatic logic [CIDX_W-1:0]  gp [0:GSP2-1];
                    automatic int nn;
                    // BALANCED tree per-group reduction (log(GSZ) levels), so the
                    // grp_min register's input cone is shallow.
                    for (int j=0;j<GSP2;j++) begin
                        if (j>=GSZ) begin
                            gv[j] = CHEAP_KEY_INT32 ? 32'h7FFF_FFFF : 32'h7F80_0000; gp[j]='0;
                        end else begin
                            gv[j] = sel_score[g*GSZ+j]; gp[j] = CIDX_W'(g*GSZ+j);
                        end
                    end
                    nn = GSP2;
                    while (nn>1) begin
                        for (int i=0;i<nn/2;i++) begin
                            if (key_gt(gv[2*i], gv[2*i+1])) begin gv[i]=gv[2*i+1]; gp[i]=gp[2*i+1]; end
                            else                            begin gv[i]=gv[2*i];   gp[i]=gp[2*i];   end
                        end
                        nn = nn/2;
                    end
                    grp_min[g]     <= gv[0];
                    grp_min_pos[g] <= gp[0];
                end
            end

            // -------- OPT1F: REGISTERED per-segment running-min (the 100MHz fix) ----
            //   Recompute every segment's min each active cycle from its GSZ slots,
            //   FORWARDING this cycle's in-flight write (wr_en/wr_pos/wr_val set by the
            //   S_SELECT FILL/THRESH branch below).  Because the slot write is an NBA,
            //   sel_score[wr_pos] still reads its OLD value here; the forward substitutes
            //   wr_val so seg_min reflects the SAME-cycle write at T+1 (closes the
            //   stale-by-1 hole).  Balanced GSZ-wide tree -> shallow REGISTERED cone, so
            //   the comb that reaches the THRESH compare is only the SHORT NG-wide
            //   cmin0F tree over these registers.  LEFT-wins-ties => lowest index kept.
            if (G8_OPT1F && G8_SELPIPE) begin
                // -------- SEL_PIPE: joint (min, second-min) pair-merge recompute ----
                //   Same leaves + same-cycle write forward as OPT1F below, but each
                //   tree node carries (min,pos,sec,secpos).  min/pos reduction rule is
                //   IDENTICAL to the plain tree (left wins ties over ordered leaves =
                //   lowest index), so seg_min/seg_pos are bit-identical to OPT1F's.
                //   sec = second element in (value, position) order = min over the
                //   segment excluding the ONE slot at seg_pos — exactly what the
                //   cmin_r relocation update needs.  Merge rule per node:
                //     min = L (tie->L) or R (strictly smaller)
                //     if min==L: sec = mintb(sL, mR)  (tie->sL: all L pos < all R pos)
                //     if min==R: sec = mintb(mL, sR)  (tie->mL)
                //   NOTE the forward-mux select (wr_pos_c) is REGISTER-DERIVED under
                //   SELPIPE (fill or cmin_r_pos) — the entire cone starts at registers.
                for (int g=0; g<NG_EFF; g++) begin
                    automatic logic [SCORE_W-1:0] mv  [0:GSP2-1];
                    automatic logic [CIDX_W-1:0]  mp  [0:GSP2-1];
                    automatic logic [SCORE_W-1:0] sv2 [0:GSP2-1];
                    automatic logic [CIDX_W-1:0]  sp2 [0:GSP2-1];
                    automatic logic [SCORE_W-1:0] MAXK;
                    automatic logic [SCORE_W-1:0] mL, mR, sL, sR;
                    automatic logic [CIDX_W-1:0]  mLp, mRp, sLp, sRp;
                    automatic int nn;
                    automatic int slot;
                    MAXK = CHEAP_KEY_INT32 ? 32'h7FFF_FFFF : 32'h7F80_0000;
                    for (int j=0;j<GSP2;j++) begin
                        if (j>=GSZ) begin
                            mv[j] = MAXK; mp[j] = '0;
                        end else begin
                            slot = g*GSZ+j;
                            mv[j] = (wr_en_c && (int'(wr_pos_c)==slot)) ? wr_val_c
                                                                        : sel_score[slot];
                            mp[j] = CIDX_W'(slot);
                        end
                        sv2[j] = MAXK; sp2[j] = mp[j];   // singleton sec = +MAX sentinel
                    end
                    nn = GSP2;
                    while (nn>1) begin
                        for (int i=0;i<nn/2;i++) begin
                            mL=mv[2*i];   mLp=mp[2*i];   sL=sv2[2*i];   sLp=sp2[2*i];
                            mR=mv[2*i+1]; mRp=mp[2*i+1]; sR=sv2[2*i+1]; sRp=sp2[2*i+1];
                            if (key_gt(mL, mR)) begin
                                mv[i]=mR; mp[i]=mRp;
                                if (key_gt(mL, sR)) begin sv2[i]=sR; sp2[i]=sRp; end
                                else                begin sv2[i]=mL; sp2[i]=mLp; end
                            end else begin
                                mv[i]=mL; mp[i]=mLp;
                                if (key_gt(sL, mR)) begin sv2[i]=mR; sp2[i]=mRp; end
                                else                begin sv2[i]=sL; sp2[i]=sLp; end
                            end
                        end
                        nn = nn/2;
                    end
                    seg_min[g]    <= mv[0];
                    seg_pos[g]    <= mp[0];
                    seg_sec[g]    <= sv2[0];
                    seg_secpos[g] <= sp2[0];
                end
                // SEL_PIPE registered global min: fold this cycle's single write.
                // cmin_r@T+1 = lowest-index min over seg_min@T+1 by construction
                // (invariant proven inductively; A/B-gated per-cycle vs cmin0F).
                if (wr_en_c) begin
                    cmin_r_score <= cminp_nxt_score;
                    cmin_r_pos   <= cminp_nxt_pos;
                end
            end
            else if (G8_OPT1F) begin
                for (int g=0; g<NG_EFF; g++) begin
                    automatic logic [SCORE_W-1:0] gv [0:GSP2-1];
                    automatic logic [CIDX_W-1:0]  gp [0:GSP2-1];
                    automatic int nn;
                    automatic int slot;
                    for (int j=0;j<GSP2;j++) begin
                        if (j>=GSZ) begin
                            gv[j] = CHEAP_KEY_INT32 ? 32'h7FFF_FFFF : 32'h7F80_0000; gp[j]='0;
                        end else begin
                            slot = g*GSZ+j;
                            // FORWARD the SAME-CYCLE (combinational) in-flight write
                            // into this segment's tree (wr_*_c, NOT the NBA-delayed
                            // registered wr_*; see fix-2 RCA above).  Closes the
                            // 1-cycle stale-min window that dropped the rank-0 max.
                            gv[j] = (wr_en_c && (int'(wr_pos_c)==slot)) ? wr_val_c
                                                                        : sel_score[slot];
                            gp[j] = CIDX_W'(slot);
                        end
                    end
                    nn = GSP2;
                    while (nn>1) begin
                        for (int i=0;i<nn/2;i++) begin
                            if (key_gt(gv[2*i], gv[2*i+1])) begin gv[i]=gv[2*i+1]; gp[i]=gp[2*i+1]; end
                            else                            begin gv[i]=gv[2*i];   gp[i]=gp[2*i];   end
                        end
                        nn = nn/2;
                    end
                    seg_min[g] <= gv[0];
                    seg_pos[g] <= gp[0];
                end
            end

            if (G8_PREFETCH) begin
                // P2: deposit/evict housekeeping for the ring.  prefetch_req
                // admits a candidate into the running top-K_C; budget caps the
                // speculative fetches (drop+count when full).  prefetch_evict
                // invalidates a ring slot (row never dotted).  The ACTUAL row
                // bytes are deposited by the prefetch gather path; here we track
                // the key/valid bookkeeping and the budget.
                if (prefetch_req) begin
                    if (pf_budget < G8_PREFETCH_BUDGET[31:0])
                        pf_budget <= pf_budget + 32'd1;       // counted; deposit handled by ring writer
                    else
                        prefetch_drop <= 1'b1;                // budget exceeded -> drop, fall back to P1
                end
                if (prefetch_evict) begin
                    for (int s=0;s<PF_DEPTH;s++)
                        if (pf_valid[s] && pf_key[s]==prefetch_evict_id) pf_valid[s]<=1'b0;
                end
            end

            // ---------- watchdog abort wins (tear down both lanes) ----------
            if (wd_abort) begin
                m_axi_arvalid<=1'b0; m_axi_rready<=1'b0;
                dot_in_valid<=1'b0; dot_in_last<=1'b0; dot_clear<=1'b0;
                desc_wptr<=desc_rptr;            // drain descriptor FIFO
                rc_active<=1'b0; dot_busy<=1'b0; fin_ptr<='0;
                // G8_ROWPP teardown: park fetch/feed, drop pending owners
                // (guarded so ROWPP=0 stays byte-identical to the base module)
                if (G8_ROWPP) begin
                    wrc_active<=1'b0; fd_active<=1'b0; fd_ptr<=desc_rptr;
                    rpp_occ<='0; rpp_head<='0; rpp_tail<='0; own_rp<=own_wp;
                end
                for (int k=0;k<K_F;k++) begin fheap_score[k]<=FP32_NEG_INF; fheap_pos[k]<='0; end
                state<=S_FINAL;
            end else case (state)
            // -------------------------------------------------------------
            S_IDLE: if (start) begin
                // K1: launch-sample the runtime shortlist size for THIS run and
                // PARK the unused tail slots at the MAX key (see CHEAP_KEY_MAX
                // note) — the live set is exactly slots 0..kc_eff-1.
                kcq <= kc_eff;
                // K1 x J5-β: FOLDED at the W2 merge (2026-07-17) — the stripe
                // frontend takes .n_rows(kcq) and walks exactly kcq rows
                // (g8_stripe_frontend.sv n_rows_q: AR bound, per-channel
                // ai_done init, landing-count target).  The former fail-fast
                // $fatal guard is retired; the W1 guard contract (kc_eff
                // coherence) is unchanged.
                for (int k=0;k<K_C;k++) begin
                    sel_score[k] <= ((CIDX_W+1)'(k) < kc_eff) ? CHEAP_NEG_INF
                                                              : CHEAP_KEY_MAX;
                    sel_idx[k]   <= '0;
                end
                thr_score<=CHEAP_NEG_INF; thr_pos<='0; fill<='0;
                scan_ptr<='0; scan_min<=CHEAP_NEG_INF; scan_min_pos<='0; sel_state<=SEL_FILL;
                for (int g=0;g<NG_EFF;g++) begin
                    grp_min[g]<=CHEAP_NEG_INF; grp_min_pos[g]<=CIDX_W'(g*GSZ);
                    seg_min[g]<=CHEAP_NEG_INF; seg_pos[g]<=CIDX_W'(g*GSZ);
                    // SEL_PIPE per-run re-arm (matches the reset-time init)
                    seg_sec[g]<=CHEAP_NEG_INF;
                    seg_secpos[g]<=(GSZ>1) ? CIDX_W'(g*GSZ+1) : CIDX_W'(g*GSZ);
                end
                cmin_r_score<=CHEAP_NEG_INF; cmin_r_pos<='0;   // SEL_PIPE re-arm
                wr_en<=1'b0; wr_pos<='0; wr_val<=CHEAP_NEG_INF;
                // OPT1G systolic sorter cells re-armed per run (all empty = -inf)
                for (int k=0;k<K_C;k++) begin cell_score[k]<=CHEAP_NEG_INF; cell_idx[k]<='0; end
                seen_last<=1'b0;     // PDI#7.1 re-arm sticky last capture per run
                ai_cand<='0; ai_elem<='0; ai_done<=1'b0;
                rc_active<=1'b0; rc_elem<='0; rc_beats<='0;
                dot_busy<=1'b0; dot_owner<='0;
                desc_wptr<='0; desc_rptr<='0;
                state<=S_SELECT;
            end

            // -------- G8.1 streaming threshold-select -----------------------
            S_SELECT: begin
                // PDI#7.1 (G8_WEDGE_RCA fix b): latch the last-logit pulse the
                // instant it is observed, in ANY sub-state (FILL/THRESH/SCAN).
                if (cheap_valid && cheap_last) seen_last <= 1'b1;

                if (G8_OPT1G) begin
                    // ============= OPT1G: NEIGHBOR-ONLY SYSTOLIC SORTER ========
                    //   Insert the incoming logit e=(cheap_score,cheap_index) into the
                    //   DESCENDING cell array every cycle.  Each cell updates in
                    //   parallel looking ONLY at itself and cell[c-1]:
                    //     - if e > cell[c]:  cell c must shift; it takes the LARGER of
                    //         {e, cell[c-1]} that belongs at position c.  For c==0 the
                    //         upper neighbor is +inf, so cell[0] takes e (the new max
                    //         candidate among those > cell[0]); for c>0 it takes
                    //         cell[c-1] if cell[c-1] also > e (e inserts lower), else e.
                    //     - else (e <= cell[c]): cell c is unaffected, keep it.
                    //   The displaced minimum falls off cell[K_C-1] (discarded).  One
                    //   key_gt comparator + small mux per cell = CONSTANT-depth comb,
                    //   no K_C-wide reduction / decode.  EXACT global top-K_C by cheap
                    //   key (every kept element at its true rank; only the true min
                    //   ever drops) -> set BIT-IDENTICAL to OPT1C-flat.  cheap_ready is
                    //   HIGH every cycle (1 insert/cyc) -> ZERO drop.
                    if (cheap_valid) begin
                        for (int c = 0; c < K_C; c++) begin
                            if (CHEAP_KEY_INT32 ? cheap_gt(cheap_score, cell_score[c])
                                                : fp32_gt (cheap_score, cell_score[c])) begin
                                if (c == 0) begin
                                    // top cell: upper neighbor is +inf -> e lands here
                                    // (it is the largest value strictly above cell[0]).
                                    cell_score[c] <= cheap_score;
                                    cell_idx  [c] <= cheap_index;
                                end else if (CHEAP_KEY_INT32
                                                ? cheap_gt(cheap_score, cell_score[c-1])
                                                : fp32_gt (cheap_score, cell_score[c-1])) begin
                                    // e > upper neighbor too -> e was inserted ABOVE this
                                    // cell; cell c SHIFTS DOWN, taking its upper neighbor.
                                    cell_score[c] <= cell_score[c-1];
                                    cell_idx  [c] <= cell_idx  [c-1];
                                end else begin
                                    // e <= upper neighbor but e > cell c -> THIS is e's
                                    // insertion point; cell c takes e.
                                    cell_score[c] <= cheap_score;
                                    cell_idx  [c] <= cheap_index;
                                end
                            end
                            // else: e <= cell[c] -> cell c keeps its value (no write).
                        end
                        // last logit inserted: advance next cycle (cells settle via NBA).
                        if (cheap_last) state <= S_SEL_DONE;
                    end else if (seen_last) begin
                        state <= S_SEL_DONE;     // stream drained, set final
                    end
                end else if (G8_OPT1C) begin
                    // ============= OPT1C: NON-BLOCKING running-min select ======
                    //   `cheap_ready` is HIGH for ALL of S_SELECT (see always_comb
                    //   above), so the free-running tree_tile_stream loses ZERO
                    //   logits.  No SEL_SCAN phase: the min-value slot is the
                    //   combinational `cmin_pos`/`cmin_score` (computed this cycle
                    //   from the registered sel_score[]).  Each accepted logit:
                    //     * FILL phase (fill<K_C): write slot[fill] sequentially.
                    //     * THRESH phase (fill==K_C): if logit beats the current
                    //       minimum slot, OVERWRITE cmin_pos.  No rescan needed —
                    //       next cycle's cmin recomputes from the updated array.
                    //   Bit-exact top-K_C-by-value vs the lossless OPT1B golden;
                    //   the K_C-boundary tie member may differ (both are valid
                    //   top-K_C sets — gather TB serves such ids as zero rows and
                    //   the frontier is tie-invariant).
                    if (cheap_valid) begin
                        if (fill != kcq) begin           // K1 runtime shortlist
                            // FILL: sequential placement of the first kcq logits.
                            sel_score[fill[CIDX_W-1:0]] <= cheap_score;
                            sel_idx  [fill[CIDX_W-1:0]] <= cheap_index;
                            fill <= fill + 1'b1;
                            // OPT1D/OPT1F write-forward side-band (covers the FILL->THRESH
                            // boundary so the last-filled slot is reflected in the
                            // registered seg_min/cmin1 the very next cycle, no stale-min
                            // window).
                            if (G8_OPT1D || G8_OPT1F) begin
                                wr_en  <= 1'b1;
                                wr_pos <= fill[CIDX_W-1:0];
                                wr_val <= cheap_score;
                            end
                        end else begin
                            // THRESH: replace the current min slot if beaten.
                            // (!red_kill = SEL_PIPE RED-arm drop hook; constant 1
                            //  in every real build, see red_kill decl above.)
                            if ((CHEAP_KEY_INT32 ? cheap_gt(cheap_score, cmin_score)
                                                 : fp32_gt (cheap_score, cmin_score))
                                && !red_kill) begin
                                sel_score[cmin_pos] <= cheap_score;
                                sel_idx  [cmin_pos] <= cheap_index;
                                // OPT1D/OPT1F write-forward: fold this write into next
                                // cycle's registered seg_min/cmin1 (the written segment
                                // is recomputed including wr_val so the +1-cycle register
                                // latency carries no stale min).
                                if (G8_OPT1D || G8_OPT1F) begin
                                    wr_en  <= 1'b1;
                                    wr_pos <= cmin_pos;
                                    wr_val <= cheap_score;
                                end
                            end
                        end
                        if (cheap_last) state <= S_SEL_DONE;
                    end else if (seen_last) begin
                        // last already latched on a prior beat with no in-flight
                        // valid — stream drained, candidate set final.
                        state <= S_SEL_DONE;
                    end
                end else begin
                    // ============= legacy OPT1B (SEL_SCAN; repro only) ==========
                    unique case (sel_state)
                    SEL_FILL: if (cheap_valid) begin
                        sel_score[fill[CIDX_W-1:0]]<=cheap_score;
                        sel_idx  [fill[CIDX_W-1:0]]<=cheap_index;
                        if (fill == (kcq - 1'b1)) begin   // K1 runtime shortlist
                            fill<=fill+1'b1; scan_ptr<='0;
                            scan_min<=cheap_score; scan_min_pos<=fill[CIDX_W-1:0];
                            sel_state<=SEL_SCAN;
                            if (cheap_last) state<=S_SEL_DONE;
                        end else begin
                            fill<=fill+1'b1;
                            if (cheap_last) state<=S_SEL_DONE;
                        end
                    end
                    SEL_SCAN: if (scan_ptr == kcq) begin   // K1 runtime shortlist
                        thr_score<=scan_min; thr_pos<=scan_min_pos;
                        if (seen_last) state     <= S_SEL_DONE;
                        else           sel_state <= SEL_THRESH;
                    end else begin
                        if (CHEAP_KEY_INT32
                                ? cheap_gt(scan_min, sel_score[scan_ptr[CIDX_W-1:0]])
                                : fp32_gt (scan_min, sel_score[scan_ptr[CIDX_W-1:0]])) begin
                            scan_min<=sel_score[scan_ptr[CIDX_W-1:0]];
                            scan_min_pos<=scan_ptr[CIDX_W-1:0];
                        end
                        scan_ptr<=scan_ptr+1'b1;
                    end
                    SEL_THRESH: if (cheap_valid) begin
                        if (CHEAP_KEY_INT32 ? cheap_gt(cheap_score, thr_score)
                                            : fp32_gt(cheap_score, thr_score)) begin
                            sel_score[thr_pos]<=cheap_score; sel_idx[thr_pos]<=cheap_index;
                            scan_ptr<='0; scan_min<=cheap_score; scan_min_pos<=thr_pos;
                            sel_state<=SEL_SCAN;
                        end
                        if (cheap_last) state<=S_SEL_DONE;
                    end
                    default: sel_state<=SEL_FILL;
                    endcase
                end
            end

            S_SEL_DONE: begin
                // OPT1G: the selected set lives in the sorter cells (cell_idx);
                // OPT1C/F/D/B: in sel_idx.  Both feed cand_idx[] for the gather.
                if (G8_OPT1G)
                    for (int k=0;k<K_C;k++) cand_idx[k]<=cell_idx[k];
                else
                    for (int k=0;k<K_C;k++) cand_idx[k]<=sel_idx[k];
                ai_cand<='0; ai_elem<='0; ai_done<=1'b0;
                rc_active<=1'b0; rc_elem<='0;
                dot_busy<=1'b0; dot_owner<='0;
                // G8_WIDE consume-lane init
                wrc_active<=1'b0; wrc_grp<='0; wrc_subg<='0; wrc_beats<='0;
                wdot_busy<=1'b0; wdot_owner<='0; wdot_phase<=1'b0; wbeat_have<=1'b0;
                wb_head<='0; wb_tail<='0; wb_occ<='0;
                // G8_ROWPP per-run re-arm (matches reset-time init)
                rpp_head<='0; rpp_tail<='0; rpp_occ<='0;
                fd_ptr<='0; fd_beats<='0; fd_active<=1'b0;
                own_wp<='0; own_rp<='0;
`ifdef G8_ROWPP_RED
                red_rowcnt<='0;
`endif
                desc_wptr<='0; desc_rptr<='0;
                fin_ptr<='0;
                for (int k=0;k<K_F;k++) begin fheap_score[k]<=FP32_NEG_INF; fheap_pos[k]<='0; end
                state<=S_GATHER;
            end

            // =============================================================
            // S_GATHER — P1: AR-ISSUE lane and R-CONSUME lane run together.
            //   Two near-independent blocks in one always_ff (each touches a
            //   disjoint set of regs).  AR issues ahead up to G8_OUTSTANDING.
            //   J5-β G8_NCH>1: the WHOLE gather is DELEGATED to
            //   g8_stripe_frontend (launched by stripe_start_r on the
            //   S_SEL_DONE exit); this branch only LANDS the exported
            //   per-channel dot results into rescored[] (slots are disjoint
            //   by construction: ch c owns slots k%G8_NCH==c) and advances
            //   to the untouched S_FINAL when the front-end reports done.
            //   The legacy single-channel engine below is bypassed and its
            //   m_axi_* master stays idle (arvalid=0 since S_IDLE/start).
            // =============================================================
            S_GATHER: if (G8_NCH > 1) begin
                for (int c=0;c<G8_NCH;c++)
                    if (st_land_v[c])
                        rescored[st_land_slot[c]] <= st_land_score[c];
                if (st_gdone) begin
                    m_axi_arvalid<=1'b0; m_axi_rready<=1'b0;   // parity with legacy exit
                    fin_ptr<='0;
                    for (int k=0;k<K_F;k++) begin fheap_score[k]<=FP32_NEG_INF; fheap_pos[k]<='0; end
                    state<=S_FINAL;
                end
            end else if (G8_WIDE) begin
                // =========================================================
                // G8_WIDE GATHER — 512b master + P-wide consume dot.
                //   LANE 1 (AR): sub-bursts of <=MAX_BURST BEATS, each beat
                //   FPB fp16.  arsize = log2(AXI_DATA_W/8).  addr/elem step in
                //   beats.  descriptor.beats = beat count; .elem_base = element.
                //   LANE 2 (R/dot): each rvalid beat latched; GPB groups of
                //   WIDE_P drained one/cycle into g8_dot_fp16_wide.
                // =========================================================
                // -------------------- LANE 1: AR-ISSUE (WIDE) -------------
                if (!m_axi_arvalid && !ai_done &&
                    !desc_full && (outstanding < G8_OUTSTANDING[FIFO_AW:0])) begin
                    if (ai_cand == kcq) begin        // K1: gather kcq rows
                        ai_done <= 1'b1;
                    end else begin
                        // ---- G8WIDE-NOC FIX (ported from pdi_speed_g8wide_noc, F1): AXI 4KB
                        //   page-split.  ROOT CAUSE of pdi18 0-ARREADY: a full row =
                        //   ceil(H/FPB) beats * (AXI_DATA_W/8) B is up to 6144 B, and rows
                        //   start at base + cand*H*2 == {0,2048} mod 4096, so EVERY row burst
                        //   crosses a 4 KB boundary -> AXI4-illegal -> Versal NoC NMU lawfully
                        //   never asserts ARREADY (araddr stuck 0x0, watchdog, cheap
                        //   passthrough).  Cap blen_b by beats-to-next-4KB-page so no AR
                        //   crosses 4 KB.  The scalar path is 4KB-safe only because its 512 B
                        //   sub-bursts never cross; the WIDE lane lost that (capped by
                        //   MAX_BURST *beats*, not *bytes*).  Width-INDEPENDENT: STILL
                        //   required at 256b (a 192-beat 256b row also crosses 4 KB).
                        //   `G8_DISABLE_4KB_SPLIT is the SIM-ONLY negative-control escape for
                        //   the TB 4KB-legality assert (proves the gate has teeth); it is
                        //   NEVER defined by any synth/package script — the package gate
                        //   greps the synth log to assert it stayed undefined.
                        // remaining beats in this row, capped at MAX_BURST
                        automatic int unsigned rem_el = H - int'(ai_elem);
                        automatic int unsigned rem_bt = (rem_el + FPB - 1) / FPB;
                        automatic int unsigned blen_b = (rem_bt > MAX_BURST) ? MAX_BURST : rem_bt;
                        automatic int unsigned span_el;
                        automatic logic [AXI_ADDR_W-1:0] rb =
                            lmfull_hbm_base +
                            (AXI_ADDR_W'(cand_idx[ai_cand[CIDX_W-1:0]]) * (H*2));
                        automatic logic [AXI_ADDR_W-1:0] ar_start = rb + (AXI_ADDR_W'(ai_elem) * 2);
`ifndef G8_DISABLE_4KB_SPLIT
                        begin
                            automatic int unsigned BPB         = (AXI_DATA_W < 8) ? 1 : (AXI_DATA_W/8);
                            automatic int unsigned bytes_to_pg = 4096 - int'(ar_start[11:0]); // 1..4096
                            automatic int unsigned pg_beats    = bytes_to_pg / BPB;           // >=1 (beat-aligned start)
                            if (blen_b > pg_beats) blen_b = pg_beats;                          // 4 KB cap
                            if (blen_b == 0)       blen_b = 1;                                  // safety
                        end
`endif
                        span_el = blen_b * FPB;                                             // elements this AR
                        m_axi_araddr  <= ar_start;
                        m_axi_arlen   <= 8'(blen_b - 1);
                        m_axi_arsize  <= 3'(AXI_DATA_W <= 16 ? 1 :
                                          AXI_DATA_W==32 ? 2 : AXI_DATA_W==64 ? 3 :
                                          AXI_DATA_W==128 ? 4 : AXI_DATA_W==256 ? 5 :
                                          AXI_DATA_W==512 ? 6 : 7);
                        m_axi_arburst <= 2'b01;
                        m_axi_arid    <= '0;
                        m_axi_arvalid <= 1'b1;
                        desc_mem[desc_wptr[FIFO_AW-1:0]] <= '{
                            cand_ptr : ai_cand[CIDX_W-1:0],
                            elem_base: ai_elem,
                            beats    : ($clog2(MAX_BURST+1))'(blen_b),
                            row_first: (ai_elem == '0),
                            row_last : ((int'(ai_elem)+int'(span_el)) >= H),
                            from_ring: 1'b0
                        };
                        desc_wptr <= desc_wptr + 1'b1;
                        if ((int'(ai_elem)+int'(span_el)) >= H) begin
                            ai_elem <= '0;
                            ai_cand <= ai_cand + 1'b1;
                        end else begin
                            ai_elem <= ai_elem + AR_W'(span_el);
                        end
                    end
                end else if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                end

                // -------------------- LANE 2: R-CONSUME / WIDE dot --------
                //   DECOUPLED into (1) beat-fetch and (2) group-feed, so the
                //   registered AXI rready handshake is NOT entangled with the
                //   II=2 dot pacing (the entanglement deadlocked the naive
                //   single-block version: rready set-and-read same cycle).
                //   beat-fetch: keep rready high while active and the beat
                //     holding-reg is empty; latch arriving beat -> wrc_beatbuf,
                //     set wbeat_have, init wrc_subg.
                //   group-feed: II=2 paced (wdot_phase); drain WIDE_P-group from
                //     wrc_beatbuf into the wide dot; clear wbeat_have when the
                //     beat's GPB groups are spent so beat-fetch refills.
                wdot_in_valid <= 1'b0;
                wdot_clear    <= 1'b0;
                wdot_in_last  <= 1'b0;
                wdot_phase <= ~wdot_phase;

                if (G8_ROWPP) begin : g_feed_rowpp
                // ============ G8_ROWPP: decoupled FETCH/FEED, row overlap =========
                //   FETCH: fd_ptr walks the descriptor FIFO independently; beats are
                //     accepted CONTINUOUSLY across descriptor/row boundaries into the
                //     deep rpp ring (same registered-rready no-overrun invariant as
                //     the IL branch: raise only if occ-after-drain + the one possible
                //     in-flight beat still fits).
                //   FEED: II=1 from the ring head, walking desc_rptr; row_first pops
                //     are gated ONLY on owner-FIFO room (NOT on the dot flush).
                //   Beat->descriptor mapping is positional: same-ID AXI returns beats
                //     in AR issue order == descriptor order; fetch counts each
                //     descriptor's .beats, feed re-counts the same .beats.
                automatic logic feed_now, front_drain, acc_now, fd_have;
                automatic logic [RPP_WB_AW:0] occ_after_drain;
                automatic logic [8:0] fd_left;
                automatic logic row_last_grp, row_first_grp;
                feed_now        = wrc_active && (rpp_occ != 0);
                front_drain     = feed_now && (int'(wrc_subg) >= (GPB-1));
                acc_now         = (m_axi_rvalid && m_axi_rready);
                fd_have         = (fd_ptr != desc_wptr);
                fd_left         = fd_beats - (acc_now ? 9'd1 : 9'd0);
                occ_after_drain = rpp_occ - (front_drain ? 1'b1 : 1'b0);

                // ---- (1) FETCH lane ------------------------------------------
                if (!fd_active) begin
                    m_axi_rready <= 1'b0;
                    if (fd_have) begin
                        fd_beats  <= 9'(desc_mem[fd_ptr[FIFO_AW-1:0]].beats);
                        fd_active <= 1'b1;
                    end
                end else begin
                    // registered rready with in-flight-beat accounting (no overrun)
                    if ((fd_left != 0) &&
                        ((occ_after_drain + (m_axi_rready ? 1'b1 : 1'b0))
                          < RPP_WB_DEPTH[RPP_WB_AW:0])) begin
                        m_axi_rready <= 1'b1;
                    end else begin
                        m_axi_rready <= 1'b0;
                    end
                    if (acc_now) begin
                        rpp_mem[rpp_tail] <= m_axi_rdata;
                        rpp_tail <= (rpp_tail == RPP_WB_AW'(RPP_WB_DEPTH-1))
                                    ? '0 : (rpp_tail + 1'b1);
                        fd_beats <= fd_left;
                        if (fd_left == 0) begin
                            // descriptor's beats all fetched: chain to the next
                            // (1-cycle reload; the deep ring hides the dip).
                            fd_active <= 1'b0;
                            fd_ptr    <= fd_ptr + 1'b1;
                        end
                    end
                end

                // ---- (2) FEED descriptor pop (gate = owner room ONLY) --------
                if (!wrc_active) begin
                    if (!desc_empty &&
                        !(desc_mem[desc_rptr[FIFO_AW-1:0]].row_first &&
                          (own_occ >= (OWN_AW+1)'(ROWPP_OWN_DEPTH-1)))) begin
                        wrc_desc   <= desc_mem[desc_rptr[FIFO_AW-1:0]];
                        wrc_beats  <= desc_mem[desc_rptr[FIFO_AW-1:0]].beats;
                        wrc_grp    <= desc_mem[desc_rptr[FIFO_AW-1:0]].elem_base / WIDE_P;
                        nrm_rd_addr <= ($clog2((H<2)?2:H))'(
                                         desc_mem[desc_rptr[FIFO_AW-1:0]].elem_base);
                        wrc_subg   <= '0;
                        wrc_active <= 1'b1;
                    end
                end else if (feed_now) begin
                    // ---- (3) II=1 group feed from the ring head ---------------
                    row_last_grp  = ((int'(wrc_grp)+1) >= HGRP);
                    row_first_grp =
                        wrc_desc.row_first && (wrc_grp == (wrc_desc.elem_base/WIDE_P));
                    wdot_in_valid <= 1'b1;
                    wdot_clear    <= row_first_grp;
                    wdot_in_last  <= row_last_grp;
                    wdot_a        <= rpp_mem[rpp_head][wrc_subg*WIDE_P*16 +: WIDE_P*16];
                    wdot_b        <= nrm_rd_data_wide;
                    if (row_first_grp) begin
                        // owner FIFO push (replaces the single wdot_owner reg)
`ifdef G8_ROWPP_RED
                        // RED arm: corrupt ONE row's owner (+1 slot) — the A/B
                        // rescored[] compare MUST fail (gate teeth).
                        red_rowcnt <= red_rowcnt + 1'b1;
                        own_mem[own_wp[OWN_AW-1:0]] <=
                            wrc_desc.cand_ptr +
                            ((red_rowcnt == 10'(`G8_ROWPP_RED)) ? CIDX_W'(1) : CIDX_W'(0));
`else
                        own_mem[own_wp[OWN_AW-1:0]] <= wrc_desc.cand_ptr;
`endif
                        own_wp <= own_wp + 1'b1;
                    end
                    if (!row_last_grp) begin
                        wrc_grp     <= wrc_grp + 1'b1;
                        nrm_rd_addr <= nrm_rd_addr + WIDE_P;
                    end
                    if (front_drain) begin
                        rpp_head <= (rpp_head == RPP_WB_AW'(RPP_WB_DEPTH-1))
                                    ? '0 : (rpp_head + 1'b1);
                        wrc_subg <= '0;
                        wrc_beats <= wrc_beats - 1'b1;
                        if (wrc_beats == ($clog2(MAX_BURST+1))'(1)) begin
                            // last beat of this descriptor fed: retire it
                            wrc_active <= 1'b0;
                            desc_rptr  <= desc_rptr + 1'b1;
                        end
                    end else begin
                        wrc_subg <= wrc_subg + 1'b1;
                    end
                end

                // ---- (4) ring occupancy (single writer) -----------------------
                rpp_occ <= rpp_occ - (front_drain ? 1'b1 : 1'b0)
                                   + (acc_now ? 1'b1 : 1'b0);
                end else if (!WIDE_IL) begin : g_feed_ii2
                // ============= II=2 group-feed (SHIPPED WIDE, A/B default) ======
                // (1) descriptor pop
                if (!wrc_active) begin
                    if (!desc_empty &&
                        !(desc_mem[desc_rptr[FIFO_AW-1:0]].row_first && wdot_busy)) begin
                        wrc_desc   <= desc_mem[desc_rptr[FIFO_AW-1:0]];
                        wrc_beats  <= desc_mem[desc_rptr[FIFO_AW-1:0]].beats;
                        wrc_grp    <= desc_mem[desc_rptr[FIFO_AW-1:0]].elem_base / WIDE_P;
                        nrm_rd_addr <= ($clog2((H<2)?2:H))'(
                                         desc_mem[desc_rptr[FIFO_AW-1:0]].elem_base);
                        wrc_subg   <= '0;
                        wbeat_have <= 1'b0;
                        wrc_active <= 1'b1;
                    end
                end else begin
                    // (2) beat-fetch: rready high while active & no buffered beat
                    //     & beats remain.  On accept, latch beat, mark have.
                    if (!wbeat_have && (wrc_beats != 0)) begin
                        m_axi_rready <= 1'b1;
                        if (m_axi_rvalid && m_axi_rready) begin
                            wrc_beatbuf <= m_axi_rdata;
                            wbeat_have  <= 1'b1;
                            wrc_subg    <= '0;
                            m_axi_rready <= 1'b0;   // hold until this beat drained
                        end
                    end else begin
                        m_axi_rready <= 1'b0;
                    end

                    // (3) group-feed: II=2 paced; only on the feed phase, only
                    //     when a beat is buffered.
                    if (!wdot_phase && wbeat_have) begin
                        automatic logic row_last_grp = ((int'(wrc_grp)+1) >= HGRP);
                        automatic logic row_first_grp =
                            wrc_desc.row_first && (wrc_grp == (wrc_desc.elem_base/WIDE_P));
                        wdot_in_valid <= 1'b1;
                        wdot_clear    <= row_first_grp;
                        wdot_in_last  <= row_last_grp;
                        wdot_a        <= wrc_beatbuf[wrc_subg*WIDE_P*16 +: WIDE_P*16];
                        wdot_b        <= nrm_rd_data_wide;
                        if (row_first_grp) begin
                            wdot_busy <= 1'b1; wdot_owner <= wrc_desc.cand_ptr;
                        end
                        if (!row_last_grp) begin
                            wrc_grp     <= wrc_grp + 1'b1;
                            nrm_rd_addr <= nrm_rd_addr + WIDE_P;
                        end
                        // beat group accounting
                        if (int'(wrc_subg) >= (GPB-1)) begin
                            // beat drained -> retire it, allow refetch
                            wbeat_have <= 1'b0;
                            wrc_subg   <= '0;
                            wrc_beats  <= wrc_beats - 1'b1;
                            if (wrc_beats == 1) begin
                                // last beat of the sub-burst: retire descriptor
                                wrc_active <= 1'b0;
                                desc_rptr  <= desc_rptr + 1'b1;
                            end
                        end else begin
                            wrc_subg <= wrc_subg + 1'b1;
                        end
                    end
                end
                end else begin : g_feed_ii1
                // ============= LEVER I: II=1 group-feed, DOUBLE-BUFFERED ========
                //   front slot (wrc_beatbuf/wbeat_have, wrc_subg) is fed every
                //   cycle; back slot (wrc_beatbuf2/wbeat_have2) is prefetched while
                //   the front feeds, so the GPB-boundary promote never bubbles.
                //   `wrc_beats` = beats LEFT TO FETCH (decrements on ACCEPT).
                //   Descriptor retires when the last group of the sub-burst is fed
                //   (front drained, no back beat, nothing left to fetch).
                //
                //   rready is DECOUPLED (set-on-free-slot, registered) — same
                //   anti-deadlock contract as II=2: never gated on a same-cycle
                //   rvalid/feed combinational term.

                // (1) descriptor pop
                if (!wrc_active) begin
                    if (!desc_empty &&
                        !(desc_mem[desc_rptr[FIFO_AW-1:0]].row_first && wdot_busy)) begin
                        wrc_desc   <= desc_mem[desc_rptr[FIFO_AW-1:0]];
                        wrc_beats  <= desc_mem[desc_rptr[FIFO_AW-1:0]].beats;
                        wrc_grp    <= desc_mem[desc_rptr[FIFO_AW-1:0]].elem_base / WIDE_P;
                        nrm_rd_addr <= ($clog2((H<2)?2:H))'(
                                         desc_mem[desc_rptr[FIFO_AW-1:0]].elem_base);
                        wrc_subg    <= '0;
                        wb_head     <= '0;
                        wb_tail     <= '0;
                        wb_occ      <= '0;
                        wrc_active  <= 1'b1;
                    end
                end else begin
                    // ------- beat ring (depth WB_DEPTH) — robust occupancy -------
                    //   A small ring of buffered beats decouples the registered AXI
                    //   rready (1-cycle set latency) from the II=1 group-feed.  The
                    //   feed reads the head beat (wb_head) GPB groups at a time; an
                    //   accept writes the tail (wb_tail).  rready is raised only when
                    //   the post-accept occupancy can NOT exceed WB_DEPTH even with
                    //   the one in-flight beat the registered rready may already have
                    //   committed (occ + inflight < WB_DEPTH) — provably no overrun.
                    automatic logic feed_now    = (wb_occ != 0);
                    automatic logic front_drain =
                        feed_now && (int'(wrc_subg) >= (GPB-1));   // head beat empties
                    automatic logic acc_now     = (m_axi_rvalid && m_axi_rready);
                    // occupancy after this cycle's drain (head retire) but before accept
                    automatic logic [WB_AW:0] occ_after_drain = wb_occ - (front_drain ? 1'b1 : 1'b0);

                    // ------- (2) beat-fetch: rready when room for inflight beat -----
                    //   inflight = current registered rready (a beat may already be
                    //   committed to arrive).  Raise rready only if, after this
                    //   cycle's drain AND counting that inflight beat, there is still
                    //   a free slot AND beats remain to fetch.
                    // WIDE consumer-FSM deadlock fix (07-08, WIDE_CONSUMER_FSM_HUNT.md):
                    // raw wrc_beats keeps rready high one cycle past the last beat -> with
                    // OUTSTANDING=2 back-to-back cross-AR return, steals the next descriptor's
                    // first beat -> wrc_beats underflows -> wedge at rows=1. Subtract the
                    // in-flight accept. Guarded so RED(no-define)/GREEN(define) prove on ONE file.
                    // SHIPPING: unconditional (define-propagation through AVED IP-pack
                    // is a known fragility; RED/GREEN proof preserved in git 06c9af8 + fsm_hunt).
                    automatic logic [8:0] wb_left = wrc_beats - (acc_now ? 9'd1 : 9'd0);
                    if ((wb_left != 0) &&
                        ((occ_after_drain + (m_axi_rready ? 1'b1 : 1'b0)) < WB_DEPTH[WB_AW:0])) begin
                        m_axi_rready <= 1'b1;
                    end else begin
                        m_axi_rready <= 1'b0;
                    end

                    // ------- (3) group-feed: II=1, every cycle the head is valid ----
                    if (feed_now) begin
                        automatic logic row_last_grp = ((int'(wrc_grp)+1) >= HGRP);
                        automatic logic row_first_grp =
                            wrc_desc.row_first && (wrc_grp == (wrc_desc.elem_base/WIDE_P));
                        wdot_in_valid <= 1'b1;
                        wdot_clear    <= row_first_grp;
                        wdot_in_last  <= row_last_grp;
                        wdot_a        <= wb_mem[wb_head][wrc_subg*WIDE_P*16 +: WIDE_P*16];
                        wdot_b        <= nrm_rd_data_wide;
                        if (row_first_grp) begin
                            wdot_busy <= 1'b1; wdot_owner <= wrc_desc.cand_ptr;
                        end
                        if (!row_last_grp) begin
                            wrc_grp     <= wrc_grp + 1'b1;
                            nrm_rd_addr <= nrm_rd_addr + WIDE_P;
                        end
                        if (front_drain) begin
                            wb_head  <= (wb_head == WB_AW'(WB_DEPTH-1)) ? '0 : (wb_head + 1'b1);
                            wrc_subg <= '0;
                        end else begin
                            wrc_subg <= wrc_subg + 1'b1;
                        end
                    end

                    // ------- (4) beat-accept: push to ring tail ---------------------
                    if (acc_now) begin
                        wb_mem[wb_tail] <= m_axi_rdata;
                        wb_tail   <= (wb_tail == WB_AW'(WB_DEPTH-1)) ? '0 : (wb_tail + 1'b1);
                        wrc_beats <= wrc_beats - 1'b1;
                    end

                    // ------- occupancy update (drain retires head, accept adds) -----
                    wb_occ <= wb_occ - (front_drain ? 1'b1 : 1'b0) + (acc_now ? 1'b1 : 1'b0);

                    // ------- (5) descriptor retire: last group fed, ring drained ----
                    if (front_drain && (occ_after_drain == 0) && (wrc_beats == 0) && !acc_now) begin
                        wrc_active <= 1'b0;
                        desc_rptr  <= desc_rptr + 1'b1;
                    end
                end
                end

                // -------------------- WIDE dot flush -> rescored[] --------
                //   G8_ROWPP: results arrive strictly in row order (the combine
                //   tree is a fixed-latency pipe fed one snapshot per row), so
                //   the owner FIFO head names the slot; pop it per out_valid.
                if (wdot_out_valid) begin
                    if (G8_ROWPP) begin
                        rescored[own_mem[own_rp[OWN_AW-1:0]]] <= wdot_score_fp32;
                        own_rp <= own_rp + 1'b1;
                    end else begin
                        rescored[wdot_owner] <= wdot_score_fp32;
                        wdot_busy <= 1'b0;
                    end
                end

                // -------------------- gather complete? (WIDE) -------------
                //   G8_ROWPP: `own_wp==own_rp` (all pushed owners flushed)
                //   replaces `!wdot_busy` — it covers every row still in the
                //   dot/combine pipe, not just the last one.
                if (G8_ROWPP
                    ? (ai_done && desc_empty && !wrc_active && !wdot_in_valid
                       && (own_wp == own_rp))
                    : (ai_done && desc_empty && !wrc_active && !wdot_in_valid
                       && !wdot_busy)) begin
                    m_axi_arvalid<=1'b0; m_axi_rready<=1'b0;
                    fin_ptr<='0;
                    for (int k=0;k<K_F;k++) begin fheap_score[k]<=FP32_NEG_INF; fheap_pos[k]<='0; end
                    state<=S_FINAL;
                end
            end else begin
                // -------------------- LANE 1: AR-ISSUE --------------------
                // Issue the next sub-burst AR if: not all issued, AR slot free
                // (no AR currently pending acceptance), and the descriptor FIFO
                // has room AND fewer than G8_OUTSTANDING in flight.
                if (!m_axi_arvalid && !ai_done &&
                    !desc_full && (outstanding < G8_OUTSTANDING[FIFO_AW:0])) begin
                    if (ai_cand == kcq) begin        // K1: gather kcq rows
                        ai_done <= 1'b1;                 // all rows' ARs issued
                    end else begin
                        automatic int unsigned rem  = H - int'(ai_elem);
                        automatic int unsigned blen = (rem > MAX_BURST) ? MAX_BURST : rem;
                        automatic logic [AXI_ADDR_W-1:0] rb =
                            lmfull_hbm_base +
                            (AXI_ADDR_W'(cand_idx[ai_cand[CIDX_W-1:0]]) * (H*2));
                        // P2: is this row already in the prefetch ring?  (hit
                        // => no AR; the consume lane reads from pf_row.)  The
                        // dot math is identical, so bit-exactness is preserved.
                        automatic logic ring_hit = 1'b0;
                        if (G8_PREFETCH) begin
                            for (int s=0;s<PF_DEPTH;s++)
                                if (pf_valid[s] && pf_key[s]==cand_idx[ai_cand[CIDX_W-1:0]])
                                    ring_hit = 1'b1;
                        end
                        if (!ring_hit) begin
                            m_axi_araddr  <= rb + (AXI_ADDR_W'(ai_elem) * 2);
                            m_axi_arlen   <= 8'(blen - 1);   // <=255 sub-burst bound
                            m_axi_arsize  <= 3'b001;
                            m_axi_arburst <= 2'b01;
                            m_axi_arid    <= '0;             // SAME ID => in-order R
                            m_axi_arvalid <= 1'b1;
                        end
                        // push descriptor for this sub-burst (ring or AXI source)
                        desc_mem[desc_wptr[FIFO_AW-1:0]] <= '{
                            cand_ptr : ai_cand[CIDX_W-1:0],
                            elem_base: ai_elem,
                            beats    : ($clog2(MAX_BURST+1))'(blen),
                            row_first: (ai_elem == '0),
                            row_last : ((int'(ai_elem)+int'(blen)) >= H),
                            from_ring: ring_hit
                        };
                        desc_wptr <= desc_wptr + 1'b1;
                        // advance issue cursor
                        if ((int'(ai_elem)+int'(blen)) >= H) begin
                            ai_elem <= '0;
                            ai_cand <= ai_cand + 1'b1;   // next row
                        end else begin
                            ai_elem <= ai_elem + AR_W'(blen);
                        end
                    end
                end else if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;               // AR accepted; slot frees
                end

                // -------------------- LANE 2: R-CONSUME / dot --------------
                // Pop a descriptor when idle; drain its beats in arrival order
                // (same-ID guarantee), feed the II=2 dot.  A NEW ROW's first
                // beat (dot_clear) is gated on dot_busy==0 so the prior row's
                // single-accumulator dot has flushed (out_valid) -> bit-exact.
                if (!rc_active) begin
                    if (!desc_empty &&
                        !(desc_mem[desc_rptr[FIFO_AW-1:0]].row_first && dot_busy)) begin
                        rc_desc   <= desc_mem[desc_rptr[FIFO_AW-1:0]];
                        rc_beats  <= desc_mem[desc_rptr[FIFO_AW-1:0]].beats;
                        rc_elem   <= desc_mem[desc_rptr[FIFO_AW-1:0]].elem_base;
                        nrm_rd_addr <= ($clog2((H<2)?2:H))'(
                                         desc_mem[desc_rptr[FIFO_AW-1:0]].elem_base);
                        rc_active <= 1'b1;
                        dot_skip  <= 1'b0;
                        // for an AXI sub-burst, raise rready; ring sub-bursts
                        // self-feed (no AXI read).
                        m_axi_rready <= !desc_mem[desc_rptr[FIFO_AW-1:0]].from_ring;
                    end
                end else if (rc_desc.from_ring) begin
                    // ---- P2 RING source: feed beats from pf_row (no AXI) ----
                    if (!dot_skip) begin
                        automatic logic [FP16_W-1:0] ring_a;
                        automatic int slot; slot = 0;
                        for (int s=0;s<PF_DEPTH;s++)
                            if (pf_valid[s] && pf_key[s]==cand_idx[rc_desc.cand_ptr]) slot = s;
                        ring_a = pf_row[slot][rc_elem];
                        begin
                            automatic logic row_last_elem = (rc_elem == (H-1));
                            automatic logic sub_last_beat  = (rc_beats == ($clog2(MAX_BURST+1))'(1));
                            dot_in_valid <= 1'b1;
                            dot_clear    <= rc_desc.row_first && (rc_elem == rc_desc.elem_base);
                            dot_in_last  <= row_last_elem;
                            dot_a_fp16   <= ring_a;
                            dot_b_fp16   <= nrm_rd_data;
`ifndef SYNTHESIS
                            // pdi21 G8RESC dot_a/dot_b ALIGNMENT CANARY (RCA
                            // project_pdi20_g8resc_silicon_staging_rca): dot_a is element
                            // rc_elem of the row; dot_b=nrm[nrm_rd_addr]. nrm_rd_addr & rc_elem
                            // are updated in lockstep (elem_base at pop, +1 together) so they
                            // MUST be equal at every feed — a mismatch = beat-to-dot staging
                            // drift. Sim-only (compiled out for synth).
                            if (nrm_rd_addr !== ($clog2((H<2)?2:H))'(rc_elem))
                                $error("G8RESC ALIGN(RING): nrm_rd_addr=%0d != rc_elem=%0d (dot_a/dot_b skew)",
                                       nrm_rd_addr, rc_elem);
`endif
                            if (rc_desc.row_first && (rc_elem == rc_desc.elem_base)) begin
                                dot_busy  <= 1'b1; dot_owner <= rc_desc.cand_ptr;
                            end
                            if (!row_last_elem) begin
                                rc_elem     <= rc_elem + 1'b1;
                                nrm_rd_addr <= nrm_rd_addr + 1'b1;
                            end
                            rc_beats <= rc_beats - 1'b1;
                            if (sub_last_beat) begin
                                rc_active <= 1'b0;
                                desc_rptr <= desc_rptr + 1'b1;
                            end else begin
                                dot_skip <= 1'b1;
                            end
                        end
                    end else begin
                        dot_skip <= 1'b0;
                    end
                end else if (G8_CONSUME_HARDEN_P) begin
                    // ---- P1 AXI source (HARDENED): INTAKE lane -----------------
                    //   Read-latency hardening of the scalar gather->dot consume.
                    //   The baseline captures dot_a=lm[k] (this AXI beat) and
                    //   dot_b=nrm_rd_data COMBINATIONALLY in the SAME cycle at the
                    //   SAME element k.  On silicon nrm_buf can BRAM-demote (hidden
                    //   DOUT reg) -> dot_b=nrm[k-1] STALE-by-1, xsim-green/silicon-
                    //   only, and the address-only canary is BLIND to it.  HARDENED
                    //   mode OWNS the nrm read in an explicit FDRE (nrm_rd_data_q)
                    //   captured here at element k, and the SEPARATE feed block
                    //   (below, drains every cycle) feeds the dot from a 1-deep skid
                    //   one cycle later, so dot_a and dot_b are a MATCHED pair for
                    //   element k with a deterministic OWNED 1-cycle latency (immune
                    //   to the nrm_buf comb-read timing/attribute).  Requires
                    //   SCALAR_II1 (II=1, no dot_skip); +1 pipeline latency only.
                    //   (h_v occupancy + dot feed are owned by the standalone feed
                    //   block so the LAST beat drains after the descriptor retires.)
                    m_axi_rready <= 1'b1;                 // II=1 continuous intake
                    if (sc_hard_beat) begin              // = rvalid&&rready this cyc
                        automatic logic row_last_elem = (rc_elem == (H-1));
                        automatic logic sub_last_beat  = (rc_beats == ($clog2(MAX_BURST+1))'(1));
                        // G8_LANE_FIX (pdi24 G0): narrow-burst active lane = addr[1]
                        // == rc_elem[0] (sub-burst bases even). Odd elems on [31:16].
                        h_a       <= (G8_LANE_FIX && rc_elem[0])
                                     ? m_axi_rdata[SC_LANE_HI -: 16]
                                     : m_axi_rdata[15:0];
                        h_elem    <= rc_elem;
                        h_clear   <= rc_desc.row_first && (rc_elem == rc_desc.elem_base);
                        h_last    <= row_last_elem;
                        h_own_set <= rc_desc.row_first && (rc_elem == rc_desc.elem_base);
                        h_owner   <= rc_desc.cand_ptr;
                        // OWN the nrm read: sample the comb read at element k NOW;
                        // available (matched) at the feed cycle as nrm_rd_data_q.
                        nrm_rd_data_q <= nrm_rd_data;
                        nrm_addr_q    <= ($clog2((H<2)?2:H))'(rc_elem);
                        if (!row_last_elem) begin
                            rc_elem     <= rc_elem + 1'b1;
                            nrm_rd_addr <= nrm_rd_addr + 1'b1;
                        end
                        rc_beats <= rc_beats - 1'b1;
                        if (sub_last_beat) begin
                            m_axi_rready <= 1'b0;         // sub-burst retire
                            rc_active    <= 1'b0;
                            desc_rptr    <= desc_rptr + 1'b1;
                        end
                    end
                end else begin
                    // ---- P1 AXI source: II=2 pacing identical to g8_rescore --
                    if (!dot_skip) begin
                        m_axi_rready <= 1'b1;
                        if (m_axi_rvalid && m_axi_rready) begin
                            automatic logic row_last_elem  = (rc_elem == (H-1));
                            automatic logic sub_last_beat   = (rc_beats == ($clog2(MAX_BURST+1))'(1));
                            dot_in_valid <= 1'b1;
                            dot_clear    <= rc_desc.row_first && (rc_elem == rc_desc.elem_base);
                            dot_in_last  <= row_last_elem;
                            // G8_LANE_FIX (pdi24 G0): narrow-burst active lane =
                            // addr[1] == rc_elem[0]. Odd elems arrive on [31:16].
                            dot_a_fp16   <= (G8_LANE_FIX && rc_elem[0])
                                            ? m_axi_rdata[SC_LANE_HI -: 16]
                                            : m_axi_rdata[15:0];
                            dot_b_fp16   <= nrm_rd_data;
`ifndef SYNTHESIS
                            // pdi21 G8RESC dot_a/dot_b ALIGNMENT CANARY (RCA
                            // project_pdi20_g8resc_silicon_staging_rca): dot_a=lm[cand,rc_elem]
                            // (this AXI beat) paired with dot_b=nrm[nrm_rd_addr]; nrm_rd_addr &
                            // rc_elem are updated in lockstep (elem_base at pop, +1 together) so
                            // they MUST be equal at every feed — a mismatch = the OUTSTANDING
                            // beat-handoff drift this build removes (8->2). Sim-only.
                            if (nrm_rd_addr !== ($clog2((H<2)?2:H))'(rc_elem))
                                $error("G8RESC ALIGN(AXI): nrm_rd_addr=%0d != rc_elem=%0d (dot_a/dot_b skew)",
                                       nrm_rd_addr, rc_elem);
`endif
                            // latch dot ownership on the row's FIRST element fed
                            if (rc_desc.row_first && (rc_elem == rc_desc.elem_base)) begin
                                dot_busy  <= 1'b1; dot_owner <= rc_desc.cand_ptr;
                            end
                            if (!row_last_elem) begin
                                rc_elem     <= rc_elem + 1'b1;
                                nrm_rd_addr <= nrm_rd_addr + 1'b1;
                            end
                            rc_beats <= rc_beats - 1'b1;
                            if (sub_last_beat) begin
                                // sub-burst exhausted: retire descriptor.
                                m_axi_rready <= 1'b0;
                                rc_active    <= 1'b0;
                                desc_rptr    <= desc_rptr + 1'b1;
                                // row finalize happens when the dot flushes; the
                                // dot_owner side-band (NOT rc_desc) names the slot.
                            end else if (SCALAR_II1) begin
                                // II=1 (G8_DOT_II1.md): feed the dot EVERY cycle — keep
                                // rready high, NO dot_skip gap. rvalid still gates each
                                // feed so no consume without data; per-beat a/b pairing
                                // (rc_elem/nrm_rd_addr lockstep + canary) unchanged.
                                m_axi_rready <= 1'b1;
                            end else begin
                                m_axi_rready <= 1'b0;
                                dot_skip     <= 1'b1;        // II=2 gap
                            end
                        end
                    end else begin
                        m_axi_rready <= 1'b1;
                        dot_skip     <= 1'b0;
                    end
                end

                // ---- HARDENED matched-1-stage FEED (drains every cycle) ------
                //   Owns the skid occupancy h_v and the dot feed, so the LAST
                //   captured beat is fed AFTER the intake retires the descriptor.
                //   Feeds the beat captured LAST cycle: dot_a=h_a (element h_elem)
                //   paired with dot_b=nrm_rd_data_q (OWNED FDRE = nrm[h_elem]).
                if (G8_CONSUME_HARDEN_P && !G8_WIDE) begin
                    if (h_v) begin
                        dot_in_valid <= 1'b1;
                        dot_clear    <= h_clear;
                        dot_in_last  <= h_last;
                        dot_a_fp16   <= h_a;
                        dot_b_fp16   <= nrm_rd_data_q;
                        if (h_own_set) begin dot_busy <= 1'b1; dot_owner <= h_owner; end
                        // beat-index consistency (synthesizable teeth): the OWNED
                        // nrm address MUST equal the fed beat's element; a mismatch
                        // = a descriptor/beat handoff DRIFT (never in nominal op).
                        if (nrm_addr_q != ($clog2((H<2)?2:H))'(h_elem))
                            g8_err <= 1'b1;
`ifndef SYNTHESIS
                        dbg_fed_elem <= h_elem;           // TB DATA-VALUE canary tap
                        if (nrm_addr_q !== ($clog2((H<2)?2:H))'(h_elem))
                            $error("G8RESC HARDEN IDX: nrm_addr_q=%0d != h_elem=%0d (beat/nrm drift)",
                                   nrm_addr_q, h_elem);
`endif
                    end
                    // occupancy: full iff a beat was captured this cycle (the old
                    // skid entry, if any, was just fed above -> a clean 1-deep pipe)
                    h_v <= sc_hard_beat;
                end

                // -------------------- dot flush -> rescored[] --------------
                // A row's last element set dot_in_last; when the pipe flushes,
                // latch the score into the candidate slot it belongs to — using
                // the SIDE-BAND dot_owner (spec §4a fix), NOT rc_desc.cand_ptr
                // (which may already point at a later sub-burst's candidate).
                if (dot_out_valid) begin
                    rescored[dot_owner] <= dot_score_fp32;
                    dot_busy <= 1'b0;            // row's dot retired; next row's clear may proceed
                end

                // -------------------- gather complete? --------------------
                // All ARs issued AND FIFO drained AND no row mid-consume AND the
                // dot pipe idle (no busy row, no pending last feed) -> finalize.
                // HARDENED: also require the 1-deep consume skid drained (h_v==0)
                // so the last element's matched dot feed is not lost.
                if (ai_done && desc_empty && !rc_active && !dot_in_valid && !dot_busy
                    && (!G8_CONSUME_HARDEN_P || !h_v)) begin
                    m_axi_arvalid<=1'b0; m_axi_rready<=1'b0;
                    fin_ptr<='0;
                    for (int k=0;k<K_F;k++) begin fheap_score[k]<=FP32_NEG_INF; fheap_pos[k]<='0; end
                    state<=S_FINAL;
                end
            end

            // -------------------- G8.4 top-K_F (unchanged) ----------------
            S_FINAL: if (fin_ptr != kcq) begin   // K1: scan only the kcq rescored
                if (f_accept) begin
                    for (int k=K_F-1;k>=0;k--) begin
                        if (f_pos[k]) begin
                            fheap_score[k]<=fin_in_score; fheap_pos[k]<=fin_ptr[CIDX_W-1:0];
                        end else if (k>0) begin
                            logic any_above; any_above=1'b0;
                            for (int j=0;j<k;j++) if (f_pos[j]) any_above=1'b1;
                            if (any_above) begin
                                fheap_score[k]<=fheap_score[k-1];
                                fheap_pos[k]  <=fheap_pos[k-1];
                            end
                        end
                    end
                end
                fin_ptr<=fin_ptr+1'b1;
            end else begin res_ptr<='0; state<=S_RESOLVE; end

            S_RESOLVE: if (res_ptr != K_F[CIDX_W:0]) begin
                fr_id[res_ptr[CIDX_W-1:0]]    <= cand_idx[fheap_pos[res_ptr[CIDX_W-1:0]]];
                fr_score[res_ptr[CIDX_W-1:0]] <= rescored[fheap_pos[res_ptr[CIDX_W-1:0]]];
                res_ptr<=res_ptr+1'b1;
            end else begin done<=1'b1; state<=S_DONE; end

            S_DONE: state<=S_IDLE;
            default: state<=S_IDLE;
            endcase
        end
    end

    genvar gk;
    generate
        for (gk=0; gk<K_F; gk++) begin : g_pack
            assign frontier_ids   [gk*IDX_W   +: IDX_W]   = fr_id[gk];
            assign frontier_scores[gk*SCORE_W +: SCORE_W] = fr_score[gk];
        end
    endgenerate

endmodule

`default_nettype wire
