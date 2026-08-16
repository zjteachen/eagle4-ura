// =====================================================================
// eagle4_config.svh — THE SINGLE SOURCE OF TRUTH FOR EVERY RTL CONFIG TOGGLE
// =====================================================================
// USER MANDATE (2026-07-22): "all these config parameter toggles shall be in
// the top file and double checked ... migrate all such toggle controls to one
// single file. And change can be visible at once."
//
// WHY THIS FILE EXISTS — three real burns, all of the SAME class:
//   1. HEAD_PAR   : 0 occurrences in pkg_speedwave_v2.tcl => never strappable
//                   => PDI 0xE4B71118 shipped SERIAL G5 after a 28 h build +
//                   flash + cold-AC + full campaign (reports/perf/PDI_0xE4B71118_6WAY.md).
//   2. DUAL_PORT  : declared ONLY inside gemv_stage_array_scaled_slr.sv, never
//                   bound at any instantiation => a proven 1.28-1.54x lever is
//                   UNREACHABLE (reports/results/HBM_MULTICHANNEL_PLAN.md L1).
//   3. DEPTH_N /  : top parameters but absent from the packaging ipx list =>
//      EMBG_EN      the BD re-synthesises from component.xml DEFAULTS and the
//                   strap silently evaporates (TREE_FSM_GAP6_STRAP_CHAIN.md).
//
// RULES (enforced mechanically — do not break them):
//   R1. EVERY top-level parameter default in eagle4_drafter.sv is `EAGLE4_<NAME>
//       from THIS file. Change a value HERE, nowhere else.
//   R2. The packaging ipx user/hdl parameter list is GENERATED from this file
//       by scripts/gen_pkg_param_list.sh -> pkg/pkg_param_list.tcl. Coverage
//       therefore CANNOT drift: a new toggle added here is packaging-visible
//       the next time the generator runs (and the generator runs in the
//       packaging flow itself).
//   R3. Each toggle carries [CLASS] + one-line purpose + NETLIST SIGNATURE
//       (the thing you grep in the BUILT netlist to prove it took effect).
//   R4. Class C (declared module-local, UNWIRED to top) toggles are listed at
//       the bottom as comments. They are NOT threaded blind — the gap is
//       VISIBLE here instead of silent.
//
// CLASSES:
//   [A] live strap        - top param AND packaging-visible (strappable today)
//   [B] top param         - was packaging-INVISIBLE before this migration; the
//                           generated list in R2 promotes every B to A.
//   [D] derived           - not independently settable (function of another)
//   [C] UNWIRED           - module-local, no `.NAME(` binding anywhere (bottom)
//   [E] dead / vestigial  - present but no live effect (bottom)
//
// GATE: scripts/check_pkg_param_coverage.sh <pkg> <top>  must be all-PASS.
// =====================================================================
`ifndef EAGLE4_CONFIG_SVH
`define EAGLE4_CONFIG_SVH

// ---------------------------------------------------------------------
// DEFINE-DERIVED STRAPS (must be passed to BOTH xvlog AND xelab: -d NAME=v).
// NOTE: pkg_speedwave_v2.tcl CLEARS verilog_define on the packaged IP, so a
// `ifdef can NEVER ship in a PDI — a define-only toggle is sim-only by
// construction. Anything that must ship is a PARAMETER below.
// ---------------------------------------------------------------------
`ifndef DEPTH_N_V
  `define DEPTH_N_V 0
`endif
`ifndef EMBG_EN_V
  `define EMBG_EN_V 0
`endif
`ifndef FORREASON_V
  `define FORREASON_V 0
`endif
// F4: physical `a`-bank count. 1 = F1/depth-1 (all children share root a_0).
// >1 = F4 per-parent re-gather. 16 = the depth>=2 ship value: 8 survivor slots
// x 2 LEVEL-PARITY epochs (double-buffer). One 8-slot buffer is NOT enough —
// chain.cpp:491-495 keeps parent_for_reasoning[K_F] SEPARATE from
// child_for_reasoning[K_F], because a_d[k]->bank k would clobber a_{d-1}[k]
// mid-level (children are sequential; child k writes before a later sibling
// whose parent slot == k reads). The epoch MSB gives the ping-pong for free.
`ifndef FR_KF_BANKS_V
  `define FR_KF_BANKS_V 1
`endif

// FR_ABANK_BRAM_V — a-bank storage style (CONGFIX-1). 0 = FF register file
// (shipping/default, byte-identical); 1 = SDP BRAM (frees ~786K FF + ~286K LUT,
// costs 64 RAMB36; SLICE 97.94% -> ~82-84%). ONLY legal at FR_KF_BANKS>8, where
// the level-parity double buffer (FR_DBUF) guarantees read/write banks differ —
// the producer enforces this itself with (KF_BANKS>8); a collision would be
// SILICON-DEAD but XSIM-GREEN (real RAMB36 SDP returns undefined; both RTL
// models return pre-write data), so the guard is the only defence.
`ifndef FR_ABANK_BRAM_V
  `define FR_ABANK_BRAM_V 0
`endif
// EMB_NCH: HBM read channels for the depth-N child-embedding gather.
// 1 = the shipping single-channel S08 path (m_axi_emb -> HBM5_PORT1).
// 2 = intra-row stripe: channel c reads elements [c*H/2,(c+1)*H/2) of the
//     SAME unmodified embed_tokens image through a SECOND port of the SAME
//     HBM PC (m_axi_embb -> S09_AXI -> HBM5_PORT0) => no duplicate image,
//     no host relayout, no ABI change.  MEASURED gain is ~0 (the landing
//     buffer takes 1 chunk/cyc, which one 256b channel already saturates:
//     199 -> 197 cyc/row) => DEFAULT STAYS 1.  See
//     reports/results/EMB_GATHER_MULTICHANNEL.md for the full reconciliation.
`ifndef EMB_NCH_V
  `define EMB_NCH_V 1
`endif
// ENORM_D0: apply the depth-N 27/8 e_norm requant at LEVEL-0 too (real token
// embeds rms~0.02 saturate the 31/10 grid — the shipped-root accept lever).
// Default 1 = ACTIVE in production (a default-0 lever would be inert, the
// DUAL_PORT/HEAD_PAR trap). Only meaningful with DEPTH_N=1 (the 27/8 instance
// is generate-elided at DEPTH_N=0 -> depth-0 NETLIST byte-identical either way).
`ifndef ENORM_D0_V
  `define ENORM_D0_V 1
`endif
// PRECISION_SAFE: atomic fixed-point contract for exact root shortlist parity.
// 0 preserves the current production grids byte-for-byte. 1 requires the
// embedding input buffer to be loaded as signed INT16 frac16 and moves all
// dependent seams together; never enable only a subset of this contract.
`ifndef PRECISION_SAFE_V
  `define PRECISION_SAFE_V 0
`endif
// HEAD_PAR_V — G5 head-parallelism (ported from speedwave_v2_g5mem 0x18).
// 1 = SERIAL G5 (DEFAULT): ONE attention_head_assembly_perbank_sbuf time-
//     multiplexes all N_HEADS=24 heads (the ps_tree_fsm gate-2-4-proven config,
//     BYTE-IDENTICAL at this value). 6 = HEAD_PAR engines run in LOCKSTEP,
//     engine e bound to heads {e, e+HEAD_PAR, ...}, N_GRP = 24/6 = 4 iterations
//     => the ctx-linear attention term 552/ctx -> 92/ctx (6.00x). This is the
//     0x18 burn parameter (HEAD_PAR had 0 pkg refs => shipped SERIAL after a
//     28 h build): it MUST be packaging-visible (gen_pkg_param_list) and
//     netlist-verified. Composes with the depth-N KV lane-chain (orthogonal:
//     head-parallelism vs position-mapping) — each engine applies the SAME
//     lane-chain remapped read position; each engine carries its OWN F3
//     residency mirror bitmap (per-engine mirror, tech-lead ruling 2026-07-26).
`ifndef HEAD_PAR_V
  `define HEAD_PAR_V 1
`endif
// DUAL_PORT_V — TDP dual-read GEMV (Stage-B', WO-025). 0 = single-port lane
// (DEFAULT, bit+cycle identical). 1 = read 2 K-chunks/cycle via URAM port B
// (RD_PORTS=2 + gemv_lane_scaled_stream_dual, now WLAT-fixed for the SLR seam)
// -> per-column active cyc halves (G9 521->265). REQUIRES the body activation
// seams to source 2 chunks/cycle (a_chunk_b off chunk_idx_b_out). Threaded to
// gemv_stage_array_scaled_slr .DUAL_PORT. NETLIST SIGNATURE: URAM288 with BOTH
// port-A and port-B RDB_WR active on the weight banks AND URAM COUNT UNCHANGED
// (the [Synth 8-3971] true-dual-port -> BRAM-replication trap) AND seam->lane
// FDRE depth port-A == port-B.
`ifndef DUAL_PORT_V
  `define DUAL_PORT_V 0
`endif
// LANES_V — GEMV array width (LANES-256 work order, LANES256_GEMM_STUDY.md).
// 128 = DEFAULT (byte+cycle identical, the certified shape). 256 = double
// the output columns per pass: per-stage col-tiles halve (G4 24->12/sub,
// G7 64->32, G9 48->24, G12/TREE 1002->501) -> GEMV block exactly 2.00x;
// stacked on DUAL_PORT=1 this is the QUAD (measured-model forward
// 129,927 -> ~100,694 cyc = 1.29x on the landed 1.77x). URAM-NEUTRAL and
// BRAM-NEUTRAL (1,536 URAM / 1,024 RAMB) ONLY through the two coupled
// mechanics that now DERIVE from this value: bank cascade_height=8 packing
// (6 URAM/48K bank; naive map = 8 -> 2,048 total = DEAD) and the halved
// BANK_DEPTH/SCALE_DEPTH (a non-halved SCALE_DEPTH costs +1,024 RAMB =
// DEAD). Geometry (stage bases/cols) is lanes-PARAMETRIC end to end
// (stage_geometry.svh functions take lanes; the SG_LANES define is DELETED
// — a define could never ship, the 0xE4B71118 class). 192 is DEAD-by-
// geometry (G7 8192/192 not integer) and rejected at elaboration.
`ifndef LANES_V
  `define LANES_V 128
`endif

// =====================================================================
// GEOMETRY / DATAPATH SHAPE
// =====================================================================
// [A] LANES | GEMV output columns per pass. 128 = default (certified, byte+cycle identical); 256 = LANES-256 quad lever (strap in the build, LANES256_GEMM_STUDY.md §6). | netlist: LANES gemv_lane_* instances under u_array; @256 ALSO: every u_bank "48 K x 32 -> URAM288=6, Matrix 6x1" (cascade_height=8) + ZERO u_bank rows in the Block RAM mapping (spill signature) + zero [Synth 8-3971] + scale banks 4 RAMB36/lane (SCALE_DEPTH=8192) + G12 weight stage base 40704 (not 81408) + tree tile terminal 501
`define EAGLE4_LANES              `LANES_V
// [B] K_MAX | largest K across all GEMVs (G9 8192) — sizes chunk counters | netlist: chunk counter width = $clog2(K_MAX/K_PAR)
`define EAGLE4_K_MAX              8192
// [B] K_PAR | reduction width per URAM read (16; 32/64 ooc-round to 1,920 URAM = over the 1,925 wall) | netlist: 16 MAC taps per lane CSA tree
`define EAGLE4_K_PAR              16
// [B] A_WIDTH | activation bit width on the seams | netlist: seam data bus = LANES*A_WIDTH bits
`define EAGLE4_A_WIDTH            16
// [B] TREE_V_MAX | tree_decode vocabulary scan length (Llama-3.2-3B vocab) | netlist: vocab counter width 17b
`define EAGLE4_TREE_V_MAX         128256
// [B] ARR_GROUPS | GEMV array SLR groups (3 = 3-SLR shed) | netlist: 3 genblk arr_grp[*] scopes
`define EAGLE4_ARR_GROUPS         3
// [B] ARR_PIPE_STAGES | registered pipe stages between array groups | netlist: FDRE depth on the inter-group result bus
`define EAGLE4_ARR_PIPE_STAGES    1

// =====================================================================
// BOOT / WEIGHT-DMA MASTER GEOMETRY
// =====================================================================
// [B] BOOT_AXI_ADDR_W | m_axi_boot address width | netlist: m_axi_boot_araddr[63:0]
`define EAGLE4_BOOT_AXI_ADDR_W    64
// [B] BOOT_AXI_DATA_W | m_axi_boot data width | netlist: m_axi_boot_rdata[31:0]
`define EAGLE4_BOOT_AXI_DATA_W    32
// [B] BOOT_AXI_ID_W | m_axi_boot id width | netlist: m_axi_boot_arid[0:0]
`define EAGLE4_BOOT_AXI_ID_W      1
// [B] BOOT_BURST_LEN | boot-DMA burst beats (64; arlen must stay <=256, see feedback_axi_arlen_8bit_truncation_h3072) | netlist: arlen literal 8'h3F
`define EAGLE4_BOOT_BURST_LEN     64

// =====================================================================
// OUTPUT-DMA ABI
// =====================================================================
// [B] OUT_ABI_V24 | 1 = v24 SGLang host ABI (ids/scores/parents); 0 = M7 simple layout | netlist: u_out_dma ABI_V24 generate arm
`define EAGLE4_OUT_ABI_V24        1'b1
// [B] OUT_TOP_K | out-DMA stride (8 live + 8 pad) | netlist: out address stride literal
`define EAGLE4_OUT_TOP_K          16
// [B] OUT_PARENT_SENTINEL | root-parent sentinel value written by out-DMA | netlist: -1 constant on the parents write path
`define EAGLE4_OUT_PARENT_SENTINEL -1

// =====================================================================
// SEAMS / STAGING  (the FF-staging mandate — memory feedback_ff_staging_intermediate_seams)
// =====================================================================
// [A] SEAM_FF | 1 = all 11 activation seams are FF/regfile (stage_io_buffer_ff via _sel), NOT BRAM | netlist: FDRE present on seam nets AND ZERO RAMB36/RAM64M/SRL16E/SRL32E + no [Synth 8-6849] (scripts/check_no_bram_on_act_seams.sh)
`define EAGLE4_SEAM_FF            1'b1
// [A] SEAM_FF_LAND_P | GLM lmdown seam FF landing register (the GLM=0 fix) | netlist: FDRE stage on u_lmdown seam output
`define EAGLE4_SEAM_FF_LAND_P     1'b1
// [A] TREE_SEAM_FF | 1 = the g8->softmax-bridge per-candidate LOGIT seam is fed by g8's REGISTERED sync-read (bridge_score_ff, lookahead-addressed) instead of the async cross-SLR dbg_cand_cheap_score debug tap that COLLAPSES TO A CONSTANT in the netlist (the 0x1F tied-zero {-9,-18} tree-score defect). Default 1 = fix ACTIVE (a default-0 fix is inert — the async path is silicon-dead). Only meaningful with DEPTH_N=1. Bit-exact vs the async read (lookahead 1-cycle sync read = same slots/order). | netlist: u_g8/bridge_score_ff_reg FDRE bank (SCORE_W=32) driving u_tree_bridge/cand_logit_f32; ZERO async/comb path from g8 rescored[] to the bridge cand_q; dbg_cand_cheap_score remains async for out3 debug only
`define EAGLE4_TREE_SEAM_FF       1'b1
// [A] FF_ACT_CACHE | body GEMV activation cache in FF (0 = pdi17 silufix2-golden +1 seam) | netlist: act_cache FDRE bank in gemv_stage_array_scaled_slr
`define EAGLE4_FF_ACT_CACHE       1'b0
// [B] SEAM_R1_FULLW | R1 residual seam carries the FULL width (no chunked re-read) | netlist: R1 seam bus width = H*A_WIDTH
`define EAGLE4_SEAM_R1_FULLW      1'b1
// [A] CONNECT_SEAMS | 1 = data-CONNECTED forward (real PDI#5+ datapath); 0 = per-stage ISOLATION (PDI#4 bisect goldens) | netlist: seam mux select tied to 1 => the a_chunk external port arm is pruned
`define EAGLE4_CONNECT_SEAMS      1'b0
// [A] D1_LAT | registered GEMV activation seam (RD_LAT=1 + array WLAT=1) — the COMBREAD stale-by-1 fix | netlist: FDRE on the G6 seam read + WLAT compensation flop; 0 ships the BROKEN combread seam
`define EAGLE4_D1_LAT             1'b1
// [A] PREFETCH_COL_P | GEMV column double-buffer prefetch (bit+cycle INVARIANT vs 0) | netlist: 2x column register bank in the array
`define EAGLE4_PREFETCH_COL_P     1'b0
// [A] R4_BANKSPLIT | R4 seam bank-split (0 since pdi20 FF-staging made it redundant) | netlist: u_seam_r4g/_r4u are PLAIN stage_io_buffer (no bank genblk)
`define EAGLE4_R4_BANKSPLIT       1'b0
// [B] BUFRDREG | register the G1/G1e effective read CHUNK ADDRESS (breaks the ~14.6 ns comb chunk-address route behind the +128/+64 PVT slip) | netlist: FDRE on chunk_addr feeding the seam BRAM/FF ADDR sinks; RDATA_LAT 1->2 on G1/G1e rmsnorm
`define EAGLE4_BUFRDREG           1'b1
// [A] SLR_SAFE | registered 3-SLR crossing seams (paired with D1_LAT=1) | netlist: FDRE pairs on every inter-SLR array bus
`define EAGLE4_SLR_SAFE           1'b1
// [B] S7A_FRONT_SEQ | sequenced front-half stage handoff (S7A) | netlist: front sequencer FSM states present
`define EAGLE4_S7A_FRONT_SEQ      1'b1
// [B] BACK_SEQ | sequenced back-half stage handoff | netlist: back sequencer FSM states present
`define EAGLE4_BACK_SEQ           1'b1

// =====================================================================
// GEMV CORE (2-bit x fp16 mixed precision)
// =====================================================================
// [A] GEMV_SCALED | 1 = gemv_stage_array_scaled (per-group fp16 GPTQ dequant on DSP58, Q3.13) | netlist: DSP48E2/DSP58 scale multipliers present per lane group
`define EAGLE4_GEMV_SCALED        1'b1
// [A] DUAL_PORT | TDP dual-read GEMV (Stage-B', WO-025): 0 = single-port (default, bit+cyc identical), 1 = 2 chunks/cyc via URAM port B (gemv_lane_scaled_stream_dual + RD_PORTS=2, WLAT-fixed) -> per-col cyc halves. Requires body seams to source a_chunk_b. | netlist: URAM288 port-A AND port-B RDB_WR active on weight banks AND URAM COUNT UNCHANGED (8-3971 replication trap) AND seam->lane FDRE depth port-A==port-B
`define EAGLE4_DUAL_PORT          `DUAL_PORT_V
// [A] GEMV_SCALE_FRAC | dequant fixed-point fraction bits (13 => Q3.13, memory #169 population-best) | netlist: 13-bit right-shift constant on the scale path
`define EAGLE4_GEMV_SCALE_FRAC    13
// [A] GEMV_KPAR64 | 1 = K_PAR=64 wide-bank array (ILLEGAL with GEMV_SCALED=1, $error :1485) | netlist: gemv_stage_array_uram_kpar64 instance name
`define EAGLE4_GEMV_KPAR64        1'b0
// [B] GPTQ_QZERO_RAW | raw GPTQ zero-point code (2'b01) | netlist: 2-bit constant on the (w - z) subtractor
`define EAGLE4_GPTQ_QZERO_RAW     2'b01
// [A] BODY_BATCH_B | body batch factor (B=1: batch is the DEPTH line, not the speed line; B>1 costs +233K LUT/unit) | netlist: B copies of the body accumulator set
`define EAGLE4_BODY_BATCH_B       1
// [A] SILU_RANGE_FIX_P | 1 = R4_AW=18 Q6.12 un-clipped (M7 ROOT-CAUSE FIX). 0 truncates 288b->256b => G7p 0x264e6fa4. NEVER build 0. | netlist: R4 seam write bus is 288b wide; G7p cap-CRC must be 0xd8d6d147
`define EAGLE4_SILU_RANGE_FIX_P   1'b1
// [A] TAIL_FP16 | 1 = fp16 full-MLP tail. DECLARED BUILD HOOK ONLY — there is NO fp16 GEMV-lane RTL in tree. DO NOT set 1. | netlist: (none — would be fp16 lane instances)
`define EAGLE4_TAIL_FP16          1'b0

// =====================================================================
// G5 ATTENTION / KV
// =====================================================================
// [A] CTX_WIN | KV context window depth (1024 positions) | netlist: KV URAM/BRAM depth = CTX_WIN per head-stack
`define EAGLE4_CTX_WIN            1024
// [B] KV_V_RAM_STYLE | V cache memory style ("block" = BRAM, Layout B) | netlist: RAMB36 instances under the V cache; NOT URAM
`define EAGLE4_KV_V_RAM_STYLE     "block"
// [B] KV_K_RAM_STYLE | K cache memory style ("ultra" = URAM, Layout B) | netlist: URAM288 instances under the K cache
`define EAGLE4_KV_K_RAM_STYLE     "ultra"
// [A] COMBINE_LOCAL | WO-051 per-KV-stack LOCAL combine (kills the cross-SLR combine cone) | netlist: per-stack combine adders inside each KV genblk (not a single global tree)
`define EAGLE4_COMBINE_LOCAL      1'b1
// [A] REQUANT_HEADMUX | W3-F2 requant placed BEHIND the 24:1 head mux (1 requant unit, not 24) | netlist: exactly ONE act_rescale instance downstream of the head mux
`define EAGLE4_REQUANT_HEADMUX    1'b1
// [B] ROPE_CS_PIPE | RTL WO#8 run-11 lever: pipeline the u_rope rotation cone (run-9 THE worst path: pos_d1->g_pair[12].u_rot/x_out, -1.557 @80MHz, 14.650ns = 2.487 logic/12 lvls + 12.163 ROUTE). cs_q reg per pair splits ROM leg from DSP leg; +1 cycle per rope pass, absorbed by the rope_valid handshake (every caller waits; none count cycles); bit-exact. ONE time-muxed u_rope serves all HEAD_PAR=6 engines, so ONE pipe fix covers all. Default 1 = ACTIVE (a default-0 lever is inert — the DUAL_PORT/HEAD_PAR trap). | netlist: u_rope/g_lut_pipe.g_pair[*].cs_q_reg FDREs (64x32=2048); RAMB under u_rope MUST stay 0 (distributed-ROM guard); SRL on cs_q/start_d2 = 0
`define EAGLE4_ROPE_CS_PIPE       1'b1
// [B] ROPE_POS_REP | RTL WO#8 companion: explicit (* keep *) pos_d1 replicas (8 groups x 8 pairs) for route locality — run 9 shows auto-rep x3 still routed 12.163ns; latency-neutral, bit-exact | netlist: u_rope/g_pos_rep.pos_d1_r_reg[0..7][10:0] FDRE banks (88 FDRE)
`define EAGLE4_ROPE_POS_REP       8
// [B] KV_ADDR_REP | RTL WO#12 (run-12 lever): V-KV BANK-GROUP address replication — run-10 RESPIN50 blocker: per-stack v_rd_pos_q_reg[2][5] -> g_stack[2].u_v BRAM addr, fo=456, 13.602ns PURE ROUTE, KV_VADDR +0.028 @66 (bar >=+0.30; -0.362 @80 run 9). Each of the 6 SHARED V stacks splits into 8 width-slice xpm memories (256b each), each addressed by its own keep'd SAME-CYCLE replica (read twins in the assembly, write twins in mha_sbuf) — fo per net ~1/8, replicas place with their bank group (ROPE_POS_REP idiom). Composes with HEAD_PAR=6: split is per-stack, orthogonal to head-parallel reads (6 stacks x 8 slices = 48 xpm, storage-neutral). Latency-neutral, bit-exact; K stays per-stack. REQUIRES ROPE_POS_REP (elab guard). | netlist: u_attn/v_rd_pos_q_reg[0..47][10:0] FDRE (528) + u_attn_full/v_wr_pos_q_reg[0..47][10:0] FDRE (528); u_kv/g_stack[*].u_v/g_grp.g_slice[0..7].u_mem xpm instances (8 per stack, RAMB36 only)
`define EAGLE4_KV_ADDR_REP        8
// ---------------------------------------------------------------------
// F3 KV-RESIDENCY MASK — NOT A TOGGLE, DELIBERATELY (2026-07-24).
//   reports/results/ATTN_F3_KV_RESIDENCY_MASK.md
// The attention datapath had NO residency concept: pos_req swept
// [0, ctx_len) unconditionally and score_max_tracker admitted every score,
// so a never-written K row scored 0 -- a COMPETITIVE value -- and took
// running_max on the ~50 % of heads whose real Q·K is negative. The fix
// (a per-slot written bitmap in attention_head_assembly_perbank_sbuf that
// forces score_norm = -INF and drops exp_valid_in for non-resident slots)
// is UNCONDITIONAL in the netlist: it is a correctness repair, not a lever,
// and it is provably INERT whenever the window contains no holes
// (depth-0 seal byte-identical: frontier 8/8, G5 0xc59befef, G7p
// 0xd8d6d147, G8RESC 0x002c953c). Making it a strap would (a) add a 72nd
// top parameter and break the "param-dump 71 IDENTICAL" no-regress proof
// for zero benefit, and (b) create a configuration in which the shipped
// image can silently regress to the defect -- exactly the HEAD_PAR class.
// A/B is done with the sim-only RED define below (section [E]).
// Netlist signature: a CTX_WIN-bit `kv_resident` FDRE bank + a CTX_WIN:1
// read mux inside u_attn_full/g_combine_perbank/u_attn; ZERO RAMB/URAM.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// F3-CLR KV-RESIDENCY SESSION CLEAR — ALSO NOT A TOGGLE, SAME REASONING.
//   reports/results/KV_RESIDENCY_CLEAR_FIX.md
// `kv_resident` above was SET-ONLY: the ONLY clear was rst_n, which on this
// board is a PHYSICAL COLD-AC. Residency is a SESSION property and there
// was no way to END a session. SGLang decodes MANY requests; a new request
// reuses slots 0..N for DIFFERENT context but does not rewrite them all
// (the FSM self-appends ONE row per forward), so the previous request's
// slots stay marked resident over the previous request's K/V -> the drafter
// attends stale cross-request context that the F3 mask certifies as valid.
// Signature: depth-0 golden 7/8 on a freshly power-cycled card, 0/8 after
// ANY decode run, restored only by a cold-AC; accept pinned 1.02-1.05
// through every host-side fix.
// The repair is reg 0x154 bit[0] -> a 1-cycle `kv_res_clear` pulse into
// every engine's bitmap. UNCONDITIONAL in the netlist for exactly the F3
// reasons: it is a correctness repair, and a strap would create a shipping
// configuration that silently regresses (the HEAD_PAR class). It is INERT
// unless the host writes 0x154, so every existing gate stays byte-identical.
// Netlist signature: control_regs kv_res_clear FDRE + one res_clr_q FDRE
// bank per engine (KV_RES_CLR_REP bits x HEAD_PAR) feeding the kv_resident
// FDRE bank's synchronous clear; 0x154 present in the write+read decode.
// ---------------------------------------------------------------------
// [B] KV_RES_CLR_REP | F3-CLR clear-pulse fan-out replication INSIDE each engine's kv_resident bitmap. The clear must reach CTX_WIN=1024 FFs per engine (x HEAD_PAR=6 = 6,144 loads) — the ROPE_POS_REP / KV_ADDR_REP route class. R = number of keep'd SAME-EDGE twins, twin r driving bits [r*ceil(1024/R), ...); ONE source value => the clear stays ATOMIC over the bitmap, only the route splits. 1 = single registered pulse (default; raise only if the OOC/route report shows the clear net as a critical path). Latency-neutral and functionally identical at every R. | netlist: u_attn_full/.../u_attn/res_clr_q_reg[0..R-1] FDRE (R per engine) + kv_resident FDRE count UNCHANGED at HEAD_PAR*1024
`define EAGLE4_KV_RES_CLR_REP     1

// =====================================================================
// TREE DECODE
// =====================================================================
// [A] TREE_P | vocab-scan logits/cycle for top-K_F=8 tree decode (8 = pkpar wide; frontier INVARIANT in P) | netlist: TREE_P parallel compare lanes in tree_tile_stream_pkpar
`define EAGLE4_TREE_P             8
// [A] TREE_PP | 2-entry FF ping-pong tiler (tile t+1 GEMV under tile t stream); stream content BIT-IDENTICAL | netlist: tree_tile_stream_pkpar_pp instance name + 2-deep FF tile buffer
`define EAGLE4_TREE_PP            1'b0
// [A] STAGE_INJECT | golden-ingest infra (AXI 0x10C..0x11C). =0 generate-prunes the whole inject path (zero FF/LUT delta) | netlist: stage_inject instance present/absent
`define EAGLE4_STAGE_INJECT       1'b0
// [B] DEPTH_N | multi-level tree decode (the headline goal). 0 = single-pass drafter, BIT-IDENTICAL | netlist: depth loop states in top_fsm + the journal-replay KV path
`define EAGLE4_DEPTH_N            `DEPTH_N_V
// [B] EMBG_EN | GAP-4 RUNG C child-embedding gather (m_axi_emb master live). Only meaningful with DEPTH_N=1 | netlist: emb_gather instance + m_axi_emb NOT tied off (S08_AXI in the BD)
`define EAGLE4_EMBG_EN            `EMBG_EN_V
// [B] EMB_NCH | HBM read channels for the child-embedding gather. 1 (DEFAULT) = the shipping single-channel path (m_axi_emb -> S08_AXI -> HBM5_PORT1); 2 = intra-row stripe over a SECOND port of the SAME HBM PC (m_axi_embb -> S09_AXI -> HBM5_PORT0, bd_patch/S09_embb.patch.tcl) reading the SAME flat image at a different offset -> ZERO duplicate storage, ZERO host relayout. Only meaningful with DEPTH_N=1 && EMBG_EN=1. MEASURED end-to-end gain ~0 (sink-bound: input_emb_buf takes 1 chunk/cyc; 199->197 cyc/row, <=0.03% of a depth-N level) -> shipped OFF until the sink widens or K_F-parallel gather lands (reports/results/EMB_GATHER_MULTICHANNEL.md). | netlist: emb_gather g_nchn generate scope (NOT g_nch1) + m_axi_embb_arvalid NOT tied to 0 + 2 AR engines
`define EAGLE4_EMB_NCH            `EMB_NCH_V
// [B] FORREASON | F1 depth>=1 re-key hidden producer: G9 widened to N=2H=6144 (48 col-tiles) + a = G9[H:2H] + (block_input + G6_oproj) + the `a` bank. 0 = depth-0 BIT-IDENTICAL (G9 stays 24 tiles, zero new cells). Only meaningful with DEPTH_N=1. | netlist: for_reasoning_producer instance + g9hi_buf FDRE bank + G9 tile counter terminal 48 (NOT 24)
`define EAGLE4_FORREASON          `FORREASON_V
// [B] FR_KF_BANKS | physical `a`-bank count: 1 = F1/depth-1 (spec_utils.py:468 repeat_interleave — every child shares the root a_0); 16 = F4/depth>=2 per-parent re-gather = 8 survivor slots x 2 level-parity epochs (double-buffer, chain.cpp:491-495). Strappable via FR_KF_BANKS_V so depth-0 default (=1) stays byte-identical. | netlist: a_bank FDRE count = FR_KF_BANKS*192*256
`define EAGLE4_FR_KF_BANKS        `FR_KF_BANKS_V
// [B] FR_ABANK_BRAM | a-bank storage: 0 = FF regfile (default, byte-identical), 1 = SDP BRAM (CONGFIX-1: SLICE 97.94%->~82-84%, +64 RAMB36). Legal ONLY at FR_KF_BANKS>8 (FR_DBUF epoch separation). | netlist: a_bank RAMB36E5 count == 64 AND a_bank FDRE ~0 AND `LUT as Memory`==0 (silent LUTRAM downgrade = the failure mode)
`define EAGLE4_FR_ABANK_BRAM      `FR_ABANK_BRAM_V
// [A] HEAD_PAR | G5 head-parallelism: 1 = SERIAL (default, byte-identical to the ps_tree_fsm gate-2-4-proven config), 6 = 6-way lockstep engines (ctx-linear attn 552/ctx->92/ctx). THE 0x18 BURN param (0 pkg refs shipped SERIAL). Composes with depth-N KV lane-chain (per-engine F3 mirror). | netlist: HEAD_PAR (=6) attention_head_assembly_perbank_sbuf instances under u_attn_full/g_headpar/g_eng[*] (6-stack census) AND residency FDRE ~= HEAD_PAR*1024 (per-engine F3 mirror, F3-3) AND .kv_wr_v/.kv_wr_slot DRIVEN on every g_eng instance (F3-1 undriven-port trap)
`define EAGLE4_HEAD_PAR           `HEAD_PAR_V
// [B] ENORM_D0 | apply the depth-N 27/8 e_norm requant (SUMSQ_SHR=27,SCALE_SHR=8) at LEVEL-0 too — real token embeds rms~0.02 saturate the 31/10 scale_q12 grid at the root (silicon d2_call0012 1/8); 27/8 has +4b headroom (the shipped-root accept lever). 0 = legacy level>=1-only (31/10 at the root). Default 1 = lever ACTIVE in production. Only meaningful with DEPTH_N=1 (the 27/8 instance is generate-elided at DEPTH_N=0 -> depth-0 byte-identical either value). | netlist: use_enorm_d mux under g_enorm_depth_requant routes u_rmsnorm_g1_e_d to the G1e seam at level_i_w==0 (=1), not only level>=1 (=0)
`define EAGLE4_ENORM_D0           `ENORM_D0_V
// [B] PRECISION_SAFE | atomic candidate-boundary precision mode: frac16 embedding input; e-norm 31/10+EPS168 -> frac11; h-norm -> frac11; G4 ACT_FRAC11; R3 frac7; preMLP 29/10. 0 = current production arithmetic byte-identical. | netlist: u_rmsnorm_g1_e_d EPS_BUCKET=168 and G6-to-R3 shift=3
`define EAGLE4_PRECISION_SAFE     `PRECISION_SAFE_V

// =====================================================================
// G8 EXACT RESCORE (the SPEED wall — 49% of the forward)
// =====================================================================
// [A] G8_SPLICE | splice the cheap-path stream into the G8 gather (tree_cheap_serializer snoop) | netlist: tree_cheap_serializer instance + the snoop FIFO
`define EAGLE4_G8_SPLICE          1'b0
// [A] G8_K_C | exact-rescore candidate count (512; live floor is 64 per project_kc_shortlist_live_floor_64) | netlist: gather row counter terminal count
`define EAGLE4_G8_K_C             512
// [A] G8_MAX_BURST | max AXI beats per gather burst | netlist: arlen saturation constant
`define EAGLE4_G8_MAX_BURST       256
// [A] G8_PIPELINED | decoupled gather front-end (required by WIDE/NCH/ROWPP) | netlist: g8_stripe_frontend instance present
`define EAGLE4_G8_PIPELINED       1'b1
// [A] G8_OUTSTANDING | outstanding AR depth PER CHANNEL. 4 = the TB-certified + speedwave-0xE4B71111-silicon value (safe WITH ROWPP row-tag pairing; the pdi21 2/8 deep-AR drift was the ARRIVAL-ORDER pairing, killed by the owner FIFO). Was 2 while the config diverged from the cert TB (G8_NCH_STRIPE.md §2.1). | netlist: AR credit counter width / owner FIFO depth
`define EAGLE4_G8_OUTSTANDING     4
// [A] G8_ROWPP | decoupled fetch/feed + owner FIFO + ROW_OVERLAP (~196 cyc/row at every tested latency; REQUIRED by G8_NCH>1). 1 = the TB-certified elaboration (every rungc/fullchain gate runs ROWPP_P=1 — G8_NCH_STRIPE.md §2.1 closed the sim-vs-build divergence). | netlist: ROW_OVERLAP generate arm in g8_stripe_frontend
`define EAGLE4_G8_ROWPP           1'b1
// [B] G8_PREFETCH | master enable for the P2 prefetch hooks (repro only) | netlist: g8_prefetch_thresh instance present
`define EAGLE4_G8_PREFETCH        1'b0
// [A] G8_DATA_W | m_axi_g8 data width (256b NMU-native; 512b was the pdi18 no-ARREADY fail) | netlist: m_axi_g8_rdata[255:0]; BD S05_AXI DATA_WIDTH must MATCH
`define EAGLE4_G8_DATA_W          256
// [B] G8_OPT1C | balanced-tree log2(K_C) comparator selector (exact, zero-drop) | netlist: log2(512)=9-level comparator tree
`define EAGLE4_G8_OPT1C           1'b1
// [B] G8_OPT1D | hierarchical-min selector variant (repro only) | netlist: 2-level hierarchical min blocks
`define EAGLE4_G8_OPT1D           1'b0
// [B] G8_MIN_NG | hierarchical-min group count (with OPT1D) | netlist: 16 min-group scopes
`define EAGLE4_G8_MIN_NG          16
// [A] G8_OPT1F | PIPELINED registered 2-level min | netlist: registered cmin stage between the two min levels
`define EAGLE4_G8_OPT1F           1'b0
// [A] G8_SELPIPE | registered-cmin selector retiming — THE 100 MHz enabler (full-top OOC -2.935 -> +0.087 MET) | netlist: sel_pipe FDRE stage in g8_rescore_p
`define EAGLE4_G8_SELPIPE         1'b1
// [A] G8_OPT1G | 72 MHz selector variant (GRAND default ON) | netlist: OPT1G generate arm in the selector
`define EAGLE4_G8_OPT1G           1'b1
// [A] G8_WIDE | 256b wide gather + P=16 wide dot (frontier-exact; re-associated adder tree) | netlist: g8_dot_fp16_wide* instance + 256b rdata consumption
`define EAGLE4_G8_WIDE            1'b1
// [A] G8_WIDE_P | wide-dot lane count | netlist: 16 fp16 multipliers per wide dot
`define EAGLE4_G8_WIDE_P          16
// [A] G8_MROUND | scalar-dot round-split (fp16_mul normalize/round pipelined) | netlist: extra FDRE stage in fp16_mul_b
`define EAGLE4_G8_MROUND          1'b1
// [A] G8_SCALAR_II1 | scalar dot FADD_PIPE=0 + feed-every-cycle = 2x consume, BIT-EXACT | netlist: g8_dot_fp16 FADD_PIPE=0 arm; gate G8RESC must stay 0x002c953c
`define EAGLE4_G8_SCALAR_II1      1'b1
// [A] G8_WIDE_IL | II=1 interleaved-accumulator wide dot (898->459 gather cyc, 1.956x). HARD CONTRACT: the wide nrm read MUST stay COMBINATIONAL. | netlist: G8_WIDE_M accumulator copies in g8_dot_fp16_wide_il
`define EAGLE4_G8_WIDE_IL         1'b1
// [A] G8_WIDE_M | II=1 interleave depth (pow2, >=2) | netlist: 4 interleaved accumulators
`define EAGLE4_G8_WIDE_M          4
// [A] G8_LANE_FIX | AXI4 A3.4.3 narrow-burst LANE-SELECT by element parity. =0 = the even-only dot that produced the wrong-8 on silicon. NEVER ship 0. | netlist: parity mux on rdata[31:16] vs [15:0]
`define EAGLE4_G8_LANE_FIX        1'b1
// [A] G8_NCH | N-channel STRIPED HBM gather (2 = ch0 m_axi_g8/S05->HBM3_PORT0 + ch1 m_axi_g8b/S07->HBM3_PORT1, gather-order round-robin over the SAME flat lm-head image). Requires PIPELINED+WIDE+WIDE_IL+ROWPP. 2 SHIPPED 2026-07-26: tree->g8 101,921->51,233 cyc (1.989x, byte-identical G8RESC 0x002c953c; gates G8_NCH_STRIPE.md; S07 BD patch = w2_1f STEP2.7, silicon-proven J5B U2). | netlist: m_axi_g8b ports NOT tied off + stripe generate scope g_stripe.u_stripe + 2x g8_dot_fp16_wide_il in stripe
`define EAGLE4_G8_NCH             2
// [A] G8_CONSUME_HARDEN | explicit owned FDRE + matched skid on the scalar nrm read (removes ram_style reliance). Default follows `G8_HARDEN so the xsim gate can flip it. | netlist: owned FDRE on nrm_buf read + beat-index check teeth
`ifdef G8_HARDEN
  `define EAGLE4_G8_CONSUME_HARDEN 1'b1
`else
  `define EAGLE4_G8_CONSUME_HARDEN 1'b0
`endif

// =====================================================================
// 0x1125 ACCFIX — accept-path fixes (kvres_accfix fork, 2026-07-29)
// ALL four are ELABORATION-present datapaths with a RUNTIME enable in the
// new ACCFIX register (0x15C, reset 0x0 = every fix OFF = byte-identical to
// 0x1124 by construction — one elaboration serves baseline AND fix arms, so
// the WSNAP-restored accept A/Bs need no re-elab). Setting the elab toggle
// to 0 generate-prunes that fix's cells entirely (netlist rollback lever).
// =====================================================================
// [B] ACCFIX_SEAM | H1 CHILD_SEAM_FIX: the depth>=1 re-key hidden `a` loses 4 LSBs/level through the R3 a-arm floor (`>>>F2_SH_FR`) then re-left-shift in for_reasoning_producer (level0: 7-bit floor `>>>F2_SH_HFC` on h_fc). Capture the shift remainder at the a-arm read (fr_rem_buf, 192x16x5b FF) and re-add it in the producer's wide pre-sat sum -> the carried residual chain becomes EXACT per level (level0 lands round-half h_fc@frac9). Runtime enable = ACCFIX reg bit0. Only meaningful with DEPTH_N=1 && FORREASON=1. | netlist: fr_rem_buf FDRE bank + rem term adder in for_reasoning_producer (g_seamfix)
`define EAGLE4_ACCFIX_SEAM_P      1'b1
// [B] ACCFIX_B2P | B2' round-half on the two attention `>>>12` floors (weight = exp*inv, and per-position attn*V product). Owns ~98% of ctx error growth (relL2 .0833@512 -> .0075, ANGLE_B_NUMERIC_AT_CTX.md); ctx=1 BYTE-IDENTICAL even when ENABLED ((4096*V+2048)>>12==V). Runtime enable = ACCFIX reg bit1. | netlist: +2048 constant adders in attn_v_combine g_mac + perbank_sbuf combine_attn cone
`define EAGLE4_ACCFIX_B2P_P        1'b1
// [B] ACCFIX_KC | BRIDGE_KC: plumb runtime kc_cfg (0x150) into tree_softmax_bridge (both sweeps + rescored[] read bound — kills the stale-slot-ingest hazard at kc<512 depth-N, PHASE3 §b) AND skip the S_MAX pass by seeding max_q from the G8 frontier top-1 (bit-identical: same fp32 through the same decode). Runtime enable = ACCFIX reg bit2 (OFF = 2x512 sweeps, byte-identical). | netlist: kc port on u_tree_bridge + S_IDLE->S_SUM edge
`define EAGLE4_ACCFIX_KC_P        1'b1
// [B] ACCFIX_KVONLY | P3 KVONLY launch mode (PHASE6 fast-priming): while ACCFIX reg bit3 is set, every launch runs S_G1->S_G4->S_G5(KV append)->S_DONE, skipping G6..TREE/G8 (7.6x cheaper prefix-prime rows; KV row produced by the SAME datapath = bit-identical to a full launch's). No tree/OUT-DMA is produced (host ignores output on prime launches already). | netlist: kvonly edge terms in top_fsm S_IDLE/S_G5_ROPE
`define EAGLE4_ACCFIX_KVONLY_P      1'b1
// [B] OBS_LVL | per-level tree observability: 512-entry BRAM ring capturing the per-level candidate feed (level, branch/parent slot, token, prob fp32) at the tree_feed_seq->tree_level_fsm seam, host-readable via the 0x15C-window dbg read regs (0x160 idx autoinc / 0x164 data). Read-only tap — zero effect on the datapath. | netlist: obs_cand_ram BRAM + 0x160/0x164 decode
`define EAGLE4_OBS_LVL_P          1'b1

// =====================================================================
// [D] DERIVED — not independently settable
// =====================================================================
// D1_RDLAT = D1_LAT ; D1_WLAT = D1_LAT (kept derived in eagle4_drafter.sv so a
// single D1_LAT change moves both; the packaging generator resolves them to the
// SAME value so the BD can never see a half-applied D1 fix).
`define EAGLE4_D1_RDLAT           `EAGLE4_D1_LAT
`define EAGLE4_D1_WLAT            `EAGLE4_D1_LAT

// =====================================================================
// [C] UNWIRED — declared module-local, ZERO `.NAME(` bindings anywhere.
//     These are NOT threaded blind. Each needs a threading diff before it can
//     ever be strapped. Listed so the gap is VISIBLE, not silent.
// =====================================================================
//   DUAL_PORT     — WIRED 2026-07-26 (Stage-B' WO-025): promoted to class [A]
//                 (`DUAL_PORT_V` above). eagle4_drafter parameter DUAL_PORT ->
//                 u_array .DUAL_PORT + a_chunk_b fed from every body seam's
//                 port-B read at chunk_idx_b_out. No longer in this list.
//   MUL_PIPE      src_full/g8/g8_dot_fp16.sv:97                    default 1'b1
//                 register fp16_mul OUTPUT (prod16). UNWIRED — g8_rescore_p does not bind it,
//                 so the module default is the only reachable value.
//   WATCHDOG_EN   src_full/top/fsm_probe.sv:31                     default 1'b0
//                 FSM stall watchdog. UNWIRED — no binding; watchdog can never be enabled.
//   ROPE_SLR_REG  src_full/layers/rope_layer.sv:53                 default 1'b0
//                 registered RoPE SLR crossing. UNWIRED — no binding.
//   HEAD_PAR      NOW WIRED (2026-07-26, Stage-A): ported from speedwave_v2_g5mem
//                 as a live [A] strap (EAGLE4_HEAD_PAR above, default 1). Class
//                 moved [C]->[A]; the burn it names is now closed for this fork.
//
// =====================================================================
// [E] DEAD / SIM-ONLY DEFINES — pkg tcl CLEARS verilog_define, so NONE of these
//     can ever ship in a PDI. They are xsim RED-arm teeth only:
//     G8_ROWPP_RED, FEEDSEQ_RED_STALE, FEEDSEQ_RED_OFFBYONE, TREE_RED_PER_PARENT,
//     TREE_RED_TRANSPOSE, TREE_RED_SUM, TREE_RED_RTL_PARENT_SLOT, TREE_RED_POS,
//     TREE_RED_INDEP, TREE_P8_RED, TREE_ARB_RED, STRIPE_RED, KVREPL_RED,
//     KVLANE_RED_STRIDE, KVRES_RED_NOMASK, SIM_CTX_FORCE=<n>,
//     G8_SELPIPE_RED, EMB_RED_NOSCALE, BRIDGE_RED_NONORM,
//     G8_DISABLE_4KB_SPLIT, SER_OVF_WARN_ONLY, GEMV_LANE_DEBUG, G8_DBG,
//     BUFSPLIT_* (declaration-site macros), SYNTHESIS.
//     SG_LANES — RETIRED 2026-07-26 (LANES-256): the define is DELETED from
//     stage_geometry.svh; geometry lanes now ride the LANES PARAMETER as a
//     function argument (packaging-visible via LANES_V above), so the
//     define-cleared-at-packaging split-brain of study §3.1 cannot occur.
//     G8_HARDEN is the ONE define with a parameter shadow (G8_CONSUME_HARDEN above)
//     — that is the pattern to copy if a define ever needs to ship.
// =====================================================================

`endif // EAGLE4_CONFIG_SVH
