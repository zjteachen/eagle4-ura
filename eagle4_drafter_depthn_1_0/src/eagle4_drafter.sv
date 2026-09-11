// =====================================================================
// eagle4_drafter.sv
// EAGLE4 drafter — top-level kernel, production integration.
// =====================================================================
// RECLOCK100 fork (2026-07-10) — build-id 0xE4B71110 (PROBE_REV 8'h10).
//   = tree_batch ship base (0xE4B7110F lineage) with:
//   (1) BODY_BATCH_B=1 (speed line stays depth-0; batch is the separate line)
//   (2) G8_SELPIPE=1 (sel_pipe registered-cmin OPT1F selector retiming — the
//       100 MHz in-context fix: full-top OOC @10ns -2.935 -> +0.087 MET)
//   (3) kvrepl_port kv_rd_pos per-stack replication merge (retires the
//       recurring KV_VADDR body-WNS phys_opt rescue cone)
//   (4) FASTIN boot-DMA route into input_emb_buf/h_fc_buf (fastin fork merge
//       2026-07-10: 0x54 bits[2]/[3] emb/hfc target-selects, loader target
//       latch, 3-way dma_we split — kills the 2.77 ms/call 6,144-poke
//       emb+h_fc load; AXI-Lite trio path kept as fallback)
//   (5) packaged at FREQ_HZ=100 MHz (pkg/pkg_reclock100.tcl)
// =====================================================================
//
// Wires together:
//   - control_regs       (AXI4-Lite slave, libeagle4_host_v56_direct-compatible)
//   - top_fsm            (11-stage drafter orchestrator)
//   - gemv_stage_array_uram  (shared 128-lane PE array, all 6 GEMV stages)
//
// Stages that are NOT yet wired to compute units (G1 fc, G5 RoPE+KV,
// G7_SILU, G11, TREE) have done signals exposed as top-level ports for
// external mocking. They will bind to their respective layer modules in
// subsequent ticks.
//
// AXI4-Lite slave port: standard HLS-compatible 7-bit addr / 32-bit data.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

// =====================================================================
// SINGLE-SOURCE CONFIG (user mandate 2026-07-22). EVERY parameter default
// below is `EAGLE4_<NAME> from src_full/top/eagle4_config.svh -- that file is
// the ONE place a toggle value changes, and the packaging ipx parameter list
// is GENERATED from it (scripts/gen_pkg_param_list.sh) so coverage cannot drift.
// DO NOT write a literal default in this block.
// =====================================================================
`include "eagle4_config.svh"

// DEPTH-N strap (tree FSM stage 2). Default 0 = today's single-pass drafter,
// bit-identical. Strap with `-d DEPTH_N_V=1` to BOTH xvlog AND xelab.
// (moved to eagle4_config.svh -- DEPTH_N_V default lives there now)

// GAP-4 RUNG C strap (child-embedding gather). Default 0 = no gather graft;
// the m_axi_emb master is tied off and input_emb_buf keeps its host-only
// writer, so the design is BIT-IDENTICAL to the pre-graft drafter. Strap with
// `-d EMBG_EN_V=1` to BOTH xvlog AND xelab (mirrors DEPTH_N_V). EMBG_EN=1 is
// only meaningful with DEPTH_N=1.
// (moved to eagle4_config.svh -- EMBG_EN_V default lives there now)

module eagle4_drafter #(
    parameter int LANES      = `EAGLE4_LANES,
    parameter int K_MAX      = `EAGLE4_K_MAX,   // largest K across all GEMVs (G9)
    parameter int K_PAR      = `EAGLE4_K_PAR,
    parameter int A_WIDTH    = `EAGLE4_A_WIDTH,
    parameter int TREE_V_MAX = `EAGLE4_TREE_V_MAX, // tree_decode vocabulary scan length (Llama-3.2-3B vocab)
    // M9 boot-DMA m_axi read master geometry
    parameter int BOOT_AXI_ADDR_W = `EAGLE4_BOOT_AXI_ADDR_W,
    parameter int BOOT_AXI_DATA_W = `EAGLE4_BOOT_AXI_DATA_W,
    parameter int BOOT_AXI_ID_W   = `EAGLE4_BOOT_AXI_ID_W,
    parameter int BOOT_BURST_LEN  = `EAGLE4_BOOT_BURST_LEN,
    // OUTPUT-DMA ABI: 1 = v24 SGLang host ABI (ids/scores/parents, 8 live +
    //   8 pad, TOP_K stride, parents=root sentinel) for M8 SGLang E2E;
    //   0 = M7 simple layout (idx+score @out0, idx mirror @out1).
    parameter bit OUT_ABI_V24     = `EAGLE4_OUT_ABI_V24,
    parameter int OUT_TOP_K       = `EAGLE4_OUT_TOP_K,
    parameter int OUT_PARENT_SENTINEL = `EAGLE4_OUT_PARENT_SENTINEL,
    // =====================================================================
    // PDI#5-ASSEMBLY: data-connect the 6 broken seams (R1/R2/R4/R5/R6/R8) +
    // the 2 Llama residuals (R3/R7). ONE RTL, TWO MODES (DATAFLOW_AUDIT §6.4):
    //   CONNECT_SEAMS = 0 : per-stage ISOLATION (PDI#4). Each GEMV stage reads
    //     the external `a_chunk` port; silu/g11 inputs tied 0. This is the
    //     mode the per-stage bisect goldens rely on (each golden is an
    //     INDEPENDENT seed-42 draw, so the connected forward cannot match them
    //     downstream — only isolation can). DEFAULT — preserves the PDI#4 walk.
    //   CONNECT_SEAMS = 1 : data-CONNECTED forward. Each seam's stage_io_buffer
    //     read replaces exactly one arm of the per-stage activation mux; the
    //     two residual_add adders re-inject x / h. This is the real PDI#5
    //     single-layer forward (embed -> G1 -> ... -> G12 -> tree).
    // The mux is per-seam (the L8c template at the G6 arm), so both modes
    // coexist in one synthesizable RTL. The TB strap (CONNECT_SEAMS param)
    // selects the mode at elaboration.
    // =====================================================================
    parameter bit CONNECT_SEAMS   = `EAGLE4_CONNECT_SEAMS,
    // =====================================================================
    // PDI#5-CLOSING: fp16-MLP-tail accuracy lever (#184), PARAMETERIZED OPTION.
    //   TAIL_FP16 = 0 (DEFAULT) : the free INT16 tail. With the #190 corrected
    //     zero-clip per-seam fracs (attn_in=12, r3_h1=6, ...) + #185 SmoothQuant
    //     pow2 per-channel scale this delivers the validated set-8/8 / real-N
    //     76.45-set / 62.65-ordered / 92.55-rank0 config at ZERO extra fabric.
    //     The 2-bit weights stay in URAM; the GEMV core is unchanged. This is
    //     the build target for the closing PDI#5.
    //   TAIL_FP16 = 1 : fp16-full-MLP-tail (gate/up/silu/down in fp16) for
    //     STRICT ordered-8/8 (synthetic-14 14/14, real-N +6.9pp ordered). Cost
    //     = +280K..566K LUT + ~640..1664 DSP + ~0.85-0.9x fmax (fp16_tail §3).
    //     NOTE: there is NO fp16 GEMV-lane RTL in the tree yet — TAIL_FP16=1 is
    //     a DECLARED BUILD HOOK pending the user's accuracy/timing decision; the
    //     fp16 tail datapath is an RTL build item (not wired here). DO NOT set =1
    //     until that datapath is added — the default 0 path is the only
    //     functional tail today. Left as a parameter so the user can flip it per
    //     the pending decision without re-plumbing the top.
    // =====================================================================
    parameter bit TAIL_FP16       = `EAGLE4_TAIL_FP16,
    // =====================================================================
    // PDI#5-COMPLETE: the two DEFERRED speed pieces, wired as elaboration-time
    // generate swaps (pin-compatible, bit-exact-invariant per their forks).
    //
    //   TREE_P (#191 tree_decode_fast): vocab-scan logits/cycle for the top-K_F=8
    //     tree-decode. TREE_P=1 -> serial tree_tile_stream + tree_decode (the
    //     closing path, 160 us @200MHz P=4-equiv scan). TREE_P=8 -> the P-wide
    //     tree_tile_stream_pkpar + tree_merge_pipe (80 us, 2.00x). xsim-proven
    //     8/8 BIT-IDENTICAL frontier at P=8 on real g12 + 100 random vectors
    //     (TREE_DECODE_FAST_RESULT §2). Frontier is INVARIANT in P. DEFAULT 8.
    //
    //   GEMV_KPAR64 (#188 kpar64_allshapes): when 1, the shared GEMV array is
    //     gemv_stage_array_uram_kpar64 (K_PAR=64 wide-bank, N_SG=4). 4.00x the
    //     per-column chunk rate vs K_PAR=16, URAM-NEUTRAL (1,536 @128 lanes),
    //     BIT-IDENTICAL across all 5 shape classes (KPAR64_ALLSHAPES_RESULT §1).
    //     GEMV_KPAR64=0 -> the closing K_PAR=16 array. DEFAULT 1.
    //     NOTE: the closing top is a PILOT single-chunk (NCHUNKS=1, K_PAR-wide
    //     seam broadcast) forward; setting GEMV_KPAR64=1 widens the array AND the
    //     activation/seam chunk to K_PAR=64. The bit-exact frontier is proven by
    //     the standalone all-shapes TB (K_PAR-invariant), NOT re-derived through
    //     the pilot single-chunk top (that requires the full-H streaming rewrite,
    //     out of scope — see PDI5_COMPLETE_RESULT §scope).
    // =====================================================================
    parameter int  TREE_P         = `EAGLE4_TREE_P,
    // EXP_G12TREE_PINGPONG: TREE_PP=1 swaps the tree tiler for the 2-entry FF
    //   ping-pong variant (tree_tile_stream_pp / tree_tile_stream_pkpar_pp):
    //   tile t+1's G12 GEMV runs UNDER tile t's stream into the tree, removing
    //   the strict GEMV->stream serialization (per-tile max(W,stream) instead
    //   of W+stream). Stream content/order BIT-IDENTICAL (A/B gate:
    //   reports/results/EXP_G12TREE_PINGPONG_RESULT.md). DEFAULT 0 = baseline
    //   byte-identical elaboration.
    parameter bit  TREE_PP        = `EAGLE4_TREE_PP,
    parameter bit  GEMV_KPAR64    = `EAGLE4_GEMV_KPAR64,  // DEFAULT 0: GEMV_KPAR64=1 + GEMV_SCALED=1 (both default) is an illegal pair ($error :1485). 0 = the scaled FITTING config (the built/correctness default). Flip to 1 only for the wide speed lane WITH GEMV_SCALED=0. (hunt W2, 2026-06-17)
    // =====================================================================
    // B2 (pdi6 — SCALED GEMV): swap the unscaled integer array
    //   (gemv_stage_array_uram, Σa·(w−z)) for gemv_stage_array_scaled, which
    //   applies the per-group fp16 GPTQ dequant scale on a DSP58 (Q3.13,
    //   population-best per memory #169 — NOT Q1.15). This is the DOMINANT fix
    //   the xsim_e2e closure identified: the unscaled-INT contract is 0/8 vs
    //   eagle4_ref BY CONSTRUCTION; the per-group scale puts the GEMV output in
    //   the same numeric space as SGLang.
    //
    //   GEMV_SCALED=0 -> the unscaled K_PAR=16 array (the historical g6.hex /
    //     bisect-anchor contract; default for back-compat with the per-stage
    //     goldens). GEMV_SCALED=1 -> the scaled array. MUTUALLY EXCLUSIVE with
    //     GEMV_KPAR64 (the wide-bank array has no scaled lane) — an elaboration
    //     assert below fails loudly if both are 1.
    //
    //   The scaled array adds a per-lane scale_bram (BRAM) loaded by the
    //   weight_dma_loader's sc_* port (#157 scale-boot-port), and emits
    //   result_q_per_lane (RES_WIDTH=48, SCALE_FRAC=13 frac bits). The top
    //   requantizes result_q >> SCALE_FRAC into the 32b/lane array_acc bus the
    //   seams + tree consume — i.e. the scaled-dequant magnitude in INT space.
    // =====================================================================
    parameter bit  GEMV_SCALED    = `EAGLE4_GEMV_SCALED,
    parameter int  GEMV_SCALE_FRAC = `EAGLE4_GEMV_SCALE_FRAC,      // Q3.13 (memory #169 population-best)
    // =====================================================================
    // GPTQ ZERO-POINT (the model constant) — QZ-FIX 2026-05-27.
    // gptq2_dequant16 computes w_signed = raw_w - (qz_raw + 1). The real
    // EAGLE4 gptq_model-2bit-128g safetensors store a UNIFORM raw qzero = 1
    // for EVERY GEMV tensor (q/k/v/o_proj, gate/up/down_proj — verified all
    // 7 tensors, 100% of groups), so the correct effective zero point is 2,
    // i.e. qz_raw = 1. The external `qz_raw` port defaulted to 0 in the S8
    // harness (and any host that leaves it unset) -> w_signed was off by +1
    // on EVERY weight -> a uniform Σ_k a[k] bias per output column. That bias
    // is benign-ish for G4 (attention re-normalizes) but FATAL for G6 o_proj:
    // the bias drove the dequant to ~-1e6 for ALL 3072 columns -> uniform
    // INT16-min (-32768) at the seam, poisoning r3_h1/pre_mlp -> S8 0/8.
    // The zero point is a MODEL CONSTANT, not a per-token input, so we pin it
    // here (silicon-correct regardless of how the host drives the legacy port).
    // =====================================================================
    parameter logic [1:0] GPTQ_QZERO_RAW = `EAGLE4_GPTQ_QZERO_RAW,
    // =====================================================================
    // S3/S7a (SEAM_REWRITE §S3): the R1 (G1 RMSNorm -> G4 QKV) seam width.
    // SEAM_R1_FULLW=1 (DEFAULT) = FULL-WIDTH streaming seam (192 chunks captured
    // into a full-H stage_io_buffer, G4 reads chunk-indexed by array_chunk_idx_out).
    // SEAM_R1_FULLW=0 = legacy single-chunk head-0 broadcast pilot (NCHUNKS=1).
    // =====================================================================
    parameter bit  SEAM_R1_FULLW  = `EAGLE4_SEAM_R1_FULLW,
    // =====================================================================
    // S7a (S7_FSM_SEQUENCING_SPEC §S7a): the FRONT-HALF FSM sequencing strap.
    // S7A_FRONT_SEQ=1 (DEFAULT) wires (a) the G4 per-stage OUTPUT-TILE sequencer
    // (walk tile 0..71 inside S_G4, re-pulse array_start per tile, settle 1
    // cycle per tile, g4_done only on tile 71), and (b) the full 24-head
    // attention splice (g4_qkv_seam -> attention_full_mha) in place of the head-0
    // D_HEAD=16 pilot, with g5_done from attention_full_mha.done + a clean
    // stream-out into the G5->G6 seam. S7A_FRONT_SEQ=0 = the legacy pilot.
    // =====================================================================
    parameter bit  S7A_FRONT_SEQ  = `EAGLE4_S7A_FRONT_SEQ,
    // =====================================================================
    // S7b — BACK-HALF output-tile SEQUENCING strap. BACK_SEQ=1 (DEFAULT) wires
    // the GENERIC per-stage output-tile sequencer across G6(24)/G7g(64)/
    // G7u(64)/G9(24): the top_fsm still pulses gX_start once on entering each
    // stage, but a mini-FSM walks ALL the stage's output tiles (re-pulsing the
    // array start per tile with a per-tile col_offset_acc settle) and only
    // asserts gX_done on the LAST tile. It also drives the silu 512-pulse `en`
    // (reading the R4g/R4u seams across all 512 chunks) + the g11 R6 input feed
    // S3 fixes (off-by-one -1 + last-chunk range gate). BACK_SEQ=0 = the legacy
    // single-tile pilot (host tile_idx, single g7p_start silu pulse). See
    // S7B_BACK_FSM_RESULT.md.
    // =====================================================================
    parameter bit  BACK_SEQ       = `EAGLE4_BACK_SEQ,
    // =====================================================================
    // SLR-safe scaled-GEMV controls (SLRSAFE_SCALED_ARRAY_RESULT / PDI6_TIMING_RISK):
    // the scaled K_PAR=16 array spans 3 SLRs (1,536 URAM). SLR_SAFE=1 wraps it in
    // gemv_stage_array_scaled_slr with registered SLR-crossing seams (a_chunk /
    // qz_raw / stage_sel / tile_idx / start fwd + acc/result_q/done return,
    // PIPE_STAGES deep per group) — BIT-EXACT to SLR_SAFE=0, just +2*PIPE_STAGES cyc
    // (the done-driven handshake auto-extends). SLR_SAFE=0 (DEFAULT) = combinational
    // passthrough = byte-AND-cycle identical to the plain scaled array (the xsim
    // FUNCTIONAL config). SLR_SAFE=1 + ARR_GROUPS=3 + ARR_PIPE_STAGES=1 is the
    // build-time 125 MHz TIMING config (with the 3-SLR pblock XDC, S9).
    // =====================================================================
    // PDI#10 SILICON-TRUE DEFAULT (2026-06-08, FIX_INTEGRATION_TOP): SLR_SAFE=1
    // is the golden 3-SLR 125 MHz build config (PDI10_D1_REGREAD_CONFIRM live
    // config = SLR_SAFE=1, ARR_GROUPS=3, ARR_PIPE_STAGES=1). It supplies the +1
    // forward-seam register that the D1_LAT=1 registered-read fix is balanced
    // against — D1_LAT=1 MUST be paired with SLR_SAFE=1. xsim functional A/B can
    // still force SLR_SAFE=0/D1_LAT=0 via -generic_top to get the legacy combread
    // golden path.
    parameter bit  SLR_SAFE        = `EAGLE4_SLR_SAFE,  // SILICON DEFAULT (registered 3-SLR seams; paired with D1_LAT=1)
    parameter int  ARR_GROUPS      = `EAGLE4_ARR_GROUPS,
    parameter int  ARR_PIPE_STAGES = `EAGLE4_ARR_PIPE_STAGES,
    // =====================================================================
    // FF_ACT_CACHE (2026-06-21, pdi13_ffact) — per-SLR FF activation caches for
    // cross-SLR activation delivery, the FUNDAMENTAL version of the SLR fix.
    //   Two integrations, BOTH gated behind this single strap:
    //   (1) BODY broadcast: gemv_stage_array_scaled_slr forward seam becomes a
    //       2-stage hub(SLR1)+per-group(SLRk) pipe (lead-by-2 lookahead), so the
    //       only cross-SLR arc is a clean FF->FF hop (the RCA worst body path:
    //       g_bcast_pipe -> DSP58 in another SLR, LMTAP_REGRESSION_A_DCP_DIFF.md).
    //   (2) GLM seam: a per-SLR FF LANDING register on the lmdown input h
    //       (seam_r8_rd -> u_lmdown.a_chunk) so GLM's lanes consume a LOCAL FF
    //       copy, never a cross-SLR BRAM read. Threaded into u_lmdown via
    //       SEAM_FF_LAND, with one extra matched weight/marker delay to absorb
    //       the +1 landing-reg latency (bit-exact).
    //   FF_ACT_CACHE=0 (DEFAULT): byte+cycle identical to the canonical RTL
    //   (both integrations collapse to wire-throughs / the legacy seam).
    //   FF_ACT_CACHE=1: BIT-EXACT result, +1 forward latency on the body GEMV and
    //   +1 on the GLM activation arm, both absorbed by the done-driven handshake.
    // =====================================================================
    // ============================ pdi17 SEAMFIX (2026-06-23) =============================
    // DECOUPLED STRAPS. pdi16 conflated TWO integrations under one FF_ACT_CACHE strap:
    //   (a) BODY GEMV broadcast lead-by-2 (u_array .FF_ACT_CACHE) -- UNPROVEN; its module
    //       A/B xelab HUNG, and on silicon it deterministically broke G4k (0x4ef5ab69 vs
    //       golden 0x6c76c42f). RCA (FF16_RCA_A/D): the lead-by-2 seam lands the registered
    //       weight col_offset_acc re-base 2 cycles late vs the +1 the G4 settle budget
    //       allows -> every K/V column reads its opening chunk at the PREVIOUS column.
    //   (b) GLM lmdown seam landing reg (u_lmdown .SEAM_FF_LAND) -- the actual GLM=0 fix,
    //       independent path (lmdown_gemv_banksplit.sv), silicon-proven safe on silufix2.
    // pdi17 REVERTS the body to the silufix2-golden +1 seam (FF_ACT_CACHE=0) and KEEPS the
    // GLM fix on its OWN strap (SEAM_FF_LAND_P=1, wired to u_lmdown). This is the CLEAN
    // correctness PDI: body == silufix2-golden (G1->G11 golden) + the GLM seam fix.
    // PREFETCH_COL_P=0 + STAGE_INJECT=0 minimize the diff vs silufix2-golden (one var).
    //   FF_ACT_CACHE=0  -> body u_array legacy +1 broadcast seam (silicon-golden silufix2).
    //   SEAM_FF_LAND_P=1-> GLM u_lmdown per-SLR FF landing reg (the GLM=0 fix; UNTESTED on
    //                      silicon -> the gate MUST verify GLM CRC=0x859b57b1, not just G4k).
    // SEAM_FF (2026-06-26 pdi20 STRUCTURAL FIX, BAKED DEFAULT=1): whole-design selector
    // for the inter-stage activation buffer storage primitive. 0 = BRAM stage_io_buffer
    // (byte-identical to pdi18/pdi19 baseline); 1 = FF/REGFILE stage_io_buffer_ff
    // (NO ram_style="block", trades SLR-fragile BRAM nets for abundant FF). Threaded into
    // every stage_io_buffer_sel wrapper instance (11 seams) so the swap is ONE param, not
    // 11 hand-edits. RD_LAT/AUTO_SEQ/FRAC_BITS/DBG_TAP wiring is identical in both arms
    // (see stage_io_buffer_sel.sv). pdi20 bakes SEAM_FF=1 (the FF-staging fix).
    parameter bit  SEAM_FF         = `EAGLE4_SEAM_FF,
    parameter bit  FF_ACT_CACHE    = `EAGLE4_FF_ACT_CACHE,   // pdi17: body GEMV reverts to silufix2-golden +1 seam
    // Stage-B' WO-025: TDP dual-read GEMV. 0 = single-port lane (default,
    // bit+cyc identical, a_chunk_b/chunk_idx_b_out inert). 1 = 2 chunks/cyc
    // (port A even / port B odd) via gemv_lane_scaled_stream_dual + URAM
    // RD_PORTS=2 — per-column cycles halve. Requires the scaled arm +
    // CONNECT_SEAMS + FRONT/BACK_SEQ + SEAM_R1_FULLW + BODY_BATCH_B=1
    // (elab-asserted below).
    parameter bit  DUAL_PORT       = `EAGLE4_DUAL_PORT,
    // TREE-BATCH (sibling-batch / batched-frontier, doc 14 §2 Stage-2 P2): process
    // BODY_BATCH_B<=4 same-depth tree siblings through the ONE weight-resident body
    // GEMV array in a single pass (URAM weight read ONCE, MAC'd against B sibling
    // activation columns held in FF). Threaded into u_array's BATCH_B. DEFAULT 1 =
    // byte-AND-cycle identical to the shipping WIDE drafter (batch ports inert:
    // a_chunk_batch tied off, result_q_batch='0 in the array). B>1 requires the
    // production array config (SLR_SAFE=1, FF_ACT_CACHE=0, DUAL_PORT=0 — asserted
    // in the array). The tree/frontier feed of the B sibling columns is the P2
    // integration (staged); the batch datapath + array gates land here first.
    parameter int  BODY_BATCH_B    = `EAGLE4_BODY_BATCH_B,   // RECLOCK100 (build-id 0xE4B71110): B=1 — the SPEED line keeps
                                           // batch OFF (roadmap Night-1 Track B: batch is the depth line;
                                           // keep them separable so a batch place-fail can't sink the
                                           // reclock number). tree_batch ship default was 2:
                                           // hardware INSTANTIATED (one extra sib lane-set on the shared body
                                           // array; URAM/BRAM-neutral, +182K LUT per Gate-4 OOC). At depth-0 the
                                           // forward consumes only the b0 column (result_q_per_lane) so M7 is
                                           // preserved bit-exact; the sibling column (a_chunk_batch/result_q_batch)
                                           // is exercised ONLY by the host BATCH-TEST reg window (0x138..0x148).
    parameter bit  SEAM_FF_LAND_P  = `EAGLE4_SEAM_FF_LAND_P,
    parameter bit  TREE_SEAM_FF    = `EAGLE4_TREE_SEAM_FF,     // g8->bridge logit seam: registered sync-read (kills 0x1F tied-zero)   // pdi17: GLM lmdown seam FF landing reg (the GLM=0 fix)
    // Model-provenance strap for the synth-baked 3072->256 GLM ROM.
    // 2026-07-28: was hardcoded to p4 (ae3c bake, md5 b1008671, 2026-06-08) in
    // EVERY fork including this one, unconditionally, regardless of which
    // checkpoint's other weights were boot-streamed. Every 0xE4B71121 8815 run
    // silently ran 8815 body weights through an ae3c GLM. Proven root cause via
    // scripts/regen_8815/m8815_chimera_probe.py (host-only, zero card): 8815
    // everywhere + ae3c lmdown alone reproduces the wrong silicon frontier;
    // reverting lmdown to 8815 recovers the exact golden. p4_8815 is a verified
    // byte-complete 8815 bake (16/16 lanes, matches the probe's own derivation,
    // md5 f8402ce6 on lane0) copied non-destructively from a 2026-07-24 session
    // -- does not touch p4, which 10+ other forks still depend on for ae3c.
    // See reports/AUTONOMOUS_DRIVE_LOG.md 2026-07-28 ~18:5x and NOW.md.
    parameter string LMDOWN_MEM_DIR_P = "/home/y344shi/eagle4_artifacts/p4_8815",
    // PDI#10 D1 rework: single knob for the GEMV activation-seam registered-read
    // (RD_LAT) + paired array weight/enable delay (WLAT). D1_LAT=0 = legacy
    // combinational-read seam (sim==the GOOD 8/8 combread baseline). D1_LAT=1 =
    // registered-read seam (silicon BRAM truth) + WLAT compensation. Threaded so
    // the full-chain A/B is one generic, not 6 hand-edits.
    parameter bit  D1_LAT          = `EAGLE4_D1_LAT,  // SILICON DEFAULT (registered GEMV activation seam: RD_LAT=1 + array WLAT=1; FIX_INTEGRATION_TOP). Force 0 via -generic_top for the legacy combread A/B golden.
    // =====================================================================
    // PREFETCH_COL_P (pdi16) — G12/body GEMV column double-buffer prefetch.
    //   Threads PREFETCH_COL into the scaled-SLR array (gemv_stage_array_scaled_
    //   slr): the wrapper self-sequences a contiguous run of columns and prefetches
    //   col+1 in the previous column's drain, hiding the per-column refill bubble.
    //   PRESENT-but-OFF in ARCH_AS_BUILT; bit-exact vs PREFETCH_COL=0 (task #233,
    //   the frontier/cap-CRC is INVARIANT — only the drain bubble is hidden).
    //   1 = enabled (pdi16 default ON), 0 = single-column (byte+cyc identical).
    //   pdi17: OFF -> maximize byte-identity with silufix2-golden (one variable change).
    // =====================================================================
    parameter bit  PREFETCH_COL_P  = `EAGLE4_PREFETCH_COL_P,   // pdi17: single-column (minimize diff vs silufix2)
    // Independent knobs so RD_LAT (seam registered read) and WLAT (array weight+
    // enable delay) can be A/B'd separately. Default = D1_LAT (paired). Override
    // via -generic for the rework sweep (e.g. D1_RDLAT=1 D1_WLAT=0).
    parameter bit  D1_RDLAT        = `EAGLE4_D1_RDLAT,
    parameter bit  D1_WLAT         = `EAGLE4_D1_WLAT,
    // -----------------------------------------------------------------
    // R4_BANKSPLIT (2026-06-15, pdi11_g7pkeep_phys) — G7p PHYSICAL fix
    //   knob (Fix B, fallback for the single-SLR pblock primary fix).
    //   The G7p silicon scramble (cap-CRC 0x264e6fa4 vs golden 0xd8d6d147)
    //   is a PHYSICAL post-route effect: the 288-bit (16 lane x 18-bit
    //   Q6.12 = 4x cascaded RAMB36 READ_WIDTH_A=72) R4 silu-feed seam read
    //   net (u_seam_r4g/u_seam_r4u -> r4_silu_rd_idx broadcast) SLR-slips /
    //   lane-aliases on the live floorplan — the SAME wide-read-net class as
    //   the M7 +128 256b load bug. Gate-level funcsim is GOLDEN 16/16
    //   (R4SEAM_GATELEVEL_RCA.md) so sim cannot expose it; only on-card
    //   cap-CRC validates. R4_BANKSPLIT=1 replaces each wide u_seam_r4*
    //   stage_io_buffer with stage_io_buffer_banked (8x36b = 8 non-cascaded
    //   single-RAMB banks, BRAM-TILE-NEUTRAL: 16xRAMB18 = 8 Block RAM Tiles
    //   = same 8 tiles as 4xRAMB36) addressed by the SAME r4_silu_rd_idx, so
    //   each lane-pair is its own pinnable RAMB (easier to confine to ONE
    //   SLR than a 4-RAMB cascade). PER the +128 lesson
    //   (feedback_banksplit_did_not_fix_plus128_nonuniform) narrow ALONE
    //   morphs (does not cure) a wide-net slip — so the PRIMARY fix is the
    //   single-SLR pblock {u_seam_r4g,u_seam_r4u,silu} in the .xdc; this
    //   bank-split is the read-net-narrowing fallback that makes that pblock
    //   easier to satisfy. R4_BANKSPLIT=0 (DEFAULT) = the wide seam =
    //   BYTE-IDENTICAL to g7pkeep (depth-1 + the speed config unaffected).
    parameter bit  R4_BANKSPLIT     = `EAGLE4_R4_BANKSPLIT,
    // =====================================================================
    // CTX_WIN — attention context window (OPTION-B, FIX-G batched expose).
    // Was hardcoded 1024 at the attention_full_mha instance. Threaded as a top
    // generic so the S9 build can shrink it (e.g. -generic CTX_WIN=768) to free
    // the attention KV-window URAM that FIX-G's +lmdown bank needs. DEFAULT 1024
    // keeps the current (1024-window) behaviour bit-identical. CTX_WIN <= 1024
    // (the attn_head_assembly exp_buf depth CTX_MAX stays 1024 >= CTX_WIN).
    // =====================================================================
    parameter int  CTX_WIN         = `EAGLE4_CTX_WIN,
    // =====================================================================
    // DEPTH_N — on-chip depth-N tree level loop (tree FSM stage 2).
    // 0 (DEFAULT): single-pass drafter, BIT-IDENTICAL to the shipped design.
    // 1: one launch walks max_depth (reg 0x20) levels, K_F body passes per
    //    level >= 1, each branch on its own KV lane-chain column.
    // Strapped via `-d DEPTH_N_V=1` to BOTH xvlog and xelab (a DEFINE, mirroring
    // the EMBG_EN_V pattern) so the default costs literally zero logic.
    // =====================================================================
    parameter bit  DEPTH_N         = `EAGLE4_DEPTH_N,
    // =====================================================================
    // EMBG_EN — GAP-4 RUNG C child-embedding gather graft.
    // 0 (DEFAULT): m_axi_emb tied off, input_emb_buf host-only writer, FSM
    //    gather-wait folded away -> BIT-IDENTICAL to the pre-graft drafter.
    // 1 (needs DEPTH_N=1): the emb_gather AXI master reads the TARGET
    //    embed_tokens image at hbm_emb_base (ctrl 0x40/0x44) and lands each
    //    depth>=1 child's row into input_emb_buf via the byte-enabled write
    //    arbiter; top_fsm detours through S_LVL_EMBG before each child G1.
    // Strapped `-d EMBG_EN_V=1` to BOTH xvlog and xelab.
    // =====================================================================
    parameter bit  EMBG_EN         = `EAGLE4_EMBG_EN,
    // =====================================================================
    // EMB_NCH — MULTI-CHANNEL HBM for the child-embedding gather.
    //   Standing user directive: "when using the HBM, always use multiple
    //   channels."  1 (DEFAULT) = the shipping single-channel S08 path.
    //   2 = emb_gather splits the row into 2 contiguous element slices and
    //   fetches them through TWO masters (m_axi_emb -> S08_AXI/HBM5_PORT1,
    //   m_axi_embb -> S09_AXI/HBM5_PORT0) reading the SAME flat
    //   embed_tokens image at 0x004280000000 — SAME PC, both of its NoC
    //   ports, one master per port (fan-in 1/port, the S05/S07-on-HBM3
    //   pattern).  No duplicate image, no host relayout, no ABI change.
    //   *** MEASURED: 199 -> 197 cyc/row (build/emb_nch, AR_LAT=0). The
    //   landing buffer takes ONE 256b chunk/cycle and a single 256b channel
    //   already delivers one beat/cycle, so the row stays sink-bound at
    //   BEATS_ROW=192 writes.  <=0.03% of a depth-N level either way.
    //   SHIPPED OFF (=1) on the numbers, not on doubt — the port geometry,
    //   RTL and BD patch exist so it is a STRAP the day the sink widens
    //   (K_F-parallel per-child gather).  reports/results/EMB_GATHER_MULTICHANNEL.md.
    // =====================================================================
    parameter int  EMB_NCH         = `EAGLE4_EMB_NCH,
    // =====================================================================
    // FORREASON (F1) — the depth>=1 re-key hidden ("for_reasoning") producer.
    // 0 (DEFAULT): G9 walks 24 col-tiles (N=3072, to_logits only), no `a`
    //    producer, no `a` bank — BIT-IDENTICAL to the pre-F1 drafter (every
    //    added expression folds against a constant-0 select).
    // 1 (needs DEPTH_N=1): G9 walks 48 col-tiles (N=2H=6144 — the checkpoint's
    //    real down_proj [512,6144], enable_2hs), tiles 24..47 are captured at
    //    the accumulator-native frac8 into g9hi_buf, and
    //        a = G9_down[H:2H] + ( block_input + G6_oproj_out )
    //                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ == h1_buf_f (R3)
    //    is produced into for_reasoning_producer's FF `a` bank.
    //    Measured contract: DEPTHN_REKEY_GOLDENS.md §4 A/B/C, max rel-L2
    //    2.24e-04 over 349 real GPU drafter forwards.
    //    ZERO URAM delta: the weight bank's address map (stage_geometry.svh
    //    sg_stage_N(4)=6144) and BANK_DEPTH=96K ALREADY reserve all 48 G9
    //    columns, with G12 based above them — F1 is a CONTROL change on the
    //    tile walk, not a storage change (reports/results/TREE_FSM_DEPTHN_F1_FORREASONING.md §2).
    // Strapped `-d FORREASON_V=1` to BOTH xvlog and xelab.
    // =====================================================================
    parameter bit  FORREASON       = `EAGLE4_FORREASON,
    parameter int  FR_KF_BANKS     = `EAGLE4_FR_KF_BANKS,
    // CONGFIX-1: a-bank storage style. Parameter (NOT a `define) so it survives
    // BD re-synthesis from component.xml — packaging clears verilog_define.
    parameter bit  FR_ABANK_BRAM   = `EAGLE4_FR_ABANK_BRAM,
    // HEAD_PAR — G5 head-parallelism (ported from speedwave_v2_g5mem, Stage-A).
    // Parameter (NOT a `define) so it survives BD re-synthesis — the 0x18 burn
    // was HEAD_PAR being packaging-invisible. Default 1 = SERIAL G5 = the
    // ps_tree_fsm gate-2-4-proven config (byte-identical). 6 = 6-way lockstep.
    parameter int  HEAD_PAR        = `EAGLE4_HEAD_PAR,
    // =====================================================================
    // 0x1125 ACCFIX (kvres_accfix, 2026-07-29): elaboration presence of the
    // four accept-path fixes. Each is RUNTIME-gated by the ACCFIX register
    // (0x15C, reset 0 = every fix OFF = byte-identical to 0x1124), so one
    // elaboration serves baseline AND fix arms. Elab=0 generate-prunes the
    // fix's cells (netlist rollback). OBS_LVL_P = per-level candidate ring.
    // =====================================================================
    parameter bit  ACCFIX_SEAM_P   = `EAGLE4_ACCFIX_SEAM_P,
    parameter bit  ACCFIX_B2P_P    = `EAGLE4_ACCFIX_B2P_P,
    parameter bit  ACCFIX_KC_P     = `EAGLE4_ACCFIX_KC_P,
    parameter bit  ACCFIX_KVONLY_P = `EAGLE4_ACCFIX_KVONLY_P,
    parameter bit  OBS_LVL_P       = `EAGLE4_OBS_LVL_P,
    parameter bit  ENORM_D0        = `EAGLE4_ENORM_D0,
    // Atomic fixed-point parity experiment. Default 0 preserves every legacy
    // grid. When 1, the embedding loader must supply signed INT16 frac16.
    parameter bit  PRECISION_SAFE  = `EAGLE4_PRECISION_SAFE,
    // =====================================================================
    // SILU_RANGE_FIX — gate→sigmoid range extension (Q6.12, un-clipped silu).
    // 0 (SOURCE DEFAULT): legacy silu_layer (Q4.12 16-bit, 64-entry ±8 sigmoid
    //   LUT) — the seed-42 ±8 gate clip (125/8192 |gate|>8 channels corrupted).
    // 1: silu_layer_fix (IN_W=18 Q6.12 gate/up, 512-entry sigmoid LUT over the
    //   full gate range, round-half stage-A). Inputs un-clipped (real gate≤12.5,
    //   up≤18.0 no longer pinned to ±8). Q8.8 output contract unchanged.
    // Threaded as a PORT parameter so the full-chain frontier gate can A/B it
    // (-generic_top SILU_RANGE_FIX=1) WITHOUT a source edit. SOURCE DEFAULT is
    // OFF until the seed-42 RESCORED frontier is PROVEN UNCHANGED with it ON
    // (FIX_SILU_RANGE.md). Unit-gated bit-tight vs golden on |gate|>8 (maxabs
    // 6 LSB Q8.8) by tb_silu_range_gate; legacy clip diverges (teeth 7001 LSB).
    // =====================================================================
    parameter bit  SILU_RANGE_FIX_P = `EAGLE4_SILU_RANGE_FIX_P,  // SOURCE DEFAULT ON (Q6.12 un-clipped; R4_AW=18, no 288->256 truncation). Legacy +-8 clip (=0) is DEAD: it caused the G7p=0x264e6fa4 truncation that cost 2 cold-ACs (2026-06-17). NEVER build 0. (memory feedback_staged_componentxml_neq_built_netlist_xci)
    // =====================================================================
    // WO-053 — KV V-stack storage primitive (BRAM-shed the SLR crossing).
    //   "block" (DEFAULT) : V stacks → RAMB36 so they co-locate with the
    //                       attn_v_combine MAC in ONE SLR → kills the
    //                       −0.780 ns KV-read→combine SLR crossing.
    //   "ultra"           : original URAM behaviour (pristine baseline /
    //                       for the bit-exact A/B comparison).
    // K stays URAM. PURE implementation attribute — xsim ignores it, so the
    // frontier is BIT-IDENTICAL for both values (feedback_no_accuracy_tradeoff).
    // The S9 build sets it via -generic KV_V_RAM_STYLE=block.
    // =====================================================================
    parameter      KV_V_RAM_STYLE  = `EAGLE4_KV_V_RAM_STYLE,
    parameter      KV_K_RAM_STYLE  = `EAGLE4_KV_K_RAM_STYLE,
    // =====================================================================
    // WO-051 — per-KV-stack LOCAL attn_v_combine (root-cause SLR-crossing fix).
    //   COMBINE_LOCAL=1 (DEFAULT, 2026-06-02 WO-051-PROPAGATION-FIX) : per-bank
    //                       combine in u_attn_full — the wide V-row stays
    //                       SLR-local; only the registered reduced partial
    //                       crosses (reg->reg). BIT-EXACT to the shared combine
    //                       (tb_attn_head_assembly_perbank 2688/0). This is the
    //                       SILICON WE WANT, so it is the default — the BD/.xci
    //                       does NOT carry a COMBINE_LOCAL key, so the AVED
    //                       BD-instantiated IP synth picks up THIS default
    //                       (the forkE -generic override does NOT persist into
    //                       the .xci → previously fell back to 0 → shared
    //                       combine got synthesized → −0.780 ns SLR crossing).
    //   COMBINE_LOCAL=0   : shared combine (former golden; BIT-IDENTICAL but
    //                       has the −0.780 ns wide-V SLR crossing). Still fully
    //                       reversible via -generic / CONFIG override.
    // The forkE2 package still sets -generic COMBINE_LOCAL=1 belt-and-suspenders.
    // =====================================================================
    parameter bit  COMBINE_LOCAL   = `EAGLE4_COMBINE_LOCAL,
    // =====================================================================
    // SPEEDWAVE-V2 W3-F2 (ported from jewel J5B_F2_INTEGRATION, 2026-07-18):
    //   requant relocated behind the 24:1 head mux. =1 (DEFAULT SHIP): do NOT
    //   instantiate g4_qkv_seam (9,216 parallel rsh_sat units ≈ 440K LUT on
    //   the jewel netlist, util_truth_hier.rpt:29); pass RAW INT32 q/k/v tiles
    //   to attention_full_mha_sbuf (RAW_REQUANT=1) which head-muxes 24:1 FIRST
    //   and requants only the selected head's 3x128 lanes (384 units).
    //   Bit-exact by commutation: mux(rsh_sat(x)) == rsh_sat(mux(x)) — same
    //   per-element function, same C1-REVERT frac contract (12->10).
    //   =0: legacy g4_qkv_seam path (byte-identical fallback).
    parameter bit  REQUANT_HEADMUX = `EAGLE4_REQUANT_HEADMUX,
    // =====================================================================
    // RTL WO#8 (run-11): rope rotation-cone pipeline + pos_d1 locality
    // replicas. Threaded to BOTH rope_layer instantiations (the sbuf
    // u_attn_full/u_rope — the run-9 failing cone — and the legacy head-0
    // pilot u_rope, which is mux-bypassed at S7A_FRONT_SEQ=1 but strapped
    // identically so no un-piped rope cone can survive in any config).
    // Contract + gate evidence: rope_layer.sv header, ROPE_CONE_PIPELINE.md.
    parameter bit  ROPE_CS_PIPE    = `EAGLE4_ROPE_CS_PIPE,
    parameter int  ROPE_POS_REP    = `EAGLE4_ROPE_POS_REP,
    // RTL WO#12 (run-12 lever): V-KV bank-group address replication — the
    // run-10 RESPIN50 KV_VADDR blocker (+0.028 @66, bar >=+0.30; -0.362
    // @80). Splits each V stack into KV_ADDR_REP width-slice memories,
    // each addressed by its own keep'd same-cycle replica (read AND
    // write). Latency-neutral, bit-exact. Threaded to u_attn_full (which
    // composes it with HEAD_PAR=6). Contract: reports/results/KVADDR_REP_FIX.md.
    parameter int  KV_ADDR_REP     = `EAGLE4_KV_ADDR_REP,
    // F3-CLR (2026-07-28): fan-out replication of the host-writable KV
    // residency clear pulse inside each engine's kv_resident bitmap.
    // Contract: reports/results/KV_RESIDENCY_CLEAR_FIX.md.
    parameter int  KV_RES_CLR_REP  = `EAGLE4_KV_RES_CLR_REP,
    // =====================================================================
    // G8 SPLICE (PREP, fork pdi6_s7c_g8 — reports/G8_S7C_SPLICE_PREP_20260528.md)
    //   G8_SPLICE=0 (DEFAULT) : BIT-IDENTICAL to pdi6_s7c. g8_rescore is NOT
    //     instantiated; the output writer fires on tree_done with the tree's
    //     cheap top-K_F=8; the m_axi_g8_* gather port parks quiescent; top_fsm
    //     S_TREE -> S_DONE directly.
    //   G8_SPLICE=1 : instantiate the validated g8_rescore (fused-gather-dot
    //     frontier rescore) between the G12-fed tree vocab-walk and the output
    //     DMA. g8_rescore's internal G8.1 selector snoops the SAME G12 tile
    //     stream that feeds tree_decode (cheap top-K_C=512), gathers each
    //     candidate's lm_head_full row from HBM via the NEW read-only m_axi_g8_*
    //     master (lmfull_hbm_base @ ctrl 0x14/0x1c), fp16-dots it against the
    //     captured G11 final-norm (to_logits_norm), and re-orders to a rescored
    //     top-8 that drives the output writer. top_fsm gains the S_G8 phase.
    //   The DEFAULT path elaborates byte-for-byte the s7c top; flipping to 1 is
    //   the turnkey golden-PDI build. See report §turnkey-checklist.
    // =====================================================================
    parameter bit  G8_SPLICE       = `EAGLE4_G8_SPLICE,
    // G8 dimensions (production): V=128256, H=3072, K_C=512, K_F=8.
    parameter int  G8_K_C          = `EAGLE4_G8_K_C,
    // G8 gather AXI4 256-beat sub-burst cap (bounded multi-sub-burst, the
    // AXI-protocol-correct gather validated in pdi6_g8 strict-AXI TB).
    parameter int  G8_MAX_BURST    = `EAGLE4_G8_MAX_BURST,
    // =====================================================================
    // PDI#9 MAINLINE: G8 read optimization (g8_rescore_p).
    //   G8_PIPELINED=1 swaps the serial g8_rescore for the PIPELINED
    //   g8_rescore_p (P1: G8_OUTSTANDING same-ID ARs in flight, reorder-free
    //   consume; seen_last + all-states watchdog ported in; measured 4.42x
    //   gather cycle reduction, ordered-8/8 bit-exact vs the serial frontier).
    //   G8_PREFETCH=1 compiles in the P2 ring hit-path (inert here: top ties the
    //   snoop ports off -> every row falls through P1 -> bit-exact to pure P1).
    //   G8_DATA_W is the g8 m_axi beat width; 32 matches the silicon-proven NoC
    //   S05_AXI port (G8_DATA_W=512 deferred — would re-open the SLR seam).
    //   pdi9_mainline DEFAULT = PIPELINED ON.
    // =====================================================================
    parameter bit  G8_PIPELINED    = `EAGLE4_G8_PIPELINED,
    // pdi21 G8RESC SCALAR-FIX: drop 8->2. The deep-AR fleet (reports/results/G8_DEEP_AR.md)
    // found N=2 SATURATES the gather (no throughput loss); >2 ARs in flight let the II=2
    // dot fall behind the landing beats -> beat-to-dot pairing DRIFTS over the 512-row
    // stream = the position-dependent G8RESC 2/8 silicon corruption (RCA
    // project_pdi20_g8resc_silicon_staging_rca). N=2 removes the drift surface.
    parameter int  G8_OUTSTANDING  = `EAGLE4_G8_OUTSTANDING,
    // SPEEDWAVE (EXP_G8_PINGPONG_II1, 2026-07-11): G8_ROWPP=1 = decoupled
    //   fetch/feed (8-deep FF beat ring) + owner FIFO + ROW_OVERLAP early dot
    //   snapshot in g8_rescore_p — removes the measured 23-cyc/row dot-flush
    //   bubble AND the OUTSTANDING-retire latency collapse (408.69→196.62
    //   cyc/row @L=300 with OUTST>=4; 512-row dump BYTE-IDENTICAL md5
    //   dce78393, RED owner-corrupt teeth. EXP_G8_PINGPONG_II1_RESULT.md).
    //   Requires G8_WIDE=1 && G8_WIDE_IL=1 ($fatal inside g8_rescore_p).
    //   NOTE the pdi21 8->2 OUTSTANDING cut above was the SCALAR beat-handoff
    //   drift (dot pairing by arrival order); the WIDE path pairs positionally
    //   (lockstep rc_elem/nrm_rd_addr + dot_owner + synth g8_err teeth on
    //   nrm_addr_q != h_elem), which does not share that surface — OUTST=4
    //   with ROWPP is the gated fold config. DEFAULT 0 = byte-identical legacy.
    parameter bit  G8_ROWPP        = `EAGLE4_G8_ROWPP,
    parameter bit  G8_PREFETCH     = `EAGLE4_G8_PREFETCH,
    // GRAND_WIDE fork: 512 = the 512b m_axi_g8 wide gather master (32 fp16/beat).
    //   MUST equal G8_WIDE_P*16 when G8_WIDE=1 (checked below). Set back to 32 to
    //   build the silicon-proven scalar path (then also set G8_WIDE=0).
    parameter int  G8_DATA_W       = `EAGLE4_G8_DATA_W,  // pdi25 G8WIDE: 256b m_axi_g8 (NMU-native; 512b was the pdi18 no-ARREADY fail — G8_GATHER_SPEED_BUILDREADY_SPEC F2). BD S05_AXI DATA_WIDTH must be {256}.
    // PDI#10 G8 SEL_SCAN fix: OPT1C non-blocking running-min selector (DEFAULT
    // ON).  =0 restores the legacy OPT1B SEL_SCAN selector that drops cheap_ready
    // during its rescan (repro of the logit-drop bug only).  Threaded to
    // g8_rescore_p.G8_OPT1C; default preserves the fixed behavior.
    parameter bit  G8_OPT1C        = `EAGLE4_G8_OPT1C,
    // PDI#10 G8_OPT1C_TIMING fix: the DEFAULT G8.1 min is now a BALANCED-TREE
    // flat running-min (cmin0; OOC WNS -456 ns -> -2.478 ns @100 MHz, a 184x comb-
    // depth cut, exact + golden-passing).  G8_OPT1D=1 selects the hierarchical
    // write-forward variant (slower in OOC; repro only).  cheap_ready stays HIGH
    // (zero drop); rescored frontier == golden (tb_g8_freerun_selscan).
    parameter bit  G8_OPT1D        = `EAGLE4_G8_OPT1D,
    parameter int  G8_MIN_NG       = `EAGLE4_G8_MIN_NG,
    // PDI#10 G8_PIPELINE 100MHz fix: OPT1F is the DEFAULT G8.1 min — a PIPELINED
    // registered 2-level segmented running-min (registered per-segment GSZ tree
    // with the in-flight write FORWARDED, + a SHORT comb NG-wide top tree).  It
    // breaks the -2.478 ns flat-min critical arc into two short registered hops so
    // the selector closes 100 MHz with margin, while staying BIT-EXACT to the flat
    // min (tb_g8_opt1d_ab A/B) and dropping ZERO logits (cheap_ready HIGH every
    // cycle; tb_g8_freerun_selscan rescored ordered-8/8 == golden).  =0 falls back
    // to OPT1C's flat balanced-tree min (the timing-fail reference).
    // PDI#10 SHIPPED DEFAULT = 0 (OPT1C flat-min @72MHz golden — the deliverable).
    // OPT1F (=1) stays available for a future 100MHz min-redesign, but it is an
    // architectural wall at bit-exactness (see FULLCHAIN_REGREAD_GREEN / STATE.md).
    parameter bit  G8_OPT1F        = `EAGLE4_G8_OPT1F,
    // SEL_PIPE (2026-07-09): registered-cmin OPT1F selector retiming — the 100MHz
    // IN-CONTEXT fix for the redteam full-top WNS -2.935 (all worst arcs =
    // u_g8 seg_min_reg->seg_pos_reg).  Only meaningful with G8_OPT1F=1; =0 falls
    // back to the unmodified OPT1F selector (the A/B reference).  See
    // g8_rescore_p.sv G8_SELPIPE header for the exactness argument + gates.
    parameter bit  G8_SELPIPE      = `EAGLE4_G8_SELPIPE,
    // =====================================================================
    // GRAND-BUILD G8 SPEED LEVERS (2026-06-13 integration — see
    // GRAND_INTEGRATION_RESULT.md).  Three orthogonal g8_rescore_p features:
    //   G8_OPT1G : item4 NEIGHBOR-ONLY SYSTOLIC INSERTION SORTER selector.
    //     Constant-depth comb (no K_C-wide reduce / decode) -> eliminates the
    //     -0.854 ns OPT1C/F selector carry-chain violator; A/B BIT-EXACT to
    //     OPT1C-flat (0/512 set-mism, 6/6 seeds) AND seed-42 golden frontier.
    //     DEFAULT ON (the 72 MHz selector closure).  Priority over OPT1F/D/C.
    //   MROUND_P : item4 fp16_mul_b register-split on the SCALAR g8_dot_fp16
    //     (bit-exact, +6.981 ns @72 MHz).  DEFAULT ON.  Inert when G8_WIDE=1
    //     (the wide dot replaces the scalar dot).
    //   G8_WIDE  : speed_g8 512b master + P=WIDE_P wide-consume dot.
    //     URAM/BRAM-NEUTRAL, +~16 DSP, FRONTIER-exact (re-baselined golden).
    //     DEFAULT *OFF* AT THE TOP: enabling it requires a 512-bit m_axi_g8
    //     master + a WIDE_P-wide nrm read port, NEITHER of which exists in the
    //     silicon-proven 32-bit S05_AXI NoC BD (speed_g8's author explicitly
    //     DEFERRED 512b: "would re-open the SLR seam").  The wide branch is
    //     fully wired + module-OOC-validated and selectable, but turning it ON
    //     is a SEPARATE BD/NoC change that needs its own through-top wide gate
    //     + 512b-master OOC before it is build-safe.  See GRAND_INTEGRATION_RESULT.md.
    parameter bit  G8_OPT1G        = `EAGLE4_G8_OPT1G,   // GRAND default ON (72MHz selector)
    parameter bit  G8_WIDE         = `EAGLE4_G8_WIDE,   // pdi25 G8WIDE: 256b wide gather + P=16 wide dot. pdi18's 512b no-ARREADY was the missing 4KB page-split (AXI4-illegal bursts), FIXED in g8_rescore_p (F1) + 256b NMU-native width (F2). Frontier-exact contract (re-associated adder tree; frontier gate = arbiter).
    parameter int  G8_WIDE_P       = `EAGLE4_G8_WIDE_P,
    parameter bit  G8_MROUND       = `EAGLE4_G8_MROUND,   // GRAND default ON (scalar-dot round-split)
    // pdi21 SCALAR II=1 lever (G8_DOT_II1.md): scalar g8_dot_fp16 FADD_PIPE=0 (single-
    // accumulator, per-row BIT-EXACT) + AXI consume feeds the dot every cycle = 2x
    // consume floor on the PROVEN transacting 32b gather (the only no-NoC speed lever).
    // =1 DEFAULT; =0 reverts to the pdi20 II=2 scalar path (byte-identical fallback).
    parameter bit  G8_SCALAR_II1   = `EAGLE4_G8_SCALAR_II1,
    // ---- FIX-CANDIDATE C: G8_CONSUME_HARDEN — matched-delay owned-nrm consume ----
    //   (reports/results/G8DUMP512_FIXC_CONSUME.md.)  =0 (DEFAULT) BYTE-IDENTICAL to
    //   the shipped pdi23 scalar consume.  =1 hardens the scalar gather->dot dot_b
    //   (nrm) read: an EXPLICIT owned FDRE + a matched 1-cycle skid instead of the
    //   comb nrm_buf read (removes reliance on the ram_style attribute; deterministic
    //   OWNED read latency), plus a beat-index consistency check (g8_err teeth) and a
    //   sim-only DATA-VALUE canary tap.  Fold-candidate; requires G8_SCALAR_II1=1.
    //   Default follows `G8_HARDEN (xvlog -d G8_HARDEN) so the xsim gate can flip the
    //   strap without editing the TB; the PDI package sets the param explicitly.
    //   The `ifdef G8_HARDEN selection now lives in eagle4_config.svh (single
    //   source of truth) — `EAGLE4_G8_CONSUME_HARDEN already resolves to 1'b1
    //   under -d G8_HARDEN and 1'b0 otherwise, so this is ONE line here.
    parameter bit  G8_CONSUME_HARDEN = `EAGLE4_G8_CONSUME_HARDEN,
    // ---- pdi24 G0 fold (FIXBUILD_PLAN.md §2.2): G8 narrow-burst LANE-SELECT fix --
    //   The scalar 32b gather issues 2-byte NARROW bursts but consumed rdata[15:0]
    //   every beat; per AXI4 A3.4.3 odd elements arrive on lanes [31:16] -> even-only
    //   dot -> the exact silicon wrong-8 [6621,40756,...]. =1 (DEFAULT) selects the
    //   active lane by element parity. The WIDE path (G8_WIDE=1) has NO narrow
    //   transfers (arsize == full bus width) — this covers the co-present scalar
    //   fallback. =0 = legacy byte-identical misread (A/B only, NEVER ship).
    parameter bit  G8_LANE_FIX     = `EAGLE4_G8_LANE_FIX,
    // ---- BEST-SPEED LEVER I: WIDE_IL — II=1 interleaved-accumulator wide dot ----
    //   Recovers the ~2x WIDE conceded at II=2 (898->459 gather cyc, 1.956x measured;
    //   REAL seed-42 512-row A/B frontier ordered-8/8 == golden == WIDE II=2 — see
    //   pdi11_wide_il/WIDE_IL_RESULT.md).  Only meaningful with G8_WIDE=1.  Default
    //   ON in this best-speed fork; set to 0 to fall back to the WIDE II=2 path.
    //   HARD CONTRACT (WIDE_IL_RESULT.md §5): the wide nrm read (g_nrm_wide below)
    //   MUST be COMBINATIONAL — it already is (`assign`); do NOT pipeline it or II=1
    //   reads nrm one group stale.
    parameter bit  G8_WIDE_IL      = `EAGLE4_G8_WIDE_IL,   // BEST-SPEED default ON (II=1 wide dot)
    parameter int  G8_WIDE_M       = `EAGLE4_G8_WIDE_M,      // II=1 interleave depth (pow2, >=2)
    // ---- SPEEDWAVE-V2 J5-β G8_NCH (ported from jewel J5B W2, 2026-07-18) ----
    //   N-channel STRIPED HBM gather: gather-order round-robin over the SAME
    //   flat 788 MB lm-head image (NO host relayout; slot k -> ch k%N).
    //   =1 (DEFAULT): byte-identical legacy — stripe generate OFF inside
    //   g8_rescore_p, m_axi_g8b tied off (port shape fixed for the BD).
    //   =2 (SHIP, set at package time): ch0 rides m_axi_g8 (S05_AXI/HBM3_PORT0),
    //   ch1 rides the NEW m_axi_g8b (S07_AXI/HBM3_PORT1 via bd_patch/
    //   S07_stripe.patch.tcl — silicon-proven by the J5B U2 unit PDI).
    //   Gate-proven 196->98 cyc/row @N=2 byte-identical (EXP-J5B GO).
    //   Requires G8_PIPELINED=1 && G8_WIDE=1 && WIDE_IL=1 && G8_ROWPP=1.
    parameter int  G8_NCH          = `EAGLE4_G8_NCH,
    // =====================================================================
    // BUFRDREG (PLUS128_LOADPATH_FIX_V2 L2 — MAXPROBE): register the G1/G1e
    // effective read CHUNK ADDRESS in a flop the L3 tight SOFT pblock co-locates
    // with the bank-split BRAM, breaking the ~14.6 ns combinational chunk-address
    // route (counter-FF -> decrementer LUT -> 256 BRAM ADDR sinks) that the +128/
    // +64 PVT slip exploits. Adds ONE read-latency cycle, absorbed by bumping the
    // G1/G1e rmsnorm RDATA_LAT 1->2. BUFRDREG=0 -> legacy comb-address read,
    // BYTE-IDENTICAL to pdi11_m7 (the A/B mode-0 gate). DEFAULT ON for the build.
    // =====================================================================
    parameter bit  BUFRDREG        = `EAGLE4_BUFRDREG,
    // PDI#11 PER-STAGE GOLDEN-INJECT (stage_inject.sv). DEFAULT 0 = the entire
    // inject path is generate-pruned -> netlist + behavior BIT-IDENTICAL to the
    // no-inject top (zero FF/LUT delta; the ultimate safe-when-off, memory
    // feedback_no_accuracy_tradeoff). Flip to 1 (via -generic_top for the gate, or
    // in the package tcl for a debug PDI) to compile in the inject muxes; the
    // runtime inject_en bitmask (0x10C) then arms per-stage isolation. With
    // STAGE_INJECT=1 AND inject_en=0 behavior is still bit-identical (mux selects
    // the real upstream net) — the headline byte-identity gate.
    parameter bit  STAGE_INJECT    = `EAGLE4_STAGE_INJECT
) (
    input  wire        ap_clk,
    input  wire        ap_rst_n,

    // ---- AXI4-Lite slave (control) ----
    // ADDR WIDTH = 9 (MAXPROBE): widened from 8 to reach the liveness window
    // 0x100/0x104/0x108. BD control aperture is 4 KB so the extra bit is mappable;
    // 0x00..0xF8 decode bit-identical (their bit[8]=0). The package .tcl re-infers
    // the port width from this RTL. (Was 8-bit at PDI#7 for 0x80/0x84.)
    input  wire [8:0]  s_axi_control_awaddr,
    input  wire        s_axi_control_awvalid,
    output wire        s_axi_control_awready,
    input  wire [31:0] s_axi_control_wdata,
    input  wire [3:0]  s_axi_control_wstrb,
    input  wire        s_axi_control_wvalid,
    output wire        s_axi_control_wready,
    output wire [1:0]  s_axi_control_bresp,
    output wire        s_axi_control_bvalid,
    input  wire        s_axi_control_bready,
    input  wire [8:0]  s_axi_control_araddr,   // MAXPROBE: 9-bit (was 8-bit; see awaddr note)
    input  wire        s_axi_control_arvalid,
    output wire        s_axi_control_arready,
    output wire [31:0] s_axi_control_rdata,
    output wire [1:0]  s_axi_control_rresp,
    output wire        s_axi_control_rvalid,
    input  wire        s_axi_control_rready,

    // ---- Boot-time weight load (driven by host via a separate channel) ----
    input  wire        boot_we,
    input  wire [15:0] boot_lane_sel,
    input  wire [16:0] boot_wr_addr,
    input  wire [31:0] boot_wr_data,

    // ---- B2: TB direct-drive scale-BRAM load (xsim only; quiescent on silicon
    //      where the dma_sc_* loader path fills the scale BRAM). Tied 0 by the
    //      silicon harness; the top-level Gate TB drives these to preload the
    //      per-group Q3.13 scales into the scaled array's per-lane scale_bram. ----
    input  wire        tb_sc_we,
    input  wire [15:0] tb_sc_lane_sel,
    input  wire [16:0] tb_sc_wr_addr,
    input  wire [15:0] tb_sc_wr_data,

    // ---- M9: boot-only m_axi read master (HBM → URAM bulk weight load) ----
    // AXI4 READ-ONLY. SILENT during compute (single ARVALID driver, boot
    // phase only). The orchestrator wires this to one HBM NoC aperture in
    // create_bd_design_layout_b.tcl (model on GM0_BASE 0x4000000000).
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_boot_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_boot_araddr,
    output wire [7:0]                 m_axi_boot_arlen,
    output wire [2:0]                 m_axi_boot_arsize,
    output wire [1:0]                 m_axi_boot_arburst,
    output wire                       m_axi_boot_arvalid,
    input  wire                       m_axi_boot_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_boot_rid,
    input  wire [BOOT_AXI_DATA_W-1:0] m_axi_boot_rdata,
    input  wire [1:0]                 m_axi_boot_rresp,
    input  wire                       m_axi_boot_rlast,
    input  wire                       m_axi_boot_rvalid,
    output wire                       m_axi_boot_rready,

    // ---- OUTPUT-DMA: compute-result write master (tree top-K → HBM C2H) ----
    // AXI4 WRITE-ONLY (AW/W/B). The SOLE AXI write master in the drafter
    // (the M9 boot DMA above is read-only). AWVALID asserts ONLY for the tiny
    // transient burst after tree_done, then parks silent (AXI-fan-in rule).
    // The orchestrator wires this to an HBM NoC aperture in the BD; the host
    // reads it back C2H from out0_base (see output_dma_writer's HBM ABI).
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_out_awid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_out_awaddr,
    output wire [7:0]                 m_axi_out_awlen,
    output wire [2:0]                 m_axi_out_awsize,
    output wire [1:0]                 m_axi_out_awburst,
    output wire                       m_axi_out_awvalid,
    input  wire                       m_axi_out_awready,
    output wire [BOOT_AXI_DATA_W-1:0] m_axi_out_wdata,
    output wire [BOOT_AXI_DATA_W/8-1:0] m_axi_out_wstrb,
    output wire                       m_axi_out_wlast,
    output wire                       m_axi_out_wvalid,
    input  wire                       m_axi_out_wready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_out_bid,
    input  wire [1:0]                 m_axi_out_bresp,
    input  wire                       m_axi_out_bvalid,
    output wire                       m_axi_out_bready,

    // ---- Activation streaming inputs (driven by upstream layer modules) ----
    input  wire signed [K_PAR*A_WIDTH-1:0]  a_chunk,
    input  wire        [1:0]                qz_raw,
    input  wire        [15:0]               tile_idx,

    // ---- G8 SPLICE: lm_head_full gather m_axi READ master (AR + R only) ----
    // AXI4 READ-ONLY. The drafter's SECOND read master (the M9 boot DMA is the
    // first, also read-only). g8_rescore asserts ARVALID ONLY inside its gather
    // window (the S_G8 phase, after the tree walk) -> parks quiescent during
    // compute, so the >5-driver NoC fan-in deadlock class does not apply
    // (memory feedback_axi_driver_fanin_rule: each bundle has ONE driver).
    // When G8_SPLICE=0 these tie off quiescent (no instance drives them).
    // The orchestrator wires this to its OWN HBM NoC aperture (the 788 MB
    // lm_head_full base, distinct from the weight/emb/out apertures) in the BD.
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_g8_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_g8_araddr,
    output wire [7:0]                 m_axi_g8_arlen,
    output wire [2:0]                 m_axi_g8_arsize,
    output wire [1:0]                 m_axi_g8_arburst,
    output wire                       m_axi_g8_arvalid,
    input  wire                       m_axi_g8_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_g8_rid,
    // G8_WIDE through-top: the g8 gather R-data port width FOLLOWS G8_DATA_W
    //   (32 = silicon-proven 32-bit S05_AXI scalar path; 512 = the wide 512b
    //   master, 32 fp16/beat). The BD/NoC must expose a master of matching width
    //   on this bundle (see G8_WIDE_THROUGHTOP.md §unit-PDI). All other g8 AXI
    //   signals (ar*, rid, rresp, rlast, rvalid, rready) are width-invariant.
    input  wire [G8_DATA_W-1:0]       m_axi_g8_rdata,
    input  wire [1:0]                 m_axi_g8_rresp,
    input  wire                       m_axi_g8_rlast,
    input  wire                       m_axi_g8_rvalid,
    output wire                       m_axi_g8_rready,

    // ---- SPEEDWAVE-V2 J5-β m_axi_g8b: stripe ch1 read master ----
    //   Always present (BD drop-in shape); tied off (arvalid=0, rready=0) at
    //   G8_NCH=1 so the default package is byte-identical legacy. At G8_NCH=2
    //   it reads the SAME 0x4180000000 lm-head aperture through S07_AXI/
    //   HBM3_PORT1 (one master per HBM port — fan-in discipline, see
    //   bd_patch/S07_stripe.patch.tcl header).
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_g8b_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_g8b_araddr,
    output wire [7:0]                 m_axi_g8b_arlen,
    output wire [2:0]                 m_axi_g8b_arsize,
    output wire [1:0]                 m_axi_g8b_arburst,
    output wire                       m_axi_g8b_arvalid,
    input  wire                       m_axi_g8b_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_g8b_rid,
    input  wire [G8_DATA_W-1:0]       m_axi_g8b_rdata,
    input  wire [1:0]                 m_axi_g8b_rresp,
    input  wire                       m_axi_g8b_rlast,
    input  wire                       m_axi_g8b_rvalid,
    output wire                       m_axi_g8b_rready,

    // ---- GAP-4 RUNG C: m_axi_emb — child-embedding gather read master ----
    //   READ-ONLY (AR+R), own AXI ID / own HBM port, mimics m_axi_g8's shape.
    //   Reads the TARGET embed_tokens image at hbm_emb_base (ctrl 0x40/0x44),
    //   a SECOND ~751.5 MB HBM aperture DISTINCT from the G8 fp16 lm-head.
    //   Always present as a BD drop-in; TIED OFF (arvalid=0, rready=0) at
    //   EMBG_EN=0 so the default package is byte-identical. 256b R data
    //   (16 bf16/beat = 1 input_emb_buf chunk). A NEW NoC SI port (e.g.
    //   S08_AXI) must expose a read master of this width — the one BD edit a
    //   build needs (see RUNG-C report BUILD-READY block).
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_emb_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_emb_araddr,
    output wire [7:0]                 m_axi_emb_arlen,
    output wire [2:0]                 m_axi_emb_arsize,
    output wire [1:0]                 m_axi_emb_arburst,
    output wire                       m_axi_emb_arvalid,
    input  wire                       m_axi_emb_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_emb_rid,
    input  wire [255:0]               m_axi_emb_rdata,
    input  wire [1:0]                 m_axi_emb_rresp,
    input  wire                       m_axi_emb_rlast,
    input  wire                       m_axi_emb_rvalid,
    output wire                       m_axi_emb_rready,

    // ---- EMB_NCH ch1: m_axi_embb — second child-embedding gather master ----
    //   Always present (BD drop-in shape, the m_axi_g8b idiom); TIED OFF
    //   (arvalid=0, rready=0) at EMB_NCH=1 so the default package is
    //   byte-identical.  At EMB_NCH=2 it reads the SAME embed_tokens
    //   aperture 0x004280000000 through S09_AXI/HBM5_PORT0 — the OTHER NoC
    //   port of the SAME HBM PC, so there is NO duplicate image and NO host
    //   relayout; only the element OFFSET differs (channel 1 owns
    //   [H/2, H)).  One master per HBM port => ARVALID fan-in stays 1 per
    //   port (feedback_axi_driver_fanin_rule).  256b R data, same as ch0.
    //   BD: bd_patch/S09_embb.patch.tcl (requires S07 then S08 applied).
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_embb_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_embb_araddr,
    output wire [7:0]                 m_axi_embb_arlen,
    output wire [2:0]                 m_axi_embb_arsize,
    output wire [1:0]                 m_axi_embb_arburst,
    output wire                       m_axi_embb_arvalid,
    input  wire                       m_axi_embb_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_embb_rid,
    input  wire [255:0]               m_axi_embb_rdata,
    input  wire [1:0]                 m_axi_embb_rresp,
    input  wire                       m_axi_embb_rlast,
    input  wire                       m_axi_embb_rvalid,
    output wire                       m_axi_embb_rready,

    // ---- Status / debug ----
    output wire [3:0]                       stage_sel,
    output wire        [15:0]               array_chunk_idx_out,
    output wire                             array_chunk_idx_valid,
    output wire signed [32*LANES-1:0]       array_acc_out
);

    // ----------------------------------------------------------------
    // GRAND_WIDE through-top consistency guard (elaboration-time).
    //   When G8_WIDE=1, the top-level m_axi_g8_rdata port width (G8_DATA_W) MUST
    //   equal the wide-consume group width (G8_WIDE_P*16) so the 512b beat carries
    //   exactly an integer number of WIDE_P-wide groups (512 = 16*16*2 => 2
    //   groups/beat). A mismatch here is the kind of silent width bug that wedges
    //   the consume FSM in silicon; fail loud at elaboration instead.
    // ----------------------------------------------------------------
    if (G8_WIDE && (G8_DATA_W != G8_WIDE_P*16) && (G8_DATA_W != G8_WIDE_P*16*2))
        $error("G8_WIDE=1 requires G8_DATA_W == G8_WIDE_P*16 (=%0d) or 2x (=%0d); got %0d",
               G8_WIDE_P*16, G8_WIDE_P*16*2, G8_DATA_W);

    // ----------------------------------------------------------------
    // Internal handshakes
    // ----------------------------------------------------------------
    wire start_pulse;
    wire kernel_busy;
    wire kernel_done;

    // ---- PDI#11 PER-STAGE INJECT nets (hoisted: used by stage muxes that appear
    //   textually BEFORE the stage_inject instance below; SV continuous-assign
    //   needs decl-before-use). Tied 0 when STAGE_INJECT=0 -> all muxes pass-through.
    localparam int INJ_NSTAGE = 19;            // shared index space w/ dbg_stage_capture
    // pdi16: widened 128->256 so ONE inject buffer can hold a FULL stage OUTPUT
    //   (the GLM low_rank[256] skip-op: 16 chunks x 16 lanes = 256 words). Still
    //   plain FF (8 Kb), no URAM/BRAM -> URAM 1710 binding wall untouched.
    //   WIDX_W = clog2(256)=8; control_regs passes the full 32b widx so no decode
    //   change. With STAGE_INJECT=0 the whole stage_inject + every mux is pruned.
    localparam int INJ_MAX_W   = 256;          // 256 words (full GLM output / widest stage out)
    wire [INJ_NSTAGE-1:0]   inject_en;         // live per-stage enable bitmask (0x10C)
    wire [INJ_MAX_W*32-1:0] inj_bus;           // flat 32b/word golden inject buffer
    wire        inj_en_we_w, inj_widx_we_w, inj_data_we_w, inj_sel_we_w;
    wire [31:0] inj_en_wdata_w, inj_widx_wdata_w, inj_data_wdata_w, inj_sel_wdata_w;
    wire [31:0] inj_en_rb_w, inj_widx_rb_w, inj_status_w;

    wire g4_start,  g4_done;
    wire g5_start;
    wire g6_start,  g6_done;
    wire g7g_start, g7g_done;
    wire g7u_start, g7u_done;
    wire g7p_start;
    wire g9_start,  g9_done;
    wire g11_start;
    wire glm_start, glm_done;   // FIX-G: g11_lmdown dense GEMV handshake
    wire g12_start, g12_done;
    wire tree_start;
    wire g1_start;

    // PDI #3 integration ladder — replacing tied-1'b1 stage_dones with
    // real layer modules one at a time.

    // ----------------------------------------------------------------
    // L6 + M6 + M6b: tree_decode top-K_F=8 selection over the FULL vocab,
    // fed by a G12-tile-stepping controller (tree_tile_stream).
    //
    // C1 (M6): the real Llama-3.2-3B vocabulary is 128256. TREE_V_MAX is
    // now 128256 and TREE_IDX_W is locked to 18 bits — the index width the
    // host/SGLang token-id ABI uses for the candidate tree (128255 fits in
    // 17b, but the 18b lane is the documented EAGLE4 token-id width and
    // prevents any mod-4096 truncation; we pass IDX_W explicitly so the
    // counter, the 8*IDX_W output bus, and the heap index slots all carry
    // the full 18-bit id). $clog2(128256)=17, so we override the module
    // default to 18 at the instance.
    //
    // C2 (M6b — FULL-VOCAB FIX): the M6 selector read a SINGLE tile of
    // array_acc_out (tree_score = array_acc_out[(idx%LANES)*32 +: 32]); that
    // is single-tile-correct only. The 128-lane GEMV array emits LANES=128
    // logits per run, so V=128256 needs 1002 tiles. tree_tile_stream now
    // steps the G12 GEMV tile-by-tile and streams each tile's 128 logits into
    // tree_decode in lockstep, with vocab index = tile*LANES + lane (full
    // 18-bit). It is the SOLE driver of tree_decode's start/valid/score/
    // index/last AND of the G12 GEMV (re)start + tile_idx + stage override
    // during the tree stage. tree_decode is reused UNCHANGED.
    //
    // Multi-driver-safe: tree_decode's heap is a single always_ff; all of its
    // stream inputs come from exactly ONE block (u_tree_tiler). array_acc_out
    // is read-only. The GEMV array start/tile_idx/stage_sel are muxed so the
    // tiler owns them only while tt_busy is high (the tree stage); top_fsm /
    // the external tile_idx port own them otherwise.
    // ----------------------------------------------------------------
    localparam int TREE_IDX_W  = 18;                 // C1: Llama-3.2-3B token-id width
    localparam int TREE_N_TILES = (TREE_V_MAX + LANES - 1) / LANES;

    wire signed  [8*32-1:0]         tree_topk_scores;
    wire         [8*TREE_IDX_W-1:0] tree_topk_indices;
    wire                            tree_done;

    // SPEEDWAVE-V2 runtime-config registers (declared here, ahead of the tiler
    // generate, `default_nettype none): driven by u_ctrl (0x14C / 0x150).
    //   n_tiles_cfg — V1 FR-Spec variable vocab scan length (0 = full V).
    //   kc_cfg      — K1 runtime G8 rescore shortlist size (0 = K_C=512).
    wire [15:0] n_tiles_cfg;
    wire [15:0] kc_cfg;
    // ---- 0x1125 ACCFIX runtime enables (u_ctrl 0x15C, reset 0 = all OFF) ----
    // Each effective enable is the elaboration parameter AND the runtime bit,
    // so an elab-pruned fix can never be armed and the runtime-off arm is the
    // 0x1124-identical datapath within ONE elaboration (the WSNAP A/B vehicle).
    wire [3:0]  accfix_cfg_w;
    wire        accfix_seam_en   = ACCFIX_SEAM_P   && accfix_cfg_w[0];
    wire        accfix_b2p_en    = ACCFIX_B2P_P    && accfix_cfg_w[1];
    wire        accfix_kc_en     = ACCFIX_KC_P     && accfix_cfg_w[2];
    wire        accfix_kvonly_en = ACCFIX_KVONLY_P && accfix_cfg_w[3];
    // ---- OBS_LVL per-level candidate ring (u_ctrl 0x160/0x164) ----
    wire [10:0] obs_rd_idx_w;
    wire [31:0] obs_rd_data_w;
    wire [31:0] obs_status_word_w;
    // OBS_LVL tap nets — driven inside g_depth_n from the feed_seq->tree_level_fsm
    // candidate seam (READ-ONLY tap; tied 0 at DEPTH_N=0). Entry layout:
    //   word0 = prob fp32 ; word1 = {level[31:28], branch[27:25], 7'b0, tok[17:0]}.
    wire        obs_cand_v_w;
    wire [31:0] obs_cand_prob_w;
    wire [17:0] obs_cand_tok_w;
    wire        obs_tree_arm_w;   // lvl_start pulse = ring write-pointer reset

    // tree_tile_stream <-> tree_decode + GEMV-array stepping wires
    wire                         tt_dec_start;
    wire                         tt_valid_in;
    wire signed [31:0]           tt_score_in;
    wire        [TREE_IDX_W-1:0] tt_index_in;
    wire                         tt_last_in;

    // TREE_P8-RESTORE: G8.1 selection-threshold feedback for the pkpar-arm
    // pre-filter (tree_cheap_serializer). Driven by g_g8_splice/g_g8p (the
    // pipelined g8_rescore_p exports its lagged running-min); tied 0 (filter
    // inert = pass-all) in the bypass / non-pipelined arms.
    wire [31:0] g8_sel_thresh;
    wire        g8_sel_thresh_valid;
    wire                         tt_gemv_start;
    wire        [15:0]           tt_gemv_tile_idx;
    wire                         tt_stage_g12;       // high during the walk
    wire                         tt_busy;

    // array_done forward-declared further down (the GEMV array + capture
    // decode reference it). The tiler consumes it as the tile-valid handshake.
    wire array_done;

    // ----------------------------------------------------------------
    // PDI#5-COMPLETE: tree-decode — generate swap (TREE_P).
    //   TREE_P=1 -> serial tree_tile_stream + tree_decode (closing path).
    //   TREE_P>1 -> #191 P-wide tree_tile_stream_pkpar + tree_merge_pipe
    //     (P logits/cycle; 2.00x@P=8). Pin-compatible at the boundary:
    //     same tree_start / array_acc_out / array_done in, same
    //     gemv_tile_* G12-stepping out, same topk_{scores,indices}/done out.
    //     The P-wide (score,index) stream is INTERNAL to the pkpar pair.
    //     Frontier is INVARIANT in P (TREE_DECODE_FAST §2: 8/8 bit-identical
    //     on real g12 + 100 random vectors). Exactly one arm elaborated.
    // ----------------------------------------------------------------
    generate
    if (TREE_P <= 1) begin : g_tree_serial
        // EXP_G12TREE_PINGPONG: TREE_PP swaps in the 2-entry FF ping-pong tiler
        // (identical ports + stream contract; GEMV overlaps the tree stream).
        if (TREE_PP) begin : g_tiler_pp
        tree_tile_stream_pp #(
            .LANES(LANES), .V_MAX(TREE_V_MAX), .IDX_W(TREE_IDX_W), .SCORE_W(32),
            .N_TILES(TREE_N_TILES)
        ) u_tree_tiler (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .tree_start(tree_start),
            .n_tiles_cfg(n_tiles_cfg),         // SPEEDWAVE-V2 V1 (ctrl 0x14C)
            .array_acc_out(array_acc_out),     // read-only live tile logits
            .array_done(array_done),           // tile-valid handshake
            .gemv_tile_start(tt_gemv_start),
            .gemv_tile_idx(tt_gemv_tile_idx),
            .gemv_stage_g12(tt_stage_g12),
            .tree_dec_start(tt_dec_start),
            .valid_in(tt_valid_in),
            .score_in(tt_score_in),
            .index_in(tt_index_in),
            .last_in(tt_last_in),
            .busy(tt_busy)
        );
        end else begin : g_tiler_base
        tree_tile_stream #(
            .LANES(LANES), .V_MAX(TREE_V_MAX), .IDX_W(TREE_IDX_W), .SCORE_W(32),
            .N_TILES(TREE_N_TILES)
        ) u_tree_tiler (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .tree_start(tree_start),
            .n_tiles_cfg(n_tiles_cfg),         // SPEEDWAVE-V2 V1 (ctrl 0x14C)
            .array_acc_out(array_acc_out),     // read-only live tile logits
            .array_done(array_done),           // tile-valid handshake
            .gemv_tile_start(tt_gemv_start),
            .gemv_tile_idx(tt_gemv_tile_idx),
            .gemv_stage_g12(tt_stage_g12),
            .tree_dec_start(tt_dec_start),
            .valid_in(tt_valid_in),
            .score_in(tt_score_in),
            .index_in(tt_index_in),
            .last_in(tt_last_in),
            .busy(tt_busy)
        );
        end

        tree_decode #(.K_F(8), .V_MAX(TREE_V_MAX), .IDX_W(TREE_IDX_W), .SCORE_W(32)) u_tree (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(tt_dec_start),                 // M6b: heap-clear from the tiler
            .valid_in(tt_valid_in),
            .score_in(tt_score_in),               // M6b: per-tile G12 logit stream
            .index_in(tt_index_in),               // M6b: full 18-bit tile*LANES+lane
            .last_in(tt_last_in),                 // M6b: pulses on global idx V-1
            .topk_scores(tree_topk_scores),
            .topk_indices(tree_topk_indices),
            .done(tree_done)
        );
    end else begin : g_tree_pwide
        // P-wide internal stream between the pkpar tiler and the merge pipe.
        wire                        ttp_valid_in;
        wire signed [TREE_P*32-1:0] ttp_scores_in;
        wire [TREE_P*TREE_IDX_W-1:0] ttp_indices_in;
        wire [TREE_P-1:0]           ttp_lane_en;
        wire                        ttp_last_in;
        wire                        ttp_dec_start;

        // EXP_G12TREE_PINGPONG: TREE_PP swaps in the P-wide ping-pong tiler.
        if (TREE_PP) begin : g_tiler_pp
        tree_tile_stream_pkpar_pp #(
            .LANES(LANES), .P(TREE_P), .V_MAX(TREE_V_MAX),
            .IDX_W(TREE_IDX_W), .SCORE_W(32), .N_TILES(TREE_N_TILES)
        ) u_tree_tiler (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .tree_start(tree_start),
            .n_tiles_cfg(n_tiles_cfg),         // SPEEDWAVE-V2 V1 (ctrl 0x14C)
            .array_acc_out(array_acc_out),
            .array_done(array_done),
            .gemv_tile_start(tt_gemv_start),
            .gemv_tile_idx(tt_gemv_tile_idx),
            .gemv_stage_g12(tt_stage_g12),
            .tree_dec_start(ttp_dec_start),
            .valid_in(ttp_valid_in),
            .scores_in(ttp_scores_in),
            .indices_in(ttp_indices_in),
            .lane_en(ttp_lane_en),
            .last_in(ttp_last_in),
            .busy(tt_busy)
        );
        end else begin : g_tiler_base
        tree_tile_stream_pkpar #(
            .LANES(LANES), .P(TREE_P), .V_MAX(TREE_V_MAX),
            .IDX_W(TREE_IDX_W), .SCORE_W(32), .N_TILES(TREE_N_TILES)
        ) u_tree_tiler (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .tree_start(tree_start),
            .n_tiles_cfg(n_tiles_cfg),         // SPEEDWAVE-V2 V1 (ctrl 0x14C)
            .array_acc_out(array_acc_out),
            .array_done(array_done),
            .gemv_tile_start(tt_gemv_start),
            .gemv_tile_idx(tt_gemv_tile_idx),
            .gemv_stage_g12(tt_stage_g12),
            .tree_dec_start(ttp_dec_start),
            .valid_in(ttp_valid_in),
            .scores_in(ttp_scores_in),
            .indices_in(ttp_indices_in),
            .lane_en(ttp_lane_en),
            .last_in(ttp_last_in),
            .busy(tt_busy)
        );
        end

        tree_merge_pipe #(
            .K_F(8), .P(TREE_P), .V_MAX(TREE_V_MAX),
            .IDX_W(TREE_IDX_W), .SCORE_W(32)
        ) u_tree (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(ttp_dec_start),
            .valid_in(ttp_valid_in),
            .scores_in(ttp_scores_in),
            .indices_in(ttp_indices_in),
            .lane_en(ttp_lane_en),
            .last_in(ttp_last_in),
            .topk_scores(tree_topk_scores),
            .topk_indices(tree_topk_indices),
            .done(tree_done)
        );
        // ------------------------------------------------------------
        // TREE_P8-RESTORE (2026-07-09): re-materialize the SERIAL cheap-logit
        // stream from the P-wide one, so the G8-splice candidate snoop
        // (which consumes tt_valid_in/tt_score_in/tt_index_in/tt_last_in)
        // works UNCHANGED under TREE_P>1. The pkpar tiler emits monotonically
        // increasing indices (lane 0 lowest per group), so the serialized
        // replay is BIT- and ORDER-IDENTICAL to the TREE_P=1 serial stream —
        // G8.1's candidate SET/order and the rescored frontier are invariant
        // by construction. Rate: the P-wide burst drains at 1/cyc under the
        // conservative sel_thresh pre-filter (FIFO DEPTH=8192, overflow
        // canary). When G8_SPLICE=0 the tt_* nets are unread -> synth prunes
        // the serializer. See layers/tree_cheap_serializer.sv. Works with
        // BOTH tiler variants (g_tiler_base pkpar / g_tiler_pp pkpar_pp) —
        // the serializer taps the ttp_* merge-pipe stream, whose content and
        // order are bit-identical across the two tilers (EXP_G12TREE_PINGPONG
        // A/B gate).
        // ------------------------------------------------------------
        if (G8_SPLICE) begin : g_ser
            tree_cheap_serializer #(
                // GAP-5 2026-07-22: DEPTH 4096 -> 8192. Measured peaks vs the
                // old 4096: depth-N real-data 3,599 (1.14x margin), jewel-rate
                // golden 4,089 (7 spare), FR-Spec reorder 5,116 (OVERFLOW —
                // silent survivor corruption on silicon, no $fatal there).
                // See layers/tree_cheap_serializer.sv GAP-5 header note +
                // reports/results/TREE_FSM_GAP5_SERIALIZER_FIFO.md.
                .P(TREE_P), .IDX_W(TREE_IDX_W), .SCORE_W(32), .DEPTH(8192),
                .CHEAP_KEY_INT32(1'b1)     // MUST match the g8 instance key mode
            ) u_cheap_ser (
                .clk(ap_clk), .rst_n(ap_rst_n),
                .flush(tree_start),
                .valid_in(ttp_valid_in),
                .scores_in(ttp_scores_in),
                .indices_in(ttp_indices_in),
                .lane_en(ttp_lane_en),
                .last_in(ttp_last_in),
                .thresh(g8_sel_thresh),
                .thresh_valid(g8_sel_thresh_valid),
                .out_valid(tt_valid_in),
                .out_score(tt_score_in),
                .out_index(tt_index_in),
                .out_last(tt_last_in)
            );
        end else begin : g_ser_off
            // no G8 snoop to feed — keep the serial wires quiet (s7c behaviour)
            assign tt_valid_in = 1'b0;
            assign tt_score_in = '0;
            assign tt_index_in = '0;
            assign tt_last_in  = 1'b0;
        end
        // serial-only tree_decode start unused in this arm
        assign tt_dec_start = 1'b0;
    end
    endgenerate

    // ----------------------------------------------------------------
    // OUTPUT-DMA (SPEC_OUTPUT_DMA.md): on tree_done, burst the K_F=8
    // (score,index) pairs to HBM at out0_base so the host can read the
    // drafter's output C2H. THE SOLE AXI write master in the drafter.
    //
    // HBM ABI (canonical — matches scripts/verify_drafter_silicon_goldens.py
    // silicon_read_tree_topk):
    //   out0_base + 0   : K_F int32 INDICES (token-ids, zero-ext)
    //   out0_base + 32  : K_F int32 SCORES  (sign-ext)        [one 16-beat burst]
    //   out1_base + 0   : K_F int32 INDICES mirror             [one 8-beat burst]
    //
    // Multi-driver-safe: the writer latches a snapshot of tree_topk_* at
    // tree_done; the heap stays read-only. AWVALID is asserted ONLY for the
    // two transient bursts, then parks silent — no compute-time concurrency,
    // so the >5-driver NoC fan-in deadlock class does not apply.
    // ----------------------------------------------------------------
    wire out_dma_busy;
    wire out_dma_done;
    wire out_dma_err;   // C8 (B5): sticky OUT-DMA stall-timeout abort -> 0x78 bit2

    // ----------------------------------------------------------------
    // G8 SPLICE: the output writer's trigger + top-K source are MUXED.
    //   G8_SPLICE=0 : trigger=tree_done, frontier=tree top-K_F=8 (s7c-exact).
    //   G8_SPLICE=1 : trigger=g8_done,   frontier=g8 RESCORED top-8.
    // These wires are forward-declared here (the writer is instantiated before
    // the G8 block) and assigned in the G8 SPLICE block after the tree region.
    // ----------------------------------------------------------------
    wire                            ow_trigger;
    wire signed  [8*32-1:0]         ow_topk_scores;
    wire         [8*TREE_IDX_W-1:0] ow_topk_indices;

    // ---- PROBE V3 dump-region top wiring (out3 candidate set / out4 tree heap) ----
    wire        dump_en;        // 0xC0 bit0 (control_regs) -> writer dump gate
    wire [63:0] out3_base;      // 0xC4/0xC8
    wire [63:0] out4_base;      // 0xCC/0xD0
    // PROBE V3 out3 candidate-set dump read port (driven by output_dma_writer).
    localparam int G8_CIDX_W = (G8_K_C <= 1) ? 1 : $clog2(G8_K_C);   // 9 @ K_C=512
    wire [G8_CIDX_W-1:0] g8_cand_rdaddr;       // candidate slot select (= addr>>1)
    wire [TREE_IDX_W-1:0] g8_cand_dump_id;     // candidate vocab id (== G8_IDX_W)
    wire [31:0]          g8_cand_dump_score;   // candidate fp32 score (ASYNC debug tap)
    // SEAM-FF: g8's REGISTERED (FDRE) rescored[] read for the softmax-bridge logit
    // feed. Driven by the g8 instance; the async g8_cand_dump_score is used only by
    // the out3 debug dump. TREE_SEAM_FF selects which one feeds the bridge.
    wire [31:0]          g8_bridge_score_ff;   // registered rescored[] @ lookahead
    localparam int OW_DUMP3_AW = $clog2(G8_K_C*2);     // 10 @ K_C=512 (2 words/cand)
    wire [OW_DUMP3_AW-1:0]   ow_dump3_rdaddr;          // writer -> candidate buffer
    wire [31:0]              ow_dump3_rddata;          // candidate word back to writer
    wire [8*32-1:0]          ow_tree_parents;          // K_F parent indices (INT32)
    wire [8*32-1:0]          ow_g8_rescored;           // K_F G8 rescored fp32
    // out3 word map: word 2*c = id[c] (zero-ext), word 2*c+1 = score[c].
    //   c = rdaddr>>1; the low bit selects id vs score. g8_cand_rdaddr drives the
    //   g8_rescore candidate read port; the data returns combinationally.
    //   NOTE (stage-4, verified): g8_cand_dump_score == g8's rescored[] slot (the
    //   FINAL fp16-dot fp32 RESCORE score, NOT the cheap G12 score -- the port's
    //   `_cheap_` name is a misnomer; rescored[] is what dbg_cand_cheap_score reads
    //   in g8_rescore_p.sv:1131). That is exactly the logit the softmax bridge
    //   needs (ref/export softmax over z["G8"] == rescored).
    //
    // STAGE-4 ARBITER MUX: during the depth-N level loop the tree_softmax_bridge
    // owns this candidate read port (bridge_cand_rd_active) to sweep the 512
    // rescored logits; otherwise the output-DMA dump walk owns it (post-all-done,
    // as before). At DEPTH_N=0 bridge_cand_rd_active is tied 0 -> BIT-IDENTICAL
    // to the pre-stage-4 dump-only path.
    wire [G8_CIDX_W-1:0] bridge_cand_rdaddr;     // bridge's 512-sweep read index
    wire                 bridge_cand_rd_active;  // 1 while bridge owns the port
    // SEAM-FF: bridge's LOOKAHEAD (next) read index -> g8's registered rescored[]
    // read; and the strap-selected logit source the bridge actually consumes.
    wire [G8_CIDX_W-1:0] bridge_cand_rdaddr_nxt;
    wire [31:0]          bridge_cand_logit = TREE_SEAM_FF ? g8_bridge_score_ff
                                                          : g8_cand_dump_score;
`ifdef TREE_ARB_RED
    // RED (b): ARBITER SELECTS THE WRONG SOURCE. Hand the bridge the dump-walk
    // address (stale/0 during the level loop) instead of its own sweep index, so
    // the softmax reads the wrong rescored slots and the tree must diverge.
    assign g8_cand_rdaddr  = bridge_cand_rd_active ? ow_dump3_rdaddr[OW_DUMP3_AW-1:1]
                                                   : ow_dump3_rdaddr[OW_DUMP3_AW-1:1];
`else
    assign g8_cand_rdaddr  = bridge_cand_rd_active ? bridge_cand_rdaddr
                                                   : ow_dump3_rdaddr[OW_DUMP3_AW-1:1];
`endif
    assign ow_dump3_rddata = ow_dump3_rdaddr[0]
                           ? g8_cand_dump_score
                           : {{(32-TREE_IDX_W){1'b0}}, g8_cand_dump_id};
    // out4 tree-heap extras: parents = root sentinel (-1) for the depth-0 pilot;
    // g8 rescored = the frontier scores (= ow_topk_scores under G8 splice).
    genvar owp;
    for (owp = 0; owp < 8; owp = owp + 1) begin : g_ow_parents
        assign ow_tree_parents[owp*32 +: 32] = 32'hFFFF_FFFF;   // -1 (no-parent)
        assign ow_g8_rescored [owp*32 +: 32] = ow_topk_scores[owp*32 +: 32];
    end

    // ================================================================
    // GAP5: the physical OUT-DMA AXI write port is generate-SELECTED between
    // two write masters:
    //   * DEPTH_N=0 : the shipped depth-0 FRONTIER writer (output_dma_writer),
    //                 driving fr_* -> m_axi_out. BYTE-IDENTICAL to today.
    //   * DEPTH_N!=0: the depth-N TREE writer (tree_dma_writer), driving tr_* ->
    //                 m_axi_out, bursting the WHOLE tree (4-tensor verify ABI).
    // DEPTH_N is a BUILD-TIME param, so exactly ONE master drives m_axi_out
    // (never two concurrent AWVALID drivers — feedback_axi_driver_fanin_rule).
    // At DEPTH_N=0 the tr_* branch + tree writer do NOT elaborate, so the
    // frontier path is unchanged (see the generate mux below).
    // ================================================================
    wire fr_busy, fr_done, fr_err;
    wire [BOOT_AXI_ID_W-1:0]     fr_awid;    wire [BOOT_AXI_ADDR_W-1:0]  fr_awaddr;
    wire [7:0] fr_awlen;  wire [2:0] fr_awsize;  wire [1:0] fr_awburst;  wire fr_awvalid;
    wire [BOOT_AXI_DATA_W-1:0]   fr_wdata;   wire [BOOT_AXI_DATA_W/8-1:0] fr_wstrb;
    wire fr_wlast;  wire fr_wvalid;  wire fr_bready;
    // depth-N tree writer drivers + status (undriven/unused at DEPTH_N=0).
    wire tr_busy, tr_done, tr_err;
    wire [BOOT_AXI_ID_W-1:0]     tr_awid;    wire [BOOT_AXI_ADDR_W-1:0]  tr_awaddr;
    wire [7:0] tr_awlen;  wire [2:0] tr_awsize;  wire [1:0] tr_awburst;  wire tr_awvalid;
    wire [BOOT_AXI_DATA_W-1:0]   tr_wdata;   wire [BOOT_AXI_DATA_W/8-1:0] tr_wstrb;
    wire tr_wlast;  wire tr_wvalid;  wire tr_bready;

    output_dma_writer #(
        .K_F(8), .IDX_W(TREE_IDX_W), .SCORE_W(32),
        .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(BOOT_AXI_DATA_W),
        .AXI_ID_W(BOOT_AXI_ID_W), .EN_OUT1_MIRROR(1'b1),
        // v24 SGLang host ABI: ids@out0, scores@out1, parents@out2; 8 live +
        // (TOP_K-8) zero-pad; parents = root sentinel. (OUT_ABI_V24=0 falls
        // back to the M7 idx+score / idx-mirror layout.)
        .ABI_V24(OUT_ABI_V24), .TOP_K(OUT_TOP_K),
        .PARENT_SENTINEL(OUT_PARENT_SENTINEL),
        // PROBE V3: enable out3/out4 dump regions (gated at runtime by 0xC0 bit0).
        .EN_DUMP(1'b1), .CAND_N(G8_K_C)
    ) u_out_dma (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .tree_done(ow_trigger),            // G8 SPLICE: tree_done OR g8_done
        .topk_scores(ow_topk_scores),      // G8 SPLICE: tree OR g8 rescored
        .topk_indices(ow_topk_indices),
        .out0_base(out0_base),
        .out1_base(out1_base),
        .out2_base(out2_base),
        // ---- PROBE V3 dump regions (out3 candidate set, out4 tree heap) ----
        .dump_en(dump_en),
        .out3_base(out3_base), .out4_base(out4_base),
        .dump3_rdaddr(ow_dump3_rdaddr), .dump3_rddata(ow_dump3_rddata),
        .tree_ids(ow_topk_indices), .tree_scores(tree_topk_scores),
        .tree_parents(ow_tree_parents), .g8_rescored(ow_g8_rescored),
        // GAP5: frontier writer now drives fr_* (muxed to m_axi_out below).
        .out_busy(fr_busy),
        .out_done(fr_done),
        .out_err(fr_err),                    // C8 (B5): sticky stall-timeout abort
        .m_axi_out_awid(fr_awid),            .m_axi_out_awaddr(fr_awaddr),
        .m_axi_out_awlen(fr_awlen),          .m_axi_out_awsize(fr_awsize),
        .m_axi_out_awburst(fr_awburst),      .m_axi_out_awvalid(fr_awvalid),
        .m_axi_out_awready(m_axi_out_awready),
        .m_axi_out_wdata(fr_wdata),          .m_axi_out_wstrb(fr_wstrb),
        .m_axi_out_wlast(fr_wlast),          .m_axi_out_wvalid(fr_wvalid),
        .m_axi_out_wready(m_axi_out_wready),
        .m_axi_out_bid(m_axi_out_bid),       .m_axi_out_bresp(m_axi_out_bresp),
        .m_axi_out_bvalid(m_axi_out_bvalid), .m_axi_out_bready(fr_bready)
    );

    // ---- GAP5: OUT-DMA AXI write-port + status generate mux ----
    generate if (DEPTH_N) begin : g_out_axi_tree
        // depth-N: the TREE writer (tr_*) owns the physical write port.
        assign m_axi_out_awid    = tr_awid;    assign m_axi_out_awaddr  = tr_awaddr;
        assign m_axi_out_awlen   = tr_awlen;   assign m_axi_out_awsize  = tr_awsize;
        assign m_axi_out_awburst = tr_awburst; assign m_axi_out_awvalid = tr_awvalid;
        assign m_axi_out_wdata   = tr_wdata;   assign m_axi_out_wstrb   = tr_wstrb;
        assign m_axi_out_wlast   = tr_wlast;   assign m_axi_out_wvalid  = tr_wvalid;
        assign m_axi_out_bready  = tr_bready;
        // status 0x78 = OR of both masters (frontier's fr_* AXI is dangling here).
        assign out_dma_busy = fr_busy | tr_busy;
        assign out_dma_done = fr_done | tr_done;
        assign out_dma_err  = fr_err  | tr_err;
    end else begin : g_out_axi_frontier
        // DEPTH_N=0: BYTE-IDENTICAL frontier path (pure net pass-through).
        assign m_axi_out_awid    = fr_awid;    assign m_axi_out_awaddr  = fr_awaddr;
        assign m_axi_out_awlen   = fr_awlen;   assign m_axi_out_awsize  = fr_awsize;
        assign m_axi_out_awburst = fr_awburst; assign m_axi_out_awvalid = fr_awvalid;
        assign m_axi_out_wdata   = fr_wdata;   assign m_axi_out_wstrb   = fr_wstrb;
        assign m_axi_out_wlast   = fr_wlast;   assign m_axi_out_wvalid  = fr_wvalid;
        assign m_axi_out_bready  = fr_bready;
        assign out_dma_busy = fr_busy;
        assign out_dma_done = fr_done;
        assign out_dma_err  = fr_err;
    end endgenerate

    // ----------------------------------------------------------------
    // L3: rmsnorm_block instances drive g1_done and g11_done.
    // Two independent instances — input RMSNorm (g1) and post-attn
    // RMSNorm (g11). Each sequences sumsq → scale_int → apply with a
    // ~447-cycle deterministic latency at H=3072, K_PAR=16.
    //
    // For the integration-skeleton PDI, x_chunk / gamma_chunk are tied
    // to 0 at this site (real values come from upstream activation
    // routing in a later ladder rung). The internal FSM still walks the
    // full latency so kernel_done timing matches a real RMSNorm.
    // ----------------------------------------------------------------
    // ================================================================
    // M4 (2026-05-23): REAL input-embedding feed into G1 RMSNorm.
    //
    // Before M4 the G1 RMSNorm input was hard-tied `.x_chunk('0)`, so G1 (and
    // every downstream stage) ran on zeros and the bisect harness reported G1
    // first-divergent ("got 0 exp 1408"). M4 wires a real, chunk-indexed
    // input-embedding buffer into G1.
    //
    // Buffer + fill design
    // --------------------
    //   * input_emb_buf : H/K_PAR = 192 chunks of K_PAR=16 INT16 (Q4.12), the
    //     layer input embedding. Written by exactly ONE always_ff (the
    //     emb-fill writer below) → multi-driver-safe. Read combinationally by
    //     the chunk-indexed selector that feeds the rmsnorm FSM in lockstep.
    //   * gamma_emb_buf : matching 192 chunks of RMSNorm gamma (Q4.12). For
    //     the pilot golden gamma == 1.0 == 16'h1000 for every element.
    //   * SILICON fill path: the buffer write port (emb_we / emb_wr_chunk /
    //     emb_wr_data) is driven by an m_axi input-embedding DMA modeled on
    //     weight_dma_loader.sv (m_axi read from hbm_emb_base → this buffer,
    //     single ARVALID driver, parks idle during compute). That DMA master
    //     is NOT instantiated in this pilot (the bisect gate needs no extra
    //     m_axi driver and the AXI-fan-in rule forbids a second one here); the
    //     write port is left exposed for the orchestrator to attach it in the
    //     PDI integration rung. The xsim PRELOAD below maps 1:1 onto that DMA
    //     fill: the same {chunk,data} stream the DMA would produce.
    //   * xsim PRELOAD: the bisect TB drops the golden input at
    //     +EMB_HEX=<path>; under simulation we $readmemh it into emb_init and
    //     the fill writer copies it in at reset. The pilot uses D_HEAD=16 = one
    //     real chunk; the rest of the 192 chunks are zero.
    //
    // Off-by-one note (proven by $display trace, not assumed): rmsnorm_block's
    // sumsq_en/apply_en are registered one cycle after the FSM enters S_SUMSQ/
    // S_APPLY, so the chunk consumed while en is high is x_req_chunk starting
    // at 1, i.e. the chunk at index 0 of the en window is x_req_chunk==1. The
    // single real pilot chunk is therefore placed at buffer slot EMB_DATA_SLOT
    // (=1) so it is summed/applied exactly once. (Full-H production loads all
    // 192 slots; the harness pilot loads one.)
    // ================================================================
    localparam int EMB_H        = 3072;
    localparam int EMB_K_PAR    = 16;
    localparam int EMB_AW       = 16;
    localparam int EMB_NCHUNKS  = EMB_H / EMB_K_PAR;            // 192
    localparam int EMB_CW       = $clog2(EMB_NCHUNKS + 1);
    localparam int EMB_DATA_SLOT = 1;  // see off-by-one note above
    localparam logic signed [15:0] GAMMA_ONE_Q12 = 16'sh1000;  // 1.0 in Q4.12

    // =====================================================================
    // MAXPROBE liveness build-id (MAXPROBE_INSTRUMENTATION_SPEC §4.1). Bumped
    // each PDI; the host reads 0x100 FIRST and asserts {0xE4B7, BUILD_TAG,
    // PROBE_REV}; a mismatch => the probe AXI window is dead/mis-decoded -> STOP
    // before trusting any probe. Pure-constant hardwired assign (no reg, no load).
    // =====================================================================
    localparam logic [7:0] BUILD_TAG = 8'h11;   // PDI#11 max-probe
    localparam logic [7:0] PROBE_REV = 8'h43;  // UNIFIED build 0xE4B71143 (pipeline-staged; fork source stays 8'h15)  // SPEEDWAVE-V2 LEVERS (build-id 0xE4B71115): routable-fallback monolith =
                                               // SPEEDWAVE 0xE4B71111 + floorplan-independent jewel-week levers: K1 kc_cfg
                                               // (ctrl 0x150) + V1 n_tiles_cfg (ctrl 0x14C) + F2 requant-behind-headmux +
                                               // J5-β G8_NCH=2 stripe (m_axi_g8b/S07) + W3-C1-P1 wr-pos replication @80 MHz.
                                               // Was 8'h11: SPEEDWAVE (build-id 0xE4B71111): consolidated speed-wave PDI on the
                                               // RECLOCK100 66MHz merged base (SELPIPE + kvrepl + FASTIN + B=1):
                                               // TREE_P=8 + TREE_PP=1 (pkpar ping-pong walk + tree_cheap_serializer
                                               // under G8_SPLICE=1, EXP_G12TREE_PINGPONG + TREE_P8_FOLD) +
                                               // FIX-SBUF (attention_full_mha_sbuf score_buf, EXP_G5_CTXSCAN 1.609x G5) +
                                               // G8_ROWPP=1 + G8_OUTSTANDING=4 (EXP_G8_PINGPONG_II1, ~196 cyc/row).
                                               // NOTE: 0x11 was used FORK-LOCALLY by verilog/forks/tree_p8 (never
                                               // built into a PDI); speedwave SUBSUMES that fold — tree_p8 is
                                               // superseded and must not be built. Was 8'h10: RECLOCK100+FASTIN (build-id 0xE4B71110): 100 MHz reclock of the
                                               // ship WIDE lineage on the 0xE4B7110F-validated base —
                                               // BODY_BATCH_B=1 (speed line, batch off) +
                                               // G8_SELPIPE=1 registered-cmin OPT1F selector retiming (kills the
                                               // seg_min->seg_pos 63-level wall: full-top OOC @10ns -2.935 ->
                                               // +0.087 MET, sel_pipe MANIFEST) + kvrepl_port kv_rd_pos per-stack
                                               // replication (retires the KV_VADDR phys_opt rescue cone) +
                                               // FASTIN boot-DMA route into input_emb_buf/h_fc_buf (0x54
                                               // bits[2]/[3] emb/hfc target-selects, flat walk n_cols=1/cpc=1536/
                                               // stage_base=0; kills the 2.77ms/call 6,144-AXI-Lite-poke emb+h_fc
                                               // load, RTT_MEASUREMENTS.md; trio path kept intact as fallback;
                                               // gates FASTIN_RESULT.md re-run on THIS merged source).
                                               // Was 8'h0F: TREE-BATCH B=2 (build-id 0xE4B7110F): BODY_BATCH_B=2 sibling-batch on the shipping WIDE array (keeps WIDE-256 G8 + consumer-FSM fix + M7 correctness of 0xE4B7110E) + host BATCH-TEST reg window 0x138(ctrl)/0x13c/0x140(b0 lo/hi)/0x144/0x148(sib lo/hi) proving the B=2 weight-reuse datapath on silicon (inert in normal forward). Was 8'h0E: WIDE-FIX variant (build-id 0xE4B7110E): consumer-FSM rready deassert fix (g8_rescore_p wb_left=wrc_beats-acc_now, RED/GREEN proven 07-08 WIDE_CONSUMER_FSM_HUNT.md) on the 0xE4B7110B WIDE base. Was:   // probe-rev 11 (pdi25 G8WIDE SPEED: build-id 0xE4B7110B; 256b WIDE m_axi_g8 gather (G8_WIDE=1, G8_DATA_W=256, P=16 II=1 wide dot) + the F1 4KB page-split in the WIDE AR lane (the pdi18 0-ARREADY root cause) + fixC G8_CONSUME_HARDEN strap (inert at G8_WIDE=1; hardened scalar consume available at G8_WIDE=0). Base = pdi23 OPT1F, prior rev 7: G8_OPT1G=0 + G8_OPT1F=1 = non-blocking running-min select with REGISTERED pipelined 2-level min, replacing the placement-fragile OPT1G systolic broadcast sorter (G8RESC_BUILDINVARIANT_RCA). Adds DETERMINISTIC g8 frontier/cand-set dump regs (0x130 seek / 0x134 readback; FREE addrs, NOT the 0x110/0x114 stage-inject regs) replacing the racy cap[18] cap-CRC. Otherwise == pdi21: scalar 32b gather G8_WIDE=0/G8_DATA_W=32, G8_OUTSTANDING=2, G8_SCALAR_II1=1, SEAM_FF=1)

    // =====================================================================
    // M7 GLM=0 WIRED-TAP LOCALIZER (2026-06-20, GLM_TAP_LOCALIZER_BUILD).
    // DBG_LMTAP=1 in this build: tap the lmdown activation-delivery cone at the
    // first GLM S_RUN cycle, lane 0 -> 3 AXI-Lite readback regs (0x120/0x124/0x128).
    //   (1) seam_r8 BRAM DOUT lane0 (pre-guard)  = u_seam_r8.dbg_rd_chunk_q0
    //   (2) rd_guard_q (RD_LAT guard)            = u_seam_r8.dbg_rd_guard_q
    //   (3) a_d lane0 (MUL A operand, captured)  = u_lmdown.dbg_a_d0
    // Decision rule (Hunter E §5 / Hunter F §5): seam DOUT==0 w/ G11 golden =>
    // seam read net dead; seam DOUT!=0 but a_d==0 => the a_chunk->a_d crossing dead.
    // The taps are READ-ONLY continuous reads off existing nets (drive nothing back)
    // -> golden frontier BIT-IDENTICAL (feedback_no_accuracy_tradeoff). DBG_LMTAP=0
    // => all tap ports tie 0 + DCE -> byte-identical to the no-tap netlist.
    // pdi16: DBG_LMTAP=0 — the GLM wired-taps REGRESSED placement (tap fanout
    //   evacuated SLR1 -> re-triggered the G4q Problem-A margin loss, FWD_03 §note).
    //   The overnight build must NOT carry them. =0 => all tap ports tie 0 + DCE ->
    //   byte-identical to the no-tap netlist (feedback_no_accuracy_tradeoff). The
    //   skip-op golden-ingest (stage_inject 0x10C..0x11C) REPLACES the diagnostic
    //   role of the taps for the GLM-isolation use case.
    localparam bit DBG_LMTAP = 1'b1;   // pdi18: re-enable on-card 3-node seam/a_d read (coloc pblock buys back the room; per GLM_FLEET_DECISION §4)

    // =================================================================
    // BUFSPLIT (2026-06-11, BUFSPLIT_FIX): the six wide 256-bit boot
    // buffers (h_fc_buf / input_emb_buf / 4 gammas) are split into 8
    // SEPARATELY-ADDRESSED 32-bit banks each, so synthesis cannot mis-bias
    // the chunk address of a wide partial-write/full-read BRAM (the M7 +128
    // chunk-rotation class). Each 256b chunk is stored across 8 banks of
    // 32b x 192. Boot writes hit exactly ONE bank (the 32b word index
    // word=ptr&7) at chunk=ptr>>3 — no partial-bit-select into a wide word.
    // Reads concatenate the 8 banks at the same chunk index back into the
    // 256b chunk via the `_rd(idx)` function — so EVERY downstream read site
    // stays byte-identical (BUF[idx] -> BUF_rd(idx)).
    //
    //   bank b (b=0..7) holds bits [b*32 +: 32] of the 256b chunk, i.e. the
    //   32b word that the host streams at flat-word-addr (chunk*8 + b).
    //   That 32b word packs lanes (2*b) and (2*b+1) (16b each, Q4.12).
    //
    // BUFSPLIT_DECL declares the 8 ram_style=block banks; BUFSPLIT_RDFN
    // declares the chunk-reconstruction function. EMB_NBANKS == 8.
    // =================================================================
    localparam int EMB_NBANKS = (EMB_K_PAR*EMB_AW) / 32;   // 256/32 = 8 banks
    `define BUFSPLIT_DECL(NAME) \
        (* ram_style = "block" *) reg [31:0] NAME``_b0 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b1 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b2 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b3 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b4 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b5 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b6 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b7 [0:EMB_NCHUNKS-1];
    // reconstruct the 256b chunk from the 8 banks (bank7 = MSBs .. bank0 = LSBs)
    `define BUFSPLIT_RDFN(NAME) \
        function automatic signed [EMB_K_PAR*EMB_AW-1:0] NAME``_rd(input [EMB_CW-1:0] idx); \
            NAME``_rd = $signed({ NAME``_b7[idx], NAME``_b6[idx], NAME``_b5[idx], NAME``_b4[idx], \
                                  NAME``_b3[idx], NAME``_b2[idx], NAME``_b1[idx], NAME``_b0[idx] }); \
        endfunction
    // per-word boot write: data32 lands in bank `word` (0..7) at `chunk`. One
    // bank fires per write -> no partial-bit-select into a wide word.
    `define BUFSPLIT_WR(NAME, WE, CHUNK, WORD, D32) \
        if ((WE) && (WORD)==3'd0) NAME``_b0[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd1) NAME``_b1[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd2) NAME``_b2[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd3) NAME``_b3[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd4) NAME``_b4[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd5) NAME``_b5[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd6) NAME``_b6[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd7) NAME``_b7[CHUNK] <= (D32);

    // ---- input-embedding buffer + gamma buffer (one always_ff writer) ----
    // BUFSPLIT: input_emb_buf (192 chunks x 256b) -> 8 x (32b x 192) banks.
    `BUFSPLIT_DECL(input_emb_buf)
    `BUFSPLIT_RDFN(input_emb_buf)
    // BUFSPLIT: gamma_emb_buf (hidden_norm gamma) -> 8 banks.
    `BUFSPLIT_DECL(gamma_emb_buf)
    `BUFSPLIT_RDFN(gamma_emb_buf)
    // FIX-E (OPTION A): a NEW sibling buffer for the host-precomputed h_fc =
    // fc(hidden_cat3) (3072 elems, Q4.12). SAME geometry as input_emb_buf
    // (192 chunks x 256b). Read by G1 h_norm (frac12) AND the R3
    // post-attn residual a-arm (frac5 via arith >>>7). The embedding stays in
    // input_emb_buf and feeds ONLY G1 e_norm. Loaded via the new boot_is_hfc
    // (0x24) target-select through the same M1c 0x4c/0x50 stream.
    // BUFSPLIT: h_fc_buf -> 8 banks.
    `BUFSPLIT_DECL(h_fc_buf)
    `BUFSPLIT_RDFN(h_fc_buf)
    // S7d DUAL-INPUT: 2nd gamma image for the e_norm / input_layernorm path
    // (the NEW u_rmsnorm_g1_e below). The EXISTING u_rmsnorm_g1 keeps gamma_emb_buf
    // (h_norm / HIGH half); the e-path reads input_emb_buf (same embed input) with
    // THIS distinct gamma. xsim preload via +GAMMA_E_HEX (else default 1.0).
    // BUFSPLIT: gamma_e_buf -> 8 banks.
    `BUFSPLIT_DECL(gamma_e_buf)
    `BUFSPLIT_RDFN(gamma_e_buf)
    // FIX-D: gamma image for the post-attention RMSNorm (post_attn_norm_g).
    // Consumed by u_rmsnorm_premlp (added in the residual block) which norms the
    // R3 residual h1 -> pre_mlp (frac=12). xsim preload via +GAMMA_POSTATTN_HEX
    // (else default 1.0). Q4.12, 3072 words, same packing as gamma_emb_buf.
    // BUFSPLIT: gamma_postattn_buf -> 8 banks.
    `BUFSPLIT_DECL(gamma_postattn_buf)
    `BUFSPLIT_RDFN(gamma_postattn_buf)
    // FIX-F: gamma image for the FINAL RMSNorm (final_norm_g, Q4.12 absmax 1.430).
    // Consumed by u_rmsnorm_g11 (was hardcoded gamma=1.0). xsim preload via
    // +GAMMA_FINAL_HEX (else default 1.0). Same packing as gamma_emb_buf.
    // BUFSPLIT: gamma_final_buf -> 8 banks.
    `BUFSPLIT_DECL(gamma_final_buf)
    `BUFSPLIT_RDFN(gamma_final_buf)

    // C5 (B1) input-embedding write port. The host streams the input embedding
    // through the SAME M1c 0x48/0x4c/0x50 trio as weights, with boot_is_emb=1
    // (0x7c bit0) routing the 32-bit words here instead of URAM. A full chunk is
    // EMB_K_PAR*EMB_AW = 256 bits = EMB_WORDS_PER_CHUNK=8 words of 32 bits; the
    // flat 32b-word address selects the chunk (>>3) and the word-in-chunk (low 3).
    // Driven by ctrl_boot_we_emb below (decl-before-use: the source nets are the
    // ctrl_boot_* wires declared near the boot mux). The xsim preload (+EMB_FULL_HEX
    // / +EMB_HEX) still seeds the array directly for functional sims.
    localparam int EMB_WORDS_PER_CHUNK = (EMB_K_PAR*EMB_AW) / 32;   // 256/32 = 8
    localparam int EMB_WPC_LOG2        = $clog2(EMB_WORDS_PER_CHUNK); // 3
    wire                              emb_we;
    wire [16:0]                       emb_word_addr;   // flat 32b-word addr (M1c ptr)
    wire [EMB_CW-1:0]                 emb_wr_chunk;    // = emb_word_addr >> 3
    wire [EMB_WPC_LOG2-1:0]           emb_wr_word;     // = emb_word_addr[2:0]
    wire [31:0]                       emb_wr_data32;
    // FIX-E: h_fc write-port nets (mirror the emb_* port; driven by the new
    // ctrl_boot_we_hfc 3-way boot-we split near the boot mux).
    wire                              hfc_we;
    wire [16:0]                       hfc_word_addr;   // flat 32b-word addr (M1c ptr)
    wire [EMB_CW-1:0]                 hfc_wr_chunk;    // = hfc_word_addr >> 3
    wire [EMB_WPC_LOG2-1:0]           hfc_wr_word;     // = hfc_word_addr[2:0]
    wire [31:0]                       hfc_wr_data32;
    // GAMMA-LOAD: gamma write-port nets (mirror the emb_* / hfc_* ports; driven
    // by the ctrl_boot_we_gamma split near the boot mux). Forward-declared here
    // so the gamma always_ff writer (below the buffer decls) sees declared nets
    // (xvlog VRFC 10-3380 requires declaration-before-use even procedurally).
    wire                              gamma_we_hidden, gamma_we_enorm,
                                      gamma_we_postat, gamma_we_final;
    wire [16:0]                       gamma_word_addr; // flat 32b-word addr (M1c ptr)
    wire [EMB_CW-1:0]                 gamma_wr_chunk;  // = gamma_word_addr >> 3
    wire [EMB_WPC_LOG2-1:0]           gamma_wr_word;   // = gamma_word_addr[2:0]
    wire [31:0]                       gamma_wr_data32;

    // xsim preload memory (loaded from +EMB_HEX; synthesis ignores the initial)
    //
    // OUTDMA NOTE (2026-05-23): this entire xsim-only preload block is wrapped
    // in the simulation-only guarded region so the integrated top is SYNTHESIZABLE
    // (the `string emb_hex_path` decl is not a synthesizable type — it tripped
    // `[Synth 8-27] string type not supported` and blocked the OUTDMA top-region
    // ooc-synth gate). The pragma is a comment: xsim still runs the preload
    // unchanged (M4 behavior identical in simulation); only synthesis skips it,
    // which is exactly the documented intent ("synthesis ignores the initial").
    // No M4 logic is altered — this is a synth-visibility guard only.
    // synthesis translate_off
    // BUFSPLIT sim-only helpers: the preload paths below address the buffers
    // as flat 16-bit lanes (element e -> chunk e/16, lane e%16). With the bank
    // split, lane (e%16) lives in bank (e%16)/2 at half (e%16)%2. These macros
    // zero a whole chunk / write one 16b lane across the 8 banks, so the sim
    // preload populates the SAME bits the silicon boot writer (per-32b-word)
    // would. Sim-only (inside translate_off); synth never sees them.
    `define BUFSPLIT_ZERO(NAME, IDX) begin \
        NAME``_b0[IDX]='0; NAME``_b1[IDX]='0; NAME``_b2[IDX]='0; NAME``_b3[IDX]='0; \
        NAME``_b4[IDX]='0; NAME``_b5[IDX]='0; NAME``_b6[IDX]='0; NAME``_b7[IDX]='0; end
    `define BUFSPLIT_W16(NAME, EE, W16) begin \
        case (((EE)%EMB_K_PAR)/2) \
          0: NAME``_b0[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          1: NAME``_b1[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          2: NAME``_b2[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          3: NAME``_b3[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          4: NAME``_b4[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          5: NAME``_b5[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          6: NAME``_b6[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          7: NAME``_b7[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
        endcase end
    reg signed [EMB_K_PAR*EMB_AW-1:0] emb_init [0:EMB_NCHUNKS-1];
    integer emb_i;
    string  emb_hex_path;
    initial begin : g1_emb_preload
        for (emb_i = 0; emb_i < EMB_NCHUNKS; emb_i = emb_i + 1) begin
            `BUFSPLIT_ZERO(input_emb_buf, emb_i)
            `BUFSPLIT_ZERO(h_fc_buf, emb_i)        // FIX-E: default h_fc (smoke runs no preload, like input_emb_buf)
            `BUFSPLIT_ZERO(gamma_emb_buf, emb_i)
            `BUFSPLIT_ZERO(gamma_e_buf, emb_i)     // S7d: e_norm gamma (init 0; loaded below)
            `BUFSPLIT_ZERO(gamma_postattn_buf, emb_i) // FIX-D: post-attn gamma (init 0; loaded below)
            `BUFSPLIT_ZERO(gamma_final_buf, emb_i) // FIX-F: g11 final-norm gamma (init 0; loaded below)
            emb_init[emb_i]      = '0;
        end
        if ($value$plusargs("EMB_FULL_HEX=%s", emb_hex_path)) begin
            // S3 FULL-H preload: +EMB_FULL_HEX = 3072 INT16 Q4.12 words (the FULL
            // seed-42 G1 input). Load all 3072 words and pack 16 consecutive words
            // into each of the 192 chunk slots, in NATURAL element order (element e
            // -> slot e/16, lane e%16). Drives the full G1 RMSNorm so its streamed
            // output exercises the full-width R1 seam (S3 gate).
            begin : g1_load_full
                reg [15:0] word_mem [0:EMB_H-1];
                integer    e;
                for (e = 0; e < EMB_H; e = e + 1) word_mem[e] = 16'h0;
                $readmemh(emb_hex_path, word_mem);
                for (e = 0; e < EMB_H; e = e + 1)
                    `BUFSPLIT_W16(input_emb_buf, e, word_mem[e])
            end
            // gamma: if +GAMMA_FULL_HEX given, load the real 3072-word Q4.12 gamma
            // (input_layernorm_g) so the seam matches golden. Else default 1.0.
            begin : g1_gamma_full
                reg [15:0] g_mem [0:EMB_H-1];
                integer    e2;
                string     gamma_hex_path;
                for (e2 = 0; e2 < EMB_H; e2 = e2 + 1) g_mem[e2] = GAMMA_ONE_Q12;
                if ($value$plusargs("GAMMA_FULL_HEX=%s", gamma_hex_path)) begin
                    $readmemh(gamma_hex_path, g_mem);
                    $display("[S3] G1 FULL-H gamma preload from %s", gamma_hex_path);
                end
                for (e2 = 0; e2 < EMB_H; e2 = e2 + 1)
                    `BUFSPLIT_W16(gamma_emb_buf, e2, g_mem[e2])
            end
            // S7d DUAL-INPUT: 2nd gamma image (input_layernorm_g, Q4.12) for the
            // e_norm path. +GAMMA_E_HEX = 3072-word hex; else default 1.0.
            begin : g1e_gamma_full
                reg [15:0] ge_mem [0:EMB_H-1];
                integer    e3;
                string     gamma_e_hex_path;
                for (e3 = 0; e3 < EMB_H; e3 = e3 + 1) ge_mem[e3] = GAMMA_ONE_Q12;
                if ($value$plusargs("GAMMA_E_HEX=%s", gamma_e_hex_path)) begin
                    $readmemh(gamma_e_hex_path, ge_mem);
                    $display("[S7d] G1_E FULL-H gamma preload from %s", gamma_e_hex_path);
                end
                for (e3 = 0; e3 < EMB_H; e3 = e3 + 1)
                    `BUFSPLIT_W16(gamma_e_buf, e3, ge_mem[e3])
            end
            // FIX-D: post-attention RMSNorm gamma (post_attn_norm_g, Q4.12) for
            // u_rmsnorm_premlp. +GAMMA_POSTATTN_HEX = 3072-word hex; else 1.0.
            begin : premlp_gamma_full
                reg [15:0] gp_mem [0:EMB_H-1];
                integer    e4;
                string     gamma_pa_hex_path;
                for (e4 = 0; e4 < EMB_H; e4 = e4 + 1) gp_mem[e4] = GAMMA_ONE_Q12;
                if ($value$plusargs("GAMMA_POSTATTN_HEX=%s", gamma_pa_hex_path)) begin
                    $readmemh(gamma_pa_hex_path, gp_mem);
                    $display("[FIX-D] post-attn gamma preload from %s", gamma_pa_hex_path);
                end
                for (e4 = 0; e4 < EMB_H; e4 = e4 + 1)
                    `BUFSPLIT_W16(gamma_postattn_buf, e4, gp_mem[e4])
            end
            // FIX-F: final RMSNorm gamma (final_norm_g, Q4.12) for u_rmsnorm_g11.
            // +GAMMA_FINAL_HEX = 3072-word hex; else default 1.0. Mirrors the
            // hidden_norm gamma loader (GAMMA_FULL_HEX) for the g11 norm.
            begin : final_gamma_full
                reg [15:0] gf_mem [0:EMB_H-1];
                integer    e5;
                string     gamma_fn_hex_path;
                for (e5 = 0; e5 < EMB_H; e5 = e5 + 1) gf_mem[e5] = GAMMA_ONE_Q12;
                if ($value$plusargs("GAMMA_FINAL_HEX=%s", gamma_fn_hex_path)) begin
                    $readmemh(gamma_fn_hex_path, gf_mem);
                    $display("[FIX-F] final-norm gamma preload from %s", gamma_fn_hex_path);
                end
                for (e5 = 0; e5 < EMB_H; e5 = e5 + 1)
                    `BUFSPLIT_W16(gamma_final_buf, e5, gf_mem[e5])
            end
            $display("[S3] G1 FULL-H input preload: %0d words from %s into %0d chunk slots",
                     EMB_H, emb_hex_path, EMB_NCHUNKS);
        end else if ($value$plusargs("EMB_HEX=%s", emb_hex_path)) begin
            // golden g1_in.hex = 16 INT16 Q4.12 words (one chunk). Load them as
            // 16 separate 16-bit words, pack into the single real chunk slot.
            begin : g1_load
                reg [15:0] word_mem [0:EMB_K_PAR-1];
                integer    w;
                for (w = 0; w < EMB_K_PAR; w = w + 1) word_mem[w] = 16'h0;
                $readmemh(emb_hex_path, word_mem);
                for (w = 0; w < EMB_K_PAR; w = w + 1)
                    `BUFSPLIT_W16(input_emb_buf, EMB_DATA_SLOT*EMB_K_PAR + w, word_mem[w])
            end
            // gamma = 1.0 (Q4.12) for every element of the real chunk
            for (emb_i = 0; emb_i < EMB_K_PAR; emb_i = emb_i + 1)
                `BUFSPLIT_W16(gamma_emb_buf, EMB_DATA_SLOT*EMB_K_PAR + emb_i, GAMMA_ONE_Q12)
            $display("[M4] G1 input-embedding preload: %0d words from %s into slot %0d",
                     EMB_K_PAR, emb_hex_path, EMB_DATA_SLOT);
        end
    end
    // synthesis translate_on

    // ================================================================
    // B6: SYNTH-VISIBLE gamma ROM bake (silicon-correctness fix).
    // (B6_GAMMA_BAKE_SPEC_20260527.md) The 4 RMSNorm gammas are STATIC model
    // constants; the only OTHER fill is inside the simulation-only guarded
    // region above, so the FLASHED PDI would run un-init (0/X) gamma -> every
    // RMSNorm wrong on silicon. BAKE them into the netlist as ROM init.
    // ================================================================
    // GAMMA-BUFFER-FIX (2026-05-27, GAMMA_BUFFER_FIX_20260527.md): the previous
    // PROCEDURAL bake form ($readmemh into a flat temp `gb_mem`, then a
    // for-loop slice-pack `array[r][slice]=temp[..]`) was SILENTLY DROPPED by
    // Vivado synthesis (no-driver [Synth 8-3848]) despite being synth-visible —
    // confirmed by BAKE_DRIVER_AUDIT_20260527.md (the 4 gammas appeared in the
    // oocfit2 drop list WITH this exact bake present). The temp->slice copy is
    // NOT recognized as ROM init. Replaced with the PROVEN-SURVIVING idiom (A):
    // a DIRECT `initial $readmemh("file", array)` straight into the target
    // packed array, exactly like rope_layer.sv:49 (`cs_lut`, ROM-inferred,
    // survives on xcv80). To make a direct load possible the flat 3072-INT16
    // goldens are PRE-PACKED into 192 x 256-bit chunk lines (lane15..lane0,
    // MSB..LSB, matching gamma_buf[c][lane*16 +: 16]) by scripts/pack_gamma_hex.py
    // -> /tmp/p4/<name>_packed.hex. Values are bit-identical to the prior bake
    // and to the xsim-validated buffer contents (round-trip verified).
    //
    // No plusarg gate / no temp: the direct readmemh is the whole driver (ROM).
    // Sim TBs that wish to override a gamma drive the +GAMMA_*_HEX path through
    // the translate_off `g1_emb_preload` block ABOVE; in those sims that block
    // packs the array AFTER t=0 elaboration of this initial only if it runs
    // later. (Current s7c TBs supply NO +GAMMA_*_HEX, so the baked ROM is the
    // sole gamma source in every active gate — preserving bit-exactness.)
    //
    // GAMMA-LOAD FIX (2026-06-04, GAMMA_LOAD_FIX.md): the prior idiom-A bake here
    //   initial $readmemh("/tmp/p4/<name>_packed.hex", gamma_<N>_buf);
    // was BROKEN on silicon two ways: (1) the WIDE 256-bit gamma word is the
    // lmdown LUT-ROM / no-driver trap class ([[feedback_nodriver_bake_drop_synth_only]],
    // "TWO axes": a 256b idiom-A $readmemh either drops with [Synth 8-3848] or
    // maps to a LUT-ROM, NOT a driven array); (2) the absolute /tmp/p4 path only
    // exists on THIS build host, so any other build synthesizes un-init (0/X)
    // gamma -> every RMSNorm output 0 -> all-zero hidden -> identity frontier
    // [0..7] on all 8 seeds (the exact ZERO_LOGIT_BISECTION silicon fingerprint).
    //
    // REPLACED with idiom-B: a real always_ff WRITE port fed by the host boot
    // trio (0x48/0x4c/0x50, target-selected by boot_gamma_sel @0x08). This gives
    // each gamma array a synth-tracked driver (survives synth, BRAM-mapped via the
    // ram_style="block" attr above) AND a host load path (session-init streams the
    // 4 Q4.12 gammas once per cold-AC). The zero-init in the translate_off
    // g1_emb_preload block above seeds X->0 for sim; sim TBs that want a specific
    // gamma either drive +GAMMA_*_HEX (translate_off) OR exercise this write port.
    // 32-bit sub-word write via the forward-declared gamma_* nets (driven near the
    // boot mux, exactly like emb_we / emb_wr_chunk). One driver per gamma array.
    // BUFSPLIT: per-32b-word bank write (data32 -> bank=gamma_wr_word at
    // chunk=gamma_wr_chunk). Exactly one bank fires per gamma write strobe.
    always_ff @(posedge ap_clk) begin : gamma_load_wr
        `BUFSPLIT_WR(gamma_emb_buf,      gamma_we_hidden, gamma_wr_chunk, gamma_wr_word, gamma_wr_data32)
        `BUFSPLIT_WR(gamma_e_buf,        gamma_we_enorm,  gamma_wr_chunk, gamma_wr_word, gamma_wr_data32)
        `BUFSPLIT_WR(gamma_postattn_buf, gamma_we_postat, gamma_wr_chunk, gamma_wr_word, gamma_wr_data32)
        `BUFSPLIT_WR(gamma_final_buf,    gamma_we_final,  gamma_wr_chunk, gamma_wr_word, gamma_wr_data32)
    end

    // ---- GAP-4 RUNG C: emb_gather -> input_emb_buf chunk-write bus + handshake ----
    // Module-scope nets so the write arbiter below sees the gather source; driven
    // by u_emb_gather in the DEPTH_N generate (EMBG_EN=1) or tied 0 (g_embg_tieoff).
    wire                        embg_o_we;
    wire [7:0]                  embg_o_chunk;    // 0..191
    wire [EMB_K_PAR*EMB_AW-1:0] embg_o_data;     // 256b = 16 x Q4.12
    wire                        emb_start_w;     // top_fsm S_LVL_EMBG launch pulse
    wire                        emb_done_w;      // emb_gather live-start-masked done

    // Single always_ff writer for input_emb_buf.
    //   EMBG_EN=0 : the EXACT pre-graft host-only per-word BUFSPLIT_WR
    //     (byte-identical — no-regress by construction).
    //   EMBG_EN=1 : the OOC-proven byte-enabled write ARBITER
    //     (embwrap_bufsplit_ooc.sv, 138 LUT / 8xRAMB18E5 / 0 URAM). ONE always_comb
    //     resolves host per-word vs emb_gather chunk-wide (gather priority) into a
    //     single {en,chunk,bmask,data} bus, then EXACTLY ONE write statement per
    //     bank -> Vivado infers 8 block-RAM banks with per-32b-lane write enable
    //     (NOT [Synth 8-4767] "multiple writes via different ports" -> FF dissolve
    //     -> the 192:1 read mux). See TREE_FSM_GAP4_EMBGATHER.md §A.2.
    generate if (EMBG_EN) begin : g_emb_arb
        logic                        emb_wr_en;
        logic [EMB_CW-1:0]           emb_wr_chunk_a;   // ONE shared address net (all 8 banks)
        logic [EMB_NBANKS-1:0]       emb_wr_bmask;     // per-bank write-enable mask
        logic [EMB_K_PAR*EMB_AW-1:0] emb_wr_data256;
        always_comb begin
            if (embg_o_we) begin                       // gather wins the collision
                emb_wr_en      = 1'b1;
                emb_wr_chunk_a = embg_o_chunk[EMB_CW-1:0];
                emb_wr_bmask   = {EMB_NBANKS{1'b1}};                 // chunk = all 8 banks
                emb_wr_data256 = embg_o_data;
            end else begin
                emb_wr_en      = emb_we;
                emb_wr_chunk_a = emb_wr_chunk[EMB_CW-1:0];
                emb_wr_bmask   = emb_we ? (EMB_NBANKS'(1) << emb_wr_word) : '0; // one bank
                emb_wr_data256 = {EMB_NBANKS{emb_wr_data32}};        // same 32b on all lanes
            end
        end
        always_ff @(posedge ap_clk) begin
            if (emb_wr_en && emb_wr_bmask[0]) input_emb_buf_b0[emb_wr_chunk_a] <= emb_wr_data256[0*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[1]) input_emb_buf_b1[emb_wr_chunk_a] <= emb_wr_data256[1*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[2]) input_emb_buf_b2[emb_wr_chunk_a] <= emb_wr_data256[2*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[3]) input_emb_buf_b3[emb_wr_chunk_a] <= emb_wr_data256[3*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[4]) input_emb_buf_b4[emb_wr_chunk_a] <= emb_wr_data256[4*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[5]) input_emb_buf_b5[emb_wr_chunk_a] <= emb_wr_data256[5*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[6]) input_emb_buf_b6[emb_wr_chunk_a] <= emb_wr_data256[6*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[7]) input_emb_buf_b7[emb_wr_chunk_a] <= emb_wr_data256[7*32 +: 32];
        end
    end else begin : g_emb_hostonly
        always_ff @(posedge ap_clk) begin
            `BUFSPLIT_WR(input_emb_buf, emb_we, emb_wr_chunk, emb_wr_word, emb_wr_data32)
        end
    end endgenerate

    // GAP-4 RUNG C: tie off m_axi_emb + the gather bus + emb_done whenever the
    // gather is not present (DEPTH_N=0 OR EMBG_EN=0). u_emb_gather (in g_depth_n,
    // gated on EMBG_EN) drives them in the live case. Mutually exclusive with the
    // instance -> exactly ONE driver per net. emb_start_w is ALWAYS driven by the
    // top_fsm output (tied 0 inside the FSM when EMBG_EN=0).
    generate if (!(DEPTH_N && EMBG_EN)) begin : g_embg_tieoff
        assign m_axi_emb_arid    = '0;
        assign m_axi_emb_araddr  = '0;
        assign m_axi_emb_arlen   = '0;
        assign m_axi_emb_arsize  = '0;
        assign m_axi_emb_arburst = '0;
        assign m_axi_emb_arvalid = 1'b0;
        assign m_axi_emb_rready  = 1'b0;
        assign embg_o_we    = 1'b0;
        assign embg_o_chunk = '0;
        assign embg_o_data  = '0;
        assign emb_done_w   = 1'b0;
    end endgenerate
    // EMB_NCH ch1 tie-off: m_axi_embb is parked unless the gather is live AND
    // strapped to >=2 channels.  Mutually exclusive with the g_embg driver
    // below -> exactly ONE driver per net at every strap combination.
    generate if (!(DEPTH_N && EMBG_EN && (EMB_NCH > 1))) begin : g_embb_tieoff
        assign m_axi_embb_arid    = '0;
        assign m_axi_embb_araddr  = '0;
        assign m_axi_embb_arlen   = '0;
        assign m_axi_embb_arsize  = '0;
        assign m_axi_embb_arburst = '0;
        assign m_axi_embb_arvalid = 1'b0;
        assign m_axi_embb_rready  = 1'b0;
    end endgenerate
    // FIX-E: single always_ff writer for h_fc_buf (silicon DMA path). Mirrors the
    // input_emb_buf writer; driven by hfc_we (= ctrl_boot_we_hfc). One driver per word.
    // BUFSPLIT: per-32b-word bank write for h_fc.
    always_ff @(posedge ap_clk) begin
        `BUFSPLIT_WR(h_fc_buf, hfc_we, hfc_wr_chunk, hfc_wr_word, hfc_wr_data32)
    end

    // ---- chunk-indexed read, streamed in lockstep with the G1 rmsnorm FSM ----
    wire [EMB_CW-1:0] g1_req_chunk;
    wire              g1_req_valid;
    // S3 off-by-one (proven by $display trace, 2026-05-26): rmsnorm_block's
    // apply_en/sumsq_en are REGISTERED one cycle after the FSM increments
    // chunk_cnt, so when en is actually high the FSM presents x_req_chunk =
    // {1,2,...,N_CHUNKS} — slot 0 is skipped and slot N_CHUNKS (=192, OOB) reads 0.
    // The legacy single-chunk pilot masked this (EMB_DATA_SLOT=1). For the FULL-H
    // streaming feed (SEAM_R1_FULLW=1) we compensate the read index by -1 so the
    // FSM samples input slots 0..191 in natural element order. Read-mux only (no
    // FSM change → all H=16 goldens unaffected), gated by SEAM_R1_FULLW so the
    // legacy pilot read path is byte-identical.
    wire [EMB_CW-1:0] g1_eff_chunk = SEAM_R1_FULLW
        ? (g1_req_chunk - EMB_CW'(1))   // full-H: undo the apply/sumsq pre-increment
        : g1_req_chunk;                 // legacy pilot: slot == req_chunk (slot1)
    // S3: the full-H input read needs one cycle beyond x_req_valid. The LAST
    // sumsq/apply request (req_chunk=N_CHUNKS) lands in the first drain cycle,
    // where x_req_valid is already LOW; gating on x_req_valid alone returns 0
    // for chunk 191. Do not leave the read enabled on the range alone, however:
    // rmsnorm_block parks chunk_cnt at N_CHUNKS after done, which made
    // g1_rd_ir stay high through the later R3 walk and falsely trip C4 on every
    // cycle. Keep exactly the required one-cycle falling-edge tail.
    reg g1_req_valid_d1;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) g1_req_valid_d1 <= 1'b0;
        else           g1_req_valid_d1 <= g1_req_valid;
    end
    wire g1_req_live = g1_req_valid || g1_req_valid_d1;
    wire g1_eff_inrange = SEAM_R1_FULLW
        ? (g1_req_live &&
           g1_req_chunk >= EMB_CW'(1) && g1_req_chunk <= EMB_CW'(EMB_NCHUNKS))
        : (g1_req_valid && g1_req_chunk <  EMB_CW'(EMB_NCHUNKS));
    // FIX-E R1: G1 h_norm (hidden_norm) reads h_fc (precomputed fc(hidden_cat3))
    // at frac=12 (fc_h.hex), NOT the embedding. e_norm (below) keeps input_emb_buf.
    // COMBREAD FIX (2026-06-08, RMSNORM_COMBREAD_CONFIRM): h_fc_buf/gamma_emb_buf
    // are (* ram_style="block" *) BRAM — the read MUST be REGISTERED (silicon
    // BRAM is a 1-cycle synchronous read; the old combinational `assign` was a
    // stale-by-1 trap that xsim async hid). Register the read here and set
    // RDATA_LAT=1 on u_rmsnorm_g1 so the sumsq/apply engines consume the data one
    // cycle later, exactly when the registered read delivers buf[eff_chunk]. The
    // address (g1_eff_chunk = g1_req_chunk-1) still leads the data by 1 cycle.
    reg signed [EMB_K_PAR*EMB_AW-1:0] g1_x_chunk;
    reg signed [EMB_K_PAR*EMB_AW-1:0] g1_gamma_chunk;
    // =====================================================================
    // F2 (FORREASON DEPTH-MUX, TREE_FSM_DEPTHN_F2_DEPTHMUX.md) — module-scope
    // plumbing. The F1 producer's registered read port (u_forreason.rd_chunk,
    // RD_LAT=1) is surfaced on these two nets by g_resid_fullh.g_forreason and
    // tied off in every other generate arm, so the TWO block-input consumers
    // (G1 h_norm just below, R3 residual a-arm inside g_resid_fullh) can
    // depth-mux between h_fc_buf (level 0) and the producer's `a` (level>=1).
    //   f2_lvl_ge1_w    : FORREASON && DEPTH_N && (level_i_w != 0). Stable
    //                     across a whole child body pass (level_i_w only steps
    //                     between levels), so it needs no per-cycle alignment.
    //                     Constant 0 at FORREASON=0 or DEPTH_N=0 -> both muxes
    //                     fold away, depth-0 elaboration byte-identical.
    //   f2_fr_rd_data_w : the producer's a, Q(FR_A_FRAC)=Q6.9. addr->data
    //                     latency is 1 cycle == the registered h_fc_buf reads
    //                     it replaces on BOTH consumers (sim==silicon).
    // =====================================================================
    wire signed [EMB_K_PAR*EMB_AW-1:0] f2_fr_rd_data_w;
    wire                               f2_lvl_ge1_w;
    // =====================================================================
    // L2 (PLUS128_LOADPATH_FIX_V2 §3) MAXPROBE: register the EFFECTIVE READ
    // CHUNK ADDRESS in a flop the L3 tight SOFT pblock co-locates with the bank-
    // split BRAM. The +128/+64 lives on the ~14.6 ns COMBINATIONAL chunk-address
    // route (counter-FF -> g1_eff_chunk decrementer LUT -> 256 BRAM ADDR sinks).
    // Registering the address once makes the only in-cycle path ADDR-FF -> BRAM
    // (sub-ns when co-located), removing the hostile net. Adds 1 read-latency
    // cycle, re-aligned by RDATA_LAT(2) on u_rmsnorm_g1 below. (* keep *) +
    // (* max_fanout=8 *) (L4) so the tool replicates the addr driver per bank
    // cluster. BUFRDREG=0 -> byte-identical legacy comb-address read.
    // =====================================================================
    (* keep = "true", max_fanout = 8 *) reg [EMB_CW-1:0] g1_rd_chunk_q;
    reg                                                  g1_rd_inrange_q;
    always_ff @(posedge ap_clk) begin
        g1_rd_chunk_q   <= g1_eff_chunk;     // register the comb decrement ONCE
        g1_rd_inrange_q <= g1_eff_inrange;
    end
    wire [EMB_CW-1:0] g1_rd_idx = BUFRDREG ? g1_rd_chunk_q   : g1_eff_chunk;
    wire              g1_rd_ir  = BUFRDREG ? g1_rd_inrange_q : g1_eff_inrange;
    // ---- PER-STAGE INJECT: G1 (RMSNorm input, idx 0) ----
    // When inject_en[0] is set, the G1 RMSNorm input chunk comes from the golden
    // inject buffer (low EMB_AW bits of the first EMB_K_PAR words) instead of the
    // h_fc buffer read — isolating the RMSNorm transform from any h_fc load fault.
    // gamma is left as the real read (RMSNorm needs the real gamma to match golden).
    // inject_en/inj_bus=0 when STAGE_INJECT=0 -> bit-identical to the original read.
    wire signed [EMB_K_PAR*EMB_AW-1:0] inj_g1_chunk;
    genvar gig1;
    generate
        for (gig1 = 0; gig1 < EMB_K_PAR; gig1 = gig1 + 1) begin : g_inj_g1
            assign inj_g1_chunk[gig1*EMB_AW +: EMB_AW] = inj_bus[gig1*32 +: EMB_AW];
        end
    endgenerate
    always_ff @(posedge ap_clk) begin
        g1_x_chunk     <= inject_en[0] ? inj_g1_chunk
                                       : (g1_rd_ir ? h_fc_buf_rd(g1_rd_idx) : '0);  // BUFSPLIT read @ REGISTERED addr
        g1_gamma_chunk <= g1_rd_ir ? gamma_emb_buf_rd(g1_rd_idx) : '0;  // BUFSPLIT read @ REGISTERED addr
    end
    // ---- F2 (DEPTH-MUX) consumer #1: the G1 h_norm input ----------------
    // Level 0 keeps the h_fc_buf register above BYTE-IDENTICALLY. Level >= 1
    // reads the producer's `a` INSTEAD (the child block input a_{d-1};
    // DEPTHN_REKEY_SPEC.md / TREE_FSM_DEPTHN_BODY_REKEY_RCA.md F2).
    //
    // RE-GRADE (F1 §3.2: ~3 bits, NOT the old >>>7/<<<2 9-bit prescription):
    // `a` is presented to G1 at its OWN grid, Q(FR_A_FRAC)=Q6.9 — a ZERO-bit
    // physical shift, so it is EXACT (no saturation possible, no rounding).
    // The ~3-bit re-grade vs h_fc's Q3.12 is carried entirely by the frac
    // relabel (12 -> 9 = 3 bits), which matches the measured magnitude growth
    // ||a||/||h_fc|| = 125.9..154.5/16.9 = 7.4..9.1x (~2.9..3.2 bits): the
    // engine's INT operating point at level>=1 is 0.93x the proven level-0
    // envelope. RMSNorm is scale-invariant (y = x/rms(x)*gamma consumes the
    // dimensionless ratio of the SAME ints), so no downstream frac changes.
    //
    // LATENCY: the producer read is addr@t -> data@t+1 (RD_LAT=1), the same
    // as the registered h_fc read above; the in-range zero-gate is applied at
    // the DATA cycle via g1_rd_ir delayed 1 (f2_g1_ir_q). The producer's
    // rd_idx is driven with this same g1_rd_idx inside g_forreason.
    reg f2_g1_ir_q;
    always_ff @(posedge ap_clk) f2_g1_ir_q <= g1_rd_ir;
`ifdef F2_RED_HFC
    // RED ARM (the F2 gate MUST FAIL): consumer pinned to the OLD h_fc path
    // at level>=1 — the exact pre-F2 stale-root defect, re-armed.
    wire f2_g1_use_fr = 1'b0;
`else
    wire f2_g1_use_fr = f2_lvl_ge1_w && !inject_en[0];
`endif
    wire signed [EMB_K_PAR*EMB_AW-1:0] g1_x_chunk_eff = f2_g1_use_fr
        ? (f2_g1_ir_q ? f2_fr_rd_data_w : '0)
        : g1_x_chunk;
    // §6 READ-SIDE residency taps (PLUS128_LOADPATH_FIX_V2 §6): the WRITE-side CRC
    // (res_sel 0-4) is BLIND to the read-side +128 slip (write addr is clean). Fold
    // the chunk index the read ACTUALLY presents (g1_rd_idx) together with the data
    // it gets back (g1_x_chunk[31:0]) in apply order, so the flashed card shows
    // per-buffer whether the read MAP became golden. g1_rd_stb = the cycle
    // g1_x_chunk holds a valid in-range read (g1_rd_ir delayed 1 cycle = the data
    // cycle). idx_tap = the address used to fetch THIS word (g1_rd_idx delayed 1).
    reg              g1_rd_stb;
    reg [EMB_CW-1:0] g1_rd_idx_tap;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin g1_rd_stb <= 1'b0; g1_rd_idx_tap <= '0; end
        else begin
            g1_rd_stb     <= g1_rd_ir;     // data lands 1 cyc after g1_rd_ir sampled
            g1_rd_idx_tap <= g1_rd_idx;    // the addr that produced g1_x_chunk this cycle
        end
    end
    wire [31:0] g1_rd_word_tap = g1_x_chunk_eff[31:0];   // low 32b (lane0/1) — order+value sensitive (F2: tap follows the EFFECTIVE consumer data; ==g1_x_chunk at level 0)
    wire              g1_apply_en;     // S3/M5: G1 apply-phase strobe tap

    wire signed [16*16-1:0] rms_g1_out_chunk_w;
    wire                    rms_g1_out_valid;
    wire                    g1_done;
    // FULL-H AVERAGING FIX (rmsnorm_fullh, merged 2026-05-23): G1 input is the
    // Q4.12 input-embedding (f=12, B=2, small RMS → gain≈20). The legacy
    // SUMSQ_RSHIFT=12 lands the Q-format but leaves the TALOS engine's hardcoded
    // /16 average in place → at full H=3072 the RMSNorm output was sqrt(192)≈13.86×
    // too small (a real correctness bug, NOT a uniform scale — it corrupts the
    // residual-add sum). The constant-coefficient multiply (MULT=43691≈(2/3)·2^16,
    // SHR=31) makes the engine's /16 yield the correct /3072 mean-of-squares;
    // SCALE_SHR=10 (=12-B) gives the Q(6.10) scale headroom for gain>8. Strict
    // generalization of the legacy shift (H=16 reduces to the pure shift; verified).
    // Params from rmsnorm_fullh/goldens/fullh_params.json (xsim bit-exact, g1).
    rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                    .SUMSQ_RSHIFT(12),                 // legacy fallback (unused: MULT!=0)
                    .SUMSQ_MULT(43691), .SUMSQ_SHR(31), .SCALE_SHR(10),
                    .XS_WIDTH(18),
                    .RDATA_LAT(BUFRDREG ? 2 : 1)) u_rmsnorm_g1 (  // COMBREAD FIX + L2 reg-addr (+1)
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g1_start),
        .x_chunk(g1_x_chunk_eff),      // M4: h_fc chunk stream (F2: == g1_x_chunk at level 0; the producer's `a` at level>=1)
        .gamma_chunk(g1_gamma_chunk),  // M4: gamma (1.0 Q4.12 for pilot)
        .out_chunk(rms_g1_out_chunk_w),
        .out_valid(rms_g1_out_valid),
        .done(g1_done),
        .x_req_chunk(g1_req_chunk),
        .x_req_valid(g1_req_valid),
        .apply_en_o(g1_apply_en)        // S3/M5: capture-strobe source (apply_en+AP_LAT)
    );

    // ================================================================
    // S7d DUAL-INPUT: 2nd RMSNorm — e_norm = input_layernorm(embeds)
    // ================================================================
    // FULLK_DUALINPUT_ANALYSIS_20260526 §4.1 + ENORM_VERIFY_RESULT (pdi6_enorm):
    // G4's true input is K=6144 = cat(e_norm, h_norm). The EXISTING u_rmsnorm_g1
    // is the h_norm path (HIGH half [3072:6143] -> seam chunks [192..383]); THIS
    // new instance is the e_norm / input_layernorm path (LOW half [0:3071] ->
    // seam chunks [0..191]). IDENTICAL engine + params to u_rmsnorm_g1 (same /3
    // constant-mult full-H averaging fix, XS_WIDTH=18); the ONLY deltas are the
    // gamma stream (gamma_e_buf = input_layernorm_g) and that its captured output
    // is written into the LOW half of the 6144-wide R1 seam below. ENORM verify
    // proved this wiring bit-exact-class vs the g1_e_norm golden (max|d|=5,
    // cosine 0.99999844, no global scale). Don't swap the halves (CAVEAT).
    //
    // Both norms co-launch on g1_start and run independently (same FSM cadence);
    // each captures its 192 chunks (apply_en + AP_LAT=2) into its seam half. The
    // G4 array_start gate (seam_r1_full = all 384 chunks in) covers both writers.
    wire [EMB_CW-1:0] g1e_req_chunk;
    wire              g1e_req_valid;
    // Mirror the g1 full-H off-by-one undo (-1) and range-gate, gated by SEAM_R1_FULLW.
    wire [EMB_CW-1:0] g1e_eff_chunk = SEAM_R1_FULLW
        ? (g1e_req_chunk - EMB_CW'(1))
        : g1e_req_chunk;
    wire g1e_eff_inrange = SEAM_R1_FULLW
        ? (g1e_req_chunk >= EMB_CW'(1) && g1e_req_chunk <= EMB_CW'(EMB_NCHUNKS))
        : (g1e_req_valid && g1e_req_chunk <  EMB_CW'(EMB_NCHUNKS));
    // INPUT = the embed buffer (same as u_rmsnorm_g1 in this pilot — see
    // ENORM_VERIFY §"Exact instantiation"). GAMMA = gamma_e_buf (input_layernorm_g).
    // COMBREAD FIX (2026-06-08): input_emb_buf/gamma_e_buf are BRAM — register the
    // read; u_rmsnorm_g1_e carries RDATA_LAT=1 to consume the registered data one
    // cycle later (see u_rmsnorm_g1).
    reg signed [EMB_K_PAR*EMB_AW-1:0] g1e_x_chunk;
    reg signed [EMB_K_PAR*EMB_AW-1:0] g1e_gamma_chunk;
    // L2 (MAXPROBE) e-path: identical registered-read-address structure as g1.
    (* keep = "true", max_fanout = 8 *) reg [EMB_CW-1:0] g1e_rd_chunk_q;
    reg                                                  g1e_rd_inrange_q;
    always_ff @(posedge ap_clk) begin
        g1e_rd_chunk_q   <= g1e_eff_chunk;
        g1e_rd_inrange_q <= g1e_eff_inrange;
    end
    wire [EMB_CW-1:0] g1e_rd_idx = BUFRDREG ? g1e_rd_chunk_q   : g1e_eff_chunk;
    wire              g1e_rd_ir  = BUFRDREG ? g1e_rd_inrange_q : g1e_eff_inrange;
    always_ff @(posedge ap_clk) begin
        g1e_x_chunk     <= g1e_rd_ir ? input_emb_buf_rd(g1e_rd_idx) : '0;  // BUFSPLIT read @ REGISTERED addr
        g1e_gamma_chunk <= g1e_rd_ir ? gamma_e_buf_rd(g1e_rd_idx)   : '0;  // BUFSPLIT read @ REGISTERED addr
    end
    // §6 READ-SIDE residency taps for the e-path (input_emb_buf read map).
    reg              g1e_rd_stb;
    reg [EMB_CW-1:0] g1e_rd_idx_tap;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin g1e_rd_stb <= 1'b0; g1e_rd_idx_tap <= '0; end
        else begin
            g1e_rd_stb     <= g1e_rd_ir;
            g1e_rd_idx_tap <= g1e_rd_idx;
        end
    end
    wire [31:0] g1e_rd_word_tap = g1e_x_chunk[31:0];

    wire signed [16*16-1:0] rms_g1e_out_chunk_w;
    wire                    rms_g1e_out_valid;
    wire                    g1e_done;
    wire                    g1e_apply_en;     // e-path apply-phase strobe (seam LOW write)
    wire                    g1e_apply_en_eff; // capture cadence of the selected e-norm engine

    rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                    .SUMSQ_RSHIFT(12),
                    .SUMSQ_MULT(43691), .SUMSQ_SHR(31), .SCALE_SHR(10),
                    .XS_WIDTH(18),
                    .RDATA_LAT(BUFRDREG ? 2 : 1)) u_rmsnorm_g1_e (  // == u_rmsnorm_g1 (+L2 reg-addr)
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g1_start),                 // co-launch with G1 (both norms, then concat)
        .x_chunk(g1e_x_chunk),
        .gamma_chunk(g1e_gamma_chunk),
        .out_chunk(rms_g1e_out_chunk_w),
        .out_valid(rms_g1e_out_valid),
        .done(g1e_done),
        .x_req_chunk(g1e_req_chunk),
        .x_req_valid(g1e_req_valid),
        .apply_en_o(g1e_apply_en)         // capture strobe (apply_en + AP_LAT) for seam LOW
    );

    // =====================================================================
    // DEPTH-N e_norm REQUANT FIX  (TREE_FSM_DEPTHN_L1_ENORM_FIX, 2026-07-23)
    // =====================================================================
    // ROOT CAUSE — offline bit-accurate model of rmsnorm_block/sumsq/scale_int/
    // apply (calib_depthn_hnorm.py + a2_enorm_requant_proof.py), relL2-per-stage
    // curve for pass=1, parent tok 622:
    //     EMB  (gathered child embed)          relL2 ~0       (EMBCRC 8/8, clean)
    //     ENORM h-half (a-arm / hidden_norm)   relL2 0.0007   (a-arm EXONERATED)
    //     ENORM e-half (input_layernorm)       relL2 0.4103   <-- THE JUMP
    //     G4 concat input (6144-wide)          relL2 ~0.24    (== golden §4.7 0.2407)
    // The e_norm engine's 31/10 grid (u_rmsnorm_g1_e: SUMSQ_SHR=31,SCALE_SHR=10)
    // is calibrated for the depth-0 seed embed (RMS~0.05). At depth>=1 the e_norm
    // input is a GATHERED REAL TOKEN embedding row (RMS~0.018 measured over the 8
    // seed tokens). scale = 1/rms is then LARGE and scale_q12 (Q4.12, max 32767)
    // SATURATES for every gathered token (measured 400/400 random tokens, 31-62%
    // per-element error) -> the e_norm output collapses -> G4 QKV direction is
    // corrupted -> wrong L1 children (the token-55 cross-parent collapse in C1).
    // NOTE: this is the e-half (embed) requant, NOT the a-half — the RCA prior
    // pointed at the a-arm; the relL2 curve refutes that (h-half is 0.0007).
    //
    // FIX: a depth-gated 2nd e_norm instance at 27/8 (SUMSQ_SHR=27,SCALE_SHR=8)
    // relocates the scale_q12 saturation headroom by +4 bits while preserving the
    // frac12 output magnitude (matched (SUMSQ_SHR,SCALE_SHR) pair). Measured:
    // e-half relL2 0.4103 -> 0.0157 (Q-grid floor); 0/400 tokens saturate; all 8
    // gate tokens <1.5% rel. (29/9, an earlier proposal, still saturates 13/400
    // min-RMS tokens -> rejected in favour of 27/8.)
    //
    // DEPTH-0 INERT: generate-gated on DEPTH_N (DEPTH_N=0 builds never see it),
    // and the output is muxed into the e-half seam ONLY at f2_lvl_ge1_w (level>=1);
    // at level 0 the seam write uses the ORIGINAL u_rmsnorm_g1_e verbatim ->
    // depth-0 byte-identical (param-dump 70==70, no top-level param added).
    localparam int ENORM_D_SUMSQ_SHR = PRECISION_SAFE ? 31 : 27;
    localparam int ENORM_D_SCALE_SHR = PRECISION_SAFE ? 10 : 8;
    wire signed [16*16-1:0] rms_g1e_out_chunk_eff;   // e-half seam write source
    generate if (DEPTH_N) begin : g_enorm_depth_requant
        wire signed [16*16-1:0] rms_g1e_d_out_chunk_w;
        // 31/10 can early-exit rmsnorm_scale_int at denom<=512, whereas 27/8
        // takes the 25-cycle divider. Co-launch therefore does not guarantee
        // lockstep APPLY. The selected 27/8 engine needs an independent
        // request-indexed BRAM feed and its own seam-capture strobe.
        wire [EMB_CW-1:0] g1ed_req_chunk;
        wire              g1ed_req_valid;
        wire [EMB_CW-1:0] g1ed_eff_chunk = SEAM_R1_FULLW
            ? (g1ed_req_chunk - EMB_CW'(1))
            : g1ed_req_chunk;
        wire g1ed_eff_inrange = SEAM_R1_FULLW
            ? (g1ed_req_chunk >= EMB_CW'(1) &&
               g1ed_req_chunk <= EMB_CW'(EMB_NCHUNKS))
            : (g1ed_req_valid && g1ed_req_chunk < EMB_CW'(EMB_NCHUNKS));
        (* keep = "true", max_fanout = 8 *) reg [EMB_CW-1:0] g1ed_rd_chunk_q;
        reg                                                  g1ed_rd_inrange_q;
        always_ff @(posedge ap_clk) begin
            g1ed_rd_chunk_q   <= g1ed_eff_chunk;
            g1ed_rd_inrange_q <= g1ed_eff_inrange;
        end
        wire [EMB_CW-1:0] g1ed_rd_idx =
            BUFRDREG ? g1ed_rd_chunk_q : g1ed_eff_chunk;
        wire g1ed_rd_ir = BUFRDREG ? g1ed_rd_inrange_q : g1ed_eff_inrange;
        reg signed [EMB_K_PAR*EMB_AW-1:0] g1ed_x_chunk;
        reg signed [EMB_K_PAR*EMB_AW-1:0] g1ed_gamma_chunk;
        always_ff @(posedge ap_clk) begin
            g1ed_x_chunk <= g1ed_rd_ir ? input_emb_buf_rd(g1ed_rd_idx) : '0;
            g1ed_gamma_chunk <= g1ed_rd_ir ? gamma_e_buf_rd(g1ed_rd_idx) : '0;
        end
        wire g1ed_apply_en;
        rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                        .SUMSQ_RSHIFT(12),
                        .SUMSQ_MULT(43691), .SUMSQ_SHR(ENORM_D_SUMSQ_SHR),
                        .SCALE_SHR(ENORM_D_SCALE_SHR),
                        .SCALE_EPS_BUCKET(PRECISION_SAFE ? 168 : 1),
                        .XS_WIDTH(18),
                        .RDATA_LAT(BUFRDREG ? 2 : 1)) u_rmsnorm_g1_e_d (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(g1_start),                 // identical FSM cadence to u_rmsnorm_g1_e
            .x_chunk(g1ed_x_chunk),
            .gamma_chunk(g1ed_gamma_chunk),
            .out_chunk(rms_g1e_d_out_chunk_w),
            .out_valid(), .done(),
            .x_req_chunk(g1ed_req_chunk), .x_req_valid(g1ed_req_valid),
            .out_chunk_strobe(), .out_chunk_idx_o(),
            .apply_en_o(g1ed_apply_en)
        );
        // RED arm: `-d ENORM_RED_SAT` re-pins the depth>=1 e-half to the ORIGINAL
        // saturating 31/10 output -> the gate MUST go back RED (proves the gate is
        // sensitive to the e_norm requant, not just to depth-N wiring).
`ifdef ENORM_RED_SAT
        wire use_enorm_d = 1'b0;
`elsif ENORM_GREEN_L0
        // RCA experiment (TREE_FSM_DEPTHN_G2B_ROOT_RCA): force the 27/8 requant at
        // ALL levels INCLUDING level-0. Discriminates whether the depth-0 real-root
        // 1/8 is the level-0 31/10 e_norm saturation (B) — if this flips G2B to 8/8,
        // the injection is faithful and the golden is correct.
        wire use_enorm_d = 1'b1;
`else
        // SHIPPED default (ENORM_D0 strap, eagle4_config.svh): apply the 27/8
        // requant at ALL levels INCLUDING level-0 real embeds — the shipped-root
        // accept lever (silicon d2_call0012 root 1/8 traced to level-0 31/10
        // scale_q12 saturation on real token embeds rms~0.02). ENORM_D0=1
        // (default) => 27/8 at level-0 too; ENORM_D0=0 => legacy level>=1-only
        // (f2_lvl_ge1_w). NOT gated behind f2_lvl_ge1_w/FORREASON: gating it off
        // at the root makes the accept lever inert (the DUAL_PORT/HEAD_PAR trap).
        wire use_enorm_d = PRECISION_SAFE ? 1'b1
                                         : (ENORM_D0 ? 1'b1 : f2_lvl_ge1_w);
`endif
        assign rms_g1e_out_chunk_eff = use_enorm_d ? rms_g1e_d_out_chunk_w
                                                   : rms_g1e_out_chunk_w;
        assign g1e_apply_en_eff = use_enorm_d ? g1ed_apply_en : g1e_apply_en;
    end else begin : g_enorm_depth_requant_off
        assign rms_g1e_out_chunk_eff = rms_g1e_out_chunk_w;   // DEPTH_N=0: byte-identical
        assign g1e_apply_en_eff = g1e_apply_en;
    end endgenerate

    // M4: hold the FIRST valid output chunk (the real-data chunk) so the
    // harness tap `rms_g1_out_chunk`, latched at g1_done, observes the
    // real-data chunk's RMSNorm result rather than the last (zero) streamed
    // chunk. Single always_ff driver → multi-driver-safe.
    //
    // Pipeline-skew note (proven by $display trace): rmsnorm_apply's `valid`
    // strobe rises ONE cycle AFTER the matching `out_chunk` data is on the bus
    // (out_arr is registered on en_a at cycle C+2, while valid <= en_b lands at
    // C+3). So when out_valid FIRST rises, the correct chunk result is on the
    // PREVIOUS cycle's out_chunk. We therefore capture a 1-cycle-delayed copy
    // of out_chunk at the first out_valid rising edge — yielding the real
    // chunk's RMSNorm output bit-for-bit (verified vs g1.hex, max|d|=1 LSB).
    reg signed [16*16-1:0] rms_g1_out_chunk_d1;   // out_chunk delayed 1 cycle
    reg signed [16*16-1:0] rms_g1_out_first;
    reg                    rms_g1_first_seen;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            rms_g1_out_chunk_d1 <= '0;
            rms_g1_out_first    <= '0;
            rms_g1_first_seen   <= 1'b0;
        end else begin
            rms_g1_out_chunk_d1 <= rms_g1_out_chunk_w;
            if (g1_start) begin
                rms_g1_first_seen <= 1'b0;        // re-arm per G1 invocation
            end else if (rms_g1_out_valid && !rms_g1_first_seen) begin
                rms_g1_out_first  <= rms_g1_out_chunk_d1;
                rms_g1_first_seen <= 1'b1;
            end
        end
    end
    wire signed [16*16-1:0] rms_g1_out_chunk = rms_g1_out_first;

    wire signed [16*16-1:0] rms_g11_out_chunk;
    wire                    rms_g11_out_valid;
    wire                    g11_done;
    // S6: clean per-chunk output strobe + index from the g11 RMSNorm, aligned
    // to rms_g11_out_chunk (apply_en + AP_LAT). Drives the FULL-WIDTH R8 seam
    // write (all 192 chunks), replacing the single-chunk pilot capture.
    localparam int G11_NCHUNKS = 3072/16;                 // 192
    localparam int G11_CW      = $clog2(G11_NCHUNKS + 1); // 8 (matches rmsnorm tap width)
    wire                  g11_out_strobe;
    wire [G11_CW-1:0]     g11_out_chunk_idx;
    // F3 (residual_propagate, 2026-05-23): the g11 / FINAL RMSNorm CONSUMES the
    // R6 seam, which carries the R7 residual output (h1+mlp). Per RESIDUAL-FORMAT
    // the residual stream is Q12.4 (FRAC_BITS=4), NOT Q4.12 — r7_out reaches ±857
    // (Q4.12 saturates 2722/3072 elems, downstream RMSNorm cosine collapses 0.83).
    // A Q12.4 INT16 input puts sumsq at sumsq_real*2^8; the TALOS scale engine
    // needs sumsq_real*2^12, so SUMSQ_RSHIFT = 2*FRAC_BITS-12 = 2*4-12 = -4
    // (a LEFT-shift). Derived + verified bit-exact at the engine-exact H=16
    // granularity (forks/residual_propagate/ref/derive_g11_q124_h16.py: Q12.4 in
    // -> Q12.4 out, worst|d|=1 LSB; legacy +12 on a Q12.4 input is off by 2^8).
    // S7C C2: the g11hp RMSNorm OUTPUT is Q6.10 (frac=10), requantized Q6.10 -> Q4.12
    // (<<2) at the R8 seam below (act_rescale) so G12 still consumes Q4.12 (final_norm
    // absmax 3.7 fits). (G1's input g1-in IS Q4.12 -> it keeps SUMSQ_RSHIFT=12.)
    // FIX-B/C (S7C_FRAC_CONFIG_AUDIT §4): the two residual seams carry DIFFERENT
    // golden fracs — r3_h1 = Q11.5 (frac=5).  Preserve G9's accumulator-native
    // Q8.8 for r7_out instead of discarding five fractional bits at the seam.
    // single RESID_FRAC=4 matched NEITHER golden binary point. Split into two
    // params. No COMPOSITE_RTLFRACS report exists proving frac=4 reaches golden,
    // so per the audit's option (ii) we carry the golden fracs and re-tune the
    // downstream RMSNorms (g11 for the Q8.8 r7 input; the new post-attn norm for
    // the Q11.5 r3 input — FIX-D). Golden ranges fit INT16 at both fracs
    // (r3_h1 absmax 5319, r7_out absmax 6804; sat=0).
    localparam int R3_FRAC = PRECISION_SAFE ? 7 : 5;
    localparam int R7_FRAC = 8;                          // Q8.8 r7_out/to_logits (-> g11)
    localparam int G11_SUMSQ_RSHIFT = 2*R7_FRAC - 12;    // legacy fallback (unused: MULT!=0)

    // PDI#5-CLOSING: guard the fp16-tail build hook. TAIL_FP16=1 requires the
    // (not-yet-built) fp16 MLP-tail datapath; fail elaboration loudly so a build
    // can't silently ship the INT16 tail when fp16 was requested. Default 0 is a
    // no-op. (References TAIL_FP16 so it surfaces as an IP GUI generic.)
    initial begin
        if (TAIL_FP16) begin
            $error("eagle4_drafter: TAIL_FP16=1 selected, but the fp16 MLP-tail datapath is NOT wired (fp16_tail #184 is a deferred RTL build item). Set TAIL_FP16=0 or add the fp16 tail lanes first.");
        end
        // B2: the scaled array and the K_PAR=64 wide-bank array are mutually
        // exclusive (the wide-bank array has no scaled lane). The FITTING config
        // is GEMV_SCALED=1 + GEMV_KPAR64=0 (memory #203/#204: K_PAR=64/32 overflow
        // URAM; only K_PAR=16 fits — and the scaled lane is K_PAR=16).
        if (GEMV_SCALED && GEMV_KPAR64) begin
            $error("eagle4_drafter: GEMV_SCALED=1 and GEMV_KPAR64=1 are mutually exclusive (no scaled K_PAR=64 lane). Set GEMV_KPAR64=0 for the scaled FITTING config.");
        end
        // EMB_NCH: the emb_gather MODULE is proven at NCH=1/2/4
        // (scripts/run_emb_nch_gate.sh), but the TOP exposes only two
        // embed-gather AXI masters (m_axi_emb -> S08, m_axi_embb -> S09).
        // NCH>2 needs 2 more top ports + S10/S11 BD patches; fail LOUD rather
        // than silently dropping channels 2..N-1 (the un-threaded-parameter
        // class this whole config header exists to prevent).
        if (EMB_NCH < 1 || EMB_NCH > 2) begin
            $error("eagle4_drafter: EMB_NCH=%0d is out of the TOP-supported range [1,2] (only m_axi_emb + m_axi_embb exist). Add the S10/S11 masters + BD patches first.", EMB_NCH);
        end
        if ((EMB_NCH > 1) && !(DEPTH_N && EMBG_EN)) begin
            $error("eagle4_drafter: EMB_NCH=%0d requires DEPTH_N=1 && EMBG_EN=1 (there is no gather to stripe otherwise).", EMB_NCH);
        end
        // 3072 elements / (NCH channels * 16 elems-per-beat) must be integral,
        // or the chunk map c*BEATS_PER_CH+n stops being a bijection onto [0,192).
        if ((3072 % (EMB_NCH*16)) != 0) begin
            $error("eagle4_drafter: H=3072 must be a multiple of EMB_NCH*16 (got EMB_NCH=%0d).", EMB_NCH);
        end
    end
    wire signed [16*16-1:0] seam_r6_rd_c;   // R6 connected arm (driven below)
    // S5B: g11 RMSNorm chunk-request taps (M4 idiom) → drive the FULL-WIDTH
    // R6 seam read index so g11 streams its full 3072-element input chunk-by-
    // chunk (NOT the single-chunk head-0 broadcast pilot).
    wire [G11_CW-1:0]       g11_x_req_chunk;
    wire                    g11_x_req_valid;
    // COMBREAD FIX (2026-06-08): seam_r6_rd_c is the COMBINATIONAL output of
    // u_seam_r6 (stage_io_buffer, ram_style="block" -> registered 1-cycle BRAM on
    // silicon) and gamma_final_buf is ram_style="block". Both are read with a
    // same-cycle address advance (r6_rd_idx / g11_g_eff_chunk derive from
    // g11_x_req_chunk). Register BOTH reads here and set RDATA_LAT=1 on
    // u_rmsnorm_g11 so the engines consume the data one cycle later (aligned to the
    // registered read). seam_r6_full gating is unchanged (start already waits).
    reg signed [16*16-1:0] g11_x_in;
    always_ff @(posedge ap_clk) begin
        g11_x_in <= CONNECT_SEAMS ? seam_r6_rd_c : '0;
    end
    // FIX-F: feed the REAL final_norm gamma (gamma_final_buf, Q4.12, final_norm_g.hex
    // absmax 1.430), chunk-indexed in lockstep with the g11 input stream. Mirror the
    // g11 input's +1 off-by-one undo (rmsnorm apply/sumsq pre-increment) so gamma slot
    // n aligns with x slot n. BACK_SEQ uses g11_x_req_chunk-1 (= r6_rd_idx); the legacy
    // pilot uses the bare index. Out-of-range / isolation -> 0 (matches g11_x_in).
    wire [G11_CW-1:0] g11_g_eff_chunk = BACK_SEQ
        ? (g11_x_req_chunk - G11_CW'(1))
        : g11_x_req_chunk;
    wire g11_g_inrange = BACK_SEQ
        ? (g11_x_req_chunk >= G11_CW'(1) && g11_x_req_chunk <= G11_CW'(G11_NCHUNKS))
        : 1'b1;
    reg signed [16*16-1:0] g11_gamma_in;    // COMBREAD FIX: registered BRAM read (gamma_final_buf)
    always_ff @(posedge ap_clk) begin
        g11_gamma_in <= (CONNECT_SEAMS && g11_g_inrange)
                        ? gamma_final_buf_rd(g11_g_eff_chunk) : '0;  // BUFSPLIT read
    end
    // FULL-H averaging uses MULT=43691 to convert the engine's /16 convention
    // to /3072. G9 Q3->Q8 makes input codes 32x and sumsq 1024x relative to the
    // prior 19/6 configuration. The +10 SUMSQ shift and +5 apply shift preserve
    // the same Q6.10 G11 output while retaining five more G9 fractional bits.
    localparam int G11_SUMSQ_SHR  = 29;
    localparam int G11_SCALE_SHR  = 11;
    localparam int G11_OUT_FRAC   = 10;
    localparam int G11_SCALE_WIDTH = 17;
    // -----------------------------------------------------------------
    // G11 START-GATE (R6 READ-BEFORE-WRITE FIX, 2026-05-27).
    // ROOT CAUSE (proven by $display timeline, NOT the pre_mlp net-width class):
    // the top_fsm pulses g11_start the instant g9_done (S_G9->S_G11), but the R6
    // seam (u_seam_r6) is filled by the R7 residual driver which only BEGINS
    // streaming g9down_buf into R6 ~hundreds of ns LATER (g9_done -> bk_g9_done_r
    // -> r7 a-phase 192cyc -> r7 b-phase 192cyc -> r6 writes). Trace evidence:
    //   g11 SUMSQ reads R6 @t=301,982-302,010 ns  -> seam_r6_rd_c = X (unwritten)
    //   r7 WRITES R6 slot0..5 @t=302,754-302,774 ns (≈770 ns too LATE)
    // so g11's sumsq accumulated X -> scale X/0 -> EVERY g11 output chunk 0
    // (g11_in_norm cos 0.0). The GEMV stages (G7g/G7u/G9) already gate on
    // bk_input_full (seam_premlp_full / seam_r5_full); g11 is the dedicated
    // rmsnorm_block and had NO analogous "R6 fully written" gate.
    // FIX: hold the FSM's 1-cycle g11_start as `g11_start_pending` until the R6
    // seam reports `full` (all 192 R7 writes landed), then issue ONE gated start
    // pulse g11_start_eff. Latched so the FSM pulse is never lost; cleared when the
    // gated start fires. Pure caller-side sequencing (no rmsnorm_block change, no
    // FSM-module change) -> the tb_s6 ARM-A bit-exact engine path is untouched.
    // Functional-only: lengthens the g11 wait (FSM stays in S_G11 for g11_done);
    // zero datapath/Q-format change. seam_r6_full = u_seam_r6.full (wired below).
    wire seam_r6_full;
    reg  g11_start_pending;
    wire g11_start_eff = CONNECT_SEAMS ? (g11_start_pending && seam_r6_full)
                                       : g11_start;   // isolation: legacy direct start
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n)               g11_start_pending <= 1'b0;
        else if (g11_start)          g11_start_pending <= 1'b1;   // latch FSM pulse
        else if (g11_start_eff)      g11_start_pending <= 1'b0;   // consumed at gated start
    end
    rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                    .SUMSQ_RSHIFT(G11_SUMSQ_RSHIFT),    // legacy fallback (unused: MULT!=0)
                    .SUMSQ_MULT(43691), .SUMSQ_SHR(G11_SUMSQ_SHR), .SCALE_SHR(G11_SCALE_SHR),
                    .RDATA_LAT(1), .SCALE_WIDTH(G11_SCALE_WIDTH)) u_rmsnorm_g11 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g11_start_eff),       // R6 READ-BEFORE-WRITE FIX: wait for seam_r6_full
        .x_chunk(g11_x_in),          // R6: to_logits/R7 output Q8.8 (or 0 in isolation)
        .gamma_chunk(g11_gamma_in),  // R6: gamma 1.0 Q4.12 (or 0 in isolation)
        .out_chunk(rms_g11_out_chunk), // g11hp Q6.10 frac=10 (#166); requantized <<2 at R8
        .out_valid(rms_g11_out_valid),
        .done(g11_done),
        .x_req_chunk(g11_x_req_chunk),       // S5B: R6 read index (full-H chunk request)
        .x_req_valid(g11_x_req_valid),       // S5B: high during SUMSQ/APPLY streaming
        .out_chunk_strobe(g11_out_strobe),   // S6: clean per-chunk strobe (apply_en+AP_LAT)
        .out_chunk_idx_o(g11_out_chunk_idx), // S6: that chunk's logical index 0..191
        .apply_en_o()                        // M5: G1 tap unused for g11 (R8 seam uses S6 strobe)
    );

    // ----------------------------------------------------------------
    // L2: silu_layer integration — drives g7p_done from silu.valid
    // Caller-side, FSM time-multiplexes over H_INTER/K_PAR=8192/16=512
    // cycles. For the integration-skeleton PDI, gate_chunk/up_chunk are
    // held at 0 (real values come from G7g/G7u GEMV outputs when fused
    // dataflow lands in L8). We pulse `en` on g7p_start; silu.valid one
    // cycle later (2-stage internal pipeline) latches g7p_done.
    // ----------------------------------------------------------------
    wire signed [16*16-1:0] silu_out_chunk;
    wire                    silu_valid;
    reg                     g7p_done_r;
    // S7b silu-sweep completion (driven in the silu en-FSM block far below; declared
    // here so g7p_done can reference it — xelab needs decl-before-use for regs).
    reg                     silu_sweep_done_r;
    // S5a: FULL-WIDTH R4 gate/up seams. G7g gate_proj + G7u up_proj GEMV
    // outputs (H_INTER=8192 = G7_NCHUNKS=512 chunks of K_PAR=16) STREAM into
    // two full-width stage_io_buffers; silu reads them CHUNK-INDEXED in lockstep
    // over all 512 chunks (no single-broadcast head-0 pilot). The seam reads
    // (seam_r4g_rd_c / seam_r4u_rd_c) are addressed by r4_silu_rd_idx, the
    // silu-feed chunk index. S7b drives the 512-pulse silu `en` + read pointer.
    localparam int G7_H_INTER = 8192;                 // gate/up intermediate width
    localparam int G7_NCHUNKS = G7_H_INTER / K_PAR;   // 512
    localparam int G7_CW      = $clog2(G7_NCHUNKS);   // 9
    wire [G7_CW-1:0] r4_silu_rd_idx;                  // silu-feed read pointer (driven below)
    // PDI#5-ASSEMBLY R4: G7g gate / G7u up GEMV outputs feed silu when
    // CONNECT_SEAMS=1 (seam_r4g_rd / seam_r4u_rd, captured below); otherwise
    // tied 0 (PDI#4 isolation). The seam reads are forward-declared in the
    // PDI#5-ASSEMBLY block. ONE driver per net (combinational mux).
    // SILU-Q612-FIX (2026-06-06): R4 gate/up seams widened 16->18 bit (Q4.12->Q6.12)
    // so the real gate(<=12.5)/up(<=18.0) activations no longer clip at +-8.0. The
    // packed bus is now K_PAR*18 = 288-bit (was 256). Output (silu->R5) stays 16-bit
    // Q8.8 — only the INPUT side of silu widens.
    // SILU-RANGE-FIX gate (FIX_SILU_RANGE.md, 2026-06-08): silu_layer_fix is
    // unit-gated bit-tight vs golden on the 125 seed-42 |gate|>8 channels
    // (tb_silu_range_gate: maxabs 6 LSB-Q8.8; legacy clip diverges 7001 LSB =
    // teeth). Now derived from the PORT parameter SILU_RANGE_FIX_P so the
    // full-chain frontier gate A/Bs it via -generic_top — SOURCE DEFAULT is OFF
    // (legacy) until the seed-42 RESCORED frontier is PROVEN UNCHANGED with it ON.
    // R4_AW (seam buffers / GEMV sat) tracks this so OFF re-introduces the ±8 clip
    // bit-for-bit (legacy silu_layer + sigmoid_pwl_q12, Q4.12 16-bit).
    localparam bit SILU_RANGE_FIX = SILU_RANGE_FIX_P;  // 1 = Q6.12 un-clipped, 0 = legacy ±8 clip (source default OFF)
    localparam int R4_AW = SILU_RANGE_FIX ? 18 : 16;  // Q6.12 (fix) / Q4.12 (legacy) element width
    wire signed [16*R4_AW-1:0] seam_r4g_rd_c;   // gate -> silu (connected arm, Q6.12)
    wire signed [16*R4_AW-1:0] seam_r4u_rd_c;   // up   -> silu (connected arm, Q6.12)
    wire signed [16*R4_AW-1:0] silu_gate_real = CONNECT_SEAMS ? seam_r4g_rd_c : '0;
    wire signed [16*R4_AW-1:0] silu_up_real   = CONNECT_SEAMS ? seam_r4u_rd_c : '0;
    // ---- PER-STAGE INJECT: G7p (silu, idx 8) — the KNOWN-BROKEN R4 seam ----
    // When inject_en[8] is set, the silu gate/up operands are sourced from the
    // golden inject buffer (gate = words[0..15], up = words[16..31], low R4_AW bits)
    // instead of the broken R4 seam read — so the captured silu output CRC isolates
    // the silu transform itself (golden in -> must produce 0xd8d6d147). This is the
    // GATE-B capability stage. inject_en/inj_bus are 0 when STAGE_INJECT=0 ->
    // silu_gate_in == silu_gate_real, bit-identical. (Declared as wires so the
    // generate-pruned inject path leaves these as the plain seam mux.)
    wire signed [16*R4_AW-1:0] inj_silu_gate;
    wire signed [16*R4_AW-1:0] inj_silu_up;
    genvar gis;
    generate
        for (gis = 0; gis < 16; gis = gis + 1) begin : g_inj_silu
            assign inj_silu_gate[gis*R4_AW +: R4_AW] = inj_bus[gis*32 +: R4_AW];
            assign inj_silu_up  [gis*R4_AW +: R4_AW] = inj_bus[(16+gis)*32 +: R4_AW];
        end
    endgenerate
    wire signed [16*R4_AW-1:0] silu_gate_in = inject_en[8] ? inj_silu_gate : silu_gate_real;
    wire signed [16*R4_AW-1:0] silu_up_in   = inject_en[8] ? inj_silu_up   : silu_up_real;
    // S7b drives the 512-pulse silu `en` (silu_en_eff) so silu sweeps all 512
    // gate/up chunks; BACK_SEQ=0 falls back to the single g7p_start pulse. The
    // en-FSM + r4_silu_rd_idx are declared in the BACK-HALF SEQUENCER block.
    wire silu_en_eff;
    // FIX-A (S7C_FRAC_CONFIG_AUDIT §4): OUT_FRAC=8 -> silu emits Q8.8 (g7p seam
    // frac=8). At the legacy Q4.12 the silu*up product (real ±91) saturated INT16
    // (373k >> 32767); Q8.8 spans ±128 so it no longer clips (golden g7_silu sat=0).
    // SILU-RANGE-FIX (2026-06-08): SILU_RANGE_FIX=1 -> silu_layer_fix (IN_W=18
    // Q6.12 gate/up, 512-entry sigmoid LUT over the FULL gate range, round-half
    // stage-A). Inputs un-clipped (real gate<=12.5/up<=18.0 no longer pinned to
    // ±8.0). Q8.8 output contract unchanged. SILU_RANGE_FIX=0 -> legacy
    // silu_layer (Q4.12 16-bit, 64-entry ±8 sigmoid LUT, ±8 gate clip) ONLY for
    // a forced timing fallback (see SILU_RANGE_FIX localparam note). Gate-proof:
    // FIX_SILU_RANGE.md (seed-42 |gate|>8 maxabs=6, frontier 8/8 == canonical).
    generate
    if (SILU_RANGE_FIX) begin : g_silu_fix
        silu_layer_fix #(.K_PAR(16), .IN_W(R4_AW), .OUT_FRAC(8)) u_silu (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .en(silu_en_eff),
            .gate_chunk(silu_gate_in),  // R4: G7g GEMV output Q6.12 (or 0 in isolation)
            .up_chunk(silu_up_in),      // R4: G7u GEMV output Q6.12 (or 0 in isolation)
            .out_chunk(silu_out_chunk), // Q8.8 (16-bit) -> R5 seam (unchanged contract)
            .valid(silu_valid)
        );
    end else begin : g_silu_legacy
        // LEGACY (R4_AW==16, Q4.12 ±8-clipping): re-introduces the seed-42
        // |gate|>8 corruption. Use ONLY if a future timing pass forces it.
        silu_layer #(.K_PAR(16), .OUT_FRAC(8)) u_silu (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .en(silu_en_eff),
            .gate_chunk(silu_gate_in),  // Q4.12 16-bit (clipped at the R4 seam)
            .up_chunk(silu_up_in),      // Q4.12 16-bit
            .out_chunk(silu_out_chunk), // Q8.8 (16-bit) -> R5 seam
            .valid(silu_valid)
        );
    end
    endgenerate
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) g7p_done_r <= 1'b0;
        else           g7p_done_r <= silu_valid;
    end
    // S7b: under BACK_SEQ g7p_done waits for the FULL 512-chunk silu sweep
    // (silu_sweep_done_r, pulsed on the 512th silu_valid in the en-FSM block);
    // the legacy pilot keeps the first-valid done. (silu_sweep_done_r forward-
    // declared in the silu en-FSM block.)
    wire g7p_done = BACK_SEQ ? silu_sweep_done_r : g7p_done_r;

    // ----------------------------------------------------------------
    // L1 + L5: g5 stage = RoPE → KV-append (placeholder) → attention.
    //
    // Substage FSM:
    //   G5_IDLE   : wait for g5_start
    //   G5_ROPE   : pulse rope_layer.start; wait for rope.valid
    //   G5_ATTN   : pulse attention_head_assembly.start; wait for attn.done
    //   G5_DONE   : pulse g5_done (1 cycle), back to IDLE
    //
    // For the integration-skeleton, kv_vec inputs to the attention head
    // are tied to 0 (real KV storage wires in during L8 fused dataflow).
    // The attention FSM still walks all 9 internal phases — proves the
    // chain end-to-end without needing the KV BD in place yet.
    //
    // Pilot D_HEAD=16 here (matches the ooc-synth pilot for L4c). The
    // production design will fan out 24 heads at D_HEAD=128, with this
    // sub-FSM either time-multiplexing across heads or replicating the
    // assembly per head — that scaling choice is L8/L9 territory.
    // ----------------------------------------------------------------
    localparam int G5_D_HEAD = 16;          // pilot; full = 128
    localparam int G5_CTX_MAX = 32;         // pilot; full = 2048

    wire signed [16*128-1:0] rope_vec_out;
    wire                     rope_valid;

    // M5: g5 now sequences TWO RoPE passes (Q then K) before attention, so
    // Q and K are rotated independently and V is routed UN-rotated. The pilot
    // captures head-0 (D_HEAD=16) of each of the three G4 sub-tiles (q/k/v).
    typedef enum logic [2:0] {
        G5_S_IDLE   = 3'd0,
        G5_S_ROPE_Q = 3'd1,   // rotate Q-tile → latch q_rotated, kick K rotate
        G5_S_ROPE_K = 3'd2,   // rotate K-tile → K-write + V-write, kick attn
        G5_S_ATTN   = 3'd3,
        G5_S_DONE   = 3'd4
    } g5_state_e;

    g5_state_e g5_state;
    logic      g5_rope_start;
    logic      g5_rope_sel;        // 0 = rotate Q-tile, 1 = rotate K-tile
    logic      g5_attn_start;
    logic      g5_kv_wr;           // 1-cycle KV K+V write strobe
    wire       g5_attn_done;
    reg        g5_done_r;

    // ----------------------------------------------------------------
    // M5 (L8a): G4 sub-tile capture — REAL QKV split.
    //
    // G4 is the concatenated QKV projection. The shared GEMV array runs
    // it as three sub-tiles (q_proj / k_proj / v_proj, each [*,3072]),
    // selected by the host-driven `tile_idx`. For the pilot we adopt the
    // convention tile_idx[1:0] = {0:Q, 1:K, 2:V}. Each sub-tile completes
    // with array_done while stage_sel==G4 (4'd0); on that edge we latch the
    // head-0 slice (first G5_D_HEAD=16 lanes, low 16 bits of each INT32
    // accumulator) into the matching capture buffer.
    //
    // Multi-driver-safe: each buffer is a single packed reg driven by ONE
    // always_ff (the whole vector latched at once). The capture write-enable
    // is decoded from tile_idx, so the three buffers never share a driver.
    // Production (M2/M3) replaces the head-0 slice with full D_HEAD=128 and
    // the INT32→Q4.12 saturate, and sequences tile_idx 0..23 per projection.
    // ----------------------------------------------------------------
    localparam int G4_STAGE_SEL = 4'd0;
    localparam int G4_OUT       = 3072;                        // per sub-tile (Q_OUT)
    localparam int G4_NTILES    = (G4_OUT + LANES - 1) / LANES;// 24 col-tiles/sub
    localparam int G4_TOT_TILES = 3 * G4_NTILES;               // 72 concat local cols
    localparam int G4_TW        = $clog2(G4_NTILES);           // col-tile index width

    // Live 128-lane INT32 array output (the whole G4 column-tile this done).
    wire signed [32*LANES-1:0] g4_tile_slice = array_acc_out;

    // Forward declarations (the GEMV array + stage hold-flags are wired
    // further down; the M5 capture decode references them here).
    // NOTE: `array_done` is now declared up in the M6b tree-tiler region
    // (the tiler consumes it as its tile-valid handshake); only g4_start_d
    // is declared here to avoid a duplicate net.
    reg  g4_start_d;
    // S7a: forward-declare the array column-tile index (driven by the per-tile
    // sequencer / tree tiler / host mux further down). The G4 capture decode
    // below keys on it, so it must be declared before first use (VRFC 10-3380).
    wire [15:0] array_tile_idx;

    // S7a full-width capture decode: flat concat-col index (0..71) -> sub-tile
    // + col-tile. Keyed on `array_tile_idx` (= the per-tile sequencer's
    // g4_seq_tile_idx under S7A_FRONT_SEQ, held STABLE the whole in-flight
    // window, or the host tile_idx in the legacy pilot) so array_done(N) writes
    // the correct q/k/v_full_buf[col_tile(N)] slot.
    wire             g4_tile_done = array_done & g4_start_d & (stage_sel == G4_STAGE_SEL);
    wire [15:0]      g4_lcol      = array_tile_idx;                 // 0..71 concat col
    wire [1:0]       g4_subtile   = (g4_lcol < 16'(G4_NTILES))     ? 2'd0   // Q
                                  : (g4_lcol < 16'(2*G4_NTILES))   ? 2'd1   // K
                                  : (g4_lcol < 16'(G4_TOT_TILES))  ? 2'd2   // V
                                  :                                  2'd3;  // OOB
    // exact col-tile index (G4_NTILES=24 is not a power of two): subtract base.
    wire [15:0]      g4_subbase   = (g4_subtile == 2'd0) ? 16'd0
                                  : (g4_subtile == 2'd1) ? 16'(G4_NTILES)
                                  :                        16'(2*G4_NTILES);
    wire [G4_TW-1:0] g4_ctile     = (g4_lcol - g4_subbase);        // 0..23
    wire             cap_q_we     = g4_tile_done & (g4_subtile == 2'd0);
    wire             cap_k_we     = g4_tile_done & (g4_subtile == 2'd1);
    wire             cap_v_we     = g4_tile_done & (g4_subtile == 2'd2);

    // Three FULL-WIDTH capture buffers: G4_NTILES slots of 128 INT32 lanes
    // each (== 3072 INT32 elements per sub-tile).
    reg signed [32*LANES-1:0] q_full_buf [0:G4_NTILES-1];
    reg signed [32*LANES-1:0] k_full_buf [0:G4_NTILES-1];
    reg signed [32*LANES-1:0] v_full_buf [0:G4_NTILES-1];
    always_ff @(posedge ap_clk) begin
        if (cap_q_we) q_full_buf[g4_ctile] <= g4_tile_slice;
        if (cap_k_we) k_full_buf[g4_ctile] <= g4_tile_slice;
        if (cap_v_we) v_full_buf[g4_ctile] <= g4_tile_slice;
    end

    // Head-0 (D_HEAD=16) views of the full buffers — preserve the existing
    // RoPE/attention head-0 pilot consumers (head-0 = col-tile 0, lanes
    // 0..G5_D_HEAD-1, low 16 bits of each INT32 accumulator). The downstream
    // head-0 RoPE/KV path is unchanged by the full-width capture (and is bypassed
    // when S7A_FRONT_SEQ=1 — the 24-head attn splice owns G5 then).
    wire signed [16*G5_D_HEAD-1:0] q_tile_buf;
    wire signed [16*G5_D_HEAD-1:0] k_tile_buf;
    wire signed [16*G5_D_HEAD-1:0] v_tile_buf;
    genvar gq;
    generate
        for (gq = 0; gq < G5_D_HEAD; gq = gq + 1) begin : g_g4_head0_view
            assign q_tile_buf[gq*16 +: 16] = q_full_buf[0][gq*32 +: 16];
            assign k_tile_buf[gq*16 +: 16] = k_full_buf[0][gq*32 +: 16];
            assign v_tile_buf[gq*16 +: 16] = v_full_buf[0][gq*32 +: 16];
        end
    endgenerate

    // ----------------------------------------------------------------
    // M5 (L8a cont): single RoPE engine, time-multiplexed across Q then K.
    // Pass 1 rotates the Q-tile (g5_rope_sel=0), pass 2 the K-tile
    // (g5_rope_sel=1). V is NEVER routed through RoPE. The rotated Q is
    // latched (q_rotated_buf) for the attention query; the rotated K drives
    // the K-cache write on the same cycle as the un-rotated V drives the
    // V-cache write.
    // ----------------------------------------------------------------
    wire signed [16*G5_D_HEAD-1:0] rope_in_head0 =
        g5_rope_sel ? k_tile_buf : q_tile_buf;

    // rope_layer is D_HEAD=128 wide; the pilot only uses head-0 (low 16
    // elements). Zero-extend the unused high lanes (they fall outside the
    // head-0 slice we consume on the output side).
    wire signed [16*128-1:0] rope_vec_in =
        {{(16*(128-G5_D_HEAD)){1'b0}}, rope_in_head0};

    rope_layer #(
        .D_HEAD(128), .N_PAIRS(64), .MAX_CTX(2048), .A_WIDTH(16),
        .ROPE_CS_PIPE(ROPE_CS_PIPE), .ROPE_POS_REP(ROPE_POS_REP)   // RTL WO#8
    ) u_rope (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g5_rope_start),
        .pos(kv_append_pos),        // KV_WRPOS_FIX: rotate at new-token index (ctx_len-1)
        .vec_in(rope_vec_in),       // M5: Q-tile (pass1) or K-tile (pass2)
        .vec_out(rope_vec_out),
        .valid(rope_valid)
    );

    // rope_vec_out head-0 slice (the rotated D_HEAD=16 vector this pass).
    wire signed [16*G5_D_HEAD-1:0] rope_out_head0 = rope_vec_out[16*G5_D_HEAD-1:0];

    // Latch the rotated Q (pass 1) as the attention query.
    reg signed [16*G5_D_HEAD-1:0] q_rotated_buf;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n)                              q_rotated_buf <= '0;
        else if (g5_state == G5_S_ROPE_Q && rope_valid)
                                                    q_rotated_buf <= rope_out_head0;
    end

    // ----------------------------------------------------------------
    // M5 (L8b): K + V caches for head-0 (pilot D_HEAD=16, MAX_CTX=32).
    //   K-write : ROTATED K  (rope_vec_out head-0, captured in pass 2)
    //   V-write : RAW V       (v_tile_buf — the REAL G4 V sub-tile, NO RoPE)
    // Both write at wr_pos = kv_append_pos (= ctx_len-1, the NEW-TOKEN INDEX) on
    // the single g5_kv_wr strobe so K[pos] and V[pos] are appended together.
    // KV_WRPOS_FIX (2026-06-06): was wr_pos=ctx_len (window length); shared signal
    // now used at every KV write site so the convention can't fork.
    // ----------------------------------------------------------------
    wire        u_attn_rd_en;
    wire        u_attn_rd_kind;
    wire [10:0] u_attn_rd_pos;
    wire signed [16*G5_D_HEAD-1:0] k_vec_from_cache;
    wire signed [16*G5_D_HEAD-1:0] v_vec_from_cache;

    // K write — ROTATED K (rope output during pass 2)
    wire signed [16*G5_D_HEAD-1:0] k_write_vec = rope_out_head0;
    // V write — REAL V tile (un-rotated). This is the M5 fix: no longer the
    // K slice / live array output — it is the captured G4 V sub-tile.
    wire signed [16*G5_D_HEAD-1:0] v_write_vec = v_tile_buf;

    kv_head_wide #(.D_HEAD(G5_D_HEAD), .A_WIDTH(16), .MAX_CTX(G5_CTX_MAX)) u_kv_k_head0 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_en(g5_kv_wr),
        .wr_pos(kv_append_pos),               // KV_WRPOS_FIX: new-token index (ctx_len-1)
        .wr_vec(k_write_vec),
        .rd_en(u_attn_rd_en & ~u_attn_rd_kind),     // kind=0 → K
        .rd_pos(u_attn_rd_pos),
        .rd_vec(k_vec_from_cache)
    );

    kv_head_wide #(.D_HEAD(G5_D_HEAD), .A_WIDTH(16), .MAX_CTX(G5_CTX_MAX)) u_kv_v_head0 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_en(g5_kv_wr),                            // co-write with K, real V
        .wr_pos(kv_append_pos),               // KV_WRPOS_FIX: new-token index (ctx_len-1)
        .wr_vec(v_write_vec),
        .rd_en(u_attn_rd_en & u_attn_rd_kind),       // kind=1 → V
        .rd_pos(u_attn_rd_pos),
        .rd_vec(v_vec_from_cache)
    );

    wire signed [32*G5_D_HEAD-1:0] g5_attn_out_vec;
    attention_head_assembly #(
        .D_HEAD(G5_D_HEAD), .A_WIDTH(16), .CTX_MAX(G5_CTX_MAX)
    ) u_attn_head (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g5_attn_start),
        .ctx_len(ctx_len[10:0]),
        .q_vec(q_rotated_buf),           // M5: rotated Q (pass 1 latch)
        .kv_rd_en(u_attn_rd_en),         // L8b: live KV read port
        .kv_rd_kind(u_attn_rd_kind),
        .kv_rd_pos(u_attn_rd_pos),
        .k_vec_in(k_vec_from_cache),
        .v_vec_in(v_vec_from_cache),
        .out_vec(g5_attn_out_vec),
        .done(g5_attn_done)
    );

    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            g5_state      <= G5_S_IDLE;
            g5_rope_start <= 1'b0;
            g5_rope_sel   <= 1'b0;
            g5_attn_start <= 1'b0;
            g5_kv_wr      <= 1'b0;
            g5_done_r     <= 1'b0;
        end else begin
            // pulse defaults
            g5_rope_start <= 1'b0;
            g5_attn_start <= 1'b0;
            g5_kv_wr      <= 1'b0;
            g5_done_r     <= 1'b0;
            case (g5_state)
                G5_S_IDLE: begin
                    if (g5_start) begin
                        // Q/K/V tiles already captured during G4. Kick RoPE
                        // pass 1 = rotate the Q-tile.
                        g5_rope_sel   <= 1'b0;
                        g5_rope_start <= 1'b1;
                        g5_state      <= G5_S_ROPE_Q;
                    end
                end
                G5_S_ROPE_Q: begin
                    // rope_valid → rotated Q is on rope_vec_out this cycle
                    // (latched into q_rotated_buf by its own always_ff). Kick
                    // RoPE pass 2 = rotate the K-tile.
                    if (rope_valid) begin
                        g5_rope_sel   <= 1'b1;
                        g5_rope_start <= 1'b1;
                        g5_state      <= G5_S_ROPE_K;
                    end
                end
                G5_S_ROPE_K: begin
                    // rope_valid → rotated K is on rope_vec_out this cycle.
                    // Strobe the KV write: K-cache <= rotated K, V-cache <=
                    // RAW V (v_tile_buf). Then start attention.
                    if (rope_valid) begin
                        g5_kv_wr      <= 1'b1;
                        g5_attn_start <= 1'b1;
                        g5_state      <= G5_S_ATTN;
                    end
                end
                G5_S_ATTN: begin
                    if (g5_attn_done) begin
                        g5_done_r <= 1'b1;
                        g5_state  <= G5_S_DONE;
                    end
                end
                G5_S_DONE: begin
                    g5_state <= G5_S_IDLE;
                end
                default: g5_state <= G5_S_IDLE;
            endcase
        end
    end


    // Config registers exposed from control_regs (unused for now, will route
    // to layer modules in subsequent ticks)
    wire [31:0] ctx_len, num_frontiers, max_depth;
    wire [63:0] out0_base, out1_base, out2_base, hbm_emb_base;
    wire [63:0] lmfull_hbm_base;   // G8 SPLICE: 788 MB lm_head_full HBM byte base (ctrl 0x14/0x1c)

    // ----------------------------------------------------------------
    // S7a 24-head attention splice (relocated below the ctx_len/config-reg
    // declarations so it can reference ctx_len — the splice consumes the
    // config-reg ctx_len for the attention window). Replaces the head-0 pilot.
    // ----------------------------------------------------------------
    // ================================================================
    // S7a — FULL 24-head attention splice (replaces the head-0 D_HEAD=16 pilot)
    // ================================================================
    // S4-verified chain (forks/pdi6_s4, Linf=1 / 0-over-2-LSB vs the GEMV-native
    // g5 reference): g4_qkv_seam (q/k/v_full_buf frac13 INT32 -> frac10 INT16,
    // round-half >>3 + sat) -> attention_full_mha (24 heads, D_HEAD=128, ctx_len
    // driven by the config reg with a >=1 floor, do_kv_fill=1). The 3072 INT16
    // attn-out is then STREAMED into the G5->G6 seam (a full-H stage_io_buffer)
    // with a CLEAN per-chunk write strobe (NOT out_valid), and g5_done fires only
    // after the stream-out completes (full-depth drain, H3). G6 reads this seam
    // chunk-indexed when S7A_FRONT_SEQ=1; the pilot L8c island is bypassed.
    // ----------------------------------------------------------------
    localparam int S7A_NHEADS = 24;
    localparam int S7A_DHEAD  = 128;
    localparam int S7A_H      = S7A_NHEADS * S7A_DHEAD;     // 3072
    localparam int S7A_OUT_W  = S7A_H * A_WIDTH;
    localparam int S7A_ROW_W  = S7A_DHEAD * A_WIDTH;
    localparam int S7A_NCH    = S7A_H / K_PAR;              // 192 chunks
    localparam int S7A_CW     = (S7A_NCH > 1) ? $clog2(S7A_NCH) : 1;

    // ----------------------------------------------------------------
    // LANES-256 — PER-HEAD VIEW of the G4 capture buffers.
    //
    // attention_full_mha_sbuf (and the legacy g4_qkv_seam) consume the raw
    // Q/K/V as [0:N_HEADS-1] slots of 32*D_HEAD bits. At LANES=128 the G4
    // capture buffer q_full_buf[G4_NTILES=24][32*128] happened to BE that
    // shape (D_HEAD == LANES — the 128-lane coincidence the study's §4.4
    // coupling grep missed). At LANES=256 the capture is [12][32*256]: each
    // tile slot holds TWO heads. This re-slice is pure wiring — head h lives
    // in capture slot (h*D_HEAD)/LANES at bit offset ((h*D_HEAD)%LANES)*32 —
    // and at LANES=128 it degenerates to the identity (slot=h, ofs=0), so
    // the default netlist is unchanged. D_HEAD must divide LANES (128/256
    // both OK; guarded at elaboration below).
    // ----------------------------------------------------------------
    wire signed [32*S7A_DHEAD-1:0] q_head_view [0:S7A_NHEADS-1];
    wire signed [32*S7A_DHEAD-1:0] k_head_view [0:S7A_NHEADS-1];
    wire signed [32*S7A_DHEAD-1:0] v_head_view [0:S7A_NHEADS-1];
    generate
        for (genvar hv = 0; hv < S7A_NHEADS; hv = hv + 1) begin : g_head_view
            localparam int HV_SLOT = (hv * S7A_DHEAD) / LANES;
            localparam int HV_OFS  = ((hv * S7A_DHEAD) % LANES) * 32;
            assign q_head_view[hv] = q_full_buf[HV_SLOT][HV_OFS +: 32*S7A_DHEAD];
            assign k_head_view[hv] = k_full_buf[HV_SLOT][HV_OFS +: 32*S7A_DHEAD];
            assign v_head_view[hv] = v_full_buf[HV_SLOT][HV_OFS +: 32*S7A_DHEAD];
        end
    endgenerate

    // ctx_len floor: attention needs >=1 cached position (the just-appended token).
    wire [10:0] s7a_ctx_len_reg = (ctx_len[10:0] == 11'd0) ? 11'd1 : ctx_len[10:0];

`ifdef SIM_CTX_FORCE
    // =====================================================================
    // GATE-ONLY ctx override — F3 KV-residency HOLES gate.
    //
    // The full-chain TB writes reg 0x10 = 1 unconditionally, so there is no
    // host-side way to run the seed-42 stimulus with a LONGER attention
    // window than the number of KV rows the RTL self-appended. -d
    // SIM_CTX_FORCE=<n> elaborates the attention window to <n> while the KV
    // store still receives exactly ONE write (at kv_append_pos = n-1) --
    // i.e. a window with n-1 HOLES, which is precisely the phase-C stimulus
    // that produced the historical ctx=4 "ordered 2/8" silicon result.
    //
    // It overrides ONLY the attention window; every other consumer of the
    // ctx_len register is untouched, so the change is confined to G5.
    //
    // CAN NEVER SHIP: the packaging tcl clears verilog_define (config header
    // section [E]), and if this ever DID reach synthesis the guard below is
    // an unresolved-module elaboration ERROR rather than a silent strap.
    // =====================================================================
 `ifdef SYNTHESIS
    sim_ctx_force_must_never_be_synthesised u_sim_ctx_force_guard();
 `endif
    wire [10:0] s7a_ctx_len = 11'(`SIM_CTX_FORCE);
    initial $display("[KVRES] SIM_CTX_FORCE: attention window forced to ctx_len=%0d (reg 0x10 ignored by G5 only)",
                     11'(`SIM_CTX_FORCE));
`else
    wire [10:0] s7a_ctx_len = s7a_ctx_len_reg;
`endif

    // KV_WRPOS_FIX (2026-06-06, propagated from pdi6_s7c_g8): ONE shared KV write
    // position so the convention can't fork between the pilot KV stacks, the pilot
    // rope, and the S7a 24-head splice. New token written/rotated at the NEW-TOKEN
    // INDEX = ctx_len-1 (matching eagle4_ref rope/kv_append(position) + read window
    // [0,ctx_len)). Guarded for the ctx_len==0 floor like s7a_ctx_len. (NOTE: in
    // pdi6 the all-zero-G5 root cause was a TB scale-backdoor K/V collision, NOT
    // this; this is convention-hardening, bit-neutral on the active S7a path.)
    wire [10:0] kv_append_pos = s7a_ctx_len - 11'd1;   // (ctx_len==0)?0:(ctx_len-1)

    // ---- DEPTH-N level-loop bus (declared here: the KV lane-chain addressing
    // below consumes branch_j_w/level_i_w, and top_fsm is instantiated later
    // in the file). At DEPTH_N=0 every one of these is inert.
    wire                     lvl_start_w;
    wire [2:0]               branch_j_w;
    wire [3:0]               level_i_w;
    // F4 (FR_KF_BANKS>1): the frontier SLOT (0..7) of the parent of the branch
    // about to run — i.e. which of the previous level's 8 body passes produced
    // this survivor (chain.cpp:483 `parent_branch = ord[s]/K_F`). This is the
    // `a`-bank the child re-gathers its for_reasoning seed FROM. Driven from
    // tree_level_fsm's committed surv_parent snapshot inside g_depth_n; tied 0
    // when DEPTH_N=0. fr_par_node_w is the parent's POOL index: its low 3 bits
    // are the slot ONLY while the parents are level-0 nodes (depth<=2). It is
    // kept because the G3 RED arm (`G3_RED_POOLLOW`) reinstates that
    // pre-inversion mapping on purpose.
    wire [tree_score_pkg::NODE_W-1:0] fr_par_node_w;
    // G3 (2026-07-23): the SAME parent as an actual frontier SLOT, published by
    // tree_level_fsm (surv_parent_slot_o) -- exact at EVERY depth, because the
    // FSM stores the slot in lockstep with the pool index at the one place both
    // are known (the scored-candidate sink). THIS is what F4's rd_branch uses
    // now. Tied 0 at DEPTH_N=0 (g_no_depth_n), so depth-0 is unchanged.
    wire [tree_score_pkg::SLOT_W-1:0] fr_par_slot_w;
    wire                     branch_feed_w;
    wire                     out_walk_start_w;
    // Fed back INTO the FSM; driven only inside the DEPTH_N generate block.
    wire                     lvl_level_done_w;
    wire                     lvl_level_last_w;   // co-timed "that was the LAST level"
    wire                     lvl_tree_done_w;
    wire                     lvl_out_walk_done_w;

    // =====================================================================
    // DEPTH-N KV LANE-CHAIN addressing (all three fold to the legacy value
    // when DEPTH_N=0 or level_i==0, so depth-0 is bit-identical).
    //
    // Level 0 IS the legacy single body pass: it appends at ctx_len-1 and
    // sweeps [0, ctx_len). Levels >=1 are the lane-chain:
    //
    //   slot (j,i)   = ctx_len + j*num_steps + (i-1)   <- branch j's own column
    //   sweep (i)    = ctx_len + i                     <- prefix + own tokens
    //   angle (0)    = ctx_len - 1                     <- current root token
    //   angle (i>0)  = ctx_len + i                     <- child body position
    //
    // The slot/angle split is the whole reason ROPE_POS_DECOUPLE exists: the
    // slot is per-branch, the angle is NOT. Conflating them (as the shipped
    // block does, correctly, at depth 0) would rotate branch 7 at ctx_len+56.
    //
    // num_steps == max_depth, matching SGLang spec_utils.py:290
    //   token_pool_ptr + seq_len + topk_id*num_steps + arange(iter_upper)
    // =====================================================================
    wire        lane_active   = DEPTH_N && (level_i_w != 4'd0);
    wire [10:0] lane_num_step = max_depth[10:0];
    reg         s7a_attn_start;
    // DUAL-SOURCE KILL (F3 report §5): the write SLOT now comes from the SAME
    // single-source package remap the READ side uses (kv_lane_chain_pkg), so
    // write and read can never silently diverge (and the KVLANE_RED_STRIDE RED
    // arm poisons both together). Only valid under lane_active (level_i_w != 0),
    // exactly as before. Behavior-neutral vs the old open-coded
    //   s7a_ctx_len + branch_j_w*num_steps + (level_i_w - 1)
    // at the default straps (no KVLANE_RED_STRIDE): the pkg computes
    //   ctx_len_prefix + lane_j*num_steps + level_i, level_i = (level_i_w - 1).
    wire [10:0] lane_slot     = 11'(kv_lane_chain_pkg::kv_lane_slot(
                                    {1'b0, s7a_ctx_len}, {1'b0, branch_j_w},
                                    {1'b0, lane_num_step}, 12'(level_i_w) - 12'd1));

    wire [10:0] kv_append_pos_eff = lane_active ? lane_slot     : kv_append_pos;
    wire [10:0] s7a_ctx_len_eff   = lane_active ? (s7a_ctx_len + 11'(level_i_w))
                                                : s7a_ctx_len;
    wire [10:0] rope_pos_eff      = 11'(kv_lane_chain_pkg::kv_lane_rope_pos(
                                                {1'b0, s7a_ctx_len},
                                                12'(level_i_w)));

`ifndef SYNTHESIS
    // CTX_WIN truncation is SILENT in kv_head_wide_win (wr_pos[POS_W-1:0]), so
    // an over-range slot would alias instead of failing. Catch it in sim.
    always_ff @(posedge ap_clk)
        if (ap_rst_n && lane_active)
            assert (kv_append_pos_eff < 11'(CTX_WIN))
                else $error("KV lane-chain slot %0d >= CTX_WIN %0d (ctx_len=%0d j=%0d i=%0d ns=%0d)",
                            kv_append_pos_eff, CTX_WIN, s7a_ctx_len,
                            branch_j_w, level_i_w, lane_num_step);

`endif

    // FRAC FIX: the captured q/k/v_full_buf carries the GEMV ACTIVATION frac, NOT
    // the scale frac. requant_int32 = round(result_q >> SCALE_FRAC) strips the
    // per-group scale (frac = SCALE_FRAC + ACT_FRAC) back down to frac = ACT_FRAC.
    // So the g4_qkv_seam shifts by (ACT_FRAC - G5_FRAC).
    //
    // C1-REVERT (synthetic-vs-real correction, 2026-05-27): the R1 seam now
    // passes G1's frac=12 output VERBATIM (no `<<1`). The earlier FRAC_CONTRACT
    // `<<1`->frac13 was derived against a SYNTHETIC frac13 golden; the REAL
    // eagle4_ref uses attn_in=12 (the #190-validated contract). The residfrac4
    // real-input guard (N=2200) was decisive: attn_in=13 (R1 `<<1`) is the
    // WEAKEST contract on real std~=0.93 activations (rank0 91.77% — it
    // saturates), while attn_in=12 (verbatim R1, #190) is correct (93-95.6%).
    // The "verbatim = 2x too small at G4" claim was vs the WRONG (frac13)
    // golden; against the real attn_in=12 contract verbatim is exactly right
    // (G1-out frac12 -> seam frac12 -> g4_qkv_seam shifts 12-10 = 2, >>2).
    localparam int ACT_FRAC = PRECISION_SAFE ? 11 : 12;
    // g4_qkv_seam: the captured full-width Q/K/V (q/k/v_full_buf, 24 col-tiles of
    // 128 INT32 lanes) requanted frac(ACT_FRAC=13)->frac10 into flattened INT16
    // attn inputs (>>3 round+sat).
    wire signed [S7A_OUT_W-1:0] s7a_q_in, s7a_k_in, s7a_v_in;
    // SPEEDWAVE-V2 W3-F2: the parallel seam is generate-guarded away when
    // REQUANT_HEADMUX=1 — the requant lives behind the head mux inside
    // attention_full_mha_sbuf (RAW_REQUANT=1). Mirrors jewel
    // attn_stage_wrap.sv g_seam_requant/g_seam_bypass (J5B commit 096fc74).
    generate if (!REQUANT_HEADMUX) begin : g_seam_requant
        // legacy: the full parallel requant (9,216 rsh_sat units — the ~440K LUT)
        g4_qkv_seam #(
            .N_HEADS(S7A_NHEADS), .D_HEAD(S7A_DHEAD), .A_WIDTH(A_WIDTH),
            .G4_FRAC(ACT_FRAC), .G5_FRAC(10)
        ) u_g4_qkv_seam (
            // LANES-256: per-head re-slice (identity at LANES=128)
            .q_full_buf(q_head_view), .k_full_buf(k_head_view), .v_full_buf(v_head_view),
            .q_in(s7a_q_in), .k_in(s7a_k_in), .v_in(s7a_v_in)
        );
    end else begin : g_seam_bypass
        // W3-F2: INT16 vectors unused — synth prunes the dead INT16 mux arm
        // inside the attention (g_head_int16 not elaborated at RAW_REQUANT=1).
        assign s7a_q_in = '0;
        assign s7a_k_in = '0;
        assign s7a_v_in = '0;
    end endgenerate

    wire                        s7a_attn_done;
    wire signed [S7A_OUT_W-1:0] s7a_attn_out_vec;
    // F3 KV-residency count (see the .kv_res_cnt() connection below).
    (* keep = "true" *) wire [11:0] s7a_kv_res_cnt;

`ifndef SYNTHESIS
    // The ctx_len <-> residency COUPLING the reference gets structurally
    // (reference/src/attn.cpp:53 -- ctx_len is incremented BY kv_append, so an
    // unwritten position cannot exist) and this RTL never had. Reported, not
    // enforced: the F3 mask already neutralises the holes; this line exists so
    // a ctx>1 run can never again be mis-read as a datapath verdict when the
    // window was simply longer than the KV the RTL had written.
    logic kvres_warned;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) kvres_warned <= 1'b0;
        // Sampled at attn_done (all 24 heads have self-appended by then, so
        // res_cnt is settled for this forward; sampling at attn_start would
        // race the very first append and fire on every healthy run).
        else if (s7a_attn_done && !kvres_warned &&
                 ({1'b0, s7a_ctx_len_eff} > s7a_kv_res_cnt)) begin
            kvres_warned <= 1'b1;
            $display("[KVRES] DECOUPLED t=%0t: attention sweep=%0d > KV rows actually written=%0d (reg 0x10 ctx_len=%0d) — %0d slot(s) are HOLES and are masked to -INF",
                     $time, s7a_ctx_len_eff, s7a_kv_res_cnt, ctx_len[10:0],
                     {1'b0, s7a_ctx_len_eff} - s7a_kv_res_cnt);
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge ap_clk)
        if (ap_rst_n && s7a_attn_start) begin
            assert (!$isunknown({level_i_w, rope_pos_eff, s7a_ctx_len_eff}))
                else $error("KV lane-chain RoPE coordinate contains X/Z");
            if (lane_active)
                assert (rope_pos_eff == s7a_ctx_len_eff)
                    else $error("KV child RoPE early/late: rope=%0d sweep=%0d level=%0d",
                                rope_pos_eff, s7a_ctx_len_eff, level_i_w);
            else
                assert ((rope_pos_eff + 11'd1) == s7a_ctx_len_eff)
                    else $error("KV root RoPE mismatch: rope=%0d sweep=%0d",
                                rope_pos_eff, s7a_ctx_len_eff);
        end
`endif
`endif

    // SPEEDWAVE FIX-SBUF (EXP_G5_CTXSCAN, 2026-07-11): attention_full_mha_sbuf
    // is the base attention_full_mha with the COMBINE_LOCAL=1 perbank assembly
    // swapped for attention_head_assembly_perbank_sbuf: the pass-1 score_norm
    // is buffered in score_buf (32b x CTX_MAX = 1 RAMB36) and the EXP pass
    // becomes a 3-state buffered read — ZERO K re-reads, ZERO dot recompute.
    // Score arithmetic + MAX-pass cadence + running_max order UNTOUCHED;
    // bit-exact A/B 0/18,432 mismatches at ctx {8,64,256,620,1024,2047}
    // (reports/results/EXP_G5_CTXSCAN_RESULT.md). G5 law 888*ctx -> 552*ctx
    // (1.609x). Module swap (not a strap): keeps instance name/hierarchy for
    // TB XMRs; the wrapper is a pure rename of the base otherwise (diff-proven
    // at assembly). Netlist gate: score_buf MUST survive as RAMB (1 RAMB36,
    // not LUTRAM — [Synth 8-6849] check in the speedwave OOC gate).
    attention_full_mha_sbuf #(
        .N_HEADS(S7A_NHEADS), .D_HEAD(S7A_DHEAD), .A_WIDTH(A_WIDTH),
        .CTX_WIN(CTX_WIN), .HEADS_PER_STACK(4), .CTX_MAX(1024),   // FIX-G OPTION-B: CTX_WIN top generic (default 1024)
        .SCORE_LATENCY(13), .SCORE_NORM_SHR(11),
        .K_RAM_STYLE(KV_K_RAM_STYLE), .V_RAM_STYLE(KV_V_RAM_STYLE),  // WO-053: V→BRAM
        .COMBINE_LOCAL(COMBINE_LOCAL),                               // WO-051: per-bank combine select
        // SPEEDWAVE-V2 W3-F2: requant relocation (RAW_REQUANT tracks the top
        // strap; the frac pair is the SAME C1-REVERT contract the seam used).
        .RAW_REQUANT(REQUANT_HEADMUX), .G4_FRAC(ACT_FRAC), .G5_FRAC(10),
        .ROPE_POS_DECOUPLE(DEPTH_N),
        .HEAD_PAR(HEAD_PAR),                 // Stage-A: G5 head-parallelism (1=serial default, 6=6-way)
        .ROPE_CS_PIPE(ROPE_CS_PIPE), .ROPE_POS_REP(ROPE_POS_REP),   // RTL WO#8: rope cone pipe
        .KV_ADDR_REP(KV_ADDR_REP),          // RTL WO#12: V bank-group address split
        .RES_CLR_REP(KV_RES_CLR_REP)        // F3-CLR: residency-clear fan-out split
    ) u_attn_full (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(s7a_attn_start),
        .ctx_len(s7a_ctx_len_eff),
        .append_pos(kv_append_pos_eff),         // KV_WRPOS_FIX / DEPTH-N lane-chain slot
        .rope_pos_in(rope_pos_eff),             // DEPTH-N: ANGLE, decoupled from the slot
        // -----------------------------------------------------------------
        // F3 KV lane-chain READ remap coordinates (F3 report §5, the F2-side
        // top-level diff that closes the depth-N read seam). KV_LANE_CHAIN
        // follows ROPE_POS_DECOUPLE(DEPTH_N) by parameter default, so NO param
        // edit is needed. Sources: branch_j_w/level_i_w are top_fsm outputs
        // (output reg, reset to 0), so at DEPTH_N=0 these carry DEFINED 0 (the
        // contiguous-from-0 / identity coords) -- no $isunknown can fire, and
        // the KV_LANE_CHAIN=0 arm ties the assembly lane ports to '0 so they
        // never reach live logic (depth-0 byte-identical). At DEPTH_N=1 the
        // real lane coords flow, replacing the X/Z that made the combined run
        // $fatal ("lane inputs are X/Z -- top-level lane wiring missing").
        // -----------------------------------------------------------------
        .ctx_len_prefix(s7a_ctx_len),           // SHARED prefix length (>=1)
        .lane_j({1'b0, branch_j_w}),            // branch / topk id  (3b -> 4b)
        .num_steps(max_depth[10:0]),            // SGLang num_steps == max_depth
        .level_i({7'd0, level_i_w}),            // level within the lane (0 = root pass => identity)
        .done(s7a_attn_done),
        .q_in(s7a_q_in), .k_in(s7a_k_in), .v_in(s7a_v_in),
        // W3-F2 RAW INT32 tiles (the LIVE path at RAW_REQUANT=1).
        // LANES-256: per-head re-slice (identity at LANES=128).
        .q_raw_buf(q_head_view), .k_raw_buf(k_head_view), .v_raw_buf(v_head_view),
        // =================================================================
        // DEPTH-N KV LANE-CHAIN write side.
        //
        // do_kv_fill stays 1: the attention FSM self-appends the freshly
        // RoPE'd K and raw V at `append_pos`. So the lane-chain does NOT need
        // the pl_* preload port for its own writes — it only needs
        // `append_pos` to carry the lane-chain SLOT instead of ctx_len-1.
        // That keeps ONE writer (the FSM), which matters because pl_we and
        // the FSM's T_APPEND share k_wr_*/v_wr_* with NO arbiter: asserting
        // both would let T_APPEND's later non-blocking assignment silently
        // drop the preload write.
        //
        // pl_we therefore remains tied 0 here BY DESIGN, not by omission.
        // The port stays available for a future host-driven KV preload; it is
        // exercised today only by top/attn_unit_top.sv.
        // =================================================================
        .pl_we(1'b0), .pl_head('0), .pl_pos('0),
        .pl_k_vec('0), .pl_v_vec('0), .do_kv_fill(1'b1),
        .out_vec(s7a_attn_out_vec), .dbg_head(),
        // F3 KV-RESIDENCY: # of DISTINCT KV slots the RTL has actually
        // written this session. `ctx_len` (reg 0x10) is a HOST assertion with
        // no coupling to this number -- that decoupling is what made every
        // ctx>1 measurement on this image ambiguous. Kept so the divergence
        // is nameable in sim/waveform and so a future control_regs slot can
        // publish it to the host without re-plumbing the attention block.
        .kv_res_cnt(s7a_kv_res_cnt),
        // =================================================================
        // F3-CLR HOST-WRITABLE KV-RESIDENCY CLEAR (control reg 0x154 bit[0]).
        // THE accept-length repair: `kv_resident` was set-only and cleared
        // ONLY by rst_n (a physical cold-AC), so a new SGLang request
        // inherited the previous request's residency over the previous
        // request's KV rows and attended stale cross-request context that
        // the F3 mask certified as legitimate. Depth-0 golden 7/8 fresh
        // cold-AC -> 0/8 after ANY decode run; accept pinned 1.02-1.05.
        // The host pulses this at the START of every new sequence, before
        // that sequence's first draft launch. INERT until written, so every
        // existing gate/seal stays byte-identical.
        // RCA + gate: reports/results/KV_RESIDENCY_CLEAR_FIX.md
        // =================================================================
        .kv_res_clear(kv_res_clear_w),
        // B2' (ACCFIX bit1): round-half on the two attention >>>12 floors.
        .b2p_round_en(accfix_b2p_en)
    );

    // S7a G5 control FSM: on g5_start, kick attention; on attn_done, stream the
    // 3072 attn-out into the G5->G6 seam (K_PAR=16 per cycle, AUTO_SEQ), then
    // pulse g5_done. Drains the full attention pipeline (waits on attn_done) and
    // the full 192-chunk stream-out before signalling done (H3). No valid-high in
    // any drain: s7a_seam_wr is high exactly during the 192 stream cycles.
    typedef enum logic [1:0] {
        S7G5_IDLE   = 2'd0,
        S7G5_ATTN   = 2'd1,   // wait for attention_full_mha.done (full drain)
        S7G5_STREAM = 2'd2,   // walk 192 chunks into the seam (clean strobe)
        S7G5_DONE   = 2'd3
    } s7g5_state_e;
    s7g5_state_e          s7g5_state;
    wire                  s7a_seam_wr;        // COMBINATIONAL (aligned to cnt)
    reg [S7A_CW-1:0]      s7a_wr_chunk_cnt;
    reg                   s7a_seam_clear;
    reg                   s7a_g5_done_r;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            s7g5_state       <= S7G5_IDLE;
            s7a_attn_start   <= 1'b0;
            s7a_wr_chunk_cnt <= '0;
            s7a_seam_clear   <= 1'b0;
            s7a_g5_done_r    <= 1'b0;
        end else begin
            s7a_attn_start <= 1'b0;   // pulse
            s7a_seam_clear <= 1'b0;
            s7a_g5_done_r  <= 1'b0;
            case (s7g5_state)
                S7G5_IDLE: begin
                    if (S7A_FRONT_SEQ && g5_start) begin
                        s7a_attn_start <= 1'b1;
                        s7a_seam_clear <= 1'b1;        // re-arm AUTO_SEQ counter
                        s7g5_state     <= S7G5_ATTN;
                    end
                end
                S7G5_ATTN: begin
                    if (s7a_attn_done) begin
                        s7a_wr_chunk_cnt <= '0;
                        s7g5_state       <= S7G5_STREAM;
                    end
                end
                S7G5_STREAM: begin
                    // write one K_PAR=16 chunk per cycle; out_vec is stable
                    // (combinational concat of the per-head latched results).
                    // NOTE: s7a_seam_wr is COMBINATIONAL (= in S7G5_STREAM, below)
                    // so it is aligned with the CURRENT s7a_wr_chunk_cnt — the
                    // registered-pulse version skewed the write 1 chunk ahead of
                    // the counter (seq_idx 0 got chunk 1, chunk 0 dropped).
                    if (s7a_wr_chunk_cnt == S7A_CW'(S7A_NCH-1)) begin
                        s7g5_state <= S7G5_DONE;
                    end else begin
                        s7a_wr_chunk_cnt <= s7a_wr_chunk_cnt + 1'b1;
                    end
                end
                S7G5_DONE: begin
                    s7a_g5_done_r <= 1'b1;             // pulse g5_done
                    s7g5_state    <= S7G5_IDLE;
                end
                default: s7g5_state <= S7G5_IDLE;
            endcase
        end
    end
    // seam write enable: high for every cycle the FSM is in S7G5_STREAM, so it
    // is aligned with the CURRENT s7a_wr_chunk_cnt (no registered-pulse skew).
    // No valid-high outside STREAM (H2 drain discipline).
    assign s7a_seam_wr = (s7g5_state == S7G5_STREAM);
    // current chunk's 16 elements (combinational slice of the attn-out vector).
    wire signed [K_PAR*A_WIDTH-1:0] s7a_cur_chunk =
        s7a_attn_out_vec[(s7a_wr_chunk_cnt*K_PAR)*A_WIDTH +: K_PAR*A_WIDTH];

    // G5->G6 seam buffer (full-H). FRAC_BITS=10 = g5 Q-format (matches S4 gate).
    wire signed [K_PAR*A_WIDTH-1:0] seam_g5g6_rd;
    wire                            seam_g5g6_full;
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(S7A_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                      .AUTO_SEQ(1'b1), .FRAC_BITS(10), .RD_LAT(D1_RDLAT)) u_seam_g5g6 (  // D1: G6 activation seam, registered (sim==silicon, paired w/ array WLAT=1)
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(s7a_seam_wr), .wr_chunk_idx('0), .wr_chunk(s7a_cur_chunk),
        .wr_clear(s7a_seam_clear), .wr_count(), .full(seam_g5g6_full),
        .rd_valid(1'b1), .rd_chunk_idx(array_chunk_idx_out[S7A_CW-1:0]),
        .rd_chunk(seam_g5g6_rd),
        // WO-025 port-B: parallel ODD-chunk read (same RD_LAT contract).
        .rd_chunk_idx_b(array_chunk_idx_b_out[S7A_CW-1:0]),
        .rd_chunk_b(seam_g5g6_rd_b)
    );

    // g5_done: S7a stream-out completion, or the legacy pilot's g5_done_r.
    wire g5_done = S7A_FRONT_SEQ ? s7a_g5_done_r : g5_done_r;

    // ----------------------------------------------------------------
    // PDI#7.1 dbg G5 tap fix (G8_WEDGE_RCA fix d / debug-only):
    //   The dbg_observe port g5_attn_out_vec is 512b (32*16 = the LEGACY
    //   head-0 D_HEAD=16 pilot, attn_v_combine.out_vec). Under S7A_FRONT_SEQ=1
    //   (default) that pilot island is BYPASSED — the REAL attention output is
    //   the 24-head D_HEAD=128 vector s7a_attn_out_vec (3072*16b). The old tap
    //   therefore sampled a stale/zero pilot bus, so act[2] never reflected the
    //   real attention output (false "G5 zero" on bisection). Re-tap the dbg
    //   inputs to the REAL S7A path when S7A_FRONT_SEQ=1: feed the low 512b of
    //   s7a_attn_out_vec (an OR over it is non-zero iff the real attn output is
    //   non-zero — sufficient for the 0x84 act[2] nz flag) and strobe with
    //   s7a_attn_done. Pure read-only taps; datapath is BIT-IDENTICAL.
    wire signed [32*16-1:0] dbg_g5_attn_vec =
        S7A_FRONT_SEQ ? s7a_attn_out_vec[32*16-1:0] : g5_attn_out_vec;
    wire                    dbg_g5_attn_done =
        S7A_FRONT_SEQ ? s7a_attn_done : g5_attn_done;


    // ----------------------------------------------------------------
    // AXI4-Lite slave register file
    // ----------------------------------------------------------------
    // pdi23 deterministic G8 dump bus (sticky-latched below; read at 0x130/0x134).
    wire [16*32-1:0] g8_dump_flat;
    wire [1:0]       g8_dump_status;
    // TREE-BATCH B=2 batch-test bus (host ctrl 0x138 -> array; sticky readback
    // 0x13c..0x148 <- array result capture, driven in the g_array_scaled block).
    wire [31:0] batch_test_ctrl;
    wire [31:0] batch_b0_lo_w, batch_b0_hi_w, batch_sib_lo_w, batch_sib_hi_w;
    // ISA PMEM & instruction fetcher wiring
    wire [31:0] pmem_write_data_w;
    wire [7:0]  pmem_write_addr_w;
    wire        pmem_we_w;
    wire [7:0]  fetch_instr_w;
    wire [6:0]  fetch_pc_w;

    control_regs #(.COL_W(20)) u_ctrl (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .s_axi_control_awaddr(s_axi_control_awaddr), .s_axi_control_awvalid(s_axi_control_awvalid), .s_axi_control_awready(s_axi_control_awready),
        .s_axi_control_wdata(s_axi_control_wdata),   .s_axi_control_wstrb(s_axi_control_wstrb),    .s_axi_control_wvalid(s_axi_control_wvalid),  .s_axi_control_wready(s_axi_control_wready),
        .s_axi_control_bresp(s_axi_control_bresp),   .s_axi_control_bvalid(s_axi_control_bvalid),  .s_axi_control_bready(s_axi_control_bready),
        .s_axi_control_araddr(s_axi_control_araddr), .s_axi_control_arvalid(s_axi_control_arvalid), .s_axi_control_arready(s_axi_control_arready),
        .s_axi_control_rdata(s_axi_control_rdata),   .s_axi_control_rresp(s_axi_control_rresp),    .s_axi_control_rvalid(s_axi_control_rvalid),  .s_axi_control_rready(s_axi_control_rready),
        .start_pulse(start_pulse), .kernel_busy(kernel_busy), .kernel_done(kernel_done),
        .ctx_len(ctx_len), .num_frontiers(num_frontiers), .max_depth(max_depth),
        .out0_base(out0_base), .out1_base(out1_base),
        .out2_base(out2_base), .hbm_emb_base(hbm_emb_base),
        .lmfull_hbm_base(lmfull_hbm_base),   // G8 SPLICE: ctrl 0x14/0x1c
        // SPEEDWAVE-V2 runtime config (monolith offsets — NOT the jewel 0x138/0x13C)
        .n_tiles_cfg(n_tiles_cfg),           // V1: 0x14C variable vocab scan
        .kc_cfg(kc_cfg),                     // K1: 0x150 runtime K_C shortlist
        // 0x1125 ACCFIX runtime enables (0x15C) + OBS_LVL window (0x160/0x164)
        .accfix_cfg(accfix_cfg_w),
        .obs_rd_idx(obs_rd_idx_w),
        .obs_rd_data(obs_rd_data_w),
        .obs_status_word(obs_status_word_w),
        // M1c: AXI-Lite weight-load path → URAM banks
        .boot_lane_sel(ctrl_boot_lane_sel), .boot_wr_addr(ctrl_boot_wr_addr),
        .boot_wr_data(ctrl_boot_wr_data),   .boot_we(ctrl_boot_we),
        .boot_is_emb(ctrl_boot_is_emb),     // C5 (B1): 0x7c target-select
        .boot_is_hfc(ctrl_boot_is_hfc),     // FIX-E: 0x24 target-select
        .boot_gamma_sel(ctrl_boot_gamma_sel), // GAMMA-LOAD: 0x08 target-select (0..4)
        // M9: boot-DMA control/status (#157 scale-boot-port: +is_scale, COL_W=20)
        .boot_dma_start(ctrl_boot_dma_start),
        .boot_dma_is_scale(ctrl_boot_dma_is_scale),   // #157 NEW: 0=weight(URAM),1=scale(BRAM)
        .boot_dma_is_emb(ctrl_boot_dma_is_emb),       // FASTIN: 0x54[2] → input_emb_buf
        .boot_dma_is_hfc(ctrl_boot_dma_is_hfc),       // FASTIN: 0x54[3] → h_fc_buf
        .boot_hbm_base(ctrl_boot_hbm_base),
        .boot_dma_n_cols(ctrl_boot_dma_n_cols),
        .boot_dma_col_start(ctrl_boot_dma_col_start),
        .boot_dma_col_step(ctrl_boot_dma_col_step),
        .boot_dma_cpc(ctrl_boot_dma_cpc),
        .boot_dma_stage_base(ctrl_boot_dma_stage_base),
        .boot_dma_busy(dma_busy),
        .boot_dma_done(dma_done),
        .boot_dma_timeout(dma_timeout),   // M9 FIX-B: 0x74 bit[2] sticky watchdog
        // OUTPUT-DMA status (RO 0x78): bit[0]=busy, bit[1]=done
        .out_dma_busy(out_dma_busy),
        .out_dma_done(out_dma_done),
        .out_dma_err(out_dma_err),           // C8 (B5): sticky abort -> 0x78 bit2
        // PDI#7 silicon-hardening: G8 watchdog (0x78 bit3) + debug regs 0x80/0x84
        .g8_wd_err(g8_dbg_err),
        .dbg_fsmstate(dbg_fsmstate_word),
        .dbg_activity(dbg_activity_word),
        // ---- DEBUG BLOCK V2 (PDI#8): 0x88 select + 0x8C..0xA4 read-backs ----
        .dbg_sel(dbg_sel_w),
        .dbg_cyclecount(dbg_cyclecount_word),
        .dbg_checksum(dbg_checksum_word),
        .dbg_g8progress(dbg_g8progress_word),
        .dbg_g8araddr_lo(dbg_g8araddr_lo_word),
        .dbg_g8araddr_hi(dbg_g8araddr_hi_word),
        .dbg_rearm(dbg_rearm_word),
        .dbg_bootstream(dbg_bootstream_word),
        // ---- PROBE V3 (0xA8..0xBC): content-probe select + read-backs ----
        .probe_sel(prb_sel_w), .probe_sel_we(prb_sel_we),
        .probe_addr_we(prb_addr_we), .probe_addr_wdata(prb_addr_wdata),
        .probe_data_re(prb_data_re),
        .probe_addr(prb_addr_word), .probe_data(prb_data_word),
        .probe_xor(prb_xor_word), .probe_sum(prb_sum_word), .probe_count(prb_count_word),
        // ---- PROBE V3 tree/cand dump enable + out3/out4 base (0xC0..0xD0) ----
        .dump_en(dump_en), .out3_base(out3_base), .out4_base(out4_base),
        // ---- PDI#10 PROBE EXPANSION: per-stage-output content sample/count ----
        .dbg_stageout_sample(dbg_stageout_sample_word),
        .dbg_stageout_count(dbg_stageout_count_word),
        // ---- PDI#10 PER-STAGE VALUE READBACK (0xDC/0xE0/0xE4/0xE8) ----
        .dbg_cap_stage_sel(dbg_cap_stage_sel_w),
        .dbg_cap_widx_we(dbg_cap_widx_we),
        .dbg_cap_widx_wdata(dbg_cap_widx_wdata),
        .dbg_cap_val_re(dbg_cap_val_re),
        .dbg_cap_widx_word(dbg_cap_widx_word),
        .dbg_cap_val_word(dbg_cap_val_word),
        .dbg_cap_meta_word(dbg_cap_meta_word),
        // ---- PDI#11 PROBE_EXPANSION: BOOT-LOAD RESIDENCY CRC (0xEC/0xF0/0xF4/0xF8) ----
        .res_sel(dbg_res_sel_w),
        .res_clear(dbg_res_clear_w),
        .res_crc_word(dbg_res_crc_word),
        .res_sum_word(dbg_res_sum_word),
        .res_count_word(dbg_res_count_word),
        // ---- F3-CLR KV-RESIDENCY SESSION CLEAR (0x154) ----
        .kv_res_clear(kv_res_clear_w),
        .kv_res_status_word(kv_res_status_word),
        // ---- MAXPROBE liveness window (0x100 build-id / 0x104 selftest / 0x108 cap-crc) ----
        .maxprobe_build_id({16'hE4B7, BUILD_TAG, PROBE_REV}),
        .cap_crc_word(dbg_cap_crc_word),
        // ---- PDI#11 PER-STAGE INJECT (0x10C..0x11C) ----
        .inj_en_we(inj_en_we_w),       .inj_en_wdata(inj_en_wdata_w),
        .inj_widx_we(inj_widx_we_w),   .inj_widx_wdata(inj_widx_wdata_w),
        .inj_data_we(inj_data_we_w),   .inj_data_wdata(inj_data_wdata_w),
        .inj_sel_we(inj_sel_we_w),     .inj_sel_wdata(inj_sel_wdata_w),
        .inj_en_word(inj_en_rb_w),     .inj_widx_word(inj_widx_rb_w),
        .inj_status_word(inj_status_w),
        // ---- M7 GLM=0 WIRED-TAP LOCALIZER (0x120/0x124/0x128) ----
        .g8_dump_flat(g8_dump_flat),         // pdi23 0x134 deterministic frontier/cand dump
        .g8_dump_status(g8_dump_status),     // pdi23 0x130 {resc_latched, cand_latched}
        .lmtap_seam_word(lmtap_word_seam),
        .lmtap_guard_word(lmtap_word_guard),
        .lmtap_ad_word(lmtap_word_ad),
        // ---- TREE-BATCH B=2 SILICON BATCH-TEST WINDOW (0x138..0x148) ----
        .batch_test_ctrl(batch_test_ctrl),
        .batch_b0_lo(batch_b0_lo_w),   .batch_b0_hi(batch_b0_hi_w),
        .batch_sib_lo(batch_sib_lo_w), .batch_sib_hi(batch_sib_hi_w),
        // ISA PMEM write port & status read-back
        .pmem_write_data(pmem_write_data_w),
        .pmem_write_addr(pmem_write_addr_w),
        .pmem_we(pmem_we_w),
        .fetch_instr(fetch_instr_w),
        .fetch_pc(fetch_pc_w)
    );

    // ================================================================
    // ISA Prototype Instruction Fetcher with PMEM
    // ================================================================
    instruction_fetcher u_fetcher (
        .clk        (ap_clk),
        .rst_en     (ap_rst_n),
        .adv_en     (start_pulse),      // Step PC on each kernel launch pulse
        .write_en   (pmem_we_w),
        .write_addr (pmem_write_addr_w),
        .write_data (pmem_write_data_w),
        .instr      (fetch_instr_w),
        .pc_out     (fetch_pc_w)
    );

    // ================================================================
    // PDI#11 PER-STAGE GOLDEN-INJECT (stage_inject.sv) — the INPUT half of the
    // capture+inject isolation pair. Gated behind STAGE_INJECT (default 0):
    //   STAGE_INJECT=0 -> the whole block is generate-pruned; inject_en is tied 0,
    //     inj_bus tied 0, the read-backs RAZ, the per-stage muxes (below) collapse
    //     to pass-through -> netlist + behavior BIT-IDENTICAL to the no-inject top.
    //   STAGE_INJECT=1 -> the runtime inject_en bitmask (0x10C) arms isolation.
    // (memory feedback_no_accuracy_tradeoff: additive, never perturb when off.)
    // (INJ_NSTAGE/INJ_MAX_W + the inj_* nets are HOISTED near the top decls because
    //  the stage muxes appear textually before this instance.)
    // ----------------------------------------------------------------
    generate
    if (STAGE_INJECT) begin : g_stage_inject
        stage_inject #(
            .NSTAGE(INJ_NSTAGE), .MAX_W(INJ_MAX_W), .WIDX_W($clog2(INJ_MAX_W))
        ) u_stage_inject (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .en_we(inj_en_we_w),       .en_wdata(inj_en_wdata_w[INJ_NSTAGE-1:0]),
            .widx_we(inj_widx_we_w),   .widx_wdata(inj_widx_wdata_w[$clog2(INJ_MAX_W)-1:0]),
            .data_we(inj_data_we_w),   .data_wdata(inj_data_wdata_w),
            .sel_we(inj_sel_we_w),     .sel_wdata(inj_sel_wdata_w[4:0]),
            .en_word(inj_en_rb_w),     .widx_word(inj_widx_rb_w),
            .status_word(inj_status_w),
            .inject_en(inject_en),     .inj_bus(inj_bus)
        );
    end else begin : g_no_inject
        // STAGE_INJECT=0: tie everything off -> the muxes below pass-through and the
        // read-backs RAZ (host reads 0x11C and sees NO 0x1A canary -> INCONCLUSIVE,
        // never "stage clean"; memory feedback_probe_reg_needs_real_axi_write...).
        assign inject_en    = '0;
        assign inj_bus      = '0;
        assign inj_en_rb_w  = 32'b0;
        assign inj_widx_rb_w = 32'b0;
        assign inj_status_w = 32'b0;   // no canary -> infra absent
    end
    endgenerate

    // M9: control_regs boot-DMA config wires (#157 scale-boot-port: COL_W=20)
    wire        ctrl_boot_dma_start;
    wire        ctrl_boot_dma_is_scale;   // #157 NEW: bank select (weight URAM vs scale BRAM)
    wire        ctrl_boot_dma_is_emb;     // FASTIN: 0x54[2] weight-stream → input_emb_buf
    wire        ctrl_boot_dma_is_hfc;     // FASTIN: 0x54[3] weight-stream → h_fc_buf
    wire [63:0] ctrl_boot_hbm_base;
    wire [19:0] ctrl_boot_dma_n_cols;     // COL_W=20 (G12 lm N=128256 > 16 bits)
    wire [19:0] ctrl_boot_dma_col_start;
    wire [19:0] ctrl_boot_dma_col_step;
    wire [15:0] ctrl_boot_dma_cpc;
    wire [16:0] ctrl_boot_dma_stage_base;

    // ----------------------------------------------------------------
    // M9: boot-only m_axi read master + HBM→URAM streaming FSM.
    // Drives a SECOND write source into the shared URAM write port. The
    // mux below gives boot-DMA priority while it is busy; the M1c AXI-Lite
    // and legacy TB paths keep working when boot-DMA is idle.
    // ----------------------------------------------------------------
    wire        dma_busy;
    wire        dma_done;
    wire        dma_timeout;   // M9 FIX-B: sticky R-wait watchdog flag -> 0x74[2]
    wire        dma_we;
    wire [15:0] dma_lane_sel;
    wire [16:0] dma_wr_addr;
    wire [31:0] dma_wr_data;
    // FASTIN: latched per-load activation targets from the loader (stable across
    // a whole load; qualify every dma_we pulse of that load).
    wire        dma_tgt_emb;
    wire        dma_tgt_hfc;

    // #157 scale-boot-port: BRAM scale write port (drives scale_bram sc_* via
    // gemv_stage_array_scaled once the scaled-array swap is wired at package
    // time — Step-5 edit A). Declared here so the unified loader elaborates.
    wire        dma_sc_we;
    wire [15:0] dma_sc_lane_sel;
    wire [16:0] dma_sc_wr_addr;
    wire [15:0] dma_sc_wr_data;

    weight_dma_loader #(
        .LANES(LANES), .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(BOOT_AXI_DATA_W),
        .AXI_ID_W(BOOT_AXI_ID_W), .BURST_LEN(BOOT_BURST_LEN), .URAM_ADDR_W(17),
        .COL_W(20)
    ) u_dma_loader (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .boot_dma_start(ctrl_boot_dma_start),
        .boot_is_scale(ctrl_boot_dma_is_scale),        // #157 NEW: weight/scale bank select
        .boot_is_emb(ctrl_boot_dma_is_emb),            // FASTIN: activation target selects
        .boot_is_hfc(ctrl_boot_dma_is_hfc),
        .dma_tgt_emb(dma_tgt_emb),                     // FASTIN: latched per-load targets
        .dma_tgt_hfc(dma_tgt_hfc),
        .boot_hbm_base(ctrl_boot_hbm_base),
        .boot_n_cols(ctrl_boot_dma_n_cols),
        .boot_col_start(ctrl_boot_dma_col_start),
        .boot_col_step(ctrl_boot_dma_col_step),
        .boot_cpc(ctrl_boot_dma_cpc),
        .boot_stage_base(ctrl_boot_dma_stage_base),
        .boot_busy(dma_busy),
        .boot_done(dma_done),
        .boot_timeout(dma_timeout),   // M9 FIX-B: sticky watchdog -> control_regs 0x74[2]
        .dma_we(dma_we), .dma_lane_sel(dma_lane_sel),
        .dma_wr_addr(dma_wr_addr), .dma_wr_data(dma_wr_data),
        // #157 NEW: BRAM scale write port (consumed by gemv_stage_array_scaled at package time)
        .sc_we(dma_sc_we), .sc_lane_sel(dma_sc_lane_sel),
        .sc_wr_addr(dma_sc_wr_addr), .sc_wr_data(dma_sc_wr_data),
        .m_axi_boot_arid(m_axi_boot_arid),     .m_axi_boot_araddr(m_axi_boot_araddr),
        .m_axi_boot_arlen(m_axi_boot_arlen),   .m_axi_boot_arsize(m_axi_boot_arsize),
        .m_axi_boot_arburst(m_axi_boot_arburst), .m_axi_boot_arvalid(m_axi_boot_arvalid),
        .m_axi_boot_arready(m_axi_boot_arready),
        .m_axi_boot_rid(m_axi_boot_rid),       .m_axi_boot_rdata(m_axi_boot_rdata),
        .m_axi_boot_rresp(m_axi_boot_rresp),   .m_axi_boot_rlast(m_axi_boot_rlast),
        .m_axi_boot_rvalid(m_axi_boot_rvalid), .m_axi_boot_rready(m_axi_boot_rready)
    );

    // Weight-load mux — THREE write sources into the shared per-lane URAM
    // write port, in priority order:
    //   1. M9 boot-DMA (dma_we)        — production HBM→URAM bulk load
    //   2. M1c AXI-Lite (ctrl_boot_we) — PDI #4 register-streamed load
    //   3. Legacy top-level boot_*     — array-level xsim TB direct drive
    // Boot-DMA wins while it is streaming (dma_we high); the M1c/TB paths
    // are quiescent during a boot-DMA (the host does not interleave them),
    // so the priority mux is multi-driver-safe (one write per cycle).
    wire [15:0] ctrl_boot_lane_sel;
    wire [16:0] ctrl_boot_wr_addr;
    wire [31:0] ctrl_boot_wr_data;
    wire        ctrl_boot_we;
    wire        ctrl_boot_is_emb;   // C5 (B1): 0x7c bit[0] target-select
    wire        ctrl_boot_is_hfc;   // FIX-E: 0x24 bit[0] target-select
    wire [2:0]  ctrl_boot_gamma_sel; // GAMMA-LOAD: 0x08[2:0] target-select (0=off,1..4)

    // C5 (B1) + FIX-E: split the M1c boot_we 3 ways (WEIGHT / EMB / HFC).
    // boot_is_emb (0x7c) routes the trio to input_emb_buf; boot_is_hfc (0x24)
    // routes it to h_fc_buf; neither → URAM weight banks. The URAM eff_boot_we
    // excludes BOTH emb and hfc; each buffer's write port is driven by its own
    // exclusive pulse. All three mutually exclusive → single-driver-safe per target.
    // GAMMA-LOAD: a non-zero boot_gamma_sel steals the trio for a gamma buffer.
    // It is mutually exclusive with emb/hfc/weight: when boot_gamma_sel != 0 the
    // host has NOT set boot_is_emb/boot_is_hfc (the session-init clears them
    // before driving gammas), but we gate explicitly so no two write ports can
    // ever fire on the same cycle (single-driver-safe per target).
    wire        ctrl_boot_gamma_on = (ctrl_boot_gamma_sel != 3'd0);
    wire        ctrl_boot_we_wt  = ctrl_boot_we & ~ctrl_boot_is_emb & ~ctrl_boot_is_hfc & ~ctrl_boot_gamma_on;  // → URAM weight banks
    wire        ctrl_boot_we_emb = ctrl_boot_we &  ctrl_boot_is_emb & ~ctrl_boot_is_hfc & ~ctrl_boot_gamma_on;  // → input_emb_buf
    wire        ctrl_boot_we_hfc = ctrl_boot_we &  ctrl_boot_is_hfc & ~ctrl_boot_is_emb & ~ctrl_boot_gamma_on;  // → h_fc_buf (FIX-E)
    // GAMMA-LOAD: one exclusive write strobe per gamma buffer (sel 1..4).
    wire        ctrl_boot_we_gamma = ctrl_boot_we & ctrl_boot_gamma_on
                                     & ~ctrl_boot_is_emb & ~ctrl_boot_is_hfc;
    // GAMMA-LOAD: drive the forward-declared gamma write-port nets (decl'd up by
    // the gamma buffers so the writer there sees them). Mirror the emb/hfc ports.
    assign gamma_we_hidden = ctrl_boot_we_gamma & (ctrl_boot_gamma_sel == 3'd1); // gamma_emb_buf
    assign gamma_we_enorm  = ctrl_boot_we_gamma & (ctrl_boot_gamma_sel == 3'd2); // gamma_e_buf
    assign gamma_we_postat = ctrl_boot_we_gamma & (ctrl_boot_gamma_sel == 3'd3); // gamma_postattn_buf
    assign gamma_we_final  = ctrl_boot_we_gamma & (ctrl_boot_gamma_sel == 3'd4); // gamma_final_buf
    assign gamma_word_addr = ctrl_boot_wr_addr;
    assign gamma_wr_chunk  = gamma_word_addr[16:EMB_WPC_LOG2];  // chunk = wa >> 3
    assign gamma_wr_word   = gamma_word_addr[EMB_WPC_LOG2-1:0]; // word-in-chunk
    assign gamma_wr_data32 = ctrl_boot_wr_data;
    // ================================================================
    // FASTIN (2026-07-09): split the boot-DMA weight-mode stream 3 ways
    // (WEIGHT URAM / input_emb_buf / h_fc_buf), mirroring the AXI-Lite
    // ctrl_boot_we_wt/_emb/_hfc split above. dma_tgt_emb/hfc are LATCHED at
    // load start in weight_dma_loader (like is_scale_r) and stable for the
    // whole load, so every dma_we pulse routes to exactly ONE target —
    // single-driver-safe per write port. The dma_we strobe itself is the
    // proven single-cycle-clean idiom (default-low + set-per-accepted-beat),
    // NOT the wvalid&&wready 2-cycle class that doubled the AXI-Lite trio
    // (BOOT-WE-PULSE FIX) — no +128/doubling exposure on this path.
    // For emb/hfc the host programs the FLAT walk (n_cols=1, col_start=0,
    // col_step=1, cpc=1536, stage_base=0) so dma_wr_addr = flat 32b word
    // 0..1535 — the SAME flat address the AXI-Lite trio pointer walks.
    // ================================================================
    wire        dma_we_emb = dma_we &  dma_tgt_emb & ~dma_tgt_hfc;   // → input_emb_buf
    wire        dma_we_hfc = dma_we &  dma_tgt_hfc & ~dma_tgt_emb;   // → h_fc_buf
    wire        dma_we_wt  = dma_we & ~dma_tgt_emb & ~dma_tgt_hfc;   // → URAM weight banks

    // C5 (B1): drive the input_emb_buf 32-bit sub-word write port.
    // FASTIN: DMA route wins while its (exclusive) strobe is high; the AXI-Lite
    // trio path is quiescent during a DMA load (host serializes them), so the
    // priority mux is single-writer-per-cycle — same idiom as eff_boot_* below.
    assign emb_we        = dma_we_emb | ctrl_boot_we_emb;
    assign emb_word_addr = dma_we_emb ? dma_wr_addr : ctrl_boot_wr_addr;
    assign emb_wr_chunk  = emb_word_addr[16:EMB_WPC_LOG2];      // chunk = wa >> 3
    assign emb_wr_word   = emb_word_addr[EMB_WPC_LOG2-1:0];     // word-in-chunk
    assign emb_wr_data32 = dma_we_emb ? dma_wr_data : ctrl_boot_wr_data;
    // FIX-E: drive the h_fc_buf 32-bit sub-word write port (mirror the emb port).
    assign hfc_we        = dma_we_hfc | ctrl_boot_we_hfc;
    assign hfc_word_addr = dma_we_hfc ? dma_wr_addr : ctrl_boot_wr_addr;
    assign hfc_wr_chunk  = hfc_word_addr[16:EMB_WPC_LOG2];      // chunk = wa >> 3
    assign hfc_wr_word   = hfc_word_addr[EMB_WPC_LOG2-1:0];     // word-in-chunk
    assign hfc_wr_data32 = dma_we_hfc ? dma_wr_data : ctrl_boot_wr_data;

    // FASTIN: the URAM weight port now takes dma_we_wt (weight-target loads
    // only) — an emb/hfc DMA load must NOT touch URAM.
    wire        eff_boot_we       = dma_we_wt | ctrl_boot_we_wt | boot_we;
    wire [15:0] eff_boot_lane_sel = dma_we_wt     ? dma_lane_sel
                                  : ctrl_boot_we_wt ? ctrl_boot_lane_sel
                                                    : boot_lane_sel;
    wire [16:0] eff_boot_wr_addr  = dma_we_wt     ? dma_wr_addr
                                  : ctrl_boot_we_wt ? ctrl_boot_wr_addr
                                                    : boot_wr_addr;
    wire [31:0] eff_boot_wr_data  = dma_we_wt     ? dma_wr_data
                                  : ctrl_boot_we_wt ? ctrl_boot_wr_data
                                                    : boot_wr_data;

    // ----------------------------------------------------------------
    // Top FSM — 11-stage drafter orchestrator
    // ----------------------------------------------------------------
    // G8 SPLICE handshake wires (declared here; g8_rescore instance + g8_done
    // generation are in the G8 SPLICE block after the tree region).
    wire g8_start;
    wire g8_done;

    // =====================================================================
    // DEPTH-N LEVEL LOOP (tree FSM stage 2) — strapped OFF by default.
    //
    // Strap with `-d DEPTH_N_V=1` to BOTH xvlog AND xelab. (xelab -prj
    // RE-ANALYZES the sources, so a define handed only to xvlog is silently
    // dropped and you get a dead arm that "passes" while testing the good
    // build — the 07-09/07-06 compiled-out-tooth trap.)
    //
    // At DEPTH_N=0 this whole region is a generate-false: tree_level_fsm is
    // not instantiated, top_fsm's use_depth_loop folds to constant 0, and the
    // depth-0 datapath is bit-identical. That is gated by the 19 stage CRCs +
    // frontier 8/8 in scripts/run_speedwave_fullchain_gate.sh (ARM=full).
    // =====================================================================
    top_fsm #(
        .G8_SPLICE(G8_SPLICE),
        .DEPTH_N  (DEPTH_N),
        .K_F      (8),
        .EMBG_EN  (EMBG_EN),     // RUNG C: child-embedding gather-wait
        .KVONLY_EN(ACCFIX_KVONLY_P) // ACCFIX: KV-only launch mode presence
    ) u_fsm (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(start_pulse), .busy(kernel_busy), .done(kernel_done),
        // DEPTH-N: this port is what makes control reg 0x20 LIVE. Before this
        // it was declared (:2338) and passed to control_regs (:2579) and then
        // read by NOTHING.
        .max_depth      (max_depth),
        // ACCFIX KVONLY (bit3): KV-append-only launch mode (prime rows).
        .kvonly_i       (accfix_kvonly_en),
        .lvl_start      (lvl_start_w),
        .branch_j       (branch_j_w),
        .level_i        (level_i_w),
        .branch_feed    (branch_feed_w),
        .level_done     (lvl_level_done_w),
        .level_last     (lvl_level_last_w),
        .tree_all_done  (lvl_tree_done_w),
        .out_walk_start (out_walk_start_w),
        .out_walk_done  (lvl_out_walk_done_w),
        // RUNG C: child-embedding gather-wait handshake (inert at EMBG_EN=0)
        .emb_start      (emb_start_w),
        .emb_done       (emb_done_w),
        .g1_start(g1_start), .g1_done(g1_done),
        .g4_start(g4_start), .g4_done(g4_done),
        .g5_start(g5_start), .g5_done(g5_done),
        .g6_start(g6_start), .g6_done(g6_done_stage),
        .g7g_start(g7g_start), .g7g_done(g7g_done_stage),
        .g7u_start(g7u_start), .g7u_done(g7u_done_stage),
        .g7p_start(g7p_start), .g7p_done(g7p_done),
        .g9_start(g9_start), .g9_done(g9_done_stage),
        .g11_start(g11_start), .g11_done(g11_done),
        .glm_start(glm_start), .glm_done(glm_done),   // FIX-G
        .g12_start(g12_start), .g12_done(g12_done),
        .tree_start(tree_start), .tree_done(tree_done),
        .g8_start(g8_start), .g8_done(g8_done),        // G8 SPLICE
        .stage_sel(stage_sel)
    );

    // =====================================================================
    // DEPTH-N: tree_level_fsm + the KV lane-chain write driver.
    //
    // ADDRESS RULE (ONE source of truth for both write and read):
    //     tree_slot(j, i) = ctx_len + j*num_steps + i
    // This is SGLang's stock lane-chain, spec_utils.py:290
    //     token_pool_ptr + seq_len + topk_id*num_steps + arange(iter_upper)
    // i.e. branch j reads the prefix plus ITS OWN column. It is deliberately
    // NOT exact per-branch ancestry: reports/results/EXACT_ANCESTRY_AB.md
    // measured exact ancestry at -0.0018/-0.0010/-0.0029 accept for depth
    // 3/5/8 with every 95% CI straddling zero, and the capture audit found the
    // GPU's own KV tail IS the slot chain in 192/192 cases (true-ancestor only
    // 32.81%). So the lane-chain is not an approximation of the golden — it is
    // the golden, and a kv_anc_lane-style copy-forward would BREAK the match.
    // Fixed stride, no data-dependent indirection, zero new URAM/BRAM.
    // =====================================================================
    generate if (DEPTH_N) begin : g_depth_n

        // ---- the level loop ----
        // NOTE: `start` is lvl_start_w, which top_fsm pulses ONCE PER TREE
        // (S_IDLE->S_LVL_ARM), never per level. tree_level_fsm owns the
        // frontier across levels; re-pulsing start would wipe it.
        wire [tree_score_pkg::K_F*tree_score_pkg::NODE_W-1:0] tl_surv_node;
        wire [tree_score_pkg::K_F*tree_score_pkg::NODE_W-1:0] tl_surv_parent;
        wire [tree_score_pkg::K_F*tree_score_pkg::SLOT_W-1:0] tl_surv_parent_slot; // G3
        wire [tree_score_pkg::K_F*tree_score_pkg::TOK_W-1:0]  tl_surv_token;
        wire [tree_score_pkg::K_F*tree_score_pkg::SCORE_W-1:0] tl_surv_score;
        wire [tree_score_pkg::DEPTH_W-1:0]                    tl_level_idx;
        wire [tree_score_pkg::POS_W-1:0]                      tl_level_pos;
        wire                                                  tl_cand_req;
        wire [tree_score_pkg::NODE_W-1:0]                     tl_n_nodes;
        wire [tree_score_pkg::DEPTH_W-1:0]                    tl_levels_run;

        // tree readback bus (tree_level_fsm <-> tree_out_walk)
        wire [tree_score_pkg::NODE_W-1:0]  tn_walk_idx;
        wire [tree_score_pkg::TOK_W-1:0]   tn_walk_token;
        wire [tree_score_pkg::NODE_W-1:0]  tn_walk_parent;
        wire [tree_score_pkg::DEPTH_W-1:0] tn_walk_depth;
        wire [tree_score_pkg::POS_W-1:0]   tn_walk_pos;
        wire signed [tree_score_pkg::SCORE_W-1:0] tn_walk_score; // COSTDRAFTTREE-ABI (G1)

        // ---- STAGE-4: the scoring producer (bridge) + per-branch feed sequencer.
        // The bridge turns each branch's 512 rescored logits into 8 fp32 softmax
        // probs; tree_feed_seq latches the branch frontier, runs the bridge, and
        // streams the 8 (prob,token) into the FSM. See tree_feed_seq.sv +
        // reports/results/TREE_FSM_STAGE4.md.
        wire                                 br_start;
        wire [tree_score_pkg::K_F*32-1:0]    br_frontier_f32;
        wire [tree_score_pkg::K_F*tree_score_pkg::TOK_W-1:0] br_frontier_tok;
        wire                                 br_done;
        wire [tree_score_pkg::K_F*32-1:0]    br_prob_f32;
        wire [tree_score_pkg::K_F*tree_score_pkg::TOK_W-1:0] br_tok;
        wire                                 fseq_cand_valid;
        wire [31:0]                          fseq_cand_prob_f32;
        wire [tree_score_pkg::TOK_W-1:0]     fseq_cand_token;
        wire                                 fseq_feed_busy;

        tree_level_fsm u_tree_level (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start        (lvl_start_w),
            .ctx_len_i    (ctx_len),
            .max_depth_i  (max_depth),
            // ---- candidate feed ----
            // STAGE-3 SEAM -- PRODUCER NOW EXISTS AND IS VALIDATED (this fork).
            // cand_prob_f32 = p(child|parent) as IEEE-754 binary32. The producer
            // is `layers/tree_softmax_bridge.sv`: per parent body pass it sweeps
            // the 512 G8 rescore logits (g8_cand_dump_score @ g8_cand_rdaddr),
            // computes the softmax DENOMINATOR (the load-bearing cross-parent
            // normalizer), and emits the 8 selected children's fp32 probs. It is
            // accurate to 5.2e-07 rel vs fp32 softmax (bridge micro-gate) -- 15,000x
            // tighter than the parked Q4.12 softmax_frontier_q, which the analysis
            // rejects (it collapses distinct child probs into artificial ties,
            // A6 gate up to 0.76%). The bridge+FSM scoring path is GREEN on the
            // captured golden: fed-loop gate SET 100% d1..d4 AND d7_draft8 (the
            // deepest real depth), RED arms NONORM/INDEP/SUM all fire. See
            // reports/results/TREE_FSM_STAGE3.md.
            //
            // TOP-LEVEL WIRING (stage-4 integration, NOT built here -- NO BUILD):
            //   1. arbiter mux g8_cand_rdaddr (line ~1072): a6_active ? bridge_rdaddr
            //      : ow_dump3_rdaddr[.. :1]  (the dump only reads post-all-done).
            //   2. latch the 8 frontier scores/ids off ow_topk_scores/ow_topk_indices
            //      on the g8_done_w edge (per branch_j) -> bridge.frontier_f32/_tok.
            //   3. per-branch feed sequencer: on each g8_done, run the bridge, then
            //      stream its 8 (prob,token) into cand_valid/cand_prob_f32/cand_token
            //      over 8 cycles (the FSM accumulates 8x8=64/level via ord_cnt).
            //      This sequencing is exactly what tb_tree_fed_loop.sv validates.
            //   Do NOT bit-cast the INT32 tt_score_in into cand_prob_f32 -- tree_log2_fxp
            //   cracks real fp32 exponent/mantissa and would produce garbage scores.
            .cand_req     (tl_cand_req),
            .cand_valid   (fseq_cand_valid),     // stage-4: <- tree_feed_seq
            .cand_prob_f32(fseq_cand_prob_f32),  // stage-4: <- tree_softmax_bridge prob (via feed_seq)
            .cand_token   (fseq_cand_token),     // stage-4: <- tree_softmax_bridge tok  (via feed_seq)
            // ---- per-level result ----
            .level_done   (lvl_level_done_w),
            .level_last   (lvl_level_last_w),
            .level_idx_o  (tl_level_idx),
            .level_pos_o  (tl_level_pos),
            .surv_node_o  (tl_surv_node),
            .surv_parent_o(tl_surv_parent),
            .surv_parent_slot_o(tl_surv_parent_slot),   // G3 pool->slot inversion
            .surv_token_o (tl_surv_token),
            .surv_score_o (tl_surv_score),
            // ---- accumulated-tree readback (drives the output walk) ----
            .tn_rd_idx    (tn_walk_idx),
            .tn_rd_token  (tn_walk_token),
            .tn_rd_parent (tn_walk_parent),
            .tn_rd_depth  (tn_walk_depth),
            .tn_rd_pos    (tn_walk_pos),
            .tn_rd_score  (tn_walk_score),   // COSTDRAFTTREE-ABI (G1) per-node cumu log2 score
            // ---- status ----
            .busy         (),
            .done         (lvl_tree_done_w),
            .levels_run_o (tl_levels_run),
            .n_nodes_o    (tl_n_nodes)
        );

        // ---- F4/G3: parent of the branch about to run, BOTH coordinates ----
        // Same snapshot + same branch_j_w index the emb_gather uses for the
        // survivor TOKEN, here for the survivor's PARENT. Both come off the
        // committed T_COMMIT snapshot (stable for the whole level).
        //   fr_par_node_w = parent POOL index  (RED arm only, from G3 on)
        //   fr_par_slot_w = parent FRONTIER SLOT = the `a`-bank to re-gather
        // G3 SUPERSEDED "low 3 bits of the pool index are the slot": true only
        // while the parents are level-0 nodes (depth<=2). See tree_level_fsm
        // cs_pslot + reports/results/TREE_FSM_DEPTHN_G3_POOL_SLOT_INVERSION.md.
        assign fr_par_node_w =
            tl_surv_parent[branch_j_w*tree_score_pkg::NODE_W +: tree_score_pkg::NODE_W];
        // G3: the same snapshot, same branch_j_w index, but the SLOT coordinate.
        // No inversion is performed here on purpose -- the FSM already published
        // it (see tree_level_fsm cs_pslot); re-deriving it at the consumer would
        // be the second source of truth this project keeps getting bitten by.
        assign fr_par_slot_w =
            tl_surv_parent_slot[branch_j_w*tree_score_pkg::SLOT_W +: tree_score_pkg::SLOT_W];

        // ---- GAP-4 RUNG C: child-embedding gather (emb_gather top graft) ----
        // Fetches the CHILD branch's parent-token embedding row from the TARGET
        // embed_tokens image (hbm_emb_base) into input_emb_buf via the byte-enabled
        // write arbiter above, before each depth>=1 body pass. tok_id = the branch's
        // committed survivor (tl_surv_token[branch_j_w]); the top_fsm S_LVL_EMBG
        // state pulses emb_start_w and waits emb_done_w. Present only at EMBG_EN=1;
        // otherwise g_embg_tieoff drives the bus 0 and m_axi_emb is parked.
        if (EMBG_EN) begin : g_embg
            // survivor token for the branch about to run (branch_j_w already
            // advanced by top_fsm before S_LVL_EMBG). TOK_W==IDX_W==18.
            wire [tree_score_pkg::TOK_W-1:0] embg_tok =
                tl_surv_token[branch_j_w*tree_score_pkg::TOK_W +: tree_score_pkg::TOK_W];
            // ---- EMB_NCH channel fan-out ----
            //   emb_gather's AXI master ports are packed [EMB_NCH*W-1:0].
            //   Channel 0 -> m_axi_emb  (S08_AXI / HBM5_PORT1)
            //   Channel 1 -> m_axi_embb (S09_AXI / HBM5_PORT0)  [EMB_NCH=2]
            //   Each channel keeps its OWN port: ARVALID fan-in per NoC SI is
            //   1 at every EMB_NCH (feedback_axi_driver_fanin_rule), and the
            //   channels are NEVER muxed onto a shared bundle.
            //   EMB_NCH>2 needs 2 more top ports + S10/S11 BD patches; the
            //   emb_gather module itself is proven at NCH=4 in the unit gate
            //   (scripts/run_emb_nch_gate.sh) but the TOP is bounded at 2
            //   deliberately — burning SI ports that measurably buy nothing
            //   is exactly what EMB_GATHER_MULTICHANNEL.md argues against.
            wire [EMB_NCH-1:0]                     embg_arvalid;
            wire [EMB_NCH-1:0]                     embg_arready;
            wire [EMB_NCH*BOOT_AXI_ADDR_W-1:0]     embg_araddr;
            wire [EMB_NCH*8-1:0]                   embg_arlen;
            wire [EMB_NCH*3-1:0]                   embg_arsize;
            wire [EMB_NCH*2-1:0]                   embg_arburst;
            wire [EMB_NCH-1:0]                     embg_rvalid;
            wire [EMB_NCH-1:0]                     embg_rready;
            wire [EMB_NCH*256-1:0]                 embg_rdata;
            wire [EMB_NCH-1:0]                     embg_rlast;

            emb_gather #(
                .H          (3072),
                .IDX_W      (tree_score_pkg::TOK_W),   // 18
                .AXI_ADDR_W (BOOT_AXI_ADDR_W),
                .AXI_DATA_W (256),                     // 16 bf16/beat = 1 chunk
                .EMB_FMT    (1'b1),                    // bf16 source -> Q4.12 in fabric
                .NCH        (EMB_NCH)                  // 1 = shipping single channel
            ) u_emb_gather (
                .clk         (ap_clk),
                .rst_n       (ap_rst_n),
                .start       (emb_start_w),
                .tok_id      (embg_tok),
                .hbm_emb_base(hbm_emb_base),
                .m_arvalid   (embg_arvalid),
                .m_arready   (embg_arready),
                .m_araddr    (embg_araddr),
                .m_arlen     (embg_arlen),
                .m_arsize    (embg_arsize),
                .m_arburst   (embg_arburst),
                .m_rvalid    (embg_rvalid),
                .m_rready    (embg_rready),
                .m_rdata     (embg_rdata),
                .m_rlast     (embg_rlast),
                .o_we        (embg_o_we),
                .o_chunk     (embg_o_chunk),
                .o_data      (embg_o_data),
                .done        (emb_done_w),
                .beats_dbg   ()
            );
            // ---- channel 0 <-> m_axi_emb (unchanged at EMB_NCH=1) ----
            assign m_axi_emb_arvalid = embg_arvalid[0];
            assign m_axi_emb_araddr  = embg_araddr[0*BOOT_AXI_ADDR_W +: BOOT_AXI_ADDR_W];
            assign m_axi_emb_arlen   = embg_arlen  [0*8 +: 8];
            assign m_axi_emb_arsize  = embg_arsize [0*3 +: 3];
            assign m_axi_emb_arburst = embg_arburst[0*2 +: 2];
            assign m_axi_emb_rready  = embg_rready [0];
            assign embg_arready[0]   = m_axi_emb_arready;
            assign embg_rvalid [0]   = m_axi_emb_rvalid;
            assign embg_rlast  [0]   = m_axi_emb_rlast;
            assign embg_rdata[0*256 +: 256] = m_axi_emb_rdata;
            // emb_gather has no arid port (single-ID master); tie the top ARID.
            assign m_axi_emb_arid = '0;

            // ---- channel 1 <-> m_axi_embb (present only at EMB_NCH>=2) ----
            if (EMB_NCH > 1) begin : g_embb
                assign m_axi_embb_arvalid = embg_arvalid[1];
                assign m_axi_embb_araddr  = embg_araddr[1*BOOT_AXI_ADDR_W +: BOOT_AXI_ADDR_W];
                assign m_axi_embb_arlen   = embg_arlen  [1*8 +: 8];
                assign m_axi_embb_arsize  = embg_arsize [1*3 +: 3];
                assign m_axi_embb_arburst = embg_arburst[1*2 +: 2];
                assign m_axi_embb_rready  = embg_rready [1];
                assign embg_arready[1]    = m_axi_embb_arready;
                assign embg_rvalid [1]    = m_axi_embb_rvalid;
                assign embg_rlast  [1]    = m_axi_embb_rlast;
                assign embg_rdata[1*256 +: 256] = m_axi_embb_rdata;
                assign m_axi_embb_arid    = '0;
            end
        end

        // ---- STAGE-4: softmax bridge (the p(child|parent) producer) ----
        // Reads the 512 rescored logits via the arbiter-muxed candidate port
        // (bridge_cand_rd_active gives it g8_cand_rdaddr; g8_cand_dump_score is
        // g8's rescored[] slot). Emits 8 fp32 probs per parent. 0 URAM / 0 BRAM.
        tree_softmax_bridge #(
            .K_C(G8_K_C), .CIDX_W(G8_CIDX_W)
        ) u_tree_bridge (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start          (br_start),
            // BRIDGE_KC (ACCFIX bit2): runtime shortlist bound + S_MAX skip.
            .kc_cfg         (kc_cfg),
            .kc_en          (accfix_kc_en),
            .cand_rdaddr    (bridge_cand_rdaddr),      // -> arbiter mux (top)
            .cand_rd_active (bridge_cand_rd_active),   // -> arbiter mux (top)
            .cand_rdaddr_nxt(bridge_cand_rdaddr_nxt),  // SEAM-FF lookahead -> g8 reg read
            .cand_logit_f32 (bridge_cand_logit),       // SEAM-FF: registered g8 rescored[] (TREE_SEAM_FF=1)
            .frontier_f32   (br_frontier_f32),
            .frontier_tok   (br_frontier_tok),
            .done           (br_done),
            .prob_f32_o     (br_prob_f32),
            .tok_o          (br_tok)
        );

        // ---- STAGE-4: per-branch feed sequencer (the top-level wiring recipe) ----
        // Trigger = top_fsm branch_feed (fires after each branch's g8_done). It
        // latches the branch's rescored frontier (ow_topk_scores/_indices == the
        // g8 frontier under G8_SPLICE), kicks the bridge, and streams its 8
        // (prob,token) into the FSM candidate feed (cand_req stays high the whole
        // level -> per-branch gaps are fine; branch j -> ordinals j*8..j*8+7).
        tree_feed_seq #(
            .G8_IDX_W(TREE_IDX_W)
        ) u_tree_feed_seq (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .branch_feed        (branch_feed_w),
            .g8_frontier_scores (ow_topk_scores),
            .g8_frontier_ids    (ow_topk_indices),
            .bridge_start       (br_start),
            .bridge_frontier_f32(br_frontier_f32),
            .bridge_frontier_tok(br_frontier_tok),
            .bridge_done        (br_done),
            .bridge_prob_f32    (br_prob_f32),
            .bridge_tok         (br_tok),
            .cand_req           (tl_cand_req),
            .cand_valid         (fseq_cand_valid),
            .cand_prob_f32      (fseq_cand_prob_f32),
            .cand_token         (fseq_cand_token),
            .feed_busy          (fseq_feed_busy)
        );

        // ---- OBS_LVL tap: the per-level candidate stream, as consumed by
        // tree_level_fsm (64 entries per level at depth>=1; 8 at level 0).
        assign obs_cand_v_w    = fseq_cand_valid;
        assign obs_cand_prob_w = fseq_cand_prob_f32;
        assign obs_cand_tok_w  = fseq_cand_token[17:0];
        assign obs_tree_arm_w  = lvl_start_w;

        // ---- output walk: tree -> the 4-tensor verify ABI ----
        // draft_token / retrive_index (identity, emitted as a CONSTANT) /
        // retrive_next_token / retrive_next_sibling. custom_mask and positions
        // are NOT built: they serve the target forward, not the verify kernel.
        wire                                          ow_emit_valid;
        wire [tree_score_pkg::NODE_W-1:0]             ow_emit_idx;
        wire [tree_score_pkg::TOK_W-1:0]              ow_emit_token;
        wire signed [tree_score_pkg::NODE_W:0]        ow_emit_next_tok;
        wire signed [tree_score_pkg::NODE_W:0]        ow_emit_next_sib;
        wire [tree_score_pkg::POS_W-1:0]              ow_emit_pos;
        wire signed [tree_score_pkg::SCORE_W-1:0]     ow_emit_score; // COSTDRAFTTREE-ABI (G1)

        tree_out_walk u_tree_out_walk (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start        (out_walk_start_w),
            .n_nodes      (tl_n_nodes),
            .done         (lvl_out_walk_done_w),
            .tn_rd_idx    (tn_walk_idx),
            .tn_rd_token  (tn_walk_token),
            .tn_rd_parent (tn_walk_parent),
            .tn_rd_depth  (tn_walk_depth),
            .tn_rd_pos    (tn_walk_pos),
            .tn_rd_score  (tn_walk_score),   // COSTDRAFTTREE-ABI (G1)
            .emit_valid   (ow_emit_valid),
            .emit_idx     (ow_emit_idx),
            .emit_token   (ow_emit_token),
            .emit_next_tok(ow_emit_next_tok),
            .emit_next_sib(ow_emit_next_sib),
            .emit_pos     (ow_emit_pos),
            .emit_score   (ow_emit_score)    // COSTDRAFTTREE-ABI (G1)
        );

        // ---- GAP5: DEPTH-N tree OUT-DMA — burst the 4-tensor verify ABI ----
        // Consumes tree_out_walk's node-major emit stream, buffers it in FF
        // register files, and on lvl_out_walk_done_w bursts draft_token /
        // retrive_index (identity constant) / retrive_next_token /
        // retrive_next_sibling as SoA arrays (stride TREE_STRIDE) off out0_base.
        // Drives the physical m_axi_out via the tr_* generate mux above.
        tree_dma_writer #(
            .NODE_MAX(tree_score_pkg::N_MAX),
            .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(BOOT_AXI_DATA_W),
            .AXI_ID_W(BOOT_AXI_ID_W)
        ) u_tree_dma (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .emit_valid   (ow_emit_valid),
            .emit_idx     (ow_emit_idx),
            .emit_token   (ow_emit_token),
            .emit_next_tok(ow_emit_next_tok),
            .emit_next_sib(ow_emit_next_sib),
            .emit_score   (ow_emit_score),   // COSTDRAFTTREE-ABI (G1) 5th SoA region
            .walk_done    (lvl_out_walk_done_w),
            .n_nodes      (tl_n_nodes),
            .tree_base    (out0_base),
            .out_busy     (tr_busy),
            .out_done     (tr_done),
            .out_err      (tr_err),
            .m_axi_out_awid(tr_awid),        .m_axi_out_awaddr(tr_awaddr),
            .m_axi_out_awlen(tr_awlen),      .m_axi_out_awsize(tr_awsize),
            .m_axi_out_awburst(tr_awburst),  .m_axi_out_awvalid(tr_awvalid),
            .m_axi_out_awready(m_axi_out_awready),
            .m_axi_out_wdata(tr_wdata),      .m_axi_out_wstrb(tr_wstrb),
            .m_axi_out_wlast(tr_wlast),      .m_axi_out_wvalid(tr_wvalid),
            .m_axi_out_wready(m_axi_out_wready),
            .m_axi_out_bid(m_axi_out_bid),   .m_axi_out_bresp(m_axi_out_bresp),
            .m_axi_out_bvalid(m_axi_out_bvalid), .m_axi_out_bready(tr_bready)
        );

    end else begin : g_no_depth_n
        // DEPTH_N=0: hold every feedback input at its inert value so top_fsm's
        // S_LVL_* states are not merely unreachable but un-enterable.
        assign lvl_level_done_w    = 1'b0;
        assign lvl_level_last_w    = 1'b0;
        assign lvl_tree_done_w     = 1'b0;
        assign lvl_out_walk_done_w = 1'b0;
        // STAGE-4: bridge never owns the candidate port -> the arbiter mux folds
        // to the pre-stage-4 dump-only path (BIT-IDENTICAL depth-0).
        assign bridge_cand_rd_active = 1'b0;
        assign bridge_cand_rdaddr    = '0;
        assign bridge_cand_rdaddr_nxt = '0;   // SEAM-FF lookahead inert at DEPTH_N=0
        // F4: no tree FSM at DEPTH_N=0 -> parent slot inert (u_forreason itself
        // only exists at FORREASON, which is meaningful only with DEPTH_N).
        assign fr_par_node_w         = '0;
        assign fr_par_slot_w         = '0;   // G3: inert at DEPTH_N=0
        // OBS_LVL: no candidate seam exists at DEPTH_N=0.
        assign obs_cand_v_w    = 1'b0;
        assign obs_cand_prob_w = '0;
        assign obs_cand_tok_w  = '0;
        assign obs_tree_arm_w  = 1'b0;
    end endgenerate

    // =====================================================================
    // OBS_LVL (0x1125): per-level tree candidate ring — 512 entries x 2
    // words (1 BRAM-class array), write pointer reset at every tree arm
    // (lvl_start). Captures EVERY (level, branch, token, prob) the feed
    // sequencer hands to tree_level_fsm, so the host can reconstruct the
    // per-level child sets (H1 child-overlap discriminator) without any
    // DMA-format change. Read via 0x160 (word idx, status) / 0x164 (data).
    // Pure read-side tap: zero effect on the datapath.
    // =====================================================================
    generate if (OBS_LVL_P) begin : g_obs_lvl
        (* ram_style = "block" *)
        reg [31:0] obs_ring [0:1023];
        reg [9:0]  obs_wp;          // ENTRY pointer (2 words/entry)
        reg        obs_wrap;
        reg [31:0] obs_rdata_q;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                obs_wp <= '0; obs_wrap <= 1'b0;
            end else if (obs_tree_arm_w) begin
                obs_wp <= '0; obs_wrap <= 1'b0;
            end else if (obs_cand_v_w) begin
                obs_ring[{obs_wp[8:0], 1'b0}] <= obs_cand_prob_w;
                obs_ring[{obs_wp[8:0], 1'b1}] <= {level_i_w, branch_j_w, 7'b0,
                                                  obs_cand_tok_w};
                if (obs_wp[8:0] == 9'h1FF) obs_wrap <= 1'b1;
                obs_wp <= obs_wp + 1'b1;
            end
            obs_rdata_q <= obs_ring[obs_rd_idx_w[9:0]];
        end
        assign obs_rd_data_w     = obs_rdata_q;
        assign obs_status_word_w = {8'h0B, 13'b0, obs_wrap, obs_wp};
    end else begin : g_no_obs_lvl
        assign obs_rd_data_w     = 32'b0;
        assign obs_status_word_w = 32'b0;
    end endgenerate

    // ----------------------------------------------------------------
    // L8c: route head-0 attention output back into the GEMV array as
    // the G6 input activation.
    //
    // g5_attn_out_vec is G5_D_HEAD=16 × INT32. The array's a_chunk is
    // K_PAR=16 × INT16. We latch on g5_attn_done, sign-extend-truncate
    // each lane (low 16 bits = INT16 with saturation), and pack into
    // a 16×16=256-bit register. During the G6 stage (stage_sel=4'd1)
    // this latched register replaces the external a_chunk port.
    //
    // SCALE CONTRACT (C5, RESOLVED 2026-05-23 — see MIXED_PRECISION §3.1):
    //   g5_attn_out_vec[d] = attn_v_combine.out_vec[d]
    //                      = Σ_p softmax_frac[p] · V_raw[p,d]   (RAW scale).
    //   attn_v_combine internally does (attn_q12·V_raw) >> 12, which CANCELS
    //   the Q4.12 scale of the softmax weight (attn_q12 = softmax·2^12) against
    //   a RAW INT16 V element — so the accumulator is RAW-magnitude, NOT Q4.12.
    //   G6's GEMV input activation is ALSO raw INT (g6_act golden, range ≈ ±8),
    //   so we feed the raw low-16 bits DIRECTLY. There is NO >>12 here, and
    //   adding one would be the 2^12 double-scale bug the SOTA review flagged.
    //   Verified: bit-accurate trace reproduces g5.hex exactly; bisect_stages.py
    //   g6-anchor is bit-exact (lane0 acc = 60 == g6.hex col0).
    //
    // All other stages still see the external a_chunk (production will
    // replace stage by stage as upstream layer modules land).
    // ----------------------------------------------------------------
    function automatic logic signed [15:0] sat_int32_to_int16;
        input signed [31:0] v;
        begin
            if (v > 32'sd32767)         sat_int32_to_int16 = 16'sd32767;
            else if (v < -32'sd32768)   sat_int32_to_int16 = 16'sh8000;
            else                        sat_int32_to_int16 = v[15:0];
        end
    endfunction

    reg signed [16*K_PAR-1:0] attn_out_chunk_latched;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            attn_out_chunk_latched <= '0;
        end else if (g5_attn_done) begin
            for (int lane = 0; lane < G5_D_HEAD && lane < K_PAR; lane = lane + 1) begin
                attn_out_chunk_latched[lane*16 +: 16] <=
                    sat_int32_to_int16(g5_attn_out_vec[lane*32 +: 32]);
            end
        end
    end

    // Mux: G6 stage takes attention output; otherwise pass external a_chunk.
    // PDI#5-ASSEMBLY: when CONNECT_SEAMS=1 the other GEMV stages take their
    // activation from the upstream seam buffer (R1/R2/R5/R8) instead of the
    // external `a_chunk` port. The seam-read wires are forward-declared in the
    // PDI#5-ASSEMBLY block below (after the array). The G6 arm is UNCHANGED
    // (the original L8c real island). One arm per stage_sel = the §6.4 template.
    //   stage_sel 0 = G4   <- R1 (G1 RMSNorm out)
    //   stage_sel 1 = G6   <- attn_out_chunk_latched (the L8c island, kept)
    //   stage_sel 2 = G7g  <- R2 (G6 o_proj out, sat-INT16)
    //   stage_sel 3 = G7u  <- R2 (same buffer feeds both gate+up)
    //   stage_sel 4 = G9   <- R5 (silu out)
    //   stage_sel 5 = G12  <- R8 (G11 RMSNorm out)
    wire signed [K_PAR*A_WIDTH-1:0] seam_r1_rd;   // G1 -> G4
    wire signed [K_PAR*A_WIDTH-1:0] seam_r2_rd;   // G6 -> (R3 residual + post-attn norm)
    wire signed [K_PAR*A_WIDTH-1:0] seam_r5_rd;   // silu -> G9
    wire signed [K_PAR*A_WIDTH-1:0] seam_r8_rd;   // G11 RMSNorm -> g11_lmdown input (FIX-G)
    wire signed [K_PAR*A_WIDTH-1:0] low_rank_rd;  // FIX-G: g11_lmdown out (low_rank[256]) -> G12
    // FIX-D: pre_mlp seam = post-attention RMSNorm(h1) output (frac=12). Feeds
    // G7g/G7u (stage_sel 2/3) instead of the raw G6 o_proj (seam_r2_rd), which
    // bypassed the R3 residual + post-attn norm. Forward-declared; the pre_mlp
    // buffer + post-attn norm are built in the residual block (h1 visible there).
    localparam int PREMLP_NCHUNKS = 3072 / K_PAR;                       // 192
    localparam int PREMLP_CW      = (PREMLP_NCHUNKS > 1) ? $clog2(PREMLP_NCHUNKS) : 1; // 8
    wire signed [K_PAR*A_WIDTH-1:0] seam_premlp_rd;   // pre_mlp chunk (G7g/G7u input)
    wire                            seam_premlp_full;  // post-attn norm done (all 192 in)
    wire [PREMLP_CW-1:0]            premlp_rd_idx;      // chunk read pointer (driven below)
    // S7b: input-full gate taps (driven by the seam stage_io_buffer .full ports)
    wire                            seam_r2_full; // R2 (G6 o_proj) all 192 chunks in
    wire                            seam_r5_full; // R5 (silu out) all 512 chunks in

    // ---- WO-025 Stage-B' (TDP dual-read) port-B plumbing ----------------
    // The array's ODD-chunk index (chunk_idx_b_out). At DUAL_PORT=0 it mirrors
    // the even index (a_chunk_b ignored by the array); at DUAL_PORT=1 the lane
    // pair-walks: port A = even chunk c, port B = odd chunk c+1 (lockstep).
    // Every body-GEMV activation seam sources a PARALLEL read at this index.
    wire [15:0]                     array_chunk_idx_b_out;
    wire signed [K_PAR*A_WIDTH-1:0] seam_r1_rd_b;      // G4  (e/h concat, port B)
    wire signed [K_PAR*A_WIDTH-1:0] seam_g5g6_rd_b;    // G6  (S7a seam, port B)
    wire signed [K_PAR*A_WIDTH-1:0] seam_premlp_rd_b;  // G7g/G7u (pre_mlp, port B)
    wire signed [K_PAR*A_WIDTH-1:0] seam_r5_rd_b;      // G9  (silu out, port B)
    wire signed [K_PAR*A_WIDTH-1:0] low_rank_rd_b;     // G12 (low_rank, port B)

    // S7a: G6 reads the full-width G5->G6 seam (24-head attn-out, chunk-indexed
    // by array_chunk_idx_out) when S7A_FRONT_SEQ=1; otherwise the legacy L8c
    // head-0 broadcast island (attn_out_chunk_latched).
    wire signed [K_PAR*A_WIDTH-1:0] g6_act_in =
        S7A_FRONT_SEQ ? seam_g5g6_rd : attn_out_chunk_latched;
    // WO-025 port-B: the legacy L8c arm is a single latched BROADCAST chunk
    // (index-independent), so the odd-chunk value is the same latched chunk.
    wire signed [K_PAR*A_WIDTH-1:0] g6_act_in_b =
        S7A_FRONT_SEQ ? seam_g5g6_rd_b : attn_out_chunk_latched;

    // FIX-D: G7g/G7u now read the pre_mlp seam (post-attn RMSNorm of h1), NOT
    // the raw G6 o_proj. The read pointer tracks the G7 GEMV's current chunk
    // request, exactly as seam_r2_rd did for the bypassed path.
    assign premlp_rd_idx = array_chunk_idx_out[PREMLP_CW-1:0];
    // WO-025 port-B: parallel ODD-chunk read pointer for the pre_mlp seam.
    wire [PREMLP_CW-1:0] premlp_rd_idx_b = array_chunk_idx_b_out[PREMLP_CW-1:0];
    // ACT-STAGE-SEL FIX (ported from pdi6_s7c_g8, rescue_20260606): the
    // activation mux must key on the EFFECTIVE stage (`act_stage_sel`), NOT the
    // bare top_fsm `stage_sel`. During the TREE walk the GEMV array runs G12
    // (array_stage_sel forced to 4'd5) while the top_fsm sits in S_TREE with
    // stage_sel=4'd0 (default). Keying the mux on the bare stage_sel made the
    // tree-driven G12 read seam_r1_rd (the G1 seam) instead of low_rank_rd, so
    // the cheap-logit GEMV got the wrong activation and the tree saw no valid
    // logits. `act_stage_sel == stage_sel` for every body stage
    // (tt_stage_g12=0), so this is a no-op for G4/G6/G7g/G7u/G9 and ONLY
    // repoints the tree-walk G12 to low_rank_rd. (Identical expression to
    // `array_stage_sel` defined further down, recomputed here because that net
    // is declared after this mux.)
    wire [3:0] act_stage_sel = tt_stage_g12 ? 4'd5 : stage_sel;
    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_connected =
          (act_stage_sel == 4'd0) ? seam_r1_rd              // G4
        : (act_stage_sel == 4'd1) ? g6_act_in               // G6 (S7a seam / L8c)
        : (act_stage_sel == 4'd2) ? seam_premlp_rd          // G7g <- pre_mlp (FIX-D)
        : (act_stage_sel == 4'd3) ? seam_premlp_rd          // G7u <- pre_mlp (FIX-D, shared)
        : (act_stage_sel == 4'd4) ? seam_r5_rd              // G9
        : (act_stage_sel == 4'd5) ? low_rank_rd             // G12 <- low_rank[256] (FIX-G)
        :                           a_chunk;

    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_real =
        CONNECT_SEAMS ? a_chunk_connected
                      : ((stage_sel == 4'd1) ? g6_act_in : a_chunk);

    // ---- WO-025 Stage-B': ODD-chunk (port-B) activation feed --------------
    // SAME act_stage_sel mux as a_chunk_connected, but every arm reads its
    // seam's port-B at array_chunk_idx_b_out. No STAGE_INJECT arm: inject is
    // a default-off debug path; DUAL_PORT=1 + stage-inject is unsupported
    // (the injected chunk is a single broadcast, not pair-indexed). '0 when
    // seams are disconnected — TB direct-feed configs run DUAL_PORT=0
    // (elab-asserted below).
    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_b_connected =
          (act_stage_sel == 4'd0) ? seam_r1_rd_b            // G4
        : (act_stage_sel == 4'd1) ? g6_act_in_b             // G6 (S7a seam / L8c broadcast)
        : (act_stage_sel == 4'd2) ? seam_premlp_rd_b        // G7g
        : (act_stage_sel == 4'd3) ? seam_premlp_rd_b        // G7u
        : (act_stage_sel == 4'd4) ? seam_r5_rd_b            // G9
        : (act_stage_sel == 4'd5) ? low_rank_rd_b           // G12
        :                           '0;
    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_b_mux =
        CONNECT_SEAMS ? a_chunk_b_connected : '0;
`ifndef SYNTHESIS
    initial if (DUAL_PORT && !(GEMV_SCALED && !GEMV_KPAR64 && CONNECT_SEAMS
                               && S7A_FRONT_SEQ && BACK_SEQ && SEAM_R1_FULLW
                               && (BODY_BATCH_B == 1)))
        $error("eagle4_drafter: DUAL_PORT=1 requires the production seam config (GEMV_SCALED=1, GEMV_KPAR64=0, CONNECT_SEAMS=1, S7A_FRONT_SEQ=1, BACK_SEQ=1, SEAM_R1_FULLW=1, BODY_BATCH_B=1) so every a_chunk_b arm is a real port-B seam read.");
    // LANES-256 elaboration guard (LANES256_GEMM_STUDY.md):
    //   - only 128 (certified default) and 256 (the quad lever) are legal.
    //     192 is DEAD-by-geometry: G7 N=8192/192 is not an integer, so
    //     sg_stage_cols floor-division would SILENTLY DROP 128 G7 output
    //     columns. Any other value breaks stage divisibility or D_HEAD
    //     packing.
    //   - LANES=256 is certified ONLY on the scaled SLR array arm (the
    //     legacy pilot arms keep internal 128-lane bank geometry).
    initial if (!(LANES == 128 || LANES == 256))
        $error("eagle4_drafter: LANES=%0d unsupported — only 128 (default) and 256 (quad lever) divide every stage N and pack D_HEAD=128 heads (192 silently drops G7 columns).", LANES);
    initial if (LANES != 128 && !(GEMV_SCALED && !GEMV_KPAR64))
        $error("eagle4_drafter: LANES=%0d requires the scaled SLR array arm (GEMV_SCALED=1, GEMV_KPAR64=0) — the pilot array arms carry fixed 128-lane bank geometry.", LANES);
`endif

    // ---- PER-STAGE INJECT: GEMV body activation feed (G4/G6/G7g/G7u/G9/G12) ----
    // Maps the active GEMV stage (act_stage_sel) to the shared inject index space:
    //   act 0=G4(idx1) 1=G6(idx5) 2=G7g(idx6) 3=G7u(idx7) 4=G9(idx9) 5=G12(idx12).
    // When the matching inject_en bit is set, the GEMV reads the golden chunk from
    // inj_bus instead of the real seam — isolating THAT GEMV's transform. The inject
    // chunk is the low A_WIDTH bits of the first K_PAR=16 32-bit words the host
    // loaded (one golden activation element per 32-bit word; simple unambiguous ABI).
    // Default-off / STAGE_INJECT=0: inject_en=0 -> a_chunk_mux == a_chunk_real,
    // bit-identical to the pre-inject top.
    wire signed [K_PAR*A_WIDTH-1:0] inj_act_chunk;
    genvar gia;
    generate
        for (gia = 0; gia < K_PAR; gia = gia + 1) begin : g_inj_act
            assign inj_act_chunk[gia*A_WIDTH +: A_WIDTH] = inj_bus[gia*32 +: A_WIDTH];
        end
    endgenerate
    wire inj_gemv_active =
          (act_stage_sel == 4'd0) ? inject_en[1]    // G4
        : (act_stage_sel == 4'd1) ? inject_en[5]    // G6
        : (act_stage_sel == 4'd2) ? inject_en[6]    // G7g
        : (act_stage_sel == 4'd3) ? inject_en[7]    // G7u
        : (act_stage_sel == 4'd4) ? inject_en[9]    // G9
        : (act_stage_sel == 4'd5) ? inject_en[12]   // G12
        :                           1'b0;
    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_mux =
        inj_gemv_active ? inj_act_chunk : a_chunk_real;

    // ----------------------------------------------------------------
    // Shared 128-lane GEMV array (handles G4, G6, G7g, G7u, G9, G12)
    // One start pulse per stage; same array, different weight bank slice.
    //
    // M6b: during the tree stage the tree_tile_stream controller re-runs the
    // G12 GEMV once per vocab tile (1002 tiles), so it muxes in its own start
    // pulse, tile_idx (column-block index) and stage_sel (forced to G12=4'd5).
    // The tiler holds `tt_stage_g12` for the whole walk; the top_fsm / external
    // tile_idx port own these signals at every other time. Exactly one driver
    // per cycle — multi-driver-safe.
    // ----------------------------------------------------------------
    // ================================================================
    // S7b — GENERIC back-half OUTPUT-TILE sequencer (the NCHUNKS=1 -> N-tile fix)
    // ================================================================
    // ONE mini-FSM reused across the four back-half shared-array GEMV stages:
    //   G6  (o_proj, stage_sel=1) : 24 tiles
    //   G7g (gate,   stage_sel=2) : 64 tiles
    //   G7u (up,     stage_sel=3) : 64 tiles
    //   G9  (down,   stage_sel=4) : 24 tiles
    // The top_fsm stays STAGE-level (pulses gX_start once on entering each stage,
    // waits on gX_done). This sequencer walks ALL the active stage's output tiles
    // while the stage FSM holds that state, re-issuing the array start per tile
    // (with a per-tile col_offset_acc SETTLE — the §3.2 H1 generalization of the
    // array_start_fsm_q stage-entry fix), and only asserts the stage's gX_done on
    // the LAST tile (§2.3). Per tile:
    //   1. hold bk_tile_idx STABLE for the whole in-flight window (so the
    //      array_done that arrives belongs to that tile + the seam write-expander
    //      keys on the right array_tile_idx).
    //   2. SETTLE 1 cycle so weight_bank's REGISTERED col_offset_acc re-bases to
    //      this tile (the array_start_fsm_q register adds another cycle, matching
    //      the "@(posedge) x2 let banks settle" the per-stage TBs used + the tree
    //      tiler's between-tile settle). INPUT-FULL gate on the first tile (§3.1).
    //   3. ISSUE the raw start (1-cycle pulse) -> array runs this tile's K-reduction.
    //   4. on array_done: tile != last -> advance + settle; tile == last -> done.
    // No valid-high in drain (H2): the start is a single-cycle pulse, then quiescent
    // until the next tile's settle. BACK_SEQ=0 disables this (legacy 1-tile pilot).
    // ----------------------------------------------------------------
    // per-stage tile counts (stage_sel-indexed); 0 for non-back-half stages.
    function automatic [15:0] bk_stage_ntiles;
        input [3:0] ssel;
        begin
            case (ssel)
                // LANES-256: tile counts are N/LANES (the old literals were the
                // 128-lane case: 24/64/64/48). @256 they halve: 12/32/32/24.
                4'd1: bk_stage_ntiles = 16'(3072 / LANES);   // G6  o_proj  N=3072
                4'd2: bk_stage_ntiles = 16'(8192 / LANES);   // G7g gate    N=8192
                4'd3: bk_stage_ntiles = 16'(8192 / LANES);   // G7u up      N=8192
                // F1 (FORREASON): G9's TRUE checkpoint width is N=2H=6144
                // (down_proj.qweight [512,6144], enable_2hs) = 48 col-tiles:
                // tiles 0..23 = to_logits (the depth-0 frontier path), tiles
                // 24..47 = for_reasoning (the depth>=1 re-key seed). The weight
                // bank + scale bank address maps ALREADY reserve all 48 columns
                // (stage_geometry.svh sg_stage_N(4)=6144, base 56,832..81,408),
                // so this is a pure tile-walk length change: ZERO URAM/BRAM delta.
                // FORREASON=0 -> literal 24, byte-identical to pre-F1.
                4'd4: bk_stage_ntiles = FORREASON ? 16'(6144 / LANES) : 16'(3072 / LANES);   // G9 down N=6144 / 3072
                default: bk_stage_ntiles = 16'd1; // not a back-half tiled stage
            endcase
        end
    endfunction
    wire bk_stage_active = BACK_SEQ && !tt_stage_g12 &&
                           ((stage_sel == 4'd1) || (stage_sel == 4'd2) ||
                            (stage_sel == 4'd3) || (stage_sel == 4'd4));
    // the start that ENTERS the current back-half stage (top_fsm's per-stage pulse)
    wire bk_stage_start = (stage_sel == 4'd1) ? g6_start
                        : (stage_sel == 4'd2) ? g7g_start
                        : (stage_sel == 4'd3) ? g7u_start
                        : (stage_sel == 4'd4) ? g9_start
                        :                       1'b0;
    // per-stage input-full gate (first tile only). FIX-D: G7g/G7u now read the
    // pre_mlp seam (post-attn RMSNorm of h1=R3 residual), so they must wait for
    // the post-attn norm to finish (seam_premlp_full = all 192 pre_mlp chunks in),
    // which itself follows G6 o_proj -> R3 residual -> norm. G9 reads R5 (silu, 512).
    // G6 input is the G5->G6 seam (S7a / preloaded); not gated here.
    wire bk_input_full = (stage_sel == 4'd2) ? seam_premlp_full
                       : (stage_sel == 4'd3) ? seam_premlp_full
                       : (stage_sel == 4'd4) ? seam_r5_full
                       :                       1'b1;   // G6 / default: no gate

    typedef enum logic [1:0] {
        BKSQ_IDLE   = 2'd0,
        BKSQ_SETTLE = 2'd1,   // 1 cycle: let col_offset_acc re-base to this tile
        BKSQ_ISSUE  = 2'd2,   // pulse the raw start for bk_tile_idx
        BKSQ_WAIT   = 2'd3    // wait for this tile's array_done
    } bkseq_state_e;
    bkseq_state_e        bkseq_state;
    reg [15:0]           bk_tile_cnt;      // current output tile within the stage
    reg [15:0]           bk_ntiles_lat;    // latched STAGE_NTILES for the active stage
    reg [3:0]            bk_stage_lat;     // which stage this walk belongs to
    reg                  bk_seq_start_r;   // 1-cycle raw start for the tile
    reg                  bk_g6_done_r, bk_g7g_done_r, bk_g7u_done_r, bk_g9_done_r;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            bkseq_state    <= BKSQ_IDLE;
            bk_tile_cnt    <= '0;
            bk_ntiles_lat  <= 16'd1;
            bk_stage_lat   <= 4'd0;
            bk_seq_start_r <= 1'b0;
            bk_g6_done_r   <= 1'b0; bk_g7g_done_r <= 1'b0;
            bk_g7u_done_r  <= 1'b0; bk_g9_done_r  <= 1'b0;
        end else begin
            bk_seq_start_r <= 1'b0;   // default: start is a 1-cycle pulse
            bk_g6_done_r   <= 1'b0;   // default: per-stage done is a 1-cycle pulse
            bk_g7g_done_r  <= 1'b0;
            bk_g7u_done_r  <= 1'b0;
            bk_g9_done_r   <= 1'b0;
            case (bkseq_state)
                BKSQ_IDLE: begin
                    // enter on the top_fsm's per-stage start while in a back-half stage
                    if (bk_stage_active && bk_stage_start) begin
                        bk_tile_cnt   <= '0;
                        bk_ntiles_lat <= bk_stage_ntiles(stage_sel);
                        bk_stage_lat  <= stage_sel;
                        bkseq_state   <= BKSQ_SETTLE;
                    end
                end
                BKSQ_SETTLE: begin
                    // bk_tile_cnt is stable; spend 1 cycle so col_offset_acc re-bases.
                    // first tile also waits for the input seam to be full (§3.1).
                    if (bk_tile_cnt != 16'd0 || bk_input_full)
                        bkseq_state <= BKSQ_ISSUE;
                end
                BKSQ_ISSUE: begin
                    bk_seq_start_r <= 1'b1;          // raw start (further delayed 1cy)
                    bkseq_state    <= BKSQ_WAIT;
                end
                BKSQ_WAIT: begin
                    if (array_done) begin
                        if (bk_tile_cnt == bk_ntiles_lat - 16'd1) begin
                            // LAST tile -> assert this stage's done (1 cycle)
                            case (bk_stage_lat)
                                4'd1: bk_g6_done_r  <= 1'b1;
                                4'd2: bk_g7g_done_r <= 1'b1;
                                4'd3: bk_g7u_done_r <= 1'b1;
                                4'd4: bk_g9_done_r  <= 1'b1;
                                default: ;
                            endcase
                            bkseq_state <= BKSQ_IDLE;
                        end else begin
                            bk_tile_cnt <= bk_tile_cnt + 16'd1;
                            bkseq_state <= BKSQ_SETTLE;
                        end
                    end
                end
                default: bkseq_state <= BKSQ_IDLE;
            endcase
        end
    end
    wire [15:0] bk_seq_tile_idx = bk_tile_cnt;

    // ----------------------------------------------------------------
    // S7b — silu 512-pulse `en` FSM.  Drives silu across ALL 512 chunks of the
    // gate/up intermediate (R4g/R4u seams) and the R5 read pointer (r4_silu_rd_idx).
    // The top_fsm pulses g7p_start once on entering the silu stage; this FSM then
    // holds silu `en` HIGH for G7_NCHUNKS=512 cycles, advancing r4_silu_rd_idx
    // 0..511 in lockstep so silu reads chunk c of BOTH seams on the cycle its `en`
    // is high for chunk c. silu's internal `valid <= en_pipe; en_pipe <= en` then
    // emits exactly 512 ordered silu_valid pulses (2-cycle latency) -> the R5 seam
    // captures all 512 (AUTO_SEQ). g7p_done is derived from silu_valid (last pulse).
    // ----------------------------------------------------------------
    reg [G7_CW:0]  silu_en_cnt;     // 0..512 (one extra bit for the terminal count)
    reg            silu_en_run;
    reg            silu_en_r;        // the held en (512 cycles)
    reg [G7_CW-1:0] silu_rd_idx_r;   // chunk index ALIGNED to silu_en_r (see fix below)
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            silu_en_cnt <= '0; silu_en_run <= 1'b0; silu_en_r <= 1'b0; silu_rd_idx_r <= '0;
        end else begin
            silu_en_r <= 1'b0;       // default low
            if (BACK_SEQ && g7p_start) begin
                // start the 512-pulse sweep next cycle (input seams R4g/R4u full)
                silu_en_cnt <= '0;
                silu_en_run <= 1'b1;
            end else if (silu_en_run) begin
                if (silu_en_cnt == (G7_CW+1)'(G7_NCHUNKS)) begin
                    silu_en_run <= 1'b0;          // 512 en-cycles emitted
                end else begin
                    silu_en_r     <= 1'b1;        // hold en high for this chunk
                    // S7b RD-IDX ALIGN FIX (idx-trace TB caught this): silu_mul
                    // samples gate/up COMBINATIONALLY on the cycle `en` is high
                    // (silu_mul.sv:65). Because silu_en_r and silu_en_cnt are set in
                    // the SAME always_ff, by the time silu_en_r is OBSERVED high
                    // silu_en_cnt has ALREADY incremented -> a combinational
                    // `r4_silu_rd_idx = silu_en_cnt` presented chunk c+1 (skipping
                    // chunk 0, reading chunk 512 last). Latch the read index in
                    // LOCKSTEP with silu_en_r so the chunk presented on the en-high
                    // cycle equals the `silu_en_cnt` value that decided to pulse en.
                    silu_rd_idx_r <= silu_en_cnt[G7_CW-1:0];
                    silu_en_cnt   <= silu_en_cnt + 1'b1;
                end
            end
        end
    end
    // read pointer: chunk presented to silu on the cycle its `en` is high. Driven
    // by the registered silu_rd_idx_r (aligned to silu_en_r) — NOT the live
    // silu_en_cnt (one ahead, see the RD-IDX ALIGN FIX above).
    //
    // FIX-G7p (R4 silu-seam COMBREAD stale-by-1, 2026-06-13, GRAND_BASE_FIXES):
    // the R4g/R4u seams below carry .RD_LAT(D1_RDLAT). When D1_RDLAT=1 the seam
    // read is REGISTERED (silicon BRAM 1-cycle synchronous read) — the data for
    // the address presented at cycle T arrives at T+1. silu_mul_fix samples
    // gate/up COMBINATIONALLY on the `en`-high cycle (silu_mul_fix.sv:68). So the
    // read index must be presented ONE cycle EARLIER (a +1 lead), so the
    // registered seam value lands the cycle silu uses it. silu_en_cnt is exactly
    // that +1 lead: at the FSM edge that sets silu_en_r=1 for chunk c (en high
    // next cycle), silu_en_cnt holds c (pre-increment) -> a combinational
    // r4_silu_rd_idx = silu_en_cnt presents c the cycle BEFORE en-high-for-c, so
    // the registered read of buf_mem[c] lands on the en-high-for-c cycle.
    // D1_RDLAT=0 keeps the legacy comb-read path: silu_rd_idx_r (latched in
    // lockstep with silu_en_r) -> byte-identical to the legacy combread golden.
    assign r4_silu_rd_idx = D1_RDLAT ? silu_en_cnt[G7_CW-1:0] : silu_rd_idx_r;
    // BACK_SEQ silu `en` replaces the single g7p_start pulse (legacy when BACK_SEQ=0).
    // (silu_en_eff forward-declared at the silu_layer instance.)
    assign silu_en_eff = BACK_SEQ ? silu_en_r : g7p_start;

    // silu sweep completion: count the 512 silu_valid pulses (cleared at g7p_start)
    // and pulse silu_sweep_done on the LAST (512th) valid. Under BACK_SEQ, g7p_done
    // must wait for the FULL sweep, NOT the first silu_valid (which would let top_fsm
    // leave the silu stage after 1 chunk — the early-exit bug the first gate caught).
    reg [G7_CW:0] silu_vld_cnt;
    // silu_sweep_done_r declared up at the silu instance (decl-before-use).
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            silu_vld_cnt <= '0; silu_sweep_done_r <= 1'b0;
        end else begin
            silu_sweep_done_r <= 1'b0;
            if (g7p_start) silu_vld_cnt <= '0;
            else if (silu_valid) begin
                silu_vld_cnt <= silu_vld_cnt + 1'b1;
                if (silu_vld_cnt == (G7_CW+1)'(G7_NCHUNKS-1))   // 512th valid (0..511)
                    silu_sweep_done_r <= 1'b1;
            end
        end
    end

    // ================================================================
    // S7a — G4 per-stage OUTPUT-TILE sequencer (the NCHUNKS=1 -> 72-tile fix)
    // ================================================================
    // S7a: forward-declare the R1 seam "full" flag so the G4 sequencer's
    // input-full gate (SETTLE state, first tile) can reference it before the
    // R1 generate block drives it further down (VRFC 10-3380).
    wire seam_r1_full;
    // The top_fsm stays stage-level: it pulses g4_start once on entering S_G4
    // and waits on g4_done. This mini-FSM walks all G4_TOT_TILES=72 output tiles
    // (Q col-tiles 0..23, K 24..47, V 48..71) while the stage FSM holds S_G4,
    // and only asserts g4_done on the LAST tile (71). Per tile:
    //   1. hold g4_seq_tile_idx STABLE during the whole in-flight window (so the
    //      array_done that arrives belongs to that tile and the S2 capture writes
    //      the right q/k/v_full_buf slot — capture keys on array_tile_idx).
    //   2. PER-TILE SETTLE (the §3.2 H1 generalization of the array_start_fsm_q
    //      stage-entry fix): each new tile re-bases weight_bank's registered
    //      col_offset_acc, so wait one cycle (G4_SEQ_SETTLE) after g4_seq_tile_idx
    //      advances before re-issuing the raw start. The existing array_start_fsm_q
    //      register adds ANOTHER cycle, so the array sees the new col_offset_acc
    //      already settled (= the "@(posedge clk) x2 let banks settle" the
    //      per-stage isolation TBs used; the tree tiler tt_gemv_start settles the
    //      same way between its 1002 tiles).
    //   3. re-issue array_start (pulsed into array_start_fsm_raw) for tile_cnt.
    //   4. on array_done: tile 0..70 -> advance + settle; tile 71 -> g4_seq_done.
    // No valid-high in drain (H2): the start pulse is a single cycle, then the
    // sequencer is quiescent until the next tile's settle.
    // S7A_FRONT_SEQ=0 disables this (legacy single-tile pilot from host tile_idx).
    // ----------------------------------------------------------------
    localparam int G4_SEQ_TW = $clog2(G4_TOT_TILES);    // 0..71 width (7)
    typedef enum logic [1:0] {
        G4SQ_IDLE   = 2'd0,   // not in G4 (or S7A_FRONT_SEQ off)
        G4SQ_SETTLE = 2'd1,   // 1 cycle: let col_offset_acc re-base to this tile
        G4SQ_ISSUE  = 2'd2,   // pulse the raw start for g4_seq_tile_idx
        G4SQ_WAIT   = 2'd3    // wait for this tile's array_done
    } g4seq_state_e;
    g4seq_state_e            g4seq_state;
    reg [G4_SEQ_TW-1:0]      g4_tile_cnt;     // current G4 output tile 0..71
    reg                      g4_seq_start_r;  // 1-cycle raw start for the tile
    reg                      g4_seq_done_r;   // pulsed on tile-71 array_done
    wire                     g4_in_stage = (stage_sel == G4_STAGE_SEL) && !tt_stage_g12;
    wire                     g4_seq_en   = S7A_FRONT_SEQ && g4_in_stage;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            g4seq_state    <= G4SQ_IDLE;
            g4_tile_cnt    <= '0;
            g4_seq_start_r <= 1'b0;
            g4_seq_done_r  <= 1'b0;
        end else begin
            g4_seq_start_r <= 1'b0;   // default: start is a 1-cycle pulse
            g4_seq_done_r  <= 1'b0;   // default: done is a 1-cycle pulse
            case (g4seq_state)
                G4SQ_IDLE: begin
                    // enter on the top_fsm's g4_start while in S_G4 with seq enabled
                    if (g4_seq_en && g4_start) begin
                        g4_tile_cnt <= '0;
                        g4seq_state <= G4SQ_SETTLE;
                    end
                end
                G4SQ_SETTLE: begin
                    // g4_tile_cnt is already stable; spend 1 cycle so the
                    // weight_bank col_offset_acc re-bases to this tile_idx.
                    // INPUT-FULL GATE (§3.1): for the FIRST tile, also wait until
                    // the R1 seam holds all 192 chunks of the G1 RMSNorm output,
                    // so the array's chunk-191 read is valid. (g4_start follows
                    // g1_done so this is normally already true; the gate makes the
                    // dependency explicit and robust.)
                    if (g4_tile_cnt != '0 || seam_r1_full)
                        g4seq_state <= G4SQ_ISSUE;
                end
                G4SQ_ISSUE: begin
                    g4_seq_start_r <= 1'b1;          // raw start (further delayed 1cy)
                    g4seq_state    <= G4SQ_WAIT;
                end
                G4SQ_WAIT: begin
                    if (array_done) begin
                        if (g4_tile_cnt == G4_SEQ_TW'(G4_TOT_TILES-1)) begin
                            g4_seq_done_r <= 1'b1;     // last tile -> stage done
                            g4seq_state   <= G4SQ_IDLE;
                        end else begin
                            g4_tile_cnt <= g4_tile_cnt + 1'b1;
                            g4seq_state <= G4SQ_SETTLE;
                        end
                    end
                end
                default: g4seq_state <= G4SQ_IDLE;
            endcase
        end
    end
    wire [15:0] g4_seq_tile_idx = {{(16-G4_SEQ_TW){1'b0}}, g4_tile_cnt};

    // S7a (M2): the G4 stage's array_start is now the per-tile sequencer pulse
    // (g4_seq_start_r) when S7A_FRONT_SEQ=1, NOT the top_fsm's single g4_start.
    // The legacy pilot keeps g4_start. All other stages unchanged. ONE
    // OR-reduction into array_start_fsm_raw through the single array_start_fsm_q.
    wire g4_array_start_raw = S7A_FRONT_SEQ ? g4_seq_start_r : g4_start;
    wire array_start_fsm_raw = g4_array_start_raw | (BACK_SEQ ? bk_seq_start_r
                               : (g6_start | g7g_start | g7u_start | g9_start)) | g12_start;
    // B2/B4 SETTLE FIX (back-to-back GEMV-stage start hazard, found by the
    // top-level G7g xsim gate): top_fsm drives `stage_sel` combinationally on
    // `state` and pulses the per-stage start the SAME cycle the FSM enters that
    // state. But weight_bank_multistage's `col_offset_acc` (= stage_base +
    // col*cpc) is REGISTERED — it updates ONE cycle AFTER stage_sel changes. So
    // a GEMV that starts coincident with its stage_sel change reads the PREVIOUS
    // stage's base for the first chunk-lookahead, corrupting the first chunk(s)
    // of the accumulator (measured: G7g unscaled acc 102 vs golden 132). The
    // per-stage isolation TBs masked this by inserting "@(posedge clk) x2 // let
    // banks settle col_base_acc" before `start`; the integrated FSM did not.
    // FIX: delay the FSM-driven array start by ONE cycle so col_offset_acc has
    // settled to the new stage base before the lane FSM leaves S_IDLE. The tree
    // tiler's start (tt_gemv_start) already settles between tiles, so only the
    // FSM arm is delayed. Single always_ff, one writer — multi-driver-safe.
    reg array_start_fsm_q;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) array_start_fsm_q <= 1'b0;
        else           array_start_fsm_q <= array_start_fsm_raw;
    end
    // TT-SETTLE-FIX (2026-06-06, ROUND-15, ported from pdi6_s7c_g8): the comment
    // above ("the tiler's start already settles between tiles") is FALSE — the
    // tiler asserts gemv_tile_idx AND gemv_tile_start on the SAME cycle while the
    // weight/scale bank col_offset_acc is a REGISTERED 1-cycle re-base, so every
    // tt tile's chunk-0 read used the PREVIOUS tile's weight column (1/16-chunk
    // cross-column contamination; inflated high-vocab logits dominated the cheap
    // top-8). Reproduced bit-exact by a 1-chunk weight-lag model
    // (XSIM_FULLBODY_BISECT ROUND-15 §4). FIX: route the tiler start through the
    // SAME one-cycle settle the body path uses. tile_idx holds across TT_WAIT,
    // so the +1 only moves the start. +1 cyc/tile; bit-identical otherwise.
    reg tt_gemv_start_q;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) tt_gemv_start_q <= 1'b0;
        else           tt_gemv_start_q <= tt_gemv_start;
    end
    wire array_start     = tt_stage_g12 ? tt_gemv_start_q   : array_start_fsm_q;
    // M3 (UNIFIED tile-index mux): the tree tiler owns it during G12; the S7a G4
    // sequencer owns it during G4 (front-half, S7A_FRONT_SEQ); the S7b back-half
    // generic sequencer owns it during G6/G7g/G7u/G9; the host tile_idx (legacy)
    // otherwise. ONE mux, no two FSMs fighting it. array_tile_idx is forward-
    // declared (as a `wire`) up at the G4 capture decode (VRFC 10-3380), so this
    // is a bare `assign`.
    assign array_tile_idx =
          tt_stage_g12      ? tt_gemv_tile_idx
        : g4_seq_en         ? g4_seq_tile_idx
        : bk_stage_active   ? bk_seq_tile_idx
        :                     tile_idx;
    wire [3:0]  array_stage_sel = tt_stage_g12 ? 4'd5 /*G12*/    : stage_sel;
    // array_done forward-declared above (M5 capture decode references it).

    // ----------------------------------------------------------------
    // PDI#5-COMPLETE: GEMV array — generate swap (GEMV_KPAR64).
    //   GEMV_KPAR64=0 -> closing K_PAR=16 array (frontier-exact, Gate-A path).
    //   GEMV_KPAR64=1 -> #188 K_PAR=64 wide-bank array (4.00x col-rate,
    //     URAM-neutral, bit-exact-invariant per KPAR64_ALLSHAPES). Pin-compatible
    //     on the compute ports (a_chunk/qz_raw/stage_sel/tile_idx ->
    //     acc_out_per_lane/done); the only delta is the wide weight-write row.
    // The kpar64 arm widens the pilot K_PAR=16 activation chunk into the low
    // 16 of the 64-lane a_chunk (upper 48 = 0) — the pilot single-chunk
    // head-0 broadcast (NCHUNKS pilot semantics). The boot row assembles 4
    // consecutive 32-bit qweight words into one 128-bit wide row. Both arms
    // drive the SAME array_acc_out / array_done — exactly one is elaborated.
    // ----------------------------------------------------------------
    localparam int ARR_KPAR = GEMV_KPAR64 ? 64 : K_PAR;
    localparam int ARR_NSG  = ARR_KPAR / 16;        // 4 @K_PAR=64, 1 @16
    localparam int ARR_ROWW = ARR_NSG * 32;         // 128 @64, 32 @16

    // widen the pilot K_PAR(=16)-wide activation chunk to ARR_KPAR lanes
    wire signed [ARR_KPAR*A_WIDTH-1:0] a_chunk_wide;
    generate
        if (ARR_KPAR == K_PAR) begin : g_awide_eq
            assign a_chunk_wide = a_chunk_mux;
        end else begin : g_awide_pad
            // low K_PAR lanes = pilot head-0 chunk, upper lanes 0
            assign a_chunk_wide = { {((ARR_KPAR-K_PAR)*A_WIDTH){1'b0}}, a_chunk_mux };
        end
    endgenerate

    // B2: array_acc_out is now driven from an internal net (array_acc_int) so
    // the scaled arm can requantize result_q -> INT into the SAME bus the
    // unscaled arms drive directly. EXACTLY ONE generate arm elaborates, so
    // exactly one writer of array_acc_int.
    wire signed [32*LANES-1:0] array_acc_int;
    assign array_acc_out = array_acc_int;

    // QZ-FIX (2026-05-27): drive the array's GPTQ zero point from the MODEL
    // CONSTANT (GPTQ_QZERO_RAW=1 => effective zero point 2), not the external
    // qz_raw port (which the S8 harness / any unset host leaves at 0, giving a
    // uniform +1 weight bias that saturated the G6 o_proj seam to -32768). All
    // 7 GPTQ tensors are uniformly raw-qzero=1, so this single constant is
    // exact for every stage (G4/G6/G7g/G7u/G9/G12). The external port is kept
    // for ABI compatibility but is no longer load-bearing.
    wire [1:0] qz_eff = GPTQ_QZERO_RAW;

    generate
    if (GEMV_SCALED && !GEMV_KPAR64) begin : g_array_scaled
        // -------------------------------------------------------------
        // B2 SCALED array — per-group fp16 GPTQ dequant scale (Q3.13).
        //   weights URAM (weight_bank_multistage) + scales BRAM (scale_bram),
        //   gemv_lane_scaled_stream(SCALED=1). result_q_per_lane is the scaled
        //   fixed-point output (RES_WIDTH=48, GEMV_SCALE_FRAC frac bits). We
        //   requantize result_q >> SCALE_FRAC, saturating to INT32, to drive
        //   array_acc_int — the dequant magnitude in the INT space the seams +
        //   tree consume. The scale-load port (sc_*) is fed by the dma_sc_*
        //   loader (#157) OR'd with a TB direct-drive scale path (tb_sc_*).
        // -------------------------------------------------------------
        localparam int RES_W = 48;
        wire signed [RES_W*LANES-1:0] result_q_per_lane;
        wire signed [32*LANES-1:0]    acc_dbg_per_lane;   // unscaled (debug, unused)

        // -------------------------------------------------------------
        // TREE-BATCH B=2 SILICON BATCH-TEST DRIVE + CAPTURE (build 0xE4B7110F).
        //   BODY_BATCH_B>1 => the array carries BB1=BODY_BATCH_B-1 sibling lane-sets.
        //   The host BATCH-TEST reg (0x138) drives the sibling activation column
        //   a_chunk_batch = batch_test_en ? (negate ? -a_chunk : a_chunk) : '0, and
        //   this block latches BOTH the b0 (result_q_per_lane) and the sibling
        //   (result_q_batch) scaled accumulator of the host-selected lane on each
        //   array `done`. Because the sibling MACs the SAME URAM weight / BRAM scale
        //   stream as b0 and result_q is LINEAR & HOMOGENEOUS in the activation, the
        //   host asserts sib == ±b0 (negate arm) — a weight-agnostic silicon invariant
        //   with teeth (dead/copy-of-b0 sibling fails the negate arm). INERT in the
        //   normal forward (batch_test_en=0 => a_chunk_batch='0 -> siblings idle; b0
        //   result_q_per_lane is INDEPENDENT of a_chunk_batch, so M7 is bit-exact).
        // -------------------------------------------------------------
        localparam int BT_BB1 = (BODY_BATCH_B > 1) ? (BODY_BATCH_B - 1) : 1;
        wire signed [BT_BB1*RES_W*LANES-1:0] result_q_batch_w;
        wire        batch_bt_en  = batch_test_ctrl[0];
        wire        batch_bt_neg = batch_test_ctrl[1];
        wire [6:0]  batch_bt_lane = batch_test_ctrl[14:8];   // 0..127
        // per-16b-word negate of the b0 activation chunk (K_PAR words)
        wire signed [K_PAR*A_WIDTH-1:0] bt_pos = a_chunk_wide[K_PAR*A_WIDTH-1:0];
        wire signed [K_PAR*A_WIDTH-1:0] bt_neg;
        genvar btw;
        for (btw = 0; btw < K_PAR; btw = btw + 1) begin : g_bt_neg
            assign bt_neg[btw*A_WIDTH +: A_WIDTH] =
                   -$signed(a_chunk_wide[btw*A_WIDTH +: A_WIDTH]);
        end
        // sibling drive: BB1 copies of the (optionally negated) b0 chunk when enabled.
        wire signed [BT_BB1*K_PAR*A_WIDTH-1:0] a_chunk_batch_drv;
        genvar bts;
        for (bts = 0; bts < BT_BB1; bts = bts + 1) begin : g_bt_drv
            assign a_chunk_batch_drv[bts*K_PAR*A_WIDTH +: K_PAR*A_WIDTH] =
                   !batch_bt_en ? '0 : (batch_bt_neg ? bt_neg : bt_pos);
        end

        // scale-load mux: DMA scale port wins while it streams; the TB
        // direct-drive scale path is quiescent during a scale DMA.
        wire        eff_sc_we       = dma_sc_we | tb_sc_we;
        wire [15:0] eff_sc_lane_sel = dma_sc_we ? dma_sc_lane_sel : tb_sc_lane_sel;
        wire [16:0] eff_sc_wr_addr  = dma_sc_we ? dma_sc_wr_addr  : tb_sc_wr_addr;
        wire [15:0] eff_sc_wr_data  = dma_sc_we ? dma_sc_wr_data  : tb_sc_wr_data;

        // SLR-safe scaled array (SLRSAFE_SCALED_ARRAY_RESULT): drop-in for
        // gemv_stage_array_scaled (identical port list) + the 3 SLR-safe params.
        // SLR_SAFE=0 (default) = combinational passthrough = bit-AND-cycle identical
        // to the plain scaled array (the xsim functional config); SLR_SAFE=1 =
        // registered SLR-crossing seams for the 125 MHz build (bit-exact, +2 cyc).
        //
        // MAXPROBE-CRASH-FIX (2026-06-12): keep_hierarchy="yes" on the scaled-array
        // instance. The maxprobe probe additions (build-id/selftest/cap-crc liveness
        // window, 9-bit AXI-Lite addr) shifted elaboration enough that Vivado 2025.1's
        // logic-opt depth-reduction pass (HARTNlOptimizeFrameWork::_crushDepth ->
        // NDup::dupModule -> dupPorts -> NDbC::uniquePrefixes(UAggType*)) tried to
        // DUPLICATE this module and SIGSEGV'd while uniquifying its wide packed-array
        // ports (K_PAR*A_WIDTH a_chunk, RES_WIDTH*LANES result_q_per_lane, ...).
        // keep_hierarchy preserves the boundary so the crushDepth flatten-then-dup
        // pass never operates on it. ADDITIVE / non-functional: it only pins the
        // hierarchy boundary, the datapath is unchanged (frontier stays 8/8).
        (* keep_hierarchy = "yes" *)
        gemv_stage_array_scaled_slr #(
            .K(K_MAX), .K_PAR(K_PAR), .LANES(LANES), .A_WIDTH(A_WIDTH),
            .GROUP(128), .SCALE_WIDTH(16), .SCALE_FRAC(GEMV_SCALE_FRAC),
            .RES_WIDTH(RES_W),
            .SLR_SAFE(SLR_SAFE), .N_GROUPS(ARR_GROUPS), .PIPE_STAGES(ARR_PIPE_STAGES),
            // FF_ACT_CACHE: hub+per-group FF activation cache forward seam (lead-by-2),
            // the FUNDAMENTAL SLR fix for the body broadcast. 0 = legacy seam (identical).
            .FF_ACT_CACHE(FF_ACT_CACHE),
            // pdi16: column double-buffer prefetch (hides the per-column refill
            // bubble). Bit+cyc-INVARIANT to PREFETCH_COL=0 (task #233) — the
            // result chunk-index/done timing the seams + tree consume is unchanged.
            .PREFETCH_COL(PREFETCH_COL_P),
            // PDI#10 D1 fix: weight+enable +1cyc to match the RD_LAT=1 (registered)
            // activation seam reads below (PDI10_D1_REGREAD_CONFIRM). Paired with
            // RD_LAT(1) on every activation stage_io_buffer that feeds this array.
            .WLAT(D1_WLAT),
            // TREE-BATCH: sibling-batch width. 1 = byte-identical (batch ports inert).
            .BATCH_B(BODY_BATCH_B),
            // WO-025 Stage-B': TDP dual-read lane (2 chunks/cyc). 0 = single-port
            // (default, bit+cyc identical; a_chunk_b ignored, chunk_idx_b_out
            // mirrors even).
            .DUAL_PORT(DUAL_PORT)
        ) u_array (
            .clk(ap_clk), .rst_n(ap_rst_n), .start(array_start),
            .a_chunk(a_chunk_wide), .qz_raw(qz_eff),     // QZ-FIX: model const qz
            .a_chunk_b(a_chunk_b_mux),                   // WO-025: ODD-chunk feed (port B)
            .stage_sel(array_stage_sel), .tile_idx(array_tile_idx),
            .lane_we_sel(eff_boot_lane_sel), .we(eff_boot_we),
            .wr_addr(eff_boot_wr_addr), .wr_data(eff_boot_wr_data),
            .sc_lane_sel(eff_sc_lane_sel), .sc_we(eff_sc_we),
            .sc_wr_addr(eff_sc_wr_addr), .sc_wr_data(eff_sc_wr_data),
            .chunk_idx_out(array_chunk_idx_out),
            .chunk_idx_b_out(array_chunk_idx_b_out),     // WO-025: ODD-chunk index
            .chunk_idx_valid(array_chunk_idx_valid),
            .acc_out_per_lane(acc_dbg_per_lane),
            .result_q_per_lane(result_q_per_lane),
            // TREE-BATCH B=2: host-driven sibling activation + sibling result readback
            // (inert at batch_bt_en=0; b0 datapath above is unaffected => M7 exact).
            .a_chunk_batch(a_chunk_batch_drv),
            .result_q_batch(result_q_batch_w),
            .done(array_done)
        );

        // TREE-BATCH capture: latch the selected lane's b0 + sibling scaled acc on
        // each array `done` while batch-test is enabled -> host reads 0x13c..0x148.
        reg signed [RES_W-1:0] bt_b0_cap, bt_sib_cap;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                bt_b0_cap  <= '0;
                bt_sib_cap <= '0;
            end else if (array_done && batch_bt_en) begin
                bt_b0_cap  <= result_q_per_lane[batch_bt_lane*RES_W +: RES_W];
                bt_sib_cap <= result_q_batch_w[batch_bt_lane*RES_W +: RES_W];
            end
        end
        assign batch_b0_lo_w  = bt_b0_cap[31:0];
        assign batch_b0_hi_w  = {16'hB000, bt_b0_cap[47:32]};
        assign batch_sib_lo_w = bt_sib_cap[31:0];
        assign batch_sib_hi_w = {16'hB500, bt_sib_cap[47:32]};

        // Requant result_q (Q with SCALE_FRAC frac bits) -> INT32, saturating.
        // The dequant value out[j] = scaled_acc / 2^SCALE_FRAC; rounding via
        // add-half then arithmetic right shift. One assign per lane (MD-safe).
        genvar sl;
        for (sl = 0; sl < LANES; sl = sl + 1) begin : g_scaled_requant
            wire signed [RES_W-1:0] rq = result_q_per_lane[sl*RES_W +: RES_W];
            wire signed [RES_W-1:0] rq_round = rq + $signed({{(RES_W-GEMV_SCALE_FRAC){1'b0}},
                                                             1'b1, {(GEMV_SCALE_FRAC-1){1'b0}}});
            wire signed [RES_W-1:0] rq_shr = rq_round >>> GEMV_SCALE_FRAC;
            assign array_acc_int[sl*32 +: 32] =
                  (rq_shr >  $signed({{(RES_W-31){1'b0}}, 31'h7FFFFFFF})) ? 32'sh7FFFFFFF
                : (rq_shr < -$signed({{(RES_W-32){1'b0}}, 32'h80000000})) ? 32'sh80000000
                :                                                          rq_shr[31:0];
        end
    end else if (!GEMV_KPAR64) begin : g_array_k16
        // WO-025: no dual-read lane in the unscaled arm (DUAL_PORT elab-asserted
        // to require the scaled arm) — mirror the even index.
        assign array_chunk_idx_b_out = array_chunk_idx_out;
        gemv_stage_array_uram #(
            .K(K_MAX), .K_PAR(K_PAR), .LANES(LANES), .A_WIDTH(A_WIDTH)
        ) u_array (
            .clk(ap_clk), .rst_n(ap_rst_n), .start(array_start),
            .a_chunk(a_chunk_wide), .qz_raw(qz_eff),     // L8c: muxed input (QZ-FIX: model const qz)
            .stage_sel(array_stage_sel), .tile_idx(array_tile_idx),  // M6b: tiler override
            .lane_we_sel(eff_boot_lane_sel), .we(eff_boot_we),   // M1c: AXI or TB boot
            .wr_addr(eff_boot_wr_addr), .wr_data(eff_boot_wr_data),
            .chunk_idx_out(array_chunk_idx_out),
            .chunk_idx_valid(array_chunk_idx_valid),
            .acc_out_per_lane(array_acc_int),
            .done(array_done)
        );
    end else begin : g_array_k64
        // WO-025: no dual-read lane in the K_PAR=64 arm — mirror the even index.
        assign array_chunk_idx_b_out = array_chunk_idx_out;
        // #188 K_PAR=64 wide-bank array. Boot: gather 4 consecutive 32-bit
        // words (ARR_NSG=4) into one 128-bit row; the wide-row WE pulses on
        // the 4th word. Single-driver per per-lane wide bank (lane_we_sel).
        reg  [ARR_ROWW-1:0] boot_row_q;
        reg  [16:0]         boot_row_addr_q;
        reg                 boot_row_we_q;
        reg  [1:0]          boot_word_cnt;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                boot_row_q <= '0; boot_row_addr_q <= '0;
                boot_row_we_q <= 1'b0; boot_word_cnt <= 2'd0;
            end else begin
                boot_row_we_q <= 1'b0;
                if (eff_boot_we) begin
                    boot_row_q[boot_word_cnt*32 +: 32] <= eff_boot_wr_data;
                    if (boot_word_cnt == 2'd0) boot_row_addr_q <= eff_boot_wr_addr >> 2;
                    if (boot_word_cnt == ARR_NSG-1) begin
                        boot_word_cnt <= 2'd0;
                        boot_row_we_q <= 1'b1;       // emit the assembled wide row
                    end else begin
                        boot_word_cnt <= boot_word_cnt + 2'd1;
                    end
                end
            end
        end
        gemv_stage_array_uram_kpar64 #(
            .K(K_MAX), .K_PAR(ARR_KPAR), .LANES(LANES), .A_WIDTH(A_WIDTH)
        ) u_array (
            .clk(ap_clk), .rst_n(ap_rst_n), .start(array_start),
            .a_chunk(a_chunk_wide), .qz_raw(qz_eff),     // QZ-FIX: model const qz
            .stage_sel(array_stage_sel), .tile_idx(array_tile_idx),
            .lane_we_sel(eff_boot_lane_sel), .we(boot_row_we_q),
            .wr_row(boot_row_addr_q), .wr_row_data(boot_row_q),
            .chunk_idx_out(array_chunk_idx_out),
            .chunk_idx_valid(array_chunk_idx_valid),
            .acc_out_per_lane(array_acc_int),
            .done(array_done)
        );
    end
    endgenerate

    // Hold-flag per stage: latch the start-pulse so done can fire later
    // (g4_start_d forward-declared above for the M5 capture decode).
    //
    // S7b: under BACK_SEQ the back-half GEMV stages run MULTIPLE output tiles,
    // each ending in one array_done. The PER-TILE done (gX_done = array_done &
    // gX_start_d) is what the seam write-expanders (R2/R4/R7) consume — it MUST
    // fire on EVERY tile. So the hold-flag for a back-half stage stays asserted
    // for the WHOLE stage (cleared on the tiler's LAST-tile done bk_*_done_r),
    // NOT on the first array_done (which would drop tiles 1..N-1's writes).
    // The legacy single-tile pilot (BACK_SEQ=0) clears on the first array_done.
    reg g6_start_d, g7g_start_d, g7u_start_d, g9_start_d, g12_start_d;
    wire g6_hold_clear  = BACK_SEQ ? bk_g6_done_r  : array_done;
    wire g7g_hold_clear = BACK_SEQ ? bk_g7g_done_r : array_done;
    wire g7u_hold_clear = BACK_SEQ ? bk_g7u_done_r : array_done;
    wire g9_hold_clear  = BACK_SEQ ? bk_g9_done_r  : array_done;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            g4_start_d  <= 1'b0; g6_start_d  <= 1'b0;
            g7g_start_d <= 1'b0; g7u_start_d <= 1'b0;
            g9_start_d  <= 1'b0; g12_start_d <= 1'b0;
        end else begin
            // S7a (M4): in sequenced mode g4_start_d must stay high across ALL 72
            // tiles' array_done pulses (so the G4 capture gate fires per tile),
            // clearing only on the sequencer's LAST-tile done. Legacy pilot keeps
            // the original "clear on any array_done" single-tile behavior.
            if (g4_start)  g4_start_d  <= 1'b1;
            else if (S7A_FRONT_SEQ ? g4_seq_done_r : array_done) g4_start_d  <= 1'b0;
            if (g6_start)  g6_start_d  <= 1'b1; else if (g6_hold_clear)  g6_start_d  <= 1'b0;
            if (g7g_start) g7g_start_d <= 1'b1; else if (g7g_hold_clear) g7g_start_d <= 1'b0;
            if (g7u_start) g7u_start_d <= 1'b1; else if (g7u_hold_clear) g7u_start_d <= 1'b0;
            if (g9_start)  g9_start_d  <= 1'b1; else if (g9_hold_clear)  g9_start_d  <= 1'b0;
            if (g12_start) g12_start_d <= 1'b1; else if (array_done)     g12_start_d <= 1'b0;
        end
    end

    // PER-TILE done (= one array_done while the stage's hold-flag is set). These
    // drive the seam write-expanders (R2 on g6_done, R4g/R4u on g7g/g7u_done,
    // R7 b-feed on g9_done) — one clean pulse per output tile.
    // S7a (M4): G4 done fires ONLY on the sequencer's LAST tile (g4_seq_done_r),
    // so top_fsm stays in S_G4 until all 72 tiles are captured. Legacy pilot keeps
    // the per-tile done (single tile == stage).
    assign g4_done  = S7A_FRONT_SEQ ? g4_seq_done_r : (array_done & g4_start_d);
    assign g6_done  = array_done & g6_start_d;

    assign g7g_done = array_done & g7g_start_d;
    assign g7u_done = array_done & g7u_start_d;
    assign g9_done  = array_done & g9_start_d;
    assign g12_done = array_done & g12_start_d;

    // STAGE done fed to top_fsm: under BACK_SEQ the tiler asserts these on the
    // LAST tile only (so top_fsm leaves the stage after ALL tiles); the legacy
    // pilot uses the bare per-tile done (single tile == stage). G4/G12 keep the
    // per-tile done (G4 = S7a's job / front-half; G12 = tree tiler).
    wire g6_done_stage  = BACK_SEQ ? bk_g6_done_r  : g6_done;
    wire g7g_done_stage = BACK_SEQ ? bk_g7g_done_r : g7g_done;
    wire g7u_done_stage = BACK_SEQ ? bk_g7u_done_r : g7u_done;
    wire g9_done_stage  = BACK_SEQ ? bk_g9_done_r  : g9_done;

    // =====================================================================
    // PDI#5-ASSEMBLY — the data-connected forward (R1–R8 + residuals R3/R7).
    //
    // Pilot single-chunk model (matches the existing L8c G5->G6 island, which
    // captures ONE 16-element head-0 chunk and broadcasts it for the whole
    // consuming GEMV stage). Every seam below instantiates a stage_io_buffer
    // with H=K_PAR (one chunk slot, NCHUNKS=1): the producer writes the head-0
    // chunk once per invocation, the consumer reads slot 0 (broadcast for all
    // of the array's chunk requests). The two residuals use residual_add. All
    // are GATED by CONNECT_SEAMS at the consumer side (the a_chunk_mux above
    // and the silu/g11 input muxes below) so CONNECT_SEAMS=0 keeps the PDI#4
    // per-stage isolation EXACTLY (each buffer still captures, but no consumer
    // reads it — the external `a_chunk` / tied-0 arms are selected).
    //
    // Producer-out Q-format per seam (DATAPATH_PRIMS_RESULT §3.3 + MIXED_PREC):
    //   R1 (G1->G4)     : Q4.12 INT16 verbatim (rmsnorm_apply out) [NORMALIZED]
    //   R2 (G6->G7g/u)  : raw INT32 acc -> sat INT16 by producer (L8c), NO >>12
    //   R3 residual     : sat(x_emb + G6_sat) -> h1  [Q11.5 R3_FRAC=5 label, FIX-C]
    //   R4 (G7*->silu)  : raw INT32 acc -> sat INT16 (L8c)
    //   R5 (silu->G9)   : Q8.8 INT16 (silu_mul OUT_FRAC=8, g7p seam) [FIX-A; was Q4.12 clip]
    //   R6 (G9->G11)    : R7 residual out -> G11 RMSNorm x  [Q8.8]
    //   R7 residual     : to_logits pass-through -> G11 x  [Q8.8 R7_FRAC=8]
    //   R8 (G11->G12)   : g11hp RMSNorm out (Q6.10) -> requant <<2 -> Q4.12 [C2/C3]
    //
    // Multi-driver-safe: every buffer/register has exactly ONE always_ff
    // writer; the producer-capture write-strobes are mutually exclusive per
    // stage; reads are pure combinational. No shared writes.
    // =====================================================================
    localparam int SEAM_H = K_PAR;   // one-chunk broadcast pilot seam (NCHUNKS=1)

    // ---- saturate the 16 head-0 lanes of a 128-lane INT32 acc into one
    //      K_PAR*A_WIDTH chunk (the L8c producer-side INT32->INT16 sat). ----
    function automatic logic signed [K_PAR*A_WIDTH-1:0] sat_acc_head0_chunk;
        input signed [32*LANES-1:0] acc;
        logic signed [K_PAR*A_WIDTH-1:0] r;
        begin
            r = '0;
            for (int l = 0; l < K_PAR; l = l + 1)
                r[l*A_WIDTH +: A_WIDTH] = sat_int32_to_int16(acc[l*32 +: 32]);
            sat_acc_head0_chunk = r;
        end
    endfunction

    // -----------------------------------------------------------------
    // R1 — G1 RMSNorm -> G4 QKV activation.
    // Producer: rmsnorm_block g1. The valid strobe lags the data by one
    // cycle (M4 pipeline-skew note), so we capture rms_g1_out_chunk_d1 on the
    // first out_valid (the head-0 chunk). AUTO_SEQ would advance per valid;
    // with NCHUNKS=1 we just want slot 0 = the first chunk, so we gate the
    // write to the FIRST valid only (rms_g1_first_seen re-uses the M4 flag's
    // intent but with its own write-enable to stay single-writer).
    // -----------------------------------------------------------------
    // C1-REVERT (synthetic-vs-real correction, 2026-05-27): the R1 seam passes
    // G1's frac=12 output VERBATIM. The earlier `<<1` REQUANT (frac 12 -> 13)
    // was derived against a SYNTHETIC frac13 golden ("verbatim = 2x too small at
    // G4" was vs that WRONG golden). REAL eagle4_ref uses attn_in=12 (#190), so
    // verbatim pass-through is exactly right: G1-out frac12 -> seam frac12 ->
    // g4_qkv_seam shifts ACT_FRAC(12)-G5_FRAC(10) = 2 (>>2). The residfrac4 guard
    // (N=2200) showed attn_in=13 saturates real std~=0.93 activations (rank0
    // 91.77% vs attn_in=12's 93-95.6%). r1_lshift1_sat now an identity pass.
    function automatic logic signed [K_PAR*A_WIDTH-1:0] r1_lshift1_sat;
        input signed [K_PAR*A_WIDTH-1:0] ch;
        integer l;
        begin
            // C1-REVERT: verbatim (no `<<1`) — pass each lane through unchanged.
            for (l = 0; l < K_PAR; l = l + 1) begin
                r1_lshift1_sat[l*A_WIDTH +: A_WIDTH] = $signed(ch[l*A_WIDTH +: A_WIDTH]);
            end
        end
    endfunction

    // PRECISION_SAFE h-half alignment: the legacy h_norm engine emits frac12,
    // while the atomic mode's G4 input grid is frac11. Use the project's
    // round-half-before-arithmetic-shift idiom. The 17-bit temporary makes
    // +32767 + 1 safe. With PRECISION_SAFE=0 this is an identity.
    function automatic logic signed [K_PAR*A_WIDTH-1:0] r1_h_align;
        input signed [K_PAR*A_WIDTH-1:0] ch;
        logic signed [A_WIDTH:0] v;
        integer l;
        begin
            for (l = 0; l < K_PAR; l = l + 1) begin
                v = $signed(ch[l*A_WIDTH +: A_WIDTH]);
                r1_h_align[l*A_WIDTH +: A_WIDTH] =
                    PRECISION_SAFE ? (($signed(v) + 17'sd1) >>> 1)
                                   : v[A_WIDTH-1:0];
            end
        end
    endfunction

    // S7a/SEAM_R1_FULLW: the R1 seam "full" flag (`seam_r1_full`) is forward-
    // declared up at the G4 sequencer (the stage-entry input-full gate references
    // it); driven here.
    generate
    if (SEAM_R1_FULLW) begin : g_seam_r1_fullw
        // ============================================================
        // S7d DUAL-INPUT: FULL-K=6144 R1 seam = cat(e_norm, h_norm)
        // ============================================================
        // FULLK_DUALINPUT_ANALYSIS §4.2. G4's input is the 6144-wide concat
        // [e_norm | h_norm]. e_norm (NEW u_rmsnorm_g1_e) = LOW half, logical
        // chunks [0..191]; h_norm (existing u_rmsnorm_g1) = HIGH half, logical
        // chunks [192..383]. The shared 128-lane array walks a FIXED 512 chunks
        // per stage (gemv_lane_scaled_stream N_CHUNKS=512); the 6144-wide
        // (384-chunk, CW=9) buffer returns the real activation for chunks
        // [0..383] and ZERO for [384..511] (stage_io_buffer read guard
        // rd_chunk_idx<384), so the extra fixed-array chunks MAC weight*0 = 0
        // and G4 accumulates EXACTLY the K=6144 reduction. (No stage_nchunks
        // edit: this base has no runtime-K table; zero-pad is the K-truncation.)
        //
        // SINGLE-WRITER-SAFE: each half captures into its OWN stage_io_buffer
        // (one always_ff writer each) and the read is MUXED by chunk-index range
        // — NOT two always_ff into one RAM. Both norms co-launch on g1_start and
        // their apply strobes fire on the SAME cycles (identical FSM), so a single
        // shared write port WOULD collide; two buffers + a read mux avoids that
        // and is the canonical single-writer realization the analysis doc calls
        // for ("explicit address mux, NOT two always_ff into the same RAM").
        //
        // CAPTURE CONTRACT (S3, both halves): apply-phase request for logical
        // chunk j fires on apply_en; rmsnorm_apply's out_chunk lands AP_LAT=2
        // cycles later. Delay each apply_en by AP_LAT, write the LIVE out_chunk
        // (C1-REVERT: VERBATIM frac12, no `<<1`) into that half's AUTO_SEQ buffer. apply_en
        // stays high one cycle into S_DRAIN_A so the LAST chunk is captured.
        localparam int R1F_AP_LAT = 2;
        localparam int R1F_HALF_CW = (3072/K_PAR > 1) ? $clog2(3072/K_PAR) : 1; // 8
        localparam int R1_FULL_H   = 2*3072;                                    // 6144
        localparam int R1F_FULL_CW = (R1_FULL_H/K_PAR > 1)
                                     ? $clog2(R1_FULL_H/K_PAR) : 1;             // 9 (384 chunks)

        // ---- h_norm (HIGH) capture: existing u_rmsnorm_g1 -> chunks [192..383] ----
        reg [R1F_AP_LAT-1:0] r1f_h_apreq_sr;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)        r1f_h_apreq_sr <= '0;
            else if (g1_start)    r1f_h_apreq_sr <= '0;       // re-arm per G1 invocation
            else                  r1f_h_apreq_sr <= {r1f_h_apreq_sr[R1F_AP_LAT-2:0], g1_apply_en};
        end
        wire r1f_h_wr = r1f_h_apreq_sr[R1F_AP_LAT-1];
        // C1-REVERT: G1-out frac12 passed VERBATIM (no `<<1`); r1_lshift1_sat = identity.
        wire signed [K_PAR*A_WIDTH-1:0] r1f_h_wr_chunk = r1_h_align(rms_g1_out_chunk_w);

        // ---- e_norm (LOW) capture: NEW u_rmsnorm_g1_e -> chunks [0..191] ----
        reg [R1F_AP_LAT-1:0] r1f_e_apreq_sr;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)        r1f_e_apreq_sr <= '0;
            else if (g1_start)    r1f_e_apreq_sr <= '0;
            else                  r1f_e_apreq_sr <= {r1f_e_apreq_sr[R1F_AP_LAT-2:0], g1e_apply_en_eff};
        end
        wire r1f_e_wr = r1f_e_apreq_sr[R1F_AP_LAT-1];
        // TREE_FSM_DEPTHN_L1_ENORM_FIX: at level>=1 the e-half carries the 27/8
        // requant (u_rmsnorm_g1_e_d) instead of the saturating 31/10 output;
        // rms_g1e_out_chunk_eff == rms_g1e_out_chunk_w at level 0 / DEPTH_N=0.
        wire signed [K_PAR*A_WIDTH-1:0] r1f_e_wr_chunk = rms_g1e_out_chunk_eff;

        // ---- read index (0..383) presented to G4 by the live array chunk idx ----
        wire [R1F_FULL_CW-1:0] r1f_rd_idx = array_chunk_idx_out[R1F_FULL_CW-1:0];   // [8:0]
        wire                   r1f_rd_low = (r1f_rd_idx < R1F_FULL_CW'(3072/K_PAR)); // <192 = e half
        // HIGH-half local index = rd_idx - 192 (only meaningful when !r1f_rd_low);
        // intermediate wire so the [HALF_CW-1:0] select is on a net, not an expr.
        wire [R1F_FULL_CW-1:0] r1f_rd_idx_hi = r1f_rd_idx - R1F_FULL_CW'(3072/K_PAR);
        wire [R1F_HALF_CW-1:0] r1f_rd_half_idx =
            r1f_rd_low ? r1f_rd_idx[R1F_HALF_CW-1:0]
                       : r1f_rd_idx_hi[R1F_HALF_CW-1:0];

        // D1 FIX (PDI#10): the two half-buffers now read REGISTERED (RD_LAT=1, the
        // silicon BRAM truth, paired with array WLAT=1). The concat SELECT signals
        // (r1f_rd_low + the >=384 zero-pad guard) are derived from the CURRENT
        // array_chunk_idx_out, but the registered half-reads correspond to the
        // PREVIOUS address -> register the select decode by the SAME 1 cycle so the
        // mux picks the half that matches the registered data (no e/h boundary slip).
        reg r1f_rd_low_d1;
        reg r1f_rd_oor_d1;        // out-of-range (>=384) zero-pad guard, delayed
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                r1f_rd_low_d1 <= 1'b1; r1f_rd_oor_d1 <= 1'b0;
            end else begin
                r1f_rd_low_d1 <= r1f_rd_low;
                r1f_rd_oor_d1 <= (r1f_rd_idx >= R1F_FULL_CW'(R1_FULL_H/K_PAR));
            end
        end

        // ---- WO-025 port-B (ODD chunk) decode: the array pair-walks with
        // B = A+1 and A EVEN, and BOTH half boundaries (192 = e/h split,
        // 384 = OOR zero-pad) are EVEN — so A and B are ALWAYS in the same
        // half/OOR class and the port-A decode (r1f_rd_low, r1f_rd_low_d1,
        // r1f_rd_oor_d1, and the buffers' rd_valid half-select gates) is
        // shared VERBATIM. Only the half-local index differs (+1).
        wire [R1F_FULL_CW-1:0] r1f_rd_idx_b = array_chunk_idx_b_out[R1F_FULL_CW-1:0];
        wire [R1F_FULL_CW-1:0] r1f_rd_idx_b_hi = r1f_rd_idx_b - R1F_FULL_CW'(3072/K_PAR);
        wire [R1F_HALF_CW-1:0] r1f_rd_half_idx_b =
            r1f_rd_low ? r1f_rd_idx_b[R1F_HALF_CW-1:0]
                       : r1f_rd_idx_b_hi[R1F_HALF_CW-1:0];

        // e half buffer (LOW, chunks [0..191] of the concat)
        wire signed [K_PAR*A_WIDTH-1:0] r1_e_rd;
        wire signed [K_PAR*A_WIDTH-1:0] r1_e_rd_b;
        wire seam_r1_e_full;
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(1), .RD_LAT(D1_RDLAT)) u_seam_r1_e (  // D1: G4 activation seam (e half), registered
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(r1f_e_wr), .wr_chunk_idx('0), .wr_chunk(r1f_e_wr_chunk),
            .wr_clear(g1_start), .wr_count(), .full(seam_r1_e_full),
            // read only when the array indexes the LOW half (else don't care; muxed below)
            .rd_valid(r1f_rd_low), .rd_chunk_idx(r1f_rd_half_idx), .rd_chunk(r1_e_rd),
            .rd_chunk_idx_b(r1f_rd_half_idx_b), .rd_chunk_b(r1_e_rd_b)   // WO-025 port-B
        );
        // h half buffer (HIGH, chunks [192..383] of the concat)
        wire signed [K_PAR*A_WIDTH-1:0] r1_h_rd;
        wire signed [K_PAR*A_WIDTH-1:0] r1_h_rd_b;
        wire seam_r1_h_full;
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(1), .RD_LAT(D1_RDLAT)) u_seam_r1_h (  // D1: G4 activation seam (h half), registered
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(r1f_h_wr), .wr_chunk_idx('0), .wr_chunk(r1f_h_wr_chunk),
            .wr_clear(g1_start), .wr_count(), .full(seam_r1_h_full),
            .rd_valid(!r1f_rd_low && (r1f_rd_idx < R1F_FULL_CW'(R1_FULL_H/K_PAR))),
            .rd_chunk_idx(r1f_rd_half_idx), .rd_chunk(r1_h_rd),
            .rd_chunk_idx_b(r1f_rd_half_idx_b), .rd_chunk_b(r1_h_rd_b)   // WO-025 port-B
        );
        // concat read mux: chunks [0..191]=e, [192..383]=h, [>=384]=0 (zero-pad
        // for the fixed-512 array's K-truncation). Select keyed off the DELAYED
        // decode (r1f_rd_low_d1 / r1f_rd_oor_d1) to align with the registered reads.
        assign seam_r1_rd = r1f_rd_oor_d1 ? '0
                          : r1f_rd_low_d1 ? r1_e_rd : r1_h_rd;
        // WO-025 port-B concat mux: same delayed decode (same-half invariant).
        assign seam_r1_rd_b = r1f_rd_oor_d1 ? '0
                            : r1f_rd_low_d1 ? r1_e_rd_b : r1_h_rd_b;
        // input-full gate: BOTH halves' 192 chunks captured = all 384 in.
        assign seam_r1_full = seam_r1_e_full & seam_r1_h_full;
    end else begin : g_seam_r1_pilot
        // LEGACY single-chunk broadcast pilot (NCHUNKS=1). C1-REVERT: verbatim
        // frac12 (r1_lshift1_sat = identity) so the pilot also delivers frac=12 to G4.
        reg  r1_armed;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)                              r1_armed <= 1'b1;
            else if (g1_start)                          r1_armed <= 1'b1;
            else if (rms_g1_out_valid && r1_armed)      r1_armed <= 1'b0;
        end
        wire r1_wr = rms_g1_out_valid && r1_armed;   // combinational first-valid
        wire signed [K_PAR*A_WIDTH-1:0] r1p_wr_chunk = r1_h_align(rms_g1_out_chunk_d1);
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(SEAM_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(1)) u_seam_r1 (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(r1_wr), .wr_chunk_idx('0), .wr_chunk(r1p_wr_chunk),
            .wr_clear(g1_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx('0), .rd_chunk(seam_r1_rd)
        );
        assign seam_r1_full = 1'b1;   // pilot: gate disabled (single chunk always ready)
        // WO-025 port-B: pilot arm is a single-chunk BROADCAST (index-free) —
        // the odd-chunk value is the same chunk. (Pilot never runs DUAL_PORT=1;
        // the elab assert requires SEAM_R1_FULLW=1.)
        assign seam_r1_rd_b = seam_r1_rd;
    end
    endgenerate

    // -----------------------------------------------------------------
    // R2 — G6 o_proj -> G7g/G7u activation (one buffer feeds both).
    // *** S7b FULL WIDTH (H=3072, NCHUNKS=192) ***  was: single-chunk head-0
    // broadcast pilot (SEAM_H, AUTO_SEQ=1, rd_chunk_idx 0).
    //
    // Producer = the shared 128-lane GEMV array running G6 (o_proj, N=3072 =
    // 24 output tiles of 128 cols). Each per-tile array_done (g6_done) carries
    // 128 INT16-saturated accumulators = 8 chunks of K_PAR=16. 24 tiles × 8 = 192
    // chunks = 3072. The L8c raw-out->raw-in contract (NO >>12) is preserved —
    // sat_acc_subchunk just saturates INT32->INT16 (= what sat_acc_head0_chunk
    // did for head-0, now per sub-chunk). A per-tile 8-cycle write-expander
    // (mirrors the S5a R4 expander) fans one clean tile-done to 8 ordered chunk
    // writes at index tile*8 + s. The consumer (G7g/G7u) reads chunk-indexed by
    // the GEMV array's array_chunk_idx_out (R6/R5/R8 idiom).
    // -----------------------------------------------------------------
    localparam int G6_OUT_N   = 3072;
    localparam int G6_NTILES  = (G6_OUT_N + LANES - 1) / LANES;  // 24 @128 / 12 @256
    localparam int R2_NCHUNKS = G6_OUT_N / K_PAR;                // 192
    localparam int R2_CW      = $clog2(R2_NCHUNKS);              // 8
    // LANES-256: sub-chunk writes per array tile = LANES/K_PAR (8 @128,
    // 16 @256). Every per-tile write expander below (R2/R4g/R4u/G9/G9hi)
    // streams one LANES-wide acc out as LANE_SUBW chunks of K_PAR, so the
    // GLOBAL chunk index tile*LANE_SUBW+cnt is LANES-invariant (the seam
    // read side is untouched). The old literals (cnt==7, tile<<3, cnt[2:0])
    // were the 128-lane special case.
    localparam int LANE_SUBW    = LANES / K_PAR;
    localparam int LANE_SUBW_CW = $clog2(LANE_SUBW);             // 3 @128 / 4 @256
    // per-tile write expander for R2 (gate on g6_done)
    reg signed [32*LANES-1:0]  r2_acc_lat;
    reg [15:0]                 r2_tile_lat;
    reg [LANE_SUBW_CW:0]       r2_we_cnt;      // 0..LANE_SUBW-1 (8 @128 / 16 @256)
    reg                        r2_we_active;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            r2_acc_lat <= '0; r2_tile_lat <= '0; r2_we_cnt <= '0; r2_we_active <= 1'b0;
        end else if (g6_done) begin
            r2_acc_lat   <= array_acc_out;
            r2_tile_lat  <= array_tile_idx;
            r2_we_cnt    <= '0;
            r2_we_active <= 1'b1;
        end else if (r2_we_active) begin
            if (r2_we_cnt == (LANE_SUBW-1)) r2_we_active <= 1'b0;
            r2_we_cnt <= r2_we_cnt + 1'b1;
        end
    end
    wire                            r2_seam_we    = r2_we_active;
    wire [R2_CW-1:0]                r2_seam_widx  = R2_CW'(int'(r2_tile_lat)*LANE_SUBW + int'(r2_we_cnt));
    // G6 o_proj seam stored at Q11.5 (frac=5): the >>>5 round-half re-frac lands the
    // o_proj's +/-167 range (acc is frac=10, the g5 frac) with no clip AND matches
    // the R3 residual contract (a-arm h_fc is >>>7 to frac5). See the
    // sat_acc_subchunk_g6_r3 comment below. (Was: sat_acc_subchunk = frac10 -> 32x
    // over-scaled + INT16-clipped r3_h1 -> pre_mlp overflow to X.)
    wire signed [K_PAR*A_WIDTH-1:0] r2_seam_wchunk =
        sat_acc_subchunk_g6_r3(r2_acc_lat, int'(r2_we_cnt) * K_PAR);
    // SECOND COPY of the G6 o_proj output (full-H, 192 chunks) for the R3 residual
    // b-stream (the R2 stage_io_buffer has only one read port; the R3 driver
    // re-reads the G6 output here without contending for that port).
    reg signed [K_PAR*A_WIDTH-1:0] g6out_buf [0:R2_NCHUNKS-1];
    always_ff @(posedge ap_clk) begin
        if (r2_seam_we) g6out_buf[r2_seam_widx] <= r2_seam_wchunk;
    end
    // consumer (G7g/G7u) reads chunk-indexed; the G7 array walks K=3072=192 chunks
    wire [R2_CW-1:0] r2_rd_idx = array_chunk_idx_out[R2_CW-1:0];
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(G6_OUT_N), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(0)) u_seam_r2 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(r2_seam_we), .wr_chunk_idx(r2_seam_widx), .wr_chunk(r2_seam_wchunk),
        .wr_clear(g6_start), .wr_count(), .full(seam_r2_full),
        .rd_valid(1'b1), .rd_chunk_idx(r2_rd_idx), .rd_chunk(seam_r2_rd)
    );

    // -----------------------------------------------------------------
    // R4 — G7g gate, G7u up -> silu gate_chunk/up_chunk.  *** S5a FULL WIDTH ***
    //
    // The G7g/G7u producer is the shared 128-lane GEMV array. It emits the
    // 8192-element gate/up output as 64 TILES × 128 LANES: for tile_idx t,
    // lane i -> output column j = i + t*128 (gemv_stage_array_uram.sv:79-89).
    // Each tile ends in ONE clean array_done pulse (g7g_done / g7u_done)
    // carrying 128 INT16-saturated accumulators = 8 chunks of K_PAR=16.
    //
    // S5a streams the FULL gate/up output into two full-width stage_io_buffers
    // (H=8192, NCHUNKS=512, addressed). The per-tile done is a clean single
    // pulse, but ONE done must fan out to 8 chunk writes; a small per-tile
    // write-expander FSM (g7*_we_state) latches array_acc_out + tile_idx on the
    // done, then emits 8 chunk writes over the next 8 cycles at index
    // t*8 + s (s=0..7), chunk t*8+s = saturated lanes s*16 .. s*16+15.
    //
    // silu reads both seams chunk-indexed by r4_silu_rd_idx (the S7b silu en-FSM
    // read pointer) over all 512 chunks -> silu sees the FULL gate/up.
    // -----------------------------------------------------------------
    // saturate 16 selected lanes (lane base s*16) of the 128-lane INT32 acc
    // into one K_PAR*A_WIDTH chunk.
    function automatic logic signed [K_PAR*A_WIDTH-1:0] sat_acc_subchunk;
        input signed [32*LANES-1:0] acc;
        input int unsigned          lane_base;   // 0,16,...,112
        logic signed [K_PAR*A_WIDTH-1:0] r;
        begin
            r = '0;
            for (int l = 0; l < K_PAR; l = l + 1)
                r[l*A_WIDTH +: A_WIDTH] = sat_int32_to_int16(acc[(lane_base+l)*32 +: 32]);
            sat_acc_subchunk = r;
        end
    endfunction

    // SILU-Q612-FIX (2026-06-06): the R4 gate/up version of sat_acc_subchunk.
    // The G7g/G7u GEMV accumulator is Q?.12 (frac=12). The old sat_int32_to_int16
    // saturated each lane to INT16 = Q4.12 (+-8.0), which CLIPPED the real gate
    // (max 12.53) and up (max 17.98) -> 254/8192 saturated elems = 74.5% of the
    // silu error (max|d|=8346 @idx85). This 18-bit Q6.12 saturate holds +-32.0
    // (well above up's 18.0) so ZERO elems clip. Output is R4_AW(18)-bit/lane.
    localparam int R4_SAT_AW = 18;
    function automatic logic signed [K_PAR*R4_SAT_AW-1:0] sat_acc_subchunk_q612;
        input signed [32*LANES-1:0] acc;
        input int unsigned          lane_base;   // 0,16,...,112
        logic signed [K_PAR*R4_SAT_AW-1:0] r;
        logic signed [31:0]                v;
        begin
            r = '0;
            for (int l = 0; l < K_PAR; l = l + 1) begin
                v = $signed(acc[(lane_base+l)*32 +: 32]);
                if      (v >  32'sd131071) r[l*R4_SAT_AW +: R4_SAT_AW] =  18'sd131071;  // +31.9998
                else if (v < -32'sd131072) r[l*R4_SAT_AW +: R4_SAT_AW] = -18'sd131072;  // -32.0
                else                       r[l*R4_SAT_AW +: R4_SAT_AW] =  v[17:0];
            end
            sat_acc_subchunk_q612 = r;
        end
    endfunction

    // G6 o_proj seam re-frac: the o_proj GEMV consumes the g5 attention output
    // (Q6.10, frac=10 — capture_goldens.py:28) and the o_proj acc inherits that
    // frac (g6_o_proj golden = "real -> g5 frac dump", capture_goldens.py:29), so
    // the array_acc_out for G6 is at frac=10, NOT frac=12. The ONLY live consumer
    // (BACK_SEQ=1) is the R3 residual b-arm, which adds it against the a-arm h_fc
    // that is already shifted >>>7 to frac=5 (Q11.5, r3_h1 contract,
    // capture_goldens.py:30). Storing the seam at frac10 made the g6 term enter the
    // add 2^(10-5)=32x too large AND INT16-sat-clipped g6's real +/-167 range
    // (167*1024 = 171008 >> 32767 -> real 8.0). This variant arithmetic-right-shifts
    // each INT32 lane >>>5 (frac10 -> frac5, round-half) to land Q11.5 (+/-1024
    // range -> holds +/-167 with NO clip) BEFORE the INT16 sat. Matches the r3_h1
    // golden contract. (sat_acc_subchunk above is unchanged for the G7g/G7u gate/up
    // paths.)  [REBISECT_QZ1 said acc=frac12->>>>7; the goldens show acc=frac10->>>>5.]
    localparam int G6_TO_R3_SH = 10 - R3_FRAC;
    function automatic logic signed [K_PAR*A_WIDTH-1:0] sat_acc_subchunk_g6_r3;
        input signed [32*LANES-1:0] acc;
        input int unsigned          lane_base;   // 0,16,...,112
        logic signed [K_PAR*A_WIDTH-1:0] r;
        logic signed [31:0]              v_rnd;
        begin
            r = '0;
            for (int l = 0; l < K_PAR; l = l + 1) begin
                // Default: frac10 -> frac5 (shift 5). PRECISION_SAFE:
                // frac10 -> frac7 (shift 3). Both round-half then sat16.
                v_rnd = $signed(acc[(lane_base+l)*32 +: 32])
                      + (32'sd1 <<< (G6_TO_R3_SH-1));
                r[l*A_WIDTH +: A_WIDTH] =
                    sat_int32_to_int16(v_rnd >>> G6_TO_R3_SH);
            end
            sat_acc_subchunk_g6_r3 = r;
        end
    endfunction


    // ---- per-tile write expander: gate seam (R4g) ----
    // Latched on g7g_done: the 128-lane acc + the tile_idx that produced it.
    reg signed [32*LANES-1:0]  g7g_acc_lat;
    reg [15:0]                 g7g_tile_lat;
    reg [LANE_SUBW_CW:0]       g7g_we_cnt;       // 0..LANE_SUBW-1 sub-chunk writes
    reg                        g7g_we_active;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            g7g_acc_lat <= '0; g7g_tile_lat <= '0; g7g_we_cnt <= '0; g7g_we_active <= 1'b0;
        end else if (g7g_done) begin
            // latch this tile's full LANES-wide output + which tile it is
            g7g_acc_lat   <= array_acc_out;
            g7g_tile_lat  <= array_tile_idx;
            g7g_we_cnt    <= '0;
            g7g_we_active <= 1'b1;
        end else if (g7g_we_active) begin
            if (g7g_we_cnt == (LANE_SUBW-1)) g7g_we_active <= 1'b0;
            g7g_we_cnt <= g7g_we_cnt + 1'b1;
        end
    end
    wire                        g7g_seam_we   = g7g_we_active;
    // chunk index = tile*LANE_SUBW + sub (@128: 64 tiles x 8; @256: 32 x 16 -> 0..511 either way)
    wire [G7_CW-1:0]            g7g_seam_widx = G7_CW'(int'(g7g_tile_lat)*LANE_SUBW + int'(g7g_we_cnt));
    // SILU-Q612-FIX: emit Q6.12 18-bit (no +-8 clip) into the widened R4g seam.
    wire signed [K_PAR*R4_AW-1:0] g7g_seam_wchunk = sat_acc_subchunk_q612(g7g_acc_lat, int'(g7g_we_cnt) * K_PAR);

    // ---- per-tile write expander: up seam (R4u) ----
    reg signed [32*LANES-1:0]  g7u_acc_lat;
    reg [15:0]                 g7u_tile_lat;
    reg [LANE_SUBW_CW:0]       g7u_we_cnt;
    reg                        g7u_we_active;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            g7u_acc_lat <= '0; g7u_tile_lat <= '0; g7u_we_cnt <= '0; g7u_we_active <= 1'b0;
        end else if (g7u_done) begin
            g7u_acc_lat   <= array_acc_out;
            g7u_tile_lat  <= array_tile_idx;
            g7u_we_cnt    <= '0;
            g7u_we_active <= 1'b1;
        end else if (g7u_we_active) begin
            if (g7u_we_cnt == (LANE_SUBW-1)) g7u_we_active <= 1'b0;
            g7u_we_cnt <= g7u_we_cnt + 1'b1;
        end
    end
    wire                        g7u_seam_we   = g7u_we_active;
    wire [G7_CW-1:0]            g7u_seam_widx = G7_CW'(int'(g7u_tile_lat)*LANE_SUBW + int'(g7u_we_cnt));
    // SILU-Q612-FIX: emit Q6.12 18-bit (no +-8 clip) into the widened R4u seam.
    wire signed [K_PAR*R4_AW-1:0] g7u_seam_wchunk = sat_acc_subchunk_q612(g7u_acc_lat, int'(g7u_we_cnt) * K_PAR);

    // ---- silu-feed read pointer + 512-pulse en — driven by the S7b silu en-FSM
    //      (declared/sequenced in the BACK-HALF SEQUENCER block below). The seam
    //      reads are chunk-indexed by r4_silu_rd_idx = the silu en-FSM pointer.
    // (seam_r4g_rd_c / seam_r4u_rd_c are declared at the silu instance and
    //  driven here — they feed the silu gate/up connected-arm mux.)
    // SILU-Q612-FIX: R4 gate/up seams carry Q6.12 (A_WIDTH=R4_AW=18, FRAC_BITS=12).
    // FIX-G7p (2026-06-13): .RD_LAT(D1_RDLAT) — these R4 silu-feed seams were the
    // ONLY active datapath stage_io_buffers left at the default RD_LAT=0. Every
    // GEMV-array activation seam (R1 e/h, G5->G6, R5, R8, G12) already carries
    // .RD_LAT(D1_RDLAT); R4 was missed because its consumer is silu_mul, not the
    // GEMV array. On silicon the ram_style="block" read is synchronous (T+1) but
    // silu samples gate/up combinationally on the en-high cycle -> SiLU computed
    // chunk[c-1] (G7p stage-cap divergence). With D1_RDLAT=1 the seam is registered
    // and r4_silu_rd_idx is led +1 (= silu_en_cnt, above) so chunk[c] lands on the
    // en-high-for-c cycle. D1_RDLAT=0 stays the legacy comb read (byte-identical).
    // R4_BANKSPLIT gate (2026-06-15, G7p PHYSICAL fix B). DEFAULT 0 = the wide
    // stage_io_buffer = BYTE-IDENTICAL to g7pkeep. =1 swaps in the bank-split
    // (8x36b non-cascaded single-RAMB banks) addressed by the SAME r4_silu_rd_idx
    // and the SAME write expander (g7*_seam_we/_widx/_wchunk) -> chunk-level
    // bit-identical; only the on-chip storage geometry changes (4xRAMB36 cascade
    // -> 8 independent RAMB18-pair tiles). Both arms keep RD_LAT(D1_RDLAT) so the
    // +1 silu_en_cnt read-lead alignment is preserved. Instance names u_seam_r4g/
    // u_seam_r4u are kept inside both generate arms so the pblock add_cells glob
    // (*u_seam_r4g*/*u_seam_r4u*) matches either geometry.
    generate
    if (R4_BANKSPLIT) begin : g_r4_banked
        stage_io_buffer_banked #(.H(G7_H_INTER), .K_PAR(K_PAR), .A_WIDTH(R4_AW), .LANES_PER_BANK(2), .RD_LAT(D1_RDLAT)) u_seam_r4g (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(g7g_seam_we), .wr_chunk_idx(g7g_seam_widx), .wr_chunk(g7g_seam_wchunk),
            .wr_clear(g7g_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx(r4_silu_rd_idx), .rd_chunk(seam_r4g_rd_c)
        );
        stage_io_buffer_banked #(.H(G7_H_INTER), .K_PAR(K_PAR), .A_WIDTH(R4_AW), .LANES_PER_BANK(2), .RD_LAT(D1_RDLAT)) u_seam_r4u (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(g7u_seam_we), .wr_chunk_idx(g7u_seam_widx), .wr_chunk(g7u_seam_wchunk),
            .wr_clear(g7u_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx(r4_silu_rd_idx), .rd_chunk(seam_r4u_rd_c)
        );
    end else begin : g_r4_wide
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(G7_H_INTER), .K_PAR(K_PAR), .A_WIDTH(R4_AW), .AUTO_SEQ(0), .FRAC_BITS(12), .RD_LAT(D1_RDLAT)) u_seam_r4g (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(g7g_seam_we), .wr_chunk_idx(g7g_seam_widx), .wr_chunk(g7g_seam_wchunk),
            .wr_clear(g7g_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx(r4_silu_rd_idx), .rd_chunk(seam_r4g_rd_c)
        );
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(G7_H_INTER), .K_PAR(K_PAR), .A_WIDTH(R4_AW), .AUTO_SEQ(0), .FRAC_BITS(12), .RD_LAT(D1_RDLAT)) u_seam_r4u (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(g7u_seam_we), .wr_chunk_idx(g7u_seam_widx), .wr_chunk(g7u_seam_wchunk),
            .wr_clear(g7u_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx(r4_silu_rd_idx), .rd_chunk(seam_r4u_rd_c)
        );
    end
    endgenerate

    // -----------------------------------------------------------------
    // R5 — silu -> G9 down activation.  S5B FULL-WIDTH REWRITE
    // (was: single-chunk head-0 broadcast pilot, captured the FIRST
    //  silu_valid chunk and broadcast it to every G9 chunk).
    //
    // The silu_layer STREAMS the full I_DIM=8192-element MLP intermediate,
    // one K_PAR=16-wide chunk per cycle when its `en` is held (the S7 FSM
    // drives `en` for I_DIM/K_PAR=512 cycles). silu_valid IS a clean
    // 512-pulse strobe: `valid <= en_pipe; en_pipe <= en`, so one valid
    // pulse per `en` cycle, in order, 2-cycle latency, aligned to
    // silu_out_chunk — NO strobe-tap needed (unlike rmsnorm out_valid).
    //
    // The full silu output streams into a full-H stage_io_buffer
    // (R5_H=8192, R5_NCHUNKS=512). AUTO_SEQ=1: each silu_valid auto-
    // increments the slot (0..511), cleared at g7p_start. G9 reads it
    // CHUNK-INDEXED by the GEMV array's array_chunk_idx_out (the array
    // walks K_MAX=8192=512 chunks; it consumes a_chunk in lockstep with
    // its current chunk_idx — see gemv_lane_scaled_stream a_lane/in_valid),
    // so G9's down_proj sees the FULL silu output.
    //
    // Q-format: FIX-A — R5 now carries the silu output at Q8.8 (g7p seam frac=8),
    // matching the composite-proven golden (g7_silu frac=8, sat=0). Was FRAC_BITS=12
    // (Q4.12) which clipped the ±91 silu*up product at ±8.0. The buffer is
    // format-agnostic; FRAC_BITS=8 just labels it (NORMALIZED seam).
    // -----------------------------------------------------------------
    localparam int R5_H       = 8192;                 // I_DIM (MLP intermediate)
    localparam int R5_NCHUNKS = R5_H / K_PAR;         // 512
    localparam int R5_CW      = $clog2(R5_NCHUNKS);   // 9
    wire r5_wr = silu_valid;                          // clean per-chunk strobe
    // G9 reads chunk-indexed by the array's current chunk request.
    wire [R5_CW-1:0] r5_rd_idx = array_chunk_idx_out[R5_CW-1:0];
    // WO-025 port-B: parallel ODD-chunk read pointer (G9 pair-walk).
    wire [R5_CW-1:0] r5_rd_idx_b = array_chunk_idx_b_out[R5_CW-1:0];
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(R5_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                      .AUTO_SEQ(1), .FRAC_BITS(8), .RD_LAT(D1_RDLAT)) u_seam_r5 (  // D1: G9 activation seam, registered
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(r5_wr), .wr_chunk_idx('0), .wr_chunk(silu_out_chunk),
        .wr_clear(g7p_start), .wr_count(), .full(seam_r5_full),
        .rd_valid(1'b1), .rd_chunk_idx(r5_rd_idx), .rd_chunk(seam_r5_rd),
        .rd_chunk_idx_b(r5_rd_idx_b), .rd_chunk_b(seam_r5_rd_b)   // WO-025 port-B
    );

    localparam int R6_NCHUNKS  = 3072 / K_PAR;          // 192
    localparam int R6_CW       = (R6_NCHUNKS > 1) ? $clog2(R6_NCHUNKS) : 1;  // 8

    generate
    if (BACK_SEQ) begin : g_resid_fullh
        // =============================================================
        // S7b — FULL-WIDTH residuals R3 (h1 = x + G6 o_proj) and R7
        // (g11_in = h1 + G9 down). Both stream all 192 chunks (H=3072).
        //
        // The two GEMV branch outputs are captured FULL-H from the shared-array
        // per-tile dones (the same write-expander idiom as R2/R4): G6 o_proj is
        // ALREADY held full-width in the R2 seam buffer; G9 down is captured here
        // into g9down_buf[192]. The layer input x for R3 is input_emb_buf[192]
        // (full-H in the connected forward / gate preload). residual_add runs in
        // STREAM-SAVE mode (load a 0..191, start, stream b 0..191 -> out 0..191).
        //
        // R3: a = x (input_emb_buf), b = G6 o_proj (r2 chunk stream) -> h1_buf[192]
        // R7: a = h1_buf,            b = G9 down (g9down_buf)         -> R6 seam
        // =============================================================
        // ---- G9 down full-H capture (per-tile write-expander, 24 tiles x 8) ----
        reg signed [32*LANES-1:0]  g9_acc_lat;
        reg [15:0]                 g9_tile_lat;
        reg [LANE_SUBW_CW:0]       g9_we_cnt;
        reg                        g9_we_active;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                g9_acc_lat <= '0; g9_tile_lat <= '0; g9_we_cnt <= '0; g9_we_active <= 1'b0;
            end else if (g9_done) begin
                g9_acc_lat   <= array_acc_out;
                g9_tile_lat  <= array_tile_idx;
                g9_we_cnt    <= '0;
                g9_we_active <= 1'b1;
            end else if (g9_we_active) begin
                if (g9_we_cnt == (LANE_SUBW-1)) g9_we_active <= 1'b0;
                g9_we_cnt <= g9_we_cnt + 1'b1;
            end
        end
        wire [R6_CW-1:0]                g9_widx = R6_CW'(int'(g9_tile_lat)*LANE_SUBW + int'(g9_we_cnt));
        wire signed [K_PAR*A_WIDTH-1:0] g9_wchunk =
            sat_acc_subchunk(g9_acc_lat, int'(g9_we_cnt) * K_PAR);
        // F1 (FORREASON): with G9 widened to N=2H=6144 the SAME per-tile write
        // expander now sees 48 tiles. Tiles 0..23 are to_logits (this buffer,
        // untouched semantics); tiles 24..47 are for_reasoning and are split off
        // below. At FORREASON=0 g9_hi_sel is a CONSTANT 0 -> the AND folds away
        // and this write is byte-identical to pre-F1.
        // LANES-256: the to_logits/for_reasoning tile split is at H/LANES
        // tiles (24 @128, 12 @256) — the OLD literal 24 was the 128 case.
        localparam int G9_LO_NTILES = 3072 / LANES;
        wire g9_hi_sel = FORREASON ? (g9_tile_lat >= 16'(G9_LO_NTILES)) : 1'b0;
        reg signed [K_PAR*A_WIDTH-1:0] g9down_buf [0:R6_NCHUNKS-1];
        always_ff @(posedge ap_clk) begin
            if (g9_we_active && !g9_hi_sel) g9down_buf[g9_widx] <= g9_wchunk;
        end


        // ---- R3 driver: a = x (input_emb_buf 0..191), b = G6 o_proj (R2) ----
        // After G6 finishes (bk_g6_done_r), stream a 0..191 (a_valid), then start,
        // then b 0..191 from the R2 seam buffer (re-read chunk-indexed).
        reg [R6_CW:0] r3_a_cnt;  reg r3_a_run;
        reg [R6_CW:0] r3_b_cnt;  reg r3_b_run;  reg r3_started;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                r3_a_cnt <= '0; r3_a_run <= 1'b0;
                r3_b_cnt <= '0; r3_b_run <= 1'b0; r3_started <= 1'b0;
            end else begin
                if (bk_g6_done_r) begin
                    r3_a_cnt <= '0; r3_a_run <= 1'b1;
                    r3_b_cnt <= '0; r3_b_run <= 1'b0; r3_started <= 1'b0;
                end
                // phase 1: load a (x) 0..191
                if (r3_a_run) begin
                    if (r3_a_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) begin
                        r3_a_run <= 1'b0;     // last a this cycle
                        r3_started <= 1'b1;   // pulse start next cycle
                    end
                    r3_a_cnt <= r3_a_cnt + 1'b1;
                end
                // phase 2: after start, stream b (G6 o_proj) 0..191
                if (r3_started && !r3_b_run) begin
                    r3_b_run <= 1'b1; r3_b_cnt <= '0; r3_started <= 1'b0;
                end else if (r3_b_run) begin
                    if (r3_b_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) r3_b_run <= 1'b0;
                    r3_b_cnt <= r3_b_cnt + 1'b1;
                end
            end
        end
        wire [R6_CW-1:0] r3_a_idx_raw  = r3_a_cnt[R6_CW-1:0];
        // D2 FIX (PDI#10, PDI10_D1_REGREAD_CONFIRM): the R3 residual a-arm reads
        // h_fc_buf (forced BRAM) COMBINATIONALLY and writes a_save_buf. On silicon
        // the BRAM read is 1-cycle registered, so a_save_buf[i] got h_fc[i-1]
        // (R3 = h_fc[k-1] + g6[k]). The COMBREAD fix registered the b-arm + the
        // a_save_buf READ (RDATA_LAT=1) but NOT this a-arm WRITE-side read. Register
        // the h_fc_buf read AND delay a_valid/a_idx (and the start pulse) by the
        // SAME 1 cycle (symmetric to the b-arm) so a_save_buf[i] <= h_fc[i].
        // RD is ALWAYS registered here -> sim==silicon. Confirmed by tb_d2_regread.
        reg signed [K_PAR*A_WIDTH-1:0] r3_a_hfc_raw_q;
        reg [R6_CW-1:0]                r3_a_idx_d1;
        reg                            r3_a_run_d1;
        reg                            r3_started_d1;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                r3_a_hfc_raw_q <= '0; r3_a_idx_d1 <= '0;
                r3_a_run_d1 <= 1'b0; r3_started_d1 <= 1'b0;
            end else begin
                r3_a_hfc_raw_q <= h_fc_buf_rd(r3_a_idx_raw);  // BUFSPLIT read
                r3_a_idx_d1    <= r3_a_idx_raw;
                r3_a_run_d1    <= r3_a_run;
                r3_started_d1  <= r3_started;
            end
        end
        wire             r3_a_wr_f     = r3_a_run_d1;          // delayed a_valid
        wire [R6_CW-1:0] r3_a_idx_f    = r3_a_idx_d1;          // delayed a_chunk_idx (matches registered read)
        // FIX-E R2: the R3 residual a-arm is h_fc (NOT the embedding). h_fc_buf is
        // loaded ONCE at frac=12 (fc_h.hex). The R3 seam golden fc_h_r3.hex is the
        // SAME vector at frac=5 (= fc_h >>> 7). Apply a per-lane ARITHMETIC >>>7
        // (no round, per spec: fc_h_r3 = fc_h>>>7) to land Q11.5 (R3_FRAC=5) before
        // residual_add. residual_add is frac-agnostic on the add, but the b-arm
        // (g6 o_proj) is the producer-set Q11.5, so a must match Q11.5 magnitude.
        wire signed [K_PAR*A_WIDTH-1:0] r3_a_hfc_raw = r3_a_hfc_raw_q;   // D2: registered read
        // ---- F2 (DEPTH-MUX) consumer #2: the R3 residual a-arm ----------
        // Level 0 keeps the h_fc arm above BYTE-IDENTICALLY (the >>>F2_SH_HFC
        // below is the pre-F2 literal `>>> 7`, now derived). Level >= 1 reads
        // the producer's `a` instead — the child block input a_{d-1}, the SAME
        // vector the G1 h_norm consumer reads (goldens: r3_a_arm.mem ==
        // a_parent_in.mem, verified byte-identical over all 9 calls).
        //
        // FRAC BOOKKEEPING — derived from the named format localparams, never
        // hand-copied shift literals:
        //   level 0   : h_fc_buf grid F2_HFC_FRAC=12 (fc_h.hex, FIX-E R2)
        //               -> shift = F2_HFC_FRAC - R3_FRAC = 7
        //   level >= 1: producer grid FR_A_FRAC=9 (F1 §3, max|a|=21.9)
        //               -> shift = FR_A_FRAC - R3_FRAC = 4
        // The 4-bit RIGHT shift is a RE-ALIGN onto the mandated Q11.5 seam
        // grid (the b-arm G6 o_proj is Q11.5): a no-round truncation identical
        // in kind to the level-0 `>>>7` idiom, so F2 introduces ZERO error
        // beyond the pre-existing R3 seam grid (F1 §3: a's accuracy is bounded
        // by the R3 seam's Q11.5, not by anything the producer/mux adds). No
        // saturation is possible on a right shift. It is NOT the old ">>>7 on
        // a" prescription — that would additionally divide `a` by 8, burning
        // the ~3 re-grade bits TWICE (banned by F1 §3.2).
        // F1 producer formats — hoisted from g_forreason (F2) so the a-arm
        // shift is derived from the SAME single-source localparams the
        // producer instance uses. Values unchanged from F1.
        localparam int FR_H1_FRAC = R3_FRAC;   // 5, the R3 seam's own frac
        localparam int FR_G9_FRAC = 8;         // G9 accumulator native
        localparam int FR_A_FRAC  = 9;         // max non-saturating (max|a|=21.9 -> 11,226/32,767)
        localparam int F2_HFC_FRAC = 12;                     // h_fc_buf load grid (fc_h.hex)
        localparam int F2_SH_HFC   = F2_HFC_FRAC - R3_FRAC;  // = 7 (pre-F2 literal, unchanged)
        localparam int F2_SH_FR    = FR_A_FRAC   - R3_FRAC;  // = 4 (Q6.9 -> Q11.5 re-align)
`ifdef F2_RED_HFC
        // RED ARM (the F2 gate MUST FAIL): a-arm pinned to the OLD h_fc path.
        wire f2_r3_use_fr = 1'b0;
`else
        wire f2_r3_use_fr = f2_lvl_ge1_w;
`endif
        wire signed [K_PAR*A_WIDTH-1:0] r3_a_chunk_f;
        genvar r3al;
        for (r3al = 0; r3al < K_PAR; r3al = r3al + 1) begin : g_r3a_to_f5
            assign r3_a_chunk_f[r3al*A_WIDTH +: A_WIDTH] = f2_r3_use_fr
                ? ($signed(f2_fr_rd_data_w[r3al*A_WIDTH +: A_WIDTH]) >>> F2_SH_FR)   // level>=1: Q6.9  -> Q11.5
                : ($signed(r3_a_hfc_raw   [r3al*A_WIDTH +: A_WIDTH]) >>> F2_SH_HFC); // level 0 : frac12 -> frac5
        end
        // =================================================================
        // CHILD_SEAM_FIX (ACCFIX bit0, 0x1125 H1) — a-arm shift-REMAINDER
        // capture. The two `>>>` truncations above floor the residual arm to
        // frac5 before residual_add; for_reasoning_producer then re-left-
        // shifts the sum by 4 — so the carried block-input loses its low 4
        // bits (level>=1) / lands as a 7-bit floor of h_fc (level 0) EVERY
        // level: the compounding child-quality gradient (PHASE5 H1). Capture
        // the non-negative remainder (x - (x>>>n)<<n, always in [0, 2^n-1])
        // at the A_FRAC=9 grid, co-timed with the a-arm write, and re-add it
        // in the producer's wide pre-sat sum:
        //   level>=1: rem9 = pa[3:0]              -> h1<<4 + rem == pa + g6<<4  (EXACT)
        //   level 0 : rem9 = (h_fc[6:0] + 4) >> 3 -> round-half(h_fc -> frac9)
        // 192-chunk x 16-lane x 5b FF buffer (15,360 FDRE), generate-pruned
        // when ACCFIX_SEAM_P=0. Runtime-off leaves the buffer written but
        // UNCONSUMED (producer lane_rem folds to 0) = byte-identical.
        // =================================================================
        localparam int FR_REM_W = 5;
        wire [K_PAR*FR_REM_W-1:0] fr_rem_now;
        for (r3al = 0; r3al < K_PAR; r3al = r3al + 1) begin : g_r3a_rem
            wire [3:0] rem_fr  = f2_fr_rd_data_w[r3al*A_WIDTH +: 4];        // pa[3:0] @frac9
            wire [6:0] rem_hfc = r3_a_hfc_raw   [r3al*A_WIDTH +: 7];        // h_fc[6:0] @frac12
            // 8-bit add: 7-bit (rem_hfc + 4) would WRAP for rem_hfc in
            // [124,127] (127+4 = 3 mod 128 -> rem9 0 instead of 16).
            wire [4:0] rem_hfc9 = ({1'b0, rem_hfc} + 8'd4) >> 3;            // round-half -> frac9, 0..16
            assign fr_rem_now[r3al*FR_REM_W +: FR_REM_W] =
                f2_r3_use_fr ? {1'b0, rem_fr} : rem_hfc9;
        end
        wire             r3_start_f    = r3_started_d1;        // D2: start delayed +1 to follow the now-1-cyc-later last a-write
        wire [R6_CW-1:0] r3_b_idx_f    = r3_b_cnt[R6_CW-1:0];
        // COMBREAD FIX (2026-06-08): g6out_buf is BRAM (registered 1-cycle read on
        // silicon). Register the b-arm read AND delay b_valid by the same 1 cycle
        // so b_chunk[k] arrives coincident with b_valid[k] (the contract
        // u_resid_r3's RDATA_LAT=1 path expects). Without this the b operand would
        // be stale-by-1 (a[k]+b[k-1]). sim==silicon (both registered).
        reg signed [K_PAR*A_WIDTH-1:0] r3_b_chunk_f;
        reg                            r3_b_valid_f;
        always_ff @(posedge ap_clk) begin
            r3_b_chunk_f <= g6out_buf[r3_b_idx_f];
            r3_b_valid_f <= r3_b_run;
        end

        wire                            r3_out_valid_f;
        wire [R6_CW-1:0]                r3_out_idx_f;
        wire signed [K_PAR*A_WIDTH-1:0] r3_out_chunk_f;
        wire                            r3_done_f;
        residual_add #(.H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                       .FRAC_BITS(R3_FRAC), .RDATA_LAT(1)) u_resid_r3 (   // FIX-C + COMBREAD FIX: registered a_save_buf read
            .clk(ap_clk), .rst_n(ap_rst_n),
            .a_valid(r3_a_wr_f), .a_chunk_idx(r3_a_idx_f), .a_chunk(r3_a_chunk_f),
            .start(r3_start_f), .b_chunk(r3_b_chunk_f), .b_valid(r3_b_valid_f),
            .out_valid(r3_out_valid_f), .out_chunk_idx(r3_out_idx_f), .out_chunk(r3_out_chunk_f),
            .done(r3_done_f)
        );
        // h1 full-H buffer
        reg signed [K_PAR*A_WIDTH-1:0] h1_buf_f [0:R6_NCHUNKS-1];
        always_ff @(posedge ap_clk) begin
            if (r3_out_valid_f) h1_buf_f[r3_out_idx_f] <= r3_out_chunk_f;
        end
        // PROBE V3: export the R3 residual_add output (post-attn residual) to the
        // top-scope dbg mirror for dbg_probe_ram stage-code 13. Read-only.
        assign r3_dbg_out_valid = r3_out_valid_f;
        assign r3_dbg_out_chunk = r3_out_chunk_f;

        // =============================================================
        // FIX-D: POST-ATTENTION RMSNorm  (the MLP-front assembly gap)
        // =============================================================
        // chain.cpp:118-124 / capture_goldens.py:221-227:
        //   h1       = residual + g6_o_proj            (R3 above, h1_buf_f)
        //   pre_mlp  = rmsnorm(h1, post_attn_norm_g)    (THIS instance, frac=12)
        //   gate/up  = GEMV(pre_mlp)                    (G7g/G7u read seam_premlp_rd)
        // Before this fix G7g/G7u read the RAW G6 o_proj (seam_r2_rd), skipping
        // BOTH the residual add AND the post-attn norm. Structurally IDENTICAL to
        // the G1 RMSNorm (same engine + frac=12 output); the only deltas are the
        // input stream (h1_buf_f, chunk-indexed by the M4 x_req tap) and the gamma
        // (gamma_postattn_buf = post_attn_norm_g). Launch on r3_done_f (h1 ready);
        // stream all 192 chunks; capture via the S6 out_chunk_strobe/idx_o into the
        // top-level premlp_buf; assert seam_premlp_full after all 192 captured.
        reg                  pa_start;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) pa_start <= 1'b0;
            else           pa_start <= r3_done_f;   // 1-cycle launch pulse after h1 done
        end
        // PRE_MLP X-FIX (2026-05-27): the rmsnorm_block x_req_chunk / out_chunk_idx_o
        // ports are $clog2((H/K_PAR)+1) = $clog2(193) = 8 bits wide ([7:0]). These
        // nets were declared [R6_CW:0] = [8:0] = 9 bits (R6_CW=$clog2(192)=8), so the
        // MSB (bit 8) was UNCONNECTED -> Z/X at sim. That X poisoned pa_eff_chunk and
        // pa_inrange (both functions of pa_x_req_chunk) -> pa_x_chunk = X -> the sumsq
        // accumulator went X -> scale_q12 X -> EVERY pre_mlp output chunk X (input-
        // independent: a single X in the shared sumsq poisons all 192 outputs). The
        // G1 instance never hit this because its g1_req_chunk net is [EMB_CW-1:0] with
        // EMB_CW=$clog2(193)=8 -> exactly the port width. Match that here: [R6_CW-1:0]
        // = [7:0] = the exact port width. (192 still fits in 8 bits; the <=192 range
        // test and the +1 slot bias are unchanged.)
        wire [R6_CW-1:0]                pa_x_req_chunk;
        wire                            pa_x_req_valid;
        // h1 input read (chunk-indexed by the norm's request tap; the engine's
        // x_req_chunk has the +1 slot bias -> read h1_buf_f[req-1] in range).
        wire [R6_CW-1:0] pa_eff_chunk = (pa_x_req_chunk >= (R6_CW+1)'(1))
                                        ? (pa_x_req_chunk - (R6_CW+1)'(1))
                                        : '0;
        wire pa_inrange = (pa_x_req_chunk >= (R6_CW+1)'(1)) &&
                          (pa_x_req_chunk <= (R6_CW+1)'(R6_NCHUNKS));
        // COMBREAD FIX (2026-06-08): h1_buf_f (192-deep -> BRAM) and
        // gamma_postattn_buf (ram_style="block") are synchronous-read BRAM —
        // register the read and set RDATA_LAT=1 on u_rmsnorm_premlp.
        reg signed [K_PAR*A_WIDTH-1:0] pa_x_chunk;
        reg signed [K_PAR*A_WIDTH-1:0] pa_gamma_chunk;
        always_ff @(posedge ap_clk) begin
            pa_x_chunk     <= pa_inrange ? h1_buf_f[pa_eff_chunk]              : '0;
            pa_gamma_chunk <= pa_inrange ? gamma_postattn_buf_rd(pa_eff_chunk) : '0;  // BUFSPLIT read
        end
        wire signed [K_PAR*A_WIDTH-1:0] pa_out_chunk;
        wire                            pa_out_strobe;
        wire [R6_CW-1:0]                pa_out_chunk_idx;   // PRE_MLP X-FIX: = port width [7:0] (was [8:0], MSB undriven)
        // FIX-D M1a: post-attn residual RMS≈17.37 → 25/8 (NOT G1's 31/10);
        // FIXD_PARAM_CHECK. The post-attn residual r3_h1 (RMS=17.37, int-absmax
        // 5319 = 347x the small-RMS embedding) needs SUMSQ_SHR=25, SCALE_SHR=8.
        // G1's 31/10 (mis-)averages the large-RMS residual into a degenerate
        // 0.25x-scale region (8413-LSB tail, real-RTL HARD FAIL). The 25/8 engine
        // emits OUT_FRAC=11 (Q5.11); the M1b <<1 seam below requants to Q4.12
        // (frac=12, pre_mlp.hex) before G7g/G7u.
        rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                        .SUMSQ_RSHIFT(12),
                        .SUMSQ_MULT(43691),
                        .SUMSQ_SHR(PRECISION_SAFE ? 29 : 25),
                        .SCALE_SHR(PRECISION_SAFE ? 10 : 8),
                        .XS_WIDTH(18),
                        .RDATA_LAT(1)) u_rmsnorm_premlp (   // COMBREAD FIX: registered BRAM read align
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(pa_start),
            .x_chunk(pa_x_chunk),
            .gamma_chunk(pa_gamma_chunk),
            .out_chunk(pa_out_chunk),
            .out_valid(),
            .done(),
            .x_req_chunk(pa_x_req_chunk),
            .x_req_valid(pa_x_req_valid),
            .out_chunk_strobe(pa_out_strobe),   // S6 clean per-chunk strobe (apply_en+AP_LAT)
            .out_chunk_idx_o(pa_out_chunk_idx),  // 0..191
            .apply_en_o()
        );
        // FIX-D M1b: the 25/8 premlp engine emits OUT_FRAC=11 (Q5.11). The
        // downstream pre_mlp golden (pre_mlp.hex) is frac=12 (Q4.12). Requant
        // per-lane with a <<1 (saturating left-shift, INT16) BEFORE storing into
        // premlp_buf. Clones the R8 sat16_lshift idiom (sat16_lshift_q124_to_q412)
        // but with shift=1 instead of 2. Left-shift by 1 is exact (no fractional
        // loss → no rounding needed); INT16-sat guards the +1 headroom bit.
        wire signed [K_PAR*A_WIDTH-1:0] pa_out_chunk_q412;
        genvar pal;
        for (pal = 0; pal < K_PAR; pal = pal + 1) begin : g_premlp_q412
            wire signed [A_WIDTH:0] pa_xe =
                $signed(pa_out_chunk[pal*A_WIDTH +: A_WIDTH]) <<< 1;   // Q5.11 -> Q4.12
            localparam logic signed [A_WIDTH-1:0] PA_PSAT = {1'b0, {(A_WIDTH-1){1'b1}}};
            localparam logic signed [A_WIDTH-1:0] PA_NSAT = {1'b1, {(A_WIDTH-1){1'b0}}};
            assign pa_out_chunk_q412[pal*A_WIDTH +: A_WIDTH] =
                (pa_xe > $signed(PA_PSAT)) ? PA_PSAT
              : (pa_xe < $signed(PA_NSAT)) ? PA_NSAT
              : pa_xe[A_WIDTH-1:0];
        end
        // pre_mlp buffer: full-H, written by the S6 strobe, read by G7g/G7u.
        reg signed [K_PAR*A_WIDTH-1:0] premlp_buf [0:R6_NCHUNKS-1];
        always_ff @(posedge ap_clk) begin
            if (pa_out_strobe) premlp_buf[pa_out_chunk_idx[R6_CW-1:0]] <= pa_out_chunk_q412;
        end
        // pre_mlp full flag: count the 192 captured chunks.
        // CROSS-FORWARD FIX (2026-07-25, CROSSFWD_SHORTLIST_LEAKAGE_RCA.md):
        // was cleared ONLY at pa_start (= r3_done_f, LATE). But the FSM enters
        // S_G7G on g6_done, and bk_input_full samples seam_premlp_full THERE —
        // so on forward N>=2 (and, in the depth loop, on every child LEVEL's
        // S_G6->S_G7G re-entry) the count was STALE-HIGH at 192 from the prior
        // pass, the G7g GEMV launched against the previous premlp_buf mid-
        // overwrite. Also clear on g6_start (the same wr_clear discipline every
        // stage_io_buffer seam already follows) so the "full" flag can never
        // carry across a pass. RED/GREEN proven on selfappend_hazard: B 1/8->8/8,
        // C 8/8 held. seam_premlp_full then re-asserts only after this pass's 192
        // chunks land, exactly as intended.
        reg [R6_CW:0] pa_cap_cnt;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)        pa_cap_cnt <= '0;
            else if (g6_start)    pa_cap_cnt <= '0;   // <-- cross-forward/level fix
            else if (pa_start)    pa_cap_cnt <= '0;
            else if (pa_out_strobe && pa_cap_cnt < (R6_CW+1)'(R6_NCHUNKS))
                                  pa_cap_cnt <= pa_cap_cnt + 1'b1;
        end
        assign seam_premlp_full = (pa_cap_cnt >= (R6_CW+1)'(R6_NCHUNKS));
        // D1 FIX (PDI#10): premlp_buf is a chunk-indexed RAM read by the G7g/G7u
        // GEMV at array_chunk_idx_out (the +1 lead). On silicon it infers BRAM =
        // 1-cycle registered read, so the legacy COMBINATIONAL `premlp_buf[idx]`
        // was stale-by-1 on silicon (W[c]·premlp[c-1]). Register the read here so
        // sim==silicon, paired with the array WLAT=1 (weight+enable +1cyc) that
        // re-aligns the now-registered activation. (G7g/G7u of the confirmed list.)
        reg signed [K_PAR*A_WIDTH-1:0] seam_premlp_rd_r;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) seam_premlp_rd_r <= '0;
            else           seam_premlp_rd_r <= premlp_buf[premlp_rd_idx];
        end
        assign seam_premlp_rd   = seam_premlp_rd_r;
        // WO-025 port-B: parallel registered ODD-chunk read (same D1 registered-
        // read semantics, paired with array WLAT=1 on both ports).
        reg signed [K_PAR*A_WIDTH-1:0] seam_premlp_rd_b_r;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) seam_premlp_rd_b_r <= '0;
            else           seam_premlp_rd_b_r <= premlp_buf[premlp_rd_idx_b];
        end
        assign seam_premlp_rd_b = seam_premlp_rd_b_r;

        // ---- R7 driver: a = ZERO (FIX-I), b = g9down_buf (0..191) ----
        // FIX-I (2026-05-27, reports/FIXI_R7_SCOPE_20260527.md): the depth-0
        // logits path must norm `to_logits = down[:H]` ALONE (chain.cpp:150-152).
        // The post-MLP residual in EAGLE4 applies ONLY to the for_reasoning half
        // (chain.cpp:143-148, a depth>=1 seed that is NOT implemented and is out
        // of depth-0 scope), NOT to the to_logits->G11->G12 frontier path.
        // RMSNorm is NOT sum-invariant, so adding h1 here rotated the normed
        // direction and corrupted g11_low -> G12 logits -> frontier.
        // Fix = drive the R6 seam with to_logits VERBATIM by zeroing the R7
        // a-operand (sat_add(0,b)=b exactly for in-range b, residual_add.sv:98-108):
        // u_resid_r7 now passes g9down_buf (= to_logits) through unchanged. The R6
        // write strobe (r7_out_valid / r7_out_chunk_idx) is preserved. The b-arm
        // still streams g9down_buf; the a-capture buffer is loaded with zeros.
        // (This realizes Option A's semantics — G11 norms to_logits alone — while
        // staying in-scope: g9_we_active/g9_widx are local to this generate block,
        // so feeding the top-level u_seam_r6 from them would need a hoist; zeroing
        // the in-scope a-arm is the minimal, bit-exact-equivalent edit against
        // r7_out.hex / g11_in_norm.hex.) DO NOT touch the R3 residual (correct).
        // After G9 finishes (bk_g9_done_r), stream a then b (same 2-phase idiom).
        reg [R6_CW:0] r7_a_cnt;  reg r7_a_run;
        reg [R6_CW:0] r7_b_cnt;  reg r7_b_run;  reg r7_started;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                r7_a_cnt <= '0; r7_a_run <= 1'b0;
                r7_b_cnt <= '0; r7_b_run <= 1'b0; r7_started <= 1'b0;
            end else begin
                if (bk_g9_done_r) begin
                    r7_a_cnt <= '0; r7_a_run <= 1'b1;
                    r7_b_cnt <= '0; r7_b_run <= 1'b0; r7_started <= 1'b0;
                end
                if (r7_a_run) begin
                    if (r7_a_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) begin
                        r7_a_run <= 1'b0; r7_started <= 1'b1;
                    end
                    r7_a_cnt <= r7_a_cnt + 1'b1;
                end
                if (r7_started && !r7_b_run) begin
                    r7_b_run <= 1'b1; r7_b_cnt <= '0; r7_started <= 1'b0;
                end else if (r7_b_run) begin
                    if (r7_b_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) r7_b_run <= 1'b0;
                    r7_b_cnt <= r7_b_cnt + 1'b1;
                end
            end
        end
        wire             r7_a_wr_f    = r7_a_run;
        wire [R6_CW-1:0] r7_a_idx_f   = r7_a_cnt[R6_CW-1:0];
        // FIX-I: a-operand forced to 0 so u_resid_r7 emits sat_add(0, g9down_buf)
        // = to_logits verbatim into the R6 seam (G11 norms to_logits ALONE).
        // (was: h1_buf_f[r7_a_idx_f] -> the spurious +h1 over-add.)
        wire signed [K_PAR*A_WIDTH-1:0] r7_a_chunk_f = '0;
        wire             r7_start_f   = r7_started;
        wire [R6_CW-1:0] r7_b_idx_f   = r7_b_cnt[R6_CW-1:0];
        // COMBREAD FIX (2026-06-08): g9down_buf is BRAM (sweep row #11). Register
        // the b-arm read + delay b_valid 1 cycle so b_chunk[k] is coincident with
        // b_valid[k] for u_resid_r7's RDATA_LAT=1 path. (a-arm is 0 -> sat_add(0,b),
        // but b must still be aligned, else to_logits is shifted by one chunk.)
        reg signed [K_PAR*A_WIDTH-1:0] r7_b_chunk_f;
        reg                            r7_b_valid_f;
        always_ff @(posedge ap_clk) begin
            r7_b_chunk_f <= g9down_buf[r7_b_idx_f];
            r7_b_valid_f <= r7_b_run;
        end

        residual_add #(.H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                       .FRAC_BITS(R7_FRAC), .RDATA_LAT(1)) u_resid_r7 (   // Q8.8 pass-through + registered read
            .clk(ap_clk), .rst_n(ap_rst_n),
            .a_valid(r7_a_wr_f), .a_chunk_idx(r7_a_idx_f), .a_chunk(r7_a_chunk_f),
            .start(r7_start_f), .b_chunk(r7_b_chunk_f), .b_valid(r7_b_valid_f),
            .out_valid(r7_out_valid), .out_chunk_idx(r7_out_chunk_idx_f), .out_chunk(r7_out_chunk),
            .done(r7_done)
        );
        assign r7_out_chunk_idx = r7_out_chunk_idx_f;

        // =============================================================
        // F1 — the depth>=1 re-key hidden ("for_reasoning") producer.
        //
        //   a_d = G9_down[H:2H] + ( block_input + G6_oproj_out )
        //                          \__________________________/
        //                            == h1_buf_f, ALREADY computed by
        //                               u_resid_r3 above (R3, FIX-C).
        //
        // Measured on 349 real GPU drafter forwards: DEPTHN_REKEY_GOLDENS.md
        // §4 checks A/B/C, max rel-L2 2.24e-04 (= the fp16 storage floor).
        // Gate: tb_forreasoning_gate (bit-exact, 72 rows, + 2 RED arms).
        //
        // Q FORMAT: the to_logits half is captured through
        // sat_acc_subchunk (acc-native frac8, the depth-0 precision contract).
        // for_reasoning is captured at the accumulator-NATIVE frac8 instead
        // (plain INT16 sat, no shift): max|G9hi| = 2.99 measured, so frac3
        // would leave ~24 codes and costs 4x accuracy (1.397e-02 vs 3.447e-03
        // rel-L2 against the fp16 golden). Do NOT "unify" the two captures.
        // =============================================================
        if (FORREASON) begin : g_forreason
            // (FR_H1_FRAC / FR_G9_FRAC / FR_A_FRAC hoisted to g_resid_fullh
            //  scope at the R3 a-arm — F2 derives its re-align shift from the
            //  same single-source localparams this instance uses.)

            // ---- capture G9 col-tiles 24..47 into the for_reasoning buffer ----
            wire [R6_CW-1:0] g9_hi_widx =
                R6_CW'((int'(g9_tile_lat) - G9_LO_NTILES)*LANE_SUBW + int'(g9_we_cnt));
            wire signed [K_PAR*A_WIDTH-1:0] g9_hi_wchunk =
                sat_acc_subchunk(g9_acc_lat, int'(g9_we_cnt) * K_PAR);   // frac8, no shift
            reg signed [K_PAR*A_WIDTH-1:0] g9hi_buf [0:R6_NCHUNKS-1];
            always_ff @(posedge ap_clk) begin
                if (g9_we_active && g9_hi_sel) g9hi_buf[g9_hi_widx] <= g9_hi_wchunk;
            end

            // ---- 192-chunk walk once the whole (48-tile) G9 has landed ----
            reg [R6_CW:0] fr_cnt;  reg fr_run;
            always_ff @(posedge ap_clk) begin
                if (!ap_rst_n) begin
                    fr_cnt <= '0; fr_run <= 1'b0;
                end else if (bk_g9_done_r) begin
                    fr_cnt <= '0; fr_run <= 1'b1;
                end else if (fr_run) begin
                    if (fr_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) fr_run <= 1'b0;
                    fr_cnt <= fr_cnt + 1'b1;
                end
            end
            // COMBREAD discipline (feedback_funcsim_slice_lutram_downgrade_trap):
            // BOTH operand reads are REGISTERED and the valid/idx are delayed by
            // the SAME one cycle, so sim == silicon regardless of how h1_buf_f /
            // g9hi_buf are mapped.
            reg signed [K_PAR*A_WIDTH-1:0] fr_h1_q, fr_g9_q;
            // CHILD_SEAM_FIX: the a-arm remainder buffer (see g_r3a_rem above).
            // Written on the R3 a-walk (same idx/data timing as a_save_buf);
            // read on the fr walk co-timed with fr_h1_q. FF, ACCFIX_SEAM_P-gated.
            // MANDATE (CLAUDE.md yellow rule; feedback_ff_staging_intermediate_seams):
            // intermediate activation seams are FF, never BRAM/LUTRAM. Without
            // this attribute synth inferred a 192x80 RAMB ("too shallow for URAM
            // -> BRAM") and ACCFIXCHECK correctly aborted the 0x1125 build at
            // fr_rem_buf_fdre=0. 192 x 80b = 15,360 FDRE = the gate's expectation.
            (* ram_style = "registers" *)
            reg [K_PAR*5-1:0] fr_rem_buf [0:R6_NCHUNKS-1];
            reg [K_PAR*5-1:0] fr_rem_q;
            always_ff @(posedge ap_clk) begin
                if (ACCFIX_SEAM_P && r3_a_wr_f) fr_rem_buf[r3_a_idx_f] <= fr_rem_now;
            end
            reg                            fr_v_q;
            reg [R6_CW-1:0]                fr_idx_q;
            always_ff @(posedge ap_clk) begin
                fr_h1_q  <= h1_buf_f[fr_cnt[R6_CW-1:0]];
                fr_rem_q <= fr_rem_buf[fr_cnt[R6_CW-1:0]];   // CHILD_SEAM_FIX remainder
                fr_g9_q  <= g9hi_buf[fr_cnt[R6_CW-1:0]];
                fr_v_q   <= fr_run;
                fr_idx_q <= fr_cnt[R6_CW-1:0];
            end

            // ---- F2 (DEPTH-MUX): the read port's REAL consumers ----------
            // ONE read port, TWO temporally-disjoint consumers. r3_a_run is
            // high for exactly the 192 R3 a-arm address cycles (mid-body,
            // after bk_g6_done_r); the G1 h_norm sweep only reads at body
            // start, while the R3 walk is idle. So the port follows the R3
            // address when its walk runs and the G1 read address otherwise.
            // Both consumers present addr@t and consume rd_chunk@t+1 —
            // identical to the registered h_fc_buf reads they replace.
            wire [R6_CW-1:0] f2_rd_addr = r3_a_run ? r3_a_idx_raw
                                                   : R6_CW'(g1_rd_idx);
            // F2: KF_BANKS=1 WRITE GUARD. At depth 1 every child must read the
            // ROOT's a_0 (spec_utils.py:468 repeat_interleave). The children
            // run the same 48-tile G9 + fr walk, so WITHOUT this guard child
            // i's own tail-write would clobber bank 0 before child i+1 reads
            // it (children are sequential; the fr walk lands well before the
            // next child's G1). Suppress the producer stream at level>=1.
            // F4 (KF_BANKS=8) lifts the guard and routes wr_branch instead.
`ifdef F2_RED_NOGUARD
            // C5 RED ARM (sim-only compile define, never set in synth): DISABLE
            // the KF_BANKS=1 sibling-clobber write guard. child i's tail-write is
            // no longer suppressed at level>=1, so it overwrites bank 0 before
            // child i+1 reads it -> the d=1 repeat_interleave semantics break.
            // A DUT define (not a forced net) so it cannot be clobbered by the
            // continuous assign (feedback_xsim_force_clobbered_by_contassign_dead_red_arm).
            wire f2_fr_wr_ok = 1'b1;
`else
            wire f2_fr_wr_ok = (FR_KF_BANKS > 1) || !f2_lvl_ge1_w;
`endif

            // ---- F4 (FR_KF_BANKS>1): per-parent `a` re-gather bank selects ----
            // WRITE: child branch j's for_reasoning a_d[j] -> its OWN bank j.
            // READ : survivor's block-input seed = a_{d-1}[parent_slot] -> read
            //        bank fr_par_slot (chain.cpp:491-495 parent_for_reasoning[s]
            //        = child_for_reasoning[parent_branch]).
            // DOUBLE-BUFFER (FR_KF_BANKS>8): a_{d-1} must survive ALL of level d's
            // reads while a_d is being produced, but child j writes its bank
            // mid-level, before a later sibling whose parent==j reads it. So the
            // read set (previous level, epoch ~level[0]) and the write set
            // (this level, epoch level[0]) are DIFFERENT 8-bank halves = the
            // level-parity ping-pong. The permutation is applied for FREE at
            // READ time via fr_par_slot; no explicit copy pass. At FR_KF_BANKS==1
            // both selects fold to 0 -> F1 byte-identical.
            localparam int FR_BW   = (FR_KF_BANKS <= 1) ? 1 : $clog2(FR_KF_BANKS);
            localparam bit FR_DBUF = (FR_KF_BANKS > 8);
            wire        fr_wr_epoch = level_i_w[0];
            wire        fr_rd_epoch = ~level_i_w[0];
`ifdef G3_RED_POOLLOW
            // G3 RED ARM (top level): the PRE-INVERSION F4 mapping -- parent
            // slot taken from the low 3 bits of the parent POOL index. Exact
            // through depth 2, WRONG from depth 3 (see tree_level_fsm cs_pslot).
            wire [2:0]  fr_par_slot = fr_par_node_w[2:0];
`else
            // G3: the parent's frontier SLOT, published by tree_level_fsm as a
            // first-class output (surv_parent_slot_o) instead of being guessed
            // from the pool index's low bits. Exact at EVERY depth.
            wire [2:0]  fr_par_slot = fr_par_slot_w;
`endif
            wire [FR_BW-1:0] fr_wr_branch =
                (FR_KF_BANKS <= 1) ? '0 :
                FR_DBUF            ? FR_BW'({fr_wr_epoch, branch_j_w}) :
                                     FR_BW'(branch_j_w);
            wire [FR_BW-1:0] fr_rd_branch =
                (FR_KF_BANKS <= 1) ? '0 :
                FR_DBUF            ? FR_BW'({fr_rd_epoch, fr_par_slot}) :
                                     FR_BW'(fr_par_slot);

            wire                            fr_out_valid;
            wire [R6_CW-1:0]                fr_out_idx;
            wire signed [K_PAR*A_WIDTH-1:0] fr_out_chunk;
            wire signed [K_PAR*A_WIDTH-1:0] fr_rd_chunk;
            for_reasoning_producer #(
                .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                .H1_FRAC(FR_H1_FRAC), .G9_FRAC(FR_G9_FRAC), .A_FRAC(FR_A_FRAC),
                .KF_BANKS(FR_KF_BANKS), .ABANK_BRAM_P(FR_ABANK_BRAM),
                .SEAM_FIX_P(ACCFIX_SEAM_P)          // CHILD_SEAM_FIX presence
            ) u_forreason (
                .clk(ap_clk), .rst_n(ap_rst_n),
                .in_valid(fr_v_q && f2_fr_wr_ok), .in_idx(fr_idx_q),
                .h1_chunk(fr_h1_q), .g9_chunk(fr_g9_q),
                .seam_en(accfix_seam_en),           // ACCFIX bit0 (runtime)
                .rem_chunk(fr_rem_q),               // CHILD_SEAM_FIX remainder
                .wr_branch(fr_wr_branch),             // F4: child j -> bank j (+epoch)
                .out_valid(fr_out_valid), .out_idx(fr_out_idx), .out_chunk(fr_out_chunk),
                .rd_branch(fr_rd_branch),             // F4: read parent slot (+prev epoch)
                .rd_idx(f2_rd_addr),                  // F2: the two consumers' shared address
                .rd_chunk(fr_rd_chunk)
            );

            // F2: surface the read data + level select on the module-scope
            // depth-mux nets (consumed at the G1 h_norm register and the R3
            // a-arm). The F1 `fr_a_rd_keep` placeholder is REPLACED by these
            // real consumers; fr_a_out_keep stays (the streamed out_* mirror
            // still has no functional consumer, and it pins the producer
            // output regs for the F1 netlist census).
            assign f2_fr_rd_data_w = fr_rd_chunk;
            assign f2_lvl_ge1_w    = DEPTH_N && (level_i_w != 4'd0);

            // ---- C4 RIDER: shared a-bank read-port temporal-disjointness -----
            // f2_rd_addr (above) is r3_a_run ? r3_a_idx_raw : g1_rd_idx — ONE
            // producer read port arbitrated between TWO consumers. That single
            // port is correct because the LIVE level>=1 consumer windows never
            // overlap within a pass: G1's h_norm sweep completes long before
            // the R3 a-walk (post-bk_g6_done_r). g1_rd_ir is now bounded by
            // g1_req_valid plus the required one-cycle final-chunk drain tail;
            // it must not be a sticky "address happens to be in range" flag.
            // If they ever coincided, the mux would hand R3's address to the G1
            // consumer (r3_a_run wins the ternary), corrupting a level>=1 read.
            // The REAL invariant, now asserted (was comment-only). Sim-only:
            // `ifndef SYNTHESIS => zero synth impact (matches producer idiom).
`ifndef SYNTHESIS
            f2_c4_port_disjoint: assert property (
                @(posedge ap_clk) disable iff (!ap_rst_n)
                !(f2_lvl_ge1_w && g1_rd_ir && r3_a_run)
            ) else $error("F2 C4 VIOLATED @ %0t: g1_rd_ir & r3_a_run both high -> shared a-bank read port collision (G1 gets R3's address)", $time);
`endif
            initial begin
                if (EMB_K_PAR*EMB_AW != K_PAR*A_WIDTH)
                    $error("F2: G1 chunk geometry %0dx%0d != producer chunk geometry %0dx%0d",
                           EMB_K_PAR, EMB_AW, K_PAR, A_WIDTH);
            end

            (* keep = "true" *) reg signed [K_PAR*A_WIDTH-1:0] fr_a_out_keep;
            always_ff @(posedge ap_clk) begin
                if (fr_out_valid) fr_a_out_keep <= fr_out_chunk;
            end
        end else begin : g_no_forreason
            // FORREASON=0: depth-mux nets constant-0 -> both consumer muxes
            // fold to the pre-F2 h_fc path, byte-identical elaboration.
            assign f2_fr_rd_data_w = '0;
            assign f2_lvl_ge1_w    = 1'b0;
        end
    end else begin : g_resid_pilot
        // F2: no producer in the pilot arm -> depth-mux nets constant-0
        // (both consumers fold to their legacy h_fc paths).
        assign f2_fr_rd_data_w = '0;
        assign f2_lvl_ge1_w    = 1'b0;
        // ---- legacy single-chunk pilot (BACK_SEQ=0) ----
        // FIX-D: the pilot arm has no full-H h1 / post-attn norm; drive the
        // pre_mlp seam from the raw R2 (legacy bypass) so the forward wires are
        // never undriven. (The smoke + S8 gates run BACK_SEQ=1, which uses the
        // real post-attn norm above.)
        assign seam_premlp_rd   = seam_r2_rd;
        assign seam_premlp_full = seam_r2_full;
        // WO-025 port-B: legacy pilot arm (BACK_SEQ=0) never runs DUAL_PORT=1
        // (elab-asserted at the a_chunk_b mux) — tie the odd-chunk read '0.
        assign seam_premlp_rd_b = '0;
        // PROBE V3: no R3 residual_add in the pilot arm -> tie the dbg mirror 0.
        assign r3_dbg_out_valid = 1'b0;
        assign r3_dbg_out_chunk = '0;
        // FIX-E R3: legacy pilot R3 a-arm -> h_fc_buf (for consistency; this arm is
        // BACK_SEQ=0, NOT exercised by the smoke/S8 green gates).
        wire signed [K_PAR*A_WIDTH-1:0] x_input_chunk = h_fc_buf_rd(EMB_CW'(EMB_DATA_SLOT));  // BUFSPLIT read
        reg r3_a_wr;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) r3_a_wr <= 1'b0; else r3_a_wr <= g6_start;
        end
        reg signed [K_PAR*A_WIDTH-1:0] r3_b_chunk_l;  reg r3_b_valid;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin r3_b_chunk_l <= '0; r3_b_valid <= 1'b0; end
            else begin
                if (g6_done) r3_b_chunk_l <= g6_out_sat_chunk_pilot;
                r3_b_valid <= g6_done;
            end
        end
        wire r3_out_valid; wire signed [K_PAR*A_WIDTH-1:0] r3_out_chunk; wire r3_done;
        residual_add #(.H(SEAM_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                       .FRAC_BITS(R3_FRAC)) u_resid_r3 (   // FIX-C: r3_h1 = Q11.5 (label; sat HW frac-agnostic)
            .clk(ap_clk), .rst_n(ap_rst_n),
            .a_valid(r3_a_wr), .a_chunk_idx('0), .a_chunk(x_input_chunk),
            .start(g6_done), .b_chunk(r3_b_chunk_l), .b_valid(r3_b_valid),
            .out_valid(r3_out_valid), .out_chunk_idx(), .out_chunk(r3_out_chunk), .done(r3_done)
        );
        reg signed [K_PAR*A_WIDTH-1:0] h1_buf;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) h1_buf <= '0; else if (r3_out_valid) h1_buf <= r3_out_chunk;
        end
        wire signed [K_PAR*A_WIDTH-1:0] g9_out_sat_chunk =
            sat_acc_subchunk(array_acc_out, 0);
        reg r7_a_wr;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) r7_a_wr <= 1'b0; else r7_a_wr <= g9_start;
        end
        reg signed [K_PAR*A_WIDTH-1:0] r7_b_chunk_l;  reg r7_b_valid;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin r7_b_chunk_l <= '0; r7_b_valid <= 1'b0; end
            else begin
                if (g9_done) r7_b_chunk_l <= g9_out_sat_chunk;
                r7_b_valid <= g9_done;
            end
        end
        localparam int R7_PILOT_CW = (SEAM_H/K_PAR > 1) ? $clog2(SEAM_H/K_PAR) : 1;
        wire [R7_PILOT_CW-1:0] r7_pilot_idx;
        residual_add #(.H(SEAM_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                       .FRAC_BITS(R7_FRAC)) u_resid_r7 (   // r7_out = Q8.8 (label; sat HW frac-agnostic)
            .clk(ap_clk), .rst_n(ap_rst_n),
            .a_valid(r7_a_wr), .a_chunk_idx('0), .a_chunk(h1_buf),
            .start(g9_done), .b_chunk(r7_b_chunk_l), .b_valid(r7_b_valid),
            .out_valid(r7_out_valid), .out_chunk_idx(r7_pilot_idx), .out_chunk(r7_out_chunk),
            .done(r7_done)
        );
        assign r7_out_chunk_idx = {{(R6_CW-R7_PILOT_CW){1'b0}}, r7_pilot_idx};
    end
    endgenerate

    // PROBE V3: top-scope mirror of the R3 (post-attn) residual_add output, for
    // dbg_probe_ram stage-code 13. Driven inside the g_resid_fullh generate block
    // (BACK_SEQ=1) and tied 0 in the pilot arm. READ-ONLY tap (no datapath load).
    wire                            r3_dbg_out_valid;
    wire signed [K_PAR*A_WIDTH-1:0] r3_dbg_out_chunk;

    // R7 output -> R6 seam: full-H (BACK_SEQ) or zero-extended pilot index.
    wire                            r7_out_valid;
    wire signed [K_PAR*A_WIDTH-1:0] r7_out_chunk;   // = out (post-MLP residual, Q12.4)
    wire                            r7_done;
    wire [R6_CW-1:0]                r7_out_chunk_idx;     // full-H residual slot 0..191
    wire [R6_CW-1:0]                r7_out_chunk_idx_f;   // R7 full-H driver out idx
    wire [R6_CW-1:0]                r6_wr_idx = r7_out_chunk_idx;
    // pilot G6 sat chunk (only used by the BACK_SEQ=0 arm)
    wire signed [K_PAR*A_WIDTH-1:0] g6_out_sat_chunk_pilot = sat_acc_head0_chunk(array_acc_out);

    // -----------------------------------------------------------------
    // R6 — G9 down (via R7 residual) -> G11 RMSNorm x_chunk.  S5B FULL-WIDTH
    // REWRITE (was: single-chunk head-0 broadcast pilot, H=K_PAR, NCHUNKS=1).
    //
    // Producer = R7 residual output (Q8.8, R7_FRAC=8). When the
    // S7 FSM streams the full post-MLP residual, u_resid_r7 emits a CLEAN
    // addressed strobe: out_valid pulses once per chunk, in order, with
    // out_chunk_idx (0..191) carrying the slot — a usable write strobe (NO
    // strobe-tap needed, unlike rmsnorm out_valid). The full 3072-element
    // residual streams into a full-H stage_io_buffer (H=3072, NCHUNKS=192,
    // AUTO_SEQ=0 addressed by r7_out_chunk_idx so order/skew can't corrupt
    // the slot map).
    //
    // The g11 RMSNorm reads it CHUNK-INDEXED by its M4 x_req_chunk tap (the
    // canonical full-H chunk request, high during the SUMSQ + APPLY streaming
    // phases — same idiom S3 uses for G1->G4), so g11 sees the FULL R7/G9
    // output. The seam moves bits verbatim (format-agnostic); g11 interprets
    // the input as Q8.8 (the full-H averaging fix SUMSQ_MULT/SHR above).
    // -----------------------------------------------------------------
    // S7b — g11 R6 INPUT FEED: the S3 off-by-one + last-chunk range-gate fix.
    // The g11 rmsnorm_block (M4 idiom) pre-increments x_req_chunk during its
    // SUMSQ/APPLY phases — so when `en` is actually high requesting chunk c, the
    // tap presents x_req_chunk = c+1 (proven by S3 $display trace for the IDENTICAL
    // module used for G1). The R6 seam holds the residual at slots 0..191, so the
    // read index must be x_req_chunk-1 (BACK_SEQ). The LAST sumsq/apply request
    // (req_chunk=NCHUNKS) lands one cycle into S_DRAIN where x_req_valid is already
    // LOW, so the read must present input whenever req_chunk in [1, NCHUNKS] (NOT
    // gated on x_req_valid) — mirroring S3's g1_eff_inrange. The legacy pilot
    // (BACK_SEQ=0) keeps the bare index (single-chunk, never hit drain).
    wire [R6_CW-1:0] r6_rd_idx = BACK_SEQ
        ? (g11_x_req_chunk - G11_CW'(1))     // full-H: undo the apply/sumsq pre-increment
        : g11_x_req_chunk[R6_CW-1:0];        // pilot: slot == req_chunk
    wire r6_rd_inrange = BACK_SEQ
        ? (g11_x_req_chunk >= G11_CW'(1) && g11_x_req_chunk <= G11_CW'(R6_NCHUNKS))
        : 1'b1;
    // (seam_r6_rd_c is declared at the g11 RMSNorm instance and driven here.)
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                      .AUTO_SEQ(0), .FRAC_BITS(R7_FRAC)) u_seam_r6 (   // R6 carries r7_out Q8.8
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(r7_out_valid), .wr_chunk_idx(r6_wr_idx), .wr_chunk(r7_out_chunk),
        .wr_clear(g9_start), .wr_count(), .full(seam_r6_full),
        .rd_valid(r6_rd_inrange), .rd_chunk_idx(r6_rd_idx), .rd_chunk(seam_r6_rd_c)
    );

    // -----------------------------------------------------------------
    // R8 — G11 RMSNorm -> g11_lmdown (FIX-G) input activation.
    // Producer: rmsnorm_block g11 out_chunk = g11hp (SCALE_SHR=11) Q6.10.
    //
    // FIX-G (2026-05-27, FIXG_LMDOWN_SPEC §4.1): seam_r8 is NO LONGER the G12
    // activation — it is the INPUT to the new dense g11_lmdown GEMV, which the
    // reference feeds at the g11_in seam frac = 10 (Q6.10). The nrm input reaches
    // absmax 11.82 cross-seed → the OLD Q4.12 requant (<<2, range ±8) CLIPPED it.
    // So seam_r8 now carries Q6.10 VERBATIM (R8_REQ_LSHIFT=0, no shift): g11hp out
    // is already frac=10 = the lmdown input frac. The lmdown GEMV reads seam_r8 at
    // frac=10 and produces low_rank at frac=10; G12 then consumes low_rank (frac=10)
    // via the low_rank_rd mux arm (re-pointed below). The saturate-to-INT16 stays
    // (defensive; with shift=0 it only clamps the already-in-range Q6.10 values).
    // -----------------------------------------------------------------
    // Q6.10 verbatim (FIX-G): SHIFT = G11_OUT_FRAC - G11_OUT_FRAC = 0 (no scale).
    //   seam_r8_int = sat16( g11hp_int << 0 )  (Q6.10 -> Q6.10, lmdown input frac)
    // width-derived rails ±2^15; SHIFT is a compile-time constant.
    localparam int R8_REQ_LSHIFT = G11_OUT_FRAC - G11_OUT_FRAC;   // 0 (Q6.10 verbatim, FIX-G)
    function automatic logic signed [A_WIDTH-1:0] sat16_lshift_q124_to_q412;
        input signed [A_WIDTH-1:0] x;
        logic signed [A_WIDTH+R8_REQ_LSHIFT:0] xe;   // headroom for the left-shift
        localparam logic signed [A_WIDTH-1:0] PSAT = {1'b0, {(A_WIDTH-1){1'b1}}};
        localparam logic signed [A_WIDTH-1:0] NSAT = {1'b1, {(A_WIDTH-1){1'b0}}};
        begin
            xe = $signed(x) <<< R8_REQ_LSHIFT;
            if (xe > $signed(PSAT))      sat16_lshift_q124_to_q412 = PSAT;
            else if (xe < $signed(NSAT)) sat16_lshift_q124_to_q412 = NSAT;
            else                         sat16_lshift_q124_to_q412 = xe[A_WIDTH-1:0];
        end
    endfunction
    // S6 FULL-WIDTH REWRITE (was: single-chunk head-0 broadcast pilot).
    // The g11 RMSNorm now STREAMS its full 3072-element output chunk-by-chunk
    // into a full-H stage_io_buffer (NCHUNKS=192). The clean per-chunk strobe
    // (g11_out_strobe) + index (g11_out_chunk_idx) come from the rmsnorm_block
    // apply_en+AP_LAT taps (NOT out_valid, which is not a clean 192-pulse strobe).
    // G12 reads the seam CHUNK-INDEXED by the GEMV array's array_chunk_idx_out
    // (the array consumes a_chunk in lockstep with its current chunk_idx — see
    // gemv_lane_scaled_stream a_lane/in_valid), so G12 sees the FULL G11 output.
    //
    // The Q6.10 -> Q4.12 (<<2 sat, C3) requant is applied PER CHUNK as the stream is
    // captured (combinational, single ONE-driver assign), preserving the R8 seam
    // contract (G12's GEMV activation consumes Q4.12). Width-derived, format-only.
    wire signed [K_PAR*A_WIDTH-1:0] r8_chunk_q412;
    genvar r8l;
    generate
        for (r8l = 0; r8l < K_PAR; r8l = r8l + 1) begin : g_r8_requant
            assign r8_chunk_q412[r8l*A_WIDTH +: A_WIDTH] =
                sat16_lshift_q124_to_q412(rms_g11_out_chunk[r8l*A_WIDTH +: A_WIDTH]);
        end
    endgenerate
    // FULL-WIDTH seam: H=3072 (NCHUNKS=192). Addressed write (AUTO_SEQ=0) using
    // the strobe's logical chunk index so order/skew can't corrupt the slot map.
    // FIX-G: seam_r8 is read by the g11_lmdown DENSE GEMV (during S_GLM), NOT by
    // the shared array. The lmdown GEMV drives the chunk request (glm_a_chunk_idx,
    // 0..191) and consumes seam_r8 at frac=10 (Q6.10).
    wire [G11_CW-1:0] glm_a_chunk_idx;
    wire              glm_a_chunk_idx_valid;
    wire [G11_CW-1:0] r8_rd_idx = glm_a_chunk_idx;
    // M7 GLM=0 WIRED-TAP: seam_r8 internal RD_LAT nodes (DBG_LMTAP=1)
    wire [15:0] lmtap_seam_dout0;   // u_seam_r8 BRAM DOUT lane0 (pre-guard)
    wire        lmtap_rd_guard_q;   // u_seam_r8 RD_LAT guard
    // CO-TRAVEL (A4 slip-invariant fix): the seam's NEW outputs deliver the chunk
    // index + valid that ride WITH the activation through the SAME RD_LAT pipe, so
    // u_lmdown can address its weight ROM by the DELIVERED index (data-driven) ->
    // dot product slip-invariant for ANY seam read-latency skew.
    wire [G11_CW-1:0] glm_seam_idx_dlv;   // u_seam_r8 rd_chunk_idx_out (co-travel)
    wire              glm_seam_valid_dlv;  // u_seam_r8 rd_valid_out    (co-travel)
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(0), .RD_LAT(D1_RDLAT), .DBG_TAP(DBG_LMTAP)) u_seam_r8 (  // D1: G11-lmdown activation seam, registered (paired w/ u_lmdown LMDOWN_SLR_REG=1)
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(g11_out_strobe), .wr_chunk_idx(g11_out_chunk_idx[G11_CW-1:0]),
        .wr_chunk(r8_chunk_q412),                                   // Q6.10 (FIX-G, verbatim)
        .wr_clear(g11_start), .wr_count(), .full(),
        .rd_valid(glm_a_chunk_idx_valid), .rd_chunk_idx(r8_rd_idx), .rd_chunk(seam_r8_rd),
        .dbg_rd_chunk_q0(lmtap_seam_dout0), .dbg_rd_guard_q(lmtap_rd_guard_q),
        .rd_chunk_idx_out(glm_seam_idx_dlv), .rd_valid_out(glm_seam_valid_dlv)  // CO-TRAVEL
    );

    // =================================================================
    // FIX-G — g11_lmdown DENSE fp16/INT16 GEMV (3072 -> 256 low_rank), NO bias.
    // Reads seam_r8 (G11 RMSNorm out, Q6.10/frac=10) chunk-by-chunk during S_GLM,
    // produces low_rank[256] (Q6.10/frac=10) into u_low_rank_buf, which G12 reads
    // via the low_rank_rd mux arm. Weights baked from /tmp/p4/lmdown_W_q15.hex
    // (INT16 Q1.15, 48..64 URAM, ram_style=ultra). Unit-checked bit-exact-class
    // vs the checkpoint golden. Q5.11 clipped valid current-checkpoint values
    // (observed max >20) at 15.9995 and lost 46/512 GPU candidates; Q6.10
    // preserves the full measured range and restores the 512/512 candidate set.
    // (FIXG_LMDOWN_SPEC §2/§3/§5.)
    // =================================================================
    localparam int LR_M       = 256;
    localparam int LR_LANES   = 16;
    localparam int LR_NCHUNKS = LR_M / LR_LANES;                 // 16 output chunks
    localparam int LR_OCW     = (LR_NCHUNKS > 1) ? $clog2(LR_NCHUNKS) : 1;

    wire                           glm_wr_valid;
    wire [LR_OCW-1:0]              glm_wr_chunk_idx;
    wire signed [LR_LANES*A_WIDTH-1:0] glm_wr_chunk;

    // GLM-zero placement-INSENSITIVE fix (LMDOWN_BANKSPLIT_RESULT.md §1, GLM_ZERO_ROOTCAUSE.md §6):
    // swap lmdown_gemv -> lmdown_gemv_banksplit (md5 b2549419). SAME port list, SAME
    // .LMDOWN_SLR_REG(D1_RDLAT). The two 256b operand-delivery nets are split into 8x32b
    // registered crossings so NO wide net spans an SLR -> placement-insensitive, needs NO
    // lmdown pblock (sidesteps BOTH GLM_ZERO §4a EXCLUDE_PLACEMENT-flip AND §4b over-packing).
    // xsim A/B bit-exact (256/256 maxdiff=0, GLM cap-CRC 0x859b57b1); OOC WNS +7.691@100MHz,
    // ROM 464 RAMB survives, bank regs survive as 256+4096 FDRE, URAM/BRAM/DSP neutral.
    lmdown_gemv_banksplit #(
        .K(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .M(LR_M), .LANES(LR_LANES),
        .IN_FRAC(10), .W_FRAC(15), .OUT_FRAC(10),
        .W_HEX("/home/y344shi/eagle4_artifacts/p4/lmdown_W_q15.hex"),
        .W_MEM_DIR(LMDOWN_MEM_DIR_P),
        // D1 FIX (PDI#10, FIX_INTEGRATION_TOP): the G11-lmdown body GEMV reads
        // seam_r8 (forced BRAM) for its a_chunk. On silicon the seam read is
        // 1-cycle registered (u_seam_r8 RD_LAT=D1_RDLAT below), so the internal
        // a_d<=a_chunk capture would otherwise eat act[c-1] (stale-by-1, the same
        // class as PDI10_D1_REGREAD_CONFIRM's "G11-lmdown seam_r8" body GEMV).
        // LMDOWN_SLR_REG=1 delays the RD-launch select (grp/chunk/run) by +1 so
        // the seam fetch and the internal weight-ROM read both reference chunk c
        // when a_d captures act[c]. Paired with RD_LAT=1 on u_seam_r8. Default 0 =
        // legacy combinational seam (== prior golden).
        .LMDOWN_SLR_REG(D1_RDLAT),
        // FF_ACT_CACHE integration #2: per-SLR FF landing register on the lmdown
        // activation input (seam_r8_rd -> u_lmdown.a_chunk) so GLM's lanes consume
        // a LOCAL FF copy, never a cross-SLR BRAM read. The +1 landing latency is
        // absorbed by the extra w_row_d2/*_rd3 weight+marker stage (bit-exact).
        // SEAM_FF_LAND=0 = wire-through, byte+cycle identical.
        // pdi17 SEAMFIX: decoupled from FF_ACT_CACHE (body GEMV strap). The GLM seam
        // landing reg now rides its OWN strap SEAM_FF_LAND_P=1 (the GLM=0 fix), while
        // the body GEMV (.FF_ACT_CACHE @3149) reverts to the silufix2-golden +1 seam.
        .SEAM_FF_LAND(SEAM_FF_LAND_P),
        // CO-TRAVEL (A4 slip-invariant fix, GLM_FLEET_A4_RDLAT.md): address the
        // weight ROM by the chunk index that PHYSICALLY rides with the activation
        // through the seam (glm_seam_idx_dlv/_valid) -> the GLM dot is golden for
        // ANY seam read-latency / per-bank slip (the +128 class), NOT just the sim
        // cadence. REPLACES the SEAM_FF_LAND/LMDOWN_SLR_REG matched-delay realign.
        .COTRAVEL(1'b1),
        .DBG_TAP(DBG_LMTAP)
    ) u_lmdown (
        .clk(ap_clk), .rst_n(ap_rst_n), .start(glm_start),
        .a_chunk(seam_r8_rd),                       // G11 RMSNorm out (Q6.10)
        .a_chunk_idx(glm_a_chunk_idx),              // drives seam_r8 read ptr (0..191)
        .a_chunk_idx_valid(glm_a_chunk_idx_valid),
        .a_chunk_idx_in(glm_seam_idx_dlv),          // CO-TRAVEL delivered index
        .a_chunk_idx_valid_in(glm_seam_valid_dlv),  // CO-TRAVEL delivered valid
        .wr_valid(glm_wr_valid), .wr_chunk_idx(glm_wr_chunk_idx), .wr_chunk(glm_wr_chunk),
        .done(glm_done),
        .dbg_a_d0(lmtap_a_d0), .dbg_run_valid(lmtap_run_valid)
    );

    // =================================================================
    // M7 GLM=0 WIRED-TAP LOCALIZER capture (DBG_LMTAP). ONE always_ff owns all
    // capture regs + the self-clearing `armed` flag (feedback_pulse_reg_multidriver_
    // reset_plus_proc: no reg written in two processes -> no synth GND-tie). Latch
    // the 3 cone nodes on the FIRST GLM S_RUN cycle (lmtap_run_valid rising while
    // armed). `armed` set by glm_start (entering GLM), cleared after the latch.
    // Per-node canary (A001/A002/A003) distinguishes a dead reg (reads 0, no canary
    // => INCONCLUSIVE) from a genuine captured 0 (canary+valid set, data 0 => the
    // zeroing site) -- feedback_probe_reg_needs_real_axi_write_and_liveness.
    // =================================================================
    wire [15:0] lmtap_a_d0;        // u_lmdown a_d lane0 (MUL A operand)
    wire        lmtap_run_valid;   // u_lmdown run_eff (GLM S_RUN)
    reg         lmtap_armed;
    reg         lmtap_fired;
    reg  [15:0] lmtap_seam_cap;    // seam BRAM DOUT lane0 captured
    reg         lmtap_guard_cap;   // rd_guard_q captured
    reg  [15:0] lmtap_ad_cap;      // a_d lane0 captured
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            lmtap_armed     <= 1'b0;
            lmtap_fired     <= 1'b0;
            lmtap_seam_cap  <= 16'h0;
            lmtap_guard_cap <= 1'b0;
            lmtap_ad_cap    <= 16'h0;
        end else begin
            if (glm_start) begin
                lmtap_armed <= 1'b1;     // arm on entering S_GLM
                lmtap_fired <= 1'b0;     // re-arm each forward
            end else if (lmtap_armed && lmtap_run_valid) begin
                // first GLM run cycle: latch all 3 cone nodes in one cycle
                lmtap_seam_cap  <= lmtap_seam_dout0;
                lmtap_guard_cap <= lmtap_rd_guard_q;
                lmtap_ad_cap    <= lmtap_a_d0;
                lmtap_armed     <= 1'b0; // self-clear
                lmtap_fired     <= 1'b1;
            end
        end
    end
    // readback words (exactly 32b): [31:24]=canary8, [16]=fired, [15:0]=data16.
    // A dead window (canary != 0xA1/A2/A3) is distinguishable from a captured zero
    // (canary present + fired=1 + data=0). 8(canary)+7(rsvd)+1(fired)+16(data)=32.
    //   0x120 seam_word : {8'hA1, 7'b0, fired, seam_dout0[15:0]}
    //   0x124 guard_word: {8'hA2, 7'b0, fired, 15'b0, guard}      (guard in bit0)
    //   0x128 ad_word   : {8'hA3, 7'b0, fired, a_d0[15:0]}
    wire [31:0] lmtap_word_seam  = {8'hA1, 7'b0, lmtap_fired, lmtap_seam_cap[15:0]};
    wire [31:0] lmtap_word_guard = {8'hA2, 7'b0, lmtap_fired, 15'b0, lmtap_guard_cap};
    wire [31:0] lmtap_word_ad    = {8'hA3, 7'b0, lmtap_fired, lmtap_ad_cap[15:0]};

    // low_rank[256] buffer: 16 chunks of 16 INT16 Q5.11. Written by the lmdown GEMV
    // (addressed by glm_wr_chunk_idx), read chunk-indexed by the G12 GEMV's current
    // chunk request (array_chunk_idx_out, stage_sel=G12). G12's K=256 (16 chunks)
    // geometry is UNCHANGED — it now walks the 256-wide low_rank, not 256-of-3072.
    // The buffer's H=256, K_PAR=16 => NCHUNKS=16, CW=clog2(16)=4 = LR_OCW, so the
    // GEMV's wr_chunk_idx (group index, 4b) maps 1:1 to the buffer chunk index.
    wire [LR_OCW-1:0] lr_rd_idx = array_chunk_idx_out[LR_OCW-1:0];
    // WO-025 port-B: parallel ODD-chunk read pointer (G12 pair-walk).
    wire [LR_OCW-1:0] lr_rd_idx_b = array_chunk_idx_b_out[LR_OCW-1:0];
    // -----------------------------------------------------------------
    // pdi16 SKIP-OP GOLDEN-INGEST for GLM (stage 11 = the live bug).
    //   When inject_en[11] is set, REPLACE the GLM GEMV OUTPUT (glm_wr_chunk)
    //   that lands in u_low_rank_buf with a host-injected golden value, so a
    //   buggy GLM is skipped and G12/TREE/G8 still validate on the golden
    //   low_rank[256]. The golden 256-word low_rank lives flat in inj_bus
    //   (word = chunk_idx*16 + lane, Q6.10), sliced by the live glm_wr_chunk_idx.
    //   STAGE_INJECT=0 OR inject_en[11]=0 -> glm_wr_chunk_mux == glm_wr_chunk
    //   (BIT-IDENTICAL passthrough; inject_en/inj_bus tied 0 when STAGE_INJECT=0).
    //   Additive-only 2:1 mux at the EXISTING wr_chunk consumer — no new wide SLR
    //   net (memory feedback_banksplit_did_not_fix_plus128_nonuniform).
    // -----------------------------------------------------------------
    localparam int INJ_WIDX_W = $clog2(INJ_MAX_W);   // 8 (256 words)
    wire signed [LR_LANES*A_WIDTH-1:0] inj_glm_chunk;
    genvar gilm;
    generate
        for (gilm = 0; gilm < LR_LANES; gilm = gilm + 1) begin : g_inj_glm
            // word index in the flat 256-word golden buffer = chunk*16 + lane.
            // gilm is a genvar localparam constant in each instance; the chunk
            // term is the live glm_wr_chunk_idx. Build the index as a sized vector.
            wire [INJ_WIDX_W-1:0] gwi =
                (INJ_WIDX_W'(glm_wr_chunk_idx) * LR_LANES) + INJ_WIDX_W'(gilm);
            assign inj_glm_chunk[gilm*A_WIDTH +: A_WIDTH] =
                inj_bus[gwi*32 +: A_WIDTH];
        end
    endgenerate
    wire signed [LR_LANES*A_WIDTH-1:0] glm_wr_chunk_mux =
        inject_en[11] ? inj_glm_chunk : glm_wr_chunk;

    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(LR_M), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(0),
                      .FRAC_BITS(10), .RD_LAT(D1_RDLAT)) u_low_rank_buf (  // D1: G12 activation seam, registered
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(glm_wr_valid),
        .wr_chunk_idx(glm_wr_chunk_idx),
        .wr_chunk(glm_wr_chunk_mux),                          // pdi16: GLM skip-op golden-ingest mux
        .wr_clear(glm_start), .wr_count(), .full(),
        .rd_valid(1'b1), .rd_chunk_idx(lr_rd_idx), .rd_chunk(low_rank_rd),
        .rd_chunk_idx_b(lr_rd_idx_b), .rd_chunk_b(low_rank_rd_b)   // WO-025 port-B
    );

    // =================================================================
    // G8 SPLICE — fused-gather-dot frontier rescore (PREP, fork pdi6_s7c_g8).
    //   reports/G8_S7C_SPLICE_PREP_20260528.md
    //
    // PLACEMENT: between the G12-fed tree vocab-walk (S_TREE) and the OUTPUT-DMA.
    // g8_rescore's internal G8.1 selector snoops the SAME serial G12 tile stream
    // that feeds tree_decode (cheap top-K_C=512); on tree_done the FSM enters
    // S_G8 (top_fsm), g8_start pulses, g8_rescore gathers each candidate row from
    // HBM (m_axi_g8_*, lmfull_hbm_base) and fp16-dots it against the captured G11
    // final-norm to produce a RESCORED top-8 -> the output writer.
    //
    // STRAP-GATED: when G8_SPLICE=0 the WHOLE block degenerates to the s7c
    // behaviour (ow_* = tree top-K_F=8, m_axi_g8_* quiescent) so the DEFAULT
    // elaboration is byte-for-byte the s7c top. The generate guard means
    // g8_rescore is NOT instantiated at all when G8_SPLICE=0.
    //
    // TREE_P CONSTRAINT — LIFTED (TREE_P8-RESTORE 2026-07-09): the cheap-logit
    // snoop consumes the SERIAL per-logit stream (tt_valid_in/tt_score_in/
    // tt_index_in/tt_last_in). Historically that stream only existed in the
    // TREE_P<=1 (g_tree_serial) arm, so G8_SPLICE=1 forced TREE_P=1 (the
    // reason the shipping lineage ran the SLOW serial tree walk). The pkpar
    // arm now drives the SAME tt_* wires via tree_cheap_serializer (a bit/
    // order-identical serial replay of its P-wide stream), so BOTH arms feed
    // the snoop identically and TREE_P=8 is legal under G8_SPLICE=1.
    //
    // NUMERICAL-CONTRACT CAVEATS (documented build-gate items, NOT silent):
    //   (a) cheap_score: the on-chip serial tree stream carries the G12 logit as
    //       a SIGNED INT32 (the scaled-GEMV accumulator), whereas g8_rescore's
    //       G8.1 orders candidates with fp32_gt (native-fp32-bit-pattern). For
    //       the SELECTION ordering to be golden, the build must present the
    //       cheap logit as the SAME numeric form the standalone golden used
    //       (fp32 bit-pattern). Wired here VERBATIM (INT32 bits as the 32b key)
    //       so the SET is well-defined and the dataflow validates; the exact
    //       golden numeric cheap key is the turnkey build item (report §3a).
    //   (b) nrm operand (to_logits_norm): g8_rescore.G8.3 expects fp16; the
    //       on-chip G11 final-norm streams INT16 Q6.10 (rms_g11_out_chunk). The
    //       capture buffer below stores the norm at fp16 width; the INT16-Q6.10
    //       -> IEEE-754-fp16 conversion is isolated in `nrm_i16q10_to_fp16` (a
    //       single replaceable function) so the build drops in the real convert
    //       (report §3b). For the wiring/dataflow gate the bits flow end-to-end.
    //   (c) cheap_ready backpressure: the serial tree tiler streams one logit/
    //       cycle and CANNOT stall; g8_rescore's default OPT1B selector drops
    //       cheap_ready during its rare min-rescan. The golden build either
    //       (i) defines G8_OPT1C for the spliced instance (cheap_ready tied high,
    //       one-logit/cycle, the timing item the report flags) OR (ii) extends
    //       the tiler to honor a combined ready. Here the live stream is wired
    //       and the targeted TB drives cheap honoring ready (report §3c).
    // =================================================================
    // TREE_P8-RESTORE (2026-07-09): the historical guard
    //   "G8_SPLICE=1 requires TREE_P<=1" ($error here)
    // is LIFTED. The g_tree_pwide arm now re-materializes the serial cheap
    // stream via tree_cheap_serializer (bit/order-identical replay of the
    // P-wide stream), so the G8 snoop below consumes the SAME tt_* wires in
    // BOTH arms. The old caveat (the pkpar stream being internal-only) no
    // longer holds; frontier P-invariance (TREE_DECODE_FAST §2) is preserved
    // and additionally the G8 candidate SET is stream-order-invariant by
    // construction of the serializer.

    localparam int G8_K_F   = 8;
    localparam int G8_H     = 3072;
    localparam int G8_V_MAX = TREE_V_MAX;            // 128256
    localparam int G8_IDX_W = TREE_IDX_W;            // 18 (match tree/writer ABI)

    // PDI#7 debug taps surfaced OUT of the G8 generate block (generate-internal
    // wires are not visible at top scope, so the dbg_observe + 0x78-bit3 routing
    // read these top-scope nets). Driven in BOTH arms below (0 in the bypass).
    wire g8_dbg_err;       // G8 gather-watchdog sticky abort -> ctrl 0x78 bit3
    wire g8_dbg_activity;  // 1 = a non-zero G8 rescored score was produced
    // DEBUG-BLOCK-V2 (PDI#8) G8 progress mirrors (driven in BOTH arms below).
    wire [3:0]  g8_dbg_sel_substate;
    wire [9:0]  g8_dbg_cand_filled;
    wire [9:0]  g8_dbg_rows_gathered;
    wire        g8_dbg_seen_last;
    wire        g8_dbg_in_gather;
    wire [BOOT_AXI_ADDR_W-1:0] g8_dbg_last_araddr;   // 64-bit AR byte addr
    wire        g8_dbg_rresp_err;
    // PDI#10 PROBE: top-scope mirror of the G8 rescored frontier (K_F=8 x 32b
    // fp32 scores) + a 1-cycle done strobe, so probe stage-code 15 and the
    // per-stage output checksum can tap it. Driven in BOTH arms (0 in bypass).
    wire [8*32-1:0] g8_dbg_scores;     // K_F=8 fp32 rescored scores
    wire            g8_dbg_scores_v;   // 1-cyc when the rescored frontier is valid

    generate
    if (G8_SPLICE) begin : g_g8_splice
        // -------------------------------------------------------------
        // (b) to_logits_norm capture buffer — 3072 fp16 words.
        // Written from the G11 final-norm stream (rms_g11_out_chunk, K_PAR=16
        // INT16 Q6.10) on g11_out_strobe at g11_out_chunk_idx; read by
        // g8_rescore.nrm_rd_addr (element-addressed, 0..H-1). ONE always_ff
        // writer -> multi-driver-safe (memory feedback_multi_driver_packed_reg).
        // The buffer holds the norm as fp16 BITS; the INT16-Q6.10 -> fp16
        // convert is the single replaceable function below (build-gate item).
        // -------------------------------------------------------------
        // pdi21 G8RESC SCALAR-FIX FF-STAGING (2026-06-27, RCA
        // project_pdi20_g8resc_silicon_staging_rca): nrm_buf is the G8 dot b-operand
        // and was the ONE G11-norm consumer the SEAM_FF whole-design FF conversion
        // MISSED (it is a top-level capture buffer, not a stage_io_buffer instance).
        // pdi20 silicon: body G1..G12->TREE->G8CHEAP ALL golden, ONLY G8RESC diverged
        // (2/8). RCA surface (1) = the scalar dot reads nrm via nrm_rd_addr->nrm_rd_data
        // from THIS un-FF-staged ram_style="block" array: on silicon the BRAM inserts a
        // hidden output register -> the dot captures nrm STALE-BY-1 vs the gathered
        // a-operand (the same comb-BRAM stale class that bit GLM). FIX = ram_style=
        // "registers" -> FDRE register-file: a true combinational FF-mux read (NO hidden
        // BRAM output reg), so silicon == the xsim-golden combinational behavior, and
        // the read stays SAME-CYCLE aligned with rc_elem (the dot_b alignment canary in
        // g8_rescore_p enforces nrm_rd_addr==rc_elem). The sibling fork
        // pdi11_nrmbuf_banksplit OOC proves this array dissolves to FDRE (0 RAMB/RAM64M/
        // SRL); the pdi21 OOC census re-confirms it. ram_style is a synth attribute only
        // -> xsim-invariant (the scalar frontier gate stays 8/8 == 0x002c953c).
        // NOTE: G8_NRM_BANKSPLIT (pdi11_nrmbuf_banksplit) is a separate WIDE-path fix
        // for the 256b wdot_b operand REGISTER inside g8_rescore_p; it is INERT when
        // G8_WIDE=0 (the pdi21 scalar build) and does NOT cover this array, so this
        // ram_style="registers" conversion is the correct scalar-path nrm_buf fix.
        (* ram_style = "registers" *)
        logic [15:0] nrm_buf [0:G8_H-1];

        // INT16 Q6.10 -> IEEE-754 fp16 convert (the REAL convert; was a PREP
        // pass-through before reports/NRM_Q610_FP16_CONVERT_20260528.md).
        //
        // CONTRACT (verilog/forks/pdi6_g8/ref/g8_real_ref.py:137-139, authoritative):
        //   nrm_real = int16 / 2^10                       (Q6.10 -> real)
        //   nrm_fp16 = float32(nrm_real).astype(float16)  (IEEE-754 half, RNE)
        // i.e. signed INT16 / 1024 then round-to-nearest-EVEN to fp16.
        //
        // RANGE FACT: |code| <= 32768 so value in [-32, +31.999]; smallest
        // nonzero magnitude = 1/1024 ~ 9.77e-4 >> fp16 min-normal (6.1e-5).
        // => EVERY nonzero code maps to a NORMAL fp16. No subnormal/inf/NaN
        // path is reachable, so this is a pure normal-number fixed->float:
        //   leading-one detect -> biased exp = p+5 (e = p-10, bias 15);
        //   10 mantissa bits below the leading one, RNE on the dropped bits;
        //   mantissa-overflow (0x400) bumps the exponent.
        // Bit-exact vs numpy over ALL 65536 codes (mismatch=0, see report §2).
        // Combinational (single replaceable function); the always_ff below
        // registers the result into nrm_buf, so it is effectively 1-cycle.
        function automatic logic [15:0] nrm_i16q10_to_fp16;
            input logic signed [15:0] q610;
            logic               sign;
            logic [16:0]        mag;      // |code|, up to 32768 (=1<<15)
            integer             p;        // leading-one index (0..15)
            integer             shift;
            logic [10:0]        mant;     // 10-bit mantissa (+1 guard for overflow)
            logic               round_bit, sticky, roundup;
            logic [4:0]         exp_b;    // biased exponent
            begin
                sign = q610[15];
                // magnitude: 16-bit two's complement for negatives, zero-extended
                // to 17b; the lone 17-bit case is -32768 (q610==0x8000 -> 32768).
                if (q610 == 16'sh8000)
                    mag = 17'd32768;                       // |-32768| (only 17b mag)
                else if (sign)
                    mag = {1'b0, (~q610 + 16'sd1)};        // 16b two's comp, zero-ext
                else
                    mag = {1'b0, q610};
                if (mag == 17'd0) begin
                    nrm_i16q10_to_fp16 = 16'h0000;   // +0
                end else begin
                    // leading-one position p
                    p = 0;
                    for (int b = 16; b >= 0; b = b - 1)
                        if (p == 0 && mag[b]) p = b;
                    exp_b = 5'(p + 5);   // e = p-10, biased +15
                    if (p >= 10) begin
                        shift     = p - 10;
                        mant      = 11'((mag >> shift) & 17'h3FF);
                        round_bit = (shift == 0) ? 1'b0 : ((mag >> (shift-1)) & 17'd1);
                        sticky    = (shift <= 1) ? 1'b0
                                    : ((mag & ((17'd1 << (shift-1)) - 17'd1)) != 17'd0);
                    end else begin
                        shift     = 10 - p;
                        mant      = 11'((mag << shift) & 17'h3FF);
                        round_bit = 1'b0;
                        sticky    = 1'b0;
                    end
                    // round-to-nearest-even
                    roundup = round_bit & (sticky | mant[0]);
                    if (roundup) begin
                        mant = mant + 11'd1;
                        if (mant == 11'h400) begin   // mantissa carry-out
                            mant  = 11'd0;
                            exp_b = exp_b + 5'd1;     // bump exponent
                        end
                    end
                    nrm_i16q10_to_fp16 = {sign, exp_b, mant[9:0]};
                end
            end
        endfunction

        // ONE writer: capture each strobed chunk's 16 elements into nrm_buf.
        // chunk c covers elements [c*16 .. c*16+15].
        always_ff @(posedge ap_clk) begin
            if (g11_out_strobe) begin
                for (int e = 0; e < K_PAR; e = e + 1) begin
                    automatic int eidx = (int'(g11_out_chunk_idx) * K_PAR) + e;
                    if (eidx < G8_H)
                        nrm_buf[eidx] <=
                            nrm_i16q10_to_fp16(rms_g11_out_chunk[e*A_WIDTH +: A_WIDTH]);
                end
            end
        end

        // g8_rescore element-addressed nrm read port
        wire [$clog2(G8_H)-1:0] g8_nrm_rd_addr;
        wire [15:0]             g8_nrm_rd_data = nrm_buf[g8_nrm_rd_addr];

        // ---- G8_WIDE through-top WIDE nrm read port -------------------------
        //   The wide dot consumes WIDE_P fp16 b-operands per group; the b-operand
        //   is the captured G11 final-norm (nrm_buf), addressed at the SAME
        //   element base g8_nrm_rd_addr that the rescore_p wide-consume lane
        //   drives, with WIDE_P CONSECUTIVE elements [addr .. addr+WIDE_P-1].
        //   Combinational read (nrm_buf is already registered at write; the wide
        //   lane uses it at II=2 so no extra read-port timing pressure). Out-of-
        //   range elements (addr+w >= G8_H) zero-pad — inert at H=3072 (16|3072)
        //   but keeps the path defined at non-multiple H. When G8_WIDE=0 the
        //   port is tied to 256'b0 at the instance (no logic, no read port).
        wire [G8_WIDE_P*16-1:0] g8_nrm_rd_data_wide;
        genvar gw;
        for (gw = 0; gw < G8_WIDE_P; gw = gw + 1) begin : g_nrm_wide
            wire [$clog2(G8_H):0] gw_addr = {1'b0, g8_nrm_rd_addr} + gw;
            assign g8_nrm_rd_data_wide[gw*16 +: 16] =
                (gw_addr < G8_H) ? nrm_buf[gw_addr[$clog2(G8_H)-1:0]] : 16'h0000;
        end

        // -------------------------------------------------------------
        // (a) cheap-logit snoop — the SERIAL G12 tile stream that feeds
        // tree_decode (TREE_P=1 arm). cheap_score = the INT32 logit bits;
        // cheap_index = the 18-bit vocab id; cheap_last = the global last
        // logit (tt_last_in). cheap_valid = tt_valid_in. The selector's
        // backpressure (cheap_ready) is observed but the serial tiler cannot
        // stall — see caveat (c); validated via the targeted TB.
        //
        // CONCURRENT-LAUNCH (KEY ORDERING — report §finding): g8_rescore's
        // candidate selector (G8.1) must CONSUME the cheap stream WHILE it is
        // live, i.e. DURING the S_TREE vocab-walk. So g8_rescore.start is driven
        // from tree_start (NOT the top_fsm g8_start, which only pulses AFTER
        // tree_done — by then the cheap stream is over and G8.1 would see an
        // empty set / hang waiting for cheap_last). g8 selection therefore runs
        // CONCURRENTLY with tree_decode (both snoop the same stream); g8's gather
        // + dot + resolve then run after its own internal cheap_last, producing
        // g8_done. top_fsm's S_G8 phase WAITS for that g8_done (S_TREE holds
        // until tree_done, then S_G8 holds until g8_done; g8's gather outlasts
        // the tree drain, so S_G8 is the real wait). top_fsm.g8_start is left
        // wired (1-cycle S_G8-entry pulse) but UNUSED as the launch — kept so a
        // future design that buffers/replays the cheap stream can launch on it
        // instead. (See report §3 ordering finding + turnkey item.)
        // -------------------------------------------------------------
        wire        g8_cheap_ready;  // observed (caveat c)
        wire        g8_launch = tree_start;   // concurrent-with-tree selection

        // g8 frontier outputs
        wire [G8_K_F*G8_IDX_W-1:0]  g8_frontier_ids;
        wire [G8_K_F*32-1:0]        g8_frontier_scores;
        wire                        g8_done_w;
        wire                        g8_err_w;   // PDI#7: gather-watchdog sticky abort

        if (G8_PIPELINED) begin : g_g8p
            // ====================================================================
            // PDI#9 MAINLINE: PIPELINED gather (g8_rescore_p).  Same ports as the
            // serial g8_rescore (incl. the DEBUG-V2/PROBE-V3 read-only taps, ported
            // in this fork) PLUS the P2 prefetch hooks (tied off here -> functionally
            // pure P1, bit-exact to the serial frontier).  Carries seen_last + the
            // all-states watchdog (reconciled with the 7.2 serial fixes).
            // ====================================================================
            wire        g8p_pf_drop;
            wire [G8_IDX_W:0] g8p_pf_cnt;

            // ---- SPEEDWAVE-V2 J5-β G8_NCH plumbing (ported from jewel
            //   glm_tree_g8_backend.sv:602-686 — the jewel backend is this
            //   monolith backend's descendant, pattern carried back verbatim).
            //   Legacy master nets (u_g8's single-channel engine; the REAL
            //   m_axi_g8_* driver at G8_NCH=1, idle at G8_NCH>1) + stripe
            //   channel nets. ch0 -> m_axi_g8_* mux; ch1 -> m_axi_g8b_*.
            wire                       g8l_arvalid;
            wire [BOOT_AXI_ADDR_W-1:0] g8l_araddr;
            wire [7:0]                 g8l_arlen;
            wire [2:0]                 g8l_arsize;
            wire [1:0]                 g8l_arburst;
            wire [BOOT_AXI_ID_W-1:0]   g8l_arid;
            wire                       g8l_rready;

            wire [G8_NCH-1:0]          st_arvalid;
            wire [G8_NCH-1:0]          st_arready;
            wire [BOOT_AXI_ADDR_W-1:0] st_araddr  [0:G8_NCH-1];
            wire [7:0]                 st_arlen   [0:G8_NCH-1];
            wire [2:0]                 st_arsize  [0:G8_NCH-1];
            wire [1:0]                 st_arburst [0:G8_NCH-1];
            wire [G8_NCH-1:0]          st_rvalid;
            wire [G8_NCH-1:0]          st_rready;
            wire [G8_DATA_W-1:0]       st_rdata   [0:G8_NCH-1];
            wire [G8_NCH-1:0]          st_rlast;
            wire [$clog2(G8_H)-1:0]    st_nrm_addr [0:G8_NCH-1];
            wire [G8_WIDE_P*16-1:0]    st_nrm_wide [0:G8_NCH-1];

            if (G8_NCH > 1) begin : g_nch_mux
                // ch0 rides the EXISTING m_axi_g8 port (S05_AXI/HBM3_PORT0)
                assign st_arready[0] = m_axi_g8_arready;
                assign st_rvalid [0] = m_axi_g8_rvalid;
                assign st_rdata  [0] = m_axi_g8_rdata;
                assign st_rlast  [0] = m_axi_g8_rlast;
                // ch1 rides the NEW m_axi_g8b port (S07_AXI/HBM3_PORT1)
                assign st_arready[1] = m_axi_g8b_arready;
                assign st_rvalid [1] = m_axi_g8b_rvalid;
                assign st_rdata  [1] = m_axi_g8b_rdata;
                assign st_rlast  [1] = m_axi_g8b_rlast;
                assign m_axi_g8b_arvalid = st_arvalid[1];
                assign m_axi_g8b_araddr  = st_araddr [1];
                assign m_axi_g8b_arlen   = st_arlen  [1];
                assign m_axi_g8b_arsize  = st_arsize [1];
                assign m_axi_g8b_arburst = st_arburst[1];
                assign m_axi_g8b_arid    = {BOOT_AXI_ID_W{1'b0}};  // one AXI ID per channel
                assign m_axi_g8b_rready  = st_rready [1];
            end else begin : g_nch_off
                // stripe arrays inert; m_axi_g8b tied off (BD drop-in shape fixed)
                assign st_arready = '0;  assign st_rvalid = '0;  assign st_rlast = '0;
                assign st_rdata[0] = '0;
                assign m_axi_g8b_arvalid = 1'b0;  assign m_axi_g8b_araddr = '0;
                assign m_axi_g8b_arlen = 8'd0;    assign m_axi_g8b_arsize = 3'd0;
                assign m_axi_g8b_arburst = 2'b01; assign m_axi_g8b_arid = {BOOT_AXI_ID_W{1'b0}};
                assign m_axi_g8b_rready = 1'b0;
            end

`ifndef SYNTHESIS
            initial if (G8_NCH > 2)
                $fatal(1, "eagle4_drafter: G8_NCH=%0d — only N<=2 is plumbed to top-level masters", G8_NCH);
`endif

            // m_axi_g8 output mux: stripe ch0 owns the port at G8_NCH>1
            assign m_axi_g8_arvalid = (G8_NCH > 1) ? st_arvalid[0] : g8l_arvalid;
            assign m_axi_g8_araddr  = (G8_NCH > 1) ? st_araddr [0] : g8l_araddr;
            assign m_axi_g8_arlen   = (G8_NCH > 1) ? st_arlen  [0] : g8l_arlen;
            assign m_axi_g8_arsize  = (G8_NCH > 1) ? st_arsize [0] : g8l_arsize;
            assign m_axi_g8_arburst = (G8_NCH > 1) ? st_arburst[0] : g8l_arburst;
            assign m_axi_g8_arid    = (G8_NCH > 1) ? {BOOT_AXI_ID_W{1'b0}} : g8l_arid;
            assign m_axi_g8_rready  = (G8_NCH > 1) ? st_rready [0] : g8l_rready;

            // per-channel WIDE nrm fanout from the FF nrm_buf (comb, same form
            // as the g_nrm_wide fanout above; FF register-file -> no BRAM seam)
            for (genvar gnc = 0; gnc < G8_NCH; gnc++) begin : g_st_nrm
                for (genvar gnw = 0; gnw < G8_WIDE_P; gnw++) begin : g_w
                    wire [$clog2(G8_H):0] a = {1'b0, st_nrm_addr[gnc]} + gnw;
                    assign st_nrm_wide[gnc][gnw*16 +: 16] =
                        (a < G8_H) ? nrm_buf[a[$clog2(G8_H)-1:0]] : 16'h0000;
                end
            end

            g8_rescore_p #(
                .V_MAX(G8_V_MAX), .H(G8_H), .K_C(G8_K_C), .K_F(G8_K_F),
                .IDX_W(G8_IDX_W), .SCORE_W(32),
                .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(G8_DATA_W),
                .AXI_ID_W(BOOT_AXI_ID_W), .MAX_BURST(G8_MAX_BURST),
                .G8_OUTSTANDING(G8_OUTSTANDING),
                .G8_PREFETCH(G8_PREFETCH),
                .G8_OPT1C(G8_OPT1C),            // PDI#10 SEL_SCAN fix (default ON)
                .G8_OPT1D(G8_OPT1D),           // PDI#10 timing repro variant
                .G8_MIN_NG(G8_MIN_NG),
                .G8_OPT1F(G8_OPT1F),           // PDI#10 100MHz fix: pipelined reg min
                .G8_SELPIPE(G8_SELPIPE),       // SEL_PIPE: registered-cmin retiming (100MHz in-context)
                .G8_OPT1G(G8_OPT1G),          // GRAND: item4 systolic-sorter selector (default ON)
                .G8_WIDE(G8_WIDE),            // GRAND: 512b wide-consume dot (default OFF, BD-gated)
                .WIDE_P(G8_WIDE_P),
                .WIDE_IL(G8_WIDE_IL),         // BEST-SPEED: II=1 interleaved-acc wide dot (default ON)
                .WIDE_M(G8_WIDE_M),           // BEST-SPEED: II=1 interleave depth (=4)
                .MROUND_P(G8_MROUND),         // GRAND: scalar-dot fp16-mul round-split (default ON)
                .SCALAR_II1(G8_SCALAR_II1),   // pdi21: scalar dot II=1 (FADD_PIPE=0 + feed-every-cycle)
                .G8_CONSUME_HARDEN_P(G8_CONSUME_HARDEN), // FIX-C: matched-delay owned-nrm consume (default 0)
                .G8_LANE_FIX(G8_LANE_FIX),    // pdi24 G0: scalar narrow-burst lane-select fix (default 1)
                .G8_ROWPP(G8_ROWPP),          // EXP_G8_PINGPONG_II1: decoupled fetch/feed + owner FIFO + ROW_OVERLAP dot snapshot (default 0)
                .G8_NCH(G8_NCH)               // SPEEDWAVE-V2 J5-β: N-channel striped gather (default 1 = legacy)
            ) u_g8 (
                .clk(ap_clk), .rst_n(ap_rst_n),
                .start(g8_launch),                // CONCURRENT launch @ tree_start
                .kc_cfg(kc_cfg),                  // SPEEDWAVE-V2 K1 (ctrl 0x150)
                .cheap_valid(tt_valid_in),
                .cheap_ready(g8_cheap_ready),
                .cheap_score(tt_score_in),
                .cheap_index(tt_index_in),
                .cheap_last(tt_last_in),
                .nrm_rd_data(g8_nrm_rd_data),
                // G8_WIDE through-top wide nrm b-operand: WIDE_P consecutive fp16
                // from nrm_buf at g8_nrm_rd_addr. Width = WIDE_P*FP16_W = 16*16 =
                // 256. The wide consume lane uses this when G8_WIDE=1; the synth
                // optimizer prunes g_nrm_wide when G8_WIDE=0 (port unread).
                .nrm_rd_data_wide(g8_nrm_rd_data_wide),
                .nrm_rd_addr(g8_nrm_rd_addr),
                // legacy single-channel master: REAL port at G8_NCH=1, parked
                // (inputs gated to 0, outputs to internal nets) at G8_NCH>1 —
                // the stripe frontend owns both physical ports then.
                .m_axi_arvalid(g8l_arvalid), .m_axi_arready((G8_NCH > 1) ? 1'b0 : m_axi_g8_arready),
                .m_axi_araddr(g8l_araddr),   .m_axi_arlen(g8l_arlen),
                .m_axi_arsize(g8l_arsize),   .m_axi_arburst(g8l_arburst),
                .m_axi_arid(g8l_arid),
                .m_axi_rvalid((G8_NCH > 1) ? 1'b0 : m_axi_g8_rvalid), .m_axi_rready(g8l_rready),
                .m_axi_rdata(m_axi_g8_rdata),  .m_axi_rlast((G8_NCH > 1) ? 1'b0 : m_axi_g8_rlast),
                .m_axi_rid(m_axi_g8_rid),      .m_axi_rresp(m_axi_g8_rresp),
                // SPEEDWAVE-V2 J5-β stripe channel ports (inert at G8_NCH=1)
                .stripe_arvalid(st_arvalid), .stripe_arready(st_arready),
                .stripe_araddr(st_araddr),   .stripe_arlen(st_arlen),
                .stripe_arsize(st_arsize),   .stripe_arburst(st_arburst),
                .stripe_rvalid(st_rvalid),   .stripe_rready(st_rready),
                .stripe_rdata(st_rdata),     .stripe_rlast(st_rlast),
                .stripe_nrm_addr(st_nrm_addr), .stripe_nrm_wide(st_nrm_wide),
                .lmfull_hbm_base(lmfull_hbm_base),
                // P2 prefetch hooks: tied off (no snoop wired this round)
                .prefetch_req(1'b0), .prefetch_req_id('0),
                .prefetch_evict(1'b0), .prefetch_evict_id('0),
                .prefetch_drop(g8p_pf_drop), .prefetch_count(g8p_pf_cnt),
                .frontier_ids(g8_frontier_ids),
                .frontier_scores(g8_frontier_scores),
                .done(g8_done_w),
                .g8_err(g8_err_w),
                // DEBUG-BLOCK-V2 (PDI#8) read-only progress taps (ported into P)
                .dbg_sel_substate(g8_dbg_sel_substate),
                .dbg_cand_filled(g8_dbg_cand_filled),
                .dbg_rows_gathered(g8_dbg_rows_gathered),
                .dbg_seen_last(g8_dbg_seen_last),
                .dbg_in_gather(g8_dbg_in_gather),
                .dbg_last_araddr(g8_dbg_last_araddr),
                .dbg_rresp_err(g8_dbg_rresp_err),
                // PROBE V3 out3 candidate-set dump read port (ASYNC debug tap):
                .dbg_cand_rdaddr(g8_cand_rdaddr),
                .dbg_cand_id(g8_cand_dump_id),
                .dbg_cand_cheap_score(g8_cand_dump_score),
                // SEAM-FF: dedicated REGISTERED bridge score-feed (kills 0x1F tied-zero)
                .bridge_rdaddr_nxt(bridge_cand_rdaddr_nxt),
                .bridge_score_ff(g8_bridge_score_ff),
                // TREE_P8-RESTORE: selection-threshold export -> pkpar pre-filter
                .sel_thresh(g8_sel_thresh),
                .sel_thresh_valid(g8_sel_thresh_valid)
            );
        end else begin : g_g8s
            // SPEEDWAVE-V2: serial arm has no stripe support — m_axi_g8b inert.
            assign m_axi_g8b_arvalid = 1'b0;  assign m_axi_g8b_araddr = '0;
            assign m_axi_g8b_arlen = 8'd0;    assign m_axi_g8b_arsize = 3'd0;
            assign m_axi_g8b_arburst = 2'b01; assign m_axi_g8b_arid = {BOOT_AXI_ID_W{1'b0}};
            assign m_axi_g8b_rready = 1'b0;
`ifndef SYNTHESIS
            initial if (G8_NCH > 1)
                $fatal(1, "eagle4_drafter: G8_NCH>1 requires G8_PIPELINED=1 (stripe frontend lives in g8_rescore_p)");
`endif
            // TREE_P>1 needs the pipelined g8's threshold export (the serial
            // g8_rescore has no sel_thresh port — pre-filter would be inert
            // and the serializer FIFO rate-infeasible).
            // synthesis translate_off
            initial begin
                if (TREE_P > 1)
                    $error("eagle4_drafter: G8_SPLICE=1 && TREE_P>1 requires G8_PIPELINED=1 (sel_thresh pre-filter export). Got TREE_P=%0d G8_PIPELINED=0.", TREE_P);
            end
            // synthesis translate_on
            assign g8_sel_thresh       = 32'h0;
            assign g8_sel_thresh_valid = 1'b0;
        g8_rescore #(
            .V_MAX(G8_V_MAX), .H(G8_H), .K_C(G8_K_C), .K_F(G8_K_F),
            .IDX_W(G8_IDX_W), .SCORE_W(32),
            .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(BOOT_AXI_DATA_W),
            .AXI_ID_W(BOOT_AXI_ID_W), .MAX_BURST(G8_MAX_BURST)
        ) u_g8 (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(g8_launch),                    // CONCURRENT launch @ tree_start
            // G8.1 cheap-logit snoop (serial tree stream)
            .cheap_valid(tt_valid_in),
            .cheap_ready(g8_cheap_ready),
            .cheap_score(tt_score_in),            // INT32 bits (caveat a)
            .cheap_index(tt_index_in),
            .cheap_last(tt_last_in),
            // G8.3 second operand: captured G11 final-norm
            .nrm_rd_data(g8_nrm_rd_data),
            .nrm_rd_addr(g8_nrm_rd_addr),
            // G8.2 HBM gather read master -> top m_axi_g8_*
            .m_axi_arvalid(m_axi_g8_arvalid), .m_axi_arready(m_axi_g8_arready),
            .m_axi_araddr(m_axi_g8_araddr),   .m_axi_arlen(m_axi_g8_arlen),
            .m_axi_arsize(m_axi_g8_arsize),   .m_axi_arburst(m_axi_g8_arburst),
            .m_axi_arid(m_axi_g8_arid),
            .m_axi_rvalid(m_axi_g8_rvalid),   .m_axi_rready(m_axi_g8_rready),
            .m_axi_rdata(m_axi_g8_rdata),     .m_axi_rlast(m_axi_g8_rlast),
            .m_axi_rid(m_axi_g8_rid),         .m_axi_rresp(m_axi_g8_rresp),
            .lmfull_hbm_base(lmfull_hbm_base),
            // output
            .frontier_ids(g8_frontier_ids),
            .frontier_scores(g8_frontier_scores),
            .done(g8_done_w),
            .g8_err(g8_err_w),              // PDI#7: gather-watchdog sticky abort
            // DEBUG-BLOCK-V2 (PDI#8) read-only progress taps
            .dbg_sel_substate(g8_dbg_sel_substate),
            .dbg_cand_filled(g8_dbg_cand_filled),
            .dbg_rows_gathered(g8_dbg_rows_gathered),
            .dbg_seen_last(g8_dbg_seen_last),
            .dbg_in_gather(g8_dbg_in_gather),
            .dbg_last_araddr(g8_dbg_last_araddr),
            .dbg_rresp_err(g8_dbg_rresp_err),
            // PROBE V3 out3 candidate-set dump read port:
            .dbg_cand_rdaddr(g8_cand_rdaddr),
            .dbg_cand_id(g8_cand_dump_id),
            .dbg_cand_cheap_score(g8_cand_dump_score)
        );
        // SEAM-FF: the SERIAL g8_rescore has no registered bridge port (and the
        // bridge/TREE_P>1 path is illegal on it — see the $error above), so mirror
        // the async score here to keep g8_bridge_score_ff driven in this arm.
        assign g8_bridge_score_ff = g8_cand_dump_score;
        end

        // PDI#7 debug taps: surface the watchdog flag + a sticky non-zero-score
        // activity flag to top scope. g8_activity sets when ANY rescored score
        // bus is non-zero on the done pulse (a clean "G8 produced output" signal
        // for the 0x84 activity word). Both cleared at the next launch.
        reg g8_act_sticky;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)      g8_act_sticky <= 1'b0;
            else if (g8_launch) g8_act_sticky <= 1'b0;
            else if (g8_done_w && (|g8_frontier_scores)) g8_act_sticky <= 1'b1;
        end
        assign g8_dbg_err      = g8_err_w;
        assign g8_dbg_activity = g8_act_sticky;
        // PDI#10 PROBE: top-scope mirror of the rescored frontier scores + done
        // pulse (probe stage-code 15 + the per-stage output checksum). Read-only.
        assign g8_dbg_scores   = g8_frontier_scores;
        assign g8_dbg_scores_v = g8_done_w;

        // STICKY g8_done latch: g8 launches CONCURRENTLY at tree_start, so its
        // done PULSE (g8_done_w) can fire EITHER during S_TREE (if the gather is
        // shorter than the tree drain — e.g. a reduced-K_C bring-up) OR during
        // S_G8 (the realistic K_C=512 gather, which outlasts the drain). Latch
        // it so top_fsm's S_G8 phase observes a level regardless of when the
        // pulse landed; cleared at the next launch (tree_start) / reset.
        logic g8_done_sticky;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)        g8_done_sticky <= 1'b0;
            else if (g8_launch)   g8_done_sticky <= 1'b0;   // new run clears it
            else if (g8_done_w)   g8_done_sticky <= 1'b1;   // capture the pulse
        end
        assign g8_done = g8_done_sticky;

        // Output writer source = the G8 RESCORED top-8. Trigger on the done
        // PULSE (g8_done_w) so the writer's tree_done-style edge fires exactly
        // once when g8 finishes (the writer latches the frontier on that edge).
        assign ow_trigger     = g8_done_w;
        assign ow_topk_scores  = g8_frontier_scores;   // K_F=8 x 32b fp32 bit-pattern
        assign ow_topk_indices = g8_frontier_ids;      // K_F=8 x 18b vocab id

    end else begin : g_g8_bypass
        // SPEEDWAVE-V2: bypass arm — m_axi_g8b inert (mirrors m_axi_g8 ties).
        assign m_axi_g8b_arvalid = 1'b0;  assign m_axi_g8b_araddr = '0;
        assign m_axi_g8b_arlen = 8'd0;    assign m_axi_g8b_arsize = 3'd1;
        assign m_axi_g8b_arburst = 2'b01; assign m_axi_g8b_arid = '0;
        assign m_axi_g8b_rready = 1'b0;
        // G8_SPLICE=0 : s7c-exact. Output writer = tree top-K_F=8 on tree_done.
        // m_axi_g8_* tie off quiescent; g8_done tied 0 (S_G8 unreachable).
        assign g8_sel_thresh       = 32'h0;   // TREE_P8-RESTORE: no g8 -> no filter
        assign g8_sel_thresh_valid = 1'b0;
        assign g8_done           = 1'b0;
        assign m_axi_g8_arvalid  = 1'b0;
        assign m_axi_g8_araddr   = '0;
        assign m_axi_g8_arlen    = '0;
        assign m_axi_g8_arsize   = 3'b001;
        assign m_axi_g8_arburst  = 2'b01;
        assign m_axi_g8_arid     = '0;
        assign m_axi_g8_rready   = 1'b0;
        assign ow_trigger        = tree_done;
        assign ow_topk_scores    = tree_topk_scores;
        assign ow_topk_indices   = tree_topk_indices;
        // PDI#7 debug taps tie off in the bypass (no G8 instance).
        assign g8_dbg_err        = 1'b0;
        assign g8_dbg_activity   = 1'b0;
        // PDI#10 PROBE: G8 frontier-score mirror ties off in the bypass.
        assign g8_dbg_scores     = '0;
        assign g8_dbg_scores_v   = 1'b0;
        // DEBUG-BLOCK-V2 (PDI#8) G8 progress mirrors tie off in the bypass.
        assign g8_dbg_sel_substate  = 4'd0;
        assign g8_dbg_cand_filled   = 10'd0;
        assign g8_dbg_rows_gathered = 10'd0;
        assign g8_dbg_seen_last     = 1'b0;
        assign g8_dbg_in_gather     = 1'b0;
        assign g8_dbg_last_araddr   = '0;
        assign g8_dbg_rresp_err     = 1'b0;
        // PROBE V3 out3 candidate dump ties off in the bypass (no G8 instance).
        assign g8_cand_dump_id      = '0;
        assign g8_cand_dump_score   = '0;
        // SEAM-FF: bridge score-feed tie-off in the bypass (no G8 instance).
        assign g8_bridge_score_ff   = '0;
    end
    endgenerate

    // ----------------------------------------------------------------
    // PDI#7 SILICON-HARDENING: debug-visibility observer (0x80 FSM-state /
    // 0x84 per-stage non-zero activity). PURELY ADDITIVE read-only taps;
    // no output feeds the datapath -> golden frontier BIT-IDENTICAL.
    // ----------------------------------------------------------------
    wire [31:0] dbg_fsmstate_word;
    wire [31:0] dbg_activity_word;
    // DEBUG-BLOCK-V2 (PDI#8) word nets + select.
    wire [3:0]  dbg_sel_w;
    wire [31:0] dbg_cyclecount_word, dbg_checksum_word, dbg_g8progress_word;
    wire [31:0] dbg_g8araddr_lo_word, dbg_g8araddr_hi_word;
    wire [31:0] dbg_rearm_word, dbg_bootstream_word;
    // PDI#10 0x90-FIX + content-sample word nets (driven by u_dbg).
    wire [31:0] dbg_stageout_xor_word, dbg_stageout_sample_word, dbg_stageout_count_word;
    // PDI#10 PER-STAGE VALUE READBACK (0xDC/0xE0/0xE4/0xE8) control + read-back nets.
    wire [4:0]  dbg_cap_stage_sel_w;    // 0xDC  (write-only stage select for read mux)
    wire        dbg_cap_widx_we;        // 1-cyc pulse on 0xE0 write (seek word idx)
    wire [4:0]  dbg_cap_widx_wdata;     // 0xE0 seek value (PROBE_EXPANSION: 5b for NW=32)
    wire        dbg_cap_val_re;         // 1-cyc pulse on 0xE4 read (auto-inc idx)
    wire [31:0] dbg_cap_widx_word;      // 0xE0 read-back (current word idx)
    wire [31:0] dbg_cap_val_word;       // 0xE4 cap[stage_sel][word_idx]
    wire [31:0] dbg_cap_meta_word;      // 0xE8 {captured, nwords} for selected stage
    wire [31:0] dbg_cap_crc_word;       // 0x108 MAXPROBE per-stage CRC of selected stage
    // PDI#11 PROBE_EXPANSION: BOOT-LOAD RESIDENCY CRC (0xEC/0xF0/0xF4/0xF8) nets.
    wire [2:0]  dbg_res_sel_w;          // 0xEC buffer select (0..4)
    wire        dbg_res_clear_w;        // 0xEC bit[3] residency clear pulse
    wire [31:0] dbg_res_crc_word;       // 0xF0 CRC signature
    wire [31:0] dbg_res_sum_word;       // 0xF4 running SUM
    wire [31:0] dbg_res_count_word;     // 0xF8 word count
    // F3-CLR: KV-RESIDENCY SESSION CLEAR (0x154 bit[0]) — NOT the 0xEC
    // boot-load-CRC clear above. 0xEC/dbg_res_* clears the EMB/HFC/GAMMA/
    // WEIGHT/SCALE load accumulators; THIS clears the attention block's
    // per-slot kv_resident bitmap (the cross-request contamination fix).
    wire        kv_res_clear_w;         // 0x154 bit[0] 1-cyc residency clear pulse
    wire [31:0] kv_res_status_word =    // 0x154 read-back: canary + live count
                {16'h4B56 /* "KV" */, 4'b0, s7a_kv_res_cnt};
    // PDI#10 0x90-FIX: forward decls of the dbg_sel-muxed stage-output feed
    // (the always_comb mux + folds are defined below, near the probe mux).
    localparam int PRB_MAXCH_FWD = LANES;
    reg  [PRB_MAXCH_FWD*32-1:0]            sout_data;
    reg  [$clog2(PRB_MAXCH_FWD+1)-1:0]     sout_words;
    reg                                    sout_valid;
    reg  [31:0]                            sout_xor_fold;
    wire [31:0]                            sout_word0 = sout_data[31:0];
    // PROBE V3 (0xA8..0xBC) control + read-back nets (driven by u_probe / u_ctrl).
    wire [3:0]  prb_sel_w;
    wire        prb_sel_we, prb_addr_we, prb_data_re;
    wire [9:0]  prb_addr_wdata;
    wire [31:0] prb_addr_word, prb_data_word, prb_xor_word, prb_sum_word, prb_count_word;
    // gamma write-strobe OR (any of the 4 gamma buffers) for the 0x90 checksum.
    wire        dbg_gamma_we = gamma_we_hidden | gamma_we_enorm |
                               gamma_we_postat | gamma_we_final;
    dbg_observe #(.LANES(LANES), .A_WIDTH(A_WIDTH), .G8_ADDR_W(BOOT_AXI_ADDR_W)) u_dbg (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start_pulse(start_pulse),
        .top_state(5'(u_fsm.state)),      // hierarchical read-only tap (5b enum)
        .g5_state(3'(g5_state)),          // attention sub-FSM (3b enum)
        .stage_sel(stage_sel),
        .g1_done(g1_done),   .g4_done(g4_done),   .g5_done(g5_done),
        .g6_done(g6_done_stage), .g7g_done(g7g_done_stage), .g7u_done(g7u_done_stage),
        .g7p_done(g7p_done), .g9_done(g9_done_stage),
        .g11_done(g11_done), .glm_done(glm_done), .g12_done(g12_done),
        .tree_done(tree_done), .g8_done(g8_done),
        // output-data taps + valid strobes
        .g1_out_chunk(rms_g1_out_chunk_w), .g1_out_valid(rms_g1_out_valid),
        .array_acc_out(array_acc_out),     .array_done(array_done),
        .g5_attn_out_vec(dbg_g5_attn_vec), .g5_attn_done(dbg_g5_attn_done),
        .silu_out_chunk(silu_out_chunk),   .silu_valid(silu_valid),
        .g11_out_chunk(rms_g11_out_chunk), .g11_out_strobe(g11_out_strobe),
        .tree_topk_scores(tree_topk_scores),
        .g8_activity(g8_dbg_activity),     .g8_wd_tripped(g8_dbg_err),
        .fsmstate_word(dbg_fsmstate_word),
        .activity_word(dbg_activity_word),
        // ---- DEBUG BLOCK V2 (PDI#8) ----
        .kernel_busy(kernel_busy), .kernel_done(kernel_done),
        .dbg_sel(dbg_sel_w),
        .emb_we(emb_we),     .emb_wr_data(emb_wr_data32),
        .hfc_we(hfc_we),     .hfc_wr_data(hfc_wr_data32),
        .gamma_we(dbg_gamma_we), .gamma_wr_data(gamma_wr_data32),
        .g8_sel_substate(g8_dbg_sel_substate),
        .g8_cand_filled(g8_dbg_cand_filled),
        .g8_rows_gathered(g8_dbg_rows_gathered),
        .g8_seen_last(g8_dbg_seen_last),
        .g8_in_gather(g8_dbg_in_gather),
        .g8_last_araddr(g8_dbg_last_araddr),
        .g8_rresp_err(g8_dbg_rresp_err),
        .boot_stream_we(dma_we), .boot_stream_data(dma_wr_data),
        .cyclecount_word(dbg_cyclecount_word),
        .checksum_word(dbg_checksum_word),
        .g8progress_word(dbg_g8progress_word),
        .g8araddr_lo_word(dbg_g8araddr_lo_word),
        .g8araddr_hi_word(dbg_g8araddr_hi_word),
        .rearm_word(dbg_rearm_word),
        .bootstream_word(dbg_bootstream_word),
        // PDI#10 0x90-FIX + content sample (stage-output checksum, dbg_sel-muxed)
        .sout_chunk_valid(sout_valid),
        .sout_chunk_w0(sout_word0),
        .sout_chunk_xor(sout_xor_fold),
        .sout_chunk_nwords({{(10-$clog2(PRB_MAXCH_FWD+1)){1'b0}}, sout_words}),
        .stageout_xor_word(dbg_stageout_xor_word),
        .stageout_sample_word(dbg_stageout_sample_word),
        .stageout_count_word(dbg_stageout_count_word)
    );

    // ================================================================
    // PROBE ARCHITECTURE V3 — intermediate-layer CONTENT probe (reg 0xA8..0xBC).
    // A host-selectable 1024x32 capture RAM snapshots the FIRST 256 output words
    // of the selected stage plus a whole-stream XOR/sum32/count checksum. PURELY
    // ADDITIVE: every tap is read-only; nothing feeds the datapath -> golden
    // frontier BIT-IDENTICAL. Allocated above dbg v1 (0x74-0x84) and v2
    // (0x88-0xA4). See reports/eagle4_drafter_m7/PROBE_V3.md for the reg map.
    // ----------------------------------------------------------------
    localparam int PRB_MAXCH = LANES;              // 128 = widest chunk (GEMV tile)

    // ---- per-stage chunk + valid mux (probe_sel selects the tapped bus) ----
    // 16b buses (256b = 16 INT16) pack 2/word -> 8 words; the 32b GEMV tile is
    // 128 words; g5 attention is the FIXED 16x32b = 16 words; tree is 8x32b.
    // The selected bus is zero-extended to PRB_MAXCH*32 and a word count emitted.
    reg  [PRB_MAXCH*32-1:0]            prb_chunk_data;
    reg  [$clog2(PRB_MAXCH+1)-1:0]     prb_chunk_words;
    reg                               prb_chunk_valid;

    // helper: a 256b (16xINT16) bus -> 8 packed 32b words {hi=lane2k+1, lo=lane2k}
    function automatic [PRB_MAXCH*32-1:0] pack16x16;
        input [16*16-1:0] b;
        integer k;
        begin
            pack16x16 = '0;
            for (k = 0; k < 8; k = k + 1)
                pack16x16[k*32 +: 32] = { b[(2*k+1)*16 +: 16], b[(2*k)*16 +: 16] };
        end
    endfunction
    // helper: a 16x32b bus (g5 attention, FIXED width) -> 16 words verbatim.
    function automatic [PRB_MAXCH*32-1:0] pack16x32;
        input [16*32-1:0] b;
        integer k;
        begin
            pack16x32 = '0;
            for (k = 0; k < 16; k = k + 1) pack16x32[k*32 +: 32] = b[k*32 +: 32];
        end
    endfunction
    // helper: an 8x32b bus (tree scores) -> 8 words verbatim.
    function automatic [PRB_MAXCH*32-1:0] pack8x32;
        input [8*32-1:0] b;
        integer k;
        begin
            pack8x32 = '0;
            for (k = 0; k < 8; k = k + 1) pack8x32[k*32 +: 32] = b[k*32 +: 32];
        end
    endfunction

    // array_acc_out belongs to the current stage_sel; its valid for a given
    // probe stage = array_done while stage_sel matches that GEMV stage.
    //   stage_sel: G4=0, G6=1, G7g=2, G7u=3, G9=4, G12=5.
    always_comb begin
        prb_chunk_data  = '0;
        prb_chunk_words = '0;
        prb_chunk_valid = 1'b0;
        case (prb_sel_w)
            4'd0: begin // G1 in-RMSNorm (16x16b)
                prb_chunk_data  = pack16x16(rms_g1_out_chunk_w);
                prb_chunk_words = 8;  prb_chunk_valid = rms_g1_out_valid;
            end
            4'd1: begin // G4 q/k/v GEMV (128x32b @ stage_sel==0)
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd0);
            end
            4'd2: begin // G5 attention out (FIXED 16x32b)
                prb_chunk_data  = pack16x32(dbg_g5_attn_vec);
                prb_chunk_words = 16; prb_chunk_valid = dbg_g5_attn_done;
            end
            4'd3: begin // G6 o-proj (stage_sel==1)
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd1);
            end
            4'd4: begin // G7g gate (stage_sel==2)
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd2);
            end
            4'd5: begin // G7u up (stage_sel==3)
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd3);
            end
            4'd6: begin // G7p SiLU*up (16x16b)
                prb_chunk_data  = pack16x16(silu_out_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = silu_valid;
            end
            4'd7: begin // G9 down (stage_sel==4)
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd4);
            end
            4'd8: begin // G11 final-RMSNorm (16x16b)
                prb_chunk_data  = pack16x16(rms_g11_out_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = g11_out_strobe;
            end
            4'd9: begin // GLM lmdown (LR_LANES=16 x16b = 256b)
                prb_chunk_data  = pack16x16(glm_wr_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = glm_wr_valid;
            end
            4'd10: begin // G12 logits (stage_sel==5)
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd5);
            end
            4'd11: begin // TREE top-K scores (8x32b)
                prb_chunk_data  = pack8x32(tree_topk_scores);
                prb_chunk_words = 8;  prb_chunk_valid = tree_done;
            end
            4'd12: begin // R1 residual seam (G1->G4, 16x16b read mux)
                prb_chunk_data  = pack16x16(seam_r1_rd);
                prb_chunk_words = 8;  prb_chunk_valid = array_done & (stage_sel == 4'd0);
            end
            4'd13: begin // R3 post-attn residual_add out (16x16b)
                prb_chunk_data  = pack16x16(r3_dbg_out_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = r3_dbg_out_valid;
            end
            4'd14: begin // R7 post-MLP residual_add out (16x16b)
                prb_chunk_data  = pack16x16(r7_out_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = r7_out_valid;
            end
            4'd15: begin // G8 rescored frontier scores (K_F=8 x 32b fp32)
                prb_chunk_data  = pack8x32(g8_dbg_scores);
                prb_chunk_words = 8;  prb_chunk_valid = g8_dbg_scores_v;
            end
            default: ;  // unmapped sel -> no capture
        endcase
    end

    // ---- 0x90-FIX: dbg_sel-muxed STAGE-OUTPUT checksum feed -----------------
    // A SECOND lightweight stage mux (selected by dbg_sel @0x88, independent of
    // the PROBE V3 0xA8 selector) produces a per-output-chunk strobe + the
    // chunk's word0 + an XOR-fold of its words + a word count. dbg_observe folds
    // these into the LIVE 0x90 stage-output XOR (the old 0x90 input-checksum read
    // 0 once boot loads finished). Reuses the SAME pack helpers + taps as the
    // probe mux; pure read-only combinational logic.
    // (sout_data / sout_words / sout_valid / sout_xor_fold / sout_word0 are
    //  forward-declared near the dbg word nets above so the u_dbg port map sees
    //  them; the mux + fold below are their sole drivers.)
    always_comb begin
        sout_data  = '0; sout_words = '0; sout_valid = 1'b0;
        case (dbg_sel_w)
            4'd0:  begin sout_data = pack16x16(rms_g1_out_chunk_w); sout_words = 8;
                         sout_valid = rms_g1_out_valid; end
            4'd1:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd0); end
            4'd2:  begin sout_data = pack16x32(dbg_g5_attn_vec); sout_words = 16;
                         sout_valid = dbg_g5_attn_done; end
            4'd3:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd1); end
            4'd4:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd2); end
            4'd5:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd3); end
            4'd6:  begin sout_data = pack16x16(silu_out_chunk); sout_words = 8;
                         sout_valid = silu_valid; end
            4'd7:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd4); end
            4'd8:  begin sout_data = pack16x16(rms_g11_out_chunk); sout_words = 8;
                         sout_valid = g11_out_strobe; end
            4'd9:  begin sout_data = pack16x16(glm_wr_chunk); sout_words = 8;
                         sout_valid = glm_wr_valid; end
            4'd10: begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd5); end
            4'd11: begin sout_data = pack8x32(tree_topk_scores); sout_words = 8;
                         sout_valid = tree_done; end
            4'd13: begin sout_data = pack16x16(r3_dbg_out_chunk); sout_words = 8;
                         sout_valid = r3_dbg_out_valid; end
            4'd14: begin sout_data = pack16x16(r7_out_chunk); sout_words = 8;
                         sout_valid = r7_out_valid; end
            4'd15: begin sout_data = pack8x32(g8_dbg_scores); sout_words = 8;
                         sout_valid = g8_dbg_scores_v; end
            default: ;  // sel 12 (R1) / unmapped -> no stage-output fold
        endcase
    end
    // XOR-fold all words of the selected chunk to one 32b value (combinational).
    integer sxk;
    always_comb begin
        sout_xor_fold = 32'd0;
        for (sxk = 0; sxk < PRB_MAXCH; sxk = sxk + 1)
            if (sxk < sout_words) sout_xor_fold = sout_xor_fold ^ sout_data[sxk*32 +: 32];
    end

    dbg_probe_ram #(.DEPTH(1024), .CAP_WORDS(256), .MAX_CHUNK(PRB_MAXCH)) u_probe (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .probe_sel(prb_sel_w), .sel_we(prb_sel_we),
        .addr_we(prb_addr_we), .addr_wdata(prb_addr_wdata), .data_re(prb_data_re),
        .chunk_valid(prb_chunk_valid), .chunk_words(prb_chunk_words), .chunk_data(prb_chunk_data),
        .probe_addr_word(prb_addr_word), .probe_data_word(prb_data_word),
        .probe_xor_word(prb_xor_word), .probe_sum_word(prb_sum_word),
        .probe_count_word(prb_count_word)
    );

    // ================================================================
    // PDI#10 PER-STAGE VALUE READBACK — capture ALL 19 stages in ONE forward
    // (reg 0xDC/0xE0/0xE4/0xE8). Lets the host walk every stage's first NW
    // output words after a single silicon forward -> first-divergent stage
    // falls out empirically. PURELY ADDITIVE: every arm is a read-only tap off
    // an existing net (the SAME nets the probe mux above already fans out), so
    // there is no new load on any functional read port and the golden frontier
    // is BIT-IDENTICAL (proven NON-PERTURBING by the regread frontier gate).
    //
    // TAP ALIGNMENT (memory feedback_valid_data_latency_skew_class — the whole
    // point of this block): each strobe fires on its stage's DATA-VALID cycle.
    //   * G1  taps rms_g1_out_chunk_d1 (the d1-latched aligned bus), NOT the +1
    //     skewed rms_g1_out_chunk_w that fooled the PDI#9/PDI#10 probe-walk into
    //     a FALSE "G1 first-divergent" (PDI10_60MHZ_SILICON_RCA.md §1).
    //   * G4 is split into q/k/v sub-tiles via cap_q_we/cap_k_we/cap_v_we on the
    //     shared g4_tile_slice bus.
    //   * every other arm reuses the datapath-aligned net the existing aligned
    //     probe consumes. Verified latch-cycle==data-ready-cycle in xsim.
    // ----------------------------------------------------------------
    localparam int DSC_NSTAGE = 19;                 // PROBE_EXPANSION: +2 G8 stages (17->19)
    localparam int DSC_NW     = 32;                 // PROBE_EXPANSION: deeper (16->32) words/stage
    localparam int DSC_MAXCH  = LANES;              // 128 (widest GEMV tile)

    // per-stage strobe / word-count / flat-chunk feeds.
    reg  [DSC_NSTAGE-1:0]                 dsc_strobe;
    reg  [DSC_NSTAGE*8-1:0]               dsc_words;
    reg  [DSC_NSTAGE*DSC_MAXCH*32-1:0]    dsc_chunk;

    // helper: place a packed MAX_CHUNK*32 chunk + a word count into stage slot s.
    // (combinational assembly; one always_comb, single driver per net.)
    always_comb begin
        dsc_strobe = '0;
        dsc_words  = '0;
        dsc_chunk  = '0;
        // 0  G1  — d1-ALIGNED bus (NOT rms_g1_out_chunk_w). strobe on out_valid.
        dsc_strobe[0]  = rms_g1_out_valid;
        dsc_words[0*8 +: 8] = 8'd8;
        dsc_chunk[0*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(rms_g1_out_chunk_d1);
        // 1  G4q — QKV GEMV q sub-tile
        dsc_strobe[1]  = cap_q_we;
        dsc_words[1*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[1*DSC_MAXCH*32 +: DSC_MAXCH*32] = g4_tile_slice;
        // 2  G4k
        dsc_strobe[2]  = cap_k_we;
        dsc_words[2*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[2*DSC_MAXCH*32 +: DSC_MAXCH*32] = g4_tile_slice;
        // 3  G4v
        dsc_strobe[3]  = cap_v_we;
        dsc_words[3*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[3*DSC_MAXCH*32 +: DSC_MAXCH*32] = g4_tile_slice;
        // 4  G5  — attention out (FIXED 16x32b)
        dsc_strobe[4]  = dbg_g5_attn_done;
        dsc_words[4*8 +: 8] = 8'd16;
        dsc_chunk[4*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x32(dbg_g5_attn_vec);
        // 5  G6  — o-proj GEMV (stage_sel==1)
        dsc_strobe[5]  = array_done & (stage_sel == 4'd1);
        dsc_words[5*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[5*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;
        // 6  G7g — gate GEMV (stage_sel==2)
        dsc_strobe[6]  = array_done & (stage_sel == 4'd2);
        dsc_words[6*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[6*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;
        // 7  G7u — up GEMV (stage_sel==3)
        dsc_strobe[7]  = array_done & (stage_sel == 4'd3);
        dsc_words[7*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[7*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;
        // 8  G7p — SiLU*up (16x16b)
        dsc_strobe[8]  = silu_valid;
        dsc_words[8*8 +: 8] = 8'd8;
        dsc_chunk[8*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(silu_out_chunk);
        // 9  G9  — down GEMV (stage_sel==4)
        dsc_strobe[9]  = array_done & (stage_sel == 4'd4);
        dsc_words[9*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[9*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;
        // 10 G11 — final-RMSNorm out (aligned strobe g11_out_strobe)
        dsc_strobe[10] = g11_out_strobe;
        dsc_words[10*8 +: 8] = 8'd8;
        dsc_chunk[10*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(rms_g11_out_chunk);
        // 11 GLM — lmdown out
        dsc_strobe[11] = glm_wr_valid;
        dsc_words[11*8 +: 8] = 8'd8;
        dsc_chunk[11*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(glm_wr_chunk);
        // 12 G12 — logits GEMV (stage_sel==5)
        dsc_strobe[12] = array_done & (stage_sel == 4'd5);
        dsc_words[12*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[12*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;
        // 13 TREE — top-K scores (8x32b)
        dsc_strobe[13] = tree_done;
        dsc_words[13*8 +: 8] = 8'd8;
        dsc_chunk[13*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack8x32(tree_topk_scores);
        // 14 R1 — G1->G4 residual seam (16x16b read mux)
        dsc_strobe[14] = array_done & (stage_sel == 4'd0);
        dsc_words[14*8 +: 8] = 8'd8;
        dsc_chunk[14*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(seam_r1_rd);
        // 15 R3 — post-attn residual_add out (16x16b)
        dsc_strobe[15] = r3_dbg_out_valid;
        dsc_words[15*8 +: 8] = 8'd8;
        dsc_chunk[15*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(r3_dbg_out_chunk);
        // 16 R7 — post-MLP residual_add out (16x16b)
        dsc_strobe[16] = r7_out_valid;
        dsc_words[16*8 +: 8] = 8'd8;
        dsc_chunk[16*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(r7_out_chunk);

        // ---- PROBE_EXPANSION: G8 rescore bisection stages ----
        // 17 G8CHEAP — the cheap candidate vocab IDs that FEED the G8 rescore
        //   (rescore INPUT selection). Captured on tree_done (cheap top-8 ready),
        //   zero-extended to 32b/word. Lets the host see whether the WRONG cheap
        //   set entered the rescore (body-divergence upstream of G8) vs a correct
        //   set mis-rescored (G8 gather/dot fault) — the G8 path bisector.
        dsc_strobe[17] = tree_done;
        dsc_words[17*8 +: 8] = 8'd8;
        for (int gi8 = 0; gi8 < 8; gi8 = gi8 + 1)
            dsc_chunk[17*DSC_MAXCH*32 + gi8*32 +: 32] =
                {{(32-TREE_IDX_W){1'b0}}, tree_topk_indices[gi8*TREE_IDX_W +: TREE_IDX_W]};

        // 18 G8RESC — the G8 rescored ORDERED top-8 vocab IDs (rescore OUTPUT).
        //   Captured on g8_done. Compared to the frontier golden directly; with
        //   G8CHEAP above this localizes a wrong frontier to cheap-in vs rescore.
        dsc_strobe[18] = g8_done;
        dsc_words[18*8 +: 8] = 8'd8;
        for (int gi8 = 0; gi8 < 8; gi8 = gi8 + 1)
            dsc_chunk[18*DSC_MAXCH*32 + gi8*32 +: 32] =
                {{(32-TREE_IDX_W){1'b0}}, ow_topk_indices[gi8*TREE_IDX_W +: TREE_IDX_W]};
    end

    dbg_stage_capture #(
        .NSTAGE(DSC_NSTAGE), .NW(DSC_NW), .MAX_CHUNK(DSC_MAXCH)
    ) u_stage_cap (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start_pulse(start_pulse),
        .strobe(dsc_strobe), .words(dsc_words), .chunk(dsc_chunk),
        .cap_stage_sel(dbg_cap_stage_sel_w),
        .widx_we(dbg_cap_widx_we), .widx_wdata(dbg_cap_widx_wdata),
        .val_re(dbg_cap_val_re),
        .cap_word_idx_word(dbg_cap_widx_word),
        .cap_val_word(dbg_cap_val_word),
        .cap_meta_word(dbg_cap_meta_word),
        .cap_crc_word(dbg_cap_crc_word)        // MAXPROBE: per-stage CRC (0x108)
    );

    // ================================================================
    // pdi23 DETERMINISTIC G8 FRONTIER/CAND DUMP (read at 0x130 sel / 0x134 word).
    // Replaces the RACY cap[18] cap-CRC (G8SIL2_ONCARD_READBACK: 3 reads -> 3 CRCs /
    // shifted-by-one) with STICKY one-shot latch registers + a FIXED-address seek+read
    // (NO auto-increment seek), so a host can re-read the same index and get a
    // bit-identical value -> a STABLE silicon ground-truth for the next M7.
    //   g8_dump_flat[0..7]  = the G8 RESCORED ordered top-8 vocab ids (= ow_topk_indices,
    //                          the SAME source OUT-DMA bursts; latched on first g8_done).
    //   g8_dump_flat[8..15] = the cheap CAND-SET top-8 vocab ids that FED the rescore
    //                          (= tree_topk_indices; latched on first tree_done).
    // One-shot per forward (gated by *_latched, cleared on start_pulse) -> stable.
    // PURELY ADDITIVE read-only taps off existing nets -> frontier BIT-IDENTICAL
    // (proven non-perturbing by the regread frontier gate).
    reg [31:0] g8_dump_r [0:15];
    reg        g8_dump_resc_latched;
    reg        g8_dump_cand_latched;
    integer gdk;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n || start_pulse) begin
            g8_dump_resc_latched <= 1'b0;
            g8_dump_cand_latched <= 1'b0;
            for (gdk = 0; gdk < 16; gdk = gdk + 1) g8_dump_r[gdk] <= 32'd0;
        end else begin
            if (tree_done && !g8_dump_cand_latched) begin
                g8_dump_cand_latched <= 1'b1;
                for (gdk = 0; gdk < 8; gdk = gdk + 1)
                    g8_dump_r[8+gdk] <= {{(32-TREE_IDX_W){1'b0}},
                                         tree_topk_indices[gdk*TREE_IDX_W +: TREE_IDX_W]};
            end
            if (g8_done && !g8_dump_resc_latched) begin
                g8_dump_resc_latched <= 1'b1;
                for (gdk = 0; gdk < 8; gdk = gdk + 1)
                    g8_dump_r[gdk] <= {{(32-TREE_IDX_W){1'b0}},
                                       ow_topk_indices[gdk*TREE_IDX_W +: TREE_IDX_W]};
            end
        end
    end
    genvar gdg;
    generate
        for (gdg = 0; gdg < 16; gdg = gdg + 1) begin : g_g8dump
            assign g8_dump_flat[gdg*32 +: 32] = g8_dump_r[gdg];
        end
    endgenerate
    assign g8_dump_status = {g8_dump_resc_latched, g8_dump_cand_latched};

    // ================================================================
    // PDI#11 PROBE_EXPANSION — BOOT-LOAD RESIDENCY CRC (0xEC/0xF0/0xF4/0xF8).
    // Per-buffer {crc, sum, count} of exactly the words the load ports WRITE into
    // each on-chip buffer, so the host can DIRECTLY confirm load correctness on
    // silicon: count==0 => load wrote NOTHING; crc!=golden => load wrote the WRONG
    // bytes (e.g. the +128 chunk rotation, project_m7_loadgeom_plus128_rotation);
    // crc==golden => buffer byte-correct. Every input is a READ-ONLY write-side
    // tap (emb/hfc/gamma AXI-Lite, weight URAM eff_boot, scale BRAM dma_sc); the
    // block drives nothing back into the datapath -> NON-PERTURBING.
    // ================================================================
    // address tags = {wr_chunk, wr_word} for the chunk-addressed boot buffers, so
    // the residency CRC catches a +128 chunk remap (correct data -> wrong chunk).
    wire [15:0] res_emb_tag   = {{(16-EMB_CW-EMB_WPC_LOG2){1'b0}}, emb_wr_chunk,   emb_wr_word};
    wire [15:0] res_hfc_tag   = {{(16-EMB_CW-EMB_WPC_LOG2){1'b0}}, hfc_wr_chunk,   hfc_wr_word};
    wire [15:0] res_gamma_tag = {{(16-EMB_CW-EMB_WPC_LOG2){1'b0}}, gamma_wr_chunk, gamma_wr_word};

    dbg_residency #(.ADDR_TAG_W(16), .RD_TAG_W(EMB_CW)) u_residency (
        .clk(ap_clk), .rst_n(ap_rst_n), .res_clear(dbg_res_clear_w),
        .emb_we(emb_we),           .emb_wr_data(emb_wr_data32),     .emb_addr_tag(res_emb_tag),
        .hfc_we(hfc_we),           .hfc_wr_data(hfc_wr_data32),     .hfc_addr_tag(res_hfc_tag),
        .gamma_we(dbg_gamma_we),   .gamma_wr_data(gamma_wr_data32), .gamma_addr_tag(res_gamma_tag),
        .w_we(eff_boot_we),        .w_wr_data(eff_boot_wr_data),
        .sc_we(dma_sc_we),         .sc_wr_data({16'b0, dma_sc_wr_data}),
        // §6 READ-SIDE taps (the +128-fix observable): idx5=HFC read, idx6=EMB read.
        .g1_rd_stb(g1_rd_stb),     .g1_rd_idx_tap(g1_rd_idx_tap),   .g1_rd_word_tap(g1_rd_word_tap),
        .g1e_rd_stb(g1e_rd_stb),   .g1e_rd_idx_tap(g1e_rd_idx_tap), .g1e_rd_word_tap(g1e_rd_word_tap),
        .res_sel(dbg_res_sel_w),
        .res_crc_word(dbg_res_crc_word),
        .res_sum_word(dbg_res_sum_word),
        .res_count_word(dbg_res_count_word)
    );

endmodule

`default_nettype wire
