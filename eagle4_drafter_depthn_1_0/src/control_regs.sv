// =====================================================================
// control_regs.sv
// AXI4-Lite slave control/status register file for the EAGLE4 drafter.
// =====================================================================
//
// Register layout mirrors libeagle4_host_v56_direct.so / Vitis HLS
// auto-generated *_control_s_axi.v patterns so the host driver can talk
// to the Verilog kernel without changes:
//
//   0x00  AP_CTRL    [0]=start, [1]=done, [2]=idle, [3]=ready, [7]=auto_restart
//   0x04  GIER       (interrupt enable global, ignored — no interrupts)
//   0x08  IP_IER     (per-IP IE, ignored)
//   0x0c  IP_ISR     (per-IP ISR, ignored)
//   0x10  ctx_len    current sequence position (uint32)
//   0x18  num_frontiers   tree frontier count (uint32)
//   0x20  max_depth  tree max depth (uint32)
//   0x28  out0_base[31:0]    output buffer 0 LSB
//   0x2c  out0_base[63:32]   output buffer 0 MSB
//   0x30  out1_base[31:0]
//   0x34  out1_base[63:32]
//   0x38  out2_base[31:0]
//   0x3c  out2_base[63:32]
//   0x40  hbm_emb_base[31:0]
//   0x44  hbm_emb_base[63:32]
//   0x78  (RO) out_dma_status  bit[0]=out_busy, bit[1]=out_done  (OUTPUT-DMA)
//   0x168 W  PMEM write_addr pointer / R {24'b0, write_addr}
//   0x16c W  PMEM write_data mailbox (auto-increments) / R {16'h504D, PC, instr}
//   0x178 W/R special-register read selector (5-bit bank address)
//   0x17c R  {16'h5A5A canary, special_regs.read_data} (synchronous bank read;
//            retried if a WRSR write suppresses the bank read)
//
// 32-bit data, 32-bit addr (low 7 bits = byte offset, ignore [1:0]).
// Standard 5-channel AXI4-Lite (AW, W, B, AR, R).
//
// AP_CTRL semantics (HLS convention):
//   - Host writes 1 to bit [0] to start the kernel
//   - Kernel clears bit [0], sets bit [2]=idle when running
//   - Kernel sets bit [1]=done when finished
//   - Reading bit [1] (status read) auto-clears it (sticky-done behavior)
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module control_regs #(
    parameter int COL_W = 20    // boot-DMA column-count width (G12 N=128256 > 16b)
) (
    input  wire        clk,
    input  wire        rst_n,

    // AXI4-Lite slave
    // ADDR WIDTH = 8 bits (PDI#7): widened from 7 to reach the new debug
    // registers at 0x80 (FSM-state) / 0x84 (per-stage activity). The BD control
    // aperture is 4 KB (component.xml range 0x1000) so 0x80/0x84 are mappable;
    // the only change is the RTL decode width. Existing 0x00..0x7c decodes are
    // bit-identical (their bit[7]=0); bit[1:0] still ignored (word-aligned).
    input  wire [8:0]  s_axi_control_awaddr,   // MAXPROBE: 9-bit (hosts 0x100/0x104/0x108)
    input  wire        s_axi_control_awvalid,
    output reg         s_axi_control_awready,

    input  wire [31:0] s_axi_control_wdata,
    input  wire [3:0]  s_axi_control_wstrb,
    input  wire        s_axi_control_wvalid,
    output reg         s_axi_control_wready,

    output reg  [1:0]  s_axi_control_bresp,
    output reg         s_axi_control_bvalid,
    input  wire        s_axi_control_bready,

    input  wire [8:0]  s_axi_control_araddr,   // MAXPROBE: 9-bit (was 8-bit; hosts 0x100/0x104/0x108)
    input  wire        s_axi_control_arvalid,
    output reg         s_axi_control_arready,

    output reg  [31:0] s_axi_control_rdata,
    output reg  [1:0]  s_axi_control_rresp,
    output reg         s_axi_control_rvalid,
    input  wire        s_axi_control_rready,

    // Kernel-facing
    output wire        start_pulse,     // 1-cycle pulse when AP_CTRL.start written
    input  wire        kernel_busy,
    input  wire        kernel_done,

    output reg  [31:0] ctx_len,
    output reg  [31:0] num_frontiers,
    output reg  [31:0] max_depth,
    output reg  [63:0] out0_base,
    output reg  [63:0] out1_base,
    output reg  [63:0] out2_base,
    output reg  [63:0] hbm_emb_base,

    // G8 SPLICE (PREP fork pdi6_s7c_g8): HBM byte base of the 788 MB
    // lm_head_full [V,H] fp16 weight tensor that g8_rescore gathers candidate
    // rows from. The host one-time-boots lm_head_full to an HBM aperture and
    // writes its byte base here. Two 32-bit halves at the previously-unused
    //   0x14  lmfull_hbm_base[31:0]
    //   0x1c  lmfull_hbm_base[63:32]
    // (0x14/0x1c are the only free single-word slots in the 7-bit reg space;
    // every other slot 0x00..0x7c is already allocated — see header map.)
    output reg  [63:0] lmfull_hbm_base,

    // SPEEDWAVE-V2 (2026-07-18, ported from jewel J5B W1/V1 — J5B_F2_INTEGRATION
    // lineage). ⚠️ OFFSETS DIFFER FROM THE PS/JEWEL SHELL: jewel used 0x138/0x13C,
    // but THIS monolith's 0x138..0x148 window is the TREE-BATCH B=2 batch-test
    // ABI (kept intact below). The two runtime-config registers live at the
    // next free single-word slots 0x14C/0x150 instead.
    //
    // LEGACY 0x14C scan-length register: retained for host ABI/readback only.
    // TREE scan length now comes solely from the actual ISA special-register
    // tile_count; this value has no shared-engine execution effect.

    // K1 RUNTIME K_C (0x150): # of G8 rescore candidate rows (shortlist size).
    // 0 = compile K_C=512 = RESET DEFAULT, bit-identical when never written.
    // Host contract: power-of-2 256/128. Launch-sampled at start in
    // g8_rescore_p (same discipline as 0x14C).
    output reg  [15:0] kc_cfg,

    // 0x1125 ACCFIX (0x15C): runtime enables for the accept-path fixes
    // (kvres_accfix fork). bit0 SEAM (H1 child-seam remainder re-add),
    // bit1 B2P (round-half on the two attention >>>12 floors), bit2 KC
    // (tree_softmax_bridge kc-aware sweeps + S_MAX-skip), bit3 KVONLY
    // (KV-append-only launch mode). RESET 0x0 = all OFF = byte-identical
    // to the 0x1124 datapath. Read-back {16'hACF1 canary, 12'b0, cfg}.
    output reg  [3:0]  accfix_cfg,
    // OBS_LVL per-level candidate-ring readback window:
    //   0x160 W  obs_rd_idx (word index into the 1024-word ring view)
    //   0x160 R  obs_status_word {8'h0B canary, wrap, entry count}
    //   0x164 R  obs_rd_data (ring word at obs_rd_idx; registered read —
    //            the separate idx-write and data-read AXI transactions
    //            guarantee settle time)
    output reg  [10:0] obs_rd_idx,
    input  wire [31:0] obs_rd_data,
    input  wire [31:0] obs_status_word,

    // M1c: AXI-Lite weight-load path (host streams GPTQ weights → URAM banks)
    //   0x48  boot_lane_sel  (which lane's bank to write)
    //   0x4c  boot_wr_addr   (flat URAM address; auto-increments after each 0x50 write)
    //   0x50  boot_wr_data   (writing this pulses boot_we for 1 cycle, then addr++)
    output reg  [15:0] boot_lane_sel,
    output reg  [16:0] boot_wr_addr,
    output reg  [31:0] boot_wr_data,
    output reg         boot_we,
    // C5 (B1 input-embedding load): the SAME 0x48/0x4c/0x50 trio fills the
    // on-chip input_emb_buf when boot_is_emb (0x7c bit[0]) is set; when 0 the
    // trio writes URAM weight banks (unchanged M1c). Sticky target-select.
    //   0x7c  boot_is_emb    bit[0]=1 → 0x48/0x4c/0x50 trio targets input_emb_buf
    output reg         boot_is_emb,   // B1: 0=weight URAM, 1=input_emb_buf
    // FIX-E (h_fc separate buffer): a THIRD target-select, mirroring boot_is_emb.
    //   0x24  boot_is_hfc    bit[0]=1 → 0x48/0x4c/0x50 trio targets h_fc_buf
    //   Mutually exclusive with boot_is_emb (0x7c) and boot_dma_is_scale (0x54[1]).
    output reg         boot_is_hfc,   // FIX-E: 0=normal, 1=h_fc_buf
    // GAMMA-LOAD (2026-06-04, GAMMA_LOAD_FIX): a FOURTH target-select for the
    // 4 RMSNorm gamma buffers, mirroring boot_is_emb/boot_is_hfc. The SAME
    //   0x48/0x4c/0x50 trio fills gamma_<N>_buf when boot_gamma_sel != 0.
    //   0x08  boot_gamma_sel[2:0] : 0=off, 1=hidden_norm_g (gamma_emb_buf),
    //         2=input_layernorm_g (gamma_e_buf), 3=post_attn_norm_g
    //         (gamma_postattn_buf), 4=final_norm_g (gamma_final_buf).
    //   Mutually exclusive with boot_is_emb (0x7c), boot_is_hfc (0x24) and the
    //   scale/weight URAM path (boot_dma_is_scale @0x54[1]).
    output reg  [2:0]  boot_gamma_sel,  // GAMMA-LOAD: 0=off, 1..4 select a gamma buf

    // M9: boot-only m_axi DMA weight/scale-load path (HBM → URAM+BRAM via
    //   weight_dma_loader). SCALE-BOOT-PORT fork: 0x54 bit[1] selects the
    //   SCALE bank (BRAM) vs WEIGHT bank (URAM) for this load; the column
    //   regs are now COL_W-wide (G12 lm N=128256 > 16 bits).
    //   0x54  boot_dma_start  bit[0]=1 pulses boot_dma_start; bit[1]=is_scale
    //         FASTIN (2026-07-09): bit[2]=is_emb, bit[3]=is_hfc — per-call
    //         activation DMA target-selects (weight-mode stream → input_emb_buf /
    //         h_fc_buf instead of URAM). Sticky like bit[1]; mutually exclusive
    //         with each other and with is_scale (loader gates: scale wins, emb
    //         over hfc). Reading 0x54 returns {is_hfc,is_emb,is_scale,0} for
    //         write+readback liveness (feedback_probe_reg_needs_real_axi_write).
    //   0x58  boot_hbm_base[31:0]   (HBM byte base of the stream)
    //   0x5c  boot_hbm_base[63:32]
    //   0x60  boot_dma_n_cols       (# output columns to walk, up to 128256)
    //   0x64  boot_dma_col_start    (first global output column)
    //   0x68  boot_dma_col_step     (column stride: 1=full stage, LANES=single-lane)
    //   0x6c  boot_dma_cpc          (inner/col: weight=K/16 chunks, scale=K/GROUP groups)
    //   0x70  boot_dma_stage_base   (flat base for the stage: URAM W base OR BRAM S base)
    //   0x74  (RO) boot_dma_status  bit[0]=boot_busy, bit[1]=boot_done,
    //              bit[2]=boot_dma_timeout (M9 FIX-B sticky R-wait watchdog).
    //              A wedged m_axi_boot read (AR accepted, no R / AR never
    //              accepted) now trips this after BOOT_RWAIT_MAX stall cycles
    //              instead of hanging forever. Host reads 0x74[2]; if set, the
    //              load failed — re-pulse 0x54 to retry (no cold-AC needed).
    //              A successful retry re-raises done(=bit1) and clears bit2.
    output reg               boot_dma_start,
    output reg               boot_dma_is_scale,  // 0=weight (URAM), 1=scale (BRAM)
    output reg               boot_dma_is_emb,    // FASTIN: 0x54[2] weight-stream → input_emb_buf
    output reg               boot_dma_is_hfc,    // FASTIN: 0x54[3] weight-stream → h_fc_buf
    output reg  [63:0]       boot_hbm_base,
    output reg  [COL_W-1:0]  boot_dma_n_cols,
    output reg  [COL_W-1:0]  boot_dma_col_start,
    output reg  [COL_W-1:0]  boot_dma_col_step,
    output reg  [15:0]       boot_dma_cpc,
    output reg  [16:0]       boot_dma_stage_base,
    input  wire              boot_dma_busy,
    input  wire              boot_dma_done,
    input  wire              boot_dma_timeout,  // M9 FIX-B: 0x74 bit[2] sticky watchdog

    // OUTPUT-DMA status (SPEC_OUTPUT_DMA.md): the tree top-K → HBM write
    // master's busy/done. Read-only at 0x78. out0/1_base (0x28..0x34, already
    // captured above) are the destination pointers consumed by the writer.
    //   0x78  (RO) out_dma_status  bit[0]=out_busy, bit[1]=out_done, bit[2]=out_err
    input  wire        out_dma_busy,
    input  wire        out_dma_done,
    // C8 (B5): sticky OUT-DMA stall-timeout abort flag from output_dma_writer.
    // Routed to 0x78 bit[2] so the host can detect a wedged OUT-DMA.
    input  wire        out_dma_err,

    // ---- PDI#7 SILICON-HARDENING: debug visibility + G8 watchdog ----
    //   0x78  (RO) out_dma_status — bit[3] = g8 gather-watchdog tripped (sticky).
    //         (bits 0..2 already: out_busy / out_done / out_err.)
    //   0x80  (RO) fsmstate_word  — {magic 0xF500, last_stage, g5_state, top_state}
    //   0x84  (RO) activity_word  — sticky per-stage NON-ZERO-output flags
    // All read-only; assembled by dbg_observe (top). PURELY ADDITIVE.
    input  wire        g8_wd_err,        // 0x78 bit[3]
    input  wire [31:0] dbg_fsmstate,     // 0x80
    input  wire [31:0] dbg_activity,     // 0x84

    // ---- DEBUG BLOCK V2 (PDI#8): additive regs 0x88..0xA4 ----
    //   0x88 W : dbg_sel[3:0] stage-select for the 0x8C/0x90 read mux.
    //   0x8C..0xA4 RO : status words assembled by dbg_observe. All purely
    //   additive; existing 0x00..0x84 decodes are bit-identical (their
    //   bit[7:5] differ). bit[1:0] still ignored (word-aligned).
    output reg  [3:0]  dbg_sel,          // 0x88 (write-only select)
    input  wire [31:0] dbg_cyclecount,   // 0x8C
    input  wire [31:0] dbg_checksum,     // 0x90
    input  wire [31:0] dbg_g8progress,   // 0x94
    input  wire [31:0] dbg_g8araddr_lo,  // 0x98
    input  wire [31:0] dbg_g8araddr_hi,  // 0x9C
    input  wire [31:0] dbg_rearm,        // 0xA0
    input  wire [31:0] dbg_bootstream,   // 0xA4

    // ---- PROBE ARCHITECTURE V3 (intermediate-layer content probe) ----
    //   0xA8 W : probe_sel[3:0] stage code 0..14. Writing pulses prb_sel_we (the
    //            probe module clears its RAM ptr / XOR / sum / count on it).
    //   0xAC RW: probe_addr — RO read pointer; WRITING it seeks (prb_addr_we).
    //   0xB0 R : probe_data = RAM[probe_addr]; READING it pulses prb_data_re
    //            (auto-increments probe_addr in the probe module).
    //   0xB4 R : stream XOR ; 0xB8 R : stream sum32 ; 0xBC R : word count.
    //   All purely additive; existing 0x00..0xA4 decodes are bit-identical.
    output reg  [3:0]  probe_sel,        // 0xA8 stage select
    output reg         probe_sel_we,     // 1-cyc pulse on 0xA8 write
    output reg         probe_addr_we,    // 1-cyc pulse on 0xAC write (seek)
    output reg  [9:0]  probe_addr_wdata, // 0xAC seek value
    output reg         probe_data_re,    // 1-cyc pulse on 0xB0 read (auto-inc)
    input  wire [31:0] probe_addr,       // 0xAC read-back (current read pointer)
    input  wire [31:0] probe_data,       // 0xB0 (RAM word at probe_addr)
    input  wire [31:0] probe_xor,        // 0xB4
    input  wire [31:0] probe_sum,        // 0xB8
    input  wire [31:0] probe_count,      // 0xBC

    // ---- PROBE V3 tree/cand DUMP enable + out3/out4 base pointers ----
    //   0xC0 W bit0 : dump_en (default 0 = bit-identical legacy out-DMA).
    //   0xC4/0xC8   : out3_base (candidate set) lo/hi.
    //   0xCC/0xD0   : out4_base (tree heap full) lo/hi.
    //   Purely additive; out-DMA legacy bursts are byte-identical when dump_en=0.
    output reg         dump_en,          // 0xC0 bit0
    output reg  [63:0] out3_base,        // 0xC4/0xC8
    output reg  [63:0] out4_base,        // 0xCC/0xD0

    // ---- PDI#10 PROBE EXPANSION: per-stage-output content sample (0x90-fix) ----
    //   0x90 (RO) dbg_checksum now carries the dbg_sel-selected stage's OUTPUT
    //             running XOR (was the dead input checksum). 0x8C unchanged
    //             (cycle count). New content-sample regs:
    //   0xD4 (RO) stageout_sample : first 32b output word of the dbg_sel stage.
    //   0xD8 (RO) stageout_count  : # 32b output words seen for the dbg_sel stage.
    input  wire [31:0] dbg_stageout_sample,  // 0xD4
    input  wire [31:0] dbg_stageout_count,   // 0xD8

    // ---- PDI#10 PER-STAGE VALUE READBACK (capture ALL 17 stages in 1 forward) ----
    //   0xDC W  dbg_cap_stage_sel[4:0] : stage code 0..16 (read-mux select; does
    //                                    NOT clear capture — the forward fills all).
    //   0xE0 RW dbg_cap_word_idx[3:0]  : word index in the selected stage. AUTO-
    //                                    INCREMENTS after each 0xE4 read (wraps NW);
    //                                    WRITING it seeks (pulses dbg_cap_widx_we).
    //   0xE4 R  dbg_cap_val            : cap[stage_sel][word_idx]; reading auto-incs
    //                                    (pulses dbg_cap_val_re).
    //   0xE8 R  dbg_cap_meta           : {captured_flag, nwords} for selected stage.
    output reg  [4:0]  dbg_cap_stage_sel,    // 0xDC stage select (0..18)
    output reg         dbg_cap_widx_we,      // 1-cyc pulse on 0xE0 write (seek)
    output reg  [4:0]  dbg_cap_widx_wdata,   // 0xE0 seek value (PROBE_EXPANSION: 5b for NW=32)
    output reg         dbg_cap_val_re,       // 1-cyc pulse on 0xE4 read (auto-inc)
    input  wire [31:0] dbg_cap_widx_word,    // 0xE0 read-back (current word idx)
    input  wire [31:0] dbg_cap_val_word,     // 0xE4 cap[stage_sel][word_idx]
    input  wire [31:0] dbg_cap_meta_word,    // 0xE8 {captured, nwords}

    // ---- PDI#11 PROBE_EXPANSION: BOOT-LOAD RESIDENCY CRC (dbg_residency) ----
    //   0xEC W  res_sel[2:0]  : buffer select 0=EMB 1=HFC 2=GAMMA 3=WEIGHT 4=SCALE.
    //          bit[3]=res_clear : writing 1 here pulses a 1-cyc clear of ALL
    //                           residency accumulators (NOT cleared on start_pulse —
    //                           loads precede AP_START; a start_pulse clear would
    //                           wipe the load history). Host pulses this before a
    //                           fresh load if it wants a clean re-measure.
    //   0xF0 R  res_crc       : rotation-sensitive CRC of the selected buffer's
    //                           WRITE stream (load-wrote-wrong vs load-wrote-right).
    //   0xF4 R  res_sum       : 32b running SUM of the write words (commutative XCK).
    //   0xF8 R  res_count     : # 32b words written to the buffer (persists across
    //                           the forward; count==0 => load wrote NOTHING).
    output reg  [2:0]  res_sel,              // 0xEC buffer select
    output reg         res_clear,            // 0xEC bit[3] -> 1-cyc clear pulse
    input  wire [31:0] res_crc_word,         // 0xF0
    input  wire [31:0] res_sum_word,         // 0xF4
    input  wire [31:0] res_count_word,       // 0xF8

    // ---- F3-CLR: KV-RESIDENCY SESSION CLEAR (0x154) ---------------------
    //   0x154 W  bit[0]=kv_res_clear : writing 1 pulses a 1-cycle clear of the
    //            ATTENTION block's per-slot `kv_resident` bitmap (+ its count).
    //            *** THIS IS NOT 0xEC bit[3]. *** 0xEC clears the BOOT-LOAD CRC
    //            accumulators (EMB/HFC/GAMMA/WEIGHT/SCALE) — a completely
    //            different thing. THIS one ends a KV SESSION.
    //            WHY IT EXISTS: kv_resident was SET-ONLY; the only clear was
    //            rst_n = a PHYSICAL COLD-AC. A new SGLang request reuses KV
    //            slots 0..N for different context without rewriting them all,
    //            so the previous request's slots stayed marked resident over
    //            the previous request's K/V and the drafter attended stale
    //            cross-request context masked as legitimate -> accept pinned
    //            at the floor (1.02-1.05) through every host-side fix, depth-0
    //            golden 7/8 fresh cold-AC but 0/8 after ANY decode run.
    //            HOST CONTRACT: write 0x00000001 to 0x154 at the START of every
    //            new sequence/request, BEFORE that sequence's first draft
    //            launch (and before priming its KV). Idempotent; never needed
    //            mid-sequence (a mid-sequence clear would drop the prefix).
    //   0x154 R  {16'h4B56 "KV" canary, 4'b0, kv_res_cnt[11:0]} : LIVE count of
    //            DISTINCT resident KV slots. This is the host's VERIFICATION
    //            hook — read it right after the pulse and it must be 0. The
    //            canary proves the 0x154 decode is alive (an aliased/RAZ bus
    //            cannot produce 0x4B56). It also makes the ctx_len(reg 0x10)
    //            <-> residency divergence observable from the host for the
    //            first time.
    output reg         kv_res_clear,         // 0x154 bit[0] -> 1-cyc clear pulse
    input  wire [31:0] kv_res_status_word,   // 0x154 read-back {canary, count}

    // ---- MAXPROBE LIVENESS WINDOW (9-bit araddr; 0x100/0x104/0x108) ----
    //   0x100 RO maxprobe_build_id : hardwired {0xE4B7, BUILD_TAG, PROBE_REV}. The
    //            host's FIRST read; mismatch => probe AXI window dead/mis-decoded.
    //   0x104 RW selftest_reg      : write latches; READ returns BITWISE-NOT of the
    //            stored value (proves write+read decode alive; an aliased/RAZ bus
    //            cannot produce the inverted pattern). Single-driver, NOT in the
    //            wide 256b mux (a clean self-contained reg).
    //   0x108 RO cap_crc_word      : per-stage rotation-sensitive CRC of the
    //            0xDC-selected stage's captured output stream.
    input  wire [31:0] maxprobe_build_id,    // 0x100 (hardwired const from top)
    input  wire [31:0] cap_crc_word,         // 0x108 (per-stage CRC from dbg_stage_capture)

    // ---- PDI#11 PER-STAGE GOLDEN-INJECT (stage_inject.sv; 9-bit araddr) ----
    //   0x10C W/R inject_en bitmask : per-stage enable (write+readback liveness).
    //   0x110 W/R inj_widx          : buffer write pointer (seek; auto-inc on 0x114).
    //   0x114 W   inj_data          : inj_buf[inj_widx] <= wdata; auto-incs widx.
    //   0x118 W   inj_sel           : informational stage tag (read-back in 0x11C).
    //   0x11C R   inject_status     : {0x1A canary, present, inj_sel, last-word lo}.
    // All single-cycle write pulses keyed off wr_fire (one-driver). These are inert
    // (the top ties wdata back / reads RAZ) when STAGE_INJECT=0.
    output reg         inj_en_we,            // 0x10C write pulse
    output reg  [31:0] inj_en_wdata,         // 0x10C data
    output reg         inj_widx_we,          // 0x110 write pulse (seek)
    output reg  [31:0] inj_widx_wdata,       // 0x110 seek value
    output reg         inj_data_we,          // 0x114 write pulse
    output reg  [31:0] inj_data_wdata,       // 0x114 data
    output reg         inj_sel_we,           // 0x118 write pulse
    output reg  [31:0] inj_sel_wdata,        // 0x118 data
    input  wire [31:0] inj_en_word,          // 0x10C read-back (live bitmask)
    input  wire [31:0] inj_widx_word,        // 0x110 read-back (write pointer)
    input  wire [31:0] inj_status_word,      // 0x11C liveness/status

    // ---- M7 GLM=0 WIRED-TAP LOCALIZER (9-bit araddr; 0x120/0x124/0x128) ----
    //   0x120 RO lmtap_seam_word  : {8'hA1, fired, seam_r8 BRAM DOUT lane0[15:0]}
    //   0x124 RO lmtap_guard_word : {8'hA2, fired, rd_guard_q}
    //   0x128 RO lmtap_ad_word    : {8'hA3, fired, a_d lane0[15:0]}
    // Latched at the first GLM S_RUN cycle; canary distinguishes dead-window from a
    // genuine captured zero. READ-ONLY (single-driver const from top tap latch).
    input  wire [31:0] lmtap_seam_word,      // 0x120
    input  wire [31:0] lmtap_guard_word,     // 0x124
    input  wire [31:0] lmtap_ad_word,        // 0x128

    // ---- pdi23 DETERMINISTIC G8 FRONTIER/CAND DUMP (9-bit araddr; 0x130/0x134) ----
    //   Replaces the RACY cap[18] cap-CRC (G8SIL2_ONCARD_READBACK: 3 reads -> 3 CRCs,
    //   shifted-by-one) with a STICKY, FIXED-ADDRESS seek+read (NO auto-increment, so
    //   re-reads are bit-identical -> a STABLE silicon ground-truth for the next M7).
    //   0x130 W/R g8dump_sel : index 0..15 (0..7 = rescored frontier ids, 8..15 =
    //                          cheap cand-set ids). Read returns {status[1:0], sel}.
    //   0x134 R   g8dump_word: the sticky-latched g8_dump_flat[sel] (id, zero-ext).
    //   g8_dump_flat is one-shot latched in the top on the first g8_done/tree_done
    //   after start_pulse (mirrors the OUT-DMA frontier source ow_topk_indices).
    input  wire [16*32-1:0] g8_dump_flat,    // 0x134 read source (16 x 32b sticky ids)
    input  wire [1:0]       g8_dump_status,  // {resc_latched, cand_latched}

    // ---- TREE-BATCH B=2 SILICON BATCH-TEST WINDOW (9-bit araddr; 0x138..0x148) ----
    //   Proves the shipping body array's sibling-batch (BODY_BATCH_B=2) weight-reuse
    //   datapath on hardware. INERT in the normal forward (batch_test_ctrl[0]=0 =>
    //   a_chunk_batch tied '0 in the top). The sibling column MACs the SAME URAM
    //   weight / BRAM scale stream as b0, so result_q is LINEAR in the sibling
    //   activation: with a_chunk_batch = ±a_chunk the host asserts sib == ±b0 on the
    //   selected lane (a HOST-known invariant that needs no weight knowledge; a dead
    //   or copy-of-b0 sibling FAILS the negate arm -> teeth, feedback_ab_twin...).
    //   0x138 W/R batch_test_ctrl : bit0=en, bit1=negate, bits[15:8]=lane_sel(0..127).
    //             Read returns {8'hBA canary, 0, batch_test_ctrl[15:0]} (decode-alive).
    //   0x13c R   batch_b0_lo   : result_q_per_lane[lane_sel][31:0]  (b0 raw scaled acc)
    //   0x140 R   batch_b0_hi   : {16'hB000, result_q_per_lane[lane_sel][47:32]}
    //   0x144 R   batch_sib_lo  : result_q_batch[lane_sel][31:0]     (sibling raw acc)
    //   0x148 R   batch_sib_hi  : {16'hB500, result_q_batch[lane_sel][47:32]}
    output reg  [31:0] batch_test_ctrl,      // 0x138 host control (en/negate/lane_sel)
    input  wire [31:0] batch_b0_lo,          // 0x13c b0  result_q[lane_sel] lo
    input  wire [31:0] batch_b0_hi,          // 0x140 b0  result_q[lane_sel] hi (+canary)
    input  wire [31:0] batch_sib_lo,         // 0x144 sib result_q[lane_sel] lo
    input  wire [31:0] batch_sib_hi,          // 0x148 sib result_q[lane_sel] hi (+canary)

    // outputs for ISA PMEM
    output reg  [31:0] pmem_write_data,
    output reg  [7:0]  pmem_write_addr,
    output reg         pmem_we,
    // observability inputs for ISA PMEM
    input  wire [7:0]  fetch_instr,
    input  wire [6:0]  fetch_pc,

    // ISA special-register bank debug readback (0x178 selector / 0x17C data).
    // The bank read port is synchronous, so a 0x17C read is deferred until
    // the bank reports read_ack; a simultaneous WRSR write suppresses the
    // bank read (write priority) and this block retries instead of latching
    // stale read_data.
    output reg  [4:0]  sr_read_sel,
    output reg         sr_read_en,
    input  wire [15:0] sr_read_data,
    input  wire        sr_read_ack,

    // Shared GEMV engine status (0x170) and active tile (0x174).
    input  wire        gemv_busy,
    input  wire        gemv_done_sticky,
    input  wire        gemv_error_sticky,
    input  wire [15:0] gemv_active_tile
);

    // ----------------------------------------------------------------
    // AP_CTRL bits
    // ----------------------------------------------------------------
    reg ap_start_reg;
    reg ap_done_reg;       // sticky, cleared on read
    reg ap_idle_reg;
    reg ap_ready_reg;
    reg ap_auto_restart;

    // DONE-CLEAR-AT-LATCH (2026-06-06): single-cycle pulse from the read
    // channel to the ap_done block. Asserted in the AR-ACCEPT branch (the
    // cycle rdata is latched) for a 0x00 read, so the sticky ap_done bit is
    // cleared on the SAME read whose data was captured — not on a stale LIVE
    // araddr at R-handshake time (the old race, see L396 fix below). Driven by
    // exactly one always_ff (the read channel) → single-driver discipline.
    reg ap_done_clr_pulse;

    // 0x17C special-register readback: the bank read is synchronous and a
    // WRSR write suppresses it (write priority). Wait for read_ack, then
    // capture the bank's registered data one cycle later.
    localparam logic [1:0] SR_RD_IDLE = 2'd0;
    localparam logic [1:0] SR_RD_WAIT = 2'd1;
    localparam logic [1:0] SR_RD_CAP  = 2'd2;
    reg [1:0] sr_rd_state;

    wire [31:0] ap_ctrl_rdata = {24'b0, ap_auto_restart, 3'b0, ap_ready_reg, ap_idle_reg, ap_done_reg, ap_start_reg};

    assign start_pulse = ap_start_reg & ~kernel_busy;

    // ----------------------------------------------------------------
    // Write channel
    // ----------------------------------------------------------------
    reg [8:0] aw_addr_reg;   // MAXPROBE: 9-bit (was 8-bit)
    reg       aw_addr_valid;

    // BOOT-WE-PULSE FIX (2026-06-13, GRAND_BASE_FIXES): wr_fire is a TRUE
    // single-cycle write-accept strobe. The legacy register-write block keyed off
    // (s_axi_control_wvalid && s_axi_control_wready). s_axi_control_wready is a
    // REGISTERED output (high the cycle AFTER the accept). On silicon the QDMA
    // AXI-lite master holds wvalid asserted until it observes wready, so the
    // overlap window let the `wvalid && wready` term re-evaluate true for TWO
    // cycles -> the 0x50 case (and hence boot_we + boot_wr_addr++) fired twice per
    // single host write -> the legendary 2:1 boot LOAD-DOUBLING (value into slots
    // P and P+1, pointer +2). wr_fire is set ONLY on the single-cycle accept event
    // the handshake FSM uses to assert bvalid (which clears aw_addr_valid the same
    // edge, so it is unambiguously ONE cycle). All register decodes + boot_we now
    // key off wr_fire, so each 0x50 write pulses boot_we exactly once and advances
    // the pointer by exactly 1. xsim drove wready as a clean 1-cycle pulse so this
    // bug was silicon-only and invisible to sim — the fix is gated by nothing
    // (it is strictly more correct: wr_fire is one cycle by construction).
    reg       wr_fire;

    // MAXPROBE 0x104 self-test register: write latches; read returns ~value.
    // Single-driver (this write block only); NOT in the wide capture/residency mux.
    reg [31:0] selftest_reg;
    reg [3:0]  g8dump_sel_reg;   // pdi23 0x130 seek index for the deterministic G8 dump

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_axi_control_awready <= 1'b0;
            s_axi_control_wready  <= 1'b0;
            s_axi_control_bvalid  <= 1'b0;
            s_axi_control_bresp   <= 2'b00;
            aw_addr_reg   <= 9'b0;
            aw_addr_valid <= 1'b0;
            wr_fire       <= 1'b0;
        end else begin
            // BOOT-WE-PULSE FIX: wr_fire defaults low every cycle; set true for
            // exactly the single accept cycle below. This is the only fire signal
            // the register-decode + boot_we now key off (NOT wvalid && wready).
            wr_fire <= 1'b0;

            // accept AWVALID once at a time
            if (s_axi_control_awvalid && !aw_addr_valid && !s_axi_control_bvalid) begin
                s_axi_control_awready <= 1'b1;
                aw_addr_reg   <= s_axi_control_awaddr;
                aw_addr_valid <= 1'b1;
            end else s_axi_control_awready <= 1'b0;

            // accept WVALID after AW captured (single-cycle accept: this branch
            // sets aw_addr_valid<=0 and bvalid<=1 the SAME edge, so the condition
            // can fire only once per write -> wr_fire is one cycle by construction)
            if (s_axi_control_wvalid && aw_addr_valid && !s_axi_control_bvalid) begin
                s_axi_control_wready  <= 1'b1;
                s_axi_control_bvalid  <= 1'b1;
                s_axi_control_bresp   <= 2'b00;
                aw_addr_valid <= 1'b0;
                wr_fire       <= 1'b1;   // single-cycle write-accept strobe
            end else s_axi_control_wready <= 1'b0;

            // clear BVALID once master acks
            if (s_axi_control_bvalid && s_axi_control_bready) s_axi_control_bvalid <= 1'b0;
        end
    end

    // Register writes — fires on the cycle after WVALID accepted
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ap_start_reg    <= 1'b0;
            ap_auto_restart <= 1'b0;
            ctx_len         <= 32'b0;
            num_frontiers   <= 32'b0;
            max_depth       <= 32'b0;
            out0_base       <= 64'b0;
            out1_base       <= 64'b0;
            out2_base       <= 64'b0;
            hbm_emb_base    <= 64'b0;
            lmfull_hbm_base <= 64'b0;   // G8 SPLICE
            kc_cfg          <= 16'd0;   // K1 (0x150): 0 = K_C=512 shortlist default
            accfix_cfg      <= 4'h0;    // ACCFIX (0x15C): reset ALL-OFF = 0x1124-identical
            obs_rd_idx      <= 11'd0;   // OBS_LVL (0x160) read index
            boot_lane_sel   <= 16'b0;
            boot_wr_addr    <= 17'b0;
            boot_wr_data    <= 32'b0;
            boot_we         <= 1'b0;
            boot_is_emb     <= 1'b0;   // C5 (B1)
            boot_is_hfc     <= 1'b0;   // FIX-E
            boot_gamma_sel  <= 3'b0;   // GAMMA-LOAD (0x08)
            boot_dma_start      <= 1'b0;
            boot_dma_is_scale   <= 1'b0;
            boot_dma_is_emb     <= 1'b0;   // FASTIN
            boot_dma_is_hfc     <= 1'b0;   // FASTIN
            boot_hbm_base       <= 64'b0;
            boot_dma_n_cols     <= '0;
            boot_dma_col_start  <= '0;
            boot_dma_col_step   <= '0;
            boot_dma_cpc        <= 16'b0;
            boot_dma_stage_base <= 17'b0;
            dbg_sel             <= 4'b0;   // DEBUG-V2 (0x88)
            probe_sel           <= 4'b0;   // PROBE-V3 (0xA8)
            probe_sel_we        <= 1'b0;
            probe_addr_we       <= 1'b0;
            probe_addr_wdata    <= 10'b0;
            dump_en             <= 1'b0;    // PROBE-V3 (0xC0) default OFF
            out3_base           <= 64'b0;   // PROBE-V3 (0xC4/0xC8)
            out4_base           <= 64'b0;   // PROBE-V3 (0xCC/0xD0)
            dbg_cap_stage_sel   <= 5'b0;    // PER-STAGE READBACK (0xDC)
            dbg_cap_widx_we     <= 1'b0;
            dbg_cap_widx_wdata  <= 5'b0;    // PROBE_EXPANSION: 5b for NW=32
            res_sel             <= 3'b0;    // PROBE_EXPANSION RESIDENCY (0xEC)
            res_clear           <= 1'b0;    // 0xEC bit[3] clear pulse
            kv_res_clear        <= 1'b0;    // 0x154 bit[0] KV-residency clear pulse
            selftest_reg        <= 32'b0;   // MAXPROBE 0x104 self-test latch
            g8dump_sel_reg      <= 4'b0;     // pdi23 0x130 deterministic G8-dump seek
            batch_test_ctrl     <= 32'b0;    // TREE-BATCH 0x138 batch-test control (inert)
            // PER-STAGE INJECT (0x10C..0x118): write pulses + data registers.
            inj_en_we           <= 1'b0;
            inj_en_wdata        <= 32'b0;
            inj_widx_we         <= 1'b0;
            inj_widx_wdata      <= 32'b0;
            inj_data_we         <= 1'b0;
            inj_data_wdata      <= 32'b0;
            inj_sel_we          <= 1'b0;
            inj_sel_wdata       <= 32'b0;

            // dbg_cap_val_re is OWNED by the read-data always_ff (default-low + 0xE4
            // pulse, ~line 483/492). Assigning it HERE too made it MULTI-DRIVEN
            // ([Synth 8-6858/6859] in d10g8_dbg) and synth tied it to GND, killing
            // the 0xE4 auto-inc. It self-clears every cycle so it needs no reset here.
            // (removed: dbg_cap_val_re <= 1'b0;)
            pmem_write_data <= 32'b0;
            pmem_write_addr <= 8'b0;
            pmem_we         <= 1'b0;
            sr_read_sel     <= 5'b0;
            // sr_read_en / sr_rd_state are OWNED by the read-channel
            // always_ff (including its reset) to keep a single driver.
        end else begin
            // auto-clear ap_start after the kernel picks it up
            if (kernel_busy) ap_start_reg <= 1'b0;

            // boot_we is a 1-cycle pulse — default low every cycle
            boot_we <= 1'b0;

            // boot_dma_start is a 1-cycle pulse — default low every cycle
            boot_dma_start <= 1'b0;
            // pmem_we is a 1-cycle pulse — default low every cycle
            pmem_we <= 1'b0;

            // Auto-increment write address when a PMEM word commits (wraps at 32 words)
            if (pmem_we) begin
                pmem_write_addr <= (pmem_write_addr == 8'd31) ? 8'd0 : (pmem_write_addr + 8'd1);
            end

            // PROBE-V3: sel-write and addr-seek pulses default low every cycle.
            probe_sel_we  <= 1'b0;
            probe_addr_we <= 1'b0;

            // PER-STAGE READBACK: widx-seek pulse defaults low every cycle.
            dbg_cap_widx_we <= 1'b0;

            // RESIDENCY: clear pulse defaults low every cycle (0xEC bit[3]).
            res_clear <= 1'b0;

            // F3-CLR: the KV-residency clear is a 1-CYCLE pulse -> default low
            // every cycle. It is set below ONLY inside `if (wr_fire)`, which is
            // single-cycle by construction, so it cannot suffer the boot_we
            // 2:1 doubling class (feedback_boot_write_strobe_2cycle_doubles_load).
            // Doubling would be harmless here (clear is idempotent), but the
            // single-driver + wr_fire discipline is kept uniform.
            kv_res_clear <= 1'b0;

            // PER-STAGE INJECT: all four write pulses default low every cycle; the
            // case below sets exactly one true for one cycle (wr_fire-gated, so each
            // is a single-cycle strobe by construction — no boot 2:1 doubling class).
            inj_en_we   <= 1'b0;
            inj_widx_we <= 1'b0;
            inj_data_we <= 1'b0;
            inj_sel_we  <= 1'b0;

            // M1c FIX (2026-05-22): advance the write pointer on the cycle the
            // write COMMITS (boot_we high), NOT on the 0x50-write edge. Both this
            // increment and weight_bank_uram's `mem[wr_addr]<=wr_data` are
            // non-blocking and sample the CURRENT boot_wr_addr at the same edge,
            // so the word lands at the host-selected addr and the pointer advances
            // for the next word. The old in-case increment off-by-one'd every word
            // (landed at addr+1) — caught by tb_m1c_axi_weightload (FAIL→fix proven
            // bit-exact). Placed BEFORE the case so a same-cycle 0x4c write wins.
            if (boot_we) boot_wr_addr <= boot_wr_addr + 17'd1;

            // BOOT-WE-PULSE FIX (2026-06-13): key the register decode off the
            // single-cycle wr_fire strobe, NOT (wvalid && wready). wready is a
            // registered output and on silicon the QDMA master held wvalid across
            // the overlap window, so wvalid && wready re-evaluated true for 2 cycles
            // -> the 0x50 case fired twice -> boot_we high 2 cycles -> 2:1 boot
            // load-doubling (pointer +2). wr_fire is one cycle by construction.
            if (wr_fire) begin
                case (aw_addr_reg)
                    7'h00: begin
                        if (s_axi_control_wdata[0]) ap_start_reg <= 1'b1;
                        ap_auto_restart <= s_axi_control_wdata[7];
                    end
                    // GAMMA-LOAD (0x08): which gamma buffer the 0x48/0x4c/0x50
                    // trio targets. 0=off (trio → URAM weights / emb / hfc).
                    7'h08: boot_gamma_sel <= s_axi_control_wdata[2:0];
                    7'h10: ctx_len       <= s_axi_control_wdata;
                    // G8 SPLICE: lmfull_hbm_base halves (free slots 0x14/0x1c)
                    7'h14: lmfull_hbm_base[31:0]  <= s_axi_control_wdata;
                    7'h1c: lmfull_hbm_base[63:32] <= s_axi_control_wdata;
                    7'h18: num_frontiers <= s_axi_control_wdata;
                    7'h20: max_depth     <= s_axi_control_wdata;
                    7'h28: out0_base[31:0]   <= s_axi_control_wdata;
                    7'h2c: out0_base[63:32]  <= s_axi_control_wdata;
                    7'h30: out1_base[31:0]   <= s_axi_control_wdata;
                    7'h34: out1_base[63:32]  <= s_axi_control_wdata;
                    7'h38: out2_base[31:0]   <= s_axi_control_wdata;
                    7'h3c: out2_base[63:32]  <= s_axi_control_wdata;
                    7'h40: hbm_emb_base[31:0]  <= s_axi_control_wdata;
                    7'h44: hbm_emb_base[63:32] <= s_axi_control_wdata;
                    // M1c weight-load path:
                    7'h48: boot_lane_sel <= s_axi_control_wdata[15:0];
                    7'h4c: boot_wr_addr  <= s_axi_control_wdata[16:0];
                    7'h50: begin
                        boot_wr_data <= s_axi_control_wdata;
                        boot_we      <= 1'b1;   // pulse; addr advances next cycle (see M1c FIX above)
                    end
                    // M9 boot-DMA path (SCALE-BOOT-PORT: 0x54 bit[1]=is_scale,
                    // wide col regs):
                    7'h54: begin
                        if (s_axi_control_wdata[0]) boot_dma_start <= 1'b1; // pulse
                        boot_dma_is_scale <= s_axi_control_wdata[1];         // bank select
                        // FASTIN: activation-buffer targets (sticky, like is_scale)
                        boot_dma_is_emb   <= s_axi_control_wdata[2];
                        boot_dma_is_hfc   <= s_axi_control_wdata[3];
                    end
                    7'h58: boot_hbm_base[31:0]   <= s_axi_control_wdata;
                    7'h5c: boot_hbm_base[63:32]  <= s_axi_control_wdata;
                    7'h60: boot_dma_n_cols       <= s_axi_control_wdata[COL_W-1:0];
                    7'h64: boot_dma_col_start    <= s_axi_control_wdata[COL_W-1:0];
                    7'h68: boot_dma_col_step     <= s_axi_control_wdata[COL_W-1:0];
                    7'h6c: boot_dma_cpc          <= s_axi_control_wdata[15:0];
                    7'h70: boot_dma_stage_base   <= s_axi_control_wdata[16:0];
                    // C5 (B1) input-emb target-select (mirrors M9 is_scale @0x54[1]):
                    //   bit[0]=1 → the 0x48/0x4c/0x50 trio fills input_emb_buf.
                    //   Sticky: host sets it, streams words, then clears it.
                    7'h7c: boot_is_emb <= s_axi_control_wdata[0];
                    // FIX-E (h_fc target-select, mirrors boot_is_emb @0x7c):
                    //   bit[0]=1 → the 0x48/0x4c/0x50 trio fills h_fc_buf.
                    7'h24: boot_is_hfc <= s_axi_control_wdata[0];
                    // DEBUG BLOCK V2 (PDI#8): stage-select for 0x8C/0x90 read mux.
                    8'h88: dbg_sel <= s_axi_control_wdata[3:0];
                    // PROBE V3 (0xA8/0xAC): stage select (pulses clear) + addr seek.
                    8'ha8: begin
                        probe_sel    <= s_axi_control_wdata[3:0];
                        probe_sel_we <= 1'b1;   // 1-cyc pulse -> probe module clears
                    end
                    8'hac: begin
                        probe_addr_wdata <= s_axi_control_wdata[9:0];
                        probe_addr_we    <= 1'b1;   // 1-cyc seek pulse
                    end
                    // PROBE V3 dump enable + out3/out4 base pointers.
                    8'hc0: dump_en          <= s_axi_control_wdata[0];
                    8'hc4: out3_base[31:0]  <= s_axi_control_wdata;
                    8'hc8: out3_base[63:32] <= s_axi_control_wdata;
                    8'hcc: out4_base[31:0]  <= s_axi_control_wdata;
                    8'hd0: out4_base[63:32] <= s_axi_control_wdata;
                    // PER-STAGE VALUE READBACK (0xDC stage-select, 0xE0 widx seek).
                    8'hdc: dbg_cap_stage_sel <= s_axi_control_wdata[4:0];
                    8'he0: begin
                        dbg_cap_widx_wdata <= s_axi_control_wdata[4:0];  // 5b for NW=32
                        dbg_cap_widx_we    <= 1'b1;   // 1-cyc seek pulse
                    end
                    // PROBE_EXPANSION RESIDENCY CRC buffer-select (0xEC) + clear pulse.
                    8'hec: begin
                        res_sel   <= s_axi_control_wdata[2:0];
                        res_clear <= s_axi_control_wdata[3];   // bit[3]=1 -> 1-cyc clear
                    end
                    // MAXPROBE 0x104 self-test write: latch the host pattern; the read
                    // decode returns ~selftest_reg (proves the write+read decode alive).
                    9'h104: selftest_reg <= s_axi_control_wdata;
                    // PER-STAGE INJECT writes (9-bit addr; pulse + data forwarded to
                    // stage_inject in the top). Each is a single-cycle wr_fire strobe.
                    9'h10c: begin inj_en_we   <= 1'b1; inj_en_wdata   <= s_axi_control_wdata; end
                    9'h110: begin inj_widx_we <= 1'b1; inj_widx_wdata <= s_axi_control_wdata; end
                    9'h114: begin inj_data_we <= 1'b1; inj_data_wdata <= s_axi_control_wdata; end
                    9'h118: begin inj_sel_we  <= 1'b1; inj_sel_wdata  <= s_axi_control_wdata; end
                    // pdi23 DETERMINISTIC G8 DUMP seek (0x130): latch the read index.
                    9'h130: g8dump_sel_reg <= s_axi_control_wdata[3:0];
                    // TREE-BATCH 0x138: latch batch-test control (en/negate/lane_sel).
                    9'h138: batch_test_ctrl <= s_axi_control_wdata;
                    // SPEEDWAVE-V2 V1 FR-Spec scan tiles (0x14C): 0 = full V.
                    // Launch-sampled at tree_start in the tiler.
                    // 0x14C is RESERVED (legacy scan register removed): writes ignored.
                    // SPEEDWAVE-V2 K1 runtime K_C (0x150): 0 = compile K_C=512.
                    // Launch-sampled at start in g8_rescore_p.
                    9'h150: kc_cfg <= s_axi_control_wdata[15:0];
                    // F3-CLR KV-RESIDENCY SESSION CLEAR (0x154): bit[0]=1 pulses
                    // a 1-cycle clear of the attention kv_resident bitmap.
                    // NOT 0xEC bit[3] (that clears the boot-load CRC accums).
                    9'h154: kv_res_clear <= s_axi_control_wdata[0];
                    // 0x1125 ACCFIX runtime enables (reset 0 = all OFF).
                    9'h15c: accfix_cfg <= s_axi_control_wdata[3:0];
                    // OBS_LVL ring read index (word granularity).
                    9'h160: obs_rd_idx <= s_axi_control_wdata[10:0];
                    // PMEM instruction data mailbox (0x16C)
                    // PMEM instruction data mailbox (0x16C): write auto-increments write_addr
                    9'h16c: begin
                        pmem_write_data <= s_axi_control_wdata;
                        pmem_we         <= 1'b1;
                    end
                    // PMEM address pointer seek (0x168)
                    9'h168: begin
                        pmem_write_addr <= s_axi_control_wdata[7:0];
                    end
                    // Special-register AXI read selector (0x178). Selects the
                    // bank address returned by the 0x17C readback port.
                    9'h178: sr_read_sel <= s_axi_control_wdata[4:0];
                    default: ;
                endcase
            end
        end
    end

    // ap_done sticky, set by kernel, cleared on read.
    // DONE-CLEAR-AT-LATCH (2026-06-06): the auto-clear now keys on
    // ap_done_clr_pulse (asserted by the read channel at AR-ACCEPT for a 0x00
    // read), NOT on the LIVE s_axi_control_araddr at R-handshake. The old form
    // (`rvalid && rready && araddr==0x00`) sampled the address of whatever AR
    // was being PRESENTED during the R phase, not the address of the read whose
    // data was latched — a pipelined host presenting the next AR while a prior
    // read's R completed could clear ap_done spuriously (missed completion) or
    // miss the clear. The pulse is generated in the same cycle/branch that
    // latched the 0x00 read's rdata (with done still visible), so the clear is
    // exactly paired to the completing read. kernel_done keeps SET-priority: a
    // kernel_done in the same cycle as the clear wins (it stays first here).
    always_ff @(posedge clk) begin
        if (!rst_n)                  ap_done_reg <= 1'b0;
        else if (kernel_done)        ap_done_reg <= 1'b1;
        else if (ap_done_clr_pulse)  ap_done_reg <= 1'b0;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin ap_idle_reg <= 1'b1; ap_ready_reg <= 1'b1; end
        else begin
            ap_idle_reg  <= ~kernel_busy;
            ap_ready_reg <= ~kernel_busy;
        end
    end

    // ----------------------------------------------------------------
    // Read channel
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_axi_control_arready <= 1'b0;
            s_axi_control_rvalid  <= 1'b0;
            s_axi_control_rresp   <= 2'b00;
            s_axi_control_rdata   <= 32'b0;
            probe_data_re         <= 1'b0;   // PROBE-V3 auto-inc pulse
            ap_done_clr_pulse     <= 1'b0;   // DONE-CLEAR-AT-LATCH pulse
            sr_read_en            <= 1'b0;
            sr_rd_state           <= SR_RD_IDLE;
        end else begin
            // PROBE-V3: probe_data_re is a 1-cyc pulse, default low. It is the
            // read of 0xB0 that auto-increments the probe read pointer; latched
            // rdata below captures RAM[rd_ptr] at the PRE-increment address.
            probe_data_re <= 1'b0;
            // PER-STAGE READBACK: dbg_cap_val_re is a 1-cyc pulse, default low. It
            // is the read of 0xE4 that auto-increments the word-index pointer.
            dbg_cap_val_re <= 1'b0;
            // DONE-CLEAR-AT-LATCH (2026-06-06): clear pulse defaults low; it is
            // asserted only in the AR-ACCEPT branch for a 0x00 read (below), so
            // the sticky ap_done is cleared on the SAME read that latched its
            // data (still showing done=1). One driver only (this block).
            ap_done_clr_pulse <= 1'b0;
            // 0x17C bank read strobe is a 1-cyc pulse per attempt.
            sr_read_en <= 1'b0;
            if (sr_rd_state == SR_RD_WAIT) begin
                s_axi_control_arready <= 1'b0;
                if (sr_read_ack) begin
                    // Read committed; read_data updates on this edge.
                    // sr_read_en stays at its default low in the capture cycle.
                    sr_rd_state <= SR_RD_CAP;
                end else begin
                    // A WRSR write won this cycle; retry.
                    sr_read_en <= 1'b1;
                end
            end else if (sr_rd_state == SR_RD_CAP) begin
                // The bank updated read_data on the edge that observed
                // read_ack, so capture it now (one cycle after the ack).
                s_axi_control_arready <= 1'b0;
                s_axi_control_rdata  <= {16'h5A5A, sr_read_data};
                s_axi_control_rvalid <= 1'b1;
                sr_rd_state <= SR_RD_IDLE;
            end else if (s_axi_control_arvalid && !s_axi_control_arready && !s_axi_control_rvalid) begin
                s_axi_control_arready <= 1'b1;
                if (s_axi_control_araddr == 8'hb0) probe_data_re <= 1'b1;
                if (s_axi_control_araddr == 8'he4) dbg_cap_val_re <= 1'b1;
                // DONE-CLEAR-AT-LATCH (2026-06-06): pulse the ap_done auto-clear
                // for the read of 0x00 that is being latched RIGHT NOW. rdata
                // captures ap_ctrl_rdata (with ap_done_reg=1) in this same case
                // below; the clear takes effect one edge later, so this read
                // still returns done=1 and the next 0x00 read returns done=0.
                if (s_axi_control_araddr == 8'h00 && ap_done_reg) ap_done_clr_pulse <= 1'b1;
                case (s_axi_control_araddr)
                    7'h00:   s_axi_control_rdata <= ap_ctrl_rdata;
                    7'h08:   s_axi_control_rdata <= {29'b0, boot_gamma_sel};  // GAMMA-LOAD read-back
                    7'h10:   s_axi_control_rdata <= ctx_len;
                    // G8 SPLICE: lmfull_hbm_base read-backs
                    7'h14:   s_axi_control_rdata <= lmfull_hbm_base[31:0];
                    7'h1c:   s_axi_control_rdata <= lmfull_hbm_base[63:32];
                    7'h18:   s_axi_control_rdata <= num_frontiers;
                    7'h20:   s_axi_control_rdata <= max_depth;
                    7'h28:   s_axi_control_rdata <= out0_base[31:0];
                    7'h2c:   s_axi_control_rdata <= out0_base[63:32];
                    7'h30:   s_axi_control_rdata <= out1_base[31:0];
                    7'h34:   s_axi_control_rdata <= out1_base[63:32];
                    7'h38:   s_axi_control_rdata <= out2_base[31:0];
                    7'h3c:   s_axi_control_rdata <= out2_base[63:32];
                    7'h40:   s_axi_control_rdata <= hbm_emb_base[31:0];
                    7'h44:   s_axi_control_rdata <= hbm_emb_base[63:32];
                    // M9 boot-DMA read-backs + status:
                    // FASTIN: 0x54 readback of the sticky mode bits (bit0 is the
                    // start pulse — always reads 0). Write+readback liveness for
                    // the new is_emb/is_hfc selects.
                    7'h54:   s_axi_control_rdata <= {28'b0, boot_dma_is_hfc, boot_dma_is_emb, boot_dma_is_scale, 1'b0};
                    7'h58:   s_axi_control_rdata <= boot_hbm_base[31:0];
                    7'h5c:   s_axi_control_rdata <= boot_hbm_base[63:32];
                    7'h60:   s_axi_control_rdata <= {{(32-COL_W){1'b0}}, boot_dma_n_cols};
                    7'h64:   s_axi_control_rdata <= {{(32-COL_W){1'b0}}, boot_dma_col_start};
                    7'h68:   s_axi_control_rdata <= {{(32-COL_W){1'b0}}, boot_dma_col_step};
                    7'h6c:   s_axi_control_rdata <= {16'b0, boot_dma_cpc};
                    7'h70:   s_axi_control_rdata <= {15'b0, boot_dma_stage_base};
                    7'h74:   s_axi_control_rdata <= {29'b0, boot_dma_timeout, boot_dma_done, boot_dma_busy};
                    // OUTPUT-DMA status read-back (PDI#7: + bit[3] g8 watchdog):
                    7'h78:   s_axi_control_rdata <= {28'b0, g8_wd_err, out_dma_err, out_dma_done, out_dma_busy};
                    // C5 (B1) input-emb target-select read-back:
                    7'h7c:   s_axi_control_rdata <= {31'b0, boot_is_emb};
                    // FIX-E h_fc target-select read-back:
                    7'h24:   s_axi_control_rdata <= {31'b0, boot_is_hfc};
                    // PDI#7 SILICON-HARDENING debug registers (8-bit addr):
                    8'h80:   s_axi_control_rdata <= dbg_fsmstate;   // FSM-state mirror
                    8'h84:   s_axi_control_rdata <= dbg_activity;   // per-stage activity
                    // DEBUG BLOCK V2 (PDI#8) read-backs:
                    8'h88:   s_axi_control_rdata <= {28'b0, dbg_sel}; // stage-select read-back
                    8'h8c:   s_axi_control_rdata <= dbg_cyclecount;  // selected-stage cycle count
                    8'h90:   s_axi_control_rdata <= dbg_checksum;    // selected input checksum
                    8'h94:   s_axi_control_rdata <= dbg_g8progress;  // G8 progress
                    8'h98:   s_axi_control_rdata <= dbg_g8araddr_lo; // last G8 araddr lo
                    8'h9c:   s_axi_control_rdata <= dbg_g8araddr_hi; // last G8 araddr hi + rresp
                    8'ha0:   s_axi_control_rdata <= dbg_rearm;       // ap_start/tree_done counts
                    8'ha4:   s_axi_control_rdata <= dbg_bootstream;  // boot stream wordcount+XOR
                    // PROBE V3 (0xA8..0xBC) read-backs:
                    8'ha8:   s_axi_control_rdata <= {28'b0, probe_sel}; // stage-select read-back
                    8'hac:   s_axi_control_rdata <= probe_addr;        // current read pointer
                    8'hb0:   s_axi_control_rdata <= probe_data;        // RAM[probe_addr] (auto-inc)
                    8'hb4:   s_axi_control_rdata <= probe_xor;         // stream XOR
                    8'hb8:   s_axi_control_rdata <= probe_sum;         // stream sum32
                    8'hbc:   s_axi_control_rdata <= probe_count;       // word count
                    // PROBE V3 dump-enable + out3/out4 base read-backs:
                    8'hc0:   s_axi_control_rdata <= {31'b0, dump_en};
                    8'hc4:   s_axi_control_rdata <= out3_base[31:0];
                    8'hc8:   s_axi_control_rdata <= out3_base[63:32];
                    8'hcc:   s_axi_control_rdata <= out4_base[31:0];
                    8'hd0:   s_axi_control_rdata <= out4_base[63:32];
                    // PDI#10 PROBE EXPANSION: per-stage-output content sample/count.
                    8'hd4:   s_axi_control_rdata <= dbg_stageout_sample;
                    8'hd8:   s_axi_control_rdata <= dbg_stageout_count;
                    // PER-STAGE VALUE READBACK (capture-all-in-one-forward):
                    8'hdc:   s_axi_control_rdata <= {27'b0, dbg_cap_stage_sel}; // stage select
                    8'he0:   s_axi_control_rdata <= dbg_cap_widx_word;          // current word idx
                    8'he4:   s_axi_control_rdata <= dbg_cap_val_word;           // cap[stage][idx]
                    8'he8:   s_axi_control_rdata <= dbg_cap_meta_word;          // {captured,nwords}
                    // PROBE_EXPANSION RESIDENCY CRC read-backs:
                    8'hec:   s_axi_control_rdata <= {29'b0, res_sel};           // buffer select
                    8'hf0:   s_axi_control_rdata <= res_crc_word;               // CRC signature
                    8'hf4:   s_axi_control_rdata <= res_sum_word;               // running SUM
                    8'hf8:   s_axi_control_rdata <= res_count_word;             // word count
                    // MAXPROBE LIVENESS WINDOW (9-bit araddr; 0x100/0x104/0x108):
                    9'h100:  s_axi_control_rdata <= maxprobe_build_id;          // {0xE4B7,TAG,REV} const
                    9'h104:  s_axi_control_rdata <= ~selftest_reg;              // bitwise-NOT self-test readback
                    9'h108:  s_axi_control_rdata <= cap_crc_word;               // per-stage CRC (0xDC sel)
                    // PER-STAGE INJECT read-backs (write+readback liveness + status):
                    9'h10c:  s_axi_control_rdata <= inj_en_word;                // live inject_en bitmask
                    9'h110:  s_axi_control_rdata <= inj_widx_word;              // buffer write pointer
                    9'h11c:  s_axi_control_rdata <= inj_status_word;            // {0x1A canary,present,sel,lo}
                    // M7 GLM=0 WIRED-TAP LOCALIZER (0x120/0x124/0x128):
                    9'h120:  s_axi_control_rdata <= lmtap_seam_word;            // seam_r8 BRAM DOUT lane0 (pre-guard)
                    9'h124:  s_axi_control_rdata <= lmtap_guard_word;           // rd_guard_q
                    9'h128:  s_axi_control_rdata <= lmtap_ad_word;              // a_d lane0 (MUL A)
                    // pdi23 DETERMINISTIC G8 FRONTIER/CAND DUMP (0x130 sel / 0x134 word):
                    9'h130:  s_axi_control_rdata <= {26'b0, g8_dump_status, g8dump_sel_reg};  // {status,sel} liveness
                    9'h134:  s_axi_control_rdata <= g8_dump_flat[g8dump_sel_reg*32 +: 32];    // sticky id @ sel
                    // TREE-BATCH B=2 batch-test window (0x138..0x148):
                    9'h138:  s_axi_control_rdata <= {8'hBA, 8'b0, batch_test_ctrl[15:0]};     // ctrl readback + canary
                    9'h13c:  s_axi_control_rdata <= batch_b0_lo;                              // b0  result_q lo
                    9'h140:  s_axi_control_rdata <= batch_b0_hi;                              // b0  result_q hi + canary
                    9'h144:  s_axi_control_rdata <= batch_sib_lo;                             // sib result_q lo
                    9'h148:  s_axi_control_rdata <= batch_sib_hi;                             // sib result_q hi + canary
                    // SPEEDWAVE-V2 runtime-config liveness read-backs
                    // (feedback_probe_reg_needs_real_axi_write_and_liveness):
                    9'h14c:  s_axi_control_rdata <= 32'b0;                    // reserved
                    9'h150:  s_axi_control_rdata <= {16'b0, kc_cfg};         // K1 read-back
                    // F3-CLR 0x154 read-back: {"KV" canary, 0, live kv_res_cnt}.
                    // Host verifies the clear TOOK EFFECT: count must read 0
                    // immediately after the pulse, and the canary proves the
                    // decode is alive (RAZ/aliased bus cannot fake 0x4B56).
                    9'h154:  s_axi_control_rdata <= kv_res_status_word;
                    // 0x1125 ACCFIX read-back: {canary 0xACF1, 0, cfg[3:0]}.
                    // The canary proves the decode is alive (RAZ cannot fake it).
                    9'h15c:  s_axi_control_rdata <= {16'hACF1, 12'b0, accfix_cfg};
                    // OBS_LVL: status (canary+count) and ring data word.
                    9'h160:  s_axi_control_rdata <= obs_status_word;
                    9'h164:  s_axi_control_rdata <= obs_rd_data;
                    // ISA PMEM status / read-back (0x168: write_addr, 0x16C: {canary, PC, fetch_instr})
                    9'h168:  s_axi_control_rdata <= {24'b0, pmem_write_addr};
                    9'h16c:  s_axi_control_rdata <= {16'h504D, 1'b0, fetch_pc, fetch_instr};
                    // Shared GEMV engine status: bit0 BUSY, bit1 sticky DONE,
                    // bit2 sticky ERROR. 0x174: active engine tile index.
                    9'h170:  s_axi_control_rdata <= {29'b0, gemv_error_sticky,
                                                     gemv_done_sticky, gemv_busy};
                    9'h174:  s_axi_control_rdata <= {16'b0, gemv_active_tile};
                    9'h178:  s_axi_control_rdata <= {27'b0, sr_read_sel};
                    9'h17c: begin
                        // Selector was latched by a previous 0x178 write.
                        // Issue a synchronous bank read; if a WRSR write wins
                        // this cycle, read_ack stays low and the WAIT state
                        // retries instead of latching stale read_data.
                        sr_read_en  <= 1'b1;
                        sr_rd_state <= SR_RD_WAIT;
                    end
                    default: s_axi_control_rdata <= 32'b0;
                endcase
                // 0x17C defers its R response until the bank read commits.
                if (s_axi_control_araddr != 9'h17c)
                    s_axi_control_rvalid <= 1'b1;
            end else begin
                s_axi_control_arready <= 1'b0;
                if (s_axi_control_rvalid && s_axi_control_rready) s_axi_control_rvalid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
