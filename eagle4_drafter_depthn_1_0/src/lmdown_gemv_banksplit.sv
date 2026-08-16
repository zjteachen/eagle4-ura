// =====================================================================
// lmdown_gemv.sv  —  FIX-G dense INT16 GEMV  (g11_lmdown: 3072 -> 256)
// =====================================================================
//
// The reference's g11_lmdown (chain.cpp:154-156, gemv.cpp:36-45):
//     low_rank[m] = Sum_{k=0..3071} W[m,k] * x[k]      (PyTorch nn.Linear, bias=False)
// where W = efficient_lm_head_down_proj.weight, fp16 [256,3072], NO bias,
// x = to_logits_norm = G11 RMSNorm output (seam_r8, 3072-wide).
//
// On the FPGA this is realized as a DENSE INT16 GEMV (NO per-group GPTQ scale,
// NO 2-bit unpack, NO bias) — the reference's fp32-acc of fp16 weights becomes
// an INT16 x INT16 -> INT accumulate, rescaled to the output frac at the end.
// It is a NEW minimal primitive (FIXG_LMDOWN_SPEC §0/§2): the shared 128-lane
// GPTQ array cannot serve a dense fp16 weight (no scale, no unpack path).
//
// Numerical contract (FIXG_LMDOWN_SPEC §4):
//   input  a_chunk = seam_r8 chunk, INT16 Q6.10 (frac=10).            (IN_FRAC)
//   weight w       = lmdown_W pre-converted to INT16 Q1.15 (frac=15). (W_FRAC)
//   acc            = Sum a*w  (INT, acc_frac = IN_FRAC + W_FRAC = 25)
//   output low_rank = act_rescale(acc) -> INT16 Q5.11 (frac=11), 0 sat. (OUT_FRAC)
//   => the single requant shift = ACC_FRAC - OUT_FRAC.
//
// Microarchitecture (FIXG_LMDOWN_SPEC §2.2, 16-lane default):
//   M=256 outputs processed as N_GRP=16 GROUPS of LANES=16 outputs.
//   For group g (0..15), lane L (0..15) computes output m = g*LANES + L.
//   Each group streams all NCHUNKS=192 chunks (K_PAR=16 elems/cycle), each
//   lane does a K_PAR-wide INT16xINT16 MAC reduced by a PIPELINED signed adder
//   tree (the dense product is INT32, not INT20 — see LMDOWN_SYNTH_FIX §FIX2),
//   accumulated across chunks. After a group's 192 chunks + pipeline drain the
//   16 lane outputs are rescaled (act_rescale) and committed to low_rank_buf
//   chunk g. Latency ~= N_GRP*(NCHUNKS + PIPE_LAT + drain).
//
// Weight storage (LMDOWN_SYNTH_FIX §FIX1, supersedes the original §5 URAM plan):
//   each lane owns a private 256-bit-wide `ram_style=block` (BRAM) ROM bank,
//   depth N_GRP*NCHUNKS = 16*192 = 3072 rows.  Lane L's bank row (g*NCHUNKS + c)
//   holds W[g*LANES+L, c*K_PAR .. c*K_PAR+15] (16 INT16 Q1.15 weights).  The
//   banks are BAKED from /tmp/p4/lmdown_W_q15.hex via a synth-visible $readmemh
//   so the netlist flashes pre-loaded.  BRAM (27.6% used) sidesteps the URAM
//   overflow the `ram_style=ultra` plan hit (the GEMV array already needs 1,536
//   URAM); a registered-read BRAM ROM is the idiom Vivado reliably infers.
//
// Multi-driver-safe: ONE always_ff per bank, per pipeline reg, per output reg.
// Drain discipline: glm_done pulses one cycle after the last group commits;
// out_valid (low_rank_buf write strobe) is high only on real commit cycles.
//
// LMDOWN_SYNTH_FIX (2026-05-27) edits vs the frozen FIX-G version:
//   FIX1: wbank ram_style "ultra" -> "block" (BRAM ROM), per-lane split banks
//         (each 786,432 bits < the 1,000,000-bit single-variable synth limit,
//         Synth 8-4556) + per-lane registered read.
//   FIX2: the 16-input INT32 sum-reduction (44 logic levels, WNS -1.526 ns)
//         is now a PIPELINED registered adder tree (SUMA: 16->4, SUMB: 4->1),
//         +2 latency cycles. Markers + FLUSH drain extended; arithmetic & the
//         final acc value are bit-IDENTICAL (only register boundaries added).
//   B/C/D/E/F (scratch oocfit2): the `string hex_path` local, the >65536-iter
//         pack loop, and the single 12.58 Mbit array are all gone (per-lane
//         banks + per-lane 49,152-iter bakes + W_HEX param read directly).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module lmdown_gemv_banksplit #(
    parameter int K        = 3072,           // input dim (to_logits_norm)
    parameter int K_PAR    = 16,             // elems per chunk
    parameter int A_WIDTH  = 16,             // activation element width
    parameter int M        = 256,            // output dim (low_rank)
    parameter int LANES    = 16,             // parallel output lanes
    parameter int IN_FRAC  = 10,             // seam_r8 frac (Q6.10)
    parameter int W_FRAC   = 15,             // baked weight frac (Q1.15)
    parameter int OUT_FRAC = 10,             // low_rank frac (Q5.11)
    parameter string W_HEX = "/home/y344shi/eagle4_artifacts/p4/lmdown_W_q15.hex",
    parameter string W_MEM_DIR = "/home/y344shi/eagle4_artifacts/p4",
    // -----------------------------------------------------------------
    // WO-051 P2/P3 SLR-seam fix (LMDOWN_SLR_REG). Worst-path-2 #3 is the
    // lm-down FSM->DSP arc: the FSM (state/chunk/grp) sits in one SLR and
    // its control (a_chunk_idx_valid gating the seam_r8 activation fetch,
    // and the activation a_chunk itself) crosses a die cut to reach the
    // per-lane MUL DSP (prod_reg = a_d * w_row) in another SLR — 97.5% of
    // the −0.779 ns is route.
    //   LMDOWN_SLR_REG=0 (DEFAULT) : the historical pipeline, bit-AND-cycle
    //       identical to the current golden / tb_lmdown_gemv_unit goldens.
    //   LMDOWN_SLR_REG=1 : insert ONE registered RD-launch stage on the
    //       SLR-crossing seam. The activation a_chunk, the weight-bank
    //       read address, AND the RD-stage markers (run_valid/first/last/grp)
    //       are ALL delayed by the SAME +1 cycle, so a(c) still meets w(c)
    //       at the MUL DSP with identical markers — the product, the SUMA/
    //       SUMB tree, the accumulator, and the rescale are arithmetically
    //       UNCHANGED (BIT-EXACT). Only WHEN each chunk launches shifts by
    //       +1; the S_FLUSH drain + done-driven handshake absorb the extra
    //       cycle. a_chunk_idx_valid (the seam read strobe) is also delayed
    //       +1 so the external activation fetch co-launches with the
    //       registered internal read (valid AND data registered together,
    //       feedback_valid_data_latency_skew_class).
    // -----------------------------------------------------------------
    parameter bit LMDOWN_SLR_REG = 1'b0,
    // -----------------------------------------------------------------
    // SEAM_FF_LAND (2026-06-21, pdi13_ffact FF_ACT_CACHE integration #2) —
    //   per-SLR FF LANDING register on the lmdown activation input `a_chunk`
    //   (= seam_r8_rd, the G11-lmdown cross-SLR seam read). GLM's lanes then
    //   consume a LOCAL FF copy of h, never a cross-SLR BRAM read.
    //   SEAM_FF_LAND=0 (DEFAULT): wire-through, byte+cycle identical to today.
    //   SEAM_FF_LAND=1: register a_chunk ONCE on entry (a_chunk_land), and
    //   add ONE MORE matched delay stage on the WEIGHT + MARKER arm
    //   (w_row_d2 / *_rd3) so the activation's +1 landing latency is absorbed
    //   and a(c) still meets w(c) at the MUL DSP -> BIT-EXACT, +1 latency
    //   (absorbed by S_FLUSH drain / done handshake). Requires LMDOWN_SLR_REG=1
    //   (the live silicon config — the +1 realign machinery it extends).
    // -----------------------------------------------------------------
    parameter bit SEAM_FF_LAND = 1'b0,
    // -----------------------------------------------------------------
    // COTRAVEL (2026-06-24, A4 slip-invariant fix — GLM_FLEET_A4_RDLAT.md,
    // proven by tb_cotravel_proof.sv). The historical realign
    // (LMDOWN_SLR_REG / SEAM_FF_LAND) launches the weight ROM read off the FSM
    // cadence and ASSUMES the activation lands at a fixed latency, then adds
    // matched whole-256b delays (w_row_d / w_row_d2) to cancel it. That cancels
    // only a UNIFORM cadence skew at the SIM number — a PHYSICAL per-bank /
    // wide-net SLR slip (the +128 class) lands the activation a DIFFERENT number
    // of cycles late than assumed, and the matched delay can no longer equalize
    // it -> a(c±1)·W(c) -> GLM wrong/zero on silicon (xsim/STA blind).
    //
    // COTRAVEL=1 REMOVES the latency assumption: the weight ROM is addressed by
    // the chunk INDEX that ARRIVES WITH the activation (data-driven addressing),
    // and the MUL/ACC markers ride the DELIVERED valid — NOT the FSM launch
    // cadence. activation and weight are then the SAME chunk BY CONSTRUCTION for
    // ANY seam read latency / per-bank skew. This REPLACES the SEAM_FF_LAND /
    // LMDOWN_SLR_REG matched-delay realign internally (w_eff = w_row, NO
    // w_row_d/w_row_d2 stage) -> the co-travel ROM read IS the alignment.
    //
    // The delivered index/valid come from the seam's NEW co-travel outputs
    // (a_chunk_idx_in / a_chunk_idx_valid_in = u_seam_r8.rd_chunk_idx_out /
    // rd_valid_out), which travel through the seam's SAME RD_LAT pipe as the
    // data. When those ports are UNCONNECTED (e.g. the shared cadence-robust
    // gate TB that uses the OLD port list -> a_chunk_idx_valid_in tied 0),
    // COTRAVEL gracefully FALLS BACK to reconstructing the delivered index
    // INTERNALLY by delaying chunk_eff/grp_eff/run_eff by the activation-arm
    // latency (ACT_SEAM_LAT = LMDOWN_SLR_REG + SEAM_FF_LAND), so the dot is
    // correct WITHOUT any TB wiring; the new seam ports are then a
    // belt-and-suspenders for the REAL physical silicon slip.
    //
    // COTRAVEL=0 : byte-identical to the existing pdi17 LMDOWN_SLR_REG /
    //              SEAM_FF_LAND matched-delay path (the A/B no-regress proof).
    parameter bit COTRAVEL = 1'b1,
    // derived (do not override)
    localparam int NCHUNKS  = K / K_PAR,                 // 192
    localparam int CW       = (NCHUNKS > 1) ? $clog2(NCHUNKS) : 1, // 8
    localparam int N_GRP    = M / LANES,                 // 16
    localparam int GW       = (N_GRP > 1) ? $clog2(N_GRP) : 1,     // 4
    localparam int BANK_DEPTH = N_GRP * NCHUNKS,         // 3072 rows per lane
    localparam int BAW      = $clog2(BANK_DEPTH),        // 12
    localparam int ROW_W    = K_PAR * A_WIDTH,           // 256
    localparam int ACC_FRAC = IN_FRAC + W_FRAC,          // 25
    localparam int OCW      = (N_GRP > 1) ? $clog2(N_GRP) : 1,
    // -----------------------------------------------------------------
    // BANK-SPLIT (2026-06-19, LMDOWN_BANKSPLIT — GLM_ZERO_ROOTCAUSE.md §6).
    // The 256-bit lmdown operand-delivery nets (activation a_d and weight
    // w_row_d feeding the per-lane MUL) are placement-FRAGILE: a single wide
    // 256b net that the router floats across an SLR (or a hard-excluding pblock
    // carves out) can SLR-slip / GND-tie / alias -> GLM zeros on silicon
    // (xsim/STA blind, the +128-proven class). REMEDY: split each 256b operand
    // net into BANKS of BANK_W=32 bits, each captured in its OWN narrow register
    // (a true reg-to-reg crossing the router/SLL can realize). The MUL operands
    // are RE-ASSEMBLED from the NB narrow registered crossings -> NO wide net
    // spans an SLR, and the +1 marker alignment is preserved BY CONSTRUCTION
    // regardless of placement -> needs NO pblock. BANK_W=32 carries exactly 2
    // INT16 elements per bank (A_WIDTH=16), NB=256/32=8 banks. The register
    // BOUNDARY and LATENCY are UNCHANGED vs the monolithic a_d / w_row_d (1
    // stage), so the captured value is BIT-IDENTICAL -> GLM cap-CRC stays
    // 0x859b57b1 (latency-only structural restructure).
    localparam int BANK_W   = 32,                        // bank width (2 INT16)
    localparam int NB       = ROW_W / BANK_W,            // 8 banks per 256b net
    localparam int EPB      = BANK_W / A_WIDTH,           // 2 elements per bank
    // DBG_TAP (2026-06-20, M7 GLM=0 WIRED-TAP LOCALIZER): expose a_d lane-0 (the MUL
    // A operand, post-capture) as a READ-ONLY boundary port for the AXI-Lite localizer.
    // DBG_TAP=0 => tied 0 -> DCE -> byte-identical to no-tap md5.
    parameter bit DBG_TAP = 1'b0
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,        // one-cycle pulse (glm_start)

    // streaming activation: ONE chunk (K_PAR x A_WIDTH) per cycle = seam_r8 chunk
    input  wire signed [K_PAR*A_WIDTH-1:0]   a_chunk,
    // which 16-elem chunk of the 3072-wide input is being presented this cycle
    output wire [CW-1:0]                     a_chunk_idx,  // drives the seam read ptr
    output wire                              a_chunk_idx_valid,

    // ---- CO-TRAVEL delivered index/valid (A4 fix) -------------------
    // The chunk index + valid that physically RODE WITH the activation through
    // the seam (= u_seam_r8.rd_chunk_idx_out / rd_valid_out, piped through the
    // seam's SAME RD_LAT registers as a_chunk). When COTRAVEL=1 the weight ROM
    // is addressed by this DELIVERED index so weight(delivered)·a_d(delivered)
    // meet at the MUL regardless of any seam read-latency slip. OPTIONAL: if
    // a_chunk_idx_valid_in is never asserted (these ports unconnected => default
    // 0, e.g. the shared cadence-robust gate TB), COTRAVEL reconstructs the
    // delivered index INTERNALLY (delay chunk_eff by ACT_SEAM_LAT). Ignored when
    // COTRAVEL=0. lmdown STILL drives a_chunk_idx/_valid (the seam read pointer).
    input  wire [CW-1:0]                     a_chunk_idx_in,
    input  wire                              a_chunk_idx_valid_in,

    // low_rank_buf write interface (consumer side reads this buffer chunk-indexed)
    output reg                               wr_valid,     // one pulse per committed group
    output reg  [OCW-1:0]                    wr_chunk_idx, // group index 0..N_GRP-1
    output reg  signed [LANES*A_WIDTH-1:0]   wr_chunk,     // 16 INT16 Q5.11 outputs

    output reg                               done,         // all M outputs committed

    // ---- DBG_TAP wired localizer (READ-ONLY; DBG_TAP=0 => DCE) ------
    output wire        [15:0]                dbg_a_d0,     // a_d lane0 (MUL A operand, post-capture)
    output wire                              dbg_run_valid // run_eff (GLM S_RUN active)
);

    // -----------------------------------------------------------------
    // bank read address for the CURRENT (group, chunk) — drives all lanes.
    // row = grp*NCHUNKS + chunk.  Declared up here so the per-lane generate
    // banks (below) can use it for their registered read.
    // -----------------------------------------------------------------
    reg [GW-1:0] grp,        grp_n;        // current output group 0..N_GRP-1
    reg [CW-1:0] chunk,      chunk_n;      // current chunk 0..NCHUNKS-1
    reg [3:0]    flush_cnt,  flush_cnt_n;  // pipeline drain countdown after last chunk

    // -----------------------------------------------------------------
    // LMDOWN_SLR_REG seam: the FSM-driven read launch (which chunk/group is
    // active, and whether we are running) is the SLR-crossing control. When
    // enabled we register chunk/grp/run/first/last ONE cycle into *_eff and
    // drive EVERYTHING that feeds the read off *_eff — the external
    // activation fetch (a_chunk_idx / a_chunk_idx_valid), the weight-bank ROM
    // address (bank_addr_full), and the RD-stage markers. Because the SAME
    // delayed chunk drives both the activation request and the weight read,
    // a(c) still meets w(c) at the MUL DSP — pure +1 launch delay, bit-exact.
    // When disabled, *_eff are wire-throughs (zero added latency, identical).
    // -----------------------------------------------------------------
    wire [GW-1:0] grp_eff;
    wire [CW-1:0] chunk_eff;
    wire          run_eff;     // == "this cycle's RD launch is a real run cycle"
    // (seam register block defined after `state` is declared, below)

    // -----------------------------------------------------------------
    // CO-TRAVEL delivered (index, group, valid) — the chunk whose ACTIVATION is
    // arriving at a_chunk THIS cycle (so the ROM addressed by it produces the
    // matching weight one cycle later, the same cycle a_d holds that activation).
    //   ACT_SEAM_LAT = the activation-arm latency from chunk_eff to a_lin:
    //     seam RD_LAT (=LMDOWN_SLR_REG, the live u_seam_r8 read register) +
    //     SEAM_FF_LAND (the per-SLR landing register on a_chunk).
    //   delivered = EXTERNAL co-travel port (a_chunk_idx_in/_valid_in) when it
    //   is asserted (the seam delivered it, capturing any PHYSICAL slip); else
    //   INTERNAL reconstruction = chunk_eff/grp_eff/run_eff delayed ACT_SEAM_LAT
    //   cycles (the nominal cadence, no TB wiring required).
    // grp is NOT delivered by the seam (chunk-only co-travel), so grp_dlv is
    // always the internally-delayed grp_eff (group is monotone, changes only at
    // group boundaries where the FLUSH drain de-asserts the valid).
    // -----------------------------------------------------------------
    localparam int ACT_SEAM_LAT = (LMDOWN_SLR_REG ? 1 : 0) + (SEAM_FF_LAND ? 1 : 0);

    // internal reconstruction shift regs (chunk/grp/run delayed ACT_SEAM_LAT).
    wire [CW-1:0] chunk_dly;
    wire [GW-1:0] grp_dly;
    wire          run_dly;
    generate
    if (ACT_SEAM_LAT == 0) begin : g_ct_dly0
        assign chunk_dly = chunk_eff;
        assign grp_dly   = grp_eff;
        assign run_dly   = run_eff;
    end else begin : g_ct_dlyN
        reg [CW-1:0] csh [0:ACT_SEAM_LAT-1];
        reg [GW-1:0] gsh [0:ACT_SEAM_LAT-1];
        reg          rsh [0:ACT_SEAM_LAT-1];
        integer di;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                for (di = 0; di < ACT_SEAM_LAT; di = di + 1) begin
                    csh[di] <= '0; gsh[di] <= '0; rsh[di] <= 1'b0;
                end
            end else begin
                csh[0] <= chunk_eff; gsh[0] <= grp_eff; rsh[0] <= run_eff;
                for (di = 1; di < ACT_SEAM_LAT; di = di + 1) begin
                    csh[di] <= csh[di-1]; gsh[di] <= gsh[di-1]; rsh[di] <= rsh[di-1];
                end
            end
        end
        assign chunk_dly = csh[ACT_SEAM_LAT-1];
        assign grp_dly   = gsh[ACT_SEAM_LAT-1];
        assign run_dly   = rsh[ACT_SEAM_LAT-1];
    end
    endgenerate

    // X-SAFE external co-travel valid. The new co-travel ports are OPTIONAL: when
    // they are left UNCONNECTED (the shared cadence-robust gate TB uses the OLD
    // port list), xsim drives the input to X, not 0. Coerce an X external valid to
    // 0 (SIM-ONLY guard) so COTRAVEL deterministically FALLS BACK to the internal
    // reconstruction (chunk_dly) instead of propagating X into the address/markers.
    // In synthesis the top ALWAYS drives this port (0/1) so the guard is a no-op
    // and is compiled out under `SYNTHESIS -> byte-identical netlist.
    wire ext_v_raw = a_chunk_idx_valid_in;
`ifdef SYNTHESIS
    wire ext_v = ext_v_raw;
`else
    wire ext_v = (ext_v_raw === 1'bx || ext_v_raw === 1'bz) ? 1'b0 : ext_v_raw;
`endif
    // delivered index/valid: external co-travel port if it is asserted, else the
    // internally-reconstructed nominal index. grp always internal (chunk-only port).
    wire          dlv_use_ext = COTRAVEL && ext_v;
    wire [CW-1:0] chunk_dlv   = dlv_use_ext ? a_chunk_idx_in : chunk_dly;
    wire          run_dlv     = COTRAVEL ? (ext_v | run_dly) : run_eff;
    wire [GW-1:0] grp_dlv     = grp_dly;

    // bank address: CO-TRAVEL addresses the ROM by the DELIVERED (index, group)
    // so weight(delivered) lands aligned with a_d(delivered) at the MUL. Legacy
    // (COTRAVEL=0) addresses by the FSM launch (grp_eff,chunk_eff) as before.
    wire [GW-1:0]  bank_grp   = COTRAVEL ? grp_dlv   : grp_eff;
    wire [CW-1:0]  bank_chunk = COTRAVEL ? chunk_dlv : chunk_eff;
    wire [BAW-1:0] bank_addr_full = bank_grp * NCHUNKS + bank_chunk;

    // registered weight read (BRAM ROM contract); ONE w_row[L] per lane,
    // driven inside the per-lane generate block (FIX1).
    reg signed [ROW_W-1:0]       w_row [0:LANES-1];

    // -----------------------------------------------------------------
    // FIX1 — Per-lane weight banks as BRAM ROM (ram_style="block"), baked.
    // The original single 3D array
    //   reg [ROW_W-1:0] wbank [0:LANES-1][0:BANK_DEPTH-1];
    // is 256b x 16 x 3072 = 12,582,912 bits -> exceeds Vivado synth's
    // 1,000,000-bit single-variable limit (Synth 8-4556), so the WHOLE thing
    // failed to synthesize. Split into LANES per-lane banks (each 256b x 3072
    // = 786,432 bits < 1M), one per generate scope, each baking ONLY its own
    // rows (49,152 < 65,536 iter cap, Synth 8-6896).  ram_style="block" infers
    // BRAM (not URAM) — sidesteps the URAM overflow; a registered-read BRAM ROM
    // is the idiom Vivado reliably infers (no "no driver" drop, Blocker A).
    // bank[g*NCHUNKS + c] = { W[g*LANES+gl, c*K_PAR+15], ..., W[g*LANES+gl, c*K_PAR] }.
    // -----------------------------------------------------------------
    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : g_wbank
            // FIX-3 (idiom C — xpm_memory_sprom; 2026-05-27, LMDOWN_WBANK_FIX).
            // The inferred-array forms ALL failed to land in BRAM on this part:
            //   - procedural temp->slice $readmemh  -> [Synth 8-3848] no-driver (dropped, FIX-1)
            //   - direct $readmemh + ram_style/rom_style="block" (idiom A) -> SURVIVES
            //     (zero 8-3848) but Vivado maps the 3072x256 ROM to LUT distributed
            //     ROM (258K LUT6), NOT BRAM — the 256-bit word width defeats ROM->BRAM
            //     inference for both ram_style AND rom_style.
            //   - never-firing write (idiom D) -> [Synth 8-3391] "memory too large to
            //     dissolve" elaboration error.
            // The reliable wide-initialized-ROM primitive is the XPM ROM macro: it
            // cascades RAMB36s for the 256-bit word automatically and carries the
            // init via MEMORY_INIT_FILE. Per-lane init = the packed .mem produced by
            // scripts/pack_lmdown_hex.py (3072 lines x 256-bit, line r=g*NCHUNKS+c for
            // output row m=g*LANES+gl, sub-lane15..0 MSB..LSB == wbank[r][ii*16+:16]) —
            // byte-identical weights to the old slice-pack -> bit-exact preserved.
            // READ_LATENCY_A=1 gives a 1-cycle registered read == the old w_row[gl]
            // RD-stage timing (douta valid one cycle after addra).
            wire [ROW_W-1:0] douta_gl;
            xpm_memory_sprom #(
                .ADDR_WIDTH_A      (BAW),
                .MEMORY_SIZE       (BANK_DEPTH * ROW_W),   // bits
                .MEMORY_PRIMITIVE  ("block"),
                .MEMORY_INIT_FILE  ($sformatf("%s/lmdown_W_q15_lane%0d.mem", W_MEM_DIR, gl)),
                .MEMORY_INIT_PARAM (""),
                .READ_DATA_WIDTH_A (ROW_W),
                .READ_LATENCY_A    (1),
                .READ_RESET_VALUE_A("0"),
                .ECC_MODE          ("no_ecc"),
                .AUTO_SLEEP_TIME   (0),
                .USE_MEM_INIT      (1),
                .WAKEUP_TIME       ("disable_sleep"),
                .MESSAGE_CONTROL   (0)
            ) u_wbank (
                .douta   (douta_gl),
                .addra   (bank_addr_full),
                .clka    (clk),
                .ena     (1'b1),
                .regcea  (1'b1),
                .rsta    (1'b0),
                .sleep   (1'b0),
                .injectsbiterra(1'b0),
                .injectdbiterra(1'b0),
                .sbiterra(),
                .dbiterra()
            );
            // alias the XPM read output into the shared w_row[gl] (== old RD stage).
            always_comb w_row[gl] = $signed(douta_gl);
        end
    endgenerate

    // -----------------------------------------------------------------
    // FSM: per group g, stream NCHUNKS chunks; lanes MAC -> PIPELINED tree -> acc;
    // on the last chunk drain the pipeline, rescale + commit chunk g; advance grp.
    // -----------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE, S_RUN, S_FLUSH, S_DONE } state_t;
    state_t state, state_n;

    // -----------------------------------------------------------------
    // LMDOWN_SLR_REG seam register (declared here, after `state`). Delays
    // the RD launch select by +1 when enabled; pure wire-through otherwise.
    // -----------------------------------------------------------------
    generate
    if (LMDOWN_SLR_REG) begin : g_rd_seam
        (* keep = "true", shreg_extract = "no" *) reg [GW-1:0] grp_q;
        (* keep = "true", shreg_extract = "no" *) reg [CW-1:0] chunk_q;
        (* keep = "true", shreg_extract = "no" *) reg          run_q;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                grp_q <= '0; chunk_q <= '0; run_q <= 1'b0;
            end else begin
                grp_q   <= grp;
                chunk_q <= chunk;
                run_q   <= (state == S_RUN);
            end
        end
        assign grp_eff   = grp_q;
        assign chunk_eff = chunk_q;
        assign run_eff   = run_q;
    end else begin : g_rd_passthru
        assign grp_eff   = grp;
        assign chunk_eff = chunk;
        assign run_eff   = (state == S_RUN);
    end
    endgenerate

    // a_chunk is presented in lockstep with the EFFECTIVE (possibly +1
    // delayed) RD-launch chunk, so the external seam fetch and the internal
    // weight-bank read both reference the same chunk.
    assign a_chunk_idx       = chunk_eff;
    assign a_chunk_idx_valid = run_eff;

    // -----------------------------------------------------------------
    // Per-lane MAC + PIPELINED reduce.  The dense INT16xINT16 product is INT32
    // (a*w up to ~2^30), so we use a direct signed adder tree (NOT csa_tree16_int20,
    // which is sized for INT20 GPTQ products).  The dense GEMV is 256x3072 = 0.79M
    // MACs, negligible resource.
    //
    // FIX2 (LMDOWN_SYNTH_FIX): the original SUM stage summed all 16 INT32 products
    // in ONE combinational cycle -> synthesized as a 16-deep DSP-ALU cascade
    // (44 logic levels, WNS -1.526 ns @125 MHz).  It is now a PIPELINED registered
    // adder tree across TWO extra register stages:
    //   SUMA : 16 INT32 -> 4 INT34 partials (2 binary-add levels)   [+1 cycle]
    //   SUMB : 4  INT34 -> 1 INT40 sum       (2 binary-add levels)   [+1 cycle]
    // Each register stage is <=2 adder levels deep -> closes 125 MHz.  The final
    // sum_reg value is ARITHMETICALLY IDENTICAL to the old single-cycle 16-input
    // add (associative integer addition) -> the acc + rescale + commit are
    // bit-EXACT.  The extra +2 latency is absorbed by the longer marker shift
    // chain + the deeper FLUSH drain (done-driven handshake auto-extends glm_done).
    //
    // Stage layout (project-standard registered ROM read):
    //   t   : present a_chunk(c), bank_addr(c)                       [S_RUN, chunk=c]
    //   t+1 : w_row[L] <= W(c) (registered ROM read), a_d <= a(c)    [RD stage]
    //   t+2 : prod_reg <= a_d * w_row                                [MUL stage]
    //   t+3 : suma_reg <= pairwise+pairwise of 16 prods -> 4         [SUMA stage]
    //   t+4 : sum_reg  <= reduce 4 -> 1                              [SUMB stage]
    //   t+5 : acc      <= (first? sum : acc+sum)                     [ACC stage]
    // -----------------------------------------------------------------
    // RD-stage markers derive from the EFFECTIVE (possibly +1 delayed) RD
    // launch so they stay co-aligned with a_d / w_row at the MUL stage
    // (when LMDOWN_SLR_REG=0 these collapse to the original chunk/state).
    // CO-TRAVEL: the RD markers derive from the DELIVERED (index,valid,group) so
    // v_rd/first_rd/last_rd/grp_rd describe the chunk whose ACTIVATION+WEIGHT meet
    // at the MUL (data-driven), NOT the FSM launch cadence. is_first/is_last from
    // the delivered index (==0 / ==NCHUNKS-1). Legacy (COTRAVEL=0) = the FSM
    // launch markers (run_eff / chunk_eff), byte-identical.
    wire [CW-1:0]                mk_chunk  = COTRAVEL ? chunk_dlv : chunk_eff;
    wire                         mk_run    = COTRAVEL ? run_dlv   : run_eff;
    wire [GW-1:0]                mk_grp    = COTRAVEL ? grp_dlv    : grp_eff;
    wire                         run_valid     = mk_run;
    wire                         is_first_chunk= (mk_chunk == '0);
    wire                         is_last_chunk = (mk_chunk == CW'(NCHUNKS-1));

    // matched activation delay (registered alongside the ROM read).
    // BANK-SPLIT: instead of ONE 256b a_d register, capture the 256b activation
    // seam in NB=8 INDEPENDENT 32b registers (a_d_bank), each a narrow reg-to-reg
    // crossing. a_d is the comb RE-ASSEMBLY of the 8 banks -> bit-identical value,
    // identical 1-cycle latency, but NO single 256b net spans an SLR.
    (* keep = "true", dont_touch = "true", shreg_extract = "no" *)
    reg  [BANK_W-1:0]              a_d_bank [0:NB-1];     // 8 x 32b activation banks
    wire signed [K_PAR*A_WIDTH-1:0] a_d;                 // a(c) re-assembled (MUL A)
    genvar abk;
    generate
        for (abk = 0; abk < NB; abk = abk + 1) begin : g_a_reassemble
            assign a_d[abk*BANK_W +: BANK_W] = a_d_bank[abk];
        end
    endgenerate

    // -----------------------------------------------------------------
    // SEAM_FF_LAND — per-SLR FF LANDING register on the activation input.
    // a_chunk (= seam_r8_rd, the cross-SLR seam read) is registered ONCE here
    // so the lanes consume a LOCAL FF copy (a_lin). When SEAM_FF_LAND=0 this is
    // a pure wire-through (a_lin == a_chunk) -> zero added latency, identical.
    // The +1 latency this adds (SEAM_FF_LAND=1) is compensated on the weight/
    // marker arm by the extra w_row_d2/*_rd3 stage below.
    // -----------------------------------------------------------------
    (* keep = "true", dont_touch = "true", shreg_extract = "no" *)
    reg  signed [K_PAR*A_WIDTH-1:0] a_chunk_land;
    always_ff @(posedge clk) begin
        if (!rst_n) a_chunk_land <= '0;
        else        a_chunk_land <= a_chunk;
    end
    wire signed [K_PAR*A_WIDTH-1:0] a_lin = SEAM_FF_LAND ? a_chunk_land : a_chunk;

    // DBG_TAP wired localizer (M7 GLM=0): READ-ONLY tap of a_d lane0 (MUL A operand)
    // + run_eff (GLM S_RUN). DBG_TAP=0 => tied 0 -> DCE -> byte-identical no-tap md5.
    generate
    if (DBG_TAP) begin : g_dbg_tap
        assign dbg_a_d0     = a_d[15:0];
        assign dbg_run_valid = run_eff;
    end else begin : g_no_dbg_tap
        assign dbg_a_d0     = 16'h0000;
        assign dbg_run_valid = 1'b0;
    end
    endgenerate

    // -----------------------------------------------------------------
    // SEAM-LATENCY REALIGN (2026-06-08, PDI#10 D1 lmdown gate — FIX_LMDOWN_SEAM_GATE).
    // With LMDOWN_SLR_REG=1 the EXTERNAL seam (u_seam_r8, RD_LAT=D1_RDLAT=1) is a
    // 1-cycle SYNCHRONOUS read: the activation a_chunk(c) appears one cycle AFTER
    // a_chunk_idx=c is presented. So from the common launch (chunk_eff) the two
    // arms have DIFFERENT latency:
    //     weight  : chunk_eff -> (XPM ROM 1cyc) -> w_row              = latency 1
    //     activ.  : chunk_eff -> a_chunk_idx -> (seam 1cyc) -> a_chunk
    //                                        -> (a_d reg 1cyc) -> a_d = latency 2
    // The original LMDOWN_SLR_REG=1 only delayed the launch SELECT (chunk_eff); it
    // did NOT compensate the seam's extra read register, so a_d landed a(c-1) while
    // w_row landed W(c) -> the MUL computed a(c-1)*W(c) (stale-by-1, the gate's
    // mode-B divergence: 253/256, maxdiff 145). Confirmed by trace: at the MUL
    // cycle a_d=a(chunk0) met w_row=W(chunk1).
    //
    // FIX: when LMDOWN_SLR_REG=1, delay the WEIGHT arm and the RD-stage markers by
    // ONE MORE cycle (w_row_d / *_rd2) so BOTH arms are latency 2 from chunk_eff and
    // a_d(c) meets w_row_d(c) at the MUL stage with co-aligned markers. The MUL/
    // SUMA/SUMB/ACC arithmetic is UNCHANGED -> bit-IDENTICAL to the combinational
    // (mode-0) golden, just one cycle later (absorbed by the S_FLUSH drain).
    // When LMDOWN_SLR_REG=0 (comb seam, latency-symmetric) the extra stage is a
    // pure wire-through (w_eff=w_row, markers = original RD stage) -> zero change.
    // -----------------------------------------------------------------
    // BANK-SPLIT: w_row_d (the +1-delayed weight arm, MODE=1) is split into
    // NB=8 INDEPENDENT 32b registers PER LANE (w_row_d_bank[L][b]), each a narrow
    // reg-to-reg crossing. w_row_d[L] is the comb RE-ASSEMBLY of the 8 banks ->
    // bit-identical value, identical +1 latency, but NO single 256b weight net
    // spans an SLR (the GLM_ZERO_ROOTCAUSE.md §6 placement-insensitive remedy).
    (* keep = "true", dont_touch = "true", shreg_extract = "no" *)
    reg  [BANK_W-1:0]            w_row_d_bank [0:LANES-1][0:NB-1]; // L x 8 x 32b
    wire signed [ROW_W-1:0]      w_row_d [0:LANES-1];     // w_row delayed +1, re-assembled
    genvar wlr, wbk;
    generate
        for (wlr = 0; wlr < LANES; wlr = wlr + 1) begin : g_wd_lane
            for (wbk = 0; wbk < NB; wbk = wbk + 1) begin : g_wd_reassemble
                assign w_row_d[wlr][wbk*BANK_W +: BANK_W] = w_row_d_bank[wlr][wbk];
            end
        end
    endgenerate
    reg                          v_rd2, first_rd2, last_rd2;
    reg [GW-1:0]                 grp_rd2;
    // SEAM_FF_LAND: ONE MORE matched weight + marker delay stage (rd3) so the
    // weight/marker arm absorbs the +1 activation-landing latency. Split into
    // NB banks per lane like w_row_d (no wide net spans an SLR). Compiled away /
    // ignored when SEAM_FF_LAND=0 (w_eff/markers select the rd2 path = today).
    (* keep = "true", dont_touch = "true", shreg_extract = "no" *)
    reg  [BANK_W-1:0]            w_row_d2_bank [0:LANES-1][0:NB-1];
    wire signed [ROW_W-1:0]      w_row_d2 [0:LANES-1];   // w_row delayed +2, re-assembled
    genvar wlr2, wbk2;
    generate
        for (wlr2 = 0; wlr2 < LANES; wlr2 = wlr2 + 1) begin : g_wd2_lane
            for (wbk2 = 0; wbk2 < NB; wbk2 = wbk2 + 1) begin : g_wd2_reassemble
                assign w_row_d2[wlr2][wbk2*BANK_W +: BANK_W] = w_row_d2_bank[wlr2][wbk2];
            end
        end
    endgenerate
    reg                          v_rd3, first_rd3, last_rd3;
    reg [GW-1:0]                 grp_rd3;
    wire signed [ROW_W-1:0]      w_eff   [0:LANES-1];
    wire                         v_mul, first_mul, last_mul;
    wire [GW-1:0]                grp_mul;

    // marker shift registers aligned to the 5 stages (rd, mul, suma, sumb, acc).
    // valid / first-chunk / last-chunk / group travel one stage per cycle so the
    // single-driver accumulator (ACC stage) knows which group and whether to
    // overwrite-then-add (first chunk) or accumulate.
    reg                          v_rd, v0, v1, v2;
    reg                          first_rd, first0, first1, first2;
    reg                          last_rd,  last0,  last1,  last2;
    reg [GW-1:0]                 grp_rd,   grp0,   grp1,   grp2;

    // SEAM-LATENCY REALIGN selects (declared after the RD markers they reference).
    // SEAM_FF_LAND adds +1 to the activation arm (a_chunk_land) -> the weight +
    // markers must also be +1 deeper (rd3) to stay chunk-aligned at the MUL.
    //   SEAM_FF_LAND : weight = w_row_d2 (latency 2), markers = *_rd3   (act = 3)
    //   LMDOWN_SLR_REG only : weight = w_row_d (latency 1), markers = *_rd2 (act = 2)
    //   neither      : weight = w_row (latency 0), markers = *_rd        (act = 1)
    // CO-TRAVEL OVERRIDE: the weight ROM is already addressed by the DELIVERED
    // index (bank_addr_full <- chunk_dlv), so w_row is ALREADY the chunk that
    // matches a_d at the MUL — NO w_row_d / w_row_d2 matched-delay realign is
    // needed (and SEAM_FF_LAND/LMDOWN_SLR_REG matched delays would now MIS-align).
    // COTRAVEL=1 therefore selects w_eff = w_row and the *_rd markers directly,
    // REPLACING the matched-delay realign. COTRAVEL=0 keeps the exact pdi17
    // SEAM_FF_LAND/LMDOWN_SLR_REG selection (byte-identical).
    genvar we;
    generate
        for (we = 0; we < LANES; we = we + 1) begin : g_weff
            assign w_eff[we] = COTRAVEL       ? w_row[we]
                             : SEAM_FF_LAND   ? w_row_d2[we]
                             : LMDOWN_SLR_REG ? w_row_d[we]
                                              : w_row[we];
        end
    endgenerate
    assign v_mul     = COTRAVEL ? v_rd     : (SEAM_FF_LAND ? v_rd3     : (LMDOWN_SLR_REG ? v_rd2     : v_rd));
    assign first_mul = COTRAVEL ? first_rd : (SEAM_FF_LAND ? first_rd3 : (LMDOWN_SLR_REG ? first_rd2 : first_rd));
    assign last_mul  = COTRAVEL ? last_rd  : (SEAM_FF_LAND ? last_rd3  : (LMDOWN_SLR_REG ? last_rd2  : last_rd));
    assign grp_mul   = COTRAVEL ? grp_rd   : (SEAM_FF_LAND ? grp_rd3   : (LMDOWN_SLR_REG ? grp_rd2   : grp_rd));

    // products per lane (registered).
    reg signed [31:0]            prod_reg [0:LANES-1][0:K_PAR-1];
    // SUMA partials: 4 per lane (each sums 4 of the 16 products) -> INT34 headroom.
    reg signed [33:0]            suma_reg [0:LANES-1][0:3];
    // SUMB: final per-lane sum of the 4 partials -> INT40 headroom (matches old).
    reg signed [39:0]            sum_reg  [0:LANES-1];
    reg signed [47:0]            acc      [0:LANES-1];   // chunk accumulate (192 terms)

    integer L, I, Q;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (L = 0; L < LANES; L = L + 1) begin
                sum_reg[L] <= '0;
                acc[L]     <= '0;
                for (I = 0; I < K_PAR; I = I + 1) prod_reg[L][I] <= '0;
                for (Q = 0; Q < 4; Q = Q + 1)     suma_reg[L][Q] <= '0;
            end
            for (I = 0; I < NB; I = I + 1) a_d_bank[I] <= '0;   // BANK-SPLIT activation banks
            v_rd <= 1'b0; v0 <= 1'b0; v1 <= 1'b0; v2 <= 1'b0;
            first_rd <= 1'b0; first0 <= 1'b0; first1 <= 1'b0; first2 <= 1'b0;
            last_rd <= 1'b0; last0 <= 1'b0; last1 <= 1'b0; last2 <= 1'b0;
            grp_rd <= '0; grp0 <= '0; grp1 <= '0; grp2 <= '0;
            v_rd2 <= 1'b0; first_rd2 <= 1'b0; last_rd2 <= 1'b0; grp_rd2 <= '0;
            for (L = 0; L < LANES; L = L + 1)                   // BANK-SPLIT weight banks
                for (I = 0; I < NB; I = I + 1) w_row_d_bank[L][I] <= '0;
            // SEAM_FF_LAND: extra +1 weight/marker stage reset
            v_rd3 <= 1'b0; first_rd3 <= 1'b0; last_rd3 <= 1'b0; grp_rd3 <= '0;
            for (L = 0; L < LANES; L = L + 1)
                for (I = 0; I < NB; I = I + 1) w_row_d2_bank[L][I] <= '0;
        end else begin
            // RD stage: matched activation delay + markers (w_row driven in gen)
            // BANK-SPLIT: capture the 256b a_chunk seam into NB=8 independent 32b
            // banks. Each bank registers a contiguous 32b slice (2 INT16 elems);
            // the comb re-assembly (a_d) is bit-identical to the old `a_d<=a_chunk`.
            for (I = 0; I < NB; I = I + 1)
                a_d_bank[I] <= a_lin[I*BANK_W +: BANK_W];  // SEAM_FF_LAND: from landed copy
            v_rd     <= run_valid;
            first_rd <= run_valid & is_first_chunk;
            last_rd  <= run_valid & is_last_chunk;
            grp_rd   <= mk_grp;   // CO-TRAVEL: delivered group (== grp_eff when COTRAVEL=0)

            // SEAM-LATENCY REALIGN (+1) — only consumed when LMDOWN_SLR_REG=1.
            // Delays the weight read + RD markers ONE extra cycle so they match the
            // seam's extra activation register; a_d(c) then meets w_eff(c) at MUL.
            // BANK-SPLIT: w_row[L] (256b) -> NB=8 independent 32b banks per lane.
            // Same +1 latency boundary; re-assembled w_row_d[L] is bit-identical.
            for (L = 0; L < LANES; L = L + 1)
                for (I = 0; I < NB; I = I + 1)
                    w_row_d_bank[L][I] <= w_row[L][I*BANK_W +: BANK_W];
            v_rd2     <= v_rd;
            first_rd2 <= first_rd;
            last_rd2  <= last_rd;
            grp_rd2   <= grp_rd;

            // SEAM_FF_LAND extra stage (+1) — only CONSUMED when SEAM_FF_LAND=1
            // (w_eff/markers select the rd3 path). Delays the weight + markers ONE
            // more cycle to match the activation's +1 landing register, so a_d(c)
            // meets w_eff(c) at MUL with co-aligned markers. Bit-identical; the
            // regs are always clocked (cheap) but unused when SEAM_FF_LAND=0.
            for (L = 0; L < LANES; L = L + 1)
                for (I = 0; I < NB; I = I + 1)
                    w_row_d2_bank[L][I] <= w_row_d_bank[L][I];
            v_rd3     <= v_rd2;
            first_rd3 <= first_rd2;
            last_rd3  <= last_rd2;
            grp_rd3   <= grp_rd2;

            // MUL stage: per-lane, per-sublane INT16 x INT16 products.
            // w_eff / *_mul are wire-throughs (MODE 0) or the +1-delayed weight +
            // markers (MODE 1) so the product is chunk-aligned in BOTH modes.
            for (L = 0; L < LANES; L = L + 1) begin
                for (I = 0; I < K_PAR; I = I + 1) begin
                    prod_reg[L][I] <=
                        $signed(a_d[I*A_WIDTH +: A_WIDTH]) *
                        $signed(w_eff[L][I*A_WIDTH +: A_WIDTH]);
                end
            end
            // markers travel with the product: *_mul = RD markers (MODE 0) or the
            // +1-realigned RD markers (MODE 1), keeping group/first/last aligned to
            // the chunk that a_d*w_eff just multiplied.
            v0     <= v_mul;
            first0 <= first_mul;
            last0  <= last_mul;
            grp0   <= grp_mul;

            // SUMA stage: reduce the 16 products to 4 partials per lane.
            // Each partial sums 4 consecutive products (2 binary-add levels):
            //   suma[q] = (p[4q]+p[4q+1]) + (p[4q+2]+p[4q+3]).
            for (L = 0; L < LANES; L = L + 1) begin
                for (Q = 0; Q < 4; Q = Q + 1) begin
                    suma_reg[L][Q] <=
                        ($signed(prod_reg[L][4*Q+0]) + $signed(prod_reg[L][4*Q+1])) +
                        ($signed(prod_reg[L][4*Q+2]) + $signed(prod_reg[L][4*Q+3]));
                end
            end
            v1     <= v0;
            first1 <= first0;
            last1  <= last0;
            grp1   <= grp0;

            // SUMB stage: reduce the 4 partials to the final per-lane sum
            //   sum = (suma[0]+suma[1]) + (suma[2]+suma[3])   (2 binary-add levels).
            for (L = 0; L < LANES; L = L + 1) begin
                sum_reg[L] <=
                    ($signed(suma_reg[L][0]) + $signed(suma_reg[L][1])) +
                    ($signed(suma_reg[L][2]) + $signed(suma_reg[L][3]));
            end
            v2     <= v1;
            first2 <= first1;
            last2  <= last1;
            grp2   <= grp1;

            // ACC stage: accumulate per lane (SINGLE DRIVER for acc).  On the FIRST
            // chunk of a group start fresh (sum_reg, not acc+sum); otherwise add.
            for (L = 0; L < LANES; L = L + 1) begin
                if (v2) begin
                    if (first2) acc[L] <= sum_reg[L];
                    else        acc[L] <= acc[L] + sum_reg[L];
                end
            end
        end
    end

    // commit when the last-chunk's sum has been ADDED into acc, i.e. ONE cycle
    // after v2&last2 (acc updated that cycle; read it the next).  No acc clear is
    // needed — the next group's first2 overwrites acc with its own sum_reg.
    reg          commit_pulse;
    reg [GW-1:0] commit_grp;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            commit_pulse <= 1'b0; commit_grp <= '0;
        end else begin
            commit_pulse <= v2 & last2;     // acc holds this group's full sum next cyc
            commit_grp   <= grp2;
        end
    end

    // -----------------------------------------------------------------
    // Output rescale + commit.  acc[L] is at ACC_FRAC; rescale to OUT_FRAC INT16.
    // act_rescale-style: round-to-nearest arithmetic shift + saturate.
    // -----------------------------------------------------------------
    localparam int SHIFT = ACC_FRAC - OUT_FRAC;   // 25 - 11 = 14
    localparam logic signed [A_WIDTH-1:0] PSAT = {1'b0, {(A_WIDTH-1){1'b1}}};
    localparam logic signed [A_WIDTH-1:0] NSAT = {1'b1, {(A_WIDTH-1){1'b0}}};
    function automatic logic signed [A_WIDTH-1:0] rescale_out;
        input signed [47:0] x;
        logic signed [49:0] xe, r;
        begin
            xe = $signed({{2{x[47]}}, x});
            if (SHIFT > 0)
                r = (xe + ($signed(50'sd1) <<< (SHIFT-1))) >>> SHIFT;
            else
                r = xe;
            if (r > $signed({{(50-A_WIDTH){1'b0}}, PSAT}))      rescale_out = PSAT;
            else if (r < $signed({{(50-A_WIDTH){1'b1}}, NSAT})) rescale_out = NSAT;
            else                                                rescale_out = r[A_WIDTH-1:0];
        end
    endfunction

    integer Lo;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_valid <= 1'b0; wr_chunk_idx <= '0; wr_chunk <= '0;
        end else begin
            wr_valid <= commit_pulse;
            if (commit_pulse) begin
                wr_chunk_idx <= commit_grp[OCW-1:0];
                for (Lo = 0; Lo < LANES; Lo = Lo + 1)
                    wr_chunk[Lo*A_WIDTH +: A_WIDTH] <= rescale_out(acc[Lo]);
            end
        end
    end

    // -----------------------------------------------------------------
    // FSM control
    // -----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; grp <= '0; chunk <= '0; flush_cnt <= '0;
        end else begin
            state <= state_n; grp <= grp_n; chunk <= chunk_n; flush_cnt <= flush_cnt_n;
        end
    end

    always_comb begin
        state_n     = state;
        grp_n       = grp;
        chunk_n     = chunk;
        flush_cnt_n = flush_cnt;
        case (state)
            S_IDLE: if (start) begin
                state_n = S_RUN; grp_n = '0; chunk_n = '0; flush_cnt_n = '0;
            end
            S_RUN: begin
                if (chunk == CW'(NCHUNKS-1)) begin
                    state_n     = S_FLUSH;
                    flush_cnt_n = '0;
                end else begin
                    chunk_n = chunk + 1'b1;
                end
            end
            // drain the BANK->MUL->SUMA->SUMB->ACC->commit pipeline before the next
            // group restarts streaming, so this group's last-chunk add + commit land
            // before the next group's first add overwrites the shared accumulator.
            // FIX2: the pipeline is now 2 cycles DEEPER (SUMA+SUMB replace the single
            // SUM stage), so the drain depth is bumped from 8 to 10 (the path is
            // rd(1)+mul(1)+suma(1)+sumb(1)+acc(1)+commit(1)+wr(1) + margin).
            // SEAM_FF_LAND adds ONE MORE RD stage (rd3) on the weight/marker arm to
            // match the +1 activation landing reg, so the drain needs +1 (11). Still
            // < the 4-bit flush_cnt max (15). (4-bit flush_cnt holds up to 15.)
            S_FLUSH: begin
                if (flush_cnt >= (SEAM_FF_LAND ? 4'd11 : 4'd10)) begin
                    if (grp == GW'(N_GRP-1)) begin
                        state_n = S_DONE;
                    end else begin
                        state_n = S_RUN;
                        grp_n   = grp + 1'b1;
                        chunk_n = '0;
                    end
                end else begin
                    flush_cnt_n = flush_cnt + 1'b1;
                end
            end
            S_DONE: state_n = S_IDLE;
            default: state_n = S_IDLE;
        endcase
    end

    // done one cycle after the last group's commit is guaranteed flushed.
    always_ff @(posedge clk) begin
        if (!rst_n) done <= 1'b0;
        else        done <= (state == S_DONE);
    end

`ifndef SYNTHESIS
    initial if (SEAM_FF_LAND && !LMDOWN_SLR_REG)
        $error("lmdown_gemv_banksplit: SEAM_FF_LAND=1 requires LMDOWN_SLR_REG=1 (it extends the +1 seam-realign with one more matched weight/marker stage).");
`endif

endmodule

`default_nettype wire
