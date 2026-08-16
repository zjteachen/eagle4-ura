// =====================================================================
// dbg_observe.sv  (PDI#7 silicon-hardening: debug visibility)
// =====================================================================
//
// MOTIVATION (reports/eagle4_drafter_m7/DEBUG_VISIBILITY_WATCHDOG.md):
//   The flashed PDI#6 image exposed only two status registers (0x74 boot-DMA,
//   0x78 out-DMA). Bisection of the "all-zero logits / identity-frontier"
//   datapath fault DEAD-ENDED because there was NO host-visible signal for
//   (a) which FSM state the kernel is parked in, (b) which stages actually
//   completed, and (c) whether any stage emitted NON-ZERO output. This block
//   adds exactly that, at two previously-FREE AXI-Lite offsets (0x80/0x84).
//
// PURELY ADDITIVE / FUNCTION-INVARIANT:
//   * Every input is a READ-ONLY tap off an existing top-level signal.
//   * No output of this block feeds the datapath; it only produces two 32-bit
//     status words muxed onto the AXI-Lite read channel by control_regs. The
//     golden frontier is therefore BIT-IDENTICAL with or without this block
//     (memory feedback_no_accuracy_tradeoff).
//
// MULTI-DRIVER SAFE (memory feedback_multi_driver_packed_reg_trap):
//   Every packed status word is assembled with a SINGLE combinational concat
//   from scalar/unpacked sources. Sticky vectors are unpacked regs each driven
//   from exactly ONE always_ff; nothing writes disjoint slices of a packed reg.
//
// SYNTH-SURVIVAL (memory feedback_nodriver_bake_drop_synth_only):
//   No baked/$readmemh memory here -> nothing to silently drop. Every reg has a
//   real clocked driver.
//
// 0x80  fsmstate_word : {magic, last_stage, g5_state, top_state}  (live mirror)
//   [31:16] = 0xF500 magic tag (host confirms it read the mirror, not RAZ)
//   [15:13] = reserved
//   [12: 8] = last_stage : top_fsm enum code of the LAST stage that asserted
//             its *_done strobe (frozen at the last completed stage on a hang)
//   [ 7: 5] = g5_state   : attention sub-FSM state (3b)
//   [ 4: 0] = top_state  : top_fsm.state (state_t, 5b)
//
// 0x84  activity_word : sticky per-stage NON-ZERO-output flags
//   A bit SETS the first cycle that stage's output strobe is high AND any
//   sample of that stage's output bus is non-zero. Cleared at each launch
//   (start_pulse). A datapath zero fault localizes to "highest set bit + 1"
//   in ONE register read.
//   [ 0] G1   embed/in-RMSNorm out   (rms_g1_out_chunk on rms_g1_out_valid)
//   [ 1] G4   q/k/v GEMV out         (array_acc_out on array_done @ stage G4)
//   [ 2] G5   attention out          (g5_attn_out_vec on g5_attn_done)
//   [ 3] G6   o-proj GEMV out        (array_acc_out on array_done @ stage G6)
//   [ 4] G7g  gate GEMV out          (array_acc_out on array_done @ stage G7G)
//   [ 5] G7u  up   GEMV out          (array_acc_out on array_done @ stage G7U)
//   [ 6] G7p  SiLU*up out            (silu_out_chunk on silu_valid)
//   [ 7] G9   down GEMV out          (array_acc_out on array_done @ stage G9)
//   [ 8] G11  final-RMSNorm out      (rms_g11_out_chunk on g11_out_strobe)
//   [ 9] G12  logits GEMV out        (array_acc_out on array_done @ stage G12)
//   [10] TREE top-K scores           (tree_topk_scores on tree_done)
//   [11] G8   rescored scores        (g8_activity, driven from the G8 block)
//   [30] g8_watchdog tripped echo    (sticky; mirrors 0x78 bit3)
//   [31] any-stage-nonzero OR        (convenience: at least one bit 0..11 set)
//
// stage_sel codes (from top_fsm): G4=0, G6=1, G7G=2, G7U=3, G9=4, G12=5.
//
// =====================================================================
// DEBUG BLOCK V2 (PDI#8) — additive, registers 0x88..0xA4 only.
// 0x80/0x84 (above) are UNCHANGED. v2 adds:
//   0x88 W  dbg_sel[3:0]  stage select (latches the v2 read mux)
//   0x8C R  per-stage CYCLE COUNT for the selected stage (g<X>_start..done)
//   0x90 R  input checksum (XOR-fold) for the selected source (emb/hfc/gamma)
//   0x94 R  G8 progress {rresp_err,seen_last,in_gather,sub[3:0],filled[9:0],rows[9:0]}
//   0x98 R  last m_axi_g8 araddr[31:0]
//   0x9C R  last m_axi_g8 araddr[63:32] (lo bits) ; [31] = sticky rresp!=OKAY
//   0xA0 R  {tree_done_cnt[15:0], ap_start_cnt[15:0]}  (re-arm verification)
//   0xA4 R  M9 boot stream {running XOR fold[15:0], word-count[15:0]}
// All read-only taps; multi-driver-safe (one always_ff per accumulator);
// no $readmemh (survives synth).
//
// 14-STAGE CYCLE-COUNT MAP (dbg_sel select -> 0x8C):
//   0=G1  1=G4  2=G5  3=G6  4=G7g 5=G7u 6=G7p 7=G9 8=G11 9=GLM 10=G12
//   11=TREE 12=G8 13=AP-BUSY (total kernel-busy cycles)
// Each counter is the free-running cycle delta between that stage's START
// (top_state ENTRY) and DONE strobe; frozen on done, cleared at launch.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module dbg_observe #(
    parameter int LANES   = 128,
    parameter int A_WIDTH = 16,
    parameter int G8_ADDR_W = 64
) (
    input  wire        clk,
    input  wire        rst_n,

    // ---- launch / FSM-state taps (read-only) ----
    input  wire        start_pulse,     // 1-cyc launch (clears stickies)
    input  wire [4:0]  top_state,       // top_fsm.state
    input  wire [2:0]  g5_state,        // attention sub-FSM
    input  wire [3:0]  stage_sel,       // shared-GEMV stage code (G4=0..G12=5)

    // ---- per-stage completion strobes (for last_stage mirror) ----
    input  wire        g1_done,
    input  wire        g4_done,
    input  wire        g5_done,
    input  wire        g6_done,
    input  wire        g7g_done,
    input  wire        g7u_done,
    input  wire        g7p_done,
    input  wire        g9_done,
    input  wire        g11_done,
    input  wire        glm_done,
    input  wire        g12_done,
    input  wire        tree_done,
    input  wire        g8_done,

    // ---- per-stage OUTPUT-DATA taps + their valid strobes (for 0x84) ----
    input  wire signed [16*16-1:0]      g1_out_chunk,     // rms_g1_out_chunk_w
    input  wire                         g1_out_valid,     // rms_g1_out_valid
    input  wire signed [32*LANES-1:0]   array_acc_out,    // shared GEMV out bus
    input  wire                         array_done,       // GEMV tile-valid
    input  wire signed [32*16-1:0]      g5_attn_out_vec,  // attention out (D_HEAD=16)
    input  wire                         g5_attn_done,
    input  wire signed [16*16-1:0]      silu_out_chunk,
    input  wire                         silu_valid,
    input  wire signed [16*16-1:0]      g11_out_chunk,    // rms_g11_out_chunk
    input  wire                         g11_out_strobe,
    input  wire signed [8*32-1:0]       tree_topk_scores,
    // G8 activity is computed inside the G8 generate block (sticky) and passed
    // in already-reduced; 0 when G8_SPLICE=0.
    input  wire                         g8_activity,      // 1 = a non-zero G8 score seen
    input  wire                         g8_wd_tripped,    // 0x78 bit3 echo

    // ================= DEBUG BLOCK V2 inputs (PDI#8) =================
    input  wire        kernel_busy,      // ap busy (for 0xA0 / busy-cycle ctr)
    input  wire        kernel_done,      // ap done strobe (tree-done family)

    // 0x88 write: stage select for 0x8C / 0x90 read mux
    input  wire [3:0]  dbg_sel,

    // input-buffer write strobes + data words (legacy input checksum, now at 0xB? — kept)
    input  wire        emb_we,           // input_emb_buf write
    input  wire [31:0] emb_wr_data,
    input  wire        hfc_we,           // h_fc_buf write
    input  wire [31:0] hfc_wr_data,
    input  wire        gamma_we,         // any gamma_*_buf write (OR of 4)
    input  wire [31:0] gamma_wr_data,

    // PDI#10 0x90-FIX — per-STAGE-OUTPUT running checksum, selected by dbg_sel.
    // The top muxes the dbg_sel-selected stage's output chunk onto this feed (one
    // strobe per output chunk); we fold a running XOR + sum over its words and a
    // word count. This makes 0x90 a LIVE per-stage output content checksum (the
    // old 0x90 input-checksum read 0 once loads finished). 0x90 = the running XOR
    // for the dbg_sel-selected stage (cleared at launch + on a dbg_sel change).
    input  wire        sout_chunk_valid, // 1-cyc per output chunk of dbg_sel stage
    input  wire [31:0] sout_chunk_w0,    // word 0 of that chunk (representative)
    input  wire [31:0] sout_chunk_xor,   // XOR-fold of all words IN that chunk
    input  wire [9:0]  sout_chunk_nwords,// # 32b words in that chunk

    // G8 progress mirrors (from g8_rescore, 0 in the G8 bypass)
    input  wire [3:0]  g8_sel_substate,
    input  wire [9:0]  g8_cand_filled,
    input  wire [9:0]  g8_rows_gathered,
    input  wire        g8_seen_last,
    input  wire        g8_in_gather,
    input  wire [G8_ADDR_W-1:0] g8_last_araddr, // last AR byte addr (full width)
    input  wire        g8_rresp_err,

    // M9 boot stream tap (HBM->URAM weight/scale loader)
    input  wire        boot_stream_we,   // one pulse per loaded word
    input  wire [31:0] boot_stream_data,

    // ---- host-readable status words (combinational; control_regs muxes) ----
    output wire [31:0] fsmstate_word,    // 0x80
    output wire [31:0] activity_word,    // 0x84
    output wire [31:0] cyclecount_word,  // 0x8C (selected stage cycle count)
    output wire [31:0] checksum_word,    // 0x90 (selected input checksum)
    output wire [31:0] g8progress_word,  // 0x94
    output wire [31:0] g8araddr_lo_word, // 0x98
    output wire [31:0] g8araddr_hi_word, // 0x9C  ([31]=rresp_err)
    output wire [31:0] rearm_word,        // 0xA0 {tree_done_cnt,ap_start_cnt}
    output wire [31:0] bootstream_word,   // 0xA4 {xor_fold[15:0],wordcount[15:0]}
    // PDI#10 0x90-FIX + content-sample outputs:
    output wire [31:0] stageout_xor_word,   // 0x90 : dbg_sel stage-output running XOR (the 0x90 fix)
    output wire [31:0] stageout_sample_word,// 0xD4 : first 32b word of the dbg_sel stage's first output chunk
    output wire [31:0] stageout_count_word  // 0xD8 : # 32b output words seen for dbg_sel stage
);

    // -----------------------------------------------------------------
    // 0x80 — FSM-state mirror. last_stage freezes at the last *_done seen.
    // -----------------------------------------------------------------
    reg [4:0] last_stage;
    always_ff @(posedge clk) begin
        if (!rst_n)           last_stage <= 5'd0;
        else if (start_pulse) last_stage <= 5'd0;
        else begin
            // priority chain (one *_done high per cycle in practice).
            // codes match the top_fsm state_t enum for legibility.
            if (g1_done)        last_stage <= 5'd1;
            else if (g4_done)   last_stage <= 5'd2;
            else if (g5_done)   last_stage <= 5'd3;
            else if (g6_done)   last_stage <= 5'd4;
            else if (g7g_done)  last_stage <= 5'd5;
            else if (g7u_done)  last_stage <= 5'd6;
            else if (g7p_done)  last_stage <= 5'd7;
            else if (g9_done)   last_stage <= 5'd8;
            else if (g11_done)  last_stage <= 5'd9;
            else if (glm_done)  last_stage <= 5'd13;
            else if (g12_done)  last_stage <= 5'd10;
            else if (tree_done) last_stage <= 5'd11;
            else if (g8_done)   last_stage <= 5'd14;
        end
    end

    assign fsmstate_word = {16'hF500,     // [31:16] magic tag
                            3'b000,       // [15:13] reserved
                            last_stage,   // [12:8]
                            g5_state,     // [7:5]
                            top_state};   // [4:0]

    // -----------------------------------------------------------------
    // 0x84 — sticky per-stage NON-ZERO output flags.
    // -----------------------------------------------------------------
    // OR-reduce of each output bus (any bit set anywhere -> non-zero sample).
    wire nz_g1   = |g1_out_chunk;
    wire nz_arr  = |array_acc_out;
    wire nz_attn = |g5_attn_out_vec;
    wire nz_silu = |silu_out_chunk;
    wire nz_g11  = |g11_out_chunk;
    wire nz_tree = |tree_topk_scores;

    // GEMV-stage attribution: the shared array's array_done belongs to the
    // current stage_sel. stage_sel codes: G4=0,G6=1,G7G=2,G7U=3,G9=4,G12=5.
    wire arr_g4  = array_done & (stage_sel == 4'd0) & nz_arr;
    wire arr_g6  = array_done & (stage_sel == 4'd1) & nz_arr;
    wire arr_g7g = array_done & (stage_sel == 4'd2) & nz_arr;
    wire arr_g7u = array_done & (stage_sel == 4'd3) & nz_arr;
    wire arr_g9  = array_done & (stage_sel == 4'd4) & nz_arr;
    wire arr_g12 = array_done & (stage_sel == 4'd5) & nz_arr;

    reg [11:0] act;   // bits 0..11 per the header map
    always_ff @(posedge clk) begin
        if (!rst_n)           act <= 12'd0;
        else if (start_pulse) act <= 12'd0;
        else begin
            if (g1_out_valid   & nz_g1)   act[0]  <= 1'b1;  // G1
            if (arr_g4)                    act[1]  <= 1'b1;  // G4
            if (g5_attn_done   & nz_attn) act[2]  <= 1'b1;  // G5
            if (arr_g6)                    act[3]  <= 1'b1;  // G6
            if (arr_g7g)                   act[4]  <= 1'b1;  // G7g
            if (arr_g7u)                   act[5]  <= 1'b1;  // G7u
            if (silu_valid     & nz_silu) act[6]  <= 1'b1;  // G7p (SiLU)
            if (arr_g9)                    act[7]  <= 1'b1;  // G9
            if (g11_out_strobe & nz_g11)  act[8]  <= 1'b1;  // G11
            if (arr_g12)                   act[9]  <= 1'b1;  // G12
            if (nz_tree)                   act[10] <= 1'b1;  // TREE (held post-decode)
            if (g8_activity)               act[11] <= 1'b1;  // G8
        end
    end

    wire any_nz = |act;

    // single combinational concat (multi-driver-safe)
    assign activity_word = {any_nz,          // [31] any-stage non-zero OR
                            g8_wd_tripped,   // [30] G8 watchdog tripped echo
                            18'd0,           // [29:12] reserved
                            act};            // [11:0]

    // =================================================================
    // DEBUG BLOCK V2 (PDI#8) — additive accumulators 0x88..0xA4.
    // Every accumulator below has EXACTLY ONE always_ff driver -> multi-
    // driver-safe; all read words are single combinational concats.
    // =================================================================

    // ---- 0x88 dbg_sel : latched in control_regs; arrives here as `dbg_sel`.

    // -----------------------------------------------------------------
    // 0x8C — per-stage CYCLE COUNT. 14 free-running cycle counters, each
    // counts while its stage is "in flight" (entered, not yet done) and
    // FREEZES at its done strobe. Cleared at launch. Stage in-flight is
    // detected from the top_state ENTRY edge (rises) to the done strobe.
    // The AP-BUSY counter (sel=13) counts all kernel_busy cycles.
    // -----------------------------------------------------------------
    // per-stage "in flight" level: set on the matching top_state, cleared
    // on that stage's *_done. Encoded as a level reg driven by ONE always_ff.
    // 24-bit counters: 2^24 cycles = 134 ms @ 125 MHz, far beyond any stage's
    // real duration (the whole forward is ~ms) — narrowing 32->24 cuts the
    // 14 parallel increment adders' LUT/FF cost (OOC gate < 2K LUT).
    localparam int CYCW = 24;
    reg [13:0]      cyc_run;              // 1 = stage counting
    reg [CYCW-1:0]  cyc_cnt [0:13];       // the 14 cycle counters
    integer ci;
    // top_state codes used elsewhere: G1=1 G4=2 G5=3 G6=4 G7g=5 G7u=6 G7p=7
    // G9=8 G11=9 G12=10 TREE=11 GLM=13. (G8 has no own top_state — gated by
    // g8 in_gather; index 12 = G8.)
    wire [13:0] stage_done = { 1'b0,                 // [13] AP-BUSY (no done; busy-gated)
                               g8_done,              // [12] G8
                               tree_done,            // [11] TREE
                               g12_done,             // [10] G12
                               glm_done,             // [ 9] GLM
                               g11_done,             // [ 8] G11
                               g9_done,              // [ 7] G9
                               g7p_done,             // [ 6] G7p
                               g7u_done,             // [ 5] G7u
                               g7g_done,             // [ 4] G7g
                               g6_done,              // [ 3] G6
                               g5_done,              // [ 2] G5
                               g4_done,              // [ 1] G4
                               g1_done };            // [ 0] G1
    // stage "active" gate: top_state match (G8 = g8_in_gather; AP-BUSY=busy).
    wire [13:0] stage_active = { kernel_busy,                  // [13] AP-BUSY
                                 g8_in_gather,                 // [12] G8
                                 (top_state == 5'd11),         // [11] TREE
                                 (top_state == 5'd10),         // [10] G12
                                 (top_state == 5'd13),         // [ 9] GLM
                                 (top_state == 5'd9),          // [ 8] G11
                                 (top_state == 5'd8),          // [ 7] G9
                                 (top_state == 5'd7),          // [ 6] G7p
                                 (top_state == 5'd6),          // [ 5] G7u
                                 (top_state == 5'd5),          // [ 4] G7g
                                 (top_state == 5'd4),          // [ 3] G6
                                 (top_state == 5'd3),          // [ 2] G5
                                 (top_state == 5'd2),          // [ 1] G4
                                 (top_state == 5'd1) };        // [ 0] G1

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cyc_run <= 14'd0;
            for (ci = 0; ci < 14; ci = ci + 1) cyc_cnt[ci] <= '0;
        end else if (start_pulse) begin
            cyc_run <= 14'd0;
            for (ci = 0; ci < 14; ci = ci + 1) cyc_cnt[ci] <= '0;
        end else begin
            for (ci = 0; ci < 14; ci = ci + 1) begin
                if (stage_done[ci]) begin
                    cyc_run[ci] <= 1'b0;            // freeze on done
                end else if (stage_active[ci]) begin
                    cyc_run[ci] <= 1'b1;
                    cyc_cnt[ci] <= cyc_cnt[ci] + 1'b1;   // count while active (sat at 2^24-1)
                end
            end
        end
    end
    assign cyclecount_word = {{(32-CYCW){1'b0}}, cyc_cnt[dbg_sel[3:0]]};

    // -----------------------------------------------------------------
    // 0x90 — input CHECKSUMS (running XOR-fold of loaded words). Three
    // independent folds: emb (sel 0), hfc (sel 1), gamma (sel 2). Cleared
    // on a new select-set (dbg_sel change) AND at launch.
    // -----------------------------------------------------------------
    reg [31:0] csum_emb, csum_hfc, csum_gamma;
    reg [3:0]  dbg_sel_q;                  // for clear-on-select-change
    wire       sel_changed = (dbg_sel != dbg_sel_q);
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            csum_emb <= 32'd0; csum_hfc <= 32'd0; csum_gamma <= 32'd0;
            dbg_sel_q <= 4'd0;
        end else begin
            dbg_sel_q <= dbg_sel;
            if (start_pulse || sel_changed) begin
                csum_emb <= 32'd0; csum_hfc <= 32'd0; csum_gamma <= 32'd0;
            end else begin
                if (emb_we)   csum_emb   <= csum_emb   ^ emb_wr_data;
                if (hfc_we)   csum_hfc   <= csum_hfc   ^ hfc_wr_data;
                if (gamma_we) csum_gamma <= csum_gamma ^ gamma_wr_data;
            end
        end
    end
    // (legacy input checksum retained internally; the 0x90 register now carries
    //  the LIVE per-stage OUTPUT checksum below — the old input-fold read 0 once
    //  boot loads finished, which was the "dead 0x90" symptom.)
    wire [31:0] input_csum_unused = (dbg_sel == 4'd1) ? csum_hfc   :
                                    (dbg_sel == 4'd2) ? csum_gamma :
                                                         csum_emb;

    // -----------------------------------------------------------------
    // 0x90-FIX — per-STAGE-OUTPUT running checksum (XOR + sum) + first-word
    // sample + word count, for the dbg_sel-selected stage. Fed by the top's
    // dbg_sel-muxed stage-output chunk feed (sout_*). Cleared at launch and on
    // a dbg_sel change so each (forward, stage) pair gets a fresh checksum.
    // One always_ff driver (multi-driver-safe); no $readmemh (synth-safe).
    // -----------------------------------------------------------------
    reg [31:0] sout_xor, sout_sum, sout_count, sout_sample;
    reg        sout_sample_seen;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sout_xor <= 32'd0; sout_sum <= 32'd0; sout_count <= 32'd0;
            sout_sample <= 32'd0; sout_sample_seen <= 1'b0;
        end else if (start_pulse || sel_changed) begin
            sout_xor <= 32'd0; sout_sum <= 32'd0; sout_count <= 32'd0;
            sout_sample <= 32'd0; sout_sample_seen <= 1'b0;
        end else if (sout_chunk_valid && sout_chunk_nwords != 10'd0) begin
            sout_xor <= sout_xor ^ sout_chunk_xor;
            sout_sum <= sout_sum + sout_chunk_w0;     // representative running sum (lightweight)
            if (sout_count != 32'hFFFF_FFFF) sout_count <= sout_count + {22'd0, sout_chunk_nwords};
            if (!sout_sample_seen) begin
                sout_sample      <= sout_chunk_w0;    // first output word of the first chunk
                sout_sample_seen <= 1'b1;
            end
        end
    end

    assign checksum_word        = sout_xor;     // 0x90 (the fix: live stage-output XOR)
    assign stageout_xor_word    = sout_xor;     // 0x90 mirror (explicit name)
    assign stageout_sample_word = sout_sample;  // 0xD4
    assign stageout_count_word  = sout_count;   // 0xD8

    // -----------------------------------------------------------------
    // 0x94 — G8 progress (single combinational concat of g8_rescore mirrors).
    // -----------------------------------------------------------------
    assign g8progress_word = { g8_rresp_err,        // [31]
                               g8_seen_last,         // [30]
                               g8_in_gather,         // [29]
                               5'd0,                 // [28:24] reserved
                               g8_sel_substate,      // [23:20]
                               g8_cand_filled,       // [19:10]
                               g8_rows_gathered };   // [ 9: 0]

    // -----------------------------------------------------------------
    // 0x98 / 0x9C — last G8 AR byte addr (split) + rresp sticky flag.
    //   0x98 = araddr[31:0]; 0x9C[30:0] = araddr[62:32]; 0x9C[31] = rresp_err.
    //   (HBM byte addrs are < 2^48 so bit 62 is always 0 -> lossless split.)
    // -----------------------------------------------------------------
    wire [G8_ADDR_W-1:0] araddr_ext = g8_last_araddr;
    assign g8araddr_lo_word = araddr_ext[31:0];
    assign g8araddr_hi_word = { g8_rresp_err, araddr_ext[62:32] };

    // -----------------------------------------------------------------
    // 0xA0 — re-arm verification: ap_start count (start_pulse) + tree_done
    // count (kernel_done family). Both saturate at 16b; NOT cleared at launch
    // (they exist to count launches across the session).
    // -----------------------------------------------------------------
    reg [15:0] ap_start_cnt, tree_done_cnt;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ap_start_cnt <= 16'd0; tree_done_cnt <= 16'd0;
        end else begin
            if (start_pulse && ap_start_cnt  != 16'hFFFF) ap_start_cnt  <= ap_start_cnt  + 16'd1;
            if (kernel_done && tree_done_cnt != 16'hFFFF) tree_done_cnt <= tree_done_cnt + 16'd1;
        end
    end
    assign rearm_word = { tree_done_cnt, ap_start_cnt };

    // -----------------------------------------------------------------
    // 0xA4 — M9 boot stream running word-count + XOR-fold (bootcheck pattern).
    //   word-count saturates at 16b; XOR-fold reduced to 16b (lo^hi of the
    //   32b running XOR). Cleared at launch (start_pulse) so each forward's
    //   boot-stream replay can be re-checksummed independently.
    // -----------------------------------------------------------------
    reg [15:0] boot_wordcnt;
    reg [31:0] boot_xor;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            boot_wordcnt <= 16'd0; boot_xor <= 32'd0;
        end else if (start_pulse) begin
            boot_wordcnt <= 16'd0; boot_xor <= 32'd0;
        end else if (boot_stream_we) begin
            if (boot_wordcnt != 16'hFFFF) boot_wordcnt <= boot_wordcnt + 16'd1;
            boot_xor <= boot_xor ^ boot_stream_data;
        end
    end
    assign bootstream_word = { (boot_xor[31:16] ^ boot_xor[15:0]), boot_wordcnt };

endmodule

`default_nettype wire
