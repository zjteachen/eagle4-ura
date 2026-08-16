// =====================================================================
// dbg_stage_capture.sv  (PDI#10 PER-STAGE VALUE READBACK — capture-ALL-in-one-forward)
// =====================================================================
//
// MOTIVATION (reports/pdi9_validation/PDI10_DEBUG_REGS.md):
//   The PDI#10 60 MHz golden image FAILS on silicon (M7 frontier wrong) while
//   RTL + netlist are proven clean in sim (PDI10_60MHZ_SILICON_RCA.md). To find
//   the FIRST-DIVERGENT stage on the REAL silicon — not inferred from sim — the
//   host must read each stage's intermediate VALUE out of the flashed device.
//
//   The existing PROBE-V3 RAM (dbg_probe_ram) captures ONE host-selected stage
//   per forward (writing probe_sel CLEARS the RAM). Walking 17 stages then needs
//   17 forwards, each re-deriving inputs — unreliable for a single first-
//   divergence verdict. This block captures the first NW output words of ALL 17
//   stages in a SINGLE forward; the host then walks them via dbg_cap_stage_sel /
//   dbg_cap_word_idx / dbg_cap_val.
//
// DESIGN — flat per-(stage,word) register array, written word-parallel on each
//   stage's data-valid strobe (NO serial replay FSM, NO shared "busy"):
//     * Each stage's data is valid for ONE cycle (its strobe). A serial one-word-
//       per-cycle replay can MISS a strobe while draining a previous stage (and
//       is fragile to the comb→ff 1-cycle race). So instead, on each stage's
//       FIRST strobe this forward, we latch its first NW words IN PARALLEL into
//       cap[stage][0..NW-1] in a single cycle and set captured[stage]. No strobe
//       can ever be dropped; stages need not be temporally disjoint.
//     * Storage = NSTAGE*NW*32 FF (17*16*32 = 8704 FF) — trivial on V80 (5.15M
//       FF). No BRAM, so no LUTRAM-downgrade / combread / readmemh-drop hazard.
//
// TAP ALIGNMENT (memory feedback_valid_data_latency_skew_class — THE WHOLE POINT):
//   Each stage's strobe MUST be the DATA-VALID cycle, NOT the valid-flag assert.
//   The G1 arm in the TOP taps rms_g1_out_chunk_d1 (the d1-latched ALIGNED bus),
//   NOT the +1-skewed rms_g1_out_chunk_w (the §1 false-G1 silicon artifact). Every
//   tap is data-valid aligned (verified in xsim: host-walk == live aligned ref).
//
// PURELY ADDITIVE / FUNCTION-INVARIANT (memory feedback_no_accuracy_tradeoff):
//   Every input is a READ-ONLY tap off an existing net (the SAME nets the probe
//   mux taps), registered into cap[]. No new load on a functional read port; the
//   block drives nothing back into the datapath -> golden frontier BIT-IDENTICAL.
//
// SYNTH-SURVIVAL (memory feedback_nodriver_bake_drop_synth_only):
//   cap[] is a plain clocked register array (one write per stage per forward, a
//   registered read mux) — real drivers, no $readmemh, nothing dropped in synth.
//
// HOST INTERFACE (regs allocated above the PROBE-V3 / dump block):
//   0xDC W  dbg_cap_stage_sel[4:0] : stage code 0..18 (read-mux select; no clear).
//   0xE0 RW dbg_cap_word_idx[4:0]  : word index; AUTO-INCS after each 0xE4 read
//                                    (natural NW=32 wrap); WRITING seeks.
//   0xE4 R  dbg_cap_val            : cap[stage_sel][word_idx]; reading auto-incs.
//   0xE8 R  dbg_cap_meta           : {bit16=captured_this_fwd, [7:0]=nwords}.
//
// STAGE CODE MAP (PROBE_EXPANSION: 19 stages, 1:1 with probe_walk_capall.py):
//   0 G1  1 G4q  2 G4k  3 G4v  4 G5  5 G6  6 G7g  7 G7u  8 G7p  9 G9
//   10 G11  11 GLM  12 G12  13 TREE  14 R1  15 R3  16 R7
//   17 G8CHEAP (rescore INPUT cheap candidate vocab ids, 8 words)
//   18 G8RESC  (rescore OUTPUT ordered top-8 vocab ids, 8 words)
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module dbg_stage_capture #(
    parameter int NSTAGE   = 19,                        // PROBE_EXPANSION: +2 G8 stages (17->19)
    parameter int NW       = 32,                        // PROBE_EXPANSION: deeper (16->32) words/stage
    parameter int MAX_CHUNK = 128,                      // widest single-strobe chunk
    parameter int SSEL_W   = 5,                         // ceil log2 NSTAGE (19 -> 5)
    parameter int WIDX_W   = 5                          // ceil log2 NW (NW=32 -> 5)
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start_pulse,                     // forward launch -> re-arm

    // ---- per-stage tap feed (flat strobe/words/chunk; muxed in TOP) ----
    input  wire [NSTAGE-1:0]               strobe,      // 1-cyc data-valid pulse / stage
    input  wire [NSTAGE*8-1:0]             words,       // 8b word-count / stage
    input  wire [NSTAGE*MAX_CHUNK*32-1:0]  chunk,       // MAX_CHUNK*32 chunk / stage

    // ---- host control ----
    input  wire [SSEL_W-1:0] cap_stage_sel,             // 0xDC read-mux select
    input  wire              widx_we,                   // 1-cyc pulse on 0xE0 write
    input  wire [WIDX_W-1:0] widx_wdata,                // 0xE0 seek value
    input  wire              val_re,                    // 1-cyc pulse on 0xE4 read

    // ---- host-readable status words ----
    output wire [31:0] cap_word_idx_word,               // 0xE0 read-back
    output wire [31:0] cap_val_word,                    // 0xE4 cap[sel][idx]
    output wire [31:0] cap_meta_word,                   // 0xE8 {captured, nwords}
    output wire [31:0] cap_crc_word                     // 0x108 per-stage rotation-sensitive CRC32
);

    // -----------------------------------------------------------------
    // MAXPROBE upgrade #1 (MAXPROBE_INSTRUMENTATION_SPEC §2): per-stage
    // ROTATION-SENSITIVE CRC32 alongside the 32-word prefix. The cross-stream
    // signature today is XOR+sum (commutative -> order-blind); a CRC fold over the
    // SAME NW words a stage latches makes a reordered-but-same-multiset output (a
    // permutation bug) detectable. Same barrel-rotate-by-1 + 4-tap parity + XOR-data
    // step as dbg_residency.crc_step. Cost: NSTAGE*32 FF + small XOR cones.
    // -----------------------------------------------------------------
    function automatic [31:0] crc_step(input [31:0] c, input [31:0] d);
        logic fb;
        begin
            fb = c[31] ^ c[21] ^ c[1] ^ c[0];
            crc_step = ({c[30:0], fb}) ^ d;
        end
    endfunction

    // -----------------------------------------------------------------
    // Flat capture array cap[stage][word] + per-stage captured/nwords meta.
    // -----------------------------------------------------------------
    reg [31:0]         cap [0:NSTAGE-1][0:NW-1];
    reg [NSTAGE-1:0]   captured;                         // 1 once stage latched this fwd
    reg [7:0]          nwords_q [0:NSTAGE-1];            // # words latched for the stage
    reg [31:0]         cap_crc  [0:NSTAGE-1];            // MAXPROBE: per-stage CRC32 of captured words

    integer s, w;
    reg [31:0] crc_acc;
    reg [7:0]  nw_eff;
    always_ff @(posedge clk) begin
        if (!rst_n || start_pulse) begin
            captured <= '0;
            for (s = 0; s < NSTAGE; s = s + 1) begin
                nwords_q[s] <= 8'd0;
                cap_crc[s]  <= 32'd0;        // re-arm CRC each forward
            end
            // cap[] contents need not reset — captured[]=0 makes meta report
            // "not captured" and the host ignores uncaptured stages. (Leaving the
            // array unreset avoids 8704 reset loads; values are overwritten on
            // capture.) For clean sim, zero it on reset only.
            if (!rst_n)
                for (s = 0; s < NSTAGE; s = s + 1)
                    for (w = 0; w < NW; w = w + 1) cap[s][w] <= 32'd0;
        end else begin
            // For each stage: on its FIRST strobe this forward, latch first NW
            // words IN PARALLEL (one cycle) and record captured + nwords. A re-
            // strobe is ignored (captured[s] gates it). All stages independent ->
            // simultaneous strobes are all captured; none can be dropped.
            for (s = 0; s < NSTAGE; s = s + 1) begin
                if (strobe[s] && !captured[s] && (words[s*8 +: 8] != 8'd0)) begin
                    captured[s] <= 1'b1;
                    nw_eff = (words[s*8 +: 8] > 8'(NW)) ? 8'(NW) : words[s*8 +: 8];
                    nwords_q[s] <= nw_eff;
                    for (w = 0; w < NW; w = w + 1)
                        cap[s][w] <= chunk[s*MAX_CHUNK*32 + w*32 +: 32];
                    // MAXPROBE: fold the nw_eff captured words (in word order) into a
                    // rotation-sensitive CRC32 — same cycle, combinational over the
                    // chunk word slices, single registered write to cap_crc[s].
                    crc_acc = 32'd0;
                    for (w = 0; w < NW; w = w + 1)
                        if (w < int'(nw_eff))
                            crc_acc = crc_step(crc_acc, chunk[s*MAX_CHUNK*32 + w*32 +: 32]);
                    cap_crc[s] <= crc_acc;
                end
            end
        end
    end

    // -----------------------------------------------------------------
    // Host read pointer (0xE0). Seek on widx_we; auto-inc on 0xE4 read. WIDX_W
    // bits => natural wrap at NW=32 (NO explicit ">=NW" compare: WIDX_W'(32)==0
    // would make it always-true and stick rd_idx at 0 — the stuck-read trap).
    //
    // SEEK WINS over auto-inc in the SAME cycle (widx_we priority): the host walk
    // does an explicit per-word seek (0xE0=w) then reads 0xE4. If a host pipelines
    // the seek-write and the value-read so that widx_we and val_re land in the
    // same cycle, the SEEK must win so the returned word is exactly w (not w+1).
    // -----------------------------------------------------------------
    reg [WIDX_W-1:0] rd_idx;
    always_ff @(posedge clk) begin
        if (!rst_n)        rd_idx <= '0;
        else if (widx_we)  rd_idx <= widx_wdata;            // SEEK priority
        else if (val_re)   rd_idx <= rd_idx + WIDX_W'(1);   // natural wrap at NW
    end

    // -----------------------------------------------------------------
    // Read-back: COMBINATIONAL read mux (cap[sel][rd_idx]) — ZERO read latency.
    //
    // DOUBLING-FIX (PDI#11 PROBE_EXPANSION): the prior REGISTERED read mux
    //   `always_ff rd_data <= cap[sel][rd_idx]`
    // lagged rd_idx by one cycle. On the AXI path, control_regs pulses val_re in
    // the SAME AR-accept cycle that latches rdata <= cap_val_word, so the latched
    // rdata reflected cap[sel][rd_idx_OLD] while rd_idx had already (or was about
    // to) advance. Combined with the host's seek+double-read, that one-cycle skew
    // made every value appear at TWO consecutive word indices ([a,a,b,b], the
    // rd_idx≈w>>1 doubling). A registered read of a register array buys nothing
    // (cap[] is plain FF, not a sync-RAM macro) and only ADDS the skew.
    //
    // cap[] is a flat FF array, so cap[sel][rd_idx] is a pure combinational LUT
    // mux (17*16 = 272 32-bit words) — NOT a BRAM/LUTRAM synchronous read, so the
    // stale-by-1 COMBREAD trap (feedback_funcsim_slice_lutram_downgrade_trap) does
    // NOT apply: there is no RAM read primitive, no READ_FIRST, nothing to
    // downgrade. control_regs registers the value once (rdata <= cap_val_word at
    // AR-accept), giving the standard 1-cycle AXI-Lite read latency with NO extra
    // skew between rd_idx and the returned word.
    //
    // Host model after this fix: seek 0xE0=w, read 0xE4 ONCE -> returns cap[w]
    // exactly. The legacy throwaway-double-read still works (both reads return
    // cap[w] before any auto-inc takes effect, and the seek re-pins each word).
    // -----------------------------------------------------------------
    assign cap_word_idx_word = {{(32-WIDX_W){1'b0}}, rd_idx};
    assign cap_val_word      = cap[cap_stage_sel][rd_idx];   // comb, zero-lag
    assign cap_meta_word     = { 15'b0, captured[cap_stage_sel],
                                 8'b0,  nwords_q[cap_stage_sel] };
    // MAXPROBE: per-stage CRC of the 0xDC-selected stage's captured stream (0x108).
    assign cap_crc_word      = cap_crc[cap_stage_sel];

endmodule

`default_nettype wire
