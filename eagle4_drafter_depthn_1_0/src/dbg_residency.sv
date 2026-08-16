// =====================================================================
// dbg_residency.sv  (PDI#11 PROBE_EXPANSION — BOOT-LOAD RESIDENCY CRC + COUNT)
// =====================================================================
//
// MOTIVATION (reports/pdi9_validation/PROBE_EXPANSION.md):
//   The PDI#10 60 MHz silicon M7 FAIL traces to WRONG INPUT DATA on silicon
//   (h_fc/gamma load-path, memory project_pdi10_g1_silicon_not_readlatency;
//   the M7 +128 chunk-rotation, project_m7_loadgeom_plus128_rotation). To tell
//   "the load wrote the WRONG bytes" from "the load wrote NOTHING" DIRECTLY on
//   the flashed device — without reading every buffer back word by word — this
//   block accumulates a per-buffer CRC-style signature + a SUM + a WORD COUNT of
//   exactly what the load ports WRITE into each on-chip buffer, in the order they
//   are written. The host reads these back and compares to a host-computed golden
//   signature of the SAME word stream:
//       count==0           -> load wrote NOTHING (port/select/addr wedge)
//       count==golden, but
//       crc != golden_crc  -> load wrote the WRONG bytes (rotation / mis-pack)
//       crc == golden_crc  -> the buffer is byte-correct on silicon
//
// COVERED BUFFERS (each = an independent {crc, sum, count} accumulator):
//   0 EMB     input_emb_buf       (emb_we   / emb_wr_data32)
//   1 HFC     h_fc_buf            (hfc_we   / hfc_wr_data32)
//   2 GAMMA   4 gamma RMSNorm bufs (gamma_we / gamma_wr_data32; gsel tags which)
//   3 WEIGHT  qweight URAM banks  (w_we     / w_wr_data : eff_boot_we/eff_boot_wr_data)
//   4 SCALE   scale BRAM          (sc_we    / sc_wr_data : dma_sc_we/dma_sc_wr_data)
//
// CRC: a cheap LFSR-style fold that is order- AND value-sensitive (so a +128
//   chunk ROTATION of the SAME words yields a DIFFERENT crc, unlike a plain
//   commutative XOR/sum). crc <= {crc[30:0], crc[31]^crc[21]^crc[1]^crc[0]} ^ data
//   i.e. a barrel-rotate-by-1 with a 4-tap parity feedback, then XOR the data.
//   Rotation-sensitive because each word is mixed at a position-dependent shift.
//   (The SUM is kept too — commutative, catches value-but-not-order errors and is
//   trivially reproducible host-side as a cross-check.)
//
// ADDRESS-AWARE FOLD (the M7 +128 load-permutation instrument — body-research
//   BODY_DIVERGENCE_RESEARCH.md §137, recommendation #1): for the chunk-addressed
//   boot buffers (EMB/HFC/GAMMA) the CRC mixes the LANDING ADDRESS {wr_chunk,
//   wr_word} together with the data: crc <= crc_step(crc, data ^ addr_tag). So if
//   the load writes the CORRECT data to the WRONG chunk (a +128 chunk remap that a
//   pure write-data CRC would miss because the data STREAM is unchanged), the
//   address tag perturbs the CRC -> it diverges from the golden, surfacing the
//   permutation on silicon. WEIGHT/SCALE fold data only (their addressing is
//   already validated bit-exact on silicon, project_pdi9_residency_clean).
//
// PURELY ADDITIVE / READ-ONLY (memory feedback_no_accuracy_tradeoff):
//   Every input is a READ-ONLY tap off a write-enable + write-data net that the
//   load path already drives into the real buffer. This block drives nothing back
//   into the datapath -> golden frontier BIT-IDENTICAL. Plain clocked registers
//   (no $readmemh, no RAM) -> synth-safe (feedback_nodriver_bake_drop_synth_only).
//
// CLEAR POLICY (FIX — do NOT clear on start_pulse): the boot loads (weights,
//   scales, emb/h_fc/gammas) ALL happen BEFORE AP_START. Clearing on start_pulse
//   would wipe the load history at the very moment the forward begins, leaving the
//   host to read count=0 ("WROTE-NOTHING") for buffers that WERE loaded — exactly
//   the false-negative the first walk gate caught. So the accumulators persist
//   across loads AND the forward, cleared ONLY on rst_n (cold-AC/reset) OR an
//   EXPLICIT host clear (0xEC write with bit[3]=1 -> res_clear pulse). The host
//   reads the residency AFTER the loads (before or after the forward — both valid);
//   to re-measure a fresh load it pulses res_clear first.
//
// HOST INTERFACE (regs above the per-stage value block 0xDC..0xE8):
//   0xEC W  res_sel[2:0]  : buffer select 0..4 (read mux). bit[3]=res_clear pulse.
//   0xF0 R  res_crc       : CRC signature of the selected buffer's write stream.
//   0xF4 R  res_sum       : 32b running SUM of the selected buffer's write words.
//   0xF8 R  res_count     : # 32b words written to the selected buffer this fwd.
//   (gamma sub-tag in res_crc readback context: the GAMMA accumulator folds ALL
//    4 gammas in load order; the host streams the 4 gammas in the SAME fixed order
//    so the golden matches. A per-gamma split is unnecessary for load-correctness.)
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module dbg_residency #(
    parameter int ADDR_TAG_W = 16,         // {wr_chunk, wr_word} tag width
    parameter int RD_TAG_W   = 8           // §6: registered read chunk-index width (EMB_CW=8)
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        res_clear,          // host-pulsed clear (0xEC bit[3]); NOT start_pulse

    // ---- write-side taps (READ-ONLY) ----
    input  wire        emb_we,
    input  wire [31:0] emb_wr_data,
    input  wire [ADDR_TAG_W-1:0] emb_addr_tag,   // {emb_wr_chunk, emb_wr_word}
    input  wire        hfc_we,
    input  wire [31:0] hfc_wr_data,
    input  wire [ADDR_TAG_W-1:0] hfc_addr_tag,
    input  wire        gamma_we,
    input  wire [31:0] gamma_wr_data,
    input  wire [ADDR_TAG_W-1:0] gamma_addr_tag,
    input  wire        w_we,
    input  wire [31:0] w_wr_data,
    input  wire        sc_we,
    input  wire [31:0] sc_wr_data,         // scale words are 16b; host zero-extends

    // ---- §6 READ-SIDE taps (PLUS128_LOADPATH_FIX_V2 §6) — observe the +128 fix.
    //   The WRITE-side CRC (idx 0-4) is BLIND to the read-side +128 slip (write addr
    //   is clean). Fold the chunk index the G1 read ACTUALLY presents together with
    //   the data it gets back, in apply order, so res_sel=5 (HFC read) / 6 (EMB read)
    //   tell the flashed card per-buffer whether the READ map became golden. The
    //   idx tag is mixed into the CRC so a +128/+64 chunk slip (same data, wrong
    //   chunk) perturbs it. READ-ONLY; drives nothing into the datapath. ----
    input  wire        g1_rd_stb,          // 1-cyc strobe when g1_x_chunk (HFC read) is valid
    input  wire [RD_TAG_W-1:0] g1_rd_idx_tap,    // the REGISTERED read index actually used
    input  wire [31:0] g1_rd_word_tap,           // low 32b of g1_x_chunk
    input  wire        g1e_rd_stb,
    input  wire [RD_TAG_W-1:0] g1e_rd_idx_tap,
    input  wire [31:0] g1e_rd_word_tap,          // low 32b of g1e_x_chunk (EMB read)

    // ---- host control + readback ----
    input  wire [2:0]  res_sel,            // 0xEC buffer select (0..6)
    output wire [31:0] res_crc_word,       // 0xF0
    output wire [31:0] res_sum_word,       // 0xF4
    output wire [31:0] res_count_word      // 0xF8
);
    localparam int NBUF = 7;               // 0EMB 1HFC 2GAMMA 3WEIGHT 4SCALE 5RDHFC 6RDEMB

    reg [31:0] crc   [0:NBUF-1];
    reg [31:0] sum   [0:NBUF-1];
    reg [31:0] cnt   [0:NBUF-1];

    // one barrel-rotate-by-1 + 4-tap parity feedback, then XOR data (position-
    // dependent mix -> rotation-sensitive). Inlined per buffer for a clean single
    // driver each.
    function automatic [31:0] crc_step(input [31:0] c, input [31:0] d);
        logic fb;
        begin
            fb = c[31] ^ c[21] ^ c[1] ^ c[0];
            crc_step = ({c[30:0], fb}) ^ d;
        end
    endfunction

    localparam int NWBUF = 5;              // the 5 WRITE-side buffers (idx 0..4)
    genvar gi;
    generate
        for (gi=0; gi<NWBUF; gi=gi+1) begin : g_acc
            wire        we_i = (gi==0) ? emb_we   :
                               (gi==1) ? hfc_we   :
                               (gi==2) ? gamma_we :
                               (gi==3) ? w_we     : sc_we;
            wire [31:0] d_i  = (gi==0) ? emb_wr_data   :
                               (gi==1) ? hfc_wr_data   :
                               (gi==2) ? gamma_wr_data :
                               (gi==3) ? w_wr_data     : sc_wr_data;
            // ADDRESS-AWARE tag: EMB/HFC/GAMMA mix the LANDING ADDRESS so a +128
            // chunk remap (correct data -> wrong chunk) perturbs the CRC. WEIGHT/
            // SCALE use 0 (data-only fold; addressing already silicon-validated).
            wire [31:0] tag_i = (gi==0) ? {{(32-ADDR_TAG_W){1'b0}}, emb_addr_tag}   :
                                (gi==1) ? {{(32-ADDR_TAG_W){1'b0}}, hfc_addr_tag}   :
                                (gi==2) ? {{(32-ADDR_TAG_W){1'b0}}, gamma_addr_tag} :
                                          32'd0;
            always_ff @(posedge clk) begin
                if (!rst_n || res_clear) begin     // FIX: NOT start_pulse (loads precede AP_START)
                    crc[gi] <= 32'd0; sum[gi] <= 32'd0; cnt[gi] <= 32'd0;
                end else if (we_i) begin
                    crc[gi] <= crc_step(crc[gi], d_i ^ tag_i);   // data XOR addr-tag
                    sum[gi] <= sum[gi] + d_i;                    // pure data sum (addr-independent xcheck)
                    if (cnt[gi] != 32'hFFFF_FFFF) cnt[gi] <= cnt[gi] + 32'd1;
                end
            end
        end
    endgenerate

    // -----------------------------------------------------------------
    // §6 READ-SIDE accumulators (idx 5 = HFC read stream, idx 6 = EMB read stream).
    // Fold (word ^ idx-tag) on the read strobe, in apply order. A single always_ff
    // each (clean single-driver). Same clear policy as the write side (rst_n or
    // host res_clear; NOT start_pulse — the read stream runs DURING the forward, but
    // the host clears explicitly before re-measuring). The host golden streams chunk
    // 0..191 in natural order with each chunk's known low-word and compares.
    // -----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n || res_clear) begin
            crc[5] <= 32'd0; sum[5] <= 32'd0; cnt[5] <= 32'd0;
            crc[6] <= 32'd0; sum[6] <= 32'd0; cnt[6] <= 32'd0;
        end else begin
            if (g1_rd_stb) begin
                crc[5] <= crc_step(crc[5],
                            g1_rd_word_tap ^ {{(32-RD_TAG_W){1'b0}}, g1_rd_idx_tap});
                sum[5] <= sum[5] + g1_rd_word_tap;
                if (cnt[5] != 32'hFFFF_FFFF) cnt[5] <= cnt[5] + 32'd1;
            end
            if (g1e_rd_stb) begin
                crc[6] <= crc_step(crc[6],
                            g1e_rd_word_tap ^ {{(32-RD_TAG_W){1'b0}}, g1e_rd_idx_tap});
                sum[6] <= sum[6] + g1e_rd_word_tap;
                if (cnt[6] != 32'hFFFF_FFFF) cnt[6] <= cnt[6] + 32'd1;
            end
        end
    end

    // combinational read mux (NBUF=7 entries -> trivial). res_sel is 3b (0..7); mux
    // clamps to 0 for the unused sel=7.
    assign res_crc_word   = crc[res_sel[2:0] < 3'(NBUF) ? res_sel : 3'd0];
    assign res_sum_word   = sum[res_sel[2:0] < 3'(NBUF) ? res_sel : 3'd0];
    assign res_count_word = cnt[res_sel[2:0] < 3'(NBUF) ? res_sel : 3'd0];

endmodule

`default_nettype wire
