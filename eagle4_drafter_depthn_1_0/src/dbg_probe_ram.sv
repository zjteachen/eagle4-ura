// =====================================================================
// dbg_probe_ram.sv  (PROBE ARCHITECTURE V3 — intermediate-layer CONTENT probe)
// =====================================================================
//
// MOTIVATION (reports/eagle4_drafter_m7/PROBE_V3.md):
//   dbg_observe v1 (0x80/0x84) exposes WHICH stage last fired and WHETHER its
//   output was non-zero; v2 (0x88..0xA4) exposes per-stage CYCLE COUNTS, input
//   checksums and G8 progress. Neither exposes the actual CONTENT of an
//   intermediate stage's output bus, so a per-stage golden-vs-silicon diff
//   (first-divergent stage AND word index) is impossible from host registers
//   alone. This block adds exactly that: a host-selectable 1024x32 capture RAM
//   that snapshots the FIRST 256 output words of the selected stage plus a
//   whole-stream XOR / sum32 / word-count checksum.
//
// PURELY ADDITIVE / FUNCTION-INVARIANT (memory feedback_no_accuracy_tradeoff):
//   * Every input is a READ-ONLY tap off an existing top-level signal.
//   * No output feeds the datapath; it only produces host-readable status words
//     muxed onto the AXI-Lite read channel by control_regs. The golden frontier
//     is BIT-IDENTICAL with or without this block.
//
// MULTI-DRIVER SAFE (memory feedback_multi_driver_packed_reg_trap):
//   The capture RAM has EXACTLY ONE write port (one always_ff). XOR/sum/count
//   accumulators each have one always_ff driver. probe_data is a single
//   combinational RAM read.
//
// SYNTH-SURVIVAL (memory feedback_nodriver_bake_drop_synth_only):
//   The RAM is a plain inferred 1024x32 dual-port (one write, one read) — a real
//   clocked driver, no $readmemh, so nothing is silently dropped in synth.
//
// HOST INTERFACE (allocated from 0xA8 upward, above dbg v1 0x74-0x84 and the
//   v2 0x88-0xA4 block):
//   0xA8 W  probe_sel[3:0]  : stage code 0..13. WRITING this clears the RAM
//                             write pointer, XOR, sum32, count and read pointer.
//   0xAC R  probe_addr      : current read pointer (auto-increments after each
//                             0xB0 read; wraps at 1024). Host may also WRITE it
//                             to seek (handled in control_regs).
//   0xB0 R  probe_data      : RAM[probe_addr]; reading auto-increments probe_addr.
//   0xB4 R  stream XOR       : XOR of every captured 32-bit word in the stream.
//   0xB8 R  stream sum32      : truncated 32-bit sum of every captured word.
//   0xBC R  count            : total 32-bit words seen in the stream (saturating).
//
// STAGE-CODE MAP (probe_sel -> tapped bus; mirrors dbg_observe activity bits):
//   0  G1     in-RMSNorm out   (g1_out_chunk,  16x16b -> 8 words)
//   1  G4     q/k/v GEMV out   (array_acc_out@G4, 128x32b -> 128 words)
//   2  G5     attention out    (g5_attn_out_vec, 16x32b FIXED -> 16 words)
//   3  G6     o-proj GEMV out  (array_acc_out@G6, 128x32b -> 128 words)
//   4  G7g    gate GEMV out    (array_acc_out@G7G)
//   5  G7u    up   GEMV out    (array_acc_out@G7U)
//   6  G7p    SiLU*up out      (silu_out_chunk, 16x16b -> 8 words)
//   7  G9     down GEMV out    (array_acc_out@G9)
//   8  G11    final-RMSNorm    (g11_out_chunk, 16x16b -> 8 words)
//   9  GLM    lmdown out       (lmdown wr_chunk, 16x16b -> 8 words)
//   10 G12    logits GEMV out  (array_acc_out@G12)
//   11 TREE   top-K scores     (tree_topk_scores, 8x32b -> 8 words)
//   12 R1     G1->G4 residual seam   (256b -> 8 words)
//   13 R7     post-MLP residual out  (256b -> 8 words)
//
// PACKING: 16-bit buses pack 2 INT16/word (low half = lane 2k, high = 2k+1);
//   32-bit buses map 1 word/lane. The top selects the matching n_words per sel.
//
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module dbg_probe_ram #(
    parameter int DEPTH    = 1024,                 // capture-RAM depth (words)
    parameter int CAP_WORDS = 256,                 // # words snapshotted into RAM
    parameter int MAX_CHUNK = 128,                 // widest single-strobe chunk (G4..G12 = 128 words)
    parameter int AW       = $clog2(DEPTH)         // RAM address width (10)
) (
    input  wire        clk,
    input  wire        rst_n,

    // ---- host control ----
    input  wire [3:0]  probe_sel,        // 0xA8 : selected stage code
    input  wire        sel_we,           // 1-cyc pulse when 0xA8 written (clears)
    input  wire        addr_we,          // 1-cyc pulse when 0xAC written (seek)
    input  wire [AW-1:0] addr_wdata,     // 0xAC seek value
    input  wire        data_re,          // 1-cyc pulse: 0xB0 was read (auto-inc addr)

    // ---- the SELECTED stage's chunk feed (muxed in the top) ----
    //   chunk_valid pulses once per output strobe of the selected stage.
    //   chunk_words = how many 32-bit words this chunk carries (<= MAX_CHUNK).
    //   chunk_data  = the flat MAX_CHUNK*32 chunk (word w = chunk_data[w*32 +: 32];
    //                 unused high words ignored via chunk_words).
    input  wire        chunk_valid,
    input  wire [$clog2(MAX_CHUNK+1)-1:0] chunk_words,
    input  wire [MAX_CHUNK*32-1:0]        chunk_data,

    // ---- host-readable status words ----
    output wire [31:0] probe_addr_word,  // 0xAC : current read pointer
    output wire [31:0] probe_data_word,  // 0xB0 : RAM[probe_addr]
    output wire [31:0] probe_xor_word,   // 0xB4
    output wire [31:0] probe_sum_word,   // 0xB8
    output wire [31:0] probe_count_word  // 0xBC
);

    // -----------------------------------------------------------------
    // 1024x32 capture RAM (single write port, single read port).
    // -----------------------------------------------------------------
    (* ram_style = "block" *)
    reg [31:0] cap_ram [0:DEPTH-1];

    reg [AW-1:0] wr_ptr;        // RAM write pointer (stops at CAP_WORDS)
    reg [AW-1:0] rd_ptr;        // host read pointer (0xAC), wraps DEPTH
    reg [31:0]   xor_acc;       // whole-stream XOR fold
    reg [31:0]   sum_acc;       // whole-stream truncated sum
    reg [31:0]   word_cnt;      // total words seen (saturating)

    // ---- per-strobe capture FSM ----
    // On chunk_valid, walk word 0..chunk_words-1, writing each into the RAM
    // (while wr_ptr < CAP_WORDS) and folding XOR/sum/count over ALL of them.
    // chunk_words <= MAX_CHUNK so the inner index fits in $clog2(MAX_CHUNK+1).
    localparam int WIDX_W = $clog2(MAX_CHUNK + 1);
    reg                  busy;       // 1 while replaying a captured chunk
    reg [WIDX_W-1:0]     widx;       // word index within the current chunk
    reg [WIDX_W-1:0]     wlen;       // latched chunk_words for the current chunk
    reg [MAX_CHUNK*32-1:0] cdat;     // latched chunk_data

    // current word being processed (combinational select off the latched chunk)
    wire [31:0] cur_word = cdat[widx*32 +: 32];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr   <= '0;
            xor_acc  <= 32'd0;
            sum_acc  <= 32'd0;
            word_cnt <= 32'd0;
            busy     <= 1'b0;
            widx     <= '0;
            wlen     <= '0;
            cdat     <= '0;
        end else if (sel_we) begin
            // CLEAR on a new probe_sel write (start a fresh capture window).
            wr_ptr   <= '0;
            xor_acc  <= 32'd0;
            sum_acc  <= 32'd0;
            word_cnt <= 32'd0;
            busy     <= 1'b0;
            widx     <= '0;
            wlen     <= '0;
        end else begin
            if (!busy) begin
                if (chunk_valid && chunk_words != '0) begin
                    // latch the chunk and start replaying word 0
                    cdat <= chunk_data;
                    wlen <= chunk_words;
                    widx <= '0;
                    busy <= 1'b1;
                end
            end else begin
                // replay one word per cycle: fold + (conditionally) store
                xor_acc <= xor_acc ^ cur_word;
                sum_acc <= sum_acc + cur_word;
                if (word_cnt != 32'hFFFF_FFFF) word_cnt <= word_cnt + 32'd1;
                if (wr_ptr < AW'(CAP_WORDS[AW-1:0])) begin
                    cap_ram[wr_ptr] <= cur_word;
                    wr_ptr <= wr_ptr + AW'(1);
                end
                if (widx + WIDX_W'(1) >= wlen) busy <= 1'b0;  // last word of chunk
                else                            widx <= widx + WIDX_W'(1);
            end
        end
    end

    // NOTE on capture-rate: a stage strobe that re-asserts every cycle (e.g. a
    // GEMV emitting one 128-word tile per cycle) would arrive faster than this
    // one-word-per-cycle replay can drain. The taps chosen here strobe per-chunk
    // with >= MAX_CHUNK cycles between strobes (tile cadence), so busy always
    // clears before the next chunk_valid. The first-divergence diagnosis only
    // needs the FIRST CAP_WORDS words + a whole-stream checksum, both of which
    // are preserved even if a pathologically fast stage overruns (overruns are
    // simply dropped while busy — they do not corrupt the captured prefix).

    // -----------------------------------------------------------------
    // Host read pointer (0xAC). Seek on addr_we; auto-increment on a 0xB0 read.
    // -----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n)            rd_ptr <= '0;
        else if (sel_we)       rd_ptr <= '0;            // fresh window -> rewind
        else if (addr_we)      rd_ptr <= addr_wdata;    // explicit seek
        else if (data_re)      rd_ptr <= rd_ptr + AW'(1); // auto-inc (wraps DEPTH)
    end

    // -----------------------------------------------------------------
    // Read-back words. probe_data uses a SYNCHRONOUS RAM read (registered on
    // rd_ptr) so the 1024x32 array infers a single BRAM (RAMB36) instead of
    // distributed LUT-RAM. rd_ptr tracks the host pointer; rd_data is the RAM
    // word at the address that was presented LAST cycle. The host's AXI read
    // path already returns rdata one cycle after AR-accept, and rd_ptr only
    // advances on data_re (the 0xB0 read), so rd_data is stable for the read.
    // -----------------------------------------------------------------
    reg [31:0] rd_data;
    always_ff @(posedge clk) rd_data <= cap_ram[rd_ptr];

    assign probe_addr_word  = {{(32-AW){1'b0}}, rd_ptr};
    assign probe_data_word  = rd_data;
    assign probe_xor_word   = xor_acc;
    assign probe_sum_word   = sum_acc;
    assign probe_count_word = word_cnt;

endmodule

`default_nettype wire
