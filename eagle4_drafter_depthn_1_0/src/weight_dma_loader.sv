// =====================================================================
// weight_dma_loader.sv
// M9: boot-only AXI4 read master + weight/scale-load FSM (HBM → URAM+BRAM).
// =====================================================================
//
// WHY (SPEC_M9_BOOT_MAXI.md):
//   M1c's AXI-Lite weight-load (0x48/4c/50) is ~9.7M register writes for
//   the full 48 MB drafter → minutes. The production route: host bulk-DMAs
//   the packed qweights to an HBM aperture (fast QDMA H2C), then THIS FSM
//   reads HBM→URAM via a single m_axi read port and streams each returned
//   word into the per-lane URAM banks at the M1c flat addresses.
//
// SCALE BOOT PORT (SCALE-BOOT-PORT fork, 2026-05-23 — closes the PDI #5
//   boot-load gap):
//   The scale_bram (per-lane Q3.13 dequant scales, inside
//   gemv_stage_array_scaled) has a SEPARATE write port
//   (sc_we/sc_lane_sel/sc_wr_addr/sc_wr_data) that NOTHING drove on silicon
//   — so the scaled GEMV booted with empty scales and the 8/8 SGLang
//   contract failed on hardware (xsim worked only because the TB
//   $readmemh-preloaded the scale_bram). This FSM is now a UNIFIED
//   weight+scale boot loader: a per-load `boot_is_scale` mode bit selects
//   which bank the same address-walk FSM streams into.
//
//   * WEIGHT mode (boot_is_scale=0): unchanged. One INT32 qweight word per
//     32-bit AXI beat → URAM weight port (dma_we/dma_lane_sel/dma_wr_addr/
//     dma_wr_data). cpc=K/16 chunks/col, stage_base = W_stage_base.
//   * SCALE  mode (boot_is_scale=1): the scale .bin is packed uint16 LE
//     (Q3.13), so each 32-bit AXI beat carries TWO consecutive scale words
//     (low half = even word, high half = odd word — matching numpy
//     `<u2`.tobytes() little-endian). The FSM emits TWO scale-bram writes
//     per beat to sc_we/sc_lane_sel/sc_wr_addr/sc_wr_data. Here "cpc" is
//     reinterpreted as gpc=K/GROUP groups/col and stage_base = S_stage_base.
//     The (col,inner) walk is IDENTICAL — only the inner extent (gpc vs
//     cpc), the destination port, and the 2-words-per-beat unpack differ.
//     All scale streams have even (N×gpc) word counts (both even) → the
//     2-per-beat unpack divides cleanly for every stage (G4..G12).
//
//   Multi-driver-safe: still ONE m_axi ARVALID driver (boot-only), ONE
//   always_ff. The weight and scale ports are written in disjoint loads
//   (the host issues weight loads then scale loads, never overlapping), and
//   each port is driven by exactly this one FSM.
//
// SILENT DURING COMPUTE (the whole point — DMA_DRIVER_GUIDE.md §6):
//   The FSM only asserts ARVALID during the boot phase (between
//   boot_dma_start and boot_done). Once boot_done latches it parks in
//   S_DONE with ARVALID/RVALID-ready deasserted — a SINGLE ARVALID driver,
//   boot-only, so the >5-driver NoC-fan-in deadlock class does NOT apply
//   (that rule is about *compute-time* concurrency).
//
// ADDRESS GENERATION (mirrors ref/m1_g6_extract_and_golden.py exactly):
//   The host arranges the HBM buffer as raw 32-bit qweight words in
//   (col, chunk) nested order — the natural extractor order. This FSM
//   walks the same nested order and DERIVES the URAM target:
//       col       = col_start + col_idx * col_step      (global out column)
//       lane      = col % LANES                          (which lane's bank)
//       local_col = col / LANES                          (col within bank)
//       flat_addr = stage_base + local_col*cpc + chunk
//   so word N of the HBM stream lands at exactly the slot the M1c
//   AXI-Lite path would have written. (For the full per-stage load:
//   col_start=0, n_cols=N, col_step=1. For a single-lane verification:
//   col_start=0, n_cols=N/LANES, col_step=LANES → all words hit lane 0.)
//
//   Geometry comes from control regs so all 6 GEMV stages load without a
//   recompile (stage_base/cpc per M1_WEIGHT_MAP.md).
//
// AXI4 READ (read-only master — no AW/W/B channels):
//   Fixed-length INCR bursts of BURST_LEN beats (32-bit each). Address is
//   the HBM byte base + word_index*4. Single outstanding burst (AR issued,
//   then drain R beats) — simplest possible, deadlock-free, and boot
//   latency is irrelevant (<1 ms target dwarfs any per-burst stall).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module weight_dma_loader #(
    parameter int LANES      = 128,
    parameter int AXI_ADDR_W = 64,
    parameter int AXI_DATA_W = 32,    // one packed qweight INT32 per beat
    parameter int AXI_ID_W   = 1,
    parameter int BURST_LEN  = 64,    // beats per INCR burst (<=256)
    parameter int URAM_ADDR_W = 17,   // weight_bank flat address width
    parameter int SCALE_ADDR_W = 17,  // scale_bram flat address width
    parameter int SCALE_W      = 16,  // Q3.13 scale word width
    parameter int COL_W        = 20,  // column-count width. MUST cover the
                                      // largest stage N (G12 lm = 128256 >
                                      // 16 bits) — the old [15:0] truncated
                                      // G12 to 62720 cols, dropping the high
                                      // local_cols of every lane (SCALE-BOOT-
                                      // PORT fork found this in xsim).
    // M9 FIX-B (2026-06-11, M9_FIX.md): bounded R-wait watchdog. Number of
    // consecutive STALL cycles (arvalid&&!arready OR rready&&!rvalid) tolerated
    // on the m_axi_boot read path before the watchdog aborts the load. 1<<20 ≈
    // 1.05M cycles ≈ 17.5 ms @ 60 MHz — orders of magnitude beyond any healthy
    // HBM4 read latency, but a tiny fraction of the "permanent hang until
    // cold-AC" the d10g8_dbg P&R produced. Converts a silicon wedge into a
    // host-detectable, re-armable failure (0x74[2] sticky).
    parameter int BOOT_RWAIT_MAX = (1 << 20),
    parameter int RWAIT_W        = 21 // $clog2(BOOT_RWAIT_MAX)+1 ; holds 0..1<<20
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- Boot-DMA control (from control_regs 0x54/0x58/0x5c/...) ----
    input  wire                       boot_dma_start,    // 1-cycle pulse: kick off boot load
    input  wire                       boot_is_scale,     // 0=weight load (URAM), 1=scale load (BRAM)
    // FASTIN (2026-07-09): per-call activation DMA targets. When set (0x54 bit[2]
    // emb / bit[3] hfc, mutually exclusive with each other and with is_scale — the
    // scale bit WINS if illegally combined), the WEIGHT-mode stream is re-routed at
    // the TOP from the URAM banks to input_emb_buf / h_fc_buf. The FSM/address walk
    // is UNCHANGED (host programs the flat walk: n_cols=1, col_start=0, cpc=1536,
    // stage_base=0 → dma_wr_addr = flat word 0..1535). Latched at start like
    // is_scale_r so a mid-load host reg change cannot disturb an in-flight load.
    input  wire                       boot_is_emb,       // 0x54[2]: stream → input_emb_buf
    input  wire                       boot_is_hfc,       // 0x54[3]: stream → h_fc_buf
    input  wire [AXI_ADDR_W-1:0]      boot_hbm_base,     // HBM byte base of the stream
    input  wire [COL_W-1:0]           boot_n_cols,       // # output columns to walk (up to 128256)
    input  wire [COL_W-1:0]           boot_col_start,    // first global column
    input  wire [COL_W-1:0]           boot_col_step,     // column stride (1=full stage, LANES=single-lane)
    input  wire [15:0]                boot_cpc,          // inner extent per col: weight=K/16 chunks, scale=K/GROUP groups
    input  wire [URAM_ADDR_W-1:0]     boot_stage_base,   // flat base for the stage (URAM W base OR BRAM S base)

    // ---- Status ----
    output reg                        boot_busy,
    output reg                        boot_done,         // sticky-high after a completed load
    // M9 FIX-B: sticky-high after the R-wait watchdog aborts a wedged load.
    // Routed to control_regs 0x74 bit[2]. Stays high until the next clean
    // load completes (boot_done re-raised) so the host can read it after a
    // wedge, retry, and confirm recovery — all without a cold-AC.
    output reg                        boot_timeout,

    // FASTIN: latched target of the CURRENT load, valid whenever dma_we pulses.
    // The top splits dma_we with these (URAM / input_emb_buf / h_fc_buf) exactly
    // like the ctrl_boot_we_wt/_emb/_hfc 3-way AXI-Lite split. Stable for the
    // whole load (latched at start), so every dma_we pulse of one load routes to
    // ONE target — single-driver-safe per write port.
    output wire                       dma_tgt_emb,
    output wire                       dma_tgt_hfc,

    // ---- URAM weight write port (muxed with M1c AXI-Lite path at top level) ----
    output reg                        dma_we,
    output reg  [15:0]                dma_lane_sel,
    output reg  [URAM_ADDR_W-1:0]     dma_wr_addr,
    output reg  [AXI_DATA_W-1:0]      dma_wr_data,

    // ---- BRAM scale write port (drives scale_bram sc_* via gemv_stage_array_scaled) ----
    output reg                        sc_we,
    output reg  [15:0]                sc_lane_sel,
    output reg  [SCALE_ADDR_W-1:0]    sc_wr_addr,
    output reg  [SCALE_W-1:0]         sc_wr_data,

    // ---- AXI4 read master (read-only: AR + R channels) ----
    output reg  [AXI_ID_W-1:0]        m_axi_boot_arid,
    output reg  [AXI_ADDR_W-1:0]      m_axi_boot_araddr,
    output reg  [7:0]                 m_axi_boot_arlen,
    output reg  [2:0]                 m_axi_boot_arsize,
    output reg  [1:0]                 m_axi_boot_arburst,
    output reg                        m_axi_boot_arvalid,
    input  wire                       m_axi_boot_arready,

    input  wire [AXI_ID_W-1:0]        m_axi_boot_rid,
    input  wire [AXI_DATA_W-1:0]      m_axi_boot_rdata,
    input  wire [1:0]                 m_axi_boot_rresp,
    input  wire                       m_axi_boot_rlast,
    input  wire                       m_axi_boot_rvalid,
    output reg                        m_axi_boot_rready
);

    localparam [2:0] ARSIZE_32 = 3'b010;   // 4 bytes/beat
    localparam [1:0] BURST_INCR = 2'b01;

    // ----------------------------------------------------------------
    // Latched geometry (snapshot on boot_dma_start so the host can change
    // regs without disturbing an in-flight load)
    // ----------------------------------------------------------------
    reg [AXI_ADDR_W-1:0]  hbm_base_r;
    reg                   is_scale_r;       // 0=weight (URAM), 1=scale (BRAM)
    // FASTIN: latched activation-buffer targets (mutually exclusive; scale wins —
    // both are force-cleared when boot_is_scale is set at the same start, and the
    // emb bit wins over an illegal emb+hfc double-set, mirroring the top's
    // ctrl_boot_we_emb/_hfc priority comments in control_regs.sv:112).
    reg                   is_emb_r;         // weight-mode stream → input_emb_buf
    reg                   is_hfc_r;         // weight-mode stream → h_fc_buf
    assign dma_tgt_emb = is_emb_r;
    assign dma_tgt_hfc = is_hfc_r;
    reg [COL_W-1:0]       n_cols_r;
    reg [COL_W-1:0]       col_start_r;
    reg [COL_W-1:0]       col_step_r;
    reg [15:0]            cpc_r;            // inner extent/col (chunks OR groups)
    reg [URAM_ADDR_W-1:0] stage_base_r;

    // total words to fetch = n_cols * cpc
    reg [31:0]            total_words_r;

    // ----------------------------------------------------------------
    // Walk counters: col_idx (0..n_cols-1), chunk (0..cpc-1).
    // These advance once per word consumed from the R channel — the URAM
    // target (lane/local_col/flat_addr) is derived combinationally.
    // ----------------------------------------------------------------
    reg [COL_W-1:0]       col_idx;       // which column (index, not global col)
    reg [15:0]            chunk_idx;
    reg [31:0]            words_done;

    // current global column = col_start + col_idx*col_step
    wire [31:0] cur_col   = col_start_r + col_idx * col_step_r;
    // lane = col % LANES ; local_col = col / LANES  (LANES is a power of two = 128)
    localparam int LANE_BITS = $clog2(LANES);
    wire [15:0] cur_lane  = {{(16-LANE_BITS){1'b0}}, cur_col[LANE_BITS-1:0]};
    wire [15:0] cur_lcol  = cur_col[LANE_BITS +: 16];
    // flat target = stage_base + local_col*inner_per_col + inner_idx. The same
    // formula serves the URAM weight bank (inner=chunk, cpc=K/16) and the BRAM
    // scale bank (inner=group, cpc=gpc=K/GROUP) — only the inner extent & base
    // differ (both supplied via the same regs by the host per load).
    wire [URAM_ADDR_W-1:0] cur_addr  = stage_base_r + (cur_lcol * cpc_r) + chunk_idx;
    // scale mode packs 2 uint16 words/beat: the beat's 2nd word lands at the
    // next inner slot (same col — gpc is even for every stage so a 2-word beat
    // never straddles a column boundary).
    wire [URAM_ADDR_W-1:0] cur_addr2 = cur_addr + {{(URAM_ADDR_W-1){1'b0}}, 1'b1};

    // ----------------------------------------------------------------
    // Burst bookkeeping: beats remaining in the in-flight burst.
    // ----------------------------------------------------------------
    reg [8:0]             beats_left;     // beats still to drain in current burst
    reg [AXI_ADDR_W-1:0]  next_araddr;    // running HBM byte address for next burst
    reg [31:0]            words_remaining; // words still to request (for last-burst sizing)
    reg [AXI_ADDR_W-1:0]  cur_burst_bytes; // bytes in the in-flight burst (beats*4)

    // ----------------------------------------------------------------
    // M9 FIX-B: R-wait watchdog. Counts CONSECUTIVE cycles the m_axi_boot read
    // path is stalled — either waiting for AR to be accepted (arvalid&&!arready)
    // or waiting for an R-beat after rready is up (rready&&!rvalid). Any forward
    // progress (an accepted AR or an accepted R-beat) resets it to 0. On expiry
    // we abort: deassert arvalid/rready, set sticky boot_timeout, drop busy, and
    // return to S_IDLE (NOT S_DONE) so the host can re-arm with a fresh start.
    // ----------------------------------------------------------------
    reg [RWAIT_W-1:0]     rwait_cnt;
    // stall this cycle = AR outstanding-unaccepted OR R-intake-open-but-empty.
    wire boot_stalled = (m_axi_boot_arvalid && !m_axi_boot_arready)
                     || (m_axi_boot_rready  && !m_axi_boot_rvalid);
    // only arm the watchdog while a load is actually in flight (not in
    // S_IDLE/S_DONE where arvalid/rready are low and boot_stalled is 0 anyway,
    // but gating on busy makes the intent explicit and keeps the counter parked).
    wire wd_expired = (rwait_cnt >= RWAIT_W'(BOOT_RWAIT_MAX));

    // ----------------------------------------------------------------
    // FSM
    // ----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE    = 3'd0,
        S_ARREQ   = 3'd1,   // assert ARVALID, wait ARREADY
        S_RDATA   = 3'd2,   // drain R beats, stream to URAM (or even scale word)
        S_SCFLUSH = 3'd3,   // scale mode only: write the odd (high) scale word
        S_DONE    = 3'd4    // parked; ARVALID low forever (silent during compute)
    } state_e;
    state_e state;

    // ---- scale-mode odd-word holding slot (single-write-port-safe) ----
    // In scale mode each 32-bit beat carries TWO uint16 words; the scale_bram
    // has ONE write port, so the EVEN word is written in S_RDATA and the ODD
    // word in S_SCFLUSH the next cycle. burst_done_after_flush remembers that
    // this beat ended the burst so the flush returns to S_ARREQ.
    reg [15:0]            sc2_lane;
    reg [URAM_ADDR_W-1:0] sc2_addr;
    reg [SCALE_W-1:0]     sc2_data;
    reg                   burst_done_after_flush;

    // ---- pending-start latch (S_DONE re-arm trap fix) ----------------------
    // A boot_dma_start pulse received while a load is IN FLIGHT (S_ARREQ/S_RDATA/
    // S_SCFLUSH) cannot start a second load immediately (single outstanding burst
    // FSM). Instead we LATCH it (+ snapshot the geometry visible at that pulse) and
    // honour it the instant the in-flight load completes. This keeps the loader
    // compatible with BOTH host flows:
    //   * unpatched host (1 pulse/load): the pulse lands in S_DONE → starts the new
    //     load directly with the freshly-programmed geometry (no skip).
    //   * patched host (re-arm + real start): the re-arm pulse lands in S_DONE with
    //     the PREVIOUS stage's stale geometry → starts a (harmless) re-load; the
    //     real start pulse arrives mid-flight → latched here with the FRESH scale
    //     geometry → runs next, so the correct stage is loaded last. Idempotent.
    reg                   pending_start;
    reg [AXI_ADDR_W-1:0]  p_hbm_base;
    reg                   p_is_scale;
    reg                   p_is_emb;      // FASTIN: pending-snapshot of the emb target
    reg                   p_is_hfc;      // FASTIN: pending-snapshot of the hfc target
    reg [COL_W-1:0]       p_n_cols;
    reg [COL_W-1:0]       p_col_start;
    reg [COL_W-1:0]       p_col_step;
    reg [15:0]            p_cpc;
    reg [URAM_ADDR_W-1:0] p_stage_base;

    // helper: words to fetch for a given (is_scale,n_cols,cpc) snapshot.
    function automatic [31:0] f_total_words(input logic is_sc,
                                            input [COL_W-1:0] ncols,
                                            input [15:0] cpc_in);
        f_total_words = is_sc ? ((32'(ncols) * 32'(cpc_in)) >> 1)
                              :  (32'(ncols) * 32'(cpc_in));
    endfunction

    // number of beats for the next burst (clamp last burst to remaining words)
    wire [8:0] this_burst_beats =
        (words_remaining >= BURST_LEN[31:0]) ? BURST_LEN[8:0] : words_remaining[8:0];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            boot_busy          <= 1'b0;
            boot_done          <= 1'b0;
            boot_timeout       <= 1'b0;   // M9 FIX-B sticky watchdog flag
            rwait_cnt          <= '0;
            dma_we             <= 1'b0;
            dma_lane_sel       <= 16'b0;
            dma_wr_addr        <= '0;
            dma_wr_data        <= '0;
            sc_we              <= 1'b0;
            sc_lane_sel        <= 16'b0;
            sc_wr_addr         <= '0;
            sc_wr_data         <= '0;
            sc2_lane           <= 16'b0;
            sc2_addr           <= '0;
            sc2_data           <= '0;
            burst_done_after_flush <= 1'b0;
            pending_start      <= 1'b0;
            p_hbm_base         <= '0;
            p_is_scale         <= 1'b0;
            p_is_emb           <= 1'b0;   // FASTIN
            p_is_hfc           <= 1'b0;   // FASTIN
            p_n_cols           <= '0;
            p_col_start        <= '0;
            p_col_step         <= '0;
            p_cpc              <= 16'b0;
            p_stage_base       <= '0;
            is_scale_r         <= 1'b0;
            is_emb_r           <= 1'b0;   // FASTIN
            is_hfc_r           <= 1'b0;   // FASTIN
            m_axi_boot_arid    <= '0;
            m_axi_boot_araddr  <= '0;
            m_axi_boot_arlen   <= 8'b0;
            m_axi_boot_arsize  <= ARSIZE_32;
            m_axi_boot_arburst <= BURST_INCR;
            m_axi_boot_arvalid <= 1'b0;
            m_axi_boot_rready  <= 1'b0;
            hbm_base_r         <= '0;
            n_cols_r           <= '0;
            col_start_r        <= '0;
            col_step_r         <= '0;
            cpc_r              <= 16'b0;
            stage_base_r       <= '0;
            total_words_r      <= 32'b0;
            col_idx            <= '0;
            chunk_idx          <= 16'b0;
            words_done         <= 32'b0;
            beats_left         <= 9'b0;
            next_araddr        <= '0;
            words_remaining    <= 32'b0;
            cur_burst_bytes    <= '0;
        end else begin
            // default single-cycle strobes
            dma_we <= 1'b0;
            sc_we  <= 1'b0;

            // -----------------------------------------------------------------
            // M9 FIX-B: R-wait watchdog counter. Increment on a stalled read
            // cycle, reset to 0 on any forward progress. Parked at 0 in
            // S_IDLE/S_DONE (boot_stalled is 0 there). This runs BEFORE the FSM
            // case so the case-arm abort below sees this cycle's count.
            if (boot_stalled)
                rwait_cnt <= rwait_cnt + 1'b1;
            else
                rwait_cnt <= '0;

            // -----------------------------------------------------------------
            // Capture a start pulse that arrives MID-FLIGHT (any state other than
            // S_IDLE/S_DONE, which start a load directly). Snapshot the geometry
            // visible at this pulse so the latched load uses the right (fresh)
            // regs even if they change before it runs. Honoured on completion.
            if (boot_dma_start && (state != S_IDLE) && (state != S_DONE)) begin
                pending_start <= 1'b1;
                p_hbm_base    <= boot_hbm_base;
                p_is_scale    <= boot_is_scale;
                // FASTIN: snapshot the activation targets with the same mutual-
                // exclusion the direct-start sites use (scale wins; emb over hfc).
                p_is_emb      <= boot_is_emb & ~boot_is_scale;
                p_is_hfc      <= boot_is_hfc & ~boot_is_scale & ~boot_is_emb;
                p_n_cols      <= boot_n_cols;
                p_col_start   <= boot_col_start;
                p_col_step    <= boot_col_step;
                p_cpc         <= boot_cpc;
                p_stage_base  <= boot_stage_base;
            end

            // -----------------------------------------------------------------
            // M9 FIX-B: watchdog ABORT. If the read path has been stalled for
            // BOOT_RWAIT_MAX consecutive cycles while a load is in flight
            // (S_ARREQ/S_RDATA/S_SCFLUSH), the m_axi_boot read is wedged (AR
            // accepted with no R, or AR never accepted). Abort cleanly: drop
            // arvalid/rready, set the sticky boot_timeout, and return to S_IDLE
            // (NOT S_DONE) so a fresh boot_dma_start re-arms a retry. This takes
            // priority over the normal case-arm for the read states. boot_done is
            // left low (this load did NOT complete) — the host distinguishes a
            // wedge (0x74 = busy=0,done=0,timeout=1) from success (done=1).
            if (wd_expired && (state == S_ARREQ || state == S_RDATA || state == S_SCFLUSH)) begin
                m_axi_boot_arvalid <= 1'b0;
                m_axi_boot_rready  <= 1'b0;
                boot_busy          <= 1'b0;
                boot_done          <= 1'b0;
                boot_timeout       <= 1'b1;   // sticky until next clean completion
                pending_start      <= 1'b0;   // drop any latched follow-on; host re-arms
                rwait_cnt          <= '0;
                state              <= S_IDLE;
            end else
            case (state)
                // --------------------------------------------------------
                S_IDLE: begin
                    boot_busy <= 1'b0;
                    if (boot_dma_start) begin
                        // a fresh start clears the sticky timeout — the host's
                        // retry is now in flight, so 0x74[2] reads "no longer
                        // wedged" once this load is launched.
                        boot_timeout    <= 1'b0;
                        // snapshot geometry
                        hbm_base_r      <= boot_hbm_base;
                        is_scale_r      <= boot_is_scale;
                        // FASTIN: latch activation targets (scale wins; emb over hfc)
                        is_emb_r        <= boot_is_emb & ~boot_is_scale;
                        is_hfc_r        <= boot_is_hfc & ~boot_is_scale & ~boot_is_emb;
                        n_cols_r        <= boot_n_cols;
                        col_start_r     <= boot_col_start;
                        col_step_r      <= boot_col_step;
                        cpc_r           <= boot_cpc;
                        stage_base_r    <= boot_stage_base;
                        // BEATS to fetch from HBM. Weight: 1 INT32 word/beat →
                        // n_cols*cpc beats. Scale: 2 uint16 words/beat →
                        // (n_cols*gpc)/2 beats (gpc even ⇒ exact).
                        total_words_r   <= boot_is_scale ? (32'(boot_n_cols) * 32'(boot_cpc)) >> 1
                                                         : (32'(boot_n_cols) * 32'(boot_cpc));
                        words_remaining <= boot_is_scale ? (32'(boot_n_cols) * 32'(boot_cpc)) >> 1
                                                         : (32'(boot_n_cols) * 32'(boot_cpc));
                        next_araddr     <= boot_hbm_base;
                        col_idx         <= '0;
                        chunk_idx       <= 16'b0;
                        words_done      <= 32'b0;
                        boot_busy       <= 1'b1;
                        boot_done       <= 1'b0;
                        state           <= S_ARREQ;
                    end
                end

                // --------------------------------------------------------
                // Issue one INCR burst. Single outstanding burst at a time.
                S_ARREQ: begin
                    if (words_remaining == 32'b0) begin
                        // all words fetched.
                        m_axi_boot_arvalid <= 1'b0;
                        if (pending_start) begin
                            // a start pulse arrived mid-flight (patched-host real
                            // start, or a back-to-back load) — run it NOW from the
                            // latched (fresh) geometry instead of parking. boot_done
                            // is NOT raised between the two loads, so the host's
                            // busy->done edge poll sees ONE clean completion for the
                            // load that actually finishes last.
                            //
                            // PENDING-PULSE-FIX (2026-06-06): a boot_dma_start pulse
                            // arriving in THIS EXACT cycle (consume of the previous
                            // pending load) must NOT be lost. The mid-flight capture
                            // block above (S_ARREQ != S_IDLE/S_DONE) already ran this
                            // cycle and re-latched p_* with the NEW pulse's geometry;
                            // if we unconditionally cleared pending_start here, that
                            // newly-captured load would be silently dropped while its
                            // p_* geometry survived (orphaned). Instead re-latch
                            // pending_start to honour a same-cycle pulse — its p_*
                            // snapshot (written above) then runs on the next consume.
                            // When no same-cycle pulse exists this is exactly the old
                            // unconditional clear, so all other paths are unchanged.
                            pending_start   <= boot_dma_start &&
                                               (state != S_IDLE) && (state != S_DONE);
                            hbm_base_r      <= p_hbm_base;
                            is_scale_r      <= p_is_scale;
                            // FASTIN: pending snapshots already mutual-exclusion-gated
                            is_emb_r        <= p_is_emb;
                            is_hfc_r        <= p_is_hfc;
                            n_cols_r        <= p_n_cols;
                            col_start_r     <= p_col_start;
                            col_step_r      <= p_col_step;
                            cpc_r           <= p_cpc;
                            stage_base_r    <= p_stage_base;
                            total_words_r   <= f_total_words(p_is_scale, p_n_cols, p_cpc);
                            words_remaining <= f_total_words(p_is_scale, p_n_cols, p_cpc);
                            next_araddr     <= p_hbm_base;
                            col_idx         <= '0;
                            chunk_idx       <= 16'b0;
                            words_done      <= 32'b0;
                            boot_busy       <= 1'b1;
                            boot_done       <= 1'b0;
                            state           <= S_ARREQ;
                        end else begin
                            boot_busy          <= 1'b0;
                            boot_done          <= 1'b1;
                            state              <= S_DONE;
                        end
                    end else if (!m_axi_boot_arvalid) begin
                        m_axi_boot_arid    <= '0;
                        m_axi_boot_araddr  <= next_araddr;
                        m_axi_boot_arlen   <= this_burst_beats[7:0] - 8'd1; // arlen = beats-1
                        m_axi_boot_arsize  <= ARSIZE_32;
                        m_axi_boot_arburst <= BURST_INCR;
                        m_axi_boot_arvalid <= 1'b1;
                        beats_left         <= this_burst_beats;
                        // bytes this burst will consume = beats * (DATA_W/8)
                        cur_burst_bytes    <= {{(AXI_ADDR_W-9){1'b0}}, this_burst_beats}
                                              * (AXI_DATA_W/8);
                    end else if (m_axi_boot_arready) begin
                        // AR handshake complete
                        m_axi_boot_arvalid <= 1'b0;
                        m_axi_boot_rready  <= 1'b1;
                        state              <= S_RDATA;
                    end
                end

                // --------------------------------------------------------
                // Drain the R beats; write each to the derived target slot.
                //   WEIGHT mode: 1 INT32/beat → URAM, advance chunk by 1.
                //   SCALE  mode: 2 uint16/beat → scale_bram. Write the EVEN
                //     word here, hand the ODD word to S_SCFLUSH (next cycle)
                //     because the scale_bram has ONE write port.
                S_RDATA: begin
                    m_axi_boot_rready <= 1'b1;
                    if (m_axi_boot_rvalid && m_axi_boot_rready) begin
                        // common per-beat bookkeeping
                        words_done      <= words_done + 32'd1;       // BEAT counter
                        words_remaining <= words_remaining - 32'd1;
                        beats_left      <= beats_left - 9'd1;

                        if (!is_scale_r) begin
                            // ---- WEIGHT: 1 INT32 word/beat → URAM ----
                            dma_we       <= 1'b1;
                            dma_lane_sel <= cur_lane;
                            dma_wr_addr  <= cur_addr;
                            dma_wr_data  <= m_axi_boot_rdata;

                            // advance (col, chunk) by 1
                            if (chunk_idx == cpc_r - 16'd1) begin
                                chunk_idx <= 16'b0;
                                col_idx   <= col_idx + 1'b1;
                            end else begin
                                chunk_idx <= chunk_idx + 16'd1;
                            end

                            if (m_axi_boot_rlast || beats_left == 9'd1) begin
                                // burst finished — advance HBM addr, request next
                                m_axi_boot_rready <= 1'b0;
                                next_araddr       <= next_araddr + cur_burst_bytes;
                                state             <= S_ARREQ;
                            end
                        end else begin
                            // ---- SCALE: write EVEN word, defer ODD to S_SCFLUSH.
                            // numpy `<u2`.tobytes() ⇒ EVEN word = low 16 bits,
                            // ODD word = high 16 bits of the 32-bit beat. They
                            // land at consecutive inner (group) slots cur_addr,
                            // cur_addr+1 of the SAME column (gpc is even for every
                            // stage so a beat never straddles a column boundary).
                            sc_we       <= 1'b1;
                            sc_lane_sel <= cur_lane;
                            sc_wr_addr  <= cur_addr;
                            sc_wr_data  <= m_axi_boot_rdata[SCALE_W-1:0];        // low/even word
                            // hand the odd (high) word to the flush cycle
                            sc2_lane    <= cur_lane;
                            sc2_addr    <= cur_addr2;
                            sc2_data    <= m_axi_boot_rdata[2*SCALE_W-1:SCALE_W]; // high/odd word

                            // advance (col, group) by 2 (two words consumed)
                            if (chunk_idx >= cpc_r - 16'd2) begin
                                chunk_idx <= 16'b0;
                                col_idx   <= col_idx + 1'b1;
                            end else begin
                                chunk_idx <= chunk_idx + 16'd2;
                            end

                            // remember whether this beat ended the burst, then
                            // go flush the odd word with rready LOW.
                            burst_done_after_flush <= (m_axi_boot_rlast || beats_left == 9'd1);
                            m_axi_boot_rready      <= 1'b0;
                            state                  <= S_SCFLUSH;
                        end
                    end
                end

                // --------------------------------------------------------
                // SCALE mode only: write the held ODD (high) word, then resume.
                // rready stays LOW this whole cycle (single write port; one
                // sc_we/cycle). The address walk already advanced in S_RDATA.
                S_SCFLUSH: begin
                    sc_we       <= 1'b1;
                    sc_lane_sel <= sc2_lane;
                    sc_wr_addr  <= sc2_addr;
                    sc_wr_data  <= sc2_data;
                    if (burst_done_after_flush) begin
                        // burst finished — advance HBM addr, request next burst
                        m_axi_boot_rready <= 1'b0;
                        next_araddr       <= next_araddr + cur_burst_bytes;
                        state             <= S_ARREQ;
                    end else begin
                        // more beats in this burst — re-open R intake
                        m_axi_boot_rready <= 1'b1;
                        state             <= S_RDATA;
                    end
                end

                // --------------------------------------------------------
                // Parked. ARVALID stays low → port is silent during compute.
                S_DONE: begin
                    m_axi_boot_arvalid <= 1'b0;
                    m_axi_boot_rready  <= 1'b0;
                    // A start pulse in S_DONE now STARTS the new load directly
                    // (snapshot + clear boot_done), mirroring S_IDLE — it does NOT
                    // merely re-arm S_DONE->S_IDLE (the old trap that skipped loads
                    // #2..#8 when the host issued one pulse per load). boot_done is
                    // dropped this cycle so the host's busy->done edge poll detects a
                    // FRESH completion of THIS load (not the sticky prior level).
                    if (boot_dma_start) begin
                        boot_timeout    <= 1'b0;   // M9 FIX-B: retry clears sticky wedge flag
                        hbm_base_r      <= boot_hbm_base;
                        is_scale_r      <= boot_is_scale;
                        // FASTIN: latch activation targets (scale wins; emb over hfc)
                        is_emb_r        <= boot_is_emb & ~boot_is_scale;
                        is_hfc_r        <= boot_is_hfc & ~boot_is_scale & ~boot_is_emb;
                        n_cols_r        <= boot_n_cols;
                        col_start_r     <= boot_col_start;
                        col_step_r      <= boot_col_step;
                        cpc_r           <= boot_cpc;
                        stage_base_r    <= boot_stage_base;
                        total_words_r   <= f_total_words(boot_is_scale, boot_n_cols, boot_cpc);
                        words_remaining <= f_total_words(boot_is_scale, boot_n_cols, boot_cpc);
                        next_araddr     <= boot_hbm_base;
                        col_idx         <= '0;
                        chunk_idx       <= 16'b0;
                        words_done      <= 32'b0;
                        boot_busy       <= 1'b1;
                        boot_done       <= 1'b0;
                        state           <= S_ARREQ;
                    end else begin
                        boot_busy <= 1'b0;
                        boot_done <= 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
