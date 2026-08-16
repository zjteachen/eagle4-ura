// =====================================================================
// emb_gather.sv — J5E (2026-07-20): DEPTH-N child-token EMBEDDING GATHER
//   One token id -> one H-element row read from the TARGET model's
//   embed_tokens table in HBM -> written into the pipeline's emb_buf.
//   Work order: reports/DEPTHN_EMB_GATHER_WORKORDER.md §2.
//
// WHY THIS EXISTS
//   Depth-0 gets its embedding from the host (the 0x7c boot trio, one
//   6 KB transfer per forward).  At depth>=1 the tree produces child
//   token ids ON CHIP, so a host round-trip per child would destroy the
//   whole point of on-chip tree decode.  This block is the fetch
//   primitive the depth-N FSM rungs will call.
//
// ***** THE SECOND-TABLE CORRECTION (work order §1 row 6, MEASURED) *****
//   The DRAFTER's lm_head is NOT the TARGET's embed_tokens.  Both are
//   [128256, 3072] (which is what fooled the first draft of the work
//   order) but they are DIFFERENT TENSORS — max abs diff 0.103 on rows
//   0 and 622.  The drafter checkpoint has NO embed_tokens key at all;
//   the target's embed_tokens is tied to the target's output head.
//   => this block MUST read from `hbm_emb_base` (a SECOND HBM region,
//      the target embed_tokens image), NOT from `lmfull_hbm_base` (the
//      drafter lm-head image G8 gathers).  The two bases are separate
//      ports on this module's parent for exactly that reason.  Silently
//      reusing the G8 base would return plausible-but-wrong vectors —
//      the worst possible failure mode (no crash, degraded accept).
//
// ***** FORMAT DECISION (EMB_FMT), stated and TESTED, not assumed *****
//   emb_buf is `signed [15:0]` Q4.12 (eagle4_drafter_ps.sv:523 + the G1
//   e-arm read at :655).  The TARGET's embed_tokens ships as bf16.
//   Two staging choices, both supported by the EMB_FMT strap:
//     EMB_FMT=1 (DEFAULT, "native"): HBM holds the bf16 table VERBATIM
//       as it appears in the safetensors; this block converts bf16 ->
//       Q4.12 in the datapath (bf16_to_q412 below, 16 converters, one
//       per beat lane).  No 751.5 MB host-side repack pass.
//     EMB_FMT=0 ("pre-baked"): HBM holds an already-Q4.12 int16 image;
//       beats are copied verbatim.  Cheaper in fabric, costs a host-side
//       conversion pass at session init.
//   Both arms are gated (J5E_EMB_GATHER_IMPL.md §3).  Rounding contract
//   for EMB_FMT=1: round-half-AWAY-from-zero, saturate to [-32768,32767].
//
// ***** MULTI-CHANNEL HBM (NCH), 2026-07-24 — READ THE COST/BENEFIT *****
//   Standing user directive: "when using the HBM, be mindful to always
//   use multiple channels."  NCH implements that for this block:
//     * NCH=1 (DEFAULT): the g_nch1 arm below is the VERBATIM, unmodified
//       shipping text.  Byte-identical, zero delta.
//     * NCH>1: the row's H elements are split into NCH CONTIGUOUS slices
//       of H/NCH elements.  Channel c fetches elements
//       [c*H/NCH, (c+1)*H/NCH) of the SAME flat image from its OWN AXI
//       master port (own single ID, own NoC SI, own HBM port) and writes
//       chunk c*BEATS_PER_CH + <its own consume counter>.
//       ⇒ NO image duplication, NO host-side interleave, NO ABI change:
//         every channel reads the identical unmodified embed_tokens
//         image at a different offset.  This is the cheapest striping
//         form that exists (see reports/results/EMB_GATHER_MULTICHANNEL.md).
//   *** HONEST LIMIT — the sink, not the channel, is the wall. ***
//       The landing buffer (input_emb_buf, eagle4_drafter.sv g_emb_arb)
//       accepts EXACTLY ONE 256b chunk per cycle.  One 256b AXI channel
//       already delivers one beat per cycle (G8 silicon: 97.148 cyc for a
//       96-beat/channel row, 1.2% off the beat floor).  So at NCH>1 the
//       channels are ARBITRATED onto that one write port and the row
//       still takes BEATS_ROW writes: the throughput gain is ~0 until the
//       sink widens.  NCH>1 buys only AR/first-word-latency overlap today.
//       Measured numbers + the exact condition that makes NCH>1 pay
//       (multi-sink per-child gather / K_F-parallel banks) are in
//       reports/results/EMB_GATHER_MULTICHANNEL.md.  DEFAULT STAYS 1.
//
// REUSE (work order: "reuse the proven AR/burst/beat machinery")
//   The AR sub-burst geometry — MAX_BURST cap, the AXI4 4 KB page split,
//   full-width arsize, single AXI ID / in-order beats, OUTST throttle —
//   is a line-faithful lift of g8_stripe_frontend.sv:227-284 (itself the
//   shipping G8WIDE-NOC F1 geometry, g8_rescore_p.sv:1586-1604, silicon
//   proven on the S07 path).  What is STRIPPED: the owner FIFO, the
//   candidate table port, the dot pipeline.  The N-channel round robin is
//   now KEPT (in the g_nchn arm) but at SLICE granularity, not row
//   granularity: G8 stripes 512 rows across channels; this block fetches
//   ONE row, so only an intra-row split can use a second channel at all.
//   A row here is 6144 B and does NOT divide 4096, so the page split is
//   LOAD-BEARING, not decorative (a 6 KB row starting mid-page splits).
//
// PAST-LESSON COMPLIANCE
//   * lane-true consume: full-width arsize only
//     (project_g8resc_rootcause_axi_narrow_lane — arsize=2B gave an
//      even-lane-only dot on silicon).
//   * arlen is 8 bits (feedback_axi_arlen_8bit_truncation_h3072).
//   * ONE always_ff owns every register — no multi-driver
//     (feedback_pulse_reg_multidriver_reset_plus_proc).  At NCH>1 the
//     SHARED write port + all per-channel state live in ONE always_ff in
//     g_nchn (a per-channel generate with its own always_ff would give
//     o_we/o_chunk/o_data NCH drivers — the exact multi-driver trap).
//   * AR fan-in (feedback_axi_driver_fanin_rule): each channel drives its
//     OWN m_arvalid bit on its OWN NoC SI port ⇒ ARVALID fan-in per port
//     stays 1 at every NCH.  Channels are NEVER muxed onto one port.
//   * `done` is a sticky level masked with the LIVE start pulse so a
//     caller entering its wait state on the same cycle as start can
//     never sample the PREVIOUS run's done
//     (feedback_sticky_done_rearm_trap) — same idiom as
//     g8_stripe_frontend's gather_done.
//   * beats land by ISSUE ORDER off a single AXI ID PER CHANNEL; the write
//     chunk index is (channel base + that channel's consume counter),
//     never an arrival-derived address (the pdi21 arrival-order lesson).
//     Cross-channel arrival order is therefore irrelevant by construction.
//
// RED TEETH:
//   `EMB_RED_NOSCALE`  drops the *2 byte scaling from the row address
//     (tok_id*H instead of tok_id*H*2).  The load gate MUST fail for any
//     tok_id != 0.
//   `EMB_NCH_RED_MAP`  (NCH>1 only) drops the per-channel element base
//     from the AR address while KEEPING it in the write chunk index —
//     i.e. every channel re-fetches slice 0 and files it under a
//     different chunk range.  The reassembled row must MISMATCH.  This is
//     the canonical stripe-mapping defect (the H1 silent-ABI class).
//
// OUT OF SCOPE (named, not half-built — work order §2):
//   which child ids to fetch and when (tree/scheduler), KV re-parenting,
//   ctx_len_eff muxing.  Those are the depth-N FSM rungs.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

// (* use_dsp = "no" *) — the ONLY multiply in this block is the row-address
//   constant `tok_id * ROW_BYTES`.  At the shipping geometry ROW_BYTES = 6144
//   = 0x1800 = 2^12 + 2^11, i.e. TWO set bits: fabric does it in one adder.
//   Left to itself Vivado maps it to a DSP58 (measured: "emb_gather |
//   A2*(B:0x1800) | A 18 | B 13" in the first OOC pass) — a whole DSP tile
//   spent on two shifts.  Address arithmetic does not belong in a DSP.
(* use_dsp = "no" *)
module emb_gather #(
    parameter int H          = 3072,   // embedding dim
    parameter int IDX_W      = 18,     // EAGLE4 token-id width (128256 < 2^18)
    parameter int AXI_ADDR_W = 64,
    parameter int AXI_DATA_W = 256,
    parameter int MAX_BURST  = 256,    // AXI4 8-bit arlen cap
    parameter int OUTST      = 4,      // outstanding sub-bursts (per channel)
    parameter bit EMB_FMT    = 1'b1,   // 1 = bf16 source (convert), 0 = raw Q4.12
    parameter int NCH        = 1       // HBM read channels (1 = shipping default)
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // ---- request ----
    input  wire                      start,      // 1-cycle pulse
    input  wire [IDX_W-1:0]          tok_id,     // sampled at start
    input  wire [AXI_ADDR_W-1:0]     hbm_emb_base,  // TARGET embed_tokens base

    // ---- AXI4 read masters (ONE PER CHANNEL: own ID / own NoC SI / own
    //      HBM port).  Packed [NCH*W-1:0]; at NCH=1 every vector is exactly
    //      the legacy scalar width, so the shipping instantiation binds
    //      unchanged and the netlist is byte-identical. ----
    output logic [NCH-1:0]             m_arvalid,
    input  wire  [NCH-1:0]             m_arready,
    output logic [NCH*AXI_ADDR_W-1:0]  m_araddr,
    output logic [NCH*8-1:0]           m_arlen,
    output logic [NCH*3-1:0]           m_arsize,
    output logic [NCH*2-1:0]           m_arburst,
    input  wire  [NCH-1:0]             m_rvalid,
    output logic [NCH-1:0]             m_rready,
    input  wire  [NCH*AXI_DATA_W-1:0]  m_rdata,
    input  wire  [NCH-1:0]             m_rlast,

    // ---- emb_buf write port (CHUNK-WIDE: one beat = one chunk) ----
    //   The consumer array is 16-wide-read (G1 e-arm), so a chunk-wide
    //   write is the natural, stall-free landing: 1 beat = FPB elements
    //   = 1 write.  An element-wide port would need FPB cycles/beat and
    //   throttle rready 16:1 for no benefit.
    //   *** THIS PORT IS THE NCH>1 BOTTLENECK — see the header. ***
    output logic                     o_we,
    output logic [$clog2((H+AXI_DATA_W/16-1)/(AXI_DATA_W/16))-1:0] o_chunk, // 0..H/FPB-1
    output logic [AXI_DATA_W-1:0]    o_data,     // Q4.12 signed int16 x FPB

    // ---- completion ----
    output wire                      done,       // sticky, cleared on start
    output logic [31:0]              beats_dbg   // beats consumed this run
);
    localparam int FPB       = AXI_DATA_W/16;             // 16 elems/beat
    localparam int BEATS_ROW = (H + FPB - 1)/FPB;         // 192
    localparam int CHW       = $clog2(BEATS_ROW);         // 8
    localparam int EW        = $clog2(H)+1;               // element cursor width
    localparam int ROW_BYTES = H*2;                       // 6144
    // ---- NCH slice geometry (contiguous, equal, image-layout-invariant) ----
    localparam int ELEM_PER_CH  = H / NCH;                // 3072 @NCH=1, 1536 @2
    localparam int BEATS_PER_CH = ELEM_PER_CH / FPB;      // 192 @NCH=1, 96 @2

    // -----------------------------------------------------------------
    // bf16 -> Q4.12 (round half away from zero, saturating)
    //   bf16 = {s, e[7:0], m[6:0]};  v = (-1)^s * 2^(e-127) * 1.m
    //   q = v * 4096 = sig * 2^(e-127-7+12) = sig * 2^(e-122),
    //   with sig = {1'b1, m} (8 bits, range 128..255).
    //   sh = e-122 :  sh>=8  -> |v|>=8   -> saturate
    //                 sh<=-9 -> |q|<0.5  -> 0
    //   sh in [0,7] : mag = sig << sh   (max 255<<7 = 32640, no overflow)
    //   sh in [-8,-1]: mag = (sig + 2^(rs-1)) >> rs,  rs = -sh
    //   Checked: v=1.0 (e=127,m=0) -> sig=128, sh=5 -> 4096 = 1.0 Q4.12.
    // -----------------------------------------------------------------
    function automatic logic signed [15:0] bf16_to_q412(input logic [15:0] b);
        logic        sgn;
        logic [7:0]  e;
        logic [8:0]  sig;
        int          sh, rs;
        logic [31:0] mag;
        begin
            sgn = b[15];
            e   = b[14:7];
            sig = {1'b1, b[6:0]};
            if (e == 8'd0) begin
                bf16_to_q412 = 16'sd0;                  // zero / subnormal -> 0
            end else begin
                sh = int'(e) - 122;
                if (sh >= 8)       mag = 32'd65535;     // |v| >= 8 -> saturate
                else if (sh <= -9) mag = 32'd0;
                else if (sh >= 0)  mag = (32'(sig) << sh);
                else begin
                    rs  = -sh;                          // 1..8
                    mag = (32'(sig) + (32'd1 << (rs-1))) >> rs;
                end
                if (sgn) bf16_to_q412 = (mag >= 32'd32768) ? -16'sd32768
                                                           : -$signed({1'b0, mag[14:0]});
                else     bf16_to_q412 = (mag >= 32'd32767) ?  16'sd32767
                                                           :  $signed({1'b0, mag[14:0]});
            end
        end
    endfunction

    // -----------------------------------------------------------------
    // beat -> chunk payload (comb; the FORMAT DECISION lives here).
    //   ONE converter set TOTAL, never one per channel.  The write port
    //   consumes exactly one beat per cycle at every NCH, so exactly one
    //   beat needs converting per cycle: at NCH>1 the RDATA is muxed by
    //   the grant FIRST and converted once (g_nchn below).  Measured: the
    //   per-channel form cost +4,767 LUT at NCH=2 (32 bf16 converters);
    //   the shared form costs a 2:1 256b mux instead.
    // -----------------------------------------------------------------
    function automatic logic [FPB*16-1:0] beat_conv(input logic [AXI_DATA_W-1:0] raw);
        begin
            for (int k = 0; k < FPB; k++)
                beat_conv[k*16 +: 16] = EMB_FMT
                    ? bf16_to_q412(raw[k*16 +: 16])   // native bf16 table
                    : raw[k*16 +: 16];                // pre-baked Q4.12 table
        end
    endfunction

    typedef enum logic [0:0] { S_IDLE, S_RUN } state_e;

    logic done_r;
    // live-start mask: kills the sticky-done re-arm trap at the source
    assign done = done_r && !start;

    generate
    // =================================================================
    // g_nch1 — THE SHIPPING SINGLE-CHANNEL ARM.
    //   VERBATIM pre-2026-07-24 text.  The ONLY textual deltas are the
    //   [0] index on the now-vectored AXI ports and beat_q412[0] (the
    //   converter array).  Both are width-1 selects of width-1 vectors:
    //   elaborated logic, and therefore the netlist, is identical.
    // =================================================================
    if (NCH == 1) begin : g_nch1
        state_e state;
        logic [IDX_W-1:0]           tok_q;       // launch-sampled id
        logic [AXI_ADDR_W-1:0]      base_q;      // launch-sampled base
        logic [EW-1:0]              ai_elem;     // AR element cursor
        logic                       ai_done;     // all ARs issued
        logic [CHW:0]               rc_beat;     // consume/chunk counter
        logic [$clog2(OUTST+1):0]   outst_cnt;

        always_ff @(posedge clk) begin
            if (!rst_n) begin
                state<=S_IDLE; done_r<=1'b0;
                tok_q<='0; base_q<='0; ai_elem<='0; ai_done<=1'b0;
                rc_beat<='0; outst_cnt<='0; beats_dbg<='0;
                m_arvalid[0]<=1'b0; m_araddr<='0; m_arlen<='0;
                m_arsize<=3'd0; m_arburst<=2'b01; m_rready[0]<=1'b0;
                o_we<=1'b0; o_chunk<='0; o_data<='0;
            end else begin
                o_we <= 1'b0;                                  // 1-cycle write pulse

                case (state)
                // -------------------------------------------------------
                S_IDLE: if (start) begin
                    tok_q   <= tok_id;
                    base_q  <= hbm_emb_base;
                    ai_elem <= '0; ai_done <= 1'b0;
                    rc_beat <= '0; outst_cnt <= '0; beats_dbg <= '0;
                    done_r  <= 1'b0;
                    state   <= S_RUN;
                end

                // -------------------------------------------------------
                S_RUN: begin
                    automatic bit ar_issued;
                    automatic bit r_retired;
                    ar_issued = 1'b0;
                    r_retired = 1'b0;

                    m_rready[0] <= 1'b1;

                    // ---------------- AR lane ----------------
                    if (!m_arvalid[0] && !ai_done && (int'(outst_cnt) < OUTST)) begin
                        // sub-burst geometry: verbatim g8_stripe_frontend.sv:236-259
                        automatic int unsigned rem_el = H - int'(ai_elem);
                        automatic int unsigned rem_bt = (rem_el + FPB - 1)/FPB;
                        automatic int unsigned blen_b = (rem_bt > MAX_BURST) ? MAX_BURST : rem_bt;
                        automatic int unsigned span_el;
                        // ROW BASE — the second-table address (see header).
                        // RED TEETH: drop the *2 byte scaling.
                        automatic logic [AXI_ADDR_W-1:0] rb = base_q +
`ifdef EMB_RED_NOSCALE
                            (AXI_ADDR_W'(tok_q) * (H));
`else
                            (AXI_ADDR_W'(tok_q) * (ROW_BYTES));
`endif
                        automatic logic [AXI_ADDR_W-1:0] ar_start =
                            rb + (AXI_ADDR_W'(ai_elem) * 2);
                        begin
                            // AXI4 4 KB page rule.  LOAD-BEARING here: a 6144 B
                            // row does not divide 4096, so rows alternately
                            // start mid-page and MUST split.
                            automatic int unsigned BPB         = AXI_DATA_W/8;
                            automatic int unsigned bytes_to_pg = 4096 - int'(ar_start[11:0]);
                            automatic int unsigned pg_beats    = bytes_to_pg / BPB;
                            if (blen_b > pg_beats) blen_b = pg_beats;
                            if (blen_b == 0)       blen_b = 1;
                        end
                        span_el = blen_b * FPB;
                        m_araddr  <= ar_start;
                        m_arlen   <= 8'(blen_b - 1);
                        // full-width consume ONLY (narrow-lane silicon lesson)
                        m_arsize  <= 3'(AXI_DATA_W==512 ? 6 : AXI_DATA_W==256 ? 5 : 4);
                        m_arburst <= 2'b01;
                        m_arvalid[0] <= 1'b1;
                        ar_issued  = 1'b1;
                        if ((int'(ai_elem) + int'(span_el)) >= H) ai_done <= 1'b1;
                        else ai_elem <= ai_elem + EW'(span_el);
                    end else if (m_arvalid[0] && m_arready[0]) begin
                        m_arvalid[0] <= 1'b0;
                    end

                    // ---------------- R / write lane ----------------
                    //   Single AXI ID + in-order slave => beat n is chunk n.
                    if (m_rvalid[0] && m_rready[0]) begin
                        o_we      <= 1'b1;
                        o_chunk   <= rc_beat[CHW-1:0];
                        o_data    <= beat_conv(m_rdata);
                        rc_beat   <= rc_beat + 1'b1;
                        beats_dbg <= beats_dbg + 32'd1;
                        if (m_rlast[0]) r_retired = 1'b1;
                        if (int'(rc_beat) + 1 == BEATS_ROW) begin
                            m_rready[0] <= 1'b0;
                            done_r   <= 1'b1;      // sticky until next start
                            state    <= S_IDLE;    // multi-invocation re-arm
                        end
                    end

                    // single write: +issue -retire (both may fire same cycle)
                    outst_cnt <= outst_cnt + (ar_issued ? 1'b1 : 1'b0)
                                           - (r_retired ? 1'b1 : 1'b0);
                end
                endcase
            end
        end
    end // g_nch1
    // =================================================================
    // g_nchn — N-WAY INTRA-ROW STRIPE (NCH >= 2).
    //   Channel c owns elements [c*ELEM_PER_CH, (c+1)*ELEM_PER_CH) of the
    //   SAME flat image and writes chunks [c*BEATS_PER_CH, ...+BEATS_PER_CH).
    //   Per-channel: own AR engine, own ARVALID on its own port (fan-in 1),
    //   own single AXI ID, own OUTST credit, own consume counter.
    //   Shared: the ONE emb_buf write port, granted round-robin.
    // =================================================================
    else begin : g_nchn
        localparam int PW = (NCH > 1) ? $clog2(NCH) : 1;
        state_e state;
        logic [IDX_W-1:0]         tok_q;
        logic [AXI_ADDR_W-1:0]    base_q;
        logic [EW-1:0]            ai_elem   [NCH];   // cursor WITHIN the slice
        logic [NCH-1:0]           ai_done;
        logic [CHW:0]             rc_beat   [NCH];   // beats consumed on ch c
        logic [NCH-1:0]           ch_fin;            // slice complete
        logic [$clog2(OUTST+1):0] outst_cnt [NCH];
        logic [PW-1:0]            rr_ptr;            // round-robin start

        // ---- combinational round-robin grant onto the single write port ----
        //   RREADY may depend on RVALID (AXI4 A3.3.1: only the VALID side is
        //   forbidden from waiting on READY), so the grant is combinational —
        //   a registered grant would idle the port every other beat.
        logic [NCH-1:0] gnt;
        always_comb begin
            gnt = '0;
            if (state == S_RUN) begin
                for (int i = 0; i < NCH; i++) begin
                    automatic int c = (int'(rr_ptr) + i) % NCH;
                    if ((gnt == '0) && m_rvalid[c] && !ch_fin[c]) gnt[c] = 1'b1;
                end
            end
        end
        assign m_rready = gnt;

        // ---- grant mux FIRST, convert ONCE (see beat_conv header) ----
        //   Only the granted channel's beat is written this cycle, so only
        //   it needs converting.  256b OR-mux (gnt is one-hot by
        //   construction) + ONE bf16->Q4.12 converter set for all NCH.
        logic [AXI_DATA_W-1:0] rdata_sel;
        always_comb begin
            rdata_sel = '0;
            for (int c = 0; c < NCH; c++)
                if (gnt[c]) rdata_sel = m_rdata[c*AXI_DATA_W +: AXI_DATA_W];
        end

        always_ff @(posedge clk) begin
            if (!rst_n) begin
                state<=S_IDLE; done_r<=1'b0;
                tok_q<='0; base_q<='0; ai_done<='0; ch_fin<='0;
                rr_ptr<='0; beats_dbg<='0;
                m_arvalid<='0; m_araddr<='0; m_arlen<='0;
                m_arsize<='0; m_arburst<='0;
                o_we<=1'b0; o_chunk<='0; o_data<='0;
                for (int c = 0; c < NCH; c++) begin
                    ai_elem[c]<='0; rc_beat[c]<='0; outst_cnt[c]<='0;
                    m_arburst[c*2 +: 2] <= 2'b01;
                end
            end else begin
                o_we <= 1'b0;                                  // 1-cycle write pulse

                case (state)
                // -------------------------------------------------------
                S_IDLE: if (start) begin
                    tok_q  <= tok_id;
                    base_q <= hbm_emb_base;
                    ai_done<='0; ch_fin<='0; rr_ptr<='0;
                    beats_dbg<='0; done_r<=1'b0;
                    for (int c = 0; c < NCH; c++) begin
                        ai_elem[c]<='0; rc_beat[c]<='0; outst_cnt[c]<='0;
                    end
                    state  <= S_RUN;
                end

                // -------------------------------------------------------
                S_RUN: begin
                    automatic logic [NCH-1:0] fin_nxt = ch_fin;

                    for (int c = 0; c < NCH; c++) begin
                        automatic bit ar_issued = 1'b0;
                        automatic bit r_retired = 1'b0;

                        // ---------------- AR lane (per channel) ----------------
                        if (!m_arvalid[c] && !ai_done[c] && (int'(outst_cnt[c]) < OUTST)) begin
                            automatic int unsigned rem_el = ELEM_PER_CH - int'(ai_elem[c]);
                            automatic int unsigned rem_bt = (rem_el + FPB - 1)/FPB;
                            automatic int unsigned blen_b = (rem_bt > MAX_BURST) ? MAX_BURST : rem_bt;
                            automatic int unsigned span_el;
                            automatic logic [AXI_ADDR_W-1:0] rb = base_q +
`ifdef EMB_RED_NOSCALE
                                (AXI_ADDR_W'(tok_q) * (H));
`else
                                (AXI_ADDR_W'(tok_q) * (ROW_BYTES));
`endif
                            // RED TEETH (EMB_NCH_RED_MAP): drop the per-channel
                            // element base from the ADDRESS while the write chunk
                            // index below keeps it -> every channel re-fetches
                            // slice 0.  The reassembled row MUST mismatch.
                            automatic logic [AXI_ADDR_W-1:0] ar_start = rb +
`ifdef EMB_NCH_RED_MAP
                                (AXI_ADDR_W'(int'(ai_elem[c])) * 2);
`else
                                (AXI_ADDR_W'(c*ELEM_PER_CH + int'(ai_elem[c])) * 2);
`endif
                            begin
                                // AXI4 4 KB page rule (LOAD-BEARING: neither the
                                // 6144 B row nor the 3072 B slice divides 4096).
                                automatic int unsigned BPB         = AXI_DATA_W/8;
                                automatic int unsigned bytes_to_pg = 4096 - int'(ar_start[11:0]);
                                automatic int unsigned pg_beats    = bytes_to_pg / BPB;
                                if (blen_b > pg_beats) blen_b = pg_beats;
                                if (blen_b == 0)       blen_b = 1;
                            end
                            span_el = blen_b * FPB;
                            m_araddr[c*AXI_ADDR_W +: AXI_ADDR_W] <= ar_start;
                            m_arlen  [c*8 +: 8]  <= 8'(blen_b - 1);
                            // full-width consume ONLY (narrow-lane silicon lesson)
                            m_arsize [c*3 +: 3]  <= 3'(AXI_DATA_W==512 ? 6 : AXI_DATA_W==256 ? 5 : 4);
                            m_arburst[c*2 +: 2]  <= 2'b01;
                            m_arvalid[c]         <= 1'b1;
                            ar_issued = 1'b1;
                            if ((int'(ai_elem[c]) + int'(span_el)) >= ELEM_PER_CH) ai_done[c] <= 1'b1;
                            else ai_elem[c] <= ai_elem[c] + EW'(span_el);
                        end else if (m_arvalid[c] && m_arready[c]) begin
                            m_arvalid[c] <= 1'b0;
                        end

                        // ---------------- R / write lane (granted channel) ------
                        //   Single AXI ID per channel + in-order slave => the
                        //   n-th beat OF THIS CHANNEL is chunk c*BEATS_PER_CH+n.
                        //   Cross-channel arrival order is irrelevant.
                        if (gnt[c]) begin
                            o_we      <= 1'b1;
                            o_chunk   <= CHW'(c*BEATS_PER_CH + int'(rc_beat[c]));
                            o_data    <= beat_conv(rdata_sel);
                            rc_beat[c]<= rc_beat[c] + 1'b1;
                            beats_dbg <= beats_dbg + 32'd1;
                            rr_ptr    <= PW'((c + 1) % NCH);
                            if (m_rlast[c]) r_retired = 1'b1;
                            if (int'(rc_beat[c]) + 1 == BEATS_PER_CH) fin_nxt[c] = 1'b1;
                        end

                        outst_cnt[c] <= outst_cnt[c] + (ar_issued ? 1'b1 : 1'b0)
                                                     - (r_retired ? 1'b1 : 1'b0);
                    end

                    ch_fin <= fin_nxt;
                    if (&fin_nxt) begin
                        done_r <= 1'b1;        // sticky until next start
                        state  <= S_IDLE;      // multi-invocation re-arm
                    end
                end
                endcase
            end
        end
    end // g_nchn
    endgenerate

`ifndef SYNTHESIS
    initial begin
        if (H % FPB != 0)
            $fatal(1, "emb_gather: H(%0d) must be a multiple of FPB(%0d)", H, FPB);
        if (NCH < 1)
            $fatal(1, "emb_gather: NCH(%0d) must be >= 1", NCH);
        // equal contiguous slices, each an integer number of full beats:
        // the write-chunk map c*BEATS_PER_CH + n is only a bijection onto
        // [0,BEATS_ROW) when this holds.
        if (H % (NCH*FPB) != 0)
            $fatal(1, "emb_gather: H(%0d) must be a multiple of NCH*FPB(%0d)", H, NCH*FPB);
    end
`endif

endmodule

`default_nettype wire
