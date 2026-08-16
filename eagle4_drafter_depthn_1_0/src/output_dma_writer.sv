// =====================================================================
// output_dma_writer.sv
// OUTPUT-DMA: compute-result write master (URAM/heap → HBM C2H readback).
// =====================================================================
//
// WHY (SPEC_OUTPUT_DMA.md, M7 finding; SPEC_OUTDMA_V24_ABI.md, M8 finding):
//   The drafter's final product — the candidate-tree top-K_F=8 (score,index)
//   pairs from tree_decode — lived ONLY as internal/xsim wires. out0/1/2_base
//   were latched into control_regs but never reached an HBM writer, so on
//   silicon the host could not read the drafter's output. That blocks M7
//   (bit-exact proof) and M8 (SGLang). This module is the missing link: it
//   bursts the K_F (score,index) pairs to HBM on `tree_done`.
//
//   TWO selectable output ABIs (ABI_V24 param):
//     ABI_V24 = 0  (M7 SIMPLE layout, default, unchanged — bit-exact for the
//                   tb_output_dma_writer / M7 harness):
//       out0_base + 0          : K_F int32 INDICES (zero-ext)
//       out0_base + K_F*4 (=32): K_F int32 SCORES  (sign-ext)   [16-beat burst]
//       out1_base + 0          : K_F int32 INDICES mirror        [ 8-beat burst]
//
//     ABI_V24 = 1  (v24 SGLang host ABI — eagle4_v24_forward /
//                   libeagle4_host_v56_direct):
//       THREE contiguous arrays, each shaped [D · K_MAX · TOP_K] int32, in
//       HBM order  ids → scores → parents, addressed by the THREE separate
//       base pointers out0_base / out1_base / out2_base (= control_regs
//       0x28 / 0x30 / 0x38). For one (depth,frontier) node the writer fills
//       index window d·(K_MAX·TOP_K) + f·TOP_K + t with t∈[0..TOP_K):
//         t ∈ [0 .. K_F-1)      = LIVE   tree results (heap-rank 0 = largest)
//         t ∈ [K_F .. TOP_K-1)  = ZERO-PAD (ids=0, scores=0)
//       Parents window holds PARENT_SENTINEL (default -1, the SGLang
//       "no-parent / root" marker validated by cost_draft_tree_main.py's
//       `while n_parent >= 0` tree-walk terminator) in every slot.
//
//       PDI#4 pilot (D=1, num_frontiers=1, max_depth=1): a single node at
//       (d=0,f=0) → window words [0..TOP_K-1] of each of the three arrays;
//       the writer emits exactly TOP_K(16) beats per array burst, landing the
//       live K_F=8 at t∈[0..7] and zero-pad/sentinel at t∈[8..15], at the
//       base of each array (depth-0 frontier-0 = offset 0). The remaining
//       elems-TOP_K words of each [D·K_MAX·TOP_K]=128-word array stay at
//       whatever the host pre-zeroed them to (host memset/qdma_read covers the
//       full 128); for the pilot only [0..15] is meaningful and the SGLang
//       hook reads exactly [0..15].
//
// SINGLE AXI WRITE MASTER (the AXI-fan-in rule — feedback_axi_driver_fanin_rule):
//   A 2nd master-driver class (a 2nd AWVALID/ARVALID source) caused the
//   historical integration hangs. This is the ONLY AXI WRITE master in the
//   drafter (the M9 weight DMA is a READ master on a separate AR-only port).
//   AWVALID is asserted ONLY between tree_done and out_done — a short run of
//   sequential write bursts (one outstanding at a time, never concurrent),
//   then the port parks SILENT (AWVALID/WVALID low) forever. So the
//   >5-concurrent-driver NoC deadlock class does not apply.
//
// AXI4 WRITE (write master — AW/W/B channels only, no read channels):
//   Fixed INCR burst, AWSIZE=4 bytes/beat, single outstanding burst. WLAST on
//   the final beat; one BVALID per burst. Simplest deadlock-free shape — the
//   output is tiny (<=192 bytes for v24) so latency is irrelevant.
//
// Multi-driver-safe: every output reg is driven from exactly ONE always_ff.
// The K_F (score,index) snapshot is latched once at tree_done into a single
// flat staging register, so the heap (owned by tree_decode) is read-only here.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module output_dma_writer #(
    parameter int K_F            = 8,        // children per frontier (EAGLE4)
    parameter int IDX_W          = 18,       // tree token-id width (Llama-3.2-3B)
    parameter int SCORE_W        = 32,       // tree score width
    parameter int AXI_ADDR_W     = 64,
    parameter int AXI_DATA_W     = 32,       // one INT32 per beat
    parameter int AXI_ID_W       = 1,
    parameter bit EN_OUT1_MIRROR = 1'b1,     // (M7 mode) also mirror indices to out1_base
    // ---- v24 SGLang host ABI mode ----
    parameter bit ABI_V24        = 1'b0,     // 0 = M7 simple, 1 = v24 ids/scores/parents
    parameter int TOP_K          = 16,       // v24 per-(depth,frontier) slot count (8 live + 8 pad)
    parameter int PARENT_SENTINEL = -1,      // v24 root/no-parent marker (SGLang: parent < 0)
    // ---- B5: wedged-burst self-abort (host-recoverable, no cold-AC) ----
    // STALL_TIMEOUT = max ap_clk cycles the writer may wait on ANY single AXI
    // handshake (AWREADY / WREADY / BVALID) before it self-aborts to S_DONE and
    // sets the sticky out_err bit. Default 2^20 ≈ 8.4 ms @125 MHz — generous
    // enough never to trip on a healthy NoC (a full v24 192-byte write is <1 µs),
    // small enough to abort well inside the host's silicon poll window. The TB
    // overrides this to a tiny value so the timeout path simulates fast.
    parameter int STALL_TIMEOUT  = 1 << 20,
    // ---- PROBE V3: optional tree/candidate DUMP regions (out3 / out4) ----
    //   EN_DUMP=0 (default) -> dump ports tie off; the writer is BIT-IDENTICAL to
    //   the legacy build (same regions, same beats, byte-for-byte). EN_DUMP=1 +
    //   the runtime dump_en register bit gates two EXTRA sequential bursts in the
    //   SAME FSM (single write master preserved):
    //     out3 = G8.1 CANDIDATE set : CAND_N x {id, cheap_score} = 2*CAND_N words
    //     out4 = TREE HEAP full     : K_F x {id, score, parent} + K_F rescored
    //                                 = 4*K_F words
    parameter bit EN_DUMP   = 1'b0,
    parameter int CAND_N    = 512        // G8.1 candidate-set size (prod K_C=512)
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- Trigger + payload from the candidate-tree selector ----
    input  wire                       tree_done,       // 1-cycle pulse: top-K ready
    input  wire signed [K_F*SCORE_W-1:0] topk_scores,  // heap-ranked, [0]=largest
    input  wire        [K_F*IDX_W-1:0]   topk_indices,

    // ---- Destination base pointers ----
    //  M7  mode: out0_base = idx+score region, out1_base = idx mirror.
    //  v24 mode: out0_base = ids array, out1_base = scores array,
    //            out2_base = parents array (control_regs 0x28 / 0x30 / 0x38).
    input  wire [AXI_ADDR_W-1:0]      out0_base,
    input  wire [AXI_ADDR_W-1:0]      out1_base,
    input  wire [AXI_ADDR_W-1:0]      out2_base,

    // ---- PROBE V3 dump regions (used only when EN_DUMP & dump_en) ----
    //   dump_en (0xC0 bit0, default 0): gate the out3/out4 dump bursts. When 0
    //   the writer issues EXACTLY the legacy region set (bit-identical).
    input  wire                       dump_en,
    input  wire [AXI_ADDR_W-1:0]      out3_base,   // 0xC4/0xC8 : candidate set
    input  wire [AXI_ADDR_W-1:0]      out4_base,   // 0xCC/0xD0 : tree heap full
    //   out3 candidate read port: the writer drives a beat index, the top maps it
    //   to the G8 candidate buffer. Word 2*c = id[c], word 2*c+1 = cheap_score[c].
    output wire [$clog2((CAND_N*2 > 1) ? CAND_N*2 : 2)-1:0] dump3_rdaddr,
    input  wire [AXI_DATA_W-1:0]      dump3_rddata,
    //   out4 tree-heap snapshot (flat, latched at trigger): K_F entries of
    //   {id, score, parent} interleaved (3 words/entry) then K_F rescored scores.
    input  wire [K_F*IDX_W-1:0]       tree_ids,
    input  wire [K_F*SCORE_W-1:0]     tree_scores,
    input  wire [K_F*32-1:0]          tree_parents,   // parent index per child (INT32)
    input  wire [K_F*SCORE_W-1:0]     g8_rescored,    // G8 rescored fp32 per child

    // ---- Status (RO, exposed via control_regs 0x78) ----
    //   out_dma_status @ 0x78:  bit0 = out_busy, bit1 = out_done, bit2 = out_err
    // INTEGRATOR NOTE (B5): out_err is a NEW status port. Today control_regs.sv
    //   assembles 0x78 as {30'b0, out_dma_done, out_dma_busy} (control_regs.sv:309).
    //   To surface bit2, the integrator must (a) add `input wire out_dma_err` to
    //   control_regs.sv and change the 0x78 read to
    //   {29'b0, out_dma_err, out_dma_done, out_dma_busy}, and (b) route a new
    //   wire `out_dma_err` from THIS module's out_err up through eagle4_drafter.sv
    //   into control_regs (mirroring how out_dma_busy / out_dma_done are wired at
    //   eagle4_drafter.sv:387-388 / 960-961). This module only EXPORTS out_err.
    output reg                        out_busy,
    output reg                        out_done,         // sticky-high after a completed write
    output reg                        out_err,          // B5: sticky stall-timeout abort flag (status bit2)

    // ---- AXI4 write master (write-only: AW + W + B channels) ----
    output reg  [AXI_ID_W-1:0]        m_axi_out_awid,
    output reg  [AXI_ADDR_W-1:0]      m_axi_out_awaddr,
    output reg  [7:0]                 m_axi_out_awlen,
    output reg  [2:0]                 m_axi_out_awsize,
    output reg  [1:0]                 m_axi_out_awburst,
    output reg                        m_axi_out_awvalid,
    input  wire                       m_axi_out_awready,

    output reg  [AXI_DATA_W-1:0]      m_axi_out_wdata,
    output reg  [AXI_DATA_W/8-1:0]    m_axi_out_wstrb,
    output reg                        m_axi_out_wlast,
    output reg                        m_axi_out_wvalid,
    input  wire                       m_axi_out_wready,

    input  wire [AXI_ID_W-1:0]        m_axi_out_bid,
    input  wire [1:0]                 m_axi_out_bresp,
    input  wire                       m_axi_out_bvalid,
    output reg                        m_axi_out_bready
);

    localparam [2:0] AWSIZE_32 = 3'b010;          // 4 bytes/beat
    localparam [1:0] BURST_INCR = 2'b01;
    localparam int   BYTES_PER_BEAT = AXI_DATA_W/8;

    // ----------------------------------------------------------------
    // Beat counts per region, per ABI.
    //   M7  : region 0 = idx[0..K_F-1] then score[0..K_F-1] -> 2*K_F beats
    //         region 1 = idx[0..K_F-1]                      ->   K_F beats
    //   v24 : region 0 = ids    [TOP_K]  (K_F live + pad)
    //         region 1 = scores [TOP_K]
    //         region 2 = parents[TOP_K]  (all PARENT_SENTINEL)
    // The staging buffer is sized for the larger of the two ABIs.
    // ----------------------------------------------------------------
    localparam int M7_R0_BEATS  = 2*K_F;          // 16
    localparam int M7_R1_BEATS  = K_F;            // 8
    localparam int V24_BEATS    = TOP_K;          // 16

    // Staging buffer depth: M7 region 0 needs 2*K_F; v24 needs TOP_K. Both 16
    // for the canonical K_F=8 / TOP_K=16 config; take the max generically.
    localparam int STAGE_DEPTH  = (M7_R0_BEATS > V24_BEATS) ? M7_R0_BEATS : V24_BEATS;

    // Number of BASE (legacy) write regions (bursts) this ABI issues.
    //   v24 : always 3 (ids, scores, parents)
    //   M7  : 2 if EN_OUT1_MIRROR else 1
    localparam int BASE_REGIONS = ABI_V24 ? 3 : (EN_OUT1_MIRROR ? 2 : 1);
    // PROBE V3: + 2 dump regions (out3 candidate set, out4 tree heap) when the
    // host enables dump_en at runtime AND the build sets EN_DUMP. Region indices
    // for the dump bursts are BASE_REGIONS (out3) and BASE_REGIONS+1 (out4).
    localparam int R3_IDX   = BASE_REGIONS;       // out3 = candidate set
    localparam int R4_IDX   = BASE_REGIONS + 1;   // out4 = tree heap full
    localparam int CAND_BEATS = 2*CAND_N;         // out3 : {id,score} per candidate
    localparam int TREE_BEATS = 4*K_F;            // out4 : {id,score,parent}+rescored
    // AXI4 arlen ceiling: split bursts into <= MAX_SUB beats (out3 = up to 1024).
    localparam int MAX_SUB  = 256;
    // beat counters must span the WIDEST sub-burst AND the widest region. BEAT_W
    // is sized to hold MAX_SUB so the `(BEAT_W+1)'(MAX_SUB)` compares are exact.
    localparam int W1 = (EN_DUMP && CAND_BEATS > STAGE_DEPTH) ? CAND_BEATS : STAGE_DEPTH;
    localparam int MAXBEATS = (W1 > MAX_SUB) ? W1 : MAX_SUB;
    localparam int BEAT_W   = $clog2(MAXBEATS + 1);

    // ----------------------------------------------------------------
    // Staging buffers — flat arrays of STAGE_DEPTH INT32 words, latched ONCE
    // at tree_done. Single always_ff writer → multi-driver-safe.
    //
    //   M7  mode uses one buffer `stage_idx_score`:
    //     [0 .. K_F-1]     = zero-extended INDICES
    //     [K_F .. 2*K_F-1] = sign-extended  SCORES
    //   v24 mode uses three buffers (ids / scores / parents), each TOP_K deep:
    //     [0 .. K_F-1]     = LIVE  (ids zero-ext, scores sign-ext, parents=sentinel)
    //     [K_F .. TOP_K-1] = PAD   (ids=0, scores=0, parents=sentinel)
    // ----------------------------------------------------------------
    reg signed [AXI_DATA_W-1:0] stage_idx_score [0:STAGE_DEPTH-1]; // M7
    reg signed [AXI_DATA_W-1:0] stage_ids       [0:STAGE_DEPTH-1]; // v24
    reg signed [AXI_DATA_W-1:0] stage_scores    [0:STAGE_DEPTH-1]; // v24
    reg signed [AXI_DATA_W-1:0] stage_parents   [0:STAGE_DEPTH-1]; // v24
    // PROBE V3 out4 staging: 4*K_F words = K_F x {id,score,parent} interleaved
    // (3 words/child) then K_F rescored scores. Latched at trigger.
    reg signed [AXI_DATA_W-1:0] stage_tree      [0:(EN_DUMP ? TREE_BEATS : 1)-1];

    // ----------------------------------------------------------------
    // Latched destination pointers (snapshot at tree_done so the host may
    // change out{0,1,2,3,4}_base after launch without disturbing an in-flight
    // write).
    // ----------------------------------------------------------------
    reg [AXI_ADDR_W-1:0] out0_base_r;
    reg [AXI_ADDR_W-1:0] out1_base_r;
    reg [AXI_ADDR_W-1:0] out2_base_r;
    reg [AXI_ADDR_W-1:0] out3_base_r;   // PROBE V3 candidate set
    reg [AXI_ADDR_W-1:0] out4_base_r;   // PROBE V3 tree heap
    reg                  dump_en_r;     // latched dump enable for this burst run
    // runtime region count: base regions, + 2 dump regions when armed this run.
    wire [3:0] n_regions_run = EN_DUMP && dump_en_r ? BASE_REGIONS[3:0] + 4'd2
                                                    : BASE_REGIONS[3:0];

    // ----------------------------------------------------------------
    // FSM
    // ----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE   = 3'd0,
        S_AWREQ  = 3'd1,   // assert AWVALID for the current region, wait AWREADY
        S_WDATA  = 3'd2,   // stream W beats, WLAST on final
        S_BRESP  = 3'd3,   // wait BVALID, then advance region or finish
        S_DONE   = 3'd4    // parked; AWVALID/WVALID low forever (silent)
    } state_e;
    state_e state;

    reg [2:0]  region;          // 0..4 — region index (burst); +2 dump regions
    reg [BEAT_W-1:0] beat_idx;   // word index within the current REGION (out3 wide)

    // PROBE V3 / AXI-arlen rule (feedback_axi_arlen_8bit_truncation_h3072):
    //   out3 is 2*CAND_N = up to 1024 beats > the 256-beat AXI4 arlen ceiling.
    //   A burst is therefore split into <=MAX_SUB-beat SUB-BURSTS. beat_idx is
    //   the running word index within the REGION; sub_total is the beats in the
    //   CURRENT sub-burst; sub_base is its starting beat. Legacy regions (<=16
    //   beats) emit a single sub-burst, so their AW/W/B sequence is unchanged.
    reg [BEAT_W-1:0] sub_base;     // first region-beat of the current sub-burst
    reg [BEAT_W:0]   sub_total;    // beats in the current sub-burst (<=MAX_SUB)

    // PROBE V3: out3 candidate-set read address. The writer presents the beat
    // index to the top, which maps word 2*c->id[c], word 2*c+1->cheap_score[c].
    // During S_WDATA the writer prefetches beat_idx+1 (nxt_word), so the address
    // must lead by one there; in S_AWREQ it presents beat 0 (addr 0 = beat_idx).
    localparam int DUMP3_AW = $clog2((CAND_N*2 > 1) ? CAND_N*2 : 2);
    wire [BEAT_W-1:0] dump3_beat = (state == S_WDATA) ? (beat_idx + (BEAT_W)'(1))
                                                      : beat_idx;
    assign dump3_rdaddr = (EN_DUMP && (region == R3_IDX[2:0]))
                          ? dump3_beat[DUMP3_AW-1:0] : DUMP3_AW'(0);

    // ----------------------------------------------------------------
    // B5: bounded stall-timeout counter. Increments every cycle the writer is
    // waiting on an AXI handshake that has NOT yet made progress (AWVALID high
    // & AWREADY low; WVALID high & WREADY low; or BREADY high & BVALID low). It
    // RESETS to 0 the cycle any handshake advances, and is held at 0 in S_IDLE /
    // S_DONE (idle = not stalled). On reaching STALL_TIMEOUT the FSM forces
    // S_DONE and latches the sticky out_err bit so the host can detect a wedged
    // burst by reading 0x78 bit2 — recoverable without a cold-AC. Width is sized
    // to hold STALL_TIMEOUT; +1 guard bit keeps the compare clean.
    localparam int STALL_W = $clog2(STALL_TIMEOUT + 1) + 1;
    reg [STALL_W-1:0] stall_cnt;

    // --- per-region beat count + base address (combinational selects) ---
    reg  [BEAT_W:0]       cur_region_beats;
    reg  [AXI_ADDR_W-1:0] cur_region_base;
    always_comb begin
        // PROBE V3 dump regions take priority on their indices (out3 / out4).
        if (EN_DUMP && (region == R3_IDX[2:0])) begin
            cur_region_beats = (BEAT_W+1)'(CAND_BEATS);   // out3 = 2*CAND_N words
            cur_region_base  = out3_base_r;
        end else if (EN_DUMP && (region == R4_IDX[2:0])) begin
            cur_region_beats = (BEAT_W+1)'(TREE_BEATS);   // out4 = 4*K_F words
            cur_region_base  = out4_base_r;
        end else if (ABI_V24) begin
            // v24: three equal TOP_K-beat bursts -> ids/scores/parents.
            cur_region_beats = (BEAT_W+1)'(V24_BEATS);
            unique case (region)
                3'd0:    cur_region_base = out0_base_r;  // ids
                3'd1:    cur_region_base = out1_base_r;  // scores
                default: cur_region_base = out2_base_r;  // parents (region 2)
            endcase
        end else begin
            // M7: region 0 = idx+score (out0), region 1 = idx mirror (out1).
            cur_region_beats = (region == 3'd0) ? (BEAT_W+1)'(M7_R0_BEATS)
                                                : (BEAT_W+1)'(M7_R1_BEATS);
            cur_region_base  = (region == 3'd0) ? out0_base_r : out1_base_r;
        end
    end

    // --- current staging word for the current region+beat (combinational) ---
    //  M7  : both regions index stage_idx_score directly.
    //  v24 : region selects which of the three staging arrays.
    //  out3: words come from dump3_rddata (top maps the read addr to the G8
    //        candidate buffer); cur_word == nxt_word == dump3_rddata because the
    //        address presented (dump3_rdaddr below) already tracks the next beat.
    //  out4: words come from stage_tree (latched at trigger).
    // STAGE_DEPTH-bounded indices use the low bits; dump indices are wider.
    localparam int TREE_AW = $clog2((EN_DUMP ? TREE_BEATS : 2));
    wire [4:0]        base_idx = beat_idx[4:0];           // base regions <= 16 deep
    wire [4:0]        base_nxt = beat_idx[4:0] + 5'd1;
    wire [BEAT_W-1:0] beat_idx_p1 = beat_idx + (BEAT_W)'(1);
    wire [TREE_AW-1:0] tree_idx    = beat_idx[TREE_AW-1:0];
    wire [TREE_AW-1:0] tree_idx_p1 = beat_idx_p1[TREE_AW-1:0];
    // PROBE V3 sub-burst helpers (avoid index-on-expression in the FSM).
    // region_beats_total = full beats of the current region (from the selects).
    wire [BEAT_W:0]  region_beats_total = cur_region_beats;
    wire [BEAT_W:0]  sub_remaining = region_beats_total - {1'b0, sub_base};
    wire             sub_is_capped = (sub_remaining > (BEAT_W+1)'(MAX_SUB));
    wire [BEAT_W:0]  sub_len       = sub_is_capped ? (BEAT_W+1)'(MAX_SUB) : sub_remaining;
    wire [7:0]       sub_awlen     = sub_len[7:0] - 8'd1;   // beats-1 (sub_len<=256)
    wire [AXI_ADDR_W-1:0] sub_addr = cur_region_base +
                                     (AXI_ADDR_W'(sub_base) * BYTES_PER_BEAT);
    reg signed [AXI_DATA_W-1:0] cur_word;
    reg signed [AXI_DATA_W-1:0] nxt_word;
    always_comb begin
        if (EN_DUMP && (region == R3_IDX[2:0])) begin
            // out3 candidate set: top returns the word at dump3_rdaddr.
            cur_word = dump3_rddata;
            nxt_word = dump3_rddata;   // dump3_rdaddr is driven one ahead below
        end else if (EN_DUMP && (region == R4_IDX[2:0])) begin
            cur_word = stage_tree[tree_idx];
            nxt_word = stage_tree[tree_idx_p1];
        end else if (ABI_V24) begin
            unique case (region)
                3'd0:    begin cur_word = stage_ids    [base_idx];
                               nxt_word = stage_ids    [base_nxt]; end
                3'd1:    begin cur_word = stage_scores [base_idx];
                               nxt_word = stage_scores [base_nxt]; end
                default: begin cur_word = stage_parents[base_idx];
                               nxt_word = stage_parents[base_nxt]; end
            endcase
        end else begin
            cur_word = stage_idx_score[base_idx];
            nxt_word = stage_idx_score[base_nxt];
        end
    end

    // ----------------------------------------------------------------
    // B5: combinational stall-progress detector.
    //   waiting  = the writer currently has a handshake outstanding (it has
    //              asserted a *VALID/BREADY and is waiting on the slave's ready).
    //   progress = that handshake actually completed THIS cycle.
    // The counter (in the always_ff) increments while (waiting & !progress) and
    // resets on progress. These are pure functions of the registered FSM state +
    // current AXI signals → no extra driver on any AXI net (single-writer rule).
    // ----------------------------------------------------------------
    reg waiting;
    reg progress;
    always_comb begin
        waiting  = 1'b0;
        progress = 1'b0;
        case (state)
            S_AWREQ: begin
                // Stalled iff AWVALID is up but AWREADY hasn't returned.
                waiting  = m_axi_out_awvalid;
                progress = m_axi_out_awvalid && m_axi_out_awready;
            end
            S_WDATA: begin
                // Stalled iff a W beat is presented but WREADY hasn't returned.
                waiting  = m_axi_out_wvalid;
                progress = m_axi_out_wvalid && m_axi_out_wready;
            end
            S_BRESP: begin
                // Stalled iff BREADY is up but BVALID hasn't arrived.
                waiting  = m_axi_out_bready;
                progress = m_axi_out_bready && m_axi_out_bvalid;
            end
            default: begin
                waiting  = 1'b0;   // S_IDLE / S_DONE = not stalled
                progress = 1'b0;
            end
        endcase
    end

    // B5: timeout fires the cycle the stall counter would exceed STALL_TIMEOUT
    // while still waiting with no progress. Sampled in the always_ff below.
    wire stall_timeout = waiting && !progress && (stall_cnt >= STALL_TIMEOUT[STALL_W-1:0]);

    integer i;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            out_busy          <= 1'b0;
            out_done          <= 1'b0;
            out_err           <= 1'b0;
            stall_cnt         <= '0;
            region            <= 3'd0;
            beat_idx          <= '0;
            sub_base          <= '0;
            sub_total         <= '0;
            out0_base_r       <= '0;
            out1_base_r       <= '0;
            out2_base_r       <= '0;
            out3_base_r       <= '0;   // PROBE V3
            out4_base_r       <= '0;   // PROBE V3
            dump_en_r         <= 1'b0; // PROBE V3
            m_axi_out_awid    <= '0;
            m_axi_out_awaddr  <= '0;
            m_axi_out_awlen   <= 8'd0;
            m_axi_out_awsize  <= AWSIZE_32;
            m_axi_out_awburst <= BURST_INCR;
            m_axi_out_awvalid <= 1'b0;
            m_axi_out_wdata   <= '0;
            m_axi_out_wstrb   <= '0;
            m_axi_out_wlast   <= 1'b0;
            m_axi_out_wvalid  <= 1'b0;
            m_axi_out_bready  <= 1'b0;
            for (i = 0; i < STAGE_DEPTH; i = i + 1) begin
                stage_idx_score[i] <= '0;
                stage_ids[i]       <= '0;
                stage_scores[i]    <= '0;
                stage_parents[i]   <= '0;
            end
            if (EN_DUMP) for (i = 0; i < TREE_BEATS; i = i + 1) stage_tree[i] <= '0;
        end else begin
            // ----------------------------------------------------------------
            // B5: stall-counter maintenance (runs every cycle, BEFORE the FSM
            // case so the case's state writes win). Counts cycles spent waiting
            // on a handshake with no progress; resets on progress or whenever we
            // are not waiting. This does NOT touch any happy-path signal — on a
            // healthy burst `progress` pulses each handshake so stall_cnt never
            // climbs past a couple of cycles and stall_timeout never asserts.
            if (waiting && !progress) stall_cnt <= stall_cnt + 1'b1;
            else                      stall_cnt <= '0;

            case (state)
                // --------------------------------------------------------
                S_IDLE: begin
                    out_busy          <= 1'b0;
                    m_axi_out_awvalid <= 1'b0;
                    m_axi_out_wvalid  <= 1'b0;
                    m_axi_out_wlast   <= 1'b0;
                    m_axi_out_bready  <= 1'b0;
                    if (tree_done) begin
                        // Snapshot payload + destination pointers ONCE.
                        if (ABI_V24) begin
                            // --- v24: build ids/scores/parents staging ---
                            // Live K_F results: ids zero-ext, scores sign-ext,
                            // parents = root sentinel.
                            for (i = 0; i < K_F; i = i + 1) begin
                                stage_ids[i] <= $signed(
                                    {{(AXI_DATA_W-IDX_W){1'b0}},
                                     topk_indices[i*IDX_W +: IDX_W]});
                                stage_scores[i] <= $signed(
                                    {{(AXI_DATA_W-SCORE_W){topk_scores[i*SCORE_W+SCORE_W-1]}},
                                     topk_scores[i*SCORE_W +: SCORE_W]});
                                stage_parents[i] <= PARENT_SENTINEL[AXI_DATA_W-1:0];
                            end
                            // Zero-pad t ∈ [K_F .. TOP_K-1]: ids=0, scores=0,
                            // parents=sentinel.
                            for (i = K_F; i < STAGE_DEPTH; i = i + 1) begin
                                stage_ids[i]     <= '0;
                                stage_scores[i]  <= '0;
                                stage_parents[i] <= PARENT_SENTINEL[AXI_DATA_W-1:0];
                            end
                        end else begin
                            // --- M7: idx[0..K_F-1] then score[0..K_F-1] ---
                            for (i = 0; i < K_F; i = i + 1) begin
                                stage_idx_score[i] <= $signed(
                                    {{(AXI_DATA_W-IDX_W){1'b0}},
                                     topk_indices[i*IDX_W +: IDX_W]});
                                stage_idx_score[K_F + i] <= $signed(
                                    {{(AXI_DATA_W-SCORE_W){topk_scores[i*SCORE_W+SCORE_W-1]}},
                                     topk_scores[i*SCORE_W +: SCORE_W]});
                            end
                        end
                        out0_base_r <= out0_base;
                        out1_base_r <= out1_base;
                        out2_base_r <= out2_base;
                        // PROBE V3: latch dump bases + enable + tree-heap snapshot.
                        if (EN_DUMP) begin
                            out3_base_r <= out3_base;
                            out4_base_r <= out4_base;
                            dump_en_r   <= dump_en;
                            // stage_tree = K_F x {id,score,parent} interleaved,
                            // then K_F rescored scores (3*K_F + K_F = 4*K_F words).
                            for (i = 0; i < K_F; i = i + 1) begin
                                stage_tree[3*i + 0] <= $signed(
                                    {{(AXI_DATA_W-IDX_W){1'b0}}, tree_ids[i*IDX_W +: IDX_W]});
                                stage_tree[3*i + 1] <= tree_scores[i*SCORE_W +: SCORE_W];
                                stage_tree[3*i + 2] <= tree_parents[i*32 +: 32];
                                stage_tree[3*K_F + i] <= g8_rescored[i*SCORE_W +: SCORE_W];
                            end
                        end
                        region      <= 3'd0;
                        beat_idx     <= '0;     // PROBE V3: region-relative word idx
                        sub_base     <= '0;     // PROBE V3: first sub-burst at 0
                        out_busy    <= 1'b1;
                        out_done    <= 1'b0;
                        out_err     <= 1'b0;     // B5: clear sticky err on (re)launch
                        stall_cnt   <= '0;       // B5: fresh stall budget per launch
                        state       <= S_AWREQ;
                    end
                end

                // --------------------------------------------------------
                // Issue the AW for the current region. One outstanding burst.
                S_AWREQ: begin
                    // B5: a wedged AW (AWREADY never returns) self-aborts here.
                    // stall_timeout is 0 on every healthy cycle, so the else-arm
                    // is bit-identical to the pre-B5 behavior.
                    if (stall_timeout) begin
                        m_axi_out_awvalid <= 1'b0;   // clean deassert (no valid-high in drain)
                        m_axi_out_wvalid  <= 1'b0;
                        m_axi_out_wlast   <= 1'b0;
                        m_axi_out_bready  <= 1'b0;
                        out_busy          <= 1'b0;
                        out_done          <= 1'b1;
                        out_err           <= 1'b1;   // sticky abort flag (status bit2)
                        state             <= S_DONE;
                    end else if (!m_axi_out_awvalid) begin
                        // PROBE V3: bound this sub-burst to <= MAX_SUB beats so
                        // arlen never truncates (out3 = up to 1024 beats). For a
                        // legacy region (<=16 beats) the first sub-burst covers
                        // the whole region (sub_base=0), so AW/W/B are unchanged.
                        // remaining = region_beats - sub_base
                        // sub = min(MAX_SUB, remaining)
                        // sub_total holds `sub`; awaddr advances by sub_base words.
                        // (region_beats_total - sub_base) fits in BEAT_W+1.
                        sub_total <= sub_len;
                        m_axi_out_awid    <= '0;
                        m_axi_out_awaddr  <= sub_addr;
                        m_axi_out_awlen   <= sub_awlen;
                        m_axi_out_awsize  <= AWSIZE_32;
                        m_axi_out_awburst <= BURST_INCR;
                        m_axi_out_awvalid <= 1'b1;
                        beat_idx          <= sub_base;   // resume region-relative idx
                    end else if (m_axi_out_awready) begin
                        // AW accepted — drop AWVALID, present the FIRST W beat of
                        // this sub-burst (cur_word = word @ beat_idx == sub_base).
                        m_axi_out_awvalid <= 1'b0;
                        m_axi_out_wvalid  <= 1'b1;
                        m_axi_out_wstrb   <= {(AXI_DATA_W/8){1'b1}};
                        m_axi_out_wdata   <= cur_word;
                        m_axi_out_wlast   <= (sub_total == (BEAT_W+1)'(1)); // single-beat sub-burst
                        state             <= S_WDATA;
                    end
                end

                // --------------------------------------------------------
                // Stream the W beats. beat_idx = word currently on the bus.
                // On each accepted beat, advance to the next staging word, or
                // (after WLAST is accepted) move to BRESP.
                S_WDATA: begin
                    // B5: a wedged W beat (WREADY never returns) self-aborts.
                    if (stall_timeout) begin
                        m_axi_out_awvalid <= 1'b0;
                        m_axi_out_wvalid  <= 1'b0;   // clean deassert (no valid-high in drain)
                        m_axi_out_wlast   <= 1'b0;
                        m_axi_out_bready  <= 1'b0;
                        out_busy          <= 1'b0;
                        out_done          <= 1'b1;
                        out_err           <= 1'b1;
                        state             <= S_DONE;
                    end else if (m_axi_out_wvalid && m_axi_out_wready) begin
                        if (m_axi_out_wlast) begin
                            // final beat of this sub-burst accepted — close W.
                            m_axi_out_wvalid <= 1'b0;
                            m_axi_out_wlast  <= 1'b0;
                            m_axi_out_bready <= 1'b1;
                            state            <= S_BRESP;
                        end else begin
                            // present the next beat (nxt_word = word @ beat_idx+1).
                            // WLAST asserts when beat_idx+1 is the LAST word of the
                            // current sub-burst, i.e. (beat_idx+2) == sub_base+sub_total.
                            m_axi_out_wdata <= nxt_word;
                            m_axi_out_wlast <= (({1'b0, beat_idx} + (BEAT_W+1)'(2))
                                               == ({1'b0, sub_base} + sub_total));
                            beat_idx        <= beat_idx + (BEAT_W)'(1);
                        end
                    end
                end

                // --------------------------------------------------------
                // Wait for the write response, then advance region or finish.
                S_BRESP: begin
                    // B5: a wedged B (BVALID never returns) self-aborts.
                    if (stall_timeout) begin
                        m_axi_out_awvalid <= 1'b0;
                        m_axi_out_wvalid  <= 1'b0;
                        m_axi_out_wlast   <= 1'b0;
                        m_axi_out_bready  <= 1'b0;   // clean deassert
                        out_busy          <= 1'b0;
                        out_done          <= 1'b1;
                        out_err           <= 1'b1;
                        state             <= S_DONE;
                    end else begin
                        m_axi_out_bready <= 1'b1;
                        if (m_axi_out_bvalid) begin
                            m_axi_out_bready <= 1'b0;
                            // More sub-bursts left in THIS region? (out3 splits into
                            // ceil(2*CAND_N / MAX_SUB) sub-bursts.) sub_base+sub_total
                            // is the next region-beat to send.
                            if (({1'b0, sub_base} + sub_total) < region_beats_total) begin
                                sub_base <= sub_base + sub_total[BEAT_W-1:0];
                                state    <= S_AWREQ;   // next sub-burst, same region
                            end else if ({1'b0, region} + 4'd1 < n_regions_run) begin
                                // region complete -> next region, reset sub_base.
                                region   <= region + 3'd1;
                                sub_base <= '0;
                                beat_idx <= '0;
                                state    <= S_AWREQ;
                            end else begin
                                out_busy <= 1'b0;
                                out_done <= 1'b1;
                                state    <= S_DONE;
                            end
                        end
                    end
                end

                // --------------------------------------------------------
                // Parked. AWVALID/WVALID stay low → port is SILENT after the
                // burst run (single transient AWVALID driver). Re-arm on the
                // next tree_done (next draft step) — clears done, restarts.
                S_DONE: begin
                    m_axi_out_awvalid <= 1'b0;
                    m_axi_out_wvalid  <= 1'b0;
                    m_axi_out_wlast   <= 1'b0;
                    m_axi_out_bready  <= 1'b0;
                    out_busy          <= 1'b0;
                    out_done          <= 1'b1;
                    // out_err is held sticky here (no assignment = retains value)
                    // until the next launch clears it.
                    // ----------------------------------------------------------------
                    // PAIRING-FIX (2026-06-06, PDI#7.2): a tree_done pulse in S_DONE now
                    // STARTS the new burst DIRECTLY (snapshot payload + go S_AWREQ),
                    // mirroring weight_dma_loader's S_DONE re-arm fix. The OLD code
                    // merely bounced S_DONE->S_IDLE and consumed the single trigger
                    // pulse there; by the time S_IDLE ran the pulse (ow_trigger =
                    // g8_done_w, ONE pulse per forward, never repeated) was gone, so
                    // invocation #2/#4/... never latched or burst -> HBM kept the prior
                    // output -> the silicon "every-other-run stale output" pairing bug.
                    // The snapshot+arm body below is byte-identical to the S_IDLE arm.
                    // ----------------------------------------------------------------
                    if (tree_done) begin
                        // Snapshot payload + destination pointers ONCE (same as S_IDLE).
                        if (ABI_V24) begin
                            // --- v24: build ids/scores/parents staging ---
                            for (i = 0; i < K_F; i = i + 1) begin
                                stage_ids[i] <= $signed(
                                    {{(AXI_DATA_W-IDX_W){1'b0}},
                                     topk_indices[i*IDX_W +: IDX_W]});
                                stage_scores[i] <= $signed(
                                    {{(AXI_DATA_W-SCORE_W){topk_scores[i*SCORE_W+SCORE_W-1]}},
                                     topk_scores[i*SCORE_W +: SCORE_W]});
                                stage_parents[i] <= PARENT_SENTINEL[AXI_DATA_W-1:0];
                            end
                            for (i = K_F; i < STAGE_DEPTH; i = i + 1) begin
                                stage_ids[i]     <= '0;
                                stage_scores[i]  <= '0;
                                stage_parents[i] <= PARENT_SENTINEL[AXI_DATA_W-1:0];
                            end
                        end else begin
                            // --- M7: idx[0..K_F-1] then score[0..K_F-1] ---
                            for (i = 0; i < K_F; i = i + 1) begin
                                stage_idx_score[i] <= $signed(
                                    {{(AXI_DATA_W-IDX_W){1'b0}},
                                     topk_indices[i*IDX_W +: IDX_W]});
                                stage_idx_score[K_F + i] <= $signed(
                                    {{(AXI_DATA_W-SCORE_W){topk_scores[i*SCORE_W+SCORE_W-1]}},
                                     topk_scores[i*SCORE_W +: SCORE_W]});
                            end
                        end
                        out0_base_r <= out0_base;
                        out1_base_r <= out1_base;
                        out2_base_r <= out2_base;
                        // PROBE V3: latch dump bases + enable + tree-heap snapshot.
                        if (EN_DUMP) begin
                            out3_base_r <= out3_base;
                            out4_base_r <= out4_base;
                            dump_en_r   <= dump_en;
                            for (i = 0; i < K_F; i = i + 1) begin
                                stage_tree[3*i + 0] <= $signed(
                                    {{(AXI_DATA_W-IDX_W){1'b0}}, tree_ids[i*IDX_W +: IDX_W]});
                                stage_tree[3*i + 1] <= tree_scores[i*SCORE_W +: SCORE_W];
                                stage_tree[3*i + 2] <= tree_parents[i*32 +: 32];
                                stage_tree[3*K_F + i] <= g8_rescored[i*SCORE_W +: SCORE_W];
                            end
                        end
                        region    <= 3'd0;
                        beat_idx  <= '0;
                        sub_base  <= '0;
                        out_busy  <= 1'b1;
                        out_done  <= 1'b0;
                        out_err   <= 1'b0;     // B5: clear sticky err on (re)launch
                        stall_cnt <= '0;       // B5: fresh stall budget per launch
                        state     <= S_AWREQ;  // start the burst NOW (no S_IDLE bounce)
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
