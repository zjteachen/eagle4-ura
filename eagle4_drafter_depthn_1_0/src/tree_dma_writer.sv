// =====================================================================
// tree_dma_writer.sv
// DEPTH-N OUTPUT-DMA: burst the WHOLE tree (the 4-tensor SGLang verify ABI)
// to HBM, consuming tree_out_walk's node-major emit stream.
// =====================================================================
//
// WHY (reports/results/TREE_FSM_STAGE4.md §6 item 2; SGLANG_TREE_KV_SURFACE.md §A5/§C):
//   `tree_out_walk` builds the depth-N tree and emits it node-major, but the
//   shipped `output_dma_writer` only bursts the depth-0 frontier (K_F top-K
//   (score,index) pairs). At depth>0 the host must read back the WHOLE tree.
//   This module is the missing consumer: it buffers `tree_out_walk`'s emit
//   stream into FF register files and, on `walk_done`, bursts the FOUR tensors
//   the SGLang verify kernel consumes (eagle_utils.py:308, :344):
//
//     draft_token          [n_nodes] int32   node-major vocab ids
//     retrive_index        [n_nodes] int32   IDENTITY (arange) — see below
//     retrive_next_token   [n_nodes] int32   first child of each node (-1 = none)
//     retrive_next_sibling [n_nodes] int32   next sibling of each node (-1 = none)
//
//   `custom_mask` / `positions` serve the TARGET forward, NOT the verify kernel,
//   and are deliberately NOT built here (matches tree_out_walk's 4-tensor ABI).
//
// LAYOUT (SoA, host-idiomatic) — one contiguous int32 array per tensor, each at
//   a FIXED element stride TREE_STRIDE off `tree_base`, so tensor k lives at
//     tree_base + k*TREE_STRIDE*4  (bytes),  length n_nodes int32.
//   The host reads n_nodes (status), then 4 slices — each maps 1:1 onto a torch
//   tensor of shape (n_nodes,) with NO reshape (see the GAP-6 recipe). This
//   mirrors the v24 ids/scores/parents SoA convention (output_dma_writer.sv:20).
//
// EXPLOITING THE IDENTITY (SGLANG_TREE_KV_SURFACE.md D48 / :344):
//   retrive_index[i] == bid*D + i is an identity map, NOT a permutation, so it
//   costs ZERO storage: region 1 emits the beat counter directly as a CONSTANT
//   (no FF buffer). It is still DMA'd so the host gets a complete 4-tensor set
//   without special-casing; the host MAY instead skip region 1 and use
//   torch.arange(n_nodes) (documented in the GAP-6 recipe).
//
// FF-STAGING MANDATE (feedback_ff_staging_intermediate_seams):
//   The three node buffers are FF register files (`ram_style="registers"` +
//   `shreg_extract="no"`), NOT BRAM/LUTRAM/URAM — verified in the OOC netlist
//   (0 URAM / 0 RAMB / 0 SRL). URAM is the binding wall (99.95%) and is untouched.
//   The tree walk is NOT accept-critical, so cycles are traded for timing freely.
//
// SINGLE AXI WRITE MASTER (feedback_axi_driver_fanin_rule):
//   Like output_dma_writer, this is a write-only AW/W/B master, one outstanding
//   burst, AWVALID asserted only between walk_done and out_done, then SILENT.
//   At the top it is generate-selected against output_dma_writer so exactly ONE
//   of the two drives the physical m_axi_out port (DEPTH_N build-time select) —
//   never two concurrent AWVALID drivers.
//
// B5 stall-timeout (host-recoverable, no cold-AC): a wedged AXI handshake
//   self-aborts to S_DONE and latches sticky out_err (same contract as
//   output_dma_writer's out_err @ 0x78 bit2).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_dma_writer
    import tree_score_pkg::*;
#(
    parameter int NODE_MAX     = N_MAX,      // 904 @ MAX_DEPTH_HW=15
    parameter int AXI_ADDR_W   = 64,
    parameter int AXI_DATA_W   = 32,         // one INT32 per beat
    parameter int AXI_ID_W     = 1,
    parameter int N_TENSOR     = 5,          // draft_token / index / next_tok / next_sib / SCORE
                                             // region 4 = COSTDRAFTTREE-ABI (G1) per-node
                                             // cumulative log2 score, int32 Q12.20 (see below)
    parameter int TREE_STRIDE  = 1024,       // int32 elems between SoA arrays (> NODE_MAX)
    parameter int MAX_SUB      = 256,        // AXI4 arlen ceiling (<=256 beats/sub-burst)
    parameter int STALL_TIMEOUT = 1 << 20    // ~8.4 ms @125 MHz; TB overrides small
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- node-major emit stream from tree_out_walk (one node per beat) ----
    input  wire                       emit_valid,
    input  wire [NODE_W-1:0]          emit_idx,       // == retrive_index (identity)
    input  wire [TOK_W-1:0]           emit_token,     // draft_token
    input  wire signed [NODE_W:0]     emit_next_tok,  // retrive_next_token   (-1 = none)
    input  wire signed [NODE_W:0]     emit_next_sib,  // retrive_next_sibling (-1 = none)
    input  wire signed [SCORE_W-1:0]  emit_score,     // COSTDRAFTTREE-ABI (G1) cumu log2 Q12.28

    // ---- burst trigger + node count ----
    input  wire                       walk_done,      // 1-cycle pulse (tree_out_walk.done)
    input  wire [NODE_W-1:0]          n_nodes,        // nodes built this tree

    // ---- destination (SoA: tensor k at tree_base + k*TREE_STRIDE*4) ----
    input  wire [AXI_ADDR_W-1:0]      tree_base,

    // ---- status (RO) ----
    output reg                        out_busy,
    output reg                        out_done,       // sticky-high after a completed write
    output reg                        out_err,        // sticky stall-timeout abort

    // ---- AXI4 write master (AW + W + B channels only) ----
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

    localparam [2:0] AWSIZE_32  = 3'b010;   // 4 bytes/beat
    localparam [1:0] BURST_INCR = 2'b01;
    localparam int   BYTES_PER_BEAT = AXI_DATA_W/8;

    // beat counter width: spans the widest sub-burst AND the widest region
    // (a region is n_nodes <= NODE_MAX beats). Sized to hold max(NODE_MAX,MAX_SUB).
    localparam int W1       = (NODE_MAX > MAX_SUB) ? NODE_MAX : MAX_SUB;
    localparam int BEAT_W   = $clog2(W1 + 1);

    // ----------------------------------------------------------------
    // FF register files (NOT BRAM/LUTRAM/URAM). retrive_index is NOT stored
    // (emitted as a constant beat counter), so only three arrays are buffered.
    // Narrow storage (token TOK_W, links NODE_W+1); zero/sign-extended to int32
    // at the AXI word select below.
    // ----------------------------------------------------------------
    (* ram_style = "registers", shreg_extract = "no" *)
    reg        [TOK_W-1:0]  buf_tok  [0:NODE_MAX-1];
    (* ram_style = "registers", shreg_extract = "no" *)
    reg signed [NODE_W:0]   buf_ntok [0:NODE_MAX-1];
    (* ram_style = "registers", shreg_extract = "no" *)
    reg signed [NODE_W:0]   buf_nsib [0:NODE_MAX-1];
    // COSTDRAFTTREE-ABI (G1): per-node score buffered as int32 Q12.20 (the top 32 bits
    // of the Q12.28 accumulator = drop the 8 LSB fraction bits). ulp = 2^-20 = 9.54e-7,
    // 358x finer than the measured survivor-decision floor (3.42e-4 log2 units,
    // tree_score_pkg.sv) -> narrowing to int32 is order-safe. FF, not BRAM/URAM.
    localparam int SCORE_DMA_W = 32;           // Q12.20 int32
    (* ram_style = "registers", shreg_extract = "no" *)
    reg signed [SCORE_DMA_W-1:0] buf_score [0:NODE_MAX-1];

    // ----------------------------------------------------------------
    // Latched control snapshot (taken at walk_done).
    // ----------------------------------------------------------------
    reg [AXI_ADDR_W-1:0] tree_base_r;
    reg [NODE_W-1:0]     n_nodes_r;

    // ----------------------------------------------------------------
    // FSM
    // ----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE  = 3'd0,
        S_AWREQ = 3'd1,   // assert AWVALID for the current sub-burst
        S_WDATA = 3'd2,   // stream W beats, WLAST on final
        S_BRESP = 3'd3,   // wait BVALID, then advance sub-burst / region / finish
        S_DONE  = 3'd4    // parked; AWVALID/WVALID low forever (silent)
    } state_e;
    state_e state;

    reg [2:0]        region;    // 0..N_TENSOR-1 : which tensor (SoA burst; 3b for 5 tensors)
    reg [BEAT_W-1:0] beat_idx;  // region-relative node index currently on the bus
    reg [BEAT_W-1:0] sub_base;  // first region-beat of the current sub-burst
    reg [BEAT_W:0]   sub_total; // beats in the current sub-burst (<=MAX_SUB)

    // ---- per-region beat count (all regions are n_nodes beats) + base ----
    wire [BEAT_W:0]       region_beats_total = {1'b0, n_nodes_r};
    // tensor k base = tree_base + k*TREE_STRIDE*BYTES_PER_BEAT
    wire [AXI_ADDR_W-1:0] cur_region_base =
        tree_base_r + AXI_ADDR_W'({29'd0, region}) * AXI_ADDR_W'(TREE_STRIDE) * BYTES_PER_BEAT;

    // ---- sub-burst helpers (mirror output_dma_writer's arlen split) ----
    wire [BEAT_W:0]  sub_remaining = region_beats_total - {1'b0, sub_base};
    wire             sub_is_capped = (sub_remaining > (BEAT_W+1)'(MAX_SUB));
    wire [BEAT_W:0]  sub_len       = sub_is_capped ? (BEAT_W+1)'(MAX_SUB) : sub_remaining;
    wire [7:0]       sub_awlen     = sub_len[7:0] - 8'd1;   // beats-1 (sub_len<=256)
    wire [AXI_ADDR_W-1:0] sub_addr = cur_region_base +
                                     (AXI_ADDR_W'(sub_base) * BYTES_PER_BEAT);

    // ---- current / next AXI word for (region, beat) — comb, int32 ----
    wire [BEAT_W-1:0] beat_p1 = beat_idx + (BEAT_W)'(1);
    reg signed [AXI_DATA_W-1:0] cur_word;
    reg signed [AXI_DATA_W-1:0] nxt_word;

    function automatic [AXI_DATA_W-1:0] tensor_word (input [2:0] r, input [BEAT_W-1:0] node);
        case (r)
            3'd0:    tensor_word = {{(AXI_DATA_W-TOK_W){1'b0}}, buf_tok[node]};          // draft_token (zero-ext)
            3'd1:    tensor_word = {{(AXI_DATA_W-NODE_W){1'b0}}, node[NODE_W-1:0]};       // retrive_index = identity CONSTANT
            3'd2:    tensor_word = $signed(buf_ntok[node]);                               // retrive_next_token (sign-ext, -1)
            3'd3:    tensor_word = $signed(buf_nsib[node]);                               // retrive_next_sibling (sign-ext, -1)
            default: tensor_word = buf_score[node];                                       // COSTDRAFTTREE score, int32 Q12.20 log2
        endcase
    endfunction

    always_comb begin
        cur_word = tensor_word(region, beat_idx);
        nxt_word = tensor_word(region, beat_p1);
    end

    // ---- B5 stall-progress detector (pure fn of state + AXI signals) ----
    reg waiting, progress;
    always_comb begin
        waiting  = 1'b0;
        progress = 1'b0;
        case (state)
            S_AWREQ: begin waiting = m_axi_out_awvalid; progress = m_axi_out_awvalid && m_axi_out_awready; end
            S_WDATA: begin waiting = m_axi_out_wvalid;  progress = m_axi_out_wvalid  && m_axi_out_wready;  end
            S_BRESP: begin waiting = m_axi_out_bready;  progress = m_axi_out_bready  && m_axi_out_bvalid;  end
            default: begin waiting = 1'b0;              progress = 1'b0;                                   end
        endcase
    end
    localparam int STALL_W = $clog2(STALL_TIMEOUT + 1) + 1;
    reg [STALL_W-1:0] stall_cnt;
    wire stall_timeout = waiting && !progress && (stall_cnt >= STALL_TIMEOUT[STALL_W-1:0]);

    integer i;

    // ---- emit-stream capture (single writer on the buffers) ----
    always_ff @(posedge clk) begin
        if (rst_n && emit_valid) begin
            buf_tok [emit_idx] <= emit_token;
            buf_ntok[emit_idx] <= emit_next_tok;
            buf_nsib[emit_idx] <= emit_next_sib;
            // narrow Q12.28 -> Q12.20 by taking the top 32 bits (drop 8 LSB frac).
            buf_score[emit_idx] <= emit_score[SCORE_W-1 -: SCORE_DMA_W];
        end
    end

    // ---- burst FSM (single writer on every AXI/control reg) ----
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
            tree_base_r       <= '0;
            n_nodes_r         <= '0;
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
        end else begin
            // stall-counter maintenance (before the FSM case)
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
                    if (walk_done && (n_nodes != '0)) begin
                        tree_base_r <= tree_base;
                        n_nodes_r   <= n_nodes;
                        region      <= 3'd0;
                        beat_idx    <= '0;
                        sub_base    <= '0;
                        out_busy    <= 1'b1;
                        out_done    <= 1'b0;
                        out_err     <= 1'b0;
                        stall_cnt   <= '0;
                        state       <= S_AWREQ;
                    end
                end

                // --------------------------------------------------------
                S_AWREQ: begin
                    if (stall_timeout) begin
                        m_axi_out_awvalid <= 1'b0;
                        m_axi_out_wvalid  <= 1'b0;
                        m_axi_out_wlast   <= 1'b0;
                        m_axi_out_bready  <= 1'b0;
                        out_busy          <= 1'b0;
                        out_done          <= 1'b1;
                        out_err           <= 1'b1;
                        state             <= S_DONE;
                    end else if (!m_axi_out_awvalid) begin
                        sub_total         <= sub_len;
                        m_axi_out_awid    <= '0;
                        m_axi_out_awaddr  <= sub_addr;
                        m_axi_out_awlen   <= sub_awlen;
                        m_axi_out_awsize  <= AWSIZE_32;
                        m_axi_out_awburst <= BURST_INCR;
                        m_axi_out_awvalid <= 1'b1;
                        beat_idx          <= sub_base;
                    end else if (m_axi_out_awready) begin
                        // AW accepted — present the first W beat of this sub-burst.
                        m_axi_out_awvalid <= 1'b0;
                        m_axi_out_wvalid  <= 1'b1;
                        m_axi_out_wstrb   <= {(AXI_DATA_W/8){1'b1}};
                        m_axi_out_wdata   <= cur_word;
                        m_axi_out_wlast   <= (sub_total == (BEAT_W+1)'(1)); // single-beat sub-burst
                        state             <= S_WDATA;
                    end
                end

                // --------------------------------------------------------
                S_WDATA: begin
                    if (stall_timeout) begin
                        m_axi_out_awvalid <= 1'b0;
                        m_axi_out_wvalid  <= 1'b0;
                        m_axi_out_wlast   <= 1'b0;
                        m_axi_out_bready  <= 1'b0;
                        out_busy          <= 1'b0;
                        out_done          <= 1'b1;
                        out_err           <= 1'b1;
                        state             <= S_DONE;
                    end else if (m_axi_out_wvalid && m_axi_out_wready) begin
                        if (m_axi_out_wlast) begin
                            m_axi_out_wvalid <= 1'b0;
                            m_axi_out_wlast  <= 1'b0;
                            m_axi_out_bready <= 1'b1;
                            state            <= S_BRESP;
                        end else begin
                            // present next beat; WLAST when beat_idx+1 is the last
                            // word of this sub-burst: (beat_idx+2)==(sub_base+sub_total).
                            m_axi_out_wdata <= nxt_word;
                            m_axi_out_wlast <= (({1'b0, beat_idx} + (BEAT_W+1)'(2))
                                               == ({1'b0, sub_base} + sub_total));
                            beat_idx        <= beat_idx + (BEAT_W)'(1);
                        end
                    end
                end

                // --------------------------------------------------------
                S_BRESP: begin
                    if (stall_timeout) begin
                        m_axi_out_awvalid <= 1'b0;
                        m_axi_out_wvalid  <= 1'b0;
                        m_axi_out_wlast   <= 1'b0;
                        m_axi_out_bready  <= 1'b0;
                        out_busy          <= 1'b0;
                        out_done          <= 1'b1;
                        out_err           <= 1'b1;
                        state             <= S_DONE;
                    end else begin
                        m_axi_out_bready <= 1'b1;
                        if (m_axi_out_bvalid) begin
                            m_axi_out_bready <= 1'b0;
                            if (({1'b0, sub_base} + sub_total) < region_beats_total) begin
                                // more sub-bursts in THIS tensor region
                                sub_base <= sub_base + sub_total[BEAT_W-1:0];
                                state    <= S_AWREQ;
                            end else if ({1'b0, region} + 3'd1 < N_TENSOR[2:0]) begin
                                // next tensor region
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
                S_DONE: begin
                    m_axi_out_awvalid <= 1'b0;
                    m_axi_out_wvalid  <= 1'b0;
                    m_axi_out_wlast   <= 1'b0;
                    m_axi_out_bready  <= 1'b0;
                    out_busy          <= 1'b0;
                    out_done          <= 1'b1;
                    // out_err held sticky until re-launch clears it.
                    // Re-arm directly on the next walk_done (mirrors
                    // output_dma_writer's S_DONE pairing-fix — a single trigger
                    // pulse must not be lost bouncing through S_IDLE).
                    if (walk_done && (n_nodes != '0)) begin
                        tree_base_r <= tree_base;
                        n_nodes_r   <= n_nodes;
                        region      <= 3'd0;
                        beat_idx    <= '0;
                        sub_base    <= '0;
                        out_busy    <= 1'b1;
                        out_done    <= 1'b0;
                        out_err     <= 1'b0;
                        stall_cnt   <= '0;
                        state       <= S_AWREQ;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    // retrive_index identity claim is load-bearing (emitted as a constant): assert
    // that every buffered node index is in range so the constant map is valid.
    always_ff @(posedge clk)
        if (rst_n && emit_valid)
            assert (emit_idx < NODE_W'(NODE_MAX))
                else $error("tree_dma_writer: emit_idx %0d >= NODE_MAX %0d", emit_idx, NODE_MAX);
`endif

endmodule

`default_nettype wire
