// =====================================================================
// gemv_stage_array_scaled_slr.sv — SLR-aware scaled GEMV array
// =====================================================================
// Replicates activation/control per SLR group and instantiates one weight/scale
// bank and integer GEMV lane per output column. SLR_SAFE=0 is passthrough;
// SLR_SAFE=1 registers forward broadcast and return values.
// PREFETCH_COL sequences contiguous columns when enabled. DUAL_PORT reads even
// and odd chunks together; otherwise the single-port lane is used. BATCH_B>1
// reuses one weight/scale read across sibling activation columns. FF_ACT_CACHE
// adds a hub plus per-group activation register stage. WLAT delays lane
// weight/valid/tag alignment for the registered activation seam.
// All arithmetic is signed integer MAC plus fixed-point scale multiplication.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

// Shared per-stage chunk counts and bank geometry.
`include "stage_geometry.svh"

(* keep_hierarchy = "yes" *)
module gemv_stage_array_scaled_slr #(
    parameter int K           = 64,
    parameter int K_PAR       = 16,
    parameter int LANES       = 4,
    parameter int A_WIDTH     = 16,
    parameter int GROUP       = 128,
    parameter int SCALE_WIDTH = 16,
    parameter int SCALE_FRAC  = 13,         // Q3.13 production
    parameter int RES_WIDTH   = 48,
    // BATCH_B=1 compiles sibling datapaths away. Larger batches reuse one
    // weight/scale read and are restricted by the elaboration guard below.
    parameter int BATCH_B     = 1,
    // ---- SLR-safe controls ----
    parameter bit SLR_SAFE    = 1'b0,       // 0 = passthrough, 1 = registered seams
    parameter int N_GROUPS    = 3,          // = #SLRs the array spans
    parameter int PIPE_STAGES = 1,          // registered hops per crossing
    // FF_ACT_CACHE adds a hub and per-group activation register stage.
    // The extra forward stage is reflected in index lookahead and return depth.
    parameter bit FF_ACT_CACHE = 1'b0,
    // Optional self-sequencing of contiguous output tiles.
    parameter bit PREFETCH_COL = 1'b0,
    // Optional second weight read and MAC path for two chunks per cycle.
    parameter bit DUAL_PORT = 1'b0,
    // WLAT=1 delays weight and MAC-valid by one cycle for a registered
    // activation seam; pair it with the corresponding seam latency.
    parameter bit WLAT = 1'b0,
    // ---- derived ----
    parameter int N_CHUNKS    = K / K_PAR,
    parameter int CW          = (N_CHUNKS <= 1) ? 1 : $clog2(N_CHUNKS + 2),
    parameter int N_SGROUPS   = (K + GROUP - 1) / GROUP,
    parameter int GW          = (N_SGROUPS <= 1) ? 1 : $clog2(N_SGROUPS + 1),
    // Per-lane capacity follows GEOM_LANES; small tests can override geometry
    // independently of instantiated lane count. At 128 lanes, all stages need
    // 97,440 weight words and 12,180 scale entries per lane.
    parameter int GEOM_LANES  = LANES,
    parameter int BANK_DEPTH  = (96 * 1024 * 128) / GEOM_LANES,
    parameter int SCALE_DEPTH = (16 * 1024 * 128) / GEOM_LANES,
    parameter int LANES_PER_GRP = (LANES + N_GROUPS - 1) / N_GROUPS,
    // FF_ACT_CACHE forward seam = 2 deep (hub + per-group local); the return
    // seam is registered the SAME total depth so the round-trip stays symmetric.
    // When FF_ACT_CACHE=0 the return depth is the legacy PIPE_STAGES.
    parameter int RET_STAGES  = (SLR_SAFE && FF_ACT_CACHE) ? 2 : PIPE_STAGES,
    // # registered hops on the slowest in+out round trip (for callers/dwell)
    parameter int RT_DELAY    = SLR_SAFE ? (FF_ACT_CACHE ? 4 : (2 * PIPE_STAGES)) : 0,
    // Extra sibling columns; reserve one unused slot when BATCH_B==1.
    parameter int BB1         = (BATCH_B > 1) ? (BATCH_B - 1) : 1
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,

    // Activation broadcast (shared across all lanes) — registered when SLR_SAFE
    input  wire signed [K_PAR*A_WIDTH-1:0]   a_chunk,
    // Odd-chunk activation for DUAL_PORT; fetched using chunk_idx_b_out.
    input  wire signed [K_PAR*A_WIDTH-1:0]   a_chunk_b,
    // BW-4 sibling-batch activation columns (packed BB1 sets of one chunk each).
    // Fed off the SAME chunk index as a_chunk (all lanes lockstep). Inert when
    // BATCH_B==1 (BB1==1, unused). Leave UNCONNECTED at BATCH_B==1.
    input  wire signed [BB1*K_PAR*A_WIDTH-1:0] a_chunk_batch,
    input  wire        [1:0]                 qz_raw,

    // Stage selector + tile offset for time-multiplexed N>LANES
    input  wire        [3:0]                 stage_sel,
    input  wire        [15:0]                tile_idx,

    // ---- column double-buffer (PREFETCH_COL=1 only; ignored when 0) ----
    // On `start`, self-sequence n_cols contiguous output columns beginning at
    // `tile_idx` (tile_idx, tile_idx+1, ..., tile_idx+n_cols-1). n_cols==0 is
    // treated as 1 (single column = the legacy behavior). Each column's result
    // is emitted on `col_done` (1-cyc pulse) tagged with its `col_done_idx`;
    // `cols_done` pulses once after the LAST column completes.
    input  wire        [15:0]                n_cols,

    // ---- URAM weight load (one shared write port) ----
    input  wire        [15:0]                lane_we_sel,
    input  wire                              we,
    input  wire        [16:0]                wr_addr,
    input  wire        [31:0]                wr_data,

    // ---- BRAM scale load (one shared write port) ----
    input  wire        [15:0]                sc_lane_sel,
    input  wire                              sc_we,
    input  wire        [16:0]                sc_wr_addr,
    input  wire        [SCALE_WIDTH-1:0]     sc_wr_data,

    // Lane status — chunk_idx_out from lane 0 (all lockstep)
    output wire        [CW-1:0]              chunk_idx_out,
    // Odd-chunk index in dual-port mode; mirrors chunk_idx_out otherwise.
    output wire        [CW-1:0]              chunk_idx_b_out,
    output wire                              chunk_idx_valid,

    // Per-lane outputs (registered-return aligned when SLR_SAFE)
    output wire signed [32*LANES-1:0]        acc_out_per_lane,      // unscaled INT32 (debug)
    output wire signed [RES_WIDTH*LANES-1:0] result_q_per_lane,     // scaled fixed-point (b0)
    // BW-4 sibling-batch results: BB1 sets of RES_WIDTH*LANES, one per sibling
    // column, return-seam-aligned with result_q_per_lane. '0 when BATCH_B==1.
    output wire signed [BB1*RES_WIDTH*LANES-1:0] result_q_batch,
    output wire                              done,

    // ---- column double-buffer status ----
    // col_done : 1-cyc pulse as EACH column's result_q_per_lane/acc_out_per_lane
    //            becomes valid (== `done` when PREFETCH_COL=0).
    // col_done_idx : the tile_idx (output column) the just-finished col_done
    //            belongs to (== tile_idx when PREFETCH_COL=0).
    // cols_done : 1-cyc pulse after the LAST self-sequenced column completes
    //            (== `done` when PREFETCH_COL=0).
    output wire                              col_done,
    output wire        [15:0]                col_done_idx,
    output wire                              cols_done
);

    // =================================================================
    // COLUMN DOUBLE-BUFFER SEQUENCER (PREFETCH_COL=1 only).
    //   Produces the EFFECTIVE start (eff_start) + tile_idx (eff_tile_idx) the
    //   forward seam + banks consume. PREFETCH_COL=0 -> straight passthrough
    //   (eff_start=start, eff_tile_idx=tile_idx) => bit+cycle identical.
    //
    //   The `done` used here is the wrapper's aggregate lane `done` AFTER the
    //   return seam (declared below as `done`); it is referenced ahead of its
    //   `assign` (legal — `done` is an output `wire`). One column completes on
    //   each `done` pulse. On that pulse the sequencer:
    //     - emits col_done (with col_done_idx = the column just finished),
    //     - if more columns remain, re-pulses eff_start for the next column.
    //   The next column's eff_tile_idx is advanced as soon as the CURRENT
    //   column leaves S_RUN (lanes stop reading weights), so the registered
    //   bank col_offset_acc has settled to col+1 BEFORE eff_start re-fires —
    //   removing the 1-cycle settle the integrated FSM otherwise inserts.
    // =================================================================
    wire        eff_start;
    wire [15:0] eff_tile_idx;
    // NOTE: `done` and `chunk_idx_valid` are module OUTPUT wires (port list).
    // The sequencer references them ahead of their `assign` below — legal for
    // a wire. `done` (post-return-seam aggregate) marks a column complete;
    // `chunk_idx_valid` (= lane-0 S_RUN) gives the S_RUN->drain edge that gates
    // the prefetch tile_idx advance.

    generate
    if (!PREFETCH_COL) begin : g_seq_off
        // ---- passthrough: exact legacy single-column behavior ----
        assign eff_start    = start;
        assign eff_tile_idx = tile_idx;
        assign col_done     = done;
        assign col_done_idx = tile_idx;
        assign cols_done    = done;
    end else begin : g_seq_on
        // n_cols==0 -> 1 (a caller that leaves it unconnected gets single col).
        wire [15:0] ncols_eff = (n_cols == 16'd0) ? 16'd1 : n_cols;

        // Column-sequencer FSM.
        //   CS_IDLE  : wait for the caller `start`. Latch base tile_idx, ncols,
        //              issue eff_start for column 0.
        //   CS_RUN   : a column is in flight. When it has entered its drain
        //              (was in S_RUN, now chunk_idx_valid fell) and more columns
        //              remain, advance the prefetch tile_idx (pre-settle the bank
        //              col_offset_acc for col+1). When `done` pulses, that column
        //              is complete: emit col_done; if more remain re-issue
        //              eff_start (col_offset_acc already settled), else finish.
        typedef enum logic [1:0] { CS_IDLE, CS_RUN, CS_DONE } cs_t;
        cs_t cs, cs_n;

        reg  [15:0] base_tile;       // tile_idx of column 0 (latched at start)
        reg  [15:0] ncols_q;         // latched ncols
        reg  [15:0] col_ptr;         // index of the column currently IN FLIGHT
        reg  [15:0] pf_tile;         // tile_idx currently driven to the banks
        reg         pf_advanced;     // pf_tile already advanced for this column?
        reg         seq_start_r;     // registered eff_start pulse
        reg  [15:0] done_idx_r;      // tile_idx tag for the col_done about to fire
        reg         col_done_r;
        reg         cols_done_r;

        // "lanes left S_RUN this cycle" = chunk_idx_valid 1->0 edge.
        reg  cidx_valid_d;
        always_ff @(posedge clk) begin
            if (!rst_n) cidx_valid_d <= 1'b0;
            else        cidx_valid_d <= chunk_idx_valid;
        end
        wire run_to_drain = cidx_valid_d && !chunk_idx_valid;

        // more columns to launch AFTER the in-flight one?
        wire more_cols = (col_ptr + 16'd1) < ncols_q;

        always_comb begin
            cs_n = cs;
            case (cs)
                CS_IDLE: if (start) cs_n = CS_RUN;
                CS_RUN:  if (done && !more_cols) cs_n = CS_DONE;
                CS_DONE: cs_n = CS_IDLE;
                default: cs_n = CS_IDLE;
            endcase
        end

        always_ff @(posedge clk) begin
            if (!rst_n) begin
                cs <= CS_IDLE;
                base_tile <= '0; ncols_q <= 16'd1; col_ptr <= '0;
                pf_tile <= '0; pf_advanced <= 1'b0;
                seq_start_r <= 1'b0; done_idx_r <= '0;
                col_done_r <= 1'b0; cols_done_r <= 1'b0;
            end else begin
                cs <= cs_n;
                // defaults (single-cycle pulses)
                seq_start_r <= 1'b0;
                col_done_r  <= 1'b0;
                cols_done_r <= 1'b0;

                case (cs)
                    CS_IDLE: if (start) begin
                        base_tile   <= tile_idx;
                        ncols_q     <= ncols_eff;
                        col_ptr     <= 16'd0;
                        pf_tile     <= tile_idx;     // bank sees col 0
                        pf_advanced <= 1'b0;
                        seq_start_r <= 1'b1;         // launch col 0 (1-cyc settle
                                                     //  already covered by the
                                                     //  caller's stage_sel/tile
                                                     //  setup, same as legacy)
                        done_idx_r  <= tile_idx;
                    end
                    CS_RUN: begin
                        // Prefetch: once the in-flight column has entered drain
                        // and another column remains, advance the bank tile_idx
                        // so col_offset_acc settles to col+1 during the dead
                        // window. Guard with pf_advanced so it happens once.
                        if (run_to_drain && more_cols && !pf_advanced) begin
                            pf_tile     <= base_tile + col_ptr + 16'd1;
                            pf_advanced <= 1'b1;
                        end
                        // Column complete on `done`.
                        if (done) begin
                            col_done_r <= 1'b1;
                            done_idx_r <= base_tile + col_ptr;   // tag finished col
                            if (more_cols) begin
                                // launch next column: col_offset_acc already at
                                // col+1 (prefetched), so eff_start fires now with
                                // no extra settle cycle.
                                col_ptr     <= col_ptr + 16'd1;
                                seq_start_r <= 1'b1;
                                pf_advanced <= 1'b0;
                            end else begin
                                cols_done_r <= 1'b1;             // last column
                            end
                        end
                    end
                    CS_DONE: ; // one-cycle return to idle
                    default: ;
                endcase
            end
        end

        assign eff_start    = seq_start_r;
        assign eff_tile_idx = pf_tile;
        assign col_done     = col_done_r;
        assign col_done_idx = done_idx_r;
        assign cols_done    = cols_done_r;
    end
    endgenerate

    // =================================================================
    // FORWARD BROADCAST SEAM (per group): a_chunk + qz + stage + tile + start.
    // SLR_SAFE=0 -> g_bcast_through (combinational); SLR_SAFE=1 -> g_bcast_pipe
    // (PIPE_STAGES-deep registered replica per group, keep/dont_touch).
    // NOTE: driven from the EFFECTIVE start/tile_idx (eff_*) so the column
    // double-buffer (PREFETCH_COL=1) is transparent to the seam; with
    // PREFETCH_COL=0 eff_start==start and eff_tile_idx==tile_idx.
    // =================================================================
    wire signed [K_PAR*A_WIDTH-1:0] a_g    [0:N_GROUPS-1];
    wire signed [K_PAR*A_WIDTH-1:0] ab_g   [0:N_GROUPS-1];   // WO-025 odd-chunk activation
    wire        [1:0]               qz_g   [0:N_GROUPS-1];
    wire        [3:0]               st_g   [0:N_GROUPS-1];
    wire        [15:0]              tl_g   [0:N_GROUPS-1];
    wire                            sr_g   [0:N_GROUPS-1];
    // BW-4: per-group sibling-batch activation, forward-seam-aligned with a_g.
    wire signed [BB1*K_PAR*A_WIDTH-1:0] a_batch_g [0:N_GROUPS-1];
    // BW-4: per-lane sibling results (before the return seam), one per sibling.
    wire signed [RES_WIDTH-1:0]         lane_res_batch [0:LANES-1][0:BB1-1];

    genvar gg;
    generate
        if (!SLR_SAFE) begin : g_bcast_through
            for (gg = 0; gg < N_GROUPS; gg = gg + 1) begin : g_pt
                assign a_g[gg]  = a_chunk;
                assign ab_g[gg] = a_chunk_b;
                assign qz_g[gg] = qz_raw;
                assign st_g[gg] = stage_sel;
                assign tl_g[gg] = eff_tile_idx;
                assign sr_g[gg] = eff_start;
            end
        end else if (!FF_ACT_CACHE) begin : g_bcast_pipe
            // Per-group broadcast registers preserve control/data alignment.
            // keep retains the explicit group copies; max_fanout permits
            // further physical replication. Controls have tighter fanout
            // bounds than the wider activation buses.
            (* keep = "true", max_fanout = 64 *)
            reg signed [K_PAR*A_WIDTH-1:0] a_chunk_p   [0:N_GROUPS-1][0:PIPE_STAGES-1];
            (* keep = "true", max_fanout = 64 *)
            reg signed [K_PAR*A_WIDTH-1:0] a_chunk_b_p [0:N_GROUPS-1][0:PIPE_STAGES-1];
            (* keep = "true", max_fanout = 32 *)
            reg        [1:0]               qz_raw_p    [0:N_GROUPS-1][0:PIPE_STAGES-1];
            (* keep = "true", max_fanout = 32 *)
            reg        [3:0]               stage_sel_p [0:N_GROUPS-1][0:PIPE_STAGES-1];
            (* keep = "true", max_fanout = 32 *)
            reg        [15:0]              tile_idx_p  [0:N_GROUPS-1][0:PIPE_STAGES-1];
            (* keep = "true", max_fanout = 32 *)
            reg                            start_p     [0:N_GROUPS-1][0:PIPE_STAGES-1];

            genvar ss;
            for (gg = 0; gg < N_GROUPS; gg = gg + 1) begin : g_grp_pipe
                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        a_chunk_p[gg][0]   <= '0; a_chunk_b_p[gg][0] <= '0;
                        qz_raw_p[gg][0]   <= '0;
                        stage_sel_p[gg][0] <= '0; tile_idx_p[gg][0] <= '0;
                        start_p[gg][0]     <= 1'b0;
                    end else begin
                        a_chunk_p[gg][0]   <= a_chunk;
                        a_chunk_b_p[gg][0] <= a_chunk_b;
                        qz_raw_p[gg][0]    <= qz_raw;
                        stage_sel_p[gg][0] <= stage_sel;
                        tile_idx_p[gg][0]  <= eff_tile_idx;
                        start_p[gg][0]     <= eff_start;
                    end
                end
                for (ss = 1; ss < PIPE_STAGES; ss = ss + 1) begin : g_extra
                    always_ff @(posedge clk) begin
                        if (!rst_n) begin
                            a_chunk_p[gg][ss]   <= '0; a_chunk_b_p[gg][ss] <= '0;
                            qz_raw_p[gg][ss]   <= '0;
                            stage_sel_p[gg][ss] <= '0; tile_idx_p[gg][ss] <= '0;
                            start_p[gg][ss]     <= 1'b0;
                        end else begin
                            a_chunk_p[gg][ss]   <= a_chunk_p[gg][ss-1];
                            a_chunk_b_p[gg][ss] <= a_chunk_b_p[gg][ss-1];
                            qz_raw_p[gg][ss]    <= qz_raw_p[gg][ss-1];
                            stage_sel_p[gg][ss] <= stage_sel_p[gg][ss-1];
                            tile_idx_p[gg][ss]  <= tile_idx_p[gg][ss-1];
                            start_p[gg][ss]     <= start_p[gg][ss-1];
                        end
                    end
                end
                assign a_g[gg]  = a_chunk_p[gg][PIPE_STAGES-1];
                assign ab_g[gg] = a_chunk_b_p[gg][PIPE_STAGES-1];
                assign qz_g[gg] = qz_raw_p[gg][PIPE_STAGES-1];
                assign st_g[gg] = stage_sel_p[gg][PIPE_STAGES-1];
                assign tl_g[gg] = tile_idx_p[gg][PIPE_STAGES-1];
                assign sr_g[gg] = start_p[gg][PIPE_STAGES-1];
            end
        end else begin : g_bcast_ffcache
            // =============================================================
            // FF_ACT_CACHE=1 — per-SLR FF activation cache (the FUNDAMENTAL fix).
            //   STAGE 0 : ONE SHARED HUB register (single replica of the whole
            //             broadcast bus; pin to SLR1 via the .xdc). This is the
            //             driver every per-group local FF reads from.
            //   STAGE 1 : PER-GROUP local replica (one per SLR). Each group's
            //             lanes read THIS local FF (a_g/ab_g/qz_g/st_g/tl_g/sr_g),
            //             so within a group there is NO cross-SLR read.
            //   The ONLY cross-SLR arc is hub-FF (SLR1) -> per-group-FF (SLRk):
            //   a clean point-to-point FF->FF hop (registered SLL), not a global
            //   combinational fanout to 3 SLRs.
            //   Forward latency = 2 (hub + local). The chunk-index lookahead
            //   (g_idx_ffcache_lead2 below) leads by 2 so the consumer's
            //   combinational a_chunk fetch, after this +2 seam, re-aligns with
            //   weight[c] at the lane (bit-exact to PIPE_STAGES=1, +1 latency).
            // =============================================================
            // ---- STAGE 0 : shared hub (single replica) ----
            (* keep = "true", dont_touch = "true" *)
            reg signed [K_PAR*A_WIDTH-1:0] a_hub;
            (* keep = "true", dont_touch = "true" *)
            reg signed [K_PAR*A_WIDTH-1:0] a_b_hub;
            (* keep = "true", dont_touch = "true" *)
            reg        [1:0]               qz_hub;
            (* keep = "true", dont_touch = "true" *)
            reg        [3:0]               st_hub;
            (* keep = "true", dont_touch = "true" *)
            reg        [15:0]              tl_hub;
            (* keep = "true", dont_touch = "true" *)
            reg                            sr_hub;
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    a_hub <= '0; a_b_hub <= '0; qz_hub <= '0;
                    st_hub <= '0; tl_hub <= '0; sr_hub <= 1'b0;
                end else begin
                    a_hub   <= a_chunk;
                    a_b_hub <= a_chunk_b;
                    qz_hub  <= qz_raw;
                    st_hub  <= stage_sel;
                    tl_hub  <= eff_tile_idx;
                    sr_hub  <= eff_start;
                end
            end
            // ---- STAGE 1 : per-group local replica (one FF per SLR) ----
            (* keep = "true", dont_touch = "true" *)
            reg signed [K_PAR*A_WIDTH-1:0] a_loc   [0:N_GROUPS-1];
            (* keep = "true", dont_touch = "true" *)
            reg signed [K_PAR*A_WIDTH-1:0] a_b_loc [0:N_GROUPS-1];
            (* keep = "true", dont_touch = "true" *)
            reg        [1:0]               qz_loc  [0:N_GROUPS-1];
            (* keep = "true", dont_touch = "true" *)
            reg        [3:0]               st_loc  [0:N_GROUPS-1];
            (* keep = "true", dont_touch = "true" *)
            reg        [15:0]              tl_loc  [0:N_GROUPS-1];
            (* keep = "true", dont_touch = "true" *)
            reg                            sr_loc  [0:N_GROUPS-1];
            for (gg = 0; gg < N_GROUPS; gg = gg + 1) begin : g_grp_ffcache
                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        a_loc[gg]   <= '0; a_b_loc[gg] <= '0; qz_loc[gg] <= '0;
                        st_loc[gg]  <= '0; tl_loc[gg]  <= '0; sr_loc[gg] <= 1'b0;
                    end else begin
                        a_loc[gg]   <= a_hub;
                        a_b_loc[gg] <= a_b_hub;
                        qz_loc[gg]  <= qz_hub;
                        st_loc[gg]  <= st_hub;
                        tl_loc[gg]  <= tl_hub;
                        sr_loc[gg]  <= sr_hub;
                    end
                end
                assign a_g[gg]  = a_loc[gg];
                assign ab_g[gg] = a_b_loc[gg];
                assign qz_g[gg] = qz_loc[gg];
                assign st_g[gg] = st_loc[gg];
                assign tl_g[gg] = tl_loc[gg];
                assign sr_g[gg] = sr_loc[gg];
            end
        end
    endgenerate

    // =================================================================
    // BW-4 SIBLING-BATCH FORWARD SEAM. The B-1 sibling activation columns are
    // registered the SAME PIPE_STAGES depth as a_chunk (g_bcast_pipe) so they
    // reach the sibling lanes exactly when a_chunk reaches the b0 lane -> every
    // sibling MACs a_chunk_batch[c] against weight[c], byte-aligned with b0.
    // BATCH_B>1 is asserted (below) to the SLR_SAFE=1 / FF_ACT_CACHE=0 /
    // DUAL_PORT=0 production config, so PIPE_STAGES-deep is the correct depth.
    // BATCH_B==1: a_batch_g tied '0, ZERO extra logic (byte-identical array).
    // =================================================================
    generate
        if (BATCH_B > 1) begin : g_fwd_batch
            (* keep = "true", dont_touch = "true" *)
            reg signed [BB1*K_PAR*A_WIDTH-1:0] a_batch_p [0:N_GROUPS-1][0:PIPE_STAGES-1];
            genvar gb, sb;
            for (gb = 0; gb < N_GROUPS; gb = gb + 1) begin : g_bp
                always_ff @(posedge clk) begin
                    if (!rst_n) a_batch_p[gb][0] <= '0;
                    else        a_batch_p[gb][0] <= a_chunk_batch;
                end
                for (sb = 1; sb < PIPE_STAGES; sb = sb + 1) begin : g_bpx
                    always_ff @(posedge clk) begin
                        if (!rst_n) a_batch_p[gb][sb] <= '0;
                        else        a_batch_p[gb][sb] <= a_batch_p[gb][sb-1];
                    end
                end
                assign a_batch_g[gb] = a_batch_p[gb][PIPE_STAGES-1];
            end
        end else begin : g_fwd_batch_off
            genvar gb0;
            for (gb0 = 0; gb0 < N_GROUPS; gb0 = gb0 + 1) begin : g_bp0
                assign a_batch_g[gb0] = '0;
            end
        end
    endgenerate

    // =================================================================
    // PER-GROUP LANE BLOCKS (each becomes a Pblock target = one SLR).
    // Reuses the bit-exact scaled per-lane datapath: weight_bank_multistage
    // URAM + scale_bram BRAM + gemv_lane_scaled_stream (SCALED=1). UNCHANGED.
    // =================================================================
    wire [LANES-1:0]            lane_done;
    wire [CW-1:0]               lane_chunk_idx      [0:LANES-1];
    wire [CW-1:0]               lane_chunk_idx_next [0:LANES-1];
    // WO-025 dual-port: odd-chunk (port B) current + next indices per lane.
    // For DUAL_PORT=0 these mirror the even index (port B unused).
    wire [CW-1:0]               lane_chunk_idx_b      [0:LANES-1];
    wire [CW-1:0]               lane_chunk_idx_b_next [0:LANES-1];
    wire                        lane_chunk_valid    [0:LANES-1];
    wire [GW-1:0]               lane_group_idx      [0:LANES-1];
    wire [GW-1:0]               lane_group_idx_next [0:LANES-1];
    wire signed [31:0]          lane_acc_raw        [0:LANES-1];
    wire signed [RES_WIDTH-1:0] lane_res_raw        [0:LANES-1];

    // ----------------------------------------------------------------
    // EXPOSED chunk index (group 0, lane 0 — all lanes lockstep).
    //
    // ACTIVATION-ALIGNMENT (the reason this differs from the plain array):
    //   The integrating consumer fetches the activation COMBINATIONALLY off
    //   chunk_idx_out (e.g. eagle4_drafter r5_rd_idx = array_chunk_idx_out for
    //   the SiLU->G7 seam) and drives it into a_chunk. When SLR_SAFE=1 the
    //   forward seam delays a_chunk by PIPE_STAGES before the lane consumes it.
    //   If chunk_idx_out exposed the lane's UNDELAYED counter c, the consumer
    //   would fetch a[c-PIPE] (it acts PIPE cycles before the lane reaches c)
    //   -> a[c-PIPE] would pair with weight[c] = a PIPE-cycle activation/weight
    //   SKEW. To cancel the forward seam, we expose an index that LEADS the
    //   lane by PIPE_STAGES, so the consumer's combinational fetch, after the
    //   +PIPE seam, arrives at the lane aligned with weight[c]. PIPE_STAGES=1
    //   (production) uses the lane's chunk_idx_next_out (= c+1) directly. The
    //   plain (SLR_SAFE=0) path exposes the lane counter unchanged.
    // ----------------------------------------------------------------
    generate
        if (!SLR_SAFE) begin : g_idx_plain
            assign chunk_idx_out   = lane_chunk_idx[0];
            assign chunk_idx_b_out = lane_chunk_idx_b[0];      // == even idx when DUAL_PORT=0
            assign chunk_idx_valid = lane_chunk_valid[0];
        end else if (FF_ACT_CACHE) begin : g_idx_ffcache_lead2
            // Two-stage forward seam: begin the activation walk at hub-start,
            // one cycle before the lane sees start. Expose index 0 there, then
            // advance each cycle for the latched per-stage chunk count.
            // This path supports the single-port lane only.
            reg           eff_start_d1;   // == sr_hub timing
            reg  [3:0]    stage_sel_d1;   // pre-seam stage_sel, hub-aligned
            always_ff @(posedge clk) begin
                if (!rst_n) begin eff_start_d1 <= 1'b0; stage_sel_d1 <= '0; end
                else        begin eff_start_d1 <= eff_start; stage_sel_d1 <= stage_sel; end
            end
            wire [16:0] lead_chunks_real = sg_w_chunks_per_col(stage_sel_d1);
            wire [CW-1:0] lead_nc =
                (lead_chunks_real >= 17'(N_CHUNKS)) ? CW'(N_CHUNKS)
                                                    : lead_chunks_real[CW-1:0];
            wire [CW-1:0] lead_nc_eff = (lead_nc == '0) ? CW'(N_CHUNKS) : lead_nc;
            reg  [CW-1:0] lead_nc_q;
            reg  [CW-1:0] lead_idx_q;     // NEXT lead index to expose (1..NC-1)
            reg           lead_run_q;     // walk active AFTER the hub-start cycle
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    lead_idx_q <= '0; lead_run_q <= 1'b0; lead_nc_q <= '0;
                end else if (eff_start_d1) begin
                    // hub-start cycle exposes idx 0 (combinationally below). Arm the
                    // registered walk to expose idx 1 NEXT cycle, 2 after, ... NC-1
                    // last. NC==1 -> nothing after idx 0.
                    lead_nc_q  <= lead_nc_eff;
                    lead_idx_q <= CW'(1);
                    lead_run_q <= (lead_nc_eff > CW'(1));
                end else if (lead_run_q) begin
                    if (lead_idx_q == lead_nc_q - 1'b1) begin
                        lead_run_q <= 1'b0;          // just exposing NC-1, then stop
                    end else begin
                        lead_idx_q <= lead_idx_q + 1'b1;
                    end
                end
            end
            // valid this cycle: the hub-start cycle (idx 0) OR while the registered
            // walk runs (idx 1..NC-1). idx 0 exposed combinationally on the hub-start
            // cycle so it is never double-emitted.
            wire        lead_valid_now = eff_start_d1 || lead_run_q;
            wire [CW-1:0] lead_idx_now = eff_start_d1 ? CW'(0) : lead_idx_q;
            assign chunk_idx_out   = lead_idx_now;
            assign chunk_idx_b_out = lead_idx_now;   // DUAL_PORT not used with FF_ACT_CACHE
            assign chunk_idx_valid = lead_valid_now;
        end else begin : g_idx_lookahead
            // PIPE_STAGES=1 production config: lead the consumer by 1 cycle so
            // its combinational a_chunk fetch, after the +1 forward seam, lands
            // at the lane aligned with weight[c]. The exposed INDEX leads by 1
            // (= chunk_idx_next_out, which is `c` while the lane counter is
            // `c-1`, and `0` the cycle the lane sees start). The exposed VALID
            // must cover the SAME N_CHUNKS-cycle window shifted 1 cycle early:
            //   - the S_IDLE+start cycle (lane about to run): chunk_idx_next=0,
            //     reconstructed as (sr_g[0] && !lane_chunk_valid[0]);
            //   - the S_RUN cycles for chunk_idx 0..N-2 (chunk_idx_next advances,
            //     i.e. next != cur); at chunk_idx==N-1 the lane's chunk_idx_n
            //     holds (next == cur) -> drop lead-valid so the consumer does
            //     not refetch a stale index.
            wire lead_start    = sr_g[0] && !lane_chunk_valid[0];
            wire lead_run      = lane_chunk_valid[0] &&
                                 (lane_chunk_idx_next[0] != lane_chunk_idx[0]);
            assign chunk_idx_out   = lane_chunk_idx_next[0];
            // Secondary activation uses the matching next-chunk lookahead.
            assign chunk_idx_b_out = lane_chunk_idx_b_next[0];
            assign chunk_idx_valid = lead_start || lead_run;
        end
    endgenerate
`ifndef SYNTHESIS
    initial if (SLR_SAFE && PIPE_STAGES != 1)
        $error("gemv_stage_array_scaled_slr: SLR_SAFE chunk_idx look-ahead supports PIPE_STAGES=1 only (got %0d).", PIPE_STAGES);
    // Cache lookahead and sibling batching have narrower supported configs.
    initial if (FF_ACT_CACHE && !SLR_SAFE)
        $error("gemv_stage_array_scaled_slr: FF_ACT_CACHE=1 requires SLR_SAFE=1 (it restructures the SLR-crossing forward/return seam).");
    initial if (FF_ACT_CACHE && DUAL_PORT)
        $error("gemv_stage_array_scaled_slr: FF_ACT_CACHE=1 lead-by-2 lookahead is implemented for the SINGLE-PORT lane only (live config is DUAL_PORT=0).");
    // BW-4: sibling-batch is implemented only for the production seam config.
    initial if (BATCH_B > 1 && (!SLR_SAFE || FF_ACT_CACHE || DUAL_PORT))
        $error("gemv_stage_array_scaled_slr: BATCH_B>1 requires SLR_SAFE=1, FF_ACT_CACHE=0, DUAL_PORT=0 (the production seam config; got SLR_SAFE=%0d FF_ACT_CACHE=%0d DUAL_PORT=%0d).", SLR_SAFE, FF_ACT_CACHE, DUAL_PORT);
`endif

    genvar grp, li;
    generate
        for (grp = 0; grp < N_GROUPS; grp = grp + 1) begin : g_grp
            localparam int BASE = grp * LANES_PER_GRP;
            localparam int CNT  = (BASE + LANES_PER_GRP <= LANES)
                                   ? LANES_PER_GRP : (LANES - BASE);

            // Use the same delayed stage selector for chunk count and bank
            // addressing. Reduced-K test configurations clamp to capacity.
            wire [16:0] grp_chunks_real = sg_w_chunks_per_col(st_g[grp]);
            wire [CW-1:0] grp_nc_run =
                (grp_chunks_real >= 17'(N_CHUNKS)) ? CW'(N_CHUNKS)
                                                   : grp_chunks_real[CW-1:0];

            for (li = 0; li < CNT; li = li + 1) begin : g_lane
                localparam int L = BASE + li;           // global lane index
                wire [15:0] col_offset = tl_g[grp][15:0];

                wire        lane_we    = we    && (lane_we_sel == L[15:0]);
                wire        sc_lane_we = sc_we && (sc_lane_sel == L[15:0]);

                if (!DUAL_PORT) begin : g_single
                    // ============================================================
                    // SINGLE-PORT lane (DEFAULT) — byte+cycle identical to the
                    // existing g6/perbank/cycle-measure contract. UNCHANGED idiom.
                    // ============================================================
                    wire [31:0] qw_lane;
                    weight_bank_multistage #(
                        .LANES(GEOM_LANES),  // LANES-256: geometry follows GEOM_LANES (= LANES in production)
                        .DEPTH(BANK_DEPTH), .DATA_W(32), .RD_PORTS(1)
                    ) u_bank (
                        .clk(clk), .rst_n(rst_n),
                        .we(lane_we), .wr_addr(wr_addr), .wr_data(wr_data),
                        .stage_sel(st_g[grp]),
                        .col_offset(col_offset),
                        .chunk_idx({{(16-CW){1'b0}}, lane_chunk_idx_next[L]}),
                        .rd_data(qw_lane),
                        .chunk_idx_b(16'd0), .rd_data_b()    // port B unused
                    );

                    wire [SCALE_WIDTH-1:0] scale_lane;
                    scale_bram #(
                        .LANES(GEOM_LANES),  // LANES-256: geometry follows GEOM_LANES (= LANES in production)
                        .DEPTH(SCALE_DEPTH), .SCALE_W(SCALE_WIDTH), .ADDR_W(17), .GW(7),
                        .GROUP(GROUP)
                    ) u_scale (
                        .clk(clk), .rst_n(rst_n),
                        .we(sc_lane_we), .wr_addr(sc_wr_addr), .wr_data(sc_wr_data),
                        .stage_sel(st_g[grp]),
                        .col_offset(col_offset),
                        .group_idx({{(7-GW){1'b0}}, lane_group_idx_next[L]}),
                        .rd_data(scale_lane)
                    );

                    gemv_lane_scaled_stream #(
                        .K(K), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                        .GROUP(GROUP), .SCALE_WIDTH(SCALE_WIDTH),
                        .SCALE_FRAC(SCALE_FRAC), .RES_WIDTH(RES_WIDTH), .SCALED(1'b1),
                        .WLAT(WLAT)                              // D1 fix: weight+enable +1cyc
                    ) u_lane (
                        .clk(clk), .rst_n(rst_n), .start(sr_g[grp]),
                        .a_chunk(a_g[grp]),
                        .qw_word(qw_lane),
                        .qz_raw(qz_g[grp]),
                        .scale_q($signed(scale_lane)),
                        .n_chunks_run(grp_nc_run),
                        .chunk_idx_out(lane_chunk_idx[L]),
                        .chunk_idx_valid(lane_chunk_valid[L]),
                        .chunk_idx_next_out(lane_chunk_idx_next[L]),
                        .group_idx_out(lane_group_idx[L]),
                        .group_idx_next_out(lane_group_idx_next[L]),
                        .acc_out(lane_acc_raw[L]),
                        .result_q(lane_res_raw[L]),
                        .done(lane_done[L])
                    );
                    // ========================================================
                    // BW-4 SIBLING-BATCH lanes (weight/scale REUSE). B-1 extra
                    // gemv_lane_scaled_stream instances share u_bank's ONE
                    // qw_lane read and u_scale's ONE scale_lane read (URAM/BRAM
                    // NEUTRAL); each MACs its OWN sibling activation column
                    // (a_batch_g[grp] slice) against the shared weight. All
                    // lanes get the SAME start/qz/n_chunks_run and run the
                    // IDENTICAL self-timed FSM in lockstep with u_lane, so the
                    // shared qw_lane/scale_lane (1-cyc lookahead off b0's
                    // chunk_idx_next/group_idx_next) is exactly the weight/scale
                    // each sibling needs at its current chunk/group — same
                    // alignment contract as b0. Cost = (B-1) x lane MAC/DSP/FF,
                    // URAM/BRAM +0. Compiled away at BATCH_B==1.
                    // ========================================================
                    if (BATCH_B > 1) begin : g_batch_lanes
                        genvar bb;
                        for (bb = 0; bb < BB1; bb = bb + 1) begin : g_bl
                            gemv_lane_scaled_stream #(
                                .K(K), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                                .GROUP(GROUP), .SCALE_WIDTH(SCALE_WIDTH),
                                .SCALE_FRAC(SCALE_FRAC), .RES_WIDTH(RES_WIDTH),
                                .SCALED(1'b1), .WLAT(WLAT)
                            ) u_lane_b (
                                .clk(clk), .rst_n(rst_n), .start(sr_g[grp]),
                                .a_chunk(a_batch_g[grp][bb*K_PAR*A_WIDTH +: K_PAR*A_WIDTH]),
                                .qw_word(qw_lane),          // SHARED weight read
                                .qz_raw(qz_g[grp]),
                                .scale_q($signed(scale_lane)), // SHARED scale read
                                .n_chunks_run(grp_nc_run),
                                .chunk_idx_out(), .chunk_idx_valid(),
                                .chunk_idx_next_out(),
                                .group_idx_out(), .group_idx_next_out(),
                                .acc_out(),
                                .result_q(lane_res_batch[L][bb]),
                                .done()
                            );
                        end
                    end
                    // port-B index unused -> mirror even so chunk_idx_b_out == even.
                    assign lane_chunk_idx_b[L]      = lane_chunk_idx[L];
                    assign lane_chunk_idx_b_next[L] = lane_chunk_idx_next[L];
                end else begin : g_dual
                    // ============================================================
                    // WO-025 DUAL-READ-PORT lane — 2 chunks/cycle (32 MAC/lane).
                    // Port A reads the EVEN chunk (lane_chunk_idx_next), port B
                    // the ODD chunk (lane_chunk_idx_b_next). Both reads share the
                    // SAME registered col_offset_acc (one output column).
                    // ============================================================
                    wire [31:0] qw_lane, qw_lane_b;
                    weight_bank_multistage #(
                        .LANES(GEOM_LANES),  // LANES-256: geometry follows GEOM_LANES (= LANES in production)
                        .DEPTH(BANK_DEPTH), .DATA_W(32), .RD_PORTS(2)
                    ) u_bank (
                        .clk(clk), .rst_n(rst_n),
                        .we(lane_we), .wr_addr(wr_addr), .wr_data(wr_data),
                        .stage_sel(st_g[grp]),
                        .col_offset(col_offset),
                        .chunk_idx  ({{(16-CW){1'b0}}, lane_chunk_idx_next[L]}),
                        .rd_data    (qw_lane),
                        .chunk_idx_b({{(16-CW){1'b0}}, lane_chunk_idx_b_next[L]}),
                        .rd_data_b  (qw_lane_b)
                    );

                    // scale bank: group-indexed off the EVEN index's group (for
                    // even CPG, both chunks of a pair are in the same group, so
                    // the even group_idx is the pair's group). Lookahead aligns
                    // with the lane's group_idx_next exactly like single-port.
                    wire [SCALE_WIDTH-1:0] scale_lane;
                    scale_bram #(
                        .LANES(GEOM_LANES),  // LANES-256: geometry follows GEOM_LANES (= LANES in production)
                        .DEPTH(SCALE_DEPTH), .SCALE_W(SCALE_WIDTH), .ADDR_W(17), .GW(7),
                        .GROUP(GROUP)
                    ) u_scale (
                        .clk(clk), .rst_n(rst_n),
                        .we(sc_lane_we), .wr_addr(sc_wr_addr), .wr_data(sc_wr_data),
                        .stage_sel(st_g[grp]),
                        .col_offset(col_offset),
                        .group_idx({{(7-GW){1'b0}}, lane_group_idx_next[L]}),
                        .rd_data(scale_lane)
                    );

                    gemv_lane_scaled_stream_dual #(
                        .K(K), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                        .GROUP(GROUP), .SCALE_WIDTH(SCALE_WIDTH),
                        .SCALE_FRAC(SCALE_FRAC), .RES_WIDTH(RES_WIDTH),
                        .WLAT(WLAT)          // Stage-B': D1 seam fix for the dual lane
                    ) u_lane (
                        .clk(clk), .rst_n(rst_n), .start(sr_g[grp]),
                        .a_chunk(a_g[grp]),
                        .a_chunk_b(ab_g[grp]),
                        .qw_word(qw_lane),
                        .qw_word_b(qw_lane_b),
                        .qz_raw(qz_g[grp]),
                        .scale_q($signed(scale_lane)),
                        .n_chunks_run(grp_nc_run),
                        .chunk_idx_out(lane_chunk_idx[L]),
                        .chunk_idx_b_out(lane_chunk_idx_b[L]),
                        .chunk_idx_valid(lane_chunk_valid[L]),
                        .chunk_idx_next_out(lane_chunk_idx_next[L]),
                        .chunk_idx_b_next_out(lane_chunk_idx_b_next[L]),
                        .group_idx_out(lane_group_idx[L]),
                        .group_idx_next_out(lane_group_idx_next[L]),
                        .acc_out(lane_acc_raw[L]),
                        .result_q(lane_res_raw[L]),
                        .done(lane_done[L])
                    );
                end
            end
        end
    endgenerate

    // =================================================================
    // RETURN-PATH SEAM: per-lane acc_out + result_q + done crossing back.
    // The downstream requant/tree reads ONE lane per cycle (no combinational
    // all-lane reduce), so registering each lane's outputs PIPE_STAGES deep is
    // pure delay. Every group is delayed the SAME PIPE_STAGES -> values stay
    // mutually aligned -> identical. SLR_SAFE=0 = combinational passthrough.
    // =================================================================
    generate
        if (!SLR_SAFE) begin : g_ret_through
            for (grp = 0; grp < LANES; grp = grp + 1) begin : g_pt_ret
                assign acc_out_per_lane[grp*32 +: 32]               = lane_acc_raw[grp];
                assign result_q_per_lane[grp*RES_WIDTH +: RES_WIDTH] = lane_res_raw[grp];
            end
            assign done = &lane_done;
        end else begin : g_ret_pipe
            // RET_STAGES deep (= PIPE_STAGES legacy, or 2 when FF_ACT_CACHE so the
            // return matches the 2-deep forward hub+local). Pure delay line — every
            // lane delayed equally -> values unchanged, only WHEN `done`/results are
            // sampled shifts (absorbed by the done-driven handshake).
            (* keep = "true", dont_touch = "true" *)
            reg signed [31:0]          acc_ret [0:LANES-1][0:RET_STAGES-1];
            (* keep = "true", dont_touch = "true" *)
            reg signed [RES_WIDTH-1:0] res_ret [0:LANES-1][0:RET_STAGES-1];
            (* keep = "true", dont_touch = "true" *)
            reg [LANES-1:0]            done_ret [0:RET_STAGES-1];

            genvar rl, rs;
            for (rl = 0; rl < LANES; rl = rl + 1) begin : g_ret
                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        acc_ret[rl][0] <= '0; res_ret[rl][0] <= '0;
                    end else begin
                        acc_ret[rl][0] <= lane_acc_raw[rl];
                        res_ret[rl][0] <= lane_res_raw[rl];
                    end
                end
                for (rs = 1; rs < RET_STAGES; rs = rs + 1) begin : g_ret_x
                    always_ff @(posedge clk) begin
                        if (!rst_n) begin
                            acc_ret[rl][rs] <= '0; res_ret[rl][rs] <= '0;
                        end else begin
                            acc_ret[rl][rs] <= acc_ret[rl][rs-1];
                            res_ret[rl][rs] <= res_ret[rl][rs-1];
                        end
                    end
                end
                assign acc_out_per_lane[rl*32 +: 32]               = acc_ret[rl][RET_STAGES-1];
                assign result_q_per_lane[rl*RES_WIDTH +: RES_WIDTH] = res_ret[rl][RET_STAGES-1];
            end

            always_ff @(posedge clk) begin
                if (!rst_n) done_ret[0] <= '0;
                else        done_ret[0] <= lane_done;
            end
            for (rs = 1; rs < RET_STAGES; rs = rs + 1) begin : g_done_x
                always_ff @(posedge clk) begin
                    if (!rst_n) done_ret[rs] <= '0;
                    else        done_ret[rs] <= done_ret[rs-1];
                end
            end
            assign done = &done_ret[RET_STAGES-1];
        end
    endgenerate

    // =================================================================
    // BW-4 SIBLING-BATCH RETURN SEAM: register each sibling result the SAME
    // depth as res_ret (RET_STAGES when SLR_SAFE, else passthrough) so
    // result_q_batch is emitted return-seam-aligned with result_q_per_lane.
    // '0 when BATCH_B==1 (no logic).
    // =================================================================
    generate
        if (BATCH_B > 1) begin : g_ret_batch
            if (!SLR_SAFE) begin : g_rb_through
                genvar lb, cb;
                for (lb = 0; lb < LANES; lb = lb + 1) begin : g_rbl
                    for (cb = 0; cb < BB1; cb = cb + 1) begin : g_rbc
                        assign result_q_batch[(lb*BB1+cb)*RES_WIDTH +: RES_WIDTH] =
                               lane_res_batch[lb][cb];
                    end
                end
            end else begin : g_rb_pipe
                (* keep = "true", dont_touch = "true" *)
                reg signed [RES_WIDTH-1:0] resb_ret [0:LANES-1][0:BB1-1][0:RET_STAGES-1];
                genvar lb, cb, rsb;
                for (lb = 0; lb < LANES; lb = lb + 1) begin : g_rbl
                    for (cb = 0; cb < BB1; cb = cb + 1) begin : g_rbc
                        always_ff @(posedge clk) begin
                            if (!rst_n) resb_ret[lb][cb][0] <= '0;
                            else        resb_ret[lb][cb][0] <= lane_res_batch[lb][cb];
                        end
                        for (rsb = 1; rsb < RET_STAGES; rsb = rsb + 1) begin : g_rbx
                            always_ff @(posedge clk) begin
                                if (!rst_n) resb_ret[lb][cb][rsb] <= '0;
                                else        resb_ret[lb][cb][rsb] <= resb_ret[lb][cb][rsb-1];
                            end
                        end
                        assign result_q_batch[(lb*BB1+cb)*RES_WIDTH +: RES_WIDTH] =
                               resb_ret[lb][cb][RET_STAGES-1];
                    end
                end
            end
        end else begin : g_ret_batch_off
            assign result_q_batch = '0;
        end
    endgenerate

endmodule

`default_nettype wire
