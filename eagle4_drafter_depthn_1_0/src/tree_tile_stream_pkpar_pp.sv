// =====================================================================
// tree_tile_stream_pkpar_pp.sv   (EXP_G12TREE_PINGPONG, P-wide arm)
// P-wide full-vocab G12 -> tree_merge_pipe controller with a 2-entry FF
// PING-PONG tile buffer. The pkpar analogue of tree_tile_stream_pp: the
// baseline tree_tile_stream_pkpar already buffers ONE tile but still
// serializes GEMV(W) then stream(LANES/P), costing W + LANES/P + 1 per
// tile with the GEMV array idle during every stream cycle. This version
// runs tile t+1's GEMV UNDER tile t's stream: per-tile ~= max(W, LANES/P).
//
// Stream CONTRACT identical to tree_tile_stream_pkpar (monotone P-groups,
// lane 0 lowest index, tail lanes disabled via lane_en, last_in on the
// group containing V_MAX-1). tree_merge_pipe is reused UNCHANGED.
//
// FF-STAGING LAW: ping-pong buffer = (* ram_style="registers" *) FF
// register file (RE-READ seam; 2 x LANES x 32b). Netlist: FDRE only.
//
// MULTI-DRIVER SAFE: both FSMs in ONE always_ff; producer capture slot and
// consumer stream slot are provably different bits whenever both fire.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module tree_tile_stream_pkpar_pp #(
    parameter int LANES   = 128,
    parameter int P       = 4,
    parameter int V_MAX   = 128256,
    parameter int IDX_W   = 18,
    parameter int SCORE_W = 32,
    parameter int N_TILES = (V_MAX + LANES - 1) / LANES,
    parameter int LANE_W  = (LANES  <= 1) ? 1 : $clog2(LANES),
    parameter int TILE_W  = (N_TILES <= 1) ? 1 : $clog2(N_TILES),
    parameter int GRP_W   = (LANES/P <= 1) ? 1 : $clog2(LANES/P)
) (
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire                          tree_start,

    // V1 FR-Spec runtime scan length (VARIABLE_VOCAB_RTL_PLAN.md §3.6):
    // # of LANES-wide tiles to walk THIS tree scan. 0 or >N_TILES = full
    // vocab (compile N_TILES) — reset/default bit-identical. Launch-sampled
    // into ntq/lastv_q on the tree_start pulse (frozen for the whole walk).
    input  wire [15:0]                   n_tiles_cfg,

    input  wire signed [SCORE_W*LANES-1:0] array_acc_out,
    input  wire                          array_done,

    output logic                         gemv_tile_start,
    output logic [15:0]                  gemv_tile_idx,
    output logic                         gemv_stage_g12,

    output logic                         tree_dec_start,
    output logic                         valid_in,
    output logic signed [P*SCORE_W-1:0]  scores_in,
    output logic [P*IDX_W-1:0]           indices_in,
    output logic [P-1:0]                 lane_en,
    output logic                         last_in,

    output logic                         busy
);

    localparam int GRPS_PER_TILE      = LANES / P;
    localparam int LAST_TILE_I        = N_TILES - 1;
    localparam int LAST_VALID_GLOBAL  = V_MAX - 1;
    initial begin
        if ((LANES % P) != 0)
            $error("tree_tile_stream_pkpar_pp: LANES(%0d) must be a multiple of P(%0d)", LANES, P);
    end
    localparam logic [TILE_W-1:0] LAST_TILE = TILE_W'(N_TILES - 1);

    // ----- 2-entry FF ping-pong tile buffer -----
    (* ram_style = "registers", shreg_extract = "no" *)
    logic signed [SCORE_W-1:0] tile_buf [0:1][0:LANES-1];
    logic [1:0] slot_full;

    // ----- producer (GEMV stepper) -----
    typedef enum logic [1:0] { P_IDLE = 2'd0, P_ISSUE = 2'd1, P_WAIT = 2'd2 } p_state_e;
    p_state_e          pstate;
    logic [TILE_W:0]   ptile;
    wire               pslot = ptile[0];

    // ----- consumer (P-group streamer) -----
    typedef enum logic [1:0] { C_IDLE = 2'd0, C_WAITFULL = 2'd1, C_STREAM = 2'd2, C_DRAIN = 2'd3 } c_state_e;
    c_state_e          cstate;
    logic [TILE_W-1:0] ctile;
    logic [GRP_W-1:0]  grp;
    wire               cslot = ctile[0];

    // ---- V1 FR-Spec launch-sampled scan length (frozen per walk) ----
    //   ntq     : effective tile count for THIS walk (1..N_TILES)
    //   lastv_q : last valid global vocab index for THIS walk
    //   nt_eff  : clamped live register view (0/over-range -> full vocab).
    //   Sampled on tree_start in the SAME single always_ff as both FSMs
    //   (multi-driver-safe by the file's own contract); first consumer read
    //   is >=2 cycles after the sample lands (P_WAIT stop / C_STREAM).
    logic [15:0] ntq;
    logic [IDX_W-1:0] lastv_q;
    wire  [15:0] nt_eff = (n_tiles_cfg == 16'd0 || n_tiles_cfg > 16'(N_TILES))
                          ? 16'(N_TILES) : n_tiles_cfg;
    // full-scan keeps the exact V_MAX tail (guards V_MAX not a LANES multiple);
    // a truncated scan is tile-granular: last index = nt_eff*LANES - 1.
    wire  [IDX_W-1:0] lastv_eff = (nt_eff == 16'(N_TILES))
                          ? IDX_W'(V_MAX - 1)
                          : IDX_W'((32'(nt_eff) * LANES) - 1);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pstate          <= P_IDLE;
            cstate          <= C_IDLE;
            ptile           <= '0;
            ctile           <= '0;
            grp             <= '0;
            slot_full       <= 2'b00;
            gemv_tile_start <= 1'b0;
            gemv_tile_idx   <= 16'd0;
            gemv_stage_g12  <= 1'b0;
            tree_dec_start  <= 1'b0;
            valid_in        <= 1'b0;
            scores_in       <= '0;
            indices_in      <= '0;
            lane_en         <= '0;
            last_in         <= 1'b0;
            busy            <= 1'b0;
            ntq             <= 16'(N_TILES);
            lastv_q         <= IDX_W'(V_MAX - 1);
        end else begin
            // V1: launch-sample the scan length (same edge as walk entry; the
            // first read of ntq/lastv_q is >=2 cycles later — plan §3.6 note).
            if (tree_start) begin
                ntq     <= nt_eff;
                lastv_q <= lastv_eff;
            end
            gemv_tile_start <= 1'b0;
            tree_dec_start  <= 1'b0;
            valid_in        <= 1'b0;
            last_in         <= 1'b0;
            lane_en         <= '0;

            // ---------------- producer ----------------
            case (pstate)
                P_IDLE: begin
                    if (tree_start) begin
                        ptile  <= '0;
                        pstate <= P_ISSUE;
                    end
                end
                P_ISSUE: begin
                    if (!slot_full[pslot]) begin
                        gemv_tile_idx   <= 16'(ptile);
                        gemv_tile_start <= 1'b1;
                        pstate          <= P_WAIT;
                    end
                end
                P_WAIT: begin
                    gemv_tile_idx <= 16'(ptile);
                    if (array_done) begin
                        for (int l = 0; l < LANES; l++)
                            tile_buf[pslot][l] <= array_acc_out[l*SCORE_W +: SCORE_W];
                        slot_full[pslot] <= 1'b1;
                        if (ptile == (TILE_W+1)'(ntq - 16'd1)) begin   // V1 runtime stop
                            pstate <= P_IDLE;
                        end else begin
                            ptile  <= ptile + 1'b1;
                            pstate <= P_ISSUE;
                        end
                    end
                end
                default: pstate <= P_IDLE;
            endcase

            // ---------------- consumer ----------------
            case (cstate)
                C_IDLE: begin
                    gemv_stage_g12 <= 1'b0;
                    busy           <= 1'b0;
                    if (tree_start) begin
                        tree_dec_start <= 1'b1;
                        ctile          <= '0;
                        grp            <= '0;
                        gemv_stage_g12 <= 1'b1;
                        busy           <= 1'b1;
                        cstate         <= C_WAITFULL;
                    end
                end
                C_WAITFULL: begin
                    if (slot_full[cslot]) begin
                        grp    <= '0;
                        cstate <= C_STREAM;
                    end
                end
                C_STREAM: begin
                    // emit P-group `grp` of the buffered tile (baseline order).
                    valid_in <= 1'b1;
                    for (int g = 0; g < P; g++) begin
                        int unsigned gl;
                        int unsigned gv;
                        gl = grp * P + g;
                        gv = ctile * LANES + gl;
                        scores_in [g*SCORE_W +: SCORE_W] <= tile_buf[cslot][gl];
                        indices_in[g*IDX_W   +: IDX_W]   <= IDX_W'(gv);
                        lane_en[g] <= (gv <= 32'(lastv_q));            // V1 runtime tail
                    end

                    if ((ctile == TILE_W'(ntq - 16'd1)) &&
                        ((ctile * LANES + grp * P + (P-1)) >= 32'(lastv_q))) begin
                        last_in          <= 1'b1;
                        slot_full[cslot] <= 1'b0;
                        cstate           <= C_DRAIN;
                    end else if (grp == GRP_W'(GRPS_PER_TILE - 1)) begin
                        slot_full[cslot] <= 1'b0;   // tile fully streamed
                        ctile            <= ctile + 1'b1;
                        cstate           <= C_WAITFULL;
                    end else begin
                        grp <= grp + 1'b1;
                    end
                end
                C_DRAIN: begin
                    gemv_stage_g12 <= 1'b0;
                    busy           <= 1'b0;
                    cstate         <= C_IDLE;
                end
                default: cstate <= C_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && pstate == P_WAIT && array_done &&
            cstate == C_STREAM && (pslot == cslot))
            $error("tree_tile_stream_pkpar_pp: producer capture into slot %0d while consumer streams it (ptile=%0d ctile=%0d)", pslot, ptile, ctile);
    end
`endif

endmodule

`default_nettype wire
