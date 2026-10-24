// =====================================================================
// gemv_lane_scaled_stream.sv — single-port scaled GEMV lane
// =====================================================================
// Consumes one K_PAR-wide integer MAC chunk per cycle. Each runtime-sized
// group (chunks_per_scale_group, supported values 4 and 8) is multiplied by
// its signed fixed-point scale at the group boundary and accumulated into
// result_q. SCALED=0 retains the INT32 path. n_chunks_run is latched at
// start; zero selects N_CHUNKS. WLAT=1 delays weight, valid/tag, and scale
// by one cycle for a registered activation seam; WLAT=0 is direct alignment.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module gemv_lane_scaled_stream #(
    parameter int K           = 64,
    parameter int K_PAR       = 16,
    parameter int A_WIDTH     = 16,
    parameter int SCALE_WIDTH = 16,       // signed fixed-point scale width
    parameter int SCALE_FRAC  = 15,       // scale fractional bits (informational)
    parameter int RES_WIDTH   = 48,       // scaled fixed-point accumulator width
    parameter bit SCALED      = 1'b1,     // 0 = unscaled INT32 result
    // Delay weight, scale and issue tags to match an extra activation register.
    parameter bit WLAT        = 1'b0,
    parameter int N_CHUNKS    = K / K_PAR,
    parameter int CW          = (N_CHUNKS <= 1) ? 1 : $clog2(N_CHUNKS + 2),
    // Minimum supported runtime group length (chunks). The group-index width
    // covers the worst case at this length, so a runtime selection of 4 does
    // not alias two groups or truncate the group tag pipeline.
    parameter int CPG_MIN     = 4,
    parameter int N_GROUPS_MAX = (N_CHUNKS + CPG_MIN - 1) / CPG_MIN,
    parameter int GW          = (N_GROUPS_MAX <= 1) ? 1 : $clog2(N_GROUPS_MAX + 1)
) (
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire                            start,

    // Streaming activation: ONE chunk (K_PAR × A_WIDTH bits) per cycle
    input  wire signed [K_PAR*A_WIDTH-1:0] a_chunk,
    // Streaming qweight: ONE int32 word per cycle (16 raw 2-bit weights)
    input  wire        [31:0]              qw_word,
    // Per-group qzero (one 2-bit value); held for the current group
    input  wire        [1:0]               qz_raw,
    // Signed fixed-point scale, held for the current input group.
    input  wire signed [SCALE_WIDTH-1:0]   scale_q,
    // Runtime chunks per scale group. The shared controller validates this
    // to the supported set {4, 8}; the lane latches it at start so a group
    // walk cannot change underneath an active tile.
    input  wire        [15:0]              chunks_per_scale_group,

    // Number of chunks for this launch; zero selects N_CHUNKS.
    // Caller must bound nonzero counts to the elaborated capacity.
    input  wire        [CW-1:0]            n_chunks_run,

    // FSM exposes the chunk index it expects; caller drives a_chunk and
    // qw_word consistent with this chunk.
    output wire        [CW-1:0]            chunk_idx_out,
    output wire                            chunk_idx_valid,
    // Next-state chunk_idx for URAM lookahead.
    output wire        [CW-1:0]            chunk_idx_next_out,
    // Group index the lane is currently consuming (drives scale-BRAM addr).
    output wire        [GW-1:0]            group_idx_out,
    // Next-state group index for scale-BRAM 1-cycle lookahead.
    output wire        [GW-1:0]            group_idx_next_out,

    // Unscaled INT32 sum and scaled RES_WIDTH result. Scaling adds
    // SCALE_FRAC fractional bits to the activation convention; this lane
    // does not shift them away. SCALED=0 sign-extends the unscaled sum.
    output reg  signed [31:0]              acc_out,
    output reg  signed [RES_WIDTH-1:0]     result_q,
    output reg                             done
);

    // Forward declares for FSM
    wire signed [25:0] csa_sum_out;
    wire               csa_out_valid;

    localparam int CSA_LAT = 3;

    // Stream input chunks, drain the CSA pipeline, then publish completion.
    typedef enum logic [1:0] { S_IDLE, S_RUN, S_DRAIN, S_DONE } state_t;
    state_t state, state_n;
    logic [CW-1:0] chunk_idx,   chunk_idx_n;
    logic [CW-1:0] drain_count, drain_count_n;
    logic [CW-1:0] outs_recvd,  outs_recvd_n;

    // Latch the runtime bound and group length so they cannot change during
    // an active walk. chunks_per_scale_group is validated upstream to {4, 8};
    // the latch also keeps the group tag pipeline consistent if the bank is
    // rewritten between operations.
    logic [CW-1:0] nc_run_q;
    logic [15:0]   cpg_run_q;
    wire   [CW-1:0] nc_eff = (nc_run_q == '0) ? CW'(N_CHUNKS) : nc_run_q;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            nc_run_q  <= '0;
            cpg_run_q <= '0;
        end else if (state == S_IDLE && start) begin
            nc_run_q  <= n_chunks_run;
            cpg_run_q <= chunks_per_scale_group;
        end
    end

    // Runtime group geometry: 4 -> shift 2, 8 -> shift 3. The controller
    // rejects every other value, so this is not a silent fallback.
    wire [3:0] cpg_shift = (cpg_run_q == 16'd4) ? 4'd2 : 4'd3;
    wire [CW-1:0] cpg_mask = CW'(cpg_run_q - 16'd1);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; chunk_idx <= '0; drain_count <= '0; outs_recvd <= '0;
        end else begin
            state <= state_n; chunk_idx <= chunk_idx_n;
            drain_count <= drain_count_n; outs_recvd <= outs_recvd_n;
        end
    end

    always_comb begin
        state_n = state; chunk_idx_n = chunk_idx;
        drain_count_n = drain_count; outs_recvd_n = outs_recvd;
        case (state)
            S_IDLE: if (start) begin
                state_n = S_RUN; chunk_idx_n = '0;
                drain_count_n = '0; outs_recvd_n = '0;
            end
            S_RUN: begin
                if (chunk_idx == nc_eff - 1'b1) begin
                    state_n = S_DRAIN; drain_count_n = '0;
                end else chunk_idx_n = chunk_idx + 1'b1;
                if (csa_out_valid) outs_recvd_n = outs_recvd + 1'b1;
            end
            S_DRAIN: begin
                if (csa_out_valid) outs_recvd_n = outs_recvd + 1'b1;
                if (outs_recvd == nc_eff) state_n = S_DONE;
                else if (drain_count >= CSA_LAT + 2) state_n = S_DONE;
                drain_count_n = drain_count + 1'b1;
            end
            S_DONE: state_n = S_IDLE;
            default: state_n = S_IDLE;
        endcase
    end

    assign chunk_idx_out      = chunk_idx;
    assign chunk_idx_next_out = chunk_idx_n;
    assign chunk_idx_valid    = (state == S_RUN);

    // Group index = chunk_idx >> cpg_shift (runtime 4/8). Expose current +
    // next (the next index drives the scale-BRAM 1-cycle lookahead).
    wire [CW-1:0] grp_cur_full  = chunk_idx   >> cpg_shift;
    wire [CW-1:0] grp_next_full = chunk_idx_n >> cpg_shift;
    assign group_idx_out      = grp_cur_full[GW-1:0];
    assign group_idx_next_out = grp_next_full[GW-1:0];

    // Align weight and issue timing with the activation read latency.
    wire [31:0]                  qw_word_eff;
    wire                         mac_in_valid;     // CSA in_valid / scaled-issue valid (issue point)
    wire [CW-1:0]                issue_chunk_idx;  // chunk_idx used for group tagging at issue
    // Scale follows the same delay as weight and group tag; otherwise a
    // group-boundary product would use the following group's scale.
    wire signed [SCALE_WIDTH-1:0] scale_q_eff;
    generate
    if (!WLAT) begin : g_wlat0
        assign qw_word_eff     = qw_word;
        assign mac_in_valid    = chunk_idx_valid;
        assign issue_chunk_idx = chunk_idx;
        assign scale_q_eff     = scale_q;
    end else begin : g_wlat1
        reg [31:0]                  qw_word_d;
        reg                         civ_d;
        reg [CW-1:0]                cidx_d;
        reg signed [SCALE_WIDTH-1:0] scale_q_d;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                qw_word_d <= '0; civ_d <= 1'b0; cidx_d <= '0; scale_q_d <= '0;
            end else begin
                qw_word_d <= qw_word;
                civ_d     <= chunk_idx_valid;
                cidx_d    <= chunk_idx;
                scale_q_d <= scale_q;
            end
        end
        assign qw_word_eff     = qw_word_d;
        assign mac_in_valid    = civ_d;
        assign issue_chunk_idx = cidx_d;
        assign scale_q_eff     = scale_q_d;
    end
    endgenerate

    // ----------------------------------------------------------------
    // Dequant — qw_word is current-chunk consistent (same as stream lane)
    // ----------------------------------------------------------------
    wire [31:0] qz_word_synth = {30'b0, qz_raw};
    wire [47:0] w_s_packed;
    gptq2_dequant16 u_dequant (
        .qw_word    (qw_word_eff),
        .qz_word    (qz_word_synth),
        .n_idx      (4'd0),
        .w_s_packed (w_s_packed)
    );

    // 16 × MAC
    logic signed [19:0] prods [0:K_PAR-1];
    for (genvar i = 0; i < K_PAR; i = i + 1) begin : g_mac
        wire signed [A_WIDTH-1:0] a_lane    = a_chunk[i*A_WIDTH +: A_WIDTH];
        wire signed [2:0]         w_s_lane  = $signed(w_s_packed[i*3 +: 3]);
        assign prods[i] = $signed(a_lane) * $signed(w_s_lane);
    end

    // Wallace CSA tree (project standard reduce)
    csa_tree16_int20 u_csa (
        .clk(clk), .rst_n(rst_n),
        .in_valid(mac_in_valid),
        .prods(prods),
        .sum_out(csa_sum_out),
        .out_valid(csa_out_valid)
    );

    // ----------------------------------------------------------------
    // Unscaled INT32 accumulator for diagnostics and SCALED=0.
    // ----------------------------------------------------------------
    logic signed [31:0] acc;
    always_ff @(posedge clk) begin
        if (!rst_n)                            acc <= '0;
        else if (state == S_IDLE && start)     acc <= '0;
        else if (csa_out_valid)                acc <= acc + $signed(csa_sum_out);
    end

    // ================================================================
    // SCALED PATH  (compiled away when SCALED=0 — pure regression mode)
    // ================================================================
    generate
    if (SCALED) begin : g_scaled
        // --- Track which GROUP each csa_out belongs to ---------------
        // The csa tree is CSA_LAT-deep; the chunk that PRODUCED a given
        // csa_out left S_RUN CSA_LAT cycles earlier. We therefore tag each
        // chunk's group at issue time and shift it down the pipe alongside
        // the csa data so the scaled accumulate uses the RIGHT group.
        //
        // group_at_issue = issue_chunk_idx >> cpg_shift, captured when the
        // (WLAT-aware) issue fires. A CSA_LAT-stage shift register carries
        // (group, last_in_group, valid) so it lines up with csa_out_valid.
        // The mask/last test uses the LATCHED runtime group length, so both
        // 4 and 8 produce the same boundary semantics.
        wire [CW-1:0] grp_issue_full = issue_chunk_idx >> cpg_shift;
        wire [GW-1:0] grp_issue      = grp_issue_full[GW-1:0];
        // last chunk in this group? (issue_chunk_idx mod cpg == cpg-1)
        wire last_in_grp_issue = ((issue_chunk_idx & cpg_mask) == cpg_mask);
        wire issue_valid = mac_in_valid;

        // Shift register matched to csa pipeline depth (CSA_LAT cycles).
        // We also pipeline scale_q so the scale APPLIED at the group flush
        // is the scale the caller drove for that group AT ISSUE TIME — i.e.
        // the caller aligns scale_q with group_idx_out (issue side), exactly
        // as qw_word aligns with chunk_idx_out. No flush-side caller timing.
        logic [GW-1:0]               grp_sr   [0:CSA_LAT-1];
        logic                        last_sr  [0:CSA_LAT-1];
        logic                        vld_sr   [0:CSA_LAT-1];
        logic signed [SCALE_WIDTH-1:0] sc_sr  [0:CSA_LAT-1];
        integer s;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                for (s = 0; s < CSA_LAT; s = s + 1) begin
                    grp_sr[s]  <= '0; last_sr[s] <= 1'b0; vld_sr[s] <= 1'b0;
                    sc_sr[s]   <= '0;
                end
            end else begin
                grp_sr[0]  <= grp_issue;
                last_sr[0] <= last_in_grp_issue;
                vld_sr[0]  <= issue_valid;
                sc_sr[0]   <= scale_q_eff;   // D1 SCALE-ALIGN: WLAT-delayed scale
                for (s = 1; s < CSA_LAT; s = s + 1) begin
                    grp_sr[s]  <= grp_sr[s-1];
                    last_sr[s] <= last_sr[s-1];
                    vld_sr[s]  <= vld_sr[s-1];
                    sc_sr[s]   <= sc_sr[s-1];
                end
            end
        end
        // Tags aligned with csa_out_valid (the head of the SR).
        wire [GW-1:0]                 grp_now   = grp_sr[CSA_LAT-1];
        wire                          last_now  = last_sr[CSA_LAT-1];
        wire signed [SCALE_WIDTH-1:0] scale_now = sc_sr[CSA_LAT-1];

        // --- Per-group INT accumulator -------------------------------
        // Accumulate csa partial sums within a group, then on the group's
        // last chunk multiply by scale and add to the scaled accumulator.
        // Group INT range: GROUP * max|a*w| = 128 * (32767*4) ≈ 2^24, plus
        // headroom; 32 bits is safe.
        logic signed [31:0]          group_acc;
        logic signed [RES_WIDTH-1:0] scaled_acc;

        // scale_q is held by the caller for the current group; sample it
        // at the cycle the group's last csa_out lands (caller aligns it via
        // group_idx_out — same lockstep as qw_word vs chunk_idx).
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                group_acc  <= '0;
                scaled_acc <= '0;
            end else if (state == S_IDLE && start) begin
                group_acc  <= '0;
                scaled_acc <= '0;
            end else if (csa_out_valid) begin
                logic signed [31:0] g_total;
                g_total = group_acc + $signed(csa_sum_out);
                if (last_now) begin
                    // flush this group: scaled_acc += g_total * scale (the
                    // pipelined scale that the caller drove for THIS group).
                    scaled_acc <= scaled_acc +
                                  ($signed(g_total) * $signed(scale_now));
                    group_acc  <= '0;
                end else begin
                    group_acc  <= g_total;
                end
            end
        end

        // --- Outputs -------------------------------------------------
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                acc_out  <= '0; result_q <= '0; done <= 1'b0;
            end else begin
                done <= (state == S_DONE);
                if (state == S_DONE) begin
                    acc_out  <= acc;          // unscaled debug value
                    result_q <= scaled_acc;   // scaled fixed-point output
                end
            end
        end
    end else begin : g_unscaled
        // SCALED=0 — exact gemv_lane_int_stream_wallace behaviour.
        // (group_idx_out/next are assigned at the top level; scale_q ignored.)
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                acc_out  <= '0; result_q <= '0; done <= 1'b0;
            end else begin
                done <= (state == S_DONE);
                if (state == S_DONE) begin
                    acc_out  <= acc;
                    result_q <= $signed(acc);   // sign-extended to RES_WIDTH
                end
            end
        end
    end
    endgenerate

endmodule

`default_nettype wire
