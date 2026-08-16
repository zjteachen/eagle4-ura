// =====================================================================
// gemv_lane_scaled_stream_dual.sv      (WO-025 — dual-read-port GEMV lane)
// =====================================================================
//
// 32-MAC/cycle scaled GEMV lane. The bit-exact analogue of
// gemv_lane_scaled_stream that consumes TWO K-chunks per cycle (the EVEN
// chunk via URAM port A, the ODD chunk via URAM port B), so the per-column
// active-cycle count halves: ceil(nc/2) instead of nc.
//
// WHY THIS IS THE WO-025 LEVER (project_dualreadport_uram_gemv_lever):
//   The GEMV is URAM-READ-PORT bound: one K_PAR-wide row = one chunk per
//   cycle is the per-column throughput cap, NOT DSP (4-8/lane) or activation
//   fanout. URAM288 on V80 is true-dual-port and port B is IDLE during
//   compute. Driving port B as a 2nd compute read fetches 2 chunks/cycle =
//   2*K_PAR weights/cycle with the SAME URAM block count.
//
// =====================================================================
// BIT-EXACTNESS  (the hard requirement — feedback_no_accuracy_tradeoff)
// =====================================================================
// The reference per-group INT partial sum is
//     gsum_g = Σ_{chunk c ∈ group g}  ( Σ_i a[c][i]·w_s[c][i] )
// and the scaled output is  out = Σ_g  gsum_g · scale_g.
//
// This lane computes, EACH cycle, csa_e (= Σ_i a[even][i]·w_s[even][i]) and
// csa_o (= Σ_i a[odd][i]·w_s[odd][i]) with TWO independent 16-MAC + Wallace
// CSA datapaths (csa_tree16_int20, identical to the single-port lane's), then
// folds  group_acc += csa_e + csa_o. Because INT addition is associative and
// commutative, summing the chunks two-at-a-time gives the IDENTICAL gsum_g as
// summing them one-at-a-time. The scale multiply happens ONCE per group on the
// full INT group sum (same DSP58 INT-mode slot, same pipelined scale_now), so
// scaled_acc is byte-for-byte identical to the single-port lane.
//
// GROUP-BOUNDARY ALIGNMENT (why a pair never straddles two scales):
//   CPG = GROUP/K_PAR chunks per group. For EVEN CPG (production: 128/16 = 8),
//   chunks 2p and 2p+1 share the same group for every pair p (2p+1 crosses a
//   CPG boundary only when 2p ≡ CPG-1 mod CPG, impossible for even 2p). So both
//   members of a cycle's pair carry the SAME group/scale — the "fold then one
//   scale" reduction is exact. CPG odd is UNSUPPORTED (synthesis guard below);
//   the EAGLE4 datapath is always CPG=8.
//
// ODD nc / PARTIAL FINAL GROUP:
//   The FSM walks ceil(nc/2) pairs. On the final pair when nc is odd, the ODD
//   chunk index == nc (out of range) → its MAC is MASKED to 0 (a_odd forced 0)
//   so csa_o contributes nothing. The flush "last chunk of group" test is
//   computed against the CLAMPED group end min((g+1)*CPG-1, nc-1), so a partial
//   final group still flushes on its true last valid chunk. (Production nc =
//   16/192/384/512 are all multiples of CPG=8 → every group full, even nc.)
//
// LATENCY (start → done) and the SCALE alignment match the single-port lane's
// idiom exactly: caller drives scale_q for the CURRENT group (aligned to
// group_idx_out), the lane pipelines it CSA_LAT deep and applies it at flush.
// The activation for BOTH chunks of a pair is supplied by the caller off the
// exposed chunk_idx_out (even) and chunk_idx_b_out (odd) — same combinational
// fetch idiom as the single-port lane, just 2-wide.
//
// REGRESSION: instantiated ONLY when the array's DUAL_PORT=1. DUAL_PORT=0 uses
// the unchanged single-port gemv_lane_scaled_stream, so the existing g6 / perbank
// / cycle-measure contracts are untouched.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module gemv_lane_scaled_stream_dual #(
    parameter int K           = 64,
    parameter int K_PAR       = 16,
    parameter int A_WIDTH     = 16,
    parameter int GROUP       = 128,      // GPTQ group_size
    parameter int SCALE_WIDTH = 16,       // signed scale
    parameter int SCALE_FRAC  = 15,       // fractional bits (informational)
    parameter int RES_WIDTH   = 48,       // scaled fixed-point accumulator width
    // WLAT (Stage-B' D1 activation-seam fix for the DUAL lane): mirror of the
    // single lane's WLAT. On the SLR_SAFE seam the activation (a_chunk/a_chunk_b)
    // arrives ONE register deeper than the weights (qw_word/qw_word_b), so at
    // WLAT=0 the dual lane computes W[pair]·act[pair-1] on the body GEMV (the D1
    // stale-by-1 class). WLAT=1 delays BOTH weight chunks + the MAC enable + the
    // per-pair issue index (group tag + odd-mask) + scale_q by ONE cycle so the
    // (already +1-deeper) activation and the delayed weights meet. BIT-EXACT: all
    // operands and their tags shift by the SAME 1 cycle; the extra cycle is
    // absorbed by S_DRAIN (CSA_LAT+2 margin). WLAT=0 = legacy (bit+cycle identical
    // to the isolation-proven dual lane).
    parameter bit WLAT        = 1'b0,
    parameter int N_CHUNKS    = K / K_PAR,
    parameter int CW          = (N_CHUNKS <= 1) ? 1 : $clog2(N_CHUNKS + 2),
    parameter int CPG         = GROUP / K_PAR,                    // chunks/group (8)
    parameter int N_GROUPS    = (K + GROUP - 1) / GROUP,
    parameter int GW          = (N_GROUPS <= 1) ? 1 : $clog2(N_GROUPS + 1),
    // pair-walk count width (ceil(N_CHUNKS/2))
    parameter int N_PAIRS     = (N_CHUNKS + 1) / 2,
    parameter int PW          = (N_PAIRS <= 1) ? 1 : $clog2(N_PAIRS + 2)
) (
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire                            start,

    // Streaming activation: EVEN chunk (port A) + ODD chunk (port B) per cycle
    input  wire signed [K_PAR*A_WIDTH-1:0] a_chunk,      // even chunk (2p)
    input  wire signed [K_PAR*A_WIDTH-1:0] a_chunk_b,    // odd  chunk (2p+1)
    // Streaming qweight: even word (port A) + odd word (port B)
    input  wire        [31:0]              qw_word,       // even chunk weights
    input  wire        [31:0]              qw_word_b,      // odd  chunk weights
    // Per-group qzero (one 2-bit value); held for the current group
    input  wire        [1:0]               qz_raw,
    // Per-group dequant scale, held for the CURRENT group
    input  wire signed [SCALE_WIDTH-1:0]   scale_q,

    // RUNTIME terminal chunk count for the FSM walk (G12_K_PATH FIX-1 idiom).
    // The lane is ELABORATED for the worst-case K_MAX (N_CHUNKS) but only walks
    // n_chunks_run chunks per start; 0 -> full N_CHUNKS (unconnected-safe).
    input  wire        [CW-1:0]            n_chunks_run,

    // FSM exposes the EVEN chunk index it expects (port A); the ODD index
    // (port B) is +1. Caller drives a_chunk/qw_word off chunk_idx_out and
    // a_chunk_b/qw_word_b off chunk_idx_b_out.
    output wire        [CW-1:0]            chunk_idx_out,     // even (2p)
    output wire        [CW-1:0]            chunk_idx_b_out,   // odd  (2p+1)
    output wire                            chunk_idx_valid,
    // Next-state EVEN/ODD chunk_idx for URAM 1-cycle lookahead (ports A/B).
    output wire        [CW-1:0]            chunk_idx_next_out,
    output wire        [CW-1:0]            chunk_idx_b_next_out,
    // Group index the lane is currently consuming (drives scale-BRAM addr).
    output wire        [GW-1:0]            group_idx_out,
    output wire        [GW-1:0]            group_idx_next_out,

    // Final UNSCALED INT32 accumulator (debug; == single-port lane's acc_out).
    output reg  signed [31:0]              acc_out,
    // Final SCALED fixed-point accumulator (Q with SCALE_FRAC frac bits).
    output reg  signed [RES_WIDTH-1:0]     result_q,
    output reg                             done
);

`ifndef SYNTHESIS
    initial if (CPG[0] != 1'b0)
        $error("gemv_lane_scaled_stream_dual: CPG (GROUP/K_PAR=%0d) must be EVEN for the dual-port pair/group alignment.", CPG);
`endif

    localparam int CSA_LAT  = 3;
    localparam int CPG_BITS = (CPG <= 1) ? 1 : $clog2(CPG);

    // Forward declares for the FSM (driven by the two CSA trees below).
    wire signed [25:0] csa_sum_e, csa_sum_o;
    wire               csa_out_valid_e, csa_out_valid_o;
    wire               csa_out_valid;     // == csa_out_valid_e (identical timing)
    wire signed [26:0] csa_pair_sum;      // csa_sum_e + csa_sum_o per cycle

    // ----------------------------------------------------------------
    // FSM — pair walk. Mirrors gemv_lane_scaled_stream but the run counter
    // is the PAIR index (advances by 1 pair = 2 chunks per cycle).
    // ----------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE, S_RUN, S_DRAIN, S_DONE } state_t;
    state_t state, state_n;
    wire state_is_idle = (state == S_IDLE);

    // ----------------------------------------------------------------
    // Effective terminal chunk count, latched at start (same as single-port).
    // ----------------------------------------------------------------
    logic [CW-1:0] nc_run_q;
    wire  [CW-1:0] nc_eff = (nc_run_q == '0) ? CW'(N_CHUNKS) : nc_run_q;
    always_ff @(posedge clk) begin
        if (!rst_n)                            nc_run_q <= '0;
        else if (state_is_idle && start)       nc_run_q <= n_chunks_run;
    end

    // number of PAIRS to walk = ceil(nc_eff/2)
    wire [PW-1:0] np_eff = PW'((nc_eff + CW'(1)) >> 1);

    logic [PW-1:0] pair_idx,    pair_idx_n;
    logic [CW-1:0] drain_count, drain_count_n;
    logic [PW-1:0] outs_recvd,  outs_recvd_n;   // # pair-CSA results received

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; pair_idx <= '0; drain_count <= '0; outs_recvd <= '0;
        end else begin
            state <= state_n; pair_idx <= pair_idx_n;
            drain_count <= drain_count_n; outs_recvd <= outs_recvd_n;
        end
    end

    always_comb begin
        state_n = state; pair_idx_n = pair_idx;
        drain_count_n = drain_count; outs_recvd_n = outs_recvd;
        case (state)
            S_IDLE: if (start) begin
                state_n = S_RUN; pair_idx_n = '0;
                drain_count_n = '0; outs_recvd_n = '0;
            end
            S_RUN: begin
                if (pair_idx == np_eff - 1'b1) begin
                    state_n = S_DRAIN; drain_count_n = '0;
                end else pair_idx_n = pair_idx + 1'b1;
                if (csa_out_valid) outs_recvd_n = outs_recvd + 1'b1;
            end
            S_DRAIN: begin
                if (csa_out_valid) outs_recvd_n = outs_recvd + 1'b1;
                if (outs_recvd == np_eff) state_n = S_DONE;
                else if (drain_count >= CSA_LAT + 2) state_n = S_DONE;
                drain_count_n = drain_count + 1'b1;
            end
            S_DONE: state_n = S_IDLE;
            default: state_n = S_IDLE;
        endcase
    end

    // EVEN chunk = 2*pair_idx ; ODD chunk = 2*pair_idx + 1
    wire [CW-1:0] even_cur  = CW'(pair_idx)      << 1;
    wire [CW-1:0] even_next = CW'(pair_idx_n)    << 1;
    assign chunk_idx_out        = even_cur;
    assign chunk_idx_b_out      = even_cur  + CW'(1);
    assign chunk_idx_next_out   = even_next;
    assign chunk_idx_b_next_out = even_next + CW'(1);
    assign chunk_idx_valid      = (state == S_RUN);

    // Group index = even_chunk / CPG (== odd_chunk/CPG for even CPG). Current +
    // next for the scale-BRAM 1-cycle lookahead.
    wire [CW-1:0] grp_cur_full  = (CPG <= 1) ? even_cur  : (even_cur  >> CPG_BITS);
    wire [CW-1:0] grp_next_full = (CPG <= 1) ? even_next : (even_next >> CPG_BITS);
    assign group_idx_out      = grp_cur_full[GW-1:0];
    assign group_idx_next_out = grp_next_full[GW-1:0];

    // ----------------------------------------------------------------
    // WLAT (D1 dual-lane seam fix): delay BOTH weight chunks + MAC enable +
    // the per-pair issue index (group tag + odd-mask) + scale_q by ONE cycle
    // to meet the +1-deeper SLR-seam activation. Pass-through at WLAT=0
    // (bit+cycle identical to the isolation-proven dual lane). chunk_idx_out /
    // group_idx_out stay UNDELAYED (the caller's fetch index; the seam adds the
    // register externally) — same idiom as the single lane.
    // ----------------------------------------------------------------
    wire [31:0]                   qw_word_eff, qw_word_b_eff;
    wire                          mac_in_valid;
    wire [CW-1:0]                 even_issue;
    wire signed [SCALE_WIDTH-1:0] scale_q_eff;
    generate
    if (!WLAT) begin : g_wlat0
        assign qw_word_eff   = qw_word;
        assign qw_word_b_eff = qw_word_b;
        assign mac_in_valid  = chunk_idx_valid;
        assign even_issue    = even_cur;
        assign scale_q_eff   = scale_q;
    end else begin : g_wlat1
        reg [31:0]                   qw_d, qwb_d;
        reg                          civ_d;
        reg [CW-1:0]                 ec_d;
        reg signed [SCALE_WIDTH-1:0] sc_d;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                qw_d <= '0; qwb_d <= '0; civ_d <= 1'b0; ec_d <= '0; sc_d <= '0;
            end else begin
                qw_d  <= qw_word;  qwb_d <= qw_word_b; civ_d <= chunk_idx_valid;
                ec_d  <= even_cur; sc_d  <= scale_q;
            end
        end
        assign qw_word_eff   = qw_d;
        assign qw_word_b_eff = qwb_d;
        assign mac_in_valid  = civ_d;
        assign even_issue    = ec_d;
        assign scale_q_eff   = sc_d;
    end
    endgenerate

    // ----------------------------------------------------------------
    // ODD-chunk validity mask: when 2p+1 >= nc_eff (odd nc final pair), the
    // odd MAC must contribute 0. Force a_odd = 0 so csa_o = 0 exactly.
    // (Uses even_issue = the WLAT-aligned index.)
    // ----------------------------------------------------------------
    wire odd_valid_issue = ((even_issue + CW'(1)) < nc_eff);
    wire signed [K_PAR*A_WIDTH-1:0] a_odd_eff =
        odd_valid_issue ? a_chunk_b : '0;

    // ----------------------------------------------------------------
    // Dequant (even + odd) — combinational, identical to single-port lane.
    // ----------------------------------------------------------------
    wire [31:0] qz_word_synth = {30'b0, qz_raw};

    wire [47:0] w_s_packed_e;
    gptq2_dequant16 u_dequant_e (
        .qw_word(qw_word_eff),   .qz_word(qz_word_synth), .n_idx(4'd0),
        .w_s_packed(w_s_packed_e)
    );
    wire [47:0] w_s_packed_o;
    gptq2_dequant16 u_dequant_o (
        .qw_word(qw_word_b_eff), .qz_word(qz_word_synth), .n_idx(4'd0),
        .w_s_packed(w_s_packed_o)
    );

    // 16 × MAC, EVEN path
    logic signed [19:0] prods_e [0:K_PAR-1];
    for (genvar i = 0; i < K_PAR; i = i + 1) begin : g_mac_e
        wire signed [A_WIDTH-1:0] a_lane   = a_chunk[i*A_WIDTH +: A_WIDTH];
        wire signed [2:0]         w_s_lane = $signed(w_s_packed_e[i*3 +: 3]);
        assign prods_e[i] = $signed(a_lane) * $signed(w_s_lane);
    end
    // 16 × MAC, ODD path (a_odd_eff is zero-masked on the partial final pair)
    logic signed [19:0] prods_o [0:K_PAR-1];
    for (genvar i = 0; i < K_PAR; i = i + 1) begin : g_mac_o
        wire signed [A_WIDTH-1:0] a_lane   = a_odd_eff[i*A_WIDTH +: A_WIDTH];
        wire signed [2:0]         w_s_lane = $signed(w_s_packed_o[i*3 +: 3]);
        assign prods_o[i] = $signed(a_lane) * $signed(w_s_lane);
    end

    // ----------------------------------------------------------------
    // Two Wallace CSA trees (even + odd), lock-step (same latency CSA_LAT).
    // ----------------------------------------------------------------
    assign csa_out_valid = csa_out_valid_e;             // identical timing

    csa_tree16_int20 u_csa_e (
        .clk(clk), .rst_n(rst_n), .in_valid(mac_in_valid),
        .prods(prods_e), .sum_out(csa_sum_e), .out_valid(csa_out_valid_e)
    );
    csa_tree16_int20 u_csa_o (
        .clk(clk), .rst_n(rst_n), .in_valid(mac_in_valid),
        .prods(prods_o), .sum_out(csa_sum_o), .out_valid(csa_out_valid_o)
    );

    // ----------------------------------------------------------------
    // UNSCALED INT32 accumulator (debug; matches single-port acc_out exactly).
    // Per cycle accumulate (csa_e + csa_o). INT26 + INT26 = up to INT27.
    // ----------------------------------------------------------------
    assign csa_pair_sum = $signed(csa_sum_e) + $signed(csa_sum_o);
    logic signed [31:0] acc;
    always_ff @(posedge clk) begin
        if (!rst_n)                          acc <= '0;
        else if (state == S_IDLE && start)   acc <= '0;
        else if (csa_out_valid)              acc <= acc + $signed(csa_pair_sum);
    end

    // ================================================================
    // SCALED PATH — per-group fold then one DSP scale.
    // ================================================================
    // Tag each PAIR at issue with its group, last-in-group, and the scale the
    // caller drove for that group; shift CSA_LAT deep to line up with the
    // pair-CSA result. Identical idiom to gemv_lane_scaled_stream.
    wire [CW-1:0] grp_issue_full = (CPG <= 1) ? even_issue : (even_issue >> CPG_BITS);
    wire [GW-1:0] grp_issue      = grp_issue_full[GW-1:0];

    // group's last VALID chunk = min((grp+1)*CPG-1, nc_eff-1).
    wire [CW-1:0] grp_end_full = ((grp_issue_full + CW'(1)) << CPG_BITS) - CW'(1);
    wire [CW-1:0] grp_last     = (grp_end_full < (nc_eff - CW'(1)))
                                 ? grp_end_full : (nc_eff - CW'(1));
    // does THIS pair contain grp_last? odd member if odd_valid, else even member.
    wire last_in_grp_issue = odd_valid_issue ? ((even_issue + CW'(1)) == grp_last)
                                             : (even_issue == grp_last);
    wire issue_valid = mac_in_valid;

    logic [GW-1:0]                 grp_sr  [0:CSA_LAT-1];
    logic                         last_sr [0:CSA_LAT-1];
    logic                         vld_sr  [0:CSA_LAT-1];
    logic signed [SCALE_WIDTH-1:0] sc_sr  [0:CSA_LAT-1];
    integer s;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (s = 0; s < CSA_LAT; s = s + 1) begin
                grp_sr[s] <= '0; last_sr[s] <= 1'b0; vld_sr[s] <= 1'b0; sc_sr[s] <= '0;
            end
        end else begin
            grp_sr[0]  <= grp_issue;
            last_sr[0] <= last_in_grp_issue;
            vld_sr[0]  <= issue_valid;
            sc_sr[0]   <= scale_q_eff;
            for (s = 1; s < CSA_LAT; s = s + 1) begin
                grp_sr[s] <= grp_sr[s-1]; last_sr[s] <= last_sr[s-1];
                vld_sr[s] <= vld_sr[s-1]; sc_sr[s]   <= sc_sr[s-1];
            end
        end
    end
    wire                          last_now  = last_sr[CSA_LAT-1];
    wire signed [SCALE_WIDTH-1:0] scale_now = sc_sr[CSA_LAT-1];

    // Per-group INT accumulator (fold pair sums), flush*scale at group end.
    logic signed [31:0]          group_acc;
    logic signed [RES_WIDTH-1:0] scaled_acc;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            group_acc  <= '0; scaled_acc <= '0;
        end else if (state == S_IDLE && start) begin
            group_acc  <= '0; scaled_acc <= '0;
        end else if (csa_out_valid) begin
            logic signed [31:0] g_total;
            g_total = group_acc + $signed(csa_pair_sum);
            if (last_now) begin
                scaled_acc <= scaled_acc + ($signed(g_total) * $signed(scale_now));
                group_acc  <= '0;
            end else begin
                group_acc  <= g_total;
            end
        end
    end

    // --- Outputs ---
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_out  <= '0; result_q <= '0; done <= 1'b0;
        end else begin
            done <= (state == S_DONE);
            if (state == S_DONE) begin
                acc_out  <= acc;
                result_q <= scaled_acc;
            end
        end
    end

endmodule

`default_nettype wire
