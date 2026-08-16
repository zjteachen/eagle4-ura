// =====================================================================
// gemv_lane_scaled_stream.sv      (M1b — per-group dequant scale apply)
// =====================================================================
//
// Streaming-interface GEMV lane that adds the REAL GPTQ per-group
// dequant scale to gemv_lane_int_stream_wallace's INT path.
//
// Numerical contract (group_size=GROUP, here 128):
//   raw GEMV (UNSCALED, the existing contract — proven bit-exact vs g6.hex):
//       acc[j] = Σ_k a[k]·(raw_w[k,j] − (qz+1))
//   REAL dequant GEMV (this module, SCALED=1):
//       out[j] = Σ_g scale_q[g,j] · ( Σ_{k∈group g} a[k]·(raw_w[k,j]−(qz+1)) )
//   i.e. the per-group INT partial sum is multiplied by the group's
//   Q1.15 scale on ONE DSP58 (INT mode, group_acc[INT] × scale[16] → 48b)
//   and the scaled products are accumulated across groups.
//
// Reuse, not reinvent:
//   - Same FSM / chunk streaming / Wallace CSA reduce as
//     gemv_lane_int_stream_wallace.sv (the chunk MAC + csa_tree16_int20
//     pipeline is byte-for-byte the same).
//   - The scale multiply mirrors gemv_lane_qscale.sv's DSP58 INT-mode
//     `$signed(acc) * $signed(scale_q)` slot; the only new logic is the
//     per-GROUP boundary detection + scaled accumulation.
//
// Selectable path (SCALED parameter):
//   SCALED=0 : behaves EXACTLY like gemv_lane_int_stream_wallace —
//              acc_out is the unscaled INT32 accumulator, result_q is the
//              same value sign-extended. This keeps g6.hex (tol 0)
//              regression green and proves we didn't break the contract.
//   SCALED=1 : acc_out still carries the unscaled INT32 acc (debug), and
//              result_q carries the scaled fixed-point output. The
//              consumer interprets result_q as having SCALE_FRAC fractional
//              bits (Q1.15 ⇒ 15 frac bits); the lane does NOT renormalize
//              (preserves full precision, same as gemv_lane_qscale).
//
// Scale stream interface (SCALED=1):
//   The caller supplies scale_q valid for the CURRENT group. Group index
//   advances every GROUP/K_PAR chunks. scale_q is driven by the scale-BRAM
//   read in gemv_stage_array_uram (mirrors qw_word streaming). The lane
//   exposes group_idx_out so the caller's scale-BRAM lookahead aligns
//   exactly like the weight-bank chunk_idx_next lookahead.
//
// Latency (start → done):
//   = streaming lane latency + 1 cycle (final scale-accumulate flush)
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module gemv_lane_scaled_stream #(
    parameter int K           = 64,
    parameter int K_PAR       = 16,
    parameter int A_WIDTH     = 16,
    parameter int GROUP       = 128,      // GPTQ group_size
    parameter int SCALE_WIDTH = 16,       // Q1.15 signed scale
    parameter int SCALE_FRAC  = 15,       // fractional bits of the scale
    parameter int RES_WIDTH   = 48,       // scaled fixed-point accumulator width
    parameter bit SCALED      = 1'b1,     // 0 = unscaled INT path (g6.hex regression)
    // WLAT (2026-06-08, PDI#10 D1 — GEMV ACTIVATION COMBREAD stale-by-1 fix,
    // PDI10_D1_REGREAD_CONFIRM): the SLR_SAFE=1 activation seam delivers a_chunk to
    // this lane through ONE MORE register than the weight on silicon — the activation
    // crosses [stage_io_buffer registered BRAM read] + [array forward-seam register]
    // (2 registers from the +1-lead address) while the weight crosses only the
    // [weight_bank registered read] (1 register from its +1-lead address). The net is
    // a_chunk == act[c-1] when the MAC is at chunk c -> W[c]·act[c-1] on every body
    // GEMV. WLAT=1 EQUALIZES the depths by delaying the WEIGHT + the MAC ENABLE
    // (in_valid / group-tag issue) by ONE cycle inside the lane, so weight and the
    // (already +1-deeper) activation arrive together AND the valid window tracks the
    // delayed operands. BIT-EXACT: every (a,w) product and its group tag shift by the
    // SAME 1 cycle, so the per-group INT sum, the scaled accumulate and result_q are
    // identical; the extra cycle is absorbed by S_DRAIN (CSA_LAT+2 margin). Boundary-
    // safe: the FIRST issue is gated by the registered in_valid (no pre-start garbage)
    // and the LAST real chunk's product still lands before drain completes.
    //   WLAT=0 : legacy (weight + activation both 1-deep) — bit+cycle identical to all
    //            prior goldens. Use for any consumer whose activation is NOT on the
    //            2-register SLR_SAFE seam.
    //   WLAT=1 : weight+enable delayed 1 cycle to match the 2-register activation seam.
    parameter bit WLAT        = 1'b0,
    parameter int N_CHUNKS    = K / K_PAR,
    parameter int CW          = (N_CHUNKS <= 1) ? 1 : $clog2(N_CHUNKS + 2),
    // chunks per group, and group-count width
    parameter int CPG         = GROUP / K_PAR,
    parameter int N_GROUPS    = (K + GROUP - 1) / GROUP,
    parameter int GW          = (N_GROUPS <= 1) ? 1 : $clog2(N_GROUPS + 1)
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
    // Per-group dequant scale (Q1.15), held for the CURRENT group
    input  wire signed [SCALE_WIDTH-1:0]   scale_q,

    // ----------------------------------------------------------------
    // G12_K_PATH FIX-1 (per-stage-K chunk count). RUNTIME terminal count
    // for the FSM walk. The lane is ELABORATED for the worst-case K_MAX
    // (N_CHUNKS chunks) so every buffer/width is sized for the max; but
    // the FSM only walks `n_chunks_run` chunks per start. The caller
    // drives this from the per-stage chunk count that already lives in
    // stage_geometry.svh (sg_w_chunks_per_col(stage_sel)), CLAMPED to the
    // elaborated N_CHUNKS so it can never exceed the array depth.
    //
    // BIT-IDENTICAL: the dropped chunks (n_chunks_run .. N_CHUNKS-1) are
    // MAC-by-zero (zero-padded activation beyond a stage's real K), so the
    // accumulator result is unchanged; this can only RELAX timing (fewer
    // cycles). When tied to N_CHUNKS it is exactly the old fixed behavior.
    // Width = CW (same as chunk_idx). 0 is treated as N_CHUNKS (a safe
    // default for any caller that leaves it unconnected/zero).
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

    // Final UNSCALED INT32 accumulator (same value as the stream lane).
    output reg  signed [31:0]              acc_out,
    // Final SCALED fixed-point accumulator (Q with SCALE_FRAC frac bits).
    // When SCALED=0 this is acc_out sign-extended to RES_WIDTH.
    output reg  signed [RES_WIDTH-1:0]     result_q,
    output reg                             done
);

    // Forward declares for FSM
    wire signed [25:0] csa_sum_out;
    wire               csa_out_valid;

    localparam int CSA_LAT = 3;

    // ----------------------------------------------------------------
    // FSM — identical to gemv_lane_int_stream_wallace
    // ----------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE, S_RUN, S_DRAIN, S_DONE } state_t;
    state_t state, state_n;
    logic [CW-1:0] chunk_idx,   chunk_idx_n;
    logic [CW-1:0] drain_count, drain_count_n;
    logic [CW-1:0] outs_recvd,  outs_recvd_n;

    // G12_K_PATH FIX-1: effective terminal chunk count for this start.
    // n_chunks_run==0 (unconnected caller) -> full N_CHUNKS (old behavior).
    // Otherwise the caller-supplied (already-clamped to N_CHUNKS) per-stage
    // count. Latched at start so a mid-walk stage_sel change cannot perturb
    // an in-flight walk (the same self-timing the SLR seam relies on).
    logic [CW-1:0] nc_run_q;
    wire   [CW-1:0] nc_eff = (nc_run_q == '0) ? CW'(N_CHUNKS) : nc_run_q;
    always_ff @(posedge clk) begin
        if (!rst_n)                        nc_run_q <= '0;
        else if (state == S_IDLE && start) nc_run_q <= n_chunks_run;
    end

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

    // Group index = chunk_idx / CPG  (CPG is a power of two for GROUP/K_PAR).
    // Expose current + next for the scale-BRAM 1-cycle lookahead.
    localparam int CPG_BITS = (CPG <= 1) ? 1 : $clog2(CPG);
    wire [CW-1:0] grp_cur_full  = (CPG <= 1) ? chunk_idx   : (chunk_idx   >> CPG_BITS);
    wire [CW-1:0] grp_next_full = (CPG <= 1) ? chunk_idx_n : (chunk_idx_n >> CPG_BITS);
    assign group_idx_out      = grp_cur_full[GW-1:0];
    assign group_idx_next_out = grp_next_full[GW-1:0];

    // ----------------------------------------------------------------
    // WLAT (D1 activation-seam fix): when WLAT=1, delay the WEIGHT word and the
    // MAC-ENABLE / group-tag issue by ONE cycle so they line up with the activation
    // that arrives one register deeper on the SLR_SAFE silicon seam. When WLAT=0
    // these are pass-throughs (bit+cycle identical to the legacy lane).
    // ----------------------------------------------------------------
    wire [31:0]                  qw_word_eff;
    wire                         mac_in_valid;     // CSA in_valid / scaled-issue valid (issue point)
    wire [CW-1:0]                issue_chunk_idx;  // chunk_idx used for group tagging at issue
    // PDI#10 D1 SCALE-ALIGN FIX: scale_q must travel the SAME WLAT delay as the
    // weight + group tag. The original D1 fix delayed qw_word/enable/group-tag
    // (issue_chunk_idx) by 1 cycle under WLAT=1, but left scale_q UNDELAYED — so
    // at a per-group boundary the scale APPLIED to a group's INT sum was the
    // NEXT group's scale (off-by-one-group). The unscaled INT acc stayed bit-
    // exact (no scale), so the bug ONLY corrupted the SCALED result (the G4..G12
    // body taps), which is what regressed the full chain. Delaying scale_q by the
    // SAME 1 cycle re-aligns it with its group tag. Bit+cycle identical when WLAT=0.
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
    // UNSCALED INT32 accumulator (UNCHANGED — drives the regression path)
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
        // group_at_issue = chunk_idx / CPG, captured when in S_RUN. A
        // CSA_LAT-stage shift register carries (group, last_in_group, valid)
        // so it lines up with csa_out_valid.
        localparam int CPG_B = CPG_BITS;

        // WLAT-aware issue: use the (optionally 1-cycle-delayed) issue_chunk_idx +
        // mac_in_valid so the group tag tracks the delayed weight/enable when WLAT=1
        // (and is bit+cycle identical to chunk_idx/(state==S_RUN) when WLAT=0).
        wire [CW-1:0] grp_issue_full = (CPG <= 1) ? issue_chunk_idx : (issue_chunk_idx >> CPG_B);
        wire [GW-1:0] grp_issue      = grp_issue_full[GW-1:0];
        // last chunk in this group? (issue_chunk_idx mod CPG == CPG-1)
        wire last_in_grp_issue = (CPG <= 1) ? 1'b1
                                            : (issue_chunk_idx[CPG_B-1:0] == (CPG-1));
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
