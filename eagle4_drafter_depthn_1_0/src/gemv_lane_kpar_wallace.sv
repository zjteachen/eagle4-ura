// =====================================================================
// gemv_lane_kpar_wallace.sv  (explore/kpar — perf-headroom study)
// =====================================================================
// Parametric-K_PAR streaming GEMV lane. Generalises
// gemv_lane_int_stream_wallace (which is hardwired to K_PAR=16 via a
// single csa_tree16_int20) to K_PAR in {16, 32, 64} by tiling the
// reuse-blessed primitives:
//
//   * gptq2_dequant16  — one per 16-lane SUBGROUP (K_PAR/16 instances)
//   * csa_tree16_int20 — one per 16-lane SUBGROUP (K_PAR/16 instances)
//   * partial-sum CSA  — combinational 3:2 tree that reduces the
//                        N_SG=K_PAR/16 sub-tree outputs into one INT32,
//                        registered (1 extra cycle) so the deeper
//                        reduction is pipelined, not flattened into the
//                        csa_tree16 critical path.
//
// Per cycle the lane consumes K_PAR activations (one chunk) and produces
// no output; it accumulates N_CHUNKS = K/K_PAR partial sums.  Latency to
// final acc = N_CHUNKS + pipeline depth.  Cycles/column = K/K_PAR — the
// quantity the K_PAR sweep is measuring.
//
// Bit-exactness: the integer sum is identical for every K_PAR (only the
// chunking changes), so all three configs must match the K=3072 golden
// acc_int32.hex.
//
// Streaming interface (same contract as the K_PAR=16 stream lane):
//   a_chunk  — K_PAR activations (K_PAR*A_WIDTH bits) for this chunk
//   qw_word  — N_SG int32 words (one per 16-lane subgroup), packed
//   qz_raw   — per-group 2-bit zero (broadcast to all subgroups; the
//              column shares one group within a 128-row group_size)
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module gemv_lane_kpar_wallace #(
    parameter int K        = 3072,
    parameter int K_PAR    = 16,
    parameter int A_WIDTH  = 16,
    parameter int N_SG      = K_PAR / 16,        // 16-lane subgroups
    parameter int N_CHUNKS  = K / K_PAR,
    parameter int CW        = (N_CHUNKS <= 1) ? 1 : $clog2(N_CHUNKS + 2)
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,

    // Streaming activation: ONE chunk (K_PAR × A_WIDTH bits) per cycle
    input  wire signed [K_PAR*A_WIDTH-1:0]   a_chunk,
    // Streaming qweight: N_SG int32 words per cycle (16 raw 2-bit weights each)
    input  wire        [N_SG*32-1:0]         qw_word,
    // Per-group qzero (one 2-bit value, shared across subgroups)
    input  wire        [1:0]                 qz_raw,
    // Runtime terminal row count, latched at start; zero selects N_CHUNKS.
    input  wire        [CW-1:0]              n_chunks_run,

    output wire        [CW-1:0]              chunk_idx_out,
    output wire                              chunk_idx_valid,
    output wire        [CW-1:0]              chunk_idx_next_out,

    output reg  signed [31:0]                acc_out,
    output reg                               done
);

`ifdef GEMV_LANE_DEBUG
    initial $display("[%m] elaborated K=%0d K_PAR=%0d N_SG=%0d N_CHUNKS=%0d CW=%0d",
                     K, K_PAR, N_SG, N_CHUNKS, CW);
`endif

    // ----------------------------------------------------------------
    // FSM — identical structure to the K_PAR=16 stream lane.
    // The csa_tree16 latency is 3; the partial-sum reduce adds 1 reg
    // stage, so the drain depth grows by 1 when N_SG > 1.
    // ----------------------------------------------------------------
    localparam int CSA_LAT  = 3;
    localparam int PSUM_LAT = (N_SG > 1) ? 1 : 0;
    localparam int PIPE_LAT = CSA_LAT + PSUM_LAT;

    typedef enum logic [1:0] { S_IDLE, S_RUN, S_DRAIN, S_DONE } state_t;
    state_t state, state_n;
    logic [CW-1:0] chunk_idx,   chunk_idx_n;
    logic [CW-1:0] drain_count, drain_count_n;
    logic [CW-1:0] outs_recvd,  outs_recvd_n;

    wire psum_valid;   // valid of the final (post-partial-sum) result

    // Latch the runtime terminal count (zero selects the elaborated maximum).
    logic [CW-1:0] nc_run_q;
    wire  [CW-1:0] nc_eff = (nc_run_q == '0) ? CW'(N_CHUNKS) : nc_run_q;
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
                if (psum_valid) outs_recvd_n = outs_recvd + 1'b1;
            end
            S_DRAIN: begin
                if (psum_valid) outs_recvd_n = outs_recvd + 1'b1;
                if (outs_recvd == nc_eff) state_n = S_DONE;
                else if (drain_count >= PIPE_LAT + 2) state_n = S_DONE;
                drain_count_n = drain_count + 1'b1;
            end
            S_DONE: state_n = S_IDLE;
            default: state_n = S_IDLE;
        endcase
    end

    assign chunk_idx_out      = chunk_idx;
    assign chunk_idx_next_out = chunk_idx_n;
    assign chunk_idx_valid    = (state == S_RUN);

    // ----------------------------------------------------------------
    // N_SG subgroups: each does gptq2_dequant16 → 16 INT20 products →
    // csa_tree16_int20 → INT26 sub-sum.
    // ----------------------------------------------------------------
    wire [31:0] qz_word_synth = {30'b0, qz_raw};

    wire signed [25:0] sg_sum   [0:N_SG-1];
    wire               sg_valid [0:N_SG-1];

    genvar sg, i;
    generate
        for (sg = 0; sg < N_SG; sg = sg + 1) begin : g_sg
            wire [31:0]  qw_sub = qw_word[sg*32 +: 32];
            wire [47:0]  w_s_packed;

            gptq2_dequant16 u_dequant (
                .qw_word    (qw_sub),
                .qz_word    (qz_word_synth),
                .n_idx      (4'd0),
                .w_s_packed (w_s_packed)
            );

            logic signed [19:0] prods [0:15];
            for (i = 0; i < 16; i = i + 1) begin : g_mac
                wire signed [A_WIDTH-1:0] a_lane   =
                        a_chunk[(sg*16 + i)*A_WIDTH +: A_WIDTH];
                wire signed [2:0]         w_s_lane =
                        $signed(w_s_packed[i*3 +: 3]);
                assign prods[i] = $signed(a_lane) * $signed(w_s_lane);
            end

            csa_tree16_int20 u_csa (
                .clk(clk), .rst_n(rst_n),
                .in_valid(chunk_idx_valid),
                .prods(prods),
                .sum_out(sg_sum[sg]),
                .out_valid(sg_valid[sg])
            );
        end
    endgenerate

    // ----------------------------------------------------------------
    // Partial-sum reduction over the N_SG sub-sums.
    //   N_SG=1 (K_PAR=16): pass-through, no extra reg.
    //   N_SG=2 (K_PAR=32): one add, registered.
    //   N_SG=4 (K_PAR=64): one 3:2 CSA layer + final CPA, registered.
    // Registered → keeps the deepening reduction OUT of the csa_tree16
    // critical path (the WNS question the spec asks).
    // ----------------------------------------------------------------
    logic signed [31:0] psum_comb;
    always_comb begin
        psum_comb = '0;
        for (int s = 0; s < N_SG; s = s + 1)
            psum_comb = psum_comb + $signed(sg_sum[s]);
    end

    logic signed [31:0] psum_reg;
    logic               psum_valid_reg;

    generate
        if (N_SG > 1) begin : g_psum_reg
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    psum_reg <= '0; psum_valid_reg <= 1'b0;
                end else begin
                    psum_reg       <= psum_comb;
                    psum_valid_reg <= sg_valid[0];   // all subgroups in lockstep
                end
            end
            assign psum_valid = psum_valid_reg;
        end else begin : g_psum_pass
            assign psum_reg   = psum_comb;       // unused reg tied off below
            assign psum_valid = sg_valid[0];
        end
    endgenerate

    wire signed [31:0] psum_now = (N_SG > 1) ? psum_reg : psum_comb;

    // ----------------------------------------------------------------
    // INT32 accumulator
    // ----------------------------------------------------------------
    logic signed [31:0] acc;
    always_ff @(posedge clk) begin
        if (!rst_n)                        acc <= '0;
        else if (state == S_IDLE && start) acc <= '0;
        else if (psum_valid)               acc <= acc + psum_now;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_out <= '0; done <= 1'b0;
        end else begin
            done <= (state == S_DONE);
            if (state == S_DONE) acc_out <= acc;
        end
    end

endmodule

`default_nettype wire
