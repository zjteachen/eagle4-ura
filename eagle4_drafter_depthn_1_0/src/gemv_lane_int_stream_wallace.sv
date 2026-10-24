// =====================================================================
// gemv_lane_int_stream_wallace.sv
// V9.5 — streaming-interface variant of gemv_lane_int_pilot_wallace.
// =====================================================================
//
// Fixes the K=3072 scaling problem from V9 ooc-synth (flat 49-kbit
// a_pack bus + chunk_idx mux blew the lane up to 19,876 LUT and broke
// 300 MHz).  Same Wallace pipeline downstream; the only change is the
// activation interface:
//
//   OLD: input  a_pack [K*A_WIDTH-1:0]  (flat, K can be big)
//        internal mux: a_lane = a_pack[(chunk_idx*K_PAR + i)*A_WIDTH ...]
//
//   NEW: input  a_chunk[K_PAR*A_WIDTH-1:0]  (one chunk, 256 bits)
//        no internal mux — caller has already selected the right chunk
//        based on the lane's exposed chunk_idx_out/chunk_idx_valid.
//        The lane FSM drives chunk_idx; the caller responds.
//
// This matches the V6 OS weight-bus pattern (caller drives the
// right chunk per cycle).  At LANES=128 production, an H-tree
// activation broadcast network supplies the same a_chunk to all lanes
// every cycle.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module gemv_lane_int_stream_wallace #(
    parameter int K       = 64,
    parameter int K_PAR   = 16,
    parameter int A_WIDTH = 16,
    parameter int N_CHUNKS = K / K_PAR,
    parameter int CW      = (N_CHUNKS <= 1) ? 1 : $clog2(N_CHUNKS + 2)
) (
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           start,

    // Streaming activation: ONE chunk (K_PAR × A_WIDTH bits) per cycle
    input  wire signed [K_PAR*A_WIDTH-1:0] a_chunk,
    // Streaming qweight: ONE int32 word per cycle (16 raw 2-bit weights)
    input  wire        [31:0]              qw_word,
    // Per-group qzero (one 2-bit value)
    input  wire        [1:0]               qz_raw,
    // Runtime terminal chunk count, latched at start; zero selects N_CHUNKS.
    // The shared controller bounds this to the elaborated capacity.
    input  wire        [CW-1:0]            n_chunks_run,

    // FSM exposes the chunk index it expects; caller drives a_chunk and
    // qw_word consistent with this chunk.
    output wire        [CW-1:0]            chunk_idx_out,
    output wire                            chunk_idx_valid,
    // Next-state chunk_idx for URAM lookahead (use as weight-bank address
    // at cycle T → URAM data at T+1 aligns with lane's chunk_idx@T+1).
    output wire        [CW-1:0]            chunk_idx_next_out,

    // Final INT32 accumulator
    output reg  signed [31:0]              acc_out,
    output reg                             done
);

    // Forward declares for FSM
    wire signed [25:0] csa_sum_out;
    wire               csa_out_valid;

    // DEBUG: dump elaborated parameters at sim time 0 (also gated — 128 lanes
    // print 128 banner lines into the integrated-top log otherwise).
`ifdef GEMV_LANE_DEBUG
    initial $display("[%m] elaborated K=%0d K_PAR=%0d N_CHUNKS=%0d CW=%0d A_WIDTH=%0d",
                     K, K_PAR, N_CHUNKS, CW, A_WIDTH);
`endif

    // FSM. Note: chunk_idx_n (next-state, combinational) is exposed via
    // chunk_idx_next_out so that URAM-backed weight banks can read with
    // 1-cycle lookahead — URAM data at cycle T+1 then equals qw[chunk_idx@T+1],
    // perfectly aligning with the lane's combinational consumption.
    typedef enum logic [1:0] { S_IDLE, S_RUN, S_DRAIN, S_DONE } state_t;
    state_t state, state_n;
    logic [CW-1:0] chunk_idx,   chunk_idx_n;
    logic [CW-1:0] drain_count, drain_count_n;
    logic [CW-1:0] outs_recvd,  outs_recvd_n;

    localparam int CSA_LAT = 3;

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
    assign chunk_idx_valid = (state == S_RUN);

    // Dequant — qw_word is already current-chunk consistent
    wire [31:0] qz_word_synth = {30'b0, qz_raw};
    wire [47:0] w_s_packed;
    gptq2_dequant16 u_dequant (
        .qw_word    (qw_word),
        .qz_word    (qz_word_synth),
        .n_idx      (4'd0),
        .w_s_packed (w_s_packed)
    );

    // 16 × MAC — activations come straight from a_chunk (no mux!)
    logic signed [19:0] prods [0:K_PAR-1];
    for (genvar i = 0; i < K_PAR; i = i + 1) begin : g_mac
        wire signed [A_WIDTH-1:0] a_lane    = a_chunk[i*A_WIDTH +: A_WIDTH];
        wire signed [2:0]         w_s_lane  = $signed(w_s_packed[i*3 +: 3]);
        assign prods[i] = $signed(a_lane) * $signed(w_s_lane);
    end

    // Wallace CSA tree
    csa_tree16_int20 u_csa (
        .clk(clk), .rst_n(rst_n),
        .in_valid(chunk_idx_valid),
        .prods(prods),
        .sum_out(csa_sum_out),
        .out_valid(csa_out_valid)
    );

    // INT32 accumulator
    logic signed [31:0] acc;
    always_ff @(posedge clk) begin
        if (!rst_n)                            acc <= '0;
        else if (state == S_IDLE && start)     acc <= '0;
        else if (csa_out_valid)                acc <= acc + $signed(csa_sum_out);
    end

    // Output latch
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_out <= '0; done <= 1'b0;
        end else begin
            done <= (state == S_DONE);
            if (state == S_DONE) acc_out <= acc;
        end
    end

    // DEBUG: print FSM-finish stats. Gated behind GEMV_LANE_DEBUG — at
    // LANES=128 this per-DONE $display floods the integrated-top xsim log
    // (128 lanes × every stage), making the bisection harness log unreadable.
    // Define +define+GEMV_LANE_DEBUG at xvlog time to re-enable per-lane prints.
`ifdef GEMV_LANE_DEBUG
    always_ff @(posedge clk) begin
        if (state == S_DONE)
            $display("[t=%0t] DONE: outs_recvd=%0d chunk_idx=%0d acc=%0d",
                     $time, outs_recvd, chunk_idx, acc);
    end
`endif

endmodule

`default_nettype wire
