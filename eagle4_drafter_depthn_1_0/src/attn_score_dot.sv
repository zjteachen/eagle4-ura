// =====================================================================
// attn_score_dot.sv
// One attention score: sum_i (Q[i] * K[i]) over D_HEAD elements.
// =====================================================================
//
// Computes the Q · K[p] dot product used in attention scoring.
//   - Q is the current-token query (D_HEAD fp16/Q4.12 values)
//   - K[p] is the cached key for one position (D_HEAD elements)
//   - Output: one INT32 score (signed accumulator before scale)
//
// Pilot: D_HEAD=16 → one Wallace CSA tree chunk (combinational feed).
// Production: D_HEAD=128 → 8 chunks of 16 each, summed serially over 8
// cycles into a wider accumulator (chunk-serial FSM).
//
// Reuses csa_tree16_int20 from the GEMV path. Same primitive,
// different data source.
//
// Parameterized by N_CHUNKS_D = D_HEAD/16:
//   N_CHUNKS_D == 1  → pilot path, latency = 4 cycles (unchanged)
//   N_CHUNKS_D >  1  → chunk-serial FSM path
//
// Bit-exactness note: per-element 32-bit products are truncated to
// INT20 (prod_full[19:0]) EXACTLY as the pilot does, then 16 are
// reduced per chunk by csa_tree16_int20 → INT26 sum, and the
// N_CHUNKS_D chunk-sums are accumulated in a 32-bit accumulator.
// This is bit-identical to a flat sum of all D_HEAD truncated products
// (CSA + integer add are both exact).
//
// Latency (production, N_CHUNKS_D=8 for D_HEAD=128):
//   1 (start latch / first chunk presented)
//   + 8 (feed 8 chunks back-to-back; csa II=1)
//   + 2 (csa internal pipeline tail: last chunk's sum lands 3 cycles
//        after its feed, accumulator absorbs it)
//   + 1 (final accumulator → score_out latch)
//   = 12 cycles per score   (was 4 for the pilot)
//   *** REPORT: SCORE_LATENCY in attention_head_assembly must become 12
//       and flash_attn_head V_DELAY ripples from this (orchestrator). ***
//
// Per attention forward at ctx_len=2048, this fires 2048 times per
// head × 24 heads = ~50K invocations.  Production fans out per-head
// (compute all 24 heads' scores in parallel) and time-multiplexes
// within a head over ctx_len positions.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module attn_score_dot #(
    parameter int D_HEAD  = 16,      // pilot; production = 128
    parameter int A_WIDTH = 16
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,
    input  wire signed [A_WIDTH*D_HEAD-1:0]  q_pack,        // D_HEAD × INT16
    input  wire signed [A_WIDTH*D_HEAD-1:0]  k_pack,        // D_HEAD × INT16
    output reg  signed [31:0]                score_out,
    output reg                               done
);

    localparam int N_CHUNKS_D = D_HEAD / 16;     // 1 (pilot) .. 8 (D_HEAD=128)

    initial begin
        if ((D_HEAD % 16) != 0)
            $display("ERROR: attn_score_dot needs D_HEAD divisible by 16; got %0d", D_HEAD);
    end

    generate
    if (N_CHUNKS_D == 1) begin : g_pilot
        // ============================================================
        // PILOT PATH (D_HEAD=16) — single combinational CSA feed.
        // Behaviour identical to the original module: latency 4 cycles.
        // ============================================================
        // D_HEAD parallel INT16 × INT16 multiplies (DSP58 INT mode).
        // Each product truncated to 20-bit to feed csa_tree16_int20.
        logic signed [19:0] prods [0:D_HEAD-1];
        for (genvar i = 0; i < D_HEAD; i = i + 1) begin : g_mac
            wire signed [A_WIDTH-1:0] q_lane = q_pack[i*A_WIDTH +: A_WIDTH];
            wire signed [A_WIDTH-1:0] k_lane = k_pack[i*A_WIDTH +: A_WIDTH];
            wire signed [31:0] prod_full = $signed(q_lane) * $signed(k_lane);
            // FULL product (no normalize shift) — softmax max-subtract makes
            // the absolute scale irrelevant; truncate-to-fit INT20.
            assign prods[i] = prod_full[19:0];
        end

        wire        csa_in_valid = start;
        wire signed [25:0] csa_sum;
        wire        csa_valid;

        csa_tree16_int20 u_csa (
            .clk(clk), .rst_n(rst_n),
            .in_valid(csa_in_valid),
            .prods(prods),
            .sum_out(csa_sum),
            .out_valid(csa_valid)
        );

        always_ff @(posedge clk) begin
            if (!rst_n) begin
                score_out <= '0;
                done      <= 1'b0;
            end else begin
                done <= csa_valid;
                if (csa_valid) score_out <= $signed({{6{csa_sum[25]}}, csa_sum});
            end
        end

    end else begin : g_serial
        // ============================================================
        // PRODUCTION PATH (D_HEAD=128) — chunk-serial FSM.
        // Feed one 16-product chunk per cycle into csa_tree16_int28,
        // accumulate the N_CHUNKS_D INT32 chunk-sums into a 32-bit acc.
        //
        // PRECISION FIX (2026-06-20, depth-N attention-score blocker):
        //   The OLD path truncated every q*k product to INT20 (pf[19:0],
        //   real +/-0.5 at frac20) and used csa_tree16_int20.  At depth-1
        //   (ctx_len>=2, the first forward where the score weights the
        //   output) the per-element q*k products reach +/-41 real -> 231/256
        //   OVERFLOW INT20, sign-wrap, and shrink the true Q.K dot ~5x
        //   (RTL 7.79 vs fp32-implied 38.15) -> the softmax weights the wrong
        //   KV position -> wrong depth-1 child token (28051 vs golden 315).
        //   Measured: ctxlen2_probe2_d2s42 + scripts/attn_score_decomp.py.
        //   FIX = retain each product to INT28 (real +/-128 at frac20, 3.1x
        //   headroom over the 41.2 max) + the widened csa_tree16_int28.
        //   ctx_len=1 (seed) output is V[0] regardless of score -> byte-safe.
        // ============================================================
        localparam int CW = (N_CHUNKS_D <= 1) ? 1 : $clog2(N_CHUNKS_D);

        // ----- chunk multiplexer into the CSA -----
        // chunk_idx selects which 16 of the D_HEAD products feed the tree.
        logic [CW-1:0]      chunk_idx;          // exact width: which chunk to feed
        logic               feeding;            // high while feeding chunks
        logic signed [31:0] chunk_prods [0:15]; // the 16 products of the selected chunk (INT32; was INT28)

        // ----- the single CSA tree (one instance, fed serially) -----
        wire signed [31:0] csa_sum;
        wire               csa_valid;

        csa_tree16_int28 u_csa (
            .clk(clk), .rst_n(rst_n),
            .in_valid(feeding),
            .prods(chunk_prods),
            .sum_out(csa_sum),
            .out_valid(csa_valid)
        );

        // ----- FSM: walk N_CHUNKS_D chunks, then accumulate the sums -----
        typedef enum logic [1:0] {
            S_IDLE = 2'd0,
            S_FEED = 2'd1,
            S_DRAIN= 2'd2,
            S_DONE = 2'd3
        } sd_state_e;

        sd_state_e         st;
        logic [CW:0]       feed_cnt;            // counts chunks fed (0..N_CHUNKS_D)
        logic [CW:0]       collect_cnt;         // counts chunk-sums accumulated
        logic signed [31:0] acc;                // wide enough for 8 × INT32 chunk-sums
        logic signed [A_WIDTH*D_HEAD-1:0] q_lat, k_lat;  // latched operands

        // D_HEAD parallel products over the LATCHED operands. Latching makes
        // the module self-contained across the multi-cycle walk (the caller
        // may change q_pack/k_pack while we are still summing).
        // PRECISION FIX (2026-06-20): retain each product to INT28 (was INT20).
        // INT32-WIDEN (2026-06-30, depth-2 v=64 residual root cause): the
        // "41.2 real max" was UNDER-measured. At depth-1 the root-key dot for
        // heads h11/h17 reaches |q*k| = 137.9 real at frac20 -> OVERFLOWS
        // INT28 (2^27/2^20=128), sign-wraps one element, flips pos0's dot
        // -204 -> +52, softmax spuriously picks pos0 -> out_vec=V_pos0 ->
        // wrong child token 8545 vs golden 315. Only h11/h17 have a product
        // >128 (matches the 2 bad heads exactly). FIX = retain the FULL INT32
        // product (measured max chunk-sum 298 real / max dot 600 real both fit
        // INT32; the CSA already reduces in INT32). Non-overflowing scores are
        // BIT-IDENTICAL, so depth-0 (ctx_len=1 -> out=V[0]) is byte-safe.
        // Evidence: reports/results/DEPTHN_SCORE_INT28_OVERFLOW_FIX.md.
        logic signed [31:0] prods_lat [0:D_HEAD-1];
        for (genvar i = 0; i < D_HEAD; i = i + 1) begin : g_mac_lat
            wire signed [A_WIDTH-1:0] ql = q_lat[i*A_WIDTH +: A_WIDTH];
            wire signed [A_WIDTH-1:0] kl = k_lat[i*A_WIDTH +: A_WIDTH];
            wire signed [31:0] pf = $signed(ql) * $signed(kl);
            assign prods_lat[i] = pf;   // full INT32 (was pf[27:0])
        end

        // chunk_prods is the single mux of the selected chunk's 16 products.
        always_comb begin
            for (int j = 0; j < 16; j++)
                chunk_prods[j] = prods_lat[chunk_idx*16 + j];
        end

        always_ff @(posedge clk) begin
            if (!rst_n) begin
                st          <= S_IDLE;
                chunk_idx   <= '0;
                feeding     <= 1'b0;
                feed_cnt    <= '0;
                collect_cnt <= '0;
                acc         <= '0;
                score_out   <= '0;
                done        <= 1'b0;
                q_lat       <= '0;
                k_lat       <= '0;
            end else begin
                done    <= 1'b0;
                feeding <= 1'b0;

                // Accumulate any chunk-sum that pops out of the CSA tree.
                // csa_sum is now INT32 (csa_tree16_int28); no sign-extend needed.
                if (csa_valid && collect_cnt < N_CHUNKS_D) begin
                    acc         <= acc + csa_sum;
                    collect_cnt <= collect_cnt + 1'b1;
                end

                case (st)
                    S_IDLE: begin
                        if (start) begin
                            q_lat       <= q_pack;
                            k_lat       <= k_pack;
                            acc         <= '0;
                            chunk_idx   <= '0;
                            feed_cnt    <= '0;
                            collect_cnt <= '0;
                            feeding     <= 1'b1;       // present chunk 0 next edge
                            st          <= S_FEED;
                        end
                    end

                    S_FEED: begin
                        // We are presenting chunk_idx this cycle (feeding high).
                        feeding <= 1'b1;
                        if (feed_cnt == N_CHUNKS_D - 1) begin
                            // last chunk presented this cycle
                            feeding  <= 1'b1;
                            feed_cnt <= feed_cnt + 1'b1;
                            st       <= S_DRAIN;
                        end else begin
                            chunk_idx <= chunk_idx + 1'b1;
                            feed_cnt  <= feed_cnt + 1'b1;
                        end
                    end

                    S_DRAIN: begin
                        // wait until all chunk-sums have been accumulated
                        feeding <= 1'b0;
                        if (collect_cnt == N_CHUNKS_D) begin
                            score_out <= acc;
                            done      <= 1'b1;
                            st        <= S_DONE;
                        end
                    end

                    S_DONE: begin
                        st <= S_IDLE;
                    end

                    default: st <= S_IDLE;
                endcase
            end
        end

    end
    endgenerate

endmodule

`default_nettype wire
