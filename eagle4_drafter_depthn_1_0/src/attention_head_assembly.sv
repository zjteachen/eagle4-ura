// =====================================================================
// attention_head_assembly.sv
// One per-head attention FSM — composes the 5 attn primitives + a small
// exp-value buffer. KV storage is EXTERNAL (driven through k_vec_in /
// v_vec_in ports) so the same assembly can target either kv_head_uram
// (L5) or a future wide-row BRAM (L8 fused dataflow).
// =====================================================================
//
// Pipeline (3-pass softmax — L7 will later add an online-softmax variant
// that fuses passes 1+2):
//
//   PHASE_SCORE_MAX (ctx_len cycles)
//     For p = 0..ctx_len-1:
//       request K[p] via kv_rd_pos; receive k_vec_in 1 cycle later;
//       compute score_p = dot(q, K[p]) via attn_score_dot;
//       update running_max via score_max_tracker.
//
//   PHASE_DRAIN_MAX (SCORE_LATENCY cycles)
//     wait for the last score to settle through the attn_score_dot
//     pipeline; running_max latched.
//
//   PHASE_EXP_SUM (ctx_len cycles)
//     For p = 0..ctx_len-1:
//       request K[p] again; receive k_vec_in;
//       compute score_p = dot(q, K[p]);
//       feed score_p + running_max into softmax_exp_sum, which produces
//         exp_val_p and accumulates sum_so_far;
//       buffer exp_val_p in exp_buf[p] for the combine pass.
//
//   PHASE_DRAIN_EXP (SCORE_LATENCY cycles)
//     wait for the last exp_val to land in exp_buf; sum_so_far settled.
//
//   PHASE_INV (~25 cycles)
//     pulse inv_q12.start; wait for inv_q12.done → inv_sum (Q4.12).
//
//   PHASE_COMBINE (ctx_len cycles)
//     For p = 0..ctx_len-1:
//       request V[p] via kv_rd_pos; receive v_vec_in;
//       compute attn_q12 = (exp_buf[p] * inv_sum) >> 12;
//       feed attn_q12 + v_vec_in into attn_v_combine, which accumulates.
//
//   PHASE_DONE — out_vec valid; pulse `done`.
//
// External KV read protocol:
//   kv_rd_pos / kv_rd_kind / kv_rd_en : request driven combinationally
//                                       from this module.
//                                       kv_rd_kind: 0=K, 1=V.
//   k_vec_in / v_vec_in : caller drives the D_HEAD-wide vector on the
//                         NEXT cycle (single-cycle RAM-style latency).
//
// Caller responsibilities (in L5 wiring):
//   - Provide ctx_len from control_regs.
//   - Tie the caller's KV storage to the kv_rd_* + k_vec_in / v_vec_in
//     ports. For multi-head, instantiate this module per head and arbitrate
//     the KV ports OR provide per-head KV memories.
//
// Multi-driver-safe: out_vec is the unbroken output of attn_v_combine.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module attention_head_assembly #(
    parameter int D_HEAD        = 16,
    parameter int A_WIDTH       = 16,
    parameter int CTX_MAX       = 32,         // pilot; production wires to 2048
    // attn_score_dot latency (start-sampled → done-sampled).
    //   D_HEAD=16  (pilot, II=1 streaming) : 4
    //   D_HEAD=128 (chunk-serial, II=8 FSM): 13   <-- measured 2026-05-22
    // The score collection is now HANDSHAKE-driven (waits on score_done),
    // so SCORE_LATENCY is informational and used only for sizing; it no
    // longer gates correctness. Kept as a parameter for the orchestrator.
    parameter int SCORE_LATENCY = 4,
    // -----------------------------------------------------------------
    // SCORE_NORM_SHR — score normalization for the softmax seam (ATTN-FULL).
    // attn_score_dot emits the RAW Q·K accumulator (sum of frac×frac element
    // products, NOT >>2·frac normalized). The exp_pwl_q12 LUT expects its
    // delta in Q4.12 (one LUT index ≈ 1024 ≈ exp(-0.25)). Feeding the raw
    // delta makes |score−max| huge ⇒ exp collapses to 1 ⇒ softmax degenerates
    // to HARDMAX (argmax). To produce a FAITHFUL soft softmax matching the
    // fp32 reference (which applies 1/sqrt(d_head)), right-shift the score
    // by SCORE_NORM_SHR before the max-tracker + exp seam. At D_HEAD=128 with
    // act_frac=10 inputs the raw score is real·2^(2·10), and the reference
    // scale is real/sqrt(128); the matching shift is ≈ 2·frac − 12 + log2(sqrt(d))
    // ≈ 8 + ~3.5 ⇒ ~11. SHR has NO effect at ctx_len=1 (score==max ⇒ delta=0
    // ⇒ exp=4096 regardless) so the depth-0 V-passthrough stays BIT-EXACT.
    // Default 0 = legacy pilot behavior (hardmax); set to ~11 for ctx_len>1.
    // -----------------------------------------------------------------
    parameter int SCORE_NORM_SHR = 0,
    // ---------------------------------------------------------------
    // DEPTH_TEMP_FIX — the 1/sqrt(2) softmax-temperature correction (F1), done
    // HERE as a multiply.  DEFAULT FLIPPED 1 -> 0 on 2026-07-23 (F1+F2 commit).
    //
    // *** DO NOT SET THIS BACK TO 1 WITHOUT READING THIS. ***
    // F1 already ships, folded into the exp table values in exp_pwl_q12.sv
    // (EXP_PWL_LUT[i] = 4096*exp(-(i*2^7/2^12)/sqrt(2))).  Enabling BOTH applies
    // 1/sqrt(2) TWICE -> an effective temperature of 1/(sqrt(128)*sqrt(2)) = 1/16,
    // i.e. 1.414x TOO FLAT — strictly worse than the original bug.
    // scripts/ctxn_attn_numerics_model.py section 6 has a mechanical tooth that
    // RED-fires on exactly that combination.
    //
    // The table fold was chosen over this multiply because it measures better and
    // costs less: ctx=4 L1 vs fp32 attn.cpp 0.0369 (this multiply, which also
    // COARSENS the exp grid to a 0.25 true-exponent step) vs 0.0033 (fold, 0.0221
    // step); zero multipliers instead of one 32x15 per engine (6 at HEAD_PAR=6) on
    // the score->max_tracker path; and score_norm stays BIT-IDENTICAL, so
    // score_max_tracker / score_buf / the max compare are untouched.
    // Full argument + numbers: reports/results/ATTN_F1F2_SOFTMAX_SHARPNESS.md
    //
    // Kept (not deleted) so the multiply form stays available as an A/B arm:
    // override to 1 ONLY together with reverting the sqrt(2) out of the table.
    // ---------------------------------------------------------------
    parameter int DEPTH_TEMP_FIX = 0
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // ----- control -----
    input  wire                              start,
    input  wire [10:0]                       ctx_len,           // # of positions to attend over
    input  wire signed [A_WIDTH*D_HEAD-1:0]  q_vec,             // current query (latched on start)

    // ----- external KV read port -----
    output logic                             kv_rd_en,
    output logic                             kv_rd_kind,        // 0=K, 1=V
    output logic [10:0]                      kv_rd_pos,
    input  wire  signed [A_WIDTH*D_HEAD-1:0] k_vec_in,          // valid one cycle after rd_en
    input  wire  signed [A_WIDTH*D_HEAD-1:0] v_vec_in,

    // ----- result -----
    output wire  signed [32*D_HEAD-1:0]      out_vec,
    output logic                             done
);

    localparam int CTX_W = $clog2(CTX_MAX);

    // Score collection is handshake-driven (per-position):
    //   *_RD    — issue kv_rd_en for pos; k_vec_in arrives next cycle
    //   *_START — assert score_start (k_vec_in now valid); FSM moves on
    //   *_WAIT  — wait for score_done, then consume the score, advance pos
    // This works for BOTH the pilot II=1 score (done in 4 cyc) and the
    // production chunk-serial II=8 score (done in 13 cyc) — no streaming
    // assumption.
    typedef enum logic [4:0] {
        S_IDLE        = 5'd0,
        S_MAX_RD      = 5'd1,
        S_MAX_START   = 5'd2,
        S_MAX_WAIT    = 5'd3,
        S_EXP_RD      = 5'd4,
        S_EXP_START   = 5'd5,
        S_EXP_WAIT    = 5'd6,
        S_EXP_BUF     = 5'd7,
        S_INV         = 5'd8,
        S_COMB_RD     = 5'd9,
        S_COMB_WAIT   = 5'd10,
        S_COMB_MAC    = 5'd11,
        S_COMB_MAC2   = 5'd14,   // TIMING-FIX: registered-product commit cycle
        S_DRAIN_COMB  = 5'd12,
        S_DONE        = 5'd13
    } state_e;

    state_e state;
    logic [10:0]      pos_req;          // position currently being processed
    logic [10:0]      ctx_len_latched;
    logic signed [A_WIDTH*D_HEAD-1:0] q_vec_latched;

    // -----------------------------------------------------------------
    // Pass-through to attn_score_dot (handshake: start pulse, wait done).
    // -----------------------------------------------------------------
    logic                       score_start;
    logic signed [31:0]         score_out;
    logic                       score_done;

    attn_score_dot #(.D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH)) u_score (
        .clk(clk), .rst_n(rst_n),
        .start(score_start),
        .q_pack(q_vec_latched),
        .k_pack(k_vec_in),                  // direct from external KV bus
        .score_out(score_out),
        .done(score_done)
    );

    // Score normalization for the softmax seam (ATTN-FULL). Arithmetic
    // right-shift the raw Q·K score into the exp_pwl_q12 Q4.12 delta range so
    // the INT softmax is SOFT (not hardmax). At ctx_len=1 this is a no-op
    // (single score == max ⇒ delta 0). SCORE_NORM_SHR=0 = legacy pilot.
    localparam int TEMP_MUL = 181;
    localparam int TEMP_SHR = 19;
    wire signed [47:0] score_mul = $signed(score_out) * TEMP_MUL;
    wire signed [31:0] score_norm =
          (DEPTH_TEMP_FIX != 0) ? $signed(score_mul >>> TEMP_SHR)
        : (SCORE_NORM_SHR == 0) ? score_out
        :                         $signed(score_out >>> SCORE_NORM_SHR);

    // -----------------------------------------------------------------
    // score_max_tracker (pass 1)
    // -----------------------------------------------------------------
    logic                       max_clear;
    logic                       max_valid;
    wire signed [31:0]          running_max;

    score_max_tracker u_max (
        .clk(clk), .rst_n(rst_n),
        .clear(max_clear),
        .valid(max_valid),
        .score(score_norm),
        .running_max(running_max)
    );

    // -----------------------------------------------------------------
    // softmax_exp_sum (pass 2)
    // -----------------------------------------------------------------
    logic                       exp_clear;
    logic                       exp_valid_in;
    wire        [31:0]          exp_val;
    wire                        exp_valid;
    wire  signed [47:0]         sum_so_far;

    softmax_exp_sum u_softmax (
        .clk(clk), .rst_n(rst_n),
        .clear(exp_clear),
        .valid_in(exp_valid_in),
        .score_in(score_norm),
        .max_q(running_max),
        .exp_val(exp_val),
        .exp_valid(exp_valid),
        .sum_so_far(sum_so_far)
    );

    // -----------------------------------------------------------------
    // exp_buf — buffers exp_val per position for pass 3 (combine).
    // CTX_MAX entries × 16-bit (the saturated low half of exp_val Q4.12).
    // (* ram_style = "block" *) keeps this in a BRAM at synth time.
    // -----------------------------------------------------------------
    (* ram_style = "block" *)
    logic [15:0] exp_buf [0:CTX_MAX-1];

    logic                       exp_buf_we;
    logic [CTX_W-1:0]           exp_buf_waddr;
    logic [CTX_W-1:0]           exp_buf_raddr;
    logic [15:0]                exp_buf_rdata;

    always_ff @(posedge clk) begin
        if (exp_buf_we) exp_buf[exp_buf_waddr] <= exp_val[15:0];
        exp_buf_rdata <= exp_buf[exp_buf_raddr];
    end

    // -----------------------------------------------------------------
    // inv_q12 (pass 3, one-shot)
    // -----------------------------------------------------------------
    logic                       inv_start;
    logic                       inv_busy;
    logic                       inv_done;
    logic signed [15:0]         inv_sum;

    inv_q12 u_inv (
        .clk(clk), .rst_n(rst_n),
        .start(inv_start),
        .sum_q12(sum_so_far),
        .busy(inv_busy),
        .done(inv_done),
        .inv_q12_out(inv_sum)
    );

    // -----------------------------------------------------------------
    // attn_v_combine (pass 4)
    // TIMING-FIX (2026-05-23, fixattn fork): combine_attn_q12 and
    // combine_valid are now REGISTERED (always_ff), not combinational.
    // Previously combine_attn_q12 = sat16(exp_buf_rdata * inv_sum) was an
    // always_comb that fed straight into attn_v_combine's MAC, putting TWO
    // DSP58 multiplies + a saturate + a 26-bit accumulate in ONE cycle
    // (the -1.395 ns / 24-logic-level OOC offender). Registering it splits
    // the (exp * inv_sum) multiply and the (attn * V) multiply into separate
    // cycles. v_vec_in = V[pos] is held by the caller across the combine
    // states (kv_rd_en only pulses in S_COMB_RD), so the registered weight
    // stays co-aligned with V[pos] in the added commit cycle S_COMB_MAC2.
    // +1 pipeline-fill cycle per combine position; attention math unchanged.
    // -----------------------------------------------------------------
    logic                       combine_clear;
    logic                       combine_valid;      // registered (see FSM/always_ff)
    logic signed [15:0]         combine_attn_q12;   // registered (see always_ff below)

    attn_v_combine #(.D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH)) u_combine (
        .clk(clk), .rst_n(rst_n),
        .clear(combine_clear),
        .valid_in(combine_valid),
        .round_en(1'b0),                 // B2' N/A on the legacy shared arm (not the shipped path)
        .attn_q12(combine_attn_q12),
        .v_vec(v_vec_in),
        .out_vec(out_vec)
    );

    function automatic signed [15:0] sat16_q12;
        input signed [31:0] v;
        begin
            // shift right 12 with sign-extend, then saturate to int16
            logic signed [31:0] shifted;
            shifted = {{12{v[31]}}, v[31:12]};
            if (shifted > 32'sd32767)         sat16_q12 = 16'sd32767;
            else if (shifted < -32'sd32768)   sat16_q12 = 16'sh8000;
            else                              sat16_q12 = shifted[15:0];
        end
    endfunction

    // combine_attn_q12 = sat16( (exp_buf_rdata * inv_sum) >> 12 )
    // TIMING-FIX: the first multiply (exp_buf_rdata * inv_sum) + saturate is
    // now isolated in its own cycle by latching combine_attn_q12 in always_ff.
    // It is computed in S_COMB_MAC (when exp_buf_rdata = exp[pos] is valid)
    // and presented in S_COMB_MAC2, co-aligned with the held v_vec_in = V[pos].
    // The second multiply (attn * V) then happens inside attn_v_combine in the
    // S_COMB_MAC2 cycle — one DSP58 multiply per cycle, breaking the chain.
    logic signed [31:0] combine_attn_full;
    always_comb begin
        combine_attn_full = $signed({1'b0, exp_buf_rdata}) * $signed(inv_sum);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            combine_attn_q12 <= 16'sd0;
            combine_valid    <= 1'b0;
        end else begin
            // Latch the saturated Q4.12 weight when exp[pos] is valid
            // (S_COMB_MAC); it is consumed by the MAC in the next cycle.
            combine_attn_q12 <= sat16_q12(combine_attn_full);
            // combine_valid pulses one cycle later than the old combinational
            // (state==S_COMB_MAC), i.e. exactly in S_COMB_MAC2, so it stays
            // co-aligned with the latched weight + held V[pos].
            combine_valid    <= (state == S_COMB_MAC);
        end
    end

    // -----------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            pos_req          <= '0;
            ctx_len_latched  <= '0;
            q_vec_latched    <= '0;
            kv_rd_en         <= 1'b0;
            kv_rd_kind       <= 1'b0;
            kv_rd_pos        <= '0;
            score_start      <= 1'b0;
            max_clear        <= 1'b0;
            max_valid        <= 1'b0;
            exp_clear        <= 1'b0;
            exp_valid_in     <= 1'b0;
            exp_buf_we       <= 1'b0;
            exp_buf_waddr    <= '0;
            exp_buf_raddr    <= '0;
            inv_start        <= 1'b0;
            combine_clear    <= 1'b0;
            done             <= 1'b0;
        end else begin
            // defaults
            kv_rd_en      <= 1'b0;
            score_start   <= 1'b0;
            max_clear     <= 1'b0;
            max_valid     <= 1'b0;
            exp_clear     <= 1'b0;
            exp_valid_in  <= 1'b0;
            exp_buf_we    <= 1'b0;
            inv_start     <= 1'b0;
            combine_clear <= 1'b0;
            done          <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        ctx_len_latched <= ctx_len;
                        q_vec_latched   <= q_vec;
                        max_clear       <= 1'b1;
                        exp_clear       <= 1'b1;
                        combine_clear   <= 1'b1;
                        pos_req         <= '0;
                        state           <= S_MAX_RD;
                    end
                end

                // ===== PASS 1: scores → running max (handshake per pos) =====
                S_MAX_RD: begin
                    // Request K[pos_req]; k_vec_in valid next cycle.
                    kv_rd_en   <= 1'b1;
                    kv_rd_kind <= 1'b0;          // K
                    kv_rd_pos  <= pos_req;
                    state      <= S_MAX_START;
                end

                S_MAX_START: begin
                    // k_vec_in is valid this cycle → kick off the score dot.
                    // attn_score_dot latches q/k on the start edge.
                    score_start <= 1'b1;
                    state       <= S_MAX_WAIT;
                end

                S_MAX_WAIT: begin
                    if (score_done) begin
                        max_valid <= 1'b1;       // feed score_out to max tracker
                        if (pos_req == ctx_len_latched - 11'd1) begin
                            pos_req <= '0;
                            state   <= S_EXP_RD;
                        end else begin
                            pos_req <= pos_req + 11'd1;
                            state   <= S_MAX_RD;
                        end
                    end
                end

                // ===== PASS 2: exp(score-max) + sum, buffer exp per pos =====
                S_EXP_RD: begin
                    kv_rd_en   <= 1'b1;
                    kv_rd_kind <= 1'b0;          // K
                    kv_rd_pos  <= pos_req;
                    state      <= S_EXP_START;
                end

                S_EXP_START: begin
                    score_start <= 1'b1;
                    state       <= S_EXP_WAIT;
                end

                S_EXP_WAIT: begin
                    if (score_done) begin
                        // Pulse exp valid_in this cycle; exp_val + exp_valid
                        // land NEXT cycle (softmax_exp_sum is registered).
                        exp_valid_in <= 1'b1;
                        state        <= S_EXP_BUF;
                    end
                end

                S_EXP_BUF: begin
                    // exp_valid is high now (1 cycle after valid_in); the
                    // accumulator already absorbed exp_val into sum_so_far.
                    // Buffer this position's exp_val for the combine pass.
                    exp_buf_we    <= 1'b1;
                    exp_buf_waddr <= pos_req[CTX_W-1:0];
                    if (pos_req == ctx_len_latched - 11'd1) begin
                        inv_start <= 1'b1;       // sum_so_far is final → reciprocal
                        state     <= S_INV;
                    end else begin
                        pos_req <= pos_req + 11'd1;
                        state   <= S_EXP_RD;
                    end
                end

                S_INV: begin
                    if (inv_done) begin
                        pos_req       <= '0;
                        exp_buf_raddr <= '0;
                        state         <= S_COMB_RD;
                    end
                end

                // ===== PASS 4: combine (handshake per pos) ==============
                // S_COMB_RD  : drive kv_rd_en/kv_rd_pos + exp_buf_raddr (these
                //              are REGISTERED, so go high the NEXT cycle).
                // S_COMB_WAIT: kv_rd_en high now → V[pos] latches at end of
                //              this cycle; exp_buf_rdata also latches now.
                // S_COMB_MAC : exp[pos] valid → the (exp*inv_sum) multiply +
                //              saturate is computed and LATCHED into
                //              combine_attn_q12 this edge (TIMING-FIX). No MAC
                //              commit yet. V[pos] is held by the caller.
                // S_COMB_MAC2: combine_attn_q12 + combine_valid are now valid
                //              (registered) and V[pos] still held → attn_v_combine
                //              commits acc[d] += (attn_q12*V[d])>>12 this edge.
                //              The second DSP58 multiply lives here, one cycle
                //              after the first → chain broken.
                S_COMB_RD: begin
                    kv_rd_en      <= 1'b1;
                    kv_rd_kind    <= 1'b1;              // V
                    kv_rd_pos     <= pos_req;
                    exp_buf_raddr <= pos_req[CTX_W-1:0];
                    state         <= S_COMB_WAIT;
                end

                S_COMB_WAIT: begin
                    // wait one cycle for V + exp_buf reads to land
                    state <= S_COMB_MAC;
                end

                S_COMB_MAC: begin
                    // v_vec_in = V[pos_req], exp_buf_rdata = exp[pos_req].
                    // combine_attn_q12 latches (exp*inv_sum)>>12 this edge;
                    // combine_valid will pulse next cycle (S_COMB_MAC2). We do
                    // NOT advance pos_req here so V[pos_req] stays held.
                    state <= S_COMB_MAC2;
                end

                S_COMB_MAC2: begin
                    // combine_attn_q12 + combine_valid valid now; V[pos_req]
                    // still held → MAC commits acc[d] += (attn_q12*V[d])>>12.
                    if (pos_req == ctx_len_latched - 11'd1) begin
                        state <= S_DRAIN_COMB;       // last MAC commits this edge
                    end else begin
                        pos_req <= pos_req + 11'd1;
                        state   <= S_COMB_RD;
                    end
                end

                S_DRAIN_COMB: begin
                    // 1-cycle slack so the final acc register settles before
                    // we publish done (out_vec is combinational from acc[]).
                    done  <= 1'b1;
                    state <= S_DONE;
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
