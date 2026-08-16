// =====================================================================
// rmsnorm_block.sv
// EAGLE4 RMSNorm composition: sumsq -> scale_int -> apply.
// =====================================================================
//
// Sequences the three RMSNorm primitives behind a single start/done
// handshake. One full RMSNorm of an H-element vector takes:
//
//   PHASE_SUMSQ : H/K_PAR cycles (en=1) + ~3 cycles pipeline drain
//   PHASE_SCALE : ~57 cycles (iterative isqrt + iterative div)
//   PHASE_APPLY : H/K_PAR cycles (en=1) + ~3 cycles pipeline drain
//
// Total for H=3072, K_PAR=16: ~192 + 3 + 57 + 192 + 3 ~= 447 cycles
//   = 1.66 us at 270 MHz, 2.24 us at 200 MHz.
//
// For PDI #3 integration-skeleton: x_chunk / gamma_chunk are tied to 0
// at the caller (eagle4_drafter.sv g1 and g11 instances) until upstream
// activation routing lands in a later ladder rung. The FSM still walks
// the full deterministic latency, so kernel_done timing matches what a
// real RMSNorm would consume.
//
// Multi-driver-safe: the apply.out_chunk is wired straight to a single
// output port; no shared packed registers driven from multiple always
// blocks.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module rmsnorm_block #(
    parameter int H       = 3072,
    parameter int K_PAR   = 16,
    parameter int A_WIDTH = 16,
    // M4 (2026-05-23): right-shift applied to the 64-bit sum-of-squares BEFORE
    // it enters rmsnorm_scale_int. DEFAULT 0 preserves the legacy behaviour
    // bit-for-bit (g11 and every existing instance are unchanged). The G1
    // instance sets SUMSQ_RSHIFT=12 because its activations are Q4.12 INT16:
    // sumsq is then in Q8.24, and the TALOS scale engine (numerator 2^24,
    // hardcoded /16 averaging) otherwise yields a scale 64× too small. Feeding
    // sumsq>>12 lands the Q4.12 scale exactly (see MIXED_PRECISION §1; verified
    // bit-accurate against ref/golden/g1.hex, max|d|=1 LSB). This shift is a
    // pure caller-side scaling parameter — it adds one constant shift, no new
    // driver, and does not perturb the FSM or latency.
    //
    // ───────────────────────────────────────────────────────────────────────
    // FULL-H AVERAGING FIX (rmsnorm_fullh, 2026-05-23) — supersedes the pure
    // SUMSQ_RSHIFT for any H≠16. The TALOS scale engine hardcodes a /16 average
    // (radicand=((sumsq>>4)+1)<<12). SUMSQ_RSHIFT only applies a power-of-2
    // shift, so it CANNOT turn the engine's /16 into /H when H/16 is not a power
    // of 2 (H=3072 → H/16=192, NOT a power of 2). The pure shift only lands the
    // Q-format; the AVERAGE stayed /16, so at full H the RMSNorm output was
    // sqrt(H/16)=sqrt(192)≈13.86× too SMALL (cosine≈1.0, magnitude wrong).
    // (Masked before because g1/g11 goldens were only validated at H=16, where
    //  /16 == /H exactly.)
    //
    // FIX: replace `sumsq >> SUMSQ_RSHIFT` with a per-stage constant multiply
    //   sumsq_eng = (sumsq * SUMSQ_MULT) >> SUMSQ_SHR
    // chosen so the engine's internal /16 yields the CORRECT mean-of-squares
    // SS/H, Q-format-aligned. This is a STRICT GENERALIZATION of the legacy
    // shift: at H=16 the chosen (MULT,SHR) reduces to a pure 2^k shift and the
    // datapath is bit-identical. (target ratio = 2^(16+2B-2f)/H; see
    // rmsnorm_fullh/ref/derive_fix3.py / RMSNORM_FULLH_RESULT.md.)
    //
    // SUMSQ_MULT=0 selects the LEGACY pure-shift path (back-compat default), so
    // existing H=16 instances and goldens are bit-identical. SUMSQ_MULT>0 picks
    // the constant-multiply averaging fix. SCALE_SHR carries the apply-stage
    // shift (12-B) for small-RMS stages whose dimensionless gain exceeds the
    // Q4.12 ceiling (B headroom bits; B=0 ⇒ SCALE_SHR=12 ⇒ legacy apply).
    // The multiply is one constant-coefficient DSP/LUT mult; no new driver,
    // FSM, or latency change (sumsq_eng is a pure combinational wire).
    parameter int SUMSQ_RSHIFT = 0,
    parameter longint unsigned SUMSQ_MULT = 0,         // 0 ⇒ legacy pure-shift
    parameter int              SUMSQ_SHR  = 0,
    parameter int              SCALE_SHR  = 12,         // apply x×scale shift = 12-B
    // C7 G1_SATURATION_FIX (2026-05-26): stage-A x_scaled intermediate width in
    // rmsnorm_apply. Default A_WIDTH = legacy (bit-identical). The G1 instance
    // sets XS_WIDTH=18 so the 4 largest pre-norm hiddens (|x_scaled|≤65312) stop
    // saturating ±32767. Pure wire/register width — no FSM/latency/Q-format
    // change (output still saturated to A_WIDTH at the ×gamma stage).
    parameter int              XS_WIDTH   = A_WIDTH,
    // ─────────────────────────────────────────────────────────────────────────
    // RDATA_LAT (2026-06-08, COMBREAD stale-by-1 fix — RMSNORM_COMBREAD_CONFIRM):
    // the caller feeds x_chunk/gamma_chunk from a (* ram_style="block" *) BRAM
    // indexed by x_req_chunk = chunk_cnt. On silicon the BRAM read is REGISTERED
    // (1-cycle): the data for address-presented-at-T arrives at T+RDATA_LAT, but
    // the sumsq/apply engines sample x_chunk the SAME cycle they raise their
    // enable — so every chunk is consumed stale-by-1 (cap[0]=0, cap[1]≈A[0]).
    // FIX: delay the sumsq_en / apply_en strobes that drive the engines by
    // RDATA_LAT cycles so they sample x_chunk exactly when the registered read
    // delivers buf[that-chunk]. The address (x_req_chunk=chunk_cnt) still LEADS
    // the data by RDATA_LAT, matching the registered read. The caller must use a
    // REGISTERED read (rd_data <= buf[addr]) so sim==silicon (both 1-cycle); see
    // eagle4_drafter.sv g1/g1e/premlp/g11 registered-read taps.
    //   RDATA_LAT=0 : LEGACY combinational-read path, bit-identical to all prior
    //                 H=16 goldens and the async-LUTRAM slice funcsim.
    //   RDATA_LAT=1 : registered-BRAM-read alignment (the body G1/G1e/pre-MLP/G11
    //                 instances). One extra latency cycle per phase; the strobe
    //                 taps (out_chunk_strobe/idx, apply_en_o) shift with it.
    parameter int              RDATA_LAT  = 0,
    parameter int              SCALE_WIDTH = 16,
    parameter longint unsigned SCALE_EPS_BUCKET = 1
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,
    input  wire signed [K_PAR*A_WIDTH-1:0]   x_chunk,       // streaming input
    input  wire signed [K_PAR*A_WIDTH-1:0]   gamma_chunk,   // matching gamma row
    output wire signed [K_PAR*A_WIDTH-1:0]   out_chunk,     // streaming output
    output wire                              out_valid,
    output logic                             done,
    // M4: chunk-request taps so a caller can stream a chunk-indexed buffer in
    // lockstep with the internal SUMSQ/APPLY phases. `x_req_chunk` is the chunk
    // index the FSM consumes THIS cycle; `x_req_valid` is high during the two
    // streaming phases. The caller drives x_chunk = buf[x_req_chunk] purely
    // combinationally (the buffer read is combinational), so the value is
    // present on the same cycle sumsq_en / apply_en sample it. Existing
    // instances leave these outputs unconnected (legacy behaviour preserved).
    output wire  [$clog2((H/K_PAR)+1)-1:0]   x_req_chunk,
    output wire                              x_req_valid,
    // S6 (G11->G12 full-width seam): a CLEAN per-chunk output strobe + index
    // ALIGNED to out_chunk, for streaming the full-H norm output into a
    // stage_io_buffer seam. `out_valid` (= rmsnorm_apply.valid) is NOT a clean
    // 192-pulse strobe (the first apply chunk has no valid pulse and the last
    // arrives during S_DRAIN_A, shifted -1 from data — see tb_rmsnorm_block_g11hp.sv
    // notes). Instead we derive the strobe from the registered apply_en delayed
    // by the deterministic rmsnorm_apply latency AP_LAT=2: the chunk requested
    // when apply_en is high at cycle T has its out_chunk on the bus at T+2.
    // `out_chunk_idx_o` carries that chunk's logical index (0..N_CHUNKS-1).
    // Exactly N_CHUNKS strobes per RMSNorm, in order, one per cycle. Existing
    // instances leave these unconnected (no behaviour change — pure taps).
    output reg                               out_chunk_strobe,
    output reg   [$clog2((H/K_PAR)+1)-1:0]   out_chunk_idx_o,
    // S3 (G1 tap, merged from S7a/M5): expose the registered apply-phase strobe so
    // a caller can capture the FULL streamed output bit-exactly using the
    // deterministic apply_en+AP_LAT rule (the same rule the golden generator
    // tb_rmsnorm_block_g11hp.sv uses) — NOT the unreliable out_valid (out_valid
    // count != apply-chunk count; data is shifted -1 from valid). Gating on
    // apply_en (high one cycle into S_DRAIN_A for the LAST chunk) captures all
    // N_CHUNKS. Additive output; existing instances leave it unconnected.
    output wire                              apply_en_o
);

    localparam int N_CHUNKS = H / K_PAR;
    localparam int CHUNK_W  = $clog2(N_CHUNKS + 1);

    typedef enum logic [2:0] {
        S_IDLE    = 3'd0,
        S_SUMSQ   = 3'd1,   // streaming x_chunk into rmsnorm_sumsq
        S_DRAIN_S = 3'd2,   // wait 3 cycles for sumsq pipeline drain
        S_SCALE   = 3'd3,   // scale_int iterative sqrt+div
        S_APPLY   = 3'd4,   // streaming x_chunk + gamma into rmsnorm_apply
        S_DRAIN_A = 3'd5,   // wait 3 cycles for apply pipeline drain
        S_DONE    = 3'd6
    } state_e;

    state_e state;
    logic [CHUNK_W-1:0] chunk_cnt;
    logic [3:0]         drain_cnt;

    // -----------------------------------------------------------------
    // Sub-instance handshakes
    // -----------------------------------------------------------------
    logic              sumsq_clear;
    logic              sumsq_en;
    logic [63:0]       sumsq_out;

    logic              scale_start;
    logic              scale_busy;
    logic              scale_done;
    logic signed [SCALE_WIDTH-1:0] scale_q12;

    logic              apply_en;

    // ─────────────────────────────────────────────────────────────────────────
    // RDATA_LAT alignment: delay the FSM-generated enables (and clear) by
    // RDATA_LAT cycles before they reach the sumsq/apply engines, so the engine
    // samples x_chunk on the cycle the REGISTERED caller read delivers buf[addr]
    // for the address the FSM presented RDATA_LAT cycles earlier. RDATA_LAT=0
    // collapses to a direct wire (bit-identical legacy). One extra register per
    // signal per latency cycle; no datapath change.
    logic              sumsq_en_q;     // delayed sumsq enable -> engine
    logic              sumsq_clear_q;  // delayed clear        -> engine
    logic              apply_en_q;     // delayed apply enable -> engine
    generate
        if (RDATA_LAT == 0) begin : g_rdlat0
            assign sumsq_en_q    = sumsq_en;
            assign sumsq_clear_q = sumsq_clear;
            assign apply_en_q    = apply_en;
        end else if (RDATA_LAT == 1) begin : g_rdlat1
            // SYNTH-CLEAN single-flop case. A {sr[RDATA_LAT-2:0], x} concat would
            // be {sr[-1:0], x} = an illegal/negative part-select at RDATA_LAT==1
            // ([Synth 8-524/8-525/8-6156]); xsim tolerated it, synth rejects it.
            // One register per signal — identical timing to the shift-vector tail
            // at depth 1 (delay-by-1), bit-exact to the intended COMBREAD align.
            logic sumsq_en_d1, sumsq_clr_d1, apply_en_d1;
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    sumsq_en_d1  <= 1'b0;
                    sumsq_clr_d1 <= 1'b0;
                    apply_en_d1  <= 1'b0;
                end else begin
                    sumsq_en_d1  <= sumsq_en;
                    sumsq_clr_d1 <= sumsq_clear;
                    apply_en_d1  <= apply_en;
                end
            end
            assign sumsq_en_q    = sumsq_en_d1;
            assign sumsq_clear_q = sumsq_clr_d1;
            assign apply_en_q    = apply_en_d1;
        end else begin : g_rdlat
            // RDATA_LAT >= 2 : shift-register tail. [RDATA_LAT-2:0] is in-range
            // (>= [0:0]) for every RDATA_LAT >= 2, so this concat is synth-legal.
            logic [RDATA_LAT-1:0] sumsq_en_sr, sumsq_clr_sr, apply_en_sr;
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    sumsq_en_sr  <= '0;
                    sumsq_clr_sr <= '0;
                    apply_en_sr  <= '0;
                end else begin
                    sumsq_en_sr  <= {sumsq_en_sr [RDATA_LAT-2:0], sumsq_en};
                    sumsq_clr_sr <= {sumsq_clr_sr[RDATA_LAT-2:0], sumsq_clear};
                    apply_en_sr  <= {apply_en_sr [RDATA_LAT-2:0], apply_en};
                end
            end
            assign sumsq_en_q    = sumsq_en_sr [RDATA_LAT-1];
            assign sumsq_clear_q = sumsq_clr_sr[RDATA_LAT-1];
            assign apply_en_q    = apply_en_sr [RDATA_LAT-1];
        end
    endgenerate
    assign apply_en_o = apply_en_q;   // S3/M5 output tap, aligned to the engine's sample cycle

    rmsnorm_sumsq #(.K_PAR(K_PAR), .A_WIDTH(A_WIDTH)) u_sumsq (
        .clk(clk), .rst_n(rst_n),
        .clear(sumsq_clear_q), .en(sumsq_en_q),
        .x_chunk(x_chunk),
        .sumsq_out(sumsq_out)
    );

    // Q-format + averaging scaling of sumsq before the TALOS scale engine.
    //   SUMSQ_MULT==0 : LEGACY pure power-of-2 shift (signed: ≥0 right, <0 left).
    //                   Bit-identical to the M4 behaviour for all existing
    //                   instances and the H=16 goldens.
    //   SUMSQ_MULT>0  : FULL-H averaging fix — constant-coefficient multiply
    //                   then right-shift, so the engine's hardcoded /16 lands
    //                   the correct /H mean-of-squares. One combinational wire
    //                   (no extra always block / driver / FSM / latency change).
    // The product is 64×~40 bits → use a wide intermediate, then take [63:0].
    wire [127:0] sumsq_mul   = sumsq_out * SUMSQ_MULT;
    wire [63:0]  sumsq_fix   = sumsq_mul[63+SUMSQ_SHR -: 64];   // (sumsq*MULT)>>SHR
    wire [63:0]  sumsq_legacy = (SUMSQ_RSHIFT >= 0)
                                 ? (sumsq_out >> SUMSQ_RSHIFT)
                                 : (sumsq_out << (-SUMSQ_RSHIFT));
    wire [63:0] sumsq_scaled = (SUMSQ_MULT != 0) ? sumsq_fix : sumsq_legacy;

    rmsnorm_scale_int #(.SCALE_WIDTH(SCALE_WIDTH),
                        .EPS_BUCKET(SCALE_EPS_BUCKET)) u_scale (
        .clk(clk), .resetn(rst_n),
        .start(scale_start),
        .sumsq(sumsq_scaled),
        .busy(scale_busy),
        .done(scale_done),
        .scale_q12(scale_q12)
    );

    // M4: chunk-request taps. The FSM consumes x_chunk during S_SUMSQ and
    // S_APPLY, indexed by chunk_cnt. Expose both so a caller can stream a
    // chunk-indexed buffer in lockstep (single combinational mux on the
    // caller side — see eagle4_drafter.sv G1 input-embedding feed).
    assign x_req_chunk = chunk_cnt;
    assign x_req_valid = (state == S_SUMSQ) || (state == S_APPLY);

    rmsnorm_apply #(.K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .SCALE_SHR(SCALE_SHR),
                    .XS_WIDTH(XS_WIDTH), .SCALE_WIDTH(SCALE_WIDTH)) u_apply (
        .clk(clk), .rst_n(rst_n),
        .en(apply_en_q),               // RDATA_LAT-delayed: samples x_chunk when the registered read is ready
        .scale_q12(scale_q12),
        .x_chunk(x_chunk),
        .gamma_chunk(gamma_chunk),
        .out_chunk(out_chunk),
        .valid(out_valid)
    );

    // -----------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            chunk_cnt    <= '0;
            drain_cnt    <= '0;
            sumsq_clear  <= 1'b0;
            sumsq_en     <= 1'b0;
            scale_start  <= 1'b0;
            apply_en     <= 1'b0;
            done         <= 1'b0;
        end else begin
            // defaults
            sumsq_clear <= 1'b0;
            sumsq_en    <= 1'b0;
            scale_start <= 1'b0;
            apply_en    <= 1'b0;
            done        <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        sumsq_clear <= 1'b1;
                        chunk_cnt   <= '0;
                        state       <= S_SUMSQ;
                    end
                end
                S_SUMSQ: begin
                    sumsq_en <= 1'b1;
                    chunk_cnt <= chunk_cnt + 1'b1;
                    if (chunk_cnt == N_CHUNKS[CHUNK_W-1:0] - 1'b1) begin
                        // +RDATA_LAT: the delayed sumsq_en_q tail lands RDATA_LAT
                        // cycles later than the FSM raises sumsq_en, so the last
                        // accumulate completes RDATA_LAT cycles later. Extend the
                        // pipeline drain so sumsq_out is settled before scale_start.
                        drain_cnt <= 4'd3 + RDATA_LAT[3:0];
                        state     <= S_DRAIN_S;
                    end
                end
                S_DRAIN_S: begin
                    if (drain_cnt == 0) begin
                        scale_start <= 1'b1;
                        state       <= S_SCALE;
                    end else begin
                        drain_cnt <= drain_cnt - 1'b1;
                    end
                end
                S_SCALE: begin
                    if (scale_done) begin
                        chunk_cnt <= '0;
                        state     <= S_APPLY;
                    end
                end
                S_APPLY: begin
                    apply_en <= 1'b1;
                    chunk_cnt <= chunk_cnt + 1'b1;
                    if (chunk_cnt == N_CHUNKS[CHUNK_W-1:0] - 1'b1) begin
                        // +RDATA_LAT: matches the delayed apply_en_q tail so the
                        // final out_chunk lands (and its strobe fires) before done.
                        drain_cnt <= 4'd3 + RDATA_LAT[3:0];
                        state     <= S_DRAIN_A;
                    end
                end
                S_DRAIN_A: begin
                    if (drain_cnt == 0) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        drain_cnt <= drain_cnt - 1'b1;
                    end
                end
                S_DONE: begin
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------
    // S6: clean per-chunk output strobe/index, aligned to out_chunk.
    //   apply_en high at cycle T (S_APPLY) drives u_apply.en at T; chunk_cnt@T is
    //   that chunk's logical index. rmsnorm_apply is a 2-stage pipe (AP_LAT=2):
    //   out_chunk for en@T is on the bus at T+2. So out_chunk_strobe/idx_o must be
    //   apply_en/chunk_cnt delayed by EXACTLY AP_LAT=2 registers. Implemented as a
    //   1-deep (AP_LAT-1) shift feeding the output register => 2 total stages.
    //   Exactly N_CHUNKS ordered strobes, one per cycle, aligned to out_chunk.
    // Single always_ff writer for these regs (multi-driver-safe).
    // -----------------------------------------------------------------
    // The LOGICAL applied-chunk index (0,1,2,...) is NOT chunk_cnt: when apply_en
    // first goes high chunk_cnt is already 1 (the FSM increments it the same cycle
    // it raises apply_en, and the slot+1 contract means chunk_cnt=k applies logical
    // chunk k-1). So count apply outputs locally: ap_logical increments on each
    // apply_en cycle, giving 0..N_CHUNKS-1 for the N_CHUNKS applied chunks, in
    // exactly the order rmsnorm_apply emits them.
    localparam int AP_LAT = 2;   // rmsnorm_apply en -> out_chunk latency
    logic [CHUNK_W-1:0] ap_logical;       // logical applied-chunk counter (0..N-1)
    logic               ap_en_d1;         // apply_en delayed AP_LAT-1 = 1
    logic [CHUNK_W-1:0] ap_idx_d1;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ap_logical <= '0;
            ap_en_d1 <= 1'b0;  ap_idx_d1 <= '0;
            out_chunk_strobe <= 1'b0;  out_chunk_idx_o <= '0;
        end else begin
            // Key the strobe off apply_en_q (the RDATA_LAT-delayed enable that
            // actually drives u_apply), NOT the raw FSM apply_en: the engine
            // produces out_chunk at apply_en_q + AP_LAT. With RDATA_LAT=0,
            // apply_en_q == apply_en (bit-identical legacy strobe).
            if (start)            ap_logical <= '0;       // re-arm per RMSNorm
            else if (apply_en_q)  ap_logical <= ap_logical + 1'b1;
            // stage 1 (T+1): capture the apply request fired THIS cycle (its
            // logical index = ap_logical BEFORE this cycle's increment).
            ap_en_d1  <= apply_en_q;
            ap_idx_d1 <= ap_logical;
            // stage 2 / output (T+2): aligns with out_chunk on the bus NOW
            out_chunk_strobe <= ap_en_d1;
            out_chunk_idx_o  <= ap_idx_d1;
        end
    end

endmodule

`default_nettype wire
