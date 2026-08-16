// =====================================================================
// residual_add.sv  —  PDI#5 datapath primitive (DATAFLOW-AUDIT R3 / R7)
// =====================================================================
// The Llama decoder residual add, over a K_PAR-wide Q4.12 INT16 chunk
// stream:
//
//     out[i] = sat_q4_12( a[i] + b[i] )
//
// Used twice per layer:
//   R3:  h1  = input  + attn_out   (after G6 o_proj, before post-attn RMSNorm)
//   R7:  out = h1     + mlp_out     (after G9 down_proj, before final RMSNorm)
//
// Numerical contract (MIXED_PRECISION_DECISIONS.md §1):
//   - Both operands are signed Q4.12 INT16 (range ±8.0, 1 LSB = 2^-12).
//   - Sum done in a widened (INT17) accumulator, then SATURATED back to
//     INT16 at the Q4.12 extremes:  > +32767 -> +32767 ; < -32768 -> -32768.
//     +32767 == +7.99975 (one LSB below +8.0); -32768 == -8.0.
//
// This module pairs an INPUT-SAVE BUFFER with the adder: the residand `a`
// (the layer input `x`, or the post-attn hidden `h1`) is captured while the
// attention / MLP branch computes on the shared GEMV array, then replayed
// chunk-by-chunk to be added to the branch output `b`.
//
// Design patterns reused (do not reinvent):
//   * input-save buffer  -> M4 `input_emb_buf` (eagle4_drafter.sv:319-377):
//       a (* ram_style="block" *) chunk array, single always_ff writer,
//       chunk-indexed combinational read.
//   * saturate-to-INT16  -> L8c `sat_int32_to_int16` (eagle4_drafter.sv:885).
//
// Multi-driver-safe: the save buffer has exactly ONE always_ff writer
// (the `a` capture path). The adder output register has one writer. The
// `b` stream is read-only.  No tri-state, no shared writes.
//
// Two usage modes (both exercised by the testbench):
//   (1) STREAM-SAVE mode  — drive `a` chunks with `a_valid` to fill the save
//       buffer (capture during the branch), then assert `start` to replay the
//       buffer against the `b` chunks. This is the PDI#5 R3/R7 mode.
//   (2) PASS-THROUGH check — `a` and `b` presented together (the save buffer
//       is still used: a_valid latches a into the buffer, then start replays).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module residual_add #(
    parameter int H        = 3072,          // hidden dim (3072 -> 192 chunks)
    parameter int K_PAR    = 16,            // elements per chunk
    parameter int A_WIDTH  = 16,            // element width (bits)
    // FRAC_BITS documents the residual-stream Q-format carried on this seam.
    //   Q4.12  (FRAC_BITS=12) -> ±8.0      : the NORMALIZED-seam format (R1/R5/R8)
    //   Q12.4  (FRAC_BITS=4)  -> ±2048.0   : the RESIDUAL/HIDDEN-stream format (R3/R7)
    // The saturating-add HARDWARE is identical for either choice — the rails are
    // the signed-A_WIDTH integer extremes ±2^(A_WIDTH-1). FRAC_BITS only changes
    // the *interpretation* (1 LSB = 2^-FRAC_BITS) at the producer/consumer seams.
    // residual_range_and_format.py measured: on the real seed-42 forward the
    // residual stream reaches r3_h1=±166, r7_out=±857; Q4.12 saturates 1980/2722
    // of 3072 elems (RMSNorm-out cosine collapses to 0.83); Q12.4 holds the range
    // with ZERO saturation and RMSNorm-out cosine 0.9999999 (fp16-indistinguishable).
    parameter int FRAC_BITS = 4,            // 4 => Q12.4 residual stream (R3/R7)
    // RDATA_LAT (2026-06-08, COMBREAD stale-by-1 fix — RMSNORM_COMBREAD_CONFIRM):
    // a_save_buf is (* ram_style="block" *) BRAM. The legacy path read it
    // COMBINATIONALLY (`a_rd = a_save_buf[rd_idx]`) and advanced rd_idx the SAME
    // cycle it added a_rd+b_chunk -> on silicon (registered 1-cycle BRAM read) the
    // add consumed a_save_buf[rd_idx-1] = a[k-1]+b[k] (1-chunk misalignment on R3/R7).
    //   RDATA_LAT=0 : LEGACY combinational read (bit-identical to all prior goldens).
    //   RDATA_LAT=1 : REGISTERED read alignment — a_rd is registered (matches the
    //                 silicon BRAM 1-cycle read) and b_chunk/b_valid/rd_idx are
    //                 delayed one cycle so the add consumes the matched (a[k],b[k])
    //                 pair. sim==silicon (both 1-cycle). Adds 1 cycle of latency.
    parameter int RDATA_LAT = 0,
    // derived (do not override)
    localparam int NCHUNKS = H / K_PAR,                 // 3072/16 = 192
    localparam int CW      = (NCHUNKS > 1) ? $clog2(NCHUNKS) : 1
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // ---- residand `a` capture (the saved input / h1) -----------------
    // One chunk per a_valid; a_chunk_idx selects the destination slot.
    input  wire                              a_valid,
    input  wire [CW-1:0]                     a_chunk_idx,
    input  wire signed [K_PAR*A_WIDTH-1:0]   a_chunk,

    // ---- replay + add against the branch output `b` ------------------
    input  wire                              start,        // begin the add walk
    input  wire signed [K_PAR*A_WIDTH-1:0]   b_chunk,      // branch output chunk
    input  wire                              b_valid,      // b_chunk is live this cycle

    output reg                               out_valid,    // out_chunk live
    output reg  [CW-1:0]                     out_chunk_idx,// which chunk this is
    output reg  signed [K_PAR*A_WIDTH-1:0]   out_chunk,    // sat_q4_12(a + b)
    output reg                               done          // all NCHUNKS emitted
);

    // -----------------------------------------------------------------
    // Saturating add of two signed A_WIDTH operands (format-agnostic).
    //   sum done in INT(A_WIDTH+2) (two guard bits) — cannot overflow for
    //   two A_WIDTH adds. Saturate to the signed-A_WIDTH integer extremes
    //   (rails are width-derived, NOT a hardcoded Q4.12 constant — so the
    //   SAME hardware serves Q4.12 ±8 and Q12.4 ±2048; only FRAC_BITS scale
    //   interpretation differs at the seams).
    //   For A_WIDTH=16: +SAT = +32767, -SAT = -32768.
    //     Q4.12  : +32767 = +7.99975 , -32768 = -8.0
    //     Q12.4  : +32767 = +2047.94 , -32768 = -2048.0
    // -----------------------------------------------------------------
    localparam logic signed [A_WIDTH-1:0] POS_SAT =  {1'b0, {(A_WIDTH-1){1'b1}}};  // +2^(W-1)-1
    localparam logic signed [A_WIDTH-1:0] NEG_SAT =  {1'b1, {(A_WIDTH-1){1'b0}}};  // -2^(W-1)

    function automatic logic signed [A_WIDTH-1:0] sat_add;
        input signed [A_WIDTH-1:0] x;
        input signed [A_WIDTH-1:0] y;
        logic signed [A_WIDTH+1:0] s;   // two guard bits: room for two A_WIDTH adds
        begin
            s = $signed({{2{x[A_WIDTH-1]}}, x}) + $signed({{2{y[A_WIDTH-1]}}, y});
            if (s > $signed({{2{POS_SAT[A_WIDTH-1]}}, POS_SAT}))        sat_add = POS_SAT;
            else if (s < $signed({{2{NEG_SAT[A_WIDTH-1]}}, NEG_SAT}))   sat_add = NEG_SAT;
            else                                                        sat_add = s[A_WIDTH-1:0];
        end
    endfunction

    // -----------------------------------------------------------------
    // Input-save buffer (M4 input_emb_buf pattern): block RAM, single
    // always_ff writer, chunk-indexed combinational read.
    // -----------------------------------------------------------------
    (* ram_style = "block" *)
    reg signed [K_PAR*A_WIDTH-1:0] a_save_buf [0:NCHUNKS-1];

    always_ff @(posedge clk) begin
        if (a_valid) a_save_buf[a_chunk_idx] <= a_chunk;   // ONE writer
    end

    // -----------------------------------------------------------------
    // Replay-and-add FSM.  On `start`, walk chunk 0..NCHUNKS-1 in lockstep
    // with the incoming b stream (b_valid gates the advance), adding the
    // saved `a` chunk to the live `b` chunk and emitting the saturated sum.
    // -----------------------------------------------------------------
    reg            running;
    reg [CW-1:0]   rd_idx;

    generate
    if (RDATA_LAT == 0) begin : g_combread
        // ---- LEGACY combinational read path (bit-identical to prior goldens) ----
        wire signed [K_PAR*A_WIDTH-1:0] a_rd = a_save_buf[rd_idx];

        integer lane;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                running       <= 1'b0;
                rd_idx        <= '0;
                out_valid     <= 1'b0;
                out_chunk_idx <= '0;
                out_chunk     <= '0;
                done          <= 1'b0;
            end else begin
                out_valid <= 1'b0;
                done      <= 1'b0;

                if (start && !running) begin
                    running <= 1'b1;
                    rd_idx  <= '0;
                end else if (running && b_valid) begin
                    // combinational add of saved a_chunk[rd_idx] + this b_chunk
                    for (lane = 0; lane < K_PAR; lane = lane + 1) begin
                        out_chunk[lane*A_WIDTH +: A_WIDTH] <=
                            sat_add(a_rd       [lane*A_WIDTH +: A_WIDTH],
                                    b_chunk    [lane*A_WIDTH +: A_WIDTH]);
                    end
                    out_chunk_idx <= rd_idx;
                    out_valid     <= 1'b1;

                    if (rd_idx == CW'(NCHUNKS-1)) begin
                        running <= 1'b0;
                        done    <= 1'b1;
                        rd_idx  <= '0;
                    end else begin
                        rd_idx <= rd_idx + 1'b1;
                    end
                end
            end
        end
    end else begin : g_regread
        // ---- REGISTERED read alignment (COMBREAD FIX, RDATA_LAT=1) ----
        // Stage 0 (this cycle): present rd_idx to a_save_buf; the registered read
        // delivers a_rd_r = a_save_buf[rd_idx] on the NEXT cycle. Pipeline the
        // matching b_chunk / b_valid / rd_idx one cycle so the add (stage 1)
        // consumes the matched (a[k], b[k]) pair. Mirrors the silicon BRAM 1-cycle
        // read -> sim==silicon. The walk-complete (done) keys off the stage-1 idx.
        reg signed [K_PAR*A_WIDTH-1:0] a_rd_r;        // registered a_save_buf read
        reg signed [K_PAR*A_WIDTH-1:0] b_chunk_d1;    // b_chunk delayed to align with a_rd_r
        reg                            b_valid_d1;    // b_valid delayed
        reg [CW-1:0]                   rd_idx_d1;     // rd_idx that produced a_rd_r

        integer lane;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                running       <= 1'b0;
                rd_idx        <= '0;
                a_rd_r        <= '0;
                b_chunk_d1    <= '0;
                b_valid_d1    <= 1'b0;
                rd_idx_d1     <= '0;
                out_valid     <= 1'b0;
                out_chunk_idx <= '0;
                out_chunk     <= '0;
                done          <= 1'b0;
            end else begin
                out_valid  <= 1'b0;
                done       <= 1'b0;
                b_valid_d1 <= 1'b0;

                // stage 0: registered read + pipeline the matching b/idx
                a_rd_r     <= a_save_buf[rd_idx];

                if (start && !running) begin
                    running <= 1'b1;
                    rd_idx  <= '0;
                end else if (running && b_valid) begin
                    // capture b_chunk and the rd_idx that addressed a_save_buf
                    // THIS cycle; a_rd_r holds a_save_buf[rd_idx] next cycle.
                    b_chunk_d1 <= b_chunk;
                    b_valid_d1 <= 1'b1;
                    rd_idx_d1  <= rd_idx;
                    if (rd_idx != CW'(NCHUNKS-1))
                        rd_idx <= rd_idx + 1'b1;
                    else
                        running <= 1'b0;   // last address presented; drain stage 1
                end

                // stage 1: matched add of a_save_buf[rd_idx_d1] + b_chunk_d1
                if (b_valid_d1) begin
                    for (lane = 0; lane < K_PAR; lane = lane + 1) begin
                        out_chunk[lane*A_WIDTH +: A_WIDTH] <=
                            sat_add(a_rd_r    [lane*A_WIDTH +: A_WIDTH],
                                    b_chunk_d1[lane*A_WIDTH +: A_WIDTH]);
                    end
                    out_chunk_idx <= rd_idx_d1;
                    out_valid     <= 1'b1;
                    if (rd_idx_d1 == CW'(NCHUNKS-1))
                        done <= 1'b1;
                end
            end
        end
    end
    endgenerate

endmodule

`default_nettype wire
