// =====================================================================
// stage_io_buffer.sv  —  PDI#5 datapath primitive (DATAFLOW-AUDIT R1/R2/R4/R5/R6/R8)
// =====================================================================
// Generic inter-stage activation buffer. It captures a producer stage's
// K_PAR-wide output chunk stream into an H-deep on-chip buffer, then
// presents the captured activation as the next stage's `a_chunk` input
// (replacing the broken `a_chunk`-external-port / tied-`'0'` seams found
// by the DATAFLOW AUDIT).
//
// This is the reusable building block for the SIX broken seams:
//   R1  G1   RMSNorm  -> G4   QKV      activation
//   R2  G6   o_proj   -> G7g/G7u       activation (one buffer feeds both)
//   R5  silu          -> G9   down     activation
//   R6  G9   down     -> G11  RMSNorm  x_chunk
//   R8  G11  RMSNorm  -> G12  LM head  activation
//   (R4 gate/up -> silu uses the same pattern, two narrow instances)
//
// It is the generalization of two patterns already in the top:
//   * M4 `input_emb_buf` (eagle4_drafter.sv:319-377): block-RAM chunk
//     array, single always_ff writer, chunk-indexed combinational read.
//   * L8c `attn_out_chunk_latched` (eagle4_drafter.sv:894-908): producer
//     output captured into a latch consumed by the next stage's mux arm.
//
// Capture interface (producer side):
//   - `wr_valid` + `wr_chunk_idx` + `wr_chunk` : write one chunk per
//     wr_valid.  Mirrors the M4 DMA write port (emb_we/emb_wr_chunk/
//     emb_wr_data) and the L8c "latch on producer-done" model.
//   - Optional auto-incrementing capture (`wr_seq`): if the producer just
//     streams chunk-after-chunk with no index (the rmsnorm/gemv out_valid
//     style), tie wr_chunk_idx to the internal sequential counter by using
//     wr_seq=1 — the buffer assigns slots 0,1,2,... on each wr_valid.
//
// Present interface (consumer side):
//   - `rd_chunk_idx` selects the chunk; `rd_chunk` is the combinational
//     read (the next stage's `a_chunk`).  A `rd_valid` qualifier mirrors
//     M4's g1_req_valid guard so out-of-range reads return '0.
//
// Q-format: this buffer is FORMAT-AGNOSTIC. It moves K_PAR*A_WIDTH-bit
// chunks verbatim. The producer is responsible for delivering the chunk
// in the seam's contracted format (Q4.12 INT16 for the RMSNorm/silu seams;
// raw-INT16 saturated for the GEMV-output seams — see MIXED_PRECISION
// §2/§3 and the DATAPATH_PRIMS_RESULT.md seam table). The buffer never
// rescales; if a seam needs INT32->INT16 saturation it is done by the
// producer's capture stage (the L8c sat_int32_to_int16 pattern) BEFORE
// wr_chunk, exactly as the audit's R-rungs prescribe.
//
// Multi-driver-safe: exactly ONE always_ff writer for the RAM, one for the
// sequential write counter. Reads are pure combinational lookups.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module stage_io_buffer #(
    parameter int H        = 3072,          // activation dim of this seam
    parameter int K_PAR    = 16,            // elements per chunk
    parameter int A_WIDTH  = 16,            // element width (bits)
    parameter bit AUTO_SEQ = 1'b0,          // 1: ignore wr_chunk_idx, auto-increment
    // FRAC_BITS documents the Q-format the seam carries (the buffer itself is
    // FORMAT-AGNOSTIC — it moves A_WIDTH-bit chunks verbatim and never rescales).
    // It labels the seam unambiguously for the integrator:
    //   NORMALIZED seams  R1/R5/R8 (post-RMSNorm/silu, ±8)  -> FRAC_BITS=12 (Q4.12)
    //   RESIDUAL seam     R6 (G9 down -> G11 RMSNorm x, ±857) -> FRAC_BITS=4  (Q12.4)
    // Producer + consumer MUST agree on FRAC_BITS so the RMSNorm input is scaled
    // right. residual_range_and_format.py measured the R6 residual (= r7_out,
    // absmax 857) saturates 2722/3072 in Q4.12 but 0/3072 in Q12.4.
    parameter int FRAC_BITS = 12,           // default = NORMALIZED seam (Q4.12)
    // RD_LAT (2026-06-08, PDI#10 D1 — GEMV ACTIVATION COMBREAD stale-by-1 fix,
    // PDI10_D1_REGREAD_CONFIRM): buf_mem is forced (* ram_style="block" *) BRAM.
    // The legacy read was COMBINATIONAL (`assign rd_chunk = buf_mem[rd_chunk_idx]`)
    // — but on silicon a forced block-RAM read is a 1-cycle SYNCHRONOUS read, so
    // the data for the address presented at cycle T arrives at T+1. The activation
    // seam (gemv_lane_scaled_stream consumer, SLR_SAFE=1) drove the read at the
    // weight-bank-symmetric +1 lead (chunk_idx_next_out) and the forward seam
    // registered a_chunk ONCE; in the COMB-read case those two +1s cancel, but the
    // silicon BRAM read adds a SECOND +1 that nothing absorbs -> MAC computes
    // W[c]·act[c-1] on EVERY chunk (G4/G7g/G7u/G9/G11-lmdown). Confirmed shipping-
    // fatal by tb_d1_regread (mode-A == golden, mode-B (registered) DIVERGES).
    //
    //   RD_LAT=0 : LEGACY combinational read. Bit-identical to all prior async
    //              goldens. Use for buffers NOT on the SLR_SAFE activation seam
    //              (or where the consumer already self-aligns to a comb read).
    //   RD_LAT=1 : REGISTERED read alignment. The read is registered ONCE inside
    //              this module (= the silicon BRAM output register, so sim==silicon
    //              — no async-LUTRAM-downgrade ambiguity, per the funcsim trap).
    //              The CALLER must drive rd_chunk_idx at a +1 lead relative to the
    //              chunk the consumer needs, so the registered rd_chunk lands the
    //              cycle the consumer (MAC) uses it. The activation seam already
    //              leads by +1 (array_chunk_idx_out = chunk_idx_next_out), so the
    //              caller adds ONE MORE +1 lead (rd_chunk_idx = chunk_idx_next + 1,
    //              i.e. the +2 seam) when RD_LAT=1 — exactly mirroring the weight
    //              bank's registered chunk_idx_next lookahead.
    parameter int RD_LAT = 0,
    // DBG_TAP (2026-06-20, M7 GLM=0 WIRED-TAP LOCALIZER, GLM_TAP_LOCALIZER_BUILD):
    //   When 1 (and RD_LAT=1), expose the RD_LAT=1 internal read nodes as READ-ONLY
    //   boundary ports so the full drafter can latch them via AXI-Lite (the wired-tap,
    //   NOT-XMR localizer of M7_INCONTEXT_LOCALIZE_DESIGN.md §1). dbg_rd_chunk_q0 = the
    //   seam BRAM DOUT lane-0 (BEFORE the guard-AND), dbg_rd_guard_q = the RD_LAT guard.
    //   These are continuous reads off existing nets -> drive NOTHING back -> golden
    //   bit-identical (feedback_no_accuracy_tradeoff). DBG_TAP=0 => ports tie to a const
    //   and DCE away -> byte-identical to the no-tap md5 (the pre-build gate).
    parameter bit DBG_TAP = 1'b0,
    // derived (do not override)
    localparam int NCHUNKS = H / K_PAR,
    localparam int CW      = (NCHUNKS > 1) ? $clog2(NCHUNKS) : 1
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // ---- producer-side capture --------------------------------------
    input  wire                              wr_valid,
    input  wire [CW-1:0]                     wr_chunk_idx, // used when AUTO_SEQ=0
    input  wire signed [K_PAR*A_WIDTH-1:0]   wr_chunk,
    input  wire                              wr_clear,     // reset the auto-seq counter
                                                           // (assert at producer start)
    output reg  [CW-1:0]                     wr_count,     // chunks captured so far
    output reg                               full,         // NCHUNKS captured

    // ---- consumer-side present (next stage's a_chunk) ---------------
    input  wire                              rd_valid,
    input  wire [CW-1:0]                     rd_chunk_idx,
    output wire signed [K_PAR*A_WIDTH-1:0]   rd_chunk,

    // ---- DBG_TAP wired localizer (READ-ONLY; DBG_TAP=0 => DCE) ------
    output wire [15:0]                       dbg_rd_chunk_q0,  // seam BRAM DOUT lane0 (pre-guard)
    output wire                              dbg_rd_guard_q,   // RD_LAT range/valid guard

    // ---- CO-TRAVEL outputs (A4 slip-invariant fix) ------------------
    // The chunk-INDEX and VALID that ride WITH the data through the SAME RD_LAT
    // pipeline as rd_chunk, so the consumer (lmdown GEMV) can address its weight
    // ROM by the index that physically arrives alongside the delivered activation
    // (data-driven addressing) -> the dot product is slip-invariant for ANY seam
    // read-latency skew. See reports/results/GLM_FLEET_A4_RDLAT.md + tb_cotravel_proof.
    //   RD_LAT=0 : pure pass-through (rd_chunk_idx_out==rd_chunk_idx, rd_valid_out==rd_valid)
    //   RD_LAT=1 : piped through the SAME ONE register stage as the BRAM sync read,
    //              so the index/valid arrive in LOCKSTEP with rd_chunk
    //              (feedback_valid_data_latency_skew_class).
    // These are NEW OUTPUTS appended at the end of the port list -> existing callers
    // that don't connect them still elaborate (outputs may be left unconnected).
    output wire [CW-1:0]                     rd_chunk_idx_out, // index that travels WITH rd_chunk
    output wire                              rd_valid_out      // valid that travels WITH rd_chunk
);

    // -----------------------------------------------------------------
    // Buffer storage (M4 input_emb_buf pattern): block RAM, single writer.
    // -----------------------------------------------------------------
    (* ram_style = "block" *)
    reg signed [K_PAR*A_WIDTH-1:0] buf_mem [0:NCHUNKS-1];

    // sequential write index used when AUTO_SEQ=1 (producer streams without
    // explicit indices, like rmsnorm/gemv out_valid pulses)
    reg [CW-1:0] seq_idx;

    wire [CW-1:0] eff_wr_idx = AUTO_SEQ ? seq_idx : wr_chunk_idx;

    // ONE writer for the RAM.
    always_ff @(posedge clk) begin
        if (wr_valid) buf_mem[eff_wr_idx] <= wr_chunk;
    end

    // ONE writer for the sequential counter + capture bookkeeping.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            seq_idx  <= '0;
            wr_count <= '0;
            full     <= 1'b0;
        end else if (wr_clear) begin
            seq_idx  <= '0;
            wr_count <= '0;
            full     <= 1'b0;
        end else if (wr_valid) begin
            // S7b FIX (power-of-2 NCHUNKS): compare wr_count against NCHUNKS as a
            // FULL INTEGER, not the CW-bit cast `CW'(NCHUNKS)`. When NCHUNKS == 2^CW
            // (e.g. the R5 seam: H=8192 -> NCHUNKS=512, CW=$clog2(512)=9),
            // `CW'(512)` TRUNCATES to 0, so `wr_count != CW'(512)` == `wr_count != 0`
            // BLOCKED the very first increment (wr_count frozen at 0) and
            // `wr_count == CW'(511)` could never be reached -> `full` NEVER asserted
            // for power-of-2 NCHUNKS. This is the SAME truncation the S5b read-guard
            // fix (line ~141) addressed; here it broke the input-full gate the S7b
            // back-half sequencer uses to start G9 (bk_input_full = seam_r5_full).
            // Widening to (CW+1)-bit makes it correct for ALL H; for non-power-of-2
            // NCHUNKS (fits in CW bits) it is bit-identical to the old behaviour.
            if ({1'b0, wr_count} != (CW+1)'(NCHUNKS))   wr_count <= wr_count + 1'b1;
            if ({1'b0, wr_count} == (CW+1)'(NCHUNKS-1))  full <= 1'b1;
            if (AUTO_SEQ) begin
                if (seq_idx == CW'(NCHUNKS-1)) seq_idx <= '0;
                else                           seq_idx <= seq_idx + 1'b1;
            end
        end
    end

    // -----------------------------------------------------------------
    // Combinational read presented to the next stage as a_chunk.
    // Out-of-range / not-valid reads return '0 (M4 g1_req_valid guard).
    //
    // S5B FIX (power-of-2 NCHUNKS): compare against NCHUNKS as a full INTEGER,
    // NOT the CW-bit cast `CW'(NCHUNKS)`. When NCHUNKS == 2^CW (e.g. the R5
    // seam: H=8192 → NCHUNKS=512, CW=$clog2(512)=9), `CW'(512)` TRUNCATES to 0,
    // so the old guard `rd_chunk_idx < 0` was NEVER true and every read returned
    // '0 (the R5 all-zero readback). `rd_chunk_idx` (CW-bit unsigned) widened
    // against the int NCHUNKS gives the correct range check for ALL H — when
    // NCHUNKS is a power of two every CW-bit index is in range (guard always
    // true); for non-power-of-2 (e.g. R6/R8 H=3072, NCHUNKS=192, CW=8) it is
    // identical to the old behaviour (192 fits in 8 bits, no truncation).
    // -----------------------------------------------------------------
    wire signed [K_PAR*A_WIDTH-1:0] rd_chunk_comb =
        (rd_valid && ({1'b0, rd_chunk_idx} < (CW+1)'(NCHUNKS)))
            ? buf_mem[rd_chunk_idx] : '0;

    generate
    if (RD_LAT == 0) begin : g_rd_comb
        // ---- LEGACY combinational read (bit-identical to prior async goldens) ----
        assign rd_chunk = rd_chunk_comb;
    end else begin : g_rd_reg
        // ---- REGISTERED read = TRUE SYNCHRONOUS BRAM READ (1-cycle) ------------
        // The caller drives rd_chunk_idx at a +1 lead so this registered value
        // lands the cycle the consumer needs it (the D1 activation-seam fix; see
        // RD_LAT comment above).
        //
        // FIX-G7p-KEEP (2026-06-15) — ROOT-CAUSE STRUCTURAL FIX, supersedes the
        // earlier attribute-only attempt:
        //   The legacy RD_LAT=1 code was an ASYNC array read + GUARD MUX + a
        //   SEPARATE fabric register:
        //       rd_chunk_comb = guard ? buf_mem[idx] : '0;   // async read+mux
        //       rd_chunk_r   <= rd_chunk_comb;                // separate FF
        //   The guard mux sits BETWEEN the array read and the register, so the
        //   register CANNOT collapse into the BRAM's own DOUT output register.
        //   Synth then had two bad options, both observed in OOC:
        //     (a) keep buf_mem as RAMB36 and RETIME/ABSORB the loose fabric FF out
        //         of the u_seam_r4* hierarchy  -> grand build (FFs-in-seam==0):
        //         the registered-read latency relationship was relocated, breaking
        //         the +1 read-index alignment on silicon -> seam returned buf[c-1],
        //         silu computed silu(gate[c-1])*up[c-1] -> G7p cap-CRC 0x264e6fa4
        //         (stale) vs golden 0xd8d6d147 (the SHIPPED bug).
        //     (b) pin the loose FF (keep/dont_touch) -> synth declares ram_style=
        //         "block" INFEASIBLE [Synth 8-6849] and DOWNGRADES buf_mem to
        //         LUTRAM (a NEW regression: trades the BRAM for the FF).
        //   RTL-behavioral xsim is GREEN in BOTH cases because it never simulates
        //   the absorbed/relocated netlist (the trap that let (a) ship twice).
        //
        //   FIX: read buf_mem[idx] DIRECTLY inside the clocked process. This is
        //   the canonical sync-read BRAM idiom -> the 1-cycle read register IS the
        //   BRAM's intrinsic OUTPUT register (or addr/DOUT pipeline), NOT a
        //   separable fabric FF, so synth CANNOT retime/absorb it out of the seam
        //   AND keeps buf_mem as RAMB36 (no 8-6849, no LUTRAM downgrade). The
        //   out-of-range/!rd_valid guard is registered ONE cycle (rd_guard_q) and
        //   applied to the registered data, so the read latency and the masking
        //   semantics are bit-identical to the old async-read+register:
        //       old: rd_chunk_r(T) = guard(T-1) ? buf_mem[idx(T-1)] : '0
        //       new: rd_chunk_r(T) = guard_q(T)  ? buf_mem_q(T)      : '0
        //               where guard_q(T)=guard(T-1), buf_mem_q(T)=buf_mem[idx(T-1)]
        //   => identical sequence; only the implementation primitive changed.
        //   RD_LAT=0 path (g_rd_comb) is untouched -> mode-0 stays byte-identical.
        reg signed [K_PAR*A_WIDTH-1:0] rd_chunk_q;      // BRAM sync-read output reg
        reg                            rd_guard_q;      // registered range/valid guard
        wire rd_guard = rd_valid && ({1'b0, rd_chunk_idx} < (CW+1)'(NCHUNKS));
        always_ff @(posedge clk) begin
            // Unconditional synchronous read of buf_mem (no reset, no enable mux
            // in front of the array) = the BRAM-output-register inference idiom.
            rd_chunk_q <= buf_mem[rd_chunk_idx];
        end
        always_ff @(posedge clk) begin
            if (!rst_n) rd_guard_q <= 1'b0;
            else        rd_guard_q <= rd_guard;
        end
        assign rd_chunk = rd_guard_q ? rd_chunk_q : '0;
    end
    endgenerate

    // -----------------------------------------------------------------
    // DBG_TAP wired localizer outputs (M7 GLM=0). READ-ONLY: continuous reads
    // off the RD_LAT=1 internal nodes; drives NOTHING back into the datapath.
    // DBG_TAP=0 (or RD_LAT=0) => tied to const 0 -> the upstream tap chain DCEs
    // away -> byte-identical to the no-tap netlist (pre-build gate).
    // -----------------------------------------------------------------
    generate
    if (DBG_TAP && (RD_LAT != 0)) begin : g_dbg_tap
        assign dbg_rd_chunk_q0 = g_rd_reg.rd_chunk_q[15:0]; // seam BRAM DOUT lane0 (pre-guard)
        assign dbg_rd_guard_q  = g_rd_reg.rd_guard_q;       // RD_LAT range/valid guard
    end else begin : g_no_dbg_tap
        assign dbg_rd_chunk_q0 = 16'h0000;
        assign dbg_rd_guard_q  = 1'b0;
    end
    endgenerate

    // -----------------------------------------------------------------
    // CO-TRAVEL outputs (A4 slip-invariant fix). The chunk-INDEX and VALID
    // travel through the SAME RD_LAT register depth as `rd_chunk`, so they
    // arrive in LOCKSTEP with the delivered data. The downstream lmdown GEMV
    // then addresses its weight ROM by `rd_chunk_idx_out` (the index that
    // physically rode through the seam alongside the activation) -> the dot
    // is slip-invariant for ANY seam read latency / per-bank skew.
    //
    // ONE always_ff owns the pipe regs (multi-driver-safe). When RD_LAT=0
    // they are pure pass-throughs == rd_chunk_idx / rd_valid (the combinational
    // read has zero latency, so the index/valid are already aligned).
    // -----------------------------------------------------------------
    generate
    if (RD_LAT == 0) begin : g_ct_pass
        assign rd_chunk_idx_out = rd_chunk_idx;
        assign rd_valid_out     = rd_valid;
    end else begin : g_ct_reg
        // Pipe the index + valid by ONE cycle = the SAME stage as the BRAM
        // sync-read register (rd_chunk_q / rd_guard_q). idx(T)/valid(T) emerge
        // at T+1 alongside rd_chunk(T+1) = buf_mem[idx(T)].
        reg [CW-1:0] rd_chunk_idx_q;
        reg          rd_valid_q;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                rd_chunk_idx_q <= '0;
                rd_valid_q     <= 1'b0;
            end else begin
                rd_chunk_idx_q <= rd_chunk_idx;
                rd_valid_q     <= rd_valid;
            end
        end
        assign rd_chunk_idx_out = rd_chunk_idx_q;
        assign rd_valid_out     = rd_valid_q;
    end
    endgenerate

endmodule

`default_nettype wire
