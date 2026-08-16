// =====================================================================
// stage_io_buffer_ff.sv  —  FLIP-FLOP / SHIFT-REGISTER drop-in replacement
//                            for stage_io_buffer.sv (NO block RAM).
// =====================================================================
// MOTIVATION (2026-06-26, STRUCTURAL FIX keystone):
//   The baseline stage_io_buffer stages the inter-stage activation chunks
//   in a (* ram_style="block" *) BRAM array. A single BRAM block whose
//   address/data nets cross an SLR boundary is the recurring silicon
//   bug-class on this design (GLM guard-kill, COMBREAD stale-by-1, the
//   +128 load rotation). FF is the most abundant resource on the V80
//   (~12% used of 5.15M); URAM (88.83%) is the binding wall. Moving the
//   intermediate seams to flip-flop / shift-register staging removes the
//   BRAM read net entirely and trades the scarce resource (URAM/BRAM,
//   SLR-fragile) for the abundant one (FF, placeable anywhere).
//
// DROP-IN CONTRACT:
//   * IDENTICAL port list to stage_io_buffer (same names, widths, order),
//     including the DBG_TAP and CO-TRAVEL outputs appended at the end.
//   * IDENTICAL write semantics: AUTO_SEQ=0 (indexed) / AUTO_SEQ=1 (auto-
//     increment), the (CW+1)-wide full-integer wr_count/full compares
//     (power-of-2 NCHUNKS safe), wr_clear.
//   * IDENTICAL read semantics + LATENCY for BOTH RD_LAT modes:
//       RD_LAT=0 : combinational addressed read, out-of-range/!valid -> '0
//       RD_LAT=1 : registered sync read (the +1 caller lead), guard
//                  registered ONE cycle and applied to the registered data
//                  -> rd_chunk(T) = guard_q(T) ? mem_q(T) : '0
//                  with guard_q(T)=guard(T-1), mem_q(T)=mem[idx(T-1)].
//   * IDENTICAL co-travel: rd_chunk_idx_out / rd_valid_out piped at the
//     SAME RD_LAT depth as rd_chunk.
//
// STORAGE IMPL (generate-selectable, NO ram_style="block" anywhere):
//   IMPL="REGFILE" (default, GUARANTEED bit-exact for ANY access pattern):
//       (* ram_style="registers" *) FF array + addressed read mux. This is
//       a true drop-in: random / re-read / out-of-order reads are all
//       bit-identical to the BRAM reference. ZERO RAMB / ZERO URAM. Cost =
//       the NCHUNKS:1 chunk-wide read mux in LUTs (acceptable: LUT is
//       abundant, and the BRAM-SLR fragility is eliminated).
//   IMPL="SHIFT" (FIFO, sequential fill-then-drain ONLY):
//       NCHUNKS-deep shift register, chunk-wide. Writes shift in (chunk
//       order), reads shift out from the head -> NO addressed read mux, so
//       cheaper LUT, but it IGNORES wr_chunk_idx / rd_chunk_idx and relies
//       on the seam's natural fill-all-then-drain-in-order pattern (the
//       GEMV-order read at +1 lead). Bit-exact under that pattern only;
//       NOT a drop-in for random/re-read seams. Use REGFILE unless a seam
//       is proven strictly sequential.
//
// FORMAT-AGNOSTIC: like the baseline, moves K_PAR*A_WIDTH-bit chunks
// verbatim, never rescales. FRAC_BITS is documentation only.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module stage_io_buffer_ff #(
    parameter int H        = 3072,
    parameter int K_PAR    = 16,
    parameter int A_WIDTH  = 16,
    parameter bit AUTO_SEQ = 1'b0,
    parameter int FRAC_BITS = 12,
    parameter int RD_LAT   = 0,
    parameter bit DBG_TAP  = 1'b0,
    // FF storage style. REGFILE = addressed FF array (true drop-in).
    // SHIFT = sequential FIFO shift register (no read mux, sequential-only).
    parameter     IMPL     = "REGFILE",
    // derived (do not override)
    localparam int NCHUNKS = H / K_PAR,
    localparam int CW      = (NCHUNKS > 1) ? $clog2(NCHUNKS) : 1,
    localparam int CHW     = K_PAR * A_WIDTH
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // ---- producer-side capture --------------------------------------
    input  wire                              wr_valid,
    input  wire [CW-1:0]                     wr_chunk_idx,
    input  wire signed [CHW-1:0]             wr_chunk,
    input  wire                              wr_clear,
    output reg  [CW-1:0]                     wr_count,
    output reg                               full,

    // ---- consumer-side present (next stage's a_chunk) ---------------
    input  wire                              rd_valid,
    input  wire [CW-1:0]                     rd_chunk_idx,
    output wire signed [CHW-1:0]             rd_chunk,

    // Stage-B' DUAL_PORT: additive 2nd (ODD-chunk) read port. Combinational
    // regfile read at rd_chunk_idx_b, SAME RD_LAT contract as rd_chunk. Inert
    // when unconnected (rd_chunk_b unused at DUAL_PORT=0). REGFILE only (the
    // body GEMV seams are REGFILE); SHIFT ties it to '0 (no random read there).
    input  wire [CW-1:0]                     rd_chunk_idx_b,
    output wire signed [CHW-1:0]             rd_chunk_b,

    // ---- DBG_TAP wired localizer (READ-ONLY; DBG_TAP=0 => DCE) ------
    output wire [15:0]                       dbg_rd_chunk_q0,
    output wire                              dbg_rd_guard_q,

    // ---- CO-TRAVEL outputs (A4 slip-invariant fix) ------------------
    output wire [CW-1:0]                     rd_chunk_idx_out,
    output wire                              rd_valid_out
);

    // -----------------------------------------------------------------
    // Write bookkeeping — BIT-IDENTICAL to stage_io_buffer (seq counter,
    // wr_count cap, full assert; power-of-2 NCHUNKS safe via (CW+1) compare)
    // -----------------------------------------------------------------
    reg [CW-1:0] seq_idx;
    wire [CW-1:0] eff_wr_idx = AUTO_SEQ ? seq_idx : wr_chunk_idx;

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
            if ({1'b0, wr_count} != (CW+1)'(NCHUNKS))   wr_count <= wr_count + 1'b1;
            if ({1'b0, wr_count} == (CW+1)'(NCHUNKS-1))  full <= 1'b1;
            if (AUTO_SEQ) begin
                if (seq_idx == CW'(NCHUNKS-1)) seq_idx <= '0;
                else                           seq_idx <= seq_idx + 1'b1;
            end
        end
    end

    // registered guard shared by RD_LAT=1 paths (== guard(T-1))
    wire rd_guard = rd_valid && ({1'b0, rd_chunk_idx} < (CW+1)'(NCHUNKS));
    reg  rd_guard_q;
    always_ff @(posedge clk) begin
        if (!rst_n) rd_guard_q <= 1'b0;
        else        rd_guard_q <= rd_guard;
    end

    // dbg tap source nets (driven by the active IMPL's registered read)
    wire [CHW-1:0] dbg_mem_q;

    // =================================================================
    //  STORAGE / READ — generate-selected, NO ram_style="block"
    // =================================================================
    generate
    // -----------------------------------------------------------------
    if (IMPL == "SHIFT") begin : g_shift
        // FIFO shift register: fill-all-then-drain-in-order. No read mux.
        // sr[0] is the head (next chunk out). On write, shift the whole
        // register one slot toward 0 and load the new chunk at the tail
        // sr[NCHUNKS-1] -> after NCHUNKS sequential writes sr[i]=chunk[i].
        // On an accepted read, shift toward 0 (consume head). Writes and
        // reads never overlap on this seam (fill until `full`, then drain),
        // so ONE always_ff owns the array (multi-driver-safe).
        //
        // CRITICAL (2026-06-26): `ram_style="registers"` governs RAM inference
        // but does NOT block SHIFT-REGISTER extraction — on a clean
        // sr[i]<=sr[i+1] pattern Vivado would pack this into SRL16E/SRL32E,
        // which are LUTRAM primitives (the exact read-stale / LUTRAM-downgrade
        // hazard we are escaping). `shreg_extract="no"` + `keep="true"` forces
        // genuine FDRE flip-flops (SLR-distributable), matching the project's
        // SLR-crossing FF-pipe idiom (lmdown_gemv_banksplit.sv:396,
        // rope_layer.sv:75, slr_seam_pipe.sv:72). The OOC primitive census
        // MUST show FDRE and ZERO SRL16E/SRL32E/RAMB/LUTRAM.
        (* ram_style = "registers", shreg_extract = "no", keep = "true" *)
        reg signed [CHW-1:0] sr [0:NCHUNKS-1];
        integer si;
        always_ff @(posedge clk) begin
            if (wr_valid) begin
                for (si = 0; si < NCHUNKS-1; si = si + 1)
                    sr[si] <= sr[si+1];
                sr[NCHUNKS-1] <= wr_chunk;
            end else if (rd_guard) begin
                for (si = 0; si < NCHUNKS-1; si = si + 1)
                    sr[si] <= sr[si+1];
                // tail slot holds (don't-care after drain)
            end
        end

        wire signed [CHW-1:0] head_comb = rd_guard ? sr[0] : '0;

        assign rd_chunk_b = '0;   // SHIFT: no random 2nd read (DUAL_PORT uses REGFILE seams)
        if (RD_LAT == 0) begin : g_rd0
            assign rd_chunk = head_comb;
            assign dbg_mem_q = sr[0];
        end else begin : g_rd1
            reg signed [CHW-1:0] head_q;
            always_ff @(posedge clk) head_q <= sr[0];
            assign rd_chunk = rd_guard_q ? head_q : '0;
            assign dbg_mem_q = head_q;
        end
    // -----------------------------------------------------------------
    end else begin : g_regfile
        // Addressed FF register file — TRUE DROP-IN (any access pattern).
        (* ram_style = "registers" *)
        reg signed [CHW-1:0] ff_mem [0:NCHUNKS-1];
        always_ff @(posedge clk) begin
            if (wr_valid) ff_mem[eff_wr_idx] <= wr_chunk;
        end

        wire signed [CHW-1:0] rd_comb =
            (rd_valid && ({1'b0, rd_chunk_idx} < (CW+1)'(NCHUNKS)))
                ? ff_mem[rd_chunk_idx] : '0;

        // Stage-B' 2nd (ODD) read — same guard/RD_LAT contract, own index.
        wire rd_guard_b = rd_valid && ({1'b0, rd_chunk_idx_b} < (CW+1)'(NCHUNKS));
        reg  rd_guard_b_q;
        always_ff @(posedge clk) rd_guard_b_q <= rd_guard_b;
        wire signed [CHW-1:0] rd_comb_b = rd_guard_b ? ff_mem[rd_chunk_idx_b] : '0;

        if (RD_LAT == 0) begin : g_rd0
            // LEGACY combinational read — bit-identical to BRAM RD_LAT=0.
            assign rd_chunk = rd_comb;
            assign dbg_mem_q = ff_mem[rd_chunk_idx];
            assign rd_chunk_b = rd_comb_b;
        end else begin : g_rd1
            // Registered sync read: mem_q(T)=ff_mem[idx(T-1)],
            // guard_q(T)=guard(T-1). rd_chunk(T)=guard_q?mem_q:'0.
            // Bit-identical sequence to the BRAM sync-read.
            reg signed [CHW-1:0] rd_mem_q;
            always_ff @(posedge clk) rd_mem_q <= ff_mem[rd_chunk_idx];
            assign rd_chunk = rd_guard_q ? rd_mem_q : '0;
            assign dbg_mem_q = rd_mem_q;
            // 2nd port, identical RD_LAT=1 sync-read semantics.
            reg signed [CHW-1:0] rd_mem_b_q;
            always_ff @(posedge clk) rd_mem_b_q <= ff_mem[rd_chunk_idx_b];
            assign rd_chunk_b = rd_guard_b_q ? rd_mem_b_q : '0;
        end
    end
    endgenerate

    // -----------------------------------------------------------------
    // DBG_TAP wired localizer outputs (READ-ONLY). DBG_TAP=0 or RD_LAT=0
    // => tied const 0 -> DCE -> byte-identical to no-tap netlist.
    // -----------------------------------------------------------------
    generate
    if (DBG_TAP && (RD_LAT != 0)) begin : g_dbg_tap
        assign dbg_rd_chunk_q0 = dbg_mem_q[15:0];
        assign dbg_rd_guard_q  = rd_guard_q;
    end else begin : g_no_dbg_tap
        assign dbg_rd_chunk_q0 = 16'h0000;
        assign dbg_rd_guard_q  = 1'b0;
    end
    endgenerate

    // -----------------------------------------------------------------
    // CO-TRAVEL outputs — same RD_LAT depth as rd_chunk (drop-in identical
    // to stage_io_buffer's g_ct_pass / g_ct_reg).
    // -----------------------------------------------------------------
    generate
    if (RD_LAT == 0) begin : g_ct_pass
        assign rd_chunk_idx_out = rd_chunk_idx;
        assign rd_valid_out     = rd_valid;
    end else begin : g_ct_reg
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
