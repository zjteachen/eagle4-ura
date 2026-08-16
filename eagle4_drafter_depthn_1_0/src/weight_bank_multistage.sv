// =====================================================================
// weight_bank_multistage.sv
// Per-lane URAM weight bank with stage-aware addressing.
// =====================================================================
//
// Each lane owns 1 weight bank that holds quantized weights (32-bit
// chunks of 16 × 2-bit weights) for ALL GEMV stages combined.
//
// ---------------------------------------------------------------------
// CORRECTED FOOTPRINT (WEIGHT-LAYOUT-FIT, 2026-05-23) — supersedes the
// old undersized table that assumed wrong N for G4 and G9.
//
// True dims (cross-checked vs safetensors + reference/src/gptq.cpp +
// reference/include/eagle4_gemm_abstract.h + docs/M1_WEIGHT_MAP.md):
//   G4_qkv : K=6144  N=9216 (q+k+v = 3×3072 concat, share stage_sel=0)
//   G6_o   : K=3072  N=3072
//   G7_g   : K=3072  N=8192
//   G7_u   : K=3072  N=8192
//   G9_down: K=8192  N=6144 (ENABLE_2HS: to_logits + for_reasoning;
//                            BOTH halves needed — for_reasoning seeds
//                            the next tree depth, so N cannot drop to 3072)
//   G12_lm : K=256   N=128256
//
// Per-lane footprint = Σ (N/LANES)·(K/16), LANES=128:
//   stage   K     N       cpc  local_cols  entries/lane
//   G4_qkv  6144  9216     384         72         27648
//   G6_o    3072  3072     192         24          4608
//   G7_g    3072  8192     192         64         12288
//   G7_u    3072  8192     192         64         12288
//   G9_down 8192  6144     512         48         24576
//   G12_lm   256  128256    16       1002         16032
//   ---------------------------------------------- ------
//   TOTAL                                          97,440 entries
//                                  × 32 bits = 3.118 Mbit / lane
//
// Vivado maps a (DEPTH×32) RAM onto URAM288 (4096-deep × 72-wide
// columns): URAM/lane = ceil(DEPTH/4096). With DEPTH=98304 (96K, ≥ the
// 97,440 footprint, power-of-two-aligned for clean cascade) that is
// ceil(98304/4096) = 24 URAM/lane (32-bit data uses 32 of the 72 bits,
// ~44% width-packed). At LANES=128: 24×128 = 3072 URAM — OVER the 1925
// budget. The 32→72 width waste is the dominant cost; the wide-URAM
// repack primitive (weight_bank_uram_packed72.sv) is the fit mitigation
// (≈2.25× density → ~11 URAM/lane → ~1408 total). See FIT VERDICT in
// WEIGHT_LAYOUT_FIT_RESULT.md.
//
// Stage selector + chunk_idx + col_offset map to a flat URAM address.
// Per-stage base offsets are localparams; tweak if architecture changes.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

// SINGLE SOURCE OF TRUTH for per-stage geometry — shared with scale_bram.sv
// and the scaled-array TBs. (F1/F4 fix: eliminates the dual-table drift.)
`include "stage_geometry.svh"

(* keep_hierarchy = "yes" *)
module weight_bank_multistage #(
    // LANES-256 (2026-07-26): geometry follows the LANES parameter (the
    // stage_geometry.svh functions take it as an argument — the SG_LANES
    // define is dead). DEPTH default scales exactly: per-lane footprint
    // halves when lanes double (97,440@128 -> 48,720@256; every stage N
    // divides both). 128 -> 96K (98,304), 256 -> 48K (49,152).
    parameter int LANES = 128,
    parameter int DEPTH = (96 * 1024 * 128) / LANES, // >= true footprint at 128/256
    parameter int DATA_W = 32,
    parameter int RD_PORTS = 1          // WO-025: 1 = legacy 1-read, 2 = dual-read (port B)
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Boot-time write port (driven by AXI4-Lite weight-load FSM)
    input  wire                  we,
    input  wire [16:0]           wr_addr,        // flat 17-bit address
    input  wire [DATA_W-1:0]     wr_data,

    // Compute-time read port A (primary / even chunk)
    input  wire [3:0]            stage_sel,      // 0..5 GEMV stage
    input  wire [15:0]           col_offset,     // which output col within stage
    input  wire [15:0]           chunk_idx,      // K-chunk within column
    output wire [DATA_W-1:0]     rd_data,

    // Compute-time read port B (WO-025 dual-read; same column, odd chunk).
    // Ignored when RD_PORTS==1 (rd_data_b ties 0). col_offset for port B is
    // the SAME column as port A (single output col per lane), so port B reuses
    // the registered col_offset_acc — only the inner chunk index differs.
    input  wire [15:0]           chunk_idx_b,
    output wire [DATA_W-1:0]     rd_data_b
);

    // ----------------------------------------------------------------
    // Per-stage base offsets and per-column stride come from the SHARED
    // geometry (stage_geometry.svh) — NOT a local table. The corrected
    // WEIGHT-LAYOUT-FIT 2026-05-23 dims (G4 N=9216 / G9 N=6144) are now
    // derived from the single sg_stage_N()/sg_stage_K() source the scale
    // bank also uses. Documented footprint:
    //   G4_qkv : 0      .. 27648   (72 cols × 384)   — q+k+v=9216 concat
    //   G6_o   : 27648  .. 32256   (24 cols × 192)
    //   G7_g   : 32256  .. 44544   (64 cols × 192)
    //   G7_u   : 44544  .. 56832   (64 cols × 192)
    //   G9_down: 56832  .. 81408   (48 cols × 512)   — N=6144 (both 2HS halves)
    //   G12_lm : 81408  .. 97440   (1002 cols × 16)
    // ----------------------------------------------------------------
    function automatic [16:0] stage_base(input [3:0] s);
        stage_base = sg_w_stage_base(s, LANES);
    endfunction

    function automatic [16:0] stage_chunks_per_col(input [3:0] s);
        stage_chunks_per_col = sg_w_chunks_per_col(s);
    endfunction

    // ----------------------------------------------------------------
    // Read-side flat address: base + col × chunks_per_col + chunk_idx
    //
    // TIMING-OPTIMIZED: register `base + col × cpc` so the per-cycle
    // critical path is just (registered_offset + chunk_idx) — a 17-bit
    // adder, no DSP.  The expensive col × cpc multiply only updates
    // when stage_sel or col_offset changes (every N_CHUNKS cycles for
    // a given output column), so 1-cycle stale-address protection is
    // baked into the upstream FSM by gating chunk_idx_valid.
    // ----------------------------------------------------------------
    reg  [16:0] col_offset_acc;     // registered: base + col×cpc
    reg  [3:0]  stage_sel_d;
    reg  [15:0] col_offset_d;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            col_offset_acc <= '0;
            stage_sel_d    <= '0;
            col_offset_d   <= '0;
        end else begin
            // Re-evaluate when stage_sel or col_offset changes
            // F6: 2-state != (not 4-state !==) in synthesizable RTL.
            if (stage_sel != stage_sel_d || col_offset != col_offset_d) begin
                col_offset_acc <= stage_base(stage_sel) +
                                  (col_offset * stage_chunks_per_col(stage_sel));
                stage_sel_d    <= stage_sel;
                col_offset_d   <= col_offset;
            end
        end
    end

    // Per-cycle address: just an add (no multiply on critical path).
    // Port B shares col_offset_acc (same output column) — only the inner
    // chunk index differs (the odd chunk of the dual-port pair).
    wire [16:0] rd_addr   = col_offset_acc + chunk_idx;
    wire [16:0] rd_addr_b = col_offset_acc + chunk_idx_b;

    // ----------------------------------------------------------------
    // Backing URAM (instantiated via the V5d primitive).
    // RD_PORTS=2 drives the idle URAM288 port B as a 2nd compute read
    // (WO-025). RD_PORTS=1 collapses port B away (bit+cycle identical).
    // ----------------------------------------------------------------
    weight_bank_uram #(
        .DEPTH(DEPTH), .DATA_W(DATA_W), .RD_PORTS(RD_PORTS)
    ) u_uram (
        .clk(clk), .rst_n(rst_n),
        .we(we), .wr_addr(wr_addr), .wr_data(wr_data),
        .rd_addr(rd_addr),     .rd_data(rd_data),
        .rd_addr_b(rd_addr_b), .rd_data_b(rd_data_b)
    );

endmodule

`default_nettype wire
