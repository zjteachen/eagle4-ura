// =====================================================================
// scale_bram.sv      (M1b — per-group dequant scale store, BRAM-resident)
// =====================================================================
//
// Layout B: WEIGHTS live in URAM (weight_bank_multistage), SCALES live in
// BRAM. This module is the per-lane BRAM scale bank. It mirrors the
// weight_bank_multistage addressing idiom (stage_base + col*cpc-equivalent
// + index) but the inner index is the GROUP index (K/GROUP groups per
// column), NOT the K-chunk index — there is one scale per (group, column).
//
// SIZE (Layout B budget): scales are tiny vs weights. Per-lane scale entries
// summed over the 6 GEMV stages (group_size=128).
//
//   CORRECTED FOOTPRINT (F1 fix, 2026-05-23): the COLUMN COUNTS below were
//   stale (G4 48 cols / G9 24 cols, the old m1b dims) while
//   weight_bank_multistage was corrected to the true dims (G4 N=9216→72 cols,
//   G9 N=6144→48 cols). The two banks MUST agree on cols/stage or the scaled
//   GEMV mis-bases. The geometry now lives ONCE in stage_geometry.svh; this
//   table is documentation, the included functions are the source of truth.
//
//   stage    cols/lane (=N/128)   grp/col (=K/128)   entries/lane   base
//   G4_qkv   72                   48                 3456           0
//   G6_o     24                   24                  576           3456
//   G7_g     64                   24                 1536           4032
//   G7_u     64                   24                 1536           5568
//   G9_down  48                   64                 3072           7104
//   G12_lm   1002                 2                  2004           10176
//   ------------------------------------------------- ------
//   total                                            12180 entries
//   × 16 bits = ~190 Kbit / lane (depth-limited, not width); at LANES=128
//   still well within V80's 3741 RAMB36.
//
// Read latency: 1 cycle (registered BRAM output), matching the URAM weight
// bank so the scale and weight pipelines stay aligned. The caller drives
// group_idx_next for 1-cycle lookahead, identical to the URAM chunk_idx_next
// idiom — scale data at T+1 = scale[group@T+1], lining up with the lane's
// group_idx_out at T+1.
//
// Multi-driver-safe: ONE write port (boot loader), ONE read port (compute).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

// SINGLE SOURCE OF TRUTH for per-stage geometry — shared with
// weight_bank_multistage.sv and the scaled-array TBs. (F1/F4 fix.)
`include "stage_geometry.svh"

(* keep_hierarchy = "yes" *)
module scale_bram #(
    // LANES-256 (2026-07-26): geometry follows the LANES parameter (shared
    // stage_geometry.svh functions take it as an argument). DEPTH default
    // scales exactly with the per-lane footprint (12,180@128 -> 6,090@256):
    // 128 -> 16K, 256 -> 8K = 4 RAMB36/lane -> 256x4 = 1,024 total, BRAM-
    // NEUTRAL vs the golden 128x8 (measured, LANES256_GEMM_STUDY.md §2).
    // A forgotten halving costs +1,024 RAMB36 = DEAD (study §6) — deriving
    // it here makes the halving automatic under a strapped LANES=256.
    parameter int LANES      = 128,
    parameter int DEPTH      = (16 * 1024 * 128) / LANES, // per-lane scale entries (headroom)
    parameter int SCALE_W    = 16,          // Q(16-frac).frac signed scale
    parameter int ADDR_W     = 17,
    parameter int GW         = 7,           // group-index width (covers up to 64 groups)
    parameter int GROUP      = 128          // GPTQ group size (scale per K/GROUP groups)
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Boot-time write port (driven by the scale-load FSM / AXI-Lite)
    input  wire                  we,
    input  wire [ADDR_W-1:0]     wr_addr,        // flat scale address
    input  wire [SCALE_W-1:0]    wr_data,

    // Compute-time read port
    input  wire [3:0]            stage_sel,      // 0..5 GEMV stage
    input  wire [15:0]           col_offset,     // local output col within stage
    input  wire [GW-1:0]         group_idx,      // group within column
    output reg  [SCALE_W-1:0]    rd_data
);

    // ----------------------------------------------------------------
    // Per-stage base + groups-per-column come from the SHARED geometry
    // (stage_geometry.svh) — NOT a local table. This is the F1 fix: the
    // scale bank and weight_bank_multistage now derive cols/stage from the
    // SAME N/K, so they can never drift apart again. (Stale local tables
    // assuming G4=48 / G9=24 cols are gone.)
    // ----------------------------------------------------------------
    function automatic [ADDR_W-1:0] stage_base(input [3:0] s);
        stage_base = sg_s_stage_base(s, GROUP, LANES);
    endfunction

    function automatic [15:0] stage_groups_per_col(input [3:0] s);
        stage_groups_per_col = sg_s_groups_per_col(s, GROUP);
    endfunction

    // ----------------------------------------------------------------
    // Read-side flat address (same timing trick as weight_bank_multistage):
    // register base + col×gpc so the per-cycle path is only +group_idx.
    // F6: use 2-state != (not 4-state !==) in synthesizable RTL.
    // ----------------------------------------------------------------
    reg  [ADDR_W-1:0] col_base_acc;
    reg  [3:0]        stage_sel_d;
    reg  [15:0]       col_offset_d;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            col_base_acc <= '0; stage_sel_d <= '0; col_offset_d <= '0;
        end else if (stage_sel != stage_sel_d || col_offset != col_offset_d) begin
            col_base_acc <= stage_base(stage_sel) +
                            (col_offset * stage_groups_per_col(stage_sel));
            stage_sel_d  <= stage_sel;
            col_offset_d <= col_offset;
        end
    end

    wire [ADDR_W-1:0] rd_addr = col_base_acc + {{(ADDR_W-GW){1'b0}}, group_idx};

    // ----------------------------------------------------------------
    // Backing BRAM — simple dual-port (1 write, 1 read), 1-cycle read.
    // Inferred as RAMB36/RAMB18 (block_ram style hint).
    // ----------------------------------------------------------------
    (* ram_style = "block" *) reg [SCALE_W-1:0] mem [0:DEPTH-1];

`ifdef SIM_WSNAP
    // SIM-ONLY snapshot/restore (rationale in weight_bank_uram.sv).
    string  wsnap_dir;
    longint wsnap_t;
    initial begin
        if ($value$plusargs("WSNAP_LOAD=%s", wsnap_dir)) begin
            #1;
            $readmemh($sformatf("%s/%m.mem", wsnap_dir), mem);
        end
        if ($value$plusargs("WSNAP_DUMP=%s", wsnap_dir)) begin
            if (!$value$plusargs("WSNAP_T=%d", wsnap_t)) wsnap_t = 64'd63_300_000;
            #(wsnap_t);
            $writememh($sformatf("%s/%m.mem", wsnap_dir), mem);
        end
    end
`endif

    // SIM-ONLY zero-init (compiled out under SYNTHESIS): a scale cell that is
    // READ before the boot loader has WRITTEN it (e.g. a unit TB that boots
    // only a few columns, then reads a lookahead group at a column boundary)
    // would otherwise return X and poison the scaled product even though the
    // unscaled INT acc is correct. On hardware the scale loader writes every
    // live cell before compute, so this init is a no-op there. It does NOT
    // change any value the loader writes — pure X-avoidance for simulation.
`ifndef SYNTHESIS
    integer ii_init;
    initial for (ii_init = 0; ii_init < DEPTH; ii_init = ii_init + 1)
        mem[ii_init] = '0;
`endif

    always_ff @(posedge clk) begin
        if (we) mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end

endmodule

`default_nettype wire
