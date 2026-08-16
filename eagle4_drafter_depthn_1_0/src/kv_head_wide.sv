// =====================================================================
// kv_head_wide.sv
// D_HEAD-wide KV cache for one attention head.
// =====================================================================
//
// Bridges the per-element kv_head_uram primitive to the D_HEAD-wide
// vector API expected by attention_head_assembly / flash_attn_head:
//
//   wr_vec  ── D_HEAD × A_WIDTH-bit per cycle (one full K[pos] or V[pos])
//   rd_vec  ── D_HEAD × A_WIDTH-bit, 1-cycle latency
//
// Implementation: a single (D_HEAD * A_WIDTH)-bit wide BRAM, MAX_CTX
// rows deep. Synchronous read (1-cycle), synchronous write.
//
// Synth notes:
//   - Inferred as block RAM via (* ram_style = "block" *).
//   - For D_HEAD=128, A_WIDTH=16 → 2048-bit wide × 2048 deep = 4 Mbit
//     per head, 4 URAMs cascaded (272-bit URAM-wide). For pilot
//     D_HEAD=16 / MAX_CTX=32 it's ~1 BRAM.
//   - Production scaling for 24 heads × 2 (K+V) = 48 × 4 = 192 URAMs
//     at full geometry, well under V80's 1925.
//
// Multi-driver-safe: single packed register `mem` driven from a single
// always_ff. No shared packed reg issues.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module kv_head_wide #(
    parameter int D_HEAD  = 16,
    parameter int A_WIDTH = 16,
    parameter int MAX_CTX = 32,
    parameter int ADDR_W  = (MAX_CTX <= 1) ? 1 : $clog2(MAX_CTX),
    parameter int ROW_W   = D_HEAD * A_WIDTH
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Write port — full D_HEAD vector at pos
    input  wire                  wr_en,
    input  wire [10:0]           wr_pos,
    input  wire [ROW_W-1:0]      wr_vec,

    // Read port — 1-cycle latency
    input  wire                  rd_en,
    input  wire [10:0]           rd_pos,
    output reg  [ROW_W-1:0]      rd_vec
);

    (* ram_style = "block" *)
    reg [ROW_W-1:0] mem [0:MAX_CTX-1];

    initial begin
        for (int i = 0; i < MAX_CTX; i++) mem[i] = '0;
    end

    always_ff @(posedge clk) begin
        if (wr_en) begin
            mem[wr_pos[ADDR_W-1:0]] <= wr_vec;
        end
        if (rd_en) begin
            rd_vec <= mem[rd_pos[ADDR_W-1:0]];
        end
    end

endmodule

`default_nettype wire
