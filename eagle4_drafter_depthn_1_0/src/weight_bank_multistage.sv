// Per-lane weight storage for all shared GEMV stages.
// A 32-bit word holds 16 two-bit weights along the input dimension.
// Address = stage_base + local_output_column * (K/16) + chunk_idx.
// Stage dimensions and bases come from stage_geometry.svh.
//
// With 128 lanes, G4 has 72 columns per lane and 384 words per column:
// 27,648 words per lane. All stages together require 97,440 words per lane.
// Boot writes and compute reads must not overlap.

`timescale 1ns/1ps
`default_nettype none

// Shared geometry keeps weight and scale column counts consistent.
`include "stage_geometry.svh"

(* keep_hierarchy = "yes" *)
module weight_bank_multistage #(
    // Per-lane storage halves when the number of output lanes doubles.
    parameter int LANES = 128,
    parameter int DEPTH = (96 * 1024 * 128) / LANES, // >= true footprint at 128/256
    parameter int DATA_W = 32,
    parameter int RD_PORTS = 1          // 1 = primary read; 2 = primary + secondary
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Boot-time write port; the caller selects this lane and flat address.
    input  wire                  we,
    input  wire [16:0]           wr_addr,        // flat 17-bit address
    input  wire [DATA_W-1:0]     wr_data,

    // Compute-time read port A (primary / even chunk)
    input  wire [3:0]            stage_sel,      // 0..5 GEMV stage
    input  wire [15:0]           col_offset,     // which output col within stage
    input  wire [15:0]           chunk_idx,      // K-chunk within column
    output wire [DATA_W-1:0]     rd_data,

    // Optional second read of the same output column, at another chunk.
    // RD_PORTS=1 ties rd_data_b to zero.
    input  wire [15:0]           chunk_idx_b,
    output wire [DATA_W-1:0]     rd_data_b
);

    // Both banks derive stage bases and strides from the same geometry.
    function automatic [16:0] stage_base(input [3:0] s);
        stage_base = sg_w_stage_base(s, LANES);
    endfunction

    function automatic [16:0] stage_chunks_per_col(input [3:0] s);
        stage_chunks_per_col = sg_w_chunks_per_col(s);
    endfunction

    // Register the column base so the read-address path only adds chunk_idx.
    // Allow the base to settle after changing stage_sel or col_offset.
    reg  [16:0] col_offset_acc;     // registered: base + col×cpc
    reg  [3:0]  stage_sel_d;
    reg  [15:0] col_offset_d;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            col_offset_acc <= '0;
            stage_sel_d    <= '0;
            col_offset_d   <= '0;
        end else begin
            // Update only when the selected column changes.
            if (stage_sel != stage_sel_d || col_offset != col_offset_d) begin
                col_offset_acc <= stage_base(stage_sel) +
                                  (col_offset * stage_chunks_per_col(stage_sel));
                stage_sel_d    <= stage_sel;
                col_offset_d   <= col_offset;
            end
        end
    end

    // Both read ports share the registered column base.
    wire [16:0] rd_addr   = col_offset_acc + chunk_idx;
    wire [16:0] rd_addr_b = col_offset_acc + chunk_idx_b;

    // Synchronous inferred RAM; RD_PORTS selects one or two compute reads.
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
