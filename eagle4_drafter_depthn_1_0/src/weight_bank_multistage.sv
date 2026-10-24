// Per-lane weight storage for the shared GEMV engine.
// A 32-bit word holds 16 two-bit weights along the input dimension.
// Address = cfg_base + chunk_idx, where cfg_base is an EXPLICIT flat word
// address supplied by the shared operation controller (weight_base plus the
// current tile's row stride). The old stage_sel/col_offset derivation is
// gone; the controller owns tile addressing so ISA configuration is the sole
// source of bases and bounds.
//
// The base is registered, matching the synchronous RAM read: the controller
// must present cfg_base at least one cycle before the array start, then hold
// it stable for the whole tile. Boot writes and compute reads must not
// overlap.

`timescale 1ns/1ps
`default_nettype none

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

    // Explicit compute-time base, in flat 32-bit word address units.
    // weight_addr = cfg_base + chunk_idx (all lanes share cfg_base; each
    // lane owns its own bank instance).
    input  wire [16:0]           cfg_base,

    // Compute-time read port A (primary / even chunk)
    input  wire [15:0]           chunk_idx,      // K-chunk within the tile row
    output wire [DATA_W-1:0]     rd_data,

    // Optional second read of the same tile row, at another chunk.
    // RD_PORTS=1 ties rd_data_b to zero.
    input  wire [15:0]           chunk_idx_b,
    output wire [DATA_W-1:0]     rd_data_b
);

    // Register the explicit base. The controller changes it between tiles
    // and waits at least one cycle (SETTLE) before the next array start, so
    // the synchronous RAM sees the settled address on the first read.
    reg [16:0] base_q;

    always_ff @(posedge clk) begin
        if (!rst_n) base_q <= '0;
        else        base_q <= cfg_base;
    end

    // Both read ports share the registered base.
    wire [16:0] rd_addr   = base_q + chunk_idx;
    wire [16:0] rd_addr_b = base_q + chunk_idx_b;

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
