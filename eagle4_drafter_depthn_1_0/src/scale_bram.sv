// Per-lane fixed-point dequantization scales, one per input group and output
// column. Address = stage_base + local_output_column * (K/GROUP) + group_idx.
// stage_geometry.svh supplies the same dimensions used by the weight bank.
//
// With 128 lanes and GROUP=128, G4 uses 72*48=3,456 entries per lane;
// all stages together use 12,180. Each entry is a signed fixed-point code
// interpreted by the GEMV lane, not an IEEE FP16 value.
//
// The column base and RAM output are registered. After changing stage/column,
// allow the base to settle; then supply group_idx with one-cycle lookahead.
// Load scales before compute; this module does not arbitrate overlapping use.

`timescale 1ns/1ps
`default_nettype none

// Shared weight/scale geometry.
`include "stage_geometry.svh"

(* keep_hierarchy = "yes" *)
module scale_bram #(
    // Per-lane capacity scales inversely with the number of output lanes.
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

    // Derive bases and strides rather than maintaining a second dimension table.
    function automatic [ADDR_W-1:0] stage_base(input [3:0] s);
        stage_base = sg_s_stage_base(s, GROUP, LANES);
    endfunction

    function automatic [15:0] stage_groups_per_col(input [3:0] s);
        stage_groups_per_col = sg_s_groups_per_col(s, GROUP);
    endfunction

    // Register the column base; the inner read-address path only adds group_idx.
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

    // One write port and one synchronous read port; request block RAM.
    (* ram_style = "block" *) reg [SCALE_W-1:0] mem [0:DEPTH-1];

`ifdef SIM_WSNAP
    // Simulation snapshot/restore; files are named by elaborated instance path.
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

    // Simulation-only zero fill. Hardware callers must load every scale used
    // by compute; this initialization can mask incomplete simulation loads.
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
