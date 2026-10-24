// Per-lane fixed-point dequantization scales, one per input group and output
// column. Address = cfg_base + group_idx, where cfg_base is an EXPLICIT flat
// scale-entry address supplied by the shared operation controller (scale_base
// plus the current tile's row stride). The controller owns tile addressing;
// stage_geometry no longer participates in this bank.
//
// Each entry is a signed fixed-point code interpreted by the GEMV lane, not
// an IEEE FP16 value. GROUP is no longer a compile-time address stride: the
// lane computes the runtime group index from chunks_per_scale_group and this
// bank only adds it to the explicit base.
//
// The base and RAM output are registered. The controller changes the base
// between tiles and waits at least one SETTLE cycle before the next start;
// group_idx is supplied with one-cycle lookahead while a tile runs. Load
// scales before compute; this module does not arbitrate overlapping use.

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module scale_bram #(
    // Per-lane capacity scales inversely with the number of output lanes.
    parameter int LANES      = 128,
    parameter int DEPTH      = (16 * 1024 * 128) / LANES, // per-lane scale entries (headroom)
    parameter int SCALE_W    = 16,          // Q(16-frac).frac signed scale
    parameter int ADDR_W     = 17,
    parameter int GW         = 8            // group-index width (covers 0..128 groups)
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Boot-time write port (driven by the scale-load FSM / AXI-Lite)
    input  wire                  we,
    input  wire [ADDR_W-1:0]     wr_addr,        // flat scale address
    input  wire [SCALE_W-1:0]    wr_data,

    // Explicit compute-time base, in flat scale-entry address units.
    input  wire [ADDR_W-1:0]     cfg_base,

    // Compute-time read port
    input  wire [GW-1:0]         group_idx,      // runtime group within the tile row
    output reg  [SCALE_W-1:0]    rd_data
);

    // Register the explicit base (see weight_bank_multistage for the settle
    // contract). The per-cycle path only adds group_idx.
    reg [ADDR_W-1:0] base_q;
    always_ff @(posedge clk) begin
        if (!rst_n) base_q <= '0;
        else        base_q <= cfg_base;
    end

    wire [ADDR_W-1:0] rd_addr = base_q + {{(ADDR_W-GW){1'b0}}, group_idx};

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
