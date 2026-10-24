// Inferred per-lane weight RAM with synchronous, one-cycle compute reads.
// RD_PORTS=1: one write port and one read port; rd_data_b is tied to zero.
// RD_PORTS>=2: port A reads; port B writes during boot or reads during compute.
// Boot loading must finish before compute. No read/write arbitration is provided.
// Attributes request UltraRAM and a cascade height; verify actual mapping
// in synthesis reports rather than inferring resource use from this source.

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module weight_bank_uram #(
    parameter int DEPTH    = 4096,
    parameter int DATA_W   = 32,
    parameter int RD_PORTS = 1,                      // 1 = single read, >=2 = dual read
    parameter int ADDR_W   = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    // Boot write port (shares port B with the secondary read in dual-read mode).
    input  wire                  we,
    input  wire [ADDR_W-1:0]     wr_addr,
    input  wire [DATA_W-1:0]     wr_data,
    // Read port A (compute, primary / even chunk)
    input  wire [ADDR_W-1:0]     rd_addr,
    output reg  [DATA_W-1:0]     rd_data,
    // Secondary compute read; ignored when RD_PORTS==1.
    input  wire [ADDR_W-1:0]     rd_addr_b,
    output wire [DATA_W-1:0]     rd_data_b
);

    // Preserve the RAM inference and cascade attributes across refactors.
    (* ram_style = "ultra", cascade_height = 8 *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

`ifdef SIM_WSNAP
    // Simulation snapshot/restore, keyed by elaborated instance path (%m).
    // SIM_WSNAP must not be enabled for synthesis.
    string  wsnap_dir;
    longint wsnap_t;
    initial begin
        if ($value$plusargs("WSNAP_LOAD=%s", wsnap_dir)) begin
            int wsnap_fd;
            #1;
            // Hierarchy changes invalidate snapshot names. Reject missing files
            // rather than continuing with zero-initialized weights.
            wsnap_fd = $fopen($sformatf("%s/%m.mem", wsnap_dir), "r");
            if (wsnap_fd == 0)
                $fatal(1, "WSNAP_LOAD: no snapshot for %m (hierarchy mismatch? snapshot is only valid for an IDENTICAL elaboration). Re-run with a full boot load.");
            $fclose(wsnap_fd);
            $readmemh($sformatf("%s/%m.mem", wsnap_dir), mem);
        end
        if ($value$plusargs("WSNAP_DUMP=%s", wsnap_dir)) begin
            if (!$value$plusargs("WSNAP_T=%d", wsnap_t)) wsnap_t = 64'd63_300_000; // 63.3 ms in NS units (timescale 1ns) — post-load
            #(wsnap_t);
            $writememh($sformatf("%s/%m.mem", wsnap_dir), mem);
        end
    end
`endif

    // Initial zero fill aids simulation; it is not a substitute for boot loading.
    // This block is not guarded by SYNTHESIS; device initialization is tool-dependent.
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    generate
        if (RD_PORTS >= 2) begin : g_tdp
            // Two-address RAM template: pure read on A, shared read/write
            // address on B. Separate write and secondary-read address buses
            // would describe three ports and can prevent UltraRAM inference.
            wire [ADDR_W-1:0] addr_b = we ? wr_addr : rd_addr_b;
            // Port A: pure sync read
            always_ff @(posedge clk) begin
                rd_data <= mem[rd_addr];
            end
            // Port B: read-or-write, ONE address (classic TDP port, no-change read)
            reg [DATA_W-1:0] rd_data_b_r;
            always_ff @(posedge clk) begin
                if (we) mem[addr_b]  <= wr_data;
                else    rd_data_b_r  <= mem[addr_b];
            end
            assign rd_data_b = rd_data_b_r;
        end else begin : g_sdp
            // Single-read mode resets the output register, not the stored weights.
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    rd_data <= '0;
                end else begin
                    if (we) mem[wr_addr] <= wr_data;
                    rd_data <= mem[rd_addr];
                end
            end
            assign rd_data_b = '0;     // inert; port-B inputs unused
        end
    endgenerate

endmodule

`default_nettype wire
