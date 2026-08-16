// =====================================================================
// xpm_memory_stubs.sv — LOCAL SIMULATION ONLY. DO NOT PACKAGE.
//
// Behavioral stand-ins for the two Xilinx XPM primitives this IP
// instantiates (xpm_memory_sdpram, xpm_memory_sprom). The real macros
// live inside the Vivado install (XILINX_VIVADO/data/ip/xpm) and are
// pulled in automatically by Vivado's elaboration/synthesis flow — they
// are NOT part of this repo. This file exists only so Verilator (or any
// simulator without a Vivado license/install on hand) can lint and
// functionally simulate this IP's own RTL locally.
//
// Scope: covers exactly the port/parameter subset this design actually
// drives (see kv_head_wide_win.sv, lmdown_gemv_banksplit.sv). Not a
// general-purpose XPM replacement — e.g. byte-lane writes
// (BYTE_WRITE_WIDTH_A < full word), ECC, and cascading are not modeled
// because this design never enables them (WRITE_PROTECT=1, single wea
// bit, ECC_MODE="no_ecc").
//
// Keep out of any file list used for synthesis/packaging (component.xml
// src fileset, gen_pkg_param_list.sh, etc.) — only feed it to a
// simulator-only invocation (see sim_local/run_lint.sh).
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

module xpm_memory_sdpram #(
    parameter int ADDR_WIDTH_A = 6,
    parameter int ADDR_WIDTH_B = 6,
    parameter int AUTO_SLEEP_TIME = 0,
    parameter int BYTE_WRITE_WIDTH_A = 32,
    parameter int CASCADE_HEIGHT = 0,
    parameter string CLOCKING_MODE = "common_clock",
    parameter string ECC_MODE = "no_ecc",
    parameter string MEMORY_INIT_FILE = "none",
    parameter string MEMORY_INIT_PARAM = "0",
    parameter string MEMORY_OPTIMIZATION = "true",
    parameter string MEMORY_PRIMITIVE = "auto",
    parameter int MEMORY_SIZE = 2048,
    parameter int MESSAGE_CONTROL = 0,
    parameter int READ_DATA_WIDTH_B = 32,
    parameter int READ_LATENCY_B = 1,
    parameter string READ_RESET_VALUE_B = "0",
    parameter string RST_MODE_A = "SYNC",
    parameter string RST_MODE_B = "SYNC",
    parameter int SIM_ASSERT_CHK = 0,
    parameter int USE_EMBEDDED_CONSTRAINT = 0,
    parameter int USE_MEM_INIT = 1,
    parameter int USE_MEM_INIT_MMI = 0,
    parameter string WAKEUP_TIME = "disable_sleep",
    parameter int WRITE_DATA_WIDTH_A = 32,
    parameter string WRITE_MODE_B = "read_first",
    parameter int WRITE_PROTECT = 1
)(
    input  wire                          clka,
    input  wire                          ena,
    input  wire                          wea,
    input  wire [ADDR_WIDTH_A-1:0]       addra,
    input  wire [WRITE_DATA_WIDTH_A-1:0] dina,
    input  wire                          injectsbiterra,
    input  wire                          injectdbiterra,

    input  wire                          clkb,
    input  wire                          rstb,
    input  wire                          enb,
    input  wire                          regceb,
    input  wire [ADDR_WIDTH_B-1:0]       addrb,
    output wire [READ_DATA_WIDTH_B-1:0]  doutb,
    output wire                          sbiterrb,
    output wire                          dbiterrb,

    input  wire                          sleep
);
    assign sbiterrb = 1'b0;
    assign dbiterrb = 1'b0;

    localparam int DEPTH = (1 << ADDR_WIDTH_A);

    reg [WRITE_DATA_WIDTH_A-1:0] mem [0:DEPTH-1];

    // read_first idiom: separate always blocks racing on the SAME edge
    // both sample pre-edge state for nonblocking RHS, so a same-address
    // R+W naturally returns OLD data on port B, matching WRITE_MODE_B.
    always @(posedge clka) begin
        if (ena && wea) mem[addra] <= dina;
    end

    reg [READ_DATA_WIDTH_B-1:0] pipe [0:(READ_LATENCY_B > 0 ? READ_LATENCY_B : 1)-1];
    integer pi;
    always @(posedge clkb) begin
        if (rstb) begin
            for (pi = 0; pi < READ_LATENCY_B; pi = pi + 1)
                pipe[pi] <= {READ_DATA_WIDTH_B{1'b0}};
        end else if (enb) begin
            pipe[0] <= mem[addrb];
            for (pi = 1; pi < READ_LATENCY_B; pi = pi + 1)
                pipe[pi] <= pipe[pi-1];
        end
    end
    assign doutb = pipe[READ_LATENCY_B-1];

    initial begin
        if (USE_MEM_INIT) begin
            for (int ii = 0; ii < DEPTH; ii = ii + 1) mem[ii] = '0;
        end
        if (MEMORY_INIT_FILE != "none")
            $readmemh(MEMORY_INIT_FILE, mem);
    end
endmodule

module xpm_memory_sprom #(
    parameter int ADDR_WIDTH_A = 6,
    parameter int MEMORY_SIZE = 2048,
    parameter string MEMORY_PRIMITIVE = "auto",
    parameter string MEMORY_INIT_FILE = "none",
    parameter string MEMORY_INIT_PARAM = "",
    parameter int READ_DATA_WIDTH_A = 32,
    parameter int READ_LATENCY_A = 1,
    parameter string READ_RESET_VALUE_A = "0",
    parameter string ECC_MODE = "no_ecc",
    parameter int AUTO_SLEEP_TIME = 0,
    parameter int USE_MEM_INIT = 1,
    parameter string WAKEUP_TIME = "disable_sleep",
    parameter int MESSAGE_CONTROL = 0
)(
    output wire [READ_DATA_WIDTH_A-1:0] douta,
    input  wire [ADDR_WIDTH_A-1:0]      addra,
    input  wire                         clka,
    input  wire                         ena,
    input  wire                         regcea,
    input  wire                         rsta,
    input  wire                         sleep,
    input  wire                         injectsbiterra,
    input  wire                         injectdbiterra,
    output wire                         sbiterra,
    output wire                         dbiterra
);
    assign sbiterra = 1'b0;
    assign dbiterra = 1'b0;

    localparam int DEPTH = (1 << ADDR_WIDTH_A);

    reg [READ_DATA_WIDTH_A-1:0] mem [0:DEPTH-1];

    reg [READ_DATA_WIDTH_A-1:0] pipe [0:(READ_LATENCY_A > 0 ? READ_LATENCY_A : 1)-1];
    integer pi;
    always @(posedge clka) begin
        if (rsta) begin
            for (pi = 0; pi < READ_LATENCY_A; pi = pi + 1)
                pipe[pi] <= {READ_DATA_WIDTH_A{1'b0}};
        end else if (ena) begin
            pipe[0] <= mem[addra];
            for (pi = 1; pi < READ_LATENCY_A; pi = pi + 1)
                pipe[pi] <= pipe[pi-1];
        end
    end
    assign douta = pipe[READ_LATENCY_A-1];

    initial begin
        if (USE_MEM_INIT) begin
            for (int ii = 0; ii < DEPTH; ii = ii + 1) mem[ii] = '0;
        end
        if (MEMORY_INIT_FILE != "none")
            $readmemh(MEMORY_INIT_FILE, mem);
    end
endmodule

`default_nettype wire
