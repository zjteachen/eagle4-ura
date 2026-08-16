// =====================================================================
// weight_bank_uram.sv
// Per-lane URAM-resident qweight bank — Phase 1 V5d primitive.
// =====================================================================
//
// Stores DEPTH × DATA_W bits of qweight (one lane's slice).  Dual-port:
//   - Write port  (boot mode) : externally driven during weight load,
//                                e.g. via AXI4-Lite into a streaming FSM.
//   - Read  port  (compute)   : driven by the lane's chunk_idx counter
//                                during the GEMV inner loop.
//
// Synthesis intent: Vivado infers URAM288 on V80 when:
//   - DEPTH >= ~512 AND DATA_W >= 32   (sweet spot for URAM mapping)
//   - (* ram_style = "ultra" *) attribute is present
// At pilot scale (DEPTH=16 for K=N=64 pilot) Vivado may map to LUTRAM
// or BRAM instead — that's fine for functional verification; URAM
// inference is meaningfully validated at V9 (K=N=3072 production scale).
//
// Read latency: 1 cycle (registered output).
//
// ---------------------------------------------------------------------
// WO-025 DUAL-READ-PORT (2nd compute read, 2026-06-03)
// ---------------------------------------------------------------------
// URAM288 on V80 is a TRUE-DUAL-PORT macro (ports A and B, each
// independently address+read+write). In Layout B the boot write shares
// port A and port B is IDLE during compute. The GEMV is URAM-READ-PORT
// bound (one K_PAR-wide row = one chunk per cycle), so driving the idle
// port B as a SECOND compute read fetches a SECOND chunk per cycle with
// the SAME URAM block count (the port physically exists). RD_PORTS=2
// exposes that 2nd read (rd_addr_b/rd_data_b); RD_PORTS=1 (default) is
// byte+cycle identical to the original simple-dual-port (1 write, 1 read)
// — the 2nd-port logic is compiled away and rd_data_b ties to 0.
//
// The boot write stays on port A only (single-writer, multi-driver-safe;
// feedback_axi_driver_fanin_rule). Both compute reads are READ-ONLY into
// the same array, so there is no write/write or read/write hazard on the
// step datapath (the boot loader has finished before compute begins).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module weight_bank_uram #(
    parameter int DEPTH    = 4096,
    parameter int DATA_W   = 32,
    parameter int RD_PORTS = 1,                      // WO-025: 1 = legacy SDP, 2 = dual-read
    parameter int ADDR_W   = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    // Write port (boot) — port A only
    input  wire                  we,
    input  wire [ADDR_W-1:0]     wr_addr,
    input  wire [DATA_W-1:0]     wr_data,
    // Read port A (compute, primary / even chunk)
    input  wire [ADDR_W-1:0]     rd_addr,
    output reg  [DATA_W-1:0]     rd_data,
    // Read port B (compute, WO-025 dual; ignored when RD_PORTS==1)
    input  wire [ADDR_W-1:0]     rd_addr_b,
    output wire [DATA_W-1:0]     rd_data_b
);

    // Force URAM by attribute. Vivado will downgrade to BRAM / LUTRAM at
    // small depths automatically.
    //
    // LANES-256 PACKING FIX (2026-07-26, LANES256_GEMM_STUDY.md §1.2):
    // cascade_height=8 makes the mapper build ONE cascade column instead of
    // rounding each depth-block to the 4-URAM (32K-entry) Matrix shape.
    //   48K bank (LANES=256): default map = 8 URAM ("Matrix 2x1" rounding)
    //     -> 8x256 = 2,048 total = 106.4% of 1,925 = DEAD.  With cascade:
    //     "Matrix 6x1" = 6 URAM/bank -> 6x256 = 1,536 = URAM-NEUTRAL vs the
    //     golden.  Bank WNS +8.227 ns @100 MHz measured (ucasc48).
    //   96K bank (LANES=128): ideal 12 URAM either way (count can't shrink);
    //     census gate re-verifies 12 with the attribute (inert at default).
    (* ram_style = "ultra", cascade_height = 8 *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

`ifdef SIM_WSNAP
    // SIM-ONLY snapshot/restore (see pdi10_fix/weight_bank_uram.sv for the
    // full rationale). +WSNAP_DUMP=<dir> writes this instance's mem[] keyed by
    // %m after the load settles; +WSNAP_LOAD=<dir> restores it in seconds and
    // the TB can skip the boot walk. Guarded define: synthesis never sees it.
    string  wsnap_dir;
    longint wsnap_t;
    initial begin
        if ($value$plusargs("WSNAP_LOAD=%s", wsnap_dir)) begin
            int wsnap_fd;
            #1;
            // GUARD (2026-07-25): %m paths shift if the elaboration hierarchy
            // changes (e.g. a parameter like FR_KF_BANKS altering generate
            // structure). A missing file => $readmemh loads NOTHING => zero
            // weights => zero-logit mirage (frontier [0..7]). Fail LOUD instead.
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

    // Initialize for sim determinism — does not generate any logic in
    // synthesis on Versal; simulator picks it up.
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    generate
        if (RD_PORTS >= 2) begin : g_tdp
            // ---- WO-025 Stage-B' TDP — the m5 2-ADDRESS template, VERBATIM ----
            // (weight_bank_uram_tdp.sv, urambw m5 GO 2026-07-17: URAM 12==12,
            // ZERO [Synth 8-3971], util_m5_tdp32.rpt.) The earlier RD_PORTS=2
            // form presented THREE address buses (wr_addr, rd_addr, rd_addr_b)
            // -> Vivado classified TDP-with-write and DEMOTED the whole bank
            // URAM->BRAM/LUTRAM (gate-3 census 2026-07-26: URAM 1536->0).
            // m5 form: port A = pure read; port B = ONE shared address bus,
            // boot WRITE while we=1, compute READ (odd chunk) otherwise. Boot
            // and compute are temporally disjoint (boot loader completes before
            // any GEMV start), so the mux is hazard-free by construction.
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
            // ---- Legacy SDP (RD_PORTS==1): write (boot) + primary compute
            // read — UNCHANGED, byte-identical to the shipping bank. ----
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
