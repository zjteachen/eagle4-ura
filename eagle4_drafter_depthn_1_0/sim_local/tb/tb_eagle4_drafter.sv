// =====================================================================
// tb_eagle4_drafter.sv — TESTBENCH ONLY (see BUILD.md / run_sim.sh).
//
// "Tier A" smoke test: instantiate the real eagle4_drafter RTL with
// G8_SPLICE=0 / DEPTH_N=0 / EMBG_EN=0 (plain parameter overrides, no RTL
// edits — see top_fsm.sv:262 and eagle4_drafter.sv:803 for why this
// collapses the datapath to a single draft step with no HBM round-trips
// for the vocab-rescore / child-embedding-gather paths). That means this
// testbench only needs to model ONE external memory interface for real
// (m_axi_out, the output write-DMA) — everything else the DUT would
// otherwise need HBM for is simply never triggered in this config.
//
// Everything is populated with random data (fixed-point-appropriate
// ranges, not full-width garbage — see eagle4_regmap_pkg comments). This
// is an I/O + control-flow smoke test, not a numerical-accuracy check —
// that needs a real trained EAGLE-3 checkpoint, out of scope here.
//
// Load paths used (see BUILD.md primer for what these terms mean):
//   WEIGHT / SCALE : TB direct-drive pins (boot_we/tb_sc_we) — fastest,
//                    no protocol, one word per clock.
//   EMB / HFC / GAMMA : AXI4-Lite register trio (0x48/0x4c/0x50), the
//                    only path these buffers have.
//
// KNOWN GAP (found running this, not fixed here — see TASKS.md): RoPE's
// cos/sin lookup table is loaded by a literal, non-parameterized
// `$readmemh("/home/y344shi/eagle4_artifacts/p4/rope_cs.hex", cs_lut)` in
// rope_layer.sv:284/328 — a hardcoded path from the original developer's
// machine, for a file that was never checked into this repo at all (unlike
// LMDOWN_MEM_DIR_P below, there is no parameter to override; the path is
// baked directly into the $readmemh call). The read silently fails (just a
// simulator warning, not a compile error), and our simulator's default
// zero-fill for uninitialized state means every RoPE cos/sin pair reads as
// (0,0) instead of a real rotation. This did not stop the pipeline from
// completing (see
// TASKS.md for the LUT indexing/format if you want to generate a real one:
// word = {cos_t[15:0], sin_t[15:0]}, indexed by pos*N_PAIRS+pair).
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_eagle4_drafter #(
    parameter string LMDOWN_DIR = "",  // absolute path to src/ (holding lmdown_W_q15_laneN.mem); set via -G from run_sim.sh
    // Full WEIGHT/SCALE depth is ~14.7M direct-drive pokes (real hardware
    // size). For fast iteration while debugging the harness itself,
    // run_sim.sh can override these down to a handful of words per lane via
    // -G; the RTL doesn't care how much of each bank actually got written,
    // it only reads what a given stage's addressing touches. Defaults to
    // the true depth so "no override" == the real, complete load.
    parameter int TEST_WEIGHT_DEPTH = -1,
    parameter int TEST_SCALE_DEPTH  = -1
);
    import eagle4_regmap_pkg::*;

    localparam int W_DEPTH = (TEST_WEIGHT_DEPTH < 0) ? WEIGHT_DEPTH : TEST_WEIGHT_DEPTH;
    localparam int S_DEPTH = (TEST_SCALE_DEPTH  < 0) ? SCALE_DEPTH  : TEST_SCALE_DEPTH;

    // ---------------- clock / reset ----------------
    // 100 MHz, matching the header note in eagle4_drafter.sv ("packaged at
    // FREQ_HZ=100 MHz").
    logic ap_clk = 1'b0;
    always #5 ap_clk = ~ap_clk;

    logic ap_rst_n = 1'b0;
    initial begin
        repeat (20) @(posedge ap_clk);
        ap_rst_n = 1'b1;
    end

    // ---------------- AXI4-Lite control bus (host <-> control_regs) ----------------
    logic        axil_awvalid, axil_wvalid, axil_bready, axil_arvalid, axil_rready;
    logic [8:0]  axil_awaddr, axil_araddr;
    logic [31:0] axil_wdata;
    logic [3:0]  axil_wstrb;
    wire         axil_awready, axil_wready, axil_bvalid;
    wire [1:0]   axil_bresp;
    wire         axil_arready, axil_rvalid;
    wire [31:0]  axil_rdata;
    wire [1:0]   axil_rresp;

    // ---------------- TB direct-drive pins (WEIGHT / SCALE) ----------------
    logic        boot_we;
    logic [15:0] boot_lane_sel;
    logic [16:0] boot_wr_addr;
    logic [31:0] boot_wr_data;

    logic        tb_sc_we;
    logic [15:0] tb_sc_lane_sel;
    logic [16:0] tb_sc_wr_addr;
    logic [15:0] tb_sc_wr_data;

    // ---------------- external-memory master ports we don't use in Tier A ----------------
    // These are all inputs *to* the DUT (the DUT is the AXI master on every
    // one of these ports) — tying them to idle/zero is safe because
    // G8_SPLICE=0/DEPTH_N=0/EMBG_EN=0 means the DUT logic that would issue
    // transactions here is never instantiated / never fires.
    localparam int BOOT_AW = 64, BOOT_DW = 32, BOOT_IW = 1, G8_DW = 256;

    wire [BOOT_IW-1:0] m_axi_boot_arid_u; wire [BOOT_AW-1:0] m_axi_boot_araddr_u;
    wire [7:0] m_axi_boot_arlen_u; wire [2:0] m_axi_boot_arsize_u; wire [1:0] m_axi_boot_arburst_u;
    wire m_axi_boot_arvalid_u; wire m_axi_boot_rready_u;

    wire [BOOT_IW-1:0] m_axi_g8_arid_u; wire [BOOT_AW-1:0] m_axi_g8_araddr_u;
    wire [7:0] m_axi_g8_arlen_u; wire [2:0] m_axi_g8_arsize_u; wire [1:0] m_axi_g8_arburst_u;
    wire m_axi_g8_arvalid_u; wire m_axi_g8_rready_u;

    wire [BOOT_IW-1:0] m_axi_g8b_arid_u; wire [BOOT_AW-1:0] m_axi_g8b_araddr_u;
    wire [7:0] m_axi_g8b_arlen_u; wire [2:0] m_axi_g8b_arsize_u; wire [1:0] m_axi_g8b_arburst_u;
    wire m_axi_g8b_arvalid_u; wire m_axi_g8b_rready_u;

    wire [BOOT_IW-1:0] m_axi_emb_arid_u; wire [BOOT_AW-1:0] m_axi_emb_araddr_u;
    wire [7:0] m_axi_emb_arlen_u; wire [2:0] m_axi_emb_arsize_u; wire [1:0] m_axi_emb_arburst_u;
    wire m_axi_emb_arvalid_u; wire m_axi_emb_rready_u;

    wire [BOOT_IW-1:0] m_axi_embb_arid_u; wire [BOOT_AW-1:0] m_axi_embb_araddr_u;
    wire [7:0] m_axi_embb_arlen_u; wire [2:0] m_axi_embb_arsize_u; wire [1:0] m_axi_embb_arburst_u;
    wire m_axi_embb_arvalid_u; wire m_axi_embb_rready_u;

    // ---------------- output write-DMA (the one master we model for real) ----------------
    wire [BOOT_IW-1:0] out_awid; wire [BOOT_AW-1:0] out_awaddr;
    wire [7:0] out_awlen; wire [2:0] out_awsize; wire [1:0] out_awburst; wire out_awvalid;
    wire out_awready;
    wire [BOOT_DW-1:0] out_wdata; wire [BOOT_DW/8-1:0] out_wstrb; wire out_wlast; wire out_wvalid;
    wire out_wready;
    wire [BOOT_IW-1:0] out_bid; wire [1:0] out_bresp; wire out_bvalid; wire out_bready;

    // ---------------- isolation-mode ports (unused: CONNECT_SEAMS=1 default) ----------------
    localparam int K_PAR = 16, A_WIDTH = 16;

    // ================================================================
    // DUT
    // ================================================================
    eagle4_drafter #(
        .G8_SPLICE(1'b0),
        .DEPTH_N  (1'b0),
        .EMBG_EN  (1'b0),
        .LMDOWN_MEM_DIR_P(LMDOWN_DIR)
    ) dut (
        .ap_clk  (ap_clk),
        .ap_rst_n(ap_rst_n),

        .s_axi_control_awaddr (axil_awaddr), .s_axi_control_awvalid(axil_awvalid), .s_axi_control_awready(axil_awready),
        .s_axi_control_wdata  (axil_wdata),  .s_axi_control_wstrb  (axil_wstrb),   .s_axi_control_wvalid (axil_wvalid), .s_axi_control_wready(axil_wready),
        .s_axi_control_bresp  (axil_bresp),  .s_axi_control_bvalid (axil_bvalid),  .s_axi_control_bready (axil_bready),
        .s_axi_control_araddr (axil_araddr), .s_axi_control_arvalid(axil_arvalid), .s_axi_control_arready(axil_arready),
        .s_axi_control_rdata  (axil_rdata),  .s_axi_control_rresp  (axil_rresp),   .s_axi_control_rvalid (axil_rvalid), .s_axi_control_rready(axil_rready),

        .boot_we(boot_we), .boot_lane_sel(boot_lane_sel), .boot_wr_addr(boot_wr_addr), .boot_wr_data(boot_wr_data),
        .tb_sc_we(tb_sc_we), .tb_sc_lane_sel(tb_sc_lane_sel), .tb_sc_wr_addr(tb_sc_wr_addr), .tb_sc_wr_data(tb_sc_wr_data),

        .m_axi_boot_arid(m_axi_boot_arid_u), .m_axi_boot_araddr(m_axi_boot_araddr_u), .m_axi_boot_arlen(m_axi_boot_arlen_u),
        .m_axi_boot_arsize(m_axi_boot_arsize_u), .m_axi_boot_arburst(m_axi_boot_arburst_u), .m_axi_boot_arvalid(m_axi_boot_arvalid_u),
        .m_axi_boot_arready(1'b0), .m_axi_boot_rid('0), .m_axi_boot_rdata('0), .m_axi_boot_rresp('0), .m_axi_boot_rlast(1'b0), .m_axi_boot_rvalid(1'b0),
        .m_axi_boot_rready(m_axi_boot_rready_u),

        .m_axi_out_awid(out_awid), .m_axi_out_awaddr(out_awaddr), .m_axi_out_awlen(out_awlen), .m_axi_out_awsize(out_awsize),
        .m_axi_out_awburst(out_awburst), .m_axi_out_awvalid(out_awvalid), .m_axi_out_awready(out_awready),
        .m_axi_out_wdata(out_wdata), .m_axi_out_wstrb(out_wstrb), .m_axi_out_wlast(out_wlast), .m_axi_out_wvalid(out_wvalid), .m_axi_out_wready(out_wready),
        .m_axi_out_bid(out_bid), .m_axi_out_bresp(out_bresp), .m_axi_out_bvalid(out_bvalid), .m_axi_out_bready(out_bready),

        .a_chunk('0), .qz_raw('0), .tile_idx('0),

        .m_axi_g8_arid(m_axi_g8_arid_u), .m_axi_g8_araddr(m_axi_g8_araddr_u), .m_axi_g8_arlen(m_axi_g8_arlen_u),
        .m_axi_g8_arsize(m_axi_g8_arsize_u), .m_axi_g8_arburst(m_axi_g8_arburst_u), .m_axi_g8_arvalid(m_axi_g8_arvalid_u),
        .m_axi_g8_arready(1'b0), .m_axi_g8_rid('0), .m_axi_g8_rdata('0), .m_axi_g8_rresp('0), .m_axi_g8_rlast(1'b0), .m_axi_g8_rvalid(1'b0),
        .m_axi_g8_rready(m_axi_g8_rready_u),

        .m_axi_g8b_arid(m_axi_g8b_arid_u), .m_axi_g8b_araddr(m_axi_g8b_araddr_u), .m_axi_g8b_arlen(m_axi_g8b_arlen_u),
        .m_axi_g8b_arsize(m_axi_g8b_arsize_u), .m_axi_g8b_arburst(m_axi_g8b_arburst_u), .m_axi_g8b_arvalid(m_axi_g8b_arvalid_u),
        .m_axi_g8b_arready(1'b0), .m_axi_g8b_rid('0), .m_axi_g8b_rdata('0), .m_axi_g8b_rresp('0), .m_axi_g8b_rlast(1'b0), .m_axi_g8b_rvalid(1'b0),
        .m_axi_g8b_rready(m_axi_g8b_rready_u),

        .m_axi_emb_arid(m_axi_emb_arid_u), .m_axi_emb_araddr(m_axi_emb_araddr_u), .m_axi_emb_arlen(m_axi_emb_arlen_u),
        .m_axi_emb_arsize(m_axi_emb_arsize_u), .m_axi_emb_arburst(m_axi_emb_arburst_u), .m_axi_emb_arvalid(m_axi_emb_arvalid_u),
        .m_axi_emb_arready(1'b0), .m_axi_emb_rid('0), .m_axi_emb_rdata('0), .m_axi_emb_rresp('0), .m_axi_emb_rlast(1'b0), .m_axi_emb_rvalid(1'b0),
        .m_axi_emb_rready(m_axi_emb_rready_u),

        .m_axi_embb_arid(m_axi_embb_arid_u), .m_axi_embb_araddr(m_axi_embb_araddr_u), .m_axi_embb_arlen(m_axi_embb_arlen_u),
        .m_axi_embb_arsize(m_axi_embb_arsize_u), .m_axi_embb_arburst(m_axi_embb_arburst_u), .m_axi_embb_arvalid(m_axi_embb_arvalid_u),
        .m_axi_embb_arready(1'b0), .m_axi_embb_rid('0), .m_axi_embb_rdata('0), .m_axi_embb_rresp('0), .m_axi_embb_rlast(1'b0), .m_axi_embb_rvalid(1'b0),
        .m_axi_embb_rready(m_axi_embb_rready_u)

        // stage_sel / array_chunk_idx_out / array_chunk_idx_valid / array_acc_out
        // are debug-probe outputs — left unconnected, harmless.
    );

    // ================================================================
    // Fake "HBM" behind the output write-DMA
    // ================================================================
    localparam longint OUT_MEM_BASE = 64'h0000_0000;
    localparam int      OUT_MEM_WORDS = 16384; // 64 KiB, comfortably covers out0/out1/out2 below

    axi4_mem_slave #(
        .ID_W(BOOT_IW), .ADDR_W(BOOT_AW), .DATA_W(BOOT_DW),
        .BASE_ADDR(OUT_MEM_BASE), .DEPTH_WORDS(OUT_MEM_WORDS), .RD_LATENCY(2)
    ) u_out_mem (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .awid(out_awid), .awaddr(out_awaddr), .awlen(out_awlen), .awsize(out_awsize), .awburst(out_awburst),
        .awvalid(out_awvalid), .awready(out_awready),
        .wdata(out_wdata), .wstrb(out_wstrb), .wlast(out_wlast), .wvalid(out_wvalid), .wready(out_wready),
        .bid(out_bid), .bresp(out_bresp), .bvalid(out_bvalid), .bready(out_bready),
        .arid('0), .araddr('0), .arlen('0), .arsize('0), .arburst('0), .arvalid(1'b0), .arready(),
        .rid(), .rdata(), .rresp(), .rlast(), .rvalid(), .rready(1'b1)
    );

    // ================================================================
    // Output layout inside the fake HBM (byte addresses, must not overlap)
    // ================================================================
    localparam longint OUT0_BASE = OUT_MEM_BASE + 64'h1000; // ids
    localparam longint OUT1_BASE = OUT_MEM_BASE + 64'h2000; // scores
    localparam longint OUT2_BASE = OUT_MEM_BASE + 64'h3000; // parents

    // ================================================================
    // Driver tasks
    // ================================================================

    task automatic axil_write(input int addr, input logic [31:0] data);
        axil_awaddr  = addr[8:0]; axil_awvalid = 1'b1;
        axil_wdata   = data; axil_wstrb = 4'hF; axil_wvalid = 1'b1;
        axil_bready  = 1'b1;
        do @(posedge ap_clk); while (!axil_awready);
        axil_awvalid = 1'b0;
        while (!axil_wready) @(posedge ap_clk);
        axil_wvalid = 1'b0;
        while (!axil_bvalid) @(posedge ap_clk);
        @(posedge ap_clk);
        axil_bready = 1'b0;
    endtask

    task automatic axil_read(input int addr, output logic [31:0] data);
        axil_araddr = addr[8:0]; axil_arvalid = 1'b1;
        axil_rready = 1'b1;
        do @(posedge ap_clk); while (!axil_arready);
        axil_arvalid = 1'b0;
        while (!axil_rvalid) @(posedge ap_clk);
        data = axil_rdata;
        @(posedge ap_clk);
        axil_rready = 1'b0;
    endtask

    // WEIGHT / SCALE: TB direct-drive pins. Level-sensitive write enable
    // (see gemv_stage_array_scaled_slr.sv:853, a plain `we && addr==lane`
    // comparison, not an edge-triggered pulse), so we can hold *_we high
    // continuously and change lane/addr/data every cycle — one word per
    // clock, no protocol overhead.
    task automatic load_all_weight_banks();
        int lane, addr;
        boot_we = 1'b1;
        for (lane = 0; lane < LANES; lane++) begin
            boot_lane_sel = lane[15:0];
            for (addr = 0; addr < W_DEPTH; addr++) begin
                boot_wr_addr = addr[16:0];
                boot_wr_data = $urandom();
                @(posedge ap_clk);
            end
            if (lane % 16 == 0)
                $display("[%0t] load_all_weight_banks: lane %0d/%0d", $time, lane, LANES-1);
        end
        boot_we = 1'b0;
        @(posedge ap_clk);
    endtask

    task automatic load_all_scale_banks();
        int lane, addr;
        tb_sc_we = 1'b1;
        for (lane = 0; lane < LANES; lane++) begin
            tb_sc_lane_sel = lane[15:0];
            for (addr = 0; addr < S_DEPTH; addr++) begin
                tb_sc_wr_addr = addr[16:0];
                tb_sc_wr_data = $urandom();
                @(posedge ap_clk);
            end
            if (lane % 16 == 0)
                $display("[%0t] load_all_scale_banks: lane %0d/%0d", $time, lane, LANES-1);
        end
        tb_sc_we = 1'b0;
        @(posedge ap_clk);
    endtask

    // EMB / HFC / GAMMA: AXI4-Lite trio, retargeted by whichever select bit
    // the caller already set. Modest-magnitude random int16 pairs per word
    // (not full 32'hxxxxxxxx) — see eagle4_regmap_pkg header: this is a
    // deliberate, easy-to-relax choice to avoid every activation saturating
    // to the fixed-point rail on the very first RMSNorm.
    function automatic logic [31:0] rand_activation_word();
        logic signed [15:0] lo, hi;
        lo = $urandom_range(16'sd4095, -16'sd4096);
        hi = $urandom_range(16'sd4095, -16'sd4096);
        return {hi, lo};
    endfunction

    task automatic load_vector_via_trio(input int n_words);
        int i;
        axil_write(ADDR_BOOT_WR_ADDR, 32'd0); // base address, once
        for (i = 0; i < n_words; i++)
            axil_write(ADDR_BOOT_WR_DATA, rand_activation_word());
    endtask

    task automatic read_residency(input int sel, output logic [31:0] count);
        axil_write(ADDR_RES_SEL, {28'b0, sel[2:0]});
        axil_read(ADDR_RES_COUNT, count);
    endtask

    // ================================================================
    // Main sequence
    // ================================================================
    localparam int TIMEOUT_CYCLES = 30_000_000;

    initial begin
        logic [31:0] rd;
        int errors;
        errors = 0;

        axil_awvalid = 0; axil_wvalid = 0; axil_bready = 0; axil_arvalid = 0; axil_rready = 0;
        boot_we = 0; boot_lane_sel = 0; boot_wr_addr = 0; boot_wr_data = 0;
        tb_sc_we = 0; tb_sc_lane_sel = 0; tb_sc_wr_addr = 0; tb_sc_wr_data = 0;

        @(posedge ap_rst_n);
        repeat (5) @(posedge ap_clk);

        $display("[%0t] ---- loading WEIGHT banks (%0d lanes x %0d words) ----", $time, LANES, W_DEPTH);
        load_all_weight_banks();

        $display("[%0t] ---- loading SCALE banks (%0d lanes x %0d words) ----", $time, LANES, S_DEPTH);
        load_all_scale_banks();

        $display("[%0t] ---- loading GAMMA buffers ----", $time);
        axil_write(ADDR_HFC_SELECT, 0); axil_write(ADDR_EMB_SELECT, 0);
        axil_write(ADDR_GAMMA_SEL, GAMMA_HIDDEN_NORM); load_vector_via_trio(GAMMA_WORDS);
        axil_write(ADDR_GAMMA_SEL, GAMMA_E_NORM);      load_vector_via_trio(GAMMA_WORDS);
        axil_write(ADDR_GAMMA_SEL, GAMMA_POSTATTN);    load_vector_via_trio(GAMMA_WORDS);
        axil_write(ADDR_GAMMA_SEL, GAMMA_FINAL);       load_vector_via_trio(GAMMA_WORDS);
        axil_write(ADDR_GAMMA_SEL, 0);

        $display("[%0t] ---- loading EMB (token embedding, %0d words) ----", $time, EMB_WORDS);
        axil_write(ADDR_EMB_SELECT, 1);
        load_vector_via_trio(EMB_WORDS);
        axil_write(ADDR_EMB_SELECT, 0);

        $display("[%0t] ---- loading HFC (low/mid/hi features, %0d words) ----", $time, HFC_WORDS);
        axil_write(ADDR_HFC_SELECT, 1);
        load_vector_via_trio(HFC_WORDS);
        axil_write(ADDR_HFC_SELECT, 0);

        // ---- ISA PMEM loading & instruction fetch verification ----
        $display("[%0t] ---- testing ISA PMEM loading over AXI (0x168/0x16C) ----", $time);
        // 1. Reset PMEM write address to 0
        axil_write(ADDR_PMEM_ADDR, 32'd0);
        axil_read(ADDR_PMEM_ADDR, rd);
        $display("  PMEM write_addr reset: got 0x%08x (expect 0x00000000)", rd);
        if (rd[7:0] !== 8'd0) errors++;

        // Load the ISA program for the linear draft pass (13 bytes):
        // bytes 0..11 = STATE G1,G4,G5,G6,G7G,G7U,G7_SILU,G9,G11,GLM,G12,TREE
        // byte 12 = HALT. State IDs follow the top_fsm enum (G11=9, GLM=13).
        // Word 0 (bytes 0..3):  G1,G4,G5,G6
        // Word 1 (bytes 4..7):  G7G,G7U,G7_SILU,G9
        // Word 2 (bytes 8..11): G11,GLM,G12,TREE
        // Word 3 (byte 12):     HALT
        axil_write(ADDR_PMEM_DATA, 32'h0403_0201);
        axil_write(ADDR_PMEM_DATA, 32'h0807_0605);
        axil_write(ADDR_PMEM_DATA, 32'h0B0A_0D09);
        axil_write(ADDR_PMEM_DATA, 32'h0000_0040);

        // Verify write address pointer auto-incremented to 4
        axil_read(ADDR_PMEM_ADDR, rd);
        $display("  PMEM write_addr after 4 writes: got 0x%08x (expect 0x00000004)", rd);
        if (rd[7:0] !== 8'd4) errors++;

        // 3. Verify fetcher initial state: PC=0, instr=0x01 (first byte of word 0)
        axil_read(ADDR_PMEM_DATA, rd);
        $display("  PMEM status at PC=0: got 0x%08x (canary=0x%04x, PC=%0d, instr=0x%02x; expect canary=0x504D, PC=0, instr=0x01)",
                 rd, rd[31:16], rd[14:8], rd[7:0]);
        if (rd[31:16] !== 16'h504D || rd[14:8] !== 7'd0 || rd[7:0] !== 8'h01) begin
            $display("  FAIL: PMEM fetcher initial word mismatch!");
            errors++;
        end else begin
            $display("  PASS: PMEM loaded and initial instruction 0x01 fetched successfully.");
        end

        // ---- residency self-check: did each load path actually land? ----
        $display("[%0t] ---- residency self-check (dbg_residency.sv) ----", $time);
        read_residency(RES_SEL_EMB,    rd); $display("  EMB    count = %0d (expect %0d)", rd, EMB_WORDS);
        if (rd !== EMB_WORDS) errors++;
        read_residency(RES_SEL_HFC,    rd); $display("  HFC    count = %0d (expect %0d)", rd, HFC_WORDS);
        if (rd !== HFC_WORDS) errors++;
        read_residency(RES_SEL_GAMMA,  rd); $display("  GAMMA  count = %0d (expect %0d)", rd, GAMMA_WORDS*4);
        if (rd !== GAMMA_WORDS*4) errors++;
        read_residency(RES_SEL_WEIGHT, rd); $display("  WEIGHT count = %0d (expect %0d)", rd, W_DEPTH*LANES);
        if (rd !== W_DEPTH*LANES) errors++;
        // SCALE is a known RTL instrumentation gap, not a load failure: the
        // dbg_residency counter for SCALE is wired to .sc_we(dma_sc_we) only
        // (eagle4_drafter.sv:7686) -- it never observes the tb_sc_we direct-
        // drive pin, only the AXI-DMA path. The data itself lands fine (both
        // paths OR together into the same eff_sc_we that actually drives
        // scale_bram -- see eagle4_drafter.sv:4859), this counter just can't
        // see it. So we log it but do not count it as an error.
        read_residency(RES_SEL_SCALE,  rd);
        $display("  SCALE  count = %0d (dbg_residency only watches the AXI-DMA path, not tb_sc_we -- 0 here is EXPECTED when loading via direct-drive, not a failure)", rd);

        if (errors != 0) begin
            $display("[%0t] FAIL: %0d load path(s) did not land the expected word count -- see BUILD.md/TASKS.md before trusting anything below.", $time, errors);
        end

        // ---- run config: single node, single frontier (the G8_SPLICE=0/DEPTH_N=0 pilot shape) ----
        axil_write(ADDR_CTX_LEN, 32'd0);
        axil_write(ADDR_NUM_FRONTIER, 32'd1);
        axil_write(ADDR_MAX_DEPTH, 32'd1);
        axil_write(ADDR_OUT0_LO, OUT0_BASE[31:0]); axil_write(ADDR_OUT0_HI, OUT0_BASE[63:32]);
        axil_write(ADDR_OUT1_LO, OUT1_BASE[31:0]); axil_write(ADDR_OUT1_HI, OUT1_BASE[63:32]);
        axil_write(ADDR_OUT2_LO, OUT2_BASE[31:0]); axil_write(ADDR_OUT2_HI, OUT2_BASE[63:32]);

        $display("[%0t] ---- ap_start ----", $time);
        axil_write(ADDR_AP_CTRL, 32'h1);

        // NOTE (bug fixed 2026-08-17): the first version of this loop used
        // `forever ... disable wait_timeout` inside the wait_done branch.
        // disable only kills the *sibling* branch -- it does not make the
        // forever loop itself return, so wait_done never "completed" and
        // join_any (which waits for a branch to finish, not for a disable
        // to fire) blocked forever even after the DUT had genuinely
        // finished. A `while` that falls off the end once the condition is
        // met is what actually lets join_any see completion.
        fork
            begin : wait_done
                logic [31:0] status;
                status = '0;
                while (!(status[1] || status[2] || status[3]))
                    axil_read(ADDR_OUT_DMA_STATUS, status);
                $display("[%0t] out_dma_status = 0x%08x (busy=%0d done=%0d err=%0d g8_wd_err=%0d)",
                         $time, status, status[0], status[1], status[2], status[3]);
            end
            begin : wait_timeout
                repeat (TIMEOUT_CYCLES) @(posedge ap_clk);
                $display("[%0t] FAIL: TIMEOUT waiting for out_dma_status.done after %0d cycles -- likely a real hang, not just a slow run.", $time, TIMEOUT_CYCLES);
                $finish;
            end
        join_any
        disable wait_timeout;

        // ---- decode + sanity-check the v24 ABI output (ids/scores/parents) ----
        $display("[%0t] ---- output (OUT_ABI_V24: ids / scores / parents, TOP_K=%0d, K_LIVE=%0d) ----", $time, TOP_K, K_LIVE);
        for (int t = 0; t < TOP_K; t++) begin
            logic signed [31:0] id_w, score_w, parent_w;
            automatic int out0_word0 = int'((OUT0_BASE - OUT_MEM_BASE) >> 2);
            automatic int out1_word0 = int'((OUT1_BASE - OUT_MEM_BASE) >> 2);
            automatic int out2_word0 = int'((OUT2_BASE - OUT_MEM_BASE) >> 2);
            id_w     = u_out_mem.mem[out0_word0 + t];
            score_w  = u_out_mem.mem[out1_word0 + t];
            parent_w = u_out_mem.mem[out2_word0 + t];
            $display("  t=%0d  id=%0d  score=%0d  parent=%0d", t, id_w, score_w, parent_w);

            if (t < K_LIVE) begin
                if (id_w < 0 || id_w >= VOCAB) begin
                    $display("    FAIL: id out of vocab range [0,%0d)", VOCAB);
                    errors++;
                end
            end else begin
                if (id_w !== 0 || score_w !== 0) begin
                    $display("    FAIL: expected zero-pad at slot %0d", t);
                    errors++;
                end
            end
            if (parent_w !== PARENT_SENTINEL) begin
                $display("    FAIL: parent != PARENT_SENTINEL(%0d)", PARENT_SENTINEL);
                errors++;
            end
        end
        // ---- Verify program ran to HALT: PC frozen at the HALT byte ----
        axil_read(ADDR_PMEM_DATA, rd);
        $display("[%0t] PMEM status post-run: got 0x%08x (canary=0x%04x, PC=%0d, instr=0x%02x; expect PC=12, instr=0x40 HALT)",
                 $time, rd, rd[31:16], rd[14:8], rd[7:0]);
        if (rd[31:16] !== 16'h504D || rd[14:8] !== 7'd12 || rd[7:0] !== 8'h40) begin
            $display("  FAIL: program did not halt at byte 12!");
            errors++;
        end else begin
            $display("  PASS: ISA program executed G1..TREE and halted with PC frozen at the HALT byte.");
        end

        // ---- ISA micro-test: JMP lands on the word-aligned target; a
        // second ap_start resets the PC and reruns ----
        $display("[%0t] ---- ISA micro-test: JMP + relaunch ----", $time);
        // Clear the sticky ap_done from the pipeline run (read-to-clear).
        axil_read(ADDR_AP_CTRL, rd);
        if (!rd[1]) begin
            $display("  FAIL: ap_done not set after the pipeline program HALT!");
            errors++;
        end
        // Program: byte 0 = JMP word 1 (0x21), bytes 1..3 skipped,
        // byte 4 = HALT. The skipped bytes are STATE S_IDLE (would stall
        // if executed) — reaching HALT proves the jump really moved the PC.
        axil_write(ADDR_PMEM_ADDR, 32'd0);
        axil_write(ADDR_PMEM_DATA, 32'h0000_0021);   // 0x21 = JMP, target word 1
        axil_write(ADDR_PMEM_DATA, 32'h0000_0040);   // 0x40 = HALT at byte 4
        axil_write(ADDR_AP_CTRL, 32'h1);
        begin : wait_isa_done
            logic [31:0] status;
            int n;
            status = '0;
            for (n = 0; (n < 1000) && !status[1]; n = n + 1)
                axil_read(ADDR_AP_CTRL, status);
            if (!status[1]) begin
                $display("[%0t] FAIL: timeout waiting for ap_done on the JMP micro-test.", $time);
                errors++;
            end
        end
        axil_read(ADDR_PMEM_DATA, rd);
        $display("  PMEM status after JMP run: got 0x%08x (PC=%0d, instr=0x%02x; expect PC=4, instr=0x40)",
                 rd, rd[14:8], rd[7:0]);
        if (rd[31:16] !== 16'h504D || rd[14:8] !== 7'd4 || rd[7:0] !== 8'h40) begin
            $display("  FAIL: JMP did not land on the word-aligned target!");
            errors++;
        end else begin
            $display("  PASS: JMP executed autonomously to byte 4 and HALT froze the PC there.");
        end


        if (errors == 0) $display("[%0t] PASS: pipeline ran end-to-end, output parses as a well-formed OUT_ABI_V24 frontier.", $time);
        else              $display("[%0t] FAIL: %0d check(s) failed.", $time, errors);

        $finish;
    end
endmodule
`default_nettype wire
