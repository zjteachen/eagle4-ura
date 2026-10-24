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

        .a_chunk('0),

        .stage_sel(dbg_stage_sel),
        .array_chunk_idx_out(dbg_array_chunk_idx),
        .array_chunk_idx_valid(dbg_array_chunk_idx_valid),
        .array_acc_out(dbg_array_acc),

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

    );

    wire [3:0]  dbg_stage_sel;
    wire [15:0] dbg_array_chunk_idx;
    wire        dbg_array_chunk_idx_valid;
    wire signed [32*128-1:0] dbg_array_acc;

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
    // Isolated ISA WRSR + special-register bank smoke test.
    // No WEIGHT/SCALE/activation loading: this proves the ISA control path
    // (PMEM upload -> WRSR -> HALT) and the actual bank's AXI readback.
    // ================================================================
    bit sr_test_active = 1'b0;
    int sr_write_strobes = 0;
    int sr_read_collisions = 0;

    always @(posedge ap_clk) begin
        if (ap_rst_n && sr_test_active) begin
            if (dut.g1_start || dut.gemv_op_start_w || dut.g5_start || dut.g6_start ||
                dut.g7g_start || dut.g7u_start || dut.g7p_start ||
                dut.g9_start || dut.g11_start || dut.glm_start ||
                dut.tree_start || dut.g8_start)
                $fatal(1, "REGS-only: a stage started");
            if (out_awvalid || out_wvalid || m_axi_boot_arvalid_u ||
                m_axi_g8_arvalid_u || m_axi_g8b_arvalid_u ||
                m_axi_emb_arvalid_u || m_axi_embb_arvalid_u)
                $fatal(1, "REGS-only: unexpected external-memory transaction");
            if (dut.u_special_regs.write_en)
                sr_write_strobes <= sr_write_strobes + 1;
            // A read request overlapped a WRSR write, so the bank's write
            // priority suppressed that attempt and control_regs retried.
            if (dut.u_special_regs.read_en && dut.u_special_regs.write_en)
                sr_read_collisions <= sr_read_collisions + 1;
        end
    end

    task automatic sr_read_reg(input logic [4:0] sel, output logic [31:0] data);
        axil_write(ADDR_SR_SEL, {27'b0, sel});
        axil_read(ADDR_SR_DATA, data);
    endtask

    task automatic wait_pmem_ptr(input int expected, input string tag);
        logic [31:0] st;
        int n;
        st = '0;
        for (n = 0; n < 20 && st[7:0] != expected[7:0]; n++)
            axil_read(ADDR_PMEM_ADDR, st);
        if (st[7:0] !== expected[7:0])
            $fatal(1, "REGS-only: %s: PMEM pointer %0d, expected %0d",
                   tag, st[7:0], expected);
    endtask

    task automatic check_pmem(input int exp_pc, input logic [7:0] exp_instr,
                              input string tag);
        logic [31:0] st;
        axil_read(ADDR_PMEM_DATA, st);
        if (st[31:16] !== 16'h504D || st[14:8] !== exp_pc[6:0] ||
            st[7:0] !== exp_instr)
            $fatal(1, "REGS-only: %s: expected PC=%0d instr=%02x, got %08x",
                   tag, exp_pc, exp_instr, st);
    endtask

    task automatic wait_ap_done(input string tag);
        logic [31:0] st;
        int n;
        st = '0;
        for (n = 0; n < 5000 && !st[1]; n++) axil_read(ADDR_AP_CTRL, st);
        if (!st[1]) $fatal(1, "REGS-only: %s: timeout waiting for ap_done", tag);
    endtask

    task automatic pmem_write_at(input int addr, input logic [31:0] data,
                                 input int next_ptr, input string tag);
        axil_write(ADDR_PMEM_ADDR, addr);
        axil_write(ADDR_PMEM_DATA, data);
        wait_pmem_ptr(next_ptr, tag);
    endtask

    task automatic run_regs_only();
        logic [31:0] rd;
        logic [7:0]  p [0:127];
        logic [31:0] pw [0:31];
        logic [15:0] vals [0:10];
        int nb, nw, n, w, i;
        int base_writes;

        $display("[%0t] ---- ISA WRSR + special_regs AXI readback test ----", $time);

        // Deterministic nonzero values. r6/r8 set bank-address bit16; r9
        // packs source=3, destination=9, qzero=1, scale_frac=13.
        vals[0]  = 16'h0001;
        vals[1]  = 16'h2345;
        vals[2]  = 16'h0003;
        vals[3]  = 16'h0008;
        vals[4]  = 16'h0008;
        vals[5]  = 16'h0002;
        vals[6]  = 16'h0001;
        vals[7]  = 16'h3456;
        vals[8]  = 16'h0001;
        vals[9]  = 16'h1CAD;
        vals[10] = 16'hBEEF;

        // Reset state: all entries and decoded config outputs are zero, and
        // an out-of-range selector reads zero with the canary intact.
        sr_read_reg(5'd0, rd);
        if (rd !== 32'h5A5A_0000) $fatal(1, "REGS-only: reset r0 %08x", rd);
        sr_read_reg(5'd15, rd);
        if (rd !== 32'h5A5A_0000) $fatal(1, "REGS-only: reset reserved %08x", rd);
        if (dut.u_special_regs.weight_base !== 17'd0 ||
            dut.u_special_regs.scale_base !== 17'd0 ||
            dut.u_special_regs.activation_source !== 3'd0)
            $fatal(1, "REGS-only: config outputs not cleared by reset");

        // Program A: write registers 0..10, then HALT at byte 33. WRSR
        // instructions start at PCs 0,3,6,...,30, so most immediates cross
        // a 32-bit PMEM word boundary.
        nb = 0;
        for (i = 0; i < 11; i++) begin
            p[nb]     = 8'h60 | 8'(i);
            p[nb + 1] = vals[i][7:0];
            p[nb + 2] = vals[i][15:8];
            nb = nb + 3;
        end
        p[nb] = 8'h40;
        nb = nb + 1;
        nw = (nb + 3) / 4;
        for (i = nb; i < nw * 4; i++) p[i] = 8'h40;
        for (w = 0; w < nw; w++)
            pw[w] = {p[4*w + 3], p[4*w + 2], p[4*w + 1], p[4*w]};

        axil_write(ADDR_PMEM_ADDR, 0);
        for (w = 0; w < nw; w++) axil_write(ADDR_PMEM_DATA, pw[w]);
        wait_pmem_ptr(nw, "program A");
        check_pmem(0, 8'h60, "program A pre-run");

        sr_test_active = 1'b1;
        base_writes = sr_write_strobes;
        axil_write(ADDR_AP_CTRL, 1);
        wait_ap_done("program A");
        check_pmem(33, 8'h40, "program A");
        @(negedge ap_clk);
        if (sr_write_strobes - base_writes != 11)
            $fatal(1, "REGS-only: program A wrote %0d times, expected 11",
                   sr_write_strobes - base_writes);

        for (i = 0; i < 11; i++) begin
            sr_read_reg(5'(i), rd);
            if (rd[31:16] !== 16'h5A5A || rd[15:0] !== vals[i])
                $fatal(1, "REGS-only: r%0d readback %08x, expected %04x",
                       i, rd, vals[i]);
        end
        for (i = 11; i < 16; i++) begin
            sr_read_reg(5'(i), rd);
            if (rd !== 32'h5A5A_0000)
                $fatal(1, "REGS-only: reserved r%0d readback %08x", i, rd);
        end
        sr_read_reg(5'd31, rd);
        if (rd !== 32'h5A5A_0000)
            $fatal(1, "REGS-only: reserved r31 readback %08x", rd);

        // Decoded config fields (the next task's engine handoff).
        if (dut.u_special_regs.activation_base !== 16'h0001 ||
            dut.u_special_regs.weight_base !== 17'h12345 ||
            dut.u_special_regs.tile_count !== 16'h0003 ||
            dut.u_special_regs.chunks_per_tile !== 16'h0008 ||
            dut.u_special_regs.chunks_per_scale_group !== 16'h0008 ||
            dut.u_special_regs.output_base !== 16'h0002 ||
            dut.u_special_regs.scale_base !== 17'h13456 ||
            dut.u_special_regs.activation_source !== 3'd3 ||
            dut.u_special_regs.output_destination !== 4'd9 ||
            dut.u_special_regs.qzero_raw !== 2'd1 ||
            dut.u_special_regs.scale_frac !== 5'd13 ||
            dut.u_special_regs.output_conversion !== 16'hBEEF)
            $fatal(1, "REGS-only: decoded config outputs mismatch");

        // Read-to-clear ap_done followed by idle/ready.
        repeat (4) @(posedge ap_clk);
        axil_read(ADDR_AP_CTRL, rd);
        if (rd[1] || !rd[2] || !rd[3])
            $fatal(1, "REGS-only: expected idle/ready with ap_done clear, got %08x", rd);

        // Restart: the same program reruns, writing each register once more.
        base_writes = sr_write_strobes;
        axil_write(ADDR_AP_CTRL, 1);
        wait_ap_done("program A restart");
        check_pmem(33, 8'h40, "program A restart");
        @(negedge ap_clk);
        if (sr_write_strobes - base_writes != 11)
            $fatal(1, "REGS-only: restart wrote %0d times, expected 11",
                   sr_write_strobes - base_writes);
        sr_read_reg(5'd10, rd);
        if (rd !== 32'h5A5A_BEEF) $fatal(1, "REGS-only: r10 after restart %08x", rd);

        // Program B: WRSR at byte 3 (unaligned start) with its two immediate
        // bytes in word 1, plus a reserved-address write. Word bytes:
        //   0: 61 AA AA 62 | 4: 55 55 6F AD | 8: DE 40 40 40
        axil_write(ADDR_PMEM_ADDR, 0);
        axil_write(ADDR_PMEM_DATA, 32'h62AA_AA61);
        axil_write(ADDR_PMEM_DATA, 32'hAD6F_5555);
        axil_write(ADDR_PMEM_DATA, 32'h4040_40DE);
        wait_pmem_ptr(3, "program B");
        base_writes = sr_write_strobes;
        axil_write(ADDR_AP_CTRL, 1);
        wait_ap_done("program B");
        check_pmem(9, 8'h40, "program B");
        @(negedge ap_clk);
        if (sr_write_strobes - base_writes != 3)
            $fatal(1, "REGS-only: program B wrote %0d times, expected 3",
                   sr_write_strobes - base_writes);
        sr_read_reg(5'd1, rd);
        if (rd !== 32'h5A5A_AAAA) $fatal(1, "REGS-only: crossing r1 %08x", rd);
        sr_read_reg(5'd2, rd);
        if (rd !== 32'h5A5A_5555) $fatal(1, "REGS-only: crossing r2 %08x", rd);
        sr_read_reg(5'd15, rd);
        if (rd !== 32'h5A5A_0000) $fatal(1, "REGS-only: reserved write took %08x", rd);
        sr_read_reg(5'd0, rd);
        if (rd !== 32'h5A5A_0001) $fatal(1, "REGS-only: r0 disturbed %08x", rd);
        sr_read_reg(5'd3, rd);
        if (rd !== 32'h5A5A_0008) $fatal(1, "REGS-only: r3 disturbed %08x", rd);

        // Program C: end-of-PMEM WRSR. JMP at byte 0 targets word 31; two
        // WRSRs at bytes 124 and 127. The byte-127 instruction's immediate
        // bytes are the wrapped bytes 0 (low) and 1 (high); PC then wraps
        // to byte 2, where HALT sits.
        pmem_write_at(0,  32'h4040_5A3F, 1, "program C word0");
        pmem_write_at(31, 32'h6512_3464, 0, "program C word31");
        base_writes = sr_write_strobes;
        axil_write(ADDR_AP_CTRL, 1);
        wait_ap_done("program C");
        check_pmem(2, 8'h40, "program C");
        @(negedge ap_clk);
        if (sr_write_strobes - base_writes != 2)
            $fatal(1, "REGS-only: program C wrote %0d times, expected 2",
                   sr_write_strobes - base_writes);
        sr_read_reg(5'd4, rd);
        if (rd !== 32'h5A5A_1234) $fatal(1, "REGS-only: end-wrap r4 %08x", rd);
        sr_read_reg(5'd5, rd);
        if (rd !== 32'h5A5A_5A3F) $fatal(1, "REGS-only: end-wrap r5 %08x", rd);
        sr_read_reg(5'd1, rd);
        if (rd !== 32'h5A5A_AAAA) $fatal(1, "REGS-only: r1 lost %08x", rd);
        sr_read_reg(5'd2, rd);
        if (rd !== 32'h5A5A_5555) $fatal(1, "REGS-only: r2 lost %08x", rd);
        sr_read_reg(5'd7, rd);
        if (rd !== 32'h5A5A_3456) $fatal(1, "REGS-only: r7 lost %08x", rd);

        // Reserved opcode (0xE0): fail-safe HALT, no bank write, no stage.
        pmem_write_at(0, 32'h4040_40E0, 1, "reserved opcode");
        base_writes = sr_write_strobes;
        axil_write(ADDR_AP_CTRL, 1);
        wait_ap_done("reserved opcode");
        check_pmem(0, 8'hE0, "reserved opcode");
        @(negedge ap_clk);
        if (sr_write_strobes - base_writes != 0)
            $fatal(1, "REGS-only: reserved opcode wrote the bank");

        // JMP semantics preserved: JMP word 0 -> word 1; HALT at byte 4.
        axil_write(ADDR_PMEM_ADDR, 0);
        axil_write(ADDR_PMEM_DATA, 32'h0000_0021);
        axil_write(ADDR_PMEM_DATA, 32'h0000_0040);
        wait_pmem_ptr(2, "JMP program");
        base_writes = sr_write_strobes;
        axil_write(ADDR_AP_CTRL, 1);
        wait_ap_done("JMP program");
        check_pmem(4, 8'h40, "JMP program");
        @(negedge ap_clk);
        if (sr_write_strobes - base_writes != 0)
            $fatal(1, "REGS-only: JMP wrote the bank");

        // Arbitration: a 22-WRSR + JMP loop writes the bank on 22 of every
        // 23 cycles. AXI 0x17C reads issued while it runs must retry through
        // write-suppressed bank reads and return the live value, not stale
        // pre-loop data. r0 changes from 0x0001 to 0x1234 on the first
        // iteration; r1 becomes 0x5678.
        nb = 0;
        for (i = 0; i < 21; i++) begin
            p[nb]     = 8'h60;
            p[nb + 1] = 8'h34;
            p[nb + 2] = 8'h12;
            nb = nb + 3;
        end
        p[nb]     = 8'h61;   // byte 63: WRSR r1 (immediates at 64/65)
        p[nb + 1] = 8'h78;
        p[nb + 2] = 8'h56;
        nb = nb + 3;
        p[nb] = 8'h20;       // byte 66: JMP word 0
        nb = nb + 1;
        p[nb] = 8'h40;       // pad
        nb = nb + 1;
        nw = (nb + 3) / 4;
        for (i = nb; i < nw * 4; i++) p[i] = 8'h40;
        for (w = 0; w < nw; w++)
            pw[w] = {p[4*w + 3], p[4*w + 2], p[4*w + 1], p[4*w]};

        axil_write(ADDR_PMEM_ADDR, 0);
        for (w = 0; w < nw; w++) axil_write(ADDR_PMEM_DATA, pw[w]);
        wait_pmem_ptr(nw, "program D");
        sr_read_reg(5'd0, rd);
        if (rd !== 32'h5A5A_0001) $fatal(1, "REGS-only: r0 pre-loop %08x", rd);

        base_writes = sr_write_strobes;
        axil_write(ADDR_AP_CTRL, 1);
        for (n = 0; n < 40 && (sr_write_strobes - base_writes) < 4; n++)
            @(posedge ap_clk);
        if (sr_write_strobes - base_writes < 4)
            $fatal(1, "REGS-only: arbitration loop did not start");
        axil_read(ADDR_AP_CTRL, rd);
        if (rd[1] || rd[2])
            $fatal(1, "REGS-only: expected busy kernel during arbitration, got %08x", rd);

        for (i = 0; i < 8; i++) begin
            sr_read_reg(5'd0, rd);
            if (rd !== 32'h5A5A_1234)
                $fatal(1, "REGS-only: arbitration r0 read %08x (stale?)", rd);
        end
        sr_read_reg(5'd1, rd);
        if (rd !== 32'h5A5A_5678)
            $fatal(1, "REGS-only: arbitration r1 read %08x", rd);
        if (sr_read_collisions == 0)
            $fatal(1, "REGS-only: no bank read/WRSR collision was exercised");
        $display("PASS: WRSR readback, crossings, reserved, restart, JMP/HALT, wrap, arbitration (collisions=%0d).",
                 sr_read_collisions);
    endtask

    // Isolated ISA contract: STATE G1 must finish both full-width norms,
    // then HALT without launching a downstream stage or external DMA.
    // Use the clean output strobes, not rmsnorm_apply's delayed out_valid.
    bit g1_test_active = 1'b0;
    int g1_starts = 0, g1_completions = 0, g1e_completions = 0;
    int g1_h_chunks = 0, g1_e_chunks = 0, g1_halts = 0;
    localparam int G1_CHUNKS = D_MODEL / K_PAR;

    always @(posedge ap_clk) begin
        if (ap_rst_n && g1_test_active) begin
            if (dut.gemv_op_start_w || dut.g5_start || dut.g6_start ||
                dut.g7g_start || dut.g7u_start || dut.g7p_start ||
                dut.g9_start || dut.g11_start || dut.glm_start ||
                dut.tree_start || dut.g8_start)
                $fatal(1, "G1-only: a downstream stage started");
            if (out_awvalid || out_wvalid || m_axi_boot_arvalid_u ||
                m_axi_g8_arvalid_u || m_axi_g8b_arvalid_u ||
                m_axi_emb_arvalid_u || m_axi_embb_arvalid_u)
                $fatal(1, "G1-only: unexpected external-memory transaction");
            if (dut.g1_start) g1_starts <= g1_starts + 1;
            if (dut.u_rmsnorm_g1.out_chunk_strobe) begin
                if (dut.u_rmsnorm_g1.out_chunk_idx_o != g1_h_chunks)
                    $fatal(1, "G1-only: hidden output chunks out of order");
                g1_h_chunks <= g1_h_chunks + 1;
            end
            if (dut.u_rmsnorm_g1_e.out_chunk_strobe) begin
                if (dut.u_rmsnorm_g1_e.out_chunk_idx_o != g1_e_chunks)
                    $fatal(1, "G1-only: embedding output chunks out of order");
                g1_e_chunks <= g1_e_chunks + 1;
            end
            if (dut.g1_done) g1_completions <= g1_completions + 1;
            if (dut.g1e_done) g1e_completions <= g1e_completions + 1;
            if (dut.kernel_busy && g1_completions == 0 && dut.fetch_pc_w != 0)
                $fatal(1, "G1-only: PC advanced before G1 completed");
            if (dut.kernel_done) begin
                if (g1_completions != 1 || g1e_completions != 1 ||
                    g1_h_chunks != G1_CHUNKS || g1_e_chunks != G1_CHUNKS)
                    $fatal(1, "G1-only: HALT before both full vectors completed");
                g1_halts <= g1_halts + 1;
            end
        end
    end

    // Covers stuck AXI handshakes during upload as well as execution hangs.
    initial begin
        if ($test$plusargs("G1_ONLY")) begin
            repeat (200_000) @(posedge ap_clk);
            $fatal(1, "G1-only: global watchdog expired");
        end
        if ($test$plusargs("REGS_ONLY")) begin
            repeat (200_000) @(posedge ap_clk);
            $fatal(1, "REGS-only: global watchdog expired");
        end
        if ($test$plusargs("GEMV_ONLY")) begin
            repeat (2_000_000) @(posedge ap_clk);
            $fatal(1, "GEMV-only: global watchdog expired");
        end
    end


    // ================================================================
    // Bounded real GEMV cutover test (real top, AXI PMEM, real banks).
    //
    // Fixture is generated here and checked against an independent integer
    // oracle (no forced DUT outputs):
    //   x[k]      = ((7*k+3) % 17) - 8                 [HFC mode]
    //   x_emb[k]  = ((5*k+2) % 13) - 6                 [EMB mode]
    //   raw[l,k]  = (13*l + 7*k + 11*(k/16)) & 3
    //   w[l,k]    = raw[l,k] - 2
    //   raw48     = sum_g (sum_{k in g} x[k]*w[l,k]) * scale_g
    //   y[l]      = sat32((raw48 + 4096) >>> 13)
    // Scale-groups use DISTINCT per-group scales so a wrong runtime group
    // index cannot be masked by identical scales.
    // ================================================================
    function automatic int signed gemv_wg(input int mode, input int lane,
                                          input int tile, input int k);
        int raw;
        case (mode)
            0: raw = (13*lane + 7*k + 11*(k/16)) & 3;
            1: raw = (13*lane + 7*k + 11*(k/16) + 17*tile) & 3;
            2: raw = (5*lane + 3*k + 7*(k/16) + 9) & 3;
            default: raw = 0;
        endcase
        return raw - 2;
    endfunction

    function automatic int signed gemv_x(input int mode, input int k);
        if (mode == 2) return ((5*k + 2) % 13) - 6;
        return ((7*k + 3) % 17) - 8;
    endfunction

    function automatic int signed gemv_sat32(input longint v);
        longint r;
        r = (v + 64'sd4096) >>> 13;
        if (r > 64'sd2147483647)  return 32'sd2147483647;
        if (r < -64'sd2147483648) return -32'sd2147483648;
        return int'(r);
    endfunction

    function automatic int signed gemv_oracle(
        input int mode, input int lane, input int tile,
        input int act_base, input int cpt, input int cpg,
        input int scale0, input int scale1
    );
        int q, i, k, g;
        longint sum0, sum1, raw48;
        sum0 = 0; sum1 = 0;
        for (q = 0; q < cpt; q = q + 1) begin
            g = (cpg == 4) ? (q >> 2) : 0;
            for (i = 0; i < 16; i = i + 1) begin
                k = (act_base + q)*16 + i;
                if (g == 0)
                    sum0 = sum0 + longint'(gemv_x(mode, k)) * longint'(gemv_wg(mode, lane, tile, k));
                else
                    sum1 = sum1 + longint'(gemv_x(mode, k)) * longint'(gemv_wg(mode, lane, tile, k));
            end
        end
        raw48 = sum0 * longint'(scale0);
        if (cpg == 4) raw48 = raw48 + sum1 * longint'(scale1);
        return gemv_sat32(raw48);
    endfunction

    function automatic int signed gemv_scale0(input int lane);
        return 8192 + (37*lane) % 1000;
    endfunction
    function automatic int signed gemv_scale1(input int lane);
        // Always DISTINCT from scale0 for the same lane and group.
        return 8192 + (37*lane) % 1000 + 257 + 13*(lane % 7);
    endfunction

    // ---- tiny PMEM program assembler (byte oriented) ----
    int gv_nb;
    logic [7:0] gv_prog [0:127];

    task automatic gv_reset();
        gv_nb = 0;
    endtask
    task automatic gv_byte(input logic [7:0] b);
        gv_prog[gv_nb] = b;
        gv_nb = gv_nb + 1;
    endtask
    task automatic gv_wrsr(input int r, input int v);
        gv_byte(8'h60 | 8'(r));
        gv_byte(8'(v[7:0]));
        gv_byte(8'(v[15:8]));
    endtask
    task automatic gv_config(
        input int act_base, input int wbase, input int tiles, input int cpt,
        input int cpg, input int obase, input int sbase, input int src,
        input int dest, input int frac
    );
        gv_wrsr(0, act_base);  gv_wrsr(1, wbase & 16'hffff);
        gv_wrsr(2, tiles);     gv_wrsr(3, cpt);
        gv_wrsr(4, cpg);       gv_wrsr(5, obase);
        gv_wrsr(6, (wbase >> 16) & 1);
        gv_wrsr(7, sbase & 16'hffff); gv_wrsr(8, (sbase >> 16) & 1);
        gv_wrsr(9, (src << 11) | (dest << 7) | (1 << 5) | (frac & 31));
        gv_wrsr(10, 0);
    endtask

    task automatic gv_upload(output logic [31:0] status);
        logic [31:0] w, rd;
        int nw, b, j, n;
        axil_write(ADDR_PMEM_ADDR, 0);
        nw = (gv_nb + 3) / 4;
        for (b = 0; b < nw*4; b = b + 4) begin
            w = 32'h4040_4040;
            for (j = 0; j < 4; j = j + 1)
                if (b + j < gv_nb) w[j*8 +: 8] = gv_prog[b + j];
            axil_write(ADDR_PMEM_DATA, w);
        end
        rd = 0;
        for (n = 0; n < 20 && rd != nw; n = n + 1)
            axil_read(ADDR_PMEM_ADDR, rd);
        if (rd != nw) $fatal(1, "GEMV: PMEM upload committed %0d/%0d words", rd, nw);
        axil_read(ADDR_PMEM_DATA, status);
    endtask

    task automatic gv_load_weights(
        input int mode, input int tile, input int wbase,
        input int act_base, input int cpt
    );
        int lane, q, i, raw, waddr;
        logic [31:0] w;
        boot_we = 1'b1;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            boot_lane_sel = lane[15:0];
            for (q = 0; q < cpt; q = q + 1) begin
                w = 32'd0;
                for (i = 0; i < 16; i = i + 1) begin
                    raw = gemv_wg(mode, lane, tile, (act_base + q)*16 + i) + 2;
                    w[i*2 +: 2] = 2'(raw);
                end
                waddr = wbase + tile*cpt + q;
                boot_wr_addr = waddr[16:0];
                boot_wr_data = w;
                @(posedge ap_clk);
            end
        end
        boot_we = 1'b0;
        @(posedge ap_clk);
    endtask

    task automatic gv_load_scale(input int addr, input int lane, input int value);
        tb_sc_we = 1'b1;
        tb_sc_lane_sel = lane[15:0];
        tb_sc_wr_addr = addr[16:0];
        tb_sc_wr_data = 16'(value);
        @(posedge ap_clk);
    endtask

    task automatic gv_load_vector(input bit use_emb, input int mode, input int nchunks);
        logic signed [15:0] lo, hi;
        for (int ch = 0; ch < nchunks; ch = ch + 1) begin
            axil_write(ADDR_BOOT_WR_ADDR, ch*8);
            for (int w = 0; w < 8; w = w + 1) begin
                lo = 16'(gemv_x(mode, ch*16 + 2*w));
                hi = 16'(gemv_x(mode, ch*16 + 2*w + 1));
                axil_write(ADDR_BOOT_WR_DATA, {hi, lo});
            end
        end
    endtask

    task automatic gv_wait_op(output logic [31:0] status);
        logic [31:0] ap;
        int n;
        axil_write(ADDR_AP_CTRL, 1);
        // The ISA PC blocks on GEMV until DONE or validated ERROR, so waiting
        // for ap_done first also removes the stale-sticky-status race from
        // polling 0x170 while the new program is still executing WRSRs.
        ap = 0;
        for (n = 0; n < 40000 && !ap[1]; n = n + 1)
            axil_read(ADDR_AP_CTRL, ap);
        if (!ap[1]) $fatal(1, "GEMV: timeout waiting for ap_done");
        axil_read(ADDR_GEMV_STATUS, status);
    endtask

    task automatic gv_check_pc(input int halt_pc);
        logic [31:0] rd;
        axil_read(ADDR_PMEM_DATA, rd);
        if (rd[31:16] !== 16'h504D || rd[14:8] !== 7'(halt_pc) || rd[7:0] !== 8'h40)
            $fatal(1, "GEMV: expected HALT at PC=%0d, got %08x", halt_pc, rd);
    endtask

    bit gv_watch = 1'b0;
    int gv_starts = 0, gv_tiles = 0, gv_errors = 0;
    always @(posedge ap_clk) begin
        if (ap_rst_n && gv_watch) begin
            if (dut.gemv_array_start_w) gv_starts <= gv_starts + 1;
            if (dut.gemv_tile_valid_w)  gv_tiles  <= gv_tiles + 1;
            if (dut.gemv_op_error_w)    gv_errors <= gv_errors + 1;
        end
    end

    task automatic gv_begin_run(output logic [31:0] status);
        gv_watch = 1'b0;
        gv_starts = 0; gv_tiles = 0; gv_errors = 0;
        gv_upload(status);
        gv_watch = 1'b1;
    endtask

    task automatic run_gemv_only();
        logic [31:0] st;
        int lane;

        $display("[%0t] ---- shared GEMV cutover test (real top, AXI PMEM) ----", $time);

        // ---------------- Test A: HFC -> QKV composite, cpg=8 ----------------
        gv_reset();
        gv_config(0, 0, 1, 8, 8, 0, 0, 1, 9, 13);
        gv_byte(8'h80); gv_byte(8'h40);
        axil_write(ADDR_HFC_SELECT, 1);
        gv_load_vector(1'b0, 0, 10);
        axil_write(ADDR_HFC_SELECT, 0);
        gv_load_weights(0, 0, 0, 0, 8);
        for (lane = 0; lane < LANES; lane = lane + 1)
            gv_load_scale(0, lane, gemv_scale0(lane));
        tb_sc_we = 1'b0;
        axil_write(0'ha8, 1);           // probe stage view 0 (QKV/Q/K/V)
        gv_begin_run(st);
        gv_wait_op(st);
        if (!st[1] || st[2] || st[0]) begin
            logic [31:0] s2;
            axil_read(ADDR_PMEM_DATA, s2);
            $display("GEMV A diagnostic: status=%08x PMEM=%08x ctrl_state=%0d tile=%0d starts=%0d tiles=%0d err=%0d",
                     st, s2, dut.u_gemv_ctrl.state, dut.u_gemv_ctrl.tile_q,
                     gv_starts, gv_tiles, gv_errors);
            $fatal(1, "GEMV A: bad status %08x (want DONE, no BUSY/ERROR)", st);
        end
        gv_check_pc(gv_nb - 1);
        if (gv_starts != 1 || gv_tiles != 1 || gv_errors != 0)
            $fatal(1, "GEMV A: starts=%0d tiles=%0d errors=%0d", gv_starts, gv_tiles, gv_errors);
        // Per-lane scales differ, so re-check each lane's actual scale:
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            if ($signed(dut.q_full_buf[0][lane*32 +: 32]) !==
                32'(gemv_oracle(0, lane, 0, 0, 8, 8, gemv_scale0(lane), 0)))
                $fatal(1, "GEMV A: lane %0d Q mismatch", lane);
        end
        // Probe check with per-lane scales:
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            logic [31:0] prd;
            axil_write(0'hac, lane[9:0]);
            axil_read(0'hb0, prd);
            if ($signed(prd) !==
                32'(gemv_oracle(0, lane, 0, 0, 8, 8, gemv_scale0(lane), 0)))
                $fatal(1, "GEMV A probe: lane %0d mismatch (%0d)", lane, $signed(prd));
        end
        $display("[%0t] PASS GEMV A: HFC -> QKV composite, cpg=8, one tile, %0d array starts",
                 $time, gv_starts);

        // ---------------- Test B: runtime cpg=4 (two distinct groups) ------
        gv_reset();
        gv_config(0, 0, 1, 8, 4, 0, 0, 1, 0, 13);
        gv_byte(8'h80); gv_byte(8'h40);
        axil_write(ADDR_HFC_SELECT, 1);
        gv_load_vector(1'b0, 0, 8);
        axil_write(ADDR_HFC_SELECT, 0);
        gv_load_weights(0, 0, 0, 0, 8);
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            gv_load_scale(0, lane, gemv_scale0(lane));
            gv_load_scale(1, lane, gemv_scale1(lane));
        end
        tb_sc_we = 1'b0;
        axil_write(0'ha8, 1);
        gv_begin_run(st);
        gv_wait_op(st);
        if (!st[1] || st[2] || st[0])
            $fatal(1, "GEMV B: bad status %08x", st);
        gv_check_pc(gv_nb - 1);
        if (gv_starts != 1 || gv_tiles != 1 || gv_errors != 0)
            $fatal(1, "GEMV B: starts=%0d tiles=%0d errors=%0d", gv_starts, gv_tiles, gv_errors);
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            if ($signed(dut.q_full_buf[0][lane*32 +: 32]) !==
                32'(gemv_oracle(0, lane, 0, 0, 8, 4, gemv_scale0(lane), gemv_scale1(lane))))
                $fatal(1, "GEMV B: lane %0d cpg4 mismatch", lane);
        end
        $display("[%0t] PASS GEMV B: runtime cpg=4, all 128 lanes match 2-group oracle",
                 $time);

        // ---------------- Test C: nonzero act/w/scale/output bases, 2 tiles -
        gv_reset();
        gv_config(1, 16, 2, 8, 8, 2, 4, 1, 0, 13);
        gv_byte(8'h80); gv_byte(8'h40);
        axil_write(ADDR_HFC_SELECT, 1);
        gv_load_vector(1'b0, 0, 10);
        axil_write(ADDR_HFC_SELECT, 0);
        gv_load_weights(1, 0, 16, 1, 8);
        gv_load_weights(1, 1, 16, 1, 8);
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            gv_load_scale(4, lane, 8000 + lane % 100);       // tile 0
            gv_load_scale(5, lane, 9000 + lane % 100);       // tile 1
        end
        tb_sc_we = 1'b0;
        axil_write(0'ha8, 1);
        gv_begin_run(st);
        gv_wait_op(st);
        if (!st[1] || st[2] || st[0])
            $fatal(1, "GEMV C: bad status %08x", st);
        gv_check_pc(gv_nb - 1);
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            if ($signed(dut.q_full_buf[2][lane*32 +: 32]) !==
                32'(gemv_oracle(1, lane, 0, 1, 8, 8, 8000 + lane % 100, 0)))
                $fatal(1, "GEMV C: lane %0d tile0 mismatch", lane);
            if ($signed(dut.q_full_buf[3][lane*32 +: 32]) !==
                32'(gemv_oracle(1, lane, 1, 1, 8, 8, 9000 + lane % 100, 0)))
                $fatal(1, "GEMV C: lane %0d tile1 mismatch", lane);
        end
        // The probe replays a 128-word tile over 128 cycles, so consecutive
        // short tiles cannot both be snapshotted (documented probe depth
        // limit). Destination buffers above cover BOTH tiles; the probe
        // covers the first captured tile against the same oracle.
        for (int i = 0; i < LANES; i = i + 1) begin
            logic [31:0] prd;
            axil_write(0'hac, i[9:0]);
            axil_read(0'hb0, prd);
            if ($signed(prd) !== 32'(gemv_oracle(1, i, 0, 1, 8, 8, 8000 + i % 100, 0)))
                $fatal(1, "GEMV C probe: tile0 lane%0d mismatch", i);
        end
        axil_read(0'hbc, st);
        if (st !== 32'd128)
            $fatal(1, "GEMV C probe: expected the first tile's 128 words, got %0d", st);
        axil_read(ADDR_GEMV_TILE, st);
        if (st[15:0] !== 16'd1)
            $fatal(1, "GEMV C: active tile expected 1, got %0d", st[15:0]);
        if (gv_starts != 2 || gv_tiles != 2)
            $fatal(1, "GEMV C: expected 2 starts/tiles, got %0d/%0d", gv_starts, gv_tiles);
        $display("[%0t] PASS GEMV C: nonzero bases act=%0d w=%0d s=%0d out=%0d, 2 tiles, groups=8",
                 $time, 1, 16, 4, 2);

        // ---------------- Test D: EMB source -> K destination --------------
        gv_reset();
        gv_config(0, 64, 1, 8, 8, 3, 8, 2, 1, 13);
        gv_byte(8'h80); gv_byte(8'h40);
        axil_write(ADDR_EMB_SELECT, 1);
        gv_load_vector(1'b1, 2, 8);          // x_emb formula in EMB buffer
        axil_write(ADDR_EMB_SELECT, 0);
        gv_load_weights(2, 0, 64, 0, 8);
        for (lane = 0; lane < LANES; lane = lane + 1)
            gv_load_scale(8, lane, 6000 + (53*lane) % 700);
        tb_sc_we = 1'b0;
        axil_write(0'ha8, 1);
        gv_begin_run(st);
        gv_wait_op(st);
        if (!st[1] || st[2] || st[0])
            $fatal(1, "GEMV D: bad status %08x", st);
        gv_check_pc(gv_nb - 1);
        if (gv_starts != 1 || gv_tiles != 1 || gv_errors != 0)
            $fatal(1, "GEMV D: starts=%0d tiles=%0d errors=%0d", gv_starts, gv_tiles, gv_errors);
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            if ($signed(dut.k_full_buf[3][lane*32 +: 32]) !==
                32'(gemv_oracle(2, lane, 0, 0, 8, 8, 6000 + (53*lane) % 700, 0)))
                $fatal(1, "GEMV D: lane %0d K mismatch", lane);
        end
        $display("[%0t] PASS GEMV D: EMB source -> K destination, output_base=3", $time);

        // ---------------- Test E: invalid configs terminate without launch -
        // E1: unsupported chunks_per_scale_group=2.
        gv_reset();
        gv_config(0, 0, 1, 8, 2, 0, 0, 1, 0, 13);
        gv_byte(8'h80); gv_byte(8'h40);
        gv_begin_run(st);
        gv_wait_op(st);
        if (gv_starts != 0 || gv_tiles != 0 || gv_errors != 1)
            $fatal(1, "GEMV E1: illegal config launched (starts=%0d tiles=%0d err=%0d)",
                   gv_starts, gv_tiles, gv_errors);
        if (!st[2] || st[1] || st[0])
            $fatal(1, "GEMV E1: bad status %08x (want ERROR only)", st);
        gv_check_pc(gv_nb - 1);
        axil_read(ADDR_GEMV_TILE, st);
        if (st[15:0] !== 16'd0) $fatal(1, "GEMV E1: active tile moved");

        // E2: unsupported scale_frac=12.
        gv_reset();
        gv_config(0, 0, 1, 8, 8, 0, 0, 1, 0, 12);
        gv_byte(8'h80); gv_byte(8'h40);
        gv_begin_run(st);
        gv_wait_op(st);
        if (gv_starts != 0 || gv_tiles != 0 || gv_errors != 1)
            $fatal(1, "GEMV E2: illegal frac launched (starts=%0d tiles=%0d err=%0d)",
                   gv_starts, gv_tiles, gv_errors);
        if (!st[2] || st[1] || st[0])
            $fatal(1, "GEMV E2: bad status %08x (want ERROR only)", st);

        // E4: fixed qzero is 1; qzero=0 must be rejected.
        gv_reset();
        gv_config(0, 0, 1, 8, 8, 0, 0, 1, 0, 13);
        gv_wrsr(9, (1 << 11) | (0 << 7) | (0 << 5) | 13);
        gv_byte(8'h80); gv_byte(8'h40);
        gv_begin_run(st);
        gv_wait_op(st);
        if (gv_starts != 0 || gv_tiles != 0 || gv_errors != 1)
            $fatal(1, "GEMV E4: qzero=0 launched (starts=%0d tiles=%0d err=%0d)",
                   gv_starts, gv_tiles, gv_errors);
        if (!st[2] || st[1] || st[0])
            $fatal(1, "GEMV E4: bad status %08x (want ERROR only)", st);

        // E5: output_conversion is fixed at 0; 1 must be rejected.
        gv_reset();
        gv_config(0, 0, 1, 8, 8, 0, 0, 1, 0, 13);
        gv_wrsr(10, 1);
        gv_byte(8'h80); gv_byte(8'h40);
        gv_begin_run(st);
        gv_wait_op(st);
        if (gv_starts != 0 || gv_tiles != 0 || gv_errors != 1)
            $fatal(1, "GEMV E5: conversion=1 launched (starts=%0d tiles=%0d err=%0d)",
                   gv_starts, gv_tiles, gv_errors);
        if (!st[2] || st[1] || st[0])
            $fatal(1, "GEMV E5: bad status %08x (want ERROR only)", st);

        // E3: restart after errors clears sticky ERROR and runs again.
        gv_reset();
        gv_config(0, 0, 1, 8, 8, 0, 0, 1, 9, 13);
        gv_byte(8'h80); gv_byte(8'h40);
        gv_begin_run(st);
        gv_wait_op(st);
        if (!st[1] || st[2] || st[0])
            $fatal(1, "GEMV E3 restart: bad status %08x (ERROR must clear at START)", st);
        if (gv_starts != 1 || gv_errors != 0)
            $fatal(1, "GEMV E3: restart starts=%0d errors=%0d", gv_starts, gv_errors);
        gv_check_pc(gv_nb - 1);
        $display("[%0t] PASS GEMV E: invalid cpg/frac/qzero/conversion terminate without launch; restart clears ERROR",
                 $time);

        $display("[%0t] PASS: shared GEMV cutover (A-D) and error/restart (E); real banks, AXI PMEM, all-lane oracle.",
                 $time);
    endtask

    task automatic run_g1_only();
        logic [31:0] rd;
        logic signed [15:0] lo, hi;
        $display("[%0t] ---- ISA micro-test: STATE G1; HALT ----", $time);

        // Two 3072-element vectors, not three raw target-layer features:
        // h_fc_buf is the already-projected hidden input consumed by G1.
        // Signed, deterministic data; gamma=1 in Q4.12 for both norms.
        axil_write(ADDR_HFC_SELECT, 0);
        axil_write(ADDR_EMB_SELECT, 0);
        axil_write(ADDR_GAMMA_SEL, GAMMA_HIDDEN_NORM);
        axil_write(ADDR_BOOT_WR_ADDR, 0);
        repeat (GAMMA_WORDS) axil_write(ADDR_BOOT_WR_DATA, 32'h1000_1000);
        axil_write(ADDR_GAMMA_SEL, GAMMA_E_NORM);
        axil_write(ADDR_BOOT_WR_ADDR, 0);
        repeat (GAMMA_WORDS) axil_write(ADDR_BOOT_WR_DATA, 32'h1000_1000);
        axil_write(ADDR_GAMMA_SEL, 0);
        axil_write(ADDR_EMB_SELECT, 1);
        axil_write(ADDR_BOOT_WR_ADDR, 0);
        for (int i = 0; i < WORDS_PER_VECTOR; i++) begin
            lo = 16'(512 + i);
            hi = -lo;
            axil_write(ADDR_BOOT_WR_DATA, {hi, lo});
        end
        axil_write(ADDR_EMB_SELECT, 0);
        axil_write(ADDR_HFC_SELECT, 1);
        axil_write(ADDR_BOOT_WR_ADDR, 0);
        for (int i = 0; i < WORDS_PER_VECTOR; i++) begin
            lo = -16'(512 + i);
            hi = -lo;
            axil_write(ADDR_BOOT_WR_DATA, {hi, lo});
        end
        axil_write(ADDR_HFC_SELECT, 0);
        read_residency(RES_SEL_EMB, rd);
        if (rd !== WORDS_PER_VECTOR) $fatal(1, "G1-only: EMB upload incomplete");
        read_residency(RES_SEL_HFC, rd);
        if (rd !== WORDS_PER_VECTOR) $fatal(1, "G1-only: HFC upload incomplete");
        read_residency(RES_SEL_GAMMA, rd);
        if (rd !== 2*GAMMA_WORDS) $fatal(1, "G1-only: gamma upload incomplete");

        // Little-endian bytes: 01=STATE G1, 40=HALT; HALT padding.
        axil_write(ADDR_PMEM_ADDR, 0);
        axil_write(ADDR_PMEM_DATA, 32'h4040_4001);
        // AXI B precedes the registered PMEM commit. Wait for its pointer
        // increment before reading the combinational instruction fetch.
        rd = 0;
        for (int n = 0; n < 10 && rd != 1; n++) axil_read(ADDR_PMEM_ADDR, rd);
        if (rd !== 32'd1) $fatal(1, "G1-only: PMEM upload did not commit");
        axil_read(ADDR_PMEM_DATA, rd);
        if (rd !== 32'h504D_0001) $fatal(1, "G1-only: initial PMEM status %08x", rd);
        g1_test_active = 1'b1;
        axil_write(ADDR_AP_CTRL, 1);
        rd = 0;
        for (int n = 0; n < 2000 && !rd[1]; n++) axil_read(ADDR_AP_CTRL, rd);
        if (!rd[1]) $fatal(1, "G1-only: timeout waiting for ap_done");

        // Leave the monitor armed after HALT to catch fall-through/relaunch.
        repeat (20) @(posedge ap_clk);
        axil_read(ADDR_AP_CTRL, rd);
        if (rd[1] || !rd[2] || !rd[3])
            $fatal(1, "G1-only: expected idle/ready with ap_done cleared, got %08x", rd);
        axil_read(ADDR_PMEM_DATA, rd);
        if (rd !== 32'h504D_0140) $fatal(1, "G1-only: expected HALT at PC=1, got %08x", rd);
        @(negedge ap_clk);
        if (g1_starts != 1 || g1_completions != 1 || g1e_completions != 1 ||
            g1_halts != 1 || g1_h_chunks != G1_CHUNKS || g1_e_chunks != G1_CHUNKS)
            $fatal(1, "G1-only: starts=%0d done_h/e=%0d/%0d halts=%0d chunks_h/e=%0d/%0d",
                   g1_starts, g1_completions, g1e_completions, g1_halts, g1_h_chunks, g1_e_chunks);
        $display("[%0t] PASS: STATE G1 only; hidden/embedding chunks=%0d/%0d; one HALT at PC=1; no downstream starts or DMA.",
                 $time, g1_h_chunks, g1_e_chunks);
    endtask

    // ================================================================
    // Main sequence
    // ================================================================
    localparam int TIMEOUT_CYCLES = 30_000_000;

    initial begin : main_sequence
        logic [31:0] rd;
        int errors;
        errors = 0;

        axil_awvalid = 0; axil_wvalid = 0; axil_bready = 0; axil_arvalid = 0; axil_rready = 0;
        boot_we = 0; boot_lane_sel = 0; boot_wr_addr = 0; boot_wr_data = 0;
        tb_sc_we = 0; tb_sc_lane_sel = 0; tb_sc_wr_addr = 0; tb_sc_wr_data = 0;

        @(posedge ap_rst_n);
        repeat (5) @(posedge ap_clk);

        if ($test$plusargs("G1_ONLY")) begin
            run_g1_only();
            $finish;
            disable main_sequence;
        end

        if ($test$plusargs("REGS_ONLY")) begin
            run_regs_only();
            $finish;
            disable main_sequence;
        end

        if ($test$plusargs("GEMV_ONLY")) begin
            run_gemv_only();
            $finish;
            disable main_sequence;
        end

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

        // Assemble the linear draft pass from WRSR + GEMV 0x80 + the
        // remaining non-shared STATE instructions. The GEMV configurations
        // are DELTA writes: only fields whose value differs from the prior
        // operation are re-written. This is the real full-pipeline program:
        //   STATE G1; GEMV(QKV src=R1 dst=QKV 72 tiles);
        //   STATE G5; GEMV(R2 src=attention dst=R2);
        //   GEMV(R4g src=preMLP dst=R4G); GEMV(R4u ... dst=R4U);
        //   STATE G7_SILU; GEMV(R5 src=R5 dst=R6/residual);
        //   STATE G11; STATE GLM; WRSR TREE config; STATE TREE; HALT.
        gv_reset();
        gv_byte(8'h01);                                   // STATE G1
        gv_config(0, 0, 72, 384, 8, 0, 0, 3, 9, 13);      // QKV: R1 -> QKV
        gv_byte(8'h80);                                   // GEMV
        gv_byte(8'h03);                                   // STATE G5
        gv_wrsr(1, 27648); gv_wrsr(2, 24); gv_wrsr(3, 192);
        gv_wrsr(7, 3456);  gv_wrsr(9, 16'h39AD);          // src7 dest3
        gv_byte(8'h80);
        gv_wrsr(1, 32256); gv_wrsr(2, 64);
        gv_wrsr(7, 4032);  gv_wrsr(9, 16'h222D);          // src4 dest4
        gv_byte(8'h80);
        gv_wrsr(1, 44544); gv_wrsr(7, 5568);
        gv_wrsr(9, 16'h22AD);                             // src4 dest5
        gv_byte(8'h80);
        gv_byte(8'h07);                                   // STATE G7_SILU
        gv_wrsr(1, 56832); gv_wrsr(2, 24); gv_wrsr(3, 512);
        gv_wrsr(7, 7104);  gv_wrsr(9, 16'h2B2D);          // src5 dest6
        gv_byte(8'h80);
        gv_byte(8'h09);                                   // STATE G11
        gv_byte(8'h0D);                                   // STATE GLM
        gv_wrsr(1, 16'h3E00); gv_wrsr(6, 1);              // G12 weight base 81408
        gv_wrsr(2, 1002); gv_wrsr(3, 16); gv_wrsr(7, 10176);
        gv_wrsr(9, 16'h33AD);                             // src6 dest7 (TREE)
        gv_byte(8'h0B);                                   // STATE TREE
        gv_byte(8'h40);                                   // HALT
        if (gv_nb > 128)
            $fatal(1, "full pipeline program %0d bytes exceeds the 128-byte PMEM", gv_nb);
        $display("  assembled full pipeline delta-WRSR program: %0d bytes (limit 128)", gv_nb);
        gv_upload(rd);

        // Verify the write pointer committed exactly the assembled words.
        axil_read(ADDR_PMEM_ADDR, rd);
        $display("  PMEM write_addr after upload: got 0x%08x (expect %0d words)",
                 rd, (gv_nb+3)/4);
        if (rd[7:0] !== 8'((gv_nb+3)/4)) errors++;

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
        $display("[%0t] PMEM status post-run: got 0x%08x (canary=0x%04x, PC=%0d, instr=0x%02x; expect PC=%0d, instr=0x40 HALT)",
                 $time, rd, rd[31:16], rd[14:8], rd[7:0], gv_nb-1);
        if (rd[31:16] !== 16'h504D || rd[14:8] !== 7'(gv_nb-1) || rd[7:0] !== 8'h40) begin
            $display("  FAIL: program did not halt at byte %0d!", gv_nb-1);
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
