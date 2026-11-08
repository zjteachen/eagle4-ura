// Simulator time units are left to Verilator --timing; the DUT has no
// timescale directive, so the testbench deliberately does not add one.
// Standalone self-checking testbench for isa/axi_lite.sv.
// Drives the AXI-Lite channels independently at negedge, samples after
// posedge, and covers full-size and smaller PMEM_SIZE configurations.
module tb_axi_lite;

  localparam int FULL_WORDS  = 128;
  localparam int SMALL_WORDS = 16;
  localparam int MAX_WORDS   = 128;
  localparam int MAX_WAIT    = 100;
  localparam logic [1:0] OKAY   = 2'b00;
  localparam logic [1:0] DECERR = 2'b11;

  logic clk;

  // Test clock: 10 time-unit period; all stimulus changes at negedge.
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  logic [1:0]  rst_n;
  logic [8:0]  awaddr  [2];
  logic [1:0]  awvalid;
  logic [1:0]  awready;
  logic [31:0] wdata   [2];
  logic [3:0]  wstrb   [2];
  logic [1:0]  wvalid;
  logic [1:0]  wready;
  logic [1:0]  bresp   [2];
  logic [1:0]  bvalid;
  logic [1:0]  bready;
  logic [8:0]  araddr  [2];
  logic [1:0]  arvalid;
  logic [1:0]  arready;
  logic [31:0] rdata   [2];
  logic [1:0]  rresp   [2];
  logic [1:0]  rvalid;
  logic [1:0]  rready;

  int errors = 0;
  int checks = 0;
  int b_acked [2];
  int r_acked [2];
  int b_count [2];
  int r_count [2];

  logic [31:0] shadow [2][MAX_WORDS];

  // DUT with the default PMEM_SIZE=128.
  axi_lite dut_full (
    .clk(clk),
    .rst_n(rst_n[0]),
    .s_axi_control_awaddr(awaddr[0]),
    .s_axi_control_awvalid(awvalid[0]),
    .s_axi_control_awready(awready[0]),
    .s_axi_control_wdata(wdata[0]),
    .s_axi_control_wstrb(wstrb[0]),
    .s_axi_control_wvalid(wvalid[0]),
    .s_axi_control_wready(wready[0]),
    .s_axi_control_bresp(bresp[0]),
    .s_axi_control_bvalid(bvalid[0]),
    .s_axi_control_bready(bready[0]),
    .s_axi_control_araddr(araddr[0]),
    .s_axi_control_arvalid(arvalid[0]),
    .s_axi_control_arready(arready[0]),
    .s_axi_control_rdata(rdata[0]),
    .s_axi_control_rresp(rresp[0]),
    .s_axi_control_rvalid(rvalid[0]),
    .s_axi_control_rready(rready[0])
  );

  // DUT with a smaller parameter to exercise the window boundary.
  axi_lite #(.PMEM_SIZE(SMALL_WORDS)) dut_small (
    .clk(clk),
    .rst_n(rst_n[1]),
    .s_axi_control_awaddr(awaddr[1]),
    .s_axi_control_awvalid(awvalid[1]),
    .s_axi_control_awready(awready[1]),
    .s_axi_control_wdata(wdata[1]),
    .s_axi_control_wstrb(wstrb[1]),
    .s_axi_control_wvalid(wvalid[1]),
    .s_axi_control_wready(wready[1]),
    .s_axi_control_bresp(bresp[1]),
    .s_axi_control_bvalid(bvalid[1]),
    .s_axi_control_bready(bready[1]),
    .s_axi_control_araddr(araddr[1]),
    .s_axi_control_arvalid(arvalid[1]),
    .s_axi_control_arready(arready[1]),
    .s_axi_control_rdata(rdata[1]),
    .s_axi_control_rresp(rresp[1]),
    .s_axi_control_rvalid(rvalid[1]),
    .s_axi_control_rready(rready[1])
  );

  // Response-count observation: count each accepted B/R handshake, reset with
  // the DUT so stale counts from an aborted transaction cannot hide a duplicate.
  always @(posedge clk) begin
    for (int d = 0; d < 2; d++) begin
      if (!rst_n[d]) begin
        b_count[d] <= 0;
        r_count[d] <= 0;
      end else begin
        if (bvalid[d] && bready[d]) b_count[d] <= b_count[d] + 1;
        if (rvalid[d] && rready[d]) r_count[d] <= r_count[d] + 1;
      end
    end
  end

  function automatic int words(input logic d);
    return (d == 0) ? FULL_WORDS : SMALL_WORDS;
  endfunction

  function automatic int bytes(input logic d);
    return words(d) * 4;
  endfunction

  function automatic bit address_hit(input logic d, input logic [8:0] addr);
    return (addr[1:0] == 2'b00) && (int'(addr) < bytes(d));
  endfunction

  task automatic check(input bit cond, input string msg);
    checks++;
    if (!cond) begin
      errors++;
      $display("FAIL @%0t: %s", $time, msg);
    end
  endtask

  task automatic check_eq32(input logic [31:0] got, input logic [31:0] exp,
                            input string msg);
    checks++;
    if (got !== exp) begin
      errors++;
      $display("FAIL @%0t: %s got=0x%08x exp=0x%08x", $time, msg, got, exp);
    end
  endtask

  task automatic check_counts(input logic d);
    check(b_count[d] == b_acked[d],
          $sformatf("B response count mismatch for DUT %0d", d));
    check(r_count[d] == r_acked[d],
          $sformatf("R response count mismatch for DUT %0d", d));
  endtask

  task automatic drive_idle(input logic d);
    awvalid[d] = 0;
    wvalid[d]  = 0;
    arvalid[d] = 0;
    bready[d]  = 0;
    rready[d]  = 0;
  endtask

  task automatic reset_dut(input logic d);
    @(negedge clk); #1;
    rst_n[d] = 0;
    drive_idle(d);
    b_acked[d] = 0;
    r_acked[d] = 0;
    repeat (2) @(negedge clk); #1;
    rst_n[d] = 1;
    @(negedge clk); #1;
    check(awready[d] && wready[d] && arready[d],
          "READY high after reset release");
    check(!bvalid[d] && !rvalid[d], "responses clear after reset");
    for (int i = 0; i < MAX_WORDS; i++) shadow[d][i] = 32'b0;
  endtask

  task automatic model_write(input logic d, input logic [8:0] addr,
                             input logic [31:0] data, input logic [3:0] strb);
    for (int b = 0; b < 4; b++) begin
      if (strb[b]) shadow[d][int'(addr) / 4][b*8 +: 8] = data[b*8 +: 8];
    end
  endtask

  task automatic aw_handshake(input logic d, input logic [8:0] addr);
    int guard;
    guard = 0;
    @(negedge clk); #1;
    awaddr[d]  = addr;
    awvalid[d] = 1;
    while (!awready[d]) begin
      @(negedge clk); #1;
      guard++;
      if (guard > MAX_WAIT) $fatal(1, "timeout waiting for AWREADY");
    end
    @(posedge clk); #1;
    @(negedge clk); #1;
    awvalid[d] = 0;
  endtask

  task automatic w_handshake(input logic d, input logic [31:0] data,
                             input logic [3:0] strb);
    int guard;
    guard = 0;
    @(negedge clk); #1;
    wdata[d]  = data;
    wstrb[d]  = strb;
    wvalid[d] = 1;
    while (!wready[d]) begin
      @(negedge clk); #1;
      guard++;
      if (guard > MAX_WAIT) $fatal(1, "timeout waiting for WREADY");
    end
    @(posedge clk); #1;
    @(negedge clk); #1;
    wvalid[d] = 0;
  endtask

  task automatic aw_w_handshake(input logic d, input logic [8:0] addr,
                                input logic [31:0] data, input logic [3:0] strb);
    int guard;
    guard = 0;
    @(negedge clk); #1;
    awaddr[d]  = addr;
    awvalid[d] = 1;
    wdata[d]   = data;
    wstrb[d]   = strb;
    wvalid[d]  = 1;
    while (!(awready[d] && wready[d])) begin
      @(negedge clk); #1;
      guard++;
      if (guard > MAX_WAIT) $fatal(1, "timeout waiting for AW/WREADY");
    end
    @(posedge clk); #1;
    @(negedge clk); #1;
    awvalid[d] = 0;
    wvalid[d]  = 0;
  endtask

  task automatic ar_start(input logic d, input logic [8:0] addr);
    araddr[d]  = addr;
    arvalid[d] = 1;
  endtask

  task automatic ar_finish(input logic d);
    int guard;
    guard = 0;
    while (!arready[d]) begin
      @(negedge clk); #1;
      guard++;
      if (guard > MAX_WAIT) $fatal(1, "timeout waiting for ARREADY");
    end
    @(posedge clk); #1;
    @(negedge clk); #1;
    arvalid[d] = 0;
  endtask

  task automatic ar_handshake(input logic d, input logic [8:0] addr);
    @(negedge clk); #1;
    ar_start(d, addr);
    ar_finish(d);
  endtask

  task automatic wait_bvalid(input logic d, input logic [1:0] exp_resp,
                             input int hold_cycles);
    int guard;
    guard = 0;
    while (!bvalid[d]) begin
      @(negedge clk); #1;
      guard++;
      if (guard > MAX_WAIT) $fatal(1, "timeout waiting for BVALID");
    end
    check(bresp[d] === exp_resp, "BRESP value");
    check(!awready[d] && !wready[d], "AW/W blocked while B pending");
    repeat (hold_cycles) begin
      @(negedge clk); #1;
      check(bvalid[d] === 1'b1, "BVALID held under backpressure");
      check(bresp[d] === exp_resp, "BRESP held under backpressure");
    end
  endtask

  task automatic ack_b(input logic d);
    bready[d] = 1;
    @(posedge clk); #1;
    bready[d] = 0;
    b_acked[d]++;
    @(negedge clk); #1;
    check(!bvalid[d], "BVALID cleared after BREADY");
  endtask

  task automatic expect_b(input logic d, input logic [1:0] exp_resp,
                          input int hold_cycles);
    wait_bvalid(d, exp_resp, hold_cycles);
    ack_b(d);
  endtask

  task automatic wait_rvalid(input logic d, input logic [31:0] exp_data,
                             input logic [1:0] exp_resp, input int hold_cycles);
    int guard;
    guard = 0;
    while (!rvalid[d]) begin
      @(negedge clk); #1;
      guard++;
      if (guard > MAX_WAIT) $fatal(1, "timeout waiting for RVALID");
    end
    check_eq32(rdata[d], exp_data, "RDATA value");
    check(rresp[d] === exp_resp, "RRESP value");
    check(!arready[d], "AR blocked while R outstanding");
    repeat (hold_cycles) begin
      @(negedge clk); #1;
      check(rvalid[d] === 1'b1, "RVALID held under backpressure");
      check_eq32(rdata[d], exp_data, "RDATA held under backpressure");
      check(rresp[d] === exp_resp, "RRESP held under backpressure");
    end
  endtask

  task automatic ack_r(input logic d);
    rready[d] = 1;
    @(posedge clk); #1;
    rready[d] = 0;
    r_acked[d]++;
    @(negedge clk); #1;
    check(!rvalid[d], "RVALID cleared after RREADY");
  endtask

  task automatic expect_r(input logic d, input logic [31:0] exp_data,
                          input logic [1:0] exp_resp, input int hold_cycles);
    wait_rvalid(d, exp_data, exp_resp, hold_cycles);
    ack_r(d);
  endtask

  task automatic write_word(input logic d, input logic [8:0] addr,
                            input logic [31:0] data, input logic [3:0] strb,
                            input int order, input logic [1:0] exp_resp,
                            input int hold_cycles);
    if (order == 0) begin
      aw_handshake(d, addr);
      w_handshake(d, data, strb);
    end else if (order == 1) begin
      w_handshake(d, data, strb);
      aw_handshake(d, addr);
    end else begin
      aw_w_handshake(d, addr, data, strb);
    end
    expect_b(d, exp_resp, hold_cycles);
    if (exp_resp == OKAY) begin
      check(address_hit(d, addr), "OKAY write must be aligned and in window");
      model_write(d, addr, data, strb);
    end else begin
      check(!address_hit(d, addr), "DECERR write must be misaligned/out-of-range");
    end
  endtask

  task automatic read_word(input logic d, input logic [8:0] addr,
                           input logic [1:0] exp_resp, input int hold_cycles);
    logic [31:0] exp_data;
    if (exp_resp == OKAY) begin
      check(address_hit(d, addr), "OKAY read must be aligned and in window");
      exp_data = shadow[d][addr[8:2]];
    end else begin
      check(!address_hit(d, addr), "DECERR read must be misaligned/out-of-range");
      exp_data = 32'b0;
    end
    ar_handshake(d, addr);
    expect_r(d, exp_data, exp_resp, hold_cycles);
  endtask

  task automatic read_word_expect(input logic d, input logic [8:0] addr,
                                  input logic [31:0] exp_data,
                                  input logic [1:0] exp_resp,
                                  input int hold_cycles);
    ar_handshake(d, addr);
    expect_r(d, exp_data, exp_resp, hold_cycles);
  endtask

  initial begin : main
    rst_n = 2'b00;
    for (int d = 0; d < 2; d++) begin
      awaddr[d]  = 9'b0;
      wdata[d]   = 32'b0;
      wstrb[d]   = 4'b0;
      araddr[d]  = 9'b0;
      awvalid[d] = 0;
      wvalid[d]  = 0;
      arvalid[d] = 0;
      bready[d]  = 0;
      rready[d]  = 0;
      b_acked[d] = 0;
      r_acked[d] = 0;
    end

    reset_dut(0);
    reset_dut(1);

    // Full-size reset: every one of the 128 words reads as zero, including
    // the last word at 0x1FC.
    for (int i = 0; i < FULL_WORDS; i++) begin
      read_word(0, 9'(i*4), OKAY, 0);
    end

    // Smaller-parameter reset: only its 16-word window exists.
    for (int i = 0; i < SMALL_WORDS; i++) begin
      read_word(1, 9'(i*4), OKAY, 0);
    end

    // Full-size basic ordering and the last word.
    write_word(0, 9'h000, 32'h12345678, 4'hF, 0, OKAY, 0);
    read_word(0, 9'h000, OKAY, 0);
    write_word(0, 9'h004, 32'hA5A5A5A5, 4'hF, 1, OKAY, 0);
    read_word(0, 9'h004, OKAY, 0);
    write_word(0, 9'h008, 32'h0BADF00D, 4'hF, 2, OKAY, 0);
    read_word(0, 9'h008, OKAY, 0);
    write_word(0, 9'h1FC, 32'hCAFEBABE, 4'hF, 0, OKAY, 0);
    read_word(0, 9'h1FC, OKAY, 0);

    // Individual byte strobes, then an all-zero strobe that must be OKAY
    // and leave the word unchanged.
    write_word(0, 9'h00C, 32'hAABBCCDD, 4'b0001, 0, OKAY, 0);
    write_word(0, 9'h00C, 32'h11223344, 4'b0010, 0, OKAY, 0);
    write_word(0, 9'h00C, 32'h55667788, 4'b0100, 0, OKAY, 0);
    write_word(0, 9'h00C, 32'h99AABBCC, 4'b1000, 0, OKAY, 0);
    read_word(0, 9'h00C, OKAY, 0);
    write_word(0, 9'h00C, 32'hFFFFFFFF, 4'b0000, 0, OKAY, 0);
    read_word(0, 9'h00C, OKAY, 0);

    // B response backpressure: hold BREADY low, present a new AW/W pair, and
    // confirm neither channel is accepted and no extra response appears.
    aw_handshake(0, 9'h010);
    w_handshake(0, 32'h55AA55AA, 4'hF);
    wait_bvalid(0, OKAY, 0);
    awaddr[0]  = 9'h014;
    awvalid[0] = 1;
    wdata[0]   = 32'hBAAAAAAD;
    wstrb[0]   = 4'hF;
    wvalid[0]  = 1;
    #1;
    check(!awready[0] && !wready[0],
          "AW/W refused while B response is pending");
    repeat (3) begin
      @(negedge clk); #1;
      check(bvalid[0] && bresp[0] == OKAY, "B response held under backpressure");
      check(!awready[0] && !wready[0], "AW/W remain blocked while B pending");
    end
    awvalid[0] = 0;
    wvalid[0]  = 0;
    ack_b(0);
    model_write(0, 9'h010, 32'h55AA55AA, 4'hF);
    @(negedge clk); #1;
    check(!bvalid[0], "no extra B after backpressure write");
    read_word(0, 9'h010, OKAY, 0);
    read_word(0, 9'h014, OKAY, 0);

    // Changing AWADDR after AW acceptance and W data/strobe after W
    // acceptance must not affect the committed write.
    aw_handshake(0, 9'h014);
    awaddr[0] = 9'h1FC;
    @(negedge clk); #1;
    wdata[0]  = 32'hDEADBEEF;
    wstrb[0]  = 4'hF;
    wvalid[0] = 1;
    while (!wready[0]) begin
      @(negedge clk); #1;
    end
    @(posedge clk); #1;
    wdata[0] = 32'h00000000;
    wstrb[0] = 4'b0000;
    @(negedge clk); #1;
    wvalid[0] = 0;
    expect_b(0, OKAY, 0);
    model_write(0, 9'h014, 32'hDEADBEEF, 4'hF);
    read_word(0, 9'h014, OKAY, 0);
    read_word(0, 9'h1FC, OKAY, 0);

    // Read is independent of the write path: perform it while a B response
    // is still outstanding.
    write_word(0, 9'h018, 32'h0F0F0F0F, 4'hF, 0, OKAY, 0);
    aw_handshake(0, 9'h01C);
    w_handshake(0, 32'h11112222, 4'hF);
    wait_bvalid(0, OKAY, 0);
    read_word(0, 9'h018, OKAY, 0);
    check(bvalid[0] && bresp[0] == OKAY,
          "B held while independent read completes");
    ack_b(0);
    model_write(0, 9'h01C, 32'h11112222, 4'hF);
    read_word(0, 9'h01C, OKAY, 0);

    // AR accepted on the same edge as a write commit must return the old
    // word; the write then becomes visible to a later read.
    write_word(0, 9'h020, 32'h11112222, 4'hF, 0, OKAY, 0);
    aw_handshake(0, 9'h020);
    w_handshake(0, 32'h33334444, 4'hF);
    ar_start(0, 9'h020);
    ar_finish(0);
    expect_r(0, 32'h11112222, OKAY, 0);
    expect_b(0, OKAY, 0);
    model_write(0, 9'h020, 32'h33334444, 4'hF);
    read_word(0, 9'h020, OKAY, 0);

    // Read-response backpressure: accept an AR, hold RREADY low, then commit
    // a new value to the same word. RDATA/RRESP must keep the old snapshot.
    write_word(0, 9'h038, 32'h1234ABCD, 4'hF, 0, OKAY, 0);
    ar_handshake(0, 9'h038);
    wait_rvalid(0, 32'h1234ABCD, OKAY, 0);
    write_word(0, 9'h038, 32'h5678EF01, 4'hF, 0, OKAY, 0);
    check(rvalid[0] && rdata[0] == 32'h1234ABCD && rresp[0] == OKAY,
          "RDATA/RRESP held while backend changes");
    ack_r(0);
    read_word(0, 9'h038, OKAY, 0);

    // Full-size invalid accesses: DECERR, read data zero, no storage change.
    write_word(0, 9'h002, 32'hFFFFFFFF, 4'hF, 0, DECERR, 0);
    write_word(0, 9'h001, 32'h00000000, 4'b0000, 0, DECERR, 0);
    write_word(0, 9'h1FE, 32'h12345678, 4'hF, 0, DECERR, 0);
    read_word(0, 9'h002, DECERR, 0);
    read_word(0, 9'h1FE, DECERR, 0);
    read_word(0, 9'h000, OKAY, 0);
    read_word(0, 9'h1FC, OKAY, 0);
    // With PMEM_SIZE=128, every aligned 9-bit address is inside the window;
    // there is no aligned out-of-range address at full size.

    // No duplicate commit if WVALID stays high through the response. The
    // design may accept a later W-only transfer, but it cannot commit again
    // without a matching AW.
    reset_dut(0);
    aw_handshake(0, 9'h034);
    @(negedge clk); #1;
    wdata[0]  = 32'hD00DF00D;
    wstrb[0]  = 4'hF;
    wvalid[0] = 1;
    while (!wready[0]) begin
      @(negedge clk); #1;
    end
    @(posedge clk); #1;
    expect_b(0, OKAY, 0);
    repeat (5) begin
      @(negedge clk); #1;
      check(!bvalid[0], "no duplicate B with WVALID held high");
    end
    wvalid[0] = 0;
    check_counts(0);
    model_write(0, 9'h034, 32'hD00DF00D, 4'hF);
    read_word(0, 9'h034, OKAY, 0);

    // Reset with only AW queued: reset clears the buffer, so a later W alone
    // must not commit.
    reset_dut(0);
    aw_handshake(0, 9'h024);
    check(!awready[0], "AW buffer occupied before reset");
    reset_dut(0);
    w_handshake(0, 32'hDEADBEEF, 4'hF);
    repeat (5) begin
      @(negedge clk); #1;
      check(!bvalid[0], "W-only after reset does not commit");
    end
    read_word(0, 9'h024, OKAY, 0);
    reset_dut(0);

    // Reset with only W queued: a later AW alone must not commit.
    reset_dut(0);
    w_handshake(0, 32'hFEEDFACE, 4'hF);
    check(!wready[0], "W buffer occupied before reset");
    reset_dut(0);
    aw_handshake(0, 9'h028);
    repeat (5) begin
      @(negedge clk); #1;
      check(!bvalid[0], "AW-only after reset does not commit");
    end
    read_word(0, 9'h028, OKAY, 0);
    reset_dut(0);

    // Reset with an outstanding B response.
    reset_dut(0);
    aw_handshake(0, 9'h02C);
    w_handshake(0, 32'hB00B00B0, 4'hF);
    wait_bvalid(0, OKAY, 0);
    reset_dut(0);
    check(!bvalid[0], "BVALID cleared by reset");
    read_word(0, 9'h02C, OKAY, 0);

    // Reset with an outstanding R response.
    reset_dut(0);
    ar_handshake(0, 9'h030);
    wait_rvalid(0, 32'h00000000, OKAY, 0);
    reset_dut(0);
    check(!rvalid[0], "RVALID cleared by reset");
    check_eq32(rdata[0], 32'h0, "RDATA cleared by reset");
    read_word(0, 9'h030, OKAY, 0);
    check_counts(0);

    // Smaller PMEM_SIZE boundary and aliasing checks.
    write_word(1, 9'h000, 32'h11223344, 4'hF, 0, OKAY, 0);
    read_word(1, 9'h000, OKAY, 0);
    write_word(1, 9'h03C, 32'hAABBCCDD, 4'hF, 2, OKAY, 0);
    read_word(1, 9'h03C, OKAY, 0);
    write_word(1, 9'h004, 32'h01020304, 4'b0101, 0, OKAY, 0);
    read_word(1, 9'h004, OKAY, 0);
    write_word(1, 9'h008, 32'h55667788, 4'hF, 1, OKAY, 0);
    read_word(1, 9'h008, OKAY, 0);
    write_word(1, 9'h004, 32'hFFFFFFFF, 4'b0000, 0, OKAY, 0);
    read_word(1, 9'h004, OKAY, 0);

    // 0x040 is the first aligned address past the 16-word window; 0x100 and
    // 0x1FC stay out of range. Misaligned in-window and out-of-window
    // addresses also return DECERR and change nothing.
    write_word(1, 9'h040, 32'hFFFFFFFF, 4'hF, 0, DECERR, 0);
    write_word(1, 9'h044, 32'h00000000, 4'b0000, 0, DECERR, 0);
    write_word(1, 9'h100, 32'hFFFFFFFF, 4'hF, 0, DECERR, 0);
    write_word(1, 9'h1FC, 32'hFFFFFFFF, 4'hF, 0, DECERR, 0);
    read_word(1, 9'h040, DECERR, 0);
    read_word(1, 9'h044, DECERR, 0);
    read_word(1, 9'h100, DECERR, 0);
    read_word(1, 9'h1FC, DECERR, 0);
    write_word(1, 9'h002, 32'hFFFFFFFF, 4'hF, 0, DECERR, 0);
    write_word(1, 9'h03E, 32'hFFFFFFFF, 4'hF, 0, DECERR, 0);
    write_word(1, 9'h041, 32'hFFFFFFFF, 4'hF, 0, DECERR, 0);
    read_word(1, 9'h002, DECERR, 0);
    read_word(1, 9'h03E, DECERR, 0);
    read_word(1, 9'h041, DECERR, 0);
    read_word(1, 9'h000, OKAY, 0);
    read_word(1, 9'h03C, OKAY, 0);
    check_counts(1);

    if (errors != 0) begin
      $fatal(1, "TB FAIL: %0d errors / %0d checks", errors, checks);
    end
    $display("TB PASS: %0d checks", checks);
    $finish;
  end

endmodule
