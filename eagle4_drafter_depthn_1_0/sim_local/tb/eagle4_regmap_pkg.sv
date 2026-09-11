// =====================================================================
// eagle4_regmap_pkg.sv — TESTBENCH ONLY (see sim_local/README in run_sim.sh
// header). Named constants for the AXI4-Lite register map documented in
// control_regs.sv, so the testbench reads like the register map instead of
// a page of magic hex numbers. Verified against the actual case-statement
// decode in control_regs.sv (not just its comments — the comments there
// have drifted from the code before, e.g. the top-of-file header still
// claims 0x08 is the unused IP_IER register; the real decode at line 619
// repurposes it as boot_gamma_sel).
// =====================================================================
package eagle4_regmap_pkg;

    // ---- AP_CTRL handshake (Xilinx HLS convention) ----
    localparam int ADDR_AP_CTRL      = 'h00;  // W bit0=start | R bit1=done,bit2=idle,bit3=ready
    localparam int ADDR_GAMMA_SEL    = 'h08;  // W [2:0]: 0=off,1=hidden_norm_g,2=e_norm,3=postattn,4=final
    localparam int ADDR_CTX_LEN      = 'h10;
    localparam int ADDR_NUM_FRONTIER = 'h18;
    localparam int ADDR_MAX_DEPTH    = 'h20;
    localparam int ADDR_OUT0_LO      = 'h28;
    localparam int ADDR_OUT0_HI      = 'h2c;
    localparam int ADDR_OUT1_LO      = 'h30;
    localparam int ADDR_OUT1_HI      = 'h34;
    localparam int ADDR_OUT2_LO      = 'h38;
    localparam int ADDR_OUT2_HI      = 'h3c;

    // ---- M1c "AXI-Lite trio": one address (0x4c), one data mailbox (0x50,
    // auto-increments the address after each write), retargeted by which of
    // gamma_sel/is_hfc/is_emb is set. Default target (all zero) = the shared
    // WEIGHT URAM banks, but we load those via the faster TB direct-drive
    // pins instead (see tb_eagle4_drafter.sv) and only use this trio for
    // EMB / HFC / GAMMA, which have no direct-drive equivalent. ----
    localparam int ADDR_BOOT_LANE_SEL = 'h48;  // irrelevant for emb/hfc/gamma, only matters for WEIGHT target
    localparam int ADDR_BOOT_WR_ADDR  = 'h4c;  // set once per load; auto-increments internally after each 0x50 write
    localparam int ADDR_BOOT_WR_DATA  = 'h50;  // each write here pulses the load strobe for whichever target is selected

    localparam int ADDR_HFC_SELECT    = 'h24;  // W bit0=1 -> trio targets h_fc_buf
    localparam int ADDR_EMB_SELECT    = 'h7c;  // W bit0=1 -> trio targets input_emb_buf

    localparam int ADDR_OUT_DMA_STATUS = 'h78; // R bit0=busy,bit1=done,bit2=err,bit3=g8_watchdog_err

    // ---- ISA PMEM mailbox & status ----
    localparam int ADDR_PMEM_ADDR      = 'h168; // W write_addr pointer | R {24'b0, write_addr}
    localparam int ADDR_PMEM_DATA      = 'h16c; // W write_data (autoincs write_addr) | R {16'h504D, 1'b0, PC, instr}

    // ---- dbg_residency.sv self-check: "did my load actually land" ----
    localparam int ADDR_RES_SEL   = 'hec; // W [2:0] buffer select (0=EMB,1=HFC,2=GAMMA,3=WEIGHT,4=SCALE), bit3=clear pulse
    localparam int ADDR_RES_CRC   = 'hf0; // R
    localparam int ADDR_RES_SUM   = 'hf4; // R
    localparam int ADDR_RES_COUNT = 'hf8; // R  <- the one we actually care about for a smoke test: nonzero == "load path wrote something"

    localparam int RES_SEL_EMB    = 0;
    localparam int RES_SEL_HFC    = 1;
    localparam int RES_SEL_GAMMA  = 2;
    localparam int RES_SEL_WEIGHT = 3;
    localparam int RES_SEL_SCALE  = 4;

    // ---- Gamma target codes (boot_gamma_sel), from eagle4_drafter.sv:3608-3611 ----
    localparam int GAMMA_HIDDEN_NORM = 1; // gamma_emb_buf
    localparam int GAMMA_E_NORM      = 2; // gamma_e_buf
    localparam int GAMMA_POSTATTN    = 3; // gamma_postattn_buf
    localparam int GAMMA_FINAL       = 4; // gamma_final_buf

    // ---- Buffer geometry, all H=3072 / K_PAR=16 / A_WIDTH=16 packing
    // (2 elements per 32-bit word) confirmed at the EMB buffer's own
    // localparams (eagle4_drafter.sv:1456-1459) and the rmsnorm_block
    // instantiation parameters used for every gamma buffer. ----
    localparam int D_MODEL          = 3072;
    localparam int WORDS_PER_VECTOR = D_MODEL / 2;        // 1536 32-bit words for one D_MODEL-wide buffer
    localparam int EMB_WORDS        = WORDS_PER_VECTOR;   // input_emb_buf:  1 x D_MODEL
    localparam int HFC_WORDS        = WORDS_PER_VECTOR*3; // h_fc_buf:       3 x D_MODEL (your low/mid/hi features, concatenated)
    localparam int GAMMA_WORDS      = WORDS_PER_VECTOR;   // each of the 4 gamma (RMSNorm scale) buffers

    // ---- Shared GEMV weight array (gemv_stage_array_scaled_slr.sv):
    // ONE physical bank per lane, LANES=128 lanes total, each bank deep
    // enough to hold every stage's weight tile back-to-back. ----
    localparam int LANES        = 128;
    localparam int WEIGHT_DEPTH = (96*1024*128) / LANES;  // gemv_stage_array_scaled_slr.sv:219 BANK_DEPTH, GEOM_LANES=LANES -> 98304
    localparam int SCALE_DEPTH  = (16*1024*128) / LANES;  // gemv_stage_array_scaled_slr.sv:220 SCALE_DEPTH -> 16384

    localparam int VOCAB     = 128256; // TREE_V_MAX, eagle4_config_intent.txt
    localparam int TOP_K     = 16;     // OUT_TOP_K
    localparam int K_LIVE    = 8;      // live tree results per frontier (slots 0..7); 8..15 are zero-pad
    localparam int PARENT_SENTINEL = -1;

endpackage
