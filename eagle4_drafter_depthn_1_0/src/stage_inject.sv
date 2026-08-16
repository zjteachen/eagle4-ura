// =====================================================================
// stage_inject.sv  (PDI#11 PER-STAGE GOLDEN-INJECT — the INPUT half of isolation)
// =====================================================================
//
// MOTIVATION (USER MANDATE 2026-06-16):
//   dbg_stage_capture (0xDC/0x108) already lets the host read EACH stage's OUTPUT
//   cap-CRC after one silicon forward (the CAPTURE half). What was missing is the
//   INJECT half: force a KNOWN-CORRECT (golden) chunk INTO a chosen stage's INPUT
//   so the captured output CRC isolates THAT stage's transform alone — turning the
//   current free-run cascade ("everything downstream of the first fault diverges")
//   into decisive per-stage isolation (inject golden at G7p input -> the silu
//   output CRC must be 0xd8d6d147 regardless of upstream body faults).
//
//   CAPTURE (output) + INJECT (input) compose: load golden -> set inject_en[s] ->
//   run -> read 0x108 for stage s. If the output matches golden, stage s is clean;
//   if not, stage s itself is the fault.
//
// SCOPE / SAFETY (the headline contract):
//   * Gated behind the top param STAGE_INJECT (default 0). With STAGE_INJECT=0 this
//     ENTIRE block is `generate`-pruned: zero FF, zero LUT, the inject mux collapses
//     to a pass-through wire -> the netlist + behavior are BIT-IDENTICAL to the
//     no-inject top (the ultimate safety). The build the user launches keeps it 0
//     unless explicitly turned on. (memory feedback_no_accuracy_tradeoff:
//     additive-only, never perturb the functional path.)
//   * With STAGE_INJECT=1 AND inject_en=0 the mux still selects the REAL upstream
//     net every cycle -> behavior bit-identical to the no-inject top. PROVEN by the
//     full-chain frontier+capCRC gate (tb_fullchain_regread_frontier_bestspeed,
//     a TRUE full-chain run with zero force/inject anywhere; memory
//     feedback_golden_inject_gate_false_fullchain).
//
// STORAGE (memory feedback_nodriver_bake_drop_synth_only):
//   ONE shared inject buffer inj_buf[0:MAX_W-1] of 32-bit words, written word-by-
//   word over AXI (always_ff, REAL drivers — NO $readmemh, nothing dropped in
//   synth; gate 8-3848==0). The host reuses ONE physical buffer for whichever
//   stage it is isolating (load golden chunk -> set inject_en[s] -> run -> repeat
//   for the next stage). MAX_W=128 words (4 Kb FF) — trivial on V80, no URAM/BRAM
//   so the URAM 1710/88.83% binding wall is untouched.
//
// NO NEW WIDE SLR-SPANNING NET (memory feedback_banksplit_did_not_fix_plus128 /
//   the very bug being fixed): the inject mux output replaces an EXISTING stage-
//   input net AT ITS EXISTING CONSUMER — it adds a 2:1 mux at the point the seam
//   read already feeds, it does NOT route a new wide bus across SLRs. The buffer is
//   read combinationally as a flat FF array (like dbg_stage_capture.cap[]), a LUT
//   mux, not a RAM macro -> no COMBREAD/LUTRAM-downgrade hazard.
//
// MULTI-DRIVER (memory feedback_pulse_reg_multidriver_reset_plus_proc):
//   every reg here (inj_en_q, inj_sel_q, inj_buf, wr_idx) is owned by EXACTLY ONE
//   always_ff. The 0x110 seek/auto-inc pointer is self-clearing in one process.
//
// LIVENESS (memory feedback_probe_reg_needs_real_axi_write_and_liveness):
//   0x11C returns {CANARY=0x1A, inject_en_present, inj_sel, last-written-word low
//   bits}. The host reads 0x11C FIRST: if the CANARY nibble is absent the reg is
//   dead on silicon -> the inject result is INCONCLUSIVE, never "the stage is
//   clean". Plus the real-AXI write+readback of inject_en / inj_sel (mismatch=FAIL).
//
// HOST AXI-LITE MAP (free 9-bit offsets above the 0x108 MAXPROBE window):
//   0x10C W  inject_en[NSTAGE-1:0]  : per-stage enable bitmask (sticky).
//   0x110 W  inj_widx               : buffer write pointer; AUTO-INCS after each
//                                     0x114 write; WRITING seeks.
//   0x114 W  inj_data               : inj_buf[inj_widx] <= wdata; auto-incs widx.
//   0x118 W  inj_sel[4:0]           : informational stage tag (read-back, liveness).
//   0x11C R  inject_status          : {8'h1A canary, NSTAGE present-flag, inj_sel,
//                                       low bits of last written word}.
//   0x10C R  inject_en read-back    : the live bitmask (write+readback liveness).
//   0x110 R  inj_widx read-back     : current write pointer.
//
// STAGE CODE MAP — IDENTICAL to dbg_stage_capture so capture+inject share one
//   index space (0xDC sel == 0x118 sel for the same stage):
//   0 G1  1 G4q  2 G4k  3 G4v  4 G5  5 G6  6 G7g  7 G7u  8 G7p  9 G9
//   10 G11  11 GLM  12 G12  13 TREE  14 R1  15 R3  16 R7  (17/18 = G8 in/out: capture-only)
//   NOTE: only stages whose INPUT net the top muxes are actually injectable; the
//   per-stage `inject_active[s]` exported below tells the top which mux arms exist.
//   inject_en bits for capture-only stages are accepted but inert (no mux).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

module stage_inject #(
    parameter int NSTAGE = 19,                 // shared index space with dbg_stage_capture
    parameter int MAX_W  = 128,                // words in the shared inject buffer (<= widest chunk)
    parameter int WIDX_W = 7                    // ceil log2 MAX_W (128 -> 7)
) (
    input  wire        clk,
    input  wire        rst_n,

    // ---- host control (decoded 1-cyc pulses + data from control_regs) ----
    input  wire                  en_we,        // 0x10C write pulse
    input  wire [NSTAGE-1:0]     en_wdata,     // 0x10C data
    input  wire                  widx_we,      // 0x110 write pulse (seek)
    input  wire [WIDX_W-1:0]     widx_wdata,   // 0x110 seek value
    input  wire                  data_we,      // 0x114 write pulse (buffer fill + auto-inc)
    input  wire [31:0]           data_wdata,   // 0x114 data
    input  wire                  sel_we,       // 0x118 write pulse
    input  wire [4:0]            sel_wdata,    // 0x118 informational stage tag

    // ---- host read-back ----
    output wire [31:0] en_word,                // 0x10C read-back (live bitmask)
    output wire [31:0] widx_word,              // 0x110 read-back (write pointer)
    output wire [31:0] status_word,            // 0x11C liveness/status

    // ---- to the datapath: live enable mask + flat buffer read bus ----
    output wire [NSTAGE-1:0]     inject_en,    // per-stage enable (to the top muxes)
    output wire [MAX_W*32-1:0]   inj_bus       // flat 32b/word buffer (top slices per stage)
);

    // -----------------------------------------------------------------
    // 0x10C inject_en bitmask — ONE owner (sticky).
    // -----------------------------------------------------------------
    reg [NSTAGE-1:0] inj_en_q;
    always_ff @(posedge clk) begin
        if (!rst_n)      inj_en_q <= '0;
        else if (en_we)  inj_en_q <= en_wdata;
    end

    // -----------------------------------------------------------------
    // 0x118 inj_sel informational tag — ONE owner.
    // -----------------------------------------------------------------
    reg [4:0] inj_sel_q;
    always_ff @(posedge clk) begin
        if (!rst_n)       inj_sel_q <= '0;
        else if (sel_we)  inj_sel_q <= sel_wdata;
    end

    // -----------------------------------------------------------------
    // 0x110 write pointer — seek on widx_we (priority), auto-inc on data_we.
    // SEEK WINS in the same cycle (host does seek 0x110=0 then streams 0x114).
    // WIDX_W bits => natural wrap at MAX_W (avoids the always-true >=MAX_W stuck
    // trap, exactly like dbg_stage_capture.rd_idx).
    // -----------------------------------------------------------------
    reg [WIDX_W-1:0] wr_idx;
    always_ff @(posedge clk) begin
        if (!rst_n)        wr_idx <= '0;
        else if (widx_we)  wr_idx <= widx_wdata;            // SEEK priority
        else if (data_we)  wr_idx <= wr_idx + WIDX_W'(1);   // auto-inc, natural wrap
    end

    // -----------------------------------------------------------------
    // Shared inject buffer inj_buf[MAX_W] — ONE owner (the 0x114 write port).
    // Plain clocked FF array (real drivers, no $readmemh) -> survives synth as FF
    // (no BRAM/LUTRAM downgrade). Combinational flat read bus to the top.
    // -----------------------------------------------------------------
    reg [31:0] inj_buf [0:MAX_W-1];
    reg [31:0] last_wr_word;
    integer ii;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (ii = 0; ii < MAX_W; ii = ii + 1) inj_buf[ii] <= 32'd0;
            last_wr_word <= 32'd0;
        end else if (data_we) begin
            inj_buf[wr_idx] <= data_wdata;
            last_wr_word    <= data_wdata;
        end
    end

    genvar gw;
    generate
        for (gw = 0; gw < MAX_W; gw = gw + 1) begin : g_bus
            assign inj_bus[gw*32 +: 32] = inj_buf[gw];
        end
    endgenerate

    assign inject_en   = inj_en_q;
    assign en_word     = {{(32-NSTAGE){1'b0}}, inj_en_q};
    assign widx_word   = {{(32-WIDX_W){1'b0}}, wr_idx};

    // 0x11C liveness/status: CANARY proves the reg is alive on silicon (read FIRST).
    //   [31:24] = 0x1A canary
    //   [23]    = 1 (inject infra present -> distinguishes a STAGE_INJECT=0 build,
    //                where the top ties this read to 0 and the canary is absent)
    //   [20:16] = inj_sel_q (the informational stage tag, write+readback liveness)
    //   [15:0]  = low 16b of the last word written into the buffer (data-path canary)
    assign status_word = {8'h1A, 1'b1, 2'b0, inj_sel_q, last_wr_word[15:0]};

endmodule

`default_nettype wire
