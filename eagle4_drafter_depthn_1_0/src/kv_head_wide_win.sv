// =====================================================================
// kv_head_wide_win.sv  (M3 — KV-memory architecture)
// =====================================================================
//
// WIDE single-cycle KV memory for the 24-head MHA drafter, with a bounded
// on-chip context WINDOW and DEPTH-PACKING of multiple logical heads into
// one physical memory stack. The storage primitive (URAM vs BRAM) is
// selected per-instance via RAM_STYLE.
//
// WHY THIS EXISTS (the M2 KV scale wall):
//   M2 proved D_HEAD=128 × 24-head COMPUTE fits the V80 (32% DSP) but
//   `kv_head_wide` at full ctx=2048 costs 114 RAMB36 / memory → 24 heads
//   × 2 (K+V) = 5,472 RAMB36 = 146% BRAM, or 1,392 URAM in the naive
//   per-head-wide layout. Layout B already commits ~80% URAM (~1,536) to
//   qweights, leaving only ~389 free. The fit must come from the KV
//   MEMORY ARCHITECTURE (EAGLE4 is MHA, no GQA — KV cannot be shrunk by
//   grouping queries).
//
// THE ARCHITECTURE (Arch D — depth-packed wide memory):
//   - A URAM block is 4096 deep × 72 bit. The drafter's per-step working
//     context (CTX_WIN) is <= 1024, so each per-head KV vector only uses
//     CTX_WIN of the 4096 depth. We therefore pack HEADS_PER_STACK logical
//     heads back-to-back along the depth of ONE physical stack:
//       physical_addr = head_in_stack * CTX_WIN + pos
//   - The wide row (D_HEAD × A_WIDTH = 2048 bit) is laid across the
//     primitive columns. Single-cycle wide read (1-cycle registered
//     latency) — preserves the attention_head_assembly / flash_attn_head
//     wide-vector API with NO extra latency (unlike a narrow
//     element-addressed cache that would need D_HEAD/4 = 32 sub-cycles).
//
// =====================================================================
// WO-053: XPM-BACKED STORAGE (this revision)
// =====================================================================
//   The previous revision declared the packed store as a `reg [ROW_W-1:0]
//   mem [...]` with an `(* ram_style = RAM_STYLE *)` ATTRIBUTE. On the V80
//   that attribute was SILENTLY IGNORED for the wide V stacks: the routed
//   netlist showed the V arrays staying in URAM288 (174 cells proven), so
//   the V store could NOT co-locate in the same SLR as the attn_v_combine
//   MAC → the −0.780 ns KV-read→combine SLR crossing survived.
//
//   This revision instantiates `xpm_memory_sdpram` and FORCES the primitive
//   via MEMORY_PRIMITIVE (a hard synthesis directive, not a hint):
//       RAM_STYLE="block" → MEMORY_PRIMITIVE="block" → RAMB36   (V stacks)
//       RAM_STYLE="ultra" → MEMORY_PRIMITIVE="ultra" → URAM288  (K stacks)
//
//   PURE implementation attribute — ZERO functional / numerical change
//   (memory feedback_no_accuracy_tradeoff). The write/read PORTS, the
//   address arithmetic, the 1-cycle registered read latency, the
//   read-enable "hold last value" semantic, and every stored/returned bit
//   are IDENTICAL to the reg-array revision:
//     * Write: full ROW_W vector written when wr_en, at (wr_head,wr_pos).
//       BYTE_WRITE_WIDTH_A = ROW_W → single-bit wea, whole-word write.
//     * Read : READ_LATENCY_B = 1 → exactly one output register, gated by
//       enb=rd_en. doutb holds its previous value while rd_en is low —
//       this matches `if (rd_en) rd_vec <= mem[rd_addr];`. WRITE_MODE_B =
//       "read_first": on a concurrent SAME-ADDRESS read+write, doutb
//       returns the OLD (pre-write) contents. This is EXACTLY the
//       reg-array's nonblocking-assignment semantic — the read RHS
//       `mem[rd_addr]` was evaluated before the write NBA committed, so the
//       reg-array also returned the old value. (read_first is the only mode
//       that matches the reg-array AND is legal for BOTH the block/RAMB36
//       and the ultra/URAM288 primitives; "no_change" is illegal for URAM
//       SDP per XPM DRC 40-8.)
//     * Init : USE_MEM_INIT=1 (default) zero-fills mem in sim, matching the
//       old `initial mem[i]='0`. READ_RESET_VALUE_B="0".
//
//   CLOCKING_MODE="common_clock" (single clk on both ports). rstb tied 1'b0
//   (the read port had no reset in the reg-array form; rd_vec was only ever
//   driven by reads).
//
// API: identical to kv_head_wide (drop-in), plus a HEAD select for the
// depth-packed stack. With HEADS_PER_STACK=1 this degenerates to the
// per-head wide cache.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module kv_head_wide_win #(
    parameter int D_HEAD          = 128,
    parameter int A_WIDTH         = 16,
    parameter int CTX_WIN         = 1024,   // on-chip context window (<= 2048)
    parameter int HEADS_PER_STACK = 4,      // logical heads packed per physical stack
    // ---------------------------------------------------------------
    // WO-053: select the storage primitive for the packed store.
    //   "ultra" → URAM288  (default, K stacks — original behaviour)
    //   "block" → RAMB36   (V stacks — BRAM-shed so the V store
    //                        co-locates in the SAME SLR as the
    //                        attn_v_combine MAC; kills the −0.780 ns
    //                        KV-read→combine SLR crossing).
    // Now a HARD directive via xpm_memory_sdpram.MEMORY_PRIMITIVE,
    // not an ignorable (* ram_style *) hint. ZERO functional change.
    // ---------------------------------------------------------------
    parameter         RAM_STYLE       = "ultra",
    // ---------------------------------------------------------------
    // RTL WO#12 (KVADDR_REP): number of same-cycle ADDRESS-REPLICA GROUPS
    // the packed store is split into along the ROW width. The run-10
    // RESPIN50 blocker (KVADDR_REP_FIX.md) is the read-position replica
    // -> V-BRAM ADDR net: ONE xpm has ONE addrb port, so the per-stack
    // replica still fans out to every RAMB36 of the ~228-BRAM expansion
    // (fo=456, 13.602 ns pure route, 0 logic levels, SLR1->SLR0).
    //   ADDR_REP=1: the historical single xpm (byte-identical netlist).
    //   ADDR_REP=G: G xpm slices of ROW_W/G bits, SAME depth/latency/
    //       write-mode; slice g is addressed from rd_pos_rep/wr_pos_rep
    //       slice g (a same-edge replica register upstream), so each
    //       address net serves only ~228/G BRAMs and the placer can pull
    //       each replica to its own bank group (the ROPE_POS_REP idiom).
    //   BIT-EXACT: every slice carries the same address value each cycle;
    //   the row read/written is the identical ROW_W vector.
    //   LATENCY-NEUTRAL: READ_LATENCY_B=1 per slice, same enb/regceb.
    // ---------------------------------------------------------------
    parameter int ADDR_REP        = 1,
    parameter int ROW_W           = D_HEAD * A_WIDTH,
    parameter int STACK_DEPTH     = CTX_WIN * HEADS_PER_STACK,
    parameter int HSEL_W          = (HEADS_PER_STACK <= 1) ? 1 : $clog2(HEADS_PER_STACK),
    parameter int POS_W           = (CTX_WIN <= 1) ? 1 : $clog2(CTX_WIN)
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Write port — full D_HEAD vector at (head_in_stack, pos)
    input  wire                  wr_en,
    input  wire [HSEL_W-1:0]     wr_head,
    input  wire [10:0]           wr_pos,
    input  wire [ROW_W-1:0]      wr_vec,

    // Read port — 1-cycle latency
    input  wire                  rd_en,
    input  wire [HSEL_W-1:0]     rd_head,
    input  wire [10:0]           rd_pos,
    output wire [ROW_W-1:0]      rd_vec,

    // RTL WO#12 (KVADDR_REP): per-group replicated positions, used ONLY by
    // the ADDR_REP>1 arm (slice g addresses memory slice g). Same value on
    // every slice each cycle — bit-exact. Legacy callers (ADDR_REP=1) may
    // leave them unconnected: the mono arm never reads them.
    input  wire [ADDR_REP*11-1:0] rd_pos_rep,
    input  wire [ADDR_REP*11-1:0] wr_pos_rep
);

    localparam int ADDR_W = (STACK_DEPTH <= 1) ? 1 : $clog2(STACK_DEPTH);

    // physical address = head_in_stack * CTX_WIN + pos  (depth-packing) —
    // identical arithmetic to the reg-array revision.
    wire [ADDR_W-1:0] wr_addr =
        ADDR_W'(({{ (32-HSEL_W){1'b0} }, wr_head} * CTX_WIN)
                + {{ (32-POS_W){1'b0} }, wr_pos[POS_W-1:0]});
    wire [ADDR_W-1:0] rd_addr =
        ADDR_W'(({{ (32-HSEL_W){1'b0} }, rd_head} * CTX_WIN)
                + {{ (32-POS_W){1'b0} }, rd_pos[POS_W-1:0]});

    // ---------------------------------------------------------------
    // XPM simple-dual-port RAM.
    //   Port A = write-only, Port B = read-only (1-cycle registered).
    //   MEMORY_PRIMITIVE hard-selected from RAM_STYLE.
    // RTL WO#12: ADDR_REP=1 keeps the historical single instance
    // (byte-identical); ADDR_REP=G splits the store into G width-slices,
    // each addressed from its own same-cycle replica slice.
    // ---------------------------------------------------------------
    generate
    // Elaboration guards: a non-dividing group count or a rep bus that
    // cannot be driven correctly must fail loudly, not truncate silently.
    if (ADDR_REP > 1 && (ROW_W % ADDR_REP) != 0) begin : g_cfg_err
        kv_addr_rep_must_divide_row_w u_cfg_err();
    end
    endgenerate

    generate
    if (ADDR_REP <= 1) begin : g_mono
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A        (ADDR_W),
        .ADDR_WIDTH_B        (ADDR_W),
        .AUTO_SLEEP_TIME     (0),
        .BYTE_WRITE_WIDTH_A  (ROW_W),         // whole-word write → 1-bit wea
        .CASCADE_HEIGHT      (0),
        .CLOCKING_MODE       ("common_clock"),
        .ECC_MODE            ("no_ecc"),
        .MEMORY_INIT_FILE    ("none"),
        .MEMORY_INIT_PARAM   ("0"),
        .MEMORY_OPTIMIZATION ("true"),
        .MEMORY_PRIMITIVE    (RAM_STYLE),     // "ultra"→URAM, "block"→BRAM
        .MEMORY_SIZE         (STACK_DEPTH * ROW_W),  // bits
        .MESSAGE_CONTROL     (0),
        .READ_DATA_WIDTH_B   (ROW_W),
        .READ_LATENCY_B      (1),             // 1-cycle registered read (matches old always_ff)
        .READ_RESET_VALUE_B  ("0"),
        .RST_MODE_A          ("SYNC"),
        .RST_MODE_B          ("SYNC"),
        .SIM_ASSERT_CHK      (0),
        .USE_EMBEDDED_CONSTRAINT (0),
        .USE_MEM_INIT        (1),             // zero-fill in sim (matches old initial)
        .USE_MEM_INIT_MMI    (0),
        .WAKEUP_TIME         ("disable_sleep"),
        .WRITE_DATA_WIDTH_A  (ROW_W),
        .WRITE_MODE_B        ("read_first"),  // same-addr R+W → OLD data (matches reg-array NBA; legal for BRAM+URAM)
        .WRITE_PROTECT       (1)
    ) u_mem (
        // Port A — write
        .clka            (clk),
        .ena             (wr_en),
        .wea             (1'b1),              // BYTE_WRITE_WIDTH_A==ROW_W → 1 bit
        .addra           (wr_addr),
        .dina            (wr_vec),
        .injectsbiterra  (1'b0),
        .injectdbiterra  (1'b0),

        // Port B — read (1-cycle)
        .clkb            (clk),
        .rstb            (1'b0),              // read port had no reset in reg-array form
        .enb             (rd_en),             // gate output register → hold-last-value when low
        .regceb          (1'b1),
        .addrb           (rd_addr),
        .doutb           (rd_vec),
        .sbiterrb        (/* unused */),
        .dbiterrb        (/* unused */),

        // Common
        .sleep           (1'b0)
    );
    end else begin : g_grp
        // =============================================================
        // RTL WO#12 (KVADDR_REP) — per-bank-group split. G width-slices,
        // each its own xpm with its OWN addra/addrb port driven from its
        // own replica slice. Every slice sees the SAME (head, pos) pair
        // each cycle (replicas are same-edge twins upstream), so the
        // aggregate row is bit-identical to the mono arm:
        //   * same depth, same head*CTX_WIN+pos arithmetic per slice
        //   * READ_LATENCY_B=1, enb=rd_en, regceb=1 -> same 1-cycle
        //     registered read + hold-last-value semantic per slice
        //   * WRITE_MODE_B="read_first" per slice == read_first aggregate
        //   * USE_MEM_INIT zero-fill per slice == zero-filled row
        // The point is PHYSICAL: address fanout per net drops from the
        // full ~ROW_W/9 BRAM expansion to 1/G of it, and each keep'd
        // replica register can place WITH its group.
        // =============================================================
        localparam int GRP_W = ROW_W / ADDR_REP;
        for (genvar g = 0; g < ADDR_REP; g = g + 1) begin : g_slice
            wire [10:0] g_rd_pos = rd_pos_rep[g*11 +: 11];
            wire [10:0] g_wr_pos = wr_pos_rep[g*11 +: 11];
            wire [ADDR_W-1:0] g_wr_addr =
                ADDR_W'(({{ (32-HSEL_W){1'b0} }, wr_head} * CTX_WIN)
                        + {{ (32-POS_W){1'b0} }, g_wr_pos[POS_W-1:0]});
            wire [ADDR_W-1:0] g_rd_addr =
                ADDR_W'(({{ (32-HSEL_W){1'b0} }, rd_head} * CTX_WIN)
                        + {{ (32-POS_W){1'b0} }, g_rd_pos[POS_W-1:0]});

            xpm_memory_sdpram #(
                .ADDR_WIDTH_A        (ADDR_W),
                .ADDR_WIDTH_B        (ADDR_W),
                .AUTO_SLEEP_TIME     (0),
                .BYTE_WRITE_WIDTH_A  (GRP_W),        // whole-slice write → 1-bit wea
                .CASCADE_HEIGHT      (0),
                .CLOCKING_MODE       ("common_clock"),
                .ECC_MODE            ("no_ecc"),
                .MEMORY_INIT_FILE    ("none"),
                .MEMORY_INIT_PARAM   ("0"),
                .MEMORY_OPTIMIZATION ("true"),
                .MEMORY_PRIMITIVE    (RAM_STYLE),    // "ultra"→URAM, "block"→BRAM
                .MEMORY_SIZE         (STACK_DEPTH * GRP_W),  // bits
                .MESSAGE_CONTROL     (0),
                .READ_DATA_WIDTH_B   (GRP_W),
                .READ_LATENCY_B      (1),            // same 1-cycle registered read
                .READ_RESET_VALUE_B  ("0"),
                .RST_MODE_A          ("SYNC"),
                .RST_MODE_B          ("SYNC"),
                .SIM_ASSERT_CHK      (0),
                .USE_EMBEDDED_CONSTRAINT (0),
                .USE_MEM_INIT        (1),            // zero-fill in sim (matches mono)
                .USE_MEM_INIT_MMI    (0),
                .WAKEUP_TIME         ("disable_sleep"),
                .WRITE_DATA_WIDTH_A  (GRP_W),
                .WRITE_MODE_B        ("read_first"), // same-addr R+W → OLD data (matches mono)
                .WRITE_PROTECT       (1)
            ) u_mem (
                // Port A — write (slice)
                .clka            (clk),
                .ena             (wr_en),
                .wea             (1'b1),             // BYTE_WRITE_WIDTH_A==GRP_W → 1 bit
                .addra           (g_wr_addr),
                .dina            (wr_vec[g*GRP_W +: GRP_W]),
                .injectsbiterra  (1'b0),
                .injectdbiterra  (1'b0),

                // Port B — read (slice, 1-cycle)
                .clkb            (clk),
                .rstb            (1'b0),
                .enb             (rd_en),
                .regceb          (1'b1),
                .addrb           (g_rd_addr),
                .doutb           (rd_vec[g*GRP_W +: GRP_W]),
                .sbiterrb        (/* unused */),
                .dbiterrb        (/* unused */),

                // Common
                .sleep           (1'b0)
            );
        end
        // The shared-scalar rd_addr/wr_addr of the mono arm are unused
        // here (each slice derives its own); reference them so lint does
        // not flag the module-scope wires in this configuration.
        wire _unused_mono_addr = ^{wr_addr, rd_addr};
    end
    endgenerate

    // rst_n is part of the stable interface but unused by the read/write
    // datapath (the reg-array revision did not reset mem or rd_vec either).
    wire _unused_rst_n = rst_n;

endmodule

`default_nettype wire
