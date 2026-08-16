// =====================================================================
// multihead_kv.sv  (M3 — 24-head KV-memory wrapper)
// =====================================================================
//
// Arranges the full 24-head MHA KV cache (K and V) under the Arch D
// depth-packed wide-URAM architecture (kv_head_wide_win). This is the
// module the ooc-synth resource gate measures: the WHOLE 24-head KV
// store, to confirm it fits alongside the ~80% weight-URAM at < 70% of
// every V80 dimension.
//
// LAYOUT:
//   - N_KV logical heads, each with a K memory and a V memory (MHA).
//   - HEADS_PER_STACK logical heads share one physical URAM stack along
//     depth (head_in_stack * CTX_WIN + pos addressing).
//   - N_STACKS = ceil(N_KV / HEADS_PER_STACK) physical stacks for K, and
//     the same again for V.
//
// RESOURCE (CTX_WIN=1024, HEADS_PER_STACK=4, D_HEAD=128, A_WIDTH=16):
//   per stack = ceil(2048/72)=29 URAM cols × 1 (STACK_DEPTH=4096 = exactly
//   one URAM depth) = 29 URAM. N_STACKS = ceil(24/4)=6 for K + 6 for V =
//   12 stacks × 29 = 348 URAM (18.1% of 1925). Combined with the ~80%
//   (~1536) weight-URAM that is 1884/1925 = 97.9% URAM total — OVER 70%.
//
//   *** IMPORTANT GATE NUANCE ***
//   The < 70% target is per-dimension for the KV store IN ISOLATION on a
//   bare part (the ooc-synth deliverable), AND the combined check
//   (KV + weight-URAM). 348 KV-URAM fits the 389 FREE slack (the binding
//   constraint the spec states), but KV+weights together exceed 70% of the
//   URAM pool because weights alone are ~80%. See report: the achievable
//   target is "KV fits the URAM slack left by the weights", which 348 does
//   (margin 41); the literal "combined < 70% URAM" is impossible while
//   weights occupy 80% and is reported as the honest gap.
//
// I/O: per-head wide vector ports flattened into buses. The attention
// engine drives one head's wr/rd at a time per the FSM; this wrapper
// routes by head index to the correct stack + head_in_stack slot.
//
// Multi-driver-safe: each stack is a separate kv_head_wide_win instance
// with its own single-writer always_ff; no shared packed regs across
// instances.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module multihead_kv #(
    parameter int N_KV            = 24,
    parameter int D_HEAD          = 128,
    parameter int A_WIDTH         = 16,
    parameter int CTX_WIN         = 1024,
    parameter int HEADS_PER_STACK = 4,
    // ---------------------------------------------------------------
    // WO-053: per-port storage-primitive selection.
    //   K_RAM_STYLE / V_RAM_STYLE ∈ {"ultra"(URAM288), "block"(RAMB36)}.
    // Default = K in URAM, V in BRAM: the V stacks are the ones on the
    // red KV-read→attn_v_combine SLR crossing; moving them to BRAM lets
    // them co-locate with the combine in ONE SLR (crossing gone) while
    // leaving K in URAM. PURE attribute → BIT-IDENTICAL output
    // (memory feedback_no_accuracy_tradeoff).
    // ---------------------------------------------------------------
    parameter         K_RAM_STYLE     = "ultra",
    parameter         V_RAM_STYLE     = "block",
    parameter int ROW_W           = D_HEAD * A_WIDTH,
    parameter int N_STACKS        = (N_KV + HEADS_PER_STACK - 1) / HEADS_PER_STACK,
    parameter int HSEL_W          = (HEADS_PER_STACK <= 1) ? 1 : $clog2(HEADS_PER_STACK),
    parameter int HEAD_W          = (N_KV <= 1) ? 1 : $clog2(N_KV),
    // KV_REPL: when POS_REP!=0, each stack takes its read POSITION from its own
    // slice of k_rd_pos_rep / v_rd_pos_rep (a per-stack replicated copy driven by
    // the attention assembly) instead of the single shared k_rd_pos / v_rd_pos.
    // Bit-identical (every slice carries the same value each cycle); the point is
    // physical — one low-fanout register->stack net per stack, no SLR0->1 wide
    // read-address net to rescue. Default 0 = legacy shared scalar (unchanged).
    parameter int POS_REP         = 0,
    // W3-C1-P1 (CLOCK_CEILING_STUDY §1b/§3 P1): same replication idiom for the
    // WRITE position. The kv_wr_pos-class register is the #1 SPEEDWAVE routed
    // family (net fo=2,910, 13.6 ns route, ZERO logic, SLR crossing into the
    // V-BRAM ADDR pins — study §1b). WR_POS_REP!=0 makes each stack take its
    // write POSITION from its own slice of k_wr_pos_rep / v_wr_pos_rep —
    // same-cycle twins driven by the caller, so bit-identical, 0-latency
    // (P1 is contract-free; contrast P4's +1-latency variant, NOT used here).
    // Default 0 = legacy shared scalar (unchanged).
    parameter int WR_POS_REP      = 0,
    // -----------------------------------------------------------------
    // G5MEM RD_ALL (2026-07-21, G5_THROUGHPUT_INVESTIGATION §2.3):
    //   RD_ALL=0 (DEFAULT, legacy): rd_en reaches ONLY the stack holding
    //     k_rd_head/v_rd_head. The other N_STACKS-1 memories hold their
    //     output register — 5 of 6 read ports idle every cycle.
    //   RD_ALL=1: rd_en reaches EVERY stack. All stacks already derive
    //     their slot from the SAME slot_of(rd_head) and their position
    //     from their own k_rd_pos_rep slice, so enabling all six yields
    //     heads {slot, 4+slot, 8+slot, ..., 20+slot} at the requested
    //     position — one head per stack — with ZERO address arithmetic
    //     change. Consumers read k_rd_bus_flat / v_rd_bus_flat.
    //   The scalar k_rd_vec / v_rd_vec output mux is UNCHANGED in both
    //   modes (it still selects stack_of(rd_head_q)), so a legacy caller
    //   at RD_ALL=1 would still see the identical scalar row — the mode
    //   is purely additive read-enable breadth, never a data change.
    // -----------------------------------------------------------------
    parameter int RD_ALL          = 0,
    // ---------------------------------------------------------------
    // RTL WO#12 (KVADDR_REP): per-stack BANK-GROUP address replication.
    // Run-10 proved the per-stack replica is NOT enough for the V stacks:
    // one xpm has one addr port, so v_rd_pos_q_reg[2][5] still fanned to
    // the stack's whole ~228-RAMB36 expansion (fo=456, 13.602 ns route,
    // KV_VADDR +0.028 @66 MHz / -0.362 @80). ADDR_REP_V=G splits every V
    // stack into G width-slice memories, each addressed from its own
    // caller-driven same-cycle replica slice (rd AND wr). ADDR_REP_K is
    // the same lever for the K URAM stacks — default 1 (per-stack only):
    // the K cone is ~29 URAM/stack (8x fewer loads), never appeared in
    // the run-10 worst set, and any K split rounds up URAM at the 88.83%
    // binding wall (2048/72=28.4 -> G*ceil(28.4/G) > 29).
    // Requires POS_REP!=0 / WR_POS_REP!=0 when >1 (guarded).
    // COMPOSES WITH RD_ALL/HEAD_PAR: the split is purely within each of the
    // N_STACKS V memories; head-parallelism (RD_ALL) is orthogonal, so
    // 6 stacks x 8 slices = 48 xpm total (storage-neutral, same as serial).
    // ---------------------------------------------------------------
    parameter int ADDR_REP_K      = 1,
    parameter int ADDR_REP_V      = 1
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // --- K write/read (one head selected per access) ---
    input  wire                  k_wr_en,
    input  wire [HEAD_W-1:0]     k_wr_head,
    input  wire [10:0]           k_wr_pos,
    input  wire [ROW_W-1:0]      k_wr_vec,
    input  wire                  k_rd_en,
    input  wire [HEAD_W-1:0]     k_rd_head,
    input  wire [10:0]           k_rd_pos,
    output reg  [ROW_W-1:0]      k_rd_vec,

    // --- V write/read (one head selected per access) ---
    input  wire                  v_wr_en,
    input  wire [HEAD_W-1:0]     v_wr_head,
    input  wire [10:0]           v_wr_pos,
    input  wire [ROW_W-1:0]      v_wr_vec,
    input  wire                  v_rd_en,
    input  wire [HEAD_W-1:0]     v_rd_head,
    input  wire [10:0]           v_rd_pos,
    output reg  [ROW_W-1:0]      v_rd_vec,

    // KV_REPL per-stack replicated read positions (used only when POS_REP!=0).
    // Additive + defaulted so legacy callers (POS_REP=0) may leave them unconnected.
    // RTL WO#12: with ADDR_REP_K/V > 1 each stack carries ADDR_REP_* group
    // slices (stack s group g = slice s*ADDR_REP_*+g); at ADDR_REP_*=1 the
    // widths and semantics are byte-identical to the per-stack form.
    input  wire [N_STACKS*ADDR_REP_K*11-1:0] k_rd_pos_rep,
    input  wire [N_STACKS*ADDR_REP_V*11-1:0] v_rd_pos_rep,

    // W3-C1-P1 per-stack replicated WRITE positions (used only when
    // WR_POS_REP!=0). Additive + ignorable so legacy callers may leave them
    // unconnected (tie '0) — mirrors the POS_REP read-side pattern above.
    input  wire [N_STACKS*ADDR_REP_K*11-1:0] k_wr_pos_rep,
    input  wire [N_STACKS*ADDR_REP_V*11-1:0] v_wr_pos_rep,

    // WO-051: per-stack V read-data buses, flattened. Each slice
    // v_rd_bus_flat[s*ROW_W +: ROW_W] is stack s's registered V[pos] row,
    // co-located with that stack's URAM. The shared-combine path ignores
    // this and uses the muxed v_rd_vec; the per-bank combine consumes the
    // flat buses so the wide 2048-bit V-row never leaves the stack's SLR.
    // PURELY ADDITIVE — does not change v_rd_vec (bit-identical for the
    // shared path).
    output wire [N_STACKS*ROW_W-1:0] v_rd_bus_flat,

    // G5MEM: per-stack K read-data buses, flattened. Exact mirror of
    // v_rd_bus_flat above (rename + concat of the existing internal
    // k_rd_bus[] array — no new logic, no muxing). Needed because the
    // 6-way head-parallel score path consumes all six K rows in the same
    // cycle, whereas the scalar k_rd_vec is a 6:1 mux of them.
    // PURELY ADDITIVE — k_rd_vec is untouched.
    output wire [N_STACKS*ROW_W-1:0] k_rd_bus_flat
);

    // head -> (stack, head_in_stack)
    function automatic int stack_of(input int h);  return h / HEADS_PER_STACK; endfunction
    function automatic int slot_of (input int h);  return h % HEADS_PER_STACK; endfunction

    // Per-stack read-data buses
    wire [ROW_W-1:0] k_rd_bus [0:N_STACKS-1];
    wire [ROW_W-1:0] v_rd_bus [0:N_STACKS-1];

    // Registered selects (read latency is 1 cycle in the stack; we mux the
    // bus combinationally after the registered stack output, so we register
    // the read-head select to align with the data).
    reg [HEAD_W-1:0] k_rd_head_q, v_rd_head_q;
    always_ff @(posedge clk) begin
        k_rd_head_q <= k_rd_head;
        v_rd_head_q <= v_rd_head;
    end

    // RTL WO#12 elaboration guards: a >1 group count without the caller-driven
    // replica buses (POS_REP/WR_POS_REP) would address the split slices from
    // an undriven bus — refuse to elaborate instead.
    generate
    if (ADDR_REP_K > 1 && (POS_REP == 0 || WR_POS_REP == 0)) begin : g_cfg_err_k
        kv_addr_rep_requires_pos_and_wrpos_rep u_cfg_err();
    end
    if (ADDR_REP_V > 1 && (POS_REP == 0 || WR_POS_REP == 0)) begin : g_cfg_err_v
        kv_addr_rep_requires_pos_and_wrpos_rep u_cfg_err();
    end
    endgenerate

    genvar s;
    generate
        for (s = 0; s < N_STACKS; s = s + 1) begin : g_stack
            // KV_REPL: per-stack read position (replicated copy when POS_REP!=0,
            // else the legacy shared scalar). Same value either way — bit-exact.
            // RTL WO#12: group slice 0 of the stack doubles as the scalar
            // (every slice carries the same value each cycle).
            wire [10:0] k_rd_pos_s = (POS_REP != 0) ? k_rd_pos_rep[s*ADDR_REP_K*11 +: 11] : k_rd_pos;
            wire [10:0] v_rd_pos_s = (POS_REP != 0) ? v_rd_pos_rep[s*ADDR_REP_V*11 +: 11] : v_rd_pos;
            // W3-C1-P1: per-stack write position (replicated copy when
            // WR_POS_REP!=0, else the legacy shared scalar). Same value
            // either way — bit-exact; the win is purely physical route.
            wire [10:0] k_wr_pos_s = (WR_POS_REP != 0) ? k_wr_pos_rep[s*ADDR_REP_K*11 +: 11] : k_wr_pos;
            wire [10:0] v_wr_pos_s = (WR_POS_REP != 0) ? v_wr_pos_rep[s*ADDR_REP_V*11 +: 11] : v_wr_pos;

            // K stack
            wire                k_wr_hit = k_wr_en && (stack_of(k_wr_head) == s);
            // G5MEM: RD_ALL=1 ungates every stack (see the RD_ALL header note).
            wire                k_rd_hit = k_rd_en && ((RD_ALL != 0) || (stack_of(k_rd_head) == s));
            wire [HSEL_W-1:0]   k_wr_slot = HSEL_W'(slot_of(k_wr_head));
            wire [HSEL_W-1:0]   k_rd_slot = HSEL_W'(slot_of(k_rd_head));

            kv_head_wide_win #(
                .D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH),
                .CTX_WIN(CTX_WIN), .HEADS_PER_STACK(HEADS_PER_STACK),
                .RAM_STYLE(K_RAM_STYLE),
                .ADDR_REP(ADDR_REP_K)                 // RTL WO#12: K bank groups
            ) u_k (
                .clk(clk), .rst_n(rst_n),
                .wr_en(k_wr_hit), .wr_head(k_wr_slot), .wr_pos(k_wr_pos_s), .wr_vec(k_wr_vec),
                .rd_en(k_rd_hit), .rd_head(k_rd_slot), .rd_pos(k_rd_pos_s), .rd_vec(k_rd_bus[s]),
                .rd_pos_rep(k_rd_pos_rep[s*ADDR_REP_K*11 +: ADDR_REP_K*11]),
                .wr_pos_rep(k_wr_pos_rep[s*ADDR_REP_K*11 +: ADDR_REP_K*11])
            );

            // V stack
            wire                v_wr_hit = v_wr_en && (stack_of(v_wr_head) == s);
            wire                v_rd_hit = v_rd_en && ((RD_ALL != 0) || (stack_of(v_rd_head) == s));
            wire [HSEL_W-1:0]   v_wr_slot = HSEL_W'(slot_of(v_wr_head));
            wire [HSEL_W-1:0]   v_rd_slot = HSEL_W'(slot_of(v_rd_head));

            kv_head_wide_win #(
                .D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH),
                .CTX_WIN(CTX_WIN), .HEADS_PER_STACK(HEADS_PER_STACK),
                .RAM_STYLE(V_RAM_STYLE),
                .ADDR_REP(ADDR_REP_V)                 // RTL WO#12: V bank groups
            ) u_v (
                .clk(clk), .rst_n(rst_n),
                .wr_en(v_wr_hit), .wr_head(v_wr_slot), .wr_pos(v_wr_pos_s), .wr_vec(v_wr_vec),
                .rd_en(v_rd_hit), .rd_head(v_rd_slot), .rd_pos(v_rd_pos_s), .rd_vec(v_rd_bus[s]),
                .rd_pos_rep(v_rd_pos_rep[s*ADDR_REP_V*11 +: ADDR_REP_V*11]),
                .wr_pos_rep(v_wr_pos_rep[s*ADDR_REP_V*11 +: ADDR_REP_V*11])
            );
        end
    endgenerate

    // WO-051: expose the per-stack V buses (flattened). Each stack's
    // v_rd_bus[s] is the registered output of its own kv_head_wide_win, so
    // this is just a rename/concat — no new logic, no muxing. The per-bank
    // combine picks the active stack downstream (gated by stack_sel).
    genvar gs;
    generate
        for (gs = 0; gs < N_STACKS; gs = gs + 1) begin : g_vflat
            assign v_rd_bus_flat[gs*ROW_W +: ROW_W] = v_rd_bus[gs];
        end
    endgenerate

    // G5MEM: same rename/concat for the K side (mirror of g_vflat).
    genvar gk;
    generate
        for (gk = 0; gk < N_STACKS; gk = gk + 1) begin : g_kflat
            assign k_rd_bus_flat[gk*ROW_W +: ROW_W] = k_rd_bus[gk];
        end
    endgenerate

    // Output mux: select the stack that holds the (registered) read head.
    integer i;
    always @(*) begin
        k_rd_vec = '0;
        v_rd_vec = '0;
        for (i = 0; i < N_STACKS; i = i + 1) begin
            if (stack_of(k_rd_head_q) == i) k_rd_vec = k_rd_bus[i];
            if (stack_of(v_rd_head_q) == i) v_rd_vec = v_rd_bus[i];
        end
    end

endmodule

`default_nettype wire
