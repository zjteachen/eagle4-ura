// =====================================================================
// attn_v_combine_perbank.sv   (WO-051 Layer-1 — per-KV-stack local combine)
// =====================================================================
//
// ROOT-CAUSE SLR-CROSSING FIX for the −0.780 ns worst path.
//
// THE PROBLEM (current top):
//   ONE shared attn_v_combine reads the wide V-row (D_HEAD*A_WIDTH =
//   128*16 = 2048-bit) that comes out of multihead_kv. multihead_kv
//   time-muxes ALL N_STACKS=6 KV stacks through one read port, so for
//   SOME head the URAM that sources V[pos] is in a DIFFERENT SLR than the
//   single combine MAC. The 2048-bit raw V-row then crosses the SLR
//   boundary COMBINATIONALLY (unregistered SLL) straight into a 13-deep
//   DSP MAC -> −0.780 ns.
//
// THE FIX (this module):
//   Instantiate ONE attn_v_combine PER KV stack (N_STACKS instances).
//   Each combine is fed ONLY by its own stack's V read-data bus
//   (v_rd_bus[s]) and is PBLOCK-co-located with that stack's URAM in the
//   SAME SLR (see eagle4_drafter_perbank_pblock.xdc). The wide 2048-bit
//   V-row therefore NEVER leaves the stack's SLR.
//
//   Only ONE stack is active at a time (the time-mux walks heads
//   sequentially; head_sel -> stack_of(head_sel) selects exactly one
//   stack). So combine_valid is gated per stack by `stack_hit[s]`:
//   exactly the active stack accumulates; the other five hold.
//
//   The N_STACKS D_HEAD-wide INT32 partials are REGISTERED, then reduced
//   by a registered select-mux (the active stack's partial). Only this
//   reduced D_HEAD*32 = 4096-bit REGISTERED scalar bus crosses SLRs — and
//   it crosses from a register output to a register input (a clean SLL
//   timing arc), not from a URAM through a combinational MAC.
//
// BIT-EXACTNESS (the HARD requirement, memory feedback-no-accuracy-tradeoff):
//   Because exactly ONE stack accumulates per head (the others are gated
//   off), the active stack's local attn_v_combine performs the IDENTICAL
//   INT32 accumulation, in the IDENTICAL order, over the IDENTICAL
//   attn_q12 * V[d] >> 12 products, as the current single shared combine.
//   The reduction is a pure SELECT (not a sum) of the one active stack's
//   already-final partial. There is no re-association, no extra rounding,
//   no width change -> the result is BIT-IDENTICAL by construction (it is
//   literally the same arithmetic, relocated + registered).
//
//   NB: this is STRONGER than "associativity makes it equal" — there is
//   no summation across stacks at all for a given head, so even the
//   associativity argument is unnecessary. (A future GQA / multi-stack-
//   per-head variant WOULD lean on INT32-add associativity; this MHA
//   time-mux variant does not.)
//
// LATENCY CONTRACT (memory feedback-valid-data-latency-skew-class):
//   The MAC commit cycle is UNCHANGED from attn_v_combine — acc updates
//   on valid_in. The ONLY added latency is +1 cycle for the registered
//   partial + registered reduction mux. The caller (attention_head_
//   assembly) reads out_vec ONE cycle later than before; the FSM adds one
//   drain cycle (S_DRAIN_COMB -> S_DRAIN_COMB2) so `done` still co-aligns
//   with a settled out_vec. valid/data co-alignment is preserved: the
//   reduced partial is registered from acc[] which is registered from the
//   MAC — out_vec_reg lands exactly one cycle after the last acc update.
//
// Multi-driver-safe (memory feedback-multi-driver-packed-reg-trap):
//   out_vec is driven from a single unpacked INT32 array (out_acc_arr)
//   via one combinational concat. Each stack's acc lives in its own
//   attn_v_combine instance (distinct driver). No packed-reg slice
//   sharing across genvar iterations.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module attn_v_combine_perbank #(
    parameter int D_HEAD   = 128,
    parameter int A_WIDTH  = 16,
    parameter int N_STACKS = 6,
    parameter int SSEL_W   = (N_STACKS <= 1) ? 1 : $clog2(N_STACKS)
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              clear,
    input  wire                              valid_in,         // active-stack MAC strobe
    // B2' (ACCFIX bit1): round-half pass-through to every per-stack combine.
    input  wire                              round_en,
    input  wire        [SSEL_W-1:0]          stack_sel,        // stack_of(head_sel) — REGISTER-ALIGNED to v_rd_bus
    input  wire signed [A_WIDTH-1:0]         attn_q12,         // Q4.12 weight (same for all stacks)
    // Per-stack wide V read-data buses (each co-located with its stack URAM).
    // Flattened: v_rd_bus_flat[s*ROW_W +: ROW_W] is stack s's V[pos] row.
    input  wire signed [N_STACKS*A_WIDTH*D_HEAD-1:0] v_rd_bus_flat,
    // Reduced, REGISTERED D_HEAD INT32 partial of the active stack.
    output wire signed [32*D_HEAD-1:0]       out_vec
);

    localparam int ROW_W = A_WIDTH * D_HEAD;

    // -----------------------------------------------------------------
    // Per-stack local combine. valid gated so ONLY the active stack
    // accumulates this head. The non-selected stacks see valid=0 -> hold.
    // `clear` broadcasts to all stacks at head start (acc[]=0).
    // -----------------------------------------------------------------
    wire signed [32*D_HEAD-1:0] stack_out [0:N_STACKS-1];

    genvar s;
    generate
        for (s = 0; s < N_STACKS; s = s + 1) begin : g_bank
            wire signed [ROW_W-1:0] v_bank = v_rd_bus_flat[s*ROW_W +: ROW_W];
            wire                    bank_valid = valid_in && (stack_sel == SSEL_W'(s));

            attn_v_combine #(.D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH)) u_combine (
                .clk      (clk),
                .rst_n    (rst_n),
                .clear    (clear),
                .valid_in (bank_valid),
                .round_en (round_en),          // B2'
                .attn_q12 (attn_q12),
                .v_vec    (v_bank),
                .out_vec  (stack_out[s])
            );
        end
    endgenerate

    // -----------------------------------------------------------------
    // Registered select-mux reduction. Only the active stack's partial is
    // selected (exactly one stack accumulated this head). This register
    // is the ONLY net that crosses the SLR boundary (register->register).
    //
    // stack_sel is held by the caller across the whole combine pass for a
    // head (it derives from head_sel, which does not change mid-head), so
    // registering the select introduces no skew vs the data.
    // -----------------------------------------------------------------
    logic signed [31:0] out_acc_arr [0:D_HEAD-1];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int d = 0; d < D_HEAD; d++) out_acc_arr[d] <= '0;
        end else begin
            // select the active stack's D_HEAD INT32 partial
            for (int d = 0; d < D_HEAD; d++)
                out_acc_arr[d] <= stack_out[stack_sel][d*32 +: 32];
        end
    end

    // single combinational concat to the output (multi-driver-safe)
    genvar d;
    generate
        for (d = 0; d < D_HEAD; d = d + 1) begin : g_pack
            assign out_vec[d*32 +: 32] = out_acc_arr[d];
        end
    endgenerate

endmodule

`default_nettype wire
