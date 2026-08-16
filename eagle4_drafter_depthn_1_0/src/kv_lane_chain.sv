// =====================================================================
// kv_lane_chain.sv -- SGLang stock lane-chain KV address mapper
// =====================================================================
//
// SINGLE SOURCE OF TRUTH for the tree-draft KV slot rule. Both the WRITE
// side (eagle4_drafter.sv pl_pos / attention_full_mha_sbuf T_APPEND) and
// the READ side (attention_head_assembly_perbank_sbuf) must call into
// this package. A second copy of the formula anywhere else is the
// "dual-source table" bug class -- do not create one.
//
// SEMANTICS (SGLang spec_utils.py:290, verbatim):
//     token_pool_ptr + seq_len + topk_id * num_steps + tl.arange(0, iter_upper)
// =>  tree_slot(j,i) = ctx_len_prefix + j*num_steps + i
//         j in [0, K_F)          branch / topk lane
//         i in [0, max_depth)    level within the lane
//
// Branch j at level i reads   prefix [0, ctx_len_prefix)
//                       PLUS  its own column
//                             [ctx_len_prefix + j*num_steps,
//                              ctx_len_prefix + j*num_steps + i)
// and NEVER another branch's column.
//
// FIXED STRIDE, NO data-dependent indirection. This is deliberately NOT
// exact per-branch ancestry -- see reports/results/EXACT_ANCESTRY_AB.md
// (exact ancestry measured at -0.0018/-0.0010/-0.0029 accept @ depth
// 3/5/8, every 95% CI straddling zero; the GPU's own tail IS the slot
// chain in 192/192 captures). The lane-chain IS the golden. Do NOT add
// write-position-rewrite, journal replay, or a parent-indexed lookup --
// those BREAK the bit-match.
//
// DEPTH-0 REDUCTION (mandatory, by construction):
//   lane_j=0, level_i=0  =>  sweep_len = ctx_len_prefix + 0 = ctx_len_prefix
//   pos_req in [0, ctx_len_prefix) is ALWAYS < ctx_len_prefix, so
//   kv_lane_eff_pos() returns the untouched pos_req via the true-arm of
//   the ternary for EVERY pos_req, for ANY num_steps. The remapped path
//   is bit-identical to the dense sweep.
//
// ADDRESS BOUND: kv_head_wide_win.sv:117-126 forms
//   addr = (head%HEADS_PER_STACK)*CTX_WIN + rd_pos[POS_W-1:0]
// with POS_W = $clog2(CTX_WIN) = 10 for CTX_WIN=1024. rd_pos is an
// 11-bit port that is SILENTLY TRUNCATED to 10 bits. Any eff_pos >=
// CTX_WIN aliases into another head's window. Hence the hard bound
// check below.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

package kv_lane_chain_pkg;

    // Internal width is deliberately wider (12b) than the 11b port so an
    // out-of-range slot is VISIBLE to the bound check instead of already
    // having wrapped.
    localparam int KVLC_W = 12;

    // -----------------------------------------------------------------
    // WRITE slot: where branch j's level-i token is STORED.
    // -----------------------------------------------------------------
    function automatic [KVLC_W-1:0] kv_lane_slot
    (
        input [KVLC_W-1:0] ctx_len_prefix,
        input [3:0]        lane_j,
        input [KVLC_W-1:0] num_steps,
        input [KVLC_W-1:0] level_i
    );
        reg [KVLC_W-1:0] stride;
        begin
`ifdef KVLANE_RED_STRIDE
            // RED arm ONLY: column mis-strided by one. Must be detected by
            // the isolation check, never shipped.
            stride = KVLC_W'({{(KVLC_W-4){1'b0}}, lane_j} * (num_steps + KVLC_W'(1)));
`else
            stride = KVLC_W'({{(KVLC_W-4){1'b0}}, lane_j} * num_steps);
`endif
            kv_lane_slot = ctx_len_prefix + stride + level_i;
        end
    endfunction

    // -----------------------------------------------------------------
    // READ remap of the dense sweep counter.
    //   pos_req <  ctx_len_prefix -> shared prefix, untouched
    //   pos_req >= ctx_len_prefix -> own column, offset (pos_req - prefix)
    //
    // IDENTITY (the reason there is one source of truth):
    //   kv_lane_eff_pos(ctx, j, ns, ctx + i) == kv_lane_slot(ctx, j, ns, i)
    // so a level-i token is read back from exactly where it was written.
    // -----------------------------------------------------------------
    function automatic [KVLC_W-1:0] kv_lane_eff_pos
    (
        input [KVLC_W-1:0] ctx_len_prefix,
        input [3:0]        lane_j,
        input [KVLC_W-1:0] num_steps,
        input [KVLC_W-1:0] pos_req
    );
        begin
            kv_lane_eff_pos = (pos_req < ctx_len_prefix)
                ? pos_req
                : kv_lane_slot(ctx_len_prefix, lane_j, num_steps,
                               pos_req - ctx_len_prefix);
        end
    endfunction

    // -----------------------------------------------------------------
    // Effective sweep length for a level-i read: prefix + i own-column
    // entries (levels 0..i-1 of this lane are already resident).
    // level_i = 0 -> exactly ctx_len_prefix -> depth-0 reduction.
    // -----------------------------------------------------------------
    function automatic [KVLC_W-1:0] kv_lane_sweep_len
    (
        input [KVLC_W-1:0] ctx_len_prefix,
        input [KVLC_W-1:0] level_i
    );
        begin
            kv_lane_sweep_len = ctx_len_prefix + level_i;
        end
    endfunction

    // -----------------------------------------------------------------
    // RoPE position for a body call. The root consumes the current token
    // at prefix-1. A speculative child is one position beyond the tokens
    // already visible in its dense logical sweep, so its angle is
    // prefix+level (equal to the child sweep length).
    // -----------------------------------------------------------------
    function automatic [KVLC_W-1:0] kv_lane_rope_pos
    (
        input [KVLC_W-1:0] ctx_len_prefix,
        input [KVLC_W-1:0] level_i
    );
        begin
            if (level_i == '0)
                kv_lane_rope_pos = ctx_len_prefix - 1'b1;
`ifdef KVLANE_RED_ROPE_EARLY
            else
                kv_lane_rope_pos = ctx_len_prefix + level_i - 1'b1;
`else
            else
                kv_lane_rope_pos = ctx_len_prefix + level_i;
`endif
        end
    endfunction

    // -----------------------------------------------------------------
    // Bound predicate: 1 = safe for the 10-bit-truncated KV RAM address.
    // -----------------------------------------------------------------
    function automatic kv_lane_in_bounds
    (
        input [KVLC_W-1:0] eff_pos,
        input int          ctx_win
    );
        begin
            kv_lane_in_bounds = (32'(eff_pos) < ctx_win);
        end
    endfunction

endpackage : kv_lane_chain_pkg


// =====================================================================
// Thin combinational wrapper. Pure -- no state, no clock dependence on
// the mapping itself. `clk`/`rst_n` are present only for the bound
// assertion (simulation-only).
// =====================================================================
(* keep_hierarchy = "yes" *)
module kv_lane_chain #(
    parameter int K_F     = 8,
    parameter int CTX_WIN = 1024
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [11:0] ctx_len_prefix,
    input  wire [3:0]  lane_j,
    input  wire [11:0] num_steps,
    input  wire [11:0] level_i,
    input  wire [11:0] pos_req,

    output wire [11:0] eff_pos,
    output wire [11:0] sweep_len,
    output wire [11:0] write_slot,
    output wire        in_bounds
);
    import kv_lane_chain_pkg::*;

    assign eff_pos    = kv_lane_eff_pos (ctx_len_prefix, lane_j, num_steps, pos_req);
    assign sweep_len  = kv_lane_sweep_len(ctx_len_prefix, level_i);
    assign write_slot = kv_lane_slot    (ctx_len_prefix, lane_j, num_steps, level_i);
    assign in_bounds  = kv_lane_in_bounds(eff_pos, CTX_WIN);

    // synthesis translate_off
    always @(posedge clk) begin
        if (rst_n) begin
            if (!in_bounds)
                $fatal(1, "kv_lane_chain: eff_pos %0d >= CTX_WIN %0d (would truncate to %0d)",
                       eff_pos, CTX_WIN, eff_pos % CTX_WIN);
            if (32'(lane_j) >= K_F)
                $fatal(1, "kv_lane_chain: lane_j %0d >= K_F %0d", lane_j, K_F);
        end
    end
    // synthesis translate_on

endmodule

`default_nettype wire
