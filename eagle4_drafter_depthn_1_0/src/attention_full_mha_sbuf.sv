// =====================================================================
// attention_full_mha.sv  (ATTN-FULL — the de-piloted 24-head attention)
// =====================================================================
//
// FULL Llama-3.2-3B EAGLE4 attention: N_HEADS=24 query heads, D_HEAD=128,
// MHA (n_kv == n_q, h_kv = h — NO GQA; reference/src/attn.cpp:13,71).
// Replaces the head-0 / D_HEAD=16 / NCHUNKS=1 pilot in the top.
//
// ARCHITECTURE — TIME-MULTIPLEX (the #176 / attn_full verdict)
// -------------------------------------------------------------
// ONE shared D_HEAD=128 attention_head_assembly (M2) + ONE multihead_kv
// (M3) walk heads 0..N_HEADS-1 sequentially. Chosen over a 24x spatial
// replicate because:
//   - DSP stays at the single-head cost (~146 DSP) instead of ~3,504 for
//     a full spatial replicate (measured in forks/attn_full ooc).
//   - M3 already packs all 24 heads into ONE URAM store with a head select;
//     M2's KV read port is external -> time-mux needs ZERO change to either
//     verified primitive (instantiate + wire only, per the MANDATE).
//   - Attention is NOT the forward-path bottleneck (the six 2-bit GEMV
//     stages dominate); serializing 24 heads is cheap latency, big area win.
//
// DATAPATH per the eagle4_ref contract (attn.cpp):
//   1. RoPE(Q) and RoPE(K) per head: rotate pair (j, j+d/2) by per-position
//      theta_j, base ROPE_THETA=500000 (rope_layer + /tmp/p4/rope_cs.hex).
//      cos/sin are SHARED across heads for a given position.
//   2. KV append: rotated K and raw V written into multihead_kv at the
//      current position. (In this block the KV store is filled by the
//      caller / a preload FSM; this module READS K/V for the score+combine
//      and, when DO_KV_FILL=1, also writes the freshly-RoPE'd K + V for the
//      current append position.)
//   3. Per head: 3-pass attention (attention_head_assembly):
//        scores = Q . K[t] (INT20-trunc per elem, csa_tree16 -> INT32 acc)
//        softmax: max-subtract, exp_pwl_q12, inv_q12 reciprocal
//        out[h,d] = sum_t attn[t] * V[t,h,d]   (Q-format combine)
//   4. Output: out_vec[h*D_HEAD + d] packed into a N_HEADS*D_HEAD INT16
//      vector (g5 attention output, RAW INT16 in the g5 Q-format).
//
// NUMERICAL CONTRACT (honest, see ATTN_FULL_RESULT.md):
//   - At ctx_len=1 (the depth-0 seam the debug_suite g5 golden captures),
//     softmax over one position = 1.0, so out[h,d] = V[0,h,d] EXACTLY in
//     the input Q-format. This block is BIT-EXACT to g5_attn_out.hex.
//   - At ctx_len>1 the INT16 softmax (exp_pwl LUT + 24-cycle integer
//     reciprocal) is an APPROXIMATION of the fp32 eagle4_ref softmax; the
//     two agree within a documented Q-format tolerance, not bit-exact.
//     (The fp32 1/sqrt(d) score scale is folded into the per-element
//     Q-format; the INT softmax max-subtract makes the absolute score
//     scale irrelevant up to LUT granularity.)
//
// Multi-driver-safe: out_vec is a single combinational concat of an
// unpacked head-output array (each head writes its own distinct slot).
// No valid-high-in-drain (the per-head FSM lives inside the unchanged M2).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module attention_full_mha_sbuf #(
    parameter int N_HEADS        = 24,
    parameter int D_HEAD         = 128,
    parameter int A_WIDTH        = 16,
    parameter int CTX_WIN        = 1024,
    parameter int HEADS_PER_STACK= 4,
    parameter int CTX_MAX        = 1024,        // attn_head_assembly exp_buf depth
    parameter int SCORE_LATENCY  = 13,          // D_HEAD=128 chunk-serial
    // Score normalization for the softmax seam (see attention_head_assembly).
    // 0 = legacy hardmax (correct only at ctx_len=1). ~11 = soft softmax that
    // matches the fp32 reference at ctx_len>1. ctx_len=1 is bit-exact either way.
    parameter int SCORE_NORM_SHR = 0,
    // WO-053: KV storage-primitive selection (threaded to multihead_kv).
    //   K_RAM_STYLE/V_RAM_STYLE ∈ {"ultra"(URAM), "block"(BRAM)}.
    // Default V→BRAM shifts the V stacks off URAM so they co-locate with
    // the attn_v_combine MAC in one SLR (kills the KV-read→combine SLR
    // crossing). PURE attribute → bit-identical (feedback_no_accuracy_tradeoff).
    parameter         K_RAM_STYLE    = "ultra",
    parameter         V_RAM_STYLE    = "block",
    // -----------------------------------------------------------------
    // WO-051: per-KV-stack LOCAL attn_v_combine select.
    //   (DEFAULT 2026-06-02 WO-051-PROPAGATION-FIX flipped to 1 to match the
    //    top eagle4_drafter default; the top always passes .COMBINE_LOCAL()
    //    explicitly, so this default only guards against a future bare
    //    instantiation silently reverting to the shared combine.)
    //   COMBINE_LOCAL=0 : the shared attention_head_assembly
    //       (ONE attn_v_combine reads the muxed wide V-row). BIT-IDENTICAL
    //       to the current golden top; the −0.780 SLR crossing path exists.
    //   COMBINE_LOCAL=1 (DEFAULT) : attention_head_assembly_perbank (ONE
    //       attn_v_combine PER KV stack, co-located with the stack URAM).
    //       The wide 2048-bit V-row never crosses the SLR; only the
    //       registered D_HEAD*32 reduced partial does (reg->reg). BIT-EXACT
    //       to COMBINE_LOCAL=0 (same arithmetic, relocated + registered;
    //       proven by tb_attn_head_assembly_perbank + the through-top
    //       golden xsim). +1 publish-cycle latency, absorbed by the
    //       assembly's extra drain state (done still co-aligns out_vec).
    // -----------------------------------------------------------------
    parameter bit COMBINE_LOCAL    = 1'b1,
    // RAW_REQUANT=0 consumes pre-requantized packed INT16 Q/K/V vectors.
    // RAW_REQUANT=1 selects a head from the raw INT32 buffers, then applies
    // the G4_FRAC->G5_FRAC round/shift/clamp conversion to that head only.
    // The top-level REQUANT_HEADMUX parameter selects this path and omits
    // the parallel full-vector g4_qkv_seam conversion.
    parameter bit RAW_REQUANT      = 1'b0,
    // HEAD_PAR — G5 head-parallelism (ported from speedwave_v2_g5mem, Stage-A).
    // 1 = SERIAL (DEFAULT): ONE assembly time-muxes all N_HEADS (the ps_tree_fsm
    //     gate-2-4-proven config, byte-identical). N = HEAD_PAR lockstep engines
    //     (engine e -> KV stack e), N_GRP = N_HEADS/HEAD_PAR groups. STEP-2 of the
    //     Stage-A merge ports the g_headpar generate + N_GRP FSM + k_rd_bus_flat +
    //     the per-engine depth-N lane-chain wiring; until then HEAD_PAR>1 is a LOUD
    //     elaboration error (g_headpar_todo below) so no half-ported strap can ship.
    parameter int HEAD_PAR         = 1,
    parameter int G4_FRAC          = 12,   // raw tile Q-format (C1-REVERT contract)
    parameter int G5_FRAC          = 10,   // attention-input Q-format
    // ROPE_POS_DECOUPLE: 0 = rope angle IS append_pos (shipped, bit-identical);
    // 1 = rope angle comes from rope_pos_in (depth-N KV lane-chain). See the
    // rope_pos_in port comment for why conflating them corrupts depth>=1.
    parameter bit ROPE_POS_DECOUPLE = 1'b0,
    // -----------------------------------------------------------------
    // F3 (TREE_FSM_DEPTHN_BODY_REKEY_RCA defect c): KV lane-chain READ
    // remap enable, threaded to attention_head_assembly_perbank_sbuf.
    //
    // DEFAULT IS TIED TO ROPE_POS_DECOUPLE ON PURPOSE (no separate strap):
    // the eagle4_drafter instantiation already passes
    // .ROPE_POS_DECOUPLE(DEPTH_N), and ROPE_POS_DECOUPLE *is* the write
    // side of the very same lane-chain semantics (slot/angle split). The
    // WRITE side (append_pos = lane slot) is unconditionally lane-chained
    // under DEPTH_N at level>=1 — a READ side that could be strapped off
    // independently would recreate the exact half-wired write/read split
    // this fix closes (KV_LANE_CHAIN declared, never overridden, RCA §4).
    // So: DEPTH_N=0 => 0 (g_lane not elaborated, byte-identical legacy),
    // DEPTH_N=1 => 1, with no new eagle4_config.svh toggle to forget.
    // Still overridable for unit gates / A-B twins.
    parameter int KV_LANE_CHAIN     = (ROPE_POS_DECOUPLE ? 1 : 0),
    parameter int K_F               = 8,
    // -----------------------------------------------------------------
    // RTL WO#8 (run-11): rope rotation-cone pipeline + pos_d1 locality
    // replicas — threaded to u_rope (the single time-muxed rope; ONE
    // instance serves all HEAD_PAR engines, so ONE pipe fix covers all).
    // RTL WO#12 (run-12): V-KV bank-group address replication — threaded
    // to u_kv (ADDR_REP_V), the assembly (v_rd_pos_rep group twins), and
    // the write-position twins (v_wr_pos_q). Composes with HEAD_PAR: the
    // V store is the SHARED 6-stack multihead_kv split 8-ways per stack
    // (48 xpm total, storage-neutral), engines read per-stack buses.
    // -----------------------------------------------------------------
    parameter bit ROPE_CS_PIPE      = 1'b0,
    parameter int ROPE_POS_REP      = 1,
    parameter int KV_ADDR_REP       = 1,
    // F3-CLR: kv_res_clear fan-out replication inside each engine's
    // residency bitmap (see attention_head_assembly_perbank_sbuf).
    parameter int RES_CLR_REP       = 1,
    parameter int ROW_W          = D_HEAD * A_WIDTH,
    parameter int OUT_W          = N_HEADS * D_HEAD * A_WIDTH,
    parameter int HEAD_W         = (N_HEADS <= 1) ? 1 : $clog2(N_HEADS),
    parameter int KVHEAD_W       = (N_HEADS <= 1) ? 1 : $clog2(N_HEADS)
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // ----- control -----
    input  wire                       start,            // 1-cycle pulse
    input  wire [10:0]                ctx_len,          // # cached positions (>=1)
    input  wire [10:0]                append_pos,       // position to write the new K/V at
    // =====================================================================
    // ROPE_POS_DECOUPLE — split the ROTATION ANGLE from the STORAGE SLOT.
    //
    // `append_pos` historically served BOTH roles, which is correct only while
    // they coincide (depth 0). Under the depth-N KV lane-chain they diverge:
    //   storage slot  = ctx_len + j*num_steps + i   (branch j's own column)
    //   rotation angle= ctx_len + i                 (the TREE DEPTH, level-uniform)
    // With them conflated, branch j=7 at level 0 with num_steps=8 would be
    // rotated at angle ctx_len+56 instead of ctx_len. That is SILENT and it
    // corrupts every depth>=1 token, so the two are separated here.
    //
    // rope_layer is instantiated MAX_CTX(2048), so both values are
    // representable — this is a semantics fix, not a capacity change.
    // ROPE_POS_DECOUPLE=0 (DEFAULT) keeps rope_pos == append_pos, i.e.
    // bit-identical to the shipped block.
    // =====================================================================
    input  wire [10:0]                rope_pos_in,      // angle when ROPE_POS_DECOUPLE=1
    // ----- F3 KV lane-chain READ remap inputs (consumed ONLY when
    // KV_LANE_CHAIN=1; a legacy caller leaves them unconnected — the
    // KV_LANE_CHAIN=0 generate arm ties the assembly's lane ports to '0
    // so nothing floats into live logic). Semantics (kv_lane_chain_pkg):
    //   ctx_len_prefix = SHARED prefix length (drafter s7a_ctx_len, >=1)
    //   lane_j         = branch / topk id      (drafter branch_j_w)
    //   num_steps      = SGLang num_steps      (drafter max_depth[10:0])
    //   level_i        = level within the lane (drafter level_i_w; 0 = root
    //                    pass => sweep = prefix, remap = identity, so the
    //                    depth-N level-0 pass is bit-identical to depth-0)
    // INVARIANT (asserted, sim-only): the legacy ctx_len port must carry
    // the SWEEP = ctx_len_prefix + level_i (= drafter s7a_ctx_len_eff).
    input  wire [10:0]                ctx_len_prefix,
    input  wire [3:0]                 lane_j,
    input  wire [10:0]                num_steps,
    input  wire [10:0]                level_i,
    output logic                      done,             // 1-cycle pulse when all heads done

    // ----- per-head Q / K / V for the CURRENT (append) token -----
    // Flattened N_HEADS * D_HEAD * A_WIDTH. q/k are PRE-RoPE here; this block
    // applies RoPE. v is raw (no RoPE on V).
    input  wire signed [OUT_W-1:0]    q_in,             // [N_HEADS][D_HEAD]
    input  wire signed [OUT_W-1:0]    k_in,
    input  wire signed [OUT_W-1:0]    v_in,

    // Per-head raw INT32 Q/K/V buffers, used only with RAW_REQUANT=1.
    input  wire signed [32*D_HEAD-1:0] q_raw_buf [0:N_HEADS-1],
    input  wire signed [32*D_HEAD-1:0] k_raw_buf [0:N_HEADS-1],
    input  wire signed [32*D_HEAD-1:0] v_raw_buf [0:N_HEADS-1],

    // ----- KV preload port (TB / fill FSM) — writes into multihead_kv -----
    // Lets the caller stage a multi-position context (rotated K + raw V) at
    // arbitrary positions before `start`. When the block runs DO_KV_FILL it
    // also self-appends the current token at append_pos.
    input  wire                       pl_we,
    input  wire [KVHEAD_W-1:0]        pl_head,
    input  wire [10:0]                pl_pos,
    input  wire signed [ROW_W-1:0]    pl_k_vec,         // ROTATED K
    input  wire signed [ROW_W-1:0]    pl_v_vec,         // raw V
    input  wire                       do_kv_fill,       // 1 = self-append cur token

    // ----- result -----
    output logic signed [OUT_W-1:0]   out_vec,          // [N_HEADS][D_HEAD] g5 attn-out
    output logic [HEAD_W-1:0]         dbg_head,         // head currently processed

    // ----- F3 KV-RESIDENCY status (ADDITIVE, read-only) -----------------
    // # of DISTINCT absolute KV slots written since reset. This is the
    // quantity the host-side ctx_len register (reg 0x10) has NO coupling to
    // and whose absence produced the ctx>=2 "attention against zero padding"
    // artifact (reports/results/ATTN_F3_KV_RESIDENCY_MASK.md). Purely
    // observational -- nothing downstream consumes it, so a legacy caller may
    // leave it unconnected.
    output wire  [11:0]               kv_res_cnt,

    // ----- F3-CLR HOST-WRITABLE RESIDENCY CLEAR (reg 0x154 bit[0]) ------
    // 1-cycle pulse. Broadcast IDENTICALLY to every HEAD_PAR engine so the
    // per-engine mirror bitmaps stay in lockstep (the same discipline as
    // the kv_wr_v/kv_wr_slot write mirror). Marks the end of one SGLang
    // SEQUENCE: after the pulse no slot is resident until it is rewritten,
    // so a new request can never attend the previous request's KV rows.
    // Tie 1'b0 in a legacy/unit caller -> byte-identical to the pre-fix
    // revision. FULL RCA: attention_head_assembly_perbank_sbuf.sv port
    // comment + reports/results/KV_RESIDENCY_CLEAR_FIX.md
    input  wire                       kv_res_clear,
    // B2' (ACCFIX bit1): round-half on the two softmax->V combine floors,
    // broadcast identically to every perbank_sbuf engine. 0 = byte-identical.
    input  wire                       b2p_round_en
);

    localparam int N_PAIRS = D_HEAD / 2;

    // G5MEM: head-parallel geometry. N_GRP = head-loop trip count.
    //   HEAD_PAR=1 -> N_GRP=N_HEADS, head_of(0,g)=g   (the shipped serial walk)
    //   HEAD_PAR=6 -> N_GRP=4,       head_of(e,g)=e*4+g == stack e, slot g
    // HEAD_PAR must divide N_HEADS and equal N_STACKS (engine e reads stack e).
    localparam int N_GRP  = N_HEADS / HEAD_PAR;
    localparam int GRP_W  = (N_GRP    <= 1) ? 1 : $clog2(N_GRP);
    localparam int ESEL_W = (HEAD_PAR <= 1) ? 1 : $clog2(HEAD_PAR);

    // -----------------------------------------------------------------
    // Per-head Q/K/V slices (combinational from the flattened inputs)
    // -----------------------------------------------------------------
    // G5MEM: head_sel is DERIVED from (eng_sel, grp_sel), not its own counter.
    // At HEAD_PAR=1 eng_sel is constant 0 and grp_sel walks 0..N_HEADS-1 on the
    // same T_NEXT edge the old head_sel counter incremented on -> cycle-identical.
    logic [ESEL_W-1:0]           eng_sel;   // engine being RoPE'd / appended
    logic [GRP_W-1:0]            grp_sel;   // head-group index
    wire  [HEAD_W-1:0]           head_sel = HEAD_W'(eng_sel * N_GRP + grp_sel);
    wire signed [ROW_W-1:0] q_head, k_head, v_head;

    generate if (!RAW_REQUANT) begin : g_head_int16
        // Select a head from the already converted INT16 vectors.
        assign q_head = q_in[head_sel*ROW_W +: ROW_W];
        assign k_head = k_in[head_sel*ROW_W +: ROW_W];
        assign v_head = v_in[head_sel*ROW_W +: ROW_W];
    end else begin : g_head_raw
        // Select before requantization to share the converters across heads.
        // As in g4_qkv_seam, the 32-bit rounding add can wrap near INT32_MAX.
        localparam int SHR = G4_FRAC - G5_FRAC;   // requires SHR >= 1
        function automatic logic signed [A_WIDTH-1:0] rsh_sat;
            input signed [31:0] v;
            logic signed [31:0] r;
            begin
                r = (v + (32'sd1 <<< (SHR-1))) >>> SHR;   // round-half, arith shift
                if      (r >  32'sd32767) rsh_sat = 16'sd32767;
                else if (r < -32'sd32768) rsh_sat = 16'sh8000;
                else                       rsh_sat = r[A_WIDTH-1:0];
            end
        endfunction

        // the 2x-wide (INT32) head mux — the honest added cost F2 budgets
        wire signed [32*D_HEAD-1:0] q_raw = q_raw_buf[head_sel];
        wire signed [32*D_HEAD-1:0] k_raw = k_raw_buf[head_sel];
        wire signed [32*D_HEAD-1:0] v_raw = v_raw_buf[head_sel];

        genvar dm;
        for (dm = 0; dm < D_HEAD; dm = dm + 1) begin : g_rq
            assign q_head[dm*A_WIDTH +: A_WIDTH] = rsh_sat(q_raw[dm*32 +: 32]);
            assign k_head[dm*A_WIDTH +: A_WIDTH] = rsh_sat(k_raw[dm*32 +: 32]);
            assign v_head[dm*A_WIDTH +: A_WIDTH] = rsh_sat(v_raw[dm*32 +: 32]);
        end
    end endgenerate

    // =================================================================
    // RoPE — ONE rope_layer, time-multiplexed for Q then K of head_sel.
    // cos/sin LUT is position-indexed and shared across heads (attn.cpp:27).
    // =================================================================
    logic                       rope_start;
    logic [10:0]                rope_pos;
    logic signed [ROW_W-1:0]    rope_vec_in;
    wire  signed [ROW_W-1:0]    rope_vec_out;
    wire                        rope_valid;

    rope_layer #(.D_HEAD(D_HEAD), .N_PAIRS(N_PAIRS), .MAX_CTX(2048), .A_WIDTH(A_WIDTH),
                 .ROPE_CS_PIPE(ROPE_CS_PIPE), .ROPE_POS_REP(ROPE_POS_REP)) u_rope (
        .clk(clk), .rst_n(rst_n),
        .start(rope_start), .pos(rope_pos), .vec_in(rope_vec_in),
        .vec_out(rope_vec_out), .valid(rope_valid)
    );

    // Latched rotated Q (the attention query), one per engine.
    // HEAD_PAR=1 -> a 1-element array == the old scalar register.
    logic signed [ROW_W-1:0]    q_rot_lat [0:HEAD_PAR-1];
    // Rotated K to append for the current token (engine-local staging).
    logic signed [ROW_W-1:0]    k_rot_lat [0:HEAD_PAR-1];

    // =================================================================
    // multihead_kv — the 24-head depth-packed wide URAM KV store (M3).
    // Write port multiplexes: (a) caller preload (pl_*), (b) self-append.
    // Read port driven by the attention head assembly's external KV bus.
    // =================================================================
    logic                       k_wr_en, v_wr_en;
    logic [KVHEAD_W-1:0]        kv_wr_head;
    logic [10:0]                kv_wr_pos;
    logic signed [ROW_W-1:0]    k_wr_vec, v_wr_vec;
    // W3-C1-P1 (CLOCK_CEILING_STUDY §1b/§3 P1): per-stack SAME-CYCLE twin
    // copies of the write position. kv_wr_pos_reg is the SPEEDWAVE #1 routed
    // family (fo=2,910, 13.6 ns pure route into V-BRAM/K-URAM ADDR pins,
    // study §1b); the twins are assigned in the SAME always_ff, from the SAME
    // next-state values, on the SAME cycle — bit-identical, 0 latency
    // (contract-free, unlike the P4 +1-cycle variant which hits the ctx=1
    // stale-by-1 class). (* keep *) stops synthesis from merging them back
    // (the attention_head_assembly_perbank_sbuf.sv:94-95 rd-side idiom).
    // NOTE these are COMPUTE-path twins only; boot-load write strobes are
    // excluded by construction (study §3 "not latency-tolerant" note).
    localparam int N_STACKS_P1 = (N_HEADS + HEADS_PER_STACK - 1) / HEADS_PER_STACK;
    // RTL WO#12 (KVADDR_REP): V write twins carry per-stack bank-GROUP replicas
    // (stack s group g = index s*KV_AR_L+g). K stays per-stack (ADDR_REP_K=1).
    localparam int KV_AR_L     = (KV_ADDR_REP < 1) ? 1 : KV_ADDR_REP;
    localparam int V_WR_NREP   = N_STACKS_P1 * KV_AR_L;
    (* keep = "true" *) logic [10:0] k_wr_pos_q [0:N_STACKS_P1-1];
    (* keep = "true" *) logic [10:0] v_wr_pos_q [0:V_WR_NREP-1];
    wire [N_STACKS_P1*11-1:0] k_wr_pos_rep_flat;
    wire [V_WR_NREP*11-1:0]   v_wr_pos_rep_flat;
    genvar gwr;
    generate for (gwr = 0; gwr < N_STACKS_P1; gwr = gwr + 1) begin : g_wr_rep
        assign k_wr_pos_rep_flat[gwr*11 +: 11] = k_wr_pos_q[gwr];
    end endgenerate
    generate for (gwr = 0; gwr < V_WR_NREP; gwr = gwr + 1) begin : g_wr_rep_v
        assign v_wr_pos_rep_flat[gwr*11 +: 11] = v_wr_pos_q[gwr];
    end endgenerate

    // attention head assembly's KV read request
    wire                        ah_kv_rd_en;
    wire                        ah_kv_rd_kind;   // 0=K, 1=V
    wire  signed [ROW_W-1:0]    k_rd_vec, v_rd_vec;

    // WO-051: per-stack V buses (flattened) for the per-bank combine path.
    localparam int N_STACKS = (N_HEADS + HEADS_PER_STACK - 1) / HEADS_PER_STACK;
    localparam int SSEL_W   = (N_STACKS <= 1) ? 1 : $clog2(N_STACKS);
    wire signed [N_STACKS*ROW_W-1:0] v_rd_bus_flat;
    // G5MEM: per-stack K read-data buses (consumed by the 6-way score path).
    wire signed [N_STACKS*ROW_W-1:0] k_rd_bus_flat;
    // G5MEM: at HEAD_PAR>1 the read head is the SLOT index (== grp_sel); every
    // stack is ungated (RD_ALL) and takes slot_of(grp_sel)==grp_sel, yielding
    // heads {grp_sel, N_GRP+grp_sel, ...} one per stack. At HEAD_PAR=1 this is
    // head_sel, exactly as shipped.
    wire [KVHEAD_W-1:0] kv_rd_head_sel = (HEAD_PAR == 1) ? head_sel
                                                         : KVHEAD_W'(grp_sel);
    // KV_REPL: per-stack replicated read positions from the assembly, one 11-bit
    // slice per KV stack, wired straight to multihead_kv's per-stack rd_pos.
    wire [N_STACKS*11-1:0]         ah_k_rd_pos_rep;
    // RTL WO#12: V read positions carry KV_AR_L bank-group twins per stack.
    wire [N_STACKS*KV_AR_L*11-1:0] ah_v_rd_pos_rep;
    // stack_of(head_sel): which physical KV stack holds the active head.
    // Held stable across a head's whole attention pass (head_sel changes
    // only at T_NEXT_HEAD, after ah_done). The perbank assembly LATCHES it
    // at the ah_start pulse, so no skew vs the registered v_rd_bus.
    wire [SSEL_W-1:0] ah_stack_sel = SSEL_W'(head_sel / HEADS_PER_STACK);

    // The assembly drives ONE rd port with a kind flag; multihead_kv has
    // separate K and V read ports. Issue BOTH reads with the same pos/head
    // each request and let the assembly pick (k_vec_in vs v_vec_in). The
    // assembly only consumes the one it needs in each phase.
    multihead_kv #(
        .N_KV(N_HEADS), .D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH),
        .CTX_WIN(CTX_WIN), .HEADS_PER_STACK(HEADS_PER_STACK),
        .K_RAM_STYLE(K_RAM_STYLE), .V_RAM_STYLE(V_RAM_STYLE),
        .POS_REP(1),                         // KV_REPL: per-stack replicated rd_pos
        .WR_POS_REP(1),                      // W3-C1-P1: per-stack replicated wr_pos
        .ADDR_REP_V(KV_AR_L),                // RTL WO#12: V bank-group split (K stays 1)
        .RD_ALL(HEAD_PAR != 1)               // G5MEM: HEAD_PAR>1 ungates all stacks
    ) u_kv (
        .clk(clk), .rst_n(rst_n),
        .k_wr_en(k_wr_en), .k_wr_head(kv_wr_head), .k_wr_pos(kv_wr_pos), .k_wr_vec(k_wr_vec),
        // scalar k_rd_pos/v_rd_pos unused when POS_REP=1; tie to slice 0 for a valid width.
        .k_rd_en(ah_kv_rd_en), .k_rd_head(kv_rd_head_sel), .k_rd_pos(ah_k_rd_pos_rep[10:0]), .k_rd_vec(k_rd_vec),
        .v_wr_en(v_wr_en), .v_wr_head(kv_wr_head), .v_wr_pos(kv_wr_pos), .v_wr_vec(v_wr_vec),
        .v_rd_en(ah_kv_rd_en), .v_rd_head(kv_rd_head_sel), .v_rd_pos(ah_v_rd_pos_rep[10:0]), .v_rd_vec(v_rd_vec),
        .k_rd_pos_rep(ah_k_rd_pos_rep), .v_rd_pos_rep(ah_v_rd_pos_rep),  // KV_REPL per-stack
        .k_wr_pos_rep(k_wr_pos_rep_flat), .v_wr_pos_rep(v_wr_pos_rep_flat), // W3-C1-P1
        .v_rd_bus_flat(v_rd_bus_flat),       // WO-051: per-stack V buses (additive)
        .k_rd_bus_flat(k_rd_bus_flat)        // G5MEM: per-stack K buses (additive)
    );

    // =================================================================
    // attention_head_assembly (M2) — ONE shared D_HEAD=128 head engine.
    // External KV bus tied to multihead_kv.
    // =================================================================
    logic                       ah_start;
    logic [10:0]                ah_ctx_len;
    // G5MEM: one output/done/residency-count per engine (HEAD_PAR=1 -> 1-elem).
    wire  signed [32*D_HEAD-1:0] ah_out_vec [0:HEAD_PAR-1];
    wire  [HEAD_PAR-1:0]        ah_done_bus;
    // Engines share ONE start/ctx_len and run a deterministic FSM -> lockstep by
    // construction; engine 0's done sequences the top FSM (asserted below).
    wire                        ah_done = ah_done_bus[0];

    // F3 lane-chain seam into the assembly. KV_LANE_CHAIN=0: ah_asm_ctx_len
    // is a plain alias of ah_ctx_len (swept by synthesis — byte-identical
    // legacy netlist) and the lane ports are tied '0. KV_LANE_CHAIN=1: the
    // assembly's ctx_len port carries the SHARED PREFIX (it derives the
    // sweep itself via kv_lane_sweep_len) and the lane ports carry the
    // start-latched lane coordinates. Drivers live in g_lanechain_* at the
    // bottom of the module (after tstate is declared).
    wire [10:0]                 ah_asm_ctx_len;
    wire [3:0]                  ah_lane_j;
    wire [10:0]                 ah_num_steps;
    wire [10:0]                 ah_level_i;
    // F3: # of DISTINCT KV slots ever written (residency count). Read-only.
    // Every engine sees the SAME kv_wr_v/kv_wr_slot mirror -> identical bitmap;
    // engine 0's count is published.
    wire [11:0]                 ah_kv_res_cnt [0:HEAD_PAR-1];

    // WO-051: compile-time select between the SHARED combine (default,
    // golden, −0.780 SLR crossing) and the PER-BANK local combine (the SLR
    // fix). Exactly one assembly is instantiated; both drive the SAME
    // ah_out_vec / ah_done seam, so everything downstream is unchanged.
    generate
    // F3: the lane-chain remap exists ONLY in the perbank-sbuf assembly.
    // Enabling it against the legacy shared assembly would silently fall
    // back to the dense sweep — the exact defect being fixed — so make
    // that configuration an ELABORATION failure, sim and synth alike.
    if (KV_LANE_CHAIN != 0 && COMBINE_LOCAL == 1'b0) begin : g_cfg_err
        kv_lane_chain_requires_combine_local_perbank u_cfg_err();
    end
    endgenerate

    generate
    if (COMBINE_LOCAL == 1'b0) begin : g_combine_shared
        // Legacy shared combine still uses the OLD scalar-kv_rd_pos assembly.
        // KV_REPL: fan its single position onto every per-stack slice so the
        // multihead_kv (POS_REP=1) interface is driven identically (bit-exact).
        wire [10:0] ah_kv_rd_pos_sh;
        attention_head_assembly #(
            .D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH), .CTX_MAX(CTX_MAX),
            .SCORE_LATENCY(SCORE_LATENCY), .SCORE_NORM_SHR(SCORE_NORM_SHR)
        ) u_attn (
            .clk(clk), .rst_n(rst_n),
            .start(ah_start), .ctx_len(ah_ctx_len), .q_vec(q_rot_lat[0]),
            .kv_rd_en(ah_kv_rd_en), .kv_rd_kind(ah_kv_rd_kind), .kv_rd_pos(ah_kv_rd_pos_sh),
            .k_vec_in(k_rd_vec), .v_vec_in(v_rd_vec),
            .out_vec(ah_out_vec[0]), .done(ah_done_bus[0])
        );
        genvar gsh;
        for (gsh = 0; gsh < N_STACKS; gsh = gsh + 1) begin : g_sh_fan
            assign ah_k_rd_pos_rep[gsh*11 +: 11] = ah_kv_rd_pos_sh;
        end
        // RTL WO#12: fan the shared scalar onto EVERY V group twin.
        genvar gshv;
        for (gshv = 0; gshv < N_STACKS*KV_AR_L; gshv = gshv + 1) begin : g_sh_fan_v
            assign ah_v_rd_pos_rep[gshv*11 +: 11] = ah_kv_rd_pos_sh;
        end
        // F3: the legacy SHARED-combine assembly has no residency mask (it is
        // NOT the shipped path -- COMBINE_LOCAL=1 selects the perbank_sbuf
        // engine, netlist-verified g_combine_perbank). Report 0 so a consumer
        // can tell "no residency tracking here" from "nothing written".
        assign ah_kv_res_cnt[0] = 12'd0;
    end else if (HEAD_PAR != 1) begin : g_headpar
        // =============================================================
        // G5MEM 6-WAY HEAD-PARALLEL — HEAD_PAR engines, engine e -> KV stack e.
        // All engines see the SAME start / ctx_len -> lockstep FSMs; every
        // engine issues its K read (MAX pass) and V read (COMB pass) on the
        // SAME cycle. multihead_kv runs RD_ALL=1 so all read ports fire
        // together (one head per stack) instead of 1-of-6. Each engine takes
        // N_STACKS=1 (its own stack); its combine is fed only by its own V row.
        //
        // DEPTH-N COMPOSE (tech-lead ruling 2026-07-26): the KV lane-chain
        // remap (ah_asm_ctx_len/ah_lane_j/ah_num_steps/ah_level_i) is computed
        // ONCE in g_lanechain_lat and BROADCAST identically to every engine —
        // head-parallelism (which heads run) is orthogonal to position-mapping
        // (which physical slot a position reads), so one remap serves all.
        // Each engine carries its OWN F3 residency mirror bitmap fed the SAME
        // kv_wr_v/kv_wr_slot (per-engine mirror, avoids the 1024-bit shared-read
        // fanout that killed the 0x1E route).
        // =============================================================
        wire [HEAD_PAR-1:0] eng_kv_rd_en;
        wire [HEAD_PAR-1:0] eng_kv_rd_kind;

        genvar ge;
        for (ge = 0; ge < HEAD_PAR; ge = ge + 1) begin : g_eng
            wire [10:0]           eng_k_rd_pos;
            // RTL WO#12: this engine's ONE stack, KV_AR_L V bank-group twins.
            wire [KV_AR_L*11-1:0] eng_v_rd_pos;

            // RED ARM (gate tooth, never synthesised): mis-map engine ge onto
            // the NEXT stack's K row. The head->stack binding is the whole
            // correctness claim; the gate must be able to see it broken.
`ifdef G5MEM_RED
            localparam int KSTK = (ge + 1) % HEAD_PAR;
`else
            localparam int KSTK = ge;
`endif

            attention_head_assembly_perbank_sbuf #(
                .D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH), .CTX_MAX(CTX_MAX),
                .SCORE_LATENCY(SCORE_LATENCY), .SCORE_NORM_SHR(SCORE_NORM_SHR),
                .N_STACKS(1),                    // one bank: this engine's stack
                // DEPTH-N: the same lane-chain remap the serial path uses.
                .KV_LANE_CHAIN(KV_LANE_CHAIN), .K_F(K_F), .CTX_WIN(CTX_WIN),
                .KV_ADDR_REP(KV_AR_L),           // RTL WO#12: V group read twins (this stack)
                .RES_CLR_REP(RES_CLR_REP)        // F3-CLR: clear fan-out split
            ) u_attn (
                .clk(clk), .rst_n(rst_n),
                .start(ah_start), .ctx_len(ah_asm_ctx_len), .q_vec(q_rot_lat[ge]),
                // DEPTH-N broadcast (identical to all engines).
                .lane_j(ah_lane_j), .num_steps(ah_num_steps), .level_i(ah_level_i),
                .stack_sel(1'b0),                // the only bank
                .kv_rd_en(eng_kv_rd_en[ge]), .kv_rd_kind(eng_kv_rd_kind[ge]),
                .k_rd_pos_rep(eng_k_rd_pos),
                .v_rd_pos_rep(eng_v_rd_pos),
                .k_vec_in(k_rd_bus_flat[KSTK*ROW_W +: ROW_W]),  // THIS stack's K row
                .v_rd_bus_flat(v_rd_bus_flat[ge*ROW_W +: ROW_W]), // THIS stack's V row
                // F3 KV-RESIDENCY MASK write mirror — PER-ENGINE bitmap, SAME
                // muxed write port on every engine (F3-1 driven-port check).
                .kv_wr_v(k_wr_en), .kv_wr_slot(kv_wr_pos),
                .kv_res_clear(kv_res_clear),   // F3-CLR: same pulse, all engines
                .b2p_round_en(b2p_round_en),   // B2' broadcast (identical, all engines)
                .kv_res_cnt(ah_kv_res_cnt[ge]),
                .out_vec(ah_out_vec[ge]), .done(ah_done_bus[ge])
            );

            // Engine e owns stack e's replicated read-position register.
            // K: one per-stack slice. V: this stack's KV_AR_L bank-group twins,
            // placed at the stack's group-stride window (stack ge == KV stack ge).
            assign ah_k_rd_pos_rep[ge*11 +: 11]                   = eng_k_rd_pos;
            assign ah_v_rd_pos_rep[ge*KV_AR_L*11 +: KV_AR_L*11]   = eng_v_rd_pos;
        end

        // Lockstep engines: engine 0 drives the shared read strobes.
        assign ah_kv_rd_en   = eng_kv_rd_en[0];
        assign ah_kv_rd_kind = eng_kv_rd_kind[0];

`ifndef SYNTHESIS
        // LOCKSTEP TOOTH: assert every cycle rather than trusting it — a skew
        // would mis-capture silently (feedback_valid_data_latency_skew_class).
        always_ff @(posedge clk) if (rst_n) begin
            if (eng_kv_rd_en   !== {HEAD_PAR{eng_kv_rd_en[0]}} ||
                eng_kv_rd_kind !== {HEAD_PAR{eng_kv_rd_kind[0]}} ||
                ah_done_bus    !== {HEAD_PAR{ah_done_bus[0]}}) begin
                $display("[G5MEM] FATAL lockstep break @%0t rd_en=%b kind=%b done=%b",
                         $time, eng_kv_rd_en, eng_kv_rd_kind, ah_done_bus);
                $fatal(1, "G5MEM engine lockstep violated");
            end
        end
`endif
    end else begin : g_combine_perbank
        attention_head_assembly_perbank_sbuf #(
            .D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH), .CTX_MAX(CTX_MAX),
            .SCORE_LATENCY(SCORE_LATENCY), .SCORE_NORM_SHR(SCORE_NORM_SHR),
            .N_STACKS(N_STACKS),
            // F3: KV lane-chain READ remap (KV_LANE_CHAIN=0 => g_lane not
            // elaborated in the assembly; ctx_len/lane wiring collapses to
            // the byte-identical legacy form).
            .KV_LANE_CHAIN(KV_LANE_CHAIN), .K_F(K_F), .CTX_WIN(CTX_WIN),
            .KV_ADDR_REP(KV_AR_L),           // RTL WO#12: V group read twins (6 stacks)
            .RES_CLR_REP(RES_CLR_REP)        // F3-CLR: clear fan-out split
        ) u_attn (
            .clk(clk), .rst_n(rst_n),
            .start(ah_start), .ctx_len(ah_asm_ctx_len), .q_vec(q_rot_lat[0]),
            .lane_j(ah_lane_j), .num_steps(ah_num_steps), .level_i(ah_level_i),
            .stack_sel(ah_stack_sel),
            .kv_rd_en(ah_kv_rd_en), .kv_rd_kind(ah_kv_rd_kind),
            .k_rd_pos_rep(ah_k_rd_pos_rep),      // KV_REPL: per-stack replicated rd_pos
            .v_rd_pos_rep(ah_v_rd_pos_rep),
            .k_vec_in(k_rd_vec),                 // score path: scalar K (unchanged)
            .v_rd_bus_flat(v_rd_bus_flat),       // per-stack V buses (SLR-local combine)
            // ---------------------------------------------------------
            // F3 KV-RESIDENCY MASK write mirror. k_wr_en / kv_wr_pos are the
            // MUXED KV write port (pl_* preload OR the T_APPEND self-append),
            // registered, so the residency bit sets on the SAME clock edge as
            // the K/V row itself -- a slot can never read back resident before
            // its data exists. Read side uses the same absolute slot number
            // (pos_rd == the lane-chain eff_pos), so the two agree without a
            // second copy of the slot formula.
            //   T_APPEND -> k_wr_en on the NEXT cycle -> write edge and bit-set
            //   edge coincide; ah_start rises on that same edge and the
            //   assembly's first S_MAX_RD is >=2 cycles later. Margin proven
            //   in ATTN_F3_KV_RESIDENCY_MASK.md §3.
            // ---------------------------------------------------------
            .kv_wr_v(k_wr_en), .kv_wr_slot(kv_wr_pos),
            .kv_res_clear(kv_res_clear),       // F3-CLR host session boundary
            .b2p_round_en(b2p_round_en),       // B2'
            .kv_res_cnt(ah_kv_res_cnt[0]),
            .out_vec(ah_out_vec[0]), .done(ah_done_bus[0])
        );
    end
    endgenerate

    // =================================================================
    // Per-head output buffer (unpacked array — distinct driver per head).
    // attention_head_assembly emits INT32 accumulators; the g5 seam is RAW
    // INT16 (the low 16 bits, no >>12 — MIXED_PRECISION §3.1). At ctx_len=1
    // out[d] = V[d] (<= int16), so the low-16 truncation is lossless there.
    // =================================================================
    logic signed [A_WIDTH-1:0] head_out_arr [0:N_HEADS*D_HEAD-1];

    // single combinational pack to the output
    always_comb begin
        for (int k = 0; k < N_HEADS*D_HEAD; k++)
            out_vec[k*A_WIDTH +: A_WIDTH] = head_out_arr[k];
    end

    assign dbg_head = head_sel;
    // F3: publish the residency count from engine 0 (all engines identical).
    assign kv_res_cnt = ah_kv_res_cnt[0];

    // =================================================================
    // Top FSM — per head: RoPE-Q, RoPE-K, (optional self-append), run M2,
    // capture head output, advance head.
    // =================================================================
    typedef enum logic [3:0] {
        T_IDLE       = 4'd0,
        T_ROPE_Q     = 4'd1,   // present Q to rope_layer
        T_ROPE_Q_W   = 4'd2,   // wait rope_valid -> latch q_rot
        T_ROPE_K     = 4'd3,   // present K to rope_layer
        T_ROPE_K_W   = 4'd4,   // wait rope_valid -> latch k_rot
        T_APPEND     = 4'd5,   // write rotated K + raw V at append_pos
        T_ATTN_START = 4'd6,   // pulse M2 start
        T_ATTN_WAIT  = 4'd7,   // wait M2 done
        T_CAPTURE    = 4'd8,   // copy M2 out_vec -> head slot
        T_NEXT       = 4'd9,   // advance head / finish
        T_DONE       = 4'd10
    } tstate_e;

    tstate_e          tstate;
    logic [10:0]      ctx_len_lat;
    logic [10:0]      append_pos_lat;
    logic [10:0]      rope_pos_lat;    // ROTATION ANGLE (== append_pos_lat when ROPE_POS_DECOUPLE=0)
    logic             fill_lat;
    logic [GRP_W-1:0] cap_grp;    // head-group being captured (== grp_sel)

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            tstate         <= T_IDLE;
            eng_sel        <= '0;
            grp_sel        <= '0;
            ctx_len_lat    <= 11'd1;
            append_pos_lat <= '0;
            rope_pos_lat   <= '0;
            fill_lat       <= 1'b0;
            rope_start     <= 1'b0;
            rope_pos       <= '0;
            rope_vec_in    <= '0;
            for (int e = 0; e < HEAD_PAR; e++) begin
                q_rot_lat[e] <= '0;
                k_rot_lat[e] <= '0;
            end
            k_wr_en        <= 1'b0;
            v_wr_en        <= 1'b0;
            kv_wr_head     <= '0;
            kv_wr_pos      <= '0;
            // W3-C1-P1 twins: same value, same cycle, every site.
            // RTL WO#12: V carries V_WR_NREP bank-group twins (K per-stack).
            for (int s = 0; s < N_STACKS_P1; s++) k_wr_pos_q[s] <= '0;
            for (int s = 0; s < V_WR_NREP;   s++) v_wr_pos_q[s] <= '0;
            k_wr_vec       <= '0;
            v_wr_vec       <= '0;
            ah_start       <= 1'b0;
            ah_ctx_len     <= 11'd1;
            done           <= 1'b0;
            for (int i = 0; i < N_HEADS*D_HEAD; i++) head_out_arr[i] <= '0;
        end else begin
            // defaults (1-cycle pulses)
            rope_start <= 1'b0;
            k_wr_en    <= 1'b0;
            v_wr_en    <= 1'b0;
            ah_start   <= 1'b0;
            done       <= 1'b0;

            // ---- caller preload writes (independent of the head FSM) ----
            // pl_we wins the KV write port when not self-appending. The TB
            // stages context here BEFORE start. (Single writer to k_wr_*/
            // v_wr_* via these registered assigns — the FSM self-append in
            // T_APPEND is mutually exclusive in time with preload.)
            if (pl_we) begin
                k_wr_en    <= 1'b1;
                v_wr_en    <= 1'b1;
                kv_wr_head <= pl_head;
                kv_wr_pos  <= pl_pos;
                // W3-C1-P1 twins (same next-state value, same cycle)
                for (int s = 0; s < N_STACKS_P1; s++) k_wr_pos_q[s] <= pl_pos;
                for (int s = 0; s < V_WR_NREP;   s++) v_wr_pos_q[s] <= pl_pos;   // RTL WO#12
                k_wr_vec   <= pl_k_vec;
                v_wr_vec   <= pl_v_vec;
            end

            case (tstate)
                T_IDLE: begin
                    if (start) begin
                        ctx_len_lat    <= (ctx_len == 0) ? 11'd1 : ctx_len;
                        append_pos_lat <= append_pos;
                        // storage slot and rotation angle latched SEPARATELY
                        rope_pos_lat   <= ROPE_POS_DECOUPLE ? rope_pos_in : append_pos;
                        fill_lat       <= do_kv_fill;
                        eng_sel        <= '0;
                        grp_sel        <= '0;
                        tstate         <= T_ROPE_Q;
                    end
                end

                // ===== RoPE Q for head_sel (== engine eng_sel of group grp_sel) =====
                T_ROPE_Q: begin
                    rope_vec_in <= q_head;
                    rope_pos    <= rope_pos_lat;   // ANGLE, not the storage slot
                    rope_start  <= 1'b1;
                    tstate      <= T_ROPE_Q_W;
                end
                T_ROPE_Q_W: begin
                    if (rope_valid) begin
                        q_rot_lat[eng_sel] <= rope_vec_out;
                        tstate             <= T_ROPE_K;
                    end
                end

                // ===== RoPE K for head_sel =====
                T_ROPE_K: begin
                    rope_vec_in <= k_head;
                    rope_pos    <= rope_pos_lat;   // ANGLE, not the storage slot
                    rope_start  <= 1'b1;
                    tstate      <= T_ROPE_K_W;
                end
                T_ROPE_K_W: begin
                    if (rope_valid) begin
                        k_rot_lat[eng_sel] <= rope_vec_out;
                        // G5MEM: when not self-appending, advance the engine loop
                        // here (T_APPEND does it otherwise). At HEAD_PAR=1 eng_sel
                        // is always 0 -> collapses to the shipped
                        // `(fill_lat) ? T_APPEND : T_ATTN_START`.
                        if (fill_lat) begin
                            tstate  <= T_APPEND;
                        end else if (eng_sel == ESEL_W'(HEAD_PAR-1)) begin
                            eng_sel <= '0;
                            tstate  <= T_ATTN_START;
                        end else begin
                            eng_sel <= eng_sel + 1'b1;
                            tstate  <= T_ROPE_Q;
                        end
                    end
                end

                // ===== self-append rotated K + raw V at append_pos =====
                T_APPEND: begin
                    // mutually exclusive with pl_we (TB drops pl_we before start)
                    k_wr_en    <= 1'b1;
                    v_wr_en    <= 1'b1;
                    kv_wr_head <= head_sel;   // derived = eng_sel*N_GRP + grp_sel
                    kv_wr_pos  <= append_pos_lat;
                    // W3-C1-P1 twins (same next-state value, same cycle)
                    for (int s = 0; s < N_STACKS_P1; s++) k_wr_pos_q[s] <= append_pos_lat;
                    for (int s = 0; s < V_WR_NREP;   s++) v_wr_pos_q[s] <= append_pos_lat;  // RTL WO#12
                    k_wr_vec   <= k_rot_lat[eng_sel];
                    v_wr_vec   <= v_head;
                    // G5MEM: advance the engine loop; at HEAD_PAR=1 unconditionally
                    // `tstate <= T_ATTN_START` as shipped.
                    if (eng_sel == ESEL_W'(HEAD_PAR-1)) begin
                        eng_sel <= '0;
                        tstate  <= T_ATTN_START;
                    end else begin
                        eng_sel <= eng_sel + 1'b1;
                        tstate  <= T_ROPE_Q;
                    end
                end

                // ===== run the HEAD_PAR-parallel 3-pass attention =====
                T_ATTN_START: begin
                    ah_ctx_len <= ctx_len_lat;
                    ah_start   <= 1'b1;
                    cap_grp    <= grp_sel;
                    tstate     <= T_ATTN_WAIT;
                end
                T_ATTN_WAIT: begin
                    if (ah_done) tstate <= T_CAPTURE;
                end

                // ===== capture all HEAD_PAR engines' outputs in one cycle =====
                // Engine e owns head e*N_GRP + cap_grp — distinct slots, one
                // driver each (feedback_multi_driver_packed_reg_trap). RAW INT16
                // g5 seam: low 16 bits of each INT32 acc. At HEAD_PAR=1 this is
                // the shipped single-head capture of head cap_grp.
                T_CAPTURE: begin
                    for (int e = 0; e < HEAD_PAR; e++)
                        for (int d = 0; d < D_HEAD; d++)
                            head_out_arr[(e*N_GRP + cap_grp)*D_HEAD + d]
                                <= ah_out_vec[e][d*32 +: 16];
                    tstate <= T_NEXT;
                end

                T_NEXT: begin
                    if (grp_sel == GRP_W'(N_GRP-1)) begin
                        tstate <= T_DONE;
                    end else begin
                        grp_sel <= grp_sel + 1'b1;
                        tstate  <= T_ROPE_Q;
                    end
                end

                T_DONE: begin
                    done   <= 1'b1;
                    tstate <= T_IDLE;
                end

                default: tstate <= T_IDLE;
            endcase
        end
    end

    // =================================================================
    // F3: lane-chain input latches (KV_LANE_CHAIN=1 only).
    //
    // Latched at the SAME accept condition as the main FSM's T_IDLE latch
    // (start && tstate==T_IDLE) so a spurious mid-pass start pulse cannot
    // skew the lane coordinates against ctx_len_lat/append_pos_lat. The
    // assembly re-latches its own copies on each per-head ah_start, so the
    // values only need to be stable across the whole multi-head pass —
    // which this latch guarantees regardless of what the top does with
    // branch_j_w/level_i_w after start.
    //
    // KV_LANE_CHAIN=0 arm: pure '0 ties + the ah_ctx_len alias. No FF, no
    // logic — the pre-F3 netlist is reproduced exactly (generate-gated,
    // depth-0 byte-identical requirement).
    // =================================================================
    generate
    if (KV_LANE_CHAIN != 0) begin : g_lanechain_lat
        logic [10:0] ctx_prefix_l;
        logic [3:0]  lane_j_l;
        logic [10:0] num_steps_l;
        logic [10:0] level_i_l;

        always_ff @(posedge clk) begin
            if (!rst_n) begin
                ctx_prefix_l <= 11'd1;
                lane_j_l     <= '0;
                num_steps_l  <= '0;
                level_i_l    <= '0;
            end else if (start && tstate == T_IDLE) begin
                // same ==0 floor as ctx_len_lat (drafter guarantees >=1)
                ctx_prefix_l <= (ctx_len_prefix == 11'd0) ? 11'd1 : ctx_len_prefix;
                lane_j_l     <= lane_j;
                num_steps_l  <= num_steps;
                level_i_l    <= level_i;
            end
        end

        assign ah_asm_ctx_len = ctx_prefix_l;
        assign ah_lane_j      = lane_j_l;
        assign ah_num_steps   = num_steps_l;
        assign ah_level_i     = level_i_l;

        // synthesis translate_off
        // Loud-fail guards for the half-wired top state this fix exists to
        // kill: (1) lane inputs left floating (F2 diff not applied) must
        // X-fatal, not silently address garbage; (2) the legacy ctx_len
        // port must carry the SWEEP invariant prefix+level (drafter
        // s7a_ctx_len_eff), or the write side and read side have diverged.
        always @(posedge clk) if (rst_n && start && tstate == T_IDLE) begin
            if ($isunknown({ctx_len_prefix, lane_j, num_steps, level_i}))
                $fatal(1, "attention_full_mha_sbuf: KV_LANE_CHAIN=1 but lane inputs are X/Z — top-level lane wiring (F3's F2-side diff) is missing");
            if (11'((ctx_len_prefix == 11'd0 ? 11'd1 : ctx_len_prefix) + level_i[10:0])
                != (ctx_len == 11'd0 ? 11'd1 : ctx_len))
                $fatal(1, "attention_full_mha_sbuf: ctx_len(sweep)=%0d != ctx_len_prefix=%0d + level_i=%0d — write/read lane-chain views diverged",
                       ctx_len, ctx_len_prefix, level_i);
        end
        // synthesis translate_on
    end else begin : g_lanechain_off
        assign ah_asm_ctx_len = ah_ctx_len;   // byte-identical legacy path
        assign ah_lane_j      = '0;
        assign ah_num_steps   = '0;
        assign ah_level_i     = '0;
    end
    endgenerate

endmodule

`default_nettype wire
