// =====================================================================
// attention_head_assembly_perbank_sbuf.sv  (EXP_G5_CTXSCAN — FIX-SBUF)
// =====================================================================
//
// SPEED-ONLY variant of attention_head_assembly_perbank. The MEASURED
// per-position bound of the baseline is NOT the URAM K read (full-row
// 2048b, RD_LAT=2, fully hidden) — it is that the Q·K score dot
// (chunk-serial, ~13 cycles) is computed TWICE per position:
//   pass 1 (S_MAX_*): dot -> running max          ~15 cy/pos
//   pass 2 (S_EXP_*): SAME dot recomputed -> exp  ~16 cy/pos
//
// FIX: buffer the pass-1 score (score_norm, post-SHR) in a 32b x CTX_MAX
// BRAM (score_buf, 1 RAMB36). Pass 2 becomes a 3-cycle buffered-score
// read (ARD/AWAIT/BUF) with ZERO K reads and ZERO dot recompute.
//
// BIT-EXACT by construction: the stored value is the exact score_norm the
// baseline recomputes (same latched q_vec, same K row, integer dot is
// deterministic); the softmax consumes the identical INT32 through the
// identical exp_pwl/inv path in the identical position order, so
// running_max, sum_so_far, exp_buf and out_vec are bit-identical.
// Score ARITHMETIC (attn_score_dot / csa_tree16_int28 / INT32 widths) is
// UNTOUCHED (INT20/INT28 overflow lessons). MAX pass cadence unchanged.
// Gate: tb_g5_ab_sbuf.sv (REF = unmodified attention_full_mha).
//
// Diff vs base (verilog/forks/reclock100/src_full/layers/
// attention_head_assembly_perbank.sv):
//   + score_buf BRAM + sb_we/sb_waddr/sb_raddr/sb_rdata
//   + S_MAX_WAIT writes score_norm on score_done
//   ~ S_EXP_RD/S_EXP_START/S_EXP_WAIT -> S_EXP_ARD/S_EXP_AWAIT (buffered)
//   ~ softmax score_in = sb_rdata (the buffered pass-1 score)
//   - EXP pass issues NO kv_rd_en (K stacks idle in pass 2)
// Everything else (MAX pass, INV, COMB pass, perbank combine, KV_REPL
// position replication, drain states) is byte-identical to the base.
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module attention_head_assembly_perbank_sbuf #(
    parameter int D_HEAD         = 128,
    parameter int A_WIDTH        = 16,
    parameter int CTX_MAX        = 1024,
    parameter int SCORE_LATENCY  = 13,
    parameter int SCORE_NORM_SHR = 0,
    // ---------------------------------------------------------------
    // DEPTH_TEMP_FIX — the 1/sqrt(2) softmax-temperature correction (F1), done
    // HERE as a multiply.  DEFAULT FLIPPED 1 -> 0 on 2026-07-23 (F1+F2 commit).
    //
    // *** DO NOT SET THIS BACK TO 1 WITHOUT READING THIS. ***
    // F1 already ships, folded into the exp table values in exp_pwl_q12.sv
    // (EXP_PWL_LUT[i] = 4096*exp(-(i*2^7/2^12)/sqrt(2))).  Enabling BOTH applies
    // 1/sqrt(2) TWICE -> an effective temperature of 1/(sqrt(128)*sqrt(2)) = 1/16,
    // i.e. 1.414x TOO FLAT — strictly worse than the original bug.
    // scripts/ctxn_attn_numerics_model.py section 6 has a mechanical tooth that
    // RED-fires on exactly that combination.
    //
    // The table fold was chosen over this multiply because it measures better and
    // costs less: ctx=4 L1 vs fp32 attn.cpp 0.0369 (this multiply, which also
    // COARSENS the exp grid to a 0.25 true-exponent step) vs 0.0033 (fold, 0.0221
    // step); zero multipliers instead of one 32x15 per engine (6 at HEAD_PAR=6) on
    // the score->max_tracker path; and score_norm stays BIT-IDENTICAL, so
    // score_max_tracker / score_buf / the max compare are untouched.
    // Full argument + numbers: reports/results/ATTN_F1F2_SOFTMAX_SHARPNESS.md
    //
    // Kept (not deleted) so the multiply form stays available as an A/B arm:
    // override to 1 ONLY together with reverting the sqrt(2) out of the table.
    // ---------------------------------------------------------------
    parameter int DEPTH_TEMP_FIX = 0,
    parameter int N_STACKS       = 6,
    parameter int SSEL_W         = (N_STACKS <= 1) ? 1 : $clog2(N_STACKS),
    // ---- lane-chain KV (SGLang stock semantics) -------------------------
    // 0 = legacy dense sweep, structurally identical to the pre-lane-chain
    //     revision (the remap generate block is not elaborated at all).
    // 1 = lane-chain remap of the KV RAM address ONLY.
    parameter int KV_LANE_CHAIN  = 0,
    parameter int K_F            = 8,
    parameter int CTX_WIN        = 1024,
    // ---------------------------------------------------------------
    // RTL WO#12 (KVADDR_REP): V read-position BANK-GROUP replicas per
    // stack. Run-10 (66 MHz) measured the per-stack replica
    // v_rd_pos_q_reg[2][5] -> g_stack[2].u_v BRAM addr at fo=456 /
    // 13.602 ns pure route / KV_VADDR +0.028 (< the +0.30 flash bar;
    // -0.362 @80 MHz run 9): ONE xpm addr port re-aggregates the whole
    // ~228-RAMB36 stack behind a single net. KV_ADDR_REP=G gives every
    // stack G keep'd same-edge twins, each addressing a ROW_W/G memory
    // slice (multihead_kv/kv_head_wide_win split). Same next-state value,
    // same always_ff, same cycle => bit-identical, 0 latency — the
    // ROPE_POS_REP / W3-C1-P1 idiom, one granularity finer.
    // ---------------------------------------------------------------
    parameter int KV_ADDR_REP    = 1,
    // ---------------------------------------------------------------
    // F3-CLR (2026-07-28): fan-out replication factor for the
    // host-writable `kv_res_clear` pulse. The clear must reach ALL
    // CTX_WIN=1024 kv_resident FFs of THIS engine (x HEAD_PAR engines),
    // so a single un-replicated net is a 1024/6144-load control signal —
    // the ROPE_POS_REP / KV_ADDR_REP class of route problem.
    // RES_CLR_REP = R registers R keep'd SAME-EDGE twins of the pulse,
    // twin r driving bits [r*SEG, (r+1)*SEG). All twins share one source
    // register value, so the clear is still ATOMIC across the bitmap
    // (every bit clears on the same edge); only the ROUTE is split.
    // R=1 degenerates to a single registered pulse.
    // ---------------------------------------------------------------
    parameter int RES_CLR_REP    = 1
) (
    input  wire                              clk,
    input  wire                              rst_n,

    input  wire                              start,
    // KV_LANE_CHAIN=0: total dense sweep length (legacy meaning, unchanged).
    // KV_LANE_CHAIN=1: length of the SHARED PREFIX. The own-column entries
    //   are added by kv_lane_sweep_len(); at level_i=0 the two coincide.
    input  wire [10:0]                       ctx_len,
    input  wire signed [A_WIDTH*D_HEAD-1:0]  q_vec,
    input  wire [SSEL_W-1:0]                 stack_sel,

    // ---- lane-chain inputs (ignored when KV_LANE_CHAIN=0) ----
    input  wire [3:0]                        lane_j,      // branch / topk id
    input  wire [10:0]                       num_steps,   // SGLang num_steps
    input  wire [10:0]                       level_i,     // level within lane

    output logic                             kv_rd_en,
    output logic                             kv_rd_kind,        // 0=K, 1=V
    output logic [N_STACKS*11-1:0]           k_rd_pos_rep,
    // RTL WO#12 (KVADDR_REP): the V read position carries KV_ADDR_REP
    // same-cycle group replicas per stack (stack s group g = slice
    // s*KV_ADDR_REP+g). KV_ADDR_REP=1 degenerates to the per-stack form
    // byte-identically. K stays per-stack (run-10: only the V-BRAM cone
    // failed; a K split costs URAM at the binding wall).
    output logic [N_STACKS*KV_ADDR_REP*11-1:0] v_rd_pos_rep,
    input  wire  signed [A_WIDTH*D_HEAD-1:0] k_vec_in,
    input  wire  signed [N_STACKS*A_WIDTH*D_HEAD-1:0] v_rd_bus_flat,

    // =====================================================================
    // F3 KV-RESIDENCY MASK inputs (reports/results/ATTN_F3_KV_RESIDENCY_MASK.md)
    //
    // kv_wr_v / kv_wr_slot MIRROR the KV store's write port exactly (the
    // parent's registered k_wr_en / kv_wr_pos, i.e. the MUXED port that
    // carries BOTH the FSM self-append and the pl_* preload). One bit per
    // absolute KV slot is set on the SAME clock edge the K/V row is written,
    // so a slot can never read back "resident" before its data exists.
    // Leaving kv_wr_v tied 0 is legal (a unit TB that only checks addresses)
    // -- the sweep is then reported as an EMPTY-WINDOW in sim.
    // =====================================================================
    input  wire                              kv_wr_v,
    input  wire [10:0]                       kv_wr_slot,
    // =====================================================================
    // F3-CLR — HOST-WRITABLE RESIDENCY CLEAR (reg 0x154 bit[0]).
    //
    // THE DEFECT THIS REPAIRS: `kv_resident` was SET-ONLY, cleared by
    // rst_n alone — and on this board rst_n is a PHYSICAL COLD-AC. The
    // "residency is a SESSION property" rationale below is sound only
    // inside ONE continuous sequence. SGLang decodes MANY requests; each
    // new sequence reuses slots 0..N for a DIFFERENT context but does NOT
    // rewrite every slot (the FSM self-appends ONE row per forward). So
    // slots the PREVIOUS sequence wrote stay marked resident while the KV
    // RAM still holds the PREVIOUS sequence's K/V -> the drafter attends
    // stale cross-request context that the F3 mask certifies as
    // legitimate. Observed: depth-0 golden 7/8 on a freshly power-cycled
    // card, 0/8 after ANY decode run, restored only by a cold-AC; accept
    // pinned 1.02-1.05 across every host-side fix, because the host had
    // no way to clear this bitmap.
    //
    // 1-cycle pulse, asynchronous to any launch. The host issues it at the
    // START of every new sequence/request, before the first draft launch.
    // Tying it to 1'b0 leaves this module BYTE-IDENTICAL to the pre-fix
    // revision (the clear branch is then unreachable) — that is how the
    // depth-0 seal stays intact and how a legacy/unit caller stays valid.
    // =====================================================================
    input  wire                              kv_res_clear,
    // =====================================================================
    // B2' (ACCFIX bit1, 2026-07-29): round-half instead of floor on the two
    // >>>12 truncations of the softmax->V combine (the ~98%-of-ctx-error
    // floors, ANGLE_B_NUMERIC_AT_CTX.md §B2'): (a) the per-position weight
    // attn_q12 = sat16((exp*inv + 2048) >> 12) computed HERE, and (b) the
    // per-position (attn*V + 2048) >> 12 product inside attn_v_combine.
    // 0 = legacy floor, BYTE-IDENTICAL. ctx=1 is byte-identical either way.
    // =====================================================================
    input  wire                              b2p_round_en,
    // # of DISTINCT KV slots written since reset (the residency count the
    // host-side ctx_len register has no coupling to). Read-only status.
    output wire  [11:0]                      kv_res_cnt,

    output wire  signed [32*D_HEAD-1:0]      out_vec,
    output logic                             done
);

    localparam int CTX_W = $clog2(CTX_MAX);

    typedef enum logic [4:0] {
        S_IDLE        = 5'd0,
        S_MAX_RD      = 5'd1,
        S_MAX_START   = 5'd2,
        S_MAX_WAIT    = 5'd3,
        S_EXP_ARD     = 5'd4,   // FIX-SBUF: issue score_buf read (was S_EXP_RD)
        S_EXP_AWAIT   = 5'd5,   // FIX-SBUF: rdata commits; arm exp valid (was S_EXP_START)
        S_EXP_BUF     = 5'd7,   // unchanged encoding; S_EXP_WAIT (6) retired
        S_INV         = 5'd8,
        S_COMB_RD     = 5'd9,
        S_COMB_WAIT   = 5'd10,
        S_COMB_MAC    = 5'd11,
        S_COMB_MAC2   = 5'd14,
        S_DRAIN_COMB  = 5'd12,
        S_DRAIN_COMB2 = 5'd15,
        S_DONE        = 5'd13
    } state_e;

    state_e state;
    logic [10:0]      pos_req;
    logic [10:0]      ctx_len_latched;
    logic signed [A_WIDTH*D_HEAD-1:0] q_vec_latched;
    logic [SSEL_W-1:0] stack_sel_latched;

    // RTL WO#12: V replicas are per-stack-per-group (N_STACKS*KV_ADDR_REP);
    // K replicas stay per-stack. All are same-edge twins of the same
    // next-state value (pos_rd) — bit-identical, latency-neutral.
    localparam int KV_AR   = (KV_ADDR_REP < 1) ? 1 : KV_ADDR_REP;
    localparam int V_NREP  = N_STACKS * KV_AR;
    (* keep = "true" *) logic [10:0] k_rd_pos_q [0:N_STACKS-1];
    (* keep = "true" *) logic [10:0] v_rd_pos_q [0:V_NREP-1];
    genvar gpr;
    generate
        for (gpr = 0; gpr < N_STACKS; gpr = gpr + 1) begin : g_posrep
            assign k_rd_pos_rep[gpr*11 +: 11] = k_rd_pos_q[gpr];
        end
        for (gpr = 0; gpr < V_NREP; gpr = gpr + 1) begin : g_posrep_v
            assign v_rd_pos_rep[gpr*11 +: 11] = v_rd_pos_q[gpr];
        end
    endgenerate

    // =================================================================
    // Lane-chain KV address remap (KV_LANE_CHAIN=1 only).
    //
    // ONLY the KV RAM address is remapped. score_buf / exp_buf stay on
    // the DENSE pos_req -- they are contiguous scratch bounded by
    // CTX_MAX and must not inherit the sparse slot numbering.
    //
    // KV_LANE_CHAIN=0: g_lane is not elaborated; pos_rd is a plain alias
    // of pos_req, so both read sites keep the byte-identical assignment
    // `k_rd_pos_q[r] <= pos_req`. The per-stack KV_REPL replication is
    // untouched -- every stack still gets its own registered copy.
    // =================================================================
    logic [3:0]  lane_j_lat;
    logic [10:0] num_steps_lat;
    logic [10:0] ctx_prefix_lat;   // SHARED PREFIX (ctx_len_latched is the SWEEP)

    wire [11:0] sweep_len_w =
        kv_lane_chain_pkg::kv_lane_sweep_len({1'b0, ctx_len}, {1'b0, level_i});

    wire [10:0] pos_rd;
    generate
        if (KV_LANE_CHAIN != 0) begin : g_lane
            wire [11:0] pos_eff = kv_lane_chain_pkg::kv_lane_eff_pos(
                {1'b0, ctx_prefix_lat}, lane_j_lat,
                {1'b0, num_steps_lat},   {1'b0, pos_req});
            assign pos_rd = pos_eff[10:0];
            // synthesis translate_off
            always @(posedge clk) if (rst_n && state != S_IDLE) begin
                if (!kv_lane_chain_pkg::kv_lane_in_bounds(pos_eff, CTX_WIN))
                    $fatal(1, "assembly: lane-chain eff_pos %0d >= CTX_WIN %0d",
                           pos_eff, CTX_WIN);
            end
            // synthesis translate_on
        end else begin : g_dense
            assign pos_rd = pos_req;      // bit-identical legacy path
        end
    endgenerate

    // =================================================================
    // F3 — KV RESIDENCY MASK  (reports/results/ATTN_F3_KV_RESIDENCY_MASK.md)
    //
    // THE DEFECT: this datapath had NO residency concept. `pos_req` sweeps
    // [0, sweep_len) unconditionally and score_max_tracker admits every
    // score. A never-written K row dots to score_out == 0 EXACTLY, and 0 is
    // a COMPETITIVE score, not a neutral one: for the ~50% of heads whose
    // real Q·K is negative the padding TAKES running_max and the head output
    // collapses (measured gain 1.000/0.500/0.285/0.165/0.090/0.048 at ctx
    // 1/2/4/8/16/32 -- CTXN_ATTN_NUMERICS_RCA.md §5).
    // In the reference this is structurally impossible: attn.cpp:53,65 --
    // ctx_len is incremented BY kv_append, so an unwritten position cannot
    // exist. Here ctx_len is a host register (reg 0x10) with ZERO coupling
    // to what was actually written.
    //
    // MECHANISM: a per-SLOT written bitmap, one bit per absolute KV slot,
    // set on the same clock edge as the KV row write (kv_wr_v/kv_wr_slot
    // mirror the store's write port). NOT a high-water mark:
    //   * under the depth-N lane-chain a branch writes slot
    //     prefix + j*num_steps + i, so the written set is NON-CONTIGUOUS and
    //     interleaved across branches; a single "count" cannot describe it.
    //   * a bitmap is exact under ANY write order, including the pl_*
    //     preload port, a host that jumps ctx_len non-monotonically, and a
    //     future out-of-order tree walk. It is indexed by the SAME absolute
    //     slot the KV RAM is (pos_rd == the lane-chain eff_pos), so read and
    //     write agree by construction -- no second copy of the slot formula
    //     (the dual-source-table bug class kv_lane_chain.sv warns about).
    //   * cost: CTX_WIN FFs + one CTX_WIN:1 read mux. 1,024 FF = 0.02 % of
    //     the V80 FF pool.
    //
    // ONE SHARED BITMAP IS CORRECT FOR ALL 24 HEADS: the parent FSM does
    // T_APPEND(head h) BEFORE T_ATTN_START(head h), so at the moment head h
    // attends, slot `append_pos` has been written for head h. Every earlier
    // slot was written for every head in an earlier forward. The bitmap is
    // therefore a per-slot, head-invariant property.
    //
    // NOT CLEARED PER FORWARD -- residency is a SESSION property (it mirrors
    // the KV RAM contents, which survive `start`).
    //
    // *** F3-CLR AMENDMENT (2026-07-28) — READ THIS WITH THE LINE ABOVE. ***
    // "SESSION property" is right; the bug was that there was no way to end
    // a SESSION short of rst_n (= a physical cold-AC). A new SGLang request
    // IS a new session: it reuses slots 0..N for different context without
    // rewriting them all. `kv_res_clear` (reg 0x154 bit[0]) is that missing
    // session boundary. See the port comment above for the full RCA.
    //
    // DEPTH-0 / ctx=1 IS INERT BY CONSTRUCTION: the sweep is [0, ctx_len)
    // and the just-appended row lives at ctx_len-1, so with a host that
    // walks ctx_len 1,2,3,... every swept slot is resident and score_masked
    // == score_norm on every cycle -> BIT-IDENTICAL (proved by the depth-0
    // seal: frontier 8/8, G5 0xc59befef, G7p 0xd8d6d147, G8RESC 0x002c953c).
    //
    // RED ARM: -d KVRES_RED_NOMASK forces res_bit_now=1, restoring the
    // pre-fix behaviour bit-for-bit (sim-only define; the packaging tcl
    // clears verilog_define so it can never ship).
    // =================================================================
    localparam int RES_N  = CTX_WIN;
    localparam int RES_AW = (RES_N <= 1) ? 1 : $clog2(RES_N);
    // Most-negative INT32 == score_max_tracker's NEG_INF reset value. A real
    // score can never reach it: |score_out| <= ~6.3e8 (INT32 dot of Q5.10
    // operands) and score_norm = score_out >>> SCORE_NORM_SHR, so
    // score_norm ∈ ~[-2^20, 2^20]. -2^31 is therefore an unambiguous
    // "not resident" token, and because the tracker compares with a STRICT
    // `>` against its own NEG_INF reset, a masked score can never take
    // running_max even when it is the first position swept.
    localparam logic signed [31:0] SCORE_NEG_INF = 32'sh80000000;

    logic [RES_N-1:0] kv_resident;
    logic [11:0]      res_cnt_r;
    wire              res_slot_new = kv_wr_v && !kv_resident[kv_wr_slot[RES_AW-1:0]];

    // ---------------------------------------------------------------------
    // F3-CLR: the clear pulse, registered and (optionally) replicated.
    //
    // ONE source register, RES_CLR_REP keep'd same-edge twins -> the clear
    // is ATOMIC over the whole bitmap (all bits clear on one edge) while the
    // ROUTE fan-out per net drops to RES_N/RES_CLR_REP. Registering costs a
    // 1-cycle delay on a purely host-issued control pulse that is never in
    // the same cycle as a launch, so it is free.
    //
    // RED ARM: -d KVRESCLR_RED_NOCLEAR makes the pulse inert (ties the twins
    // to 0), restoring the pre-fix set-only bitmap bit-for-bit. This is the
    // tooth for tb_kv_residency_multiseq: with it armed, the CLEAN phase
    // MUST collapse onto the DIRTY phase. Sim-only define; the packaging tcl
    // clears verilog_define so it can never ship.
    // ---------------------------------------------------------------------
    localparam int RES_CLR_REP_E = (RES_CLR_REP < 1) ? 1 : RES_CLR_REP;
    localparam int RES_CLR_SEG   = (RES_N + RES_CLR_REP_E - 1) / RES_CLR_REP_E;

`ifdef KVRESCLR_RED_NOCLEAR
    wire [RES_CLR_REP_E-1:0] res_clr_q = '0;   // RED: clear compiled OUT
`else
    (* keep = "true" *) logic [RES_CLR_REP_E-1:0] res_clr_q;
    always_ff @(posedge clk) begin
        if (!rst_n) res_clr_q <= '0;
        else        res_clr_q <= {RES_CLR_REP_E{kv_res_clear}};
    end
`endif

    // The set decoder, made explicit so the clear branch can express
    // "clear everything EXCEPT a slot written on this very edge" without a
    // second always_ff. At kv_res_clear==0 this is the identical 10:1024
    // address decoder the indexed assignment already synthesised.
    logic [RES_N-1:0] res_set_mask;
    always_comb begin
        res_set_mask = '0;
        if (kv_wr_v) res_set_mask[kv_wr_slot[RES_AW-1:0]] = 1'b1;
    end

    // SINGLE always_ff, SINGLE packed reg -> multi-driver-safe
    // (feedback_multi_driver_packed_reg_trap).
    //
    // PRIORITY (documented, deliberate): rst_n > clear > set, with ONE
    // exception that is not really an exception —
    //   CLEAR WINS over the RETAINED bitmap, but a KV row written on the
    //   SAME edge as the clear still registers its bit (res_set_mask is
    //   OR'd into the cleared value). Rationale: kv_wr_v means the K/V row
    //   IS in the RAM after this edge, so dropping its residency bit would
    //   mask a LIVE row to -INF -- a correctness hole strictly worse than
    //   the one being fixed. The host pulses the clear before any KV write
    //   of the new sequence, so the coincidence is not expected in the
    //   shipping flow; this branch just makes it safe if it ever happens.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            kv_resident <= '0;
            res_cnt_r   <= '0;
        end else begin
            for (int b = 0; b < RES_N; b++) begin
                if (res_clr_q[b / RES_CLR_SEG]) kv_resident[b] <= res_set_mask[b];
                else if (res_set_mask[b])       kv_resident[b] <= 1'b1;
            end
            if (res_clr_q[0]) res_cnt_r <= kv_wr_v ? 12'd1 : 12'd0;
            else if (res_slot_new) res_cnt_r <= res_cnt_r + 12'd1;
        end
    end
    assign kv_res_cnt = res_cnt_r;

`ifdef KVRES_RED_NOMASK
    // RED ARM (sim only): no residency concept -> the historical defect.
    wire res_bit_now = 1'b1;
`else
    wire res_bit_now = kv_resident[pos_rd[RES_AW-1:0]];
`endif

    // Residency of the position currently in flight. Latched at the state
    // that issues the KV read (MAX pass) / the score_buf read (EXP pass);
    // pos_req is stable from there until the position retires, so the bit
    // stays aligned with the score it belongs to across the ~15-cycle dot.
    logic res_lat;

    // ---- attn_score_dot (UNCHANGED — arithmetic untouched) ----
    logic                       score_start;
    logic signed [31:0]         score_out;
    logic                       score_done;

    attn_score_dot #(.D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH)) u_score (
        .clk(clk), .rst_n(rst_n),
        .start(score_start),
        .q_pack(q_vec_latched),
        .k_pack(k_vec_in),
        .score_out(score_out),
        .done(score_done)
    );

    localparam int TEMP_MUL = 181;
    localparam int TEMP_SHR = 19;
    wire signed [47:0] score_mul = $signed(score_out) * TEMP_MUL;
    wire signed [31:0] score_norm =
          (DEPTH_TEMP_FIX != 0) ? $signed(score_mul >>> TEMP_SHR)
        : (SCORE_NORM_SHR == 0) ? score_out
        :                         $signed(score_out >>> SCORE_NORM_SHR);

    // F3: THE MASK. A non-resident slot presents -INF instead of its
    // dot-against-garbage. Consumed by BOTH softmax consumers:
    //   (a) score_max_tracker  -- -INF can never take running_max;
    //   (b) score_buf -> softmax_exp_sum -- the EXP pass additionally
    //       DE-ASSERTS exp_valid_in for the slot, so it adds nothing to
    //       sum_so_far. That matters because exp_pwl_q12.sv:79 floors the
    //       LUT at 1/4096 (never 0) -- masking the score alone would still
    //       leak 2.441e-4 of denominator per hole. Gating valid_in makes
    //       the fix independent of the sibling exp-floor change (F2).
    wire signed [31:0] score_masked = res_lat ? score_norm : SCORE_NEG_INF;

    // ---- score_max_tracker (arithmetic UNCHANGED; input is now masked) ----
    logic                       max_clear;
    logic                       max_valid;
    wire signed [31:0]          running_max;

    score_max_tracker u_max (
        .clk(clk), .rst_n(rst_n),
        .clear(max_clear), .valid(max_valid),
        .score(score_masked), .running_max(running_max)
    );

    // =================================================================
    // FIX-SBUF: pass-1 score buffer (32b x CTX_MAX = 1 RAMB36).
    // Written once per position in S_MAX_WAIT (on score_done, when
    // score_out is freshly latched and stable); read back in the EXP
    // pass instead of recomputing the dot. Registered read (RD_LAT=1
    // from raddr latch), consumed one full state later — the same
    // discipline as exp_buf (no comb-read stale-by-1 class).
    // =================================================================
    (* ram_style = "block" *)
    logic signed [31:0] score_buf [0:CTX_MAX-1];
    logic               sb_we;
    logic [CTX_W-1:0]   sb_waddr;
    logic [CTX_W-1:0]   sb_raddr;
    logic signed [31:0] sb_rdata;

    // F3: the buffer stores the MASKED score, so pass 2 sees exactly what
    // pass 1's max tracker saw (one value, one source -- a hole can never
    // be resurrected by the EXP pass even if its valid gate were bypassed).
    always_ff @(posedge clk) begin
        if (sb_we) score_buf[sb_waddr] <= score_masked;
        sb_rdata <= score_buf[sb_raddr];
    end

    // ---- softmax_exp_sum: score_in = the BUFFERED pass-1 score ----
    logic                       exp_clear;
    logic                       exp_valid_in;
    wire        [31:0]          exp_val;
    wire                        exp_valid;
    wire  signed [47:0]         sum_so_far;

    softmax_exp_sum u_softmax (
        .clk(clk), .rst_n(rst_n),
        .clear(exp_clear), .valid_in(exp_valid_in),
        .score_in(sb_rdata), .max_q(running_max),   // FIX-SBUF: was score_norm
        .exp_val(exp_val), .exp_valid(exp_valid), .sum_so_far(sum_so_far)
    );

    // ---- exp_buf (UNCHANGED) ----
    (* ram_style = "block" *)
    logic [15:0] exp_buf [0:CTX_MAX-1];
    logic                       exp_buf_we;
    logic [CTX_W-1:0]           exp_buf_waddr;
    logic [CTX_W-1:0]           exp_buf_raddr;
    logic [15:0]                exp_buf_rdata;
    // F3: residency of the position exp_buf_waddr refers to, captured in the
    // SAME state that captures exp_buf_waddr so the two can never skew.
    // A hole must be written as an EXPLICIT 0, not skipped: exp_buf is a
    // BRAM that retains the PREVIOUS forward's value at that address, so a
    // skipped write would combine stale weight into acc[d].
    logic                       exp_buf_wres;

    always_ff @(posedge clk) begin
        if (exp_buf_we) exp_buf[exp_buf_waddr] <= exp_buf_wres ? exp_val[15:0] : 16'd0;
        exp_buf_rdata <= exp_buf[exp_buf_raddr];
    end

    // ---- inv_q12 (UNCHANGED) ----
    logic                       inv_start;
    logic                       inv_busy;
    logic                       inv_done;
    logic signed [15:0]         inv_sum;

    inv_q12 u_inv (
        .clk(clk), .rst_n(rst_n),
        .start(inv_start), .sum_q12(sum_so_far),
        .busy(inv_busy), .done(inv_done), .inv_q12_out(inv_sum)
    );

    // ---- attn_v_combine_perbank (UNCHANGED) ----
    logic                       combine_clear;
    logic                       combine_valid;
    logic signed [15:0]         combine_attn_q12;

    attn_v_combine_perbank #(
        .D_HEAD(D_HEAD), .A_WIDTH(A_WIDTH), .N_STACKS(N_STACKS)
    ) u_combine (
        .clk(clk), .rst_n(rst_n),
        .clear(combine_clear),
        .valid_in(combine_valid),
        .round_en(b2p_round_en),        // B2' floor (b): (attn*V+2048)>>12
        .stack_sel(stack_sel_latched),
        .attn_q12(combine_attn_q12),
        .v_rd_bus_flat(v_rd_bus_flat),
        .out_vec(out_vec)
    );

    function automatic signed [15:0] sat16_q12;
        input signed [31:0] v;
        begin
            logic signed [31:0] shifted;
            shifted = {{12{v[31]}}, v[31:12]};
            if (shifted > 32'sd32767)         sat16_q12 = 16'sd32767;
            else if (shifted < -32'sd32768)   sat16_q12 = 16'sh8000;
            else                              sat16_q12 = shifted[15:0];
        end
    endfunction

    logic signed [31:0] combine_attn_full;
    always_comb begin
        combine_attn_full = $signed({1'b0, exp_buf_rdata}) * $signed(inv_sum);
        // B2' floor (a): round-half the weight truncation. exp (>=0, <=~4096)
        // * inv (>=0, <=32767) < 2^28, so +2048 can never overflow 32b.
        if (b2p_round_en) combine_attn_full = combine_attn_full + 32'sd2048;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            combine_attn_q12 <= 16'sd0;
            combine_valid    <= 1'b0;
        end else begin
            combine_attn_q12 <= sat16_q12(combine_attn_full);
            combine_valid    <= (state == S_COMB_MAC);
        end
    end

    // ---- FSM ----
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            pos_req          <= '0;
            ctx_len_latched  <= '0;
            q_vec_latched    <= '0;
            stack_sel_latched<= '0;
            kv_rd_en         <= 1'b0;
            kv_rd_kind       <= 1'b0;
            for (int r = 0; r < N_STACKS; r++) k_rd_pos_q[r] <= '0;
            for (int r = 0; r < V_NREP;   r++) v_rd_pos_q[r] <= '0;
            score_start      <= 1'b0;
            max_clear        <= 1'b0;
            max_valid        <= 1'b0;
            exp_clear        <= 1'b0;
            exp_valid_in     <= 1'b0;
            exp_buf_we       <= 1'b0;
            exp_buf_waddr    <= '0;
            exp_buf_raddr    <= '0;
            exp_buf_wres     <= 1'b0;
            res_lat          <= 1'b0;
            sb_we            <= 1'b0;
            sb_waddr         <= '0;
            sb_raddr         <= '0;
            inv_start        <= 1'b0;
            combine_clear    <= 1'b0;
            done             <= 1'b0;
        end else begin
            kv_rd_en      <= 1'b0;
            score_start   <= 1'b0;
            max_clear     <= 1'b0;
            max_valid     <= 1'b0;
            exp_clear     <= 1'b0;
            exp_valid_in  <= 1'b0;
            exp_buf_we    <= 1'b0;
            sb_we         <= 1'b0;
            inv_start     <= 1'b0;
            combine_clear <= 1'b0;
            done          <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        // KV_LANE_CHAIN=0 -> sweep_len_w is not used; the
                        // latch is byte-identical to the legacy assignment.
                        ctx_len_latched   <= (KV_LANE_CHAIN != 0)
                                             ? sweep_len_w[10:0] : ctx_len;
                        ctx_prefix_lat    <= ctx_len;
                        lane_j_lat        <= lane_j;
                        num_steps_lat     <= num_steps;
                        q_vec_latched     <= q_vec;
                        stack_sel_latched <= stack_sel;
                        max_clear         <= 1'b1;
                        exp_clear         <= 1'b1;
                        combine_clear     <= 1'b1;
                        pos_req           <= '0;
                        state             <= S_MAX_RD;
                    end
                end

                // ===== pass 1: K read + dot + max track + score_buf fill =====
                S_MAX_RD: begin
                    kv_rd_en   <= 1'b1;
                    kv_rd_kind <= 1'b0;
                    for (int r = 0; r < N_STACKS; r++)
                        k_rd_pos_q[r] <= pos_rd;   // == pos_req when KV_LANE_CHAIN=0
                    for (int r = 0; r < V_NREP; r++)
                        v_rd_pos_q[r] <= pos_rd;   // RTL WO#12: all group twins, same edge
                    // F3: capture this slot's residency with its KV address.
                    // pos_req is frozen until score_done (S_MAX_WAIT), so the
                    // bit is still this position's when score_norm lands.
                    res_lat    <= res_bit_now;
                    state      <= S_MAX_START;
                end
                S_MAX_START: begin
                    score_start <= 1'b1;
                    state       <= S_MAX_WAIT;
                end
                S_MAX_WAIT: begin
                    if (score_done) begin
                        max_valid <= 1'b1;
                        // FIX-SBUF: capture this position's score for pass 2.
                        // score_out latched this edge; stable through the write
                        // edge (next dot's score_out lands >=14 cycles away).
                        sb_we     <= 1'b1;
                        sb_waddr  <= pos_req[CTX_W-1:0];
                        if (pos_req == ctx_len_latched - 11'd1) begin
                            pos_req <= '0;
                            state   <= S_EXP_ARD;
                        end else begin
                            pos_req <= pos_req + 11'd1;
                            state   <= S_MAX_RD;
                        end
                    end
                end

                // ===== pass 2: buffered-score exp (NO K read, NO dot) =====
                S_EXP_ARD: begin
                    sb_raddr <= pos_req[CTX_W-1:0];
                    // F3: same slot, same residency lookup as the MAX pass
                    // (kv_resident is monotone within a pass -- no KV write
                    // happens between T_APPEND and attention `done`).
                    res_lat  <= res_bit_now;
                    state    <= S_EXP_AWAIT;
                end
                S_EXP_AWAIT: begin
                    // sb_rdata commits this edge; valid during S_EXP_BUF —
                    // exp_valid_in aligns valid_in with valid score_in exactly
                    // as the baseline aligned it with the recomputed score.
                    // F3: a non-resident slot is NOT presented to
                    // softmax_exp_sum at all -> it contributes 0 to
                    // sum_so_far (not exp_pwl's 1/4096 floor).
                    exp_valid_in <= res_lat;
                    state        <= S_EXP_BUF;
                end
                S_EXP_BUF: begin
                    exp_buf_we    <= 1'b1;
                    exp_buf_waddr <= pos_req[CTX_W-1:0];
                    exp_buf_wres  <= res_lat;
                    if (pos_req == ctx_len_latched - 11'd1) begin
                        inv_start <= 1'b1;
                        state     <= S_INV;
                    end else begin
                        pos_req <= pos_req + 11'd1;
                        state   <= S_EXP_ARD;
                    end
                end

                S_INV: begin
                    if (inv_done) begin
                        pos_req       <= '0;
                        exp_buf_raddr <= '0;
                        state         <= S_COMB_RD;
                    end
                end

                // ===== pass 3: V combine (UNCHANGED) =====
                S_COMB_RD: begin
                    kv_rd_en      <= 1'b1;
                    kv_rd_kind    <= 1'b1;
                    for (int r = 0; r < N_STACKS; r++)
                        k_rd_pos_q[r] <= pos_rd;   // == pos_req when KV_LANE_CHAIN=0
                    for (int r = 0; r < V_NREP; r++)
                        v_rd_pos_q[r] <= pos_rd;   // RTL WO#12: all group twins, same edge
`ifdef KVADDR_REP_RED
                    // RED-ARM TOOTH (compile-out; NEVER in a build): model the
                    // exact per-GROUP failure the WO#12 gate must catch — ONE
                    // bank-group replica of stack 3 latching a wrong V read
                    // position on the combine pass. A gate that stays golden
                    // with this armed has NO teeth on the group twins (at
                    // KV_ADDR_REP=1 this index degenerates to the historical
                    // KVREPL_RED stack-3 tooth).
                    v_rd_pos_q[3*KV_AR + (KV_AR-1)] <= pos_rd ^ 11'h001;
`endif
                    exp_buf_raddr <= pos_req[CTX_W-1:0];
                    state         <= S_COMB_WAIT;
                end
                S_COMB_WAIT: begin
                    state <= S_COMB_MAC;
                end
                S_COMB_MAC: begin
                    state <= S_COMB_MAC2;
                end
                S_COMB_MAC2: begin
                    if (pos_req == ctx_len_latched - 11'd1) begin
                        state <= S_DRAIN_COMB;
                    end else begin
                        pos_req <= pos_req + 11'd1;
                        state   <= S_COMB_RD;
                    end
                end

                S_DRAIN_COMB: begin
                    state <= S_DRAIN_COMB2;
                end
                S_DRAIN_COMB2: begin
                    done  <= 1'b1;
                    state <= S_DONE;
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // =================================================================
    // F3 residency OBSERVABILITY (simulation only — zero netlist cells).
    //
    // This is the ctx_len <-> residency COUPLING the reference gets for
    // free (attn.cpp:53 -- ctx_len is incremented BY kv_append) and this
    // RTL never had. It does not gate anything; it makes a holey window
    // VISIBLE so a ctx>1 measurement can never again be read as a datapath
    // verdict when it was really a padding artifact.
    //   [KVRES] HOLES  ... sweep contains non-resident slots (masked out)
    //   [KVRES] EMPTY  ... NOTHING in the sweep is resident (attention has
    //                      no data at all -- always a wiring bug)
    // The gate greps these: the depth-0 seal must show NEITHER; the
    // holes arm must show HOLES.
    // =================================================================
    // synthesis translate_off
    int unsigned kvres_pass_res;
    int unsigned kvres_pass_tot;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            kvres_pass_res <= 0;
            kvres_pass_tot <= 0;
        end else begin
            if (state == S_IDLE && start) begin
                kvres_pass_res <= 0;
                kvres_pass_tot <= 0;
            end else if (state == S_MAX_RD) begin
                kvres_pass_tot <= kvres_pass_tot + 1;
                if (res_bit_now) kvres_pass_res <= kvres_pass_res + 1;
            end else if (state == S_INV && inv_start) begin
                if (kvres_pass_res == 0)
                    $display("[KVRES] EMPTY  t=%0t sweep=%0d resident=0 res_cnt=%0d — attention window has NO written KV slot",
                             $time, kvres_pass_tot, res_cnt_r);
                else if (kvres_pass_res != kvres_pass_tot)
                    $display("[KVRES] HOLES  t=%0t sweep=%0d resident=%0d masked=%0d res_cnt=%0d",
                             $time, kvres_pass_tot, kvres_pass_res,
                             kvres_pass_tot - kvres_pass_res, res_cnt_r);
            end
        end
    end
    // synthesis translate_on

endmodule

`default_nettype wire
