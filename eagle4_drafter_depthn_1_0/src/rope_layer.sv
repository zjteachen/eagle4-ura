// =====================================================================
// rope_layer.sv
// Per-head RoPE rotation: applies the cos/sin LUT-based rotation to
// one Q (or K) head vector of D_HEAD=128 elements per cycle.
// =====================================================================
//
// Architecture:
//   - N_PAIRS=64 instances of rope_pair_rotate in parallel
//   - LUT (fabric ROM) holds cos/sin per (pos, pair_idx) = 2048 × 64 entries
//   - Per cycle: 1 head's rotation completes (all 64 pairs in lockstep)
//   - Caller time-multiplexes across N_HEADS heads
//
// Cost (LANES=N_PAIRS=64):
//   ~64 × ~80 LUT/rotator = ~5,120 LUT
//   ~64 × ~3 DSP = ~192 DSP (Q4.12 multiplies)
//   cos/sin LUT: async 64-way per-pair read → fabric ROM/mux tree in
//   practice (a "block" attribute is not honorable for an async read).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module rope_layer #(
    parameter int D_HEAD  = 128,
    parameter int N_PAIRS = 64,        // D_HEAD/2
    parameter int MAX_CTX = 2048,
    parameter int A_WIDTH = 16,
    // -----------------------------------------------------------------
    // WO-051 P2/P3 SLR-seam fix (ROPE_SLR_REG). Worst-path-2 #4 is the
    // 16-level RoPE DSP rotate chain whose INPUTS (vec_in / pos / start)
    // arrive across an SLR die cut (the wide vec_in slice is muxed in the
    // caller from q_in/k_in that land in another SLR). Registering the
    // module inputs ONE cycle here turns the long pre-DSP combinational
    // SLR-crossing arc into a reg->reg hop on the SLR1 side just before
    // the rotate, so the SLL delay is absorbed as +1 latency instead of
    // counting against setup.
    //   ROPE_SLR_REG=0 (DEFAULT) : combinational pass-through of the
    //       inputs == bit-AND-cycle identical to the historical golden.
    //   ROPE_SLR_REG=1           : +1 input-register stage. The cos/sin
    //       LUT already addresses off a +1-delayed pos (pos_d1), so we
    //       keep that relationship by feeding the LUT the DELAYED pos and
    //       starting the rotators off the DELAYED start; vec_in is delayed
    //       the SAME +1 so x_in/y_in stay co-aligned with cos/sin. `valid`
    //       is delayed +1 to co-align with vec_out (latency-skew rule:
    //       register valid AND data together, feedback_valid_data_latency_skew_class).
    //   Pure delay line on the seam -> BIT-EXACT to ROPE_SLR_REG=0 (the
    //       rotated bits are unchanged, only delayed by +1 cycle; the
    //       caller's rope_valid handshake absorbs the extra cycle).
    // -----------------------------------------------------------------
    parameter bit ROPE_SLR_REG = 1'b0,
    // -----------------------------------------------------------------
    // RTL WO#8 (run-11 lever): ROPE_CS_PIPE — pipeline the rotation cone.
    //
    // Run-9 routed timing (80 MHz usr clock, Midyard aved_depthn
    // impl_1/route_report_timing_summary_0.rpt): THE design worst path is
    //   u_rope/pos_d1_reg[10]_rep__2 -> u_rope/g_pair[12].u_rot/x_out_reg[12]
    //   slack -1.557 @ 12.500 ns req; data path 14.650 ns
    //   = 2.487 logic (12 levels, through DSP58) + 12.163 ROUTE (83%).
    // The single-cycle cone pos_d1 -> async cs_lut ROM read (fabric mux
    // tree) -> 4x DSP58 multiply -> add -> saturate -> x_out_reg is both
    // deep AND physically spread (64 pair rotators + their ROM slices).
    //
    //   ROPE_CS_PIPE=0 (module default): the historical single-cycle cone,
    //       bit-AND-cycle identical to the shipped golden.
    //   ROPE_CS_PIPE=1: one pipeline register cs_q per pair on the ROM
    //       output CUTS the cone in two:
    //         leg A: pos_d1 -> ROM mux tree -> cs_q          (no DSP)
    //         leg B: cs_q/vec_in_d2 -> DSP58 -> sat -> x_out (local)
    //       The rotate itself moves one cycle later, so the enable and the
    //       operand vector move WITH the cos/sin data (start_d2 / vec_in_d2)
    //       and `valid` extends to a 4-stage delay of start_s. Rotated BITS
    //       are unchanged — the module is bit-exact to ROPE_CS_PIPE=0 with
    //       +1 cycle latency, absorbed by every caller's rope_valid
    //       handshake (all callers wait for `valid`; none count cycles).
    //       HEAD-0 skew class (project_rope_head0_ctxlen_ge2_latency_bug)
    //       is re-proven by gates, not assumed: en/operands/cos-sin all
    //       advance to the SAME stage, and the RED arm
    //       (-d ROPE_CS_PIPE_RED_SKEW, sim-only) fires the rotator one
    //       cycle early against stale cs_q — it must FAIL ctx>=2 gates.
    //   NETLIST SIGNATURE (rule 13): 64 FDRE banks
    //       u_rope/g_lut_pipe.g_pair[*].cs_q_reg[31:0] present; rotator
    //       cos/sin inputs sourced from cs_q, NOT from the ROM mux tree.
    //   BRAM GUARD: at ROPE_CS_PIPE=1 the LUT is DECLARED ram_style=
    //       "distributed" — with a registered read Vivado could otherwise
    //       absorb cs_q into 64 private sync-read BRAMs (2 RAMB36 each =
    //       +128 RAMB36; BRAM is at the wall). The historical arm keeps the
    //       (ineffective-for-async) "block" attribute untouched.
    // -----------------------------------------------------------------
    parameter bit ROPE_CS_PIPE = 1'b0,
    // -----------------------------------------------------------------
    // RTL WO#8 companion (route-locality lever): ROPE_POS_REP — number of
    // explicit (* keep *) replicas of the pos_d1 LUT-address register, each
    // serving a contiguous group of N_PAIRS/ROPE_POS_REP pairs. Run 9 shows
    // Vivado auto-replicated pos_d1 x3 (`_rep__2`) and still routed 12.163ns
    // — explicit per-group replicas let the placer drop each copy next to
    // its 8-pair ROM slice. Latency-NEUTRAL and bit-exact: every replica
    // latches pos_s on the same edge with the same reset.
    //   ROPE_POS_REP=1 (module default): the historical single pos_d1 reg.
    //   ROPE_POS_REP=8: run-11 ship value (8 replicas x 8 pairs).
    //   NETLIST SIGNATURE: g_pos_rep.pos_d1_r_reg[k][10:0] FDRE banks,
    //       k = 0..ROPE_POS_REP-1 (keep = "true" stops re-merging).
    // -----------------------------------------------------------------
    parameter int ROPE_POS_REP = 1
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,           // 1 cycle pulse
    input  wire        [10:0]                pos,             // KV position
    input  wire signed [D_HEAD*A_WIDTH-1:0]  vec_in,          // one head vector
    output logic signed [D_HEAD*A_WIDTH-1:0] vec_out,         // rotated
    output reg                               valid
);

    // -----------------------------------------------------------------
    // ROPE_SLR_REG input seam: register the SLR-crossing inputs one cycle
    // (single-driver shift, multi-driver-safe). When disabled, a pure
    // combinational pass-through (no added latency, identical to golden).
    // -----------------------------------------------------------------
    wire                              start_s;
    wire        [10:0]                pos_s;
    wire signed [D_HEAD*A_WIDTH-1:0]  vec_in_s;

    generate
    if (ROPE_SLR_REG) begin : g_in_seam
        (* keep = "true", shreg_extract = "no" *)
        reg                              start_q;
        (* keep = "true", shreg_extract = "no" *)
        reg        [10:0]                pos_q;
        (* keep = "true", shreg_extract = "no" *)
        reg signed [D_HEAD*A_WIDTH-1:0]  vec_in_q;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                start_q  <= 1'b0;
                pos_q    <= '0;
                vec_in_q <= '0;
            end else begin
                start_q  <= start;
                pos_q    <= pos;
                vec_in_q <= vec_in;
            end
        end
        assign start_s  = start_q;
        assign pos_s    = pos_q;
        assign vec_in_s = vec_in_q;
    end else begin : g_in_passthru
        assign start_s  = start;
        assign pos_s    = pos;
        assign vec_in_s = vec_in;
    end
    endgenerate

    // ----------------------------------------------------------------
    // Cos/Sin LUT (driven by $readmemh from /home/y344shi/eagle4_artifacts/p4/rope_cs.hex)
    // 2048 positions × 64 pairs × 32 bits packed (cos16 || sin16).
    // Declared once per ROPE_CS_PIPE generate arm (below) so the historical
    // arm keeps its declaration byte-for-byte while the piped arm pins
    // ram_style="distributed" (see BRAM GUARD note above).
    // ----------------------------------------------------------------
    localparam int LUT_DEPTH = MAX_CTX * N_PAIRS;

    // ----------------------------------------------------------------
    // Per-pair (cos, sin) lookup — combinational from registered pos.
    // pos_d1 is pos_s delayed ONE cycle (the cos/sin LUT address register).
    // ROPE_POS_REP>1 keeps the SAME single-cycle relationship but builds
    // POS_NREP physical copies (identical value each edge) so the 64-pair
    // address fanout is served locally per pair group.
    // ----------------------------------------------------------------
    localparam int POS_NREP =
        (ROPE_POS_REP < 1)       ? 1 :
        (ROPE_POS_REP > N_PAIRS) ? N_PAIRS : ROPE_POS_REP;
    localparam int PAIRS_PER_REP = (N_PAIRS + POS_NREP - 1) / POS_NREP;

    wire [10:0] pos_d1_pair [0:N_PAIRS-1];

    generate
    if (POS_NREP == 1) begin : g_pos_single
        // historical single register (name kept: pos_d1)
        reg [10:0] pos_d1;
        always_ff @(posedge clk) begin
            if (!rst_n) pos_d1 <= 0;
            else        pos_d1 <= pos_s;
        end
        for (genvar pp = 0; pp < N_PAIRS; pp = pp + 1) begin : g_fan
            assign pos_d1_pair[pp] = pos_d1;
        end
    end else begin : g_pos_rep
        // W3-C1-P1 idiom: same next-state value, same cycle, every replica;
        // (* keep *) stops synthesis from merging them back.
        (* keep = "true" *) reg [10:0] pos_d1_r [0:POS_NREP-1];
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                for (int r = 0; r < POS_NREP; r++) pos_d1_r[r] <= '0;
            end else begin
                for (int r = 0; r < POS_NREP; r++) pos_d1_r[r] <= pos_s;
            end
        end
        for (genvar pp = 0; pp < N_PAIRS; pp = pp + 1) begin : g_fan
            assign pos_d1_pair[pp] = pos_d1_r[pp/PAIRS_PER_REP];
        end
    end
    endgenerate

    // -----------------------------------------------------------------
    // CRITICAL ALIGNMENT (head-0 cold-pipeline RoPE bug):
    // The cos/sin LUT is addressed by pos_d1 (pos_s delayed +1). The
    // rotator must therefore be ENABLED, and consume vec_in, on the SAME
    // cycle its cos/sin operands are valid:
    //   ROPE_CS_PIPE=0: cos/sin are combinational from pos_d1 -> rotate at
    //       start_d1 with vec_in_d1 (one cycle after start_s). The original
    //       pre-fix code enabled the rotator with start_s (T0) while cos/sin
    //       were still for pos_d1's PREVIOUS value (stale): on the first
    //       call after reset pos_d1==0 => identity rotation (q_rot ==
    //       q_raw); on later calls pos_d1 happened to already hold the
    //       (constant) pos so it looked correct -> the first forward's
    //       head-0 rotation silently differed from every subsequent
    //       forward's.
    //   ROPE_CS_PIPE=1: cos/sin are REGISTERED (cs_q, valid at T2) -> the
    //       rotate advances WITH them to start_d2/vec_in_d2 (T2). Same
    //       invariant, one stage later.
    // (feedback_valid_data_latency_skew_class: drive the operands and the
    // table-address pipeline on the SAME stage.)
    // -----------------------------------------------------------------
    reg                              start_d1;
    reg signed [D_HEAD*A_WIDTH-1:0]  vec_in_d1;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            start_d1  <= 1'b0;
            vec_in_d1 <= '0;
        end else begin
            start_d1  <= start_s;
            vec_in_d1 <= vec_in_s;
        end
    end

    // Rotator enable + operand vector, at the stage matching the cos/sin.
    wire                              rot_en;
    wire signed [D_HEAD*A_WIDTH-1:0]  rot_vec;
    generate
    if (ROPE_CS_PIPE) begin : g_rot_d2
        (* keep = "true", shreg_extract = "no" *)
        reg                              start_d2;
        (* shreg_extract = "no" *)
        reg signed [D_HEAD*A_WIDTH-1:0]  vec_in_d2;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                start_d2  <= 1'b0;
                vec_in_d2 <= '0;
            end else begin
                start_d2  <= start_d1;
                vec_in_d2 <= vec_in_d1;
            end
        end
`ifdef ROPE_CS_PIPE_RED_SKEW
        // RED ARM (sim-only, gates must FAIL): fire the rotator one cycle
        // EARLY (T1) while cos/sin are piped (cs_q settles at T2) — the
        // exact head-0 latency-skew class. First call after reset rotates
        // with the reset/previous-pos cs_q (≈identity); ctx>=2
        // distinct-token gates must catch it (ctx=1 is blind by
        // construction, feedback_ctx1_blind_to_all_qk_path_defects).
        assign rot_en  = start_d1;
        assign rot_vec = vec_in_d1;
`else
        assign rot_en  = start_d2;
        assign rot_vec = vec_in_d2;
`endif
    end else begin : g_rot_d1
        assign rot_en  = start_d1;
        assign rot_vec = vec_in_d1;
    end
    endgenerate

    // Internal unpacked array — each generate iteration drives ONE entry
    // (legal: distinct drivers per entry, no slice contention).
    logic signed [A_WIDTH-1:0] vec_out_arr [0:D_HEAD-1];

    // Per-pair rotation. The cs_lut ROM is declared once per arm so the
    // ROPE_CS_PIPE=0 netlist keeps the historical declaration exactly.
    genvar p;
    generate
    if (ROPE_CS_PIPE) begin : g_lut_pipe
        (* ram_style = "distributed" *)
        reg [31:0] cs_lut [0:LUT_DEPTH-1];
        initial $readmemh("/home/y344shi/eagle4_artifacts/p4/rope_cs.hex", cs_lut);

        for (p = 0; p < N_PAIRS; p = p + 1) begin : g_pair
            // leg A: ROM mux tree only, cut by the cs_q register
            wire [31:0] cs_word_c = cs_lut[pos_d1_pair[p] * N_PAIRS + p];
            (* keep = "true", shreg_extract = "no" *)
            reg  [31:0] cs_q;
            always_ff @(posedge clk) begin
                if (!rst_n) cs_q <= '0;
                else        cs_q <= cs_word_c;   // free-running: tracks pos_d1
            end
            wire signed [A_WIDTH-1:0] cos_t = $signed(cs_q[31:16]);
            wire signed [A_WIDTH-1:0] sin_t = $signed(cs_q[15:0]);

            // leg B: local DSP58 rotate off registered operands
            wire signed [A_WIDTH-1:0] x_in = rot_vec[p*A_WIDTH +: A_WIDTH];
            wire signed [A_WIDTH-1:0] y_in = rot_vec[(p+N_PAIRS)*A_WIDTH +: A_WIDTH];

            wire signed [A_WIDTH-1:0] x_out, y_out;
            wire pair_valid;

            rope_pair_rotate u_rot (
                .clk(clk), .rst_n(rst_n),
                .en(rot_en),
                .x_in(x_in), .y_in(y_in),
                .cos_t(cos_t), .sin_t(sin_t),
                .x_out(x_out), .y_out(y_out),
                .valid(pair_valid)
            );

            // Each iteration drives unique array entries → legal multi-driver
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    vec_out_arr[p]         <= '0;
                    vec_out_arr[p+N_PAIRS] <= '0;
                end else if (pair_valid) begin
                    vec_out_arr[p]         <= x_out;
                    vec_out_arr[p+N_PAIRS] <= y_out;
                end
            end
        end
    end else begin : g_lut_hist
        (* ram_style = "block" *)
        reg [31:0] cs_lut [0:LUT_DEPTH-1];
        initial $readmemh("/home/y344shi/eagle4_artifacts/p4/rope_cs.hex", cs_lut);

        for (p = 0; p < N_PAIRS; p = p + 1) begin : g_pair
            wire [31:0] cs_word = cs_lut[pos_d1_pair[p] * N_PAIRS + p];
            wire signed [A_WIDTH-1:0] cos_t = $signed(cs_word[31:16]);
            wire signed [A_WIDTH-1:0] sin_t = $signed(cs_word[15:0]);

            wire signed [A_WIDTH-1:0] x_in = rot_vec[p*A_WIDTH +: A_WIDTH];
            wire signed [A_WIDTH-1:0] y_in = rot_vec[(p+N_PAIRS)*A_WIDTH +: A_WIDTH];

            wire signed [A_WIDTH-1:0] x_out, y_out;
            wire pair_valid;

            rope_pair_rotate u_rot (
                .clk(clk), .rst_n(rst_n),
                .en(rot_en),
                .x_in(x_in), .y_in(y_in),
                .cos_t(cos_t), .sin_t(sin_t),
                .x_out(x_out), .y_out(y_out),
                .valid(pair_valid)
            );

            // Each iteration drives unique array entries → legal multi-driver
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    vec_out_arr[p]         <= '0;
                    vec_out_arr[p+N_PAIRS] <= '0;
                end else if (pair_valid) begin
                    vec_out_arr[p]         <= x_out;
                    vec_out_arr[p+N_PAIRS] <= y_out;
                end
            end
        end
    end
    endgenerate

    // Single combinational concat from array → output reg
    always_comb begin
        for (int k = 0; k < D_HEAD; k++)
            vec_out[k*A_WIDTH +: A_WIDTH] = vec_out_arr[k];
    end

    // `valid` must co-align with the cycle vec_out_arr SETTLES.
    //
    // ROPE_CS_PIPE=0 (historical 3-stage pipe from start_s):
    //   start_s  @T0
    //     -> pos_d1 @T1, start_d1 @T1, vec_in_d1 @T1          [stage 1]
    //     -> rope_pair_rotate.x_out @T2 (en=start_d1@T1; pair_valid @T2)
    //                                       cos/sin from pos_d1@T1 (correct) [stage 2]
    //     -> vec_out_arr @T3 (latched on pair_valid@T2)        [stage 3]
    //   valid asserts at T3 = start_s delayed by 3.
    //
    // ROPE_CS_PIPE=1 (4-stage pipe from start_s):
    //   start_s  @T0
    //     -> pos_d1 @T1, start_d1 @T1, vec_in_d1 @T1          [stage 1]
    //     -> cs_q @T2 (ROM read off pos_d1@T1 — correct on the very first
    //        call after reset), start_d2 @T2, vec_in_d2 @T2    [stage 2]
    //     -> rope_pair_rotate.x_out @T3 (en=start_d2@T2; pair_valid @T3)
    //                                       cos/sin from cs_q@T2 (correct)  [stage 3]
    //     -> vec_out_arr @T4 (latched on pair_valid@T3)        [stage 4]
    //   valid asserts at T4 = start_s delayed by 4.
    //
    // The caller latches q_rot/k_rot on the valid edge; this guarantees it
    // captures the FULLY settled, CORRECTLY-rotated vector on every call —
    // including the very first call after reset (the head-0 cold-pipeline
    // case). ctx_len=1 is unaffected: softmax over one position = 1.0 =>
    // out=V[0] independent of Q; the extra valid latency only shifts the
    // handshake.
    generate
    if (ROPE_CS_PIPE) begin : g_valid4
        reg valid_p1, valid_p2, valid_p3;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                valid_p1 <= 1'b0;
                valid_p2 <= 1'b0;
                valid_p3 <= 1'b0;
                valid    <= 1'b0;
            end else begin
                valid_p1 <= start_s;     // T1: pos_d1/start_d1/vec_in_d1
                valid_p2 <= valid_p1;    // T2: cs_q/start_d2/vec_in_d2
                valid_p3 <= valid_p2;    // T3: rotate registered (x_out/pair_valid)
                valid    <= valid_p3;    // T4: vec_out_arr settled -> co-aligned
            end
        end
    end else begin : g_valid3
        reg valid_p1, valid_p2;
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                valid_p1 <= 1'b0;
                valid_p2 <= 1'b0;
                valid    <= 1'b0;
            end else begin
                valid_p1 <= start_s;     // T1
                valid_p2 <= valid_p1;    // T2: rotate registered (x_out/pair_valid)
                valid    <= valid_p2;    // T3: vec_out_arr settled -> co-aligned
            end
        end
    end
    endgenerate

endmodule

`default_nettype wire
