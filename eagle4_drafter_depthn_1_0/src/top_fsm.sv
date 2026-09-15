// top_fsm.sv: top-level forward-pass controller.
//
// Stage order:
//   G1    fc_in (fp16 -> 3072)
//   G4    QKV projection (3072 -> 6144), split into Q/K/V
//   G5    RoPE and KV-cache append
//   G6    output projection (3072 -> 3072)
//   G7g   gate projection (3072 -> 8192)
//   G7u   up projection (3072 -> 8192)
//   G7'   SiLU(gate) * up
//   G9    down projection (8192 -> 3072)
//   G11   output projection (3072 -> 3072)
//   G12   language-model head (3072 -> 128256)
//   TREE  tree decode and top-K selection
//
// The 2-bit GEMV stages share a 128-lane PE array selected by stage_sel.
// G1 and G11 use the dedicated fp16 GEMV unit.
// The FSM provides stage sequencing and start/done handshakes.

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module top_fsm #(
    // Optional G8 frontier-rescore stage.
    // 0: TREE -> DONE; 1: TREE -> G8 -> DONE.
    parameter bit G8_SPLICE = 1'b0,

    // Optional depth-N tree level loop.
    // 0: single tree pass; 1: process up to max_depth levels.
    // Level 0 uses one body pass; later levels use K_F passes.
    parameter bit DEPTH_N = 1'b0,
    parameter int K_F     = 8,

    // Optional child-embedding gather before depth-N child passes.
    // Requires DEPTH_N=1; level-0 passes use the host-loaded embedding.
    parameter bit EMBG_EN = 1'b0,

    // Optional KV-only launch mode.
    // When enabled and selected at runtime, stop after G5.
    parameter bit KVONLY_EN = 1'b0
) (
    input  wire  clk,
    input  wire  rst_n,

    // Top-level start and completion handshake.
    input  wire  start,
    output reg   busy,
    output reg   done,

    // Depth-N level-loop control and status.
    input  wire [31:0]           max_depth,
    // KV-only mode is sampled at launch.
    input  wire                  kvonly_i,
    output reg                   lvl_start,      // one pulse per tree
    output reg  [2:0]             branch_j,       // current frontier branch
    output reg  [3:0]             level_i,        // current tree level
    output reg                   branch_feed,    // current branch ready
    input  wire                  level_done,     // level selection complete
    input  wire                  level_last,     // level_done marks final level
    input  wire                  tree_all_done,  // tree-level controller done
    output reg                   out_walk_start, // start output node walk
    input  wire                  out_walk_done,

    // Child-embedding gather handshake.
    output reg                   emb_start,
    input  wire                  emb_done,

    // Per-stage start/done handshakes.
    output reg   g1_start,   input  wire  g1_done,
    output reg   g4_start,   input  wire  g4_done,
    output reg   g5_start,   input  wire  g5_done,    // RoPE + KV append
    output reg   g6_start,   input  wire  g6_done,
    output reg   g7g_start,  input  wire  g7g_done,
    output reg   g7u_start,  input  wire  g7u_done,
    output reg   g7p_start,  input  wire  g7p_done,   // SiLU + multiply
    output reg   g9_start,   input  wire  g9_done,
    output reg   g11_start,  input  wire  g11_done,
    output reg   glm_start,  input  wire  glm_done,   // G11 language-model projection
    output reg   g12_start,  input  wire  g12_done,
    output reg   tree_start, input  wire  tree_done,
    // Optional G8 rescore handshake.
    output reg   g8_start,   input  wire  g8_done,

    // Shared GEMV stage selector.
    output reg [3:0] stage_sel
);

    typedef enum logic [4:0] {
        S_IDLE     = 5'd0,
        S_G1       = 5'd1,
        S_G4       = 5'd2,
        S_G5_ROPE  = 5'd3,
        S_G6       = 5'd4,
        S_G7G      = 5'd5,
        S_G7U      = 5'd6,
        S_G7_SILU  = 5'd7,
        S_G9       = 5'd8,
        S_G11      = 5'd9,
        S_G12      = 5'd10,
        S_TREE     = 5'd11,
        S_DONE     = 5'd12,
        S_GLM      = 5'd13,  // G11 language-model projection
        S_G8       = 5'd14,  // Optional frontier rescore
        // Depth-N level loop; unreachable when DEPTH_N=0.
        S_LVL_ARM  = 5'd15,  // Initialize level loop
        S_LVL_FEED = 5'd16,  // Submit branch children
        S_LVL_SEL  = 5'd17,  // Select level frontier
        S_LVL_OUT  = 5'd18,  // Walk tree output
        S_LVL_EMBG = 5'd19   // Gather child embedding
    } state_t;

    state_t state, state_n;

    // Depth-N loop enable. Constant 0 when DEPTH_N=0.
    wire use_depth_loop = DEPTH_N && (max_depth > 32'd1);

    // Latch KV-only mode at launch.
    reg kvonly_q;
    always_ff @(posedge clk) begin
        if (!rst_n)               kvonly_q <= 1'b0;
        else if (state == S_IDLE) kvonly_q <= kvonly_i;
    end
    wire kvonly_act = KVONLY_EN && kvonly_q;

    // Child-embedding gather enable. Requires the depth-N loop.
    wire use_embg = EMBG_EN && use_depth_loop;
    // Final-level indication.
    wire tree_last_w = level_last || tree_all_done;

    // Number of body passes for the current level.
    localparam int BJ_W  = $clog2(K_F);       // branch index width
    localparam int BJC_W = BJ_W + 1;          // counter width
    wire [BJC_W-1:0] n_branch_this_level =
        (level_i == 4'd0) ? BJC_W'(1) : BJC_W'(K_F);
    wire last_branch = ((BJC_W'(branch_j) + BJC_W'(1)) >= n_branch_this_level);

    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= state_n;
    end

    always_comb begin
        state_n = state;
        case (state)
            // Arm the depth-N loop before the first body pass.
            // KV-only launches use the regular stage sequence.
            S_IDLE:    if (start)   state_n = (use_depth_loop && !kvonly_act)
                                              ? S_LVL_ARM : S_G1;
            S_LVL_ARM:              state_n = S_G1;
            S_G1:      if (g1_done)   state_n = S_G4;
            S_G4:      if (g4_done)   state_n = S_G5_ROPE;
            // KV-only launches stop after KV append.
            S_G5_ROPE: if (g5_done)   state_n = kvonly_act ? S_DONE : S_G6;
            S_G6:      if (g6_done)   state_n = S_G7G;
            S_G7G:     if (g7g_done)  state_n = S_G7U;
            S_G7U:     if (g7u_done)  state_n = S_G7_SILU;
            S_G7_SILU: if (g7p_done)  state_n = S_G9;
            S_G9:      if (g9_done)   state_n = S_G11;
            S_G11:     if (g11_done)  state_n = S_GLM;
            S_GLM:     if (glm_done)  state_n = S_G12;
            S_G12:     if (g12_done)  state_n = S_TREE;
            // Optional rescore, followed by the depth loop or DONE.
            S_TREE:    if (tree_done) state_n = G8_SPLICE ? S_G8
                                               : (use_depth_loop ? S_LVL_FEED : S_DONE);
            S_G8:      if (g8_done)   state_n = use_depth_loop ? S_LVL_FEED : S_DONE;

            // Submit each branch, then select the next level frontier.
            S_LVL_FEED:               state_n = last_branch ? S_LVL_SEL
                                              : (use_embg ? S_LVL_EMBG : S_G1);
            // Wait for child-embedding gather completion.
            S_LVL_EMBG: if (emb_done)  state_n = S_G1;
            // Select the level frontier or start the next level.
            S_LVL_SEL: if (level_done) state_n = tree_last_w ? S_LVL_OUT
                                              : (use_embg ? S_LVL_EMBG : S_G1);
            S_LVL_OUT: if (out_walk_done) state_n = S_DONE;

            S_DONE:                   state_n = S_IDLE;
            default:                  state_n = S_IDLE;
        endcase
    end

    // Depth-N counters and one-cycle control pulses.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            branch_j <= '0; level_i <= '0;
            lvl_start <= 1'b0; branch_feed <= 1'b0; out_walk_start <= 1'b0;
        end else begin
            // Default pulse values.
            lvl_start      <= 1'b0;
            branch_feed    <= 1'b0;
            out_walk_start <= 1'b0;

            // Reset loop counters at every new launch.
            if (state == S_IDLE && state_n != S_IDLE) begin
                branch_j  <= '0;
                level_i   <= '0;
            end
            if (state == S_IDLE && state_n == S_LVL_ARM) begin
                // Start the tree-level controller once per tree.
                lvl_start <= 1'b1;
            end

            if (state == S_LVL_FEED) begin
                branch_feed <= 1'b1;
                if (!last_branch) branch_j <= branch_j + 1'b1;
            end

            if (state == S_LVL_SEL && level_done && !tree_last_w) begin
                level_i  <= level_i + 4'd1;
                branch_j <= '0;
            end

            if (state == S_LVL_SEL && level_done && tree_last_w)
                out_walk_start <= 1'b1;
        end
    end

    // Generate one-cycle pulses when entering stage states.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            g1_start <= 0; g4_start <= 0; g5_start <= 0; g6_start <= 0;
            g7g_start <= 0; g7u_start <= 0; g7p_start <= 0; g9_start <= 0;
            g11_start <= 0; glm_start <= 0; g12_start <= 0; tree_start <= 0;
            g8_start  <= 0; emb_start <= 0;
        end else begin
            g1_start   <= (state_n == S_G1)      && (state != S_G1);
            g4_start   <= (state_n == S_G4)      && (state != S_G4);
            g5_start   <= (state_n == S_G5_ROPE) && (state != S_G5_ROPE);
            g6_start   <= (state_n == S_G6)      && (state != S_G6);
            g7g_start  <= (state_n == S_G7G)     && (state != S_G7G);
            g7u_start  <= (state_n == S_G7U)     && (state != S_G7U);
            g7p_start  <= (state_n == S_G7_SILU) && (state != S_G7_SILU);
            g9_start   <= (state_n == S_G9)      && (state != S_G9);
            g11_start  <= (state_n == S_G11)     && (state != S_G11);
            glm_start  <= (state_n == S_GLM)     && (state != S_GLM);
            g12_start  <= (state_n == S_G12)     && (state != S_G12);
            tree_start <= (state_n == S_TREE)    && (state != S_TREE);
            // Launch the optional frontier rescore.
            g8_start   <= (state_n == S_G8)      && (state != S_G8);
            // Launch child-embedding gather.
            emb_start  <= (state_n == S_LVL_EMBG) && (state != S_LVL_EMBG);
        end
    end

    // Select the weight-bank slice for the active GEMV stage.
    always_comb begin
        case (state)
            S_G4:  stage_sel = 4'd0;
            S_G6:  stage_sel = 4'd1;
            S_G7G: stage_sel = 4'd2;
            S_G7U: stage_sel = 4'd3;
            S_G9:  stage_sel = 4'd4;
            S_G12: stage_sel = 4'd5;
            default: stage_sel = 4'd0;
        endcase
    end

    // Top-level busy and done status.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 0; done <= 0;
        end else begin
            busy <= (state != S_IDLE) && (state != S_DONE);
            done <= (state == S_DONE);
        end
    end

endmodule

`default_nettype wire
