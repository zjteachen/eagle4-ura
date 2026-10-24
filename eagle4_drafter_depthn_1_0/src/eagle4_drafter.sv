
// EAGLE4 drafter top-level integration.
`timescale 1ns/1ps
`default_nettype none

// Parameter defaults are defined in the shared configuration header.
`include "eagle4_config.svh"

// Coordinates control, staged GEMV, attention, tree decode, and output DMA.
module eagle4_drafter #(
    // Datapath geometry and output ABI.
    parameter int LANES      = `EAGLE4_LANES,
    parameter int K_MAX      = `EAGLE4_K_MAX,
    parameter int K_PAR      = `EAGLE4_K_PAR,
    parameter int A_WIDTH    = `EAGLE4_A_WIDTH,
    parameter int TREE_V_MAX = `EAGLE4_TREE_V_MAX,

    parameter int BOOT_AXI_ADDR_W = `EAGLE4_BOOT_AXI_ADDR_W,
    parameter int BOOT_AXI_DATA_W = `EAGLE4_BOOT_AXI_DATA_W,
    parameter int BOOT_AXI_ID_W   = `EAGLE4_BOOT_AXI_ID_W,
    parameter int BOOT_BURST_LEN  = `EAGLE4_BOOT_BURST_LEN,

    parameter bit OUT_ABI_V24     = `EAGLE4_OUT_ABI_V24,
    parameter int OUT_TOP_K       = `EAGLE4_OUT_TOP_K,
    parameter int OUT_PARENT_SENTINEL = `EAGLE4_OUT_PARENT_SENTINEL,

    // Isolated stage inputs or connected forward seams.
    parameter bit CONNECT_SEAMS   = `EAGLE4_CONNECT_SEAMS,

    // Optional FP16 MLP tail; the default path is INT16.
    parameter bit TAIL_FP16       = `EAGLE4_TAIL_FP16,

    // Tree scan and GEMV implementation choices.
    parameter int  TREE_P         = `EAGLE4_TREE_P,

    parameter bit  TREE_PP        = `EAGLE4_TREE_PP,
    parameter bit  GEMV_KPAR64    = `EAGLE4_GEMV_KPAR64,

    // GEMV dequantization and GPTQ configuration.
    parameter bit  GEMV_SCALED    = `EAGLE4_GEMV_SCALED,
    parameter int  GEMV_SCALE_FRAC = `EAGLE4_GEMV_SCALE_FRAC,

    parameter logic [1:0] GPTQ_QZERO_RAW = `EAGLE4_GPTQ_QZERO_RAW,

    // Activation seam width and stage sequencing.
    parameter bit  SEAM_R1_FULLW  = `EAGLE4_SEAM_R1_FULLW,

    parameter bit  S7A_FRONT_SEQ  = `EAGLE4_S7A_FRONT_SEQ,

    parameter bit  BACK_SEQ       = `EAGLE4_BACK_SEQ,

    // Physical implementation and seam timing.
    parameter bit  SLR_SAFE        = `EAGLE4_SLR_SAFE,
    parameter int  ARR_GROUPS      = `EAGLE4_ARR_GROUPS,
    parameter int  ARR_PIPE_STAGES = `EAGLE4_ARR_PIPE_STAGES,

    parameter bit  SEAM_FF         = `EAGLE4_SEAM_FF,
    parameter bit  FF_ACT_CACHE    = `EAGLE4_FF_ACT_CACHE,

    parameter bit  DUAL_PORT       = `EAGLE4_DUAL_PORT,

    parameter int  BODY_BATCH_B    = `EAGLE4_BODY_BATCH_B,

    parameter bit  SEAM_FF_LAND_P  = `EAGLE4_SEAM_FF_LAND_P,
    parameter bit  TREE_SEAM_FF    = `EAGLE4_TREE_SEAM_FF,

    parameter string LMDOWN_MEM_DIR_P = "/home/y344shi/eagle4_artifacts/p4_8815",

    parameter bit  D1_LAT          = `EAGLE4_D1_LAT,

    parameter bit  D1_RDLAT        = `EAGLE4_D1_RDLAT,
    parameter bit  D1_WLAT         = `EAGLE4_D1_WLAT,

    parameter bit  R4_BANKSPLIT     = `EAGLE4_R4_BANKSPLIT,

    parameter int  CTX_WIN         = `EAGLE4_CTX_WIN,

    // Optional depth-N execution and child-input paths.
    parameter bit  DEPTH_N         = `EAGLE4_DEPTH_N,

    parameter bit  EMBG_EN         = `EAGLE4_EMBG_EN,

    parameter int  EMB_NCH         = `EAGLE4_EMB_NCH,

    parameter bit  FORREASON       = `EAGLE4_FORREASON,
    parameter int  FR_KF_BANKS     = `EAGLE4_FR_KF_BANKS,

    parameter bit  FR_ABANK_BRAM   = `EAGLE4_FR_ABANK_BRAM,

    parameter int  HEAD_PAR        = `EAGLE4_HEAD_PAR,

    // Optional accuracy fixes and observability.
    parameter bit  ACCFIX_SEAM_P   = `EAGLE4_ACCFIX_SEAM_P,
    parameter bit  ACCFIX_B2P_P    = `EAGLE4_ACCFIX_B2P_P,
    parameter bit  ACCFIX_KC_P     = `EAGLE4_ACCFIX_KC_P,
    parameter bit  ACCFIX_KVONLY_P = `EAGLE4_ACCFIX_KVONLY_P,
    parameter bit  OBS_LVL_P       = `EAGLE4_OBS_LVL_P,
    parameter bit  ENORM_D0        = `EAGLE4_ENORM_D0,

    parameter bit  PRECISION_SAFE  = `EAGLE4_PRECISION_SAFE,

    parameter bit  SILU_RANGE_FIX_P = `EAGLE4_SILU_RANGE_FIX_P,

    // KV storage and attention implementation.
    parameter      KV_V_RAM_STYLE  = `EAGLE4_KV_V_RAM_STYLE,
    parameter      KV_K_RAM_STYLE  = `EAGLE4_KV_K_RAM_STYLE,

    parameter bit  COMBINE_LOCAL   = `EAGLE4_COMBINE_LOCAL,

    parameter bit  REQUANT_HEADMUX = `EAGLE4_REQUANT_HEADMUX,

    parameter bit  ROPE_CS_PIPE    = `EAGLE4_ROPE_CS_PIPE,
    parameter int  ROPE_POS_REP    = `EAGLE4_ROPE_POS_REP,

    parameter int  KV_ADDR_REP     = `EAGLE4_KV_ADDR_REP,

    parameter int  KV_RES_CLR_REP  = `EAGLE4_KV_RES_CLR_REP,

    // Optional G8 vocabulary rescore path.
    parameter bit  G8_SPLICE       = `EAGLE4_G8_SPLICE,

    parameter int  G8_K_C          = `EAGLE4_G8_K_C,

    parameter int  G8_MAX_BURST    = `EAGLE4_G8_MAX_BURST,

    parameter bit  G8_PIPELINED    = `EAGLE4_G8_PIPELINED,

    parameter int  G8_OUTSTANDING  = `EAGLE4_G8_OUTSTANDING,

    parameter bit  G8_ROWPP        = `EAGLE4_G8_ROWPP,
    parameter bit  G8_PREFETCH     = `EAGLE4_G8_PREFETCH,

    parameter int  G8_DATA_W       = `EAGLE4_G8_DATA_W,

    parameter bit  G8_OPT1C        = `EAGLE4_G8_OPT1C,

    parameter bit  G8_OPT1D        = `EAGLE4_G8_OPT1D,
    parameter int  G8_MIN_NG       = `EAGLE4_G8_MIN_NG,

    parameter bit  G8_OPT1F        = `EAGLE4_G8_OPT1F,

    parameter bit  G8_SELPIPE      = `EAGLE4_G8_SELPIPE,

    parameter bit  G8_OPT1G        = `EAGLE4_G8_OPT1G,
    parameter bit  G8_WIDE         = `EAGLE4_G8_WIDE,
    parameter int  G8_WIDE_P       = `EAGLE4_G8_WIDE_P,
    parameter bit  G8_MROUND       = `EAGLE4_G8_MROUND,

    parameter bit  G8_SCALAR_II1   = `EAGLE4_G8_SCALAR_II1,

    parameter bit  G8_CONSUME_HARDEN = `EAGLE4_G8_CONSUME_HARDEN,

    parameter bit  G8_LANE_FIX     = `EAGLE4_G8_LANE_FIX,

    parameter bit  G8_WIDE_IL      = `EAGLE4_G8_WIDE_IL,
    parameter int  G8_WIDE_M       = `EAGLE4_G8_WIDE_M,

    parameter int  G8_NCH          = `EAGLE4_G8_NCH,

    // Input-buffer read timing and stage isolation.
    parameter bit  BUFRDREG        = `EAGLE4_BUFRDREG,

    parameter bit  STAGE_INJECT    = `EAGLE4_STAGE_INJECT
) (
    input  wire        ap_clk,
    input  wire        ap_rst_n,

    input  wire [8:0]  s_axi_control_awaddr,
    input  wire        s_axi_control_awvalid,
    output wire        s_axi_control_awready,
    input  wire [31:0] s_axi_control_wdata,
    input  wire [3:0]  s_axi_control_wstrb,
    input  wire        s_axi_control_wvalid,
    output wire        s_axi_control_wready,
    output wire [1:0]  s_axi_control_bresp,
    output wire        s_axi_control_bvalid,
    input  wire        s_axi_control_bready,
    input  wire [8:0]  s_axi_control_araddr,
    input  wire        s_axi_control_arvalid,
    output wire        s_axi_control_arready,
    output wire [31:0] s_axi_control_rdata,
    output wire [1:0]  s_axi_control_rresp,
    output wire        s_axi_control_rvalid,
    input  wire        s_axi_control_rready,

    // Host weight and activation loading.
    input  wire        boot_we,
    input  wire [15:0] boot_lane_sel,
    input  wire [16:0] boot_wr_addr,
    input  wire [31:0] boot_wr_data,

    // Simulation-only scale-memory loading.
    input  wire        tb_sc_we,
    input  wire [15:0] tb_sc_lane_sel,
    input  wire [16:0] tb_sc_wr_addr,
    input  wire [15:0] tb_sc_wr_data,

    // Boot-time AXI read master.
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_boot_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_boot_araddr,
    output wire [7:0]                 m_axi_boot_arlen,
    output wire [2:0]                 m_axi_boot_arsize,
    output wire [1:0]                 m_axi_boot_arburst,
    output wire                       m_axi_boot_arvalid,
    input  wire                       m_axi_boot_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_boot_rid,
    input  wire [BOOT_AXI_DATA_W-1:0] m_axi_boot_rdata,
    input  wire [1:0]                 m_axi_boot_rresp,
    input  wire                       m_axi_boot_rlast,
    input  wire                       m_axi_boot_rvalid,
    output wire                       m_axi_boot_rready,

    // Result AXI write master.
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_out_awid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_out_awaddr,
    output wire [7:0]                 m_axi_out_awlen,
    output wire [2:0]                 m_axi_out_awsize,
    output wire [1:0]                 m_axi_out_awburst,
    output wire                       m_axi_out_awvalid,
    input  wire                       m_axi_out_awready,
    output wire [BOOT_AXI_DATA_W-1:0] m_axi_out_wdata,
    output wire [BOOT_AXI_DATA_W/8-1:0] m_axi_out_wstrb,
    output wire                       m_axi_out_wlast,
    output wire                       m_axi_out_wvalid,
    input  wire                       m_axi_out_wready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_out_bid,
    input  wire [1:0]                 m_axi_out_bresp,
    input  wire                       m_axi_out_bvalid,
    output wire                       m_axi_out_bready,

    // External activation-source stream (activation_source = 0). The
    // operation configuration (including qzero) comes only from the actual
    // special-register bank latched by gemv_shared_ctrl.
    input  wire signed [K_PAR*A_WIDTH-1:0]  a_chunk,

    // G8 lm-head gather master.
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_g8_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_g8_araddr,
    output wire [7:0]                 m_axi_g8_arlen,
    output wire [2:0]                 m_axi_g8_arsize,
    output wire [1:0]                 m_axi_g8_arburst,
    output wire                       m_axi_g8_arvalid,
    input  wire                       m_axi_g8_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_g8_rid,

    input  wire [G8_DATA_W-1:0]       m_axi_g8_rdata,
    input  wire [1:0]                 m_axi_g8_rresp,
    input  wire                       m_axi_g8_rlast,
    input  wire                       m_axi_g8_rvalid,
    output wire                       m_axi_g8_rready,

    // Optional second G8 gather channel.
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_g8b_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_g8b_araddr,
    output wire [7:0]                 m_axi_g8b_arlen,
    output wire [2:0]                 m_axi_g8b_arsize,
    output wire [1:0]                 m_axi_g8b_arburst,
    output wire                       m_axi_g8b_arvalid,
    input  wire                       m_axi_g8b_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_g8b_rid,
    input  wire [G8_DATA_W-1:0]       m_axi_g8b_rdata,
    input  wire [1:0]                 m_axi_g8b_rresp,
    input  wire                       m_axi_g8b_rlast,
    input  wire                       m_axi_g8b_rvalid,
    output wire                       m_axi_g8b_rready,

    // Child-embedding gather master.
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_emb_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_emb_araddr,
    output wire [7:0]                 m_axi_emb_arlen,
    output wire [2:0]                 m_axi_emb_arsize,
    output wire [1:0]                 m_axi_emb_arburst,
    output wire                       m_axi_emb_arvalid,
    input  wire                       m_axi_emb_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_emb_rid,
    input  wire [255:0]               m_axi_emb_rdata,
    input  wire [1:0]                 m_axi_emb_rresp,
    input  wire                       m_axi_emb_rlast,
    input  wire                       m_axi_emb_rvalid,
    output wire                       m_axi_emb_rready,

    // Optional second embedding channel.
    output wire [BOOT_AXI_ID_W-1:0]   m_axi_embb_arid,
    output wire [BOOT_AXI_ADDR_W-1:0] m_axi_embb_araddr,
    output wire [7:0]                 m_axi_embb_arlen,
    output wire [2:0]                 m_axi_embb_arsize,
    output wire [1:0]                 m_axi_embb_arburst,
    output wire                       m_axi_embb_arvalid,
    input  wire                       m_axi_embb_arready,
    input  wire [BOOT_AXI_ID_W-1:0]   m_axi_embb_rid,
    input  wire [255:0]               m_axi_embb_rdata,
    input  wire [1:0]                 m_axi_embb_rresp,
    input  wire                       m_axi_embb_rlast,
    input  wire                       m_axi_embb_rvalid,
    output wire                       m_axi_embb_rready,

    // Status and observability outputs.
    output wire [3:0]                       stage_sel,
    output wire        [15:0]               array_chunk_idx_out,
    output wire                             array_chunk_idx_valid,
    output wire signed [32*LANES-1:0]       array_acc_out
);

    if (G8_WIDE && (G8_DATA_W != G8_WIDE_P*16) && (G8_DATA_W != G8_WIDE_P*16*2))
        $error("G8_WIDE=1 requires G8_DATA_W == G8_WIDE_P*16 (=%0d) or 2x (=%0d); got %0d",
               G8_WIDE_P*16, G8_WIDE_P*16*2, G8_DATA_W);

    wire start_pulse;
    wire kernel_busy;
    wire kernel_done;

    localparam int INJ_NSTAGE = 19;

    localparam int INJ_MAX_W   = 256;
    wire [INJ_NSTAGE-1:0]   inject_en;
    wire [INJ_MAX_W*32-1:0] inj_bus;
    wire        inj_en_we_w, inj_widx_we_w, inj_data_we_w, inj_sel_we_w;
    wire [31:0] inj_en_wdata_w, inj_widx_wdata_w, inj_data_wdata_w, inj_sel_wdata_w;
    wire [31:0] inj_en_rb_w, inj_widx_rb_w, inj_status_w;

    wire g4_done;
    wire g5_start;
    wire g6_start,  g6_done;
    wire g7g_start, g7g_done;
    wire g7u_start, g7u_done;
    wire g7p_start;
    wire g9_start,  g9_done;
    wire g11_start;
    wire glm_start, glm_done;
    wire g12_done;
    wire tree_start;
    wire g1_start;

    // Tree vocabulary scan and top-K selection.
    localparam int TREE_IDX_W  = 18;
    localparam int TREE_N_TILES = (TREE_V_MAX + LANES - 1) / LANES;

    wire signed  [8*32-1:0]         tree_topk_scores;
    wire         [8*TREE_IDX_W-1:0] tree_topk_indices;
    wire                            tree_done;

    wire [15:0] n_tiles_cfg;
    wire [15:0] kc_cfg;

    wire [3:0]  accfix_cfg_w;
    wire        accfix_seam_en   = ACCFIX_SEAM_P   && accfix_cfg_w[0];
    wire        accfix_b2p_en    = ACCFIX_B2P_P    && accfix_cfg_w[1];
    wire        accfix_kc_en     = ACCFIX_KC_P     && accfix_cfg_w[2];
    // kvonly truncation is now expressed by program content (a shorter
    // ISA program); the reg is kept for the depth-N re-implementation.
    wire        accfix_kvonly_en = ACCFIX_KVONLY_P && accfix_cfg_w[3];

    wire [10:0] obs_rd_idx_w;
    wire [31:0] obs_rd_data_w;
    wire [31:0] obs_status_word_w;

    wire        obs_cand_v_w;
    wire [31:0] obs_cand_prob_w;
    wire [17:0] obs_cand_tok_w;
    wire        obs_tree_arm_w;

    wire                         tt_dec_start;
    wire                         tt_valid_in;
    wire signed [31:0]           tt_score_in;
    wire        [TREE_IDX_W-1:0] tt_index_in;
    wire                         tt_last_in;

    wire [31:0] g8_sel_thresh;
    wire        g8_sel_thresh_valid;
    wire                         tt_gemv_start;
    wire        [15:0]           tt_gemv_tile_idx;
    wire                         tt_stage_g12;
    wire                         tt_busy;

    wire array_done;

    generate
    if (TREE_P <= 1) begin : g_tree_serial

        if (TREE_PP) begin : g_tiler_pp
        tree_tile_stream_pp #(
            .LANES(LANES), .V_MAX(TREE_V_MAX), .IDX_W(TREE_IDX_W), .SCORE_W(32),
            .N_TILES(TREE_N_TILES)
        ) u_tree_tiler (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .tree_start(tree_start_eff),
            .n_tiles_cfg(n_tiles_cfg),
            .array_acc_out(array_acc_out),
            .array_done(array_done),
            .gemv_tile_start(tt_gemv_start),
            .gemv_tile_idx(tt_gemv_tile_idx),
            .gemv_stage_g12(tt_stage_g12),
            .tree_dec_start(tt_dec_start),
            .valid_in(tt_valid_in),
            .score_in(tt_score_in),
            .index_in(tt_index_in),
            .last_in(tt_last_in),
            .busy(tt_busy)
        );
        end else begin : g_tiler_base
        tree_tile_stream #(
            .LANES(LANES), .V_MAX(TREE_V_MAX), .IDX_W(TREE_IDX_W), .SCORE_W(32),
            .N_TILES(TREE_N_TILES)
        ) u_tree_tiler (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .tree_start(tree_start_eff),
            .n_tiles_cfg(n_tiles_cfg),
            .array_acc_out(array_acc_out),
            .array_done(array_done),
            .gemv_tile_start(tt_gemv_start),
            .gemv_tile_idx(tt_gemv_tile_idx),
            .gemv_stage_g12(tt_stage_g12),
            .tree_dec_start(tt_dec_start),
            .valid_in(tt_valid_in),
            .score_in(tt_score_in),
            .index_in(tt_index_in),
            .last_in(tt_last_in),
            .busy(tt_busy)
        );
        end

        tree_decode #(.K_F(8), .V_MAX(TREE_V_MAX), .IDX_W(TREE_IDX_W), .SCORE_W(32)) u_tree (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(tt_dec_start),
            .valid_in(tt_valid_in),
            .score_in(tt_score_in),
            .index_in(tt_index_in),
            .last_in(tt_last_in),
            .topk_scores(tree_topk_scores),
            .topk_indices(tree_topk_indices),
            .done(tree_done)
        );
    end else begin : g_tree_pwide

        wire                        ttp_valid_in;
        wire signed [TREE_P*32-1:0] ttp_scores_in;
        wire [TREE_P*TREE_IDX_W-1:0] ttp_indices_in;
        wire [TREE_P-1:0]           ttp_lane_en;
        wire                        ttp_last_in;
        wire                        ttp_dec_start;

        if (TREE_PP) begin : g_tiler_pp
        tree_tile_stream_pkpar_pp #(
            .LANES(LANES), .P(TREE_P), .V_MAX(TREE_V_MAX),
            .IDX_W(TREE_IDX_W), .SCORE_W(32), .N_TILES(TREE_N_TILES)
        ) u_tree_tiler (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .tree_start(tree_start_eff),
            .n_tiles_cfg(n_tiles_cfg),
            .array_acc_out(array_acc_out),
            .array_done(array_done),
            .gemv_tile_start(tt_gemv_start),
            .gemv_tile_idx(tt_gemv_tile_idx),
            .gemv_stage_g12(tt_stage_g12),
            .tree_dec_start(ttp_dec_start),
            .valid_in(ttp_valid_in),
            .scores_in(ttp_scores_in),
            .indices_in(ttp_indices_in),
            .lane_en(ttp_lane_en),
            .last_in(ttp_last_in),
            .busy(tt_busy)
        );
        end else begin : g_tiler_base
        tree_tile_stream_pkpar #(
            .LANES(LANES), .P(TREE_P), .V_MAX(TREE_V_MAX),
            .IDX_W(TREE_IDX_W), .SCORE_W(32), .N_TILES(TREE_N_TILES)
        ) u_tree_tiler (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .tree_start(tree_start_eff),
            .n_tiles_cfg(n_tiles_cfg),
            .array_acc_out(array_acc_out),
            .array_done(array_done),
            .gemv_tile_start(tt_gemv_start),
            .gemv_tile_idx(tt_gemv_tile_idx),
            .gemv_stage_g12(tt_stage_g12),
            .tree_dec_start(ttp_dec_start),
            .valid_in(ttp_valid_in),
            .scores_in(ttp_scores_in),
            .indices_in(ttp_indices_in),
            .lane_en(ttp_lane_en),
            .last_in(ttp_last_in),
            .busy(tt_busy)
        );
        end

        tree_merge_pipe #(
            .K_F(8), .P(TREE_P), .V_MAX(TREE_V_MAX),
            .IDX_W(TREE_IDX_W), .SCORE_W(32)
        ) u_tree (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(ttp_dec_start),
            .valid_in(ttp_valid_in),
            .scores_in(ttp_scores_in),
            .indices_in(ttp_indices_in),
            .lane_en(ttp_lane_en),
            .last_in(ttp_last_in),
            .topk_scores(tree_topk_scores),
            .topk_indices(tree_topk_indices),
            .done(tree_done)
        );

        if (G8_SPLICE) begin : g_ser
            tree_cheap_serializer #(

                .P(TREE_P), .IDX_W(TREE_IDX_W), .SCORE_W(32), .DEPTH(8192),
                .CHEAP_KEY_INT32(1'b1)
            ) u_cheap_ser (
                .clk(ap_clk), .rst_n(ap_rst_n),
                .flush(tree_start),
                .valid_in(ttp_valid_in),
                .scores_in(ttp_scores_in),
                .indices_in(ttp_indices_in),
                .lane_en(ttp_lane_en),
                .last_in(ttp_last_in),
                .thresh(g8_sel_thresh),
                .thresh_valid(g8_sel_thresh_valid),
                .out_valid(tt_valid_in),
                .out_score(tt_score_in),
                .out_index(tt_index_in),
                .out_last(tt_last_in)
            );
        end else begin : g_ser_off

            assign tt_valid_in = 1'b0;
            assign tt_score_in = '0;
            assign tt_index_in = '0;
            assign tt_last_in  = 1'b0;
        end

        assign tt_dec_start = 1'b0;
    end
    endgenerate

    wire out_dma_busy;
    wire out_dma_done;
    wire out_dma_err;

    wire                            ow_trigger;
    wire signed  [8*32-1:0]         ow_topk_scores;
    wire         [8*TREE_IDX_W-1:0] ow_topk_indices;

    wire        dump_en;
    wire [63:0] out3_base;
    wire [63:0] out4_base;

    localparam int G8_CIDX_W = (G8_K_C <= 1) ? 1 : $clog2(G8_K_C);
    wire [G8_CIDX_W-1:0] g8_cand_rdaddr;
    wire [TREE_IDX_W-1:0] g8_cand_dump_id;
    wire [31:0]          g8_cand_dump_score;

    wire [31:0]          g8_bridge_score_ff;
    localparam int OW_DUMP3_AW = $clog2(G8_K_C*2);
    wire [OW_DUMP3_AW-1:0]   ow_dump3_rdaddr;
    wire [31:0]              ow_dump3_rddata;
    wire [8*32-1:0]          ow_tree_parents;
    wire [8*32-1:0]          ow_g8_rescored;

    wire [G8_CIDX_W-1:0] bridge_cand_rdaddr;
    wire                 bridge_cand_rd_active;

    wire [G8_CIDX_W-1:0] bridge_cand_rdaddr_nxt;
    wire [31:0]          bridge_cand_logit = TREE_SEAM_FF ? g8_bridge_score_ff
                                                          : g8_cand_dump_score;
`ifdef TREE_ARB_RED

    assign g8_cand_rdaddr  = bridge_cand_rd_active ? ow_dump3_rdaddr[OW_DUMP3_AW-1:1]
                                                   : ow_dump3_rdaddr[OW_DUMP3_AW-1:1];
`else
    assign g8_cand_rdaddr  = bridge_cand_rd_active ? bridge_cand_rdaddr
                                                   : ow_dump3_rdaddr[OW_DUMP3_AW-1:1];
`endif
    assign ow_dump3_rddata = ow_dump3_rdaddr[0]
                           ? g8_cand_dump_score
                           : {{(32-TREE_IDX_W){1'b0}}, g8_cand_dump_id};

    genvar owp;
    for (owp = 0; owp < 8; owp = owp + 1) begin : g_ow_parents
        assign ow_tree_parents[owp*32 +: 32] = 32'hFFFF_FFFF;
        assign ow_g8_rescored [owp*32 +: 32] = ow_topk_scores[owp*32 +: 32];
    end

    wire fr_busy, fr_done, fr_err;
    wire [BOOT_AXI_ID_W-1:0]     fr_awid;    wire [BOOT_AXI_ADDR_W-1:0]  fr_awaddr;
    wire [7:0] fr_awlen;  wire [2:0] fr_awsize;  wire [1:0] fr_awburst;  wire fr_awvalid;
    wire [BOOT_AXI_DATA_W-1:0]   fr_wdata;   wire [BOOT_AXI_DATA_W/8-1:0] fr_wstrb;
    wire fr_wlast;  wire fr_wvalid;  wire fr_bready;

    wire tr_busy, tr_done, tr_err;
    wire [BOOT_AXI_ID_W-1:0]     tr_awid;    wire [BOOT_AXI_ADDR_W-1:0]  tr_awaddr;
    wire [7:0] tr_awlen;  wire [2:0] tr_awsize;  wire [1:0] tr_awburst;  wire tr_awvalid;
    wire [BOOT_AXI_DATA_W-1:0]   tr_wdata;   wire [BOOT_AXI_DATA_W/8-1:0] tr_wstrb;
    wire tr_wlast;  wire tr_wvalid;  wire tr_bready;

    // Format and write the selected top-K results.
    output_dma_writer #(
        .K_F(8), .IDX_W(TREE_IDX_W), .SCORE_W(32),
        .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(BOOT_AXI_DATA_W),
        .AXI_ID_W(BOOT_AXI_ID_W), .EN_OUT1_MIRROR(1'b1),

        .ABI_V24(OUT_ABI_V24), .TOP_K(OUT_TOP_K),
        .PARENT_SENTINEL(OUT_PARENT_SENTINEL),

        .EN_DUMP(1'b1), .CAND_N(G8_K_C)
    ) u_out_dma (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .tree_done(ow_trigger),
        .topk_scores(ow_topk_scores),
        .topk_indices(ow_topk_indices),
        .out0_base(out0_base),
        .out1_base(out1_base),
        .out2_base(out2_base),

        .dump_en(dump_en),
        .out3_base(out3_base), .out4_base(out4_base),
        .dump3_rdaddr(ow_dump3_rdaddr), .dump3_rddata(ow_dump3_rddata),
        .tree_ids(ow_topk_indices), .tree_scores(tree_topk_scores),
        .tree_parents(ow_tree_parents), .g8_rescored(ow_g8_rescored),

        .out_busy(fr_busy),
        .out_done(fr_done),
        .out_err(fr_err),
        .m_axi_out_awid(fr_awid),            .m_axi_out_awaddr(fr_awaddr),
        .m_axi_out_awlen(fr_awlen),          .m_axi_out_awsize(fr_awsize),
        .m_axi_out_awburst(fr_awburst),      .m_axi_out_awvalid(fr_awvalid),
        .m_axi_out_awready(m_axi_out_awready),
        .m_axi_out_wdata(fr_wdata),          .m_axi_out_wstrb(fr_wstrb),
        .m_axi_out_wlast(fr_wlast),          .m_axi_out_wvalid(fr_wvalid),
        .m_axi_out_wready(m_axi_out_wready),
        .m_axi_out_bid(m_axi_out_bid),       .m_axi_out_bresp(m_axi_out_bresp),
        .m_axi_out_bvalid(m_axi_out_bvalid), .m_axi_out_bready(fr_bready)
    );

    generate if (DEPTH_N) begin : g_out_axi_tree

        assign m_axi_out_awid    = tr_awid;    assign m_axi_out_awaddr  = tr_awaddr;
        assign m_axi_out_awlen   = tr_awlen;   assign m_axi_out_awsize  = tr_awsize;
        assign m_axi_out_awburst = tr_awburst; assign m_axi_out_awvalid = tr_awvalid;
        assign m_axi_out_wdata   = tr_wdata;   assign m_axi_out_wstrb   = tr_wstrb;
        assign m_axi_out_wlast   = tr_wlast;   assign m_axi_out_wvalid  = tr_wvalid;
        assign m_axi_out_bready  = tr_bready;

        assign out_dma_busy = fr_busy | tr_busy;
        assign out_dma_done = fr_done | tr_done;
        assign out_dma_err  = fr_err  | tr_err;
    end else begin : g_out_axi_frontier

        assign m_axi_out_awid    = fr_awid;    assign m_axi_out_awaddr  = fr_awaddr;
        assign m_axi_out_awlen   = fr_awlen;   assign m_axi_out_awsize  = fr_awsize;
        assign m_axi_out_awburst = fr_awburst; assign m_axi_out_awvalid = fr_awvalid;
        assign m_axi_out_wdata   = fr_wdata;   assign m_axi_out_wstrb   = fr_wstrb;
        assign m_axi_out_wlast   = fr_wlast;   assign m_axi_out_wvalid  = fr_wvalid;
        assign m_axi_out_bready  = fr_bready;
        assign out_dma_busy = fr_busy;
        assign out_dma_done = fr_done;
        assign out_dma_err  = fr_err;
    end endgenerate

    // Input, hidden-state, and RMSNorm gamma buffers.
    localparam int EMB_H        = 3072;
    localparam int EMB_K_PAR    = 16;
    localparam int EMB_AW       = 16;
    localparam int EMB_NCHUNKS  = EMB_H / EMB_K_PAR;
    localparam int EMB_CW       = $clog2(EMB_NCHUNKS + 1);
    localparam int EMB_DATA_SLOT = 1;
    localparam logic signed [15:0] GAMMA_ONE_Q12 = 16'sh1000;

    localparam logic [7:0] BUILD_TAG = 8'h11;
    localparam logic [7:0] PROBE_REV = 8'h43;

    localparam bit DBG_LMTAP = 1'b1;

    localparam int EMB_NBANKS = (EMB_K_PAR*EMB_AW) / 32;
    `define BUFSPLIT_DECL(NAME) \
        (* ram_style = "block" *) reg [31:0] NAME``_b0 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b1 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b2 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b3 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b4 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b5 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b6 [0:EMB_NCHUNKS-1]; \
        (* ram_style = "block" *) reg [31:0] NAME``_b7 [0:EMB_NCHUNKS-1];

    `define BUFSPLIT_RDFN(NAME) \
        function automatic signed [EMB_K_PAR*EMB_AW-1:0] NAME``_rd(input [EMB_CW-1:0] idx); \
            NAME``_rd = $signed({ NAME``_b7[idx], NAME``_b6[idx], NAME``_b5[idx], NAME``_b4[idx], \
                                  NAME``_b3[idx], NAME``_b2[idx], NAME``_b1[idx], NAME``_b0[idx] }); \
        endfunction

    `define BUFSPLIT_WR(NAME, WE, CHUNK, WORD, D32) \
        if ((WE) && (WORD)==3'd0) NAME``_b0[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd1) NAME``_b1[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd2) NAME``_b2[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd3) NAME``_b3[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd4) NAME``_b4[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd5) NAME``_b5[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd6) NAME``_b6[CHUNK] <= (D32); \
        if ((WE) && (WORD)==3'd7) NAME``_b7[CHUNK] <= (D32);

    `BUFSPLIT_DECL(input_emb_buf)
    `BUFSPLIT_RDFN(input_emb_buf)

    `BUFSPLIT_DECL(gamma_emb_buf)
    `BUFSPLIT_RDFN(gamma_emb_buf)

    `BUFSPLIT_DECL(h_fc_buf)
    `BUFSPLIT_RDFN(h_fc_buf)

    `BUFSPLIT_DECL(gamma_e_buf)
    `BUFSPLIT_RDFN(gamma_e_buf)

    `BUFSPLIT_DECL(gamma_postattn_buf)
    `BUFSPLIT_RDFN(gamma_postattn_buf)

    `BUFSPLIT_DECL(gamma_final_buf)
    `BUFSPLIT_RDFN(gamma_final_buf)

    localparam int EMB_WORDS_PER_CHUNK = (EMB_K_PAR*EMB_AW) / 32;
    localparam int EMB_WPC_LOG2        = $clog2(EMB_WORDS_PER_CHUNK);
    wire                              emb_we;
    wire [16:0]                       emb_word_addr;
    wire [EMB_CW-1:0]                 emb_wr_chunk;
    wire [EMB_WPC_LOG2-1:0]           emb_wr_word;
    wire [31:0]                       emb_wr_data32;

    wire                              hfc_we;
    wire [16:0]                       hfc_word_addr;
    wire [EMB_CW-1:0]                 hfc_wr_chunk;
    wire [EMB_WPC_LOG2-1:0]           hfc_wr_word;
    wire [31:0]                       hfc_wr_data32;

    wire                              gamma_we_hidden, gamma_we_enorm,
                                      gamma_we_postat, gamma_we_final;
    wire [16:0]                       gamma_word_addr;
    wire [EMB_CW-1:0]                 gamma_wr_chunk;
    wire [EMB_WPC_LOG2-1:0]           gamma_wr_word;
    wire [31:0]                       gamma_wr_data32;

// synthesis translate_off

    `define BUFSPLIT_ZERO(NAME, IDX) begin \
        NAME``_b0[IDX]='0; NAME``_b1[IDX]='0; NAME``_b2[IDX]='0; NAME``_b3[IDX]='0; \
        NAME``_b4[IDX]='0; NAME``_b5[IDX]='0; NAME``_b6[IDX]='0; NAME``_b7[IDX]='0; end
    `define BUFSPLIT_W16(NAME, EE, W16) begin \
        case (((EE)%EMB_K_PAR)/2) \
          0: NAME``_b0[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          1: NAME``_b1[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          2: NAME``_b2[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          3: NAME``_b3[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          4: NAME``_b4[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          5: NAME``_b5[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          6: NAME``_b6[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
          7: NAME``_b7[(EE)/EMB_K_PAR][(((EE)%2)*16) +: 16] = (W16); \
        endcase end
    reg signed [EMB_K_PAR*EMB_AW-1:0] emb_init [0:EMB_NCHUNKS-1];
    integer emb_i;
    string  emb_hex_path;
    initial begin : g1_emb_preload
        for (emb_i = 0; emb_i < EMB_NCHUNKS; emb_i = emb_i + 1) begin
            `BUFSPLIT_ZERO(input_emb_buf, emb_i)
            `BUFSPLIT_ZERO(h_fc_buf, emb_i)
            `BUFSPLIT_ZERO(gamma_emb_buf, emb_i)
            `BUFSPLIT_ZERO(gamma_e_buf, emb_i)
            `BUFSPLIT_ZERO(gamma_postattn_buf, emb_i)
            `BUFSPLIT_ZERO(gamma_final_buf, emb_i)
            emb_init[emb_i]      = '0;
        end
        if ($value$plusargs("EMB_FULL_HEX=%s", emb_hex_path)) begin

            begin : g1_load_full
                reg [15:0] word_mem [0:EMB_H-1];
                integer    e;
                for (e = 0; e < EMB_H; e = e + 1) word_mem[e] = 16'h0;
                $readmemh(emb_hex_path, word_mem);
                for (e = 0; e < EMB_H; e = e + 1)
                    `BUFSPLIT_W16(input_emb_buf, e, word_mem[e])
            end

            begin : g1_gamma_full
                reg [15:0] g_mem [0:EMB_H-1];
                integer    e2;
                string     gamma_hex_path;
                for (e2 = 0; e2 < EMB_H; e2 = e2 + 1) g_mem[e2] = GAMMA_ONE_Q12;
                if ($value$plusargs("GAMMA_FULL_HEX=%s", gamma_hex_path)) begin
                    $readmemh(gamma_hex_path, g_mem);
                    $display("[S3] G1 FULL-H gamma preload from %s", gamma_hex_path);
                end
                for (e2 = 0; e2 < EMB_H; e2 = e2 + 1)
                    `BUFSPLIT_W16(gamma_emb_buf, e2, g_mem[e2])
            end

            begin : g1e_gamma_full
                reg [15:0] ge_mem [0:EMB_H-1];
                integer    e3;
                string     gamma_e_hex_path;
                for (e3 = 0; e3 < EMB_H; e3 = e3 + 1) ge_mem[e3] = GAMMA_ONE_Q12;
                if ($value$plusargs("GAMMA_E_HEX=%s", gamma_e_hex_path)) begin
                    $readmemh(gamma_e_hex_path, ge_mem);
                    $display("[S7d] G1_E FULL-H gamma preload from %s", gamma_e_hex_path);
                end
                for (e3 = 0; e3 < EMB_H; e3 = e3 + 1)
                    `BUFSPLIT_W16(gamma_e_buf, e3, ge_mem[e3])
            end

            begin : premlp_gamma_full
                reg [15:0] gp_mem [0:EMB_H-1];
                integer    e4;
                string     gamma_pa_hex_path;
                for (e4 = 0; e4 < EMB_H; e4 = e4 + 1) gp_mem[e4] = GAMMA_ONE_Q12;
                if ($value$plusargs("GAMMA_POSTATTN_HEX=%s", gamma_pa_hex_path)) begin
                    $readmemh(gamma_pa_hex_path, gp_mem);
                    $display("[FIX-D] post-attn gamma preload from %s", gamma_pa_hex_path);
                end
                for (e4 = 0; e4 < EMB_H; e4 = e4 + 1)
                    `BUFSPLIT_W16(gamma_postattn_buf, e4, gp_mem[e4])
            end

            begin : final_gamma_full
                reg [15:0] gf_mem [0:EMB_H-1];
                integer    e5;
                string     gamma_fn_hex_path;
                for (e5 = 0; e5 < EMB_H; e5 = e5 + 1) gf_mem[e5] = GAMMA_ONE_Q12;
                if ($value$plusargs("GAMMA_FINAL_HEX=%s", gamma_fn_hex_path)) begin
                    $readmemh(gamma_fn_hex_path, gf_mem);
                    $display("[FIX-F] final-norm gamma preload from %s", gamma_fn_hex_path);
                end
                for (e5 = 0; e5 < EMB_H; e5 = e5 + 1)
                    `BUFSPLIT_W16(gamma_final_buf, e5, gf_mem[e5])
            end
            $display("[S3] G1 FULL-H input preload: %0d words from %s into %0d chunk slots",
                     EMB_H, emb_hex_path, EMB_NCHUNKS);
        end else if ($value$plusargs("EMB_HEX=%s", emb_hex_path)) begin

            begin : g1_load
                reg [15:0] word_mem [0:EMB_K_PAR-1];
                integer    w;
                for (w = 0; w < EMB_K_PAR; w = w + 1) word_mem[w] = 16'h0;
                $readmemh(emb_hex_path, word_mem);
                for (w = 0; w < EMB_K_PAR; w = w + 1)
                    `BUFSPLIT_W16(input_emb_buf, EMB_DATA_SLOT*EMB_K_PAR + w, word_mem[w])
            end

            for (emb_i = 0; emb_i < EMB_K_PAR; emb_i = emb_i + 1)
                `BUFSPLIT_W16(gamma_emb_buf, EMB_DATA_SLOT*EMB_K_PAR + emb_i, GAMMA_ONE_Q12)
            $display("[M4] G1 input-embedding preload: %0d words from %s into slot %0d",
                     EMB_K_PAR, emb_hex_path, EMB_DATA_SLOT);
        end
    end
// synthesis translate_on

    always_ff @(posedge ap_clk) begin : gamma_load_wr
        `BUFSPLIT_WR(gamma_emb_buf,      gamma_we_hidden, gamma_wr_chunk, gamma_wr_word, gamma_wr_data32)
        `BUFSPLIT_WR(gamma_e_buf,        gamma_we_enorm,  gamma_wr_chunk, gamma_wr_word, gamma_wr_data32)
        `BUFSPLIT_WR(gamma_postattn_buf, gamma_we_postat, gamma_wr_chunk, gamma_wr_word, gamma_wr_data32)
        `BUFSPLIT_WR(gamma_final_buf,    gamma_we_final,  gamma_wr_chunk, gamma_wr_word, gamma_wr_data32)
    end

    wire                        embg_o_we;
    wire [7:0]                  embg_o_chunk;
    wire [EMB_K_PAR*EMB_AW-1:0] embg_o_data;
    wire                        emb_start_w;
    wire                        emb_done_w;

    generate if (EMBG_EN) begin : g_emb_arb
        logic                        emb_wr_en;
        logic [EMB_CW-1:0]           emb_wr_chunk_a;
        logic [EMB_NBANKS-1:0]       emb_wr_bmask;
        logic [EMB_K_PAR*EMB_AW-1:0] emb_wr_data256;
        always_comb begin
            if (embg_o_we) begin
                emb_wr_en      = 1'b1;
                emb_wr_chunk_a = embg_o_chunk[EMB_CW-1:0];
                emb_wr_bmask   = {EMB_NBANKS{1'b1}};
                emb_wr_data256 = embg_o_data;
            end else begin
                emb_wr_en      = emb_we;
                emb_wr_chunk_a = emb_wr_chunk[EMB_CW-1:0];
                emb_wr_bmask   = emb_we ? (EMB_NBANKS'(1) << emb_wr_word) : '0;
                emb_wr_data256 = {EMB_NBANKS{emb_wr_data32}};
            end
        end
        always_ff @(posedge ap_clk) begin
            if (emb_wr_en && emb_wr_bmask[0]) input_emb_buf_b0[emb_wr_chunk_a] <= emb_wr_data256[0*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[1]) input_emb_buf_b1[emb_wr_chunk_a] <= emb_wr_data256[1*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[2]) input_emb_buf_b2[emb_wr_chunk_a] <= emb_wr_data256[2*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[3]) input_emb_buf_b3[emb_wr_chunk_a] <= emb_wr_data256[3*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[4]) input_emb_buf_b4[emb_wr_chunk_a] <= emb_wr_data256[4*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[5]) input_emb_buf_b5[emb_wr_chunk_a] <= emb_wr_data256[5*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[6]) input_emb_buf_b6[emb_wr_chunk_a] <= emb_wr_data256[6*32 +: 32];
            if (emb_wr_en && emb_wr_bmask[7]) input_emb_buf_b7[emb_wr_chunk_a] <= emb_wr_data256[7*32 +: 32];
        end
    end else begin : g_emb_hostonly
        always_ff @(posedge ap_clk) begin
            `BUFSPLIT_WR(input_emb_buf, emb_we, emb_wr_chunk, emb_wr_word, emb_wr_data32)
        end
    end endgenerate

    generate if (!(DEPTH_N && EMBG_EN)) begin : g_embg_tieoff
        assign m_axi_emb_arid    = '0;
        assign m_axi_emb_araddr  = '0;
        assign m_axi_emb_arlen   = '0;
        assign m_axi_emb_arsize  = '0;
        assign m_axi_emb_arburst = '0;
        assign m_axi_emb_arvalid = 1'b0;
        assign m_axi_emb_rready  = 1'b0;
        assign embg_o_we    = 1'b0;
        assign embg_o_chunk = '0;
        assign embg_o_data  = '0;
        assign emb_done_w   = 1'b0;
    end endgenerate

    generate if (!(DEPTH_N && EMBG_EN && (EMB_NCH > 1))) begin : g_embb_tieoff
        assign m_axi_embb_arid    = '0;
        assign m_axi_embb_araddr  = '0;
        assign m_axi_embb_arlen   = '0;
        assign m_axi_embb_arsize  = '0;
        assign m_axi_embb_arburst = '0;
        assign m_axi_embb_arvalid = 1'b0;
        assign m_axi_embb_rready  = 1'b0;
    end endgenerate

    always_ff @(posedge ap_clk) begin
        `BUFSPLIT_WR(h_fc_buf, hfc_we, hfc_wr_chunk, hfc_wr_word, hfc_wr_data32)
    end

    // G1 hidden RMSNorm input. Full-width requests use slots 1..192 for
    // logical chunks 0..191. Keep the final read live through phase drain.
    wire [EMB_CW-1:0] g1_req_chunk;
    wire              g1_req_valid;

    wire [EMB_CW-1:0] g1_eff_chunk = SEAM_R1_FULLW
        ? (g1_req_chunk - EMB_CW'(1))
        : g1_req_chunk;

    reg g1_req_valid_d1;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) g1_req_valid_d1 <= 1'b0;
        else           g1_req_valid_d1 <= g1_req_valid;
    end
    wire g1_req_live = g1_req_valid || g1_req_valid_d1;
    wire g1_eff_inrange = SEAM_R1_FULLW
        ? (g1_req_live &&
           g1_req_chunk >= EMB_CW'(1) && g1_req_chunk <= EMB_CW'(EMB_NCHUNKS))
        : (g1_req_valid && g1_req_chunk <  EMB_CW'(EMB_NCHUNKS));

    reg signed [EMB_K_PAR*EMB_AW-1:0] g1_x_chunk;
    reg signed [EMB_K_PAR*EMB_AW-1:0] g1_gamma_chunk;

    wire signed [EMB_K_PAR*EMB_AW-1:0] f2_fr_rd_data_w;
    wire                               f2_lvl_ge1_w;

    // Optional address register plus the data register below: match the
    // RMSNorm engine's RDATA_LAT to this path (2 with BUFRDREG, otherwise 1).
    (* keep = "true", max_fanout = 8 *) reg [EMB_CW-1:0] g1_rd_chunk_q;
    reg                                                  g1_rd_inrange_q;
    always_ff @(posedge ap_clk) begin
        g1_rd_chunk_q   <= g1_eff_chunk;
        g1_rd_inrange_q <= g1_eff_inrange;
    end
    wire [EMB_CW-1:0] g1_rd_idx = BUFRDREG ? g1_rd_chunk_q   : g1_eff_chunk;
    wire              g1_rd_ir  = BUFRDREG ? g1_rd_inrange_q : g1_eff_inrange;

    wire signed [EMB_K_PAR*EMB_AW-1:0] inj_g1_chunk;
    genvar gig1;
    generate
        for (gig1 = 0; gig1 < EMB_K_PAR; gig1 = gig1 + 1) begin : g_inj_g1
            assign inj_g1_chunk[gig1*EMB_AW +: EMB_AW] = inj_bus[gig1*32 +: EMB_AW];
        end
    endgenerate
    always_ff @(posedge ap_clk) begin
        g1_x_chunk     <= inject_en[0] ? inj_g1_chunk
                                       : (g1_rd_ir ? h_fc_buf_rd(g1_rd_idx) : '0);
        g1_gamma_chunk <= g1_rd_ir ? gamma_emb_buf_rd(g1_rd_idx) : '0;
    end

    reg f2_g1_ir_q;
    always_ff @(posedge ap_clk) f2_g1_ir_q <= g1_rd_ir;
`ifdef F2_RED_HFC

    wire f2_g1_use_fr = 1'b0;
`else
    wire f2_g1_use_fr = f2_lvl_ge1_w && !inject_en[0];
`endif
    wire signed [EMB_K_PAR*EMB_AW-1:0] g1_x_chunk_eff = f2_g1_use_fr
        ? (f2_g1_ir_q ? f2_fr_rd_data_w : '0)
        : g1_x_chunk;

    reg              g1_rd_stb;
    reg [EMB_CW-1:0] g1_rd_idx_tap;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin g1_rd_stb <= 1'b0; g1_rd_idx_tap <= '0; end
        else begin
            g1_rd_stb     <= g1_rd_ir;
            g1_rd_idx_tap <= g1_rd_idx;
        end
    end
    wire [31:0] g1_rd_word_tap = g1_x_chunk_eff[31:0];
    wire              g1_apply_en;

    wire signed [16*16-1:0] rms_g1_out_chunk_w;
    wire                    rms_g1_out_valid;
    wire                    g1_done;

    // S = sum(x*x); engine input = (S*43691)>>31. The apply path shifts
    // x*scale by 10 into an 18-bit intermediate, then multiplies gamma (>>12).
    // SUMSQ_RSHIFT is unused when SUMSQ_MULT is nonzero.
    rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                    .SUMSQ_RSHIFT(12),
                    .SUMSQ_MULT(43691), .SUMSQ_SHR(31), .SCALE_SHR(10),
                    .XS_WIDTH(18),
                    .RDATA_LAT(BUFRDREG ? 2 : 1)) u_rmsnorm_g1 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g1_start),
        .x_chunk(g1_x_chunk_eff),
        .gamma_chunk(g1_gamma_chunk),
        .out_chunk(rms_g1_out_chunk_w),
        .out_valid(rms_g1_out_valid),
        .done(g1_done),
        .x_req_chunk(g1_req_chunk),
        .x_req_valid(g1_req_valid),
        .apply_en_o(g1_apply_en)
    );

    wire [EMB_CW-1:0] g1e_req_chunk;
    wire              g1e_req_valid;

    wire [EMB_CW-1:0] g1e_eff_chunk = SEAM_R1_FULLW
        ? (g1e_req_chunk - EMB_CW'(1))
        : g1e_req_chunk;
    wire g1e_eff_inrange = SEAM_R1_FULLW
        ? (g1e_req_chunk >= EMB_CW'(1) && g1e_req_chunk <= EMB_CW'(EMB_NCHUNKS))
        : (g1e_req_valid && g1e_req_chunk <  EMB_CW'(EMB_NCHUNKS));

    reg signed [EMB_K_PAR*EMB_AW-1:0] g1e_x_chunk;
    reg signed [EMB_K_PAR*EMB_AW-1:0] g1e_gamma_chunk;

    (* keep = "true", max_fanout = 8 *) reg [EMB_CW-1:0] g1e_rd_chunk_q;
    reg                                                  g1e_rd_inrange_q;
    always_ff @(posedge ap_clk) begin
        g1e_rd_chunk_q   <= g1e_eff_chunk;
        g1e_rd_inrange_q <= g1e_eff_inrange;
    end
    wire [EMB_CW-1:0] g1e_rd_idx = BUFRDREG ? g1e_rd_chunk_q   : g1e_eff_chunk;
    wire              g1e_rd_ir  = BUFRDREG ? g1e_rd_inrange_q : g1e_eff_inrange;
    always_ff @(posedge ap_clk) begin
        g1e_x_chunk     <= g1e_rd_ir ? input_emb_buf_rd(g1e_rd_idx) : '0;
        g1e_gamma_chunk <= g1e_rd_ir ? gamma_e_buf_rd(g1e_rd_idx)   : '0;
    end

    reg              g1e_rd_stb;
    reg [EMB_CW-1:0] g1e_rd_idx_tap;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin g1e_rd_stb <= 1'b0; g1e_rd_idx_tap <= '0; end
        else begin
            g1e_rd_stb     <= g1e_rd_ir;
            g1e_rd_idx_tap <= g1e_rd_idx;
        end
    end
    wire [31:0] g1e_rd_word_tap = g1e_x_chunk[31:0];

    wire signed [16*16-1:0] rms_g1e_out_chunk_w;
    wire                    rms_g1e_out_valid;
    wire                    g1e_done;
    wire                    g1e_apply_en;
    wire                    g1e_apply_en_eff;

    rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                    .SUMSQ_RSHIFT(12),
                    .SUMSQ_MULT(43691), .SUMSQ_SHR(31), .SCALE_SHR(10),
                    .XS_WIDTH(18),
                    .RDATA_LAT(BUFRDREG ? 2 : 1)) u_rmsnorm_g1_e (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g1_start),
        .x_chunk(g1e_x_chunk),
        .gamma_chunk(g1e_gamma_chunk),
        .out_chunk(rms_g1e_out_chunk_w),
        .out_valid(rms_g1e_out_valid),
        .done(g1e_done),
        .x_req_chunk(g1e_req_chunk),
        .x_req_valid(g1e_req_valid),
        .apply_en_o(g1e_apply_en)
    );

    localparam int ENORM_D_SUMSQ_SHR = PRECISION_SAFE ? 31 : 27;
    localparam int ENORM_D_SCALE_SHR = PRECISION_SAFE ? 10 : 8;
    wire signed [16*16-1:0] rms_g1e_out_chunk_eff;
    generate if (DEPTH_N) begin : g_enorm_depth_requant
        wire signed [16*16-1:0] rms_g1e_d_out_chunk_w;

        wire [EMB_CW-1:0] g1ed_req_chunk;
        wire              g1ed_req_valid;
        wire [EMB_CW-1:0] g1ed_eff_chunk = SEAM_R1_FULLW
            ? (g1ed_req_chunk - EMB_CW'(1))
            : g1ed_req_chunk;
        wire g1ed_eff_inrange = SEAM_R1_FULLW
            ? (g1ed_req_chunk >= EMB_CW'(1) &&
               g1ed_req_chunk <= EMB_CW'(EMB_NCHUNKS))
            : (g1ed_req_valid && g1ed_req_chunk < EMB_CW'(EMB_NCHUNKS));
        (* keep = "true", max_fanout = 8 *) reg [EMB_CW-1:0] g1ed_rd_chunk_q;
        reg                                                  g1ed_rd_inrange_q;
        always_ff @(posedge ap_clk) begin
            g1ed_rd_chunk_q   <= g1ed_eff_chunk;
            g1ed_rd_inrange_q <= g1ed_eff_inrange;
        end
        wire [EMB_CW-1:0] g1ed_rd_idx =
            BUFRDREG ? g1ed_rd_chunk_q : g1ed_eff_chunk;
        wire g1ed_rd_ir = BUFRDREG ? g1ed_rd_inrange_q : g1ed_eff_inrange;
        reg signed [EMB_K_PAR*EMB_AW-1:0] g1ed_x_chunk;
        reg signed [EMB_K_PAR*EMB_AW-1:0] g1ed_gamma_chunk;
        always_ff @(posedge ap_clk) begin
            g1ed_x_chunk <= g1ed_rd_ir ? input_emb_buf_rd(g1ed_rd_idx) : '0;
            g1ed_gamma_chunk <= g1ed_rd_ir ? gamma_e_buf_rd(g1ed_rd_idx) : '0;
        end
        wire g1ed_apply_en;
        rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                        .SUMSQ_RSHIFT(12),
                        .SUMSQ_MULT(43691), .SUMSQ_SHR(ENORM_D_SUMSQ_SHR),
                        .SCALE_SHR(ENORM_D_SCALE_SHR),
                        .SCALE_EPS_BUCKET(PRECISION_SAFE ? 168 : 1),
                        .XS_WIDTH(18),
                        .RDATA_LAT(BUFRDREG ? 2 : 1)) u_rmsnorm_g1_e_d (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(g1_start),
            .x_chunk(g1ed_x_chunk),
            .gamma_chunk(g1ed_gamma_chunk),
            .out_chunk(rms_g1e_d_out_chunk_w),
            .out_valid(), .done(),
            .x_req_chunk(g1ed_req_chunk), .x_req_valid(g1ed_req_valid),
            .out_chunk_strobe(), .out_chunk_idx_o(),
            .apply_en_o(g1ed_apply_en)
        );

`ifdef ENORM_RED_SAT
        wire use_enorm_d = 1'b0;
`elsif ENORM_GREEN_L0

        wire use_enorm_d = 1'b1;
`else

        wire use_enorm_d = PRECISION_SAFE ? 1'b1
                                         : (ENORM_D0 ? 1'b1 : f2_lvl_ge1_w);
`endif
        assign rms_g1e_out_chunk_eff = use_enorm_d ? rms_g1e_d_out_chunk_w
                                                   : rms_g1e_out_chunk_w;
        assign g1e_apply_en_eff = use_enorm_d ? g1ed_apply_en : g1e_apply_en;
    end else begin : g_enorm_depth_requant_off
        assign rms_g1e_out_chunk_eff = rms_g1e_out_chunk_w;
        assign g1e_apply_en_eff = g1e_apply_en;
    end endgenerate

    reg signed [16*16-1:0] rms_g1_out_chunk_d1;
    reg signed [16*16-1:0] rms_g1_out_first;
    reg                    rms_g1_first_seen;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            rms_g1_out_chunk_d1 <= '0;
            rms_g1_out_first    <= '0;
            rms_g1_first_seen   <= 1'b0;
        end else begin
            rms_g1_out_chunk_d1 <= rms_g1_out_chunk_w;
            if (g1_start) begin
                rms_g1_first_seen <= 1'b0;
            end else if (rms_g1_out_valid && !rms_g1_first_seen) begin
                rms_g1_out_first  <= rms_g1_out_chunk_d1;
                rms_g1_first_seen <= 1'b1;
            end
        end
    end
    wire signed [16*16-1:0] rms_g1_out_chunk = rms_g1_out_first;

    wire signed [16*16-1:0] rms_g11_out_chunk;
    wire                    rms_g11_out_valid;
    wire                    g11_done;

    localparam int G11_NCHUNKS = 3072/16;
    localparam int G11_CW      = $clog2(G11_NCHUNKS + 1);
    wire                  g11_out_strobe;
    wire [G11_CW-1:0]     g11_out_chunk_idx;

    localparam int R3_FRAC = PRECISION_SAFE ? 7 : 5;
    localparam int R7_FRAC = 8;
    localparam int G11_SUMSQ_RSHIFT = 2*R7_FRAC - 12;

    initial begin
        if (TAIL_FP16) begin
            $error("eagle4_drafter: TAIL_FP16=1 selected, but the fp16 MLP-tail datapath is NOT wired (fp16_tail #184 is a deferred RTL build item). Set TAIL_FP16=0 or add the fp16 tail lanes first.");
        end

        if (GEMV_SCALED && GEMV_KPAR64) begin
            $error("eagle4_drafter: GEMV_SCALED=1 and GEMV_KPAR64=1 are mutually exclusive (no scaled K_PAR=64 lane). Set GEMV_KPAR64=0 for the scaled FITTING config.");
        end

        if (EMB_NCH < 1 || EMB_NCH > 2) begin
            $error("eagle4_drafter: EMB_NCH=%0d is out of the TOP-supported range [1,2] (only m_axi_emb + m_axi_embb exist). Add the S10/S11 masters + BD patches first.", EMB_NCH);
        end
        if ((EMB_NCH > 1) && !(DEPTH_N && EMBG_EN)) begin
            $error("eagle4_drafter: EMB_NCH=%0d requires DEPTH_N=1 && EMBG_EN=1 (there is no gather to stripe otherwise).", EMB_NCH);
        end

        if ((3072 % (EMB_NCH*16)) != 0) begin
            $error("eagle4_drafter: H=3072 must be a multiple of EMB_NCH*16 (got EMB_NCH=%0d).", EMB_NCH);
        end
    end
    wire signed [16*16-1:0] seam_r6_rd_c;

    wire [G11_CW-1:0]       g11_x_req_chunk;
    wire                    g11_x_req_valid;

    reg signed [16*16-1:0] g11_x_in;
    always_ff @(posedge ap_clk) begin
        g11_x_in <= CONNECT_SEAMS ? seam_r6_rd_c : '0;
    end

    wire [G11_CW-1:0] g11_g_eff_chunk = BACK_SEQ
        ? (g11_x_req_chunk - G11_CW'(1))
        : g11_x_req_chunk;
    wire g11_g_inrange = BACK_SEQ
        ? (g11_x_req_chunk >= G11_CW'(1) && g11_x_req_chunk <= G11_CW'(G11_NCHUNKS))
        : 1'b1;
    reg signed [16*16-1:0] g11_gamma_in;
    always_ff @(posedge ap_clk) begin
        g11_gamma_in <= (CONNECT_SEAMS && g11_g_inrange)
                        ? gamma_final_buf_rd(g11_g_eff_chunk) : '0;
    end

    localparam int G11_SUMSQ_SHR  = 29;
    localparam int G11_SCALE_SHR  = 11;
    localparam int G11_OUT_FRAC   = 10;
    localparam int G11_SCALE_WIDTH = 17;

    wire seam_r6_full;
    reg  g11_start_pending;
    wire g11_start_eff = CONNECT_SEAMS ? (g11_start_pending && seam_r6_full)
                                       : g11_start;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n)               g11_start_pending <= 1'b0;
        else if (g11_start)          g11_start_pending <= 1'b1;
        else if (g11_start_eff)      g11_start_pending <= 1'b0;
    end
    rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                    .SUMSQ_RSHIFT(G11_SUMSQ_RSHIFT),
                    .SUMSQ_MULT(43691), .SUMSQ_SHR(G11_SUMSQ_SHR), .SCALE_SHR(G11_SCALE_SHR),
                    .RDATA_LAT(1), .SCALE_WIDTH(G11_SCALE_WIDTH)) u_rmsnorm_g11 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g11_start_eff),
        .x_chunk(g11_x_in),
        .gamma_chunk(g11_gamma_in),
        .out_chunk(rms_g11_out_chunk),
        .out_valid(rms_g11_out_valid),
        .done(g11_done),
        .x_req_chunk(g11_x_req_chunk),
        .x_req_valid(g11_x_req_valid),
        .out_chunk_strobe(g11_out_strobe),
        .out_chunk_idx_o(g11_out_chunk_idx),
        .apply_en_o()
    );

    wire signed [16*16-1:0] silu_out_chunk;
    wire                    silu_valid;
    reg                     g7p_done_r;

    reg                     silu_sweep_done_r;

    localparam int G7_H_INTER = 8192;
    localparam int G7_NCHUNKS = G7_H_INTER / K_PAR;
    localparam int G7_CW      = $clog2(G7_NCHUNKS);
    wire [G7_CW-1:0] r4_silu_rd_idx;

    localparam bit SILU_RANGE_FIX = SILU_RANGE_FIX_P;
    localparam int R4_AW = SILU_RANGE_FIX ? 18 : 16;
    wire signed [16*R4_AW-1:0] seam_r4g_rd_c;
    wire signed [16*R4_AW-1:0] seam_r4u_rd_c;
    wire signed [16*R4_AW-1:0] silu_gate_real = CONNECT_SEAMS ? seam_r4g_rd_c : '0;
    wire signed [16*R4_AW-1:0] silu_up_real   = CONNECT_SEAMS ? seam_r4u_rd_c : '0;

    wire signed [16*R4_AW-1:0] inj_silu_gate;
    wire signed [16*R4_AW-1:0] inj_silu_up;
    genvar gis;
    generate
        for (gis = 0; gis < 16; gis = gis + 1) begin : g_inj_silu
            assign inj_silu_gate[gis*R4_AW +: R4_AW] = inj_bus[gis*32 +: R4_AW];
            assign inj_silu_up  [gis*R4_AW +: R4_AW] = inj_bus[(16+gis)*32 +: R4_AW];
        end
    endgenerate
    wire signed [16*R4_AW-1:0] silu_gate_in = inject_en[8] ? inj_silu_gate : silu_gate_real;
    wire signed [16*R4_AW-1:0] silu_up_in   = inject_en[8] ? inj_silu_up   : silu_up_real;

    wire silu_en_eff;

    generate
    if (SILU_RANGE_FIX) begin : g_silu_fix
        silu_layer_fix #(.K_PAR(16), .IN_W(R4_AW), .OUT_FRAC(8)) u_silu (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .en(silu_en_eff),
            .gate_chunk(silu_gate_in),
            .up_chunk(silu_up_in),
            .out_chunk(silu_out_chunk),
            .valid(silu_valid)
        );
    end else begin : g_silu_legacy

        silu_layer #(.K_PAR(16), .OUT_FRAC(8)) u_silu (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .en(silu_en_eff),
            .gate_chunk(silu_gate_in),
            .up_chunk(silu_up_in),
            .out_chunk(silu_out_chunk),
            .valid(silu_valid)
        );
    end
    endgenerate
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) g7p_done_r <= 1'b0;
        else           g7p_done_r <= silu_valid;
    end

    wire g7p_done = BACK_SEQ ? silu_sweep_done_r : g7p_done_r;

    localparam int G5_D_HEAD = 16;
    localparam int G5_CTX_MAX = 32;

    wire signed [16*128-1:0] rope_vec_out;
    wire                     rope_valid;

    typedef enum logic [2:0] {
        G5_S_IDLE   = 3'd0,
        G5_S_ROPE_Q = 3'd1,
        G5_S_ROPE_K = 3'd2,
        G5_S_ATTN   = 3'd3,
        G5_S_DONE   = 3'd4
    } g5_state_e;

    g5_state_e g5_state;
    logic      g5_rope_start;
    logic      g5_rope_sel;
    logic      g5_attn_start;
    logic      g5_kv_wr;
    wire       g5_attn_done;
    reg        g5_done_r;

    // QKV composite destination: each projection has 3072 outputs.
    // At LANES=128 this is 24 placement tiles per projection, 72 total.
    localparam int G4_OUT       = 3072;
    localparam int G4_NTILES    = (G4_OUT + LANES - 1) / LANES;
    localparam int G4_TOT_TILES = 3 * G4_NTILES;
    localparam int G4_TW        = $clog2(G4_NTILES);

    wire signed [32*LANES-1:0] g4_tile_slice = array_acc_out;

    // Q/K/V destination adapter. Destination 9 is the QKV composite: the
    // first G4_NTILES placement tiles store Q, the next K, the final V.
    // Destinations 0/1/2 store one projection at output_base + tile.
    // gemv_store_idx_w already includes output_base.
    wire [15:0]      g4_lcol      = gemv_store_idx_w;
    wire             g4_comp      = (gemv_tile_dest_w == 4'd9);
    wire [1:0]       g4_subtile   = (g4_lcol < 16'(G4_NTILES))     ? 2'd0
                                  : (g4_lcol < 16'(2*G4_NTILES))   ? 2'd1
                                  : (g4_lcol < 16'(G4_TOT_TILES))  ? 2'd2
                                  :                                  2'd3;
    wire [15:0]      g4_subbase   = (g4_subtile == 2'd0) ? 16'd0
                                  : (g4_subtile == 2'd1) ? 16'(G4_NTILES)
                                  :                        16'(2*G4_NTILES);
    wire [15:0]      g4_local     = g4_lcol - g4_subbase;
    wire [G4_TW-1:0] g4_ctile     = g4_comp ? g4_local[G4_TW-1:0]
                                             : g4_lcol[G4_TW-1:0];
    wire             cap_q_we     = gemv_tile_qkv &&
                                    ((gemv_tile_dest_w == 4'd0) ||
                                     (g4_comp && (g4_subtile == 2'd0)));
    wire             cap_k_we     = gemv_tile_qkv &&
                                    ((gemv_tile_dest_w == 4'd1) ||
                                     (g4_comp && (g4_subtile == 2'd1)));
    wire             cap_v_we     = gemv_tile_qkv &&
                                    ((gemv_tile_dest_w == 4'd2) ||
                                     (g4_comp && (g4_subtile == 2'd2)));

    reg signed [32*LANES-1:0] q_full_buf [0:G4_NTILES-1];
    reg signed [32*LANES-1:0] k_full_buf [0:G4_NTILES-1];
    reg signed [32*LANES-1:0] v_full_buf [0:G4_NTILES-1];
    always_ff @(posedge ap_clk) begin
        if (cap_q_we) q_full_buf[g4_ctile] <= g4_tile_slice;
        if (cap_k_we) k_full_buf[g4_ctile] <= g4_tile_slice;
        if (cap_v_we) v_full_buf[g4_ctile] <= g4_tile_slice;
    end

    wire signed [16*G5_D_HEAD-1:0] q_tile_buf;
    wire signed [16*G5_D_HEAD-1:0] k_tile_buf;
    wire signed [16*G5_D_HEAD-1:0] v_tile_buf;
    genvar gq;
    generate
        for (gq = 0; gq < G5_D_HEAD; gq = gq + 1) begin : g_g4_head0_view
            assign q_tile_buf[gq*16 +: 16] = q_full_buf[0][gq*32 +: 16];
            assign k_tile_buf[gq*16 +: 16] = k_full_buf[0][gq*32 +: 16];
            assign v_tile_buf[gq*16 +: 16] = v_full_buf[0][gq*32 +: 16];
        end
    endgenerate

    wire signed [16*G5_D_HEAD-1:0] rope_in_head0 =
        g5_rope_sel ? k_tile_buf : q_tile_buf;

    wire signed [16*128-1:0] rope_vec_in =
        {{(16*(128-G5_D_HEAD)){1'b0}}, rope_in_head0};

    // RoPE for the head-0 attention path.
    rope_layer #(
        .D_HEAD(128), .N_PAIRS(64), .MAX_CTX(2048), .A_WIDTH(16),
        .ROPE_CS_PIPE(ROPE_CS_PIPE), .ROPE_POS_REP(ROPE_POS_REP)
    ) u_rope (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g5_rope_start),
        .pos(kv_append_pos),
        .vec_in(rope_vec_in),
        .vec_out(rope_vec_out),
        .valid(rope_valid)
    );

    wire signed [16*G5_D_HEAD-1:0] rope_out_head0 = rope_vec_out[16*G5_D_HEAD-1:0];

    reg signed [16*G5_D_HEAD-1:0] q_rotated_buf;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n)                              q_rotated_buf <= '0;
        else if (g5_state == G5_S_ROPE_Q && rope_valid)
                                                    q_rotated_buf <= rope_out_head0;
    end

    wire        u_attn_rd_en;
    wire        u_attn_rd_kind;
    wire [10:0] u_attn_rd_pos;
    wire signed [16*G5_D_HEAD-1:0] k_vec_from_cache;
    wire signed [16*G5_D_HEAD-1:0] v_vec_from_cache;

    wire signed [16*G5_D_HEAD-1:0] k_write_vec = rope_out_head0;

    wire signed [16*G5_D_HEAD-1:0] v_write_vec = v_tile_buf;

    kv_head_wide #(.D_HEAD(G5_D_HEAD), .A_WIDTH(16), .MAX_CTX(G5_CTX_MAX)) u_kv_k_head0 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_en(g5_kv_wr),
        .wr_pos(kv_append_pos),
        .wr_vec(k_write_vec),
        .rd_en(u_attn_rd_en & ~u_attn_rd_kind),
        .rd_pos(u_attn_rd_pos),
        .rd_vec(k_vec_from_cache)
    );

    kv_head_wide #(.D_HEAD(G5_D_HEAD), .A_WIDTH(16), .MAX_CTX(G5_CTX_MAX)) u_kv_v_head0 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_en(g5_kv_wr),
        .wr_pos(kv_append_pos),
        .wr_vec(v_write_vec),
        .rd_en(u_attn_rd_en & u_attn_rd_kind),
        .rd_pos(u_attn_rd_pos),
        .rd_vec(v_vec_from_cache)
    );

    wire signed [32*G5_D_HEAD-1:0] g5_attn_out_vec;
    attention_head_assembly #(
        .D_HEAD(G5_D_HEAD), .A_WIDTH(16), .CTX_MAX(G5_CTX_MAX)
    ) u_attn_head (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(g5_attn_start),
        .ctx_len(ctx_len[10:0]),
        .q_vec(q_rotated_buf),
        .kv_rd_en(u_attn_rd_en),
        .kv_rd_kind(u_attn_rd_kind),
        .kv_rd_pos(u_attn_rd_pos),
        .k_vec_in(k_vec_from_cache),
        .v_vec_in(v_vec_from_cache),
        .out_vec(g5_attn_out_vec),
        .done(g5_attn_done)
    );

    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            g5_state      <= G5_S_IDLE;
            g5_rope_start <= 1'b0;
            g5_rope_sel   <= 1'b0;
            g5_attn_start <= 1'b0;
            g5_kv_wr      <= 1'b0;
            g5_done_r     <= 1'b0;
        end else begin

            g5_rope_start <= 1'b0;
            g5_attn_start <= 1'b0;
            g5_kv_wr      <= 1'b0;
            g5_done_r     <= 1'b0;
            case (g5_state)
                G5_S_IDLE: begin
                    if (g5_start) begin

                        g5_rope_sel   <= 1'b0;
                        g5_rope_start <= 1'b1;
                        g5_state      <= G5_S_ROPE_Q;
                    end
                end
                G5_S_ROPE_Q: begin

                    if (rope_valid) begin
                        g5_rope_sel   <= 1'b1;
                        g5_rope_start <= 1'b1;
                        g5_state      <= G5_S_ROPE_K;
                    end
                end
                G5_S_ROPE_K: begin

                    if (rope_valid) begin
                        g5_kv_wr      <= 1'b1;
                        g5_attn_start <= 1'b1;
                        g5_state      <= G5_S_ATTN;
                    end
                end
                G5_S_ATTN: begin
                    if (g5_attn_done) begin
                        g5_done_r <= 1'b1;
                        g5_state  <= G5_S_DONE;
                    end
                end
                G5_S_DONE: begin
                    g5_state <= G5_S_IDLE;
                end
                default: g5_state <= G5_S_IDLE;
            endcase
        end
    end

    wire [31:0] ctx_len, num_frontiers, max_depth;
    wire [63:0] out0_base, out1_base, out2_base, hbm_emb_base;
    wire [63:0] lmfull_hbm_base;

    localparam int S7A_NHEADS = 24;
    localparam int S7A_DHEAD  = 128;
    localparam int S7A_H      = S7A_NHEADS * S7A_DHEAD;
    localparam int S7A_OUT_W  = S7A_H * A_WIDTH;
    localparam int S7A_ROW_W  = S7A_DHEAD * A_WIDTH;
    localparam int S7A_NCH    = S7A_H / K_PAR;
    localparam int S7A_CW     = (S7A_NCH > 1) ? $clog2(S7A_NCH) : 1;

    wire signed [32*S7A_DHEAD-1:0] q_head_view [0:S7A_NHEADS-1];
    wire signed [32*S7A_DHEAD-1:0] k_head_view [0:S7A_NHEADS-1];
    wire signed [32*S7A_DHEAD-1:0] v_head_view [0:S7A_NHEADS-1];
    generate
        for (genvar hv = 0; hv < S7A_NHEADS; hv = hv + 1) begin : g_head_view
            localparam int HV_SLOT = (hv * S7A_DHEAD) / LANES;
            localparam int HV_OFS  = ((hv * S7A_DHEAD) % LANES) * 32;
            assign q_head_view[hv] = q_full_buf[HV_SLOT][HV_OFS +: 32*S7A_DHEAD];
            assign k_head_view[hv] = k_full_buf[HV_SLOT][HV_OFS +: 32*S7A_DHEAD];
            assign v_head_view[hv] = v_full_buf[HV_SLOT][HV_OFS +: 32*S7A_DHEAD];
        end
    endgenerate

    wire [10:0] s7a_ctx_len_reg = (ctx_len[10:0] == 11'd0) ? 11'd1 : ctx_len[10:0];

`ifdef SIM_CTX_FORCE

 `ifdef SYNTHESIS
    sim_ctx_force_must_never_be_synthesised u_sim_ctx_force_guard();
 `endif
    wire [10:0] s7a_ctx_len = 11'(`SIM_CTX_FORCE);
    initial $display("[KVRES] SIM_CTX_FORCE: attention window forced to ctx_len=%0d (reg 0x10 ignored by G5 only)",
                     11'(`SIM_CTX_FORCE));
`else
    wire [10:0] s7a_ctx_len = s7a_ctx_len_reg;
`endif

    wire [10:0] kv_append_pos = s7a_ctx_len - 11'd1;

    wire                     lvl_start_w;
    wire [2:0]               branch_j_w;
    wire [3:0]               level_i_w;

    wire [tree_score_pkg::NODE_W-1:0] fr_par_node_w;

    wire [tree_score_pkg::SLOT_W-1:0] fr_par_slot_w;
    wire                     branch_feed_w;
    wire                     out_walk_start_w;

    wire                     lvl_level_done_w;
    wire                     lvl_level_last_w;
    wire                     lvl_tree_done_w;
    wire                     lvl_out_walk_done_w;

    wire        lane_active   = DEPTH_N && (level_i_w != 4'd0);
    wire [10:0] lane_num_step = max_depth[10:0];
    reg         s7a_attn_start;

    wire [10:0] lane_slot     = 11'(kv_lane_chain_pkg::kv_lane_slot(
                                    {1'b0, s7a_ctx_len}, {1'b0, branch_j_w},
                                    {1'b0, lane_num_step}, 12'(level_i_w) - 12'd1));

    wire [10:0] kv_append_pos_eff = lane_active ? lane_slot     : kv_append_pos;
    wire [10:0] s7a_ctx_len_eff   = lane_active ? (s7a_ctx_len + 11'(level_i_w))
                                                : s7a_ctx_len;
    wire [10:0] rope_pos_eff      = 11'(kv_lane_chain_pkg::kv_lane_rope_pos(
                                                {1'b0, s7a_ctx_len},
                                                12'(level_i_w)));

`ifndef SYNTHESIS

    always_ff @(posedge ap_clk)
        if (ap_rst_n && lane_active)
            assert (kv_append_pos_eff < 11'(CTX_WIN))
                else $error("KV lane-chain slot %0d >= CTX_WIN %0d (ctx_len=%0d j=%0d i=%0d ns=%0d)",
                            kv_append_pos_eff, CTX_WIN, s7a_ctx_len,
                            branch_j_w, level_i_w, lane_num_step);

`endif

    localparam int ACT_FRAC = PRECISION_SAFE ? 11 : 12;

    wire signed [S7A_OUT_W-1:0] s7a_q_in, s7a_k_in, s7a_v_in;

    // Either requantize the complete Q/K/V vectors here, or pass the raw
    // INT32 buffers to attention_full_mha_sbuf for selected-head requantization.
    generate if (!REQUANT_HEADMUX) begin : g_seam_requant

        g4_qkv_seam #(
            .N_HEADS(S7A_NHEADS), .D_HEAD(S7A_DHEAD), .A_WIDTH(A_WIDTH),
            .G4_FRAC(ACT_FRAC), .G5_FRAC(10)
        ) u_g4_qkv_seam (

            .q_full_buf(q_head_view), .k_full_buf(k_head_view), .v_full_buf(v_head_view),
            .q_in(s7a_q_in), .k_in(s7a_k_in), .v_in(s7a_v_in)
        );
    end else begin : g_seam_bypass

        assign s7a_q_in = '0;
        assign s7a_k_in = '0;
        assign s7a_v_in = '0;
    end endgenerate

    wire                        s7a_attn_done;
    wire signed [S7A_OUT_W-1:0] s7a_attn_out_vec;

    (* keep = "true" *) wire [11:0] s7a_kv_res_cnt;

`ifndef SYNTHESIS

    logic kvres_warned;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) kvres_warned <= 1'b0;

        else if (s7a_attn_done && !kvres_warned &&
                 ({1'b0, s7a_ctx_len_eff} > s7a_kv_res_cnt)) begin
            kvres_warned <= 1'b1;
            $display("[KVRES] DECOUPLED t=%0t: attention sweep=%0d > KV rows actually written=%0d (reg 0x10 ctx_len=%0d) — %0d slot(s) are HOLES and are masked to -INF",
                     $time, s7a_ctx_len_eff, s7a_kv_res_cnt, ctx_len[10:0],
                     {1'b0, s7a_ctx_len_eff} - s7a_kv_res_cnt);
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge ap_clk)
        if (ap_rst_n && s7a_attn_start) begin
            assert (!$isunknown({level_i_w, rope_pos_eff, s7a_ctx_len_eff}))
                else $error("KV lane-chain RoPE coordinate contains X/Z");
            if (lane_active)
                assert (rope_pos_eff == s7a_ctx_len_eff)
                    else $error("KV child RoPE early/late: rope=%0d sweep=%0d level=%0d",
                                rope_pos_eff, s7a_ctx_len_eff, level_i_w);
            else
                assert ((rope_pos_eff + 11'd1) == s7a_ctx_len_eff)
                    else $error("KV root RoPE mismatch: rope=%0d sweep=%0d",
                                rope_pos_eff, s7a_ctx_len_eff);
        end
`endif
`endif

    // Full multi-head attention with buffered score reuse.
    attention_full_mha_sbuf #(
        .N_HEADS(S7A_NHEADS), .D_HEAD(S7A_DHEAD), .A_WIDTH(A_WIDTH),
        .CTX_WIN(CTX_WIN), .HEADS_PER_STACK(4), .CTX_MAX(1024),
        .SCORE_LATENCY(13), .SCORE_NORM_SHR(11),
        .K_RAM_STYLE(KV_K_RAM_STYLE), .V_RAM_STYLE(KV_V_RAM_STYLE),
        .COMBINE_LOCAL(COMBINE_LOCAL),

        .RAW_REQUANT(REQUANT_HEADMUX), .G4_FRAC(ACT_FRAC), .G5_FRAC(10),
        .ROPE_POS_DECOUPLE(DEPTH_N),
        .HEAD_PAR(HEAD_PAR),
        .ROPE_CS_PIPE(ROPE_CS_PIPE), .ROPE_POS_REP(ROPE_POS_REP),
        .KV_ADDR_REP(KV_ADDR_REP),
        .RES_CLR_REP(KV_RES_CLR_REP)
    ) u_attn_full (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start(s7a_attn_start),
        .ctx_len(s7a_ctx_len_eff),
        .append_pos(kv_append_pos_eff),
        .rope_pos_in(rope_pos_eff),

        .ctx_len_prefix(s7a_ctx_len),
        .lane_j({1'b0, branch_j_w}),
        .num_steps(max_depth[10:0]),
        .level_i({7'd0, level_i_w}),
        .done(s7a_attn_done),
        .q_in(s7a_q_in), .k_in(s7a_k_in), .v_in(s7a_v_in),

        .q_raw_buf(q_head_view), .k_raw_buf(k_head_view), .v_raw_buf(v_head_view),

        .pl_we(1'b0), .pl_head('0), .pl_pos('0),
        .pl_k_vec('0), .pl_v_vec('0), .do_kv_fill(1'b1),
        .out_vec(s7a_attn_out_vec), .dbg_head(),

        .kv_res_cnt(s7a_kv_res_cnt),

        .kv_res_clear(kv_res_clear_w),

        .b2p_round_en(accfix_b2p_en)
    );

    typedef enum logic [1:0] {
        S7G5_IDLE   = 2'd0,
        S7G5_ATTN   = 2'd1,
        S7G5_STREAM = 2'd2,
        S7G5_DONE   = 2'd3
    } s7g5_state_e;
    s7g5_state_e          s7g5_state;
    wire                  s7a_seam_wr;
    reg [S7A_CW-1:0]      s7a_wr_chunk_cnt;
    reg                   s7a_seam_clear;
    reg                   s7a_g5_done_r;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            s7g5_state       <= S7G5_IDLE;
            s7a_attn_start   <= 1'b0;
            s7a_wr_chunk_cnt <= '0;
            s7a_seam_clear   <= 1'b0;
            s7a_g5_done_r    <= 1'b0;
        end else begin
            s7a_attn_start <= 1'b0;
            s7a_seam_clear <= 1'b0;
            s7a_g5_done_r  <= 1'b0;
            case (s7g5_state)
                S7G5_IDLE: begin
                    if (S7A_FRONT_SEQ && g5_start) begin
                        s7a_attn_start <= 1'b1;
                        s7a_seam_clear <= 1'b1;
                        s7g5_state     <= S7G5_ATTN;
                    end
                end
                S7G5_ATTN: begin
                    if (s7a_attn_done) begin
                        s7a_wr_chunk_cnt <= '0;
                        s7g5_state       <= S7G5_STREAM;
                    end
                end
                S7G5_STREAM: begin

                    if (s7a_wr_chunk_cnt == S7A_CW'(S7A_NCH-1)) begin
                        s7g5_state <= S7G5_DONE;
                    end else begin
                        s7a_wr_chunk_cnt <= s7a_wr_chunk_cnt + 1'b1;
                    end
                end
                S7G5_DONE: begin
                    s7a_g5_done_r <= 1'b1;
                    s7g5_state    <= S7G5_IDLE;
                end
                default: s7g5_state <= S7G5_IDLE;
            endcase
        end
    end

    assign s7a_seam_wr = (s7g5_state == S7G5_STREAM);

    wire signed [K_PAR*A_WIDTH-1:0] s7a_cur_chunk =
        s7a_attn_out_vec[(s7a_wr_chunk_cnt*K_PAR)*A_WIDTH +: K_PAR*A_WIDTH];

    wire signed [K_PAR*A_WIDTH-1:0] seam_g5g6_rd;
    wire                            seam_g5g6_full;
    // Inter-stage activation seam storage.
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(S7A_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                      .AUTO_SEQ(1'b1), .FRAC_BITS(10), .RD_LAT(D1_RDLAT)) u_seam_g5g6 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(s7a_seam_wr), .wr_chunk_idx('0), .wr_chunk(s7a_cur_chunk),
        .wr_clear(s7a_seam_clear), .wr_count(), .full(seam_g5g6_full),
        .rd_valid(1'b1), .rd_chunk_idx(act_addr_w[S7A_CW-1:0]),
        .rd_chunk(seam_g5g6_rd),

        .rd_chunk_idx_b(act_addr_b_w[S7A_CW-1:0]),
        .rd_chunk_b(seam_g5g6_rd_b)
    );

    wire g5_done = S7A_FRONT_SEQ ? s7a_g5_done_r : g5_done_r;

    wire signed [32*16-1:0] dbg_g5_attn_vec =
        S7A_FRONT_SEQ ? s7a_attn_out_vec[32*16-1:0] : g5_attn_out_vec;
    wire                    dbg_g5_attn_done =
        S7A_FRONT_SEQ ? s7a_attn_done : g5_attn_done;

    wire [16*32-1:0] g8_dump_flat;
    wire [1:0]       g8_dump_status;

    wire [31:0] batch_test_ctrl;
    wire [31:0] batch_b0_lo_w, batch_b0_hi_w, batch_sib_lo_w, batch_sib_hi_w;

    wire [31:0] pmem_write_data_w;
    wire [7:0]  pmem_write_addr_w;
    wire        pmem_we_w;
    wire [7:0]  fetch_instr_w;
    wire [6:0]  fetch_pc_w;
    wire        isa_state_pulse_w;
    wire [4:0]  isa_state_id_w;
    wire        isa_is_state_w;
    wire        isa_running_w;
    wire        isa_halted_w;
    wire        isa_sr_write_en_w;
    wire [4:0]  isa_sr_write_sel_w;
    wire [15:0] isa_sr_write_data_w;
    wire        sr_read_en_w;
    wire [4:0]  sr_read_sel_w;
    wire [15:0] sr_read_data_w;
    wire        sr_read_ack_w;

    // Execution-config fields decoded by the actual special-register bank.
    // gemv_shared_ctrl latches these at GEMV START / TREE session start and
    // they are the sole source of bases, bounds, routing and numeric mode.
    wire [15:0] sr_activation_base_w;
    wire [16:0] sr_weight_base_w;
    wire [15:0] sr_tile_count_w;
    wire [15:0] sr_chunks_per_tile_w;
    wire [15:0] sr_chunks_per_scale_group_w;
    wire [15:0] sr_output_base_w;
    wire [16:0] sr_scale_base_w;
    wire [2:0]  sr_activation_source_w;
    wire [3:0]  sr_output_destination_w;
    wire [1:0]  sr_qzero_raw_w;
    wire [4:0]  sr_scale_frac_w;
    wire [15:0] sr_output_conversion_w;

    // Shared GEMV operation controller handoff (ISA START and TREE client).
    wire        fetch_gemv_pulse_w;
    wire        fetch_is_gemv_w;
    wire        gemv_busy_w;
    wire        gemv_done_w;
    wire        gemv_error_w;
    wire [15:0] gemv_active_tile_w;
    wire        gemv_array_start_w;
    wire [16:0] gemv_arr_weight_base_w;
    wire [16:0] gemv_arr_scale_base_w;
    wire [15:0] gemv_arr_chunks_per_tile_w;
    wire [15:0] gemv_arr_chunks_per_scale_group_w;
    wire        gemv_tile_valid_w;
    wire [3:0]  gemv_tile_dest_w;
    wire [15:0] gemv_store_idx_w;
    wire        gemv_store_ack_w;
    wire        gemv_client_error_w;
    wire        gemv_client_cfg_ok_w;
    wire        gemv_op_start_w;
    wire [3:0]  gemv_op_start_dest_w;
    wire        gemv_op_done_w;
    wire [3:0]  gemv_op_done_dest_w;
    wire        gemv_op_error_w;
    wire [2:0]  gemv_act_src_w;
    wire [15:0] gemv_act_base_w;
    wire [1:0]  gemv_qzero_w;
    wire [4:0]  gemv_scale_frac_w;
    wire        r6_store_ack;   // dest6 adapter drain (assigned per build)

    // AXI4-Lite register file.
    control_regs #(.COL_W(20)) u_ctrl (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .s_axi_control_awaddr(s_axi_control_awaddr), .s_axi_control_awvalid(s_axi_control_awvalid), .s_axi_control_awready(s_axi_control_awready),
        .s_axi_control_wdata(s_axi_control_wdata),   .s_axi_control_wstrb(s_axi_control_wstrb),    .s_axi_control_wvalid(s_axi_control_wvalid),  .s_axi_control_wready(s_axi_control_wready),
        .s_axi_control_bresp(s_axi_control_bresp),   .s_axi_control_bvalid(s_axi_control_bvalid),  .s_axi_control_bready(s_axi_control_bready),
        .s_axi_control_araddr(s_axi_control_araddr), .s_axi_control_arvalid(s_axi_control_arvalid), .s_axi_control_arready(s_axi_control_arready),
        .s_axi_control_rdata(s_axi_control_rdata),   .s_axi_control_rresp(s_axi_control_rresp),    .s_axi_control_rvalid(s_axi_control_rvalid),  .s_axi_control_rready(s_axi_control_rready),
        .start_pulse(start_pulse), .kernel_busy(kernel_busy), .kernel_done(kernel_done),
        .ctx_len(ctx_len), .num_frontiers(num_frontiers), .max_depth(max_depth),
        .out0_base(out0_base), .out1_base(out1_base),
        .out2_base(out2_base), .hbm_emb_base(hbm_emb_base),
        .lmfull_hbm_base(lmfull_hbm_base),

        .n_tiles_cfg(n_tiles_cfg),
        .kc_cfg(kc_cfg),

        .accfix_cfg(accfix_cfg_w),
        .obs_rd_idx(obs_rd_idx_w),
        .obs_rd_data(obs_rd_data_w),
        .obs_status_word(obs_status_word_w),

        .boot_lane_sel(ctrl_boot_lane_sel), .boot_wr_addr(ctrl_boot_wr_addr),
        .boot_wr_data(ctrl_boot_wr_data),   .boot_we(ctrl_boot_we),
        .boot_is_emb(ctrl_boot_is_emb),
        .boot_is_hfc(ctrl_boot_is_hfc),
        .boot_gamma_sel(ctrl_boot_gamma_sel),

        .boot_dma_start(ctrl_boot_dma_start),
        .boot_dma_is_scale(ctrl_boot_dma_is_scale),
        .boot_dma_is_emb(ctrl_boot_dma_is_emb),
        .boot_dma_is_hfc(ctrl_boot_dma_is_hfc),
        .boot_hbm_base(ctrl_boot_hbm_base),
        .boot_dma_n_cols(ctrl_boot_dma_n_cols),
        .boot_dma_col_start(ctrl_boot_dma_col_start),
        .boot_dma_col_step(ctrl_boot_dma_col_step),
        .boot_dma_cpc(ctrl_boot_dma_cpc),
        .boot_dma_stage_base(ctrl_boot_dma_stage_base),
        .boot_dma_busy(dma_busy),
        .boot_dma_done(dma_done),
        .boot_dma_timeout(dma_timeout),

        .out_dma_busy(out_dma_busy),
        .out_dma_done(out_dma_done),
        .out_dma_err(out_dma_err),

        .g8_wd_err(g8_dbg_err),
        .dbg_fsmstate(dbg_fsmstate_word),
        .dbg_activity(dbg_activity_word),

        .dbg_sel(dbg_sel_w),
        .dbg_cyclecount(dbg_cyclecount_word),
        .dbg_checksum(dbg_checksum_word),
        .dbg_g8progress(dbg_g8progress_word),
        .dbg_g8araddr_lo(dbg_g8araddr_lo_word),
        .dbg_g8araddr_hi(dbg_g8araddr_hi_word),
        .dbg_rearm(dbg_rearm_word),
        .dbg_bootstream(dbg_bootstream_word),

        .probe_sel(prb_sel_w), .probe_sel_we(prb_sel_we),
        .probe_addr_we(prb_addr_we), .probe_addr_wdata(prb_addr_wdata),
        .probe_data_re(prb_data_re),
        .probe_addr(prb_addr_word), .probe_data(prb_data_word),
        .probe_xor(prb_xor_word), .probe_sum(prb_sum_word), .probe_count(prb_count_word),

        .dump_en(dump_en), .out3_base(out3_base), .out4_base(out4_base),

        .dbg_stageout_sample(dbg_stageout_sample_word),
        .dbg_stageout_count(dbg_stageout_count_word),

        .dbg_cap_stage_sel(dbg_cap_stage_sel_w),
        .dbg_cap_widx_we(dbg_cap_widx_we),
        .dbg_cap_widx_wdata(dbg_cap_widx_wdata),
        .dbg_cap_val_re(dbg_cap_val_re),
        .dbg_cap_widx_word(dbg_cap_widx_word),
        .dbg_cap_val_word(dbg_cap_val_word),
        .dbg_cap_meta_word(dbg_cap_meta_word),

        .res_sel(dbg_res_sel_w),
        .res_clear(dbg_res_clear_w),
        .res_crc_word(dbg_res_crc_word),
        .res_sum_word(dbg_res_sum_word),
        .res_count_word(dbg_res_count_word),

        .kv_res_clear(kv_res_clear_w),
        .kv_res_status_word(kv_res_status_word),

        .maxprobe_build_id({16'hE4B7, BUILD_TAG, PROBE_REV}),
        .cap_crc_word(dbg_cap_crc_word),

        .inj_en_we(inj_en_we_w),       .inj_en_wdata(inj_en_wdata_w),
        .inj_widx_we(inj_widx_we_w),   .inj_widx_wdata(inj_widx_wdata_w),
        .inj_data_we(inj_data_we_w),   .inj_data_wdata(inj_data_wdata_w),
        .inj_sel_we(inj_sel_we_w),     .inj_sel_wdata(inj_sel_wdata_w),
        .inj_en_word(inj_en_rb_w),     .inj_widx_word(inj_widx_rb_w),
        .inj_status_word(inj_status_w),

        .g8_dump_flat(g8_dump_flat),
        .g8_dump_status(g8_dump_status),
        .lmtap_seam_word(lmtap_word_seam),
        .lmtap_guard_word(lmtap_word_guard),
        .lmtap_ad_word(lmtap_word_ad),

        .batch_test_ctrl(batch_test_ctrl),
        .batch_b0_lo(batch_b0_lo_w),   .batch_b0_hi(batch_b0_hi_w),
        .batch_sib_lo(batch_sib_lo_w), .batch_sib_hi(batch_sib_hi_w),

        .pmem_write_data(pmem_write_data_w),
        .pmem_write_addr(pmem_write_addr_w),
        .pmem_we(pmem_we_w),
        .fetch_instr(fetch_instr_w),
        .fetch_pc(fetch_pc_w),
        .sr_read_sel(sr_read_sel_w),
        .sr_read_en(sr_read_en_w),
        .sr_read_data(sr_read_data_w),
        .sr_read_ack(sr_read_ack_w),
        .gemv_busy(gemv_busy_w),
        .gemv_done_sticky(gemv_done_w),
        .gemv_error_sticky(gemv_error_w),
        .gemv_active_tile(gemv_active_tile_w)
    );

    instruction_fetcher u_fetcher (
        .clk        (ap_clk),
        .rst_en     (ap_rst_n),
        .start_en   (start_pulse),
        .adv_en     (isa_adv),
        .write_en   (pmem_we_w),
        .write_addr (pmem_write_addr_w),
        .write_data (pmem_write_data_w),
        .instr      (fetch_instr_w),
        .pc_out     (fetch_pc_w),
        .state_pulse(isa_state_pulse_w),
        .state_id   (isa_state_id_w),
        .is_state   (isa_is_state_w),
        .running    (isa_running_w),
        .halted     (isa_halted_w),
        .sr_write_en  (isa_sr_write_en_w),
        .sr_write_sel (isa_sr_write_sel_w),
        .sr_write_data(isa_sr_write_data_w),
        .gemv_pulse (fetch_gemv_pulse_w),
        .is_gemv    (fetch_is_gemv_w),
        // Implemented STATE operands: G1=1, G5=3, G7_SILU=7, G11=9,
        // TREE=11, GLM=13, G8=14. Retired shared-stage IDs (G4=2, G6=4,
        // G7G=5, G7U=6, G9=8, G12=10) and unassigned IDs decode as HALT.
        .state_valid_mask (32'h0000_6A8A)
    );

    special_regs u_special_regs (
        .clk        (ap_clk),
        .rst_en     (ap_rst_n),
        .write_en   (isa_sr_write_en_w),
        .write_select(isa_sr_write_sel_w),
        .write_data (isa_sr_write_data_w),
        .read_en    (sr_read_en_w),
        .read_select(sr_read_sel_w),
        .read_data  (sr_read_data_w),
        .read_ack   (sr_read_ack_w),
        .activation_base       (sr_activation_base_w),
        .weight_base           (sr_weight_base_w),
        .tile_count            (sr_tile_count_w),
        .chunks_per_tile       (sr_chunks_per_tile_w),
        .chunks_per_scale_group(sr_chunks_per_scale_group_w),
        .output_base           (sr_output_base_w),
        .scale_base            (sr_scale_base_w),
        .activation_source     (sr_activation_source_w),
        .output_destination    (sr_output_destination_w),
        .qzero_raw             (sr_qzero_raw_w),
        .scale_frac            (sr_scale_frac_w),
        .output_conversion     (sr_output_conversion_w)
    );

    generate
    if (STAGE_INJECT) begin : g_stage_inject
        stage_inject #(
            .NSTAGE(INJ_NSTAGE), .MAX_W(INJ_MAX_W), .WIDX_W($clog2(INJ_MAX_W))
        ) u_stage_inject (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .en_we(inj_en_we_w),       .en_wdata(inj_en_wdata_w[INJ_NSTAGE-1:0]),
            .widx_we(inj_widx_we_w),   .widx_wdata(inj_widx_wdata_w[$clog2(INJ_MAX_W)-1:0]),
            .data_we(inj_data_we_w),   .data_wdata(inj_data_wdata_w),
            .sel_we(inj_sel_we_w),     .sel_wdata(inj_sel_wdata_w[4:0]),
            .en_word(inj_en_rb_w),     .widx_word(inj_widx_rb_w),
            .status_word(inj_status_w),
            .inject_en(inject_en),     .inj_bus(inj_bus)
        );
    end else begin : g_no_inject

        assign inject_en    = '0;
        assign inj_bus      = '0;
        assign inj_en_rb_w  = 32'b0;
        assign inj_widx_rb_w = 32'b0;
        assign inj_status_w = 32'b0;
    end
    endgenerate

    wire        ctrl_boot_dma_start;
    wire        ctrl_boot_dma_is_scale;
    wire        ctrl_boot_dma_is_emb;
    wire        ctrl_boot_dma_is_hfc;
    wire [63:0] ctrl_boot_hbm_base;
    wire [19:0] ctrl_boot_dma_n_cols;
    wire [19:0] ctrl_boot_dma_col_start;
    wire [19:0] ctrl_boot_dma_col_step;
    wire [15:0] ctrl_boot_dma_cpc;
    wire [16:0] ctrl_boot_dma_stage_base;

    wire        dma_busy;
    wire        dma_done;
    wire        dma_timeout;
    wire        dma_we;
    wire [15:0] dma_lane_sel;
    wire [16:0] dma_wr_addr;
    wire [31:0] dma_wr_data;

    wire        dma_tgt_emb;
    wire        dma_tgt_hfc;

    wire        dma_sc_we;
    wire [15:0] dma_sc_lane_sel;
    wire [16:0] dma_sc_wr_addr;
    wire [15:0] dma_sc_wr_data;

    // HBM-to-on-chip boot loader.
    weight_dma_loader #(
        .LANES(LANES), .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(BOOT_AXI_DATA_W),
        .AXI_ID_W(BOOT_AXI_ID_W), .BURST_LEN(BOOT_BURST_LEN), .URAM_ADDR_W(17),
        .COL_W(20)
    ) u_dma_loader (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .boot_dma_start(ctrl_boot_dma_start),
        .boot_is_scale(ctrl_boot_dma_is_scale),
        .boot_is_emb(ctrl_boot_dma_is_emb),
        .boot_is_hfc(ctrl_boot_dma_is_hfc),
        .dma_tgt_emb(dma_tgt_emb),
        .dma_tgt_hfc(dma_tgt_hfc),
        .boot_hbm_base(ctrl_boot_hbm_base),
        .boot_n_cols(ctrl_boot_dma_n_cols),
        .boot_col_start(ctrl_boot_dma_col_start),
        .boot_col_step(ctrl_boot_dma_col_step),
        .boot_cpc(ctrl_boot_dma_cpc),
        .boot_stage_base(ctrl_boot_dma_stage_base),
        .boot_busy(dma_busy),
        .boot_done(dma_done),
        .boot_timeout(dma_timeout),
        .dma_we(dma_we), .dma_lane_sel(dma_lane_sel),
        .dma_wr_addr(dma_wr_addr), .dma_wr_data(dma_wr_data),

        .sc_we(dma_sc_we), .sc_lane_sel(dma_sc_lane_sel),
        .sc_wr_addr(dma_sc_wr_addr), .sc_wr_data(dma_sc_wr_data),
        .m_axi_boot_arid(m_axi_boot_arid),     .m_axi_boot_araddr(m_axi_boot_araddr),
        .m_axi_boot_arlen(m_axi_boot_arlen),   .m_axi_boot_arsize(m_axi_boot_arsize),
        .m_axi_boot_arburst(m_axi_boot_arburst), .m_axi_boot_arvalid(m_axi_boot_arvalid),
        .m_axi_boot_arready(m_axi_boot_arready),
        .m_axi_boot_rid(m_axi_boot_rid),       .m_axi_boot_rdata(m_axi_boot_rdata),
        .m_axi_boot_rresp(m_axi_boot_rresp),   .m_axi_boot_rlast(m_axi_boot_rlast),
        .m_axi_boot_rvalid(m_axi_boot_rvalid), .m_axi_boot_rready(m_axi_boot_rready)
    );

    wire [15:0] ctrl_boot_lane_sel;
    wire [16:0] ctrl_boot_wr_addr;
    wire [31:0] ctrl_boot_wr_data;
    wire        ctrl_boot_we;
    wire        ctrl_boot_is_emb;
    wire        ctrl_boot_is_hfc;
    wire [2:0]  ctrl_boot_gamma_sel;

    wire        ctrl_boot_gamma_on = (ctrl_boot_gamma_sel != 3'd0);
    wire        ctrl_boot_we_wt  = ctrl_boot_we & ~ctrl_boot_is_emb & ~ctrl_boot_is_hfc & ~ctrl_boot_gamma_on;
    wire        ctrl_boot_we_emb = ctrl_boot_we &  ctrl_boot_is_emb & ~ctrl_boot_is_hfc & ~ctrl_boot_gamma_on;
    wire        ctrl_boot_we_hfc = ctrl_boot_we &  ctrl_boot_is_hfc & ~ctrl_boot_is_emb & ~ctrl_boot_gamma_on;

    wire        ctrl_boot_we_gamma = ctrl_boot_we & ctrl_boot_gamma_on
                                     & ~ctrl_boot_is_emb & ~ctrl_boot_is_hfc;

    assign gamma_we_hidden = ctrl_boot_we_gamma & (ctrl_boot_gamma_sel == 3'd1);
    assign gamma_we_enorm  = ctrl_boot_we_gamma & (ctrl_boot_gamma_sel == 3'd2);
    assign gamma_we_postat = ctrl_boot_we_gamma & (ctrl_boot_gamma_sel == 3'd3);
    assign gamma_we_final  = ctrl_boot_we_gamma & (ctrl_boot_gamma_sel == 3'd4);
    assign gamma_word_addr = ctrl_boot_wr_addr;
    assign gamma_wr_chunk  = gamma_word_addr[16:EMB_WPC_LOG2];
    assign gamma_wr_word   = gamma_word_addr[EMB_WPC_LOG2-1:0];
    assign gamma_wr_data32 = ctrl_boot_wr_data;

    wire        dma_we_emb = dma_we &  dma_tgt_emb & ~dma_tgt_hfc;
    wire        dma_we_hfc = dma_we &  dma_tgt_hfc & ~dma_tgt_emb;
    wire        dma_we_wt  = dma_we & ~dma_tgt_emb & ~dma_tgt_hfc;

    assign emb_we        = dma_we_emb | ctrl_boot_we_emb;
    assign emb_word_addr = dma_we_emb ? dma_wr_addr : ctrl_boot_wr_addr;
    assign emb_wr_chunk  = emb_word_addr[16:EMB_WPC_LOG2];
    assign emb_wr_word   = emb_word_addr[EMB_WPC_LOG2-1:0];
    assign emb_wr_data32 = dma_we_emb ? dma_wr_data : ctrl_boot_wr_data;

    assign hfc_we        = dma_we_hfc | ctrl_boot_we_hfc;
    assign hfc_word_addr = dma_we_hfc ? dma_wr_addr : ctrl_boot_wr_addr;
    assign hfc_wr_chunk  = hfc_word_addr[16:EMB_WPC_LOG2];
    assign hfc_wr_word   = hfc_word_addr[EMB_WPC_LOG2-1:0];
    assign hfc_wr_data32 = dma_we_hfc ? dma_wr_data : ctrl_boot_wr_data;

    wire        eff_boot_we       = dma_we_wt | ctrl_boot_we_wt | boot_we;
    wire [15:0] eff_boot_lane_sel = dma_we_wt     ? dma_lane_sel
                                  : ctrl_boot_we_wt ? ctrl_boot_lane_sel
                                                    : boot_lane_sel;
    wire [16:0] eff_boot_wr_addr  = dma_we_wt     ? dma_wr_addr
                                  : ctrl_boot_we_wt ? ctrl_boot_wr_addr
                                                    : boot_wr_addr;
    wire [31:0] eff_boot_wr_data  = dma_we_wt     ? dma_wr_data
                                  : ctrl_boot_we_wt ? ctrl_boot_wr_data
                                                    : boot_wr_data;

    wire g8_start;
    wire g8_done;

    // ------------------------------------------------------------------
    // ISA stage scheduler and kernel handshake.
    //
    // The instruction fetcher drives the stage scheduling that top_fsm
    // used to: a STATE instruction asserts the same one-cycle stage-start
    // pulse the FSM issued on state entry, and the stage's done advances
    // the PC to the next instruction. State IDs match the top_fsm state
    // enum numerically (see COMPONENTS.md "ISA State Encodings &
    // Instruction Set" and ISA_TESTING_PLAN.md). top_fsm.sv is retained
    // uninstantiated as the reference for the future depth-N
    // re-implementation.
    // ------------------------------------------------------------------
    localparam logic [4:0] ISA_S_IDLE     = 5'd0;
    localparam logic [4:0] ISA_S_G1       = 5'd1;   // implemented
    localparam logic [4:0] ISA_S_G5       = 5'd3;   // implemented
    localparam logic [4:0] ISA_S_G7_SILU  = 5'd7;   // implemented
    localparam logic [4:0] ISA_S_G11      = 5'd9;   // implemented
    localparam logic [4:0] ISA_S_TREE     = 5'd11;  // implemented
    localparam logic [4:0] ISA_S_GLM      = 5'd13;  // implemented
    localparam logic [4:0] ISA_S_G8       = 5'd14;  // implemented
    // Retired shared-stage IDs: G4=2, G6=4, G7G=5, G7U=6, G9=8, G12=10.
    // The fetcher masks them to HALT (state_valid_mask above); GEMV opcode
    // 0x80 is the only path that launches the shared engine now.

    // A STATE instruction stays at its PMEM byte until its stage done
    // fires; a GEMV byte stays until the operation reports done or a
    // validated configuration error. The wait register selects WHICH event
    // advances the PC plus a GEMV tag, so no raw OR of dones can free-run
    // the PC and a rejected GEMV can never hang the program.
    reg        isa_wait;
    reg [4:0]  isa_wait_id;
    reg        isa_wait_gemv;
    wire       isa_adv;

    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            isa_wait      <= 1'b0;
            isa_wait_id   <= ISA_S_IDLE;
            isa_wait_gemv <= 1'b0;
        end else if (start_pulse) begin
            // A host restart clears any pending wait of the previous program.
            isa_wait      <= 1'b0;
            isa_wait_id   <= ISA_S_IDLE;
            isa_wait_gemv <= 1'b0;
        end else if (isa_state_pulse_w) begin
            isa_wait      <= 1'b1;
            isa_wait_id   <= isa_state_id_w;
            isa_wait_gemv <= 1'b0;
        end else if (fetch_gemv_pulse_w) begin
            isa_wait      <= 1'b1;
            isa_wait_id   <= ISA_S_IDLE;
            isa_wait_gemv <= 1'b1;
        end else if (isa_adv) begin
            isa_wait      <= 1'b0;
            isa_wait_gemv <= 1'b0;
        end
    end

    // G1 advances on the hidden norm's done, not a join of both norms.
    // TREE also terminates on a rejected client request so an invalid G12
    // configuration cannot hang the ISA program.
    wire isa_state_done =
          (isa_wait_id == ISA_S_G1)      ? g1_done        :
          (isa_wait_id == ISA_S_G5)      ? g5_done        :
          (isa_wait_id == ISA_S_G7_SILU) ? g7p_done       :
          (isa_wait_id == ISA_S_G11)     ? g11_done       :
          (isa_wait_id == ISA_S_TREE)    ? (tree_done | gemv_client_error_w) :
          (isa_wait_id == ISA_S_GLM)     ? glm_done       :
          (isa_wait_id == ISA_S_G8)      ? g8_done        : 1'b0;
    wire isa_gemv_done = gemv_op_done_w | gemv_op_error_w;
    assign isa_adv = isa_wait && (isa_wait_gemv ? isa_gemv_done : isa_state_done);

    // STATE starts remain for the non-shared stages that still own real
    // side effects (norm, attention, SiLU, G11, GLM, TREE, G8).
    assign g1_start   = isa_state_pulse_w && (isa_state_id_w == ISA_S_G1);
    assign g5_start   = isa_state_pulse_w && (isa_state_id_w == ISA_S_G5);
    assign g7p_start  = isa_state_pulse_w && (isa_state_id_w == ISA_S_G7_SILU);
    assign g11_start  = isa_state_pulse_w && (isa_state_id_w == ISA_S_G11);
    assign tree_start = isa_state_pulse_w && (isa_state_id_w == ISA_S_TREE);
    assign glm_start  = isa_state_pulse_w && (isa_state_id_w == ISA_S_GLM);
    assign g8_start   = isa_state_pulse_w && (isa_state_id_w == ISA_S_G8);

    // Shared-engine destination start/clear pulses now come from the GEMV
    // operation controller, not retired STATE operands. They retain the
    // seam wr_clear / residual-trigger side effects of the old stages.
    assign g6_start  = gemv_op_start_w && (gemv_op_start_dest_w == 4'd3);
    assign g7g_start = gemv_op_start_w && (gemv_op_start_dest_w == 4'd4);
    assign g7u_start = gemv_op_start_w && (gemv_op_start_dest_w == 4'd5);
    assign g9_start  = gemv_op_start_w && (gemv_op_start_dest_w == 4'd6);
    // Standalone G12 STATE program start removed: TREE is the sole G12
    // client and drives tt_gemv_start through the same controller.

    // Observation-only stage view: while a shared GEMV operation (including
    // a TREE client tile) is active, expose the destination's legacy stage
    // code so probes/stage capture keep their existing attribution. Non-
    // GEMV cycles expose 0 (the shared engine is the only array driver).
    wire [3:0] gemv_stage_view =
          ((gemv_tile_dest_w == 4'd9) || (gemv_tile_dest_w == 4'd0) ||
           (gemv_tile_dest_w == 4'd1) || (gemv_tile_dest_w == 4'd2)) ? 4'd0 :
          (gemv_tile_dest_w == 4'd3) ? 4'd1 :
          (gemv_tile_dest_w == 4'd4) ? 4'd2 :
          (gemv_tile_dest_w == 4'd5) ? 4'd3 :
          (gemv_tile_dest_w == 4'd6) ? 4'd4 :
          (gemv_tile_dest_w == 4'd7) ? 4'd5 : 4'd0;
    assign stage_sel = gemv_busy_w ? gemv_stage_view : 4'd0;

    // Kernel handshake: busy while the program runs, one-cycle done on
    // HALT (control_regs clears ap_start off kernel_busy, so start_pulse
    // is a clean single cycle).
    assign kernel_busy = isa_running_w;
    assign kernel_done = isa_halted_w;
    // Debug tap: state ID for a STATE instruction, 31 for GEMV, else 0.
    wire [4:0] isa_dbg_state = isa_is_state_w ? isa_state_id_w :
                               (fetch_is_gemv_w ? 5'd31 : ISA_S_IDLE);

    // ------------------------------------------------------------------
    // Sole shared GEMV operation controller. ISA GEMV (0x80) and the TREE
    // client both use the SAME latched special-register configuration.
    // ------------------------------------------------------------------
    wire gemv_act_ready_w;   // source readiness, assigned with the source mux
    gemv_shared_ctrl #(
        .LANES(LANES), .K_PAR(K_PAR), .K_MAX(K_MAX),
        .G9_NTILES(FORREASON ? (6144 / LANES) : (3072 / LANES))
    ) u_gemv_ctrl (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .isa_start(fetch_gemv_pulse_w),
        .cfg_activation_base(sr_activation_base_w),
        .cfg_weight_base(sr_weight_base_w),
        .cfg_tile_count(sr_tile_count_w),
        .cfg_chunks_per_tile(sr_chunks_per_tile_w),
        .cfg_chunks_per_scale_group(sr_chunks_per_scale_group_w),
        .cfg_output_base(sr_output_base_w),
        .cfg_scale_base(sr_scale_base_w),
        .cfg_activation_source(sr_activation_source_w),
        .cfg_output_destination(sr_output_destination_w),
        .cfg_qzero_raw(sr_qzero_raw_w),
        .cfg_scale_frac(sr_scale_frac_w),
        .cfg_output_conversion(sr_output_conversion_w),
        .act_ready(gemv_act_ready_w),
        .array_done(array_done),
        .array_start(gemv_array_start_w),
        .arr_weight_base(gemv_arr_weight_base_w),
        .arr_scale_base(gemv_arr_scale_base_w),
        .arr_chunks_per_tile(gemv_arr_chunks_per_tile_w),
        .arr_chunks_per_scale_group(gemv_arr_chunks_per_scale_group_w),
        .tile_valid(gemv_tile_valid_w),
        .tile_dest(gemv_tile_dest_w),
        .store_idx(gemv_store_idx_w),
        .store_ack(gemv_store_ack_w),
        .client_session_start(tree_start),
        .client_active(tt_stage_g12),
        .client_start(tt_gemv_start),
        .client_tile_idx(tt_gemv_tile_idx),
        .client_error(gemv_client_error_w),
        .client_config_ok(gemv_client_cfg_ok_w),
        .busy(gemv_busy_w),
        .done_sticky(gemv_done_w),
        .error_sticky(gemv_error_w),
        .active_tile(gemv_active_tile_w),
        .op_start(gemv_op_start_w),
        .op_start_dest(gemv_op_start_dest_w),
        .op_done(gemv_op_done_w),
        .op_done_dest(gemv_op_done_dest_w),
        .op_error(gemv_op_error_w),
        .latched_valid(),
        .latched_activation_source(gemv_act_src_w),
        .latched_activation_base(gemv_act_base_w),
        .latched_qzero_raw(gemv_qzero_w),
        .latched_scale_frac(gemv_scale_frac_w),
        .latched_destination(),
        .latched_output_conversion(),
        .latched_tile_count()
    );

    // An invalid TREE configuration must not start an empty tree walk. The
    // controller still sees the raw tree_start so it can latch/validate and
    // terminate the ISA wait via client_error; only the tiler start is gated.
    wire tree_start_eff = tree_start && gemv_client_cfg_ok_w;

    // Depth-N loop control is not expressible in the current ISA (its
    // sequencing is data-dependent on branch_j/level_i and the level_done
    // / level_last / tree_all_done handshakes — see ISA_TESTING_PLAN.md).
    // Tied off; re-implement from top_fsm.sv when conditional jumps land.
    assign lvl_start_w      = 1'b0;
    assign branch_feed_w    = 1'b0;
    assign out_walk_start_w = 1'b0;
    assign emb_start_w      = 1'b0;
    assign branch_j_w       = 3'd0;
    assign level_i_w        = 4'd0;

    // Fail loudly rather than silently producing no output: with the loop
    // signals tied off, a DEPTH_N build's tree walk never runs (m_axi_out
    // is muxed to the tree path) while ap_done still completes. Guard
    // until conditional-jump support re-enables the loop (ISA_TESTING_PLAN.md).
    generate if (DEPTH_N) begin : g_depthn_unsupported
        initial begin
            $fatal(1, "DEPTH_N=1 is not supported by the ISA scheduler yet (depth-N loop tied off); use a DEPTH_N=0 build");
        end
    end endgenerate


    generate if (DEPTH_N) begin : g_depth_n

        wire [tree_score_pkg::K_F*tree_score_pkg::NODE_W-1:0] tl_surv_node;
        wire [tree_score_pkg::K_F*tree_score_pkg::NODE_W-1:0] tl_surv_parent;
        wire [tree_score_pkg::K_F*tree_score_pkg::SLOT_W-1:0] tl_surv_parent_slot;
        wire [tree_score_pkg::K_F*tree_score_pkg::TOK_W-1:0]  tl_surv_token;
        wire [tree_score_pkg::K_F*tree_score_pkg::SCORE_W-1:0] tl_surv_score;
        wire [tree_score_pkg::DEPTH_W-1:0]                    tl_level_idx;
        wire [tree_score_pkg::POS_W-1:0]                      tl_level_pos;
        wire                                                  tl_cand_req;
        wire [tree_score_pkg::NODE_W-1:0]                     tl_n_nodes;
        wire [tree_score_pkg::DEPTH_W-1:0]                    tl_levels_run;

        wire [tree_score_pkg::NODE_W-1:0]  tn_walk_idx;
        wire [tree_score_pkg::TOK_W-1:0]   tn_walk_token;
        wire [tree_score_pkg::NODE_W-1:0]  tn_walk_parent;
        wire [tree_score_pkg::DEPTH_W-1:0] tn_walk_depth;
        wire [tree_score_pkg::POS_W-1:0]   tn_walk_pos;
        wire signed [tree_score_pkg::SCORE_W-1:0] tn_walk_score;

        wire                                 br_start;
        wire [tree_score_pkg::K_F*32-1:0]    br_frontier_f32;
        wire [tree_score_pkg::K_F*tree_score_pkg::TOK_W-1:0] br_frontier_tok;
        wire                                 br_done;
        wire [tree_score_pkg::K_F*32-1:0]    br_prob_f32;
        wire [tree_score_pkg::K_F*tree_score_pkg::TOK_W-1:0] br_tok;
        wire                                 fseq_cand_valid;
        wire [31:0]                          fseq_cand_prob_f32;
        wire [tree_score_pkg::TOK_W-1:0]     fseq_cand_token;
        wire                                 fseq_feed_busy;

        tree_level_fsm u_tree_level (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start        (lvl_start_w),
            .ctx_len_i    (ctx_len),
            .max_depth_i  (max_depth),

            .cand_req     (tl_cand_req),
            .cand_valid   (fseq_cand_valid),
            .cand_prob_f32(fseq_cand_prob_f32),
            .cand_token   (fseq_cand_token),

            .level_done   (lvl_level_done_w),
            .level_last   (lvl_level_last_w),
            .level_idx_o  (tl_level_idx),
            .level_pos_o  (tl_level_pos),
            .surv_node_o  (tl_surv_node),
            .surv_parent_o(tl_surv_parent),
            .surv_parent_slot_o(tl_surv_parent_slot),
            .surv_token_o (tl_surv_token),
            .surv_score_o (tl_surv_score),

            .tn_rd_idx    (tn_walk_idx),
            .tn_rd_token  (tn_walk_token),
            .tn_rd_parent (tn_walk_parent),
            .tn_rd_depth  (tn_walk_depth),
            .tn_rd_pos    (tn_walk_pos),
            .tn_rd_score  (tn_walk_score),

            .busy         (),
            .done         (lvl_tree_done_w),
            .levels_run_o (tl_levels_run),
            .n_nodes_o    (tl_n_nodes)
        );

        assign fr_par_node_w =
            tl_surv_parent[branch_j_w*tree_score_pkg::NODE_W +: tree_score_pkg::NODE_W];

        assign fr_par_slot_w =
            tl_surv_parent_slot[branch_j_w*tree_score_pkg::SLOT_W +: tree_score_pkg::SLOT_W];

        if (EMBG_EN) begin : g_embg

            wire [tree_score_pkg::TOK_W-1:0] embg_tok =
                tl_surv_token[branch_j_w*tree_score_pkg::TOK_W +: tree_score_pkg::TOK_W];

            wire [EMB_NCH-1:0]                     embg_arvalid;
            wire [EMB_NCH-1:0]                     embg_arready;
            wire [EMB_NCH*BOOT_AXI_ADDR_W-1:0]     embg_araddr;
            wire [EMB_NCH*8-1:0]                   embg_arlen;
            wire [EMB_NCH*3-1:0]                   embg_arsize;
            wire [EMB_NCH*2-1:0]                   embg_arburst;
            wire [EMB_NCH-1:0]                     embg_rvalid;
            wire [EMB_NCH-1:0]                     embg_rready;
            wire [EMB_NCH*256-1:0]                 embg_rdata;
            wire [EMB_NCH-1:0]                     embg_rlast;

            emb_gather #(
                .H          (3072),
                .IDX_W      (tree_score_pkg::TOK_W),
                .AXI_ADDR_W (BOOT_AXI_ADDR_W),
                .AXI_DATA_W (256),
                .EMB_FMT    (1'b1),
                .NCH        (EMB_NCH)
            ) u_emb_gather (
                .clk         (ap_clk),
                .rst_n       (ap_rst_n),
                .start       (emb_start_w),
                .tok_id      (embg_tok),
                .hbm_emb_base(hbm_emb_base),
                .m_arvalid   (embg_arvalid),
                .m_arready   (embg_arready),
                .m_araddr    (embg_araddr),
                .m_arlen     (embg_arlen),
                .m_arsize    (embg_arsize),
                .m_arburst   (embg_arburst),
                .m_rvalid    (embg_rvalid),
                .m_rready    (embg_rready),
                .m_rdata     (embg_rdata),
                .m_rlast     (embg_rlast),
                .o_we        (embg_o_we),
                .o_chunk     (embg_o_chunk),
                .o_data      (embg_o_data),
                .done        (emb_done_w),
                .beats_dbg   ()
            );

            assign m_axi_emb_arvalid = embg_arvalid[0];
            assign m_axi_emb_araddr  = embg_araddr[0*BOOT_AXI_ADDR_W +: BOOT_AXI_ADDR_W];
            assign m_axi_emb_arlen   = embg_arlen  [0*8 +: 8];
            assign m_axi_emb_arsize  = embg_arsize [0*3 +: 3];
            assign m_axi_emb_arburst = embg_arburst[0*2 +: 2];
            assign m_axi_emb_rready  = embg_rready [0];
            assign embg_arready[0]   = m_axi_emb_arready;
            assign embg_rvalid [0]   = m_axi_emb_rvalid;
            assign embg_rlast  [0]   = m_axi_emb_rlast;
            assign embg_rdata[0*256 +: 256] = m_axi_emb_rdata;

            assign m_axi_emb_arid = '0;

            if (EMB_NCH > 1) begin : g_embb
                assign m_axi_embb_arvalid = embg_arvalid[1];
                assign m_axi_embb_araddr  = embg_araddr[1*BOOT_AXI_ADDR_W +: BOOT_AXI_ADDR_W];
                assign m_axi_embb_arlen   = embg_arlen  [1*8 +: 8];
                assign m_axi_embb_arsize  = embg_arsize [1*3 +: 3];
                assign m_axi_embb_arburst = embg_arburst[1*2 +: 2];
                assign m_axi_embb_rready  = embg_rready [1];
                assign embg_arready[1]    = m_axi_embb_arready;
                assign embg_rvalid [1]    = m_axi_embb_rvalid;
                assign embg_rlast  [1]    = m_axi_embb_rlast;
                assign embg_rdata[1*256 +: 256] = m_axi_embb_rdata;
                assign m_axi_embb_arid    = '0;
            end
        end

        tree_softmax_bridge #(
            .K_C(G8_K_C), .CIDX_W(G8_CIDX_W)
        ) u_tree_bridge (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start          (br_start),

            .kc_cfg         (kc_cfg),
            .kc_en          (accfix_kc_en),
            .cand_rdaddr    (bridge_cand_rdaddr),
            .cand_rd_active (bridge_cand_rd_active),
            .cand_rdaddr_nxt(bridge_cand_rdaddr_nxt),
            .cand_logit_f32 (bridge_cand_logit),
            .frontier_f32   (br_frontier_f32),
            .frontier_tok   (br_frontier_tok),
            .done           (br_done),
            .prob_f32_o     (br_prob_f32),
            .tok_o          (br_tok)
        );

        tree_feed_seq #(
            .G8_IDX_W(TREE_IDX_W)
        ) u_tree_feed_seq (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .branch_feed        (branch_feed_w),
            .g8_frontier_scores (ow_topk_scores),
            .g8_frontier_ids    (ow_topk_indices),
            .bridge_start       (br_start),
            .bridge_frontier_f32(br_frontier_f32),
            .bridge_frontier_tok(br_frontier_tok),
            .bridge_done        (br_done),
            .bridge_prob_f32    (br_prob_f32),
            .bridge_tok         (br_tok),
            .cand_req           (tl_cand_req),
            .cand_valid         (fseq_cand_valid),
            .cand_prob_f32      (fseq_cand_prob_f32),
            .cand_token         (fseq_cand_token),
            .feed_busy          (fseq_feed_busy)
        );

        assign obs_cand_v_w    = fseq_cand_valid;
        assign obs_cand_prob_w = fseq_cand_prob_f32;
        assign obs_cand_tok_w  = fseq_cand_token[17:0];
        assign obs_tree_arm_w  = lvl_start_w;

        wire                                          ow_emit_valid;
        wire [tree_score_pkg::NODE_W-1:0]             ow_emit_idx;
        wire [tree_score_pkg::TOK_W-1:0]              ow_emit_token;
        wire signed [tree_score_pkg::NODE_W:0]        ow_emit_next_tok;
        wire signed [tree_score_pkg::NODE_W:0]        ow_emit_next_sib;
        wire [tree_score_pkg::POS_W-1:0]              ow_emit_pos;
        wire signed [tree_score_pkg::SCORE_W-1:0]     ow_emit_score;

        tree_out_walk u_tree_out_walk (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start        (out_walk_start_w),
            .n_nodes      (tl_n_nodes),
            .done         (lvl_out_walk_done_w),
            .tn_rd_idx    (tn_walk_idx),
            .tn_rd_token  (tn_walk_token),
            .tn_rd_parent (tn_walk_parent),
            .tn_rd_depth  (tn_walk_depth),
            .tn_rd_pos    (tn_walk_pos),
            .tn_rd_score  (tn_walk_score),
            .emit_valid   (ow_emit_valid),
            .emit_idx     (ow_emit_idx),
            .emit_token   (ow_emit_token),
            .emit_next_tok(ow_emit_next_tok),
            .emit_next_sib(ow_emit_next_sib),
            .emit_pos     (ow_emit_pos),
            .emit_score   (ow_emit_score)
        );

        tree_dma_writer #(
            .NODE_MAX(tree_score_pkg::N_MAX),
            .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(BOOT_AXI_DATA_W),
            .AXI_ID_W(BOOT_AXI_ID_W)
        ) u_tree_dma (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .emit_valid   (ow_emit_valid),
            .emit_idx     (ow_emit_idx),
            .emit_token   (ow_emit_token),
            .emit_next_tok(ow_emit_next_tok),
            .emit_next_sib(ow_emit_next_sib),
            .emit_score   (ow_emit_score),
            .walk_done    (lvl_out_walk_done_w),
            .n_nodes      (tl_n_nodes),
            .tree_base    (out0_base),
            .out_busy     (tr_busy),
            .out_done     (tr_done),
            .out_err      (tr_err),
            .m_axi_out_awid(tr_awid),        .m_axi_out_awaddr(tr_awaddr),
            .m_axi_out_awlen(tr_awlen),      .m_axi_out_awsize(tr_awsize),
            .m_axi_out_awburst(tr_awburst),  .m_axi_out_awvalid(tr_awvalid),
            .m_axi_out_awready(m_axi_out_awready),
            .m_axi_out_wdata(tr_wdata),      .m_axi_out_wstrb(tr_wstrb),
            .m_axi_out_wlast(tr_wlast),      .m_axi_out_wvalid(tr_wvalid),
            .m_axi_out_wready(m_axi_out_wready),
            .m_axi_out_bid(m_axi_out_bid),   .m_axi_out_bresp(m_axi_out_bresp),
            .m_axi_out_bvalid(m_axi_out_bvalid), .m_axi_out_bready(tr_bready)
        );

    end else begin : g_no_depth_n

        assign lvl_level_done_w    = 1'b0;
        assign lvl_level_last_w    = 1'b0;
        assign lvl_tree_done_w     = 1'b0;
        assign lvl_out_walk_done_w = 1'b0;

        assign bridge_cand_rd_active = 1'b0;
        assign bridge_cand_rdaddr    = '0;
        assign bridge_cand_rdaddr_nxt = '0;

        assign fr_par_node_w         = '0;
        assign fr_par_slot_w         = '0;

        assign obs_cand_v_w    = 1'b0;
        assign obs_cand_prob_w = '0;
        assign obs_cand_tok_w  = '0;
        assign obs_tree_arm_w  = 1'b0;
    end endgenerate

    generate if (OBS_LVL_P) begin : g_obs_lvl
        (* ram_style = "block" *)
        reg [31:0] obs_ring [0:1023];
        reg [9:0]  obs_wp;
        reg        obs_wrap;
        reg [31:0] obs_rdata_q;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                obs_wp <= '0; obs_wrap <= 1'b0;
            end else if (obs_tree_arm_w) begin
                obs_wp <= '0; obs_wrap <= 1'b0;
            end else if (obs_cand_v_w) begin
                obs_ring[{obs_wp[8:0], 1'b0}] <= obs_cand_prob_w;
                obs_ring[{obs_wp[8:0], 1'b1}] <= {level_i_w, branch_j_w, 7'b0,
                                                  obs_cand_tok_w};
                if (obs_wp[8:0] == 9'h1FF) obs_wrap <= 1'b1;
                obs_wp <= obs_wp + 1'b1;
            end
            obs_rdata_q <= obs_ring[obs_rd_idx_w[9:0]];
        end
        assign obs_rd_data_w     = obs_rdata_q;
        assign obs_status_word_w = {8'h0B, 13'b0, obs_wrap, obs_wp};
    end else begin : g_no_obs_lvl
        assign obs_rd_data_w     = 32'b0;
        assign obs_status_word_w = 32'b0;
    end endgenerate

    function automatic logic signed [15:0] sat_int32_to_int16;
        input signed [31:0] v;
        begin
            if (v > 32'sd32767)         sat_int32_to_int16 = 16'sd32767;
            else if (v < -32'sd32768)   sat_int32_to_int16 = 16'sh8000;
            else                        sat_int32_to_int16 = v[15:0];
        end
    endfunction

    reg signed [16*K_PAR-1:0] attn_out_chunk_latched;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            attn_out_chunk_latched <= '0;
        end else if (g5_attn_done) begin
            for (int lane = 0; lane < G5_D_HEAD && lane < K_PAR; lane = lane + 1) begin
                attn_out_chunk_latched[lane*16 +: 16] <=
                    sat_int32_to_int16(g5_attn_out_vec[lane*32 +: 32]);
            end
        end
    end

    wire signed [K_PAR*A_WIDTH-1:0] seam_r1_rd;
    wire signed [K_PAR*A_WIDTH-1:0] seam_r2_rd;
    wire signed [K_PAR*A_WIDTH-1:0] seam_r5_rd;
    wire signed [K_PAR*A_WIDTH-1:0] seam_r8_rd;
    wire signed [K_PAR*A_WIDTH-1:0] low_rank_rd;

    localparam int PREMLP_NCHUNKS = 3072 / K_PAR;
    localparam int PREMLP_CW      = (PREMLP_NCHUNKS > 1) ? $clog2(PREMLP_NCHUNKS) : 1;
    wire signed [K_PAR*A_WIDTH-1:0] seam_premlp_rd;
    wire                            seam_premlp_full;
    wire [PREMLP_CW-1:0]            premlp_rd_idx;

    wire                            seam_r2_full;
    wire                            seam_r5_full;

    wire [15:0]                     array_chunk_idx_b_out;
    wire signed [K_PAR*A_WIDTH-1:0] seam_r1_rd_b;
    wire signed [K_PAR*A_WIDTH-1:0] seam_g5g6_rd_b;
    wire signed [K_PAR*A_WIDTH-1:0] seam_premlp_rd_b;
    wire signed [K_PAR*A_WIDTH-1:0] seam_r5_rd_b;
    wire signed [K_PAR*A_WIDTH-1:0] low_rank_rd_b;

    wire signed [K_PAR*A_WIDTH-1:0] g6_act_in =
        S7A_FRONT_SEQ ? seam_g5g6_rd : attn_out_chunk_latched;

    wire signed [K_PAR*A_WIDTH-1:0] g6_act_in_b =
        S7A_FRONT_SEQ ? seam_g5g6_rd_b : attn_out_chunk_latched;

    // ------------------------------------------------------------------
    // Activation-source routing. act_addr = latched activation_base + the
    // array's exposed chunk lookahead. Adding the base to the ADDRESS (not
    // just muxing source data) keeps the SLR lookahead/alignment contract
    // and lets one operation start at any supported chunk offset.
    // Sources: 0 external stream, 1 h_fc, 2 input_emb, 3 R1, 4 pre-MLP/R2,
    //          5 R5, 6 low-rank, 7 attention (G5 -> G6).
    // External has no addressable base (the controller requires base 0).
    // ------------------------------------------------------------------
    wire [16:0] act_addr_w   = {1'b0, gemv_act_base_w} + {1'b0, array_chunk_idx_out};
    wire [16:0] act_addr_b_w = {1'b0, gemv_act_base_w} + {1'b0, array_chunk_idx_b_out};

    assign premlp_rd_idx = act_addr_w[PREMLP_CW-1:0];

    wire [PREMLP_CW-1:0] premlp_rd_idx_b = act_addr_b_w[PREMLP_CW-1:0];

    // HFC / input-embedding banks are combinational array reads; register
    // them once to match the RD_LAT=1 seam buffers on the other sources.
    reg signed [K_PAR*A_WIDTH-1:0] hfc_act_rd_w,  hfc_act_rd_b_w;
    reg signed [K_PAR*A_WIDTH-1:0] emb_act_rd_w,  emb_act_rd_b_w;
    always_ff @(posedge ap_clk) begin
        hfc_act_rd_w   <= h_fc_buf_rd(act_addr_w[EMB_CW-1:0]);
        hfc_act_rd_b_w <= h_fc_buf_rd(act_addr_b_w[EMB_CW-1:0]);
        emb_act_rd_w   <= input_emb_buf_rd(act_addr_w[EMB_CW-1:0]);
        emb_act_rd_b_w <= input_emb_buf_rd(act_addr_b_w[EMB_CW-1:0]);
    end

    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_src =
          (gemv_act_src_w == 3'd0) ? a_chunk
        : (gemv_act_src_w == 3'd1) ? hfc_act_rd_w
        : (gemv_act_src_w == 3'd2) ? emb_act_rd_w
        : (gemv_act_src_w == 3'd3) ? seam_r1_rd
        : (gemv_act_src_w == 3'd4) ? seam_premlp_rd
        : (gemv_act_src_w == 3'd5) ? seam_r5_rd
        : (gemv_act_src_w == 3'd6) ? low_rank_rd
        : (gemv_act_src_w == 3'd7) ? g6_act_in
        :                            '0;

    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_b_src =
          (gemv_act_src_w == 3'd0) ? '0
        : (gemv_act_src_w == 3'd1) ? hfc_act_rd_b_w
        : (gemv_act_src_w == 3'd2) ? emb_act_rd_b_w
        : (gemv_act_src_w == 3'd3) ? seam_r1_rd_b
        : (gemv_act_src_w == 3'd4) ? seam_premlp_rd_b
        : (gemv_act_src_w == 3'd5) ? seam_r5_rd_b
        : (gemv_act_src_w == 3'd6) ? low_rank_rd_b
        : (gemv_act_src_w == 3'd7) ? g6_act_in_b
        :                            '0;

`ifndef SYNTHESIS
    initial if (DUAL_PORT && !(GEMV_SCALED && !GEMV_KPAR64 && CONNECT_SEAMS
                               && S7A_FRONT_SEQ && BACK_SEQ && SEAM_R1_FULLW
                               && (BODY_BATCH_B == 1)))
        $error("eagle4_drafter: DUAL_PORT=1 requires the production seam config (GEMV_SCALED=1, GEMV_KPAR64=0, CONNECT_SEAMS=1, S7A_FRONT_SEQ=1, BACK_SEQ=1, SEAM_R1_FULLW=1, BODY_BATCH_B=1) so every a_chunk_b arm is a real port-B seam read.");

    initial if (!(LANES == 128 || LANES == 256))
        $error("eagle4_drafter: LANES=%0d unsupported — only 128 (default) and 256 (quad lever) divide every stage N and pack D_HEAD=128 heads (192 silently drops G7 columns).", LANES);
    initial if (LANES != 128 && !(GEMV_SCALED && !GEMV_KPAR64))
        $error("eagle4_drafter: LANES=%0d requires the scaled SLR array arm (GEMV_SCALED=1, GEMV_KPAR64=0) — the pilot array arms carry fixed 128-lane bank geometry.", LANES);
`endif

    wire signed [K_PAR*A_WIDTH-1:0] inj_act_chunk;
    genvar gia;
    generate
        for (gia = 0; gia < K_PAR; gia = gia + 1) begin : g_inj_act
            assign inj_act_chunk[gia*A_WIDTH +: A_WIDTH] = inj_bus[gia*32 +: A_WIDTH];
        end
    endgenerate
    // Debug injection stays a source-side override; the enable bit follows
    // the selected activation source's historical stage slot.
    wire inj_gemv_active =
          (gemv_act_src_w == 3'd3) ? inject_en[1]
        : (gemv_act_src_w == 3'd7) ? inject_en[5]
        : (gemv_act_src_w == 3'd4) ? inject_en[6]
        : (gemv_act_src_w == 3'd5) ? inject_en[9]
        : (gemv_act_src_w == 3'd6) ? inject_en[12]
        :                            1'b0;
    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_mux =
        inj_gemv_active ? inj_act_chunk : a_chunk_src;
    wire signed [K_PAR*A_WIDTH-1:0] a_chunk_b_mux =
        inj_gemv_active ? '0 : a_chunk_b_src;

    // First-tile source readiness: the controller must not launch before the
    // producing stage's seam buffer is full. This replaces the old bkseq
    // SETTLE input-full gate without reintroducing a second tile owner.
    assign gemv_act_ready_w =
          (gemv_act_src_w == 3'd3) ? seam_r1_full :
          (gemv_act_src_w == 3'd4) ? seam_premlp_full :
          (gemv_act_src_w == 3'd5) ? seam_r5_full :
          (gemv_act_src_w == 3'd7) ? (S7A_FRONT_SEQ ? seam_g5g6_full : 1'b1) :
                                     1'b1;

    reg [G7_CW:0]  silu_en_cnt;
    reg            silu_en_run;
    reg            silu_en_r;
    reg [G7_CW-1:0] silu_rd_idx_r;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            silu_en_cnt <= '0; silu_en_run <= 1'b0; silu_en_r <= 1'b0; silu_rd_idx_r <= '0;
        end else begin
            silu_en_r <= 1'b0;
            if (BACK_SEQ && g7p_start) begin

                silu_en_cnt <= '0;
                silu_en_run <= 1'b1;
            end else if (silu_en_run) begin
                if (silu_en_cnt == (G7_CW+1)'(G7_NCHUNKS)) begin
                    silu_en_run <= 1'b0;
                end else begin
                    silu_en_r     <= 1'b1;

                    silu_rd_idx_r <= silu_en_cnt[G7_CW-1:0];
                    silu_en_cnt   <= silu_en_cnt + 1'b1;
                end
            end
        end
    end

    assign r4_silu_rd_idx = D1_RDLAT ? silu_en_cnt[G7_CW-1:0] : silu_rd_idx_r;

    assign silu_en_eff = BACK_SEQ ? silu_en_r : g7p_start;

    reg [G7_CW:0] silu_vld_cnt;

    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            silu_vld_cnt <= '0; silu_sweep_done_r <= 1'b0;
        end else begin
            silu_sweep_done_r <= 1'b0;
            if (g7p_start) silu_vld_cnt <= '0;
            else if (silu_valid) begin
                silu_vld_cnt <= silu_vld_cnt + 1'b1;
                if (silu_vld_cnt == (G7_CW+1)'(G7_NCHUNKS-1))
                    silu_sweep_done_r <= 1'b1;
            end
        end
    end

    wire seam_r1_full;

    // The shared array start, per-tile bases and bounds come from the sole
    // operation controller. No stage-derived launch or tile counter remains.

    localparam int ARR_KPAR = GEMV_KPAR64 ? 64 : K_PAR;
    localparam int ARR_NSG  = ARR_KPAR / 16;
    localparam int ARR_ROWW = ARR_NSG * 32;

    wire signed [ARR_KPAR*A_WIDTH-1:0] a_chunk_wide;
    generate
        if (ARR_KPAR == K_PAR) begin : g_awide_eq
            assign a_chunk_wide = a_chunk_mux;
        end else begin : g_awide_pad

            assign a_chunk_wide = { {((ARR_KPAR-K_PAR)*A_WIDTH){1'b0}}, a_chunk_mux };
        end
    endgenerate

    wire signed [32*LANES-1:0] array_acc_int;
    assign array_acc_out = array_acc_int;

    generate
    if (GEMV_SCALED && !GEMV_KPAR64) begin : g_array_scaled

        localparam int RES_W = 48;
        wire signed [RES_W*LANES-1:0] result_q_per_lane;
        wire signed [32*LANES-1:0]    acc_dbg_per_lane;

        localparam int BT_BB1 = (BODY_BATCH_B > 1) ? (BODY_BATCH_B - 1) : 1;
        wire signed [BT_BB1*RES_W*LANES-1:0] result_q_batch_w;
        wire        batch_bt_en  = batch_test_ctrl[0];
        wire        batch_bt_neg = batch_test_ctrl[1];
        wire [6:0]  batch_bt_lane = batch_test_ctrl[14:8];

        wire signed [K_PAR*A_WIDTH-1:0] bt_pos = a_chunk_wide[K_PAR*A_WIDTH-1:0];
        wire signed [K_PAR*A_WIDTH-1:0] bt_neg;
        genvar btw;
        for (btw = 0; btw < K_PAR; btw = btw + 1) begin : g_bt_neg
            assign bt_neg[btw*A_WIDTH +: A_WIDTH] =
                   -$signed(a_chunk_wide[btw*A_WIDTH +: A_WIDTH]);
        end

        wire signed [BT_BB1*K_PAR*A_WIDTH-1:0] a_chunk_batch_drv;
        genvar bts;
        for (bts = 0; bts < BT_BB1; bts = bts + 1) begin : g_bt_drv
            assign a_chunk_batch_drv[bts*K_PAR*A_WIDTH +: K_PAR*A_WIDTH] =
                   !batch_bt_en ? '0 : (batch_bt_neg ? bt_neg : bt_pos);
        end

        wire        eff_sc_we       = dma_sc_we | tb_sc_we;
        wire [15:0] eff_sc_lane_sel = dma_sc_we ? dma_sc_lane_sel : tb_sc_lane_sel;
        wire [16:0] eff_sc_wr_addr  = dma_sc_we ? dma_sc_wr_addr  : tb_sc_wr_addr;
        wire [15:0] eff_sc_wr_data  = dma_sc_we ? dma_sc_wr_data  : tb_sc_wr_data;

        (* keep_hierarchy = "yes" *)
        gemv_stage_array_scaled_slr #(
            .K(K_MAX), .K_PAR(K_PAR), .LANES(LANES), .A_WIDTH(A_WIDTH),
            .SCALE_WIDTH(16), .SCALE_FRAC(GEMV_SCALE_FRAC),
            .RES_WIDTH(RES_W),
            .SLR_SAFE(SLR_SAFE), .N_GROUPS(ARR_GROUPS), .PIPE_STAGES(ARR_PIPE_STAGES),

            .FF_ACT_CACHE(FF_ACT_CACHE),

            .WLAT(D1_WLAT),

            .BATCH_B(BODY_BATCH_B),

            .DUAL_PORT(DUAL_PORT)
        ) u_array (
            .clk(ap_clk), .rst_n(ap_rst_n), .start(gemv_array_start_w),
            .a_chunk(a_chunk_wide), .qz_raw(gemv_qzero_w),
            .a_chunk_b(a_chunk_b_mux),
            .cfg_weight_base(gemv_arr_weight_base_w),
            .cfg_scale_base(gemv_arr_scale_base_w),
            .cfg_chunks_per_tile(gemv_arr_chunks_per_tile_w),
            .cfg_chunks_per_scale_group(gemv_arr_chunks_per_scale_group_w),
            .lane_we_sel(eff_boot_lane_sel), .we(eff_boot_we),
            .wr_addr(eff_boot_wr_addr), .wr_data(eff_boot_wr_data),
            .sc_lane_sel(eff_sc_lane_sel), .sc_we(eff_sc_we),
            .sc_wr_addr(eff_sc_wr_addr), .sc_wr_data(eff_sc_wr_data),
            .chunk_idx_out(array_chunk_idx_out),
            .chunk_idx_b_out(array_chunk_idx_b_out),
            .chunk_idx_valid(array_chunk_idx_valid),
            .acc_out_per_lane(acc_dbg_per_lane),
            .result_q_per_lane(result_q_per_lane),

            .a_chunk_batch(a_chunk_batch_drv),
            .result_q_batch(result_q_batch_w),
            .done(array_done)
        );

        reg signed [RES_W-1:0] bt_b0_cap, bt_sib_cap;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                bt_b0_cap  <= '0;
                bt_sib_cap <= '0;
            end else if (array_done && batch_bt_en) begin
                bt_b0_cap  <= result_q_per_lane[batch_bt_lane*RES_W +: RES_W];
                bt_sib_cap <= result_q_batch_w[batch_bt_lane*RES_W +: RES_W];
            end
        end
        assign batch_b0_lo_w  = bt_b0_cap[31:0];
        assign batch_b0_hi_w  = {16'hB000, bt_b0_cap[47:32]};
        assign batch_sib_lo_w = bt_sib_cap[31:0];
        assign batch_sib_hi_w = {16'hB500, bt_sib_cap[47:32]};

        // Remove the scale's fractional bits from the 48-bit group-summed
        // result: add half, arithmetic-shift, then saturate to signed INT32.
        // The shift comes from the ACTUAL latched scale_frac field (validated
        // to 13 at START), not a compile-time constant pretending to be
        // programmable.
        wire signed [RES_W-1:0] rq_half =
            48'sd1 <<< (gemv_scale_frac_w - 5'd1);
        genvar sl;
        for (sl = 0; sl < LANES; sl = sl + 1) begin : g_scaled_requant
            wire signed [RES_W-1:0] rq = result_q_per_lane[sl*RES_W +: RES_W];
            wire signed [RES_W-1:0] rq_round = rq + rq_half;
            wire signed [RES_W-1:0] rq_shr = rq_round >>> gemv_scale_frac_w;
            assign array_acc_int[sl*32 +: 32] =
                  (rq_shr >  $signed({{(RES_W-31){1'b0}}, 31'h7FFFFFFF})) ? 32'sh7FFFFFFF
                : (rq_shr < -$signed({{(RES_W-32){1'b0}}, 32'h80000000})) ? 32'sh80000000
                :                                                          rq_shr[31:0];
        end
    end else if (!GEMV_KPAR64) begin : g_array_k16

        assign array_chunk_idx_b_out = array_chunk_idx_out;
        // Unscaled K_PAR-wide GEMV array.
        gemv_stage_array_uram #(
            .K(K_MAX), .K_PAR(K_PAR), .LANES(LANES), .A_WIDTH(A_WIDTH)
        ) u_array (
            .clk(ap_clk), .rst_n(ap_rst_n), .start(gemv_array_start_w),
            .a_chunk(a_chunk_wide), .qz_raw(gemv_qzero_w),
            .cfg_weight_base(gemv_arr_weight_base_w),
            .cfg_chunks_per_tile(gemv_arr_chunks_per_tile_w),
            .lane_we_sel(eff_boot_lane_sel), .we(eff_boot_we),
            .wr_addr(eff_boot_wr_addr), .wr_data(eff_boot_wr_data),
            .chunk_idx_out(array_chunk_idx_out),
            .chunk_idx_valid(array_chunk_idx_valid),
            .acc_out_per_lane(array_acc_int),
            .done(array_done)
        );
    end else begin : g_array_k64

        assign array_chunk_idx_b_out = array_chunk_idx_out;

        reg  [ARR_ROWW-1:0] boot_row_q;
        reg  [16:0]         boot_row_addr_q;
        reg                 boot_row_we_q;
        reg  [1:0]          boot_word_cnt;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                boot_row_q <= '0; boot_row_addr_q <= '0;
                boot_row_we_q <= 1'b0; boot_word_cnt <= 2'd0;
            end else begin
                boot_row_we_q <= 1'b0;
                if (eff_boot_we) begin
                    boot_row_q[boot_word_cnt*32 +: 32] <= eff_boot_wr_data;
                    if (boot_word_cnt == 2'd0) boot_row_addr_q <= eff_boot_wr_addr >> 2;
                    if (boot_word_cnt == ARR_NSG-1) begin
                        boot_word_cnt <= 2'd0;
                        boot_row_we_q <= 1'b1;
                    end else begin
                        boot_word_cnt <= boot_word_cnt + 2'd1;
                    end
                end
            end
        end
        // Wide-bank K_PAR=64 GEMV array.
        gemv_stage_array_uram_kpar64 #(
            .K(K_MAX), .K_PAR(ARR_KPAR), .LANES(LANES), .A_WIDTH(A_WIDTH)
        ) u_array (
            .clk(ap_clk), .rst_n(ap_rst_n), .start(gemv_array_start_w),
            .a_chunk(a_chunk_wide), .qz_raw(gemv_qzero_w),
            .cfg_weight_base(gemv_arr_weight_base_w),
            .cfg_chunks_per_tile(gemv_arr_chunks_per_tile_w),
            .lane_we_sel(eff_boot_lane_sel), .we(boot_row_we_q),
            .wr_row(boot_row_addr_q), .wr_row_data(boot_row_q),
            .chunk_idx_out(array_chunk_idx_out),
            .chunk_idx_valid(array_chunk_idx_valid),
            .acc_out_per_lane(array_acc_int),
            .done(array_done)
        );
    end
    endgenerate

    // Per-tile destination events used by the destination adapter and by the
    // downstream start/done side effects. tile_valid is registered one cycle
    // after array_done with the array result bus still stable.
    wire gemv_tile_qkv  = gemv_tile_valid_w && ((gemv_tile_dest_w == 4'd9) ||
                          (gemv_tile_dest_w == 4'd0) || (gemv_tile_dest_w == 4'd1) ||
                          (gemv_tile_dest_w == 4'd2));
    wire gemv_tile_r2   = gemv_tile_valid_w && (gemv_tile_dest_w == 4'd3);
    wire gemv_tile_r4g  = gemv_tile_valid_w && (gemv_tile_dest_w == 4'd4);
    wire gemv_tile_r4u  = gemv_tile_valid_w && (gemv_tile_dest_w == 4'd5);
    wire gemv_tile_r6   = gemv_tile_valid_w && (gemv_tile_dest_w == 4'd6);
    wire gemv_tile_tree = gemv_tile_valid_w && (gemv_tile_dest_w == 4'd7);

    // Final-operation pulses used by residual/RMSNorm sequencing and debug.
    wire gemv_done_qkv = gemv_op_done_w && (gemv_op_done_dest_w == 4'd9);
    wire gemv_done_r2  = gemv_op_done_w && (gemv_op_done_dest_w == 4'd3);
    wire gemv_done_r4g = gemv_op_done_w && (gemv_op_done_dest_w == 4'd4);
    wire gemv_done_r4u = gemv_op_done_w && (gemv_op_done_dest_w == 4'd5);
    wire gemv_done_r6  = gemv_op_done_w && (gemv_op_done_dest_w == 4'd6);

    assign g4_done  = gemv_done_qkv;
    assign g6_done  = gemv_tile_r2;           // per-tile capture trigger
    assign g7g_done = gemv_tile_r4g;
    assign g7u_done = gemv_tile_r4u;
    assign g9_done  = gemv_tile_r6;
    assign g12_done = gemv_tile_tree;
    wire g6_done_stage  = gemv_done_r2;
    wire g7g_done_stage = gemv_done_r4g;
    wire g7u_done_stage = gemv_done_r4u;
    wire g9_done_stage  = gemv_done_r6;

    // Destination store handshake to the controller. Q/K/V (single or
    // composite) have a one-cycle buffer write; R2/R4/R6 report the last of
    // their LANE_SUBW serialized subchunk writes. The controller stays in
    // STORE until this ack, so operation DONE cannot precede the last store.
    reg        gemv_tile_valid_q;
    reg [3:0]  gemv_tile_dest_q;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            gemv_tile_valid_q <= 1'b0;
            gemv_tile_dest_q  <= 4'd0;
        end else begin
            gemv_tile_valid_q <= gemv_tile_valid_w;
            gemv_tile_dest_q  <= gemv_tile_dest_w;
        end
    end
    wire gemv_store_ack_imm = gemv_tile_valid_q &&
        ((gemv_tile_dest_q == 4'd0) || (gemv_tile_dest_q == 4'd1) ||
         (gemv_tile_dest_q == 4'd2) || (gemv_tile_dest_q == 4'd9));
    assign gemv_store_ack_w = gemv_store_ack_imm | r2_store_ack |
                              g7g_store_ack | g7u_store_ack | r6_store_ack;

    localparam int SEAM_H = K_PAR;

    function automatic logic signed [K_PAR*A_WIDTH-1:0] sat_acc_head0_chunk;
        input signed [32*LANES-1:0] acc;
        logic signed [K_PAR*A_WIDTH-1:0] r;
        begin
            r = '0;
            for (int l = 0; l < K_PAR; l = l + 1)
                r[l*A_WIDTH +: A_WIDTH] = sat_int32_to_int16(acc[l*32 +: 32]);
            sat_acc_head0_chunk = r;
        end
    endfunction

    function automatic logic signed [K_PAR*A_WIDTH-1:0] r1_lshift1_sat;
        input signed [K_PAR*A_WIDTH-1:0] ch;
        integer l;
        begin

            for (l = 0; l < K_PAR; l = l + 1) begin
                r1_lshift1_sat[l*A_WIDTH +: A_WIDTH] = $signed(ch[l*A_WIDTH +: A_WIDTH]);
            end
        end
    endfunction

    function automatic logic signed [K_PAR*A_WIDTH-1:0] r1_h_align;
        input signed [K_PAR*A_WIDTH-1:0] ch;
        logic signed [A_WIDTH:0] v;
        integer l;
        begin
            for (l = 0; l < K_PAR; l = l + 1) begin
                v = $signed(ch[l*A_WIDTH +: A_WIDTH]);
                r1_h_align[l*A_WIDTH +: A_WIDTH] =
                    PRECISION_SAFE ? (($signed(v) + 17'sd1) >>> 1)
                                   : v[A_WIDTH-1:0];
            end
        end
    endfunction

    // R1 stores both complete G1 outputs. The array reads embedding chunks
    // first, then hidden chunks. Delay each apply enable by two cycles to
    // align writes with data; raw RMSNorm out_valid is one cycle later.
    generate
    if (SEAM_R1_FULLW) begin : g_seam_r1_fullw

        localparam int R1F_AP_LAT = 2;
        localparam int R1F_HALF_CW = (3072/K_PAR > 1) ? $clog2(3072/K_PAR) : 1;
        localparam int R1_FULL_H   = 2*3072;
        localparam int R1F_FULL_CW = (R1_FULL_H/K_PAR > 1)
                                     ? $clog2(R1_FULL_H/K_PAR) : 1;

        reg [R1F_AP_LAT-1:0] r1f_h_apreq_sr;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)        r1f_h_apreq_sr <= '0;
            else if (g1_start)    r1f_h_apreq_sr <= '0;
            else                  r1f_h_apreq_sr <= {r1f_h_apreq_sr[R1F_AP_LAT-2:0], g1_apply_en};
        end
        wire r1f_h_wr = r1f_h_apreq_sr[R1F_AP_LAT-1];

        wire signed [K_PAR*A_WIDTH-1:0] r1f_h_wr_chunk = r1_h_align(rms_g1_out_chunk_w);

        reg [R1F_AP_LAT-1:0] r1f_e_apreq_sr;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)        r1f_e_apreq_sr <= '0;
            else if (g1_start)    r1f_e_apreq_sr <= '0;
            else                  r1f_e_apreq_sr <= {r1f_e_apreq_sr[R1F_AP_LAT-2:0], g1e_apply_en_eff};
        end
        wire r1f_e_wr = r1f_e_apreq_sr[R1F_AP_LAT-1];

        wire signed [K_PAR*A_WIDTH-1:0] r1f_e_wr_chunk = rms_g1e_out_chunk_eff;

        wire [R1F_FULL_CW-1:0] r1f_rd_idx = act_addr_w[R1F_FULL_CW-1:0];
        wire                   r1f_rd_low = (r1f_rd_idx < R1F_FULL_CW'(3072/K_PAR));

        wire [R1F_FULL_CW-1:0] r1f_rd_idx_hi = r1f_rd_idx - R1F_FULL_CW'(3072/K_PAR);
        wire [R1F_HALF_CW-1:0] r1f_rd_half_idx =
            r1f_rd_low ? r1f_rd_idx[R1F_HALF_CW-1:0]
                       : r1f_rd_idx_hi[R1F_HALF_CW-1:0];

        reg r1f_rd_low_d1;
        reg r1f_rd_oor_d1;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                r1f_rd_low_d1 <= 1'b1; r1f_rd_oor_d1 <= 1'b0;
            end else begin
                r1f_rd_low_d1 <= r1f_rd_low;
                r1f_rd_oor_d1 <= (r1f_rd_idx >= R1F_FULL_CW'(R1_FULL_H/K_PAR));
            end
        end

        wire [R1F_FULL_CW-1:0] r1f_rd_idx_b = act_addr_b_w[R1F_FULL_CW-1:0];
        wire [R1F_FULL_CW-1:0] r1f_rd_idx_b_hi = r1f_rd_idx_b - R1F_FULL_CW'(3072/K_PAR);
        wire [R1F_HALF_CW-1:0] r1f_rd_half_idx_b =
            r1f_rd_low ? r1f_rd_idx_b[R1F_HALF_CW-1:0]
                       : r1f_rd_idx_b_hi[R1F_HALF_CW-1:0];

        wire signed [K_PAR*A_WIDTH-1:0] r1_e_rd;
        wire signed [K_PAR*A_WIDTH-1:0] r1_e_rd_b;
        wire seam_r1_e_full;
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(1), .RD_LAT(D1_RDLAT)) u_seam_r1_e (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(r1f_e_wr), .wr_chunk_idx('0), .wr_chunk(r1f_e_wr_chunk),
            .wr_clear(g1_start), .wr_count(), .full(seam_r1_e_full),

            .rd_valid(r1f_rd_low), .rd_chunk_idx(r1f_rd_half_idx), .rd_chunk(r1_e_rd),
            .rd_chunk_idx_b(r1f_rd_half_idx_b), .rd_chunk_b(r1_e_rd_b)
        );

        wire signed [K_PAR*A_WIDTH-1:0] r1_h_rd;
        wire signed [K_PAR*A_WIDTH-1:0] r1_h_rd_b;
        wire seam_r1_h_full;
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(1), .RD_LAT(D1_RDLAT)) u_seam_r1_h (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(r1f_h_wr), .wr_chunk_idx('0), .wr_chunk(r1f_h_wr_chunk),
            .wr_clear(g1_start), .wr_count(), .full(seam_r1_h_full),
            .rd_valid(!r1f_rd_low && (r1f_rd_idx < R1F_FULL_CW'(R1_FULL_H/K_PAR))),
            .rd_chunk_idx(r1f_rd_half_idx), .rd_chunk(r1_h_rd),
            .rd_chunk_idx_b(r1f_rd_half_idx_b), .rd_chunk_b(r1_h_rd_b)
        );

        assign seam_r1_rd = r1f_rd_oor_d1 ? '0
                          : r1f_rd_low_d1 ? r1_e_rd : r1_h_rd;

        assign seam_r1_rd_b = r1f_rd_oor_d1 ? '0
                            : r1f_rd_low_d1 ? r1_e_rd_b : r1_h_rd_b;

        assign seam_r1_full = seam_r1_e_full & seam_r1_h_full;
    end else begin : g_seam_r1_pilot

        reg  r1_armed;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)                              r1_armed <= 1'b1;
            else if (g1_start)                          r1_armed <= 1'b1;
            else if (rms_g1_out_valid && r1_armed)      r1_armed <= 1'b0;
        end
        wire r1_wr = rms_g1_out_valid && r1_armed;
        wire signed [K_PAR*A_WIDTH-1:0] r1p_wr_chunk = r1_h_align(rms_g1_out_chunk_d1);
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(SEAM_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(1)) u_seam_r1 (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(r1_wr), .wr_chunk_idx('0), .wr_chunk(r1p_wr_chunk),
            .wr_clear(g1_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx('0), .rd_chunk(seam_r1_rd)
        );
        assign seam_r1_full = 1'b1;

        assign seam_r1_rd_b = seam_r1_rd;
    end
    endgenerate

    localparam int G6_OUT_N   = 3072;
    localparam int G6_NTILES  = (G6_OUT_N + LANES - 1) / LANES;
    localparam int R2_NCHUNKS = G6_OUT_N / K_PAR;
    localparam int R2_CW      = $clog2(R2_NCHUNKS);

    localparam int LANE_SUBW    = LANES / K_PAR;
    localparam int LANE_SUBW_CW = $clog2(LANE_SUBW);

    reg signed [32*LANES-1:0]  r2_acc_lat;
    reg [15:0]                 r2_tile_lat;
    reg [LANE_SUBW_CW:0]       r2_we_cnt;
    reg                        r2_we_active;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            r2_acc_lat <= '0; r2_tile_lat <= '0; r2_we_cnt <= '0; r2_we_active <= 1'b0;
        end else if (gemv_tile_r2) begin
            r2_acc_lat   <= array_acc_out;
            r2_tile_lat  <= gemv_store_idx_w;
            r2_we_cnt    <= '0;
            r2_we_active <= 1'b1;
        end else if (r2_we_active) begin
            if (r2_we_cnt == (LANE_SUBW-1)) r2_we_active <= 1'b0;
            r2_we_cnt <= r2_we_cnt + 1'b1;
        end
    end
    wire                            r2_seam_we    = r2_we_active;
    // Destination-store completion: the last of LANE_SUBW subchunk writes.
    wire                            r2_store_ack =
        r2_we_active && (r2_we_cnt == (LANE_SUBW-1));
    wire [R2_CW-1:0]                r2_seam_widx  = R2_CW'(int'(r2_tile_lat)*LANE_SUBW + int'(r2_we_cnt));

    wire signed [K_PAR*A_WIDTH-1:0] r2_seam_wchunk =
        sat_acc_subchunk_g6_r3(r2_acc_lat, int'(r2_we_cnt) * K_PAR);

    reg signed [K_PAR*A_WIDTH-1:0] g6out_buf [0:R2_NCHUNKS-1];
    always_ff @(posedge ap_clk) begin
        if (r2_seam_we) g6out_buf[r2_seam_widx] <= r2_seam_wchunk;
    end

    wire [R2_CW-1:0] r2_rd_idx = act_addr_w[R2_CW-1:0];
    // Inter-stage activation seam storage.
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(G6_OUT_N), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(0)) u_seam_r2 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(r2_seam_we), .wr_chunk_idx(r2_seam_widx), .wr_chunk(r2_seam_wchunk),
        .wr_clear(g6_start), .wr_count(), .full(seam_r2_full),
        .rd_valid(1'b1), .rd_chunk_idx(r2_rd_idx), .rd_chunk(seam_r2_rd)
    );

    function automatic logic signed [K_PAR*A_WIDTH-1:0] sat_acc_subchunk;
        input signed [32*LANES-1:0] acc;
        input int unsigned          lane_base;
        logic signed [K_PAR*A_WIDTH-1:0] r;
        begin
            r = '0;
            for (int l = 0; l < K_PAR; l = l + 1)
                r[l*A_WIDTH +: A_WIDTH] = sat_int32_to_int16(acc[(lane_base+l)*32 +: 32]);
            sat_acc_subchunk = r;
        end
    endfunction

    localparam int R4_SAT_AW = 18;
    function automatic logic signed [K_PAR*R4_SAT_AW-1:0] sat_acc_subchunk_q612;
        input signed [32*LANES-1:0] acc;
        input int unsigned          lane_base;
        logic signed [K_PAR*R4_SAT_AW-1:0] r;
        logic signed [31:0]                v;
        begin
            r = '0;
            for (int l = 0; l < K_PAR; l = l + 1) begin
                v = $signed(acc[(lane_base+l)*32 +: 32]);
                if      (v >  32'sd131071) r[l*R4_SAT_AW +: R4_SAT_AW] =  18'sd131071;
                else if (v < -32'sd131072) r[l*R4_SAT_AW +: R4_SAT_AW] = -18'sd131072;
                else                       r[l*R4_SAT_AW +: R4_SAT_AW] =  v[17:0];
            end
            sat_acc_subchunk_q612 = r;
        end
    endfunction

    localparam int G6_TO_R3_SH = 10 - R3_FRAC;
    function automatic logic signed [K_PAR*A_WIDTH-1:0] sat_acc_subchunk_g6_r3;
        input signed [32*LANES-1:0] acc;
        input int unsigned          lane_base;
        logic signed [K_PAR*A_WIDTH-1:0] r;
        logic signed [31:0]              v_rnd;
        begin
            r = '0;
            for (int l = 0; l < K_PAR; l = l + 1) begin

                v_rnd = $signed(acc[(lane_base+l)*32 +: 32])
                      + (32'sd1 <<< (G6_TO_R3_SH-1));
                r[l*A_WIDTH +: A_WIDTH] =
                    sat_int32_to_int16(v_rnd >>> G6_TO_R3_SH);
            end
            sat_acc_subchunk_g6_r3 = r;
        end
    endfunction

    reg signed [32*LANES-1:0]  g7g_acc_lat;
    reg [15:0]                 g7g_tile_lat;
    reg [LANE_SUBW_CW:0]       g7g_we_cnt;
    reg                        g7g_we_active;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            g7g_acc_lat <= '0; g7g_tile_lat <= '0; g7g_we_cnt <= '0; g7g_we_active <= 1'b0;
        end else if (g7g_done) begin

            g7g_acc_lat   <= array_acc_out;
            g7g_tile_lat  <= gemv_store_idx_w;
            g7g_we_cnt    <= '0;
            g7g_we_active <= 1'b1;
        end else if (g7g_we_active) begin
            if (g7g_we_cnt == (LANE_SUBW-1)) g7g_we_active <= 1'b0;
            g7g_we_cnt <= g7g_we_cnt + 1'b1;
        end
    end
    wire                        g7g_seam_we   = g7g_we_active;
    wire                        g7g_store_ack =
        g7g_we_active && (g7g_we_cnt == (LANE_SUBW-1));

    wire [G7_CW-1:0]            g7g_seam_widx = G7_CW'(int'(g7g_tile_lat)*LANE_SUBW + int'(g7g_we_cnt));

    wire signed [K_PAR*R4_AW-1:0] g7g_seam_wchunk = sat_acc_subchunk_q612(g7g_acc_lat, int'(g7g_we_cnt) * K_PAR);

    reg signed [32*LANES-1:0]  g7u_acc_lat;
    reg [15:0]                 g7u_tile_lat;
    reg [LANE_SUBW_CW:0]       g7u_we_cnt;
    reg                        g7u_we_active;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            g7u_acc_lat <= '0; g7u_tile_lat <= '0; g7u_we_cnt <= '0; g7u_we_active <= 1'b0;
        end else if (g7u_done) begin
            g7u_acc_lat   <= array_acc_out;
            g7u_tile_lat  <= gemv_store_idx_w;
            g7u_we_cnt    <= '0;
            g7u_we_active <= 1'b1;
        end else if (g7u_we_active) begin
            if (g7u_we_cnt == (LANE_SUBW-1)) g7u_we_active <= 1'b0;
            g7u_we_cnt <= g7u_we_cnt + 1'b1;
        end
    end
    wire                        g7u_seam_we   = g7u_we_active;
    wire                        g7u_store_ack =
        g7u_we_active && (g7u_we_cnt == (LANE_SUBW-1));
    wire [G7_CW-1:0]            g7u_seam_widx = G7_CW'(int'(g7u_tile_lat)*LANE_SUBW + int'(g7u_we_cnt));

    wire signed [K_PAR*R4_AW-1:0] g7u_seam_wchunk = sat_acc_subchunk_q612(g7u_acc_lat, int'(g7u_we_cnt) * K_PAR);

    generate
    if (R4_BANKSPLIT) begin : g_r4_banked
        stage_io_buffer_banked #(.H(G7_H_INTER), .K_PAR(K_PAR), .A_WIDTH(R4_AW), .LANES_PER_BANK(2), .RD_LAT(D1_RDLAT)) u_seam_r4g (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(g7g_seam_we), .wr_chunk_idx(g7g_seam_widx), .wr_chunk(g7g_seam_wchunk),
            .wr_clear(g7g_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx(r4_silu_rd_idx), .rd_chunk(seam_r4g_rd_c)
        );
        stage_io_buffer_banked #(.H(G7_H_INTER), .K_PAR(K_PAR), .A_WIDTH(R4_AW), .LANES_PER_BANK(2), .RD_LAT(D1_RDLAT)) u_seam_r4u (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(g7u_seam_we), .wr_chunk_idx(g7u_seam_widx), .wr_chunk(g7u_seam_wchunk),
            .wr_clear(g7u_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx(r4_silu_rd_idx), .rd_chunk(seam_r4u_rd_c)
        );
    end else begin : g_r4_wide
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(G7_H_INTER), .K_PAR(K_PAR), .A_WIDTH(R4_AW), .AUTO_SEQ(0), .FRAC_BITS(12), .RD_LAT(D1_RDLAT)) u_seam_r4g (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(g7g_seam_we), .wr_chunk_idx(g7g_seam_widx), .wr_chunk(g7g_seam_wchunk),
            .wr_clear(g7g_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx(r4_silu_rd_idx), .rd_chunk(seam_r4g_rd_c)
        );
        stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(G7_H_INTER), .K_PAR(K_PAR), .A_WIDTH(R4_AW), .AUTO_SEQ(0), .FRAC_BITS(12), .RD_LAT(D1_RDLAT)) u_seam_r4u (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .wr_valid(g7u_seam_we), .wr_chunk_idx(g7u_seam_widx), .wr_chunk(g7u_seam_wchunk),
            .wr_clear(g7u_start), .wr_count(), .full(),
            .rd_valid(1'b1), .rd_chunk_idx(r4_silu_rd_idx), .rd_chunk(seam_r4u_rd_c)
        );
    end
    endgenerate

    localparam int R5_H       = 8192;
    localparam int R5_NCHUNKS = R5_H / K_PAR;
    localparam int R5_CW      = $clog2(R5_NCHUNKS);
    wire r5_wr = silu_valid;

    wire [R5_CW-1:0] r5_rd_idx = act_addr_w[R5_CW-1:0];

    wire [R5_CW-1:0] r5_rd_idx_b = act_addr_b_w[R5_CW-1:0];
    // Inter-stage activation seam storage.
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(R5_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                      .AUTO_SEQ(1), .FRAC_BITS(8), .RD_LAT(D1_RDLAT)) u_seam_r5 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(r5_wr), .wr_chunk_idx('0), .wr_chunk(silu_out_chunk),
        .wr_clear(g7p_start), .wr_count(), .full(seam_r5_full),
        .rd_valid(1'b1), .rd_chunk_idx(r5_rd_idx), .rd_chunk(seam_r5_rd),
        .rd_chunk_idx_b(r5_rd_idx_b), .rd_chunk_b(seam_r5_rd_b)
    );

    localparam int R6_NCHUNKS  = 3072 / K_PAR;
    localparam int R6_CW       = (R6_NCHUNKS > 1) ? $clog2(R6_NCHUNKS) : 1;

    generate
    // Full-width residual and post-attention normalization paths.
    if (BACK_SEQ) begin : g_resid_fullh

        reg signed [32*LANES-1:0]  g9_acc_lat;
        reg [15:0]                 g9_tile_lat;
        reg [LANE_SUBW_CW:0]       g9_we_cnt;
        reg                        g9_we_active;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                g9_acc_lat <= '0; g9_tile_lat <= '0; g9_we_cnt <= '0; g9_we_active <= 1'b0;
            end else if (g9_done) begin
                g9_acc_lat   <= array_acc_out;
                g9_tile_lat  <= gemv_store_idx_w;
                g9_we_cnt    <= '0;
                g9_we_active <= 1'b1;
            end else if (g9_we_active) begin
                if (g9_we_cnt == (LANE_SUBW-1)) g9_we_active <= 1'b0;
                g9_we_cnt <= g9_we_cnt + 1'b1;
            end
        end
        assign r6_store_ack =
            g9_we_active && (g9_we_cnt == (LANE_SUBW-1));
        wire [R6_CW-1:0]                g9_widx = R6_CW'(int'(g9_tile_lat)*LANE_SUBW + int'(g9_we_cnt));
        wire signed [K_PAR*A_WIDTH-1:0] g9_wchunk =
            sat_acc_subchunk(g9_acc_lat, int'(g9_we_cnt) * K_PAR);

        localparam int G9_LO_NTILES = 3072 / LANES;
        wire g9_hi_sel = FORREASON ? (g9_tile_lat >= 16'(G9_LO_NTILES)) : 1'b0;
        reg signed [K_PAR*A_WIDTH-1:0] g9down_buf [0:R6_NCHUNKS-1];
        always_ff @(posedge ap_clk) begin
            if (g9_we_active && !g9_hi_sel) g9down_buf[g9_widx] <= g9_wchunk;
        end

        reg [R6_CW:0] r3_a_cnt;  reg r3_a_run;
        reg [R6_CW:0] r3_b_cnt;  reg r3_b_run;  reg r3_started;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                r3_a_cnt <= '0; r3_a_run <= 1'b0;
                r3_b_cnt <= '0; r3_b_run <= 1'b0; r3_started <= 1'b0;
            end else begin
                if (gemv_done_r2) begin
                    r3_a_cnt <= '0; r3_a_run <= 1'b1;
                    r3_b_cnt <= '0; r3_b_run <= 1'b0; r3_started <= 1'b0;
                end

                if (r3_a_run) begin
                    if (r3_a_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) begin
                        r3_a_run <= 1'b0;
                        r3_started <= 1'b1;
                    end
                    r3_a_cnt <= r3_a_cnt + 1'b1;
                end

                if (r3_started && !r3_b_run) begin
                    r3_b_run <= 1'b1; r3_b_cnt <= '0; r3_started <= 1'b0;
                end else if (r3_b_run) begin
                    if (r3_b_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) r3_b_run <= 1'b0;
                    r3_b_cnt <= r3_b_cnt + 1'b1;
                end
            end
        end
        wire [R6_CW-1:0] r3_a_idx_raw  = r3_a_cnt[R6_CW-1:0];

        reg signed [K_PAR*A_WIDTH-1:0] r3_a_hfc_raw_q;
        reg [R6_CW-1:0]                r3_a_idx_d1;
        reg                            r3_a_run_d1;
        reg                            r3_started_d1;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                r3_a_hfc_raw_q <= '0; r3_a_idx_d1 <= '0;
                r3_a_run_d1 <= 1'b0; r3_started_d1 <= 1'b0;
            end else begin
                r3_a_hfc_raw_q <= h_fc_buf_rd(r3_a_idx_raw);
                r3_a_idx_d1    <= r3_a_idx_raw;
                r3_a_run_d1    <= r3_a_run;
                r3_started_d1  <= r3_started;
            end
        end
        wire             r3_a_wr_f     = r3_a_run_d1;
        wire [R6_CW-1:0] r3_a_idx_f    = r3_a_idx_d1;

        wire signed [K_PAR*A_WIDTH-1:0] r3_a_hfc_raw = r3_a_hfc_raw_q;

        localparam int FR_H1_FRAC = R3_FRAC;
        localparam int FR_G9_FRAC = 8;
        localparam int FR_A_FRAC  = 9;
        localparam int F2_HFC_FRAC = 12;
        localparam int F2_SH_HFC   = F2_HFC_FRAC - R3_FRAC;
        localparam int F2_SH_FR    = FR_A_FRAC   - R3_FRAC;
`ifdef F2_RED_HFC

        wire f2_r3_use_fr = 1'b0;
`else
        wire f2_r3_use_fr = f2_lvl_ge1_w;
`endif
        wire signed [K_PAR*A_WIDTH-1:0] r3_a_chunk_f;
        genvar r3al;
        for (r3al = 0; r3al < K_PAR; r3al = r3al + 1) begin : g_r3a_to_f5
            assign r3_a_chunk_f[r3al*A_WIDTH +: A_WIDTH] = f2_r3_use_fr
                ? ($signed(f2_fr_rd_data_w[r3al*A_WIDTH +: A_WIDTH]) >>> F2_SH_FR)
                : ($signed(r3_a_hfc_raw   [r3al*A_WIDTH +: A_WIDTH]) >>> F2_SH_HFC);
        end

        localparam int FR_REM_W = 5;
        wire [K_PAR*FR_REM_W-1:0] fr_rem_now;
        for (r3al = 0; r3al < K_PAR; r3al = r3al + 1) begin : g_r3a_rem
            wire [3:0] rem_fr  = f2_fr_rd_data_w[r3al*A_WIDTH +: 4];
            wire [6:0] rem_hfc = r3_a_hfc_raw   [r3al*A_WIDTH +: 7];

            wire [4:0] rem_hfc9 = ({1'b0, rem_hfc} + 8'd4) >> 3;
            assign fr_rem_now[r3al*FR_REM_W +: FR_REM_W] =
                f2_r3_use_fr ? {1'b0, rem_fr} : rem_hfc9;
        end
        wire             r3_start_f    = r3_started_d1;
        wire [R6_CW-1:0] r3_b_idx_f    = r3_b_cnt[R6_CW-1:0];

        reg signed [K_PAR*A_WIDTH-1:0] r3_b_chunk_f;
        reg                            r3_b_valid_f;
        always_ff @(posedge ap_clk) begin
            r3_b_chunk_f <= g6out_buf[r3_b_idx_f];
            r3_b_valid_f <= r3_b_run;
        end

        wire                            r3_out_valid_f;
        wire [R6_CW-1:0]                r3_out_idx_f;
        wire signed [K_PAR*A_WIDTH-1:0] r3_out_chunk_f;
        wire                            r3_done_f;
        residual_add #(.H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                       .FRAC_BITS(R3_FRAC), .RDATA_LAT(1)) u_resid_r3 (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .a_valid(r3_a_wr_f), .a_chunk_idx(r3_a_idx_f), .a_chunk(r3_a_chunk_f),
            .start(r3_start_f), .b_chunk(r3_b_chunk_f), .b_valid(r3_b_valid_f),
            .out_valid(r3_out_valid_f), .out_chunk_idx(r3_out_idx_f), .out_chunk(r3_out_chunk_f),
            .done(r3_done_f)
        );

        reg signed [K_PAR*A_WIDTH-1:0] h1_buf_f [0:R6_NCHUNKS-1];
        always_ff @(posedge ap_clk) begin
            if (r3_out_valid_f) h1_buf_f[r3_out_idx_f] <= r3_out_chunk_f;
        end

        assign r3_dbg_out_valid = r3_out_valid_f;
        assign r3_dbg_out_chunk = r3_out_chunk_f;

        reg                  pa_start;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) pa_start <= 1'b0;
            else           pa_start <= r3_done_f;
        end

        wire [R6_CW-1:0]                pa_x_req_chunk;
        wire                            pa_x_req_valid;

        wire [R6_CW-1:0] pa_eff_chunk = (pa_x_req_chunk >= (R6_CW+1)'(1))
                                        ? (pa_x_req_chunk - (R6_CW+1)'(1))
                                        : '0;
        wire pa_inrange = (pa_x_req_chunk >= (R6_CW+1)'(1)) &&
                          (pa_x_req_chunk <= (R6_CW+1)'(R6_NCHUNKS));

        reg signed [K_PAR*A_WIDTH-1:0] pa_x_chunk;
        reg signed [K_PAR*A_WIDTH-1:0] pa_gamma_chunk;
        always_ff @(posedge ap_clk) begin
            pa_x_chunk     <= pa_inrange ? h1_buf_f[pa_eff_chunk]              : '0;
            pa_gamma_chunk <= pa_inrange ? gamma_postattn_buf_rd(pa_eff_chunk) : '0;
        end
        wire signed [K_PAR*A_WIDTH-1:0] pa_out_chunk;
        wire                            pa_out_strobe;
        wire [R6_CW-1:0]                pa_out_chunk_idx;

        rmsnorm_block #(.H(3072), .K_PAR(16), .A_WIDTH(16),
                        .SUMSQ_RSHIFT(12),
                        .SUMSQ_MULT(43691),
                        .SUMSQ_SHR(PRECISION_SAFE ? 29 : 25),
                        .SCALE_SHR(PRECISION_SAFE ? 10 : 8),
                        .XS_WIDTH(18),
                        .RDATA_LAT(1)) u_rmsnorm_premlp (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(pa_start),
            .x_chunk(pa_x_chunk),
            .gamma_chunk(pa_gamma_chunk),
            .out_chunk(pa_out_chunk),
            .out_valid(),
            .done(),
            .x_req_chunk(pa_x_req_chunk),
            .x_req_valid(pa_x_req_valid),
            .out_chunk_strobe(pa_out_strobe),
            .out_chunk_idx_o(pa_out_chunk_idx),
            .apply_en_o()
        );

        wire signed [K_PAR*A_WIDTH-1:0] pa_out_chunk_q412;
        genvar pal;
        for (pal = 0; pal < K_PAR; pal = pal + 1) begin : g_premlp_q412
            wire signed [A_WIDTH:0] pa_xe =
                $signed(pa_out_chunk[pal*A_WIDTH +: A_WIDTH]) <<< 1;
            localparam logic signed [A_WIDTH-1:0] PA_PSAT = {1'b0, {(A_WIDTH-1){1'b1}}};
            localparam logic signed [A_WIDTH-1:0] PA_NSAT = {1'b1, {(A_WIDTH-1){1'b0}}};
            assign pa_out_chunk_q412[pal*A_WIDTH +: A_WIDTH] =
                (pa_xe > $signed(PA_PSAT)) ? PA_PSAT
              : (pa_xe < $signed(PA_NSAT)) ? PA_NSAT
              : pa_xe[A_WIDTH-1:0];
        end

        reg signed [K_PAR*A_WIDTH-1:0] premlp_buf [0:R6_NCHUNKS-1];
        always_ff @(posedge ap_clk) begin
            if (pa_out_strobe) premlp_buf[pa_out_chunk_idx[R6_CW-1:0]] <= pa_out_chunk_q412;
        end

        reg [R6_CW:0] pa_cap_cnt;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)        pa_cap_cnt <= '0;
            else if (g6_start)    pa_cap_cnt <= '0;
            else if (pa_start)    pa_cap_cnt <= '0;
            else if (pa_out_strobe && pa_cap_cnt < (R6_CW+1)'(R6_NCHUNKS))
                                  pa_cap_cnt <= pa_cap_cnt + 1'b1;
        end
        assign seam_premlp_full = (pa_cap_cnt >= (R6_CW+1)'(R6_NCHUNKS));

        reg signed [K_PAR*A_WIDTH-1:0] seam_premlp_rd_r;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) seam_premlp_rd_r <= '0;
            else           seam_premlp_rd_r <= premlp_buf[premlp_rd_idx];
        end
        assign seam_premlp_rd   = seam_premlp_rd_r;

        reg signed [K_PAR*A_WIDTH-1:0] seam_premlp_rd_b_r;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) seam_premlp_rd_b_r <= '0;
            else           seam_premlp_rd_b_r <= premlp_buf[premlp_rd_idx_b];
        end
        assign seam_premlp_rd_b = seam_premlp_rd_b_r;

        reg [R6_CW:0] r7_a_cnt;  reg r7_a_run;
        reg [R6_CW:0] r7_b_cnt;  reg r7_b_run;  reg r7_started;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin
                r7_a_cnt <= '0; r7_a_run <= 1'b0;
                r7_b_cnt <= '0; r7_b_run <= 1'b0; r7_started <= 1'b0;
            end else begin
                if (gemv_done_r6) begin
                    r7_a_cnt <= '0; r7_a_run <= 1'b1;
                    r7_b_cnt <= '0; r7_b_run <= 1'b0; r7_started <= 1'b0;
                end
                if (r7_a_run) begin
                    if (r7_a_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) begin
                        r7_a_run <= 1'b0; r7_started <= 1'b1;
                    end
                    r7_a_cnt <= r7_a_cnt + 1'b1;
                end
                if (r7_started && !r7_b_run) begin
                    r7_b_run <= 1'b1; r7_b_cnt <= '0; r7_started <= 1'b0;
                end else if (r7_b_run) begin
                    if (r7_b_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) r7_b_run <= 1'b0;
                    r7_b_cnt <= r7_b_cnt + 1'b1;
                end
            end
        end
        wire             r7_a_wr_f    = r7_a_run;
        wire [R6_CW-1:0] r7_a_idx_f   = r7_a_cnt[R6_CW-1:0];

        wire signed [K_PAR*A_WIDTH-1:0] r7_a_chunk_f = '0;
        wire             r7_start_f   = r7_started;
        wire [R6_CW-1:0] r7_b_idx_f   = r7_b_cnt[R6_CW-1:0];

        reg signed [K_PAR*A_WIDTH-1:0] r7_b_chunk_f;
        reg                            r7_b_valid_f;
        always_ff @(posedge ap_clk) begin
            r7_b_chunk_f <= g9down_buf[r7_b_idx_f];
            r7_b_valid_f <= r7_b_run;
        end

        residual_add #(.H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                       .FRAC_BITS(R7_FRAC), .RDATA_LAT(1)) u_resid_r7 (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .a_valid(r7_a_wr_f), .a_chunk_idx(r7_a_idx_f), .a_chunk(r7_a_chunk_f),
            .start(r7_start_f), .b_chunk(r7_b_chunk_f), .b_valid(r7_b_valid_f),
            .out_valid(r7_out_valid), .out_chunk_idx(r7_out_chunk_idx_f), .out_chunk(r7_out_chunk),
            .done(r7_done)
        );
        assign r7_out_chunk_idx = r7_out_chunk_idx_f;

        // Optional for-reasoning activation producer.
        if (FORREASON) begin : g_forreason

            wire [R6_CW-1:0] g9_hi_widx =
                R6_CW'((int'(g9_tile_lat) - G9_LO_NTILES)*LANE_SUBW + int'(g9_we_cnt));
            wire signed [K_PAR*A_WIDTH-1:0] g9_hi_wchunk =
                sat_acc_subchunk(g9_acc_lat, int'(g9_we_cnt) * K_PAR);
            reg signed [K_PAR*A_WIDTH-1:0] g9hi_buf [0:R6_NCHUNKS-1];
            always_ff @(posedge ap_clk) begin
                if (g9_we_active && g9_hi_sel) g9hi_buf[g9_hi_widx] <= g9_hi_wchunk;
            end

            reg [R6_CW:0] fr_cnt;  reg fr_run;
            always_ff @(posedge ap_clk) begin
                if (!ap_rst_n) begin
                    fr_cnt <= '0; fr_run <= 1'b0;
                end else if (gemv_done_r6) begin
                    fr_cnt <= '0; fr_run <= 1'b1;
                end else if (fr_run) begin
                    if (fr_cnt == (R6_CW+1)'(R6_NCHUNKS-1)) fr_run <= 1'b0;
                    fr_cnt <= fr_cnt + 1'b1;
                end
            end

            reg signed [K_PAR*A_WIDTH-1:0] fr_h1_q, fr_g9_q;

            (* ram_style = "registers" *)
            reg [K_PAR*5-1:0] fr_rem_buf [0:R6_NCHUNKS-1];
            reg [K_PAR*5-1:0] fr_rem_q;
            always_ff @(posedge ap_clk) begin
                if (ACCFIX_SEAM_P && r3_a_wr_f) fr_rem_buf[r3_a_idx_f] <= fr_rem_now;
            end
            reg                            fr_v_q;
            reg [R6_CW-1:0]                fr_idx_q;
            always_ff @(posedge ap_clk) begin
                fr_h1_q  <= h1_buf_f[fr_cnt[R6_CW-1:0]];
                fr_rem_q <= fr_rem_buf[fr_cnt[R6_CW-1:0]];
                fr_g9_q  <= g9hi_buf[fr_cnt[R6_CW-1:0]];
                fr_v_q   <= fr_run;
                fr_idx_q <= fr_cnt[R6_CW-1:0];
            end

            wire [R6_CW-1:0] f2_rd_addr = r3_a_run ? r3_a_idx_raw
                                                   : R6_CW'(g1_rd_idx);

`ifdef F2_RED_NOGUARD

            wire f2_fr_wr_ok = 1'b1;
`else
            wire f2_fr_wr_ok = (FR_KF_BANKS > 1) || !f2_lvl_ge1_w;
`endif

            localparam int FR_BW   = (FR_KF_BANKS <= 1) ? 1 : $clog2(FR_KF_BANKS);
            localparam bit FR_DBUF = (FR_KF_BANKS > 8);
            wire        fr_wr_epoch = level_i_w[0];
            wire        fr_rd_epoch = ~level_i_w[0];
`ifdef G3_RED_POOLLOW

            wire [2:0]  fr_par_slot = fr_par_node_w[2:0];
`else

            wire [2:0]  fr_par_slot = fr_par_slot_w;
`endif
            wire [FR_BW-1:0] fr_wr_branch =
                (FR_KF_BANKS <= 1) ? '0 :
                FR_DBUF            ? FR_BW'({fr_wr_epoch, branch_j_w}) :
                                     FR_BW'(branch_j_w);
            wire [FR_BW-1:0] fr_rd_branch =
                (FR_KF_BANKS <= 1) ? '0 :
                FR_DBUF            ? FR_BW'({fr_rd_epoch, fr_par_slot}) :
                                     FR_BW'(fr_par_slot);

            wire                            fr_out_valid;
            wire [R6_CW-1:0]                fr_out_idx;
            wire signed [K_PAR*A_WIDTH-1:0] fr_out_chunk;
            wire signed [K_PAR*A_WIDTH-1:0] fr_rd_chunk;
            for_reasoning_producer #(
                .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                .H1_FRAC(FR_H1_FRAC), .G9_FRAC(FR_G9_FRAC), .A_FRAC(FR_A_FRAC),
                .KF_BANKS(FR_KF_BANKS), .ABANK_BRAM_P(FR_ABANK_BRAM),
                .SEAM_FIX_P(ACCFIX_SEAM_P)
            ) u_forreason (
                .clk(ap_clk), .rst_n(ap_rst_n),
                .in_valid(fr_v_q && f2_fr_wr_ok), .in_idx(fr_idx_q),
                .h1_chunk(fr_h1_q), .g9_chunk(fr_g9_q),
                .seam_en(accfix_seam_en),
                .rem_chunk(fr_rem_q),
                .wr_branch(fr_wr_branch),
                .out_valid(fr_out_valid), .out_idx(fr_out_idx), .out_chunk(fr_out_chunk),
                .rd_branch(fr_rd_branch),
                .rd_idx(f2_rd_addr),
                .rd_chunk(fr_rd_chunk)
            );

            assign f2_fr_rd_data_w = fr_rd_chunk;
            assign f2_lvl_ge1_w    = DEPTH_N && (level_i_w != 4'd0);

`ifndef SYNTHESIS
            f2_c4_port_disjoint: assert property (
                @(posedge ap_clk) disable iff (!ap_rst_n)
                !(f2_lvl_ge1_w && g1_rd_ir && r3_a_run)
            ) else $error("F2 C4 VIOLATED @ %0t: g1_rd_ir & r3_a_run both high -> shared a-bank read port collision (G1 gets R3's address)", $time);
`endif
            initial begin
                if (EMB_K_PAR*EMB_AW != K_PAR*A_WIDTH)
                    $error("F2: G1 chunk geometry %0dx%0d != producer chunk geometry %0dx%0d",
                           EMB_K_PAR, EMB_AW, K_PAR, A_WIDTH);
            end

            (* keep = "true" *) reg signed [K_PAR*A_WIDTH-1:0] fr_a_out_keep;
            always_ff @(posedge ap_clk) begin
                if (fr_out_valid) fr_a_out_keep <= fr_out_chunk;
            end
        end else begin : g_no_forreason

            assign f2_fr_rd_data_w = '0;
            assign f2_lvl_ge1_w    = 1'b0;
        end
    end else begin : g_resid_pilot

        // Pilot path stores one chunk per tile without the 8-cycle serializer;
        // the controller's store handshake completes immediately.
        assign r6_store_ack = 1'b1;

        assign f2_fr_rd_data_w = '0;
        assign f2_lvl_ge1_w    = 1'b0;

        assign seam_premlp_rd   = seam_r2_rd;
        assign seam_premlp_full = seam_r2_full;

        assign seam_premlp_rd_b = '0;

        assign r3_dbg_out_valid = 1'b0;
        assign r3_dbg_out_chunk = '0;

        wire signed [K_PAR*A_WIDTH-1:0] x_input_chunk = h_fc_buf_rd(EMB_CW'(EMB_DATA_SLOT));
        reg r3_a_wr;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) r3_a_wr <= 1'b0; else r3_a_wr <= g6_start;
        end
        reg signed [K_PAR*A_WIDTH-1:0] r3_b_chunk_l;  reg r3_b_valid;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin r3_b_chunk_l <= '0; r3_b_valid <= 1'b0; end
            else begin
                if (g6_done) r3_b_chunk_l <= g6_out_sat_chunk_pilot;
                r3_b_valid <= g6_done;
            end
        end
        wire r3_out_valid; wire signed [K_PAR*A_WIDTH-1:0] r3_out_chunk; wire r3_done;
        residual_add #(.H(SEAM_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                       .FRAC_BITS(R3_FRAC)) u_resid_r3 (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .a_valid(r3_a_wr), .a_chunk_idx('0), .a_chunk(x_input_chunk),
            .start(g6_done), .b_chunk(r3_b_chunk_l), .b_valid(r3_b_valid),
            .out_valid(r3_out_valid), .out_chunk_idx(), .out_chunk(r3_out_chunk), .done(r3_done)
        );
        reg signed [K_PAR*A_WIDTH-1:0] h1_buf;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) h1_buf <= '0; else if (r3_out_valid) h1_buf <= r3_out_chunk;
        end
        wire signed [K_PAR*A_WIDTH-1:0] g9_out_sat_chunk =
            sat_acc_subchunk(array_acc_out, 0);
        reg r7_a_wr;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) r7_a_wr <= 1'b0; else r7_a_wr <= g9_start;
        end
        reg signed [K_PAR*A_WIDTH-1:0] r7_b_chunk_l;  reg r7_b_valid;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n) begin r7_b_chunk_l <= '0; r7_b_valid <= 1'b0; end
            else begin
                if (g9_done) r7_b_chunk_l <= g9_out_sat_chunk;
                r7_b_valid <= g9_done;
            end
        end
        localparam int R7_PILOT_CW = (SEAM_H/K_PAR > 1) ? $clog2(SEAM_H/K_PAR) : 1;
        wire [R7_PILOT_CW-1:0] r7_pilot_idx;
        residual_add #(.H(SEAM_H), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                       .FRAC_BITS(R7_FRAC)) u_resid_r7 (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .a_valid(r7_a_wr), .a_chunk_idx('0), .a_chunk(h1_buf),
            .start(g9_done), .b_chunk(r7_b_chunk_l), .b_valid(r7_b_valid),
            .out_valid(r7_out_valid), .out_chunk_idx(r7_pilot_idx), .out_chunk(r7_out_chunk),
            .done(r7_done)
        );
        assign r7_out_chunk_idx = {{(R6_CW-R7_PILOT_CW){1'b0}}, r7_pilot_idx};
    end
    endgenerate

    wire                            r3_dbg_out_valid;
    wire signed [K_PAR*A_WIDTH-1:0] r3_dbg_out_chunk;

    wire                            r7_out_valid;
    wire signed [K_PAR*A_WIDTH-1:0] r7_out_chunk;
    wire                            r7_done;
    wire [R6_CW-1:0]                r7_out_chunk_idx;
    wire [R6_CW-1:0]                r7_out_chunk_idx_f;
    wire [R6_CW-1:0]                r6_wr_idx = r7_out_chunk_idx;

    wire signed [K_PAR*A_WIDTH-1:0] g6_out_sat_chunk_pilot = sat_acc_head0_chunk(array_acc_out);

    wire [R6_CW-1:0] r6_rd_idx = BACK_SEQ
        ? (g11_x_req_chunk - G11_CW'(1))
        : g11_x_req_chunk[R6_CW-1:0];
    wire r6_rd_inrange = BACK_SEQ
        ? (g11_x_req_chunk >= G11_CW'(1) && g11_x_req_chunk <= G11_CW'(R6_NCHUNKS))
        : 1'b1;

    // Inter-stage activation seam storage.
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH),
                      .AUTO_SEQ(0), .FRAC_BITS(R7_FRAC)) u_seam_r6 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(r7_out_valid), .wr_chunk_idx(r6_wr_idx), .wr_chunk(r7_out_chunk),
        .wr_clear(g9_start), .wr_count(), .full(seam_r6_full),
        .rd_valid(r6_rd_inrange), .rd_chunk_idx(r6_rd_idx), .rd_chunk(seam_r6_rd_c)
    );

    localparam int R8_REQ_LSHIFT = G11_OUT_FRAC - G11_OUT_FRAC;
    function automatic logic signed [A_WIDTH-1:0] sat16_lshift_q124_to_q412;
        input signed [A_WIDTH-1:0] x;
        logic signed [A_WIDTH+R8_REQ_LSHIFT:0] xe;
        localparam logic signed [A_WIDTH-1:0] PSAT = {1'b0, {(A_WIDTH-1){1'b1}}};
        localparam logic signed [A_WIDTH-1:0] NSAT = {1'b1, {(A_WIDTH-1){1'b0}}};
        begin
            xe = $signed(x) <<< R8_REQ_LSHIFT;
            if (xe > $signed(PSAT))      sat16_lshift_q124_to_q412 = PSAT;
            else if (xe < $signed(NSAT)) sat16_lshift_q124_to_q412 = NSAT;
            else                         sat16_lshift_q124_to_q412 = xe[A_WIDTH-1:0];
        end
    endfunction

    wire signed [K_PAR*A_WIDTH-1:0] r8_chunk_q412;
    genvar r8l;
    generate
        for (r8l = 0; r8l < K_PAR; r8l = r8l + 1) begin : g_r8_requant
            assign r8_chunk_q412[r8l*A_WIDTH +: A_WIDTH] =
                sat16_lshift_q124_to_q412(rms_g11_out_chunk[r8l*A_WIDTH +: A_WIDTH]);
        end
    endgenerate

    wire [G11_CW-1:0] glm_a_chunk_idx;
    wire              glm_a_chunk_idx_valid;
    wire [G11_CW-1:0] r8_rd_idx = glm_a_chunk_idx;

    wire [15:0] lmtap_seam_dout0;
    wire        lmtap_rd_guard_q;

    wire [G11_CW-1:0] glm_seam_idx_dlv;
    wire              glm_seam_valid_dlv;
    // Inter-stage activation seam storage.
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(0), .RD_LAT(D1_RDLAT), .DBG_TAP(DBG_LMTAP)) u_seam_r8 (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(g11_out_strobe), .wr_chunk_idx(g11_out_chunk_idx[G11_CW-1:0]),
        .wr_chunk(r8_chunk_q412),
        .wr_clear(g11_start), .wr_count(), .full(),
        .rd_valid(glm_a_chunk_idx_valid), .rd_chunk_idx(r8_rd_idx), .rd_chunk(seam_r8_rd),
        .dbg_rd_chunk_q0(lmtap_seam_dout0), .dbg_rd_guard_q(lmtap_rd_guard_q),
        .rd_chunk_idx_out(glm_seam_idx_dlv), .rd_valid_out(glm_seam_valid_dlv)
    );

    localparam int LR_M       = 256;
    localparam int LR_LANES   = 16;
    localparam int LR_NCHUNKS = LR_M / LR_LANES;
    localparam int LR_OCW     = (LR_NCHUNKS > 1) ? $clog2(LR_NCHUNKS) : 1;

    wire                           glm_wr_valid;
    wire [LR_OCW-1:0]              glm_wr_chunk_idx;
    wire signed [LR_LANES*A_WIDTH-1:0] glm_wr_chunk;

    lmdown_gemv_banksplit #(
        .K(3072), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .M(LR_M), .LANES(LR_LANES),
        .IN_FRAC(10), .W_FRAC(15), .OUT_FRAC(10),
        .W_HEX("/home/y344shi/eagle4_artifacts/p4/lmdown_W_q15.hex"),
        .W_MEM_DIR(LMDOWN_MEM_DIR_P),

        .LMDOWN_SLR_REG(D1_RDLAT),

        .SEAM_FF_LAND(SEAM_FF_LAND_P),

        .COTRAVEL(1'b1),
        .DBG_TAP(DBG_LMTAP)
    ) u_lmdown (
        .clk(ap_clk), .rst_n(ap_rst_n), .start(glm_start),
        .a_chunk(seam_r8_rd),
        .a_chunk_idx(glm_a_chunk_idx),
        .a_chunk_idx_valid(glm_a_chunk_idx_valid),
        .a_chunk_idx_in(glm_seam_idx_dlv),
        .a_chunk_idx_valid_in(glm_seam_valid_dlv),
        .wr_valid(glm_wr_valid), .wr_chunk_idx(glm_wr_chunk_idx), .wr_chunk(glm_wr_chunk),
        .done(glm_done),
        .dbg_a_d0(lmtap_a_d0), .dbg_run_valid(lmtap_run_valid)
    );

    wire [15:0] lmtap_a_d0;
    wire        lmtap_run_valid;
    reg         lmtap_armed;
    reg         lmtap_fired;
    reg  [15:0] lmtap_seam_cap;
    reg         lmtap_guard_cap;
    reg  [15:0] lmtap_ad_cap;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            lmtap_armed     <= 1'b0;
            lmtap_fired     <= 1'b0;
            lmtap_seam_cap  <= 16'h0;
            lmtap_guard_cap <= 1'b0;
            lmtap_ad_cap    <= 16'h0;
        end else begin
            if (glm_start) begin
                lmtap_armed <= 1'b1;
                lmtap_fired <= 1'b0;
            end else if (lmtap_armed && lmtap_run_valid) begin

                lmtap_seam_cap  <= lmtap_seam_dout0;
                lmtap_guard_cap <= lmtap_rd_guard_q;
                lmtap_ad_cap    <= lmtap_a_d0;
                lmtap_armed     <= 1'b0;
                lmtap_fired     <= 1'b1;
            end
        end
    end

    wire [31:0] lmtap_word_seam  = {8'hA1, 7'b0, lmtap_fired, lmtap_seam_cap[15:0]};
    wire [31:0] lmtap_word_guard = {8'hA2, 7'b0, lmtap_fired, 15'b0, lmtap_guard_cap};
    wire [31:0] lmtap_word_ad    = {8'hA3, 7'b0, lmtap_fired, lmtap_ad_cap[15:0]};

    wire [LR_OCW-1:0] lr_rd_idx = act_addr_w[LR_OCW-1:0];

    wire [LR_OCW-1:0] lr_rd_idx_b = act_addr_b_w[LR_OCW-1:0];

    localparam int INJ_WIDX_W = $clog2(INJ_MAX_W);
    wire signed [LR_LANES*A_WIDTH-1:0] inj_glm_chunk;
    genvar gilm;
    generate
        for (gilm = 0; gilm < LR_LANES; gilm = gilm + 1) begin : g_inj_glm

            wire [INJ_WIDX_W-1:0] gwi =
                (INJ_WIDX_W'(glm_wr_chunk_idx) * LR_LANES) + INJ_WIDX_W'(gilm);
            assign inj_glm_chunk[gilm*A_WIDTH +: A_WIDTH] =
                inj_bus[gwi*32 +: A_WIDTH];
        end
    endgenerate
    wire signed [LR_LANES*A_WIDTH-1:0] glm_wr_chunk_mux =
        inject_en[11] ? inj_glm_chunk : glm_wr_chunk;

    // Inter-stage activation seam storage.
    stage_io_buffer_sel #(.SEAM_FF(SEAM_FF), .H(LR_M), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .AUTO_SEQ(0),
                      .FRAC_BITS(10), .RD_LAT(D1_RDLAT)) u_low_rank_buf (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .wr_valid(glm_wr_valid),
        .wr_chunk_idx(glm_wr_chunk_idx),
        .wr_chunk(glm_wr_chunk_mux),
        .wr_clear(glm_start), .wr_count(), .full(),
        .rd_valid(1'b1), .rd_chunk_idx(lr_rd_idx), .rd_chunk(low_rank_rd),
        .rd_chunk_idx_b(lr_rd_idx_b), .rd_chunk_b(low_rank_rd_b)
    );

    localparam int G8_K_F   = 8;
    localparam int G8_H     = 3072;
    localparam int G8_V_MAX = TREE_V_MAX;
    localparam int G8_IDX_W = TREE_IDX_W;

    wire g8_dbg_err;
    wire g8_dbg_activity;

    wire [3:0]  g8_dbg_sel_substate;
    wire [9:0]  g8_dbg_cand_filled;
    wire [9:0]  g8_dbg_rows_gathered;
    wire        g8_dbg_seen_last;
    wire        g8_dbg_in_gather;
    wire [BOOT_AXI_ADDR_W-1:0] g8_dbg_last_araddr;
    wire        g8_dbg_rresp_err;

    wire [8*32-1:0] g8_dbg_scores;
    wire            g8_dbg_scores_v;

    generate
    // G8 candidate gather and rescoring.
    if (G8_SPLICE) begin : g_g8_splice

        (* ram_style = "registers" *)
        logic [15:0] nrm_buf [0:G8_H-1];

        function automatic logic [15:0] nrm_i16q10_to_fp16;
            input logic signed [15:0] q610;
            logic               sign;
            logic [16:0]        mag;
            integer             p;
            integer             shift;
            logic [10:0]        mant;
            logic               round_bit, sticky, roundup;
            logic [4:0]         exp_b;
            begin
                sign = q610[15];

                if (q610 == 16'sh8000)
                    mag = 17'd32768;
                else if (sign)
                    mag = {1'b0, (~q610 + 16'sd1)};
                else
                    mag = {1'b0, q610};
                if (mag == 17'd0) begin
                    nrm_i16q10_to_fp16 = 16'h0000;
                end else begin

                    p = 0;
                    for (int b = 16; b >= 0; b = b - 1)
                        if (p == 0 && mag[b]) p = b;
                    exp_b = 5'(p + 5);
                    if (p >= 10) begin
                        shift     = p - 10;
                        mant      = 11'((mag >> shift) & 17'h3FF);
                        round_bit = (shift == 0) ? 1'b0 : ((mag >> (shift-1)) & 17'd1);
                        sticky    = (shift <= 1) ? 1'b0
                                    : ((mag & ((17'd1 << (shift-1)) - 17'd1)) != 17'd0);
                    end else begin
                        shift     = 10 - p;
                        mant      = 11'((mag << shift) & 17'h3FF);
                        round_bit = 1'b0;
                        sticky    = 1'b0;
                    end

                    roundup = round_bit & (sticky | mant[0]);
                    if (roundup) begin
                        mant = mant + 11'd1;
                        if (mant == 11'h400) begin
                            mant  = 11'd0;
                            exp_b = exp_b + 5'd1;
                        end
                    end
                    nrm_i16q10_to_fp16 = {sign, exp_b, mant[9:0]};
                end
            end
        endfunction

        always_ff @(posedge ap_clk) begin
            if (g11_out_strobe) begin
                for (int e = 0; e < K_PAR; e = e + 1) begin
                    automatic int eidx = (int'(g11_out_chunk_idx) * K_PAR) + e;
                    if (eidx < G8_H)
                        nrm_buf[eidx] <=
                            nrm_i16q10_to_fp16(rms_g11_out_chunk[e*A_WIDTH +: A_WIDTH]);
                end
            end
        end

        wire [$clog2(G8_H)-1:0] g8_nrm_rd_addr;
        wire [15:0]             g8_nrm_rd_data = nrm_buf[g8_nrm_rd_addr];

        wire [G8_WIDE_P*16-1:0] g8_nrm_rd_data_wide;
        genvar gw;
        for (gw = 0; gw < G8_WIDE_P; gw = gw + 1) begin : g_nrm_wide
            wire [$clog2(G8_H):0] gw_addr = {1'b0, g8_nrm_rd_addr} + gw;
            assign g8_nrm_rd_data_wide[gw*16 +: 16] =
                (gw_addr < G8_H) ? nrm_buf[gw_addr[$clog2(G8_H)-1:0]] : 16'h0000;
        end

        wire        g8_cheap_ready;
        wire        g8_launch = tree_start;

        wire [G8_K_F*G8_IDX_W-1:0]  g8_frontier_ids;
        wire [G8_K_F*32-1:0]        g8_frontier_scores;
        wire                        g8_done_w;
        wire                        g8_err_w;

        if (G8_PIPELINED) begin : g_g8p

            wire        g8p_pf_drop;
            wire [G8_IDX_W:0] g8p_pf_cnt;

            wire                       g8l_arvalid;
            wire [BOOT_AXI_ADDR_W-1:0] g8l_araddr;
            wire [7:0]                 g8l_arlen;
            wire [2:0]                 g8l_arsize;
            wire [1:0]                 g8l_arburst;
            wire [BOOT_AXI_ID_W-1:0]   g8l_arid;
            wire                       g8l_rready;

            wire [G8_NCH-1:0]          st_arvalid;
            wire [G8_NCH-1:0]          st_arready;
            wire [BOOT_AXI_ADDR_W-1:0] st_araddr  [0:G8_NCH-1];
            wire [7:0]                 st_arlen   [0:G8_NCH-1];
            wire [2:0]                 st_arsize  [0:G8_NCH-1];
            wire [1:0]                 st_arburst [0:G8_NCH-1];
            wire [G8_NCH-1:0]          st_rvalid;
            wire [G8_NCH-1:0]          st_rready;
            wire [G8_DATA_W-1:0]       st_rdata   [0:G8_NCH-1];
            wire [G8_NCH-1:0]          st_rlast;
            wire [$clog2(G8_H)-1:0]    st_nrm_addr [0:G8_NCH-1];
            wire [G8_WIDE_P*16-1:0]    st_nrm_wide [0:G8_NCH-1];

            if (G8_NCH > 1) begin : g_nch_mux

                assign st_arready[0] = m_axi_g8_arready;
                assign st_rvalid [0] = m_axi_g8_rvalid;
                assign st_rdata  [0] = m_axi_g8_rdata;
                assign st_rlast  [0] = m_axi_g8_rlast;

                assign st_arready[1] = m_axi_g8b_arready;
                assign st_rvalid [1] = m_axi_g8b_rvalid;
                assign st_rdata  [1] = m_axi_g8b_rdata;
                assign st_rlast  [1] = m_axi_g8b_rlast;
                assign m_axi_g8b_arvalid = st_arvalid[1];
                assign m_axi_g8b_araddr  = st_araddr [1];
                assign m_axi_g8b_arlen   = st_arlen  [1];
                assign m_axi_g8b_arsize  = st_arsize [1];
                assign m_axi_g8b_arburst = st_arburst[1];
                assign m_axi_g8b_arid    = {BOOT_AXI_ID_W{1'b0}};
                assign m_axi_g8b_rready  = st_rready [1];
            end else begin : g_nch_off

                assign st_arready = '0;  assign st_rvalid = '0;  assign st_rlast = '0;
                assign st_rdata[0] = '0;
                assign m_axi_g8b_arvalid = 1'b0;  assign m_axi_g8b_araddr = '0;
                assign m_axi_g8b_arlen = 8'd0;    assign m_axi_g8b_arsize = 3'd0;
                assign m_axi_g8b_arburst = 2'b01; assign m_axi_g8b_arid = {BOOT_AXI_ID_W{1'b0}};
                assign m_axi_g8b_rready = 1'b0;
            end

`ifndef SYNTHESIS
            initial if (G8_NCH > 2)
                $fatal(1, "eagle4_drafter: G8_NCH=%0d — only N<=2 is plumbed to top-level masters", G8_NCH);
`endif

            assign m_axi_g8_arvalid = (G8_NCH > 1) ? st_arvalid[0] : g8l_arvalid;
            assign m_axi_g8_araddr  = (G8_NCH > 1) ? st_araddr [0] : g8l_araddr;
            assign m_axi_g8_arlen   = (G8_NCH > 1) ? st_arlen  [0] : g8l_arlen;
            assign m_axi_g8_arsize  = (G8_NCH > 1) ? st_arsize [0] : g8l_arsize;
            assign m_axi_g8_arburst = (G8_NCH > 1) ? st_arburst[0] : g8l_arburst;
            assign m_axi_g8_arid    = (G8_NCH > 1) ? {BOOT_AXI_ID_W{1'b0}} : g8l_arid;
            assign m_axi_g8_rready  = (G8_NCH > 1) ? st_rready [0] : g8l_rready;

            for (genvar gnc = 0; gnc < G8_NCH; gnc++) begin : g_st_nrm
                for (genvar gnw = 0; gnw < G8_WIDE_P; gnw++) begin : g_w
                    wire [$clog2(G8_H):0] a = {1'b0, st_nrm_addr[gnc]} + gnw;
                    assign st_nrm_wide[gnc][gnw*16 +: 16] =
                        (a < G8_H) ? nrm_buf[a[$clog2(G8_H)-1:0]] : 16'h0000;
                end
            end

            g8_rescore_p #(
                .V_MAX(G8_V_MAX), .H(G8_H), .K_C(G8_K_C), .K_F(G8_K_F),
                .IDX_W(G8_IDX_W), .SCORE_W(32),
                .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(G8_DATA_W),
                .AXI_ID_W(BOOT_AXI_ID_W), .MAX_BURST(G8_MAX_BURST),
                .G8_OUTSTANDING(G8_OUTSTANDING),
                .G8_PREFETCH(G8_PREFETCH),
                .G8_OPT1C(G8_OPT1C),
                .G8_OPT1D(G8_OPT1D),
                .G8_MIN_NG(G8_MIN_NG),
                .G8_OPT1F(G8_OPT1F),
                .G8_SELPIPE(G8_SELPIPE),
                .G8_OPT1G(G8_OPT1G),
                .G8_WIDE(G8_WIDE),
                .WIDE_P(G8_WIDE_P),
                .WIDE_IL(G8_WIDE_IL),
                .WIDE_M(G8_WIDE_M),
                .MROUND_P(G8_MROUND),
                .SCALAR_II1(G8_SCALAR_II1),
                .G8_CONSUME_HARDEN_P(G8_CONSUME_HARDEN),
                .G8_LANE_FIX(G8_LANE_FIX),
                .G8_ROWPP(G8_ROWPP),
                .G8_NCH(G8_NCH)
            ) u_g8 (
                .clk(ap_clk), .rst_n(ap_rst_n),
                .start(g8_launch),
                .kc_cfg(kc_cfg),
                .cheap_valid(tt_valid_in),
                .cheap_ready(g8_cheap_ready),
                .cheap_score(tt_score_in),
                .cheap_index(tt_index_in),
                .cheap_last(tt_last_in),
                .nrm_rd_data(g8_nrm_rd_data),

                .nrm_rd_data_wide(g8_nrm_rd_data_wide),
                .nrm_rd_addr(g8_nrm_rd_addr),

                .m_axi_arvalid(g8l_arvalid), .m_axi_arready((G8_NCH > 1) ? 1'b0 : m_axi_g8_arready),
                .m_axi_araddr(g8l_araddr),   .m_axi_arlen(g8l_arlen),
                .m_axi_arsize(g8l_arsize),   .m_axi_arburst(g8l_arburst),
                .m_axi_arid(g8l_arid),
                .m_axi_rvalid((G8_NCH > 1) ? 1'b0 : m_axi_g8_rvalid), .m_axi_rready(g8l_rready),
                .m_axi_rdata(m_axi_g8_rdata),  .m_axi_rlast((G8_NCH > 1) ? 1'b0 : m_axi_g8_rlast),
                .m_axi_rid(m_axi_g8_rid),      .m_axi_rresp(m_axi_g8_rresp),

                .stripe_arvalid(st_arvalid), .stripe_arready(st_arready),
                .stripe_araddr(st_araddr),   .stripe_arlen(st_arlen),
                .stripe_arsize(st_arsize),   .stripe_arburst(st_arburst),
                .stripe_rvalid(st_rvalid),   .stripe_rready(st_rready),
                .stripe_rdata(st_rdata),     .stripe_rlast(st_rlast),
                .stripe_nrm_addr(st_nrm_addr), .stripe_nrm_wide(st_nrm_wide),
                .lmfull_hbm_base(lmfull_hbm_base),

                .prefetch_req(1'b0), .prefetch_req_id('0),
                .prefetch_evict(1'b0), .prefetch_evict_id('0),
                .prefetch_drop(g8p_pf_drop), .prefetch_count(g8p_pf_cnt),
                .frontier_ids(g8_frontier_ids),
                .frontier_scores(g8_frontier_scores),
                .done(g8_done_w),
                .g8_err(g8_err_w),

                .dbg_sel_substate(g8_dbg_sel_substate),
                .dbg_cand_filled(g8_dbg_cand_filled),
                .dbg_rows_gathered(g8_dbg_rows_gathered),
                .dbg_seen_last(g8_dbg_seen_last),
                .dbg_in_gather(g8_dbg_in_gather),
                .dbg_last_araddr(g8_dbg_last_araddr),
                .dbg_rresp_err(g8_dbg_rresp_err),

                .dbg_cand_rdaddr(g8_cand_rdaddr),
                .dbg_cand_id(g8_cand_dump_id),
                .dbg_cand_cheap_score(g8_cand_dump_score),

                .bridge_rdaddr_nxt(bridge_cand_rdaddr_nxt),
                .bridge_score_ff(g8_bridge_score_ff),

                .sel_thresh(g8_sel_thresh),
                .sel_thresh_valid(g8_sel_thresh_valid)
            );
        end else begin : g_g8s

            assign m_axi_g8b_arvalid = 1'b0;  assign m_axi_g8b_araddr = '0;
            assign m_axi_g8b_arlen = 8'd0;    assign m_axi_g8b_arsize = 3'd0;
            assign m_axi_g8b_arburst = 2'b01; assign m_axi_g8b_arid = {BOOT_AXI_ID_W{1'b0}};
            assign m_axi_g8b_rready = 1'b0;
`ifndef SYNTHESIS
            initial if (G8_NCH > 1)
                $fatal(1, "eagle4_drafter: G8_NCH>1 requires G8_PIPELINED=1 (stripe frontend lives in g8_rescore_p)");
`endif

// synthesis translate_off
            initial begin
                if (TREE_P > 1)
                    $error("eagle4_drafter: G8_SPLICE=1 && TREE_P>1 requires G8_PIPELINED=1 (sel_thresh pre-filter export). Got TREE_P=%0d G8_PIPELINED=0.", TREE_P);
            end
// synthesis translate_on
            assign g8_sel_thresh       = 32'h0;
            assign g8_sel_thresh_valid = 1'b0;
        g8_rescore #(
            .V_MAX(G8_V_MAX), .H(G8_H), .K_C(G8_K_C), .K_F(G8_K_F),
            .IDX_W(G8_IDX_W), .SCORE_W(32),
            .AXI_ADDR_W(BOOT_AXI_ADDR_W), .AXI_DATA_W(BOOT_AXI_DATA_W),
            .AXI_ID_W(BOOT_AXI_ID_W), .MAX_BURST(G8_MAX_BURST)
        ) u_g8 (
            .clk(ap_clk), .rst_n(ap_rst_n),
            .start(g8_launch),

            .cheap_valid(tt_valid_in),
            .cheap_ready(g8_cheap_ready),
            .cheap_score(tt_score_in),
            .cheap_index(tt_index_in),
            .cheap_last(tt_last_in),

            .nrm_rd_data(g8_nrm_rd_data),
            .nrm_rd_addr(g8_nrm_rd_addr),

            .m_axi_arvalid(m_axi_g8_arvalid), .m_axi_arready(m_axi_g8_arready),
            .m_axi_araddr(m_axi_g8_araddr),   .m_axi_arlen(m_axi_g8_arlen),
            .m_axi_arsize(m_axi_g8_arsize),   .m_axi_arburst(m_axi_g8_arburst),
            .m_axi_arid(m_axi_g8_arid),
            .m_axi_rvalid(m_axi_g8_rvalid),   .m_axi_rready(m_axi_g8_rready),
            .m_axi_rdata(m_axi_g8_rdata),     .m_axi_rlast(m_axi_g8_rlast),
            .m_axi_rid(m_axi_g8_rid),         .m_axi_rresp(m_axi_g8_rresp),
            .lmfull_hbm_base(lmfull_hbm_base),

            .frontier_ids(g8_frontier_ids),
            .frontier_scores(g8_frontier_scores),
            .done(g8_done_w),
            .g8_err(g8_err_w),

            .dbg_sel_substate(g8_dbg_sel_substate),
            .dbg_cand_filled(g8_dbg_cand_filled),
            .dbg_rows_gathered(g8_dbg_rows_gathered),
            .dbg_seen_last(g8_dbg_seen_last),
            .dbg_in_gather(g8_dbg_in_gather),
            .dbg_last_araddr(g8_dbg_last_araddr),
            .dbg_rresp_err(g8_dbg_rresp_err),

            .dbg_cand_rdaddr(g8_cand_rdaddr),
            .dbg_cand_id(g8_cand_dump_id),
            .dbg_cand_cheap_score(g8_cand_dump_score)
        );

        assign g8_bridge_score_ff = g8_cand_dump_score;
        end

        reg g8_act_sticky;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)      g8_act_sticky <= 1'b0;
            else if (g8_launch) g8_act_sticky <= 1'b0;
            else if (g8_done_w && (|g8_frontier_scores)) g8_act_sticky <= 1'b1;
        end
        assign g8_dbg_err      = g8_err_w;
        assign g8_dbg_activity = g8_act_sticky;

        assign g8_dbg_scores   = g8_frontier_scores;
        assign g8_dbg_scores_v = g8_done_w;

        logic g8_done_sticky;
        always_ff @(posedge ap_clk) begin
            if (!ap_rst_n)        g8_done_sticky <= 1'b0;
            else if (g8_launch || start_pulse) g8_done_sticky <= 1'b0;
            else if (g8_done_w)   g8_done_sticky <= 1'b1;
        end
        assign g8_done = g8_done_sticky;

        assign ow_trigger     = g8_done_w;
        assign ow_topk_scores  = g8_frontier_scores;
        assign ow_topk_indices = g8_frontier_ids;

    end else begin : g_g8_bypass

        assign m_axi_g8b_arvalid = 1'b0;  assign m_axi_g8b_araddr = '0;
        assign m_axi_g8b_arlen = 8'd0;    assign m_axi_g8b_arsize = 3'd1;
        assign m_axi_g8b_arburst = 2'b01; assign m_axi_g8b_arid = '0;
        assign m_axi_g8b_rready = 1'b0;

        assign g8_sel_thresh       = 32'h0;
        assign g8_sel_thresh_valid = 1'b0;
        assign g8_done           = 1'b0;
        assign m_axi_g8_arvalid  = 1'b0;
        assign m_axi_g8_araddr   = '0;
        assign m_axi_g8_arlen    = '0;
        assign m_axi_g8_arsize   = 3'b001;
        assign m_axi_g8_arburst  = 2'b01;
        assign m_axi_g8_arid     = '0;
        assign m_axi_g8_rready   = 1'b0;
        assign ow_trigger        = tree_done;
        assign ow_topk_scores    = tree_topk_scores;
        assign ow_topk_indices   = tree_topk_indices;

        assign g8_dbg_err        = 1'b0;
        assign g8_dbg_activity   = 1'b0;

        assign g8_dbg_scores     = '0;
        assign g8_dbg_scores_v   = 1'b0;

        assign g8_dbg_sel_substate  = 4'd0;
        assign g8_dbg_cand_filled   = 10'd0;
        assign g8_dbg_rows_gathered = 10'd0;
        assign g8_dbg_seen_last     = 1'b0;
        assign g8_dbg_in_gather     = 1'b0;
        assign g8_dbg_last_araddr   = '0;
        assign g8_dbg_rresp_err     = 1'b0;

        assign g8_cand_dump_id      = '0;
        assign g8_cand_dump_score   = '0;

        assign g8_bridge_score_ff   = '0;
    end
    endgenerate

    wire [31:0] dbg_fsmstate_word;
    wire [31:0] dbg_activity_word;

    wire [3:0]  dbg_sel_w;
    wire [31:0] dbg_cyclecount_word, dbg_checksum_word, dbg_g8progress_word;
    wire [31:0] dbg_g8araddr_lo_word, dbg_g8araddr_hi_word;
    wire [31:0] dbg_rearm_word, dbg_bootstream_word;

    wire [31:0] dbg_stageout_xor_word, dbg_stageout_sample_word, dbg_stageout_count_word;

    wire [4:0]  dbg_cap_stage_sel_w;
    wire        dbg_cap_widx_we;
    wire [4:0]  dbg_cap_widx_wdata;
    wire        dbg_cap_val_re;
    wire [31:0] dbg_cap_widx_word;
    wire [31:0] dbg_cap_val_word;
    wire [31:0] dbg_cap_meta_word;
    wire [31:0] dbg_cap_crc_word;

    wire [2:0]  dbg_res_sel_w;
    wire        dbg_res_clear_w;
    wire [31:0] dbg_res_crc_word;
    wire [31:0] dbg_res_sum_word;
    wire [31:0] dbg_res_count_word;

    wire        kv_res_clear_w;
    wire [31:0] kv_res_status_word =
                {16'h4B56 /* "KV" */, 4'b0, s7a_kv_res_cnt};

    localparam int PRB_MAXCH_FWD = LANES;
    reg  [PRB_MAXCH_FWD*32-1:0]            sout_data;
    reg  [$clog2(PRB_MAXCH_FWD+1)-1:0]     sout_words;
    reg                                    sout_valid;
    reg  [31:0]                            sout_xor_fold;
    wire [31:0]                            sout_word0 = sout_data[31:0];

    wire [3:0]  prb_sel_w;
    wire        prb_sel_we, prb_addr_we, prb_data_re;
    wire [9:0]  prb_addr_wdata;
    wire [31:0] prb_addr_word, prb_data_word, prb_xor_word, prb_sum_word, prb_count_word;

    wire        dbg_gamma_we = gamma_we_hidden | gamma_we_enorm |
                               gamma_we_postat | gamma_we_final;
    dbg_observe #(.LANES(LANES), .A_WIDTH(A_WIDTH), .G8_ADDR_W(BOOT_AXI_ADDR_W)) u_dbg (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start_pulse(start_pulse),
        .top_state(isa_dbg_state),
        .g5_state(3'(g5_state)),
        .stage_sel(stage_sel),
        .g1_done(g1_done),   .g4_done(g4_done),   .g5_done(g5_done),
        .g6_done(g6_done_stage), .g7g_done(g7g_done_stage), .g7u_done(g7u_done_stage),
        .g7p_done(g7p_done), .g9_done(g9_done_stage),
        .g11_done(g11_done), .glm_done(glm_done), .g12_done(g12_done),
        .tree_done(tree_done), .g8_done(g8_done),

        .g1_out_chunk(rms_g1_out_chunk_w), .g1_out_valid(rms_g1_out_valid),
        .array_acc_out(array_acc_out),     .array_done(array_done),
        .g5_attn_out_vec(dbg_g5_attn_vec), .g5_attn_done(dbg_g5_attn_done),
        .silu_out_chunk(silu_out_chunk),   .silu_valid(silu_valid),
        .g11_out_chunk(rms_g11_out_chunk), .g11_out_strobe(g11_out_strobe),
        .tree_topk_scores(tree_topk_scores),
        .g8_activity(g8_dbg_activity),     .g8_wd_tripped(g8_dbg_err),
        .fsmstate_word(dbg_fsmstate_word),
        .activity_word(dbg_activity_word),

        .kernel_busy(kernel_busy), .kernel_done(kernel_done),
        .dbg_sel(dbg_sel_w),
        .emb_we(emb_we),     .emb_wr_data(emb_wr_data32),
        .hfc_we(hfc_we),     .hfc_wr_data(hfc_wr_data32),
        .gamma_we(dbg_gamma_we), .gamma_wr_data(gamma_wr_data32),
        .g8_sel_substate(g8_dbg_sel_substate),
        .g8_cand_filled(g8_dbg_cand_filled),
        .g8_rows_gathered(g8_dbg_rows_gathered),
        .g8_seen_last(g8_dbg_seen_last),
        .g8_in_gather(g8_dbg_in_gather),
        .g8_last_araddr(g8_dbg_last_araddr),
        .g8_rresp_err(g8_dbg_rresp_err),
        .boot_stream_we(dma_we), .boot_stream_data(dma_wr_data),
        .cyclecount_word(dbg_cyclecount_word),
        .checksum_word(dbg_checksum_word),
        .g8progress_word(dbg_g8progress_word),
        .g8araddr_lo_word(dbg_g8araddr_lo_word),
        .g8araddr_hi_word(dbg_g8araddr_hi_word),
        .rearm_word(dbg_rearm_word),
        .bootstream_word(dbg_bootstream_word),

        .sout_chunk_valid(sout_valid),
        .sout_chunk_w0(sout_word0),
        .sout_chunk_xor(sout_xor_fold),
        .sout_chunk_nwords({{(10-$clog2(PRB_MAXCH_FWD+1)){1'b0}}, sout_words}),
        .stageout_xor_word(dbg_stageout_xor_word),
        .stageout_sample_word(dbg_stageout_sample_word),
        .stageout_count_word(dbg_stageout_count_word)
    );

    localparam int PRB_MAXCH = LANES;

    reg  [PRB_MAXCH*32-1:0]            prb_chunk_data;
    reg  [$clog2(PRB_MAXCH+1)-1:0]     prb_chunk_words;
    reg                               prb_chunk_valid;

    function automatic [PRB_MAXCH*32-1:0] pack16x16;
        input [16*16-1:0] b;
        integer k;
        begin
            pack16x16 = '0;
            for (k = 0; k < 8; k = k + 1)
                pack16x16[k*32 +: 32] = { b[(2*k+1)*16 +: 16], b[(2*k)*16 +: 16] };
        end
    endfunction

    function automatic [PRB_MAXCH*32-1:0] pack16x32;
        input [16*32-1:0] b;
        integer k;
        begin
            pack16x32 = '0;
            for (k = 0; k < 16; k = k + 1) pack16x32[k*32 +: 32] = b[k*32 +: 32];
        end
    endfunction

    function automatic [PRB_MAXCH*32-1:0] pack8x32;
        input [8*32-1:0] b;
        integer k;
        begin
            pack8x32 = '0;
            for (k = 0; k < 8; k = k + 1) pack8x32[k*32 +: 32] = b[k*32 +: 32];
        end
    endfunction

    always_comb begin
        prb_chunk_data  = '0;
        prb_chunk_words = '0;
        prb_chunk_valid = 1'b0;
        case (prb_sel_w)
            4'd0: begin
                prb_chunk_data  = pack16x16(rms_g1_out_chunk_w);
                prb_chunk_words = 8;  prb_chunk_valid = rms_g1_out_valid;
            end
            4'd1: begin
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd0);
            end
            4'd2: begin
                prb_chunk_data  = pack16x32(dbg_g5_attn_vec);
                prb_chunk_words = 16; prb_chunk_valid = dbg_g5_attn_done;
            end
            4'd3: begin
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd1);
            end
            4'd4: begin
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd2);
            end
            4'd5: begin
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd3);
            end
            4'd6: begin
                prb_chunk_data  = pack16x16(silu_out_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = silu_valid;
            end
            4'd7: begin
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd4);
            end
            4'd8: begin
                prb_chunk_data  = pack16x16(rms_g11_out_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = g11_out_strobe;
            end
            4'd9: begin
                prb_chunk_data  = pack16x16(glm_wr_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = glm_wr_valid;
            end
            4'd10: begin
                prb_chunk_data  = array_acc_out;
                prb_chunk_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                prb_chunk_valid = array_done & (stage_sel == 4'd5);
            end
            4'd11: begin
                prb_chunk_data  = pack8x32(tree_topk_scores);
                prb_chunk_words = 8;  prb_chunk_valid = tree_done;
            end
            4'd12: begin
                prb_chunk_data  = pack16x16(seam_r1_rd);
                prb_chunk_words = 8;  prb_chunk_valid = array_done & (stage_sel == 4'd0);
            end
            4'd13: begin
                prb_chunk_data  = pack16x16(r3_dbg_out_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = r3_dbg_out_valid;
            end
            4'd14: begin
                prb_chunk_data  = pack16x16(r7_out_chunk);
                prb_chunk_words = 8;  prb_chunk_valid = r7_out_valid;
            end
            4'd15: begin
                prb_chunk_data  = pack8x32(g8_dbg_scores);
                prb_chunk_words = 8;  prb_chunk_valid = g8_dbg_scores_v;
            end
            default: ;
        endcase
    end

    always_comb begin
        sout_data  = '0; sout_words = '0; sout_valid = 1'b0;
        case (dbg_sel_w)
            4'd0:  begin sout_data = pack16x16(rms_g1_out_chunk_w); sout_words = 8;
                         sout_valid = rms_g1_out_valid; end
            4'd1:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd0); end
            4'd2:  begin sout_data = pack16x32(dbg_g5_attn_vec); sout_words = 16;
                         sout_valid = dbg_g5_attn_done; end
            4'd3:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd1); end
            4'd4:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd2); end
            4'd5:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd3); end
            4'd6:  begin sout_data = pack16x16(silu_out_chunk); sout_words = 8;
                         sout_valid = silu_valid; end
            4'd7:  begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd4); end
            4'd8:  begin sout_data = pack16x16(rms_g11_out_chunk); sout_words = 8;
                         sout_valid = g11_out_strobe; end
            4'd9:  begin sout_data = pack16x16(glm_wr_chunk); sout_words = 8;
                         sout_valid = glm_wr_valid; end
            4'd10: begin sout_data = array_acc_out; sout_words = PRB_MAXCH[$clog2(PRB_MAXCH+1)-1:0];
                         sout_valid = array_done & (stage_sel == 4'd5); end
            4'd11: begin sout_data = pack8x32(tree_topk_scores); sout_words = 8;
                         sout_valid = tree_done; end
            4'd13: begin sout_data = pack16x16(r3_dbg_out_chunk); sout_words = 8;
                         sout_valid = r3_dbg_out_valid; end
            4'd14: begin sout_data = pack16x16(r7_out_chunk); sout_words = 8;
                         sout_valid = r7_out_valid; end
            4'd15: begin sout_data = pack8x32(g8_dbg_scores); sout_words = 8;
                         sout_valid = g8_dbg_scores_v; end
            default: ;
        endcase
    end

    integer sxk;
    always_comb begin
        sout_xor_fold = 32'd0;
        for (sxk = 0; sxk < PRB_MAXCH; sxk = sxk + 1)
            if (sxk < sout_words) sout_xor_fold = sout_xor_fold ^ sout_data[sxk*32 +: 32];
    end

    // Runtime-selected datapath probe capture.
    dbg_probe_ram #(.DEPTH(1024), .CAP_WORDS(256), .MAX_CHUNK(PRB_MAXCH)) u_probe (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .probe_sel(prb_sel_w), .sel_we(prb_sel_we),
        .addr_we(prb_addr_we), .addr_wdata(prb_addr_wdata), .data_re(prb_data_re),
        .chunk_valid(prb_chunk_valid), .chunk_words(prb_chunk_words), .chunk_data(prb_chunk_data),
        .probe_addr_word(prb_addr_word), .probe_data_word(prb_data_word),
        .probe_xor_word(prb_xor_word), .probe_sum_word(prb_sum_word),
        .probe_count_word(prb_count_word)
    );

    localparam int DSC_NSTAGE = 19;
    localparam int DSC_NW     = 32;
    localparam int DSC_MAXCH  = LANES;

    reg  [DSC_NSTAGE-1:0]                 dsc_strobe;
    reg  [DSC_NSTAGE*8-1:0]               dsc_words;
    reg  [DSC_NSTAGE*DSC_MAXCH*32-1:0]    dsc_chunk;

    always_comb begin
        dsc_strobe = '0;
        dsc_words  = '0;
        dsc_chunk  = '0;

        dsc_strobe[0]  = rms_g1_out_valid;
        dsc_words[0*8 +: 8] = 8'd8;
        dsc_chunk[0*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(rms_g1_out_chunk_d1);

        dsc_strobe[1]  = cap_q_we;
        dsc_words[1*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[1*DSC_MAXCH*32 +: DSC_MAXCH*32] = g4_tile_slice;

        dsc_strobe[2]  = cap_k_we;
        dsc_words[2*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[2*DSC_MAXCH*32 +: DSC_MAXCH*32] = g4_tile_slice;

        dsc_strobe[3]  = cap_v_we;
        dsc_words[3*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[3*DSC_MAXCH*32 +: DSC_MAXCH*32] = g4_tile_slice;

        dsc_strobe[4]  = dbg_g5_attn_done;
        dsc_words[4*8 +: 8] = 8'd16;
        dsc_chunk[4*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x32(dbg_g5_attn_vec);

        dsc_strobe[5]  = array_done & (stage_sel == 4'd1);
        dsc_words[5*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[5*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;

        dsc_strobe[6]  = array_done & (stage_sel == 4'd2);
        dsc_words[6*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[6*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;

        dsc_strobe[7]  = array_done & (stage_sel == 4'd3);
        dsc_words[7*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[7*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;

        dsc_strobe[8]  = silu_valid;
        dsc_words[8*8 +: 8] = 8'd8;
        dsc_chunk[8*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(silu_out_chunk);

        dsc_strobe[9]  = array_done & (stage_sel == 4'd4);
        dsc_words[9*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[9*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;

        dsc_strobe[10] = g11_out_strobe;
        dsc_words[10*8 +: 8] = 8'd8;
        dsc_chunk[10*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(rms_g11_out_chunk);

        dsc_strobe[11] = glm_wr_valid;
        dsc_words[11*8 +: 8] = 8'd8;
        dsc_chunk[11*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(glm_wr_chunk);

        dsc_strobe[12] = array_done & (stage_sel == 4'd5);
        dsc_words[12*8 +: 8] = 8'(DSC_NW);
        dsc_chunk[12*DSC_MAXCH*32 +: DSC_MAXCH*32] = array_acc_out;

        dsc_strobe[13] = tree_done;
        dsc_words[13*8 +: 8] = 8'd8;
        dsc_chunk[13*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack8x32(tree_topk_scores);

        dsc_strobe[14] = array_done & (stage_sel == 4'd0);
        dsc_words[14*8 +: 8] = 8'd8;
        dsc_chunk[14*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(seam_r1_rd);

        dsc_strobe[15] = r3_dbg_out_valid;
        dsc_words[15*8 +: 8] = 8'd8;
        dsc_chunk[15*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(r3_dbg_out_chunk);

        dsc_strobe[16] = r7_out_valid;
        dsc_words[16*8 +: 8] = 8'd8;
        dsc_chunk[16*DSC_MAXCH*32 +: DSC_MAXCH*32] = pack16x16(r7_out_chunk);

        dsc_strobe[17] = tree_done;
        dsc_words[17*8 +: 8] = 8'd8;
        for (int gi8 = 0; gi8 < 8; gi8 = gi8 + 1)
            dsc_chunk[17*DSC_MAXCH*32 + gi8*32 +: 32] =
                {{(32-TREE_IDX_W){1'b0}}, tree_topk_indices[gi8*TREE_IDX_W +: TREE_IDX_W]};

        dsc_strobe[18] = g8_done;
        dsc_words[18*8 +: 8] = 8'd8;
        for (int gi8 = 0; gi8 < 8; gi8 = gi8 + 1)
            dsc_chunk[18*DSC_MAXCH*32 + gi8*32 +: 32] =
                {{(32-TREE_IDX_W){1'b0}}, ow_topk_indices[gi8*TREE_IDX_W +: TREE_IDX_W]};
    end

    // Per-stage value capture and readback.
    dbg_stage_capture #(
        .NSTAGE(DSC_NSTAGE), .NW(DSC_NW), .MAX_CHUNK(DSC_MAXCH)
    ) u_stage_cap (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .start_pulse(start_pulse),
        .strobe(dsc_strobe), .words(dsc_words), .chunk(dsc_chunk),
        .cap_stage_sel(dbg_cap_stage_sel_w),
        .widx_we(dbg_cap_widx_we), .widx_wdata(dbg_cap_widx_wdata),
        .val_re(dbg_cap_val_re),
        .cap_word_idx_word(dbg_cap_widx_word),
        .cap_val_word(dbg_cap_val_word),
        .cap_meta_word(dbg_cap_meta_word),
        .cap_crc_word(dbg_cap_crc_word)
    );

    reg [31:0] g8_dump_r [0:15];
    reg        g8_dump_resc_latched;
    reg        g8_dump_cand_latched;
    integer gdk;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n || start_pulse) begin
            g8_dump_resc_latched <= 1'b0;
            g8_dump_cand_latched <= 1'b0;
            for (gdk = 0; gdk < 16; gdk = gdk + 1) g8_dump_r[gdk] <= 32'd0;
        end else begin
            if (tree_done && !g8_dump_cand_latched) begin
                g8_dump_cand_latched <= 1'b1;
                for (gdk = 0; gdk < 8; gdk = gdk + 1)
                    g8_dump_r[8+gdk] <= {{(32-TREE_IDX_W){1'b0}},
                                         tree_topk_indices[gdk*TREE_IDX_W +: TREE_IDX_W]};
            end
            if (g8_done && !g8_dump_resc_latched) begin
                g8_dump_resc_latched <= 1'b1;
                for (gdk = 0; gdk < 8; gdk = gdk + 1)
                    g8_dump_r[gdk] <= {{(32-TREE_IDX_W){1'b0}},
                                       ow_topk_indices[gdk*TREE_IDX_W +: TREE_IDX_W]};
            end
        end
    end
    genvar gdg;
    generate
        for (gdg = 0; gdg < 16; gdg = gdg + 1) begin : g_g8dump
            assign g8_dump_flat[gdg*32 +: 32] = g8_dump_r[gdg];
        end
    endgenerate
    assign g8_dump_status = {g8_dump_resc_latched, g8_dump_cand_latched};

    wire [15:0] res_emb_tag   = {{(16-EMB_CW-EMB_WPC_LOG2){1'b0}}, emb_wr_chunk,   emb_wr_word};
    wire [15:0] res_hfc_tag   = {{(16-EMB_CW-EMB_WPC_LOG2){1'b0}}, hfc_wr_chunk,   hfc_wr_word};
    wire [15:0] res_gamma_tag = {{(16-EMB_CW-EMB_WPC_LOG2){1'b0}}, gamma_wr_chunk, gamma_wr_word};

    // Write-side residency checks for loaded buffers.
    dbg_residency #(.ADDR_TAG_W(16), .RD_TAG_W(EMB_CW)) u_residency (
        .clk(ap_clk), .rst_n(ap_rst_n), .res_clear(dbg_res_clear_w),
        .emb_we(emb_we),           .emb_wr_data(emb_wr_data32),     .emb_addr_tag(res_emb_tag),
        .hfc_we(hfc_we),           .hfc_wr_data(hfc_wr_data32),     .hfc_addr_tag(res_hfc_tag),
        .gamma_we(dbg_gamma_we),   .gamma_wr_data(gamma_wr_data32), .gamma_addr_tag(res_gamma_tag),
        .w_we(eff_boot_we),        .w_wr_data(eff_boot_wr_data),
        .sc_we(dma_sc_we),         .sc_wr_data({16'b0, dma_sc_wr_data}),

        .g1_rd_stb(g1_rd_stb),     .g1_rd_idx_tap(g1_rd_idx_tap),   .g1_rd_word_tap(g1_rd_word_tap),
        .g1e_rd_stb(g1e_rd_stb),   .g1e_rd_idx_tap(g1e_rd_idx_tap), .g1e_rd_word_tap(g1e_rd_word_tap),
        .res_sel(dbg_res_sel_w),
        .res_crc_word(dbg_res_crc_word),
        .res_sum_word(dbg_res_sum_word),
        .res_count_word(dbg_res_count_word)
    );

endmodule

`default_nettype wire
