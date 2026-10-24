// =====================================================================
// gemv_stage_array_uram.sv
// Production GEMV stage array: per-lane URAM weight bank + streaming lane.
// =====================================================================
//
// Differences from gemv_stage_array.sv:
//   - qw_word for each lane comes from a per-lane weight_bank_multistage
//     (URAM-resident), NOT an external port.
//   - The lane's chunk_idx_out drives the weight_bank_multistage's
//     chunk_idx, so each lane reads its weights in lockstep with its FSM.
//   - The shared operation controller supplies the explicit per-tile weight
//     base (cfg_weight_base) and runtime chunk bound (cfg_chunks_per_tile).
//     This dedicated unscaled engine is outside the scaled ISA numeric path,
//     but it consumes the same controller API so there is ONE owner of the
//     tile loop and base settle.
//
// At LANES=128:
//   - URAM banks: 128 × 12 URAMs = 1,536 URAMs = 79.8% V80
//   - LUT (array): 164K = 6.4% V80
//   - DSP58: 128 = 1.2% V80
//   - Critical path: 2.515 ns (closes 300 MHz)
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module gemv_stage_array_uram #(
    parameter int K       = 64,
    parameter int K_PAR   = 16,
    parameter int LANES   = 4,
    parameter int A_WIDTH = 16,
    parameter int N_CHUNKS = K / K_PAR,
    parameter int CW      = (N_CHUNKS <= 1) ? 1 : $clog2(N_CHUNKS + 2),
    parameter int BANK_DEPTH = 96 * 1024  // G12_LM_BANK_DEPTH_FIX (rescue_20260606): 76K < G12 base 81408 -> OOB alias into G9 region
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,

    // Activation broadcast (shared across all lanes)
    input  wire signed [K_PAR*A_WIDTH-1:0]   a_chunk,
    input  wire        [1:0]                 qz_raw,

    // Explicit per-tile weight base (flat 32-bit word address) and runtime
    // per-tile chunk bound, both supplied by the shared operation controller.
    input  wire        [16:0]                cfg_weight_base,
    input  wire        [15:0]                cfg_chunks_per_tile,

    // Boot-time weight load (single shared write port; the upstream
    // boot-mode FSM streams writes to one lane at a time via lane_we_sel)
    input  wire        [15:0]                lane_we_sel,      // which lane to write
    input  wire                              we,
    input  wire        [16:0]                wr_addr,
    input  wire        [31:0]                wr_data,

    // Lane status — chunk_idx_out from lane 0 (all lockstep)
    output wire        [CW-1:0]              chunk_idx_out,
    output wire                              chunk_idx_valid,

    // Per-lane accumulator outputs + done aggregation
    output wire signed [32*LANES-1:0]        acc_out_per_lane,
    output wire                              done
);

    wire [LANES-1:0] lane_done;
    wire [CW-1:0]    lane_chunk_idx      [0:LANES-1];
    wire [CW-1:0]    lane_chunk_idx_next [0:LANES-1];
    wire             lane_chunk_valid    [0:LANES-1];

    assign chunk_idx_out   = lane_chunk_idx[0];
    assign chunk_idx_valid = lane_chunk_valid[0];
    assign done = &lane_done;

    // Lane now has its own S_WARMUP state that gives URAM 1 cycle to
    // pre-read chunk 0 before MAC accumulation begins. No external
    // pipeline alignment needed.

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : g_lane
            // The controller supplies the flat per-tile weight base in word
            // address units; the generate-loop index already selects this
            // lane's physical bank, so no lane term appears in the address.

            // Per-lane weight bank
            wire [31:0] qw_lane;
            wire        lane_we = we && (lane_we_sel == i[15:0]);

            // Bank uses chunk_idx_NEXT (combinational from lane FSM):
            // at cycle T, addr = chunk_idx_n@T → URAM data at T+1 = qw[chunk_idx@T+1].
            // Lane at T+1 consumes qw_word combinationally; alignment exact.
            weight_bank_multistage #(
                .DEPTH(BANK_DEPTH), .DATA_W(32)
            ) u_bank (
                .clk(clk), .rst_n(rst_n),
                .we(lane_we), .wr_addr(wr_addr), .wr_data(wr_data),
                .cfg_base(cfg_weight_base),
                .chunk_idx({{(16-CW){1'b0}}, lane_chunk_idx_next[i]}),
                .rd_data(qw_lane),
                .chunk_idx_b(16'd0), .rd_data_b()
            );

            // Per-lane streaming GEMV
            wire signed [31:0] lane_acc;
            gemv_lane_int_stream_wallace #(
                .K(K), .K_PAR(K_PAR), .A_WIDTH(A_WIDTH)
            ) u_lane (
                .clk(clk), .rst_n(rst_n), .start(start),
                .a_chunk(a_chunk),
                .qw_word(qw_lane),
                .qz_raw(qz_raw),
                .n_chunks_run(cfg_chunks_per_tile[CW-1:0]),
                .chunk_idx_out(lane_chunk_idx[i]),
                .chunk_idx_valid(lane_chunk_valid[i]),
                .chunk_idx_next_out(lane_chunk_idx_next[i]),
                .acc_out(lane_acc),
                .done(lane_done[i])
            );

            assign acc_out_per_lane[i*32 +: 32] = lane_acc;
        end
    endgenerate

endmodule

`default_nettype wire
