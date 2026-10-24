// =====================================================================
// gemv_shared_ctrl.sv — sole owner of shared GEMV operation sequencing
// =====================================================================
// One ISA GEMV instruction (opcode 100, byte 0x80) latches the actual
// special-register bank outputs, validates them, then owns:
//   * the outer tile loop and weight/scale tile-base advance,
//   * the bank base SETTLE cycle before each launch,
//   * the array start pulse,
//   * result-valid qualification and the destination store handshake,
//   * DONE only after the destination adapter reports the final store done,
//   * sticky BUSY/DONE/ERROR status.
//
// TREE is a client of the same engine: tree_start latches the same config
// (destination must be TREE=7) and every tt_gemv_start runs exactly one
// client-specified tile with the same base/activation addressing. TREE's
// own stream FSM provides the backpressure (it requests the next tile only
// after it has drained the previous tile's 128 logits). A standalone G12
// STATE program start no longer exists.
//
// Numeric contract (validated, not pretended programmable):
//   qzero_raw = 1, scale_frac = 13, output_conversion = 0,
//   chunks_per_scale_group in {4, 8}, cpt % cpg == 0.
// The destination-specific conversion/serialize remains in the top-level
// destination adapter; this controller requires an explicit store_ack for
// every ISA tile, so DONE cannot precede the last destination write.
//
// No stage-derived base, stride, or chunk bound is present here.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none

module gemv_shared_ctrl #(
    parameter int LANES      = 128,
    parameter int K_PAR      = 16,
    parameter int K_MAX      = 8192,
    parameter int G9_NTILES  = 24,       // FORREASON ? 48 : 24 (pass from top)
    parameter int WEIGHT_DEPTH = (96 * 1024 * 128) / LANES,
    parameter int SCALE_DEPTH  = (16 * 1024 * 128) / LANES
) (
    input  wire        clk,
    input  wire        rst_n,

    // ---- ISA GEMV launch (opcode 100 / byte 0x80) ----
    input  wire        isa_start,
    input  wire [15:0] cfg_activation_base,
    input  wire [16:0] cfg_weight_base,
    input  wire [15:0] cfg_tile_count,
    input  wire [15:0] cfg_chunks_per_tile,
    input  wire [15:0] cfg_chunks_per_scale_group,
    input  wire [15:0] cfg_output_base,
    input  wire [16:0] cfg_scale_base,
    input  wire [2:0]  cfg_activation_source,
    input  wire [3:0]  cfg_output_destination,
    input  wire [1:0]  cfg_qzero_raw,
    input  wire [4:0]  cfg_scale_frac,
    input  wire [15:0] cfg_output_conversion,
    input  wire [3:0]  weight_base_align,   // per-arm word/row alignment

    // ---- activation-source readiness (first tile of an operation) ----
    // source 3 R1, 4 pre-MLP/R2, 5 R5, 7 attention; others always ready.
    input  wire        act_ready,

    // ---- array handshake ----
    input  wire        array_done,
    output logic       array_start,
    output logic [16:0] arr_weight_base,
    output logic [16:0] arr_scale_base,
    output logic [15:0] arr_chunks_per_tile,
    output logic [15:0] arr_chunks_per_scale_group,

    // ---- destination store handshake ----
    // tile_valid qualifies one tile's result; the destination adapter
    // captures it and returns store_ack when its conversion/serialize and
    // final buffer write are complete. Client (TREE) tiles need no ack.
    output logic        tile_valid,
    output logic [3:0]  tile_dest,
    output logic [15:0] store_idx,      // placement tile inside destination
                                        // (active_tile exposes the engine tile)
    input  wire         store_ack,

    // ---- TREE client ----
    input  wire        client_session_start,  // tree_start: latch config
    input  wire        client_active,         // tiler owns the engine this walk
    input  wire        client_start,          // tt_gemv_start: one tile
    input  wire [15:0] client_n_tiles_cfg,    // legacy scan-length selector
    input  wire        client_session_done,   // tree_done: walk + stream drained
    input  wire [15:0] client_tile_idx,
    output logic       client_error,          // pulse: request/config rejected
    output logic       client_config_ok,      // combinational: TREE config valid
    output logic [15:0] client_scan_tiles,    // accepted scan length for tiler

    // ---- status/config exposure ----
    output logic        busy,
    output logic        done_sticky,
    output logic        error_sticky,
    output logic [15:0] active_tile,
    output logic        op_start,             // accepted ISA operation begins
    output logic [3:0]  op_start_dest,
    output logic        op_done,              // ISA operation finished
    output logic [3:0]  op_done_dest,
    output logic        op_error,             // invalid ISA config (PC may advance)
    output logic        latched_valid,
    output logic [2:0]  latched_activation_source,
    output logic [15:0] latched_activation_base,
    output logic [1:0]  latched_qzero_raw,
    output logic [4:0]  latched_scale_frac,
    output logic [3:0]  latched_destination,
    output logic [15:0] latched_output_conversion,
    output logic [15:0] latched_tile_count
);

    localparam int N_CHUNKS  = K_MAX / K_PAR;
    localparam int LANE_SUBW = LANES / K_PAR;
    // Destination capacities, in output tiles. These mirror the physical
    // buffer geometry in eagle4_drafter.sv; they are capacity limits, not
    // address derivation.
    localparam int G4_NT  = 3072 / LANES;                 // Q, K or V alone
    localparam int G4_TOT = 3 * G4_NT;                    // QKV composite
    localparam int R2_NT  = 192 / LANE_SUBW;              // pre-MLP/R2 buffer
    localparam int R4_NT  = (8192 / K_PAR) / LANE_SUBW;   // R4 gate/up buffer

    // ------------------------------------------------------------------
    // Per-destination and per-source capacities.
    // ------------------------------------------------------------------
    function automatic [15:0] dest_tiles(input [3:0] d);
        case (d)
            4'd0, 4'd1, 4'd2: dest_tiles = 16'(G4_NT);
            4'd3:             dest_tiles = 16'(R2_NT);
            4'd4, 4'd5:       dest_tiles = 16'(R4_NT);
            4'd6:             dest_tiles = 16'(G9_NTILES);
            4'd9:             dest_tiles = 16'(G4_TOT);
            default:          dest_tiles = 16'd0;
        endcase
    endfunction

    function automatic logic dest_is_isa(input [3:0] d);
        case (d)
            4'd0, 4'd1, 4'd2, 4'd3, 4'd4, 4'd5, 4'd6, 4'd9: dest_is_isa = 1'b1;
            default: dest_is_isa = 1'b0;
        endcase
    endfunction

    // Activation-source capacities in 16-element chunks. Source 0 is the
    // external stream port and has no addressable base (base must be 0).
    function automatic [15:0] src_chunks(input [2:0] s);
        case (s)
            3'd1:    src_chunks = 16'd192;   // h_fc
            3'd2:    src_chunks = 16'd192;   // input_emb
            3'd3:    src_chunks = 16'd384;   // R1 = 2*3072/16
            3'd4:    src_chunks = 16'd192;   // pre-MLP / R2
            3'd5:    src_chunks = 16'd512;   // R5
            3'd6:    src_chunks = 16'd16;    // low-rank
            3'd7:    src_chunks = 16'd192;   // attention / G6 input
            default: src_chunks = 16'hFFFF;
        endcase
    endfunction

    // ------------------------------------------------------------------
    // Configuration validation. Wide (32-bit) span arithmetic keeps the
    // bound checks safe before any address is narrowed to 17 bits.
    // ------------------------------------------------------------------
    logic [15:0] rows_v;
    logic [31:0] w_span, s_span, o_span, a_span;
    logic        common_ok, isa_cfg_ok, client_cfg_ok;
    wire [15:0]  client_n_eff =
        (client_n_tiles_cfg == 16'd0) ? cfg_tile_count : client_n_tiles_cfg;

    always_comb begin
        rows_v = (cfg_chunks_per_scale_group == 16'd4) ? (cfg_chunks_per_tile >> 2)
               : (cfg_chunks_per_scale_group == 16'd8) ? (cfg_chunks_per_tile >> 3)
                                                       : 16'd0;
        w_span = 32'(cfg_weight_base) + 32'(cfg_tile_count) * 32'(cfg_chunks_per_tile);
        s_span = 32'(cfg_scale_base)  + 32'(cfg_tile_count) * 32'(rows_v);
        o_span = 32'(cfg_output_base) + 32'(cfg_tile_count);
        a_span = 32'(cfg_activation_base) + 32'(cfg_chunks_per_tile);

        common_ok =
            (cfg_tile_count != 16'd0) &&
            (cfg_chunks_per_tile != 16'd0) &&
            (cfg_chunks_per_tile <= 16'(N_CHUNKS)) &&
            ((cfg_chunks_per_scale_group == 16'd4) ||
             (cfg_chunks_per_scale_group == 16'd8)) &&
            ((cfg_chunks_per_tile & (cfg_chunks_per_scale_group - 16'd1)) == 16'd0) &&
            (cfg_qzero_raw == 2'b01) &&
            (cfg_scale_frac == 5'd13) &&
            (cfg_output_conversion == 16'd0) &&
            ((cfg_weight_base % 17'(weight_base_align)) == 17'd0) &&
            ((cfg_chunks_per_tile % weight_base_align) == 16'd0) &&
            (w_span <= 32'(WEIGHT_DEPTH)) &&
            (s_span <= 32'(SCALE_DEPTH)) &&
            ((cfg_activation_source != 3'd0) ||
             (cfg_activation_base == 16'd0)) &&
            ((cfg_activation_source == 3'd0) ||
             (a_span <= 32'(src_chunks(cfg_activation_source))));

        isa_cfg_ok = common_ok &&
                     dest_is_isa(cfg_output_destination) &&
                     (o_span <= 32'(dest_tiles(cfg_output_destination)));

        // TREE scan length is a selector WITHIN the ISA-configured operation:
        // 0 means the full configured span; any explicit count must fit it.
        client_cfg_ok = common_ok &&
                        (cfg_output_destination == 4'd7) &&
                        (cfg_output_base == 16'd0) &&
                        (client_n_eff <= cfg_tile_count);
    end

    assign client_config_ok = client_cfg_ok;

    // ------------------------------------------------------------------
    // Operation sequencer.
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE   = 3'd0,
        S_SETTLE = 3'd1,
        S_ISSUE  = 3'd2,
        S_WAIT   = 3'd3,
        S_STORE  = 3'd4
    } state_t;

    state_t state;
    logic       isa_mode_q;

    logic [15:0] tile_q, store_idx_q, tiles_q, cpt_q, cpg_q, rows_q;
    logic [16:0] wbase0_q, sbase0_q, wbase_t_q, sbase_t_q;
    logic [3:0]  dest_q;
    logic        client_valid_q;
    logic        client_session_q;

    assign busy         = (state != S_IDLE) || client_session_q;
    assign active_tile  = tile_q;
    assign client_scan_tiles = client_n_eff;
    assign tile_dest    = dest_q;
    assign store_idx    = store_idx_q;
    assign arr_weight_base = wbase_t_q;
    assign arr_scale_base  = sbase_t_q;
    assign arr_chunks_per_tile = cpt_q;
    assign arr_chunks_per_scale_group = cpg_q;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            isa_mode_q       <= 1'b0;
            array_start      <= 1'b0;
            tile_valid       <= 1'b0;
            tile_q           <= '0;
            store_idx_q      <= '0;
            tiles_q          <= '0;
            cpt_q            <= '0;
            cpg_q            <= '0;
            rows_q           <= '0;
            wbase0_q         <= '0;
            sbase0_q         <= '0;
            wbase_t_q        <= '0;
            sbase_t_q        <= '0;
            dest_q           <= '0;
            client_valid_q   <= 1'b0;
            client_session_q <= 1'b0;
            done_sticky      <= 1'b0;
            error_sticky     <= 1'b0;
            op_start         <= 1'b0;
            op_start_dest    <= '0;
            op_done          <= 1'b0;
            op_done_dest     <= '0;
            op_error         <= 1'b0;
            client_error     <= 1'b0;
            latched_valid    <= 1'b0;
            latched_activation_source <= '0;
            latched_activation_base   <= '0;
            latched_qzero_raw <= '0;
            latched_scale_frac <= '0;
            latched_destination <= '0;
            latched_output_conversion <= '0;
            latched_tile_count <= '0;
        end else begin
            // One-cycle strobes default low.
            array_start  <= 1'b0;
            tile_valid   <= 1'b0;
            op_start     <= 1'b0;
            op_done      <= 1'b0;
            op_error     <= 1'b0;
            client_error <= 1'b0;

            case (state)
                // ------------------------------------------------------
                S_IDLE: begin
                    if (isa_start) begin
                        // A new operation clears the sticky status.
                        done_sticky  <= 1'b0;
                        error_sticky <= 1'b0;
                        client_session_q <= 1'b0;

                        // Latch the actual bank outputs at START.
                        tiles_q  <= cfg_tile_count;
                        cpt_q    <= cfg_chunks_per_tile;
                        cpg_q    <= cfg_chunks_per_scale_group;
                        rows_q   <= rows_v;
                        wbase0_q <= cfg_weight_base;
                        sbase0_q <= cfg_scale_base;
                        dest_q   <= cfg_output_destination;
                        latched_activation_source <= cfg_activation_source;
                        latched_activation_base   <= cfg_activation_base;
                        latched_qzero_raw         <= cfg_qzero_raw;
                        latched_scale_frac        <= cfg_scale_frac;
                        latched_destination       <= cfg_output_destination;
                        latched_output_conversion <= cfg_output_conversion;
                        latched_tile_count        <= cfg_tile_count;

                        if (isa_cfg_ok && (state == S_IDLE)) begin
                            isa_mode_q     <= 1'b1;
                            client_valid_q <= 1'b0;
                            latched_valid  <= 1'b1;
                            tile_q         <= '0;
                            store_idx_q    <= cfg_output_base;
                            wbase_t_q      <= cfg_weight_base;
                            sbase_t_q      <= cfg_scale_base;
                            op_start       <= 1'b1;
                            op_start_dest  <= cfg_output_destination;
                            state          <= S_SETTLE;
                        end else begin
                            // Defined fail-safe: no lane start, latch error,
                            // pulse op_error so the ISA PC cannot hang.
                            error_sticky <= 1'b1;
                            op_error     <= 1'b1;
                            latched_valid <= 1'b0;
                            isa_mode_q   <= 1'b0;
                            client_valid_q <= 1'b0;
                            tile_q       <= '0;
                            store_idx_q  <= '0;
                            state        <= S_IDLE;
                        end
                    end else if (client_session_start) begin
                        // TREE session: latch the same configuration fields.
                        done_sticky   <= 1'b0;
                        error_sticky  <= 1'b0;
                        client_session_q <= 1'b0;
                        tiles_q  <= cfg_tile_count;
                        cpt_q    <= cfg_chunks_per_tile;
                        cpg_q    <= cfg_chunks_per_scale_group;
                        rows_q   <= rows_v;
                        wbase0_q <= cfg_weight_base;
                        sbase0_q <= cfg_scale_base;
                        dest_q   <= cfg_output_destination;
                        latched_activation_source <= cfg_activation_source;
                        latched_activation_base   <= cfg_activation_base;
                        latched_qzero_raw         <= cfg_qzero_raw;
                        latched_scale_frac        <= cfg_scale_frac;
                        latched_destination       <= cfg_output_destination;
                        latched_output_conversion <= cfg_output_conversion;
                        latched_tile_count        <= cfg_tile_count;
                        latched_valid             <= client_cfg_ok;
                        client_valid_q            <= client_cfg_ok;
                        client_session_q          <= client_cfg_ok;
                        isa_mode_q                <= 1'b0;
                        if (!client_cfg_ok) begin
                            error_sticky <= 1'b1;
                            client_error <= 1'b1;
                        end
                    end else if (client_start) begin
                        if (client_active && client_valid_q &&
                            (state == S_IDLE) &&
                            (client_tile_idx < tiles_q)) begin
                            isa_mode_q  <= 1'b0;
                            tile_q      <= client_tile_idx;
                            store_idx_q <= client_tile_idx;
                            // Base = operation base + tile * row stride.
                            wbase_t_q <= wbase0_q +
                                         17'(32'(client_tile_idx) * 32'(cpt_q));
                            sbase_t_q <= sbase0_q +
                                         17'(32'(client_tile_idx) * 32'(rows_q));
                            state <= S_SETTLE;
                        end else begin
                            // Real backpressure: never launch outside the
                            // validated config or while another tile is active.
                            error_sticky     <= 1'b1;
                            client_error     <= 1'b1;
                            client_session_q <= 1'b0;
                            client_valid_q   <= 1'b0;
                        end
                    end else if (client_session_done && client_session_q) begin
                        // Explicit session completion: the TREE walk includes
                        // its streaming drain; DONE becomes sticky only here.
                        client_session_q <= 1'b0;
                        client_valid_q   <= 1'b0;
                        done_sticky      <= 1'b1;
                    end
                end
                // ------------------------------------------------------
                S_SETTLE: begin
                    // The bank base registers capture the new tile base at
                    // the end of this cycle. For the first ISA tile wait for
                    // the activation source to report ready (input belt).
                    if (isa_mode_q && (tile_q == 16'd0) && !act_ready)
                        state <= S_SETTLE;
                    else
                        state <= S_ISSUE;
                end
                // ------------------------------------------------------
                S_ISSUE: begin
                    array_start <= 1'b1;
                    state       <= S_WAIT;
                end
                // ------------------------------------------------------
                S_WAIT: begin
                    if (array_done) begin
                        tile_valid <= 1'b1;
                        if (!isa_mode_q) begin
                            // Client tile complete; TREE now owns the result
                            // stream and will request the next tile only after
                            // it has drained this one.
                            state <= S_IDLE;
                        end else begin
                            state <= S_STORE;
                        end
                    end
                end
                // ------------------------------------------------------
                S_STORE: begin
                    if (store_ack) begin
                        if (tile_q == (tiles_q - 16'd1)) begin
                            // DONE only after the destination adapter has
                            // finished the final store/serialize.
                            done_sticky  <= 1'b1;
                            op_done      <= 1'b1;
                            op_done_dest <= dest_q;
                            state        <= S_IDLE;
                        end else begin
                            tile_q      <= tile_q + 16'd1;
                            store_idx_q <= store_idx_q + 16'd1;
                            wbase_t_q   <= wbase_t_q + 17'(cpt_q);
                            sbase_t_q   <= sbase_t_q + 17'(rows_q);
                            state       <= S_SETTLE;
                        end
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
