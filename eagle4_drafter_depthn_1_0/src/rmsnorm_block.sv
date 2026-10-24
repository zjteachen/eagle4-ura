// Two-pass RMSNorm controller: accumulate x*x, compute an integer inverse-RMS
// scale, then reread x with gamma for the apply pass. H must be divisible by K_PAR.
// Completion includes pipeline drain; scale latency depends on saturation.
//
// The FSM registers each engine enable while advancing chunk_cnt. The G1
// full-width caller maps the resulting slots 1..H/K_PAR to chunks 0..N-1.
// RDATA_LAT delays those enables again to match caller-side registered reads.
// Capture data using out_chunk_strobe/index, not the later out_valid signal.

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module rmsnorm_block #(
    parameter int H       = 3072,
    parameter int K_PAR   = 16,
    parameter int A_WIDTH = 16,
    // MULT!=0: scale-engine input = (sumsq*MULT)>>SUMSQ_SHR.
    // Otherwise SUMSQ_RSHIFT selects a right shift (positive) or left shift
    // (negative). Choose constants for H and the activation Q-format;
    // the downstream scale engine itself always divides its input by 16.
    parameter int SUMSQ_RSHIFT = 0,
    parameter longint unsigned SUMSQ_MULT = 0,         // 0 ⇒ legacy pure-shift
    parameter int              SUMSQ_SHR  = 0,
    parameter int              SCALE_SHR  = 12,         // apply x×scale shift = 12-B
    // Stage-A x×scale intermediate width; final gamma result is A_WIDTH.
    parameter int              XS_WIDTH   = A_WIDTH,
    // Delay engine strobes by RDATA_LAT to align with registered caller RAM.
    // RDATA_LAT=0 is the combinational-read path; otherwise the caller presents
    // the address before its corresponding data.
    parameter int              RDATA_LAT  = 0,
    parameter int              SCALE_WIDTH = 16,
    parameter longint unsigned SCALE_EPS_BUCKET = 1
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,
    input  wire signed [K_PAR*A_WIDTH-1:0]   x_chunk,       // streaming input
    input  wire signed [K_PAR*A_WIDTH-1:0]   gamma_chunk,   // matching gamma row
    output wire signed [K_PAR*A_WIDTH-1:0]   out_chunk,     // streaming output
    output wire                              out_valid,
    output logic                             done,
    // FSM request slot and streaming-phase indicator. Account for registered
    // enables and RDATA_LAT when mapping slots to caller buffer addresses.
    output wire  [$clog2((H/K_PAR)+1)-1:0]   x_req_chunk,
    output wire                              x_req_valid,
    // Data-aligned output strobe and logical chunk index, 0..H/K_PAR-1.
    output reg                               out_chunk_strobe,
    output reg   [$clog2((H/K_PAR)+1)-1:0]   out_chunk_idx_o,
    // The delayed enable that actually samples x/gamma in the apply engine.
    output wire                              apply_en_o
);

    localparam int N_CHUNKS = H / K_PAR;
    localparam int CHUNK_W  = $clog2(N_CHUNKS + 1);

    typedef enum logic [2:0] {
        S_IDLE    = 3'd0,
        S_SUMSQ   = 3'd1,   // stream input chunks to sumsq
        S_DRAIN_S = 3'd2,   // drain sumsq and caller read pipeline
        S_SCALE   = 3'd3,   // scale_int iterative sqrt+div
        S_APPLY   = 3'd4,   // streaming x_chunk + gamma into rmsnorm_apply
        S_DRAIN_A = 3'd5,   // drain apply and caller read pipeline
        S_DONE    = 3'd6
    } state_e;

    state_e state;
    logic [CHUNK_W-1:0] chunk_cnt;
    logic [3:0]         drain_cnt;

    // -----------------------------------------------------------------
    // Sub-instance handshakes
    // -----------------------------------------------------------------
    logic              sumsq_clear;
    logic              sumsq_en;
    logic [63:0]       sumsq_out;

    logic              scale_start;
    logic              scale_busy;
    logic              scale_done;
    logic signed [SCALE_WIDTH-1:0] scale_q12;

    logic              apply_en;

    // Delay engine enables and clear to match the caller's read latency.
    logic              sumsq_en_q;     // delayed sumsq enable -> engine
    logic              sumsq_clear_q;  // delayed clear        -> engine
    logic              apply_en_q;     // delayed apply enable -> engine
    generate
        if (RDATA_LAT == 0) begin : g_rdlat0
            assign sumsq_en_q    = sumsq_en;
            assign sumsq_clear_q = sumsq_clear;
            assign apply_en_q    = apply_en;
        end else if (RDATA_LAT == 1) begin : g_rdlat1
            // Separate one-cycle case avoids a negative shift-register slice.
            logic sumsq_en_d1, sumsq_clr_d1, apply_en_d1;
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    sumsq_en_d1  <= 1'b0;
                    sumsq_clr_d1 <= 1'b0;
                    apply_en_d1  <= 1'b0;
                end else begin
                    sumsq_en_d1  <= sumsq_en;
                    sumsq_clr_d1 <= sumsq_clear;
                    apply_en_d1  <= apply_en;
                end
            end
            assign sumsq_en_q    = sumsq_en_d1;
            assign sumsq_clear_q = sumsq_clr_d1;
            assign apply_en_q    = apply_en_d1;
        end else begin : g_rdlat
            // Multi-cycle read-latency pipeline.
            logic [RDATA_LAT-1:0] sumsq_en_sr, sumsq_clr_sr, apply_en_sr;
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    sumsq_en_sr  <= '0;
                    sumsq_clr_sr <= '0;
                    apply_en_sr  <= '0;
                end else begin
                    sumsq_en_sr  <= {sumsq_en_sr [RDATA_LAT-2:0], sumsq_en};
                    sumsq_clr_sr <= {sumsq_clr_sr[RDATA_LAT-2:0], sumsq_clear};
                    apply_en_sr  <= {apply_en_sr [RDATA_LAT-2:0], apply_en};
                end
            end
            assign sumsq_en_q    = sumsq_en_sr [RDATA_LAT-1];
            assign sumsq_clear_q = sumsq_clr_sr[RDATA_LAT-1];
            assign apply_en_q    = apply_en_sr [RDATA_LAT-1];
        end
    endgenerate
    assign apply_en_o = apply_en_q;   // aligned to the engine's input sample

    rmsnorm_sumsq #(.K_PAR(K_PAR), .A_WIDTH(A_WIDTH)) u_sumsq (
        .clk(clk), .rst_n(rst_n),
        .clear(sumsq_clear_q), .en(sumsq_en_q),
        .x_chunk(x_chunk),
        .sumsq_out(sumsq_out)
    );

    // Keep a 128-bit multiply intermediate; select 64 bits after the shift.
    wire [127:0] sumsq_mul   = sumsq_out * SUMSQ_MULT;
    wire [63:0]  sumsq_fix   = sumsq_mul[63+SUMSQ_SHR -: 64];   // (sumsq*MULT)>>SHR
    wire [63:0]  sumsq_legacy = (SUMSQ_RSHIFT >= 0)
                                 ? (sumsq_out >> SUMSQ_RSHIFT)
                                 : (sumsq_out << (-SUMSQ_RSHIFT));
    wire [63:0] sumsq_scaled = (SUMSQ_MULT != 0) ? sumsq_fix : sumsq_legacy;

    rmsnorm_scale_int #(.SCALE_WIDTH(SCALE_WIDTH),
                        .EPS_BUCKET(SCALE_EPS_BUCKET)) u_scale (
        .clk(clk), .resetn(rst_n),
        .start(scale_start),
        .sumsq(sumsq_scaled),
        .busy(scale_busy),
        .done(scale_done),
        .scale_q12(scale_q12)
    );

    // Request address precedes the delayed engine sampling enable.
    assign x_req_chunk = chunk_cnt;
    assign x_req_valid = (state == S_SUMSQ) || (state == S_APPLY);

    rmsnorm_apply #(.K_PAR(K_PAR), .A_WIDTH(A_WIDTH), .SCALE_SHR(SCALE_SHR),
                    .XS_WIDTH(XS_WIDTH), .SCALE_WIDTH(SCALE_WIDTH)) u_apply (
        .clk(clk), .rst_n(rst_n),
        .en(apply_en_q),               // RDATA_LAT-delayed: samples x_chunk when the registered read is ready
        .scale_q12(scale_q12),
        .x_chunk(x_chunk),
        .gamma_chunk(gamma_chunk),
        .out_chunk(out_chunk),
        .valid(out_valid)
    );

    // -----------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            chunk_cnt    <= '0;
            drain_cnt    <= '0;
            sumsq_clear  <= 1'b0;
            sumsq_en     <= 1'b0;
            scale_start  <= 1'b0;
            apply_en     <= 1'b0;
            done         <= 1'b0;
        end else begin
            // defaults
            sumsq_clear <= 1'b0;
            sumsq_en    <= 1'b0;
            scale_start <= 1'b0;
            apply_en    <= 1'b0;
            done        <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        sumsq_clear <= 1'b1;
                        chunk_cnt   <= '0;
                        state       <= S_SUMSQ;
                    end
                end
                S_SUMSQ: begin
                    sumsq_en <= 1'b1;
                    chunk_cnt <= chunk_cnt + 1'b1;
                    if (chunk_cnt == N_CHUNKS[CHUNK_W-1:0] - 1'b1) begin
                        // +RDATA_LAT: the delayed sumsq_en_q tail lands RDATA_LAT
                        // cycles later than the FSM raises sumsq_en, so the last
                        // accumulate completes RDATA_LAT cycles later. Extend the
                        // pipeline drain so sumsq_out is settled before scale_start.
                        drain_cnt <= 4'd3 + RDATA_LAT[3:0];
                        state     <= S_DRAIN_S;
                    end
                end
                S_DRAIN_S: begin
                    if (drain_cnt == 0) begin
                        scale_start <= 1'b1;
                        state       <= S_SCALE;
                    end else begin
                        drain_cnt <= drain_cnt - 1'b1;
                    end
                end
                S_SCALE: begin
                    if (scale_done) begin
                        chunk_cnt <= '0;
                        state     <= S_APPLY;
                    end
                end
                S_APPLY: begin
                    apply_en <= 1'b1;
                    chunk_cnt <= chunk_cnt + 1'b1;
                    if (chunk_cnt == N_CHUNKS[CHUNK_W-1:0] - 1'b1) begin
                        // +RDATA_LAT: matches the delayed apply_en_q tail so the
                        // final out_chunk lands (and its strobe fires) before done.
                        drain_cnt <= 4'd3 + RDATA_LAT[3:0];
                        state     <= S_DRAIN_A;
                    end
                end
                S_DRAIN_A: begin
                    if (drain_cnt == 0) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        drain_cnt <= drain_cnt - 1'b1;
                    end
                end
                S_DONE: begin
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // Delay the engine's apply enable and logical index by two registers.
    // Count logical chunks separately: the FSM slot already advanced when
    // it asserted the registered enable and may now be further ahead.
    localparam int AP_LAT = 2;   // rmsnorm_apply en -> out_chunk latency
    logic [CHUNK_W-1:0] ap_logical;       // logical applied-chunk counter (0..N-1)
    logic               ap_en_d1;         // apply_en delayed AP_LAT-1 = 1
    logic [CHUNK_W-1:0] ap_idx_d1;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ap_logical <= '0;
            ap_en_d1 <= 1'b0;  ap_idx_d1 <= '0;
            out_chunk_strobe <= 1'b0;  out_chunk_idx_o <= '0;
        end else begin
            // Key output timing to the delayed engine enable, not the FSM phase.
            if (start)            ap_logical <= '0;       // re-arm per RMSNorm
            else if (apply_en_q)  ap_logical <= ap_logical + 1'b1;
            // stage 1 (T+1): capture the apply request fired THIS cycle (its
            // logical index = ap_logical BEFORE this cycle's increment).
            ap_en_d1  <= apply_en_q;
            ap_idx_d1 <= ap_logical;
            // stage 2 / output (T+2): aligns with out_chunk on the bus NOW
            out_chunk_strobe <= ap_en_d1;
            out_chunk_idx_o  <= ap_idx_d1;
        end
    end

endmodule

`default_nettype wire
