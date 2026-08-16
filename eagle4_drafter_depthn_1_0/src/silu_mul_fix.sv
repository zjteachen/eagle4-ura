// SILU-Q612-FIX (2026-06-06): ported into pdi6_s7c_g8 from silu_mathcheck.
// =====================================================================
// silu_mul_fix.sv  (FIX-SILU experiment copy of silu_mul.sv)
// silu(gate)*up with the two math defects fixed:
//   FIX-1: gate/up inputs widened to Q6.12 (18-bit, range +-32) so the R4
//          seam no longer clips real gate(<=12.5)/up(<=18.0) to +-8 (Q4.12).
//          [was: 16-bit Q4.12 in silu_mul.sv -> 254 elems saturated, 74.5%
//           of the silu error, max|d|=8346 @idx85 where up=17.98->cap 8.0]
//   FIX-2: 512-entry sigmoid LUT (0.03125 buckets) replaces the 64-entry
//          0.25-bucket PWL (the broad ~3.9% rel-err floor).
//   FIX-3: stage-A >>12 now rounds-to-nearest (+half-LSB) like stage B.
// OUT_FRAC=8 (Q8.8 R5 contract) unchanged.
// =====================================================================
`timescale 1ns/1ps
`default_nettype none
(* keep_hierarchy = "yes" *)
module silu_mul_fix #(
    parameter int OUT_FRAC = 8,
    parameter int IN_W     = 18          // FIX-1: Q6.12 input width
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  en,
    input  wire signed [IN_W-1:0] gate_q,    // Q6.12 signed
    input  wire signed [IN_W-1:0] up_q,      // Q6.12 signed
    output reg  signed [15:0]    out_q,       // Q8.8 signed
    output reg                   valid
);
    // sigmoid (combinational), takes Q6.12-widened gate
    wire [15:0] sigmoid_q;
    sigmoid_pwl512_q12 u_sig (.x_q12(gate_q), .sigmoid_q12(sigmoid_q));

    // stage A: silu = gate(Q6.12) * sig(Q4.12) -> Q?.24; >>12 round; sat Q8.12 (20b)
    function automatic signed [19:0] sat_q12_wide;
        input signed [39:0] v;          // widened product
        reg   signed [39:0] s;
        begin
            s = (v + (40'sd1 <<< 11)) >>> 12;   // FIX-3: round-to-nearest
            if (s >  40'sd524287)       sat_q12_wide =  20'sd524287;
            else if (s < -40'sd524288)  sat_q12_wide = -20'sd524288;
            else                        sat_q12_wide = s[19:0];
        end
    endfunction

    localparam int SHR_B = 24 - OUT_FRAC;     // 16
    function automatic signed [15:0] sat_out;
        input signed [47:0] v;
        reg signed [47:0]   r;
        begin
            r = (SHR_B > 0) ? ((v + (48'sd1 <<< (SHR_B-1))) >>> SHR_B) : v;
            if (r > 48'sd32767)        sat_out = 16'sd32767;
            else if (r < -48'sd32768)  sat_out = 16'sh8000;
            else                       sat_out = r[15:0];
        end
    endfunction

    // gate(18b Q6.12) * sigmoid(16b Q4.12) = 34b Q?.24
    (* USE_DSP = "yes" *)
    wire signed [39:0] silu_q24_comb = $signed(gate_q) * $signed({1'b0, sigmoid_q});

    reg signed [19:0]  silu_q_reg;     // Q8.12
    reg signed [IN_W-1:0] up_q_pipe;
    reg                en_pipe;
    always_ff @(posedge clk) begin
        if (!rst_n) begin silu_q_reg<='0; up_q_pipe<='0; en_pipe<=1'b0; end
        else begin
            en_pipe <= en;
            if (en) begin
                silu_q_reg <= sat_q12_wide(silu_q24_comb);
                up_q_pipe  <= up_q;
            end
        end
    end

    // stage B: silu_q(Q8.12,20b) * up(Q6.12,18b) = Q14.24 (38b); requant Q8.8
    (* USE_DSP = "yes" *)
    wire signed [47:0] out_q24_comb = $signed(silu_q_reg) * $signed(up_q_pipe);

    always_ff @(posedge clk) begin
        if (!rst_n) begin out_q<='0; valid<=1'b0; end
        else begin
            valid <= en_pipe;
            if (en_pipe) out_q <= sat_out(out_q24_comb);
        end
    end
endmodule
`default_nettype wire
