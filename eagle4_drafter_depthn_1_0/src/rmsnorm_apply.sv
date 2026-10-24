// Two registered multiply stages per lane:
//   A: shift x*scale by SCALE_SHR into an XS_WIDTH-bit intermediate.
//   B: shift intermediate*gamma by 12 and saturate to INT16.
// Gamma has 12 fractional bits; the scale convention is set by the caller.
// out_chunk changes after two register stages; valid is en delayed by three
// registers, one cycle later than the corresponding data. For aligned output
// capture, use rmsnorm_block.out_chunk_strobe instead.

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module rmsnorm_apply #(
    parameter int K_PAR     = 16,
    parameter int A_WIDTH   = 16,
    // Fractional bits removed from x*scale; G1 uses 10.
    parameter int SCALE_SHR = 12,
    // Width of stage-A x_scaled; output remains A_WIDTH after gamma.
    parameter int XS_WIDTH = A_WIDTH,
    parameter int SCALE_WIDTH = 16
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              en,
    input  wire signed [SCALE_WIDTH-1:0]     scale_q12,       // from rmsnorm_scale_int
    input  wire signed [K_PAR*A_WIDTH-1:0]   x_chunk,
    input  wire signed [K_PAR*A_WIDTH-1:0]   gamma_chunk,
    output logic signed [K_PAR*A_WIDTH-1:0]  out_chunk,
    output logic                             valid
);
    // Product widths: PS_WIDTH for x×scale and PB_WIDTH for x_scaled×gamma.
    localparam int PS_WIDTH = A_WIDTH + SCALE_WIDTH;
    localparam int PB_WIDTH = XS_WIDTH + A_WIDTH;

    // Gamma multiply: arithmetic truncation by 12 bits, then INT16 saturation.
    // No rounding bias is added.
    function automatic signed [A_WIDTH-1:0] sat_q12;
        input signed [PB_WIDTH-1:0] v;
        begin
            // The shifted A_WIDTH result is valid only when all guard bits match sign.
            if (v[PB_WIDTH-1 : A_WIDTH+12-1] != {(PB_WIDTH-(A_WIDTH+12-1)){v[PB_WIDTH-1]}})
                sat_q12 = v[PB_WIDTH-1] ? 16'sh8000 : 16'sd32767;
            else
                sat_q12 = v[A_WIDTH+12-1 : 12];
        end
    endfunction

    // Stage-A overflow guard excludes the retained result's sign bit.
    // Values just outside the signed XS_WIDTH range can wrap rather than clamp.
    // Preserve this current behavior until an explicit arithmetic fix.
    localparam signed [XS_WIDTH-1:0] XS_MAX =  (1 <<< (XS_WIDTH-1)) - 1;
    localparam signed [XS_WIDTH-1:0] XS_MIN = -(1 <<< (XS_WIDTH-1));
    function automatic signed [XS_WIDTH-1:0] sat_scale;
        input signed [PS_WIDTH-1:0] v;
        begin
            if (v[PS_WIDTH-1:SCALE_SHR+XS_WIDTH] !=
                {(PS_WIDTH-(SCALE_SHR+XS_WIDTH)){v[PS_WIDTH-1]}})
                sat_scale = v[PS_WIDTH-1] ? XS_MIN : XS_MAX;
            else
                sat_scale = v[SCALE_SHR+XS_WIDTH-1 -: XS_WIDTH];
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (SCALE_WIDTH < 16)
            $error("rmsnorm_apply SCALE_WIDTH=%0d must be >=16", SCALE_WIDTH);
        if (XS_WIDTH < A_WIDTH)
            $error("rmsnorm_apply XS_WIDTH=%0d must be >=A_WIDTH=%0d",
                   XS_WIDTH, A_WIDTH);
        if (PS_WIDTH <= SCALE_SHR + XS_WIDTH)
            $error("rmsnorm_apply invalid guard width PS=%0d SHR=%0d XS=%0d",
                   PS_WIDTH, SCALE_SHR, XS_WIDTH);
    end
`endif

    // Internal unpacked array — distinct drivers per lane (multi-driver-safe)
    logic signed [XS_WIDTH-1:0] x_scaled_arr [0:K_PAR-1];
    logic signed [A_WIDTH-1:0] out_arr      [0:K_PAR-1];
    logic                      en_a, en_b;
    logic signed [A_WIDTH-1:0] gamma_d1     [0:K_PAR-1];

    genvar i;
    generate
        for (i = 0; i < K_PAR; i = i + 1) begin : g_lane
            wire signed [A_WIDTH-1:0] x_i     = x_chunk    [i*A_WIDTH +: A_WIDTH];
            wire signed [A_WIDTH-1:0] gamma_i = gamma_chunk[i*A_WIDTH +: A_WIDTH];

            // Stage A
            (* USE_DSP = "yes" *)
            wire signed [PS_WIDTH-1:0] x_scale =
                $signed(x_i) * $signed(scale_q12);

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    x_scaled_arr[i] <= '0;
                    gamma_d1[i]     <= '0;
                end else if (en) begin
                    x_scaled_arr[i] <= sat_scale(x_scale);
                    gamma_d1[i]     <= gamma_i;
                end
            end

            // Stage B
            (* USE_DSP = "yes" *)
            wire signed [PB_WIDTH-1:0] x_gamma = $signed(x_scaled_arr[i]) * $signed(gamma_d1[i]);

            always_ff @(posedge clk) begin
                if (!rst_n) out_arr[i] <= '0;
                else if (en_a) out_arr[i] <= sat_q12(x_gamma);
            end
        end
    endgenerate

    // valid pipeline
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            en_a <= 1'b0; en_b <= 1'b0; valid <= 1'b0;
        end else begin
            en_a  <= en;
            en_b  <= en_a;
            valid <= en_b;
        end
    end

    // Single combinational concat array → packed output
    always_comb begin
        for (int k = 0; k < K_PAR; k++)
            out_chunk[k*A_WIDTH +: A_WIDTH] = out_arr[k];
    end

endmodule

`default_nettype wire
