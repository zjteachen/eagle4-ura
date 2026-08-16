// =====================================================================
// sigmoid_pwl_q12.sv
// 64-entry PWL approximation of sigmoid(x) in Q4.12 fixed-point.
// =====================================================================
//
// Index = clip( (x_q12 + 32768) >> 10, 0, 63 )
// Each bucket is 0.25 wide in real-valued x; LUT centered at bucket midpoint.
// Range: x in [-8, +8] (Q4.12 max range).
//
// Output is unsigned Q4.12 in [1, 4094] (saturating to ~0 and ~1.0).
//
// Companion to exp_pwl_q12.sv; together with a multiplier this gives
// SiLU(x) = x * sigmoid(x).
// =====================================================================

`timescale 1ns/1ps
`default_nettype none

(* keep_hierarchy = "yes" *)
module sigmoid_pwl_q12 (
    input  wire signed [15:0]  x_q12,
    output reg         [15:0]  sigmoid_q12
);

    logic [5:0] index;
    logic signed [31:0] shifted;

    always_comb begin
        shifted = $signed({{16{x_q12[15]}}, x_q12}) + 32'sd32768;
        if (shifted < 0)
            index = 6'd0;
        else if (shifted >= 32'sd65536)
            index = 6'd63;
        else
            index = 6'(shifted >>> 10);

        case (index)
            6'd 0: sigmoid_q12 = 16'd2;
            6'd 1: sigmoid_q12 = 16'd2;
            6'd 2: sigmoid_q12 = 16'd3;
            6'd 3: sigmoid_q12 = 16'd3;
            6'd 4: sigmoid_q12 = 16'd4;
            6'd 5: sigmoid_q12 = 16'd5;
            6'd 6: sigmoid_q12 = 16'd7;
            6'd 7: sigmoid_q12 = 16'd9;
            6'd 8: sigmoid_q12 = 16'd11;
            6'd 9: sigmoid_q12 = 16'd15;
            6'd10: sigmoid_q12 = 16'd19;
            6'd11: sigmoid_q12 = 16'd24;
            6'd12: sigmoid_q12 = 16'd31;
            6'd13: sigmoid_q12 = 16'd40;
            6'd14: sigmoid_q12 = 16'd51;
            6'd15: sigmoid_q12 = 16'd65;
            6'd16: sigmoid_q12 = 16'd83;
            6'd17: sigmoid_q12 = 16'd106;
            6'd18: sigmoid_q12 = 16'd136;
            6'd19: sigmoid_q12 = 16'd172;
            6'd20: sigmoid_q12 = 16'd219;
            6'd21: sigmoid_q12 = 16'd277;
            6'd22: sigmoid_q12 = 16'd349;
            6'd23: sigmoid_q12 = 16'd437;
            6'd24: sigmoid_q12 = 16'd545;
            6'd25: sigmoid_q12 = 16'd674;
            6'd26: sigmoid_q12 = 16'd827;
            6'd27: sigmoid_q12 = 16'd1004;
            6'd28: sigmoid_q12 = 16'd1205;
            6'd29: sigmoid_q12 = 16'd1428;
            6'd30: sigmoid_q12 = 16'd1668;
            6'd31: sigmoid_q12 = 16'd1920;
            6'd32: sigmoid_q12 = 16'd2176;
            6'd33: sigmoid_q12 = 16'd2428;
            6'd34: sigmoid_q12 = 16'd2668;
            6'd35: sigmoid_q12 = 16'd2891;
            6'd36: sigmoid_q12 = 16'd3092;
            6'd37: sigmoid_q12 = 16'd3269;
            6'd38: sigmoid_q12 = 16'd3422;
            6'd39: sigmoid_q12 = 16'd3551;
            6'd40: sigmoid_q12 = 16'd3659;
            6'd41: sigmoid_q12 = 16'd3747;
            6'd42: sigmoid_q12 = 16'd3819;
            6'd43: sigmoid_q12 = 16'd3877;
            6'd44: sigmoid_q12 = 16'd3924;
            6'd45: sigmoid_q12 = 16'd3960;
            6'd46: sigmoid_q12 = 16'd3990;
            6'd47: sigmoid_q12 = 16'd4013;
            6'd48: sigmoid_q12 = 16'd4031;
            6'd49: sigmoid_q12 = 16'd4045;
            6'd50: sigmoid_q12 = 16'd4056;
            6'd51: sigmoid_q12 = 16'd4065;
            6'd52: sigmoid_q12 = 16'd4072;
            6'd53: sigmoid_q12 = 16'd4077;
            6'd54: sigmoid_q12 = 16'd4081;
            6'd55: sigmoid_q12 = 16'd4085;
            6'd56: sigmoid_q12 = 16'd4087;
            6'd57: sigmoid_q12 = 16'd4089;
            6'd58: sigmoid_q12 = 16'd4091;
            6'd59: sigmoid_q12 = 16'd4092;
            6'd60: sigmoid_q12 = 16'd4093;
            6'd61: sigmoid_q12 = 16'd4093;
            6'd62: sigmoid_q12 = 16'd4094;
            default: sigmoid_q12 = 16'd4094;
        endcase
    end

endmodule

`default_nettype wire
