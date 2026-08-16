// =====================================================================
// g8_fp16_pkg.sv — synthesizable IEEE-754 fp16-multiply + fp32-add functions.
//
// Fork: verilog/forks/pdi6_g8  (NEW standalone primitive; single source of
//        truth for the G8.3 dot arithmetic).  Imported by g8_dot_fp16.sv (the
//        streaming pipe) and by tb_g8_dot_fp16.sv (the per-op unit tests), so
//        the unit-tested functions ARE the ones the pipe synthesizes.
//
// CONTRACT (bit-exact vs ref/g8_dot_fp16_ref.py):
//   fp16_mul : IEEE binary16 multiply, exact product rounded to nearest-even.
//              == numpy (fp16*fp16).
//   fp32_add : IEEE binary32 add, round-to-nearest-even.
//              == numpy fp32 + fp32.
//   fp16_to_fp32 : exact widen.
//
// All combinational, no `real`, fully synthesizable.  The 11x11 mantissa
// multiply maps to a DSP58 (INT mode); the rest is LUT logic.
//
// (s7c_g8 SPLICE fork: added `timescale so the package mixes cleanly with the
//  top-level eagle4_drafter, which has `timescale 1ns/1ps — without it xelab
//  data-flow analysis errors XSIM 43-4099 "doesn't have a timescale but at
//  least one module in design has a timescale". This is a fork-local copy.)
// =====================================================================
`timescale 1ns/1ps
`ifndef G8_FP16_PKG_SV
`define G8_FP16_PKG_SV

package g8_fp16_pkg;

    // ----------------------------------------------------------------
    // fp16 multiply : a * b -> RNE fp16
    // ----------------------------------------------------------------
    function automatic logic [15:0] fp16_mul(input logic [15:0] x, input logic [15:0] y);
        logic        sx, sy, sr;
        logic [4:0]  ex, ey;
        logic [9:0]  mx, my;
        logic [10:0] fx, fy;
        logic signed [7:0] e_eff_x, e_eff_y;
        logic [21:0] prod;
        logic signed [9:0]  e_sum;
        logic [21:0] shifted;
        logic signed [9:0]  e_norm;
        logic [9:0]  frac10;
        logic        guard, round_bit, sticky, round_up;
        logic [10:0] frac_rnd;
        logic signed [9:0]  e_final;
        logic [4:0]  e_out;
        logic [9:0]  m_out;
        int          i, lead;
        begin
            sx = x[15]; ex = x[14:10]; mx = x[9:0];
            sy = y[15]; ey = y[14:10]; my = y[9:0];
            sr = sx ^ sy;

            if ((ex == 5'd31 && mx != 0) || (ey == 5'd31 && my != 0)) begin
                fp16_mul = {1'b0, 5'd31, 10'h200};                  // NaN in -> qNaN
            end else if (ex == 5'd31 || ey == 5'd31) begin
                if ((ex == 5'd31 && ey == 5'd0 && my == 0) ||
                    (ey == 5'd31 && ex == 5'd0 && mx == 0))
                     fp16_mul = {1'b0, 5'd31, 10'h200};             // inf*0 -> NaN
                else fp16_mul = {sr, 5'd31, 10'h000};               // signed inf
            end else if ((ex == 5'd0 && mx == 0) || (ey == 5'd0 && my == 0)) begin
                fp16_mul = {sr, 5'd0, 10'd0};                       // signed zero
            end else begin
                if (ex == 5'd0) begin fx = {1'b0, mx}; e_eff_x = -8'sd14; end
                else            begin fx = {1'b1, mx}; e_eff_x = $signed({3'b000, ex}) - 8'sd15; end
                if (ey == 5'd0) begin fy = {1'b0, my}; e_eff_y = -8'sd14; end
                else            begin fy = {1'b1, my}; e_eff_y = $signed({3'b000, ey}) - 8'sd15; end

                prod  = fx * fy;
                e_sum = e_eff_x + e_eff_y;

                if (prod == 0) begin
                    fp16_mul = {sr, 5'd0, 10'd0};
                end else begin
                    lead = 0;
                    for (i = 21; i >= 0; i = i - 1)
                        if (prod[i] && lead == 0) lead = i;
                    shifted = prod << (21 - lead);                  // MSB(leading 1) -> bit21
                    // value = shifted/2^21 * 2^e_norm, with shifted[21]=1 => 1.f.
                    e_norm  = e_sum + (lead - 20);

                    if (e_norm <= -15) begin
                        // ---- SUBNORMAL / underflow result ----
                        // subnormal 10-bit mant M = round( shifted * 2^(e_norm+3) ).
                        // rsh = -(e_norm+3) right shifts of the FULL 22-bit product,
                        // so guard/round/sticky keep all the product's low bits
                        // (no precision loss from a pre-round).
                        int          rsh;
                        logic [10:0] sub_m;          // 10-bit mant + carry room
                        logic        gg, rr, ss;
                        rsh = -(e_norm + 3);          // >= 12
                        if (rsh > 22) begin
                            fp16_mul = {sr, 5'd0, 10'd0};           // flush to zero
                        end else begin
                            sub_m = 11'(shifted >> rsh);
                            gg = (rsh >= 1) ? shifted[rsh-1] : 1'b0;
                            rr = (rsh >= 2) ? shifted[rsh-2] : 1'b0;
                            ss = (rsh >= 3) ? (|(shifted & (({22{1'b1}} >> (22-(rsh-2)))))) : 1'b0;
                            if (gg & (rr | ss | sub_m[0])) sub_m = sub_m + 11'd1;
                            if (sub_m[10]) fp16_mul = {sr, 5'd1, sub_m[9:0]}; // rounded up to smallest normal
                            else           fp16_mul = {sr, 5'd0, sub_m[9:0]};
                        end
                    end else begin
                        // ---- NORMAL result ----
                        frac10    = shifted[20:11];
                        guard     = shifted[10];
                        round_bit = shifted[9];
                        sticky    = |shifted[8:0];
                        round_up  = guard & (round_bit | sticky | frac10[0]);
                        frac_rnd  = {1'b0, frac10} + (round_up ? 11'd1 : 11'd0);
                        e_final   = e_norm;
                        // rounding carry 1.11..1 + ulp -> 10.00..0: frac_rnd[9:0]
                        // is already 0 (max frac_rnd = 1024), so just bump exp.
                        if (frac_rnd[10]) e_final = e_final + 1;

                        if (e_final + 15 >= 31) begin
                            fp16_mul = {sr, 5'd31, 10'd0};          // overflow inf
                        end else if (e_final + 15 <= 0) begin
                            // round-up pushed a near-min-normal into subnormal exp;
                            // its mantissa is 0 with exp at the boundary => smallest
                            // normal (handled by the e_norm<=-15 path normally; here
                            // e_final dipped only via the carry, mant=0).
                            fp16_mul = {sr, 5'd0, frac_rnd[9:0]};
                        end else begin
                            e_out = 5'(e_final + 15);
                            m_out = frac_rnd[9:0];
                            fp16_mul = {sr, e_out, m_out};
                        end
                    end
                end
            end
        end
    endfunction

    // ----------------------------------------------------------------
    // fp16 multiply SPLIT into two pipeline stages (the DSP58 "MREG" lever).
    //
    //   fp16_mul_b( fp16_mul_a(x,y) ) == fp16_mul(x,y)  EXACTLY (same code,
    //   cut right AFTER the 11x11 mantissa multiply prod = fx*fy / e_sum, i.e.
    //   at the DSP58 product register).  Stage A is the DSP multiply + the
    //   special-case (NaN/inf/zero) decode; stage B is the leading-one
    //   normalize + round-to-nearest-even.  Used by g8_dot_fp16 with MREG=1
    //   to register the raw DSP product BEFORE the normalize/round LUT chain,
    //   so neither the multiply nor the round is on the same combinational
    //   path (the §10 mul->s0_prod16_reg critical path is cut at the product).
    //
    //   Intermediate IR (40 bits):
    //     [39]    is_special  (NaN/inf/zero result fully decided in stage A)
    //     [38:23] spec_res     (the 16-bit fp16 result for the special case)
    //     [22]    sr           (result sign for the normal path)
    //     [21:12] e_sum (signed 10b) = e_eff_x + e_eff_y
    //     [11:0]  prod_hi      (NOT used — prod is 22b, packed below)
    //   The 22-bit raw product does not fit beside e_sum+sr+spec in <=40b, so
    //   the IR is 50 bits: [49] special, [48:33] spec_res, [32] sr,
    //   [31:22] e_sum(10b signed), [21:0] prod(22b).
    // ----------------------------------------------------------------
    function automatic logic [49:0] fp16_mul_a(input logic [15:0] x, input logic [15:0] y);
        logic        sx, sy, sr;
        logic [4:0]  ex, ey;
        logic [9:0]  mx, my;
        logic [10:0] fx, fy;
        logic signed [7:0] e_eff_x, e_eff_y;
        logic [21:0] prod;
        logic signed [9:0]  e_sum;
        begin
            sx = x[15]; ex = x[14:10]; mx = x[9:0];
            sy = y[15]; ey = y[14:10]; my = y[9:0];
            sr = sx ^ sy;

            if ((ex == 5'd31 && mx != 0) || (ey == 5'd31 && my != 0)) begin
                fp16_mul_a = {1'b1, {1'b0, 5'd31, 10'h200}, 1'b0, 10'd0, 22'd0};      // NaN -> qNaN
            end else if (ex == 5'd31 || ey == 5'd31) begin
                if ((ex == 5'd31 && ey == 5'd0 && my == 0) ||
                    (ey == 5'd31 && ex == 5'd0 && mx == 0))
                     fp16_mul_a = {1'b1, {1'b0, 5'd31, 10'h200}, 1'b0, 10'd0, 22'd0}; // inf*0 -> NaN
                else fp16_mul_a = {1'b1, {sr, 5'd31, 10'h000}, 1'b0, 10'd0, 22'd0};   // signed inf
            end else if ((ex == 5'd0 && mx == 0) || (ey == 5'd0 && my == 0)) begin
                fp16_mul_a = {1'b1, {sr, 5'd0, 10'd0}, 1'b0, 10'd0, 22'd0};           // signed zero
            end else begin
                if (ex == 5'd0) begin fx = {1'b0, mx}; e_eff_x = -8'sd14; end
                else            begin fx = {1'b1, mx}; e_eff_x = $signed({3'b000, ex}) - 8'sd15; end
                if (ey == 5'd0) begin fy = {1'b0, my}; e_eff_y = -8'sd14; end
                else            begin fy = {1'b1, my}; e_eff_y = $signed({3'b000, ey}) - 8'sd15; end

                prod  = fx * fy;
                e_sum = e_eff_x + e_eff_y;
                // not-special: carry sr, e_sum, prod to stage B (prod==0 handled there)
                fp16_mul_a = {1'b0, 16'd0, sr, e_sum, prod};
            end
        end
    endfunction

    function automatic logic [15:0] fp16_mul_b(input logic [49:0] ir);
        logic        is_special, sr;
        logic [15:0] spec_res;
        logic signed [9:0]  e_sum;
        logic [21:0] prod;
        logic [21:0] shifted;
        logic signed [9:0]  e_norm;
        logic [9:0]  frac10;
        logic        guard, round_bit, sticky, round_up;
        logic [10:0] frac_rnd;
        logic signed [9:0]  e_final;
        logic [4:0]  e_out;
        logic [9:0]  m_out;
        int          i, lead;
        begin
            is_special = ir[49];
            spec_res   = ir[48:33];
            sr         = ir[32];
            e_sum      = $signed(ir[31:22]);
            prod       = ir[21:0];

            if (is_special) begin
                fp16_mul_b = spec_res;
            end else if (prod == 0) begin
                fp16_mul_b = {sr, 5'd0, 10'd0};
            end else begin
                lead = 0;
                for (i = 21; i >= 0; i = i - 1)
                    if (prod[i] && lead == 0) lead = i;
                shifted = prod << (21 - lead);
                e_norm  = e_sum + (lead - 20);

                if (e_norm <= -15) begin
                    int          rsh;
                    logic [10:0] sub_m;
                    logic        gg, rr, ss;
                    rsh = -(e_norm + 3);
                    if (rsh > 22) begin
                        fp16_mul_b = {sr, 5'd0, 10'd0};
                    end else begin
                        sub_m = 11'(shifted >> rsh);
                        gg = (rsh >= 1) ? shifted[rsh-1] : 1'b0;
                        rr = (rsh >= 2) ? shifted[rsh-2] : 1'b0;
                        ss = (rsh >= 3) ? (|(shifted & (({22{1'b1}} >> (22-(rsh-2)))))) : 1'b0;
                        if (gg & (rr | ss | sub_m[0])) sub_m = sub_m + 11'd1;
                        if (sub_m[10]) fp16_mul_b = {sr, 5'd1, sub_m[9:0]};
                        else           fp16_mul_b = {sr, 5'd0, sub_m[9:0]};
                    end
                end else begin
                    frac10    = shifted[20:11];
                    guard     = shifted[10];
                    round_bit = shifted[9];
                    sticky    = |shifted[8:0];
                    round_up  = guard & (round_bit | sticky | frac10[0]);
                    frac_rnd  = {1'b0, frac10} + (round_up ? 11'd1 : 11'd0);
                    e_final   = e_norm;
                    if (frac_rnd[10]) e_final = e_final + 1;

                    if (e_final + 15 >= 31) begin
                        fp16_mul_b = {sr, 5'd31, 10'd0};
                    end else if (e_final + 15 <= 0) begin
                        fp16_mul_b = {sr, 5'd0, frac_rnd[9:0]};
                    end else begin
                        e_out = 5'(e_final + 15);
                        m_out = frac_rnd[9:0];
                        fp16_mul_b = {sr, e_out, m_out};
                    end
                end
            end
        end
    endfunction

    // ----------------------------------------------------------------
    // fp16_mul_b SPLIT into two pipeline stages (the MROUND lever).
    //
    //   fp16_mul_b2( fp16_mul_b1(ir) ) == fp16_mul_b(ir)  EXACTLY (same code,
    //   cut at the leading-one-normalize / round-and-pack boundary).  Stage B1
    //   does the leading-zero-detect priority encode + barrel-shift normalize +
    //   exponent (e_norm) — the LUT2-6 / LOOKAHEAD8 mux-tree half of the §2
    //   critical cone.  Stage B2 does guard/round/sticky, the 11-bit round-add
    //   carry chain, exponent fixup, and the overflow/underflow/subnormal clamp
    //   — the LUTCY round-add carry half.  Used by g8_dot_fp16 with MROUND=1 to
    //   register the normalized `shifted`/`e_norm` BETWEEN the two halves, so
    //   neither the LZD+shift nor the round+pack is on the same combinational
    //   path (the §2 DSP58 -> fp16_mul_b -> s0_prod16_reg 32-LL critical cone is
    //   cut at its natural midpoint).
    //
    //   Intermediate IR B1 (50 bits):
    //     [49]    is_special  (NaN/inf/zero/prod==0 fully decided in stage B1)
    //     [48:33] spec_res     (the 16-bit fp16 result for the special case)
    //     [32]    sr           (result sign for the normal path)
    //     [31:22] e_norm (signed 10b) = e_sum + (lead - 20)
    //     [21:0]  shifted (22b) = prod << (21 - lead)  (leading 1 at bit 21)
    // ----------------------------------------------------------------
    function automatic logic [49:0] fp16_mul_b1(input logic [49:0] ir);
        logic        is_special, sr;
        logic [15:0] spec_res;
        logic signed [9:0]  e_sum;
        logic [21:0] prod;
        logic [21:0] shifted;
        logic signed [9:0]  e_norm;
        int          i, lead;
        begin
            is_special = ir[49];
            spec_res   = ir[48:33];
            sr         = ir[32];
            e_sum      = $signed(ir[31:22]);
            prod       = ir[21:0];

            if (is_special) begin
                // pass the special result straight through B1's IR.
                fp16_mul_b1 = {1'b1, spec_res, 1'b0, 10'd0, 22'd0};
            end else if (prod == 0) begin
                // prod==0 resolves to signed zero here -> make it a special.
                fp16_mul_b1 = {1'b1, {sr, 5'd0, 10'd0}, 1'b0, 10'd0, 22'd0};
            end else begin
                lead = 0;
                for (i = 21; i >= 0; i = i - 1)
                    if (prod[i] && lead == 0) lead = i;
                shifted = prod << (21 - lead);
                e_norm  = e_sum + (lead - 20);
                fp16_mul_b1 = {1'b0, 16'd0, sr, e_norm, shifted};
            end
        end
    endfunction

    function automatic logic [15:0] fp16_mul_b2(input logic [49:0] ir);
        logic        is_special, sr;
        logic [15:0] spec_res;
        logic signed [9:0]  e_norm;
        logic [21:0] shifted;
        logic [9:0]  frac10;
        logic        guard, round_bit, sticky, round_up;
        logic [10:0] frac_rnd;
        logic signed [9:0]  e_final;
        logic [4:0]  e_out;
        logic [9:0]  m_out;
        begin
            is_special = ir[49];
            spec_res   = ir[48:33];
            sr         = ir[32];
            e_norm     = $signed(ir[31:22]);
            shifted    = ir[21:0];

            if (is_special) begin
                fp16_mul_b2 = spec_res;
            end else if (e_norm <= -15) begin
                int          rsh;
                logic [10:0] sub_m;
                logic        gg, rr, ss;
                rsh = -(e_norm + 3);
                if (rsh > 22) begin
                    fp16_mul_b2 = {sr, 5'd0, 10'd0};
                end else begin
                    sub_m = 11'(shifted >> rsh);
                    gg = (rsh >= 1) ? shifted[rsh-1] : 1'b0;
                    rr = (rsh >= 2) ? shifted[rsh-2] : 1'b0;
                    ss = (rsh >= 3) ? (|(shifted & (({22{1'b1}} >> (22-(rsh-2)))))) : 1'b0;
                    if (gg & (rr | ss | sub_m[0])) sub_m = sub_m + 11'd1;
                    if (sub_m[10]) fp16_mul_b2 = {sr, 5'd1, sub_m[9:0]};
                    else           fp16_mul_b2 = {sr, 5'd0, sub_m[9:0]};
                end
            end else begin
                frac10    = shifted[20:11];
                guard     = shifted[10];
                round_bit = shifted[9];
                sticky    = |shifted[8:0];
                round_up  = guard & (round_bit | sticky | frac10[0]);
                frac_rnd  = {1'b0, frac10} + (round_up ? 11'd1 : 11'd0);
                e_final   = e_norm;
                if (frac_rnd[10]) e_final = e_final + 1;

                if (e_final + 15 >= 31) begin
                    fp16_mul_b2 = {sr, 5'd31, 10'd0};
                end else if (e_final + 15 <= 0) begin
                    fp16_mul_b2 = {sr, 5'd0, frac_rnd[9:0]};
                end else begin
                    e_out = 5'(e_final + 15);
                    m_out = frac_rnd[9:0];
                    fp16_mul_b2 = {sr, e_out, m_out};
                end
            end
        end
    endfunction

    // ----------------------------------------------------------------
    // fp16 -> fp32 widen (exact)
    // ----------------------------------------------------------------
    function automatic logic [31:0] fp16_to_fp32(input logic [15:0] h);
        logic        s; logic [4:0] e; logic [9:0] m;
        logic [7:0]  e32; logic [22:0] m32;
        int          lead, i, sh;
        begin
            s = h[15]; e = h[14:10]; m = h[9:0];
            if (e == 5'd31) begin
                fp16_to_fp32 = {s, 8'hFF, (m != 0) ? {1'b1, 22'd0} : 23'd0};
            end else if (e == 5'd0 && m == 0) begin
                fp16_to_fp32 = {s, 8'd0, 23'd0};
            end else if (e == 5'd0) begin
                // subnormal fp16 -> normalized fp32.
                // value = m * 2^-24 (m in 1..1023).  Let lead = index of the
                // top set bit of m (0..9).  Normalized: 1.f * 2^(lead-24).
                // fp32 biased exp = (lead-24)+127 = lead+103.
                // fraction = m with the leading 1 removed, left-justified into 23 bits:
                //   remaining low bits = m[lead-1:0] (lead bits) -> shift up to bit22.
                lead = 0;
                for (i = 9; i >= 0; i = i - 1)
                    if (m[i] && lead == 0) lead = i;
                sh  = 9 - lead;
                e32 = 8'(lead + 103);
                // {m, 13'd0} puts m[9] at bit22; shift left by (sh+1) to drop the
                // leading 1 (now at bit23, off the top) and align the fraction.
                m32 = (({m, 13'd0}) << (sh + 1));     // 23-bit truncation keeps frac
                fp16_to_fp32 = {s, e32, m32};
            end else begin
                e32 = 8'({3'b000, e} + 8'd112);       // rebias 127-15=112
                m32 = {m, 13'd0};
                fp16_to_fp32 = {s, e32, m32};
            end
        end
    endfunction

    // ----------------------------------------------------------------
    // fp32 add : IEEE binary32, RNE.  GRS scheme (3 guard/round/sticky bits),
    // single add/sub, leading-one normalize, round-to-nearest-even.
    // Validated bit-exact vs numpy fp32+fp32 on 2M+ random + edge vectors
    // (incl subnormals, cancellation, magnitude-disparate, inf/nan).
    // The 24-bit significand sits at bits [26:3] of a 28-bit `res`; bits[2:0]
    // are guard/round/sticky.  Add may carry to bit27; cancellation may drop
    // the leading one far down (LZD + left shift).
    // ----------------------------------------------------------------
    function automatic logic [31:0] fp32_add(input logic [31:0] a, input logic [31:0] b);
        logic        sa, sb;
        logic [7:0]  ea, eb;
        logic [22:0] fa, fb;
        logic [23:0] ma, mb;             // implicit + 23 frac
        logic [7:0]  eae, ebe, e_big, ediff;
        logic [23:0] m_big, m_small;
        logic        s_big, s_small, s_res, op_sub;
        logic [26:0] M_big;              // m_big << 3
        logic [26:0] full_sm, M_sm;      // m_small << 3, then shifted
        logic        sticky;
        logic [27:0] res;                // up to carry bit 27
        int          i, L, sh;
        logic        st_norm;
        logic [23:0] mant;               // implicit + 23 frac after normalize
        logic        g3, r3, s3, ru;
        logic signed [10:0] e_final;
        logic [26:0] sm_mask;
        begin
            sa = a[31]; ea = a[30:23]; fa = a[22:0];
            sb = b[31]; eb = b[30:23]; fb = b[22:0];

            if (ea == 8'hFF) begin
                if (fa != 0) fp32_add = a;                          // a NaN
                else if (eb == 8'hFF && fb == 0 && (sa != sb))
                     fp32_add = {1'b0, 8'hFF, 23'h400000};          // inf-inf
                else fp32_add = a;                                  // a inf
            end else if (eb == 8'hFF) begin
                fp32_add = b;
            end else begin
                ma  = {(ea == 8'd0) ? 1'b0 : 1'b1, fa};
                mb  = {(eb == 8'd0) ? 1'b0 : 1'b1, fb};
                eae = (ea == 8'd0) ? 8'd1 : ea;
                ebe = (eb == 8'd0) ? 8'd1 : eb;

                if (eae > ebe || (eae == ebe && ma >= mb)) begin
                    e_big = eae; m_big = ma; s_big = sa;
                    m_small = mb; s_small = sb; ediff = eae - ebe;
                end else begin
                    e_big = ebe; m_big = mb; s_big = sb;
                    m_small = ma; s_small = sa; ediff = ebe - eae;
                end
                s_res  = s_big;
                op_sub = (s_big != s_small);

                M_big = {m_big, 3'b000};                            // 24b<<3 -> [26:3]
                if (ediff >= 8'd30) begin
                    M_sm   = 27'd0;
                    sticky = (m_small != 0);
                end else begin
                    full_sm = {m_small, 3'b000};                   // 27b: small<<3
                    M_sm    = full_sm >> ediff;
                    sm_mask = (27'b1 << ediff) - 27'b1;
                    sticky  = |(full_sm & sm_mask);
                end
                M_sm = M_sm | {26'd0, sticky};                     // fold sticky into LSB

                if (op_sub) res = {1'b0, M_big} - {1'b0, M_sm};
                else        res = {1'b0, M_big} + {1'b0, M_sm};

                if (res == 0) begin
                    fp32_add = {1'b0, 8'd0, 23'd0};                 // exact +0
                end else begin
                    // leading-one position
                    L = 0;
                    for (i = 27; i >= 0; i = i - 1)
                        if (res[i] && L == 0) L = i;
                    e_final = $signed({3'b000, e_big}) + (L - 26);
                    // align leading-one to bit26
                    if (L > 26) begin
                        sh      = L - 26;
                        st_norm = |(res & ((28'b1 << sh) - 28'b1));
                        res     = (res >> sh) | {27'd0, st_norm};
                    end else if (L < 26) begin
                        res = res << (26 - L);
                    end
                    // res now: bit26=leading1, [25:3]=23 frac, [2:0]=g/r/s
                    mant = res[26:3];
                    g3 = res[2]; r3 = res[1]; s3 = res[0];
                    ru = g3 & (r3 | s3 | mant[0]);
                    mant = mant + (ru ? 24'd1 : 24'd0);
                    // carry out of 24-bit mantissa (1.111..1 + ulp -> 10.000..0).
                    // mant always had bit23 set (normalized 1.f), so a post-round
                    // value of 0 can ONLY be the 0xFFFFFF+1 wrap => bump exponent,
                    // new significand is 1.0 (0x800000).
                    if (mant == 24'd0) begin
                        mant    = 24'h800000;
                        e_final = e_final + 1;
                    end

                    if (e_final >= 255) begin
                        fp32_add = {s_res, 8'hFF, 23'd0};           // overflow inf
                    end else if (e_final <= 0) begin
                        // subnormal result: shift mant right by (1 - e_final), RNE
                        begin
                            int shs; logic [23:0] subm; logic gg, rr, ss2;
                            shs = 1 - e_final;
                            if (shs > 24) fp32_add = {s_res, 8'd0, 23'd0};
                            else begin
                                subm = mant >> shs;
                                gg = (shs >= 1) ? mant[shs-1] : 1'b0;
                                rr = (shs >= 2) ? mant[shs-2] : 1'b0;
                                ss2 = (shs >= 3) ? (|(mant & ((24'b1 << (shs-2)) - 24'b1))) : 1'b0;
                                if (gg & (rr | ss2 | subm[0])) subm = subm + 24'd1;
                                if (subm[23]) fp32_add = {s_res, 8'd1, subm[22:0]};
                                else          fp32_add = {s_res, 8'd0, subm[22:0]};
                            end
                        end
                    end else begin
                        fp32_add = {s_res, e_final[7:0], mant[22:0]};
                    end
                end
            end
        end
    endfunction

    // ----------------------------------------------------------------
    // fp32 add, SPLIT into two pipeline stages for fmax.  Combining
    //   fp32_add_b(fp32_add_a(a,b)) == fp32_add(a,b) EXACTLY (same algorithm,
    //   cut at the raw-add / normalize boundary).  Used by the II=2 streaming
    //   accumulator so the per-cycle recurrence is ~half the logic depth.
    //
    // Intermediate IR (72 bits): packs the stage-A result so a single register
    //   carries it to stage B.
    //   [71]    is_special  (NaN/Inf/exact-zero -> stage B just returns spec_res)
    //   [70:39] spec_res     (the 32-bit result for the special/zero case)
    //   [38]    s_res        (result sign for the normal path)
    //   [37:30] e_big        (the larger operand's biased exponent)
    //   [29:2]  res28        (the 28-bit raw sum: bit26=nominal leading, [2:0]=grs)
    //   [1:0]   0
    // ----------------------------------------------------------------
    function automatic logic [71:0] fp32_add_a(input logic [31:0] a, input logic [31:0] b);
        logic        sa, sb;
        logic [7:0]  ea, eb;
        logic [22:0] fa, fb;
        logic [23:0] ma, mb;
        logic [7:0]  eae, ebe, e_big, ediff;
        logic [23:0] m_big, m_small;
        logic        s_big, s_small, s_res, op_sub;
        logic [26:0] M_big, full_sm, M_sm, sm_mask;
        logic        sticky;
        logic [27:0] res;
        begin
            sa = a[31]; ea = a[30:23]; fa = a[22:0];
            sb = b[31]; eb = b[30:23]; fb = b[22:0];

            if (ea == 8'hFF) begin
                if (fa != 0)                                   fp32_add_a = {1'b1, a, 39'd0};
                else if (eb == 8'hFF && fb == 0 && (sa != sb)) fp32_add_a = {1'b1, {1'b0,8'hFF,23'h400000}, 39'd0};
                else                                           fp32_add_a = {1'b1, a, 39'd0};
            end else if (eb == 8'hFF) begin
                fp32_add_a = {1'b1, b, 39'd0};
            end else begin
                ma  = {(ea == 8'd0) ? 1'b0 : 1'b1, fa};
                mb  = {(eb == 8'd0) ? 1'b0 : 1'b1, fb};
                eae = (ea == 8'd0) ? 8'd1 : ea;
                ebe = (eb == 8'd0) ? 8'd1 : eb;
                if (eae > ebe || (eae == ebe && ma >= mb)) begin
                    e_big = eae; m_big = ma; s_big = sa;
                    m_small = mb; s_small = sb; ediff = eae - ebe;
                end else begin
                    e_big = ebe; m_big = mb; s_big = sb;
                    m_small = ma; s_small = sa; ediff = ebe - eae;
                end
                s_res  = s_big;
                op_sub = (s_big != s_small);
                M_big = {m_big, 3'b000};
                if (ediff >= 8'd30) begin
                    M_sm = 27'd0; sticky = (m_small != 0);
                end else begin
                    full_sm = {m_small, 3'b000};
                    M_sm    = full_sm >> ediff;
                    sm_mask = (27'b1 << ediff) - 27'b1;
                    sticky  = |(full_sm & sm_mask);
                end
                M_sm = M_sm | {26'd0, sticky};
                if (op_sub) res = {1'b0, M_big} - {1'b0, M_sm};
                else        res = {1'b0, M_big} + {1'b0, M_sm};

                if (res == 0)
                    fp32_add_a = {1'b1, {1'b0,8'd0,23'd0}, 39'd0};  // exact zero -> special
                else
                    fp32_add_a = {1'b0, 32'd0, s_res, e_big, res, 2'd0};
            end
        end
    endfunction

    function automatic logic [31:0] fp32_add_b(input logic [71:0] ir);
        logic        is_special, s_res;
        logic [31:0] spec_res;
        logic [7:0]  e_big;
        logic [27:0] res;
        int          i, L;
        logic        st_norm;
        logic [23:0] mant;
        logic        ru;
        logic signed [10:0] e_final;
        int          sh;
        begin
            is_special = ir[71];
            spec_res   = ir[70:39];
            s_res      = ir[38];
            e_big      = ir[37:30];
            res        = ir[29:2];

            if (is_special) begin
                fp32_add_b = spec_res;
            end else begin
                L = 0;
                for (i = 27; i >= 0; i = i - 1)
                    if (res[i] && L == 0) L = i;
                e_final = $signed({3'b000, e_big}) + (L - 26);
                if (L > 26) begin
                    sh      = L - 26;
                    st_norm = |(res & ((28'b1 << sh) - 28'b1));
                    res     = (res >> sh) | {27'd0, st_norm};
                end else if (L < 26) begin
                    res = res << (26 - L);
                end
                mant = res[26:3];
                ru = res[2] & (res[1] | res[0] | mant[0]);
                mant = mant + (ru ? 24'd1 : 24'd0);
                if (mant == 24'd0) begin
                    mant    = 24'h800000;
                    e_final = e_final + 1;
                end
                if (e_final >= 255) begin
                    fp32_add_b = {s_res, 8'hFF, 23'd0};
                end else if (e_final <= 0) begin
                    begin
                        int shs; logic [23:0] subm; logic gg, rr, ss2;
                        shs = 1 - e_final;
                        if (shs > 24) fp32_add_b = {s_res, 8'd0, 23'd0};
                        else begin
                            subm = mant >> shs;
                            gg = (shs >= 1) ? mant[shs-1] : 1'b0;
                            rr = (shs >= 2) ? mant[shs-2] : 1'b0;
                            ss2 = (shs >= 3) ? (|(mant & ((24'b1 << (shs-2)) - 24'b1))) : 1'b0;
                            if (gg & (rr | ss2 | subm[0])) subm = subm + 24'd1;
                            if (subm[23]) fp32_add_b = {s_res, 8'd1, subm[22:0]};
                            else          fp32_add_b = {s_res, 8'd0, subm[22:0]};
                        end
                    end
                end else begin
                    fp32_add_b = {s_res, e_final[7:0], mant[22:0]};
                end
            end
        end
    endfunction

    // ----------------------------------------------------------------
    // fp32 strict greater-than : a > b, RNE-irrelevant (pure ordering).
    //   Synthesizable replacement for the skeleton's `$bitstoshortreal(a) >
    //   $bitstoshortreal(b)` selector compare.  IEEE-754 fp32 ordering via
    //   the standard sign-magnitude -> monotonic-key transform:
    //     key(x) = x[31] ? ~x : (x | 0x80000000)
    //   so unsigned key(x) increases monotonically with the real value of x
    //   (key(-inf) is smallest, key(+inf) largest).  a > b  <=>  key(a) > key(b).
    //
    //   ZERO HANDLING (match the `real` semantics it replaces): the SV `real`
    //   compare treated +0.0 and -0.0 as EQUAL, so `+0 > -0` was false.  The
    //   monotonic key gives key(+0)=0x80000000 > key(-0)=0x7FFFFFFF, which would
    //   spuriously prefer +0 over -0.  We special-case both-zero -> false so the
    //   ordering (and the stable-insertion tie-break) is bit-identical to the
    //   retired `real` model.  NaN is not produced on the score path (the dot is
    //   finite fp32), so NaN ordering is left at the natural key extreme.
    // ----------------------------------------------------------------
    function automatic logic fp32_gt(input logic [31:0] a, input logic [31:0] b);
        logic [31:0] ka, kb;
        logic        a_zero, b_zero;
        begin
            a_zero = (a[30:0] == 31'd0);   // +0 or -0
            b_zero = (b[30:0] == 31'd0);
            if (a_zero && b_zero) begin
                fp32_gt = 1'b0;            // +0 == -0 == 0  (match `real` ==)
            end else begin
                ka = a[31] ? ~a : (a | 32'h8000_0000);
                kb = b[31] ? ~b : (b | 32'h8000_0000);
                fp32_gt = (ka > kb);
            end
        end
    endfunction

    // ----------------------------------------------------------------
    // cheap_gt : SIGNED two's-complement strict greater-than, a > b.
    //
    //   The cheap-logit path (G8.1 candidate selection) carries the G12 logit
    //   as a SIGNED INT32 (the scaled-GEMV requant accumulator), NOT an
    //   IEEE-754 fp32 bit-pattern.  fp32_gt's sign-magnitude->monotonic-key
    //   transform (x[31]? ~x : x|0x80000000) ORDERS NEGATIVE two's-complement
    //   values BACKWARDS (IEEE negatives sort by magnitude-descending, two's-
    //   complement negatives sort by magnitude-ascending), so feeding INT32
    //   bits to fp32_gt scrambles the candidate set whenever negative logits
    //   sit near the K_C cutoff.
    //
    //   This is the GOLDEN-correct comparator for the cheap key: it is
    //   BIT-IDENTICAL to tree_decode.sv's heap compare
    //       `($signed(score_in) > heap_score[k])`  (tree_decode.sv:71)
    //   so the G8.1 K_C=512 set is selected with the SAME ordering tree_decode
    //   uses for its top-K_F=8, and the INT32 requant is monotonic in the fp32
    //   logit (same scale path for every column of a stage), so signed-int
    //   order == fp32-logit order == eagle4_ref selection order.
    //
    //   STRICT (>): equal keys do NOT displace an incumbent, matching fp32_gt's
    //   strict semantics and tree_decode's stable first-arrived tie-break.
    //   No zero special-case is needed: there is exactly one zero in two's
    //   complement (0x00000000), so the +0/-0 collision fp32_gt guards against
    //   cannot arise on the integer key.
    // ----------------------------------------------------------------
    function automatic logic cheap_gt(input logic [31:0] a, input logic [31:0] b);
        begin
            cheap_gt = ($signed(a) > $signed(b));
        end
    endfunction

endpackage

`endif
