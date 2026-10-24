// Shared GEMV dimensions and per-lane weight/scale address geometry.
// Used by weight_bank_multistage, scale_bram, and the array controller.
//
// Geometry at LANES=128, K_PAR=16, GROUP=128:
//   stage     K     N        cols/lane  grp/col  chunks/col
//   G4_qkv  6144  9216 (qkv)       72       48        384
//   G6_o    3072  3072            24       24        192
//   G7_g    3072  8192            64       24        192
//   G7_u    3072  8192            64       24        192
//   G9_down 8192  6144 (2HS)      48       64        512
//   G12_lm   256  128256       1002        2         16
//
// Weight entries per column = K/16; scale entries per column = K/GROUP.
// Both bank bases sum the preceding stages' per-lane storage footprints.

`ifndef STAGE_GEOMETRY_SVH
`define STAGE_GEOMETRY_SVH

// Callers pass the physical lane count explicitly; stage N must divide it.

// ---- raw stage dims (K, N) — the ONLY place dims live ----
function automatic int sg_stage_K(input int unsigned s);
    case (s)
        0: sg_stage_K = 6144;    // G4_qkv  (q+k+v share K)
        1: sg_stage_K = 3072;    // G6_o
        2: sg_stage_K = 3072;    // G7_g
        3: sg_stage_K = 3072;    // G7_u
        4: sg_stage_K = 8192;    // G9_down
        5: sg_stage_K =  256;    // G12_lm
        default: sg_stage_K = 0;
    endcase
endfunction

function automatic int sg_stage_N(input int unsigned s);
    case (s)
        0: sg_stage_N =   9216;  // G4_qkv  q+k+v=3x3072 concat
        1: sg_stage_N =   3072;  // G6_o
        2: sg_stage_N =   8192;  // G7_g
        3: sg_stage_N =   8192;  // G7_u
        4: sg_stage_N =   6144;  // G9_down ENABLE_2HS both halves
        5: sg_stage_N = 128256;  // G12_lm
        default: sg_stage_N = 0;
    endcase
endfunction

// Supported top-level lane counts (128 and 256) divide every stage N.
// A non-divisor would silently truncate the integer column count.
function automatic int sg_stage_cols(input int unsigned s, input int unsigned lanes);
    sg_stage_cols = sg_stage_N(s) / lanes;
endfunction

// ---- WEIGHT bank geometry (chunk-indexed: K/16 chunks per col) ----
function automatic [16:0] sg_w_chunks_per_col(input [3:0] s);
    sg_w_chunks_per_col = 17'(sg_stage_K(s) / 16);
endfunction

function automatic [16:0] sg_w_stage_base(input [3:0] s, input int unsigned lanes);
    int unsigned acc;
    acc = 0;
    for (int unsigned i = 0; i < s; i++)
        acc += sg_stage_cols(i, lanes) * (sg_stage_K(i) / 16);
    sg_w_stage_base = 17'(acc);
endfunction

// ---- SCALE bank geometry (group-indexed: K/GROUP groups per col) ----
// GROUP is the number of input weights sharing a scale.
function automatic [15:0] sg_s_groups_per_col(input [3:0] s, input int unsigned GROUP);
    sg_s_groups_per_col = 16'(sg_stage_K(s) / GROUP);
endfunction

function automatic [16:0] sg_s_stage_base(input [3:0] s, input int unsigned GROUP,
                                          input int unsigned lanes);
    int unsigned acc;
    acc = 0;
    for (int unsigned i = 0; i < s; i++)
        acc += sg_stage_cols(i, lanes) * (sg_stage_K(i) / GROUP);
    sg_s_stage_base = 17'(acc);
endfunction

`endif // STAGE_GEOMETRY_SVH
