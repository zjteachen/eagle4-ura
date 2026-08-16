// =====================================================================
// stage_geometry.svh   — SINGLE SOURCE OF TRUTH for per-stage GEMV geometry
// =====================================================================
//
// THE DUAL-TABLE TRAP (F1/F4, PRE_PDI5_CODE_REVIEW_20260523, memory
// feedback_scale_bram_weightbank_geometry_coupling):
//   weight_bank_multistage.sv and scale_bram.sv used to each carry their
//   OWN per-stage geometry table (stage_base + per-col stride). When the
//   weight bank was corrected to the true dims (WEIGHT-LAYOUT-FIT,
//   2026-05-23) and scale_bram was NOT, the scaled GEMV silently mis-based
//   for every stage >= G6 and TBs that hardcoded the OLD bases hid it.
//
// FIX: the geometry is now defined ONCE, here. weight_bank_multistage.sv,
//   scale_bram.sv, and every scaled-array TB `include this file and call
//   these functions. A future dims change touches exactly one place; the
//   two banks and the TB can NEVER drift apart again.
//
// True dims (cross-checked vs safetensors + reference/src/gptq.cpp +
// reference/include/eagle4_gemm_abstract.h + docs/M1_WEIGHT_MAP.md,
// group_size=128, LANES=128):
//   stage     K     N        cols/lane  grp/col  chunks/col
//   G4_qkv  6144  9216 (qkv)       72       48        384
//   G6_o    3072  3072            24       24        192
//   G7_g    3072  8192            64       24        192
//   G7_u    3072  8192            64       24        192
//   G9_down 8192  6144 (2HS)      48       64        512
//   G12_lm   256  128256       1002        2         16
//
// Weight bank: entries/col = K/16 (chunks). Scale bank: entries/col = K/128
// (groups). cols/col-count (= N/LANES) is the COUPLING field — it appears
// in BOTH tables and is what drifted. group_idx-stride and chunk-stride are
// derived here from the same N/K so they cannot disagree.
// =====================================================================

`ifndef STAGE_GEOMETRY_SVH
`define STAGE_GEOMETRY_SVH

// LANES-PARAMETRIC (2026-07-26, LANES256_GEMM_STUDY.md §3.1 — the SG_LANES
// define-trap fix): per-lane column counts are N / lanes, and `lanes` is now
// an explicit FUNCTION ARGUMENT threaded from each module's LANES PARAMETER.
// The old `SG_LANES macro is DELETED: the pkg tcl CLEARS verilog_define on
// the packaged IP, so a define-derived geometry can NEVER follow a strapped
// LANES=256 build — the bank bases would stay 128-lane while the array walks
// 256 lanes (geometry split-brain, the 0xE4B71118 netlist-vs-intent class).
// Callers (weight_bank_multistage, scale_bram, TBs) pass their own LANES.

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

// per-lane output columns for a stage = N / lanes  (THE coupling field).
// Every stage N divides both 128 and 256 exactly (LANES256_GEMM_STUDY.md
// §1.1); LANES=192 does NOT divide G7 (8192/192 = 42.67 -> silent floor
// drop of 128 output columns) and is DEAD-by-geometry — the drafter's
// elaboration guard rejects it.
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
// GROUP is the GPTQ group size (128). Kept as an arg so a future re-group
// touches one call site.
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
