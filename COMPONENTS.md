# EAGLE-4 Drafter IP — RTL Component Encyclopedia

Complete technical reference for all 84 SystemVerilog source files (`.sv`, `.svh`) in the `eagle4_drafter_depthn_1_0/src/` IP core targeting the AMD Xilinx Alveo V80 FPGA (`xcv80-lsva4737-2MHP-e-S`).

## Architecture & Wiring Overview

The EAGLE-4 Drafter is a dedicated hardware accelerator for the EAGLE-3 speculative drafting algorithm. It implements a multi-stage transformer forward pass combined with an autoregressive speculative tree generation engine.

### Pipeline Stage Sequence

| Stage | Identification | Operation | Dimension / Datapath | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | Hidden and embedding RMSNorm | Normalize the two vectors independently, then expose `[embedding_norm, hidden_norm]` through R1 | Two 3072-element INT16 vectors; 16 elements per chunk | Two `rmsnorm_block` instances with DEPTH_N=0; no GEMV |
| **G4** | QKV projection | Three projections of the same 6144-element input | $6144 \rightarrow 9216$ (Q, K, V: 3072 each) | Shared scaled integer GEMV array; 72 tiles at LANES=128 |
| **G5** | Rotary embedding, KV append and attention | Rotate Q/K, append K/V and compute attention | 24 heads × 128 dimensions | `attention_full_mha_sbuf` |
| **G6** | Attention output projection | $O = W_o \cdot \text{attention}$ | $3072 \rightarrow 3072$ | Shared GEMV array, stage_sel=1 |
| **G7G/U**| MLP Gate & Up Projections | $G = W_{gate} \cdot x$, $U = W_{up} \cdot x$ | $1536 \rightarrow 4096$ each | Shared GEMV Array |
| **G7_SILU**| SwiGLU Activation | $A = \text{SiLU}(G) \odot U$ | 4096-dim vector | Pipelined DSP mul |
| **G9** | MLP Down Projection | $Y = W_{down} \cdot A + x_{residual}$ | $4096 \rightarrow 1536$ | Shared GEMV Array |
| **G11** | Pre-Head RMSNorm | $x_{norm} = \text{RMSNorm}(Y)$ | 1536-dim vector | Dedicated RMSNorm |
| **GLM** | LM Down-Projection | $Z = W_{lmdown} \cdot x_{norm}$ | $3072 \rightarrow 256$ low-rank | Dense ROM weights |
| **G12** | LM Head Projection | $\text{logits}_{cheap} = W_{head} \cdot Z$ | $256 \rightarrow \text{vocab}$ | Shared GEMV Array |
| **TREE** | Speculative Draft Tree Engine | Tile streaming, logit scaling, top-K selection, branch expansion | Multi-level depth loop ($D \le \text{max\_depth}$) | Tree level FSM |
| **G8** | Vocabulary Rescore (G8_SPLICE)| Fused gather-dot exact logit validation | P-parallel FP16 inner products | External HBM embeddings |
| **OUT** | Output Formatting | Format frontier graph into `OUT_ABI_V24` | Token IDs, scores, parent indices | AXI4 Master Write-DMA |

### G1 and G4 RTL review map

Paths below are relative to `eagle4_drafter_depthn_1_0/src/`.
Start with the top-level integration, then follow the arithmetic primitives:

| Area | RTL file / symbol | Responsibility |
| :--- | :--- | :--- |
| G1 input/control | `eagle4_drafter.sv`: `u_rmsnorm_g1`, `u_rmsnorm_g1_e` | Hidden/embedding/gamma reads and concurrent starts |
| G1 sequencing | `rmsnorm_block.sv` | Sum pass, scale computation, apply pass and pipeline drain |
| G1 reduction | `rmsnorm_sumsq.sv` | Signed squares, chunk reduction, 64-bit accumulation |
| G1 scale | `rmsnorm_scale_int.sv` | Integer square root and reciprocal division, with early saturation |
| G1 apply | `rmsnorm_apply.sv` | Two shifted multiplies, intermediate-width handling and output saturation |
| G1→G4 storage | `eagle4_drafter.sv`: `g_seam_r1_fullw`; `stage_io_buffer_sel.sv`, `stage_io_buffer_ff.sv` | Two complete vectors; embedding half precedes hidden half |
| G4 control/capture | `eagle4_drafter.sv`: `g4seq_state`, `q_full_buf`, `k_full_buf`, `v_full_buf` | Wait for R1, issue all tiles, capture each result and signal completion |
| G4 array | `gemv_stage_array_scaled_slr.sv` | Per-group broadcast registers, per-lane memories, lane instances and result registers |
| G4 arithmetic | `gemv_lane_scaled_stream.sv`; optional `gemv_lane_scaled_stream_dual.sv` | Integer dot products, 128-element group sums and fixed-point scale multiplication |
| Weight decoding/reduction | `gptq2_dequant16.sv`, `csa_tree16_int20.sv` | Two-bit code minus zero-point; three-stage carry-save reduction |
| Weight/scale storage | `stage_geometry.svh`, `weight_bank_multistage.sv`, `weight_bank_uram.sv`, `scale_bram.sv` | Shared stage dimensions, flat per-lane addressing and synchronous reads |
| G4 output conversion | `eagle4_drafter.sv`: `g_scaled_requant`; `attention_full_mha_sbuf.sv`: `g_head_raw` | Remove scale fractional bits into INT32, then convert the selected head to INT16 |
| Optional full-vector conversion | `g4_qkv_seam.sv` | Used instead of selected-head conversion when `REQUANT_HEADMUX=0` |

Current source defaults select the scaled, single-port, 128-lane array with
16 activation elements per chunk, 128-element scale groups and 13 fractional
scale bits. The lane accumulates signed integer group sums times signed scales
into 48 bits; it does not perform FP16 arithmetic.

**Configuration matters:** `eagle4_config.svh` currently sets `CONNECT_SEAMS=0`.
G4 therefore selects external `a_chunk`, not R1 data, unless a build overrides
that parameter or stage injection selects another source. The local smoke
test ties the external input to zero. The G4 scheduler still waits for R1 full.
`REQUANT_HEADMUX=1` selects conversion inside the attention module rather than
instantiating `g4_qkv_seam`. Packaged intent files may describe different values.

**Arithmetic/control caveats retained by the comment-only cleanup:**
- G1's `sat_scale` overflow check omits the retained result sign bit; some
  out-of-range intermediates wrap instead of saturating.
- ISA G1 completion observes hidden `g1_done`, not both norms. G4's full-width
  first-tile wait joins their output buffers, but `STATE G1; HALT` is not itself
  a general completion barrier for unequal norm latencies.
- G4's INT32-to-INT16 rounding adds the bias in INT32, so values near INT32_MAX
  can wrap before clamping. This applies to both output-conversion paths.


### Top-Level Datapath & System Wiring

```
======================================================================================================
                               EAGLE-4 DRAFTER TOP-LEVEL ARCHITECTURE & DATAPATH
======================================================================================================

  AXI4-Lite Slave (s_axi_ctrl)
        |
        v
  +--------------------------------------------------------------------------------------------------+
  |                                        control_regs.sv                                           |
  |  Config registers, execution triggers (start), hyperparams (max_depth, k_live), residency taps  |
  +--------------------------------------------------------------------------------------------------+
        |                                                                           |
        | start, mode controls                                                      | Weight/scale/gamma
        v                                                                           | direct-drive loads
  +-------------------------------------------------------------+                   v
  |                         top_fsm.sv                          |       +----------------------------+
  |  Pipeline Sequencer (20 states: S_IDLE -> S_G1 .. S_DONE)   |       |   Weight / Scale Banks     |
  +-------------------------------------------------------------+       | - weight_bank_multistage   |
        |                                                               | - scale_bram               |
        | Stage Select & Done Handshakes                                | - weight_dma_loader        |
        v                                                               +----------------------------+
======================================================================================|===============
DATAPATH PIPELINE                                                                     |
======================================================================================|===============
                                                                                      |
 [Input Activation]                                                                   |
   emb_gather.sv / stage_inject.sv                                                    |
        |                                                                             |
        v                                                                             |
 +-------------------------------------+                                              |
 | G1: Hidden / Embedding RMSNorm      |                                              |
 |   - two rmsnorm_block instances     |                                              |
 +-------------------------------------+                                              |
        | R1: [embedding_norm, hidden_norm]                                           |
        v                                                                             |
 +-------------------------------------+                                              |
 | G4: QKV Projection                 |                                              |
 |   - gemv_stage_array_scaled_slr.sv  |<---------------------------------------------+
 +-------------------------------------+                                              |
        | Q, K, V activations                                                         |
        v                                                                             |
 +-------------------------------------+                                              |
 | G5: RoPE, KV Append & Attention    |                                              |
 |   - selected-head QKV requant      |                                              |
 |   - rope_layer.sv (rope_pair_rotate)|                                              |
 |   - multihead_kv.sv / kv_head_wide  |                                              |
 +-------------------------------------+                                              |
        | Attention output                                                            |
        v                                                                             |
 +--------------------------------------------------------------------+               |
 | G6: Attention Output Projection                                    |               |
 |   - gemv_stage_array_scaled_slr.sv                                 |<--------------+
 +--------------------------------------------------------------------+               |
        | Hidden states                                                               |
        v                                                                             |
 +-------------------------------------+                                              |
 | G7: MLP Gate / Up Projections       |                                              |
 |   - rmsnorm_block.sv (u_rmsnorm_g7) |                                              |
 |   - gemv_stage_array_scaled_slr.sv  |<---------------------------------------------+
 |   - silu_mul_fix.sv (SwiGLU)        |                                              |
 +-------------------------------------+                                              |
        | SwiGLU activations                                                          |
        v                                                                             |
 +-------------------------------------+                                              |
 | G9: MLP Down Projection             |                                              |
 |   - gemv_stage_array_scaled_slr.sv  |<---------------------------------------------+
 |   - residual_add.sv (Residual Add)  |
 +-------------------------------------+
        | Layer output
        v
 +-------------------------------------+
 | G11 & GLM: Final Norm & Low-Rank    |
 |   - rmsnorm_block.sv (u_rmsnorm_g11)|
 |   - lmdown_gemv_banksplit.sv        | (Dense 3072 -> 256 GEMV via lmdown_W_q15_lane*.mem)
 +-------------------------------------+
        | Low-rank representation (256-dim)
        v
 +-------------------------------------+
 | G12: LM Head Projection             |
 |   - gemv_stage_array_scaled_slr.sv  |<---------------------------------------------+
 +-------------------------------------+
        | Cheap Logits
        v
 +------------------------------------------------------------------------------------+
 | TREE: Speculative Draft Tree Engine                                                |
 |   - tree_tile_stream_pp.sv / tree_tile_stream_pkpar_pp.sv (tile stream)             |
 |   - tree_softmax_bridge.sv    (logit to probability)                               |
 |   - tree_decode.sv            (top-k candidate token selection)                    |
 |   - tree_level_fsm.sv         (multi-level speculative tree coordinator)           |
 |   - tree_merge_pipe.sv        (candidate path merger)                              |
 |   - tree_cheap_serializer.sv  (serialization for rescoring)                        |
 +------------------------------------------------------------------------------------+
        | Candidate tokens & cheap logits
        v
 +------------------------------------------------------------------------------------+
 | G8: Vocab Rescore Engine (Optional G8_SPLICE)                                      |
 |   - g8_rescore_p.sv / g8_rescore.sv                                                |
 |   - g8_stripe_frontend.sv     (striped token embedding fetch)                      |
 |   - g8_dot_fp16_wide_il.sv    (FP16 dot product scoring)                           |
 +------------------------------------------------------------------------------------+
        | Verified tree candidates
        v
 +------------------------------------------------------------------------------------+
 | OUT: Output Formatting & DMA Write                                                 |
 |   - tree_out_walk.sv          (canonical node-major tree traversal)                |
 |   - output_dma_writer.sv      (OUT_ABI_V24 packaging: ids, scores, parents)        |
 +------------------------------------------------------------------------------------+
        |
        v
  AXI4 Master (m_axi_out) -> Host / External Memory
```

### Scaled GEMV Array & Weight Subsystem Wiring

```
======================================================================================================
                          SHARED SCALED GEMV ARRAY & WEIGHT SUBSYSTEM
======================================================================================================

                     +---------------------------------------+
                     |         weight_dma_loader.sv          |
                     |  AXI4 Master burst reads from DRAM    |
                     +---------------------------------------+
                                         |
                                         v
   +-----------------------------------------------------------------------------------+
   |                               weight_bank_multistage.sv                           |
   |  UltraRAM weight banks: URAM288 primitives partitioned across stages (G1..G12)   |
   |  Base addresses and strides derived from stage_geometry.svh                       |
   +-----------------------------------------------------------------------------------+
                                         |
                       Quantized Weights | [ARR_GROUPS x LANES]
                                         v
    Streaming Activation   +---------------------------------------+   Scale Factors (BRAM)
    Input (stage_io_buf) ->|     gemv_stage_array_scaled_slr.sv    |<- scale_bram.sv
                           |  Organized into SLR-crossing groups   |
                           +---------------------------------------+
                                         |
                   +---------------------+---------------------+
                   |                                           |
                   v                                           v
  +-----------------------------------+       +-----------------------------------+
  |     gemv_lane_scaled_stream.sv    |  ...  |     gemv_lane_scaled_stream.sv    |
  |  Lane 0                           |       |  Lane 127                         |
  |  - K_PAR-wide multiplier array    |       |  - K_PAR-wide multiplier array    |
  |  - csa_tree16_int20/int28 adder   |       |  - csa_tree16_int20/int28 adder   |
  |  - Fixed-point scale multiplier   |       |  - Fixed-point scale multiplier   |
  +-----------------------------------+       +-----------------------------------+
                   |                                           |
                   +---------------------+---------------------+
                                         v
                           [Accumulated Output Vector]
                           -> stage_io_buffer_sel.sv
```

### Multi-Head Attention (G6) Subsystem Wiring

```
======================================================================================================
                            MULTI-HEAD ATTENTION (G6) SUBSYSTEM
======================================================================================================

                       Concatenated QKV from G1/G4 GEMV
                                     |
                                     v
                       +---------------------------+
                       |       g4_qkv_seam.sv      |
                       |  Q/K/V splitting & timing |
                       +---------------------------+
                        /            |                                   /             |                          Query Q  /       Key K  |       Value                       v               v               v
            +---------------------------------+  +----------------------+
            |         rope_layer.sv           |  |                      |
            |  Rotary Position Embedding      |  |                      |
            |  rope_pair_rotate (2D rotation) |  |                      |
            +---------------------------------+  |                      |
                     |               |           |                      |
            Rotated Q|      Rotated K|           |                      |
                     |               v           |                      |
                     |      +-----------------+  |                      |
                     |      |  multihead_kv   |  |                      |
                     |      |  kv_head_wide   |  |                      |
                     |      |  kv_lane_chain  |  |                      |
                     |      |  (KV Cache BRAM)|  |                      |
                     |      +-----------------+  |                      |
                     |               | Cached K  | Cached V             |
                     v               v           |                      |
            +---------------------------------+  |                      |
            |        attn_score_dot.sv        |  |                      |
            |  Query-Key inner product Q*K^T  |  |                      |
            +---------------------------------+  |                      |
                             | Raw Scores        |                      |
                             v                   |                      |
            +---------------------------------+  |                      |
            |        softmax_exp_sum.sv       |  |                      |
            |  exp(S - max) / sum(exp)        |  |                      |
            +---------------------------------+  |                      |
                             | Probabilities P   |                      |
                             v                   v                      |
            +-----------------------------------------------------------+
            |                  attn_v_combine_perbank.sv                |
            |            Context Accumulation: out = sum(P * V)         |
            +-----------------------------------------------------------+
                                         |
                                         v
                            Attention Output Context
                            -> residual_add.sv
```

### Speculative Tree & Rescore Subsystem Wiring

```
======================================================================================================
                     SPECULATIVE DRAFT TREE & RESCORING (TREE & G8)
======================================================================================================

                     Cheap Logits from G12 LM Head (256 -> vocab)
                                         |
                                         v
   +-----------------------------------------------------------------------------------+
   |             tree_tile_stream_pp.sv / tree_tile_stream_pkpar_pp.sv                 |
   |  Double-buffered tile streamer: chunks vocab stream into parallel candidate tiles |
   +-----------------------------------------------------------------------------------+
                                         |
                                         v
   +-----------------------------------------------------------------------------------+
   |                             tree_softmax_bridge.sv                                |
   |  Logit-to-probability bridge: evaluates local exp / max for numerical stability    |
   +-----------------------------------------------------------------------------------+
                                         |
                                         v
   +-----------------------------------------------------------------------------------+
   |                                 tree_decode.sv                                    |
   |  Top-K candidate token extraction & branch comparator tree                        |
   +-----------------------------------------------------------------------------------+
                     |                                         |
     Candidate Tokens|                         Tree Path Graph |
                     v                                         v
   +-----------------------------------+     +-----------------------------------------+
   |      tree_cheap_serializer.sv     |     |            tree_level_fsm.sv            |
   |  Serializes cheap logits for      |     |  Multi-level tree expansion coordinator |
   |  G8 verification comparison       |     |  Manages branch pruning & depth loops   |
   +-----------------------------------+     +-----------------------------------------+
                     |                                         |
                     | Candidate token IDs                     | Merged tree structure
                     v                                         |
   +-----------------------------------+                       |
   |        g8_rescore_p.sv            |                       |
   |  - g8_stripe_frontend.sv (HBM)    |                       |
   |  - g8_dot_fp16_wide_il.sv         |                       |
   |  Computes exact target-model logit|                       |
   +-----------------------------------+                       |
                     |                                         |
                     v                                         v
   +-----------------------------------------------------------------------------------+
   |                               tree_out_walk.sv                                    |
   |  Node-major tree graph traversal: flattens accepted paths into linear buffer      |
   +-----------------------------------------------------------------------------------+
                                         |
                                         v
   +-----------------------------------------------------------------------------------+
   |                              output_dma_writer.sv                                 |
   |  Packages tokens, scores, and parent indices into OUT_ABI_V24 frontier format     |
   |  Drives AXI4 Master write bursts onto m_axi_out                                  |
   +-----------------------------------------------------------------------------------+
```


## ISA State Encodings & Instruction Set

The instruction fetcher (`instruction_fetcher.sv`) executes byte-granular
programs out of a 32-word PMEM (128 bytes). Each instruction is one byte:
`instr[7:5]` is the opcode, `instr[4:0]` is the operand.

| Opcode | Mnemonic | Operand | Behavior |
| :--- | :--- | :--- | :--- |
| `000` | STATE | state ID (below) | Pulse the matching stage start; hold until that stage's done advances the PC |
| `001` | JMP | PMEM word address (0..31) | `PC <- {operand, 2'b00}`; executes in one cycle, target must be word-aligned |
| `010` | HALT | reserved (0) | Stop execution; one-cycle `halted` pulse (kernel done); PC stays at the HALT byte |
| `011` | WRSR | special-register address (0..10) | Three bytes: header `0x60\|addr`, immediate low byte, immediate high byte. Executes in one cycle: exactly one `special_regs` write, `PC += 3` |
| `100`-`111` | reserved | — | Decode as HALT (fail-safe stop; an under-programmed PMEM cannot free-run) |

WRSR immediates are little-endian. The two immediate bytes are fetched
combinationally like every other instruction byte, so they may cross any
32-bit PMEM word boundary. Byte addressing wraps modulo the 128-byte PMEM
(existing policy): at PC 126 the low/high immediate bytes are 127/0 and the
next instruction starts at PC 1; at PC 127 they are bytes 0/1 and the next
instruction starts at PC 2. A `start_en` pulse in the same cycle as a WRSR
wins (the bank write is suppressed along with the old PC). WRSR to addresses
11..31 is still one bank write strobe,
but the bank ignores it (reserved writes do not change storage).

The 5-bit STATE operand selects one of the 20 sequencer states below. The
values reuse the `top_fsm` state enum numerically, so debug taps
(`dbg_stage_capture` `top_state`, PMEM status readback) keep their meaning
across the FSM-to-ISA migration.

| Value | State | Description |
| :--- | :--- | :--- |
| `5'd0` | `S_IDLE` | Idle; no stage active. Reserved in ISA mode (a program starts at `PC=0` on `ap_start`, never in IDLE). |
| `5'd1` | `S_G1` | Separate 3072-element RMSNorms over hidden-state and embedding streams (two parallel `rmsnorm_block` instances with DEPTH_N=0; a third embedding-requantization instance exists only with DEPTH_N=1). No QKV projection; that is G4. |
| `5'd2` | `S_G4` | QKV projection GEMV on the shared array (`stage_sel=0`). |
| `5'd3` | `S_G5_ROPE` | Rotary embedding on Q/K and KV-cache append; KV-only launches terminate here. |
| `5'd4` | `S_G6` | Multi-head attention over the windowed KV cache (`stage_sel=1`). |
| `5'd5` | `S_G7G` | MLP gate projection GEMV (`stage_sel=2`). |
| `5'd6` | `S_G7U` | MLP up projection GEMV (`stage_sel=3`). |
| `5'd7` | `S_G7_SILU` | SwiGLU activation, `SiLU(gate) * up`. |
| `5'd8` | `S_G9` | MLP down projection GEMV + residual add (`stage_sel=4`). |
| `5'd9` | `S_G11` | Pre-head RMSNorm (`u_rmsnorm_g11`). |
| `5'd10` | `S_G12` | LM head projection GEMV to cheap vocab logits (`stage_sel=5`). |
| `5'd11` | `S_TREE` | Speculative tree scan and top-K selection over the cheap logits. |
| `5'd12` | `S_DONE` | FSM kernel completion. In ISA mode use the HALT instruction instead; executing `STATE 12` stalls (reserved). |
| `5'd13` | `S_GLM` | Low-rank LM down projection, 3072 -> 256 (`lmdown_gemv_banksplit`). |
| `5'd14` | `S_G8` | Optional frontier rescore against exact HBM embeddings (G8_SPLICE builds only). Must follow `S_TREE` — the G8 engine launches off `tree_start`. |
| `5'd15` | `S_LVL_ARM` | Depth-N: initialize the tree-level loop. Reserved in ISA mode (depth-N disabled). |
| `5'd16` | `S_LVL_FEED` | Depth-N: submit a branch's children to the tree engine. Reserved. |
| `5'd17` | `S_LVL_SEL` | Depth-N: wait for level frontier selection. Reserved. |
| `5'd18` | `S_LVL_OUT` | Depth-N: walk the accepted tree for output. Reserved. |
| `5'd19` | `S_LVL_EMBG` | Depth-N: gather child token embeddings. Reserved. |

Example program — the default linear draft pass (the FSM sequence at
DEPTH_N=0, G8_SPLICE=0), one byte per instruction:

```text
byte 0..11: 01 02 03 04 05 06 07 08 09 0D 0A 0B   ; G1 G4 G5 G6 G7G G7U SILU G9 G11 GLM G12 TREE
byte 12:    40                                      ; HALT
```

PMEM is programmed through the control register aperture before `ap_start`:
seek the word write pointer with `0x168`, push words through the `0x16C`
data mailbox (auto-increment), and read fetcher status (canary, PC, current
instruction) back from `0x16C`. Each `ap_start` resets `PC` to 0 and runs the
program to HALT. Jump targets must be word-aligned (multiple of 4 bytes);
pad with reserved bytes before a target if needed. PMEM writes issued while
the program is running are ignored by the fetcher (program between launches;
a mid-run write would combinationaly corrupt the instruction under
execution). Unwritten PMEM reads as HALT, so a program without a HALT traps
instead of running off the end.

### Special-register bank (`special_regs`)

WRSR writes the 11 x 16-bit ISA special-register bank in
`special_regs.sv`. Addresses 0..10 are implemented; 11..31 are reserved
(writes ignored, reads return zero). The first six addresses preserve the
original hands-on exercise contract.

| Addr | Field | Meaning |
| ---: | :--- | :--- |
| 0 | `activation_base` | Starting activation chunk |
| 1 | `weight_base[15:0]` | Per-lane weight-bank base (packed 32-bit words) |
| 2 | `tile_count` | Number of output tiles in the operation |
| 3 | `chunks_per_tile` | Activation/weight chunks per tile |
| 4 | `chunks_per_scale_group` | Chunks sharing one scale entry |
| 5 | `output_base` | First output tile within the destination buffer |
| 6 | `weight_base[16]` | Bit 16 of the weight base (bits 15:1 reserved) |
| 7 | `scale_base[15:0]` | Per-lane scale-bank base |
| 8 | `scale_base[16]` | Bit 16 of the scale base (bits 15:1 reserved) |
| 9 | packed config | `{2'b0, activation_source[2:0], output_destination[3:0], qzero_raw[1:0], scale_frac[4:0]}` |
| 10 | `output_conversion` | Downstream store-conversion selector |

The bank exposes decoded configuration outputs for the GEMV operation
controller (latched at START in the follow-on integration task); register
writes and validation of the numeric fields belong to the future engine, not
this bank. Readback is through AXI: write the 5-bit bank address to `0x178`,
then read `0x17C` for `{16'h5A5A, read_data}`. Reads use the bank's
synchronous read port; `control_regs` waits for the bank's `read_ack` and
retries when a concurrent WRSR write suppressed the bank read (write
priority), so `0x17C` never silently returns stale data.
---

## Component Reference Encyclopedia

## 1. Top-Level Orchestration & Control

### `eagle4_drafter`
- **Source File:** `eagle4_drafter_depthn_1_0/src/eagle4_drafter.sv` (7698 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Top-level wrapper module for the EAGLE-4 Drafter IP core on AMD Xilinx Alveo V80. Interfaces directly with host via AXI4-Lite control slave (s_axi_ctrl) and memory buses via AXI4 master write-DMA (m_axi_out). Interconnects all datapath stages (G1 through G12, TREE, G8 rescore), coordinates clock and reset trees, and binds debug capture/observation taps.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `\`EAGLE4_LANES` | `int` |
| `K_MAX` | `\`EAGLE4_K_MAX` | `int` |
| `K_PAR` | `\`EAGLE4_K_PAR` | `int` |
| `A_WIDTH` | `\`EAGLE4_A_WIDTH` | `int` |
| `TREE_V_MAX` | `\`EAGLE4_TREE_V_MAX` | `int` |
| `BOOT_AXI_ADDR_W` | `\`EAGLE4_BOOT_AXI_ADDR_W` | `int` |
| `BOOT_AXI_DATA_W` | `\`EAGLE4_BOOT_AXI_DATA_W` | `int` |
| `BOOT_AXI_ID_W` | `\`EAGLE4_BOOT_AXI_ID_W` | `int` |
| `BOOT_BURST_LEN` | `\`EAGLE4_BOOT_BURST_LEN` | `int` |
| `OUT_ABI_V24` | `\`EAGLE4_OUT_ABI_V24` | `bit` |
| `OUT_TOP_K` | `\`EAGLE4_OUT_TOP_K` | `int` |
| `OUT_PARENT_SENTINEL` | `\`EAGLE4_OUT_PARENT_SENTINEL` | `int` |
| `CONNECT_SEAMS` | `\`EAGLE4_CONNECT_SEAMS` | `bit` |
| `TAIL_FP16` | `\`EAGLE4_TAIL_FP16` | `bit` |
| `TREE_P` | `\`EAGLE4_TREE_P` | `int` |
| `TREE_PP` | `\`EAGLE4_TREE_PP` | `bit` |
| `GEMV_KPAR64` | `\`EAGLE4_GEMV_KPAR64` | `bit` |
| `GEMV_SCALED` | `\`EAGLE4_GEMV_SCALED` | `bit` |
| `GEMV_SCALE_FRAC` | `\`EAGLE4_GEMV_SCALE_FRAC` | `int` |
| `GPTQ_QZERO_RAW` | `\`EAGLE4_GPTQ_QZERO_RAW` | `[1:0]` |
| `SEAM_R1_FULLW` | `\`EAGLE4_SEAM_R1_FULLW` | `bit` |
| `S7A_FRONT_SEQ` | `\`EAGLE4_S7A_FRONT_SEQ` | `bit` |
| `BACK_SEQ` | `\`EAGLE4_BACK_SEQ` | `bit` |
| `SLR_SAFE` | `\`EAGLE4_SLR_SAFE` | `bit` |
| `ARR_GROUPS` | `\`EAGLE4_ARR_GROUPS` | `int` |
| `ARR_PIPE_STAGES` | `\`EAGLE4_ARR_PIPE_STAGES` | `int` |
| `SEAM_FF` | `\`EAGLE4_SEAM_FF` | `bit` |
| `FF_ACT_CACHE` | `\`EAGLE4_FF_ACT_CACHE` | `bit` |
| `DUAL_PORT` | `\`EAGLE4_DUAL_PORT` | `bit` |
| `BODY_BATCH_B` | `\`EAGLE4_BODY_BATCH_B` | `int` |
| `SEAM_FF_LAND_P` | `\`EAGLE4_SEAM_FF_LAND_P` | `bit` |
| `TREE_SEAM_FF` | `\`EAGLE4_TREE_SEAM_FF` | `bit` |
| `LMDOWN_MEM_DIR_P` | `"/home/y344shi/eagle4_artifacts/p4_8815"` | *untyped* |
| `D1_LAT` | `\`EAGLE4_D1_LAT` | `bit` |
| `PREFETCH_COL_P` | `\`EAGLE4_PREFETCH_COL_P` | `bit` |
| `D1_RDLAT` | `\`EAGLE4_D1_RDLAT` | `bit` |
| `D1_WLAT` | `\`EAGLE4_D1_WLAT` | `bit` |
| `R4_BANKSPLIT` | `\`EAGLE4_R4_BANKSPLIT` | `bit` |
| `CTX_WIN` | `\`EAGLE4_CTX_WIN` | `int` |
| `DEPTH_N` | `\`EAGLE4_DEPTH_N` | `bit` |
| `EMBG_EN` | `\`EAGLE4_EMBG_EN` | `bit` |
| `EMB_NCH` | `\`EAGLE4_EMB_NCH` | `int` |
| `FORREASON` | `\`EAGLE4_FORREASON` | `bit` |
| `FR_KF_BANKS` | `\`EAGLE4_FR_KF_BANKS` | `int` |
| `FR_ABANK_BRAM` | `\`EAGLE4_FR_ABANK_BRAM` | `bit` |
| `HEAD_PAR` | `\`EAGLE4_HEAD_PAR` | `int` |
| `ACCFIX_SEAM_P` | `\`EAGLE4_ACCFIX_SEAM_P` | `bit` |
| `ACCFIX_B2P_P` | `\`EAGLE4_ACCFIX_B2P_P` | `bit` |
| `ACCFIX_KC_P` | `\`EAGLE4_ACCFIX_KC_P` | `bit` |
| `ACCFIX_KVONLY_P` | `\`EAGLE4_ACCFIX_KVONLY_P` | `bit` |
| `OBS_LVL_P` | `\`EAGLE4_OBS_LVL_P` | `bit` |
| `ENORM_D0` | `\`EAGLE4_ENORM_D0` | `bit` |
| `PRECISION_SAFE` | `\`EAGLE4_PRECISION_SAFE` | `bit` |
| `SILU_RANGE_FIX_P` | `\`EAGLE4_SILU_RANGE_FIX_P` | `bit` |
| `KV_V_RAM_STYLE` | `\`EAGLE4_KV_V_RAM_STYLE` | *untyped* |
| `KV_K_RAM_STYLE` | `\`EAGLE4_KV_K_RAM_STYLE` | *untyped* |
| `COMBINE_LOCAL` | `\`EAGLE4_COMBINE_LOCAL` | `bit` |
| `REQUANT_HEADMUX` | `\`EAGLE4_REQUANT_HEADMUX` | `bit` |
| `ROPE_CS_PIPE` | `\`EAGLE4_ROPE_CS_PIPE` | `bit` |
| `ROPE_POS_REP` | `\`EAGLE4_ROPE_POS_REP` | `int` |
| `KV_ADDR_REP` | `\`EAGLE4_KV_ADDR_REP` | `int` |
| `KV_RES_CLR_REP` | `\`EAGLE4_KV_RES_CLR_REP` | `int` |
| `G8_SPLICE` | `\`EAGLE4_G8_SPLICE` | `bit` |
| `G8_K_C` | `\`EAGLE4_G8_K_C` | `int` |
| `G8_MAX_BURST` | `\`EAGLE4_G8_MAX_BURST` | `int` |
| `G8_PIPELINED` | `\`EAGLE4_G8_PIPELINED` | `bit` |
| `G8_OUTSTANDING` | `\`EAGLE4_G8_OUTSTANDING` | `int` |
| `G8_ROWPP` | `\`EAGLE4_G8_ROWPP` | `bit` |
| `G8_PREFETCH` | `\`EAGLE4_G8_PREFETCH` | `bit` |
| `G8_DATA_W` | `\`EAGLE4_G8_DATA_W` | `int` |
| `G8_OPT1C` | `\`EAGLE4_G8_OPT1C` | `bit` |
| `G8_OPT1D` | `\`EAGLE4_G8_OPT1D` | `bit` |
| `G8_MIN_NG` | `\`EAGLE4_G8_MIN_NG` | `int` |
| `G8_OPT1F` | `\`EAGLE4_G8_OPT1F` | `bit` |
| `G8_SELPIPE` | `\`EAGLE4_G8_SELPIPE` | `bit` |
| `G8_OPT1G` | `\`EAGLE4_G8_OPT1G` | `bit` |
| `G8_WIDE` | `\`EAGLE4_G8_WIDE` | `bit` |
| `G8_WIDE_P` | `\`EAGLE4_G8_WIDE_P` | `int` |
| `G8_MROUND` | `\`EAGLE4_G8_MROUND` | `bit` |
| `G8_SCALAR_II1` | `\`EAGLE4_G8_SCALAR_II1` | `bit` |
| `G8_CONSUME_HARDEN` | `\`EAGLE4_G8_CONSUME_HARDEN` | `bit` |
| `G8_LANE_FIX` | `\`EAGLE4_G8_LANE_FIX` | `bit` |
| `G8_WIDE_IL` | `\`EAGLE4_G8_WIDE_IL` | `bit` |
| `G8_WIDE_M` | `\`EAGLE4_G8_WIDE_M` | `int` |
| `G8_NCH` | `\`EAGLE4_G8_NCH` | `int` |
| `BUFRDREG` | `\`EAGLE4_BUFRDREG` | `bit` |
| `STAGE_INJECT` | `\`EAGLE4_STAGE_INJECT` | `bit` |

**Port Interface:**

*Inputs:*
- **`ap_clk`**
- **`ap_rst_n`**
- `[8:0]` **`s_axi_control_awaddr`**
- **`s_axi_control_awvalid`**
- `[31:0]` **`s_axi_control_wdata`**
- `[3:0]` **`s_axi_control_wstrb`**
- **`s_axi_control_wvalid`**
- **`s_axi_control_bready`**
- `[8:0]` **`s_axi_control_araddr`**
- **`s_axi_control_arvalid`**
- **`s_axi_control_rready`**
- **`boot_we`**
- `[15:0]` **`boot_lane_sel`**
- `[16:0]` **`boot_wr_addr`**
- `[31:0]` **`boot_wr_data`**
- **`tb_sc_we`**
- `[15:0]` **`tb_sc_lane_sel`**
- `[16:0]` **`tb_sc_wr_addr`**
- `[15:0]` **`tb_sc_wr_data`**
- **`m_axi_boot_arready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_boot_rid`**
- `[BOOT_AXI_DATA_W-1:0]` **`m_axi_boot_rdata`**
- `[1:0]` **`m_axi_boot_rresp`**
- **`m_axi_boot_rlast`**
- **`m_axi_boot_rvalid`**
- **`m_axi_out_awready`**
- **`m_axi_out_wready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_out_bid`**
- `[1:0]` **`m_axi_out_bresp`**
- **`m_axi_out_bvalid`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `[1:0]` **`qz_raw`**
- `[15:0]` **`tile_idx`**
- **`m_axi_g8_arready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_g8_rid`**
- `[G8_DATA_W-1:0]` **`m_axi_g8_rdata`**
- `[1:0]` **`m_axi_g8_rresp`**
- **`m_axi_g8_rlast`**
- **`m_axi_g8_rvalid`**
- **`m_axi_g8b_arready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_g8b_rid`**
- `[G8_DATA_W-1:0]` **`m_axi_g8b_rdata`**
- `[1:0]` **`m_axi_g8b_rresp`**
- **`m_axi_g8b_rlast`**
- **`m_axi_g8b_rvalid`**
- **`m_axi_emb_arready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_emb_rid`**
- `[255:0]` **`m_axi_emb_rdata`**
- `[1:0]` **`m_axi_emb_rresp`**
- **`m_axi_emb_rlast`**
- **`m_axi_emb_rvalid`**
- **`m_axi_embb_arready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_embb_rid`**
- `[255:0]` **`m_axi_embb_rdata`**
- `[1:0]` **`m_axi_embb_rresp`**
- **`m_axi_embb_rlast`**
- **`m_axi_embb_rvalid`**

*Outputs:*
- **`s_axi_control_awready`**
- **`s_axi_control_wready`**
- `[1:0]` **`s_axi_control_bresp`**
- **`s_axi_control_bvalid`**
- **`s_axi_control_arready`**
- `[31:0]` **`s_axi_control_rdata`**
- `[1:0]` **`s_axi_control_rresp`**
- **`s_axi_control_rvalid`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_boot_arid`**
- `[BOOT_AXI_ADDR_W-1:0]` **`m_axi_boot_araddr`**
- `[7:0]` **`m_axi_boot_arlen`**
- `[2:0]` **`m_axi_boot_arsize`**
- `[1:0]` **`m_axi_boot_arburst`**
- **`m_axi_boot_arvalid`**
- **`m_axi_boot_rready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_out_awid`**
- `[BOOT_AXI_ADDR_W-1:0]` **`m_axi_out_awaddr`**
- `[7:0]` **`m_axi_out_awlen`**
- `[2:0]` **`m_axi_out_awsize`**
- `[1:0]` **`m_axi_out_awburst`**
- **`m_axi_out_awvalid`**
- `[BOOT_AXI_DATA_W-1:0]` **`m_axi_out_wdata`**
- `[BOOT_AXI_DATA_W/8-1:0]` **`m_axi_out_wstrb`**
- **`m_axi_out_wlast`**
- **`m_axi_out_wvalid`**
- **`m_axi_out_bready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_g8_arid`**
- `[BOOT_AXI_ADDR_W-1:0]` **`m_axi_g8_araddr`**
- `[7:0]` **`m_axi_g8_arlen`**
- `[2:0]` **`m_axi_g8_arsize`**
- `[1:0]` **`m_axi_g8_arburst`**
- **`m_axi_g8_arvalid`**
- **`m_axi_g8_rready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_g8b_arid`**
- `[BOOT_AXI_ADDR_W-1:0]` **`m_axi_g8b_araddr`**
- `[7:0]` **`m_axi_g8b_arlen`**
- `[2:0]` **`m_axi_g8b_arsize`**
- `[1:0]` **`m_axi_g8b_arburst`**
- **`m_axi_g8b_arvalid`**
- **`m_axi_g8b_rready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_emb_arid`**
- `[BOOT_AXI_ADDR_W-1:0]` **`m_axi_emb_araddr`**
- `[7:0]` **`m_axi_emb_arlen`**
- `[2:0]` **`m_axi_emb_arsize`**
- `[1:0]` **`m_axi_emb_arburst`**
- **`m_axi_emb_arvalid`**
- **`m_axi_emb_rready`**
- `[BOOT_AXI_ID_W-1:0]` **`m_axi_embb_arid`**
- `[BOOT_AXI_ADDR_W-1:0]` **`m_axi_embb_araddr`**
- `[7:0]` **`m_axi_embb_arlen`**
- `[2:0]` **`m_axi_embb_arsize`**
- `[1:0]` **`m_axi_embb_arburst`**
- **`m_axi_embb_arvalid`**
- **`m_axi_embb_rready`**
- `[3:0]` **`stage_sel`**
- `[15:0]` **`array_chunk_idx_out`**
- **`array_chunk_idx_valid`**
- `signed [32*LANES-1:0]` **`array_acc_out`**

**Wiring & Hierarchy:**
- **Instantiated By:** *Top-level IP core (instantiated in system testbenches / block designs)*
- **Instantiates:** `attention_full_mha_sbuf` (`attention_full_mha_sbuf.sv`), `attention_head_assembly` (`attention_head_assembly.sv`), `control_regs` (`control_regs.sv`), `dbg_observe` (`dbg_observe.sv`), `dbg_probe_ram` (`dbg_probe_ram.sv`), `dbg_residency` (`dbg_residency.sv`), `dbg_stage_capture` (`dbg_stage_capture.sv`), `emb_gather` (`emb_gather.sv`), `g4_qkv_seam` (`g4_qkv_seam.sv`), `gemv_stage_array_scaled_slr` (`gemv_stage_array_scaled_slr.sv`), `gemv_stage_array_uram` (`gemv_stage_array_uram.sv`), `gemv_stage_array_uram_kpar64` (`gemv_stage_array_uram_kpar64.sv`), `kv_head_wide` (`kv_head_wide.sv`), `output_dma_writer` (`output_dma_writer.sv`), `rmsnorm_block` (`rmsnorm_block.sv`), `rope_layer` (`rope_layer.sv`), `silu_layer` (`silu_layer.sv`), `silu_layer_fix` (`silu_layer_fix.sv`), `stage_inject` (`stage_inject.sv`), `stage_io_buffer_sel` (`stage_io_buffer_sel.sv`), `top_fsm` (`top_fsm.sv`), `tree_cheap_serializer` (`tree_cheap_serializer.sv`), `tree_decode` (`tree_decode.sv`), `tree_dma_writer` (`tree_dma_writer.sv`), `tree_feed_seq` (`tree_feed_seq.sv`), `tree_level_fsm` (`tree_level_fsm.sv`), `tree_merge_pipe` (`tree_merge_pipe.sv`), `tree_out_walk` (`tree_out_walk.sv`), `tree_softmax_bridge` (`tree_softmax_bridge.sv`), `tree_tile_stream` (`tree_tile_stream.sv`), `tree_tile_stream_pkpar` (`tree_tile_stream_pkpar.sv`), `tree_tile_stream_pkpar_pp` (`tree_tile_stream_pkpar_pp.sv`), `tree_tile_stream_pp` (`tree_tile_stream_pp.sv`), `weight_dma_loader` (`weight_dma_loader.sv`)

---

### `top_fsm`
- **Source File:** `eagle4_drafter_depthn_1_0/src/top_fsm.sv` (398 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Primary hardware pipeline sequencer FSM. Implements a 20-state finite state machine (S_IDLE through S_DONE) coordinating execution across all forward-pass stages (G1, G4, G5_ROPE, G6, G7G, G7U, G7_SILU, G9, G11, GLM, G12, TREE, G8) and orchestrating the multi-level depth loop (S_LVL_ARM, S_LVL_FEED, S_LVL_SEL, S_LVL_OUT, S_LVL_EMBG). Manages start/done handshakes with all compute blocks.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `G8_SPLICE` | `1'b0` | `bit` |
| `DEPTH_N` | `1'b0` | `bit` |
| `K_F` | `8` | `int` |
| `EMBG_EN` | `1'b0` | `bit` |
| `KVONLY_EN` | `1'b0` | `bit` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[31:0]` **`max_depth`**
- **`kvonly_i`**
- **`level_done`**
- **`level_last`**
- **`tree_all_done`**
- **`out_walk_done`**
- **`emb_done`**
- **`g1_done`**
- **`g4_done`**
- **`g5_done`**
- **`g6_done`**
- **`g7g_done`**
- **`g7u_done`**
- **`g7p_done`**
- **`g9_done`**
- **`g11_done`**
- **`glm_done`**
- **`g12_done`**
- **`tree_done`**
- **`g8_done`**

*Outputs:*
- **`busy`**
- **`done`**
- **`lvl_start`**
- `[2:0]` **`branch_j`**
- `[3:0]` **`level_i`**
- **`branch_feed`**
- **`out_walk_start`**
- **`emb_start`**
- **`g1_start`**
- **`g4_start`**
- **`g5_start`**
- **`g6_start`**
- **`g7g_start`**
- **`g7u_start`**
- **`g7p_start`**
- **`g9_start`**
- **`g11_start`**
- **`glm_start`**
- **`g12_start`**
- **`tree_start`**
- **`g8_start`**
- `[3:0]` **`stage_sel`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `instruction_fetcher`
- **Source File:** `eagle4_drafter_depthn_1_0/src/instruction_fetcher.sv` (171 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Program-counter sequencer and PMEM holder for the ISA control flow. Fetches one instruction byte per byte address combinationally (0-cycle fetch) from a 32-word distributed RAM, decodes the 3-bit opcode (STATE / JMP / HALT / WRSR / reserved-as-HALT), executes word-aligned jumps and three-byte special-register writes autonomously, and tracks run/halt state for the kernel busy/done handshake. WRSR immediate bytes may cross PMEM word boundaries; byte addressing wraps modulo 128. PMEM is programmed through the `control_regs` 0x168/0x16C aperture; the 5-bit STATE operand is the sequencer state ID (see *ISA State Encodings & Instruction Set* above).

**Parameters:** *(None)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_en`** — active-low synchronous reset
- **`start_en`** — 1-cycle launch pulse (ap_start): PC <- 0, begin execution
- **`adv_en`** — advance strobe: the current STATE instruction's stage done
- **`write_en`** / `[7:0]` **`write_addr`** / `[31:0]` **`write_data`** — PMEM program port

*Outputs:*
- `[7:0]` **`instr`** — current instruction byte
- `[6:0]` **`pc_out`** — current program counter (0..127)
- **`state_pulse`** — 1-cycle pulse when a STATE instruction begins executing
- `[4:0]` **`state_id`** — state ID of the current STATE instruction
- **`is_state`** — level: current instruction is STATE and running
- **`running`** — execution active (start_en .. HALT)
- **`halted`** — 1-cycle pulse when HALT (or a reserved opcode) executes
- **`sr_write_en`** — 1-cycle WRSR bank write strobe (exactly one per WRSR)
- `[4:0]` **`sr_write_sel`** — WRSR register address (header bits 4:0)
- `[15:0]` **`sr_write_data`** — WRSR immediate, little-endian byte order

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `special_regs`
- **Source File:** `eagle4_drafter_depthn_1_0/src/special_regs.sv` (97 lines)
- **Construct Type:** `module`
- **Technical Purpose:** ISA-written 11 x 16-bit execution-configuration bank. WRSR supplies the write port; an AXI debug read port returns the selected raw register through `control_regs` 0x178/0x17C. Decodes the raw registers into the activation/weight/scale bases, tile and chunk bounds, routing selectors, numeric fields, and output-conversion selector handed to the GEMV operation controller. Addresses 11..31 are reserved.

**Parameters:** *(None)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_en`** — active-low synchronous reset
- **`write_en`**, `[4:0]` **`write_select`**, `[15:0]` **`write_data`** — ISA WRSR write port
- **`read_en`**, `[4:0]` **`read_select`** — AXI debug read port

*Outputs:*
- `[15:0]` **`read_data`** — registered read data (holds when idle/suppressed)
- **`read_ack`** — combinational read-commit indicator; low when a WRSR write suppresses the read
- `[15:0]` **`activation_base`**
- `[16:0]` **`weight_base`** — `{regs[6][0], regs[1]}`
- `[15:0]` **`tile_count`**
- `[15:0]` **`chunks_per_tile`**
- `[15:0]` **`chunks_per_scale_group`**
- `[15:0]` **`output_base`**
- `[16:0]` **`scale_base`** — `{regs[8][0], regs[7]}`
- `[2:0]` **`activation_source`**
- `[3:0]` **`output_destination`**
- `[1:0]` **`qzero_raw`**
- `[4:0]` **`scale_frac`**
- `[15:0]` **`output_conversion`**

**Precedence:** reset > write > read. Any asserted write (including a reserved address) suppresses the read and holds `read_data`; reserved writes change no storage and reserved reads return zero.

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `control_regs`
- **Source File:** `eagle4_drafter_depthn_1_0/src/control_regs.sv` (993 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Memory-mapped AXI4-Lite slave register file (4KB aperture). Decodes control and configuration registers: execution triggers (ap_start, ap_done), runtime hyperparameters (max_depth, k_live, kvonly_act), base addresses for DMA channels, direct-drive memory access ports (boot_we, tb_sc_we, EMB/HFC/GAMMA register trios), and the ISA PMEM mailbox/special-register debug window (0x168/0x16C/0x178/0x17C). Provides status polling and error flags to host.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `COL_W` | `20` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- `[8:0]` **`s_axi_control_awaddr`**
- **`s_axi_control_awvalid`**
- `[31:0]` **`s_axi_control_wdata`**
- `[3:0]` **`s_axi_control_wstrb`**
- **`s_axi_control_wvalid`**
- **`s_axi_control_bready`**
- `[8:0]` **`s_axi_control_araddr`**
- **`s_axi_control_arvalid`**
- **`s_axi_control_rready`**
- **`kernel_busy`**
- **`kernel_done`**
- `[31:0]` **`obs_rd_data`**
- `[31:0]` **`obs_status_word`**
- **`boot_dma_busy`**
- **`boot_dma_done`**
- **`boot_dma_timeout`**
- **`out_dma_busy`**
- **`out_dma_done`**
- **`out_dma_err`**
- **`g8_wd_err`**
- `[31:0]` **`dbg_fsmstate`**
- `[31:0]` **`dbg_activity`**
- `[31:0]` **`dbg_cyclecount`**
- `[31:0]` **`dbg_checksum`**
- `[31:0]` **`dbg_g8progress`**
- `[31:0]` **`dbg_g8araddr_lo`**
- `[31:0]` **`dbg_g8araddr_hi`**
- `[31:0]` **`dbg_rearm`**
- `[31:0]` **`dbg_bootstream`**
- `[31:0]` **`probe_addr`**
- `[31:0]` **`probe_data`**
- `[31:0]` **`probe_xor`**
- `[31:0]` **`probe_sum`**
- `[31:0]` **`probe_count`**
- `[31:0]` **`dbg_stageout_sample`**
- `[31:0]` **`dbg_stageout_count`**
- `[31:0]` **`dbg_cap_widx_word`**
- `[31:0]` **`dbg_cap_val_word`**
- `[31:0]` **`dbg_cap_meta_word`**
- `[31:0]` **`res_crc_word`**
- `[31:0]` **`res_sum_word`**
- `[31:0]` **`res_count_word`**
- `[31:0]` **`kv_res_status_word`**
- `[31:0]` **`maxprobe_build_id`**
- `[31:0]` **`cap_crc_word`**
- `[31:0]` **`inj_en_word`**
- `[31:0]` **`inj_widx_word`**
- `[31:0]` **`inj_status_word`**
- `[31:0]` **`lmtap_seam_word`**
- `[31:0]` **`lmtap_guard_word`**
- `[31:0]` **`lmtap_ad_word`**
- `[16*32-1:0]` **`g8_dump_flat`**
- `[1:0]` **`g8_dump_status`**
- `[31:0]` **`batch_b0_lo`**
- `[31:0]` **`batch_b0_hi`**
- `[31:0]` **`batch_sib_lo`**
- `[31:0]` **`batch_sib_hi`**
- `[15:0]` **`sr_read_data`** — special-register bank read data
- **`sr_read_ack`** — bank read committed this cycle

*Outputs:*
- **`s_axi_control_awready`**
- **`s_axi_control_wready`**
- `[1:0]` **`s_axi_control_bresp`**
- **`s_axi_control_bvalid`**
- **`s_axi_control_arready`**
- `[31:0]` **`s_axi_control_rdata`**
- `[1:0]` **`s_axi_control_rresp`**
- **`s_axi_control_rvalid`**
- **`start_pulse`**
- `[31:0]` **`ctx_len`**
- `[31:0]` **`num_frontiers`**
- `[31:0]` **`max_depth`**
- `[63:0]` **`out0_base`**
- `[63:0]` **`out1_base`**
- `[63:0]` **`out2_base`**
- `[63:0]` **`hbm_emb_base`**
- `[63:0]` **`lmfull_hbm_base`**
- `[15:0]` **`n_tiles_cfg`**
- `[15:0]` **`kc_cfg`**
- `[3:0]` **`accfix_cfg`**
- `[10:0]` **`obs_rd_idx`**
- `[15:0]` **`boot_lane_sel`**
- `[16:0]` **`boot_wr_addr`**
- `[31:0]` **`boot_wr_data`**
- **`boot_we`**
- **`boot_is_emb`**
- **`boot_is_hfc`**
- `[2:0]` **`boot_gamma_sel`**
- **`boot_dma_start`**
- **`boot_dma_is_scale`**
- **`boot_dma_is_emb`**
- **`boot_dma_is_hfc`**
- `[63:0]` **`boot_hbm_base`**
- `[COL_W-1:0]` **`boot_dma_n_cols`**
- `[COL_W-1:0]` **`boot_dma_col_start`**
- `[COL_W-1:0]` **`boot_dma_col_step`**
- `[15:0]` **`boot_dma_cpc`**
- `[16:0]` **`boot_dma_stage_base`**
- `[3:0]` **`dbg_sel`**
- `[3:0]` **`probe_sel`**
- **`probe_sel_we`**
- **`probe_addr_we`**
- `[9:0]` **`probe_addr_wdata`**
- **`probe_data_re`**
- **`dump_en`**
- `[63:0]` **`out3_base`**
- `[63:0]` **`out4_base`**
- `[4:0]` **`dbg_cap_stage_sel`**
- **`dbg_cap_widx_we`**
- `[4:0]` **`dbg_cap_widx_wdata`**
- **`dbg_cap_val_re`**
- `[2:0]` **`res_sel`**
- **`res_clear`**
- **`kv_res_clear`**
- `[4:0]` **`sr_read_sel`** — special-register read selector (0x178)
- **`sr_read_en`** — bank read request strobe
- **`inj_en_we`**
- `[31:0]` **`inj_en_wdata`**
- **`inj_widx_we`**
- `[31:0]` **`inj_widx_wdata`**
- **`inj_data_we`**
- `[31:0]` **`inj_data_wdata`**
- **`inj_sel_we`**
- `[31:0]` **`inj_sel_wdata`**
- `[31:0]` **`batch_test_ctrl`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `stage_inject`
- **Source File:** `eagle4_drafter_depthn_1_0/src/stage_inject.sv` (177 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Input activation injection multiplexer. Injects external embedding vectors or token activations into the primary datapath pipeline at stage G1, arbitrating between host-driven prime launches, external DMA streaming, and recursive child-embedding feedback from multi-level speculative iterations.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `NSTAGE` | `19` | `int` |
| `MAX_W` | `128` | `int` |
| `WIDX_W` | `7` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`en_we`**
- `[NSTAGE-1:0]` **`en_wdata`**
- **`widx_we`**
- `[WIDX_W-1:0]` **`widx_wdata`**
- **`data_we`**
- `[31:0]` **`data_wdata`**
- **`sel_we`**
- `[4:0]` **`sel_wdata`**

*Outputs:*
- `[31:0]` **`en_word`**
- `[31:0]` **`widx_word`**
- `[31:0]` **`status_word`**
- `[NSTAGE-1:0]` **`inject_en`**
- `[MAX_W*32-1:0]` **`inj_bus`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `for_reasoning_producer`
- **Source File:** `eagle4_drafter_depthn_1_0/src/for_reasoning_producer.sv` (348 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Speculative tree reasoning coordination unit. Manages handshaking and token streaming between draft tree decoding and external reasoning/verification processors, controlling backpressure and candidate frontier buffering.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `H` | `3072` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `H1_FRAC` | `5` | `int` |
| `G9_FRAC` | `8` | `int` |
| `A_FRAC` | `9` | `int` |
| `KF_BANKS` | `1` | `int` |
| `ABANK_BRAM_P` | `1'b0` | `bit` |
| `SEAM_FIX_P` | `1'b0` | `bit` |
| `REM_W` | `5` | `int` |
| `NCHUNKS` | `H / K_PAR` | `int` |
| `CW` | `(NCHUNKS <= 1` | `int` |
| `BW` | `(KF_BANKS <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`in_valid`**
- `[CW-1:0]` **`in_idx`**
- `signed [K_PAR*A_WIDTH-1:0]` **`h1_chunk`**
- `signed [K_PAR*A_WIDTH-1:0]` **`g9_chunk`**
- **`seam_en`**
- `[K_PAR*REM_W-1:0]` **`rem_chunk`**
- `[BW-1:0]` **`wr_branch`**
- `[BW-1:0]` **`rd_branch`**
- `[CW-1:0]` **`rd_idx`**

*Outputs:*
- **`out_valid`**
- `[CW-1:0]` **`out_idx`**
- `signed [K_PAR*A_WIDTH-1:0]` **`out_chunk`**
- `signed [K_PAR*A_WIDTH-1:0]` **`rd_chunk`**

**Wiring & Hierarchy:**
- **Instantiated By:** *Top-level or variant conditional instantiation*
- **Instantiates:** *(Leaf module)*

---

## 2. Shared Matrix-Vector Multiplication (GEMV) & Weights

### `gemv_stage_array_scaled_slr`
- **Source File:** `eagle4_drafter_depthn_1_0/src/gemv_stage_array_scaled_slr.sv` (1129 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Cross-SLR shared matrix-vector multiplication (GEMV) engine. Dynamically time-multiplexed across transformer linear projection stages (G1 QKV projection, G7G gate projection, G7U up projection, G9 down projection, and G12 LM head). Organizes lanes into groups across Super Logic Regions (SLRs) with integrated pipeline registers for 100 MHz timing closure on Alveo V80.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K` | `64` | `int` |
| `K_PAR` | `16` | `int` |
| `LANES` | `4` | `int` |
| `A_WIDTH` | `16` | `int` |
| `GROUP` | `128` | `int` |
| `SCALE_WIDTH` | `16` | `int` |
| `SCALE_FRAC` | `13` | `int` |
| `RES_WIDTH` | `48` | `int` |
| `BATCH_B` | `1` | `int` |
| `SLR_SAFE` | `1'b0` | `bit` |
| `N_GROUPS` | `3` | `int` |
| `PIPE_STAGES` | `1` | `int` |
| `FF_ACT_CACHE` | `1'b0` | `bit` |
| `PREFETCH_COL` | `1'b0` | `bit` |
| `DUAL_PORT` | `1'b0` | `bit` |
| `WLAT` | `1'b0` | `bit` |
| `N_CHUNKS` | `K / K_PAR` | `int` |
| `CW` | `(N_CHUNKS <= 1` | `int` |
| `N_SGROUPS` | `(K + GROUP - 1` | `int` |
| `GW` | `(N_SGROUPS <= 1` | `int` |
| `GEOM_LANES` | `LANES` | `int` |
| `BANK_DEPTH` | `(96 * 1024 * 128` | `int` |
| `SCALE_DEPTH` | `(16 * 1024 * 128` | `int` |
| `LANES_PER_GRP` | `(LANES + N_GROUPS - 1` | `int` |
| `RET_STAGES` | `(SLR_SAFE && FF_ACT_CACHE` | `int` |
| `RT_DELAY` | `SLR_SAFE ? (FF_ACT_CACHE ? 4 : (2 * PIPE_STAGES` | `int` |
| `BB1` | `(BATCH_B > 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk_b`**
- `signed [BB1*K_PAR*A_WIDTH-1:0]` **`a_chunk_batch`**
- `[1:0]` **`qz_raw`**
- `[3:0]` **`stage_sel`**
- `[15:0]` **`tile_idx`**
- `[15:0]` **`n_cols`**
- `[15:0]` **`lane_we_sel`**
- **`we`**
- `[16:0]` **`wr_addr`**
- `[31:0]` **`wr_data`**
- `[15:0]` **`sc_lane_sel`**
- **`sc_we`**
- `[16:0]` **`sc_wr_addr`**
- `[SCALE_WIDTH-1:0]` **`sc_wr_data`**

*Outputs:*
- `[CW-1:0]` **`chunk_idx_out`**
- `[CW-1:0]` **`chunk_idx_b_out`**
- **`chunk_idx_valid`**
- `signed [32*LANES-1:0]` **`acc_out_per_lane`**
- `signed [RES_WIDTH*LANES-1:0]` **`result_q_per_lane`**
- `signed [BB1*RES_WIDTH*LANES-1:0]` **`result_q_batch`**
- **`done`**
- **`col_done`**
- `[15:0]` **`col_done_idx`**
- **`cols_done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `gemv_lane_scaled_stream` (`gemv_lane_scaled_stream.sv`), `gemv_lane_scaled_stream_dual` (`gemv_lane_scaled_stream_dual.sv`), `scale_bram` (`scale_bram.sv`), `weight_bank_multistage` (`weight_bank_multistage.sv`)

---

### `gemv_stage_array_uram`
- **Source File:** `eagle4_drafter_depthn_1_0/src/gemv_stage_array_uram.sv` (131 lines)
- **Construct Type:** `module`
- **Technical Purpose:** URAM-backed GEMV stage array variant. Synthesizes dense matrix-vector dot products utilizing UltraRAM blocks as high-density on-chip weight storage, supporting wide word reads and low-latency activation streaming.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K` | `64` | `int` |
| `K_PAR` | `16` | `int` |
| `LANES` | `4` | `int` |
| `A_WIDTH` | `16` | `int` |
| `N_CHUNKS` | `K / K_PAR` | `int` |
| `CW` | `(N_CHUNKS <= 1` | `int` |
| `BANK_DEPTH` | `96 * 1024` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `[1:0]` **`qz_raw`**
- `[3:0]` **`stage_sel`**
- `[15:0]` **`tile_idx`**
- `[15:0]` **`lane_we_sel`**
- **`we`**
- `[16:0]` **`wr_addr`**
- `[31:0]` **`wr_data`**

*Outputs:*
- `[CW-1:0]` **`chunk_idx_out`**
- **`chunk_idx_valid`**
- `signed [32*LANES-1:0]` **`acc_out_per_lane`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `gemv_lane_int_stream_wallace` (`gemv_lane_int_stream_wallace.sv`), `weight_bank_multistage` (`weight_bank_multistage.sv`)

---

### `gemv_stage_array_uram_kpar64`
- **Source File:** `eagle4_drafter_depthn_1_0/src/gemv_stage_array_uram_kpar64.sv` (133 lines)
- **Construct Type:** `module`
- **Technical Purpose:** High-parallelism (K_PAR=64) URAM GEMV stage array. Expands parallel dot-product reduction from 16 to 64 parallel product terms per cycle per lane, accelerating matrix operations for high-throughput configurations.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K` | `64` | `int` |
| `K_PAR` | `64` | `int` |
| `LANES` | `4` | `int` |
| `A_WIDTH` | `16` | `int` |
| `N_SG` | `K_PAR / 16` | `int` |
| `ROW_W` | `N_SG * 32` | `int` |
| `N_CHUNKS` | `K / K_PAR` | `int` |
| `CW` | `(N_CHUNKS <= 1` | `int` |
| `ROW_DEPTH` | `24 * 1024` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `[1:0]` **`qz_raw`**
- `[3:0]` **`stage_sel`**
- `[15:0]` **`tile_idx`**
- `[15:0]` **`lane_we_sel`**
- **`we`**
- `[16:0]` **`wr_row`**
- `[ROW_W-1:0]` **`wr_row_data`**

*Outputs:*
- `[CW-1:0]` **`chunk_idx_out`**
- **`chunk_idx_valid`**
- `signed [32*LANES-1:0]` **`acc_out_per_lane`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `gemv_lane_kpar_wallace` (`gemv_lane_kpar_wallace.sv`), `weight_bank_multistage_wide` (`weight_bank_multistage_wide.sv`)

---

### `gemv_lane_scaled_stream`
- **Source File:** `eagle4_drafter_depthn_1_0/src/gemv_lane_scaled_stream.sv` (418 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Single-channel scaled GEMV dot-product compute lane. Streams K-dimensional activation vectors against dequantized weight vectors, accumulates partial sums via multi-operand tree reduction, and applies per-channel fixed-point quantization scaling factors from scale_bram.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K` | `64` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `GROUP` | `128` | `int` |
| `SCALE_WIDTH` | `16` | `int` |
| `SCALE_FRAC` | `15` | `int` |
| `RES_WIDTH` | `48` | `int` |
| `SCALED` | `1'b1` | `bit` |
| `WLAT` | `1'b0` | `bit` |
| `N_CHUNKS` | `K / K_PAR` | `int` |
| `CW` | `(N_CHUNKS <= 1` | `int` |
| `CPG` | `GROUP / K_PAR` | `int` |
| `N_GROUPS` | `(K + GROUP - 1` | `int` |
| `GW` | `(N_GROUPS <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `[31:0]` **`qw_word`**
- `[1:0]` **`qz_raw`**
- `signed [SCALE_WIDTH-1:0]` **`scale_q`**
- `[CW-1:0]` **`n_chunks_run`**

*Outputs:*
- `[CW-1:0]` **`chunk_idx_out`**
- **`chunk_idx_valid`**
- `[CW-1:0]` **`chunk_idx_next_out`**
- `[GW-1:0]` **`group_idx_out`**
- `[GW-1:0]` **`group_idx_next_out`**
- `signed [31:0]` **`acc_out`**
- `signed [RES_WIDTH-1:0]` **`result_q`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `gemv_stage_array_scaled_slr` (`gemv_stage_array_scaled_slr.sv`)
- **Instantiates:** `csa_tree16_int20` (`csa_tree16_int20.sv`), `gptq2_dequant16` (`gptq2_dequant16.sv`)

---

### `gemv_lane_scaled_stream_dual`
- **Source File:** `eagle4_drafter_depthn_1_0/src/gemv_lane_scaled_stream_dual.sv` (405 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Dual-channel scaled GEMV compute lane. Implements dual parallel dot-product paths per lane (supporting simultaneous G7G gate and G7U up projections or dual-head operations), doubling compute density while sharing weight memory addressing.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K` | `64` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `GROUP` | `128` | `int` |
| `SCALE_WIDTH` | `16` | `int` |
| `SCALE_FRAC` | `15` | `int` |
| `RES_WIDTH` | `48` | `int` |
| `WLAT` | `1'b0` | `bit` |
| `N_CHUNKS` | `K / K_PAR` | `int` |
| `CW` | `(N_CHUNKS <= 1` | `int` |
| `CPG` | `GROUP / K_PAR` | `int` |
| `N_GROUPS` | `(K + GROUP - 1` | `int` |
| `GW` | `(N_GROUPS <= 1` | `int` |
| `N_PAIRS` | `(N_CHUNKS + 1` | `int` |
| `PW` | `(N_PAIRS <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk_b`**
- `[31:0]` **`qw_word`**
- `[31:0]` **`qw_word_b`**
- `[1:0]` **`qz_raw`**
- `signed [SCALE_WIDTH-1:0]` **`scale_q`**
- `[CW-1:0]` **`n_chunks_run`**

*Outputs:*
- `[CW-1:0]` **`chunk_idx_out`**
- `[CW-1:0]` **`chunk_idx_b_out`**
- **`chunk_idx_valid`**
- `[CW-1:0]` **`chunk_idx_next_out`**
- `[CW-1:0]` **`chunk_idx_b_next_out`**
- `[GW-1:0]` **`group_idx_out`**
- `[GW-1:0]` **`group_idx_next_out`**
- `signed [31:0]` **`acc_out`**
- `signed [RES_WIDTH-1:0]` **`result_q`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `gemv_stage_array_scaled_slr` (`gemv_stage_array_scaled_slr.sv`)
- **Instantiates:** `csa_tree16_int20` (`csa_tree16_int20.sv`), `gptq2_dequant16` (`gptq2_dequant16.sv`)

---

### `gemv_lane_int_stream_wallace`
- **Source File:** `eagle4_drafter_depthn_1_0/src/gemv_lane_int_stream_wallace.sv` (180 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Integer streaming GEMV compute lane with Wallace tree reduction. Utilizes carry-save addition to reduce parallel integer product terms with minimal carry-propagation latency before final accumulation.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K` | `64` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `N_CHUNKS` | `K / K_PAR` | `int` |
| `CW` | `(N_CHUNKS <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `[31:0]` **`qw_word`**
- `[1:0]` **`qz_raw`**

*Outputs:*
- `[CW-1:0]` **`chunk_idx_out`**
- **`chunk_idx_valid`**
- `[CW-1:0]` **`chunk_idx_next_out`**
- `signed [31:0]` **`acc_out`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `gemv_stage_array_uram` (`gemv_stage_array_uram.sv`)
- **Instantiates:** `csa_tree16_int20` (`csa_tree16_int20.sv`), `gptq2_dequant16` (`gptq2_dequant16.sv`)

---

### `gemv_lane_kpar_wallace`
- **Source File:** `eagle4_drafter_depthn_1_0/src/gemv_lane_kpar_wallace.sv` (223 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Parameterized K-parallel Wallace tree reduction lane. Implements a high-speed tree adder compressing K_PAR input product operands into redundant carry-save representation followed by a fast final adder.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K` | `3072` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `N_SG` | `K_PAR / 16` | `int` |
| `N_CHUNKS` | `K / K_PAR` | `int` |
| `CW` | `(N_CHUNKS <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `[N_SG*32-1:0]` **`qw_word`**
- `[1:0]` **`qz_raw`**

*Outputs:*
- `[CW-1:0]` **`chunk_idx_out`**
- **`chunk_idx_valid`**
- `[CW-1:0]` **`chunk_idx_next_out`**
- `signed [31:0]` **`acc_out`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `gemv_stage_array_uram_kpar64` (`gemv_stage_array_uram_kpar64.sv`)
- **Instantiates:** `csa_tree16_int20` (`csa_tree16_int20.sv`), `gptq2_dequant16` (`gptq2_dequant16.sv`)

---

### `csa_tree16_int20`
- **Source File:** `eagle4_drafter_depthn_1_0/src/csa_tree16_int20.sv` (170 lines)
- **Construct Type:** `module`
- **Technical Purpose:** 16-input Carry-Save Adder (CSA) reduction tree with 20-bit accumulator precision. Compresses 16 signed product terms into sum and carry vectors, eliminating carry propagation across intermediate stages.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`in_valid`**
- `signed [19:0]` **`prods [0:15]`**

*Outputs:*
- `signed [25:0]` **`sum_out`**
- **`out_valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attn_score_dot` (`attn_score_dot.sv`), `gemv_lane_int_stream_wallace` (`gemv_lane_int_stream_wallace.sv`), `gemv_lane_kpar_wallace` (`gemv_lane_kpar_wallace.sv`), `gemv_lane_scaled_stream` (`gemv_lane_scaled_stream.sv`), `gemv_lane_scaled_stream_dual` (`gemv_lane_scaled_stream_dual.sv`)
- **Instantiates:** *(Leaf module)*

---

### `csa_tree16_int28`
- **Source File:** `eagle4_drafter_depthn_1_0/src/csa_tree16_int28.sv` (159 lines)
- **Construct Type:** `module`
- **Technical Purpose:** 16-input Carry-Save Adder tree with extended 28-bit accumulator precision. Provides wide-dynamic-range operand reduction for late accumulation stages, preventing arithmetic overflow on high-dimensional vector dot products.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`in_valid`**
- `signed [31:0]` **`prods [0:15]`**

*Outputs:*
- `signed [31:0]` **`sum_out`**
- **`out_valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attn_score_dot` (`attn_score_dot.sv`)
- **Instantiates:** *(Leaf module)*

---

### `gptq2_dequant16`
- **Source File:** `eagle4_drafter_depthn_1_0/src/gptq2_dequant16.sv` (66 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Hardware dequantizer for 2-bit GPTQ (Generalized Post-Training Quantization) weights. Unpacks 2-bit quantized weight indices and maps them to 16-bit fixed-point activations using per-group scale and zero-point parameters.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- `[31:0]` **`qw_word`**
- `[31:0]` **`qz_word`**
- `[3:0]` **`n_idx`**

*Outputs:*
- `[47:0]` **`w_s_packed`**

**Wiring & Hierarchy:**
- **Instantiated By:** `gemv_lane_int_stream_wallace` (`gemv_lane_int_stream_wallace.sv`), `gemv_lane_kpar_wallace` (`gemv_lane_kpar_wallace.sv`), `gemv_lane_scaled_stream` (`gemv_lane_scaled_stream.sv`), `gemv_lane_scaled_stream_dual` (`gemv_lane_scaled_stream_dual.sv`)
- **Instantiates:** *(Leaf module)*

---

### `weight_bank_multistage`
- **Source File:** `eagle4_drafter_depthn_1_0/src/weight_bank_multistage.sv` (165 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Multistage URAM weight storage bank. Stores quantized weights for multiple pipeline stages (G1, G7, G9, G12) in unified UltraRAM blocks, translating stage-select signals into stage-base addresses using pre-compiled geometric offsets from stage_geometry.svh.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `128` | `int` |
| `DEPTH` | `(96 * 1024 * 128` | `int` |
| `DATA_W` | `32` | `int` |
| `RD_PORTS` | `1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`we`**
- `[16:0]` **`wr_addr`**
- `[DATA_W-1:0]` **`wr_data`**
- `[3:0]` **`stage_sel`**
- `[15:0]` **`col_offset`**
- `[15:0]` **`chunk_idx`**
- `[15:0]` **`chunk_idx_b`**

*Outputs:*
- `[DATA_W-1:0]` **`rd_data`**
- `[DATA_W-1:0]` **`rd_data_b`**

**Wiring & Hierarchy:**
- **Instantiated By:** `gemv_stage_array_scaled_slr` (`gemv_stage_array_scaled_slr.sv`), `gemv_stage_array_uram` (`gemv_stage_array_uram.sv`)
- **Instantiates:** `weight_bank_uram` (`weight_bank_uram.sv`)

---

### `weight_bank_multistage_wide`
- **Source File:** `eagle4_drafter_depthn_1_0/src/weight_bank_multistage_wide.sv` (156 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Wide-data-path multistage URAM weight bank. Features widened read output ports to saturate parallel compute lanes across multiple SLR groups simultaneously.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K_PAR` | `64` | `int` |
| `N_SG` | `K_PAR / 16` | `int` |
| `ROW_W` | `N_SG * 32` | `int` |
| `ROW_DEPTH` | `24 * 1024` | `int` |
| `RADDR_W` | `(ROW_DEPTH <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`we`**
- `[RADDR_W-1:0]` **`wr_row`**
- `[ROW_W-1:0]` **`wr_row_data`**
- `[3:0]` **`stage_sel`**
- `[15:0]` **`col_offset`**
- `[15:0]` **`chunk_idx`**

*Outputs:*
- `[ROW_W-1:0]` **`rd_row`**

**Wiring & Hierarchy:**
- **Instantiated By:** `gemv_stage_array_uram_kpar64` (`gemv_stage_array_uram_kpar64.sv`)
- **Instantiates:** `weight_bank_uram` (`weight_bank_uram.sv`)

---

### `weight_bank_uram`
- **Source File:** `eagle4_drafter_depthn_1_0/src/weight_bank_uram.sv` (155 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Base UltraRAM weight memory primitive wrapper. Configures native Xilinx URAM288 primitives into deep, wide dual-port memories with integrated output registers and optional simulation-time snapshot save/restore hooks (SIM_WSNAP).

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `DEPTH` | `4096` | `int` |
| `DATA_W` | `32` | `int` |
| `RD_PORTS` | `1` | `int` |
| `ADDR_W` | `(DEPTH <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`we`**
- `[ADDR_W-1:0]` **`wr_addr`**
- `[DATA_W-1:0]` **`wr_data`**
- `[ADDR_W-1:0]` **`rd_addr`**
- `[ADDR_W-1:0]` **`rd_addr_b`**

*Outputs:*
- `[DATA_W-1:0]` **`rd_data`**
- `[DATA_W-1:0]` **`rd_data_b`**

**Wiring & Hierarchy:**
- **Instantiated By:** `weight_bank_multistage` (`weight_bank_multistage.sv`), `weight_bank_multistage_wide` (`weight_bank_multistage_wide.sv`)
- **Instantiates:** *(Leaf module)*

---

### `scale_bram`
- **Source File:** `eagle4_drafter_depthn_1_0/src/scale_bram.sv` (160 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Block RAM storage array for quantization scale factors. Stores per-channel and per-group fixed-point scaling coefficients, providing synchronized scale factor lookups aligned with weight reads for the scaled GEMV lanes.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `128` | `int` |
| `DEPTH` | `(16 * 1024 * 128` | `int` |
| `SCALE_W` | `16` | `int` |
| `ADDR_W` | `17` | `int` |
| `GW` | `7` | `int` |
| `GROUP` | `128` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`we`**
- `[ADDR_W-1:0]` **`wr_addr`**
- `[SCALE_W-1:0]` **`wr_data`**
- `[3:0]` **`stage_sel`**
- `[15:0]` **`col_offset`**
- `[GW-1:0]` **`group_idx`**

*Outputs:*
- `[SCALE_W-1:0]` **`rd_data`**

**Wiring & Hierarchy:**
- **Instantiated By:** `gemv_stage_array_scaled_slr` (`gemv_stage_array_scaled_slr.sv`)
- **Instantiates:** *(Leaf module)*

---

### `weight_dma_loader`
- **Source File:** `eagle4_drafter_depthn_1_0/src/weight_dma_loader.sv` (653 lines)
- **Construct Type:** `module`
- **Technical Purpose:** High-throughput AXI4 memory loader engine. Drives AXI master burst reads to pull model weight matrices from external HBM/DDR memory and stream them directly into on-chip URAM/BRAM banks during system boot and model reconfiguration.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `128` | `int` |
| `AXI_ADDR_W` | `64` | `int` |
| `AXI_DATA_W` | `32` | `int` |
| `AXI_ID_W` | `1` | `int` |
| `BURST_LEN` | `64` | `int` |
| `URAM_ADDR_W` | `17` | `int` |
| `SCALE_ADDR_W` | `17` | `int` |
| `SCALE_W` | `16` | `int` |
| `COL_W` | `20` | `int` |
| `BOOT_RWAIT_MAX` | `(1 << 20` | `int` |
| `RWAIT_W` | `21` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`boot_dma_start`**
- **`boot_is_scale`**
- **`boot_is_emb`**
- **`boot_is_hfc`**
- `[AXI_ADDR_W-1:0]` **`boot_hbm_base`**
- `[COL_W-1:0]` **`boot_n_cols`**
- `[COL_W-1:0]` **`boot_col_start`**
- `[COL_W-1:0]` **`boot_col_step`**
- `[15:0]` **`boot_cpc`**
- `[URAM_ADDR_W-1:0]` **`boot_stage_base`**
- **`m_axi_boot_arready`**
- `[AXI_ID_W-1:0]` **`m_axi_boot_rid`**
- `[AXI_DATA_W-1:0]` **`m_axi_boot_rdata`**
- `[1:0]` **`m_axi_boot_rresp`**
- **`m_axi_boot_rlast`**
- **`m_axi_boot_rvalid`**

*Outputs:*
- **`boot_busy`**
- **`boot_done`**
- **`boot_timeout`**
- **`dma_tgt_emb`**
- **`dma_tgt_hfc`**
- **`dma_we`**
- `[15:0]` **`dma_lane_sel`**
- `[URAM_ADDR_W-1:0]` **`dma_wr_addr`**
- `[AXI_DATA_W-1:0]` **`dma_wr_data`**
- **`sc_we`**
- `[15:0]` **`sc_lane_sel`**
- `[SCALE_ADDR_W-1:0]` **`sc_wr_addr`**
- `[SCALE_W-1:0]` **`sc_wr_data`**
- `[AXI_ID_W-1:0]` **`m_axi_boot_arid`**
- `[AXI_ADDR_W-1:0]` **`m_axi_boot_araddr`**
- `[7:0]` **`m_axi_boot_arlen`**
- `[2:0]` **`m_axi_boot_arsize`**
- `[1:0]` **`m_axi_boot_arburst`**
- **`m_axi_boot_arvalid`**
- **`m_axi_boot_rready`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

## 3. Normalization, Activation & Non-Linear Functions

### `rmsnorm_block`
- **Source File:** `eagle4_drafter_depthn_1_0/src/rmsnorm_block.sv` (409 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Composite Root-Mean-Square Normalization (RMSNorm) engine. Sequentially executes sum-of-squares accumulation, reciprocal square-root scaling, and gamma weight modulation under a single unified start/done handshake.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `H` | `3072` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `SUMSQ_RSHIFT` | `0` | `int` |
| `SUMSQ_MULT` | `0` | *untyped* |
| `SUMSQ_SHR` | `0` | `int` |
| `SCALE_SHR` | `12` | `int` |
| `XS_WIDTH` | `A_WIDTH` | `int` |
| `RDATA_LAT` | `0` | `int` |
| `SCALE_WIDTH` | `16` | `int` |
| `SCALE_EPS_BUCKET` | `1` | *untyped* |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`x_chunk`**
- `signed [K_PAR*A_WIDTH-1:0]` **`gamma_chunk`**

*Outputs:*
- `signed [K_PAR*A_WIDTH-1:0]` **`out_chunk`**
- **`out_valid`**
- **`done`**
- `[$clog2((H/K_PAR)+1)-1:0]` **`x_req_chunk`**
- **`x_req_valid`**
- **`out_chunk_strobe`**
- `[$clog2((H/K_PAR)+1)-1:0]` **`out_chunk_idx_o`**
- **`apply_en_o`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `rmsnorm_apply` (`rmsnorm_apply.sv`), `rmsnorm_scale_int` (`rmsnorm_scale_int.sv`), `rmsnorm_sumsq` (`rmsnorm_sumsq.sv`)

---

### `rmsnorm_sumsq`
- **Source File:** `eagle4_drafter_depthn_1_0/src/rmsnorm_sumsq.sv` (87 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Streaming sum-of-squares vector accumulator. Computes sum(x_i^2) across H-dimensional activation vectors using fixed-point multipliers and saturation logic.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`clear`**
- **`en`**
- `signed [K_PAR*A_WIDTH-1:0]` **`x_chunk`**

*Outputs:*
- `[63:0]` **`sumsq_out`**

**Wiring & Hierarchy:**
- **Instantiated By:** `rmsnorm_block` (`rmsnorm_block.sv`)
- **Instantiates:** *(Leaf module)*

---

### `rmsnorm_apply`
- **Source File:** `eagle4_drafter_depthn_1_0/src/rmsnorm_apply.sv` (173 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Elementwise normalization modulation unit. Multiplies input activations by the computed normalization scale factor and per-channel learned gain parameters (Gamma).

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `SCALE_SHR` | `12` | `int` |
| `XS_WIDTH` | `A_WIDTH` | `int` |
| `SCALE_WIDTH` | `16` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`en`**
- `signed [SCALE_WIDTH-1:0]` **`scale_q12`**
- `signed [K_PAR*A_WIDTH-1:0]` **`x_chunk`**
- `signed [K_PAR*A_WIDTH-1:0]` **`gamma_chunk`**

*Outputs:*
- `signed [K_PAR*A_WIDTH-1:0]` **`out_chunk`**
- **`valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `rmsnorm_block` (`rmsnorm_block.sv`)
- **Instantiates:** *(Leaf module)*

---

### `rmsnorm_scale_int`
- **Source File:** `eagle4_drafter_depthn_1_0/src/rmsnorm_scale_int.sv` (189 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Fixed-point normalization scale factor generator. Takes accumulated sum-of-squares, applies inv_q12 reciprocal square-root approximation, and computes the integer normalization scaling factor.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `SCALE_WIDTH` | `16` | `int` |
| `EPS_BUCKET` | `1` | *untyped* |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`resetn`**
- **`start`**
- `[63:0]` **`sumsq`**

*Outputs:*
- **`busy`**
- **`done`**
- `signed [SCALE_WIDTH-1:0]` **`scale_q12`**

**Wiring & Hierarchy:**
- **Instantiated By:** `rmsnorm_block` (`rmsnorm_block.sv`)
- **Instantiates:** *(Leaf module)*

---

### `inv_q12`
- **Source File:** `eagle4_drafter_depthn_1_0/src/inv_q12.sv` (121 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Q12 fixed-point reciprocal and reciprocal square-root approximation module. Uses piecewise-linear interpolation and Newton-Raphson refinement steps to compute 1/sqrt(x) and 1/x with low cycle latency.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [47:0]` **`sum_q12`**

*Outputs:*
- **`busy`**
- **`done`**
- `signed [15:0]` **`inv_q12_out`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_head_assembly` (`attention_head_assembly.sv`), `attention_head_assembly_perbank_sbuf` (`attention_head_assembly_perbank_sbuf.sv`)
- **Instantiates:** *(Leaf module)*

---

### `silu_layer`
- **Source File:** `eagle4_drafter_depthn_1_0/src/silu_layer.sv` (84 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Streaming SiLU (Sigmoid Linear Unit, x * sigmoid(x)) activation module. Computes non-linear activation functions over streaming vectors using internal piecewise-linear sigmoid approximations.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `OUT_FRAC` | `12` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`en`**
- `signed [K_PAR*A_WIDTH-1:0]` **`gate_chunk`**
- `signed [K_PAR*A_WIDTH-1:0]` **`up_chunk`**

*Outputs:*
- `signed [K_PAR*A_WIDTH-1:0]` **`out_chunk`**
- **`valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `silu_mul` (`silu_mul.sv`)

---

### `silu_layer_fix`
- **Source File:** `eagle4_drafter_depthn_1_0/src/silu_layer_fix.sv` (51 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Fixed-latency optimized SiLU activation pipeline. Features deterministic multi-stage register balancing to ensure timing closure at 100 MHz on wide vector words.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K_PAR` | `16` | `int` |
| `IN_W` | `18` | `int` |
| `OUT_FRAC` | `8` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`en`**
- `signed [K_PAR*IN_W-1:0]` **`gate_chunk`**
- `signed [K_PAR*IN_W-1:0]` **`up_chunk`**

*Outputs:*
- `signed [K_PAR*16-1:0]` **`out_chunk`**
- **`valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `silu_mul_fix` (`silu_mul_fix.sv`)

---

### `silu_mul`
- **Source File:** `eagle4_drafter_depthn_1_0/src/silu_mul.sv` (122 lines)
- **Construct Type:** `module`
- **Technical Purpose:** SwiGLU activation and multiplication unit. Computes the elementwise product of SiLU-activated gate projections and linear up projections: SiLU(G7G) * G7U.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `OUT_FRAC` | `12` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`en`**
- `signed [15:0]` **`gate_q`**
- `signed [15:0]` **`up_q`**

*Outputs:*
- `signed [15:0]` **`out_q`**
- **`valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `silu_layer` (`silu_layer.sv`)
- **Instantiates:** `sigmoid_pwl_q12` (`sigmoid_pwl_q12.sv`)

---

### `silu_mul_fix`
- **Source File:** `eagle4_drafter_depthn_1_0/src/silu_mul_fix.sv` (87 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Fixed-latency pipelined SwiGLU multiplier. Aligns gate and up projection streams and computes saturated fixed-point vector products with zero-bubble throughput.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `OUT_FRAC` | `8` | `int` |
| `IN_W` | `18` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`en`**
- `signed [IN_W-1:0]` **`gate_q`**
- `signed [IN_W-1:0]` **`up_q`**

*Outputs:*
- `signed [15:0]` **`out_q`**
- **`valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `silu_layer_fix` (`silu_layer_fix.sv`)
- **Instantiates:** `sigmoid_pwl512_q12` (`sigmoid_pwl512_q12.sv`)

---

### `sigmoid_pwl_q12`
- **Source File:** `eagle4_drafter_depthn_1_0/src/sigmoid_pwl_q12.sv` (107 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Piecewise-linear (PWL) Sigmoid function calculator in Q12 fixed-point. Maps 16-bit signed inputs to [0, 1] using symmetric multi-segment linear approximations.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- `signed [15:0]` **`x_q12`**

*Outputs:*
- `[15:0]` **`sigmoid_q12`**

**Wiring & Hierarchy:**
- **Instantiated By:** `silu_mul` (`silu_mul.sv`)
- **Instantiates:** *(Leaf module)*

---

### `sigmoid_pwl512_q12`
- **Source File:** `eagle4_drafter_depthn_1_0/src/sigmoid_pwl512_q12.sv` (540 lines)
- **Construct Type:** `module`
- **Technical Purpose:** 512-segment high-precision piecewise-linear Sigmoid evaluator. Provides high-fidelity non-linear activation curve approximation for numerical sensitivity.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- `signed [17:0]` **`x_q12`**

*Outputs:*
- `[15:0]` **`sigmoid_q12`**

**Wiring & Hierarchy:**
- **Instantiated By:** `silu_mul_fix` (`silu_mul_fix.sv`)
- **Instantiates:** *(Leaf module)*

---

### `exp_pwl_q12`
- **Source File:** `eagle4_drafter_depthn_1_0/src/exp_pwl_q12.sv` (172 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Piecewise-linear fixed-point exponential function (e^x) evaluator in Q12 format. Generates exponentiated values for attention Softmax calculations without floating-point DSP overhead.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- `signed [31:0]` **`delta_q12`**

*Outputs:*
- `[31:0]` **`exp_q12`**

**Wiring & Hierarchy:**
- **Instantiated By:** `softmax_exp_sum` (`softmax_exp_sum.sv`)
- **Instantiates:** *(Leaf module)*

---

### `residual_add`
- **Source File:** `eagle4_drafter_depthn_1_0/src/residual_add.sv` (248 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Pipelined elementwise residual addition unit. Adds skip-connection activations to stage outputs with signed saturation and overflow clipping to maintain dynamic range stability.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `H` | `3072` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `FRAC_BITS` | `4` | `int` |
| `RDATA_LAT` | `0` | `int` |
| `NCHUNKS` | `H / K_PAR` | `int` |
| `CW` | `(NCHUNKS > 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`a_valid`**
- `[CW-1:0]` **`a_chunk_idx`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`b_chunk`**
- **`b_valid`**

*Outputs:*
- **`out_valid`**
- `[CW-1:0]` **`out_chunk_idx`**
- `signed [K_PAR*A_WIDTH-1:0]` **`out_chunk`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** *Top-level or variant conditional instantiation*
- **Instantiates:** *(Leaf module)*

---

## 4. Attention & KV Cache Memory Subsystem

### `attention_head_assembly`
- **Source File:** `eagle4_drafter_depthn_1_0/src/attention_head_assembly.sv` (520 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Top-level Multi-Head Attention (MHA) orchestration module. Coordinates Query, Key, and Value vector distribution across multiple attention heads, sequencing score computation, softmax, and context reduction.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `D_HEAD` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `CTX_MAX` | `32` | `int` |
| `SCORE_LATENCY` | `4` | `int` |
| `SCORE_NORM_SHR` | `0` | `int` |
| `DEPTH_TEMP_FIX` | `0` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[10:0]` **`ctx_len`**
- `signed [A_WIDTH*D_HEAD-1:0]` **`q_vec`**
- `signed [A_WIDTH*D_HEAD-1:0]` **`k_vec_in`**
- `signed [A_WIDTH*D_HEAD-1:0]` **`v_vec_in`**

*Outputs:*
- **`kv_rd_en`**
- **`kv_rd_kind`**
- `[10:0]` **`kv_rd_pos`**
- `signed [32*D_HEAD-1:0]` **`out_vec`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_full_mha_sbuf` (`attention_full_mha_sbuf.sv`), `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `attn_score_dot` (`attn_score_dot.sv`), `attn_v_combine` (`attn_v_combine.sv`), `inv_q12` (`inv_q12.sv`), `score_max_tracker` (`score_max_tracker.sv`), `softmax_exp_sum` (`softmax_exp_sum.sv`)

---

### `attention_head_assembly_perbank_sbuf`
- **Source File:** `eagle4_drafter_depthn_1_0/src/attention_head_assembly_perbank_sbuf.sv` (811 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Banked static-buffer Multi-Head Attention assembly. Implements dedicated per-bank static buffers for Q, K, and V vectors, eliminating memory arbitration stalls during attention matrix evaluation.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `D_HEAD` | `128` | `int` |
| `A_WIDTH` | `16` | `int` |
| `CTX_MAX` | `1024` | `int` |
| `SCORE_LATENCY` | `13` | `int` |
| `SCORE_NORM_SHR` | `0` | `int` |
| `DEPTH_TEMP_FIX` | `0` | `int` |
| `N_STACKS` | `6` | `int` |
| `SSEL_W` | `(N_STACKS <= 1` | `int` |
| `KV_LANE_CHAIN` | `0` | `int` |
| `K_F` | `8` | `int` |
| `CTX_WIN` | `1024` | `int` |
| `KV_ADDR_REP` | `1` | `int` |
| `RES_CLR_REP` | `1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[10:0]` **`ctx_len`**
- `signed [A_WIDTH*D_HEAD-1:0]` **`q_vec`**
- `[SSEL_W-1:0]` **`stack_sel`**
- `[3:0]` **`lane_j`**
- `[10:0]` **`num_steps`**
- `[10:0]` **`level_i`**
- `signed [A_WIDTH*D_HEAD-1:0]` **`k_vec_in`**
- `signed [N_STACKS*A_WIDTH*D_HEAD-1:0]` **`v_rd_bus_flat`**
- **`kv_wr_v`**
- `[10:0]` **`kv_wr_slot`**
- **`kv_res_clear`**
- **`b2p_round_en`**

*Outputs:*
- **`kv_rd_en`**
- **`kv_rd_kind`**
- `[N_STACKS*11-1:0]` **`k_rd_pos_rep`**
- `[N_STACKS*KV_ADDR_REP*11-1:0]` **`v_rd_pos_rep`**
- `[11:0]` **`kv_res_cnt`**
- `signed [32*D_HEAD-1:0]` **`out_vec`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_full_mha_sbuf` (`attention_full_mha_sbuf.sv`)
- **Instantiates:** `attn_score_dot` (`attn_score_dot.sv`), `attn_v_combine_perbank` (`attn_v_combine_perbank.sv`), `inv_q12` (`inv_q12.sv`), `score_max_tracker` (`score_max_tracker.sv`), `softmax_exp_sum` (`softmax_exp_sum.sv`)

---

### `attention_full_mha_sbuf`
- **Source File:** `eagle4_drafter_depthn_1_0/src/attention_full_mha_sbuf.sv` (917 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Monolithic multi-head attention processor with static SRAM buffers. Integrates full self-attention forward pass (Q*K^T scoring, Softmax normalization, and Score*V aggregation) into a self-contained execution pipeline.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `N_HEADS` | `24` | `int` |
| `D_HEAD` | `128` | `int` |
| `A_WIDTH` | `16` | `int` |
| `CTX_WIN` | `1024` | `int` |
| `HEADS_PER_STACK` | `4` | `int` |
| `CTX_MAX` | `1024` | `int` |
| `SCORE_LATENCY` | `13` | `int` |
| `SCORE_NORM_SHR` | `0` | `int` |
| `K_RAM_STYLE` | `"ultra"` | *untyped* |
| `V_RAM_STYLE` | `"block"` | *untyped* |
| `COMBINE_LOCAL` | `1'b1` | `bit` |
| `RAW_REQUANT` | `1'b0` | `bit` |
| `HEAD_PAR` | `1` | `int` |
| `G4_FRAC` | `12` | `int` |
| `G5_FRAC` | `10` | `int` |
| `ROPE_POS_DECOUPLE` | `1'b0` | `bit` |
| `KV_LANE_CHAIN` | `(ROPE_POS_DECOUPLE ? 1 : 0` | `int` |
| `K_F` | `8` | `int` |
| `ROPE_CS_PIPE` | `1'b0` | `bit` |
| `ROPE_POS_REP` | `1` | `int` |
| `KV_ADDR_REP` | `1` | `int` |
| `RES_CLR_REP` | `1` | `int` |
| `ROW_W` | `D_HEAD * A_WIDTH` | `int` |
| `OUT_W` | `N_HEADS * D_HEAD * A_WIDTH` | `int` |
| `HEAD_W` | `(N_HEADS <= 1` | `int` |
| `KVHEAD_W` | `(N_HEADS <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[10:0]` **`ctx_len`**
- `[10:0]` **`append_pos`**
- `[10:0]` **`rope_pos_in`**
- `[10:0]` **`ctx_len_prefix`**
- `[3:0]` **`lane_j`**
- `[10:0]` **`num_steps`**
- `[10:0]` **`level_i`**
- `signed [OUT_W-1:0]` **`q_in`**
- `signed [OUT_W-1:0]` **`k_in`**
- `signed [OUT_W-1:0]` **`v_in`**
- `signed [32*D_HEAD-1:0]` **`q_raw_buf [0:N_HEADS-1]`**
- `signed [32*D_HEAD-1:0]` **`k_raw_buf [0:N_HEADS-1]`**
- `signed [32*D_HEAD-1:0]` **`v_raw_buf [0:N_HEADS-1]`**
- **`pl_we`**
- `[KVHEAD_W-1:0]` **`pl_head`**
- `[10:0]` **`pl_pos`**
- `signed [ROW_W-1:0]` **`pl_k_vec`**
- `signed [ROW_W-1:0]` **`pl_v_vec`**
- **`do_kv_fill`**
- **`kv_res_clear`**
- **`b2p_round_en`**

*Outputs:*
- **`done`**
- `signed [OUT_W-1:0]` **`out_vec`**
- `[HEAD_W-1:0]` **`dbg_head`**
- `[11:0]` **`kv_res_cnt`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `attention_head_assembly` (`attention_head_assembly.sv`), `attention_head_assembly_perbank_sbuf` (`attention_head_assembly_perbank_sbuf.sv`), `multihead_kv` (`multihead_kv.sv`), `rope_layer` (`rope_layer.sv`)

---

### `multihead_kv`
- **Source File:** `eagle4_drafter_depthn_1_0/src/multihead_kv.sv` (293 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Multi-head Key-Value cache manager. Generates write addresses, manages circular context buffer pointers, and orchestrates parallel read bursts from distributed KV memory banks across context history.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `N_KV` | `24` | `int` |
| `D_HEAD` | `128` | `int` |
| `A_WIDTH` | `16` | `int` |
| `CTX_WIN` | `1024` | `int` |
| `HEADS_PER_STACK` | `4` | `int` |
| `K_RAM_STYLE` | `"ultra"` | *untyped* |
| `V_RAM_STYLE` | `"block"` | *untyped* |
| `ROW_W` | `D_HEAD * A_WIDTH` | `int` |
| `N_STACKS` | `(N_KV + HEADS_PER_STACK - 1` | `int` |
| `HSEL_W` | `(HEADS_PER_STACK <= 1` | `int` |
| `HEAD_W` | `(N_KV <= 1` | `int` |
| `POS_REP` | `0` | `int` |
| `WR_POS_REP` | `0` | `int` |
| `RD_ALL` | `0` | `int` |
| `ADDR_REP_K` | `1` | `int` |
| `ADDR_REP_V` | `1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`k_wr_en`**
- `[HEAD_W-1:0]` **`k_wr_head`**
- `[10:0]` **`k_wr_pos`**
- `[ROW_W-1:0]` **`k_wr_vec`**
- **`k_rd_en`**
- `[HEAD_W-1:0]` **`k_rd_head`**
- `[10:0]` **`k_rd_pos`**
- **`v_wr_en`**
- `[HEAD_W-1:0]` **`v_wr_head`**
- `[10:0]` **`v_wr_pos`**
- `[ROW_W-1:0]` **`v_wr_vec`**
- **`v_rd_en`**
- `[HEAD_W-1:0]` **`v_rd_head`**
- `[10:0]` **`v_rd_pos`**
- `[N_STACKS*ADDR_REP_K*11-1:0]` **`k_rd_pos_rep`**
- `[N_STACKS*ADDR_REP_V*11-1:0]` **`v_rd_pos_rep`**
- `[N_STACKS*ADDR_REP_K*11-1:0]` **`k_wr_pos_rep`**
- `[N_STACKS*ADDR_REP_V*11-1:0]` **`v_wr_pos_rep`**

*Outputs:*
- `[ROW_W-1:0]` **`k_rd_vec`**
- `[ROW_W-1:0]` **`v_rd_vec`**
- `[N_STACKS*ROW_W-1:0]` **`v_rd_bus_flat`**
- `[N_STACKS*ROW_W-1:0]` **`k_rd_bus_flat`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_full_mha_sbuf` (`attention_full_mha_sbuf.sv`)
- **Instantiates:** `kv_head_wide_win` (`kv_head_wide_win.sv`)

---

### `kv_head_wide`
- **Source File:** `eagle4_drafter_depthn_1_0/src/kv_head_wide.sv` (70 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Wide-bus KV cache memory bank. Implements dual-port BRAM/URAM storage storing multi-token Key and Value vectors packed into wide memory words.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `D_HEAD` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `MAX_CTX` | `32` | `int` |
| `ADDR_W` | `(MAX_CTX <= 1` | `int` |
| `ROW_W` | `D_HEAD * A_WIDTH` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`wr_en`**
- `[10:0]` **`wr_pos`**
- `[ROW_W-1:0]` **`wr_vec`**
- **`rd_en`**
- `[10:0]` **`rd_pos`**

*Outputs:*
- `[ROW_W-1:0]` **`rd_vec`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `kv_head_wide_win`
- **Source File:** `eagle4_drafter_depthn_1_0/src/kv_head_wide_win.sv` (312 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Windowed wide-bus KV cache memory bank with Xilinx XPM primitive integration. Utilizes xpm_memory_sdpram to provide high-bandwidth sliding-window KV history buffering.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `D_HEAD` | `128` | `int` |
| `A_WIDTH` | `16` | `int` |
| `CTX_WIN` | `1024` | `int` |
| `HEADS_PER_STACK` | `4` | `int` |
| `RAM_STYLE` | `"ultra"` | *untyped* |
| `ADDR_REP` | `1` | `int` |
| `ROW_W` | `D_HEAD * A_WIDTH` | `int` |
| `STACK_DEPTH` | `CTX_WIN * HEADS_PER_STACK` | `int` |
| `HSEL_W` | `(HEADS_PER_STACK <= 1` | `int` |
| `POS_W` | `(CTX_WIN <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`wr_en`**
- `[HSEL_W-1:0]` **`wr_head`**
- `[10:0]` **`wr_pos`**
- `[ROW_W-1:0]` **`wr_vec`**
- **`rd_en`**
- `[HSEL_W-1:0]` **`rd_head`**
- `[10:0]` **`rd_pos`**
- `[ADDR_REP*11-1:0]` **`rd_pos_rep`**
- `[ADDR_REP*11-1:0]` **`wr_pos_rep`**

*Outputs:*
- `[ROW_W-1:0]` **`rd_vec`**

**Wiring & Hierarchy:**
- **Instantiated By:** `multihead_kv` (`multihead_kv.sv`)
- **Instantiates:** *(Leaf module)*

---

### `kv_lane_chain_pkg`
- **Source File:** `eagle4_drafter_depthn_1_0/src/kv_lane_chain.sv` (203 lines)
- **Construct Type:** `package`
- **Technical Purpose:** Daisy-chained parallel KV cache streaming interconnect. Links distributed KV cache memory slices along the physical FPGA floorplan, propagating context vectors with minimal routing congestion.

**Wiring & Hierarchy:**
- **Referenced By:** *SystemVerilog package imported across modules*
- **Instantiates:** *(Leaf module)*

---

### `attn_score_dot`
- **Source File:** `eagle4_drafter_depthn_1_0/src/attn_score_dot.sv` (266 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Query-Key dot product scoring engine. Computes raw attention scores (Q * K_t^T / sqrt(d)) between the current query token and cached key tokens across all context positions.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `D_HEAD` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [A_WIDTH*D_HEAD-1:0]` **`q_pack`**
- `signed [A_WIDTH*D_HEAD-1:0]` **`k_pack`**

*Outputs:*
- `signed [31:0]` **`score_out`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_head_assembly` (`attention_head_assembly.sv`), `attention_head_assembly_perbank_sbuf` (`attention_head_assembly_perbank_sbuf.sv`)
- **Instantiates:** `csa_tree16_int20` (`csa_tree16_int20.sv`), `csa_tree16_int28` (`csa_tree16_int28.sv`)

---

### `softmax_exp_sum`
- **Source File:** `eagle4_drafter_depthn_1_0/src/softmax_exp_sum.sv` (68 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Attention Softmax exponentiation and accumulation unit. Evaluates exp(score - max_score), accumulates the denominator sum across context length, and normalizes scores into attention probabilities.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`clear`**
- **`valid_in`**
- `signed [31:0]` **`score_in`**
- `signed [31:0]` **`max_q`**

*Outputs:*
- `[31:0]` **`exp_val`**
- **`exp_valid`**
- `signed [47:0]` **`sum_so_far`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_head_assembly` (`attention_head_assembly.sv`), `attention_head_assembly_perbank_sbuf` (`attention_head_assembly_perbank_sbuf.sv`)
- **Instantiates:** `exp_pwl_q12` (`exp_pwl_q12.sv`)

---

### `attn_v_combine`
- **Source File:** `eagle4_drafter_depthn_1_0/src/attn_v_combine.sv` (75 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Attention Value combination accumulator. Computes weighted vector sums: out = sum(Prob_t * V_t), combining attention weights with historical Value vectors.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `D_HEAD` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`clear`**
- **`valid_in`**
- **`round_en`**
- `signed [A_WIDTH-1:0]` **`attn_q12`**
- `signed [A_WIDTH*D_HEAD-1:0]` **`v_vec`**

*Outputs:*
- `signed [32*D_HEAD-1:0]` **`out_vec`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_head_assembly` (`attention_head_assembly.sv`), `attn_v_combine_perbank` (`attn_v_combine_perbank.sv`)
- **Instantiates:** *(Leaf module)*

---

### `attn_v_combine_perbank`
- **Source File:** `eagle4_drafter_depthn_1_0/src/attn_v_combine_perbank.sv` (151 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Per-bank Value combination accumulator. Accumulates attention-weighted Value vectors directly into dedicated head output buffers to prevent write port contention.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `D_HEAD` | `128` | `int` |
| `A_WIDTH` | `16` | `int` |
| `N_STACKS` | `6` | `int` |
| `SSEL_W` | `(N_STACKS <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`clear`**
- **`valid_in`**
- **`round_en`**
- `[SSEL_W-1:0]` **`stack_sel`**
- `signed [A_WIDTH-1:0]` **`attn_q12`**
- `signed [N_STACKS*A_WIDTH*D_HEAD-1:0]` **`v_rd_bus_flat`**

*Outputs:*
- `signed [32*D_HEAD-1:0]` **`out_vec`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_head_assembly_perbank_sbuf` (`attention_head_assembly_perbank_sbuf.sv`)
- **Instantiates:** `attn_v_combine` (`attn_v_combine.sv`)

---

### `rope_layer`
- **Source File:** `eagle4_drafter_depthn_1_0/src/rope_layer.sv` (430 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Rotary Position Embedding (RoPE) sequencer. Manages cos/sin frequency lookup tables and streams Query and Key vector pairs through rotation arithmetic before KV cache storage and dot-product scoring.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `D_HEAD` | `128` | `int` |
| `N_PAIRS` | `64` | `int` |
| `MAX_CTX` | `2048` | `int` |
| `A_WIDTH` | `16` | `int` |
| `ROPE_SLR_REG` | `1'b0` | `bit` |
| `ROPE_CS_PIPE` | `1'b0` | `bit` |
| `ROPE_POS_REP` | `1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[10:0]` **`pos`**
- `signed [D_HEAD*A_WIDTH-1:0]` **`vec_in`**

*Outputs:*
- `signed [D_HEAD*A_WIDTH-1:0]` **`vec_out`**
- **`valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_full_mha_sbuf` (`attention_full_mha_sbuf.sv`), `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `rope_pair_rotate` (`rope_pair_rotate.sv`)

---

### `rope_pair_rotate`
- **Source File:** `eagle4_drafter_depthn_1_0/src/rope_pair_rotate.sv` (82 lines)
- **Construct Type:** `module`
- **Technical Purpose:** 2D coordinate rotation arithmetic unit. Computes rotary embedding transformations: x_out = x*cos(t) - y*sin(t), y_out = x*sin(t) + y*cos(t) using fixed-point DSP multipliers.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`en`**
- `signed [15:0]` **`x_in`**
- `signed [15:0]` **`y_in`**
- `signed [15:0]` **`cos_t`**
- `signed [15:0]` **`sin_t`**

*Outputs:*
- `signed [15:0]` **`x_out`**
- `signed [15:0]` **`y_out`**
- **`valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** `rope_layer` (`rope_layer.sv`)
- **Instantiates:** *(Leaf module)*

---

### `g4_qkv_seam`
- **Source File:** `eagle4_drafter_depthn_1_0/src/g4_qkv_seam.sv` (82 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Q/K/V staging pipeline and distribution seam register. Receives the concatenated linear projection output from G1/G4 GEMV, splits it into discrete Query, Key, and Value streams, and aligns timing for RoPE and attention.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `N_HEADS` | `24` | `int` |
| `D_HEAD` | `128` | `int` |
| `A_WIDTH` | `16` | `int` |
| `G4_FRAC` | `13` | `int` |
| `G5_FRAC` | `10` | `int` |
| `SHR` | `G4_FRAC - G5_FRAC` | `int` |
| `H` | `N_HEADS * D_HEAD` | `int` |
| `OUT_W` | `H * A_WIDTH` | `int` |

**Port Interface:**

*Inputs:*
- `signed [32*D_HEAD-1:0]` **`q_full_buf [0:N_HEADS-1]`**
- `signed [32*D_HEAD-1:0]` **`k_full_buf [0:N_HEADS-1]`**
- `signed [32*D_HEAD-1:0]` **`v_full_buf [0:N_HEADS-1]`**

*Outputs:*
- `signed [OUT_W-1:0]` **`q_in`**
- `signed [OUT_W-1:0]` **`k_in`**
- `signed [OUT_W-1:0]` **`v_in`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

## 5. Staging & Inter-Stage Buffering

### `stage_io_buffer`
- **Source File:** `eagle4_drafter_depthn_1_0/src/stage_io_buffer.sv` (324 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Configurable intermediate activation buffer. Supports ping-pong and FIFO buffering modes for streaming vector activations between asynchronous or burst-mode pipeline stages.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `H` | `3072` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `AUTO_SEQ` | `1'b0` | `bit` |
| `FRAC_BITS` | `12` | `int` |
| `RD_LAT` | `0` | `int` |
| `DBG_TAP` | `1'b0` | `bit` |
| `NCHUNKS` | `H / K_PAR` | `int` |
| `CW` | `(NCHUNKS > 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`wr_valid`**
- `[CW-1:0]` **`wr_chunk_idx`**
- `signed [K_PAR*A_WIDTH-1:0]` **`wr_chunk`**
- **`wr_clear`**
- **`rd_valid`**
- `[CW-1:0]` **`rd_chunk_idx`**

*Outputs:*
- `[CW-1:0]` **`wr_count`**
- **`full`**
- `signed [K_PAR*A_WIDTH-1:0]` **`rd_chunk`**
- `[15:0]` **`dbg_rd_chunk_q0`**
- **`dbg_rd_guard_q`**
- `[CW-1:0]` **`rd_chunk_idx_out`**
- **`rd_valid_out`**

**Wiring & Hierarchy:**
- **Instantiated By:** `stage_io_buffer_sel` (`stage_io_buffer_sel.sv`)
- **Instantiates:** *(Leaf module)*

---

### `stage_io_buffer_sel`
- **Source File:** `eagle4_drafter_depthn_1_0/src/stage_io_buffer_sel.sv` (103 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Multi-stage activation multiplexer and staging buffer. Dynamically routes intermediate results between the shared GEMV array, non-linear activation layers, and residual accumulation paths based on stage_sel.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `SEAM_FF` | `1'b0` | `bit` |
| `H` | `3072` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `AUTO_SEQ` | `1'b0` | `bit` |
| `FRAC_BITS` | `12` | `int` |
| `RD_LAT` | `0` | `int` |
| `DBG_TAP` | `1'b0` | `bit` |
| `NCHUNKS` | `H / K_PAR` | `int` |
| `CW` | `(NCHUNKS > 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`wr_valid`**
- `[CW-1:0]` **`wr_chunk_idx`**
- `signed [K_PAR*A_WIDTH-1:0]` **`wr_chunk`**
- **`wr_clear`**
- **`rd_valid`**
- `[CW-1:0]` **`rd_chunk_idx`**
- `[CW-1:0]` **`rd_chunk_idx_b`**

*Outputs:*
- `[CW-1:0]` **`wr_count`**
- **`full`**
- `signed [K_PAR*A_WIDTH-1:0]` **`rd_chunk`**
- `signed [K_PAR*A_WIDTH-1:0]` **`rd_chunk_b`**
- `[15:0]` **`dbg_rd_chunk_q0`**
- **`dbg_rd_guard_q`**
- `[CW-1:0]` **`rd_chunk_idx_out`**
- **`rd_valid_out`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `stage_io_buffer` (`stage_io_buffer.sv`), `stage_io_buffer_ff` (`stage_io_buffer_ff.sv`)

---

### `stage_io_buffer_ff`
- **Source File:** `eagle4_drafter_depthn_1_0/src/stage_io_buffer_ff.sv` (270 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Low-latency flip-flop pipeline buffer. Provides register retiming across physical SLR boundaries on the Alveo V80 to isolate long wire delays and guarantee timing closure.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `H` | `3072` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `AUTO_SEQ` | `1'b0` | `bit` |
| `FRAC_BITS` | `12` | `int` |
| `RD_LAT` | `0` | `int` |
| `DBG_TAP` | `1'b0` | `bit` |
| `IMPL` | `"REGFILE"` | *untyped* |
| `NCHUNKS` | `H / K_PAR` | `int` |
| `CW` | `(NCHUNKS > 1` | `int` |
| `CHW` | `K_PAR * A_WIDTH` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`wr_valid`**
- `[CW-1:0]` **`wr_chunk_idx`**
- `signed [CHW-1:0]` **`wr_chunk`**
- **`wr_clear`**
- **`rd_valid`**
- `[CW-1:0]` **`rd_chunk_idx`**
- `[CW-1:0]` **`rd_chunk_idx_b`**

*Outputs:*
- `[CW-1:0]` **`wr_count`**
- **`full`**
- `signed [CHW-1:0]` **`rd_chunk`**
- `signed [CHW-1:0]` **`rd_chunk_b`**
- `[15:0]` **`dbg_rd_chunk_q0`**
- **`dbg_rd_guard_q`**
- `[CW-1:0]` **`rd_chunk_idx_out`**
- **`rd_valid_out`**

**Wiring & Hierarchy:**
- **Instantiated By:** `stage_io_buffer_sel` (`stage_io_buffer_sel.sv`)
- **Instantiates:** *(Leaf module)*

---

### `stage_io_buffer_banked`
- **Source File:** `eagle4_drafter_depthn_1_0/src/stage_io_buffer_banked.sv` (94 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Multi-banked activation buffer. Provides multiple independent memory banks to allow simultaneous burst-writing of producer stage results while a consumer stage reads previous activations.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `H` | `8192` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `18` | `int` |
| `LANES_PER_BANK` | `2` | `int` |
| `RD_LAT` | `1` | `int` |
| `NCHUNKS` | `H / K_PAR` | `int` |
| `CW` | `(NCHUNKS > 1` | `int` |
| `NBANKS` | `K_PAR / LANES_PER_BANK` | `int` |
| `BANK_W` | `LANES_PER_BANK * A_WIDTH` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`wr_valid`**
- `[CW-1:0]` **`wr_chunk_idx`**
- `signed [K_PAR*A_WIDTH-1:0]` **`wr_chunk`**
- **`wr_clear`**
- **`rd_valid`**
- `[CW-1:0]` **`rd_chunk_idx`**

*Outputs:*
- `[CW-1:0]` **`wr_count`**
- **`full`**
- `signed [K_PAR*A_WIDTH-1:0]` **`rd_chunk`**

**Wiring & Hierarchy:**
- **Instantiated By:** *Top-level or variant conditional instantiation*
- **Instantiates:** *(Leaf module)*

---

## 6. Low-Rank Down-Projection & Vocab Rescore Engine

### `lmdown_gemv_banksplit`
- **Source File:** `eagle4_drafter_depthn_1_0/src/lmdown_gemv_banksplit.sv` (848 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Dense low-rank language model down-projection engine. Executes dense GEMV projecting 3072-dimensional transformer activations to 256-dimensional representations using fixed ROM weights split across 16 lane memory files (lmdown_W_q15_lane*.mem).

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K` | `3072` | `int` |
| `K_PAR` | `16` | `int` |
| `A_WIDTH` | `16` | `int` |
| `M` | `256` | `int` |
| `LANES` | `16` | `int` |
| `IN_FRAC` | `10` | `int` |
| `W_FRAC` | `15` | `int` |
| `OUT_FRAC` | `10` | `int` |
| `W_HEX` | `"/home/y344shi/eagle4_artifacts/p4/lmdown_W_q15.hex"` | *untyped* |
| `W_MEM_DIR` | `"/home/y344shi/eagle4_artifacts/p4"` | *untyped* |
| `LMDOWN_SLR_REG` | `1'b0` | `bit` |
| `SEAM_FF_LAND` | `1'b0` | `bit` |
| `COTRAVEL` | `1'b1` | `bit` |
| `NCHUNKS` | `K / K_PAR` | `int` |
| `CW` | `(NCHUNKS > 1` | `int` |
| `N_GRP` | `M / LANES` | `int` |
| `GW` | `(N_GRP > 1` | `int` |
| `BANK_DEPTH` | `N_GRP * NCHUNKS` | `int` |
| `BAW` | `$clog2(BANK_DEPTH` | `int` |
| `ROW_W` | `K_PAR * A_WIDTH` | `int` |
| `ACC_FRAC` | `IN_FRAC + W_FRAC` | `int` |
| `OCW` | `(N_GRP > 1` | `int` |
| `BANK_W` | `32` | `int` |
| `NB` | `ROW_W / BANK_W` | `int` |
| `EPB` | `BANK_W / A_WIDTH` | `int` |
| `DBG_TAP` | `1'b0` | `bit` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `signed [K_PAR*A_WIDTH-1:0]` **`a_chunk`**
- `[CW-1:0]` **`a_chunk_idx_in`**
- **`a_chunk_idx_valid_in`**

*Outputs:*
- `[CW-1:0]` **`a_chunk_idx`**
- **`a_chunk_idx_valid`**
- **`wr_valid`**
- `[OCW-1:0]` **`wr_chunk_idx`**
- `signed [LANES*A_WIDTH-1:0]` **`wr_chunk`**
- **`done`**
- `[15:0]` **`dbg_a_d0`**
- **`dbg_run_valid`**

**Wiring & Hierarchy:**
- **Instantiated By:** *Top-level or variant conditional instantiation*
- **Instantiates:** *(Leaf module)*

---

### `g8_rescore`
- **Source File:** `eagle4_drafter_depthn_1_0/src/g8_rescore.sv` (910 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Fused-gather-dot vocabulary rescoring engine (G8_SPLICE). Takes speculative draft candidate tokens, gathers their high-dimensional language model embeddings, and computes exact dot-product logits to verify speculative accuracy.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `V_MAX` | `64` | `int` |
| `H` | `16` | `int` |
| `K_C` | `16` | `int` |
| `K_F` | `8` | `int` |
| `IDX_W` | `$clog2(V_MAX` | `int` |
| `CIDX_W` | `(K_C <= 1` | `int` |
| `FP16_W` | `16` | `int` |
| `SCORE_W` | `32` | `int` |
| `AXI_ADDR_W` | `64` | `int` |
| `AXI_DATA_W` | `32` | `int` |
| `AXI_ID_W` | `1` | `int` |
| `G8_LANE_FIX` | `1'b1` | `bit` |
| `MAX_BURST` | `256` | `int` |
| `WD_LIMIT` | `32'd4_194_304` | *untyped* |
| `CHEAP_KEY_INT32` | `1'b1` | `bit` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- **`cheap_valid`**
- `[SCORE_W-1:0]` **`cheap_score`**
- `[IDX_W-1:0]` **`cheap_index`**
- **`cheap_last`**
- `[FP16_W-1:0]` **`nrm_rd_data`**
- **`m_axi_arready`**
- **`m_axi_rvalid`**
- `[AXI_DATA_W-1:0]` **`m_axi_rdata`**
- **`m_axi_rlast`**
- `[AXI_ID_W-1:0]` **`m_axi_rid`**
- `[1:0]` **`m_axi_rresp`**
- `[AXI_ADDR_W-1:0]` **`lmfull_hbm_base`**
- `[CIDX_W-1:0]` **`dbg_cand_rdaddr`**

*Outputs:*
- **`cheap_ready`**
- `[$clog2((H<2)?2:H)-1:0]` **`nrm_rd_addr`**
- **`m_axi_arvalid`**
- `[AXI_ADDR_W-1:0]` **`m_axi_araddr`**
- `[7:0]` **`m_axi_arlen`**
- `[2:0]` **`m_axi_arsize`**
- `[1:0]` **`m_axi_arburst`**
- `[AXI_ID_W-1:0]` **`m_axi_arid`**
- **`m_axi_rready`**
- `[K_F*IDX_W-1:0]` **`frontier_ids`**
- `[K_F*SCORE_W-1:0]` **`frontier_scores`**
- **`done`**
- **`g8_err`**
- `[3:0]` **`dbg_sel_substate`**
- `[9:0]` **`dbg_cand_filled`**
- `[9:0]` **`dbg_rows_gathered`**
- **`dbg_seen_last`**
- **`dbg_in_gather`**
- `[AXI_ADDR_W-1:0]` **`dbg_last_araddr`**
- **`dbg_rresp_err`**
- `[IDX_W-1:0]` **`dbg_cand_id`**
- `[SCORE_W-1:0]` **`dbg_cand_cheap_score`**

**Wiring & Hierarchy:**
- **Instantiated By:** *Top-level or variant conditional instantiation*
- **Instantiates:** `g8_dot_fp16` (`g8_dot_fp16.sv`)

---

### `g8_rescore_p`
- **Source File:** `eagle4_drafter_depthn_1_0/src/g8_rescore_p.sv` (2472 lines)
- **Construct Type:** `module`
- **Technical Purpose:** P-parallel enhanced vocabulary rescoring engine. Evaluates multiple candidate tokens in parallel, saturating memory bandwidth and accelerating multi-branch speculative verification.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `V_MAX` | `64` | `int` |
| `H` | `16` | `int` |
| `K_C` | `16` | `int` |
| `K_F` | `8` | `int` |
| `IDX_W` | `$clog2(V_MAX` | `int` |
| `CIDX_W` | `(K_C <= 1` | `int` |
| `FP16_W` | `16` | `int` |
| `SCORE_W` | `32` | `int` |
| `AXI_ADDR_W` | `64` | `int` |
| `AXI_DATA_W` | `32` | `int` |
| `AXI_ID_W` | `1` | `int` |
| `MAX_BURST` | `256` | `int` |
| `G8_OUTSTANDING` | `8` | `int` |
| `G8_PREFETCH` | `1'b0` | `bit` |
| `G8_PREFETCH_DEPTH` | `2` | `int` |
| `G8_PREFETCH_BUDGET` | `5120` | `int` |
| `WD_LIMIT` | `32'd4_194_304` | *untyped* |
| `CHEAP_KEY_INT32` | `1'b1` | `bit` |
| `G8_OPT1C` | `1'b1` | `bit` |
| `G8_OPT1D` | `1'b0` | `bit` |
| `G8_MIN_NG` | `16` | `int` |
| `G8_OPT1F` | `1'b1` | `bit` |
| `G8_OPT1G` | `1'b0` | `bit` |
| `G8_SELPIPE` | `1'b1` | `bit` |
| `G8_WIDE` | `1'b0` | `bit` |
| `WIDE_P` | `16` | `int` |
| `WIDE_IL` | `1'b0` | `bit` |
| `WIDE_M` | `4` | `int` |
| `MROUND_P` | `1'b1` | `bit` |
| `SCALAR_II1` | `1'b1` | `bit` |
| `G8_CONSUME_HARDEN_P` | `1'b0` | `bit` |
| `G8_LANE_FIX` | `1'b1` | `bit` |
| `G8_ROWPP` | `1'b0` | `bit` |
| `ROWPP_OWN_DEPTH` | `4` | `int` |
| `G8_NCH` | `1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[15:0]` **`kc_cfg`**
- **`cheap_valid`**
- `[SCORE_W-1:0]` **`cheap_score`**
- `[IDX_W-1:0]` **`cheap_index`**
- **`cheap_last`**
- `[FP16_W-1:0]` **`nrm_rd_data`**
- `[WIDE_P*FP16_W-1:0]` **`nrm_rd_data_wide`**
- **`m_axi_arready`**
- **`m_axi_rvalid`**
- `[AXI_DATA_W-1:0]` **`m_axi_rdata`**
- **`m_axi_rlast`**
- `[AXI_ID_W-1:0]` **`m_axi_rid`**
- `[1:0]` **`m_axi_rresp`**
- `[AXI_ADDR_W-1:0]` **`lmfull_hbm_base`**
- **`prefetch_req`**
- `[IDX_W-1:0]` **`prefetch_req_id`**
- **`prefetch_evict`**
- `[IDX_W-1:0]` **`prefetch_evict_id`**
- `[CIDX_W-1:0]` **`dbg_cand_rdaddr`**
- `[CIDX_W-1:0]` **`bridge_rdaddr_nxt`**
- `[G8_NCH-1:0]` **`stripe_arready`**
- `[G8_NCH-1:0]` **`stripe_rvalid`**
- `[AXI_DATA_W-1:0]` **`stripe_rdata [0:G8_NCH-1]`**
- `[G8_NCH-1:0]` **`stripe_rlast`**
- `[WIDE_P*FP16_W-1:0]` **`stripe_nrm_wide [0:G8_NCH-1]`**

*Outputs:*
- **`cheap_ready`**
- `[$clog2((H<2)?2:H)-1:0]` **`nrm_rd_addr`**
- **`m_axi_arvalid`**
- `[AXI_ADDR_W-1:0]` **`m_axi_araddr`**
- `[7:0]` **`m_axi_arlen`**
- `[2:0]` **`m_axi_arsize`**
- `[1:0]` **`m_axi_arburst`**
- `[AXI_ID_W-1:0]` **`m_axi_arid`**
- **`m_axi_rready`**
- **`prefetch_drop`**
- `[IDX_W:0]` **`prefetch_count`**
- `[K_F*IDX_W-1:0]` **`frontier_ids`**
- `[K_F*SCORE_W-1:0]` **`frontier_scores`**
- **`done`**
- **`g8_err`**
- `[3:0]` **`dbg_sel_substate`**
- `[9:0]` **`dbg_cand_filled`**
- `[9:0]` **`dbg_rows_gathered`**
- **`dbg_seen_last`**
- **`dbg_in_gather`**
- `[AXI_ADDR_W-1:0]` **`dbg_last_araddr`**
- **`dbg_rresp_err`**
- `[IDX_W-1:0]` **`dbg_cand_id`**
- `[SCORE_W-1:0]` **`dbg_cand_cheap_score`**
- `[SCORE_W-1:0]` **`bridge_score_ff`**
- `[SCORE_W-1:0]` **`sel_thresh`**
- **`sel_thresh_valid`**
- `[G8_NCH-1:0]` **`stripe_arvalid`**
- `[AXI_ADDR_W-1:0]` **`stripe_araddr [0:G8_NCH-1]`**
- `[7:0]` **`stripe_arlen [0:G8_NCH-1]`**
- `[2:0]` **`stripe_arsize [0:G8_NCH-1]`**
- `[1:0]` **`stripe_arburst [0:G8_NCH-1]`**
- `[G8_NCH-1:0]` **`stripe_rready`**
- `[$clog2((H<2)?2:H)-1:0]` **`stripe_nrm_addr [0:G8_NCH-1]`**

**Wiring & Hierarchy:**
- **Instantiated By:** *Top-level or variant conditional instantiation*
- **Instantiates:** `g8_dot_fp16` (`g8_dot_fp16.sv`), `g8_dot_fp16_wide` (`g8_dot_fp16_wide.sv`), `g8_dot_fp16_wide_il` (`g8_dot_fp16_wide_il.sv`), `g8_stripe_frontend` (`g8_stripe_frontend.sv`)

---

### `g8_stripe_frontend`
- **Source File:** `eagle4_drafter_depthn_1_0/src/g8_stripe_frontend.sv` (366 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Multi-channel striped memory fetch frontend for G8 rescoring. Formulates burst read requests across memory channels to gather non-contiguous token embedding vectors with high bus utilization.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `N_CH` | `2` | `int` |
| `H` | `3072` | `int` |
| `K_C` | `512` | `int` |
| `IDX_W` | `18` | `int` |
| `SCORE_W` | `32` | `int` |
| `AXI_ADDR_W` | `64` | `int` |
| `AXI_DATA_W` | `256` | `int` |
| `MAX_BURST` | `256` | `int` |
| `OUTST` | `4` | `int` |
| `WIDE_P` | `16` | `int` |
| `WIDE_M` | `4` | `int` |
| `ROW_OVERLAP` | `1'b1` | `bit` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[$clog2(K_C):0]` **`n_rows`**
- `[AXI_ADDR_W-1:0]` **`lmfull_hbm_base`**
- `[IDX_W-1:0]` **`cand_id_rd [0:N_CH-1]`**
- `[WIDE_P*16-1:0]` **`nrm_wide [0:N_CH-1]`**
- `[N_CH-1:0]` **`m_arready`**
- `[N_CH-1:0]` **`m_rvalid`**
- `[AXI_DATA_W-1:0]` **`m_rdata [0:N_CH-1]`**
- `[N_CH-1:0]` **`m_rlast`**

*Outputs:*
- `[$clog2(K_C)-1:0]` **`cand_addr [0:N_CH-1]`**
- `[$clog2(H)-1:0]` **`nrm_addr [0:N_CH-1]`**
- `[N_CH-1:0]` **`m_arvalid`**
- `[AXI_ADDR_W-1:0]` **`m_araddr [0:N_CH-1]`**
- `[7:0]` **`m_arlen [0:N_CH-1]`**
- `[2:0]` **`m_arsize [0:N_CH-1]`**
- `[1:0]` **`m_arburst [0:N_CH-1]`**
- `[N_CH-1:0]` **`m_rready`**
- `[N_CH-1:0]` **`land_valid`**
- `[$clog2(K_C)-1:0]` **`land_slot [0:N_CH-1]`**
- `[SCORE_W-1:0]` **`land_score [0:N_CH-1]`**
- **`gather_done`**
- `[31:0]` **`scored_cnt_dbg`**

**Wiring & Hierarchy:**
- **Instantiated By:** `g8_rescore_p` (`g8_rescore_p.sv`)
- **Instantiates:** `g8_dot_fp16_wide_il` (`g8_dot_fp16_wide_il.sv`)

---

### `g8_dot_fp16`
- **Source File:** `eagle4_drafter_depthn_1_0/src/g8_dot_fp16.sv` (336 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Single-channel half-precision (FP16) dot product engine. Computes vector inner products in native IEEE 754 half-precision floating-point arithmetic for rescore logit evaluation.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `FADD_PIPE` | `1'b1` | `bit` |
| `MUL_PIPE` | `1'b1` | `bit` |
| `MROUND` | `1'b0` | `bit` |
| `MREG` | `1'b1` | `bit` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`clear`**
- **`in_valid`**
- **`in_last`**
- `[15:0]` **`a_fp16`**
- `[15:0]` **`b_fp16`**

*Outputs:*
- **`out_valid`**
- `[31:0]` **`score_fp32`**

**Wiring & Hierarchy:**
- **Instantiated By:** `g8_rescore` (`g8_rescore.sv`), `g8_rescore_p` (`g8_rescore_p.sv`)
- **Instantiates:** *(Leaf module)*

---

### `g8_dot_fp16_wide`
- **Source File:** `eagle4_drafter_depthn_1_0/src/g8_dot_fp16_wide.sv` (199 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Wide-word FP16 dot product reduction unit. Parallelizes FP16 multiplications across wide vector slices, accumulating products into high-precision intermediate registers.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `P` | `16` | `int` |
| `MREG` | `1'b1` | `bit` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`clear`**
- **`in_valid`**
- **`in_last`**
- `[P*16-1:0]` **`a_fp16`**
- `[P*16-1:0]` **`b_fp16`**

*Outputs:*
- **`out_valid`**
- `[31:0]` **`score_fp32`**

**Wiring & Hierarchy:**
- **Instantiated By:** `g8_rescore_p` (`g8_rescore_p.sv`)
- **Instantiates:** *(Leaf module)*

---

### `g8_dot_fp16_wide_il`
- **Source File:** `eagle4_drafter_depthn_1_0/src/g8_dot_fp16_wide_il.sv` (348 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Interleaved wide-word FP16 dot product unit. Features multi-stage interleaved accumulation pipelines to maximize DSP operating frequency and hide floating-point addition latency.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `P` | `16` | `int` |
| `M` | `4` | `int` |
| `MREG` | `1'b1` | `bit` |
| `ROW_OVERLAP` | `1'b0` | `bit` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`clear`**
- **`in_valid`**
- **`in_last`**
- `[P*16-1:0]` **`a_fp16`**
- `[P*16-1:0]` **`b_fp16`**

*Outputs:*
- **`out_valid`**
- `[31:0]` **`score_fp32`**

**Wiring & Hierarchy:**
- **Instantiated By:** `g8_rescore_p` (`g8_rescore_p.sv`), `g8_stripe_frontend` (`g8_stripe_frontend.sv`)
- **Instantiates:** *(Leaf module)*

---

### `g8_fp16_pkg`
- **Source File:** `eagle4_drafter_depthn_1_0/src/g8_fp16_pkg.sv` (742 lines)
- **Construct Type:** `package`
- **Technical Purpose:** SystemVerilog arithmetic package defining FP16 data structures, exponent/mantissa manipulation functions, conversion routines (int16/fp16), and floating-point comparison operators.

**Wiring & Hierarchy:**
- **Referenced By:** *SystemVerilog package imported across modules*
- **Instantiates:** *(Leaf module)*

---

### `emb_gather`
- **Source File:** `eagle4_drafter_depthn_1_0/src/emb_gather.sv` (535 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Token embedding gather DMA unit. Fetches full embedding vectors for selected draft tokens from external memory to initialize subsequent speculative drafting passes (RUNG C).

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `H` | `3072` | `int` |
| `IDX_W` | `18` | `int` |
| `AXI_ADDR_W` | `64` | `int` |
| `AXI_DATA_W` | `256` | `int` |
| `MAX_BURST` | `256` | `int` |
| `OUTST` | `4` | `int` |
| `EMB_FMT` | `1'b1` | `bit` |
| `NCH` | `1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[IDX_W-1:0]` **`tok_id`**
- `[AXI_ADDR_W-1:0]` **`hbm_emb_base`**
- `[NCH-1:0]` **`m_arready`**
- `[NCH-1:0]` **`m_rvalid`**
- `[NCH*AXI_DATA_W-1:0]` **`m_rdata`**
- `[NCH-1:0]` **`m_rlast`**

*Outputs:*
- `[NCH-1:0]` **`m_arvalid`**
- `[NCH*AXI_ADDR_W-1:0]` **`m_araddr`**
- `[NCH*8-1:0]` **`m_arlen`**
- `[NCH*3-1:0]` **`m_arsize`**
- `[NCH*2-1:0]` **`m_arburst`**
- `[NCH-1:0]` **`m_rready`**
- **`o_we`**
- `[$clog2((H+AXI_DATA_W/16-1)/(AXI_DATA_W/16))-1:0]` **`o_chunk`**
- `[AXI_DATA_W-1:0]` **`o_data`**
- **`done`**
- `[31:0]` **`beats_dbg`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

## 7. Speculative Draft Tree Engine

### `tree_level_fsm`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_level_fsm.sv` (730 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Multi-level speculative draft tree sequencer FSM. Controls multi-level tree expansion across speculative depths (levels 0 to max_depth-1), managing candidate branching, frontier selection, and exit handshakes.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- `[31:0]` **`ctx_len_i`**
- `[31:0]` **`max_depth_i`**
- **`cand_valid`**
- `[31:0]` **`cand_prob_f32`**
- `[TOK_W-1:0]` **`cand_token`**
- `[NODE_W-1:0]` **`tn_rd_idx`**

*Outputs:*
- **`cand_req`**
- **`level_done`**
- **`level_last`**
- `[DEPTH_W-1:0]` **`level_idx_o`**
- `[POS_W-1:0]` **`level_pos_o`**
- `[K_F*NODE_W-1:0]` **`surv_node_o`**
- `[K_F*NODE_W-1:0]` **`surv_parent_o`**
- `[K_F*SLOT_W-1:0]` **`surv_parent_slot_o`**
- `[K_F*TOK_W-1:0]` **`surv_token_o`**
- `[K_F*SCORE_W-1:0]` **`surv_score_o`**
- `[TOK_W-1:0]` **`tn_rd_token`**
- `[NODE_W-1:0]` **`tn_rd_parent`**
- `[DEPTH_W-1:0]` **`tn_rd_depth`**
- `[POS_W-1:0]` **`tn_rd_pos`**
- `signed [SCORE_W-1:0]` **`tn_rd_score`**
- **`busy`**
- **`done`**
- `[DEPTH_W-1:0]` **`levels_run_o`**
- `[NODE_W-1:0]` **`n_nodes_o`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** `tree_decode` (`tree_decode.sv`), `tree_log2_fxp` (`tree_log2_fxp.sv`)

---

### `tree_feed_seq`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_feed_seq.sv` (205 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Tree feeder sequencer. Schedules candidate tokens from current tree levels into downstream compute and rescore pipelines, tracking parent-child node relationships.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:** *(Empty port list / Internal wiring)*

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_tile_stream`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_tile_stream.sv` (259 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Logit tile streamer. Slices massive vocabulary logit vectors into manageable hardware tiles, streaming them into top-K sorting and pruning networks.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `128` | `int` |
| `V_MAX` | `128256` | `int` |
| `IDX_W` | `18` | `int` |
| `SCORE_W` | `32` | `int` |
| `N_TILES` | `(V_MAX + LANES - 1` | `int` |
| `LANE_W` | `(LANES  <= 1` | `int` |
| `TILE_W` | `(N_TILES <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`tree_start`**
- `[15:0]` **`n_tiles_cfg`**
- `signed [SCORE_W*LANES-1:0]` **`array_acc_out`**
- **`array_done`**

*Outputs:*
- **`gemv_tile_start`**
- `[15:0]` **`gemv_tile_idx`**
- **`gemv_stage_g12`**
- **`tree_dec_start`**
- **`valid_in`**
- `signed [SCORE_W-1:0]` **`score_in`**
- `[IDX_W-1:0]` **`index_in`**
- **`last_in`**
- **`busy`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_tile_stream_pp`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_tile_stream_pp.sv` (252 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Ping-pong pipelined logit tile streamer. Double-buffers logit tiles to overlap logit generation with candidate selection arithmetic.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `128` | `int` |
| `V_MAX` | `128256` | `int` |
| `IDX_W` | `18` | `int` |
| `SCORE_W` | `32` | `int` |
| `N_TILES` | `(V_MAX + LANES - 1` | `int` |
| `LANE_W` | `(LANES  <= 1` | `int` |
| `TILE_W` | `(N_TILES <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`tree_start`**
- `[15:0]` **`n_tiles_cfg`**
- `signed [SCORE_W*LANES-1:0]` **`array_acc_out`**
- **`array_done`**

*Outputs:*
- **`gemv_tile_start`**
- `[15:0]` **`gemv_tile_idx`**
- **`gemv_stage_g12`**
- **`tree_dec_start`**
- **`valid_in`**
- `signed [SCORE_W-1:0]` **`score_in`**
- `[IDX_W-1:0]` **`index_in`**
- **`last_in`**
- **`busy`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_tile_stream_pkpar`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_tile_stream_pkpar.sv` (207 lines)
- **Construct Type:** `module`
- **Technical Purpose:** K-parallel logit tile streamer. Streams K_PAR candidate logit elements per clock cycle into parallel sorting comparator arrays.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `128` | `int` |
| `P` | `4` | `int` |
| `V_MAX` | `128256` | `int` |
| `IDX_W` | `18` | `int` |
| `SCORE_W` | `32` | `int` |
| `N_TILES` | `(V_MAX + LANES - 1` | `int` |
| `LANE_W` | `(LANES  <= 1` | `int` |
| `TILE_W` | `(N_TILES <= 1` | `int` |
| `GRP_W` | `(LANES/P <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`tree_start`**
- `[15:0]` **`n_tiles_cfg`**
- `signed [SCORE_W*LANES-1:0]` **`array_acc_out`**
- **`array_done`**

*Outputs:*
- **`gemv_tile_start`**
- `[15:0]` **`gemv_tile_idx`**
- **`gemv_stage_g12`**
- **`tree_dec_start`**
- **`valid_in`**
- `signed [P*SCORE_W-1:0]` **`scores_in`**
- `[P*IDX_W-1:0]` **`indices_in`**
- `[P-1:0]` **`lane_en`**
- **`last_in`**
- **`busy`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_tile_stream_pkpar_pp`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_tile_stream_pkpar_pp.sv` (239 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Ping-pong buffered K-parallel logit tile streamer. Combines parallel element streaming with double-buffered tile prefetching for uninterrupted top-K evaluation.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `128` | `int` |
| `P` | `4` | `int` |
| `V_MAX` | `128256` | `int` |
| `IDX_W` | `18` | `int` |
| `SCORE_W` | `32` | `int` |
| `N_TILES` | `(V_MAX + LANES - 1` | `int` |
| `LANE_W` | `(LANES  <= 1` | `int` |
| `TILE_W` | `(N_TILES <= 1` | `int` |
| `GRP_W` | `(LANES/P <= 1` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`tree_start`**
- `[15:0]` **`n_tiles_cfg`**
- `signed [SCORE_W*LANES-1:0]` **`array_acc_out`**
- **`array_done`**

*Outputs:*
- **`gemv_tile_start`**
- `[15:0]` **`gemv_tile_idx`**
- **`gemv_stage_g12`**
- **`tree_dec_start`**
- **`valid_in`**
- `signed [P*SCORE_W-1:0]` **`scores_in`**
- `[P*IDX_W-1:0]` **`indices_in`**
- `[P-1:0]` **`lane_en`**
- **`last_in`**
- **`busy`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_softmax_bridge`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_softmax_bridge.sv` (466 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Logit-to-probability numerical conversion bridge. Normalizes candidate cheap logits through localized exponentiation and maximum subtraction to produce valid probabilities for tree pruning.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:** *(Empty port list / Internal wiring)*

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_decode`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_decode.sv` (217 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Speculative candidate token selector. Implements sorting networks and comparator trees to extract the top-K candidate tokens with highest cumulative likelihood from streaming logit tiles.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K_F` | `8` | `int` |
| `V_MAX` | `128256` | `int` |
| `IDX_W` | `$clog2(V_MAX` | `int` |
| `SCORE_W` | `32` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- **`valid_in`**
- `signed [SCORE_W-1:0]` **`score_in`**
- `[IDX_W-1:0]` **`index_in`**
- **`last_in`**

*Outputs:*
- `signed [K_F*SCORE_W-1:0]` **`topk_scores`**
- `[K_F*IDX_W-1:0]` **`topk_indices`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`), `tree_level_fsm` (`tree_level_fsm.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_merge_pipe`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_merge_pipe.sv` (313 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Candidate path merging pipeline. Merges speculative branches across sibling nodes, pruning low-probability candidates and constructing the consolidated speculative draft frontier.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K_F` | `8` | `int` |
| `P` | `4` | `int` |
| `V_MAX` | `128256` | `int` |
| `IDX_W` | `$clog2(V_MAX` | `int` |
| `SCORE_W` | `32` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start`**
- **`valid_in`**
- `signed [P*SCORE_W-1:0]` **`scores_in`**
- `[P*IDX_W-1:0]` **`indices_in`**
- `[P-1:0]` **`lane_en`**
- **`last_in`**

*Outputs:*
- `signed [K_F*SCORE_W-1:0]` **`topk_scores`**
- `[K_F*IDX_W-1:0]` **`topk_indices`**
- **`done`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_cheap_serializer`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_cheap_serializer.sv` (332 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Cheap-logit serialization module. Deserializes and replays cheap logit streams from stage G12 into serial format required by the G8 rescoring engine.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:** *(Empty port list / Internal wiring)*

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_out_walk`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_out_walk.sv` (254 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Tree traversal graph engine. Traverses the accepted speculative token tree in canonical node-major order, resolving parent pointers and flattening the graph into a linear token buffer.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:** *(Empty port list / Internal wiring)*

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_log2_fxp`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_log2_fxp.sv` (107 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Fixed-point base-2 logarithm (log2) calculator. Computes log-probabilities for cumulative tree path scoring using tree_log2_lut.svh lookups and linear interpolation.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`valid_in`**
- `[31:0]` **`prob_f32`**

*Outputs:*
- **`valid_out`**
- `signed [SCORE_W-1:0]` **`log2_q`**

**Wiring & Hierarchy:**
- **Instantiated By:** `tree_level_fsm` (`tree_level_fsm.sv`)
- **Instantiates:** *(Leaf module)*

---

### `score_max_tracker`
- **Source File:** `eagle4_drafter_depthn_1_0/src/score_max_tracker.sv` (44 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Streaming maximum score tracking register. Dynamically tracks the maximum logit across a vocabulary stream to provide numerical scaling references for stable Softmax computation.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`clear`**
- **`valid`**
- `signed [31:0]` **`score`**

*Outputs:*
- `signed [31:0]` **`running_max`**

**Wiring & Hierarchy:**
- **Instantiated By:** `attention_head_assembly` (`attention_head_assembly.sv`), `attention_head_assembly_perbank_sbuf` (`attention_head_assembly_perbank_sbuf.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_dma_writer`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_dma_writer.sv` (419 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Speculative draft tree DMA writer. Packages accepted draft tree nodes into memory-aligned bursts and writes them to external memory buffers.

**Parameters:** *(None / Configured via `eagle4_config.svh`)*

**Port Interface:** *(Empty port list / Internal wiring)*

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `output_dma_writer`
- **Source File:** `eagle4_drafter_depthn_1_0/src/output_dma_writer.sv` (723 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Primary output write-DMA engine. Formats the final speculative tree into the standardized OUT_ABI_V24 specification (token IDs, cumulative scores, parent indices) and writes to host memory via m_axi_out.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `K_F` | `8` | `int` |
| `IDX_W` | `18` | `int` |
| `SCORE_W` | `32` | `int` |
| `AXI_ADDR_W` | `64` | `int` |
| `AXI_DATA_W` | `32` | `int` |
| `AXI_ID_W` | `1` | `int` |
| `EN_OUT1_MIRROR` | `1'b1` | `bit` |
| `ABI_V24` | `1'b0` | `bit` |
| `TOP_K` | `16` | `int` |
| `PARENT_SENTINEL` | `-1` | `int` |
| `STALL_TIMEOUT` | `1 << 20` | `int` |
| `EN_DUMP` | `1'b0` | `bit` |
| `CAND_N` | `512` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`tree_done`**
- `signed [K_F*SCORE_W-1:0]` **`topk_scores`**
- `[K_F*IDX_W-1:0]` **`topk_indices`**
- `[AXI_ADDR_W-1:0]` **`out0_base`**
- `[AXI_ADDR_W-1:0]` **`out1_base`**
- `[AXI_ADDR_W-1:0]` **`out2_base`**
- **`dump_en`**
- `[AXI_ADDR_W-1:0]` **`out3_base`**
- `[AXI_ADDR_W-1:0]` **`out4_base`**
- `[AXI_DATA_W-1:0]` **`dump3_rddata`**
- `[K_F*IDX_W-1:0]` **`tree_ids`**
- `[K_F*SCORE_W-1:0]` **`tree_scores`**
- `[K_F*32-1:0]` **`tree_parents`**
- `[K_F*SCORE_W-1:0]` **`g8_rescored`**
- **`m_axi_out_awready`**
- **`m_axi_out_wready`**
- `[AXI_ID_W-1:0]` **`m_axi_out_bid`**
- `[1:0]` **`m_axi_out_bresp`**
- **`m_axi_out_bvalid`**

*Outputs:*
- `[$clog2((CAND_N*2 > 1) ? CAND_N*2 : 2)-1:0]` **`dump3_rdaddr`**
- **`out_busy`**
- **`out_done`**
- **`out_err`**
- `[AXI_ID_W-1:0]` **`m_axi_out_awid`**
- `[AXI_ADDR_W-1:0]` **`m_axi_out_awaddr`**
- `[7:0]` **`m_axi_out_awlen`**
- `[2:0]` **`m_axi_out_awsize`**
- `[1:0]` **`m_axi_out_awburst`**
- **`m_axi_out_awvalid`**
- `[AXI_DATA_W-1:0]` **`m_axi_out_wdata`**
- `[AXI_DATA_W/8-1:0]` **`m_axi_out_wstrb`**
- **`m_axi_out_wlast`**
- **`m_axi_out_wvalid`**
- **`m_axi_out_bready`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `tree_score_pkg`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_score_pkg.sv` (102 lines)
- **Construct Type:** `package`
- **Technical Purpose:** SystemVerilog package defining draft tree data structures, candidate score formats, node descriptors, and frontier array definitions.

**Wiring & Hierarchy:**
- **Referenced By:** *SystemVerilog package imported across modules*
- **Instantiates:** *(Leaf module)*

---

## 8. Debug, Residency & Telemetry Subsystem

### `dbg_residency`
- **Source File:** `eagle4_drafter_depthn_1_0/src/dbg_residency.sv` (201 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Hardware residency accounting and verification module. Monitors write pulses to on-chip memories, counting landed words for weights, scales, embeddings, and biases to verify initialization completeness.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `ADDR_TAG_W` | `16` | `int` |
| `RD_TAG_W` | `8` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`res_clear`**
- **`emb_we`**
- `[31:0]` **`emb_wr_data`**
- `[ADDR_TAG_W-1:0]` **`emb_addr_tag`**
- **`hfc_we`**
- `[31:0]` **`hfc_wr_data`**
- `[ADDR_TAG_W-1:0]` **`hfc_addr_tag`**
- **`gamma_we`**
- `[31:0]` **`gamma_wr_data`**
- `[ADDR_TAG_W-1:0]` **`gamma_addr_tag`**
- **`w_we`**
- `[31:0]` **`w_wr_data`**
- **`sc_we`**
- `[31:0]` **`sc_wr_data`**
- **`g1_rd_stb`**
- `[RD_TAG_W-1:0]` **`g1_rd_idx_tap`**
- `[31:0]` **`g1_rd_word_tap`**
- **`g1e_rd_stb`**
- `[RD_TAG_W-1:0]` **`g1e_rd_idx_tap`**
- `[31:0]` **`g1e_rd_word_tap`**
- `[2:0]` **`res_sel`**

*Outputs:*
- `[31:0]` **`res_crc_word`**
- `[31:0]` **`res_sum_word`**
- `[31:0]` **`res_count_word`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `dbg_stage_capture`
- **Source File:** `eagle4_drafter_depthn_1_0/src/dbg_stage_capture.sv` (207 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Intermediate activation diagnostic capture buffer. Sniffs internal datapath buses at selected pipeline stages and captures activation vectors into RAM for host inspection.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `NSTAGE` | `19` | `int` |
| `NW` | `32` | `int` |
| `MAX_CHUNK` | `128` | `int` |
| `SSEL_W` | `5` | `int` |
| `WIDX_W` | `5` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start_pulse`**
- `[NSTAGE-1:0]` **`strobe`**
- `[NSTAGE*8-1:0]` **`words`**
- `[NSTAGE*MAX_CHUNK*32-1:0]` **`chunk`**
- `[SSEL_W-1:0]` **`cap_stage_sel`**
- **`widx_we`**
- `[WIDX_W-1:0]` **`widx_wdata`**
- **`val_re`**

*Outputs:*
- `[31:0]` **`cap_word_idx_word`**
- `[31:0]` **`cap_val_word`**
- `[31:0]` **`cap_meta_word`**
- `[31:0]` **`cap_crc_word`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `dbg_probe_ram`
- **Source File:** `eagle4_drafter_depthn_1_0/src/dbg_probe_ram.sv` (205 lines)
- **Construct Type:** `module`
- **Technical Purpose:** AXI-accessible dual-port probe RAM. Stores captured internal diagnostic waveforms and intermediate vectors, accessible via host register reads.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `DEPTH` | `1024` | `int` |
| `CAP_WORDS` | `256` | `int` |
| `MAX_CHUNK` | `128` | `int` |
| `AW` | `$clog2(DEPTH` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- `[3:0]` **`probe_sel`**
- **`sel_we`**
- **`addr_we`**
- `[AW-1:0]` **`addr_wdata`**
- **`data_re`**
- **`chunk_valid`**
- `[$clog2(MAX_CHUNK+1)-1:0]` **`chunk_words`**
- `[MAX_CHUNK*32-1:0]` **`chunk_data`**

*Outputs:*
- `[31:0]` **`probe_addr_word`**
- `[31:0]` **`probe_data_word`**
- `[31:0]` **`probe_xor_word`**
- `[31:0]` **`probe_sum_word`**
- `[31:0]` **`probe_count_word`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

### `dbg_observe`
- **Source File:** `eagle4_drafter_depthn_1_0/src/dbg_observe.sv` (461 lines)
- **Construct Type:** `module`
- **Technical Purpose:** Internal diagnostic signal multiplexer. Aggregates internal status lines, FIFO flags, FSM states, and residency counters, mapping them to readable AXI4-Lite status registers.

**Parameters:**
| Name | Default Value | Type |
| :--- | :--- | :--- |
| `LANES` | `128` | `int` |
| `A_WIDTH` | `16` | `int` |
| `G8_ADDR_W` | `64` | `int` |

**Port Interface:**

*Inputs:*
- **`clk`**
- **`rst_n`**
- **`start_pulse`**
- `[4:0]` **`top_state`**
- `[2:0]` **`g5_state`**
- `[3:0]` **`stage_sel`**
- **`g1_done`**
- **`g4_done`**
- **`g5_done`**
- **`g6_done`**
- **`g7g_done`**
- **`g7u_done`**
- **`g7p_done`**
- **`g9_done`**
- **`g11_done`**
- **`glm_done`**
- **`g12_done`**
- **`tree_done`**
- **`g8_done`**
- `signed [16*16-1:0]` **`g1_out_chunk`**
- **`g1_out_valid`**
- `signed [32*LANES-1:0]` **`array_acc_out`**
- **`array_done`**
- `signed [32*16-1:0]` **`g5_attn_out_vec`**
- **`g5_attn_done`**
- `signed [16*16-1:0]` **`silu_out_chunk`**
- **`silu_valid`**
- `signed [16*16-1:0]` **`g11_out_chunk`**
- **`g11_out_strobe`**
- `signed [8*32-1:0]` **`tree_topk_scores`**
- **`g8_activity`**
- **`g8_wd_tripped`**
- **`kernel_busy`**
- **`kernel_done`**
- `[3:0]` **`dbg_sel`**
- **`emb_we`**
- `[31:0]` **`emb_wr_data`**
- **`hfc_we`**
- `[31:0]` **`hfc_wr_data`**
- **`gamma_we`**
- `[31:0]` **`gamma_wr_data`**
- **`sout_chunk_valid`**
- `[31:0]` **`sout_chunk_w0`**
- `[31:0]` **`sout_chunk_xor`**
- `[9:0]` **`sout_chunk_nwords`**
- `[3:0]` **`g8_sel_substate`**
- `[9:0]` **`g8_cand_filled`**
- `[9:0]` **`g8_rows_gathered`**
- **`g8_seen_last`**
- **`g8_in_gather`**
- `[G8_ADDR_W-1:0]` **`g8_last_araddr`**
- **`g8_rresp_err`**
- **`boot_stream_we`**
- `[31:0]` **`boot_stream_data`**

*Outputs:*
- `[31:0]` **`fsmstate_word`**
- `[31:0]` **`activity_word`**
- `[31:0]` **`cyclecount_word`**
- `[31:0]` **`checksum_word`**
- `[31:0]` **`g8progress_word`**
- `[31:0]` **`g8araddr_lo_word`**
- `[31:0]` **`g8araddr_hi_word`**
- `[31:0]` **`rearm_word`**
- `[31:0]` **`bootstream_word`**
- `[31:0]` **`stageout_xor_word`**
- `[31:0]` **`stageout_sample_word`**
- `[31:0]` **`stageout_count_word`**

**Wiring & Hierarchy:**
- **Instantiated By:** `eagle4_drafter` (`eagle4_drafter.sv`)
- **Instantiates:** *(Leaf module)*

---

## 9. Configuration & Geometry Headers

### `eagle4_config.svh`
- **Source File:** `eagle4_drafter_depthn_1_0/src/eagle4_config.svh` (457 lines)
- **Construct Type:** `header`
- **Technical Purpose:** Central system configuration header. Acts as the single source of truth for all global architectural parameters, bus widths, parallelism factors, feature toggles (G8_SPLICE, DEPTH_N, EMBG_EN), and timing settings.

**Wiring & Hierarchy:**
- **Included By:** *`include directive in RTL sources*
- **Instantiates:** *(Leaf module)*

---

### `stage_geometry.svh`
- **Source File:** `eagle4_drafter_depthn_1_0/src/stage_geometry.svh` (110 lines)
- **Construct Type:** `header`
- **Technical Purpose:** Single source of truth for GEMV stage dimensions. Defines matrix row counts, column depths, base addresses, and column strides for stages G1, G7, G9, and G12 across weight banks and scale buffers.

**Wiring & Hierarchy:**
- **Included By:** *`include directive in RTL sources*
- **Instantiates:** *(Leaf module)*

---

### `tree_log2_lut.svh`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_log2_lut.svh` (135 lines)
- **Construct Type:** `header`
- **Technical Purpose:** Pre-computed 1025-entry lookup table in SystemVerilog syntax providing unsigned Q0.28 values for log2(1 + j/1024), immune to synthesis driver optimization.

**Wiring & Hierarchy:**
- **Included By:** *`include directive in RTL sources*
- **Instantiates:** *(Leaf module)*

---

### `tree_pow2_lut.svh`
- **Source File:** `eagle4_drafter_depthn_1_0/src/tree_pow2_lut.svh` (135 lines)
- **Construct Type:** `header`
- **Technical Purpose:** Pre-computed 1025-entry lookup table in SystemVerilog syntax providing unsigned Q1.28 values for 2^(j/1024), supporting fast fixed-point exponentiation.

**Wiring & Hierarchy:**
- **Included By:** *`include directive in RTL sources*
- **Instantiates:** *(Leaf module)*

---