# ISA-Controlled GEMV Plan

## Current implementation status (2026-10)

Stage 1 is implemented and committed: WRSR (opcode `011`, three-byte
little-endian immediate) writes the 11-entry ISA special-register bank, the
fetcher advances `PC += 3`, and the actual bank is readable over AXI
(`0x178` selector, `0x17C` `{16'h5A5A, read_data}` with synchronous-read
retry under WRSR arbitration). The bank exposes decoded configuration
outputs for the future operation controller. The shared GEMV datapath,
operation START/BUSY/DONE status, tile controller, input/output routing,
and GEMV opcode are **not yet implemented**; the next stage integrates the
real engine and latches these fields at START. Numeric-field validation
(e.g. supported `scale_frac`, `qzero_raw`, group size) also belongs to that
engine stage, not to the register bank.

## Scope

Replace `stage_sel`-derived GEMV configuration with ISA-programmable operation registers, while retaining the physical **128 lanes × 16 input elements per chunk**.

For `y = A x`, let `A` have `m` output rows and `n` input columns. Let `G` be the number of consecutive input elements sharing a quantization scale within one output row.

The fields below are proposed programmable configuration, not a list of registers that all exist today. Weights and scales remain resident in on-chip banks; this plan does not introduce compute-time weight streaming.

## Required operation configuration

All fields are shared by the lanes. Software does not configure each lane separately.

| Register | Meaning |
|---|---|
| `activation_source` | Selects the activation buffer supplying `x`; controls the input mux. |
| `activation_base` | Starting chunk within that buffer. |
| `weight_base` | Starting address of `A`'s region in each lane's weight memory; units: packed 32-bit words. |
| `scale_base` | Starting address of `A`'s scale region in each lane's scale memory; units: 16-bit scale entries. |
| `tile_count` | Number of output tiles: `m / 128` for aligned dimensions. |
| `chunks_per_tile` | Input chunks processed per lane: `n / 16`. |
| `chunks_per_scale_group` | Chunks sharing one scale: `G / 16`, currently 8. |
| `output_destination` | Selects the downstream buffer receiving results. |
| `output_base` | Placement of the first output tile within that destination. |

Input/output routing is part of the cutover: replacing the dimension lookup alone would leave the existing stage-specific activation selection and result-capture logic.

## Address generation

For the current contiguous layout, derive the row strides rather than adding redundant programmable registers:

```text
weight_row_stride = chunks_per_tile
scale_row_stride  = chunks_per_tile / chunks_per_scale_group
```

For tile `t`, lane `l` computes output row `128*t + l`. For input chunk `q`:

```text
activation address = activation_base + q

weight address = weight_base
               + t * weight_row_stride
               + q

scale address = scale_base
              + t * scale_row_stride
              + floor(q / chunks_per_scale_group)
```

The activation chunk is broadcast to all lanes; each lane reads its own weight and scale banks. The activation vector is replayed for every output tile.

Explicit weight/scale stride registers are needed only if padded or non-contiguous matrix storage is required. Host loading must use the same bank bases and layout as compute.

## Numeric configuration

Make these settings programmable only if operations need different numeric formats. They may remain fixed when all supported operations use the existing formats.

| Register | Current setting / purpose |
|---|---|
| `qzero_raw` | Currently 1; decoded integer weight is `raw_code - (qzero_raw + 1)`. |
| `scale_frac_bits` | Currently 13; fractional bits removed after scaled accumulation. |
| `output_conversion` | Selects the required shift/round/saturation behavior when storing results into downstream activation buffers. |

A programmable `qzero_raw` still supplies one zero-point per operation, not arbitrary per-group zero-points. The shared array remains a packed 2-bit-weight, INT16-activation engine; these fields do not make it a general FP16/FP32 GEMV.

## Execution interface

Expose:

```text
START   latch operation configuration and begin
BUSY    operation running
DONE    all output tiles completed
```

Keep the following hardware-managed rather than software-programmable:

- Running `tile_idx`.
- Each lane's running `chunk_idx`.
- Group-progress counters and accumulators.
- Per-tile broadcast lane-start pulses.
- Pipeline-valid and drain state.

Execution sequence:

```text
ISA configures the operation and pulses START
    -> latch configuration; tile_idx = 0
    -> establish tile weight/scale addresses and allow read setup
    -> broadcast one lane-start pulse
    -> each lane clears accumulators and processes chunks_per_tile chunks
    -> drain arithmetic pipeline and collect array completion
    -> advance tile_idx and repeat
    -> assert DONE after the last tile
```

All lanes receive the same bounds. Their local chunk counters advance in lockstep. There is one shared outer tile controller, not an independent tile counter per lane.

## Initial restrictions and integration requirements

Retain these restrictions for the simplest implementation:

```text
m divisible by 128
G divisible by 16
n divisible by G
```

Supporting other dimensions requires explicit padding or tail handling. In particular, a partial final scale group needs a final-group flush; changing counter bounds alone is insufficient.

`chunks_per_scale_group` is currently a compile-time quantity, and the lane uses shifts/bit tests assuming a power-of-two group length. Making it programmable requires changing or constraining the group-index and group-boundary logic, not merely adding a register.

Runtime dimensions remain bounded by the elaborated counter widths and physical buffer/bank capacities. Memory-read lookahead and matched activation, weight, scale, and valid pipelines must remain aligned when the stage-derived configuration is replaced.

Weight-loading completion remains a host responsibility, as in the current design. The host must finish loading weights, scales, and inputs before starting compute. A DMA-wait instruction or hardware load-readiness interlock would be an additional ISA feature, not a required GEMV configuration register.
