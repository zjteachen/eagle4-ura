# ISA-Controlled GEMV Plan

## Current implementation status (2026-10)

Implemented and committed in the shared-engine cutover:

- WRSR (opcode `011`, three-byte little-endian immediate) writes the 11-entry
  actual `special_regs` bank; AXI readback is `0x178`/`0x17C`.
- GEMV (opcode `100`, byte `0x80`) latches the actual bank outputs at START
  and blocks the PC until operation DONE or a validated configuration ERROR.
  Other operands of opcode 100 and all reserved opcodes decode as HALT.
- One `gemv_shared_ctrl` owns the tile loop, explicit per-tile weight/scale
  base settle, array start, result-valid/store handshake, drain, and DONE.
  Its bases/bounds are explicit: the old stage-derived base/stride and
  compile-time group lookups are gone from the scaled array and both banks.
- Runtime `chunks_per_scale_group` genuinely supports 4 and 8 (latched group
  mask/shift in the lane and its group-tag pipeline). All other values are
  rejected before launch, as are unsupported qzero/scale_frac/conversion and
  out-of-capacity spans.
- Activation sources 0..7 and destinations 0..6/9 route real address+data
  into the existing buffers; destination-specific serializers/conversions
  and downstream residual/norm/attention side effects are retained. DONE
  waits for the final destination store.
- TREE is a client of the same configured engine with real stream
  backpressure. The standalone G12 STATE program step is removed. Standard
  G9 is 24 tiles (R6 capacity), or 48 with FORREASON.
- The full pipeline is expressed as WRSR + GEMV + the remaining non-shared
  STATE instructions; the delta-WRSR program is 114 bytes (limit 128).

Verification: `run_sim.sh gemv` runs the real top with real banks, AXI PMEM,
HFC/EMB AXI loads, all-lane independent integer-oracle checks of destination
buffers and the probe, nonzero bases with two tiles, runtime groups 4/8,
routing alternatives, and invalid-config/restart. This is a bounded
fixture, not a claim of whole-model arithmetic accuracy.

## Scope

Replace `stage_sel`-derived GEMV configuration with ISA-programmable operation registers, while retaining the physical **128 lanes × 16 input elements per chunk**.

For `y = A x`, let `A` have `m` output rows and `n` input columns. Let `G` be the number of consecutive input elements sharing a quantization scale within one output row.

The fields below are the implemented configuration (actual `special_regs` bank, latched by `gemv_shared_ctrl` at START). Weights and scales remain resident in on-chip banks; there is no compute-time weight streaming or descriptor bypass.

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
| `chunks_per_scale_group` | Chunks sharing one scale: `G / 16`. Runtime value; supported set {4, 8}. |
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

These fields are validated fixed settings, not a general numeric-format menu. Unsupported values are rejected before any lane starts.

| Register | Implemented setting / contract |
|---|---|
| `qzero_raw` | Fixed 1; decoded integer weight is `raw_code - (qzero_raw + 1)` (range -2..1). Other values rejected. |
| `scale_frac_bits` | Fixed 13; Q3.13 scale, fractional bits removed after scaled accumulation. Other values rejected. |
| `output_conversion` | Fixed 0: the destination's established serializer/conversion is used (Q/K/V 32-bit store; R2/R6 16-bit; R4 18-bit q6.12). Other values rejected; this is not a programmable format selector. |

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

`chunks_per_scale_group` is runtime and genuinely supports 4 and 8: the lane latches the value at start and derives group index, group-end/flush and scale-tag timing from its mask/shift, with group-index widths sized for the worst case (4). Unsupported lengths are rejected by the controller before launch. The destination store handshake holds DONE until the final serializer write.

Runtime dimensions remain bounded by the elaborated counter widths and physical buffer/bank capacities. Memory-read lookahead and matched activation, weight, scale, and valid pipelines must remain aligned when the stage-derived configuration is replaced.

Weight-loading completion remains a host responsibility, as in the current design. The host must finish loading weights, scales, and inputs before starting compute. A DMA-wait instruction or hardware load-readiness interlock would be an additional ISA feature, not a required GEMV configuration register.
