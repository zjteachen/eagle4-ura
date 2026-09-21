# OSPREY 2-bit draft head — CPU-first NumPy runnable reference

**Synthetic weights only. This is NOT an EAGLE-3 reference, NOT verified against the GPU/SGLang path, and NOT an RTL-equivalence
result.** It is a small, readable, dependency-light implementation of the *single-midlayer EAGLE4/OSPREY draft head* that loads the
published 24-tensor storage layout in [`assets/synthetic-osprey-2bit`](../../assets/synthetic-osprey-2bit/README.md) and runs a forward
pass on a CPU. Runtime dependency: **NumPy**. Test dependency: **none** (standard-library `unittest`; pytest also collects the tests).

## Run it

```sh
cd reference/osprey-numpy
python3 -m unittest discover -s tests -t . -v        # 30 tests; process exit status 0 == pass
echo $?

# bounded forward on the published zero-filled layout (creates a ~1.7 GB SPARSE file in a NEW directory)
python3 ../../assets/synthetic-osprey-2bit/generate.py ../../assets/synthetic-osprey-2bit/tensor-manifest.json /tmp/osprey-zero --manifest
python3 run_smoke.py --model /tmp/osprey-zero/model.safetensors
```

Environment: Python ≥ 3.9 and `numpy>=1.26,<3` (`requirements.txt`). `requirements.lock.txt` lists the exact interpreter/NumPy pairs the
suite was run with — three hosts, NumPy 1.26.4 / 2.2.6 / 2.5.2, 30/30 each. On a machine without NumPy:
`python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.lock.txt`. The full-size test needs ~1 GB RAM and a
filesystem with sparse-file support; `OSPREY_REF_SKIP_FULL=1` skips it, `OSPREY_REF_TMP=<dir>` keeps all scratch files under `<dir>`
instead of a self-deleting temp directory. No GPU, CUDA, PyTorch, safetensors package, tokenizer or trained checkpoint is used.

## What is in here

| file | what |
|---|---|
| `osprey_ref/safetensors_io.py` | header parse + `np.memmap` tensors; **BF16 → float32 is exact** (`(u16 << 16)` reinterpreted: bfloat16 is the top 16 bits of an IEEE float32); refuses to cast packed `U8` storage to float |
| `osprey_ref/ternary.py` | `ternary_absmean` / `2bit_4x_uint8_rowmajor`: 4 two-bit codes per byte **LSB-first**, row-major over `[out, in]`; code `c ∈ {0,1,2}` → weight `c − 1 ∈ {−1,0,+1}`; code 3 is a hard error; `W = w · weight_scale` (one F32 scalar per tensor) |
| `osprey_ref/ops.py` | RMSNorm, SiLU, NeoX (rotate-half) RoPE, masked softmax attention — plain float32 |
| `osprey_ref/model.py` | the dataflow (below), the 24-name/shape/dtype layout check, chunked LM head |
| `run_smoke.py` | load + one bounded forward + honest verdict line |
| `tests/` | 30 tests, incl. an **independent float64 re-derivation** of the whole layer and planted negative controls |

Dataflow (hidden `H = 3072`, 32 heads × 96, MHA; MLP 8192; vocab 128256; RMSNorm eps 1e-5; RoPE θ = 10000):

```
h   = FC( concat_i RMSNorm_i( chunk_i(hidden_3H) ) )        only when the incoming hidden is 3H wide (tree root); an H-wide hidden bypasses fc
x   = concat[ input_layernorm(embed), hidden_norm(h) ]      2H
q,k,v = ternary linears 2H→H → [T,32,96]; RoPE(q,k) at EXPLICIT positions;  a = o_proj( softmax(qkᵀ/√96) v )
r1  = a + h                                                 residual = the PRE-attention hidden
s   = r1 + down( silu(gate(n)) · up(n) ),   n = post_attention_layernorm(r1)
to_logits     = norm(s) → lm_head          for_reasoning = post_mlp_layernorm(s) → next draft level's H-wide hidden
```

## What is validated — and what is not

**A. Shape plumbing on the published zero fixture (full size).** All 24 names, dtypes and shapes equal `tensor-manifest.json`; the file
loads; 7 packed matrices unpack (75 M + 57 M + 9 M codes); a 3-token forward and the 128256-way LM head run; every output is finite.
**Every output is exactly 0.0 — by construction** (zero norm gains, zero scales, zero dense weights). The test asserts that exact zero
and says so. Note the trap it also pins down: a zero *byte* is code 0 = weight **−1**, not 0; only the zero `weight_scale` makes `W = 0`.
This proves allocation/dtype/cast/unpack/dataflow plumbing at real size. **It proves nothing about numerics.**

**B. Self-consistency on RANDOM synthetic weights (tiny model, same 24-tensor scheme).** The NumPy layer agrees with an *independent*
float64 re-derivation written a different way (explicit per-head/per-token loops, complex-number RoPE, dense dequantised weights) to
rtol 3e-4; incremental decoding with a KV prefix equals one causal pass; perturbing a future token cannot move earlier outputs; BF16
decode is checked against a field-by-field IEEE decode for **all 65,536 bit patterns**; pack/unpack round-trips and has hand-derived
known answers. Three **planted mutants** (MSB-first unpack, post-norm residual, RoPE sign flip) were each shown to FAIL the suite
(6 / 2 / 3 failing tests) — the tests can say no. This validates *this implementation against its own specification*.

**C. NOT done — do not read A or B as any of these:**
- No comparison against the SGLang/GPU serving implementation or against trained weights (none are in this repository).
- No comparison against RTL/FPGA outputs; the hardware path is fixed-point, this reference is float32.
- No tokenizer, no tree drafting/verification loop, no KV-cache management beyond an optional prefix, no bfloat16 *arithmetic* emulation.
- ⚠️ `for_reasoning` rests on an **inference, not a measurement**: the layout has no `post_mlp_layernorm` tensor. The serving source
  constructs that RMSNorm when `enable_post_mlp_norm` is set and uses it whenever the module exists, so its gain stays at the constructor
  default (ones); this reference applies a unit-gain RMSNorm there and lists it in `DraftHead.UNVERIFIED`. `to_logits` does not depend on it.
- The packing convention itself was proven against real weights elsewhere (a full-tensor decode-vs-BF16 round trip in the project's
  FPGA weight packer) — that evidence is private and is **not** reproduced here; this repository only exercises it on synthetic data.

See `HANDOFF_2026-09-21.md` for sources, revisions, hashes, the exact commands/results per host, and remaining blockers.
