"""Single-midlayer EAGLE4/OSPREY draft head, float32 NumPy, driven by the 24-tensor OSPREY 2-bit storage layout.

Dataflow (transcribed from the structure of the local serving source `llama_eagle4.py`, sha256 8d0f6796ba3bfa24...,
re-implemented from scratch in NumPy -- no source code copied):

    h      = FC(concat_i RMSNorm_i(chunk_i(hidden_9216)))      only when the incoming hidden is 3*H wide (tree root);
                                                                an H-wide hidden (deeper draft levels) BYPASSES fc
    x      = concat[ input_layernorm(embed) , hidden_norm(h) ]  -> 2H
    q,k,v  = ternary linears (2H -> H each) -> [T, heads, head_dim]; NeoX RoPE on q,k at EXPLICIT positions
    a      = o_proj(softmax(q k^T / sqrt(head_dim)) v)
    r1     = a + h                                              (residual is the PRE-attention hidden)
    m      = down( silu(gate(n)) * up(n) ),  n = post_attention_layernorm(r1)
    s      = r1 + m
    to_logits     = norm(s)                 -> lm_head
    for_reasoning = post_mlp_layernorm(s)   -> fed back as the next level's H-wide hidden

!! `post_mlp_layernorm` has NO tensor in the 24-tensor layout. The serving source constructs that RMSNorm whenever
   `enable_post_mlp_norm` is set and then uses it if the module exists, so with this checkpoint its gamma stays at the
   constructor default (all ones). This reference therefore applies a UNIT-GAMMA RMSNorm there. That is an INFERENCE
   from reading the source, NOT a verified numerical fact: no captured activation has been compared. `for_reasoning`
   is flagged accordingly in DraftHead.UNVERIFIED.
"""
from dataclasses import dataclass
import numpy as np
from .safetensors_io import SafeTensorsFile, bf16_to_f32
from .ternary import unpack_ternary
from .ops import rmsnorm, rope_neox, silu, causal_attention


@dataclass(frozen=True)
class Config:
    hidden: int = 3072
    n_heads: int = 32
    head_dim: int = 96
    inter: int = 8192
    vocab: int = 128256
    rms_eps: float = 1e-5
    fc_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.n_heads * self.head_dim != self.hidden:
            raise ValueError("n_heads*head_dim must equal hidden (MHA: q, k and v are all H wide)")


OSPREY_CONFIG = Config()

_BODY = {"midlayer.self_attn.q_proj": ("hidden", "2hidden"), "midlayer.self_attn.k_proj": ("hidden", "2hidden"),
         "midlayer.self_attn.v_proj": ("hidden", "2hidden"), "midlayer.self_attn.o_proj": ("hidden", "hidden"),
         "midlayer.mlp.gate_proj": ("inter", "hidden"), "midlayer.mlp.up_proj": ("inter", "hidden"),
         "midlayer.mlp.down_proj": ("hidden", "inter")}
_NORMS = ["fc.norm.0.weight", "fc.norm.1.weight", "fc.norm.2.weight", "midlayer.hidden_norm.weight",
          "midlayer.input_layernorm.weight", "midlayer.post_attention_layernorm.weight", "norm.weight"]


def expected_layout(cfg):
    dim = {"hidden": cfg.hidden, "2hidden": 2 * cfg.hidden, "inter": cfg.inter}
    exp = {"embed_tokens.weight": ("BF16", (cfg.vocab, cfg.hidden)), "lm_head.weight": ("BF16", (cfg.vocab, cfg.hidden)),
           "fc.fc.weight": ("BF16", (cfg.hidden, 3 * cfg.hidden))}
    for n in _NORMS:
        exp[n] = ("BF16", (cfg.hidden,))
    for n, (r, c) in _BODY.items():
        exp[n + ".qweight"] = ("U8", (dim[r] * dim[c] // 4,))
        exp[n + ".weight_scale"] = ("F32", (1,))
    return exp


class TernaryLinear:
    def __init__(self, w_int8, scale):
        self.w, self.scale = w_int8, np.float32(scale)

    def __call__(self, x, row_chunk=1024):
        x = np.asarray(x, dtype=np.float32)
        out = np.empty(x.shape[:-1] + (self.w.shape[0],), dtype=np.float32)
        for a in range(0, self.w.shape[0], row_chunk):
            out[..., a:a + row_chunk] = x @ self.w[a:a + row_chunk].astype(np.float32).T
        return out * self.scale


class DraftHead:
    UNVERIFIED = ("for_reasoning: post_mlp_layernorm gamma is ASSUMED all-ones (tensor absent from the layout)",)

    def __init__(self, cfg, st):
        self.cfg, self.st = cfg, st
        exp = expected_layout(cfg)
        got = {n: st.info(n) for n in st.names()}
        missing, extra = sorted(set(exp) - set(got)), sorted(set(got) - set(exp))
        if missing or extra:
            raise ValueError("tensor name mismatch: missing=%s extra=%s" % (missing, extra))
        bad = ["%s: stored %s%s, expected %s%s" % (n, got[n][0], list(got[n][1]), exp[n][0], list(exp[n][1]))
               for n in exp if got[n] != exp[n]]
        if bad:
            raise ValueError("tensor dtype/shape mismatch:\n  " + "\n  ".join(bad))
        dim = {"hidden": cfg.hidden, "2hidden": 2 * cfg.hidden, "inter": cfg.inter}
        self.lin = {}
        for n, (r, c) in _BODY.items():
            self.lin[n.rsplit(".", 1)[1]] = TernaryLinear(unpack_ternary(st.raw(n + ".qweight"), dim[r], dim[c]),
                                                          st.raw(n + ".weight_scale")[0])
        self.g = {n: st.f32(n) for n in _NORMS}
        self.fc_w = st.f32("fc.fc.weight")

    @classmethod
    def from_file(cls, path, cfg=OSPREY_CONFIG):
        return cls(cfg, SafeTensorsFile(path))

    def embed(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= self.cfg.vocab:
            raise IndexError("token id out of range")
        return bf16_to_f32(np.ascontiguousarray(self.st.raw("embed_tokens.weight")[ids]))

    def fuse(self, hidden3):
        c = self.cfg
        h = np.asarray(hidden3, dtype=np.float32)
        parts = [rmsnorm(h[..., i * c.hidden:(i + 1) * c.hidden], self.g["fc.norm.%d.weight" % i], c.fc_eps) for i in range(3)]
        return np.concatenate(parts, axis=-1) @ self.fc_w.T

    def forward(self, embeds, hidden, positions, allow=None, past_kv=None):
        """embeds [T,H]; hidden [T,3H] (root: fused) or [T,H] (deeper level: fc bypassed); positions [T] explicit.
        past_kv: optional (k,v) each [Tp, heads, head_dim], already RoPE'd. Returns dict."""
        c = self.cfg
        embeds = np.asarray(embeds, dtype=np.float32)
        hidden = np.asarray(hidden, dtype=np.float32)
        T = embeds.shape[0]
        if hidden.shape[-1] == 3 * c.hidden:
            hidden = self.fuse(hidden)
        elif hidden.shape[-1] != c.hidden:
            raise ValueError("hidden must be H or 3H wide")
        residual = hidden
        x = np.concatenate([rmsnorm(embeds, self.g["midlayer.input_layernorm.weight"], c.rms_eps),
                            rmsnorm(hidden, self.g["midlayer.hidden_norm.weight"], c.rms_eps)], axis=-1)
        shp = (T, c.n_heads, c.head_dim)
        q = rope_neox(self.lin["q_proj"](x).reshape(shp), positions, c.rope_theta)
        k = rope_neox(self.lin["k_proj"](x).reshape(shp), positions, c.rope_theta)
        v = self.lin["v_proj"](x).reshape(shp)
        if past_kv is not None:
            k_all, v_all = np.concatenate([past_kv[0], k]), np.concatenate([past_kv[1], v])
        else:
            k_all, v_all = k, v
        a = self.lin["o_proj"](causal_attention(q, k_all, v_all, allow).reshape(T, c.hidden))
        r1 = a + residual
        n = rmsnorm(r1, self.g["midlayer.post_attention_layernorm.weight"], c.rms_eps)
        m = self.lin["down_proj"](silu(self.lin["gate_proj"](n)) * self.lin["up_proj"](n))
        s = r1 + m
        return {"to_logits": rmsnorm(s, self.g["norm.weight"], c.rms_eps),
                "for_reasoning": rmsnorm(s, np.ones(c.hidden, dtype=np.float32), c.rms_eps),   # see UNVERIFIED
                "kv": (k_all, v_all)}

    def logits(self, to_logits, row_chunk=8192):
        x = np.asarray(to_logits, dtype=np.float32)
        lm = self.st.raw("lm_head.weight")
        out = np.empty(x.shape[:-1] + (self.cfg.vocab,), dtype=np.float32)
        for a in range(0, self.cfg.vocab, row_chunk):
            out[..., a:a + row_chunk] = x @ bf16_to_f32(np.ascontiguousarray(lm[a:a + row_chunk])).T
        return out
