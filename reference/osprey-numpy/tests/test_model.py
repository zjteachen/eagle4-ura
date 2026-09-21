"""Self-consistency of the draft head on RANDOM synthetic weights against an INDEPENDENT float64 reference.

This is numerical validation of THIS implementation's plumbing (unpack -> dequant -> ops -> dataflow). It is NOT a comparison
against SGLang/GPU outputs, trained weights or RTL -- none of those are available in this repository.
"""
import os
import unittest
import numpy as np
from tests._fixtures import scratch_dir, make_tiny, TINY
from osprey_ref import DraftHead, SafeTensorsFile


def ref_forward_f64(t, cfg, embeds, hidden, positions, past=None):
    """Independent formulation: explicit per-head / per-token loops, complex-number RoPE, dense float64 weights."""
    H, nh, d = cfg.hidden, cfg.n_heads, cfg.head_dim

    def rms(x, g, eps):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g

    embeds, hidden = embeds.astype(np.float64), hidden.astype(np.float64)
    if hidden.shape[-1] == 3 * H:
        parts = [rms(hidden[:, i * H:(i + 1) * H], t["fc.norm.%d.weight" % i], cfg.fc_eps) for i in range(3)]
        hidden = np.concatenate(parts, -1) @ t["fc.fc.weight"].T
    x = np.concatenate([rms(embeds, t["midlayer.input_layernorm.weight"], cfg.rms_eps),
                        rms(hidden, t["midlayer.hidden_norm.weight"], cfg.rms_eps)], -1)
    T = x.shape[0]
    inv = 10000.0 ** (-np.arange(d // 2) * 2.0 / d)

    def rope(v, pos):
        z = (v[: d // 2] + 1j * v[d // 2:]) * np.exp(1j * pos * inv)
        return np.concatenate([z.real, z.imag])

    Q = (x @ t["midlayer.self_attn.q_proj"].T).reshape(T, nh, d)
    K = (x @ t["midlayer.self_attn.k_proj"].T).reshape(T, nh, d)
    V = (x @ t["midlayer.self_attn.v_proj"].T).reshape(T, nh, d)
    for i in range(T):
        for h in range(nh):
            Q[i, h], K[i, h] = rope(Q[i, h], positions[i]), rope(K[i, h], positions[i])
    if past is not None:
        K, V = np.concatenate([past[0], K]), np.concatenate([past[1], V])
    off = K.shape[0] - T
    A = np.zeros((T, nh, d))
    for i in range(T):
        for h in range(nh):
            sc = np.array([Q[i, h] @ K[j, h] for j in range(off + i + 1)]) / np.sqrt(d)
            p = np.exp(sc - sc.max())
            p /= p.sum()
            A[i, h] = sum(p[j] * V[j, h] for j in range(off + i + 1))
    r1 = A.reshape(T, H) @ t["midlayer.self_attn.o_proj"].T + hidden
    n = rms(r1, t["midlayer.post_attention_layernorm.weight"], cfg.rms_eps)
    g = n @ t["midlayer.mlp.gate_proj"].T
    m = ((g / (1 + np.exp(-g))) * (n @ t["midlayer.mlp.up_proj"].T)) @ t["midlayer.mlp.down_proj"].T
    s = r1 + m
    return rms(s, t["norm.weight"], cfg.rms_eps), rms(s, np.ones(H), cfg.rms_eps), (K, V)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cm = scratch_dir("model")
        cls.dir = cls._cm.__enter__()
        cls.path = os.path.join(cls.dir, "model.safetensors")
        cls.truth = make_tiny(cls.path, seed=3)
        cls.head = DraftHead.from_file(cls.path, TINY)
        cls.rng = np.random.default_rng(11)

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def test_root_path_3H_hidden_matches_independent_f64(self):
        T = 5
        ids = self.rng.integers(0, TINY.vocab, T)
        hid = self.rng.standard_normal((T, 3 * TINY.hidden)).astype(np.float32)
        pos = np.arange(7, 7 + T)
        out = self.head.forward(self.head.embed(ids), hid, pos)
        tl, fr, _ = ref_forward_f64(self.truth, TINY, self.truth["embed_tokens.weight"][ids], hid, pos)
        np.testing.assert_allclose(out["to_logits"], tl, rtol=3e-4, atol=3e-5)
        np.testing.assert_allclose(out["for_reasoning"], fr, rtol=3e-4, atol=3e-5)
        np.testing.assert_allclose(self.head.logits(out["to_logits"]), tl @ self.truth["lm_head.weight"].T, rtol=3e-4, atol=3e-4)

    def test_deeper_level_H_hidden_bypasses_fc(self):
        T = 3
        ids = self.rng.integers(0, TINY.vocab, T)
        hid = self.rng.standard_normal((T, TINY.hidden)).astype(np.float32)
        out = self.head.forward(self.head.embed(ids), hid, [0, 1, 2])
        tl, _, _ = ref_forward_f64(self.truth, TINY, self.truth["embed_tokens.weight"][ids], hid, [0, 1, 2])
        np.testing.assert_allclose(out["to_logits"], tl, rtol=3e-4, atol=3e-5)

    def test_kv_prefix_equals_full_sequence(self):   # incremental decode must equal one causal pass
        T = 6
        ids = self.rng.integers(0, TINY.vocab, T)
        hid = self.rng.standard_normal((T, 3 * TINY.hidden)).astype(np.float32)
        emb, pos = self.head.embed(ids), np.arange(T)
        full = self.head.forward(emb, hid, pos)
        a = self.head.forward(emb[:4], hid[:4], pos[:4])
        b = self.head.forward(emb[4:], hid[4:], pos[4:], past_kv=a["kv"])
        np.testing.assert_allclose(b["to_logits"], full["to_logits"][4:], rtol=2e-5, atol=2e-6)

    def test_positions_matter(self):   # planted control: RoPE is actually applied (T>1, else softmax over one key hides it)
        ids = self.rng.integers(0, TINY.vocab, 4)
        hid = self.rng.standard_normal((4, 3 * TINY.hidden)).astype(np.float32)
        emb = self.head.embed(ids)
        a = self.head.forward(emb, hid, [0, 1, 2, 3])["to_logits"]
        b = self.head.forward(emb, hid, [0, 5, 9, 40])["to_logits"]
        self.assertFalse(np.allclose(a[1:], b[1:], rtol=1e-3, atol=1e-4))
        np.testing.assert_allclose(a[0], b[0], rtol=1e-6)      # token 0 sees only itself at position 0 in both

    def test_bad_hidden_width_rejected(self):
        with self.assertRaises(ValueError):
            self.head.forward(np.zeros((1, TINY.hidden), np.float32), np.zeros((1, TINY.hidden + 4), np.float32), [0])


class TestLoaderRejects(unittest.TestCase):
    def test_missing_tensor(self):
        with scratch_dir("neg") as d:
            p = os.path.join(d, "m.safetensors")
            make_tiny(p, drop="norm.weight")
            with self.assertRaisesRegex(ValueError, "missing=.*norm.weight"):
                DraftHead.from_file(p, TINY)

    def test_wrong_shape(self):
        with scratch_dir("neg") as d:
            p = os.path.join(d, "m.safetensors")
            make_tiny(p, reshape={"fc.fc.weight": (3 * TINY.hidden, TINY.hidden)})
            with self.assertRaisesRegex(ValueError, "fc.fc.weight"):
                DraftHead.from_file(p, TINY)

    def test_wrong_config_for_file(self):   # a tiny file under the real 3072-wide config must not load
        from osprey_ref import OSPREY_CONFIG
        with scratch_dir("neg") as d:
            p = os.path.join(d, "m.safetensors")
            make_tiny(p)
            with self.assertRaises(ValueError):
                DraftHead(OSPREY_CONFIG, SafeTensorsFile(p))


if __name__ == "__main__":
    unittest.main()
