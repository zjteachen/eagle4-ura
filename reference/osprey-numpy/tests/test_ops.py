import unittest
import numpy as np
from tests import _fixtures  # noqa: F401  (sets sys.path)
from osprey_ref import rmsnorm, rope_neox, silu, causal_attention


class TestOps(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_rmsnorm_matches_float64_definition(self):
        x = self.rng.standard_normal((5, 48)).astype(np.float32)
        g = self.rng.standard_normal(48).astype(np.float32)
        x64 = x.astype(np.float64)
        want = x64 / np.sqrt((x64 ** 2).mean(-1, keepdims=True) + 1e-5) * g.astype(np.float64)
        np.testing.assert_allclose(rmsnorm(x, g, 1e-5), want, rtol=2e-6, atol=2e-6)

    def test_rmsnorm_of_zero_is_zero_not_nan(self):
        self.assertTrue((rmsnorm(np.zeros((2, 8), np.float32), np.ones(8, np.float32), 1e-5) == 0).all())

    def test_silu(self):
        x = np.linspace(-6, 6, 25).astype(np.float32)
        np.testing.assert_allclose(silu(x), x.astype(np.float64) / (1 + np.exp(-x.astype(np.float64))), rtol=1e-6, atol=1e-7)

    def test_rope_position_zero_is_identity_and_norm_preserving(self):
        x = self.rng.standard_normal((3, 4, 12)).astype(np.float32)
        np.testing.assert_allclose(rope_neox(x, [0, 0, 0], 10000.0), x, rtol=0, atol=0)
        y = rope_neox(x, [3, 11, 250], 10000.0)
        np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=2e-6)

    def test_rope_is_relative(self):   # <R_m q, R_n k> depends only on m-n
        q = self.rng.standard_normal((1, 2, 12)).astype(np.float32)
        k = self.rng.standard_normal((1, 2, 12)).astype(np.float32)
        a = (rope_neox(q, [9], 1e4) * rope_neox(k, [4], 1e4)).sum(-1)
        b = (rope_neox(q, [109], 1e4) * rope_neox(k, [104], 1e4)).sum(-1)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_rope_matches_complex_rotation(self):   # independent formulation: pair (i, i+D/2) as a complex number
        x = self.rng.standard_normal((2, 3, 12)).astype(np.float32)
        pos = np.array([5, 17])
        z = x[..., :6].astype(np.float64) + 1j * x[..., 6:].astype(np.float64)
        ang = pos[:, None] * (10000.0 ** (-np.arange(6) * 2.0 / 12))[None, :]
        zr = z * np.exp(1j * ang)[:, None, :]
        np.testing.assert_allclose(rope_neox(x, pos, 10000.0), np.concatenate([zr.real, zr.imag], -1), rtol=2e-5, atol=2e-6)

    def test_attention_single_token_returns_v(self):
        q, k, v = (self.rng.standard_normal((1, 4, 12)).astype(np.float32) for _ in range(3))
        np.testing.assert_allclose(causal_attention(q, k, v), v, rtol=1e-6)

    def test_attention_is_causal(self):   # planted perturbation of a FUTURE token must not move earlier outputs
        q, k, v = (self.rng.standard_normal((5, 4, 12)).astype(np.float32) for _ in range(3))
        base = causal_attention(q, k, v)
        k2, v2 = k.copy(), v.copy()
        k2[4] += 3.0
        v2[4] -= 2.0
        pert = causal_attention(q, k2, v2)
        np.testing.assert_array_equal(base[:4], pert[:4])
        self.assertFalse(np.allclose(base[4], pert[4]))      # and the control DOES move the row that may see it

    def test_attention_rejects_a_row_that_sees_nothing(self):
        q, k, v = (self.rng.standard_normal((2, 1, 4)).astype(np.float32) for _ in range(3))
        with self.assertRaises(ValueError):
            causal_attention(q, k, v, allow=np.array([[True, False], [False, False]]))


if __name__ == "__main__":
    unittest.main()
