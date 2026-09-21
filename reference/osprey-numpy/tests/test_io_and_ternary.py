import os
import unittest
import numpy as np
from tests._fixtures import scratch_dir, make_tiny, TINY
from osprey_ref import bf16_to_f32, f32_to_bf16, unpack_ternary, pack_ternary, SafeTensorsFile


def bf16_decode_independent(u):
    """Field-by-field IEEE decode in float64 -- a different code path from the bit-shift used by the library."""
    u = int(u)
    s, e, m = (u >> 15) & 1, (u >> 7) & 0xFF, u & 0x7F
    if e == 0xFF:
        return float("nan") if m else (float("-inf") if s else float("inf"))
    v = (m / 128.0) * 2.0 ** (-126) if e == 0 else (1.0 + m / 128.0) * 2.0 ** (e - 127)
    return -v if s else v


class TestBF16(unittest.TestCase):
    def test_exact_against_independent_field_decode_all_65536_patterns(self):
        u = np.arange(65536, dtype=np.uint16)
        with np.errstate(invalid="ignore"):        # widening a signalling-NaN bit pattern warns; NaNs are compared by isnan below
            got = bf16_to_f32(u).astype(np.float64)
        want = np.array([bf16_decode_independent(x) for x in u])
        nan = np.isnan(want)
        self.assertTrue(np.array_equal(np.isnan(got), nan))
        self.assertTrue(np.array_equal(got[~nan], want[~nan]))          # EXACT, incl. subnormals, +-0, +-inf
        self.assertTrue(np.array_equal(np.signbit(got[~nan]), np.signbit(want[~nan])))

    def test_roundtrip_bits(self):
        u = np.arange(65536, dtype=np.uint16)
        self.assertTrue(np.array_equal(f32_to_bf16(bf16_to_f32(u)), u))

    def test_rejects_non_uint16(self):
        with self.assertRaises(TypeError):
            bf16_to_f32(np.zeros(4, dtype=np.float16))


class TestTernary(unittest.TestCase):
    def test_known_answer_lsb_first(self):
        # 0x92 = 0b10_01_00_10 -> LSB-first codes [2,0,1,2] -> w [+1,-1,0,+1]
        self.assertEqual(unpack_ternary(np.array([0x92], np.uint8), 1, 4).tolist(), [[1, -1, 0, 1]])

    def test_bit_order_is_not_msb_first(self):   # planted control: an MSB-first decoder gives a DIFFERENT answer
        msb_first = [((0x92 >> s) & 3) - 1 for s in (6, 4, 2, 0)]
        self.assertNotEqual(unpack_ternary(np.array([0x92], np.uint8), 1, 4).tolist()[0], msb_first)

    def test_row_major(self):
        w = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [-1, -1, -1, -1, 1, 0, -1, 0]], np.int8)
        q = pack_ternary(w)
        # row0: w=+1 -> code 2 x4 -> 0b10101010 = 0xAA ; w=0 -> code 1 x4 -> 0x55
        # row1: w=-1 -> code 0 x4 -> 0x00 ; w=[+1,0,-1,0] -> codes [2,1,0,1] -> 2 | 1<<2 | 0<<4 | 1<<6 = 0b01_00_01_10 = 0x46
        self.assertEqual(q.tolist(), [0xAA, 0x55, 0x00, 0x46])
        self.assertTrue(np.array_equal(unpack_ternary(q, 2, 8), w))

    def test_roundtrip_random(self):
        rng = np.random.default_rng(1)
        w = rng.integers(-1, 2, size=(37, 64)).astype(np.int8)
        self.assertTrue(np.array_equal(unpack_ternary(pack_ternary(w), 37, 64), w))

    def test_zero_byte_means_minus_one_not_zero(self):   # the zero-filled fixture trap
        self.assertTrue((unpack_ternary(np.zeros(8, np.uint8), 4, 8) == -1).all())

    def test_code3_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            unpack_ternary(np.array([0x03], np.uint8), 1, 4)

    def test_size_and_divisibility_errors(self):
        with self.assertRaises(ValueError):
            unpack_ternary(np.zeros(3, np.uint8), 2, 8)
        with self.assertRaises(ValueError):
            unpack_ternary(np.zeros(3, np.uint8), 2, 6)
        with self.assertRaises(TypeError):
            unpack_ternary(np.zeros(4, np.int8), 2, 8)


class TestSafetensors(unittest.TestCase):
    def test_refuses_to_cast_packed_storage_to_float(self):
        with scratch_dir("st") as d:
            p = os.path.join(d, "model.safetensors")
            make_tiny(p)
            st = SafeTensorsFile(p)
            self.assertEqual(len(st.names()), 24)
            with self.assertRaises(TypeError):
                st.f32("midlayer.mlp.down_proj.qweight")
            self.assertEqual(st.info("embed_tokens.weight"), ("BF16", (TINY.vocab, TINY.hidden)))


if __name__ == "__main__":
    unittest.main()
