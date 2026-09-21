"""OSPREY body-weight storage: scheme `ternary_absmean`, pack `2bit_4x_uint8_rowmajor`.

Stored form (per linear layer):  `<name>.qweight`  U8 flat, length rows*cols/4
                                 `<name>.weight_scale` F32 [1]   (ONE scalar per tensor)
  * 4 two-bit codes per byte, LSB-FIRST: code_i = (byte >> 2*i) & 3, i = 0..3
  * codes are laid out ROW-MAJOR over the logical [rows=out_features, cols=in_features] matrix
  * code c in {0, 1, 2} -> ternary weight w = c - 1 in {-1, 0, +1};  code 3 NEVER occurs (hard error here)
  * dequantised weight W = w * weight_scale ;  y = x @ W.T

Provenance of this convention: the project's own FPGA weight packer decodes the local 2-bit checkpoint this way and
re-proves, on every run, a FULL-TENSOR decode-vs-BF16-checkpoint round trip for all 7 body tensors (see HANDOFF).
That proof used private trained weights and is NOT reproduced in this repository; here the convention is exercised
only on synthetic data (pack/unpack round trip + dense-reference agreement).

NOTE the zero-filled fixture: code 0 means w = -1 (NOT 0) and every weight_scale is 0.0, so W == 0 exactly.
"""
import numpy as np


def unpack_ternary(q_u8, rows, cols):
    q = np.ascontiguousarray(q_u8).reshape(-1)
    if q.dtype != np.uint8:
        raise TypeError("qweight must be uint8, got %s" % q.dtype)
    if cols % 4 != 0:
        raise ValueError("cols=%d not divisible by 4 (4 codes per byte, row-major)" % cols)
    if q.size * 4 != rows * cols:
        raise ValueError("qweight has %d bytes; [%d,%d] needs %d" % (q.size, rows, cols, rows * cols // 4))
    codes = np.empty(q.size * 4, dtype=np.uint8)
    for i in range(4):
        codes[i::4] = (q >> np.uint8(2 * i)) & np.uint8(3)
    if (codes == 3).any():
        raise ValueError("illegal 2-bit code 3 present: not a ternary_absmean tensor (or wrong bit order)")
    return (codes.astype(np.int8) - np.int8(1)).reshape(rows, cols)


def pack_ternary(w):
    w = np.asarray(w)
    if w.ndim != 2 or w.shape[1] % 4 != 0:
        raise ValueError("expected [rows, cols] with cols % 4 == 0")
    if not np.isin(w, (-1, 0, 1)).all():
        raise ValueError("weights must be ternary {-1,0,+1}")
    c = (w.astype(np.int16) + 1).astype(np.uint8).reshape(-1, 4)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).astype(np.uint8)
