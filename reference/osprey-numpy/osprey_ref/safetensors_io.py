"""Minimal safetensors reader/writer (NumPy + stdlib only). Tensors are memory-mapped, never copied whole.

dtype handling / casting path:
  BF16 has no NumPy dtype. It is exposed as a uint16 VIEW and converted with bf16_to_f32(), which is EXACT:
  bfloat16 is, bit for bit, the top 16 bits of an IEEE-754 float32 (1 sign, 8 exponent, 7 mantissa), so
  `(u16 << 16)` reinterpreted as float32 is the same real number (zeros, subnormals, inf and NaN included).
  f32_to_bf16() is the truncating inverse used ONLY to build synthetic test fixtures.
"""
import json
import struct
import numpy as np

_NP = {"BOOL": np.bool_, "U8": np.uint8, "I8": np.int8, "I16": np.int16, "U16": np.uint16, "F16": np.float16,
       "BF16": np.uint16, "I32": np.int32, "U32": np.uint32, "F32": np.float32, "I64": np.int64,
       "U64": np.uint64, "F64": np.float64}


def bf16_to_f32(u16):
    u16 = np.asarray(u16)
    if u16.dtype != np.uint16:
        raise TypeError("bf16_to_f32 expects the raw uint16 view of a BF16 tensor, got %s" % u16.dtype)
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16(x):
    """Truncate float32 -> bfloat16 bit pattern (uint16). Fixture construction only (round-toward-zero in magnitude)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return (x.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


class SafeTensorsFile:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            if not 2 <= n <= 16 * 1024 * 1024:
                raise ValueError("invalid/oversize safetensors header (%d bytes)" % n)
            self.header = json.loads(fh.read(n))
        self.data_start = 8 + n
        self.metadata = self.header.pop("__metadata__", {})
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def names(self):
        return sorted(self.header)

    def info(self, name):
        e = self.header[name]
        return e["dtype"], tuple(e["shape"])

    def raw(self, name):
        """Zero-copy view in the STORED dtype (BF16 -> uint16 view; use f32() for values)."""
        e = self.header[name]
        a, b = e["data_offsets"]
        dt = np.dtype(_NP[e["dtype"]])
        count = int(np.prod(e["shape"], dtype=np.int64)) if e["shape"] else 1
        if b - a != count * dt.itemsize:
            raise ValueError("%s: byte size %d != %d elements x %d" % (name, b - a, count, dt.itemsize))
        buf = self._mm[self.data_start + a:self.data_start + b]
        return buf.view(dt).reshape(e["shape"])

    def f32(self, name):
        """Materialise as float32 (exact for BF16/F32/F16; refuses packed U8 -- casting packed codes is NOT dequantisation)."""
        dt, _ = self.info(name)
        r = self.raw(name)
        if dt == "BF16":
            return bf16_to_f32(np.ascontiguousarray(r))
        if dt in ("F32", "F16", "F64"):
            return np.asarray(r, dtype=np.float32)
        raise TypeError("%s is %s: refuse to cast packed/integer storage to float. Use ternary.unpack_ternary()." % (name, dt))


def write_safetensors(path, tensors, metadata=None):
    """tensors: list of (name, st_dtype, shape, numpy array already in the stored representation). Create-only."""
    layout, off, blobs = {}, 0, []
    if metadata:
        layout["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for name, st_dtype, shape, arr in tensors:
        b = np.ascontiguousarray(arr).tobytes()
        layout[name] = {"dtype": st_dtype, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    enc = json.dumps(layout, separators=(",", ":")).encode()
    enc += b" " * (-len(enc) % 8)
    with open(path, "xb") as fh:
        fh.write(struct.pack("<Q", len(enc)))
        fh.write(enc)
        for b in blobs:
            fh.write(b)
