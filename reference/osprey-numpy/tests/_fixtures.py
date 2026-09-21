"""Synthetic RANDOM tiny checkpoints in the exact 24-tensor naming/packing scheme. No trained values anywhere."""
import os
import sys
import tempfile
import contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from osprey_ref.model import Config, _BODY, _NORMS          # noqa: E402
from osprey_ref.safetensors_io import write_safetensors, f32_to_bf16, bf16_to_f32   # noqa: E402
from osprey_ref.ternary import pack_ternary                 # noqa: E402

TINY = Config(hidden=48, n_heads=4, head_dim=12, inter=64, vocab=97)


@contextlib.contextmanager
def scratch_dir(tag):
    """Fresh directory. If OSPREY_REF_TMP is set the directory is created under it and KEPT (create-only hosts)."""
    base = os.environ.get("OSPREY_REF_TMP")
    if base:
        os.makedirs(base, exist_ok=True)
        yield tempfile.mkdtemp(prefix=tag + "_", dir=base)
    else:
        with tempfile.TemporaryDirectory(prefix=tag + "_") as d:
            yield d


def make_tiny(path, cfg=TINY, seed=0, drop=None, reshape=None):
    """Write a random tiny model. Returns the DENSE float64 truth {name: array} for an independent reference."""
    rng = np.random.default_rng(seed)
    dim = {"hidden": cfg.hidden, "2hidden": 2 * cfg.hidden, "inter": cfg.inter}
    tensors, truth = [], {}

    def bf(name, shape, scale=1.0, offset=0.0):
        u = f32_to_bf16((rng.standard_normal(shape) * scale + offset).astype(np.float32))
        truth[name] = bf16_to_f32(u).astype(np.float64)       # truth is what is STORED (post-truncation)
        tensors.append((name, "BF16", shape, u))

    bf("embed_tokens.weight", (cfg.vocab, cfg.hidden))
    bf("lm_head.weight", (cfg.vocab, cfg.hidden), 0.2)
    bf("fc.fc.weight", (cfg.hidden, 3 * cfg.hidden), 0.1)
    for n in _NORMS:
        bf(n, (cfg.hidden,), 0.1, 1.0)
    for n, (r, c) in _BODY.items():
        w = rng.integers(-1, 2, size=(dim[r], dim[c])).astype(np.int8)
        s = np.float32(rng.uniform(0.01, 0.06))
        truth[n] = w.astype(np.float64) * np.float64(s)
        tensors.append((n + ".qweight", "U8", (w.size // 4,), pack_ternary(w)))
        tensors.append((n + ".weight_scale", "F32", (1,), np.array([s], dtype=np.float32)))
    if drop:
        tensors = [t for t in tensors if t[0] != drop]
    if reshape:
        tensors = [(n, d, reshape[n] if n in reshape else s, a) for n, d, s, a in tensors]
    write_safetensors(path, tensors, {"synthetic": "true", "purpose": "unit test"})
    return truth
