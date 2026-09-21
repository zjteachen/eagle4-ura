"""float32 NumPy operators. Plain, unfused, readable -- this is a reference, not a fast path."""
import numpy as np


def rmsnorm(x, gamma, eps):
    x = np.asarray(x, dtype=np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(ms + np.float32(eps))).astype(np.float32) * np.asarray(gamma, dtype=np.float32)


def silu(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.float32(1.0) + np.exp(-x))


def rope_neox(x, positions, theta):
    """x: [T, H, D] (D even). NeoX / rotate-half pairing: element i pairs with element i + D/2."""
    x = np.asarray(x, dtype=np.float32)
    T, H, D = x.shape
    half = D // 2
    inv = np.float32(theta) ** (-np.arange(0, half, dtype=np.float32) * np.float32(2.0) / np.float32(D))
    ang = np.asarray(positions, dtype=np.float32)[:, None] * inv[None, :]          # [T, half]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(np.float32)


def causal_attention(q, k, v, allow=None):
    """q: [Tq,H,D], k/v: [Tk,H,D]. allow: bool [Tq,Tk] (True = may attend). Default = causal over aligned suffix."""
    Tq, H, D = q.shape
    Tk = k.shape[0]
    if allow is None:
        allow = np.arange(Tk)[None, :] <= (np.arange(Tq)[:, None] + (Tk - Tq))
    if not allow.any(axis=1).all():
        raise ValueError("a query row attends to nothing (softmax undefined)")
    sc = np.einsum("qhd,khd->hqk", q, k).astype(np.float32) / np.float32(np.sqrt(D))
    sc = np.where(allow[None, :, :], sc, np.float32(-np.inf))
    sc = sc - sc.max(axis=-1, keepdims=True)
    p = np.exp(sc)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, v).astype(np.float32)
