#!/usr/bin/env python3
"""Bounded CPU forward on a 24-tensor OSPREY 2-bit safetensors file. NumPy only.

  python3 run_smoke.py --model /path/to/model.safetensors [--tokens 0 1 2] [--seed 0] [--topk 5]

With the published ZERO-FILLED fixture every output is exactly 0.0 -- that is reported as DEGENERATE-BY-CONSTRUCTION, never as a
numerical result. Exit status: 0 = ran and outputs finite; 2 = load/shape error; 3 = non-finite output.
"""
import argparse
import sys
import time
import numpy as np
from osprey_ref import DraftHead, OSPREY_CONFIG


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    try:
        head = DraftHead.from_file(a.model, OSPREY_CONFIG)
    except (ValueError, KeyError, OSError) as e:
        print("LOAD FAILED: %s" % e)
        return 2
    print("loaded 24 tensors, unpacked 7 ternary matrices in %.1f s" % (time.time() - t0))
    for n, lin in sorted(head.lin.items()):
        vals, cnt = np.unique(lin.w, return_counts=True)
        print("  %-10s %-14s scale=%-12.6g ternary histogram %s" % (n, lin.w.shape, float(lin.scale), dict(zip(vals.tolist(), cnt.tolist()))))
    rng = np.random.default_rng(a.seed)
    T = len(a.tokens)
    hidden = rng.standard_normal((T, 3 * OSPREY_CONFIG.hidden)).astype(np.float32)   # SYNTHETIC stand-in for the target's 3 aux hidden states
    t1 = time.time()
    out = head.forward(head.embed(a.tokens), hidden, np.arange(T))
    logits = head.logits(out["to_logits"])
    print("forward T=%d in %.2f s: to_logits %s for_reasoning %s logits %s kv %s" % (
        T, time.time() - t1, out["to_logits"].shape, out["for_reasoning"].shape, logits.shape, out["kv"][0].shape))
    for u in head.UNVERIFIED:
        print("UNVERIFIED: " + u)
    if not all(np.isfinite(x).all() for x in (out["to_logits"], out["for_reasoning"], logits)):
        print("NON-FINITE OUTPUT")
        return 3
    if not logits.any():
        print("RESULT: all logits are exactly 0.0 -> DEGENERATE BY CONSTRUCTION (zero-filled fixture). Shape/dtype/unpack plumbing only.")
    else:
        top = np.argsort(-logits[-1])[:a.topk]
        print("last-position top-%d ids %s logits %s  (meaningful ONLY if the file holds real weights)" % (a.topk, top.tolist(), np.round(logits[-1][top], 4).tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
