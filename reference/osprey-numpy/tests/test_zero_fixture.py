"""SHAPE PLUMBING ONLY: the published 24-tensor zero-filled OSPREY layout loads and a bounded forward runs end to end.

Every output is EXACTLY ZERO by construction (zero norm gammas, zero scales, zero dense weights). That is asserted, not hidden:
this test proves allocation/shape/dtype/casting/unpack plumbing at FULL model size and proves nothing about numerics.
Set OSPREY_REF_SKIP_FULL=1 to skip (needs ~1.7 GB of SPARSE file + ~1 GB RAM).
"""
import json
import os
import subprocess
import sys
import unittest
import numpy as np
from tests._fixtures import scratch_dir
from osprey_ref import DraftHead, OSPREY_CONFIG, SafeTensorsFile
from osprey_ref.model import expected_layout

ASSETS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "synthetic-osprey-2bit"))


@unittest.skipIf(os.environ.get("OSPREY_REF_SKIP_FULL") == "1", "OSPREY_REF_SKIP_FULL=1")
@unittest.skipUnless(os.path.isfile(os.path.join(ASSETS, "generate.py")), "assets/synthetic-osprey-2bit not present")
class TestZeroFixture(unittest.TestCase):
    def test_expected_layout_equals_published_manifest(self):
        with open(os.path.join(ASSETS, "tensor-manifest.json")) as fh:
            man = json.load(fh)
        pub = {t["name"]: (t["dtype"], tuple(t["shape"])) for t in man["tensors"]}
        self.assertEqual(pub, expected_layout(OSPREY_CONFIG))        # all 24 names, dtypes and shapes

    def test_load_and_bounded_forward_full_size(self):
        with scratch_dir("zero") as d:
            out = os.path.join(d, "recreated")
            subprocess.run([sys.executable, os.path.join(ASSETS, "generate.py"), os.path.join(ASSETS, "tensor-manifest.json"),
                            out, "--manifest"], check=True, stdout=subprocess.DEVNULL)
            path = os.path.join(out, "model.safetensors")
            self.assertEqual(len(SafeTensorsFile(path).names()), 24)
            head = DraftHead.from_file(path)
            for lin in head.lin.values():                    # zero byte == code 0 == w -1 ; scale 0.0 -> W == 0
                self.assertTrue((lin.w == -1).all())
                self.assertEqual(float(lin.scale), 0.0)
            rng = np.random.default_rng(0)
            ids = np.array([0, 1, 128255])
            hid = rng.standard_normal((3, 3 * OSPREY_CONFIG.hidden)).astype(np.float32)
            o = head.forward(head.embed(ids), hid, [0, 1, 2])
            lg = head.logits(o["to_logits"])
            self.assertEqual(o["to_logits"].shape, (3, 3072))
            self.assertEqual(o["for_reasoning"].shape, (3, 3072))
            self.assertEqual(o["kv"][0].shape, (3, 32, 96))
            self.assertEqual(lg.shape, (3, 128256))
            for a in (o["to_logits"], o["for_reasoning"], lg):
                self.assertTrue(np.isfinite(a).all())
                self.assertTrue((a == 0).all(), "zero fixture must give EXACT zeros -- anything else means stray state")


if __name__ == "__main__":
    unittest.main()
