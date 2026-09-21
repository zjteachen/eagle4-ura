"""Portable stdlib smoke tests; no trained checkpoints or third-party packages."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('generate.py')


class ManifestGenerationTest(unittest.TestCase):
    def run_generator(self, root, tensors):
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'synthetic': True, 'tensors': tensors}))
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(manifest), str(root / 'out'), '--manifest'],
            capture_output=True, text=True,
        )

    def tensor(self):
        return dict(name='head.weight', shape=[2, 3], dtype='F32', elements=6, bytes=24)

    def test_valid_layout_is_zero_and_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_generator(root, [self.tensor()])
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / 'out/model.safetensors').open('rb') as stream:
                n = struct.unpack('<Q', stream.read(8))[0]
                header = json.loads(stream.read(n))
                self.assertEqual(header['head.weight']['shape'], [2, 3])
                self.assertEqual(header['head.weight']['dtype'], 'F32')
                self.assertEqual(header['head.weight']['data_offsets'], [0, 24])
                self.assertEqual(stream.read(), bytes(24))
            report = json.loads((root / 'out/verification.json').read_text())
            self.assertTrue(report['layout_matches_manifest'])
            self.assertFalse(report['layout_matches_source_header'])
            before = (root / 'out/model.safetensors').read_bytes()
            self.assertNotEqual(self.run_generator(root, [self.tensor()]).returncode, 0)
            self.assertEqual((root / 'out/model.safetensors').read_bytes(), before)

    def test_inconsistent_manifest_rejected_before_output(self):
        for change in ({'elements': 7}, {'bytes': 30}, {'shape': [-1]}, {'dtype': 'INVALID'}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_generator(root, [{**self.tensor(), **change}])
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / 'out').exists())

    def test_duplicate_names_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertNotEqual(self.run_generator(root, [self.tensor()] * 2).returncode, 0)
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
