"""Verify compressed and expanded fixture bytes without extracting a 1.67 GB file."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def measure(stream):
    h = hashlib.sha256()
    count = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
        count += len(chunk)
    return h.hexdigest(), count


def main():
    report = json.loads((ROOT / 'package-verification.json').read_text())
    if report.get('synthetic') is not True:
        raise ValueError('Expected a synthetic fixture')
    path = ROOT / 'model.safetensors.gz'
    with path.open('rb') as stream:
        if measure(stream) != (report['compressed_sha256'], report['compressed_bytes']):
            raise ValueError('Compressed asset integrity failure')
    with gzip.open(path, 'rb') as stream:
        if measure(stream) != (report['uncompressed_sha256'], report['uncompressed_bytes']):
            raise ValueError('Expanded asset integrity failure')
    print('PASS: compressed and expanded asset SHA256 and sizes match')


if __name__ == '__main__':
    main()
