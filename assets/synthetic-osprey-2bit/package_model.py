"""Package only the exact verified synthetic fixture, never arbitrary model weights."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED = '195d00eb9c6573ae495bc862b4295fadb6a229adb99a2c628617f967bba4e003'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if digest(args.model) != EXPECTED:
        raise ValueError('Refusing to package a file other than the verified synthetic model')
    args.output.mkdir(parents=True, exist_ok=False)
    packed = args.output / 'model.safetensors.gz'
    with args.model.open('rb') as source, packed.open('xb') as destination:
        with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as zipped:
            shutil.copyfileobj(source, zipped, 1024 * 1024)
    report = dict(synthetic=True, uncompressed_sha256=EXPECTED,
                  uncompressed_bytes=args.model.stat().st_size,
                  compressed_sha256=digest(packed), compressed_bytes=packed.stat().st_size)
    (args.output / 'package-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
