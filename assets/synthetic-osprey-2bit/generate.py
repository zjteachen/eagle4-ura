"""Header-only checkpoint inspection or manifest-only synthetic replacement.

Never reads source tensor payload or deserializes model code. Standard library only.
"""
import argparse
import json
import math
import struct
import zipfile
from pathlib import Path

SIZES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2,
         'F16': 2, 'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4,
         'I64': 8, 'U64': 8, 'F64': 8}


def header(stream):
    size = struct.unpack('<Q', stream.read(8))[0]
    if not 2 <= size <= 16 * 1024 * 1024:
        raise ValueError('Invalid/oversize header')
    return json.loads(stream.read(size)), size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--manifest', action='store_true',
                        help='Use a reviewed tensor manifest instead of a private archive')
    args = parser.parse_args()
    if args.manifest:
        manifest_input = json.loads(args.archive.read_text())
        if manifest_input.get('synthetic') is not True:
            raise ValueError('Expected a synthetic layout manifest')
        source = {}
        for item in manifest_input['tensors']:
            name = item['name']
            if not isinstance(name, str) or not name or name == '__metadata__' or name in source:
                raise ValueError('Invalid or duplicate tensor name')
            source[name] = dict(shape=item['shape'], dtype=item['dtype'],
                                data_offsets=[0, item['bytes']])
    else:
        with zipfile.ZipFile(args.archive) as archive:
            with archive.open('2bit_ckpt/model.safetensors') as stream:
                source, _ = header(stream)
    tensors = []
    for name, item in source.items():
        if name == '__metadata__':
            continue
        shape, dtype = item['shape'], item['dtype']
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid dimensions')
        count = math.prod(shape)
        size = count * SIZES[dtype]
        start, end = item['data_offsets']
        if end - start != size:
            raise ValueError('Tensor byte size mismatch: ' + name)
        tensors.append(dict(name=name, shape=shape, dtype=dtype,
                            elements=count, bytes=size))
    if args.manifest:
        for item, supplied in zip(tensors, manifest_input['tensors']):
            if item['elements'] != supplied['elements']:
                raise ValueError('Manifest element count mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'synthetic': True, 'fill': 'all-zero bytes',
                'source_variant': 'local 2bit_ckpt archive; BF16 not verified',
                'tensors': tensors}
    (args.output / 'tensor-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    layout = {'__metadata__': {'synthetic': 'true', 'purpose': 'layout testing only'}}
    offset = 0
    for item in tensors:
        layout[item['name']] = {'dtype': item['dtype'], 'shape': item['shape'],
                                'data_offsets': [offset, offset + item['bytes']]}
        offset += item['bytes']
    encoded = json.dumps(layout, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path = args.output / 'model.safetensors'
    with path.open('xb') as out:
        out.write(struct.pack('<Q', len(encoded)))
        out.write(encoded)
        # Sparse zero-filled file: logical tensor payload exists without copying weights.
        out.truncate(8 + len(encoded) + offset)
    with path.open('rb') as stream:
        actual, n = header(stream)
        for item in tensors:
            assert actual[item['name']]['shape'] == item['shape']
            assert actual[item['name']]['dtype'] == item['dtype']
        assert path.stat().st_size == 8 + n + offset
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            assert not any(chunk), 'Nonzero synthetic payload'
    report = {'tensor_count': len(tensors), 'stored_elements': sum(t['elements'] for t in tensors),
              'payload_bytes': offset, 'layout_matches_source_header': not args.manifest,
              'layout_matches_manifest': args.manifest,
              'entire_payload_verified_zero': True, 'trained_values_copied': False}
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
