# Synthetic OSPREY 2-bit storage-layout fixture

**Synthetic zero-filled weights, not trained model values.** This package contains
the actual compressed safetensors fixture, its 24-tensor manifest, a dependency-free
generator and smoke tests. It is for loading/allocation/shape plumbing only.
It is NOT an EAGLE-3/full-forward numerical reference or an RTL-equivalence result.

## Use the included model

From this directory with Python 3.9+:

```sh
python3 verify_assets.py
gzip -dk model.safetensors.gz
python3 test_generate.py
```

The extracted file is about **1.67 GB**; allow at least 2 GB free disk space.
The gzip file is small because every tensor payload byte is zero. It is a real
compressed model file, not a Git LFS pointer or a download link. No private archive,
trained checkpoint, token or third-party Python package is required.
Do not run gzip extraction if model.safetensors already exists.

Alternatively regenerate a sparse model into a NEW directory:

```sh
python3 generate.py tensor-manifest.json ./recreated --manifest
```

The SHA256 of the uncompressed model must be:

```text
195d00eb9c6573ae495bc862b4295fadb6a229adb99a2c628617f967bba4e003
```

## Layout and limits

- 24 stored tensors; 851727367 stored elements; 1668065308 tensor payload bytes.
- Names, stored shapes and dtypes match the selected local OSPREY 2-bit layout.
- BF16 embedding/LM-head tensors and U8 packed body tensors are present in this
  single mixed-storage checkpoint. This is not a second, all-BF16 model variant.
- Packed 2-bit storage cast to floats is NOT dequantization. Zero scales/norms
  can make inference degenerate. There are no meaningful numerical goldens here.
- No source checkpoint payload, model source code, prompts, vocabulary mappings,
  private operational notes or remote-machine information is included.
- No promise of compatibility with this repository's RTL geometry or weight DMA
  format: this branch adds an isolated fixture and leaves the RTL/build untouched.

`layout-verification.json` records the earlier whole-payload zero scan and manifest
match. `package-verification.json` records file integrity for the compressed asset.
`verify_assets.py` checks that compressed and expanded hashes and sizes match.
`test_generate.py` tests the generator with tiny synthetic fixtures, malformed
manifests, duplicate names and overwrite refusal; it does not run model inference.

## Regenerating the distributable

```sh
python3 generate.py tensor-manifest.json ./recreated --manifest
python3 package_model.py ./recreated/model.safetensors ./new-package
```

The new-package directory must not exist. Packaging accepts only the exact verified
synthetic-file hash above and never packages an arbitrary trained checkpoint.
