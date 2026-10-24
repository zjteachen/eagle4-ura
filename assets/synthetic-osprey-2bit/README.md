# Synthetic OSPREY 2-bit storage-layout fixture

This package contains an all-zero safetensors fixture matching a 24-tensor OSPREY 2-bit storage layout. It exercises model loading, allocation, and shape-handling code paths. The payload is all zeros, so numerical golden values do not exist.

## Use the included model
Run these commands from this directory:
```sh
python3 verify_assets.py
gzip -dk model.safetensors.gz
python3 test_generate.py
```
The expanded model is 1.67 GB; allow about 2 GB of free disk space. Its uncompressed SHA256 is:
```text
195d00eb9c6573ae495bc862b4295fadb6a229adb99a2c628617f967bba4e003
```
If `model.safetensors` already exists, `gzip -dk` asks whether to overwrite it.

## Layout
- 24 tensors, 851,727,367 stored elements, and 1,668,065,308 payload bytes.
- BF16 embedding and LM-head tensors and U8 packed 2-bit body tensors share the checkpoint; names, shapes, and dtypes are listed in `tensor-manifest.json`.
- `layout-verification.json` records the zero-payload scan and manifest match. `package-verification.json` records compressed and expanded hashes and sizes.
- `verify_assets.py` re-checks both assets without extracting the expanded file.

## Regenerate
`generate.py` writes a manifest-defined safetensors file using the Python standard library:
```sh
python3 generate.py tensor-manifest.json ./recreated --manifest
```
`package_model.py` creates the compressed distributable and accepts only the verified uncompressed SHA256:
```sh
python3 package_model.py ./recreated/model.safetensors ./new-package
```
`test_generate.py` covers generator smoke cases for tiny fixtures, malformed manifests, duplicate names, and overwrite refusal.
