# EAGLE4 URA

Hardware source and build instructions remain in [BUILD.md](BUILD.md) and
[FPGA_QUICKSTART.md](FPGA_QUICKSTART.md).

This branch adds an isolated [synthetic model asset package](assets/synthetic-osprey-2bit/README.md):
a compressed, zero-filled safetensors model, exact stored tensor layout manifest,
generator, integrity checker and smoke tests. It contains no trained weight values
and does not change RTL, build configuration or master. This is not a full draft-head
numerical reference or evidence of FPGA compatibility.

This branch also adds a [CPU-first NumPy runnable reference](reference/osprey-numpy/README.md) for that layout: loader, exact BF16 cast,
2-bit ternary unpack, a bounded single-layer draft-head forward and 30 standard-library tests. Synthetic weights only — it is not an
EAGLE-3 reference and is not verified against the GPU path or the RTL; its README states exactly what is and is not validated.
