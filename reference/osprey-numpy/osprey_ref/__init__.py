"""CPU-first NumPy runnable reference for the published synthetic OSPREY 2-bit storage layout.

NOT an EAGLE-3 reference. NOT verified numerically equivalent to the SGLang/GPU path or to any RTL.
See README.md for exactly what is and is not validated.
"""
from .safetensors_io import SafeTensorsFile, bf16_to_f32, f32_to_bf16
from .ternary import unpack_ternary, pack_ternary
from .ops import rmsnorm, rope_neox, silu, causal_attention
from .model import Config, DraftHead, OSPREY_CONFIG

__all__ = ["SafeTensorsFile", "bf16_to_f32", "f32_to_bf16", "unpack_ternary", "pack_ternary",
           "rmsnorm", "rope_neox", "silu", "causal_attention", "Config", "DraftHead", "OSPREY_CONFIG"]
