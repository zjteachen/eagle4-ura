# Instruction set design for EAGLE4
This document is for storing the current ISA design for the EAGLE4 drafter.
Iteration 1 of the ISA:
- 8 bit word width.
- bits [7:6] indicate the instruction type:
  - 00: state instruction. 
  - 01: jump instruction.
  - 10: unused.
  - 11: signal instruction, for debugging.
The goal is to begin with only state/jmp, then steadily replace state instructions
with more traditional instructions that activate specific pieces of hardware.
- bits [5:1] are a payload field, used to indicate state, or jump address.
State instructions:
0:
1:
...
