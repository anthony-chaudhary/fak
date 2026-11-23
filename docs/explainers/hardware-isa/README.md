---
title: "The hardware instruction-set ladder: five rungs down to silicon"
description: "The instruction sets a machine executes, and the software layers that carry your code down to them, explained five times with fak as the worked example."
slug: hardware-isa-ladder
keywords:
  - what is an instruction set
  - instruction set architecture explained
  - SIMD explained
  - AVX-512 vs NEON
  - PTX and SASS explained
  - tensor core mma.sync
  - how a GPU driver works
  - SPIR-V explained
  - beginner computer architecture guide
  - advanced instruction set guide
date: 2026-11-18
---

# The hardware instruction-set ladder

> **In one breath:** A processor only understands a fixed list of tiny commands. That
> list is its instruction set. Your Go code is not on that list, so something must
> translate it. This ladder follows that translation all the way down.

**One line:** the instruction set is the contract between software and silicon — the
complete, frozen vocabulary a given chip will execute — and every layer between your
source file and that chip exists to turn your intent into words the chip already knows.

> **Audience.** Five readers, one per rung. Rung 1 assumes nothing beyond "I have written
> a program." Rung 5 assumes you are comfortable with roofline reasoning and want the
> honest edges. Each rung assumes exactly what the one below it taught, so you can start
> at the bottom and climb, or drop in at the rung that matches what you already know.

## Why does this subject need a ladder?

fak runs a language model's forward pass itself, in-process. That forward pass is, at the
bottom, an enormous number of multiply-and-add operations over quantized integers. The
same arithmetic can execute on a laptop CPU, an Apple laptop's ARM cores, an NVIDIA GPU,
an AMD integrated GPU, or a phone. Each of those is a different machine with a different
instruction set, and fak reaches all of them through one interface.

That is exactly the situation where a single flat page fails. A newcomer who does not yet
know what an instruction *is* drowns in `VPMADDWD` before they learn why anyone cares. A
senior engineer who already knows what a tensor core is gets bored by the analogy. The
ladder splits the difference: five rungs, five audiences, one subject.

## The five rungs

| Level | Who it's for | What you'll get |
|---|---|---|
| **[1 · What is an instruction set?](level-1-what-is-an-instruction-set.md)** | A student or new grad. You have written code; you have never thought about what the chip does with it. | What an instruction is, why a chip has a fixed list of them, and the four-layer journey from your source file to a voltage. |
| **[2 · One instruction, many numbers](level-2-simd-one-instruction-many-numbers.md)** | A competent developer. You know what a loop is and you have heard the word "SIMD". | How one instruction operates on 16 or 32 numbers at once, why quantized inference is a dot-product problem, and how fak's AVX-512 and NEON kernels stay bit-identical to the scalar reference. |
| **[3 · The GPU instruction set](level-3-the-gpu-isa-ptx-sass-and-tensor-cores.md)** | A platform or systems engineer. You have written or read CUDA. | Why a GPU has *two* instruction sets, what `mma.sync` and `cp.async` actually ask the hardware to do, and why the same source compiles to different machine code per chip. |
| **[4 · How the ISA reaches silicon](level-4-how-the-isa-reaches-silicon.md)** | A senior engineer. You want the whole stack, not the arithmetic. | The full driver path: Go to cgo to a static library to a vendor runtime to a kernel driver to the chip, plus the shader path where the driver compiles your code at runtime. |
| **[5 · Why the ISA is not the whole story](level-5-why-the-isa-is-not-the-whole-story.md)** | A researcher or a `fable`-tier reader. You want the limits and the open edges. | Why knowing the instruction set does not predict speed, what the roofline says instead, and which claims fak has actually witnessed. |

## How to read it

- **Never thought about hardware?** Read **Rung 1**, then **Rung 2**, and stop. You will
  know what an instruction set is and why it decides how fast a model runs.
- **Writing or reading kernels?** **Rung 3** is the GPU instruction set; **Rung 4** is
  how it gets there.
- **Deciding what to believe?** **Rung 5** is the honest accounting.

Each rung ends with a nav footer linking its neighbours and back here. Every rung carries
a checkpoint you answer yourself before moving on.

## The one loop that makes this stick

Do not memorize mnemonics. Use the same five-step loop on every rung:

1. **Predict** one observable result in plain language.
2. **Run** the command or read the code the rung points at.
3. **Locate** the line that produced what you saw.
4. **Adjust** exactly one input.
5. **Rerun** and explain the difference.

That loop is the same one [`LEARNING-PATH.md`](../../../LEARNING-PATH.md) teaches for the
whole of fak. This ladder is one subject run through it five times.

## See also

- [What is a CUDA kernel?](../what-is-a-cuda-kernel.md) — the depth ladder from Go to
  silicon, and the three senses of the word "kernel". Rung 3 and Rung 4 build on it.
- [Hardware portability](../hardware-portability.md) — the HAL seam that lets one
  forward pass run on four different machines. Rung 4 is the layer beneath it.
- [The hardware matrix](../../HARDWARE-MATRIX.md) — every platform fak has measured,
  with the instruction sets and backends each one uses.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · next → [Rung 1 — What is an instruction set?](level-1-what-is-an-instruction-set.md)*
