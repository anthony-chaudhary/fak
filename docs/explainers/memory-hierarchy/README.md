---
title: "The fak memory-hierarchy ladder: five guides, five audiences"
description: "One subject — how bytes move through the memory hierarchy while a language model runs — explained five times, each rung pitched at a different reader. Start at Level 1 (what is a buffer?) and climb to Level 5 (host DRAM, offload, and the frontier). Every claim is pinned to a real file in this repository."
slug: memory-hierarchy-ladder
keywords:
  - memory hierarchy explained
  - SRAM vs HBM vs DRAM
  - shared memory CUDA
  - roofline model
  - arithmetic intensity
  - memory bandwidth bound
  - LLM inference memory
  - buffer hierarchy
date: 2026-11-18
---

# The fak memory-hierarchy ladder

> **In one breath:** A model is a pile of numbers. Running it means *moving* those
> numbers — and the machine has several places to keep them, each one faster and
> smaller than the last. Almost everything that makes inference fast or slow is a
> decision about which of those places a byte is in.

**One line:** The compute unit can only do arithmetic on numbers that are already
inside it, so every operation is preceded by a fetch, and the fetch is what costs
you. This ladder follows one byte from the disk it was stored on to the register
that multiplied it.

> **Who this is for.** One subject — *where the numbers live while the model runs,
> and what it costs to move them* — written **five times**, once for each kind of
> reader. A newcomer with no math gets Level 1. A new grad who can read Go gets
> Level 2. A working engineer who writes kernels gets Level 3. A performance
> engineer who tunes serving gets Level 4. A researcher or systems lead gets
> Level 5. Each rung assumes exactly what the rung below it taught, and nothing
> more.

Most explanations of "how inference works" describe the *math* — attention,
matrix multiplies, softmax — and stop there. But the math is the cheap part. A
modern accelerator can do an enormous number of multiplies per second and can
only *feed* them a tiny fraction of that rate, so the real story is the plumbing.
This ladder is about the plumbing.

## The one-sentence answer (all five rungs agree on this)

A compute unit can only multiply numbers that are already in its own registers.
Everything else — the model weights, the activations, the KV cache — has to be
**fetched** first, from a place that is bigger but slower. The hierarchy is the
list of those places, ordered by how close they are to the arithmetic. Inference
is fast when the bytes you need are already close, and slow when they are not.

## The five rungs

| Level | Who it's for | The question it answers |
|---|---|---|
| **[1 · What is a buffer?](level-1-what-is-a-buffer.md)** | A total newcomer. No math, no code. | "Where do the numbers actually *live* while the model is thinking?" |
| **[2 · The forward pass as a sequence of buffer moves](level-2-the-forward-pass-as-buffer-moves.md)** | A new grad who can read Go. | "Show me one token's journey through the code, and tell me which buffer each step touches." |
| **[3 · SRAM and the kernel's own scratch](level-3-sram-and-the-kernels-own-scratch.md)** | A working engineer who writes or reads kernels. | "Why does a kernel declare a tiny array at the top, and why is that array the whole trick?" |
| **[4 · HBM, VRAM, and the roofline](level-4-hbm-vram-and-the-roofline.md)** | A performance engineer. | "Which resource is the binding constraint — compute or bandwidth — and how do I tell?" |
| **[5 · Host DRAM, offload, and the frontier](level-5-host-dram-offload-and-the-frontier.md)** | A researcher or systems lead. | "What happens when the model does not fit, and what is still unsolved?" |

## How to read it

- **Curious, not technical?** Read **Level 1** and stop. It is the whole idea in
  plain words, with a coat-check analogy and no arithmetic.
- **Learning to work on inference code?** Read **1 → 2 → 3**. Level 2 walks the
  real Go forward pass; Level 3 opens the kernels it calls.
- **Tuning or serving a model?** **Level 4** is the one that changes your
  decisions — it is where "why is this slow?" gets an answer you can act on.
- **Doing research or setting direction?** **Level 5** names the frontier and,
  honestly, the parts nobody has solved.

Each rung ends with a nav footer linking its neighbours and back here.

## The hierarchy, in one table

Read this once now; every rung is a zoom-in on one row.

| Tier | Roughly how big | Roughly how far | Who owns it | In this repo |
|---|---|---|---|---|
| **Registers** | bytes per thread | zero — it *is* the arithmetic | the instruction | the multiply itself |
| **SRAM / shared memory** | tens of KB per block | on the same die | the kernel author, declared by hand | `__shared__` arrays in `internal/compute/*.cu` |
| **L1 / L2 cache** | MBs | on the same die, hardware-managed | the hardware | read-only `__ldg` loads, L2 promotion hints |
| **HBM / VRAM** | tens of GB | centimetres, off-die | the runtime, via the HAL | `internal/compute/capacity.go` |
| **Host DRAM** | hundreds of GB | across a bus | the runtime, via offload | `internal/compute/dram_ring.go` |
| **NVMe / disk** | TBs | milliseconds away | the operator | the staging ring's sink |

The pattern to notice: **each step down is bigger and slower, by orders of
magnitude.** That is not an accident of engineering; it is the physics of how
close a storage cell is to the transistor that reads it. The whole craft of
inference engineering is arranging for the bytes you need *next* to already be
one step closer than they were.

## How to check a claim on any rung

Every rung cites files as `path:line`. Those paths are real and resolve in this
repository. To check one yourself:

```sh
sed -n '483,500p' internal/model/forward.go   # the Forward entry point
```

Run that from the repository root. If a citation does not resolve, that is a bug
in the page — the pages are verified by a link-and-citation check, not by trust.

## See also

- [The hierarchy as one figure](../../../visuals/86-buffer-hierarchy-ladder.mmd) — the
  six tiers, rendered standalone (`.svg` / `.png` beside it) for slides and handouts.
- [The caching ladder](../caching/README.md) — the same five-rung pattern applied
  to fak's caching stack. This ladder is its sibling: caching is *what* to keep
  close, this ladder is *where* close is.
- [Hardware limits and capacity](../hardware-limits-and-capacity.md) — the
  capacity half of the story, and the "eighth assumption" this ladder's Level 4
  and Level 5 both lean on.
- [Hardware portability](../hardware-portability.md) — the seven shape
  assumptions the compute HAL lifts, which is why the same forward loop can run
  on a CPU, a GPU, or an integrated-memory APU.
- [The memory-bandwidth field guide](../../benchmarks/MEMORY-BANDWIDTH-FIELD-GUIDE.md)
  — the measurement discipline behind Level 4's roofline, including how to tell a
  measured ceiling from a marketing one.
- [What is a CUDA kernel?](../what-is-a-cuda-kernel.md) — the word "kernel" means
  three different things in this repository; read this before Level 3 if that
  trips you.
- [The abstraction map](../the-abstraction-map.md) — where the model sits in the
  full stack, from your terminal to the silicon.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.
- [The learning path](../../../LEARNING-PATH.md) — the prerequisite-ordered
  catalog this ladder plugs into.
