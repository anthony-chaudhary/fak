---
title: "Level 1 — Why not just the CPU?"
description: "A CPU is built to do one hard thing at a time; a language model is millions of identical easy things. Here is the arithmetic, the memory wall, and why fak still keeps a CPU path."
slug: accelerators-level-1
keywords:
  - why use a gpu for llm inference
  - gpu vs cpu inference
  - memory bandwidth bound
  - what is a gpu core
  - fak accelerator ladder
  - gpu inference for beginners
date: 2026-11-18
---

# Level 1 — Why not just the CPU?

*You are on **Level 1 of 5** of the [fak accelerator ladder](README.md).*

> **Audience.** Complete beginners. No GPU background, no systems background, no Go
> background. If you can multiply two numbers and read a table, you have everything this
> page needs.

> **Short answer.** A language model is the same tiny arithmetic — multiply, add — repeated
> hundreds of millions of times per word, with no branching and no surprises. A CPU is built
> to be *fast at one hard thing at a time*; a GPU is built to be *fast at a million easy
> things at once*. Matrix math is the most "million easy things" workload there is, so the
> GPU wins by a wide margin. And the real limit is not arithmetic at all — it is how fast you
> can *feed* the numbers in from memory, which is why "just buy a faster CPU" does not close
> the gap.

## What is a language model actually doing?

Strip away the mystique and a language model does one thing, over and over:

> It multiplies big grids of numbers together, and uses the result to pick the most likely
> next word.

That is the whole machine. There is no lookup table of sentences, no grammar rules, no
reasoning engine. There is a giant pile of numbers called **weights** (also called
*parameters*), and a small pile of numbers describing what you just typed. The model
multiplies them, gets a score for every possible next word, and picks the winner. Then it
appends that word and does it all again.

In math notation, the core operation is `y = x · W`: `x` is a row of numbers (roughly, "what
the model currently knows about the text"), `W` is a big grid of weights, `y` is the result.
That single line, repeated across many layers, is the entire forward pass. Everything else —
attention, normalization, the activation functions — is decoration around this one multiply.

## The arithmetic, done by hand

Let us use a real, small model: **SmolLM2-135M**. The "135M" means it has about **135
million weights**.

**Step 1 — how much work is one word?** For each weight, the model does one multiply and one
add. So generating a *single* token costs roughly:

```
2 × 135,000,000  =  270,000,000  multiply-adds
```

Call it **270 million operations per word**. That is the number to keep in your head.

**Step 2 — how fast is a CPU?** A modern CPU core can do maybe 4 to 16 floating-point
operations per clock cycle, and it runs at roughly 3 to 4 billion cycles per second (GHz). A
laptop has, say, 8 to 16 such cores. Best case, with everything perfect:

```
16 ops/cycle × 4,000,000,000 cycles/s × 16 cores
  = 1,024,000,000,000 ops/s
  ≈ 10^12 ops/s   (a "teraflop", on paper)
```

So in theory a CPU could do 270 million operations in about a quarter of a millisecond.
Sounds fine! Why is your laptop not generating thousands of words a second?

**Step 3 — the catch.** That `10^12` figure is a fantasy for this workload. Real code does
not hit peak ops/cycle, and — more importantly — the work is **memory-bound**: the CPU spends
most of its time *waiting for numbers to arrive from memory*, not doing arithmetic. In
practice a good CPU path on this model lands in the tens of tokens per second, not thousands.
The paper number and the real number differ by more than an order of magnitude, and the reason
is memory, not math.

**Step 4 — what a GPU does differently.** A GPU does not have 16 big, clever cores. It has
**thousands of small, simple cores**. Each one is slower and dumber than a CPU core — but
there are thousands of them, and they all do the *same* operation at the *same* time. The 270
million multiply-adds spread across thousands of lanes instead of 16, so the same work
finishes in a small fraction of the time. It is not that a GPU core is faster; it is that
there are so many more of them, and this workload lets you use all of them at once.

## Why is a GPU faster, really? It is the *shape* of the work

This is the spine of the page. If you remember one thing, remember this.

A CPU and a GPU are not "the same thing, one bigger." They are optimized for **opposite
shapes of work**.

- A **CPU** is built for **latency**: finish *one* complicated, unpredictable task as fast
  as possible. It has branch predictors, out-of-order execution, deep caches — machinery to
  guess what comes next and race ahead on a single thread. It is a sports car: fast, but it
  carries one driver.
- A **GPU** is built for **throughput**: finish a *huge number* of identical, simple tasks
  per second. It has almost no branch prediction, because there are no branches to predict.
  It is a fleet of buses: each one is slow, but together they move a city.

Now look at matrix multiplication. Every output number is computed the same way: multiply a
row by a column, add up the products. No `if`. No "it depends." The same tiny operation,
repeated millions of times, in perfect lockstep. That is the most throughput-shaped workload
there is.

```
   CPU: a few big, clever cores        GPU: thousands of small, simple cores
   (optimized for LATENCY)             (optimized for THROUGHPUT)

   +--------+  +--------+              +--+--+--+--+--+--+--+--+--+--+--+--+
   | core 0 |  | core 1 |              |  |  |  |  |  |  |  |  |  |  |  |  |
   | branch |  | branch |              +--+--+--+--+--+--+--+--+--+--+--+--+
   | predict|  | predict|              |  |  |  |  |  |  |  |  |  |  |  |  |
   |  OoO   |  |  OoO   |              +--+--+--+--+--+--+--+--+--+--+--+--+
   +--------+  +--------+              |  |  |  |  |  |  |  |  |  |  |  |  |
   ~8-16 cores                         +--+--+--+--+--+--+--+--+--+--+--+--+
   each fast, each different work      ~2,000-16,000 lanes, ALL the SAME work
```

The CPU's cleverness is *wasted* on matrix math, because there is nothing to predict and
nothing to reorder — the work is already perfectly regular. The GPU's simplicity is *exactly
right*, because regularity is the one thing it is built for.

## But my CPU has 16 cores — isn't that enough?

Sixteen is not thousands. Make it concrete: imagine adding up a million numbers. You can hire
16 brilliant mathematicians, or 4,000 ordinary ones. For "prove a hard theorem," you want the
16 — the ordinary ones cannot even start. But for "add up a million numbers," you split the
pile into 4,000 stacks and everyone adds their own. The 4,000 ordinary workers finish first,
easily, because the job was never hard — it was just *big*. Language-model math is "add up a
million numbers." It is big, not hard. That is why the crowd wins.

## The memory wall: the real reason, and the one people miss

Here is the part that surprises almost everyone. The GPU's advantage is **not** mainly about
arithmetic speed. It is about **memory bandwidth** — how fast you can move numbers from memory
into the arithmetic units. Look again at the cost of one token:

```
Work done:     270,000,000 multiply-adds
Data read:     ~135,000,000 weights  (every weight, once)
```

So for every weight the model reads, it does only about **2 operations**. Two. That is a
terrible ratio. The arithmetic units finish their two operations almost instantly and then
sit there waiting for the *next* weight to arrive from memory.

```
   The memory wall, drawn out:

   memory (the 135M weights)          arithmetic units
   +--------------------------+       +----------------+
   |  w w w w w w w w w w ... |  ---> |  [ ]  [ ]  [ ] |  mostly IDLE,
   |  w w w w w w w w w w ... |  slow |  [ ]  [ ]  [ ] |  waiting for
   |  w w w w w w w w w w ... |  pipe |  [ ]  [ ]  [ ] |  the next number
   +--------------------------+       +----------------+
        ^                                   ^
   the bottleneck lives HERE     these are fast but STARVED
```

The pipe from memory is narrow; the arithmetic units are fast. So the *pipe* sets the speed,
not the arithmetic. This is called being **memory-bandwidth-bound**, and it is the single
most important fact about running a language model.

| Where the weights live | Rough memory bandwidth |
|---|---|
| CPU + system RAM | ~50–100 GB/s |
| Discrete GPU + its own VRAM | ~300–1,000+ GB/s |

A discrete GPU pulls numbers out of its memory several times faster than a CPU pulls them out
of system RAM. Since the workload is *waiting on memory*, that is where the speedup comes
from — and it is why "just use a faster CPU" does not close the gap. A faster CPU makes the
arithmetic units faster, but they were never the bottleneck. It is also why a GPU with its
own dedicated memory (a "discrete" GPU) beats one that shares system memory with the CPU (an
"integrated" one): same arithmetic, wider pipe.

## So why does fak even have a CPU path?

If GPUs are so much better, why does fak bother with a CPU backend at all? Because the CPU
path is the **reference** — the definition of *correct*. Think of it as the answer key: every
GPU backend is graded against the CPU result, and if the GPU produces a different word than
the CPU, the GPU is wrong, full stop. The CPU path is right by construction, and everything
else has to agree with it.

fak makes this distinction explicit in code. There are exactly two correctness classes:

- **`Reference`** — bit-exact. The CPU path. Same input, same output, every single time, to
  the last decimal place. This is the answer key.
- **`Approx`** — allowed to differ in the last decimal places, but must produce the same
  *answer* (the same token). Every GPU backend is `Approx`.

Why is a GPU allowed to differ at all? Because floating-point addition is **not associative**:
`(a + b) + c` is not always exactly equal to `a + (b + c)` when the numbers are stored with
limited precision. A GPU sums the same numbers in a *different order* than the CPU does, so
the last few digits can wobble. That is fine — the wobble is far too small to change which
word wins. The rule is: **same answer, not same bits.**

You can see both halves of this contract in the code:

- `internal/compute/compute.go:279` defines `CorrectnessClass`, the type that makes the
  `Reference` vs `Approx` split impossible to forget.
- `internal/compute/compute.go:450` defines the `Backend` interface — the shape every
  backend, CPU or GPU, must present. Its `Class()` method declares which correctness class a
  backend belongs to.

The CPU backend registers itself as `cpu-ref` (`internal/compute/cpuref.go:34`), and the
harness can mechanically refuse to hold a device to bit-identity
(`internal/compute/compute.go:700`). The reference is not a convention people remember to
follow — it is enforced.

## The two AMD paths, previewed

If you have an AMD GPU, there are two genuinely different ways to reach the hardware, and
they are **not two names for one thing**:

- **Vulkan** — a cross-vendor graphics-and-compute API. AMD, NVIDIA, and Intel all implement
  it: the "speak one language to every GPU" route. It works everywhere, but carries overhead
  because it was designed for graphics first.
- **ROCm** — AMD's own native compute stack, AMD's answer to NVIDIA's CUDA. It talks to AMD
  hardware directly, with no cross-vendor translation layer, and in principle much less
  overhead.

Levels 3, 4, and 5 are about these two: how Vulkan works, how ROCm works, and how they
compare head to head. For now, hold the shape of it: same card, two doors, and the doors
behave differently.

## What is proven (and what is not)

Honesty matters more than a good story, so here is exactly where things stand.

**Vulkan is shipped and witnessed on real AMD silicon.** fak's Vulkan backend runs a full
SmolLM2-135M forward pass on an **AMD Radeon RX 7600**, and it is numerically correct:
argmax-exact greedy decode, prefill-logit cosine of 1.0. It is also still roughly **60×
slower than llama.cpp**, because of per-dispatch overhead — the cost of handing each small
operation to the GPU one at a time. Correct, but not yet fast. Full measurements:
`docs/benchmarks/VULKAN-AMD-RESULTS.md`.

**ROCm is written but never run.** The architecture taxonomy — the part that classifies AMD
GPU families and picks the right compiler target — is shipped and unit-tested on any host.
The rest of the backend, including the HIP kernels, is also written and sitting in the
repository, but it has **never been compiled**, because compiling it needs a toolchain this
machine does not have. And every acceptance bullet that requires running on silicon is
**deferred to an AMD-GPU-on-Linux node**. **No ROCm performance number exists** — not a slow
one, not an estimate, not a placeholder. Details: `internal/compute/ROCM-C002-NOTES.md`. If
you ever see a ROCm throughput number on this page or any other, treat it as a bug.

## Checkpoint: can you answer these?

Try each one before you read the answer.

1. **Roughly how many multiply-adds does it take to generate one token from a 135M-parameter
   model?** — About 270 million: two operations (one multiply, one add) per parameter, times
   135 million parameters.
2. **A CPU can do about 10^12 operations per second on paper. Why doesn't it generate
   thousands of tokens per second?** — Because the workload is memory-bound, not
   compute-bound. The model reads all 135M weights per token but does only ~2 operations per
   weight, so the arithmetic units mostly wait for numbers to arrive from memory.
3. **What is the real reason a GPU is faster here — raw clock speed, or something else?** —
   Shape, not raw speed. A GPU has thousands of small cores instead of a dozen big ones, and
   matrix math is thousands of identical, branch-free operations — exactly the shape a GPU is
   built for. Its memory bandwidth is also several times higher.
4. **Why does fak keep a CPU backend if GPUs are faster?** — Because the CPU path is the
   `Reference`: the bit-exact answer key every GPU backend is graded against. A GPU backend
   is `Approx` — it may differ in the last decimal places (floating-point addition is not
   associative, and GPUs sum in a different order) but must produce the same token.
5. **Name the two ways to reach an AMD GPU, and say whether they are the same thing.** —
   Vulkan (cross-vendor, works on AMD/NVIDIA/Intel) and ROCm (AMD's own native compute stack,
   the CUDA analogue). Genuinely different machines, not two names for one thing.

## See also

- [The accelerator ladder (README)](README.md) — the index for all five levels.
- [Level 2 — The HAL seam](level-2-the-hal-seam.md) — how a backend plugs into fak at all.
- [Hardware portability](../hardware-portability.md) — running the same model on different
  hardware.
- [Glossary](../glossary.md) — one-line definitions of the vocabulary used here.
- [Vulkan on AMD: measured results](../../benchmarks/VULKAN-AMD-RESULTS.md) — the real
  numbers behind the "shipped and witnessed" claim above.

*[Full ladder (README)](README.md) · next → [Level 2 — The HAL seam](level-2-the-hal-seam.md)*
