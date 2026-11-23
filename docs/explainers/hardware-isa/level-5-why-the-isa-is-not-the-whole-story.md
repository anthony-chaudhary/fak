---
title: "Why the instruction set is not the whole story: the limits of ISA"
description: "Knowing the instruction set does not predict speed: the memory wall, the roofline, and five other things that decide how fast a kernel runs."
slug: level-5-why-the-isa-is-not-the-whole-story
keywords:
  - instruction set limits
  - roofline model explained
  - memory wall
  - arithmetic intensity
  - why is my kernel slow
  - occupancy and latency hiding
  - bit-identical kernels
  - benchmark methodology
date: 2026-11-18
---

# Why the instruction set is not the whole story

> **In one breath:** Knowing the instruction set does not tell you how fast anything runs.
> It tells you what the chip can do. Whether that helps depends on where the time goes.
> Most of the time it goes to moving bytes.

**One line:** the instruction set is a *capability* contract, not a *performance* contract —
it tells you which operations a chip can perform and says almost nothing about which of them
will be the bottleneck, because the bottleneck is set by the ratio of arithmetic to bytes
moved, by data layout, by occupancy, by a driver you did not write, and by whether your
measurement was controlled at all.

*You are on **Level 5 of 5** of the [fak hardware instruction-set ladder](README.md).*

> **Audience.** A researcher or a `fable`-tier reader. You already know what an instruction
> set is, what SIMD is, and what a tensor core does. This rung does not re-teach rungs 1
> through 4; it assumes them. What it offers instead is the honest accounting: the limits of
> what ISA knowledge can tell you, and the edges that are genuinely still open.
>
> **Short answer.** The instruction set describes the arithmetic. It does not describe the
> bottleneck. An instruction that multiplies two numbers takes a few cycles; fetching those
> two numbers from main memory can take hundreds. So the same instruction set can produce a
> kernel that is compute-bound on one machine and memory-bound on another, and the mnemonics
> will not tell you which. This rung is about the gap between "the chip can do this" and
> "doing this is fast."

## Can I predict the speed now that I know the instruction set?

No. And the reason is worth stating precisely, because it is the whole rung in one idea.

An instruction that multiplies two numbers takes a handful of cycles. Fetching those two
numbers from main memory takes hundreds of cycles. The arithmetic is not the slow part. The
*movement* is. So a chip can have a beautiful, wide, modern instruction set and still spend
almost all of its time waiting for bytes to arrive. This is the **memory wall**, and it is why
the instruction set is a poor predictor of speed: it tells you the vocabulary of the
arithmetic and says almost nothing about the traffic.

The tool that makes this concrete is the **roofline**. In plain language: every kernel is
limited either by how fast it can compute or by how fast it can move bytes, and which one it
is depends on the ratio of arithmetic to bytes — not on the mnemonics.

```text
arithmetic intensity = operations performed / bytes moved

  high ratio  ->  the arithmetic units are the bottleneck   (compute-bound)
  low ratio   ->  the memory system is the bottleneck       (memory-bound)

The instruction set describes the numerator's vocabulary.
It says almost nothing about the denominator.
```

A kernel that does one multiply per byte moved is memory-bound on almost any machine. A
kernel that does a thousand multiplies per byte moved is compute-bound on almost any machine.
The instruction set is identical in both cases. The bottleneck is not.

fak's own research map states the honest version of this: "Bandwidth-bound" describes one
measured envelope, not an immutable verdict
([`docs/benchmarks/MEMORY-WALL-ELIMINATION-MAP.md:8`](../../benchmarks/MEMORY-WALL-ELIMINATION-MAP.md)).
That is the right posture. The roofline is a *bound*, not a prediction, and the bound moves
when you change the kernel.

**Checkpoint:** before you read on, answer this in one sentence. If two chips have the same
instruction set, can one be memory-bound on a kernel where the other is compute-bound? Yes —
and the reason is that the instruction set does not carry the memory system's bandwidth or
latency in it.

## Then what does the instruction set actually buy me?

It buys you **capability**, not speed. That distinction is the load-bearing one for this
whole rung.

A chip either has a tensor-core instruction or it does not. A chip either has `dp4a` or it
does not. That is a binary fact you can read straight off the instruction set. Whether *using*
it helps is a separate, measurable question — and the answer is sometimes no.

Here is the concrete example from this repo, and it is the whole lesson in one measured
number. The header comment of the arm64 quantized dot kernel records what happened when the
team tried to win by shrinking the data:

```text
// internal/model/quant_arm64.s:9-11
// The scalar int8 dot (qdot8scalar) is compute-bound — Go emits a per-byte
// sign-extend + scalar imul, so streaming 3.5× fewer weight bytes than f32 bought nothing
// on this arch (int8 measured 0.95× f32, i.e. no win) until the dot itself goes SIMD.
```

Read that carefully, because it is easy to skim past. The team streamed **3.5× fewer weight
bytes** than f32 — a real, large reduction in memory traffic — and measured **0.95×**, which
is to say *no win at all*. The byte reduction was real. It bought nothing. Why? Because the
kernel was not memory-bound. It was **compute-bound**: Go emitted a per-byte sign-extend plus
a scalar multiply, and that arithmetic was the bottleneck. Shrinking the bytes did not touch
the bottleneck.

The fix was not to move fewer bytes. The fix was to make the *dot itself* go SIMD — to change
the arithmetic so the compute side stopped being the wall. That is the lesson: **the ISA
change was necessary, and the byte reduction alone was not sufficient.** Capability told them
what was possible. Measurement told them what mattered.

**Worked example:** the 0.95× is the single measured number this rung will quote, and it is
quoted exactly as the file states it. It is not a speedup. It is a *null result* that taught
the team where the bottleneck actually was. A rung about not over-claiming should lead with a
number that says "this did not work."

## What else is in the way?

The memory wall is the biggest obstacle, but it is not the only one. Here are five more, each
with a real example from this repo where one exists.

### Layout: the hardware wants a shape you do not have

The hardware wants data in a specific arrangement. Your data is almost never in that
arrangement. The gap between the two is a cost, and it can exceed the cost of the arithmetic
you were trying to accelerate.

The clearest example is `ldmatrix` on the GPU. The tensor core's `mma` instruction expects its
operands in a particular per-thread register layout. That layout is not the layout you
naturally have in memory. So the kernel issues a dedicated instruction whose only job is to
load a matrix *and* transpose it into the shape the tensor core wants
([`internal/compute/cuda/fattn_mma_hd256.cuh:139`](../../../internal/compute/cuda/fattn_mma_hd256.cuh)):

```cuda
asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];" ...);
// and a second form, ldmatrix.trans, that does the transpose in the same instruction
// (internal/compute/cuda/fattn_mma_hd256.cuh:154):
asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];" ...);
```

The existence of `ldmatrix` is itself the lesson. If layout did not matter, the instruction
would not need to exist. A transpose can cost more than the multiply it feeds. The instruction
set told you the multiply was available; it did not tell you that getting the operands into
the right shape would be the expensive part.

### Occupancy: the latency you cannot see

A GPU hides memory latency by having many warps resident at once. While one warp waits for
memory, another runs. That is the whole trick. But it only works if enough warps fit.

If your kernel uses too many registers or too much shared memory, fewer warps fit, and the
latency is no longer hidden. The instruction set does not tell you the register count. It does
not tell you the occupancy. Those are properties of the *compiled kernel*, not of the
instruction set.

fak's flash-attention kernel makes this explicit. It declares its occupancy as a constant
([`internal/compute/cuda/fattn_mma_hd256.cuh:39`](../../../internal/compute/cuda/fattn_mma_hd256.cuh)),
and its own comment names the tension directly
([`internal/compute/cuda/fattn_mma_hd256.cuh:536`](../../../internal/compute/cuda/fattn_mma_hd256.cuh)):

```cuda
constexpr int occupancy = 2;
// Q in registers is faster, but register pressure is the biggest bottleneck.
```

That single line is the whole point. Keeping Q in registers is *faster per access* — and it
costs registers, which costs occupancy, which costs latency hiding. The two effects pull in
opposite directions, and which one wins is an empirical question on each chip. The instruction
set is silent on all of it.

### The driver and the runtime: someone else's compiler

The same source can compile to very different machine code on two vendors' drivers. Your
measured speed is partly a property of a driver you did not write.

fak's Vulkan backend is the example. It ships SPIR-V compute shaders, and the driver lowers
them to the device's actual machine code at runtime
([`docs/explainers/hardware-portability.md:215`](../hardware-portability.md)). The measured
throughput of that backend is, in the repo's own words, "bounded by per-dispatch CPU/driver
overhead, not numerics"
([`docs/explainers/hardware-portability.md:224`](../hardware-portability.md)). The arithmetic
is correct; the *speed* is a property of the driver path.

fak's autotune machinery treats the driver as part of the identity of a measurement, not as an
implementation detail. Its cache key includes the driver version and the SPIR-V digest
([`internal/compute/vulkan_autotune.go:11-18`](../../../internal/compute/vulkan_autotune.go)):

```go
// VulkanAutotuneKey is the exact evidence/cache key. No architecture aliases,
// rounded dimensions, driver normalization, or shader substitutions are allowed.
type VulkanAutotuneKey struct {
	DeviceUUID, DriverVersion, QuantFormat string
	M, N, K                                int
	SPIRVDigest                            string
}
```


The comment says it plainly: no driver normalization is allowed. A measurement taken on one
driver is not evidence about another. The instruction set was the same. The speed was not.

### Quantization error: a fast wrong answer is not fast

A faster instruction that computes the wrong thing is not faster. It is broken.

fak's kernels are pinned to be **bit-identical** to a scalar reference. The amd64 kernel's
header explains exactly why that is achievable and why it is worth the constraint
([`internal/model/quant_amd64.s:9-13`](../../../internal/model/quant_amd64.s)):

```text
// Integer addition is associative with no overflow (a block sum
// is bounded by 32*127*127 ≈ 5.2e5), so the lane-reduced int32 equals the scalar kernel's
// isum exactly; the per-block float combine is done in the SAME order as qdot8scalar
// (((float(isum)*dw[b])*dx[b]), accumulated block 0,1,2,…), so this is BIT-IDENTICAL to
// the scalar reference and TestQdot8AsmMatchesScalar pins it.
```

Two facts make this work. First, integer addition is associative and the block sum cannot
overflow int32, so the SIMD lane order does not change the integer result. Second, the
per-block *float* combine is done in the same order as the scalar reference, so the rounding
is identical too. The scalar reference itself is defined at
[`internal/model/quant.go:203`](../../../internal/model/quant.go), and the pin is a real test:
`TestQdot8AsmMatchesScalar` at
[`internal/model/quant_test.go:229`](../../../internal/model/quant_test.go), with the arm64
twin `TestQdot8NEONMatchesScalar` at
[`internal/model/quant_arm64_test.go:16`](../../../internal/model/quant_arm64_test.go).

So: **why is bit-identity worth giving up speed for?** Because the alternative is a kernel
that is fast and wrong, and a fast wrong answer is not a faster answer — it is a different,
incorrect answer. The bit-identity constraint is what lets fak claim its accelerated path
produces *the same tokens* as the reference, not merely similar ones. That is a correctness
claim, and correctness claims are not negotiable against speed.

### The measurement itself: a benchmark measures the machine's mood

A benchmark that is not controlled measures the machine's mood. This is the least glamorous
obstacle and the one most likely to fool you.

fak's benchmark discipline is written down, and it is worth reading before you quote any
number. The authority document states the rule bluntly: "Any number claimed elsewhere must
trace back to an entry here"
([`BENCHMARK-AUTHORITY.md:42`](../../../BENCHMARK-AUTHORITY.md)). It also fences its own
results: they are "scoped results, not a universal speedup claim"
([`BENCHMARK-AUTHORITY.md:21`](../../../BENCHMARK-AUTHORITY.md)).

The methodology page adds two rules that matter for this rung. A tuned alternative is
mandatory — "Quote the tuned baseline as the headline"
([`docs/benchmark-methodology.md:35`](../../benchmark-methodology.md)) — and reproduction gates
promotion: a result without a reproducible witness is `not yet`, not a projected gain. And the
performance page separates mechanism from outcome: "Cache reuse, context shedding, and low
kernel overhead are mechanisms or component measurements"
([`docs/performance.md:59`](../../performance.md)), while end-to-end latency and throughput are
outcomes with different witnesses.

**Expected output:** if you take one habit from this rung, take this one. Before you believe a
speed number, ask three questions. What was the baseline? What was the hardware? Can I
reproduce it? A number that cannot answer all three is not evidence.

## What is genuinely open?

This rung is the honesty rung, so it must be honest about its own edges. Here are three
genuinely unresolved questions, stated as unresolved.

**Does a hand-written assembly kernel beat the compiler's output?** This is an empirical
question that changes per generation. On one chip, hand-tuned assembly wins. On the next, the
compiler's autovectorizer catches up and the hand-written version is a maintenance liability
that no longer pays. There is no general answer. The only way to know is to measure on the
specific chip, and the answer expires when the chip does.

**Can a portable intermediate like SPIR-V ever match a hand-tuned kernel?** Unresolved. A
portable intermediate gives up the ability to exploit one vendor's specific instruction. fak's
Vulkan path is correct and witnessed, but its throughput is bounded by per-dispatch driver
overhead ([`docs/explainers/hardware-portability.md:224`](../hardware-portability.md)), and
the repo does not claim it matches a hand-tuned CUDA kernel. Whether a portable path can close
that gap, or whether the gap is structural, is an open question.

**Is the roofline a prediction?** No. It is a bound. It tells you the best you could possibly
do given the arithmetic and the traffic. It does not tell you what you will do. A kernel can
sit far below its roofline for reasons the roofline does not model — occupancy, layout,
driver overhead, launch cost. The roofline tells you where the ceiling is, not where you are.

And one more, specific to this repo: fak's own hardware matrix lists platforms that are
**targets, not results**. The sm_120 consumer Blackwell row is marked "Dev lane, nothing
measured yet"
([`docs/HARDWARE-MATRIX.md:36`](../../HARDWARE-MATRIX.md)). The honest reading is that the
instruction set is known and the speed is not. That is exactly the gap this rung is about,
sitting in the repo's own table.

## How would I check any of this myself?

You do not need a GPU to learn the loop. You need the loop. It is the same five steps the
ladder README teaches, applied to performance:

1. **Predict** one observable result in plain language. "I think this kernel is memory-bound."
2. **Measure** the thing you predicted, on a controlled machine.
3. **Change exactly one thing.** Not two. One.
4. **Measure again.**
5. **Explain the difference** — or admit you cannot, and say so.

The discipline is in step 3. If you change the kernel *and* the data layout *and* the thread
count, you have learned nothing about which one mattered. The 0.95× result in
[`internal/model/quant_arm64.s:11`](../../../internal/model/quant_arm64.s) is a good result
precisely because it isolated one variable: the byte count changed, the arithmetic did not,
and the measurement said the arithmetic was the wall.

fak's own perf discipline is written down and worth reading before you run anything:
[`BENCHMARK-AUTHORITY.md`](../../../BENCHMARK-AUTHORITY.md) for what a number must trace to,
[`docs/benchmark-methodology.md`](../../benchmark-methodology.md) for the classification rules,
and [`docs/performance.md`](../../performance.md) for the mechanism-versus-outcome split. The
honesty ledger at [`CLAIMS.md:1`](../../../CLAIMS.md) applies the same discipline to capability
claims: every claim carries a tag saying whether it is shipped, simulated, or a stub.

## Try it

```bash
# 1. Predict, in plain language, which side of the roofline the quantized dot is on.
#    Write it down before you run anything.

# 2. Measure the SIMD kernel against its scalar reference on this machine.
#    (The amd64 test runs on x86; the NEON test runs on arm64. One will skip.)
go test ./internal/model -run 'TestQdot8AsmMatchesScalar|TestQdot8NEONMatchesScalar' -v

# 3. Read the header comment that records the 0.95x null result.
#    internal/model/quant_arm64.s:9-11

# 4. Change exactly one thing. Re-measure. Explain the difference, or say you cannot.
```

**Try it:** the point of this exercise is not the pass/fail. The tests will pass; they are
pins, and they are supposed to. The point is step 1. Write your prediction down *before* you
measure, then compare. A prediction you did not record is a prediction you cannot be wrong
about, and a prediction you cannot be wrong about teaches you nothing.

## See also

- [Rung 4 — How the ISA reaches silicon](level-4-how-the-isa-reaches-silicon.md) — the full
  driver path this rung assumes, from Go to cgo to a vendor runtime to the chip.
- [Rung 2 — One instruction, many numbers](level-2-simd-one-instruction-many-numbers.md) —
  where the 0.95× measurement is first introduced, and what SIMD actually buys.
- [Hardware portability](../hardware-portability.md) — the HAL seam, and the per-dispatch
  driver overhead that bounds the Vulkan path.
- [Hardware limits and capacity](../hardware-limits-and-capacity.md) — the eighth hardware
  assumption: finite, exhaustible memory, which is a capacity fact and not a shape fact
  ([`docs/explainers/hardware-limits-and-capacity.md:71`](../hardware-limits-and-capacity.md)).
- [What fak is not](../what-fak-is-not.md) — the same honesty posture applied to the product:
  it is not a serving engine, and it says so first
  ([`docs/explainers/what-fak-is-not.md:21`](../what-fak-is-not.md)).
- [SOTA serving optimizations](../sota-optimizations.md) — where SIMD and fused kernels sit in
  the optimization stack, and why fak marks that row "partial"
  ([`docs/explainers/sota-optimizations.md:89-91`](../sota-optimizations.md)).
- [The hardware matrix](../../HARDWARE-MATRIX.md) — every platform fak has measured, and the
  ones it has not.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · prev → [Rung 4 — How the ISA reaches silicon](level-4-how-the-isa-reaches-silicon.md)*
