---
title: "The memory hierarchy — the questions a newcomer actually asks"
description: "The question ledger for the memory-hierarchy axis: the questions a newcomer asks when they first try to work out where a model's numbers actually live, each with a short answer and a pointer to the rung of the memory-hierarchy ladder that answers it in full."
slug: questions-memory-hierarchy
keywords:
  - memory hierarchy questions
  - what is a buffer
  - SRAM vs HBM vs DRAM
  - why is decode bandwidth bound
  - roofline ridge point
  - arithmetic intensity
  - kernel scratchpad
  - host DRAM offload
  - KV cache pressure
  - question loop
  - memory hierarchy ladder
date: 2026-11-19
---

# The memory hierarchy — the questions a newcomer actually asks

> **What this is.** The question ledger for the **memory-hierarchy** axis. Not a
> tutorial — a list of the questions a newcomer actually asks the first time they try
> to work out where a model's numbers *live* while it runs, each with a short answer
> and a pointer to the rung of the
> [memory-hierarchy ladder](../explainers/memory-hierarchy/README.md) that answers it
> in full. The page *is* the questions; the answers are deliberately short, because the
> rung is where the real answer lives.

"The memory hierarchy" here means one specific thing: **the ordered list of places a
number can sit, from the registers inside the arithmetic unit out to the SSD, and the
rule that a compute unit can only multiply numbers that are already in its own
registers.** A newcomer meets that sentence and immediately has questions — some
contrarian ("isn't this just caching, and isn't caching solved?"), some afraid ("what
if my model does not fit on the device at all?"), and some nobody has thought to ask
("why is the kernel's scratchpad sized at compile time?"). This page collects them,
labels them honestly, and routes each one to the rung that closes it.

## How this page works

This page is the durable surface of the **question loop** — a super-loop-family member
whose whole job is to *ask* hard questions and *not* answer them. Two loops, in two
separate context windows, keep it honest:

- **The ASK loop** (role: *questioner*) writes the questions. Each tick it appends a
  handful of `open` rows to [`asked.jsonl`](asked.jsonl) and stops. It never proposes
  fixes, files tickets, or answers itself.
- **The NEXT-STEP loop** (role: *ticketer*) turns an unanswered question into work. Each
  tick it takes **one** `open` question, decides its fate, and — only if the question
  earns it — files **one** ticket.

**Why the two loops are separated.** A questioner that also holds the ticketing pen
slides into solution mode: it starts writing questions it already knows how to close, and
quietly stops asking the uncomfortable ones. The load-bearing rule is that **the
questioner must not be the one who decides the question is answered.** Reading the open
backlog is fine and necessary; *triaging* it is what corrupts the ASK role. A fresh
context per ASK tick keeps the provocations honest. The full schema, the four categories,
and the machine-checked ledger rules live in the [question-loop README](README.md).

The categories below are the ledger's own vocabulary — **CONTRARIAN**, **AFRAID**,
**UNASKED**, and **STEELMAN** — not a new taxonomy invented for this page.

## CONTRARIAN — questions that challenge what the repo claims is correct

### Isn't the memory hierarchy just a cache, and isn't caching a solved problem?

No, and the ladder draws the line explicitly: caching is *what* to keep close, the
hierarchy is *where* close is. A cache is a policy with three parts — a budget, a
victim, and accounting — and the repo's own expert path shipped without any of them,
which is why the file says so outright: *"A memoizer, not a cache: no budget, no
victim, no accounting"* (`internal/model/expert_ring_hal.go:31`). A memoizer grows
monotonically until it converges on the whole model. And the hierarchy is not a bug to
be fixed: Level 1 says the fast tiers are fast *because* they are small, so there is no
configuration where everything is instant. "Caching is solved" is the assumption the
bounded ring exists to refute — and that ring is default-off, R0 of a ladder
(`internal/model/expert_ring_hal.go:56`).

→ full answer: [Level 5 — Host DRAM, offload, and the frontier](../explainers/memory-hierarchy/level-5-host-dram-offload-and-the-frontier.md)

### If decode is bandwidth-bound, why not just quantize harder and be done?

Because quantization moves the *ceiling*, not the wall-clock. It helps twice — fewer
resident bytes (capacity) and fewer fetched bytes (bandwidth) — but the cheaper format
adds its own costs: dequantization arithmetic, scale and metadata bytes, alignment,
occupancy, launch overhead, and a quality floor you are not allowed to cross. Level 4's
worked example is blunt: 8 GB of 2-byte weights on a 1 TB/s device ceilings at 125
tokens/s; the same weights at 4-bit ceiling at 500 tokens/s — a 4× ceiling — but the
4-bit path's compute roof is *lower* because unpacking is arithmetic, so the real gain
lands below 4×. The field guide states it directly: quantization does **not** guarantee
proportional speedup (`docs/benchmarks/MEMORY-BANDWIDTH-FIELD-GUIDE.md:82`). "Just
quantize harder" is a lever, not a finish line.

→ full answer: [Level 4 — HBM, VRAM, and the roofline](../explainers/memory-hierarchy/level-4-hbm-vram-and-the-roofline.md)

### Isn't the roofline just a benchmark chart — why does a teaching doc need it?

It is a decision procedure, not a chart. The roofline bounds attainable rate by the
smaller of two roofs — compute and memory — and the ridge point where they cross tells
you which one binds. The word doing the work is *measured*: the field guide insists you
use measured sustainable bandwidth for the relevant access pattern, not a vendor peak
(`docs/benchmarks/MEMORY-BANDWIDTH-FIELD-GUIDE.md:34`). That is what turns "why is this
slow?" from a guess into a verdict: Level 4 runs the *same* intensity twice — at 0.9
TB/s against a 1 TB/s roof you are bandwidth-bound, at 0.2 TB/s you are not, and the
next thing to check is launch count and unpacking, not the memory system. Without the
roofline you cannot tell those two apart.

→ full answer: [Level 4 — HBM, VRAM, and the roofline](../explainers/memory-hierarchy/level-4-hbm-vram-and-the-roofline.md)

### If the kernel's scratchpad is so important, why is the array size a `#define` and not tuned at runtime?

Because the compiler has to know the footprint *before* the kernel runs — to schedule
blocks and to lay the array out in SRAM — so a runtime-sized static array would make the
footprint unknowable at compile time. The footprint is a hard admission number, not a
hint: the block scheduler places a block only if the SM has room for the whole thing.
And the language does have a runtime path — it is just a different declaration. When the
tile size depends on a runtime shape (sequence length, head dimension, split count) you
declare `extern __shared__ float smem[];` and carve it by hand
(`internal/compute/cuda_v41_shared_attention.cu:44`). The `#define` is not laziness; it
is the number the scheduler reads.

→ full answer: [Level 3 — SRAM and the kernel's own scratch](../explainers/memory-hierarchy/level-3-sram-and-the-kernels-own-scratch.md)

## AFRAID — the elephant in the room

### What if my model does not fit on the device at all — is the whole hierarchy useless to me?

No, and this is the fear the ladder is built to retire. The device never had to hold
everything; it has to hold what is *live* — the weights a token actually touches, the KV
span a turn actually reads, the scratch a kernel actually needs. "The model is bigger
than the card" is a *residency* question, not a capacity wall. The fit check is built to
fail open: a backend that cannot probe its own ceiling returns `FitUnknown` and the
caller **proceeds**, never refuses (`internal/compute/capacity.go:733`). And when a plan
genuinely does not fit, the refusal is a sizing message *before* the allocation that
would otherwise panic — the remedy is to reclass the weights as host-scoped offload and
page them in, not to give up.

→ full answer: [Level 5 — Host DRAM, offload, and the frontier](../explainers/memory-hierarchy/level-5-host-dram-offload-and-the-frontier.md)

### What if the numbers I measure do not match the vendor's spec sheet?

Then you are measuring correctly, and the spec sheet is the wrong denominator. The
roofline's ridge point is compute roof divided by *measured* bandwidth
(`docs/benchmarks/MEMORY-BANDWIDTH-FIELD-GUIDE.md:34`), and the field guide is explicit
that you use measured sustainable bandwidth for the relevant access pattern, not a
vendor peak. A vendor number is a best case under an access pattern you are probably not
running; the gap between it and your measurement is the first thing the discipline
expects you to find. The profiler's whole job is to report *streamed* bytes and a
verdict against measured bandwidth, not the nominal format width
(`docs/benchmarks/MEMORY-BANDWIDTH-FIELD-GUIDE.md:82`). A mismatch is the signal, not
the error.

→ full answer: [Level 4 — HBM, VRAM, and the roofline](../explainers/memory-hierarchy/level-4-hbm-vram-and-the-roofline.md)

### What if I get the buffer sizes wrong — how would I even know?

You would know, because the footprint is a hard admission number rather than a hint: the
block scheduler places a block only if the SM has room for the whole declared footprint,
so an oversized tile simply does not get placed. And you can check the arithmetic by
hand before you ever run it — Level 3's GEMV is 32 × 33 × 4 + 128 × 4 + 4 × 4 = 4752
bytes, read straight off the three declarations
(`internal/compute/routed_expert_gemv.cu:109`). Above the kernel, the capacity model
checks a `MemoryPlan` — weights plus expected KV plus scratch — *before* anything is
allocated (`internal/compute/capacity.go:100`), so a wrong size surfaces as a refusal,
not as silent corruption. The failure mode is a message, not a mystery.

→ full answer: [Level 3 — SRAM and the kernel's own scratch](../explainers/memory-hierarchy/level-3-sram-and-the-kernels-own-scratch.md)

## UNASKED — questions nobody has thought to ask

### Why can a model be split across devices at all — and why is the split bit-exact?

Because the thing that crosses the device boundary is a *buffer*, not a computation. The
hidden state `x` is the only value a band needs from its predecessor, so a model splits
into contiguous layer bands that each run the identical per-layer instruction stream over
the identical hidden state — which is why the repo can assert the split is bit-exact
(`internal/model/partition.go:122`). `Forward` is literally `embedBand(ids)` followed by
`ForwardBand(x, 0, NumLayers, true)`, so the monolithic path is the degenerate one-band
case (`internal/model/forward.go:508`). The split is not free of *rules*: a band may not
begin mid-IndexShare group, and the code fails closed rather than producing garbage. The
buffer is the seam.

→ full answer: [Level 2 — The forward pass as a sequence of buffer moves](../explainers/memory-hierarchy/level-2-the-forward-pass-as-buffer-moves.md)

### Why does the SSD tier need a staging ring instead of just writing?

Because the SSD's scarce resource is not bandwidth, it is *write cycles*. The staging ring
exists to coalesce writes so the drive sees fewer, larger program/erase cycles — its
header names the parent issue outright: SSD endurance and a five-year lifespan budget for
high-volume local caching (`internal/compute/dram_ring.go:10`). The design follows from
that: a 2 GiB default capacity (`:15`) and a 60-second dwell timer inside a 60–120s
cooldown window (`:18`), so dirty pages accumulate in DRAM and flush lazily rather than
each write going straight to flash. The tier below the hierarchy is not just slower — it
is *consumable*, and the ring is what rations it.

→ full answer: [Level 5 — Host DRAM, offload, and the frontier](../explainers/memory-hierarchy/level-5-host-dram-offload-and-the-frontier.md)

### Why does "I could not measure the device" need its own value, distinct from "zero bytes free"?

Because they are different facts and conflating them is a correctness bug. Zero free bytes
is a real, meaningful state — the device is full, and the caller should refuse. "I could
not probe" is an absence of knowledge, and the capacity model gives it a sentinel of its
own: `FreeUnknown int64 = -1` (`internal/compute/capacity.go:45`). The distinction is what
lets the fit check fail open without lying: a backend that cannot measure returns
`FitUnknown` and the caller proceeds, "never `FitTooBig`. So adding the capability never
blocks" (`internal/compute/capacity.go:27`). A single `0` would have made every unprobeable
device look full, and the whole offload path would refuse to start.

→ full answer: [Level 4 — HBM, VRAM, and the roofline](../explainers/memory-hierarchy/level-4-hbm-vram-and-the-roofline.md)

## STEELMAN — the strongest version of the objection

### Steelman: the hierarchy is a hardware accident — a good compiler should hide it entirely, so why should a programmer ever reason about tiers?

The steelman is real: a load is a load, and a language that made you think about which
tier a number sits in would be a bad language. The counter is that the hierarchy is not
the *syntax* of a load — it is the *cost model*, and the cost model is the thing being
optimized. A compiler can hide the latency of a fetch from your source code; it cannot
hide the latency from the clock. Level 1's rule — arithmetic is nearly free, moving
numbers is what you pay for — is a property of physics, not of the language. And the
repo's own seams leak the abstraction on purpose: `DirectSlabConfig.Alignment` is a field
a *caller* must set (`internal/compute/direct_slab.go:20`), because the right alignment is
a property of the device, not of the algorithm. Hiding the hierarchy hides the objective.

→ full answer: [Level 1 — What is a buffer?](../explainers/memory-hierarchy/level-1-what-is-a-buffer.md)

### Steelman: a bigger, faster tier would make this whole ladder obsolete — why not just wait for the next memory generation?

The steelman is that the ladder is a snapshot of today's parts, and next year's parts
dissolve it. The counter is that every tier is bounded by the same two walls — capacity
and bandwidth — and they trade against each other, so a faster tier moves the ridge point
without removing it. The roofline's ridge is compute roof divided by *measured* bandwidth
(`docs/benchmarks/MEMORY-BANDWIDTH-FIELD-GUIDE.md:34`): double the bandwidth and you halve
the ridge point, which moves *more* kernels to the compute-bound side. That changes which
roof binds; it does not delete the concept. And the capacity wall bites first anyway — the
KV cache grows linearly with context, so a bigger tier buys context, not immunity. The
ladder is the shape of the trade, not a list of parts.

→ full answer: [Level 4 — HBM, VRAM, and the roofline](../explainers/memory-hierarchy/level-4-hbm-vram-and-the-roofline.md)

### Steelman: a check that proceeds when it cannot measure is not a check — is the fail-open fit test simply a bug?

The steelman is strong: a capacity check whose answer to "I don't know" is "go ahead" is
not enforcing anything, and a device that is genuinely too small still panics
mid-allocation. The counter is that the alternative is worse in the direction that
matters. A check that *refuses* when it cannot measure turns every unprobeable backend
into a hard failure, so adding a new device makes the system refuse to run on it — the
capability can never be introduced. The repo's contract is explicit and deliberate: a
backend that cannot probe returns `FitUnknown` and the caller proceeds, "never
`FitTooBig`. So adding the capability never blocks" (`internal/compute/capacity.go:27`).
The honest cost is the loud late failure, and the ladder names it as an open question
rather than pretending the trade is free.

→ full answer: [Level 5 — Host DRAM, offload, and the frontier](../explainers/memory-hierarchy/level-5-host-dram-offload-and-the-frontier.md)

## Questions still open

These are questions the repo genuinely cannot answer yet. They are marked **open** in the
ledger, and each names what would close it.

- **Does the demote-not-drop pressure sweep actually preserve recall on a served turn?**
  The post-decode sweep is default-on, but it only fires for a native in-kernel planner
  with a host-L2 byte budget; for every other serve shape (proxy, dual, upstream, CPU-only)
  the provider is nil and the edge is a fail-open no-op. Even where it fires, the ladder
  never shows a served turn whose answer depended on a span that was demoted and later
  re-read. *What would close it:* a served-turn benchmark where a demoted span is re-read
  after demotion and the answer is bit-identical to the same turn with
  `FAK_KV_PRESSURE_RELIEF=off`.
- **How much of the expert ring's residency win survives a real routing trace?** The
  bounded ring is R0 and default-off, and the "activated-expert gap" (top-8 of 256, ~3%) is
  a checkpoint property, not a measured trace. The sizing knob, the warm-start pin-set, the
  async prefetch on the miss path, and the checkpoint tier under a miss are all later rungs.
  *What would close it:* a measured per-token expert-activation trace from a real serving
  run, reporting the ring's hit rate and bytes-per-token against the unbounded memoizer on
  the same trace.
- **What does the fail-open capacity check actually cost?** `FitUnknown` makes a backend
  that cannot probe its own ceiling *proceed* rather than refuse, so a device that is too
  small still panics mid-allocation — and the classed `MemoryPlan` fit check ships only for
  the GGUF serve device-load paths, so other loaders can still reach an allocation with no
  plan at all. The ladder names the fail-open as a deliberate feature but never measures how
  often it fires. *What would close it:* a census of load paths that reach an allocation
  without a `MemoryPlan`, plus a count of the OOM panics a plan would have turned into a
  typed refusal.
- **Is the staging ring's write coalescing enough to keep the SSD inside its endurance
  budget?** The ring coalesces dirty pages and hands them to a storage callback, but it does
  not own the SSD's wear-leveling or the multi-year lifespan accounting the parent issue
  describes; the "100× fewer program/erase cycles" figure is a worked example, not a
  measured trace. *What would close it:* a measured program/erase count from a real
  high-volume local-cache workload over a multi-day run, compared against the drive's rated
  endurance.

## See also

- [The question-loop ledger](README.md) — the schema, the four categories, and the
  machine-checked rules behind this page.
- [The memory-hierarchy ladder](../explainers/memory-hierarchy/README.md) — the five rungs
  every "→ full answer" above points into.
- [The caching ladder](../explainers/caching/README.md) — the same five-rung pattern applied
  to fak's caching stack. Caching is *what* to keep close; this ladder is *where* close is.
- [Hardware limits and capacity](../explainers/hardware-limits-and-capacity.md) — the capacity
  half of the story and the "eighth assumption" Levels 4 and 5 both lean on.
- [The memory-bandwidth field guide](../benchmarks/MEMORY-BANDWIDTH-FIELD-GUIDE.md) — the
  measurement discipline behind Level 4's roofline, including how to tell a measured ceiling
  from a marketing one.

---

*[← back to the question-loop README](README.md) · [the memory-hierarchy ladder](../explainers/memory-hierarchy/README.md)*
