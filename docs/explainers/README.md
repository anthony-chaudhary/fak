---
title: "The fak explainers: where to start, whatever you already know"
description: "The entry point to fak's educational docs: question-driven ladders written for five or six kinds of reader, plus single-page explainers."
slug: explainers-index
keywords:
  - fak explainers
  - where to start learning inference
  - llm inference tutorial
  - gpu backend explained
  - vulkan vs rocm
  - question based learning
  - beginner guide to inference
  - advanced inference internals
date: 2026-11-18
---

# The fak explainers

The fak explainers are the project's educational docs: question-driven ladders, each written five or six times for different readers, plus single-page explainers, indexed by what you already know rather than by topic.

> **What this is.** The entry point to the human educational docs in this repository.
> Everything here teaches the same system — fak, an in-kernel inference and agent
> runtime — but it is written for **many different readers**, and the fastest way to
> get lost is to start on a page written for someone who already knows what you are
> about to learn. So this page is indexed by **what you already know**, not by topic.

> **How to use it.** Find the row below that sounds most like you. Read that one page.
> Every page ends with a **Checkpoint** — a handful of questions with answers. If you
> can answer them, you are done with that page; if you cannot, the page told you what
> you were missing, which is the point.

## Pick your starting point

| If this sounds like you | Start here | Why this one |
|---|---|---|
| "I have never seen a neural network run." | [Inference, Level 1 — what is a token?](inference/level-1-what-is-a-token.md) | Assumes nothing. Builds from "a token is a number" upward, with no math. |
| "I know what a token is. What actually happens to it?" | [Inference, Level 2 — the forward pass](inference/level-2-the-forward-pass.md) | The loop that turns a prompt into the next token, traced through real code. |
| "I understand the forward pass. Where does it *run*?" | [Accelerators, Level 1 — why not just the CPU?](accelerators/level-1-why-not-just-cpu.md) | The CPU-vs-GPU question, then the seam that lets fak swap one for the other. |
| "I know GPUs. What is the difference between Vulkan and ROCm?" | [Accelerators, Level 5 — Vulkan vs ROCm](accelerators/level-5-vulkan-vs-rocm.md) | The head-to-head, with the measured numbers and the honest gaps. |
| "I write kernels, or want to." | [Kernel development, Level 1](kernel-development/level-1-what-is-kernel-development.md) | Six rungs, each with a lab you run and a checkpoint you answer. |
| "I care about memory — where the bytes go." | [Memory hierarchy, Level 1 — what is a buffer?](memory-hierarchy/level-1-what-is-a-buffer.md) | SRAM, HBM/VRAM, host DRAM, and the offload frontier. |
| "I care about number formats and quantization." | [Mixed precision, Level 1 — what is a bit?](mixed-precision/level-1-what-is-a-bit-and-why-fewer.md) | The format zoo, and where measured claims end. |
| "I care about what the chip actually executes." | [Hardware ISA, Level 1 — what is an instruction set?](hardware-isa/level-1-what-is-an-instruction-set.md) | From "what is an instruction" down to PTX, SASS, and tensor cores. |
| "I care about caching and cost." | [Caching, Level 1 — what is caching?](caching/level-1-what-is-caching.md) | Prompt cache, KV cache, and the economics of a hit. |
| "I want to split one model across many workers." | [Inference parallelism, Level 1 — what is inference?](inference-parallelism/level-1-what-is-inference.md) | Pipeline, tensor, and expert parallelism, and the honest frontier. |
| "I just want the vocabulary." | [The fak/DOS glossary](glossary.md) | 23 terms, one line each. Read this first if the other pages' words are unfamiliar. |

## The eight ladders

A **ladder** is one subject written several times, once per reader. Rung 1 assumes
nothing; the top rung assumes everything the rungs below it taught. You are meant to
read **one** rung, not all of them — the rungs are parallel tellings, not a sequence
you must complete.

| Ladder | The subject | Rung 1 asks | Top rung asks | Rungs | Diagrams |
|---|---|---|---|---|---|
| [Inference](inference/README.md) | How a model turns your prompt into the next token | What is a token? | What changes at scale? | 5 | 4 |
| [Accelerators](accelerators/README.md) | How fak runs that forward pass on a GPU | Why not just the CPU? | Vulkan vs ROCm, head to head | 5 | 3 |
| [Memory hierarchy](memory-hierarchy/README.md) | How bytes move while a model runs | What is a buffer? | When the model does not fit | 5 | 0 |
| [Mixed precision](mixed-precision/README.md) | What a model's numbers are stored in | What is a bit? | Where measured claims end | 5 | 0 |
| [Hardware ISA](hardware-isa/README.md) | The instruction sets a machine executes | What is an instruction set? | The limits of ISA knowledge | 5 | 3 |
| [Caching](caching/README.md) | What to keep close, and why | What is a cache? | The caching frontier | 5 | 0 |
| [Inference parallelism](inference-parallelism/README.md) | How to split one model across workers | What is inference? | The parallelism frontier | 5 | 0 |
| [Kernel development](kernel-development/README.md) | How to change fak's kernel | What is kernel development? | What is still open | 6 | 4 |

**A suggested order, if you want one.** The two ladders that carry the main thread are
[Inference](inference/README.md) (what the computation *is*) and
[Accelerators](accelerators/README.md) (where it *runs*). Read Inference first — the
accelerator ladder assumes you know what a forward pass is. The other six ladders are
deep dives you can enter from any rung.

## The single-page explainers

Not every subject needs five tellings. These are one page each, grouped by what they
are about. They assume more than a ladder's rung 1 and less than its top rung.

### Orientation — read these first if the vocabulary is unfamiliar

- [The fak/DOS glossary](glossary.md) — 23 terms in one line each.
- [The abstraction map](the-abstraction-map.md) — from your terminal to the silicon.
- [What fak is not](what-fak-is-not.md) — the honest boundary, stated plainly.
- [One binary is the whole agent-serving surface](one-binary-one-surface.md) — why there is one artifact.
- [Two runtimes, one binary](runtime-vs-client.md) — gateway vs agent runtime vs client.
- [The cross-platform spine](cross-platform-spine.md) — one kernel from IoT to hyperscaler.
- [Why fak, now](why-fak-the-market-moment.md) — the market moved from writing code to checking it.

### Inference, hardware, and the accelerator path

- [What is a CUDA kernel?](what-is-a-cuda-kernel.md) — and why "kernel" means three things here.
- [GPU clock cycles](gpu-cycles.md) — what a kernel costs in the unit the chip counts.
- [Hardware portability via the compute HAL seam](hardware-portability.md) — the seam the accelerator ladder climbs.
- [Hardware capacity limits](hardware-limits-and-capacity.md) — the eighth assumption the HAL does not lift.
- [Multi-GPU tensor parallelism](multi-gpu-tensor-parallelism.md) — when one device is not enough.
- [AWQ 4-bit quantization](awq-quantization.md) — one quantization format, end to end.
- [What tuned SOTA serving optimizations mean in fak benchmarks](sota-optimizations.md) — reading a benchmark honestly.
- [Local-vs-frontier parity on your hardware](local-vs-frontier-parity.md) — what a local model can and cannot match.
- [TensorRT-LLM vs SGLang, and where fak fits](tensorrt-llm-vs-sglang-and-fak.md) — the serving-engine landscape.
- [The vLLM lifecycle cache-loss bridge](vllm-lifecycle-cache-loss-bridge.md) — why a restart costs you the cache.
- [Why shape and dtype parity cannot establish loader correctness](loader-semantic-transform-contracts.md) — the tensor-meaning defect class.

### Context, caching, and cost

- [Context management](context.md) — the overview.
- [Context as a variable](context-as-a-variable.md) — named, lazy, and queryable.
- [You never manage the context window](you-never-manage-the-context-window.md) — the design stance.
- [Context shedding](context-shedding.md) — trimming a long session from the middle out.
- [Context signal-to-noise](context-signal-to-noise.md) — what a cache-hit % cannot tell you.
- [Context control surfaces and the pin-audit gap](context-control-surfaces-and-the-pin-audit-gap.md) — the middle, the pins, and the gap.
- [The context tape](context-tape-visuals.md) — one visual, any trajectory.
- [The O(1) context window](o1-context-window-economics.md) — when sending less beats caching more.
- [Cache reuse: choose the layer and verify the effect](cache.md) — the practical guide.
- [What is fak's managed cache?](what-is-managed-cache.md) — a plain-English guide.
- [Addressable KV cache](addressable-kv-cache.md) — what production actually offers.
- [The addressable KV cache in 5 minutes](addressable-kv-cache-in-5-min.md) — the short version.
- [How the KV cache changes as agentic context grows](kv-cache-agentic-context.md) — the erosion story.
- [Long sessions: shed history, keep the cache hit](long-sessions-keep-the-cache-hit.md) — the tension, resolved.
- [The frozen-trajectory cache cliff](frozen-trajectory-cache-cliff.md) — why a high hit rate can go to 0%.
- [Long-session economics](long-session-economics.md) — why a growing transcript costs more.
- [What a saved token is worth](what-a-saved-token-is-worth.md) — one multiplier, two axes, six names.
- [The compounding benefits of a saved call](compounding-benefits-of-a-saved-call.md) — why one avoided call pays back four times.
- [Less context, less code](less-context-less-code.md) — where fak fits beside Caveman and Ponytail.

### Agents, the kernel, and trust

- [Agent runtime](agent-runtime.md) — category, ownership, interfaces, and proof.
- [The agent virtual filesystem](agent-virtual-filesystem.md) — the filesystem an agent sees.
- [The gateway](gateway.md) — the kernel-adjudicated wire in front of your agent.
- [Engineering is building loops](engineering-is-building-loops.md) — fak is the kernel they run on.
- [The tool call is a syscall](tool-call-is-a-syscall.md) — the analogy, made precise.
- [Tool refusal is feedback, not task completion](tool-feedback-continuation.md) — closing the wire turn.
- [REQUIRE_WITNESS as a durable tool-call continuation](require-witness-tool-call-continuations.md) — the mechanism.
- [The policy runs inside the kernel](policy-in-the-kernel.md) — and what that costs.
- [Why a capability floor beats a classifier for prompt injection](default-deny-vs-classifier.md) — the security argument.
- [Linting agent code at the kernel](code-linting-at-the-kernel.md) — the checks that run in-process.
- [The god-file growth gate](god-file-growth-gate.md) — a ratchet, not a sweep.
- [Shared workspace, positive state, and the negation operator](shared-workspace-and-the-negation-operator.md) — concurrent agents, one tree.
- [The status a peer can trust](status-a-peer-can-trust.md) — verified progress for in-flight agents.
- [Verify, don't trust: what DOS actually checks](verify-dont-trust.md) — the trust substrate.
- [vdso.Revoke is the MPI_Comm_revoke analogue](vdso-revoke-as-comm-revoke.md) — invalidate-on-refutation.
- [Change data capture for agents](change-data-capture-for-agents.md) — fak's change feeds in CDC terms.
- [Structured generation didn't die — it became infrastructure](structured-generation-became-infrastructure.md) — the shift.
- [Ultracode: a concurrent agent fleet as an operational mode](ultracode-multi-agent-dogfood.md) — the multi-agent substrate.
- [Memory engineering](memory-engineering.md) — the discipline.
- [Model routing](routing.md) — choosing a model per request.
- [Data residency and compliance](data-residency-and-compliance.md) — keep inference in-country.
- [The fak fleet benchmark suite](fleet-benchmarks.md) — five demos you can run yourself.

## How to read a page here

Two conventions are worth knowing before you trust anything you read.

**Every claim is pinned to a real file.** When a page says a mutex serializes a
backend, it names the file and line — `internal/compute/vulkan.go:69`. That is not
decoration. It is the difference between a tutorial and a story, and it means you can
always check the page against the code it describes.

**But a citation that resolves is not yet a citation that is true.** The **links**
between pages are machine-checked: `fak doc-freshness --check --corpus` walks every
markdown file in the repository (`internal/docfreshrsi/corpus_audit.go:104`) and
reports `dangling_links`, so a broken relative link fails a gate. The **citations**
are checked by hand, and that check is weaker than it looks — a gate can prove that
`internal/model/forward.go:483` *exists*, but not that line 483 *says what the page
claims it says*. When a page here is wrong, it is almost always that second failure:
the line moved, or the claim drifted off it. So read the line; do not just confirm it
is there.

## See also

- [The learning path](../../LEARNING-PATH.md) — a longer ordered route through the whole repository.
- [The docs index](../index.md) — every document, by task.
- [The glossary](glossary.md) — one-line definitions of the vocabulary.
- [The accelerator ladder](accelerators/README.md) — the ladder this index was written alongside.
- [The inference ladder](inference/README.md) — the main thread, if you want one place to start.
