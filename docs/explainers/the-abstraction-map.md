---
title: "The abstraction map — from your terminal to the silicon"
description: "The end-to-end tour of fak's layers for a human operator: your starting points and work items at the top, the two front doors, the kernel, context and memory, serving and the in-kernel model, then the compute HAL and every hardware floor below it — with the one seam you touch at each level and the doc that teaches it."
slug: the-abstraction-map
keywords:
  - abstraction map
  - layers
  - HAL
  - compute backend
  - kernel
  - operator guide
  - hardware abstraction
---

# The abstraction map — from your terminal to the silicon

This page is the floor plan. fak is built like an operating system: a stack of
layers where each layer talks only to the layer directly below it, through one
named seam (a small interface or command). You never need the whole stack in
your head — you need to know **which floor you are standing on** and **the one
seam that floor exposes**. This page names every floor, top to bottom, in plain
words.

The sentence to keep: **every floor trusts the seam below it, never a story
about it** — the same rule that makes the top of the stack honest (a done-claim
is checked against the git diff, not the commit message) makes the bottom of it
correct (a GPU kernel is checked against a bit-exact reference, not eyeballed).

## The whole map in one picture

```
 FLOOR 6 — YOU + YOUR WORK ITEMS
   your terminal · `fak help` (~20 front-door verbs) · the repo · issues ·
   lanes & leases (dos.toml, `dos arbitrate`) · done-claims (`dos verify`)
        |
 FLOOR 5 — THE TWO FRONT DOORS (the session boundary)
   `fak manage` — wraps an agent harness, adjudicates every tool call
   `fak serve` — OpenAI-compatible gateway; repoint one base URL
        |
 FLOOR 4 — THE KERNEL (the DOS)
   tool-call-as-syscall · frozen ABI (internal/abi) · verdict fold
   (Allow/Deny/Defer/Transform/Quarantine/RequireWitness) · write-time
   result gate · hash-chained journal (`fak audit`) · vDSO fast paths
        |
 FLOOR 3 — CONTEXT + MEMORY
   context MMU (internal/ctxmmu) · memory cells (internal/memq) ·
   the addressable KV cache (evict a span, bit-identical cache survives)
        |
 FLOOR 2 — SERVING + THE MODEL
   engines registry: an external engine (vLLM / SGLang / a provider API)
   OR the pure-Go in-kernel model (internal/model + internal/modelengine)
        |
 FLOOR 1 — THE COMPUTE HAL (internal/compute)
   one whole-op interface: MatMul · RMSNorm · RoPE · Attention · Argmax …
   backends register at runtime: cpu-ref (Reference) · cuda · vulkan · metal
        |
 FLOOR 0 — THE MACHINE
   build tags decide what COMPILES IN; a runtime probe (Tier) decides what
   RUNS · per-OS host memory / disk / far-memory · then the silicon itself:
   x86/ARM CPU, NVIDIA (CUDA), AMD (Vulkan), Apple (Metal)
```

## Floor 6 — you, your starting points, and your work items

You start with a terminal, a git repo, and the one binary. `fak help` shows
about twenty front-door verbs grouped by what an operator is doing (guard,
serve, run, ps, audit, model, …); the ~170 dev/repo verbs live one level down
under `fak dev`, so the front door stays readable. The reading front doors are
[`START-HERE.md`](../../START-HERE.md), [`GETTING-STARTED.md`](../../GETTING-STARTED.md),
and the prerequisite-ordered course catalog in [`LEARNING-PATH.md`](../../LEARNING-PATH.md).

Your **work items** live on this floor too: issues are the backlog, a **lane**
(declared in [`dos.toml`](../../dos.toml)) is the slice of the file tree a piece
of work is allowed to touch, and a **lease** (`dos arbitrate`) is the loan of
that lane that stops two workers editing the same files at once. When work is
claimed done, `dos verify` and `dos commit-audit` check the claim against the
git diff — the subject line is forgeable, the diff is not
([verify, don't trust](verify-dont-trust.md)). The ledger of what is shipped
vs simulated vs stub is [`CLAIMS.md`](../../CLAIMS.md).

## Floor 5 — the two front doors

Everything below this floor is invisible until an **agent** (Claude Code,
Codex, any OpenAI-compatible client) tries to *do* something. There are exactly
two ways in:

- **`fak manage -- <agent>`** wraps the agent harness itself, so every tool call
  the agent proposes is adjudicated in-process before it executes.
- **`fak serve`** is the gateway: you repoint one base URL and your existing
  agent's model traffic flows through the kernel unchanged.

Same boundary either way — one wraps the agent, the other fronts the model.

## Floor 4 — the kernel

The keystone idea, taught from scratch in
[the tool call is a syscall](tool-call-is-a-syscall.md): an OS kernel never
takes a user program's word that a write is safe, and fak applies the same
discipline to the LLM. The model **proposes** a tool call; the kernel folds a
chain of adjudicators into one verdict from a closed set — Allow, Deny, Defer,
Transform, Quarantine, RequireWitness — and only an allowed call executes.
Results pass a second, write-time gate on the way back in, so a suspicious
result is quarantined before it can poison the context. Every decision lands in
a hash-chained journal you can verify with `fak audit`.

The kernel's shape is a frozen ABI (`internal/abi`) plus registries: a new
policy rung, fast path, engine, or witness is a new package with one
`Register*()` call — the kernel walks the registries and never imports a
driver. That extension model is the whole of [`ARCHITECTURE.md`](../../ARCHITECTURE.md),
with the on-ramp in [`EXTENDING.md`](../../EXTENDING.md).

## Floor 3 — context and memory

Between the kernel and the model sits everything the agent "remembers": the
context MMU (`internal/ctxmmu`) pages cold spans of the conversation out (and
back in by content-address handle) so long sessions stay affordable; memory
cells (`internal/memq`) persist across sessions; and the **addressable KV
cache** is the property that makes the security story and the reuse story the
same story — evict a poisoned span and the cache is bit-for-bit identical to
one that never saw it ([the 5-minute version](addressable-kv-cache-in-5-min.md)).

## Floor 2 — serving and the model

The kernel does not care what answers the model traffic. An **engine** is
whatever serves tokens, attached through one registry (`RegisterEngine`): an
external engine you already run (vLLM, SGLang, llama.cpp, or a hosted provider
API) — the production path — or the **in-kernel model**, a pure-Go forward
pass (`internal/model`) bound in as an engine by `internal/modelengine`.
`fak model` resolves and caches an `hf://` model for it. Honest fence: fak is
not a serving engine and does not chase vLLM on raw tokens/sec — its field is
governance at the boundary ([what fak is not](what-fak-is-not.md)).

## Floor 1 — the compute HAL

This is the literal hardware-abstraction layer: `internal/compute`. The forward
loop never touches a device; it calls one small interface of **whole
operations** — `MatMul`, `BatchedMatMul`, `RMSNorm`, `RoPE`, `SwiGLU`,
`Attention` (one fused op), `Argmax`, plus `Upload`/`Read`/`NewKV` for
residency. A tensor carries its own `Dtype` (f32 through q4_k quantizations)
and `Layout`, so a new number format or a tiled tensor-core layout is a new
value, not a forked forward pass.

Backends **register at runtime**: `cpu-ref` is the pure-Go, stdlib-only
reference every other target must match; `cuda`, `vulkan`, and `metal` are the
device backends. Each declares a correctness `Class` — **Reference** or
**Approx** — and the harness enforces that an Approx backend is judged against
the reference's exact reduction order, never against "looks right". The full
tour of what this seam neutralizes (seven baked-in hardware assumptions) is
[hardware portability](hardware-portability.md).

## Floor 0 — the machine

Two different questions, answered by two different mechanisms: build tags
decide which backends **compile in**; the registry's `Tier()` probe (CPUID on
x86, a driver query on a GPU) decides which one **runs** on this box. Below
that sit the per-OS floors — host memory (`hostmem_linux/darwin/windows.go`),
disk, and far-memory staging in `internal/compute` — and then the silicon:
x86/ARM CPUs, NVIDIA GPUs via CUDA (witnessed on an RTX 4070 in
[`GPU.md`](../../GPU.md)), AMD GPUs via Vulkan (witnessed on a Radeon RX 7600
in [`VULKAN-AMD-RESULTS.md`](../benchmarks/VULKAN-AMD-RESULTS.md)), and Apple
GPUs via Metal. The reference backend compiles to WebAssembly unchanged — the
portable floor every device degrades to.

## One trip down and back up

You type "fix the failing test" into your agent.

1. **Floor 6** — you already hold a lease on the lane that owns those files.
2. **Floor 5** — the agent proposes `Edit(file)`; the guard intercepts it.
3. **Floor 4** — the verdict fold says Allow (the file is inside your lease's
   tree); the edit executes; the result passes the write-time gate; one journal
   row is appended.
4. **Floor 3** — the turn lands in context; cold spans page out; the cached
   prefix stays byte-identical so the next turn is cheap.
5. **Floor 2** — the next model call goes to whatever engine is attached.
6. **Floors 1–0** — if that engine is the in-kernel model, the forward loop
   issues `MatMul`/`Attention`/`Argmax` against whichever backend the probe
   picked, kernels run on your GPU (or the CPU reference), and one token at a
   time climbs back up the same stack to your screen.
7. **Floor 6 again** — you commit; the claim is checked against the diff.

## The map as a table

| Floor | You touch | The seam | Source of truth |
|---|---|---|---|
| 6 — work | `fak help`, issues, lanes, `dos verify` | lease + diff-witnessed claims | `dos.toml`, [`CLAIMS.md`](../../CLAIMS.md) |
| 5 — doors | `fak manage`, `fak serve` | one wrapped harness / one base URL | [`README.md`](../../README.md) |
| 4 — kernel | policies, `fak audit`, `fak preflight` | syscall + closed verdict set | `internal/abi`, [`ARCHITECTURE.md`](../../ARCHITECTURE.md) |
| 3 — memory | session resume, restore handles | context MMU + memory cells | `internal/ctxmmu`, `internal/memq` |
| 2 — model | `fak model`, engine config | `RegisterEngine` | `internal/model`, `internal/modelengine` |
| 1 — HAL | (nothing — it holds anyway) | `compute.Backend`, whole ops | `internal/compute` |
| 0 — machine | the box you bought | build tags + `Tier()` probe | [`GPU.md`](../../GPU.md), [Vulkan results](../benchmarks/VULKAN-AMD-RESULTS.md) |

An operator lives on floors 6–5 and can stop reading there; nothing below
requires your attention until you want it to. But the discipline is identical
on every floor — a claim is only as good as the seam that checked it — which is
why learning any one floor teaches you the shape of all seven.
