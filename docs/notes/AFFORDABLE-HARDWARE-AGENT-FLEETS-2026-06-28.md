---
title: "Affordable-hardware agent fleets: where fak's optimizations make few/old/CPU boxes usable for overnight multi-agent"
description: "A framing + showable-demo map for running multiple coding agents safely on affordable hardware (one big-RAM CPU box, one or a few older datacenter GPUs, a gaming GPU, or a Mac), including overnight/offline. Separates the witnessed-today tier from the honest not-yet, and ties each tier to the specific fak lever that makes it usable for a fleet."
---

# Affordable-hardware agent fleets — the doable tier, and what shows it

Date: 2026-06-28

Scope: a strategy/framing note, not a new benchmark. It asks one question —
*on hardware a small team can actually afford (one CPU box, one or a few older
A100/Ampere GPUs, a gaming GPU, a Mac), can you run **multiple** coding agents
**safely** and at **usable** speed, including overnight/offline?* — and answers
it from numbers that already ship. Every measured claim is cited to its
authority row; every projection is labeled. Measured ≠ projected is the whole
point.

Read with:

- [`docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md`](SCALING-LAWS-OF-AGENTS-2026-06-19.md) — the theory this note operationalizes.
- [`docs/HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md) — the four profiled platforms.
- [`docs/explainers/ultracode-multi-agent-dogfood.md`](../explainers/ultracode-multi-agent-dogfood.md) — the multi-agent *orchestration* mode and its metric.
- [`examples/fleet-reuse-demo/`](../../examples/fleet-reuse-demo/) — the runnable, GPU-free reuse curve.
- [`docs/nightrun/GPU-SERVER-OVERNIGHT-PLAN-2026-06-28.md`](../nightrun/GPU-SERVER-OVERNIGHT-PLAN-2026-06-28.md) — the honest frontier-MoE wall.

---

## 0. Thesis — the agent fleet is the workload where affordable hardware wins

The instinct is to grade a cheap box by its single-stream decode rate: an old
A100 or a CPU host runs a 7B at "only" a handful of tokens per second, "too slow
to chat with," so it looks useless. **That is the wrong meter for an agent
fleet.** A coding-agent fleet has three properties that flip the economics:

1. **It is latency-tolerant.** Overnight, offline, or just "go work the backlog,"
   nobody is watching the cursor. The meter that matters is **aggregate
   throughput under batching**, not the latency of one stream. A box that decodes
   one stream at 18 tok/s can keep dozens of agent streams saturated at once.
2. **It shares an enormous prefix.** Every agent in the fleet carries the same
   system prompt, the same tool schemas, the same house rules, often the same
   repo snapshot. The naive stack re-ingests that prefix once *per agent per
   turn*; a prefix-reuse stack pays it **once**. The work is `agents × turns ×
   working-set × reread-rate` — and reread-rate is the only term you can delete
   safely ([SCALING-LAWS §3](SCALING-LAWS-OF-AGENTS-2026-06-19.md)).
3. **It must run unattended *and* safely.** The moment no human is in the loop,
   the binding constraint stops being tok/s and becomes *trust*: did an agent
   fire a destructive call while you slept? did two agents clobber the same file?
   can you believe the diff you wake up to? That trust/coherence layer is fak's
   actual product, not a side feature.

So the honest pitch is **not** "fak makes a cheap GPU serve tokens as fast as an
H100" — it does not, and [`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
never claims it. The pitch is: **for the fleet workload, on affordable hardware,
fak deletes the duplicated work and supplies the safety floor that make the box
usable — and those wins are measured, deterministic, and reproduce on any box.**

> Use vLLM/SGLang for raw token throughput. The affordable-hardware *fleet* win
> is work-elimination + safe-unattended, not a faster token engine.

---

## 1. The honest spectrum — witnessed-today vs not-yet

| Tier | Model class | Status |
|---|---|---|
| **Works today, witnessed** | sub-frontier dense/MoE up to ~27B (SmolLM2, Qwen2.5-1.5B/3B/7B, Qwen3.6-27B) | The showable tier — every number below is measured. |
| **Not yet (named, not hidden)** | frontier 753B MoE (GLM-5.2) on CPU-offload | Sub-1 tok/s today — the `--cpu-offload-experts` host-GEMM wall (#971). |
| **Not yet (no backend)** | TPU / Coral / cloud-XLA | fak has **no** TPU backend; it is an arm64-edge-class *target*, like the Pi 5 / Jetson row in [HARDWARE-MATRIX](../HARDWARE-MATRIX.md), nothing measured. |

The frontier-MoE wall is concrete and worth stating plainly, because it is the
honest boundary of the affordable-hardware story. On the 8-GPU Ampere server,
GLM-5.2 under `--cpu-offload-experts` decodes at **0.23 tok/s** steady-state and
gets *worse* under concurrency (2-way = 0.27× of single-stream — it serializes,
it does not batch); pure-CPU `llama.cpp` on the big-RAM host is **0.89 tok/s**,
~3.8× *faster* than fak's offload path
([GPU-SERVER-OVERNIGHT-PLAN](../nightrun/GPU-SERVER-OVERNIGHT-PLAN-2026-06-28.md)). The fix is
known — GLM-5.2's *dense* path is already pure-GPU witnessed (cosine 1.0 on the
datacenter GPU, [CLAIMS.md](../../CLAIMS.md) Engine §); the DSA sparse-attention
and DSA-KV are still host-side (#86/#413). Until the experts move off the host,
**frontier-MoE-on-affordable-hardware is not a usable-agent story**, and this
note does not pretend it is.

Everything in §2 is the tier where it *is* doable.

---

## 2. The hardware tiers — box → fleet lever → witnessed number → showable demo

### Tier 0 — one big-RAM CPU box (the most accessible)

*Example: a used EPYC / Threadripper / Xeon with 256+ GB RAM, no GPU.*

- **What it serves well:** dense ≤7B at Q8/Q4_K, or a small MoE. `FAK_Q4K=1 fak
  serve --gguf <shard> --addr 127.0.0.1:8080`, every agent points at the one
  endpoint.
- **Single-stream reality (measured):** Qwen2.5-1.5B Q8 decode ~18 tok/s @ 8
  workers, Qwen2.5-7B Q8 ~3.75 tok/s @ 16 workers on a contended 32-core x86 box
  ([WORKER-SCALING-DESKTOP-X86](../../experiments/session/WORKER-SCALING-DESKTOP-X86-20260624.md));
  on an uncontended M3 Pro, 1.5B Q8 = 28.9 tok/s, 7B Q8 = 8.7 tok/s
  ([M3-LLAMACPP-RESULTS](../benchmarks/M3-LLAMACPP-RESULTS.md)). Slow for chat;
  fine for a batched overnight fleet.
- **The fleet lever (measured):** CPU **batched** decode is real — SmolLM2-135M
  Q8 on Zen5 hits ~2,916 agg tok/s at batch 960
  ([BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)).
  And the *orchestration* substrate scales hard on pure CPU: `fanrun` ran **1,024
  real agent sessions** end-to-end on a CPU-only box (no GPU, no weights, no key —
  a deterministic offline planner per sub-agent) in 364 ms serial wall, with
  3,069 real cross-agent dedup hits and exactly `(N−1)·P` prefix tokens elided
  ([CLAIMS.md](../../CLAIMS.md) fan-out §). *Honest fence: `fanrun` is **serial by
  construction** and uses an offline planner; the live-model fan-out is open
  (#982).* It proves the coordination plane, not a decode rate.
- **The safety rail that makes it overnight-safe:** the load-time **capacity
  preflight** (`RefuseMemoryPlanIfTooBig` / `EstimateF32LoadMemoryPlan`) refuses
  a model that won't fit RAM headroom *before* allocation — the gate that exists
  precisely because the all-resident path once wedged a 1 TB host (#974,
  [GPU-SERVER-OVERNIGHT-PLAN](../nightrun/GPU-SERVER-OVERNIGHT-PLAN-2026-06-28.md)).
- **Showable today:** [`examples/fleet-reuse-demo`](../../examples/fleet-reuse-demo/)
  runs the N-agent shared-prefix reuse curve **with no GPU and no model** (exact
  byte accounting, `--offline`), landing in the ~1.5–4× vs-tuned region at N=5.

### Tier 1 — one or a few older datacenter GPUs (A100 / Ampere sm_80)

*Example: one or two second-hand A100-40GB, or a single L4/L40.*

- **The headline witnessed number:** Qwen3.6-**27B** Q4_K on the 8-GPU Ampere
  server, served through fak's gateway, sustains **~1,085 aggregate completion
  tok/s at 64 concurrent streams** and converges to **97% of raw SGLang at 128
  streams** ([BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)).
  That is the shape of the claim: a 27B agent fleet *does* saturate older
  datacenter silicon, and fak's gateway tax shrinks to a few percent at fleet
  scale. *(At low concurrency the gateway is a measurable tax, not a win —
  [cost.md](../industry-scorecard/cost.md). The value here is the adjudication +
  reuse plane riding on top, not throughput.)*
- **Pure-kernel reach (measured):** SmolLM2-135M Q8 decodes at **127.8 tok/s**
  through the pure `k_q8_gemm` + `k_flash_attention` path on the datacenter GPU,
  zero cuBLAS ([CLAIMS.md](../../CLAIMS.md) Engine §). On a single GCP L4, the
  live in-kernel CUDA serve warm-decodes Qwen2.5-0.5B Q8 at **~466 tok/s**
  single-stream ([L4-INKERNEL-SERVE](L4-INKERNEL-SERVE-AND-CONCURRENCY-FIX-2026-06-25.md)).
- **The fleet lever:** continuous batching is shipped on the in-kernel lifecycle
  path (#401, [CLAIMS.md](../../CLAIMS.md) "What fak is NOT" — no longer
  SIMULATED). *Honest fence: production-grade multi-tenant **p99** scheduling is a
  separate honest no-claim; goodput-under-SLA is unmeasured ([cost.md](../industry-scorecard/cost.md)).*
- **The fleet lever that needs few GPUs:** RadixAttention prefix sharing —
  measured **86.7% cache-aware hit rate** (100% of optimal), **7.50× token-count
  ceiling**, live **4.58× → 6.95×** as the model grows 135M → 1.5B
  ([RADIXATTENTION-RESULTS](../benchmarks/RADIXATTENTION-RESULTS.md)). The hit
  rate is hardware-independent and reproduces byte-for-byte across boxes, so it is
  the same win on one old A100 as on an H100.
- **Showable today:** the 27B concurrency curve is the strongest "few-old-GPUs
  serve an agent fleet" witness already in the authority. The natural next demo
  is the in-kernel **coding loop** on one A100/27B (the #933 run contract is
  shipped; the live capstone is GPU-gated).

### Tier 2 — a gaming GPU or a Mac (the "I already own this" tier)

*Example: an RTX 4070, a Radeon RX 7600, or an M3 Pro.*

- **Single-stream (measured):** RTX 4070 in-kernel CUDA decode ~120 tok/s at
  llama.cpp Q8 parity; Qwen2.5-1.5B f16 = 36.6 tok/s (1.07× *ahead* of
  llama.cpp f16) ([GPU-QWEN-RESULTS](../benchmarks/GPU-QWEN-RESULTS.md)).
- **The fleet lever (measured):** batched multi-user decode on the 4070 peaks at
  **862 aggregate tok/s at batch 512 — 44.92× over the serial baseline**
  ([BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)).
  One consumer card multiplexes a large agent fleet.
- **Mac, end-to-end fleet (measured):** the headline 50-turn × 5-agent
  Qwen2.5-1.5B run is **19.0 min vs 78 min tuned warm-cache = 4.1×** (and 60.3×
  vs the naive re-send loop); a 5×200 fleet on Qwen2.5-7B finishes in 8.2 min,
  2.5× vs tuned ([SESSION-VALUE-STACK-RESULTS](../benchmarks/SESSION-VALUE-STACK-RESULTS.md)).
- **Portability is the point:** RX 7600 Vulkan Q8 is argmax-exact vs cpu-ref
  (cosine 1.0) — the same kernel, same gates, proven not-NVIDIA-only
  ([HARDWARE-MATRIX](../HARDWARE-MATRIX.md)).
- **Showable today:** the fleet-value stack and the RadixAttention ladder both
  run on the Mac; the deletion / turn-tax / DENY demos run offline in ~1 s.

### Tier 3 — TPU / arm64 edge (target, not yet)

No fak TPU backend exists; cloud-TPU needs XLA, Coral is int8-only. The arm64
small-SBC row (Pi 5 / Jetson Orin) is a *pending* witness in
[HARDWARE-MATRIX](../HARDWARE-MATRIX.md) — listed so the axis is honest, with no
number claimed. If TPU support is wanted, it is a new backend behind the existing
`compute.Backend` seam, scoped like the Vulkan/Metal/CUDA backends, not a config flag.

---

## 3. The lever → fleet-usability map

Why each affordable tier becomes usable *for a fleet specifically*:

| fak lever | What it does for the fleet | Status |
|---|---|---|
| **RadixAttention prefix reuse** | N agents share one prefill of the system prompt + tools + repo; the dominant cost in a fleet is paid once. 86.7% hit, 7.50× token ceiling. | SHIPPED, measured, hardware-independent |
| **Continuous batching** (in-kernel lifecycle) | One box keeps many latency-tolerant agent streams saturated; aggregate throughput, not single-stream, is the meter. | SHIPPED (#401); multi-tenant p99 = no-claim |
| **Resident Q4_K / Q6_K / Q5_K weights** | Fit a bigger model on a smaller VRAM/RAM budget; kills the mixed-quant Q8 fallback. | SHIPPED |
| **Capacity / OOM preflight + classed fit** | Refuse a too-big model *before* it wedges a shared CPU/GPU box overnight; one bounded idle-pool retry instead of a hang. | SHIPPED |
| **Demote-not-evict KV (CXL / NUMA-far / DRAM tiers)** | Hold more agents' KV resident by demoting cold prefixes to far memory instead of recomputing. | SHIPPED (policy plane; auto served-loop hook is the open edge) |
| **Bit-exact mid-run KV eviction** | A long overnight session can drop a poisoned/stale span from the *middle* of the cache, `max|Δ|=0`, without re-prefilling. | SHIPPED |

---

## 4. The safety layer — what makes "unattended overnight" actually safe

This is the "safe ways" half of the question, and it is the half no token engine
supplies. Running a fleet while you sleep is only sane if the kernel does not
believe the agents:

- **Default-deny capability floor.** The agent literally cannot fire
  `refund_payment` / force-push / `git add -A` / out-of-tree write / destructive
  shell unless the policy admits it — an in-process check on the tool-call path,
  not a detector that can be talked past
  ([`presets/coding-agent-safe.json`](../../examples/presets/coding-agent-safe.json)).
- **Disjoint-lease arbitration (`dos_arbitrate`).** Two agents are admitted
  concurrently only when their file trees are pairwise disjoint; overlap
  *serializes by refusal*, so a fleet on one shared trunk never clobbers itself
  ([ultracode-dogfood](../explainers/ultracode-multi-agent-dogfood.md) §2).
- **The trunk guard.** Structurally-decidable git hazards (`OFF_TRUNK`,
  force-push) are refused *before* `git` runs.
- **Admission / fairness.** Waiting/running queues with a token budget +
  max-concurrency, priority with aging so no agent starves, `Deny→403` /
  `Shed→429` — one runaway agent can't take the fleet's capacity (internal/gateway
  admission; the live `/metrics` fold + HTTP-429 wire is gated on the native
  scheduler being on the serve loop).
- **Commit-audit on what they did.** `dos commit-audit` binds each commit's diff
  to its claim (`diff-witnessed` vs forgeable `subject-only`), so you wake up to
  *verified* work, not self-reported "done." The shipped issue-dispatch loop uses
  exactly this: ≤ `cap` detached workers, each ships a commit citing `#N`, closure
  gated on the audit, not the worker's word ([CLAIMS.md](../../CLAIMS.md)
  issue-dispatch §).
- **Result quarantine.** A poisoned tool result is walled at first admit and
  cannot replay into the shared prefix every later agent reads.
- **RSI keep-or-revert.** A self-improving overnight loop only keeps a change on a
  non-forgeable keep-bit (strict metric gain ∧ suite-green ∧ truth-clean) in an
  isolated worktree.

Together these turn "leave N agents running overnight" from reckless into
*bounded* — the population is provably capped, collisions refuse instead of
corrupt, destructive effects are denied by structure, and the morning diff is
audited, not trusted.

---

## 5. What we can actually SHOW (ranked, witnessed-today vs next step)

| Demo | What it proves about affordable-HW fleets | Status |
|---|---|---|
| `examples/fleet-reuse-demo --offline` | N agents share one prefix → ~1.5–4× less work, walled injection — **runs on any box, no GPU/model** | witnessed today |
| The 27B @ 64–128 concurrency curve | a fleet saturates *older* datacenter GPUs; gateway tax → ~3% at scale | witnessed today (authority row) |
| 4070 batched 862 agg tok/s @512 | one gaming GPU multiplexes a large agent fleet | witnessed today |
| 50-turn × 5-agent 4.1× (Mac) | end-to-end fleet wall-clock win vs a *tuned* stack | witnessed today |
| `fanrun` 1,024 agents on a CPU box | the coordination/dedup plane scales to 1k agents with no accelerator | witnessed today (serial, offline planner) |
| In-kernel **coding loop** on 1×A100/27B | a real coding agent fleet on one affordable GPU | **next step** — #933 contract shipped, live capstone GPU-gated (#982) |
| GLM-5.2 experts off the host | closes the frontier-MoE wall so the big model joins the tier | **next step** — dense path pure-GPU witnessed; DSA host-side (#86/#413/#971) |
| Pi 5 / Jetson arm64 edge row | bottom rung of the deployment axis | **next step** — pending witness, no number |

The first five are runnable now and honest; the last three are the named open
work, with the witness or wiring each is waiting on.

---

## 6. Honest fences (carried verbatim from the authority)

- **The reuse win is self-host only.** An app that just *calls* a frontier API
  gets fak's safety floor but **not** the prompt-reuse savings.
- **The multiples are denominator-labeled.** ~1.5–4.1× is vs a **tuned
  warm-cache** stack; ~60× is only vs the **naive re-send** loop; the agent-city
  10,000×10,000 numbers are **design targets, not measurements**
  ([SCALING-LAWS §0/§3](SCALING-LAWS-OF-AGENTS-2026-06-19.md)).
- **No $ / energy / tokens-per-watt claim.** There is no power meter on the box;
  every cost/energy figure would be SIMULATED ([cost.md](../industry-scorecard/cost.md)).
  The reuse story *implies* lower infra cost at equal quality; fak commits no
  dollar number.
- **Orchestration ≠ inference.** The multi-agent *concurrency factor*
  (independent reviewed deliverables / window) and fak's *inference* 5–10×
  (decode throughput) are different axes, measured by different methods, and must
  never be multiplied or quoted as one number
  ([ultracode-dogfood §3.4](../explainers/ultracode-multi-agent-dogfood.md)).
- **Frontier MoE on cheap hardware is not yet a usable-agent story** (§1).
- **fak is not a faster token engine.** vLLM/SGLang win raw throughput; fak owns
  the agent boundary and the reuse/coherence plane.

---

## 7. One-page version

> The agent fleet is the workload where affordable hardware wins, because the
> fleet is latency-tolerant (batch it overnight), prefix-heavy (pay the setup
> once), and unattended (so trust, not tok/s, is the binding constraint). On a
> CPU box, one or a few old A100s, a gaming GPU, or a Mac, fak's measured levers —
> RadixAttention prefix reuse, continuous batching, resident-Q4_K, capacity
> preflight — delete the duplicated work, and its safety floor — default-deny
> capabilities, disjoint-lease arbitration, commit-audit, quarantine — makes
> running N agents overnight bounded instead of reckless. The doable tier today is
> sub-frontier (≤27B) and every win in it is measured; the frontier 753B MoE on
> CPU-offload is sub-1 tok/s and named as the open wall, and TPU is an unbuilt
> backend, not a config. Show it with the GPU-free reuse demo, the 27B
> concurrency curve, the 4070 batched throughput, the Mac 4.1× fleet stack, and
> the 1,024-agent CPU `fanrun`.
