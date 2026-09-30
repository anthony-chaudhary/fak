# Useful local agents, accelerated automatically

**Direction set: 2026-09-09. Status: product milestone to qualify.**

Fak's first breakthrough milestone is to make useful local agents practical on
local machines: responsive, economical, and simple to run. A developer starts
Fak, gives an agent real work, and gets the benefit of native inference,
speculative decoding, and reusable agent context through the normal workflow.
Qualified acceleration should be automatic, with no collection of tuning flags
required to obtain the supported experience.

This is the first milestone we intend to earn. It does not assert that Fak is
the world's first local-agent runtime or that every capability below is already
qualified. The [benchmark authority](../BENCHMARK-AUTHORITY.md) governs measured
claims; the [native inference goal](native-inference-goal.md) governs execution
ownership and comparisons.

## Tell the story through the work

**Short positioning:** Fak is building the open runtime for useful local agents,
with acceleration that works automatically on supported machines.

**The experience we are building:** start locally, run a real agent task, and
keep working without repeatedly rebuilding the same context. Fak should make
each model step faster and avoid model work the agent has already paid for.
Instructions, repository context, tool results, and compatible subagent prefixes
are reusable working state, subject to exact cache identity and isolation rules.

Explain mechanisms after that outcome:

| Mechanism | User benefit | Product requirement |
|---|---|---|
| Native kernels, quantized residency, batching, and memory sizing | Useful generation within the machine's compute and memory budget | Select supported paths and account for contention, setup, and memory pressure. |
| Speculative decoding | Less waiting for generated tokens | Enable by default where model/backend compatibility, correctness, and net benefit are qualified; include draft and verification overhead. |
| Agentic context and prefix/KV caching | Less repeated prefill across turns and compatible agents | Reuse valid state automatically; invalidate precisely on changed inputs, model state, or authority. Include cold-cache cost. |
| Device-resident state and direct GPU cache/storage paths | Less data movement when context is reused or paged | Preserve residency where supported; use physical direct I/O only on a supported, witnessed path. |
| Harness, durable memory, and capability floor | A complete local work loop with continuity and bounded actions | Carry acceleration through the actual agent/tool path without weakening policy or changing task semantics. |

GPU-resident reuse, unified-memory access, and direct NVMe-to-GPU DMA are distinct
mechanisms. A host-memory cache hit or a zero-copy simulation does not establish
physical GPU Direct I/O. General local runtime mechanisms belong in public Fak;
commercial cache allocation and fleet policy follow their separate ownership
rules. This milestone does not authorize a code migration by itself.

## What automatic means

1. **Detect eligibility.** Identify the model, available backend, memory budget,
   cache compatibility, and required driver/device capabilities. Start with one
   explicitly supported configuration and expand after it is qualified.
2. **Choose and enable.** Make qualified acceleration part of the normal local
   entry point. An implemented algorithm or a flag whose default is `true` does
   not satisfy this contract without a reachable, exercised execution path.
3. **Preserve correctness.** Validate speculative acceptance, cache identity,
   invalidation, and isolation. Never trade away the declared quality floor or
   tool capability floor for a throughput number.
4. **Expose the result.** Report the selected engine/backend, active acceleration,
   cache reuse, and the reason for each disabled or unavailable path. Advanced
   overrides remain available for diagnosis and reproducibility.
5. **Handle limits visibly.** Where a correct slower native path exists, report
   its use and remain operable. Otherwise return a clear unsupported result.
   Never silently switch to a hosted model or an external inference engine.
6. **Prove net benefit.** Measure the normal agent workflow after warmup, draft,
   verification, paging, and runtime overhead. A regression disables or holds
   that optimization for the affected configuration until it is qualified again.

## Current implementation is narrower than the milestone

Source inspection on 2026-09-09 at commit
`816506b53c263edd44e4b2a151ac80cac3b44031` is a wiring snapshot, not a hardware performance receipt.
Recheck these seams at the revision being shipped:

| Surface | Observed state | Source |
|---|---|---|
| Local startup and backend choice | `fak up` targets Apple Silicon with model/context memory sizing; Metal can be selected automatically. CUDA/Vulkan selection is not a universal automatic default. | [up](../cmd/fak/up.go), [model/backend selection](../cmd/fak/serve_model_load.go) |
| In-kernel prefix reuse | Default-on for eligible architecture/backend snapshot combinations. Device paths need compatible state; reuse is not universal. | [planner configuration](../internal/agent/inkernel_planner_config.go) |
| Speculative decoding | Metal hybrid-model MTP requires explicit configuration such as `FAK_SPECULATIVE=mtp`. Default-on speculation is still a milestone requirement. | [MTP eligibility](../internal/gateway/chat_completions.go), [MTP claim status](claims/qwen38-native-mtp-speculative-decode.md) |
| GPU Direct overflow | A true-valued CLI flag does not establish execution: the serve field has no runtime consumer in this snapshot. The claim ledger labels NVIDIA BaM NVMe evidence simulated. | [serve flags](../cmd/fak/serve.go), [claim ledger](../CLAIMS.md) |

## Earn the milestone on one real local workflow

Choose a bounded coding task with an independent acceptance witness: inspect a
repository, make a change, run its relevant check, and resume or reuse compatible
context for a follow-up. Keep the model, quality floor, machine, task inputs, and
resource budget explicit. Test at least one compatible subagent branch if fanout
is part of the claim. Start with one qualified model/device combination.

Before running, declare absolute task-latency and interactive-response ceilings,
memory limits, and a correctness floor appropriate to that workload. Then retain:

- Accepted task outcomes and time to verified completion; TTFT and decode rate
  explain parts of the result but do not replace task performance.
- Cold startup, first prefill, warm continuation, shared-prefix fanout when
  claimed, and cache invalidation/recovery behavior.
- Memory pressure, operator setup/intervention, and energy or cost when measured.
  Missing measurements stay unmeasured.
- A tuned baseline with matched task/model/quality and cache state, plus the
  exact automatic configuration and any fallback or disabled features.
- Durable receipts naming date, source SHA, model/artifact, device, compiler,
  driver, power/thermal state, commands, sample count, and acceptance witness.
  For repeated measurements report P50/P90 and confidence intervals; flag a
  single sample as high outlier risk. Use the existing receipt/claim process.

Software-only tests establish wiring and semantics. Simulations remain
**[SW-VERIFIED]**; physical performance and direct-I/O criteria remain unchecked
**[HW-WITNESSED]** until measured on real hardware. A successful component test
does not close the complete local-agent milestone.

## Instructions for contributing agents

Use this brief when planning, delegating, reviewing, or writing product copy:

> Deliver Fak's first local-agent milestone: useful local work, accelerated
> automatically. Name the user workflow and its current bottleneck. Trace the
> change from the normal entry point through native execution and legal context
> reuse to an independently accepted task result and durable receipt. Enable
> supported optimizations by default only after compatibility, correctness, and
> net benefit are qualified. Report active paths and fallback reasons. Keep
> shipped behavior, software verification, hardware qualification, and product
> ambition distinct.

In each worker packet name: **workflow; bottleneck; entry-point/call-path seam;
supported model/device; expected default; acceptance witness; evidence still
missing**. Prefer a bounded improvement to that path over additional knobs,
backend breadth, or factory automation without a named product dependency.

Lead public explanations with useful local work and automatic acceleration.
Explain speculative decoding and agentic caching as the mechanisms supporting
that outcome. Use "world's first", "fastest", universal "zero cold start", or
"all optimizations on" only when their exact scope is independently established.
Code presence, flag declarations, and simulated ratios are not that evidence.
